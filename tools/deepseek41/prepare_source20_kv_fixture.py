"""Pinned source-20 ratio-1 compressed-KV component fixture, offline only.

Uses unmodified official Compressor/RMSNorm/RoPE/Attention._compress_kv with
the original BF16 weights. Index selection is replaced by a pre-RoPE capture;
the existing independent NumPy FP4/E4M3 oracle replaces the GPU quantizer.
This is not a full attention/indexer reference or official GPU parity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from prepare_compressed_kv_fixture import edge_input, fp4_roundtrip
from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_mhc_fixture import KERNEL_HASH
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
    ratios = json.loads((model_dir / "config.json").read_text())["text_config"]["compress_ratios"]
    if ratios[20] != 1:
        raise ValueError("Source 20 must have ratio 1")
    torch.set_num_threads(4)
    tree = ast.parse(reference.read_bytes())
    names = {"Compressor", "RMSNorm", "precompute_freqs_cis", "apply_rotary_emb"}
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    attention = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Attention")
    method = next(node for node in attention.body if isinstance(node, ast.FunctionDef) and node.name == "_compress_kv")
    wrapper = ast.ClassDef(name="ReferenceAttention", bases=[], keywords=[], decorator_list=[], body=[method])
    shared = SimpleNamespace()
    ns = {"torch": torch, "nn": nn, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw),
          "fp4_act_quant": fp4_roundtrip, "shared_attn": shared}
    module = ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[]))
    exec(compile(module, str(reference), "exec"), ns)
    args = SimpleNamespace(compress_ratios=ratios, dim=5120, head_dim=512, norm_eps=1e-20, max_batch_size=1)
    compressor = ns["Compressor"](args, 20).eval()
    checkpoint = Checkpoint(model_dir)
    records = {}
    for part, shape in [("wkv", [512, 5120]), ("norm", [512])]:
        name = f"layers.20.attn.compressor.{part}.weight"
        if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source-20 compressor tensor {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite source-20 tensor {name}")
        getattr(compressor, part).weight.data = value
        records[name] = digest
    writer = GGUFWriter()
    TYPE_IDS["BF16"] = 30
    for key, value in [
        ("general.architecture", "deepseek41-source20-kv-fixture"),
        ("strata.deepseek41.revision", REVISION),
        ("strata.deepseek41.official_source_sha256", SOURCE_HASH),
        ("strata.deepseek41.official_kernel_sha256", KERNEL_HASH),
        ("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True)),
    ]:
        writer.add(key, value)

    def add(name, value, dtype="F32"):
        value = value.detach().contiguous()
        if not torch.isfinite(value).all():
            raise AssertionError(f"Nonfinite reference tensor {name}")
        raw = value.float().numpy().tobytes() if dtype == "F32" else value.bfloat16().view(torch.uint16).numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), dtype, raw))

    frequencies = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    sample = torch.randn(1, 3, 512, generator=torch.Generator().manual_seed(1021)).bfloat16()
    cases = [(0, edge_input()), (1, edge_input())]
    cases += [(position, sample) for position in
              [0, 1, 2, 7, 8, 9, 127, 65534, 65535, 65536, 131071, 131072, 524287, 524288, 1048573]]
    cases += [(1048574, sample[:, :2]), (1048575, sample[:, :1])]
    writer.add("fixture.cases", len(cases), "u32")
    base = torch.randn(1, 17, 5120, generator=torch.Generator().manual_seed(1020)).bfloat16()
    plans = [(base, first) for first in [17, 1, 7, 8, 9, 13]]
    plans += [(torch.zeros_like(base[:, :2]), 1), ((base[:, :2] * 1e-12).bfloat16(), 2),
              ((base[:, :2] * 256).bfloat16(), 1)]
    writer.add("fixture.plans", len(plans), "u32")
    calls = 0
    captured = {}

    def capture(_x, _qr, latent, _start, _offset, _length):
        if latent is None:
            raise AssertionError("Ratio-1 cannot delay a token")
        captured["latent"] = latent.clone()
        return None

    with torch.inference_mode():
        for case, (position, latent) in enumerate(cases):
            prefix = f"case.{case}."
            count = latent.shape[1]
            writer.add(prefix + "position", position, "u32")
            writer.add(prefix + "tokens", count, "u32")
            add(prefix + "input", latent)
            rotated = latent.clone()
            ns["apply_rotary_emb"](rotated[..., -64:], frequencies[position:position + count])
            add(prefix + "rotated", rotated)
            add(prefix + "output", fp4_roundtrip(rotated), "BF16")

        attn = ns["ReferenceAttention"]()
        attn.compressor, attn.compress_ratio, attn.is_kv_source, attn.rope_head_dim = compressor, 1, True, 64
        attn.freqs_cis = frequencies
        attn.compress_kv_cache = torch.zeros(1, 17, 512, dtype=torch.bfloat16)
        attn._compress_topk_idxs = capture
        for plan, (data, first) in enumerate(plans):
            prefix = f"plan.{plan}."
            writer.add(prefix + "tokens", data.shape[1], "u32")
            writer.add(prefix + "prefill", first, "u32")
            add(prefix + "input", data)
            attn.compress_kv_cache.zero_()
            position = 0
            while position < data.shape[1]:
                count = first if position == 0 else 1
                cache, _ = attn._compress_kv(data[:, position:position + count], None, position, 128)
                if captured["latent"].shape != (1, count, 512) or cache.shape != (1, position + count, 512):
                    raise AssertionError("Official ratio-1 latent/cache count mismatch")
                call = prefix + f"at.{position}."
                add(call + "latent", captured["latent"])
                add(call + "cache", cache, "BF16")
                position += count
                calls += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "source_hashes": records, "postprocess_cases": len(cases),
            "case_positions": [position for position, _ in cases],
            "sequence_plans": len(plans), "sequence_calls": calls,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__,
            "scope": "official CPU ratio-1 Compressor/RMSNorm/RoPE/_compress_kv; independent NumPy FP4/E4M3 roundtrip; index selection omitted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel, args.output), indent=2))


if __name__ == "__main__":
    main()

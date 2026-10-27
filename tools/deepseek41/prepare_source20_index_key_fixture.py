"""Pinned 1017 source-20 ratio-1 index key oracle, using local checkpoint tensors only.

Executes the official Indexer.forward prefix through key-cache publication; query
projection, scores and top-k are excluded. FP4/E8M0 uses independent NumPy math.
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

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_mhc_fixture import KERNEL_HASH
from prepare_index_key_fixture import edge_input, fp4_roundtrip


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
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and
                   n.name in {"Compressor", "RMSNorm", "precompute_freqs_cis", "apply_rotary_emb"}]
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    forward = next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    cut = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q")
    forward.body = forward.body[:cut]
    wrapper = ast.ClassDef(name="ReferenceIndexKey", bases=[], keywords=[], decorator_list=[], body=[forward])
    shared = SimpleNamespace()
    ns = {"torch": torch, "nn": nn, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw),
          "fp4_act_quant": fp4_roundtrip, "fp4_block_size": 32, "shared_attn": shared}
    module = ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[]))
    exec(compile(module, str(reference), "exec"), ns)
    compressor = ns["Compressor"](SimpleNamespace(compress_ratios=ratios, dim=5120,
                    head_dim=512, norm_eps=1e-20, max_batch_size=1), 20).eval()
    key = ns["ReferenceIndexKey"]()
    key.owns_k, key.compress_ratio, key.rope_head_dim = True, 1, 64
    key.wk = nn.Linear(512, 128, bias=False, dtype=torch.bfloat16).eval()
    key.k_norm = ns["RMSNorm"](128, 1e-20).eval()
    key.k_cache = torch.zeros(1, 17, 128, dtype=torch.bfloat16)
    checkpoint = Checkpoint(model_dir)
    records, key_tensors = {}, {}
    for component, parts in [("compressor", [("wkv", [512, 5120]), ("norm", [512])]),
                             ("indexer", [("wk", [128, 512]), ("k_norm", [128])])]:
        for part, shape in parts:
            name = f"layers.20.attn.{component}.{part}.weight"
            if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
                raise ValueError(f"Malformed source {name}")
            raw = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(raw).hexdigest()
            if provenance["tensors"][name]["sha256"] != digest:
                raise ValueError(f"Cached source hash mismatch: {name}")
            records[name] = digest
            value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source tensor {name}")
            target = compressor if component == "compressor" else key
            getattr(target, part).weight.data = value
            if component == "indexer":
                key_tensors[f"blk.20.attn_indexer_{part}.weight"] = value
    writer = GGUFWriter(); TYPE_IDS["BF16"] = 30
    writer.add("general.architecture", "deepseek41-source20-index-key-fixture")
    writer.add("strata.deepseek41.index_key_profile", "1017-layer20-index-key128-bf16")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    writer.add("strata.deepseek41.compressor_source_hashes",
               json.dumps({name: digest for name, digest in records.items() if ".compressor." in name}, sort_keys=True))

    def add(name, value, dtype="F32"):
        data = value.detach().contiguous()
        if not torch.isfinite(data).all():
            raise AssertionError(f"Nonfinite reference tensor {name}")
        raw = data.float().numpy().tobytes() if dtype == "F32" else data.bfloat16().view(torch.uint16).numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(data.shape)), dtype, raw))

    for name, value in key_tensors.items():
        add(name, value, "BF16")
    key.freqs_cis = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    sample = torch.randn(1, 3, 128, generator=torch.Generator().manual_seed(1022)).bfloat16()
    cases = [(0, edge_input()), (1, edge_input())]
    cases += [(position, sample) for position in
              [0, 1, 2, 7, 8, 9, 127, 65534, 65535, 65536, 131071, 131072, 524287, 524288, 1048573]]
    cases += [(1048574, sample[:, :2]), (1048575, sample[:, :1])]
    writer.add("fixture.cases", len(cases), "u32")
    captured = {}
    key.wk.register_forward_hook(lambda _module, _inputs, result: captured.update(projected=result.clone()))
    key.k_norm.register_forward_hook(lambda _module, _inputs, result: captured.update(normalized=result.clone()))
    with torch.inference_mode():
        for case, (position, normalized) in enumerate(cases):
            prefix = f"case.{case}."
            writer.add(prefix + "position", position, "u32")
            writer.add(prefix + "tokens", normalized.shape[1], "u32")
            add(prefix + "input", normalized)
            rotated = normalized.clone()
            ns["apply_rotary_emb"](rotated[..., -64:], key.freqs_cis[position:position + normalized.shape[1]])
            add(prefix + "rotated", rotated)
            add(prefix + "output", fp4_roundtrip(rotated), "BF16")
        # Identical sequence/partitions to the confirmed source-20 compressor/KV fixtures.
        base = torch.randn(1, 17, 5120, generator=torch.Generator().manual_seed(1020)).bfloat16()
        plans = [(base, first) for first in [17, 1, 7, 8, 9, 13]]
        plans += [(torch.zeros_like(base[:, :2]), 1), ((base[:, :2] * 1e-12).bfloat16(), 2),
                  ((base[:, :2] * 256).bfloat16(), 1)]
        writer.add("fixture.plans", len(plans), "u32")
        calls = 0
        for plan, (data, first) in enumerate(plans):
            writer.add(f"plan.{plan}.tokens", data.shape[1], "u32")
            writer.add(f"plan.{plan}.prefill", first, "u32")
            add(f"plan.{plan}.input", data)
            key.k_cache.zero_()
            position = 0
            while position < data.shape[1]:
                length = first if position == 0 else 1
                x = data[:, position:position + length]
                latent = compressor(x, position)
                if latent is None or latent.shape != (1, length, 512):
                    raise AssertionError("Ratio-1 must emit every token")
                key.forward(x, None, latent, position, 128)
                prefix = f"plan.{plan}.at.{position}."
                add(prefix + "latent", latent)
                add(prefix + "projected", captured["projected"])
                add(prefix + "normalized", captured["normalized"])
                add(prefix + "cache", shared.index_k[:, :position + length], "BF16")
                position += length
                calls += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "postprocess_cases": len(cases), "source_hashes": records, "sequence_plans": len(plans), "sequence_calls": calls,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__,
            "scope": "official CPU ratio-1 Compressor/Indexer key prefix/RoPE; independent E8M0/FP4; no query or selection"}


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

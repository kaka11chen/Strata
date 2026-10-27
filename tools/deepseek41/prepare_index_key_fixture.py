"""Pinned 1017 layer-2 index key oracle, using local checkpoint tensors only.

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
from prepare_compressed_kv_fixture import nearest


def fp4_roundtrip(x, block_size=32, inplace=True):
    assert block_size == 32 and inplace
    values = x.float().numpy().reshape(-1, 32)
    maximum = np.maximum(np.abs(values).max(-1, keepdims=True), np.float32(6 * 2**-126))
    # F32 multiply precedes the power-of-two ceiling in fast_round_scale.
    scaled = maximum * np.float32(1 / 6)
    fraction, exponent = np.frexp(scaled)
    scale = np.ldexp(np.ones_like(scaled), exponent - (fraction == np.float32(.5)))
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    quantized = nearest((np.abs(values) / scale).clip(0, 6), grid)
    restored = np.copysign(quantized, values) * scale
    x.copy_(torch.from_numpy(restored.reshape(x.shape)).bfloat16())
    return x


def edge_input():
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.,
                         -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5., -6.] * 2)
    blocks = [torch.zeros(32), -torch.zeros(32), ties,
              ties * 1.0078125, ties * .9921875, ties * 2.0**117,
              ties * 2**-126, ties * 2**-130,
              torch.nextafter(ties.bfloat16(), torch.full((32,), torch.inf, dtype=torch.bfloat16)).float(),
              torch.nextafter(ties.bfloat16(), torch.full((32,), -torch.inf, dtype=torch.bfloat16)).float()]
    return torch.cat([blocks[i % len(blocks)] for i in range(16)]).reshape(1, 4, 128).bfloat16()


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
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
    compressor = ns["Compressor"](SimpleNamespace(compress_ratios=[0, 0, 2], dim=5120,
                    head_dim=512, norm_eps=1e-20, max_batch_size=1), 2).eval()
    key = ns["ReferenceIndexKey"]()
    key.owns_k, key.compress_ratio, key.rope_head_dim = True, 2, 64
    key.wk = nn.Linear(512, 128, bias=False, dtype=torch.bfloat16).eval()
    key.k_norm = ns["RMSNorm"](128, 1e-20).eval()
    key.k_cache = torch.zeros(1, 8, 128, dtype=torch.bfloat16)
    checkpoint = Checkpoint(model_dir)
    records, key_tensors = {}, {}
    for component, parts in [("compressor", [("wkv", [512, 5120]), ("wgate", [512, 5120]), ("norm", [512])]),
                             ("indexer", [("wk", [128, 512]), ("k_norm", [128])])]:
        for part, shape in parts:
            name = f"layers.2.attn.{component}.{part}.weight"
            if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
                raise ValueError(f"Malformed source {name}")
            raw = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(raw).hexdigest()
            if provenance["tensors"][name]["sha256"] != digest:
                raise ValueError(f"Cached source hash mismatch: {name}")
            records[name] = digest
            value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
            target = compressor if component == "compressor" else key
            getattr(target, part).weight.data = value.float() if component == "compressor" and part != "norm" else value
            if component == "indexer":
                key_tensors[f"blk.2.attn_indexer_{part}.weight"] = value
    writer = GGUFWriter(); TYPE_IDS["BF16"] = 30
    writer.add("general.architecture", "deepseek41-index-key-fixture")
    writer.add("strata.deepseek41.index_key_profile", "1017-layer2-index-key128-bf16")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))

    def add(name, value, dtype="F32"):
        data = value.detach().contiguous()
        raw = data.float().numpy().tobytes() if dtype == "F32" else data.bfloat16().view(torch.uint16).numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(data.shape)), dtype, raw))

    for name, value in key_tensors.items():
        add(name, value, "BF16")
    key.freqs_cis = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    generator = torch.Generator().manual_seed(1017)
    # Preserve the step-6b sequence exactly for integration of both caches.
    sample512 = torch.randn(1, 3, 512, generator=generator).bfloat16()
    sample = sample512[..., :128].clone()
    cases = [(p, edge_input()) for p in [0, 65536, 1048568]]
    cases += [(p, sample[:, :1] if p == 1048574 else sample)
              for p in [0, 2, 126, 65532, 65536, 131068, 131072, 1048570, 1048574]]
    writer.add("fixture.cases", len(cases), "u32")
    captured = {}
    key.wk.register_forward_hook(lambda _module, _inputs, result: captured.update(projected=result.clone()))
    key.k_norm.register_forward_hook(lambda _module, _inputs, result: captured.update(normalized=result.clone()))
    with torch.inference_mode():
        for case, (position, normalized) in enumerate(cases):
            prefix = f"case.{case}."
            writer.add(prefix + "position", position, "u32")
            writer.add(prefix + "groups", normalized.shape[1], "u32")
            add(prefix + "input", normalized)
            rotated = normalized.clone()
            ns["apply_rotary_emb"](rotated[..., -64:], key.freqs_cis[position:position + 2 * normalized.shape[1]:2])
            add(prefix + "rotated", rotated)
            add(prefix + "output", fp4_roundtrip(rotated), "BF16")
        base = torch.randn(1, 13, 5120, generator=generator).bfloat16()
        add("sequence.input", base)
        writer.add("fixture.plans", 5, "u32")
        calls = 0
        for plan, first in enumerate([13, 5, 6, 1, 2]):
            writer.add(f"plan.{plan}.prefill", first, "u32")
            compressor.kv_state.zero_(); compressor.score_state.fill_(-torch.inf)
            key.k_cache.zero_()
            position = 0
            while position < 13:
                length = first if position == 0 else 1
                x = base[:, position:position + length]
                latent = compressor(x, position)
                key.forward(x, None, latent, position, 128)
                prefix = f"plan.{plan}.at.{position}."
                if latent is not None:
                    add(prefix + "latent", latent)
                    add(prefix + "projected", captured["projected"])
                    add(prefix + "normalized", captured["normalized"])
                count = (position + length) // 2
                if count:
                    add(prefix + "cache", shared.index_k[:, :count], "BF16")
                position += length; calls += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "postprocess_cases": len(cases), "sequence_plans": 5, "sequence_calls": calls,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__,
            "scope": "official CPU Compressor/Indexer key prefix/RoPE; independent E8M0/FP4; no query or selection"}


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

"""Pinned 1017 compressed KV oracle: official RoPE/Compressor/_compress_kv graph.

Uses local checkpoint tensors only. Index selection is replaced with a capture
hook; the FP4 kernel is represented by independent NumPy nearest-even rounding.
This is a component fixture, not full compressed attention or official GPU parity.
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


def nearest(value, grid):
    high = np.searchsorted(grid, value).clip(0, len(grid) - 1)
    low = (high - 1).clip(0)
    a, b = value - grid[low], grid[high] - value
    code = np.where((b < a) | ((b == a) & (high % 2 == 0)), high, low)
    return grid[code]


def fp4_roundtrip(x, block_size=16, inplace=True, scale_dtype=torch.float8_e4m3fn):
    assert block_size == 16 and inplace and scale_dtype == torch.float8_e4m3fn
    values = x.float().numpy().reshape(-1, 16)
    grid8 = np.array([i / 512 if i < 8 else math.ldexp(1 + (i & 7) / 8, (i >> 3) - 7)
                      for i in range(127)], dtype=np.float32)
    grid4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    maximum = np.maximum(np.abs(values).max(-1, keepdims=True), np.float32(6 / 512))
    scale = nearest(maximum / np.float32(6), grid8)
    quantized = nearest((np.abs(values) / scale).clip(0, 6), grid4)
    restored = np.copysign(quantized, values) * scale
    x.copy_(torch.from_numpy(restored.reshape(x.shape)).bfloat16())
    return x


def edge_input():
    # FP4 midpoint ties, adjacent BF16 values, E4M3 scale ties, subnormal scales,
    # signed zero and differing 16-value groups. Tail is also tested by RoPE cases.
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.,
                         -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5., -6.])
    blocks = [torch.zeros(16), -torch.zeros(16), ties,
              ties * 1.0625, ties * 1.1875, ties * 128,
              ties / 512, ties / 4096,
              torch.nextafter(ties.bfloat16(), torch.full((16,), torch.inf, dtype=torch.bfloat16)).float(),
              torch.nextafter(ties.bfloat16(), torch.full((16,), -torch.inf, dtype=torch.bfloat16)).float()]
    return torch.cat([blocks[i % len(blocks)] for i in range(32)]).reshape(1, 1, 512).bfloat16()


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
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    wrapper = ast.ClassDef(name="ReferenceAttention", bases=[], keywords=[], decorator_list=[],
                           body=[n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_kv"])
    shared = SimpleNamespace()
    ns = {"torch": torch, "nn": nn, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw),
          "fp4_act_quant": fp4_roundtrip, "shared_attn": shared}
    module = ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[]))
    exec(compile(module, str(reference), "exec"), ns)
    args = SimpleNamespace(compress_ratios=[0, 0, 2], dim=5120, head_dim=512, norm_eps=1e-20, max_batch_size=1)
    compressor = ns["Compressor"](args, 2).eval()
    checkpoint = Checkpoint(model_dir)
    records = {}
    for part, shape in [("wkv", [512, 5120]), ("wgate", [512, 5120]), ("norm", [512])]:
        name = f"layers.2.attn.compressor.{part}.weight"
        if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed compressor source {name}")
        raw = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        getattr(compressor, part).weight.data = value if part == "norm" else value.float()
    writer = GGUFWriter(); TYPE_IDS["BF16"] = 30
    writer.add("general.architecture", "deepseek41-compressed-kv-fixture")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))

    def add(name, value, dtype="F32"):
        data = value.detach().contiguous()
        raw = data.float().numpy().tobytes() if dtype == "F32" else data.bfloat16().view(torch.uint16).numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(data.shape)), dtype, raw))

    frequencies = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    generator = torch.Generator().manual_seed(1017)
    sample = torch.randn(1, 3, 512, generator=generator).bfloat16()
    cases = [(0, edge_input())]
    cases += [(position, sample[:, :1] if position == 1048574 else sample)
              for position in [0, 2, 126, 65532, 65536, 131068, 131072, 1048570, 1048574]]
    writer.add("fixture.cases", len(cases), "u32")
    with torch.inference_mode():
        for case, (position, latent) in enumerate(cases):
            prefix = f"case.{case}."
            writer.add(prefix + "position", position, "u32")
            writer.add(prefix + "groups", latent.shape[1], "u32")
            add(prefix + "input", latent)
            rotated = latent.clone()
            ns["apply_rotary_emb"](rotated[..., -64:], frequencies[position:position + 2 * latent.shape[1]:2])
            add(prefix + "rotated", rotated)
            add(prefix + "output", fp4_roundtrip(rotated), "BF16")

        attn = ns["ReferenceAttention"]()
        attn.compressor, attn.compress_ratio, attn.is_kv_source, attn.rope_head_dim = compressor, 2, True, 64
        attn.freqs_cis = frequencies
        attn.compress_kv_cache = torch.zeros(1, 8, 512, dtype=torch.bfloat16)
        captured = {}

        def capture(_x, _qr, latent, _start, _offset, _length):
            captured["latent"] = None if latent is None else latent.clone()
            return None

        attn._compress_topk_idxs = capture
        base = torch.randn(1, 13, 5120, generator=generator).bfloat16()
        add("sequence.input", base)
        writer.add("fixture.plans", 5, "u32")
        for plan, first in enumerate([13, 5, 6, 1, 2]):
            writer.add(f"plan.{plan}.prefill", first, "u32")
            compressor.kv_state.zero_(); compressor.score_state.fill_(-torch.inf)
            attn.compress_kv_cache.zero_()
            position = 0
            while position < 13:
                length = first if position == 0 else 1
                cache, _ = attn._compress_kv(base[:, position:position + length], None, position, 128)
                prefix = f"plan.{plan}.at.{position}."
                if captured["latent"] is not None:
                    add(prefix + "latent", captured["latent"])
                if cache.shape[1]:
                    add(prefix + "cache", cache, "BF16")
                position += length
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "postprocess_cases": len(cases), "sequence_plans": 5, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "torch_oracle_version": torch.__version__,
            "scope": "official CPU RoPE/Compressor/_compress_kv; independent FP4 rounding; index selection omitted"}


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

"""Pinned 1017 layer-2 Compressor.forward + RMSNorm oracle, using local BF16 weights.

Runs the official ratio-2 graph on CPU, with ordinary F32 torch.nn.Linear for
its non-quantized projections. No runtime implementation is imported, no remote
fetch is performed, and an existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION


def prepare(model_dir: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
    torch.set_num_threads(4)
    tree = ast.parse(reference.read_bytes())
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Compressor", "RMSNorm"}]
    namespace = {"torch": torch, "nn": nn, "ModelArgs": SimpleNamespace,
                 "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), namespace)
    args = SimpleNamespace(compress_ratios=[0, 0, 2], dim=5120, head_dim=512, norm_eps=1e-20, max_batch_size=1)
    compressor = namespace["Compressor"](args, 2).eval()
    checkpoint = Checkpoint(model_dir)
    writer = GGUFWriter(); TYPE_IDS["BF16"] = 30
    writer.add("general.architecture", "deepseek41-compressor-fixture")
    writer.add("strata.deepseek41.compressor_profile", "1017-layer2-ratio2-bf16")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    records = {}
    for part, shape in [("wkv", [512, 5120]), ("wgate", [512, 5120]), ("norm", [512])]:
        name = f"layers.2.attn.compressor.{part}.weight"
        if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed compressor source {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        writer.tensors.append(_Tensor(f"blk.2.attn_compressor_{part}.weight", list(reversed(shape)), "BF16", raw))
        value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite compressor source {name}")
        getattr(compressor, part).weight.data = value if part == "norm" else value.float()

    def add(name, value):
        value = value.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), "F32", value.numpy().tobytes()))

    generator = torch.Generator().manual_seed(1017)
    base = torch.randn(1, 13, 5120, generator=generator).bfloat16()
    plans = [(base, first) for first in [13, 5, 6, 1, 2]]
    plans += [(torch.zeros_like(base[:, :2]), 2), ((base[:, :2] * 1e-12).bfloat16(), 1),
              ((base[:, :2] * 256).bfloat16(), 2)]
    writer.add("fixture.plans", len(plans), "u32")
    calls = 0
    with torch.inference_mode():
        for plan, (data, first) in enumerate(plans):
            # Reset official non-persistent buffers explicitly, including a pending odd tail.
            compressor.kv_state.zero_(); compressor.score_state.fill_(-torch.inf)
            prefix = f"plan.{plan}."
            writer.add(prefix + "tokens", data.shape[1], "u32")
            writer.add(prefix + "prefill", first, "u32")
            add(prefix + "input", data)
            position = 0
            while position < data.shape[1]:
                length = first if position == 0 else 1
                output_value = compressor(data[:, position:position + length], position)
                call = prefix + f"at.{position}."
                groups = (position + length) // 2 - position // 2
                if (output_value is None) != (groups == 0):
                    raise AssertionError("Official compressor emission disagrees with ratio-2 grouping")
                if output_value is not None:
                    if output_value.shape != (1, groups, 512):
                        raise AssertionError("Unexpected official compressor shape")
                    add(call + "output", output_value)
                position += length
                # Only the active tail is a contract; stale completed-group slots are not.
                if position % 2:
                    add(call + "kv", compressor.kv_state[0, 0])
                    add(call + "scores", compressor.score_state[0, 0])
                calls += 1
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "plans": len(plans), "calls": calls, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "torch_oracle_version": torch.__version__,
            "scope": "official CPU Compressor.forward + RMSNorm; F32 projections; BF16 pre-RoPE output"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.output), indent=2))


if __name__ == "__main__":
    main()

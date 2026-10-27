"""Prepare a sparse, partial 1017 routed-layer test fixture from locally cached tensors.

Uses the hash-pinned official Gate.forward on a seeded BF16 input to choose experts.
Missing selected experts are errors. Never substitutes routes. Torch is used only
by this offline oracle; the C++ inference path has no torch dependency.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, repack_fp4

CONFIG_HASH = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
SOURCE_HASH = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def prepare(model_dir: Path, reference: Path, output: Path, *, input_tensor: torch.Tensor | None = None,
            extra_experts: tuple[int, ...] = (), layer: int = 0) -> dict:
    if layer not in (0, 1, 2, 3, 8, 14, 20):
        raise ValueError("Only fixed text layer 0, 1, 2, 3, 8, 14 or 20 fixtures are supported")
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest() != CONFIG_HASH:
        raise ValueError("Not the pinned 1017 config")
    source = reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("Not the pinned official inference_model.py")
    checkpoint = Checkpoint(model_dir)
    prefix = f"layers.{layer}.ffn.gate"
    if checkpoint.dtype(prefix + ".weight") != "BF16" or checkpoint.shape(prefix + ".weight") != [384, 5120]:
        raise ValueError("Wrong router tensor")
    if checkpoint.dtype(prefix + ".bias") != "F32" or checkpoint.shape(prefix + ".bias") != [384]:
        raise ValueError("Wrong router bias")
    router = checkpoint.array(prefix + ".weight", np.uint16)
    bias = checkpoint.array(prefix + ".bias", np.float32)
    definitions = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Gate"]
    if len(definitions) != 1:
        raise ValueError("Official Gate definition missing")
    namespace = {"torch": torch, "nn": nn, "F": F, "linear": F.linear, "ModelArgs": SimpleNamespace}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), namespace)
    args = SimpleNamespace(dim=5120, get_moe_config=lambda _: (384, 6), score_func="sqrtsoftplus", gate_temp=1.,
                           norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
    gate = namespace["Gate"](layer, args)
    gate.weight.data = torch.from_numpy(router.copy()).view(torch.bfloat16).float()
    gate.bias.data = torch.from_numpy(bias.copy())
    if input_tensor is not None and (input_tensor.dtype != torch.bfloat16 or input_tensor.shape != (1, 5120)
                                     or not torch.isfinite(input_tensor).all()):
        raise ValueError("Supplied routed fixture input must be finite BF16 [1,5120]")
    x = (torch.randn((1, 5120), generator=torch.Generator().manual_seed(1017)).bfloat16()
         if input_tensor is None else input_tensor.detach().cpu())
    with torch.inference_mode():
        weights, ids = gate(x)
    selected = ids[0].tolist()
    if any(type(expert) is not int or not 0 <= expert < 384 for expert in extra_experts):
        raise ValueError("Extra fixture experts must be original IDs in [0,384)")
    populated = sorted(set(selected).union(extra_experts))
    records = {}
    for expert in populated:
        for projection, rows, cols in [("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]:
            p = f"layers.{layer}.ffn.experts.{expert}.{projection}"
            for suffix, dtype, shape in [("weight", "I8", [rows, cols // 2]), ("scale", "F8_E8M0", [rows, cols // 32])]:
                name = p + "." + suffix
                if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
                    raise ValueError(f"Missing or malformed selected expert tensor: {name}")
                records[name] = hashlib.sha256(checkpoint.raw(name)).hexdigest()
    writer = GGUFWriter()
    writer.add("general.architecture", "deepseek41-routed-fixture")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    writer.add("strata.deepseek41.selected_ids", selected, "array:u32")
    if extra_experts:
        writer.add("strata.deepseek41.populated_ids", populated, "array:u32")
    origin = "seeded" if input_tensor is None else "supplied"
    coverage = "selected experts plus explicit fixture coverage" if extra_experts else "selected experts"
    writer.add("strata.deepseek41.scope", f"layer {layer} text; {origin} BF16 input; ONLY {coverage} populated; others are holes")
    # Reuse the existing metadata serializer. Stream tensor payloads so the 384-expert
    # address space stays sparse; memory holds at most one packed projection at a time.
    small = [
        ("router.weight", [5120, 384], 30, router.tobytes()),
        ("router.bias", [384], 0, bias.tobytes()),
        ("input", [5120], 0, x.float().numpy().tobytes()),
        ("expected.weights", [6], 0, weights.float().numpy().tobytes()),
    ]
    part_bytes = 5120 * 2304 // 32 * 17
    tensors = [(name, shape, typ, len(data)) for name, shape, typ, data in small]
    for role, rows, cols in [("gate", 2304, 5120), ("up", 2304, 5120), ("down", 5120, 2304)]:
        tensors.append((f"blk.{layer}.ffn_{role}_exps.weight", [cols, rows, 384], 39, part_bytes * 384))
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(writer.metadata)))
    for key, (typ, value) in writer.metadata.items():
        header += writer._kv_bytes(key, typ, value)
    offsets, end = [], 0
    for name, shape, typ, size in tensors:
        offsets.append(end)
        header += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", typ, end)
        end = (end + size + 31) // 32 * 32
    header += b"\0" * (-len(header) % 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        f.write(header)
        f.truncate(len(header) + end)
        for i, (_, _, _, data) in enumerate(small):
            f.seek(len(header) + offsets[i])
            f.write(data)
        for role, projection in enumerate(("w1", "w3", "w2")):
            for expert in populated:
                p = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                payload = repack_fp4(checkpoint.array(p + ".weight", np.uint8),
                                    checkpoint.array(p + ".scale", np.uint8))
                f.seek(len(header) + offsets[4 + role] + expert * part_bytes)
                f.write(payload.tobytes())
    return {"fixture": str(output), "revision": REVISION, "selected_ids": selected,
            "expected_weights": weights[0].tolist(), "logical_bytes": output.stat().st_size,
            "disk_bytes": output.stat().st_blocks * 512, "torch_oracle_version": torch.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.output), indent=2))


if __name__ == "__main__":
    main()

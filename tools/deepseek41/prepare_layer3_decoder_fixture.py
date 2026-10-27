"""Pinned layer-3 Block oracle conditioned on actual native layer-2 output.

The --handoff file is exported by the decoder-pair test's --capture mode. Its
source input reproduces layer-2 shared KV/indices through the pinned CPU graph.
Layer-3 residuals and coefficients are actual layer-2 output, including native
Q8_1 MoE. This is an integration oracle, not two official full-FP8 decoder blocks.
No network access; --discover reports original routes before fetching experts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, _Tensor
from prepare_layer2_attention_fixture import build_reference
from prepare_mhc_fixture import split_sinkhorn
from prepare_routed_fixture import REVISION, prepare as prepare_routed


def prepare(model_dir: Path, reference: Path, kernel: Path, handoff: Path, output: Path | None,
            routed_output: Path | None, discover: bool = False) -> dict:
    if not discover and (output is None or routed_output is None):
        raise ValueError("Both output paths are required for fixture preparation")
    if not discover and (output.exists() or routed_output.exists()):
        raise FileExistsError("Fixture output already exists")
    raw_handoff = handoff.read_bytes()
    if len(raw_handoff) != 16 + 5 * (2 * 20480 + 8) * 4 or struct.unpack("<8sII", raw_handoff[:16]) != (b"DS41PAIR", 1, 5):
        raise ValueError("Expected version-1 five-token DS41PAIR handoff")
    values = torch.from_numpy(np.frombuffer(raw_handoff, "<f4", offset=16).copy())
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite native handoff")
    original, original_pre, residual, incoming = torch.split(values, [5 * 20480, 5 * 4, 5 * 20480, 5 * 4])
    if not torch.equal(original, original.bfloat16().float()) or not torch.equal(residual, residual.bfloat16().float()):
        raise ValueError("Native residual handoff must be BF16-valued")
    original, residual = original.bfloat16().reshape(1, 5, 4, 5120), residual.bfloat16().reshape(1, 5, 4, 5120)
    original_pre, incoming = original_pre.reshape(1, 5, 4), incoming.reshape(1, 5, 4)
    source_attention, attn, ns, writer, records = build_reference(model_dir, reference, kernel, shared_layer3=True)
    checkpoint = Checkpoint(model_dir)
    provenance = json.loads((model_dir / "provenance.json").read_text())

    def source(name, dtype, shape, target=None):
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source tensor {name}")
        raw = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        if target:
            writer.tensors.append(_Tensor(target, list(reversed(shape)), dtype if dtype in {"F32", "BF16"} else "I8", raw))
        if dtype == "BF16":
            return torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        if dtype == "F32":
            return torch.from_numpy(np.frombuffer(raw, np.float32).copy()).reshape(shape)
        return None

    definitions = [n for n in ast.parse(reference.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name in {"Gate", "Block"}]
    ns.update(Attention=ns["ReferenceAttention"], F=F, linear=F.linear, hc_split_sinkhorn=split_sinkhorn, EngramLayout=SimpleNamespace)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), ns)
    block = ns["Block"].__new__(ns["Block"]); nn.Module.__init__(block)
    block.norm_eps, block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps = 1e-20, 4, 20, 1e-6
    block.layer_id, block.engram, block.attn = 3, None, attn
    for kind in ("attn", "ffn"):
        for part, shape in [("fn", [24, 20480]), ("scale", [3]), ("base", [24])]:
            name = f"hc_{kind}_{part}"
            setattr(block, name, source("layers.3." + name, "F32", shape, "blk.3." + name))
        norm = ns["RMSNorm"](5120, 1e-20)
        name = f"{kind}_norm.weight"
        norm.weight.data = source("layers.3." + name, "BF16", [5120], "blk.3." + name)
        setattr(block, kind + "_norm", norm)
    for proj, role, rows, cols in [("w1", "gate", 2304, 5120), ("w3", "up", 2304, 5120), ("w2", "down", 5120, 2304)]:
        for part, dtype, shape in [("weight", "F8_E4M3", [rows, cols]), ("scale", "F8_E8M0", [rows // 32, cols // 32])]:
            source(f"layers.3.ffn.shared_experts.{proj}.{part}", dtype, shape, f"blk.3.ffn_{role}_shexp.{part}")
    args = SimpleNamespace(dim=5120, get_moe_config=lambda _: (384, 6), score_func="sqrtsoftplus", gate_temp=1.,
                           norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
    gate = ns["Gate"](3, args)
    gate.weight.data = source("layers.3.ffn.gate.weight", "BF16", [384, 5120]).float()
    gate.bias.data = source("layers.3.ffn.gate.bias", "F32", [384])
    captured, selected, normalized_inputs, case_ids, joins = {}, set(), [], [], []

    class CaptureMoE(nn.Module):
        def forward(self, x, image_mask):
            assert image_mask is None
            flat = x.reshape(-1, 5120)
            weights, ids = gate(flat)
            captured.update(ffn_input=x.clone(), route_ids=ids, route_weights=weights)
            selected.update(ids.flatten().tolist()); normalized_inputs.append(flat.clone())
            return torch.zeros_like(x)  # C++ injects the declared native MoE baseline

    block.ffn = CaptureMoE()

    def capture_post(self, x, residual, post, comb):
        result = ns["Block"].hc_post(self, x, residual, post, comb)
        joins.append((result.clone(), post.clone(), comb.clone()))
        return result

    block.hc_post = MethodType(capture_post, block)
    writer.add("general.architecture", "deepseek41-layer3-decoder-fixture")
    writer.add("strata.deepseek41.mhc_profile", "1017-hc4-delayed")
    writer.add("strata.deepseek41.shared_weight_encoding", "e4m3fn-e8m0-block32")
    writer.add("strata.deepseek41.scope", "official layer-3 Block on actual native layer-2 outputs; native Q8_1 MoE supplied by test")

    def add(name, value):
        x = value.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(x.shape)), "F32", x.numpy().tobytes()))

    source_norm = ns["RMSNorm"](5120, 1e-20)
    source_norm.weight.data = source("layers.2.attn_norm.weight", "BF16", [5120])
    writer.add("strata.deepseek41.native_handoff_sha256", hashlib.sha256(raw_handoff).hexdigest())
    with torch.inference_mode():
        for case, start, length in [(0, 0, 3), (1, 3, 1), (2, 4, 1)]:
            joins.clear()
            x, pre = residual[:, start:start + length], incoming[:, start:start + length]
            source_x, source_pre = original[:, start:start + length], original_pre[:, start:start + length]
            source_attention(source_norm(block.hc_pre(source_x, source_pre)), start)
            shared = ns["shared_attn"]
            saved_kv, saved_ids = shared.compress_kv.clone(), shared.topk_idxs.clone()
            _, next_pre = block(x, start, pre, None)
            if not torch.equal(saved_kv, shared.compress_kv) or not torch.equal(saved_ids, shared.topk_idxs):
                raise ValueError("Layer-3 CPU graph changed borrowed state")
            assert len(joins) == 2
            groups = (start + length) // 2
            for name, value in [("source_input", source_x), ("source_incoming", source_pre), ("input", x), ("incoming", pre), ("next_pre", next_pre),
                                ("after_attention", joins[0][0]), ("ffn_input", captured["ffn_input"]),
                                ("post", joins[1][1]), ("comb", joins[1][2]), ("cache", attn.window_kv_cache),
                                ("source_cache", source_attention.window_kv_cache), ("compressed", source_attention.compress_kv_cache[:, :groups]),
                                ("keys", source_attention.indexer.k_cache[:, :groups]),
                                ("route_ids", captured["route_ids"]), ("route_weights", captured["route_weights"])]:
                add(f"case.{case}.{name}", value)
            case_ids.append(captured["route_ids"].tolist())
    report = {"revision": REVISION, "populated_expert_ids": sorted(selected), "case_route_ids": case_ids,
              "scope": "pinned CPU layer-3 Block conditioned on actual native layer-2 handoff + native Q8_1 MoE baseline",
              "native_handoff_sha256": hashlib.sha256(raw_handoff).hexdigest(),
              "torch_oracle_version": torch.__version__}
    if discover:
        return report
    for expert in selected:
        for proj, rows, cols in [("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]:
            for part, dtype, shape in [("weight", "I8", [rows, cols // 2]), ("scale", "F8_E8M0", [rows, cols // 32])]:
                source(f"layers.3.ffn.experts.{expert}.{proj}.{part}", dtype, shape)
    routed = prepare_routed(model_dir, reference, routed_output, input_tensor=normalized_inputs[0][:1],
                            extra_experts=tuple(sorted(selected)), layer=3)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True); writer.write(output)
    report.update(fixture=str(output), bytes=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  verified_source_tensors=len(records), routed_fixture=routed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--routed-output", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel, args.handoff,
                             args.output, args.routed_output, args.discover), indent=2))


if __name__ == "__main__":
    main()

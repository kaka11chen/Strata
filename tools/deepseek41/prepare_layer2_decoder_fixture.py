"""Local pinned source-2/source-8/source-14/source-20 Block oracle and native Q8_1 MoE composition fixture.

--discover reports original routes without reading experts or writing fixtures.
No network access. Final preparation requires every selected original expert.
Layer 8 is conditioned on the captured actual layer-7 residual/coefficients.
Layer 14 uses the pinned standalone Engram residual fixture and FP64 linear oracle.
Torch is only used by the offline oracle, never by the native decoder.
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
from gguf_reader import GGUFFile
from prepare_layer2_attention_fixture import build_reference
from prepare_mhc_fixture import split_sinkhorn
from prepare_routed_fixture import REVISION, prepare as prepare_routed


ENGRAM14_FIXTURE_HASH = "eba0e241a1975dda778486edebd704b0fa98fc6019d02218c391c6e663523c8f"

def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path | None,
            routed_output: Path | None, discover: bool = False, *, layer: int = 2, handoff: Path | None = None,
            engram_fixture: Path | None = None) -> dict:
    if layer not in (2, 8, 14, 20) or (layer == 8) != (handoff is not None) or (layer == 14) != (engram_fixture is not None):
        raise ValueError("Layer 8 requires a layer-7 handoff, layer 14 requires its pinned Engram fixture, and layers 2/20 retain seeded inputs")
    if not discover and (output is None or routed_output is None):
        raise ValueError("Both output paths are required for fixture preparation")
    if not discover and (output.exists() or routed_output.exists()):
        raise FileExistsError("Fixture output already exists")
    if engram_fixture is not None and hashlib.sha256(engram_fixture.read_bytes()).hexdigest() != ENGRAM14_FIXTURE_HASH:
        raise ValueError("Not the pinned standalone layer-14 Engram fixture")
    attn, _, ns, writer, records = build_reference(model_dir, reference, kernel, owner_layer=layer,
                                                 linear_oracle="fp64" if layer in (14, 20) else "block32")
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
    block.layer_id, block.engram, block.attn = layer, None, attn
    for kind in ("attn", "ffn"):
        for part, shape in [("fn", [24, 20480]), ("scale", [3]), ("base", [24])]:
            name = f"hc_{kind}_{part}"
            setattr(block, name, source(f"layers.{layer}." + name, "F32", shape, f"blk.{layer}." + name))
        norm = ns["RMSNorm"](5120, 1e-20)
        name = f"{kind}_norm.weight"
        norm.weight.data = source(f"layers.{layer}." + name, "BF16", [5120], f"blk.{layer}." + name)
        setattr(block, kind + "_norm", norm)
    for proj, role, rows, cols in [("w1", "gate", 2304, 5120), ("w3", "up", 2304, 5120), ("w2", "down", 5120, 2304)]:
        for part, dtype, shape in [("weight", "F8_E4M3", [rows, cols]), ("scale", "F8_E8M0", [rows // 32, cols // 32])]:
            source(f"layers.{layer}.ffn.shared_experts.{proj}.{part}", dtype, shape, f"blk.{layer}.ffn_{role}_shexp.{part}")
    args = SimpleNamespace(dim=5120, get_moe_config=lambda _: (384, 6), score_func="sqrtsoftplus", gate_temp=1.,
                           norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
    gate = ns["Gate"](layer, args)
    gate.weight.data = source(f"layers.{layer}.ffn.gate.weight", "BF16", [384, 5120]).float()
    gate.bias.data = source(f"layers.{layer}.ffn.gate.bias", "F32", [384])
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
    writer.add("general.architecture", f"deepseek41-layer{layer}-decoder-fixture")
    writer.add("strata.deepseek41.mhc_profile", "1017-hc4-delayed")
    writer.add("strata.deepseek41.shared_weight_encoding", "e4m3fn-e8m0-block32")
    writer.add("strata.deepseek41.scope", f"official layer-{layer} Block through FFN inputs and joins; native Q8_1 MoE supplied by test")

    def add(name, value):
        x = value.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(x.shape)), "F32", x.numpy().tobytes()))

    residual = torch.randn(1, 5, 4, 5120, generator=torch.Generator().manual_seed(1017)).bfloat16()
    incoming = torch.tensor([[[1., 0., 0., 0.], [.1, .7, .2, .4], [.4, .2, .6, .8], [.9, .1, .3, .7], [.2, .4, .5, .3]]])
    after_engram, tokens = None, None
    if engram_fixture is not None:
        ef = GGUFFile(engram_fixture)
        def previous(name, dtype, count):
            t = next(t for t in ef.tensors if t.name == name)
            n = count * np.dtype(dtype).itemsize
            if t.type_id != (0 if dtype == "<f4" else 24) or t.elements * (4 if t.type_id == 0 else 1) != n:
                raise ValueError("Malformed pinned Engram fixture")
            with ef.path.open("rb") as f:
                f.seek(ef.data_start + t.offset); raw = f.read(n)
            if len(raw) != n:
                raise ValueError("Truncated pinned Engram fixture")
            return torch.from_numpy(np.frombuffer(raw, dtype).copy())
        residual = previous("input.residual", "<f4", 9*4*5120).reshape(1, 9, 4, 5120)[:, :5].bfloat16()
        after_engram = previous("expected.output", "<f4", 9*4*5120).reshape(1, 9, 4, 5120)[:, :5].bfloat16()
        tokens, live = previous("tokens", "<i4", 9)[:5], previous("live", "u1", 9)[:5]
        if not live.all():
            raise ValueError("Layer-14 decoder fixture is text-only")
        writer.add("strata.deepseek41.engram_fixture_sha256", ENGRAM14_FIXTURE_HASH)
        writer.add("fixture.linear_oracle", "fp64-linears-fp32-boundaries-v1")
        writer.add("strata.deepseek41.scope", "standalone pinned layer-14 Engram -> Block with FP64 attention linears; native Q8_1 MoE supplied by test")
    handoff_hash = None
    if handoff is not None:
        raw = handoff.read_bytes()
        if len(raw) != 16 + 5 * 4 + 5 * (20480 + 4) * 4 or struct.unpack("<8sII", raw[:16]) != (b"DS41PFX7", 1, 5):
            raise ValueError("Expected version-1 five-token native layer-7 handoff")
        if np.frombuffer(raw, "<i4", count=5, offset=16).tolist() != [0, 455, 1805, 1, 20852]:
            raise ValueError("Wrong original handoff token sequence")
        values = torch.from_numpy(np.frombuffer(raw, "<f4", offset=36).copy())
        residual, incoming = torch.split(values, [5 * 20480, 5 * 4])
        if not torch.isfinite(values).all() or not torch.equal(residual, residual.bfloat16().float()):
            raise ValueError("Native layer-7 residual must be finite BF16-valued")
        if (incoming < 0).any() or (incoming > 1.000002).any():
            raise ValueError("Invalid same-token delayed coefficients")
        residual, incoming = residual.bfloat16().reshape(1, 5, 4, 5120), incoming.reshape(1, 5, 4)
        handoff_hash = hashlib.sha256(raw).hexdigest()
        writer.add("strata.deepseek41.native_handoff_sha256", handoff_hash)
    with torch.inference_mode():
        for case, start, length in [(0, 0, 3), (1, 3, 1), (2, 4, 1)]:
            joins.clear()
            x, pre = residual[:, start:start + length], incoming[:, start:start + length]
            block_input = x if after_engram is None else after_engram[:, start:start + length]
            _, next_pre = block(block_input, start, pre, None)
            assert len(joins) == 2
            groups = (start + length) // attn.compress_ratio
            for name, value in [("input", x), ("incoming", pre), ("next_pre", next_pre),
                                ("after_attention", joins[0][0]), ("ffn_input", captured["ffn_input"]),
                                ("post", joins[1][1]), ("comb", joins[1][2]), ("cache", attn.window_kv_cache),
                                ("compressed", attn.compress_kv_cache[:, :groups]), ("keys", attn.indexer.k_cache[:, :groups]),
                                ("route_ids", captured["route_ids"]), ("route_weights", captured["route_weights"])]:
                add(f"case.{case}.{name}", value)
            if layer == 20:
                add(f"case.{case}.indices", ns["shared_attn"].topk_idxs)
                mask = ns["shared_attn"].candidates
                add(f"case.{case}.candidate_mask", mask)
            if after_engram is not None:
                add(f"case.{case}.after_engram", block_input)
                writer.tensors.append(_Tensor(f"case.{case}.tokens", [length*4], "I8", tokens[start:start + length].numpy().astype("<i4").tobytes()))
            case_ids.append(captured["route_ids"].tolist())
    report = {"revision": REVISION, "populated_expert_ids": sorted(selected), "case_route_ids": case_ids,
              "scope": f"independent CPU layer-{layer} Block/attention + native Q8_1 MoE composition baseline",
              "torch_oracle_version": torch.__version__, "layer": layer, "native_handoff_sha256": handoff_hash}
    if layer == 20:
        report.update(linear_oracle="fp64-linears-fp32-boundaries-v1")
    if layer == 14:
        report.update(engram_fixture_sha256=ENGRAM14_FIXTURE_HASH, linear_oracle="fp64-linears-fp32-boundaries-v1",
                      scope="standalone pinned original Engram -> Block with FP64 attention linears -> native Q8_1 MoE composition")
    if discover:
        return report
    for expert in selected:
        for proj, rows, cols in [("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]:
            for part, dtype, shape in [("weight", "I8", [rows, cols // 2]), ("scale", "F8_E8M0", [rows, cols // 32])]:
                source(f"layers.{layer}.ffn.experts.{expert}.{proj}.{part}", dtype, shape)
    routed = prepare_routed(model_dir, reference, routed_output, input_tensor=normalized_inputs[0][:1],
                            extra_experts=tuple(sorted(selected)), layer=layer)
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--routed-output", type=Path)
    parser.add_argument("--layer", type=int, choices=(2, 8, 14, 20), default=2)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--engram-fixture", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel,
                             args.output, args.routed_output, args.discover, layer=args.layer, handoff=args.handoff, engram_fixture=args.engram_fixture), indent=2))


if __name__ == "__main__":
    main()

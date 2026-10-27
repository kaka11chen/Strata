"""Local 1017 layer-0 decoder composition fixture; no checkpoint downloads.

Runs the pinned official Block.forward with the independent attention/Sinkhorn
oracles from steps 5b/5c. Captures the real FFN input and post/comb coefficients.
The C++ test injects the already-validated CUDA MoE's Q8_1-baseline output into
an independent CPU residual join. This is not an official full-FP8 MoE oracle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION, prepare as prepare_routed
from prepare_mhc_fixture import KERNEL_HASH, split_sinkhorn
from prepare_shared_fixture import ReferenceLinear, quantize
from prepare_attention_fixture import sparse_attention


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path, routed_output: Path) -> dict:
    if output.exists() or routed_output.exists():
        raise FileExistsError("Fixture output already exists")
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    torch.set_num_threads(4)
    checkpoint = Checkpoint(model_dir)
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
    records = {}

    def source(name, dtype, shape):
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source tensor {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        if dtype in {"F8_E4M3", "F8_E8M0"}:
            return torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(
                torch.float8_e4m3fn if dtype == "F8_E4M3" else torch.float8_e8m0fnu).reshape(shape)
        if dtype == "BF16":
            return torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        return torch.from_numpy(np.frombuffer(raw, np.float32).copy()).reshape(shape)

    tree = ast.parse(reference.read_bytes())
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    wrapper = ast.ClassDef(name="ReferenceAttention", bases=[ast.parse("nn.Module", mode="eval").body], keywords=[],
        body=[n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name in {"forward", "_window_kv"}], decorator_list=[])
    names = {"RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs", "Gate", "Block"}
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]

    def act_quant(x, block_size, *_args):
        assert block_size == 32
        x.copy_(torch.from_numpy(quantize(x)))
        return x

    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "lru_cache": lru_cache,
          "ModelArgs": SimpleNamespace, "EngramLayout": SimpleNamespace, "Attention": nn.Module,
          "linear": F.linear, "act_quant": act_quant, "fp8_block_size": 32,
          "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
          "sparse_attn": sparse_attention, "hc_split_sinkhorn": split_sinkhorn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[])), str(reference), "exec"), ns)
    block = ns["Block"].__new__(ns["Block"]); nn.Module.__init__(block)
    block.norm_eps, block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps = 1e-20, 4, 20, 1e-6
    block.layer_id, block.engram = 0, None
    attn = ns["ReferenceAttention"]()
    attn.n_local_heads, attn.head_dim, attn.rope_head_dim = 64, 512, 64
    attn.n_local_groups, attn.o_lora_rank, attn.window_size = 8, 1024, 128
    attn.compress_ratio, attn.softmax_scale = 0, 512**-.5
    attn.window_kv_cache = torch.zeros(1, 128, 512, dtype=torch.bfloat16)
    attn.freqs_cis = ns["precompute_freqs_cis"](64, 16, 0, 10000., 16., 32., 1.)
    for name, rows, cols in [("wq_a", 1280, 5120), ("wq_b", 32768, 1280), ("wkv", 512, 5120),
                             ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)]:
        prefix = "layers.0.attn." + name
        raw = source(prefix + ".weight", "F8_E4M3", [rows, cols])
        scale = source(prefix + ".scale", "F8_E8M0", [rows // 32, cols // 32])
        decoded = raw.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        if not torch.isfinite(decoded).all() or not torch.equal(decoded.bfloat16().float(), decoded):
            raise ValueError("Attention weight is not losslessly BF16 representable")
        linear = ReferenceLinear()
        if name == "wo_a":
            linear.weight = nn.Parameter(decoded.bfloat16(), requires_grad=False)
        else:
            linear.decoded = decoded.numpy()
        setattr(attn, name, linear)
    for name, width in [("q_norm", 1280), ("kv_norm", 512)]:
        norm = ns["RMSNorm"](width, 1e-20)
        norm.weight.data = source("layers.0.attn." + name + ".weight", "BF16", [width])
        setattr(attn, name, norm)
    attn.attn_sink = source("layers.0.attn.attn_sink", "F32", [64])
    block.attn = attn
    for kind in ["attn", "ffn"]:
        for suffix, shape in [("fn", [24, 20480]), ("scale", [3]), ("base", [24])]:
            name = f"hc_{kind}_{suffix}"
            setattr(block, name, source("layers.0." + name, "F32", shape))
        norm = ns["RMSNorm"](5120, 1e-20)
        norm.weight.data = source(f"layers.0.{kind}_norm.weight", "BF16", [5120])
        setattr(block, kind + "_norm", norm)
    args = SimpleNamespace(dim=5120, get_moe_config=lambda _: (384, 6), score_func="sqrtsoftplus", gate_temp=1.,
                           norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
    gate = ns["Gate"](0, args)
    gate.weight.data = source("layers.0.ffn.gate.weight", "BF16", [384, 5120]).float()
    gate.bias.data = source("layers.0.ffn.gate.bias", "F32", [384])
    captured = {}; selected = set(); normalized_inputs = []; case_ids = []

    class CaptureMoE(nn.Module):
        def forward(self, x, image_mask):
            assert image_mask is None
            flattened = x.reshape(-1, 5120)
            weights, ids = gate(flattened)
            captured["ffn_input"] = x.clone()
            captured["route_ids"], captured["route_weights"] = ids, weights
            selected.update(ids.flatten().tolist())
            normalized_inputs.append(flattened.clone())
            return torch.zeros_like(x)  # explicitly replaced by the native MoE baseline in C++

    block.ffn = CaptureMoE()
    joins = []

    def capture_post(self, x, residual, post, comb):
        result = ns["Block"].hc_post(self, x, residual, post, comb)
        joins.append((result.clone(), post.clone(), comb.clone()))
        return result

    block.hc_post = MethodType(capture_post, block)
    writer = GGUFWriter()
    writer.add("general.architecture", "deepseek41-decoder-fixture")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    writer.add("strata.deepseek41.scope", "official Block graph through FFN inputs and joins; native Q8_1 MoE supplied by test")

    def add(name, value):
        x = value.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(x.shape)), "F32", x.numpy().tobytes()))

    generator = torch.Generator().manual_seed(1017)
    residual = torch.randn(1, 5, 4, 5120, generator=generator).bfloat16()
    incoming = torch.tensor([[[1., 0., 0., 0.], [.1, .7, .2, .4], [.4, .2, .6, .8], [.9, .1, .3, .7], [.2, .4, .5, .3]]])
    with torch.inference_mode():
        for case, start, length in [(0, 0, 3), (1, 3, 1), (2, 4, 1)]:
            joins.clear()
            x, pre = residual[:, start:start + length], incoming[:, start:start + length]
            _, next_pre = block(x, start, pre, None)
            assert len(joins) == 2
            for name, value in [("input", x), ("incoming", pre), ("next_pre", next_pre),
                                ("after_attention", joins[0][0]), ("ffn_input", captured["ffn_input"]),
                                ("post", joins[1][1]), ("comb", joins[1][2]), ("cache", attn.window_kv_cache),
                                ("route_ids", captured["route_ids"]), ("route_weights", captured["route_weights"])]:
                add(f"case.{case}.{name}", value)
            case_ids.append(captured["route_ids"].tolist())
    # Missing experts are errors; preserve all routes from the actual normalized inputs.
    missing = sorted(expert for expert in selected if f"layers.0.ffn.experts.{expert}.w1.weight" not in checkpoint.meta)
    if missing:
        raise ValueError(f"Required original experts are not cached: {missing}; all selected: {sorted(selected)}; routes: {case_ids}")
    for expert in selected:
        for projection, rows, cols in [("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]:
            for suffix, dtype, shape in [("weight", "I8", [rows, cols // 2]), ("scale", "F8_E8M0", [rows, cols // 32])]:
                name = f"layers.0.ffn.experts.{expert}.{projection}.{suffix}"
                if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
                    raise ValueError(f"Required original expert is not cached: {name}")
                digest = hashlib.sha256(checkpoint.raw(name)).hexdigest()
                if provenance["tensors"][name]["sha256"] != digest:
                    raise ValueError(f"Expert source hash mismatch: {name}")
                records[name] = digest
    routed = prepare_routed(model_dir, reference, routed_output,
                            input_tensor=normalized_inputs[0][:1], extra_experts=tuple(sorted(selected)))
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "verified_source_tensors": len(records),
            "calls": "prefill 3 + decode 1 + decode 1; per-token incoming mHC coefficients",
            "case_route_ids": case_ids, "populated_expert_ids": sorted(selected), "routed_fixture": routed,
            "scope": "independent CPU Block/attention graph + native Q8_1 MoE composition baseline",
            "torch_oracle_version": torch.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routed-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel, args.output, args.routed_output), indent=2))


if __name__ == "__main__":
    main()

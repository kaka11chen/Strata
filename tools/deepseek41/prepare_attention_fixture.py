"""Pinned 1017 layer-0 attention oracle, using local weights only.

Executes official Attention.forward/_window_kv, RMSNorm, RoPE and window indices.
FP8 linears use the independent NumPy block-32 oracle; sparse attention uses a
Torch translation of the pinned kernel's 64-candidate/BF16-numerator rule.
Neither TileLang accumulation order nor full-model quality is established.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_mhc_fixture import KERNEL_HASH
from prepare_shared_fixture import ReferenceLinear, quantize


def sparse_attention(q, kv, sink, indices, scale):
    # Independent CPU implementation with explicit 64-candidate boundaries.
    result = torch.empty_like(q)
    for token in range(q.shape[1]):
        maximum = torch.full((q.shape[2],), -1e30)
        denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros(q.shape[2], q.shape[3])
        for first in range(0, indices.shape[-1], 64):
            ids = indices[0, token, first:first + 64].long()
            valid = ids >= 0
            values = kv[0, ids.clamp_min(0)].float() * valid[:, None]
            scores = (q[0, token].float() @ values.T) * scale
            scores[:, ~valid] = -torch.inf
            previous = maximum
            maximum = torch.maximum(maximum, scores.max(-1).values)
            correction = (previous - maximum).exp()
            prob = (scores - maximum[:, None]).exp()
            denominator = denominator * correction + prob.sum(-1)
            accumulator = accumulator * correction[:, None] + prob.bfloat16().float() @ values
        denominator += (sink - maximum).exp()
        result[0, token] = (accumulator / denominator[:, None]).bfloat16()
    return result


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    torch.set_num_threads(4)
    tree = ast.parse(reference.read_bytes())
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    methods = [n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name in {"forward", "_window_kv"}]
    wrapper = ast.ClassDef(name="ReferenceAttention", bases=[ast.parse("nn.Module", mode="eval").body],
                           keywords=[], body=methods, decorator_list=[])
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                   n.name in {"RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs"}]
    captured = {}

    def act_quant(x, block_size, *_args):
        assert block_size == 32
        if x.shape[-2:] == (8, 4096):
            captured["core_output"] = x.clone().reshape(1, x.shape[1], 64, 512)
        x.copy_(torch.from_numpy(quantize(x)))
        return x

    namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "lru_cache": lru_cache,
                 "act_quant": act_quant, "fp8_block_size": 32, "scale_fmt": "ue8m0",
                 "scale_dtype": torch.float8_e8m0fnu, "sparse_attn": sparse_attention}
    module = ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[]))
    exec(compile(module, str(reference), "exec"), namespace)
    attn = namespace["ReferenceAttention"]()
    attn.n_local_heads, attn.head_dim, attn.rope_head_dim = 64, 512, 64
    attn.n_local_groups, attn.o_lora_rank, attn.window_size = 8, 1024, 128
    attn.compress_ratio, attn.softmax_scale = 0, 512**-.5
    attn.window_kv_cache = torch.zeros(1, 128, 512, dtype=torch.bfloat16)
    attn.freqs_cis = namespace["precompute_freqs_cis"](64, 16, 0, 10000., 16., 32., 1.)
    checkpoint = Checkpoint(model_dir)
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
    writer = GGUFWriter(); TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-attention-fixture")
    writer.add("strata.deepseek41.attention_profile", "1017-layer0-swa128-fp8-block32")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    records = {}

    def source(suffix, dtype, shape):
        name = "layers.0.attn." + suffix
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed attention tensor {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        return raw

    def add(name, tensor):
        x = tensor.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(x.shape)), "F32", x.numpy().tobytes()))

    for name, rows, cols in [("wq_a", 1280, 5120), ("wq_b", 32768, 1280), ("wkv", 512, 5120),
                             ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)]:
        raw = source(name + ".weight", "F8_E4M3", [rows, cols])
        scales = source(name + ".scale", "F8_E8M0", [rows // 32, cols // 32])
        writer.tensors.append(_Tensor(f"blk.0.attn_{name}.weight", [cols, rows], "I8", raw))
        writer.tensors.append(_Tensor(f"blk.0.attn_{name}.scale", [cols // 32, rows // 32], "I8", scales))
        codes = torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(torch.float8_e4m3fn).reshape(rows, cols)
        scale = torch.from_numpy(np.frombuffer(scales, np.uint8).copy()).view(torch.float8_e8m0fnu).reshape(rows // 32, cols // 32)
        decoded = codes.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        if not torch.isfinite(decoded).all() or not torch.equal(decoded.bfloat16().float(), decoded):
            raise ValueError("Attention weights do not expand losslessly to BF16")
        linear = ReferenceLinear()
        if name == "wo_a":
            linear.weight = nn.Parameter(decoded.bfloat16(), requires_grad=False)
        else:
            linear.decoded = decoded.numpy()
        setattr(attn, name, linear)
    for name, width in [("q_norm", 1280), ("kv_norm", 512)]:
        raw = source(name + ".weight", "BF16", [width])
        writer.tensors.append(_Tensor(f"blk.0.attn_{name}.weight", [width], "BF16", raw))
        norm = namespace["RMSNorm"](width, 1e-20)
        norm.weight.data = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16)
        setattr(attn, name, norm)
    raw = source("attn_sink", "F32", [64])
    attn.attn_sink = torch.from_numpy(np.frombuffer(raw, np.float32).copy())
    writer.tensors.append(_Tensor("blk.0.attn_sink", [64], "F32", raw))
    attn.wq_b.register_forward_hook(lambda _m, _i, out: captured.update(query=out.clone()))
    attn.kv_norm.register_forward_hook(lambda _m, _i, out: captured.update(kv=out.clone()))
    generator = torch.Generator().manual_seed(1017)
    base = torch.randn(1, 7, 5120, generator=generator).bfloat16()
    with torch.inference_mode():
        for case, start, length in [(0, 0, 5), (1, 5, 1), (2, 6, 1)]:
            x = base[:, start:start + length]
            y = attn(x, start)
            for key, value in [("input", x), ("output", y), ("query", captured["query"]), ("kv", captured["kv"]),
                               ("core", captured["core_output"]), ("cache", attn.window_kv_cache)]:
                add(f"case.{case}.{key}", value)
        attn.window_kv_cache.zero_()
        zero = torch.zeros(1, 1, 5120, dtype=torch.bfloat16)
        add("zero.output", attn(zero, 0))
    # Explicit wide-activation boundary for the reused FP8 quantizer.
    wide = torch.randn(32768, generator=generator).bfloat16()
    add("wide.input", wide)
    expected = torch.from_numpy(quantize(wide)).bfloat16().view(torch.uint16)
    writer.tensors.append(_Tensor("wide.quantized", [32768], "BF16", expected.numpy().tobytes()))
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "calls": "prefill 5 + decode 1 + decode 1; reset + zero token", "torch_oracle_version": torch.__version__,
            "scope": "official layer-0 Attention graph; independent CPU linear/sparse-attention arithmetic"}


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

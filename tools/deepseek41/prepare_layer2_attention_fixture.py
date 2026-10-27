"""Pinned 1017 owner attention oracle; source 2 optionally includes consumer 3.

Sources 2/8/14 use ratio 2; source 20 uses ratio 1 and publishes candidates. Engram and decoders are outside this oracle.

Official graph and independent CPU FP8/FP4/sparse-attention arithmetic, as in
steps 5c and 6a-6e. --linear-oracle fp64 accumulates all Linear dots in FP64,
then rounds to FP32 and the original output dtype; input quantization is kept.
Grouped BF16 wo_a and non-linear operations stay unchanged. Neither mode
establishes TileLang bit parity or full-model quality.
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
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_mhc_fixture import KERNEL_HASH
from prepare_shared_fixture import LINEAR_ORACLES, FP64Linear, ReferenceLinear, quantize
from prepare_attention_fixture import sparse_attention
from prepare_compressed_kv_fixture import fp4_roundtrip as kv_roundtrip
from prepare_index_key_fixture import fp4_roundtrip as key_roundtrip
from prepare_source20_fixture import pack_candidates


def build_reference(model_dir: Path, reference: Path, kernel: Path, shared_layer3: bool = False, owner_layer: int = 2, linear_oracle: str = "block32"):
    """Load the pinned graph and weight records for attention or decoder oracles."""
    if linear_oracle not in LINEAR_ORACLES:
        raise ValueError(f"Unsupported linear oracle: {linear_oracle}")
    if owner_layer not in (2, 8, 14, 20) or (shared_layer3 and owner_layer != 2):
        raise ValueError("Supported owners are 2/8/14/20; the existing shared consumer belongs to source 2")
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    ratio = 1 if owner_layer == 20 else 2
    capacity = 131 if owner_layer == 20 else 66
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint revision")
    torch.set_num_threads(4)
    checkpoint, writer = Checkpoint(model_dir), GGUFWriter()
    TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-shared-attention-fixture" if shared_layer3 else f"deepseek41-layer{owner_layer}-attention-fixture")
    profiles = [("attention", "1017-layer3-shared2-fp8-block32")] if shared_layer3 else [("attention", f"1017-layer{owner_layer}-ratio{ratio}-fp8-block32"),
                          ("compressor", f"1017-layer{owner_layer}-ratio{ratio}-bf16"),
                          ("index_key", f"1017-layer{owner_layer}-index-key128-bf16"),
                          ("index_query", f"1017-layer{owner_layer}-index32x128-fp8-block32")]
    for name, profile in profiles:
        writer.add("strata.deepseek41." + name + "_profile", profile)
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    if linear_oracle == "fp64":
        writer.add("fixture.linear_oracle", "fp64-linears-fp32-boundaries-v1")
    records = {}

    def raw(part, dtype, shape, layer=owner_layer):
        name = f"layers.{layer}.attn." + part
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source {name}")
        data = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(data).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        records[name] = digest
        return data

    def bf_weight(part, shape, layer=owner_layer):
        data = raw(part + ".weight", "BF16", shape, layer)
        if not shared_layer3 or layer == 3:
            writer.tensors.append(_Tensor(f"blk.{layer}.attn_" + part.replace(".", "_") + ".weight", list(reversed(shape)), "BF16", data))
        return torch.from_numpy(np.frombuffer(data, np.uint16).copy()).view(torch.bfloat16).reshape(shape)

    def fp8_linear(part, rows, cols, grouped=False, layer=owner_layer):
        decoded = []
        for suffix, dtype, shape, td in [("weight", "F8_E4M3", [rows, cols], torch.float8_e4m3fn),
                                         ("scale", "F8_E8M0", [rows // 32, cols // 32], torch.float8_e8m0fnu)]:
            data = raw(part + "." + suffix, dtype, shape, layer)
            if not shared_layer3 or layer == 3:
                writer.tensors.append(_Tensor(f"blk.{layer}.attn_" + part.replace(".", "_") + "." + suffix, list(reversed(shape)), "I8", data))
            decoded.append(torch.from_numpy(np.frombuffer(data, np.uint8).copy()).view(td).reshape(shape).float())
        values = decoded[0] * decoded[1].repeat_interleave(32, 0).repeat_interleave(32, 1)
        if not torch.isfinite(values).all() or not torch.equal(values, values.bfloat16().float()):
            raise ValueError("Weights do not expand losslessly to BF16")
        linear = ReferenceLinear(linear_oracle=linear_oracle)
        if grouped:
            linear.weight = nn.Parameter(values.bfloat16(), requires_grad=False)
        else:
            linear.decoded = values.numpy()
        return linear

    def add(name, value, dtype="BF16"):
        value = value.detach().contiguous()
        data = value.bfloat16().view(torch.uint16).numpy().tobytes() if dtype == "BF16" else value.float().numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), dtype, data))

    tree = ast.parse(reference.read_bytes())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and
                   n.name in {"Compressor", "RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs", "select_candidate_blocks"}]
    wrappers = []
    for source, target, methods in [("Attention", "ReferenceAttention", {"forward", "_window_kv", "_compress_kv", "_compress_topk_idxs"}),
                                     ("Indexer", "ReferenceIndexer", {"forward"})]:
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source)
        body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        wrappers.append(ast.ClassDef(name=target, bases=[ast.parse("nn.Module", mode="eval").body], keywords=[], decorator_list=[], body=body))

    def act_quant(x, block_size, *_args):
        assert block_size == 32
        x.copy_(torch.from_numpy(quantize(x)))
        return x

    def fp4(x, block_size, *args, **kwargs):
        assert block_size in (16, 32)
        return (kv_roundtrip if block_size == 16 else key_roundtrip)(x, block_size, *args, **kwargs)

    dense_linear = FP64Linear if linear_oracle == "fp64" else nn.Linear
    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: dense_linear(a, b, bias=False, **kw), "world_size": 1,
          "shared_attn": SimpleNamespace(), "act_quant": act_quant, "fp4_act_quant": fp4,
          "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0",
          "scale_dtype": torch.float8_e8m0fnu, "sparse_attn": sparse_attention}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions + wrappers, type_ignores=[])), str(reference), "exec"), ns)
    a = ns["ReferenceAttention"]()
    a.n_local_heads, a.head_dim, a.rope_head_dim, a.window_size = 64, 512, 64, 128
    a.n_local_groups, a.o_lora_rank = 8, 1024
    a.compress_ratio, a.softmax_scale = ratio, 512**-.5
    a.is_kv_source = a.is_index_source = True
    a.window_kv_cache = torch.zeros(1, 128, 512, dtype=torch.bfloat16)
    a.compress_kv_cache = torch.zeros(1, capacity, 512, dtype=torch.bfloat16)
    a.freqs_cis = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    a.attn_sink = torch.from_numpy(np.frombuffer(raw("attn_sink", "F32", [64]), np.float32).copy())
    if not shared_layer3:
        add(f"blk.{owner_layer}.attn_sink", a.attn_sink, "F32")
    for name, rows, cols in [("wq_a", 1280, 5120), ("wq_b", 32768, 1280), ("wkv", 512, 5120),
                             ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)]:
        setattr(a, name, fp8_linear(name, rows, cols, name == "wo_a"))
    for name, width in [("q_norm", 1280), ("kv_norm", 512)]:
        norm = ns["RMSNorm"](width, 1e-20); norm.weight.data = bf_weight(name, [width]); setattr(a, name, norm)
    c = ns["Compressor"](SimpleNamespace(compress_ratios=[0] * owner_layer + [ratio], dim=5120,
                        head_dim=512, norm_eps=1e-20, max_batch_size=1), owner_layer)
    a.compressor = c
    compressor_parts = [("wkv", [512, 5120]), ("norm", [512])]
    if ratio > 1:
        compressor_parts.insert(1, ("wgate", [512, 5120]))
    for part, shape in compressor_parts:
        value = bf_weight("compressor." + part, shape)
        getattr(c, part).weight.data = value if part == "norm" or ratio == 1 else value.float()
    index = ns["ReferenceIndexer"](); a.indexer = index
    index.owns_k, index.compress_ratio, index.rope_head_dim = True, ratio, 64
    index.n_heads, index.n_local_heads, index.index_head_dim, index.index_topk = 32, 32, 128, 512
    index.softmax_scale, index.freqs_cis = 128**-.5, a.freqs_cis
    index.is_candidate_source = owner_layer == 20
    index.uses_candidates = False
    index.candidate_topk_blocks, index.candidate_block_size = 2048, 8
    index.k_cache = torch.zeros(1, capacity, 128, dtype=torch.bfloat16)
    for attr, shape in [("wk", [128, 512]), ("weights_proj", [32, 5120])]:
        linear = dense_linear(shape[1], shape[0], bias=False, dtype=torch.bfloat16)
        linear.weight.data = bf_weight("indexer." + attr, shape); setattr(index, attr, linear)
    index.k_norm = ns["RMSNorm"](128, 1e-20)
    index.k_norm.weight.data = bf_weight("indexer.k_norm", [128])
    index.wq_b = fp8_linear("indexer.wq_b", 4096, 1280)
    if shared_layer3:
        consumer = ns["ReferenceAttention"]()
        for attr in ["n_local_heads", "head_dim", "rope_head_dim", "window_size", "n_local_groups", "o_lora_rank",
                     "compress_ratio", "softmax_scale", "freqs_cis"]:
            setattr(consumer, attr, getattr(a, attr))
        consumer.is_kv_source = consumer.is_index_source = False
        consumer.window_kv_cache = torch.zeros_like(a.window_kv_cache)
        consumer.attn_sink = torch.from_numpy(np.frombuffer(raw("attn_sink", "F32", [64], 3), np.float32).copy())
        add("blk.3.attn_sink", consumer.attn_sink, "F32")
        for name, rows, cols in [("wq_a", 1280, 5120), ("wq_b", 32768, 1280), ("wkv", 512, 5120),
                                 ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)]:
            setattr(consumer, name, fp8_linear(name, rows, cols, name == "wo_a", 3))
        for name, width in [("q_norm", 1280), ("kv_norm", 512)]:
            norm = ns["RMSNorm"](width, 1e-20); norm.weight.data = bf_weight(name, [width], 3); setattr(consumer, name, norm)
    return a, consumer if shared_layer3 else None, ns, writer, records


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path, shared_layer3: bool = False, owner_layer: int = 2, linear_oracle: str = "block32") -> dict:
    if output.exists():
        raise FileExistsError(output)
    a, consumer, ns, writer, records = build_reference(model_dir, reference, kernel, shared_layer3, owner_layer, linear_oracle)
    c, index = a.compressor, a.indexer

    def add(name, value, dtype="BF16"):
        value = value.detach().contiguous()
        data = value.bfloat16().view(torch.uint16).numpy().tobytes() if dtype == "BF16" else value.float().numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), dtype, data))

    generator = torch.Generator().manual_seed(671017)
    base = torch.randn(1, 131, 5120, generator=generator).bfloat16()
    add("sequence.input", base)
    if shared_layer3:
        # Independent hidden inputs: this fixture validates attention sharing,
        # not the intervening mHC/MoE/residual decoder graph.
        consumer_input = torch.randn(1, 131, 5120, generator=generator).bfloat16()
        add("consumer.input", consumer_input)
    plans = [(1, 9), (6, 9), (129, 131)] if shared_layer3 else [(1, 15), (5, 15), (6, 15), (13, 15), (129, 131)]
    writer.add("fixture.plans", len(plans), "u32")
    with torch.inference_mode():
        for plan, (first, end) in enumerate(plans):
            a.window_kv_cache.zero_(); a.compress_kv_cache.zero_(); index.k_cache.zero_()
            if a.compress_ratio > 1:
                c.kv_state.zero_(); c.score_state.fill_(-torch.inf)
            if shared_layer3:
                consumer.window_kv_cache.zero_()
            writer.add(f"plan.{plan}.prefill", first, "u32")
            writer.add(f"plan.{plan}.tokens", end, "u32")
            pos = 0
            while pos < end:
                tokens = first if pos == 0 else 1
                result = a(base[:, pos:pos + tokens], pos)
                prefix = f"plan.{plan}.at.{pos}."
                add(prefix + "output", result)
                add(prefix + "ring", a.window_kv_cache)
                groups = (pos + tokens) // a.compress_ratio
                if groups:
                    add(prefix + "compressed", a.compress_kv_cache[:, :groups])
                    add(prefix + "keys", index.k_cache[:, :groups])
                if owner_layer == 20:
                    shared = ns["shared_attn"]
                    add(prefix + "shared_indices", shared.topk_idxs, "F32")
                    packed = np.stack([pack_candidates(row) for row in shared.candidates.reshape(-1, groups)])
                    writer.tensors.append(_Tensor(prefix + "candidates", [packed.shape[1] * 4, tokens], "I8", packed.tobytes()))
                if shared_layer3:
                    shared = ns["shared_attn"]
                    saved_kv, saved_indices = shared.compress_kv.clone(), shared.topk_idxs.clone()
                    out3 = consumer(consumer_input[:, pos:pos + tokens], pos)
                    if not torch.equal(saved_kv, shared.compress_kv) or not torch.equal(saved_indices, shared.topk_idxs):
                        raise ValueError("Official consumer changed source-owned cache/indices")
                    add(prefix + "consumer_output", out3)
                    add(prefix + "consumer_ring", consumer.window_kv_cache)
                    if groups:
                        add(prefix + "shared_indices", shared.topk_idxs, "F32")
                pos += tokens
            print(f"reference plan={plan} prefill={first} end={end} complete", flush=True)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True); writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records), "linear_oracle": linear_oracle,
            "plans": plans, "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__, "scope": "official layer-2/source and layer-3/consumer attention; independent CPU arithmetic" if shared_layer3 else
            f"official full layer-{owner_layer} Attention graph; independent CPU arithmetic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-layer3", action="store_true", help="Write layer-3 weights and two-layer reference; use the existing layer-2 fixture for source weights")
    parser.add_argument("--layer", type=int, choices=(2, 8, 14, 20), default=2)
    parser.add_argument("--linear-oracle", choices=LINEAR_ORACLES, default="block32",
                        help="Linear accumulation: legacy arithmetic or independent FP64 dots with original FP32/output dtype boundaries")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel, args.output, args.shared_layer3, args.layer, args.linear_oracle), indent=2))


if __name__ == "__main__":
    main()

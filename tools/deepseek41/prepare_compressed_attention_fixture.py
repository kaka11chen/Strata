"""Pinned 1017 layer-2 attention core oracle, with supplied projected Q/window KV.

Executes official Attention.forward up to inverse RoPE, _window_kv, and (for the
integration sequence) _compress_kv/Indexer/Compressor with original weights.
Independent CPU FP8/FP4 and 64-candidate sparse attention stand in for GPU kernels.
Q/output projections and full-model quality are outside this component fixture.
"""
from __future__ import annotations

import argparse
import ast
import copy
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
from prepare_shared_fixture import ReferenceLinear, quantize
from prepare_attention_fixture import sparse_attention
from prepare_compressed_kv_fixture import fp4_roundtrip as kv_roundtrip
from prepare_index_key_fixture import fp4_roundtrip as key_roundtrip


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint revision")
    torch.set_num_threads(4)
    checkpoint, writer = Checkpoint(model_dir), GGUFWriter()
    TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-compressed-attention-fixture")
    for name, profile in [("compressor", "1017-layer2-ratio2-bf16"),
                          ("index_key", "1017-layer2-index-key128-bf16"),
                          ("index_query", "1017-layer2-index32x128-fp8-block32")]:
        writer.add("strata.deepseek41." + name + "_profile", profile)
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    records = {}

    def raw(part, dtype, shape):
        name = "layers.2.attn." + part
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source {name}")
        data = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(data).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        records[name] = digest
        return data

    def bf_weight(part, shape):
        data = raw(part + ".weight", "BF16", shape)
        writer.tensors.append(_Tensor("blk.2.attn_" + part.replace(".", "_") + ".weight", list(reversed(shape)), "BF16", data))
        return torch.from_numpy(np.frombuffer(data, np.uint16).copy()).view(torch.bfloat16).reshape(shape)

    def add(name, value, dtype="BF16"):
        value = value.detach().contiguous()
        data = value.bfloat16().view(torch.uint16).numpy().tobytes() if dtype == "BF16" else value.float().numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), dtype, data))

    tree = ast.parse(reference.read_bytes())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and
                   n.name in {"Compressor", "RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs"}]
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    methods = copy.deepcopy([n for n in attention.body if isinstance(n, ast.FunctionDef) and
                             n.name in {"forward", "_window_kv", "_compress_kv", "_compress_topk_idxs"}])
    forward = next(n for n in methods if n.name == "forward")
    cut = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and
               ast.unparse(n).startswith("o = o.view("))
    forward.body = forward.body[:cut] + ast.parse("return o").body
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    index_forward = copy.deepcopy(next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    wrappers = [ast.ClassDef(name=name, bases=[ast.parse("nn.Module", mode="eval").body], keywords=[], decorator_list=[], body=body)
                for name, body in [("ReferenceAttention", methods), ("ReferenceIndexer", [index_forward])]]

    def act_quant(x, block_size, *_args):
        assert block_size == 32
        x.copy_(torch.from_numpy(quantize(x)))
        return x

    def fp4(x, block_size, *args, **kwargs):
        return (kv_roundtrip if block_size == 16 else key_roundtrip)(x, block_size, *args, **kwargs)

    ns = {"torch": torch, "nn": nn, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw), "world_size": 1,
          "shared_attn": SimpleNamespace(), "act_quant": act_quant, "fp4_act_quant": fp4,
          "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0",
          "scale_dtype": torch.float8_e8m0fnu, "sparse_attn": sparse_attention}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions + wrappers, type_ignores=[])), str(reference), "exec"), ns)
    frequencies = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    rotate = ns["apply_rotary_emb"]
    sink = torch.from_numpy(np.frombuffer(raw("attn_sink", "F32", [64]), np.float32).copy())
    add("blk.2.attn_sink", sink, "F32")
    state = {}

    def attention_core():
        a = ns["ReferenceAttention"]()
        a.n_local_heads, a.head_dim, a.rope_head_dim, a.window_size = 64, 512, 64, 128
        a.compress_ratio, a.softmax_scale = 2, 512**-.5
        a.window_kv_cache = torch.zeros(1, 128, 512, dtype=torch.bfloat16)
        a.freqs_cis, a.attn_sink = frequencies, sink
        # These are the public core's input boundaries, deliberately skipping Q/KV
        # projection and norm computation; the official remaining graph is intact.
        a.wq_a = lambda _x: state.get("qr")
        a.q_norm = lambda x: x
        a.wq_b = lambda _qr: state["q"].clone().flatten(2)
        a.wkv = lambda _x: state["kv"].clone()
        a.kv_norm = lambda x: x
        return a

    generator = torch.Generator().manual_seed(661017)
    q = (torch.randn(1, 131, 64, 512, generator=generator) * .7).bfloat16()
    kv = torch.randn(1, 131, 512, generator=generator).bfloat16()
    compressed = torch.randn(1, 65, 512, generator=generator).bfloat16()
    rotate(compressed[..., -64:], frequencies[:130:2]); kv_roundtrip(compressed)
    add("sequence.q", q); add("sequence.kv", kv); add("sequence.compressed", compressed)
    plans = [1, 13, 43, 44, 65, 129]
    writer.add("fixture.plans", len(plans), "u32")
    with torch.inference_mode():
        for plan, first in enumerate(plans):
            a = attention_core()
            a._compress_kv = lambda _x, _qr, _start, _offset: (state["compressed"], state["indices"])
            writer.add(f"plan.{plan}.prefill", first, "u32")
            pos = 0
            while pos < 131:
                tokens = first if pos == 0 else 1
                groups, offset = (pos + tokens) // 2, tokens if pos == 0 else 128
                state.update(q=q[:, pos:pos + tokens], kv=kv[:, pos:pos + tokens], compressed=compressed[:, :groups])
                indices = torch.full((1, tokens, groups), -1, dtype=torch.int32)
                for t in range(tokens):
                    selected = torch.arange((pos + t + 1) // 2)
                    if plan % 2:  # caller-supplied sparse subset, including gaps and padded tail
                        selected = selected[::2]
                    indices[0, t, :len(selected)] = selected + offset
                state["indices"] = indices
                prefix = f"plan.{plan}.at.{pos}."
                if groups:
                    add(prefix + "indices", indices, "F32")
                # Execute the whole official window path on every call. Sparse
                # attention itself only needs evaluating the sampled decode rows.
                sample = pos == 0 or pos in {first, 63, 64, 127, 128, 129, 130}
                if sample:
                    result = a(torch.empty(1, tokens, 5120), pos)
                    add(prefix + "output", result)
                else:
                    a._window_kv(torch.empty(1, 1, 5120), frequencies[pos:pos + 1], pos)
                pos += tokens
            add(f"plan.{plan}.ring", a.window_kv_cache)
        # Standalone shared RoPE checks at training, extended and last positions.
        positions = [0, 1, 65535, 65536, 131071, 1048575]
        writer.add("fixture.rope_cases", len(positions), "u32")
        add("rope.input", q[:, :1])
        for case, pos in enumerate(positions):
            writer.add(f"rope.{case}.position", pos, "u32")
            for inverse in [False, True]:
                value = q[:, :1].clone(); rotate(value[..., -64:], frequencies[pos:pos + 1], inverse)
                add(f"rope.{case}.{'inverse' if inverse else 'forward'}", value)

        # Complete 6a->6b/6c->6d->6e wiring, using real compressor/indexer/sink weights.
        a = attention_core(); a.is_kv_source = a.is_index_source = True
        a.compress_kv_cache = torch.zeros(1, 8, 512, dtype=torch.bfloat16)
        compressor = ns["Compressor"](SimpleNamespace(compress_ratios=[0, 0, 2], dim=5120,
                                     head_dim=512, norm_eps=1e-20, max_batch_size=1), 2)
        for part, shape in [("wkv", [512, 5120]), ("wgate", [512, 5120]), ("norm", [512])]:
            value = bf_weight("compressor." + part, shape)
            getattr(compressor, part).weight.data = value if part == "norm" else value.float()
        compressor.register_forward_hook(lambda _m, _args, result: state.update(latent=None if result is None else result.clone()))
        a.compressor = compressor
        index = ns["ReferenceIndexer"](); a.indexer = index
        index.owns_k, index.compress_ratio, index.rope_head_dim = True, 2, 64
        index.n_heads, index.n_local_heads, index.index_head_dim, index.index_topk = 32, 32, 128, 512
        index.softmax_scale, index.freqs_cis = 128**-.5, frequencies
        index.is_candidate_source = index.uses_candidates = False
        index.k_cache = torch.zeros(1, 8, 128, dtype=torch.bfloat16)
        for attr, shape in [("wk", [128, 512]), ("weights_proj", [32, 5120])]:
            linear = nn.Linear(shape[1], shape[0], bias=False, dtype=torch.bfloat16)
            linear.weight.data = bf_weight("indexer." + attr, shape); setattr(index, attr, linear)
        index.k_norm = ns["RMSNorm"](128, 1e-20)
        index.k_norm.weight.data = bf_weight("indexer.k_norm", [128])
        parts = []
        for suffix, dtype, shape, td in [("weight", "F8_E4M3", [4096, 1280], torch.float8_e4m3fn),
                                         ("scale", "F8_E8M0", [128, 40], torch.float8_e8m0fnu)]:
            data = raw("indexer.wq_b." + suffix, dtype, shape)
            writer.tensors.append(_Tensor("blk.2.attn_indexer_wq_b." + suffix, list(reversed(shape)), "I8", data))
            parts.append(torch.from_numpy(np.frombuffer(data, np.uint8).copy()).view(td).reshape(shape).float())
        decoded = parts[0] * parts[1].repeat_interleave(32, 0).repeat_interleave(32, 1)
        if not torch.isfinite(decoded).all() or not torch.equal(decoded, decoded.bfloat16().float()):
            raise ValueError("Query weights do not expand losslessly to BF16")
        index.wq_b = ReferenceLinear(); index.wq_b.decoded = decoded.numpy()
        index.register_forward_hook(lambda _m, _args, result: state.update(indices=result.clone()))
        x = torch.randn(1, 15, 5120, generator=generator).bfloat16()
        qr = torch.randn(1, 15, 1280, generator=generator).bfloat16()
        add("integration.x", x); add("integration.qr", qr)
        for pos, tokens in [(0, 13), (13, 1), (14, 1)]:
            state.update(q=q[:, pos:pos + tokens], kv=kv[:, pos:pos + tokens], qr=qr[:, pos:pos + tokens])
            result = a(x[:, pos:pos + tokens], pos)
            prefix = f"integration.at.{pos}."
            add(prefix + "output", result)
            add(prefix + "indices", state["indices"], "F32")
            if state["latent"] is not None:
                add(prefix + "latent", state["latent"])
            groups = (pos + tokens) // 2
            add(prefix + "compressed", a.compress_kv_cache[:, :groups])
            add(prefix + "keys", index.k_cache[:, :groups])
            add(prefix + "ring", a.window_kv_cache)
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True); writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "sequence_plans": len(plans), "rope_cases": len(positions), "integration_tokens": 15,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__, "scope": "official CPU attention core with projected inputs; real 6a-6d components"}


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

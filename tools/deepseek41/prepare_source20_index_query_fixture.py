"""Pinned source-20 Indexer.forward oracle for query, causal selection and candidates.

Local original weights only. Official CPU graph, independent block-32 FP8 and
FP4 rounding; no TileLang/GPU or full-model parity claim. Tied top-k is recorded
as returned by Torch, and independently validated by cutoff membership in C++.
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
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_mhc_fixture import KERNEL_HASH
from prepare_shared_fixture import ReferenceLinear
from prepare_index_key_fixture import fp4_roundtrip
from prepare_source20_fixture import pack_candidates


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint revision")
    config = json.loads((model_dir / "config.json").read_text())["text_config"]
    if (config["compress_ratios"][20], config["candidate_source_layer_id"],
        config["candidate_topk_blocks"], config["candidate_block_size"]) != (1, 20, 2048, 8):
        raise ValueError("Wrong source-20 configuration")
    torch.set_num_threads(4)
    checkpoint, writer = Checkpoint(model_dir), GGUFWriter()
    TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-source20-index-query-fixture")
    writer.add("strata.deepseek41.index_query_profile", "1017-layer20-index32x128-fp8-block32")
    writer.add("strata.deepseek41.index_key_profile", "1017-layer20-index-key128-bf16")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    records = {}

    def raw(part, dtype, shape):
        name = "layers.20.attn." + part
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed source {name}")
        value = checkpoint.raw(name).tobytes(); digest = hashlib.sha256(value).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        records[name] = digest
        return value

    def bf_weight(part, shape, export=False):
        data = raw(part + ".weight", "BF16", shape)
        if export:
            writer.tensors.append(_Tensor("blk.20.attn_" + part.replace(".", "_") + ".weight", list(reversed(shape)), "BF16", data))
        return torch.from_numpy(np.frombuffer(data, np.uint16).copy()).view(torch.bfloat16).reshape(shape)

    def add(name, value, dtype="F32"):
        data = value.detach().contiguous()
        if dtype == "BF16":
            encoded = data.bfloat16().view(torch.uint16).numpy().tobytes()
        else:
            encoded = data.float().numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(data.shape)), dtype, encoded))

    tree = ast.parse(reference.read_bytes())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and
                   n.name in {"Compressor", "RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "select_candidate_blocks"}]
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    forward = copy.deepcopy(next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    key_forward = copy.deepcopy(forward)
    cut = next(i for i, n in enumerate(key_forward.body) if isinstance(n, ast.Assign) and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q")
    key_forward.body = key_forward.body[:cut]
    instrumented = []
    for node in forward.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.is_candidate_source":
            instrumented += ast.parse("capture('scores', index_score)").body
        instrumented.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "weights":
            instrumented += ast.parse("capture('weights', weights)").body
    forward.body = instrumented
    wrappers = [ast.ClassDef(name=name, bases=[], keywords=[], decorator_list=[], body=[method])
                for name, method in [("ReferenceIndexer", forward), ("ReferenceKey", key_forward)]]
    captured, shared = {}, SimpleNamespace()

    def capture(name, value):
        captured[name] = value.clone()

    def quant(x, *args):
        result = fp4_roundtrip(x, *args)
        if x.ndim == 4:
            capture("query", x)
        return result

    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "lru_cache": lru_cache, "ModelArgs": SimpleNamespace,
          "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw), "world_size": 1,
          "shared_attn": shared, "fp4_act_quant": quant, "fp4_block_size": 32, "capture": capture}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions + wrappers, type_ignores=[])), str(reference), "exec"), ns)
    index = ns["ReferenceIndexer"]()
    index.owns_k, index.compress_ratio, index.rope_head_dim = True, 1, 64
    index.n_heads, index.n_local_heads, index.index_head_dim, index.index_topk = 32, 32, 128, 512
    index.softmax_scale = 128**-.5
    index.is_candidate_source, index.uses_candidates = True, False
    index.candidate_topk_blocks, index.candidate_block_size = 2048, 8
    index.freqs_cis = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    index.k_cache = torch.zeros(1, 640, 128, dtype=torch.bfloat16)
    index.wk = nn.Linear(512, 128, bias=False, dtype=torch.bfloat16)
    index.wk.weight.data = bf_weight("indexer.wk", [128, 512], True)
    index.k_norm = ns["RMSNorm"](128, 1e-20)
    index.k_norm.weight.data = bf_weight("indexer.k_norm", [128], True)
    index.weights_proj = nn.Linear(5120, 32, bias=False, dtype=torch.bfloat16)
    index.weights_proj.weight.data = bf_weight("indexer.weights_proj", [32, 5120], True)
    index.wq_b = ReferenceLinear()
    parts = []
    for suffix, dtype, shape, torch_dtype in [("weight", "F8_E4M3", [4096, 1280], torch.float8_e4m3fn),
                                             ("scale", "F8_E8M0", [128, 40], torch.float8_e8m0fnu)]:
        data = raw("indexer.wq_b." + suffix, dtype, shape)
        writer.tensors.append(_Tensor("blk.20.attn_indexer_wq_b." + suffix, list(reversed(shape)), "I8", data))
        parts.append(torch.from_numpy(np.frombuffer(data, np.uint8).copy()).view(torch_dtype).reshape(shape).float())
    decoded = parts[0] * parts[1].repeat_interleave(32, 0).repeat_interleave(32, 1)
    if not torch.isfinite(decoded).all() or not torch.equal(decoded, decoded.bfloat16().float()):
        raise ValueError("Query projection does not expand losslessly to BF16")
    index.wq_b.decoded = decoded.numpy()
    index.wq_b.register_forward_hook(lambda _m, _x, result: capture("projected", result))
    compressor = ns["Compressor"](SimpleNamespace(compress_ratios=config["compress_ratios"], dim=5120,
                                  head_dim=512, norm_eps=1e-20, max_batch_size=1), 20)
    for part, shape in [("wkv", [512, 5120]), ("norm", [512])]:
        value = bf_weight("compressor." + part, shape)
        getattr(compressor, part).weight.data = value
    generator = torch.Generator().manual_seed(62017)

    def record(prefix, x, qr, indices):
        add(prefix + "x", x); add(prefix + "qr", qr)
        add(prefix + "query", captured["query"], "BF16")
        add(prefix + "weights", captured["weights"])
        add(prefix + "scores", captured["scores"])
        # Test indices are small integers exactly representable in F32. The existing
        # mapped reader supports F32 but not GGML I32; keep the core reader unchanged.
        add(prefix + "indices", indices)
        mask = shared.candidates
        packed = np.stack([pack_candidates(row) for row in mask.reshape(-1, mask.shape[-1])])
        add(prefix + "candidates", torch.from_numpy(packed).reshape(1, x.shape[1], -1))

    with torch.inference_mode():
        base = torch.randn(1, 17, 5120, generator=torch.Generator().manual_seed(1020)).bfloat16()
        base_qr = torch.randn(1, 17, 1280, generator=torch.Generator().manual_seed(6121)).bfloat16()
        plans = [(base, base_qr, first) for first in [17, 1, 7, 8, 9, 13]]
        plans += [(torch.zeros_like(base[:, :2]), torch.zeros_like(base_qr[:, :2]), 1),
                  ((base[:, :2] * 1e-12).bfloat16(), (base_qr[:, :2] * 1e-12).bfloat16(), 2),
                  ((base[:, :2] * 256).bfloat16(), (base_qr[:, :2] * 256).bfloat16(), 1)]
        writer.add("fixture.plans", len(plans), "u32")
        calls, query_rows, projected = 0, 0, None
        for plan, (x, qr, first) in enumerate(plans):
            prefix = f"plan.{plan}."
            writer.add(prefix + "tokens", x.shape[1], "u32")
            writer.add(prefix + "prefill", first, "u32")
            add(prefix + "x", x); add(prefix + "qr", qr)
            index.k_cache.zero_()
            position = 0
            while position < x.shape[1]:
                count = first if position == 0 else 1
                xx, qq = x[:, position:position + count], qr[:, position:position + count]
                latent = compressor(xx, position)
                indices = index.forward(xx, qq, latent, position, 128)
                call = prefix + f"at.{position}."
                record(call, xx, qq, indices)
                add(call + "latent", latent)
                add(call + "keys", index.k_cache[:, :position + count], "BF16")
                if projected is None:
                    projected = captured["projected"][:, :1].reshape(1, 1, 32, 128).clone()
                position += count; calls += 1; query_rows += count
        positions = [0, 1, 65535, 65536, 131071, 1048575]
        writer.add("fixture.rope_cases", len(positions), "u32")
        add("rope.input", projected)
        for case, position in enumerate(positions):
            value = projected.clone()
            ns["apply_rotary_emb"](value[..., -64:], index.freqs_cis[position:position + 1])
            fp4_roundtrip(value)
            writer.add(f"rope.{case}.position", position, "u32")
            add(f"rope.{case}.output", value, "BF16")
        key_only = ns["ReferenceKey"](); key_only.__dict__.update(index.__dict__)
        latent = torch.randn(1, 640, 512, generator=generator).bfloat16()
        key_only.forward(torch.empty(1, 640, 1), None, latent, 0, 0)
        add("long.latent", latent); add("long.keys", index.k_cache, "BF16")
        counts = [1, 7, 8, 511, 512, 513, 640, 640]
        writer.add("fixture.decode_cases", len(counts), "u32")
        for case, count in enumerate(counts):
            prefix = f"decode.{case}."
            writer.add(prefix + "tokens", count, "u32")
            x = torch.randn(1, 1, 5120, generator=generator).bfloat16()
            qr = torch.randn(1, 1, 1280, generator=generator).bfloat16()
            if case == len(counts) - 1:
                x.zero_(); qr.zero_()
            indices = index.forward(x, qr, None, count - 1, 128)
            record(prefix, x, qr, indices); query_rows += 1
        # Controlled long history, not a 16K-token original compressor sequence:
        # repeat the 640 original-weight keys to cross the actual 2048-block cap.
        index.k_cache = index.k_cache.repeat(1, 26, 1)[:, :16393].contiguous()
        add("selection.keys", index.k_cache, "BF16")
        writer.add("fixture.selection_cases", 2, "u32")
        for case, count in enumerate([16385, 16393]):
            prefix = f"selection.{case}."
            writer.add(prefix + "tokens", count, "u32")
            x = torch.randn(1, 1, 5120, generator=generator).bfloat16()
            qr = torch.randn(1, 1, 1280, generator=generator).bfloat16()
            indices = index.forward(x, qr, None, count - 1, 128)
            record(prefix, x, qr, indices); query_rows += 1
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    writer.add("strata.deepseek41.compressor_source_hashes",
               json.dumps({k: v for k, v in records.items() if ".compressor." in k}, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True); writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "source_hashes": records, "sequence_plans": len(plans), "sequence_calls": calls,
            "query_rows": query_rows, "decode_cases": len(counts), "rope_cases": len(positions), "selection_cases": 2,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__,
            "scope": "official CPU source20 Indexer/Compressor/candidates; independent FP8/FP4; tie-equivalent selection; long history repeats 640 original-weight keys"}


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

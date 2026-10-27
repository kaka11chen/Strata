"""Pinned source24/source28/source32/source36 Indexer.forward with original weights and source20 keys/masks.

CPU reference only: the official forward graph, independent FP4 rounding and
full-K FP64 linear dots, rounded at the original BF16 boundaries. Controlled
long history reuses step6aa's original-weight source20 keys and candidates;
this is not a full-model or official GPU parity claim.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_mhc_fixture import KERNEL_HASH
from prepare_shared_fixture import FP64Linear, ReferenceLinear
from prepare_index_key_fixture import fp4_roundtrip


def prepare(model_dir: Path, reference: Path, kernel: Path, source20: Path, output: Path, layer: int = 24) -> dict:
    if layer not in (24, 28, 32, 36):
        raise ValueError("Only original index sources 24, 28, 32 and 36 are supported")
    if output.exists():
        raise FileExistsError(output)
    for path, digest in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong pinned source: {path}")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong model revision")
    upstream = GGUFFile(source20)
    for key, expected in {"general.architecture": "deepseek41-source20-index-query-fixture",
                          "strata.deepseek41.revision": REVISION,
                          "strata.deepseek41.official_source_sha256": SOURCE_HASH,
                          "strata.deepseek41.official_kernel_sha256": KERNEL_HASH}.items():
        if upstream.metadata.get(key) != expected:
            raise ValueError(f"Wrong confirmed source20 fixture: {key}")
    torch.set_num_threads(4)
    checkpoint, writer, records = Checkpoint(model_dir), GGUFWriter(), {}
    TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-candidate-index-query-fixture")
    writer.add("strata.deepseek41.index_query_profile", f"1017-layer{layer}-index32x128-fp8-block32")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    writer.add("fixture.linear_oracle", "fp64")
    upstream_hash = hashlib.sha256(source20.read_bytes()).hexdigest()
    writer.add("fixture.source20_sha256", upstream_hash)

    def source_tensor(name):
        t = next(t for t in upstream.tensors if t.name == name)
        if t.type_id not in (0, 30):
            raise ValueError(f"Unexpected upstream tensor type: {name}")
        with source20.open("rb") as f:
            f.seek(upstream.data_start + t.offset)
            data = f.read(t.elements * (4 if t.type_id == 0 else 2))
        value = torch.from_numpy(np.frombuffer(data, np.float32 if t.type_id == 0 else np.uint16).copy())
        if t.type_id == 30:
            value = value.view(torch.bfloat16)
        return value.reshape(tuple(reversed(t.shape)))

    def weight(part, dtype, shape):
        name = f"layers.{layer}.attn.indexer." + part
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Wrong original tensor: {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        records[name] = digest
        writer.tensors.append(_Tensor(f"blk.{layer}.attn_indexer_" + part, list(reversed(shape)), "BF16" if dtype == "BF16" else "I8", raw))
        tensor = torch.from_numpy(np.frombuffer(raw, np.uint16 if dtype == "BF16" else np.uint8).copy())
        return tensor.view({"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu}[dtype]).reshape(shape)

    def add(name, value, dtype="F32"):
        value = value.detach().contiguous()
        raw = value.bfloat16().view(torch.uint16).numpy().tobytes() if dtype == "BF16" else value.float().numpy().tobytes()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), dtype, raw))

    tree = ast.parse(reference.read_bytes())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"precompute_freqs_cis", "apply_rotary_emb"}]
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    method = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    body = []
    for node in method.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "topk":
            body += ast.parse("capture('scores', index_score)").body
        body.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "weights":
            body += ast.parse("capture('weights', weights)").body
    method.body = body
    definitions.append(ast.ClassDef(name="ReferenceIndexer", bases=[], keywords=[], decorator_list=[], body=[method]))
    captured, shared = {}, SimpleNamespace()

    def capture(name, value):
        captured[name] = value.clone()

    def quant(x, *args):
        result = fp4_roundtrip(x, *args)
        capture("query", x)
        return result

    ns = {"torch": torch, "F": F, "math": math, "lru_cache": lru_cache, "world_size": 1,
          "shared_attn": shared, "fp4_act_quant": quant, "fp4_block_size": 32, "capture": capture}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(reference), "exec"), ns)
    index = ns["ReferenceIndexer"]()
    index.owns_k, index.compress_ratio, index.rope_head_dim = False, 1, 64
    index.n_heads, index.n_local_heads, index.index_head_dim, index.index_topk = 32, 32, 128, 512
    index.softmax_scale = 128**-.5
    index.is_candidate_source, index.uses_candidates = False, True
    index.freqs_cis = ns["precompute_freqs_cis"](64, 1048576, 65536, 160000., 16., 32., 1.)
    index.weights_proj = FP64Linear(5120, 32, bias=False, dtype=torch.bfloat16)
    index.weights_proj.weight.data = weight("weights_proj.weight", "BF16", [32, 5120])
    values = weight("wq_b.weight", "F8_E4M3", [4096, 1280]).float()
    scales = weight("wq_b.scale", "F8_E8M0", [128, 40]).float()
    decoded = values * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    if not torch.isfinite(decoded).all() or not torch.equal(decoded, decoded.bfloat16().float()):
        raise ValueError("Original query weights do not expand losslessly to BF16")
    index.wq_b = ReferenceLinear(linear_oracle="fp64")
    index.wq_b.decoded = decoded.numpy()
    plans = [("plan.0.at.0.", 0, source_tensor("plan.0.at.0.keys"))]
    for case in range(8):
        count = upstream.metadata[f"decode.{case}.tokens"]
        plans.append((f"decode.{case}.", count-1, source_tensor("long.keys")[:, :count]))
    for case in range(2):
        count = upstream.metadata[f"selection.{case}.tokens"]
        plans.append((f"selection.{case}.", count-1, source_tensor("selection.keys")[:, :count]))
    rows, pruned = 0, 0
    with torch.inference_mode():
        for case, (source, start, keys) in enumerate(plans):
            prefix = f"case.{case}."
            x, qr = source_tensor(source+"x").bfloat16(), source_tensor(source+"qr").bfloat16()
            blocks = source_tensor(source+"candidates").int()
            count, tokens = keys.shape[1], x.shape[1]
            mask = torch.zeros(1, tokens, count, dtype=torch.bool)
            for row in range(tokens):
                for block in blocks[0, row].tolist():
                    if block >= 0:
                        mask[0, row, block*8:min(count, (block+1)*8)] = True
            shared.index_k, shared.candidates = keys, mask
            offset = tokens if start == 0 else 128
            selected = index.forward(x, qr, None, start, offset)
            writer.add(prefix+"start", start, "u32")
            writer.add(prefix+"tokens", tokens, "u32")
            writer.add(prefix+"offset", offset, "u32")
            add(prefix+"x", x); add(prefix+"qr", qr); add(prefix+"keys", keys, "BF16")
            add(prefix+"candidates", blocks); add(prefix+"query", captured["query"], "BF16")
            add(prefix+"weights", captured["weights"]); add(prefix+"scores", captured["scores"]); add(prefix+"indices", selected)
            rows += tokens
            pruned += count > 16384
    writer.add("fixture.cases", len(plans), "u32")
    writer.add("fixture.query_rows", rows, "u32")
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_hashes": records,
            "source20_fixture": str(source20), "source20_sha256": upstream_hash, "linear_oracle": "fp64",
            "cases": len(plans), "query_rows": rows, "pruning_cases": pruned, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "torch_version": torch.__version__,
            "scope": "pinned official candidate-consuming Indexer.forward on CPU; independent FP64 linears/FP4 rounding; original source20 keys/masks; controlled long history, not full model or GPU parity"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference-model", type=Path, required=True)
    p.add_argument("--reference-kernel", type=Path, required=True)
    p.add_argument("--source20-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=(24, 28, 32, 36), default=24)
    a = p.parse_args()
    print(json.dumps(prepare(a.model_dir, a.reference_model, a.reference_kernel, a.source20_fixture, a.output, a.layer), indent=2))

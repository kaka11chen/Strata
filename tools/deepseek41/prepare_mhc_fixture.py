"""Pinned 1017 mHC weights and stage oracles; no model downloads or engine Torch runtime.

Runs the official Block.hc_* and RMSNorm methods with an independent Torch
translation of the hash-pinned Sinkhorn kernel. Does not run TileLang/Attention.
Optionally reuses the routed fixture writer for the actual normalized FFN input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION, prepare as prepare_routed

KERNEL_HASH = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"


def split_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    assert hc_mult == 4 and sinkhorn_iters == 20 and eps == 1e-6
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def prepare(model_dir: Path, reference: Path, kernel: Path, output: Path, routed_output: Path | None) -> dict:
    if output.exists() or (routed_output is not None and routed_output.exists()):
        raise FileExistsError("Fixture output already exists")
    for path, expected in [(model_dir / "config.json", CONFIG_HASH), (reference, SOURCE_HASH), (kernel, KERNEL_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {path}")
    torch.set_num_threads(4)
    tree = ast.parse(reference.read_bytes())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Block")
    methods = [n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in {"hc_mixes", "hc_pre", "hc_post"}]
    norm = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RMSNorm")
    wrapper = ast.ClassDef(name="ReferenceMhc", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper, norm], type_ignores=[]))
    ns = {"torch": torch, "nn": nn, "F": F, "hc_split_sinkhorn": split_sinkhorn}
    exec(compile(module, str(reference), "exec"), ns)
    block = ns["ReferenceMhc"]()
    block.norm_eps, block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps = 1e-20, 4, 20, 1e-6
    checkpoint = Checkpoint(model_dir)
    writer = GGUFWriter(); TYPE_IDS["BF16"] = 30
    writer.add("general.architecture", "deepseek41-mhc-fixture")
    writer.add("strata.deepseek41.mhc_profile", "1017-hc4-delayed")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("strata.deepseek41.official_kernel_sha256", KERNEL_HASH)
    records = {}

    def add(name, tensor):
        value = tensor.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), "F32", value.numpy().tobytes()))

    generator = torch.Generator().manual_seed(1017)
    x = torch.randn((1, 1, 4, 5120), generator=generator).bfloat16()
    mock_output = torch.randn((1, 1, 5120), generator=generator).bfloat16()
    inputs = [x, (x * 16).bfloat16(), (x * 1e-12).bfloat16(), torch.zeros_like(x)]
    incoming = [torch.tensor([[[1., 0., 0., 0.]]]), torch.tensor([[[.1, .7, .2, .4]]])]
    routed = None
    for kind in ["ffn", "attn"]:
        tensors = {}
        for suffix, dtype, shape in [(f"hc_{kind}_fn", "F32", [24, 20480]),
                                      (f"hc_{kind}_scale", "F32", [3]), (f"hc_{kind}_base", "F32", [24]),
                                      (f"{kind}_norm.weight", "BF16", [5120])]:
            source_name = "layers.0." + suffix
            if checkpoint.dtype(source_name) != dtype or checkpoint.shape(source_name) != shape:
                raise ValueError(f"Malformed mHC source {source_name}")
            raw = checkpoint.raw(source_name).tobytes()
            records[source_name] = hashlib.sha256(raw).hexdigest()
            writer.tensors.append(_Tensor("blk.0." + suffix, list(reversed(shape)), dtype, raw))
            if dtype == "BF16":
                value = torch.from_numpy(checkpoint.array(source_name, np.uint16).copy()).view(torch.bfloat16)
            else:
                value = torch.from_numpy(checkpoint.array(source_name, np.float32).copy())
            tensors[suffix] = value
        norm = ns["RMSNorm"](5120, 1e-20)
        norm.weight.data = tensors[f"{kind}_norm.weight"]
        with torch.inference_mode():
            for case, residual in enumerate(inputs):
                prev = incoming[case % 2]
                pre, post, comb = block.hc_mixes(residual, tensors[f"hc_{kind}_fn"], tensors[f"hc_{kind}_scale"], tensors[f"hc_{kind}_base"])
                normalized = norm(block.hc_pre(residual, prev))
                result = block.hc_post(mock_output, residual, post, comb)
                prefix = f"{kind}.{case}."
                for name, value in [("input", residual.flatten()), ("incoming", prev.flatten()),
                                    ("sublayer", mock_output.flatten()), ("pre", pre.flatten()), ("post", post.flatten()),
                                    ("comb", comb.flatten()), ("normalized", normalized.flatten()), ("output", result.flatten())]:
                    add(prefix + name, value)
                if kind == "ffn" and case == 0 and routed_output is not None:
                    routed = prepare_routed(model_dir, reference, routed_output, input_tensor=normalized.reshape(1, 5120))
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "cases_per_branch": len(inputs),
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "scope": "official Block.hc_* + RMSNorm; independent Torch Sinkhorn; mock sublayer output; no Attention",
            "routed_integration": routed, "torch_oracle_version": torch.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routed-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.reference_kernel, args.output, args.routed_output), indent=2))


if __name__ == "__main__":
    main()

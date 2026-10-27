"""Pinned layer-20 ratio-1 compressor and candidate-mask contract fixtures.

Runs unmodified Compressor/RMSNorm/select_candidate_blocks definitions from the
pinned reference on CPU. BF16 weights and projection outputs stay BF16. Candidate
scores are adversarial synthetic inputs, not a full layer-20 indexer execution.
Offline only; refuses to overwrite an existing output.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION

BLOCK_SIZE = 8
TOPK_BLOCKS = 2048
MAX_CONTEXT = 1048576


def build_reference(reference: Path) -> dict:
    if hashlib.sha256(reference.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Not the pinned 1017 reference model")
    names = {"Compressor", "RMSNorm", "select_candidate_blocks"}
    definitions = [node for node in ast.parse(reference.read_bytes()).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("Incomplete source-20 reference definitions")
    namespace = {"torch": torch, "nn": nn, "F": F, "ModelArgs": SimpleNamespace,
                 "Linear": lambda a, b, **kw: nn.Linear(a, b, bias=False, **kw)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), namespace)
    return namespace


def pack_candidates(mask: torch.Tensor) -> np.ndarray:
    """Lossless row encoding; includes future bits of a selected partial block."""
    if mask.ndim != 1 or mask.dtype != torch.bool or mask.numel() < 1:
        raise ValueError("Expected one nonempty bool candidate row")
    width = mask.numel()
    ids = torch.nonzero(mask[::BLOCK_SIZE], as_tuple=False).flatten()
    blocks = torch.zeros((width + BLOCK_SIZE - 1) // BLOCK_SIZE, dtype=torch.bool)
    blocks[ids] = True
    if not torch.equal(blocks.repeat_interleave(BLOCK_SIZE)[:width], mask):
        raise ValueError("Candidate mask is not constant within each block")
    stride = min(TOPK_BLOCKS, blocks.numel())
    if ids.numel() > stride:
        raise ValueError("Too many candidate blocks")
    result = np.full(stride, -1, dtype="<i4")
    result[:ids.numel()] = ids.numpy()
    return result


def candidate_cases():
    # Widths above 16384 exercise actual pruning at the model's 2048-block cap.
    cases = [(1, 1, "descending", False), (7, 1, "descending", False),
             (7, 7, "descending", False), (8, 7, "descending", False),
             (8, 8, "descending", True), (9, 8, "descending", False),
             (9, 9, "descending", True), (17, 1, "descending", False),
             (17, 7, "descending", False), (17, 8, "descending", False),
             (17, 9, "descending", False), (17, 17, "holes", True),
             (16384, 16384, "descending", False),
             (16385, 16385, "descending", True),
             (16393, 16385, "descending", False),
             (16393, 16393, "shuffled", True),
             (16393, 16393, "ties", False),
             (16393, 9, "descending", False),
             (16393, 16393, "holes", False),
             (MAX_CONTEXT, MAX_CONTEXT, "shuffled", True)]
    generator = torch.Generator().manual_seed(6120)
    for width, visible, mode, decode in cases:
        positions = torch.arange(width)
        blocks = positions // BLOCK_SIZE
        if mode == "shuffled":
            ranks = torch.randperm((width + BLOCK_SIZE - 1) // BLOCK_SIZE, generator=generator)
            logits = ranks[blocks].float() + (positions % BLOCK_SIZE).float() / 8
        elif mode == "ties":
            logits = torch.zeros(width)
        else:
            logits = (width - positions).float()
            if mode == "holes":
                logits[(blocks % 3 == 1) | (blocks == (visible - 1) // BLOCK_SIZE)] = -torch.inf
        logits[visible:] = -torch.inf
        yield logits, visible, mode, decode


def prepare(model_dir: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest() != CONFIG_HASH:
        raise ValueError("Not the pinned 1017 config")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong checkpoint provenance revision")
    config = json.loads((model_dir / "config.json").read_text())["text_config"]
    if (config["compress_ratios"][20], config["candidate_source_layer_id"],
        config["candidate_topk_blocks"], config["candidate_block_size"]) != (1, 20, TOPK_BLOCKS, BLOCK_SIZE):
        raise ValueError("Wrong source-20 configuration")
    torch.set_num_threads(4)
    namespace = build_reference(reference)
    args = SimpleNamespace(compress_ratios=config["compress_ratios"], dim=5120,
                           head_dim=512, norm_eps=1e-20, max_batch_size=1)
    compressor = namespace["Compressor"](args, 20).eval()
    if any(hasattr(compressor, name) for name in ["wgate", "kv_state", "score_state"]):
        raise AssertionError("Official ratio-1 compressor unexpectedly has pair state")
    checkpoint = Checkpoint(model_dir)
    writer = GGUFWriter()
    TYPE_IDS.update(BF16=30, I8=24)
    writer.add("general.architecture", "deepseek41-source20-contract-fixture")
    writer.add("strata.deepseek41.compressor_profile", "1017-layer20-ratio1-bf16")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    writer.add("fixture.candidate_encoding", "sorted-i32-little-endian-bytes-minus-one-padding")
    records = {}
    for part, shape in [("wkv", [512, 5120]), ("norm", [512])]:
        name = f"layers.20.attn.compressor.{part}.weight"
        if checkpoint.dtype(name) != "BF16" or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed ratio-1 source {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if provenance["tensors"][name]["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {name}")
        records[name] = digest
        writer.tensors.append(_Tensor(f"blk.20.attn_compressor_{part}.weight", list(reversed(shape)), "BF16", raw))
        value = torch.from_numpy(np.frombuffer(raw, np.uint16).copy()).view(torch.bfloat16).reshape(shape)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite compressor source {name}")
        getattr(compressor, part).weight.data = value  # Do not promote ratio-1 projection to F32.

    def add_bf16(name, value):
        value = value.detach().contiguous()
        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise AssertionError(f"Expected finite BF16 reference tensor {name}")
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), "BF16",
                                      value.view(torch.uint16).numpy().tobytes()))

    projection = []
    handle = compressor.wkv.register_forward_hook(lambda _m, _args, value: projection.append(value.clone()))
    base = torch.randn(1, 17, 5120, generator=torch.Generator().manual_seed(1020)).bfloat16()
    plans = [(base, first) for first in [17, 1, 7, 8, 9, 13]]
    plans += [(torch.zeros_like(base[:, :2]), 1), ((base[:, :2] * 1e-12).bfloat16(), 2),
              ((base[:, :2] * 256).bfloat16(), 1)]
    writer.add("fixture.plans", len(plans), "u32")
    calls = 0
    position_checks = 0
    with torch.inference_mode():
        for plan, (data, first) in enumerate(plans):
            prefix = f"plan.{plan}."
            writer.add(prefix + "tokens", data.shape[1], "u32")
            writer.add(prefix + "prefill", first, "u32")
            add_bf16(prefix + "input", data)
            position = 0
            while position < data.shape[1]:
                count = first if position == 0 else 1
                x = data[:, position:position + count]
                projection.clear()
                value = compressor(x, position)
                if value.shape != (1, count, 512) or len(projection) != 1:
                    raise AssertionError("Ratio-1 must emit one latent per token")
                call = prefix + f"at.{position}."
                add_bf16(call + "projection", projection[0])
                add_bf16(call + "output", value)
                # Ratio-1 has no pending tail and ignores start_pos for its arithmetic.
                if not torch.equal(value, compressor(x, position + 97)):
                    raise AssertionError("Ratio-1 output depends on position or stale pair state")
                position_checks += 1
                position += count
                calls += 1
    handle.remove()

    candidate_records = []
    for index, (logits, visible, mode, decode) in enumerate(candidate_cases()):
        width = logits.numel()
        # Exercise both official compress_lens forms (decode int, prefill tensor).
        lengths = visible if decode else torch.tensor([[visible]])
        mask = namespace["select_candidate_blocks"](logits.reshape(1, 1, width), lengths,
                                                     TOPK_BLOCKS, BLOCK_SIZE).flatten()
        packed = pack_candidates(mask)
        prefix = f"candidate.{index}."
        for key, value in [("width", width), ("visible", visible), ("decode", int(decode))]:
            writer.add(prefix + key, value, "u32")
        writer.add(prefix + "mode", mode)
        writer.tensors.append(_Tensor(prefix + "scores", [width], "F32", logits.numpy().tobytes()))
        writer.tensors.append(_Tensor(prefix + "mask", [width], "I8", mask.numpy().astype(np.uint8).tobytes()))
        # Reuse existing raw-byte GGUF support; do not change the default reader for test I32 payloads.
        writer.tensors.append(_Tensor(prefix + "blocks", [packed.nbytes], "I8", packed.tobytes()))
        candidate_records.append({"width": width, "visible": visible, "mode": mode, "decode": decode,
                                  "selected_blocks": int(np.count_nonzero(packed >= 0)),
                                  "future_mask_bits": int(mask[visible:].sum())})
    writer.add("fixture.candidate_rows", len(candidate_records), "u32")
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "source_tensors_verified": len(records),
            "source_hashes": records, "plans": len(plans), "calls": calls,
            "position_invariance_checks": position_checks, "candidate_rows": candidate_records,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "torch_oracle_version": torch.__version__,
            "scope": "official CPU ratio-1 BF16 projection/RMSNorm; synthetic scores through official candidate function",
            "not_validated": "native layer-20 arithmetic, full indexer/decoder, official GPU parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.output), indent=2))


if __name__ == "__main__":
    main()

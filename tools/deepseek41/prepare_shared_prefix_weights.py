"""Prepare dense weights for fixed 1017 shared-KV layers 4..7/9..13/15..19/21..39, offline and hash checked.

This bounded fixture contains one layer's attention, mHC, shared expert and
router. Routed experts are prepared separately by prepare_prefix_experts.py.
No oracle outputs or fabricated expert payloads are included.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from prepare_expert_fixture import Checkpoint, GGUFWriter, TYPE_IDS, _Tensor
from prepare_prefix_experts import CONFIG_HASH, REVISION


def prepare(model_dir: Path, layer: int, output: Path) -> dict:
    if layer not in (*range(4, 8), *range(9, 14), *range(15, 20), *range(21, 40)):
        raise ValueError("Only original source-2/8/14/20 shared-KV layers 4..7/9..13/15..19/21..39 are supported here")
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest() != CONFIG_HASH:
        raise ValueError("Wrong pinned 1017 config")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong source revision")
    TYPE_IDS.update(I8=24, BF16=30)  # Same storage IDs as the existing attention fixtures.
    checkpoint, writer, records = Checkpoint(model_dir), GGUFWriter(), {}
    writer.add("general.architecture", "deepseek41-shared-prefix-fixture")
    writer.add("strata.deepseek41.attention_profile", f"1017-layer{layer}-candidates20-fp8-block32" if layer in (24, 28, 32, 36) else
               f"1017-layer{layer}-shared{2 if layer < 8 else 8 if layer < 14 else 14 if layer < 20 else 20 if layer < 24 else layer // 4 * 4}-fp8-block32")
    if layer in (24, 28, 32, 36):
        writer.add("strata.deepseek41.index_query_profile", f"1017-layer{layer}-index32x128-fp8-block32")
    writer.add("strata.deepseek41.mhc_profile", "1017-hc4-delayed")
    writer.add("strata.deepseek41.shared_weight_encoding", "e4m3fn-e8m0-block32")
    writer.add("strata.deepseek41.model_revision", REVISION)

    def add(suffix, target, dtype, shape):
        name = f"layers.{layer}." + suffix
        if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
            raise ValueError(f"Malformed original tensor {name}")
        raw = checkpoint.raw(name).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != provenance["tensors"][name]["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        records[name] = digest
        writer.tensors.append(_Tensor(target, list(reversed(shape)), dtype if dtype in {"F32", "BF16"} else "I8", raw))

    for projection, rows, cols in [("wq_a", 1280, 5120), ("wq_b", 32768, 1280), ("wkv", 512, 5120),
                                   ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)]:
        for part, dtype, shape in [("weight", "F8_E4M3", [rows, cols]), ("scale", "F8_E8M0", [rows // 32, cols // 32])]:
            add(f"attn.{projection}.{part}", f"blk.{layer}.attn_{projection}.{part}", dtype, shape)
    for suffix, target, dtype, shape in [("q_norm.weight", "q_norm.weight", "BF16", [1280]),
                                        ("kv_norm.weight", "kv_norm.weight", "BF16", [512]), ("attn_sink", "sink", "F32", [64])]:
        add("attn." + suffix, f"blk.{layer}.attn_" + target, dtype, shape)
    if layer in (24, 28, 32, 36):
        for part, dtype, shape in [("wq_b.weight", "F8_E4M3", [4096, 1280]),
                                   ("wq_b.scale", "F8_E8M0", [128, 40]),
                                   ("weights_proj.weight", "BF16", [32, 5120])]:
            add("attn.indexer." + part, f"blk.{layer}.attn_indexer_" + part, dtype, shape)
    for kind in ("attn", "ffn"):
        for part, shape in [("fn", [24, 20480]), ("scale", [3]), ("base", [24])]:
            name = f"hc_{kind}_{part}"
            add(name, f"blk.{layer}." + name, "F32", shape)
        add(f"{kind}_norm.weight", f"blk.{layer}.{kind}_norm.weight", "BF16", [5120])
    for projection, role, rows, cols in [("w1", "gate", 2304, 5120), ("w3", "up", 2304, 5120), ("w2", "down", 5120, 2304)]:
        for part, dtype, shape in [("weight", "F8_E4M3", [rows, cols]), ("scale", "F8_E8M0", [rows // 32, cols // 32])]:
            add(f"ffn.shared_experts.{projection}.{part}", f"blk.{layer}.ffn_{role}_shexp.{part}", dtype, shape)
    add("ffn.gate.weight", "router.weight", "BF16", [384, 5120])
    add("ffn.gate.bias", "router.bias", "F32", [384])
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "layer": layer, "revision": REVISION, "source_hashes": records,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.layer, args.output), indent=2))

"""Losslessly repack ONE official FP4 expert as an MXFP4 GGUF test fixture.

This is not a model converter. The output architecture is deliberately different
from deepseek41: the full-model inspector must not accept it as a complete model.
Uses Strata's existing safetensors access and GGUF writer. No torch dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4_local_pack import Checkpoint
from gguf_writer import GGUFWriter, TYPE_IDS, _Tensor


def repack_fp4(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Adjacent HF nibbles -> low/high 16-element GGUF halves; retain all codes and scale bytes."""
    if weight.dtype != np.uint8 or scale.dtype != np.uint8 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("weight/scale must be uint8 matrices")
    rows, packed_k = weight.shape
    if rows == 0 or packed_k == 0 or packed_k % 16 or scale.shape != (rows, packed_k // 16):
        raise ValueError("FP4 weight/scale shapes disagree with 32-value blocks")
    if np.any(scale == 255):
        raise ValueError("nonfinite E8M0 scale")
    source = weight.reshape(rows, -1, 16)
    codes = np.empty((rows, packed_k // 16, 32), dtype=np.uint8)
    codes[..., 0::2] = source & 15
    codes[..., 1::2] = source >> 4
    blocks = np.empty((rows, packed_k // 16, 17), dtype=np.uint8)
    blocks[..., 0] = scale
    blocks[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return blocks


def prepare(model_dir: Path, prefix: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Checkpoint(model_dir)
    writer = GGUFWriter()
    writer.add("general.architecture", "deepseek41-expert-fixture")
    writer.add("strata.deepseek41.source_prefix", prefix)
    writer.add("strata.deepseek41.activation_baseline", "ggml-q8_0")
    records = {}
    for role, projection, rows, cols in [("gate", "w1", 2304, 5120), ("up", "w3", 2304, 5120), ("down", "w2", 5120, 2304)]:
        wn, sn = f"{prefix}.{projection}.weight", f"{prefix}.{projection}.scale"
        if checkpoint.dtype(wn) != "I8" or checkpoint.dtype(sn) != "F8_E8M0":
            raise ValueError(f"{projection}: expected packed FP4 and E8M0 scale")
        if checkpoint.shape(wn) != [rows, cols // 2] or checkpoint.shape(sn) != [rows, cols // 32]:
            raise ValueError(f"{projection}: not a 1017 expert matrix")
        weight, scale = checkpoint.array(wn, np.uint8), checkpoint.array(sn, np.uint8)
        payload = repack_fp4(weight, scale).tobytes()
        # The existing writer accepts extension types through its registry; no change to the shared writer.
        TYPE_IDS["MXFP4"] = 39
        writer.tensors.append(_Tensor(f"blk.0.ffn_{role}_exps.weight", [cols, rows, 1], "MXFP4", payload))
        records[role] = {"weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                         "scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
                         "mxfp4_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "prefix": prefix, "matrices": records,
            "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="layers.0.ffn.experts.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.prefix, args.output), indent=2))


if __name__ == "__main__":
    main()

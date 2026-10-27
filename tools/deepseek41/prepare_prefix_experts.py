"""Prepare bounded supplemental experts for the native layer-0..39 integration test.

Offline only. Each output is three concatenated standard GGUF MXFP4 projections
in gate/up/down order, consumed by the test through the existing lossless NVFP4
adapter. This is a guarded test fixture, not a full-model format or converter.
Original layer/expert IDs and pinned source hashes are retained in a sidecar.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_expert_fixture import Checkpoint, repack_fp4

REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CONFIG_HASH = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"


def validate_nvfp4_adapter(weight: np.ndarray, scale: np.ndarray) -> None:
    """Guard exact native conversion; source FP4 codes/scales remain unchanged."""
    if np.any(scale < 117) or np.any(scale > 135):
        raise ValueError("Original scales exceed the existing lossless NVFP4 adapter range")
    tiny = weight.reshape(*scale.shape, 16)[scale == 117]
    low, high = tiny & 7, (tiny >> 4) & 7
    needs_global_half = np.any((low == 1) | (low == 3) | (high == 1) | (high == 3))
    if needs_global_half and np.any(scale == 135):
        raise ValueError("No exact global scale for simultaneous scale117 tiny codes and scale135")


def prepare(model_dir: Path, layer: int, experts: list[int], output_dir: Path) -> list[dict]:
    if layer not in range(40) or not experts or len(experts) > 32 or len(set(experts)) != len(experts) or any(e not in range(384) for e in experts):
        raise ValueError("Expected 1..32 distinct original experts from a fixed prefix layer")
    if hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest() != CONFIG_HASH:
        raise ValueError("Wrong pinned 1017 config")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_revision"] != REVISION:
        raise ValueError("Wrong source revision")
    checkpoint = Checkpoint(model_dir)
    paths = [output_dir / f"layer-{layer}-expert-{expert}.mxfp4" for expert in experts]
    for path in paths:
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for expert, path in zip(experts, paths):
        records, parts = {}, []
        for projection, rows, cols in [("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304)]:
            arrays = []
            for suffix, dtype, shape in [("weight", "I8", [rows, cols // 2]), ("scale", "F8_E8M0", [rows, cols // 32])]:
                name = f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
                    raise ValueError(f"Malformed original expert tensor {name}")
                raw = checkpoint.raw(name)
                digest = hashlib.sha256(raw).hexdigest()
                if digest != provenance["tensors"][name]["sha256"]:
                    raise ValueError(f"Source hash mismatch: {name}")
                records[name] = digest
                arrays.append(checkpoint.array(name, np.uint8))
            validate_nvfp4_adapter(*arrays)
            parts.append(repack_fp4(*arrays).tobytes())
        payload = b"".join(parts)
        if len(payload) != 18800640:
            raise ValueError("Unexpected fixed expert byte count")
        record = {"revision": REVISION, "layer": layer, "expert": expert, "format": "gguf-mxfp4-gate-up-down",
                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "source_hashes": records}
        with path.open("xb") as handle:
            handle.write(payload)
        with path.with_suffix(".json").open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        results.append(record)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.layer, args.experts, args.output_dir), indent=2))

"""Raw 1017 shared FP8 weights and an independent official-Expert CPU oracle.

Offline test fixture only. Reuses Strata's safetensors reader/GGUF writer. The
oracle executes hash-pinned Expert.forward with NumPy block-32 linear operations.
ReferenceLinear also offers an explicit FP64 accumulation control for attention
fixtures. Neither mode reproduces CUDA/TileLang bit order. No checkpoint downloads.
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
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION


def quantize(x: torch.Tensor) -> np.ndarray:
    blocks = x.float().numpy().reshape(-1, 32)
    amax = np.maximum(np.max(np.abs(blocks), axis=1), np.float32(1e-4))
    mantissa, exponent = np.frexp(amax * np.float32(1 / 448))
    factors = np.ldexp(np.ones_like(amax), exponent - (mantissa == .5))
    scaled = np.clip(blocks / factors[:, None], -448, 448)
    codes = np.arange(127, dtype=np.int32)
    e, f = codes >> 3, (codes & 7).astype(np.float32)
    grid = np.where(e == 0, f * 2**-9, np.ldexp(1 + f / 8, e - 7))
    magnitudes = np.abs(scaled)
    upper = np.minimum(np.searchsorted(grid, magnitudes), 126)
    lower = np.maximum(upper - 1, 0)
    below, above = magnitudes - grid[lower], grid[upper] - magnitudes
    pick = (above < below) | ((above == below) & ((upper & 1) == 0))
    return (np.copysign(grid[np.where(pick, upper, lower)], scaled) * factors[:, None]).reshape(x.shape)


LINEAR_ORACLES = ("block32", "fp64")


def fp64_dot(activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Full-K FP64 dots, rounded once to FP32; row tiles bound scratch memory."""
    a64 = activation.astype(np.float64)
    result = np.empty((*activation.shape[:-1], weight.shape[0]), dtype=np.float32)
    for first in range(0, weight.shape[0], 256):
        result[..., first:first + 256] = a64 @ weight[first:first + 256].astype(np.float64).T
    return result


class FP64Linear(nn.Linear):
    """Offline bias-free dense projection; preserve F32/BF16 input/output types."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            raise ValueError("FP64 reference requires bias-free projections")
        result = fp64_dot(x.detach().float().numpy(), self.weight.detach().float().numpy())
        return torch.from_numpy(result).to(x.dtype)


class ReferenceLinear(nn.Module):
    def __init__(self, *_args, linear_oracle="block32", **_kwargs):
        super().__init__()
        if linear_oracle not in LINEAR_ORACLES:
            raise ValueError(f"Unsupported linear oracle: {linear_oracle}")
        self.linear_oracle = linear_oracle
        self.decoded: np.ndarray

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = quantize(x)
        if self.linear_oracle == "fp64":
            # Retain input FP8 quantization and the FP32/BF16 output boundary.
            return torch.from_numpy(fp64_dot(a, self.decoded)).to(torch.bfloat16)
        result = np.zeros((*a.shape[:-1], self.decoded.shape[0]), dtype=np.float32)
        # Official FP8 GEMM starts a new dot accumulator for each K=32 scale group.
        for first in range(0, a.shape[-1], 32):
            result += a[..., first:first + 32] @ self.decoded[:, first:first + 32].T
        return torch.from_numpy(result).to(torch.bfloat16)


def prepare(model_dir: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest() != CONFIG_HASH:
        raise ValueError("Not the pinned 1017 config")
    source = reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("Not the pinned official inference_model.py")
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Expert"]
    if len(definitions) != 1:
        raise ValueError("Official Expert definition missing")
    namespace = {"torch": torch, "nn": nn, "F": F, "Linear": ReferenceLinear}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec"), namespace)
    expert = namespace["Expert"](5120, 2304, swiglu_limit=10.)
    checkpoint = Checkpoint(model_dir)
    writer = GGUFWriter()
    TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-shared-fixture")
    writer.add("strata.deepseek41.shared_weight_encoding", "e4m3fn-e8m0-block32")
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.official_source_sha256", SOURCE_HASH)
    records = {}
    for role, projection, rows, cols in [("gate", "w1", 2304, 5120), ("up", "w3", 2304, 5120), ("down", "w2", 5120, 2304)]:
        prefix = f"layers.0.ffn.shared_experts.{projection}"
        for suffix, dtype, shape in [("weight", "F8_E4M3", [rows, cols]), ("scale", "F8_E8M0", [rows // 32, cols // 32])]:
            name = prefix + "." + suffix
            if checkpoint.dtype(name) != dtype or checkpoint.shape(name) != shape:
                raise ValueError(f"Wrong shared expert tensor: {name}")
            data = checkpoint.raw(name).tobytes()
            records[name] = hashlib.sha256(data).hexdigest()
            writer.tensors.append(_Tensor(f"blk.0.ffn_{role}_shexp.{suffix}", list(reversed(shape)), "I8", data))
        codes = torch.from_numpy(checkpoint.array(prefix + ".weight", np.uint8).copy()).view(torch.float8_e4m3fn)
        scale = torch.from_numpy(checkpoint.array(prefix + ".scale", np.uint8).copy()).view(torch.float8_e8m0fnu)
        decoded = codes.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        expanded = decoded.to(torch.bfloat16)
        if not torch.isfinite(decoded).all() or not torch.equal(expanded.float(), decoded):
            raise ValueError("Shared weights do not expand losslessly to BF16")
        writer.tensors.append(_Tensor(f"expected.{role}.bf16", [cols, rows], "BF16", expanded.view(torch.uint16).numpy().tobytes()))
        getattr(expert, projection).decoded = decoded.numpy()
    writer.add("strata.deepseek41.source_hashes", json.dumps(records, sort_keys=True))
    torch.set_num_threads(4)
    base = torch.randn((1, 5120), generator=torch.Generator().manual_seed(1017)).bfloat16()
    cases = [base, (base * 16).bfloat16(), torch.zeros_like(base)]
    with torch.inference_mode():
        for case, x in enumerate(cases):
            y = expert(x)
            writer.tensors.append(_Tensor(f"input.{case}", [5120], "F32", x.float().numpy().tobytes()))
            writer.tensors.append(_Tensor(f"expected.output.{case}", [5120], "F32", y.float().numpy().tobytes()))
            quantized = torch.from_numpy(quantize(x)).bfloat16().view(torch.uint16)
            writer.tensors.append(_Tensor(f"expected.quantized.{case}", [5120], "BF16", quantized.numpy().tobytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {"fixture": str(output), "revision": REVISION, "cases": len(cases),
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "scope": "shared expert only; official Expert graph + independent NumPy block32 linear oracle",
            "torch_oracle_version": torch.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.output), indent=2))


if __name__ == "__main__":
    main()

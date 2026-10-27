"""Pinned layer-1/14 Engram hash/state and bounded real-row lookup oracle.

Only local inputs. --discover reports original row IDs before fetching them.
The table fixture is a guarded sparse GGUF with ORIGINAL logical row positions;
it is not a complete checkpoint. No projection, residual gate or decoder here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import runpy
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer

from prepare_expert_fixture import GGUFWriter, TYPE_IDS, _Tensor
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION

ENGRAM_HASH = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
TOKENIZER_HASH = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
TOKEN_MAP_HASH = "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
HEADER_HASH = "e5c5c7fc900caec400644bef07aa5d7e28fc6ca941c8c47592b393963b218242"
ROWS = 384006168
# (original row count, safetensors header SHA256, payload start byte).
# Keep the layer-1 constants above for the confirmed projection preparer.
TABLES = {
    1: (ROWS, HEADER_HASH, 664),
    14: (384016682, "5d8fa4697d7a071c31baeee6947d493c08ea3243261c5d0f79c355ad191b8222", 672),
}


def prepare(root: Path, output: Path | None, table_output: Path | None, discover: bool = False, layer: int = 1) -> dict:
    if layer not in TABLES:
        raise ValueError("Only original Engram layers 1 and 14 are supported")
    rows, header_hash, data_start = TABLES[layer]
    if not discover and (output is None or table_output is None):
        raise ValueError("Both fixture outputs are required")
    if not discover and (output.exists() or table_output.exists()):
        raise FileExistsError("Fixture output already exists")
    for name, expected in [("config.json", CONFIG_HASH), ("inference_model.py", SOURCE_HASH),
                           ("inference_engram.py", ENGRAM_HASH), ("tokenizer.json", TOKENIZER_HASH)]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Not the pinned 1017 source: {name}")
    torch.set_num_threads(2)
    cfg = json.loads((root / "config.json").read_text())["text_config"]
    args = SimpleNamespace(**cfg, engram_pad_id=cfg["engram_pad_token_id"], max_batch_size=1, max_seq_len=1024)
    source = runpy.run_path(str(root / "inference_engram.py"))
    backend = Tokenizer.from_file(str(root / "tokenizer.json"))

    class Adapter:
        backend_tokenizer = backend
        def __len__(self):
            return backend.get_vocab_size()

    token_map, vocab = source["build_compressed_token_map"](Adapter())
    if len(token_map) != 129280 or vocab != 99092 or token_map[2] != 2:
        raise ValueError("Pinned tokenizer compressed vocabulary changed")
    map_digest = hashlib.sha256(np.asarray(token_map, "<u4").tobytes()).hexdigest()
    if map_digest != TOKEN_MAP_HASH:
        raise ValueError("Pinned compressed token map changed")
    layout = source["EngramLayout"].from_args(args)
    if (layout.layer_ids != (1, 14) or layout.num_embeddings != (ROWS, TABLES[14][0]) or
            layout.max_ngram_size != 4 or layout.n_heads != 8 or layout.head_dim != 256):
        raise ValueError("Pinned Engram layout changed")
    column = layout.layer_ids.index(layer)
    if sum(sum(group) for group in layout.primes[column]) != rows:
        raise ValueError("Hash buckets do not cover the original table")
    state = source["NgramHashState"](args, layout, Adapter())
    tokens = torch.tensor([[0, 455, 1805, 1, 20852, 525, 21716, 240, 129279]])
    live = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1]], dtype=torch.bool)
    plain = state(tokens, 0)[:, :, column].clone()
    masked = state(tokens, 0, live)[:, :, column].clone()
    generator = torch.Generator().manual_seed(611017)
    random_tokens = torch.randint(0, 129280, (1, 257), generator=generator)
    random_live = torch.rand((1, 257), generator=generator) > .14
    random_hashes = state(random_tokens, 0, random_live)[:, :, column].clone()
    required = sorted(set(masked.flatten().tolist()) | {0, 15, 16, rows - 1})
    report = {"revision": REVISION, "engram_source_sha256": ENGRAM_HASH, "tokenizer_sha256": TOKENIZER_HASH,
              "token_map_sha256": map_digest, "required_original_rows": required, "rows": len(required),
              "layer": layer, "table_rows": rows, "primes": layout.primes[column], "multipliers": state.multipliers[column].tolist(),
              "tokens": tokens.tolist(), "live_mask": live.int().tolist(), "torch_oracle_version": torch.__version__}
    if discover:
        return report
    provenance = json.loads((root / f"engram{layer}-rows/provenance.json").read_text())
    if provenance["revision"] != REVISION or provenance["header_sha256"] != header_hash:
        raise ValueError("Not the pinned table row cache")
    payloads = {}
    for row in required:
        raw = (root / f"engram{layer}-rows/{row}.bin").read_bytes()
        record = provenance["rows"][str(row)]
        if len(raw) != 264 or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Corrupt cached original row {row}")
        if record["weight_offset"] != data_start + row * 256 or record["scale_offset"] != data_start + rows * 256 + row * 8:
            raise ValueError("Cached original row offsets changed")
        payloads[row] = raw
    # Original forward and dequantization arithmetic. Only F.embedding's backing
    # storage is compact: it maps requested GLOBAL row IDs to downloaded rows.
    ids = torch.tensor(required)
    values = torch.from_numpy(np.stack([np.frombuffer(payloads[r][:256], np.uint8) for r in required]).copy()).view(torch.float8_e4m3fn)
    scales = torch.from_numpy(np.stack([np.frombuffer(payloads[r][256:], np.uint8) for r in required]).copy()).view(torch.float8_e8m0fnu)

    def sparse_embedding(indices, weight):
        positions = torch.searchsorted(ids, indices)
        if (positions >= len(ids)).any() or not torch.equal(ids[positions], indices):
            raise ValueError("Oracle attempted an unpopulated original row")
        return torch.nn.functional.embedding(positions, weight)

    definition = [n for n in ast.parse((root / "inference_model.py").read_bytes()).body
                  if isinstance(n, ast.ClassDef) and n.name == "ParallelEngramEmbedding"]
    ns = {"torch": torch, "nn": nn, "F": SimpleNamespace(embedding=sparse_embedding), "world_size": 1}
    exec(compile(ast.Module(body=definition, type_ignores=[]), str(root / "inference_model.py"), "exec"), ns)
    embed = ns["ParallelEngramEmbedding"].__new__(ns["ParallelEngramEmbedding"]); nn.Module.__init__(embed)
    embed.vocab_start_idx, embed.vocab_end_idx, embed.block_size = 0, rows, 32
    embed.weight, embed.scale = nn.Parameter(values, requires_grad=False), nn.Parameter(scales, requires_grad=False)
    with torch.inference_mode():
        expected = embed(masked)
        expected_rows = embed(ids)
    writer = GGUFWriter(); TYPE_IDS.update(I8=24, BF16=30)
    writer.add("general.architecture", "deepseek41-engram-fixture")
    writer.add("strata.deepseek41.engram_hash_profile", f"1017-layer{layer}-compressed-4gram")
    writer.add("strata.deepseek41.tokenizer_sha256", TOKENIZER_HASH)
    writer.tensors.append(_Tensor("engram.token_map", [4, len(token_map)], "I8", np.asarray(token_map, "<u4").tobytes()))
    writer.add("strata.deepseek41.revision", REVISION)
    writer.add("strata.deepseek41.engram_source_sha256", ENGRAM_HASH)
    writer.add("strata.deepseek41.engram_token_map_sha256", map_digest)
    writer.add("strata.deepseek41.row_provenance", json.dumps(provenance, sort_keys=True))

    def integers(name, x, dtype):
        raw = x.numpy().astype(dtype).tobytes()
        writer.tensors.append(_Tensor(name, [len(raw)], "I8", raw))

    for name, x, dtype in [("tokens", tokens, "<i4"), ("live", live, "u1"), ("hash.plain", plain, "<u4"),
                            ("hash.masked", masked, "<u4"), ("random.tokens", random_tokens, "<i4"),
                            ("random.live", random_live, "u1"), ("random.hash", random_hashes, "<u4"),
                            ("populated", ids, "<u4")]:
        integers(name, x, dtype)
    for name, x in [("expected.lookup", expected), ("expected.rows", expected_rows)]:
        writer.tensors.append(_Tensor(name, list(reversed(x.shape)), "BF16", x.contiguous().view(torch.uint16).numpy().tobytes()))
    output.parent.mkdir(parents=True, exist_ok=True); writer.write(output)
    table = GGUFWriter()
    table.add("general.architecture", "deepseek41-engram-table-fixture")
    table.add("strata.deepseek41.engram_table_profile", f"1017-layer{layer}-fp8-row32")
    table.add("strata.deepseek41.engram_complete", 0, "u32")
    table.add("strata.deepseek41.revision", REVISION)
    table.add("strata.deepseek41.source_header_sha256", header_hash)
    coverage = np.asarray(required, "<u4").tobytes()
    values_offset = (len(coverage) + 31) // 32 * 32
    tensors = [(f"blk.{layer}.engram_populated_rows", [4, len(required)], 0),
               (f"blk.{layer}.engram_embed.weight", [256, rows], values_offset),
               (f"blk.{layer}.engram_embed.scale", [8, rows], values_offset + rows * 256)]
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(table.metadata)))
    for key, (typ, value) in table.metadata.items():
        header += table._kv_bytes(key, typ, value)
    for name, shape, offset in tensors:
        header += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<IQQIQ", 2, *shape, 24, offset)
    header += bytes(-len(header) % 32)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    with table_output.open("xb") as f:
        f.write(header); f.write(coverage); f.truncate(len(header) + values_offset + rows * 264)
        for row, raw in payloads.items():
            f.seek(len(header) + values_offset + row * 256); f.write(raw[:256])
            f.seek(len(header) + values_offset + rows * 256 + row * 8); f.write(raw[256:])
    report.update(fixture=str(output), fixture_bytes=output.stat().st_size,
                  fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  table_fixture=str(table_output), table_logical_bytes=table_output.stat().st_size,
                  table_disk_bytes=table_output.stat().st_blocks * 512,
                  downloaded_row_bytes=len(required) * 264)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-output", type=Path)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--layer", type=int, choices=(1, 14), default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_root, args.output, args.table_output, args.discover, args.layer), indent=2))


if __name__ == "__main__":
    main()

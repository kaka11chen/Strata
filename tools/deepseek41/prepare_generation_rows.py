"""Prepare guarded, original Engram rows for one bounded completion sequence.

Offline only. --discover lists original row IDs for the pinned hash state.
Existing reference fixtures are never modified; output tables must be new.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
import struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tokenizers import Tokenizer
from prepare_engram_fixture import ENGRAM_HASH, TOKENIZER_HASH, TOKEN_MAP_HASH, TABLES
from prepare_expert_fixture import GGUFWriter
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION


def prepare(root: Path, tokens: list[int], output: Path | None = None) -> dict:
    if not 1 <= len(tokens) <= 64 or any(type(t) is not int or not 0 <= t < 129280 for t in tokens):
        raise ValueError('Expected 1..64 original token IDs')
    for name, expected in [('config.json', CONFIG_HASH), ('inference_model.py', SOURCE_HASH),
                           ('inference_engram.py', ENGRAM_HASH), ('tokenizer.json', TOKENIZER_HASH)]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Not the pinned 1017 source: {name}')
    if output is not None and output.exists():
        raise FileExistsError('Generation table output directory must be new')
    torch.set_num_threads(2)
    cfg = json.loads((root / 'config.json').read_text())['text_config']
    args = SimpleNamespace(**cfg, engram_pad_id=cfg['engram_pad_token_id'], max_batch_size=1, max_seq_len=1024)
    source = runpy.run_path(str(root / 'inference_engram.py'))
    backend = Tokenizer.from_file(str(root / 'tokenizer.json'))

    class Adapter:
        backend_tokenizer = backend
        def __len__(self):
            return backend.get_vocab_size()

    mapping, vocab = source['build_compressed_token_map'](Adapter())
    if vocab != 99092 or hashlib.sha256(np.asarray(mapping, '<u4').tobytes()).hexdigest() != TOKEN_MAP_HASH:
        raise ValueError('Pinned compressed token map changed')
    layout = source['EngramLayout'].from_args(args)
    state = source['NgramHashState'](args, layout, Adapter())
    hashes = state(torch.tensor([tokens]), 0)
    report = {'revision': REVISION, 'tokens': tokens, 'tokenizer_sha256': TOKENIZER_HASH,
              'engram_source_sha256': ENGRAM_HASH, 'token_map_sha256': TOKEN_MAP_HASH, 'layers': {}}
    tables = []
    for column, layer in enumerate((1, 14)):
        rows, header_hash, data_start = TABLES[layer]
        required = sorted(set(hashes[:, :, column].flatten().tolist()))
        if not required or len(required) > 64 * 24 or min(required) < 0 or max(required) >= rows:
            raise ValueError('Original Engram rows outside bounded range')
        record = {'layer': layer, 'table_rows': rows, 'header_sha256': header_hash,
                  'required_original_rows': required, 'hashes': hashes[:, :, column].tolist()}
        report['layers'][str(layer)] = record
        if output is None:
            continue
        cache = root / f'engram{layer}-rows'
        provenance = json.loads((cache / 'provenance.json').read_text())
        if provenance['revision'] != REVISION or provenance['header_sha256'] != header_hash:
            raise ValueError('Not the pinned row cache')
        payloads = {}
        for row in required:
            raw = (cache / f'{row}.bin').read_bytes()
            meta = provenance['rows'][str(row)]
            if (len(raw) != 264 or hashlib.sha256(raw).hexdigest() != meta['sha256'] or
                    meta['weight_offset'] != data_start + row * 256 or
                    meta['scale_offset'] != data_start + rows * 256 + row * 8):
                raise ValueError(f'Corrupt original row: layer {layer}, row {row}')
            payloads[row] = raw
        table = GGUFWriter()
        for name, value in [('general.architecture', 'deepseek41-engram-table-fixture'),
                            ('strata.deepseek41.engram_table_profile', f'1017-layer{layer}-fp8-row32'),
                            ('strata.deepseek41.revision', REVISION),
                            ('strata.deepseek41.source_header_sha256', header_hash)]:
            table.add(name, value)
        table.add('strata.deepseek41.engram_complete', 0, 'u32')
        coverage = np.asarray(required, '<u4').tobytes()
        offset = (len(coverage) + 31) // 32 * 32
        tensors = [(f'blk.{layer}.engram_populated_rows', [4, len(required)], 0),
                   (f'blk.{layer}.engram_embed.weight', [256, rows], offset),
                   (f'blk.{layer}.engram_embed.scale', [8, rows], offset + rows * 256)]
        header = bytearray(struct.pack('<IIQQ', 0x46554747, 3, 3, len(table.metadata)))
        for key, (typ, value) in table.metadata.items():
            header += table._kv_bytes(key, typ, value)
        for name, shape, position in tensors:
            header += struct.pack('<Q', len(name)) + name.encode() + struct.pack('<IQQIQ', 2, *shape, 24, position)
        header += bytes(-len(header) % 32)
        tables.append((layer, rows, offset, header, coverage, payloads))
        record['row_sha256'] = {str(row): hashlib.sha256(raw).hexdigest() for row, raw in payloads.items()}
    if output is not None:
        output.mkdir(parents=True)
        for layer, rows, offset, header, coverage, payloads in tables:
            path = output / f'engram{layer}-table.gguf'
            with path.open('xb') as f:
                f.write(header); f.write(coverage); f.truncate(len(header) + offset + rows * 264)
                for row, raw in payloads.items():
                    f.seek(len(header) + offset + row * 256); f.write(raw[:256])
                    f.seek(len(header) + offset + rows * 256 + row * 8); f.write(raw[256:])
            report['layers'][str(layer)].update(table=str(path), table_disk_bytes=path.stat().st_blocks * 512,
                                                table_header_sha256=hashlib.sha256(header).hexdigest())
        (output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--discover', action='store_true')
    a = parser.parse_args()
    if a.discover == (a.output_dir is not None):
        parser.error('Choose --discover or --output-dir')
    print(json.dumps(prepare(a.reference_root, a.tokens, a.output_dir), indent=2))

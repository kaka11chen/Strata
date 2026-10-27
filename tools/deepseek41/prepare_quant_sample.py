"""Fetch a bounded sample of the pinned GSQ-RCO IQ1_M package, never whole shards.

The output is a test artifact, not a runnable model. Reuse Strata's GGUF header
reader/writer. HTTP byte ranges must be honored; unsupported responses fail
before their body is read. Range hashes record what was received, not proof of
an entire remote shard's published SHA256. No Torch or model conversion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter, TYPE_IDS, _Tensor

REPO = 'ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF'
REVISION = 'cb02422c900df489cea5833e8a92171224059b3c'
ORIGINAL = '1bae0e1088993b9b82de08f59de5be5a34e42170'
SAMPLES = [(0, 84), (7, 6), (9, 7), (11, 4), (12, 3), (25, 1), (28, 6)]
FILES = {
    f'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-{i:05d}-of-00004.gguf': size
    for i, size in enumerate([39358830464, 39176113248, 39997334720, 2188841824], 1)
}
ENGRAM = 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
FILES[ENGRAM] = 110595290816
MAX_REQUEST = 8 * 1024 * 1024
MAX_TRANSFER = 96 * 1024 * 1024


def checked_range(response, start: int, length: int, total: int) -> bytes:
    expected = f'bytes {start}-{start + length - 1}/{total}'
    if (response.status != 206 or response.headers.get('Content-Range') != expected or
            response.headers.get('Content-Encoding', 'identity') != 'identity'):
        raise ValueError('Server did not honor the exact uncompressed byte range')
    raw = response.read(length + 1)
    if len(raw) != length:
        raise ValueError('Short/oversized range response')
    return raw


class Ranges:
    def __init__(self):
        self.records = []
        self.requested = 0

    def read(self, name: str, offset: int, size: int) -> bytes:
        total = FILES[name]
        if not 0 < size <= MAX_REQUEST or not 0 <= offset <= total - size:
            raise ValueError('Range exceeds request/file bounds')
        self.requested += size
        if self.requested > MAX_TRANSFER:
            raise ValueError('Sample exceeds 96 MiB transfer budget')
        # Distinct query keys avoid intermediary caches reusing a different Range.
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?sample={offset}-{size}'
        request = urllib.request.Request(url, headers={
            'Range': f'bytes={offset}-{offset + size - 1}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = checked_range(response, offset, size, total)
        self.records.append({'file': name, 'offset': offset, 'bytes': size,
                             'sha256': hashlib.sha256(raw).hexdigest()})
        return raw


def tensor_bytes(tensor):
    # The shared Python reader does not size the integer Engram metadata tensors.
    if tensor.type_id in (26, 27):
        return tensor.elements * (4 if tensor.type_id == 26 else 8)
    return tensor.expected_bytes()


def header(source: Ranges, name: str, path: Path) -> GGUFFile:
    raw = b''
    for size in (262144, 1048576, 4194304, 8388608):
        raw += source.read(name, len(raw), size - len(raw))
        path.write_bytes(raw)
        try:
            g = GGUFFile(path)
        except (struct.error, UnicodeError):
            continue
        if g.data_start > size:
            continue
        names = set()
        for tensor in g.tensors:
            n = tensor_bytes(tensor)
            if (tensor.name in names or n is None or n <= 0 or
                    g.data_start + tensor.offset + n > FILES[name]):
                raise ValueError('Invalid/duplicate tensor or remote file extent')
            names.add(tensor.name)
        return g
    raise ValueError('GGUF header exceeds bounded sample limit')


def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new sample directory')
    output.mkdir(parents=True)
    source = Ranges()
    try:
        headers = {name: header(source, name, output / (name + '.header')) for name in FILES}
        first = headers[next(iter(FILES))].metadata
        for key, expected in {'general.architecture': 'deepseek41', 'deepseek41.block_count': 40,
                'deepseek41.embedding_length': 5120, 'deepseek41.expert_count': 384,
                'deepseek41.expert_used_count': 6}.items():
            if first.get(key) != expected:
                raise ValueError('Wrong original source geometry: ' + key)
        for i, (name, g) in enumerate(headers.items()):
            if name != ENGRAM and (g.metadata.get('split.no') != i or g.metadata.get('split.count') != 4):
                raise ValueError('Wrong source shard identity')
        eg = headers[ENGRAM]
        if eg.metadata.get('general.architecture') != 'deepseek41-engram':
            raise ValueError('Wrong Engram source architecture')
        tables = {}
        for layer, count in ((1, 384006168), (14, 384016682)):
            matches = [t for t in eg.tensors if t.name == f'blk.{layer}.engram_embd.weight']
            if len(matches) != 1 or matches[0].type_id != 20 or matches[0].shape != [256, count]:
                raise ValueError('Unexpected original Engram table layout')
            tables[layer] = matches[0]
        tensors = {}
        for name, g in headers.items():
            for t in g.tensors:
                if t.name in tensors:
                    raise ValueError('Duplicate tensor across source shards')
                tensors[t.name] = (name, g, t)
        inventory = []
        pairs = Counter()
        for layer in range(40):
            roles = []
            for role in ('gate', 'up', 'down'):
                name = f'blk.{layer}.ffn_{role}_exps.weight'
                file, g, t = tensors[name]
                shape = [2304, 5120, 384] if role == 'down' else [5120, 2304, 384]
                if t.shape != shape or t.type_id not in (16, 19, 22, 29):
                    raise ValueError('Unexpected original expert geometry/quant type')
                roles.append(t.type_name)
                inventory.append({'tensor': name, 'file': file, 'type': t.type_name,
                                  'shape': t.shape, 'bytes': t.expected_bytes()})
            if roles[0] != roles[1]:
                raise ValueError('Gate/up use different formats')
            pairs[roles[0] + '/' + roles[2]] += 1
        writer = GGUFWriter()
        writer.add('general.architecture', 'deepseek41-quant-sample')
        writer.add('strata.deepseek41.quant_revision', REVISION)
        writer.add('strata.deepseek41.source_revision', ORIGINAL)
        writer.add('strata.deepseek41.sample_count', len(SAMPLES), 'u32')
        records = []
        sampled_pairs = set()
        for i, (layer, expert) in enumerate(SAMPLES):
            writer.add(f'sample.{i}.layer', layer, 'u32')
            writer.add(f'sample.{i}.expert', expert, 'u32')
            types = []
            for role in ('gate', 'up', 'down'):
                name = f'blk.{layer}.ffn_{role}_exps.weight'
                file, g, t = tensors[name]
                size = t.expected_bytes() // 384
                start = g.data_start + t.offset + expert * size
                raw = source.read(file, start, size)
                TYPE_IDS[t.type_name] = t.type_id
                writer.tensors.append(_Tensor(f'sample.{i}.{role}', t.shape[:2], t.type_name, raw))
                records.append({'layer': layer, 'expert': expert, 'role': role, 'type': t.type_name,
                                **source.records[-1]})
                types.append(t.type_name)
            sampled_pairs.add(types[0] + '/' + types[2])
        if sampled_pairs != set(pairs):
            raise ValueError('Samples do not cover the package format pairs')
        rows = [(1, 0), (1, 1), (1, 384006167), (14, 0), (14, 384016681)]
        row_bytes = 144
        raw = b''.join(source.read(ENGRAM, eg.data_start + tables[layer].offset + row * row_bytes, row_bytes)
                       for layer, row in rows)
        writer.add('strata.deepseek41.engram_sample_layers', [l for l, _ in rows], 'array:u32')
        writer.add('strata.deepseek41.engram_sample_rows', [r for _, r in rows], 'array:u32')
        TYPE_IDS['IQ4_NL'] = 20
        writer.tensors.append(_Tensor('engram.rows', [256, len(rows)], 'IQ4_NL', raw))
        fixture = output / 'quant-sample.gguf'
        writer.write(fixture)
        result = {'repo': REPO, 'revision': REVISION, 'declared_original_revision': ORIGINAL,
                  'scope': 'guarded sample, no full checkpoint or quality acceptance',
                  'remote_files': FILES, 'package_bytes': sum(FILES.values()),
                  'expert_type_counts': dict(Counter(x['type'] for x in inventory)),
                  'expert_format_pairs': dict(pairs), 'expert_inventory': inventory,
                  'samples': records, 'engram_rows': rows, 'ranges': source.records,
                  'requested_bytes': source.requested, 'fixture': str(fixture),
                  'fixture_bytes': fixture.stat().st_size,
                  'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest()}
        (output / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except Exception:
        (output / 'incomplete.json').write_text(json.dumps({'requested_bytes': source.requested,
            'completed_ranges': source.records}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir)
    print(json.dumps({k: result[k] for k in ('revision', 'requested_bytes', 'fixture_bytes',
        'fixture_sha256', 'expert_type_counts', 'expert_format_pairs')}, indent=2))

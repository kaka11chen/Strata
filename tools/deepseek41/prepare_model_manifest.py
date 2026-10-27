"""Describe prepared 1017 component files without copying or downloading weights.

The metadata-only GGUF reuses Strata's reader/writer. Paths are relative to the
manifest. This exporter declares guarded expert/Engram coverage; it does not
turn sparse fixtures into a complete checkpoint or authenticate their payloads.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter

REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
PREFIX = 'strata.deepseek41.manifest.'


def prepare(output: Path, reference: Path, fixtures: Path, supplemental: Path,
            boundary: Path, engram: Path, expert_gguf: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    writer = GGUFWriter()
    writer.add('general.architecture', 'deepseek41-manifest')
    def add(key, value):
        writer.add(PREFIX + key, value, 'u32' if type(value) is int else 'string')
    def location(key, path, directory=False):
        path = path.resolve(strict=True)
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f'Not the required source kind: {path}')
        add(key, os.path.relpath(path, output.resolve().parent))
    add('version', 1); add('revision', REVISION); add('profile', '1017-full40-context5')
    location('reference_root', reference, True); location('text_io', boundary)
    routed = {0: fixtures / 'real-decoder-routed0.gguf'}
    routed.update({l: fixtures / f'real-layer{l}-decoder-routed.gguf' for l in (1, 2, 3, 8, 14, 20)})
    for layer in range(40):
        key = f'layer.{layer}.'; decoder = fixtures / f'real-layer{layer}-decoder.gguf'
        for role, path in [('mhc', decoder if layer else fixtures / 'real-mhc0.gguf'),
                           ('attention', decoder if layer else fixtures / 'real-attention0.gguf'),
                           ('shared', decoder if layer else fixtures / 'real-shared0.gguf'),
                           ('router', routed.get(layer, decoder))]:
            location(key + role, path)
        add(key + 'router_weight', 'router.weight'); add(key + 'router_bias', 'router.bias')
    for layer in (1, 14):
        key = f'engram.{layer}.'
        location(key + 'projection', fixtures / f'real-engram{layer}-projection.gguf')
        location(key + 'token_map', fixtures / f'real-engram{layer}-reference.gguf')
        table = engram / f'engram{layer}-table.gguf'
        location(key + 'table', table)
        complete = GGUFFile(table).metadata.get('strata.deepseek41.engram_complete')
        if type(complete) is not int or complete not in (0, 1):
            raise ValueError('Engram table must declare coverage')
        add(key + 'complete', complete)
    add('experts_complete', int(expert_gguf is not None))
    ggufs = [(l, expert_gguf) for l in range(40)] if expert_gguf is not None else sorted(routed.items())
    populated = set()
    for i, (layer, path) in enumerate(ggufs):
        add(f'expert_gguf.{i}.layer', layer); location(f'expert_gguf.{i}.path', path)
        if expert_gguf is None:
            meta = GGUFFile(path).metadata
            if meta.get('strata.deepseek41.revision') != REVISION:
                raise ValueError('Wrong original routed revision')
            ids = meta.get('strata.deepseek41.populated_ids', meta.get('strata.deepseek41.selected_ids'))
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
                    any(type(e) is not int or not 0 <= e < 384 for e in ids)):
                raise ValueError('Missing/malformed guarded original expert IDs')
            populated.update((layer, e) for e in ids)
    add('expert_gguf_count', len(ggufs))
    raw = []
    if expert_gguf is None:
        for path in sorted(supplemental.glob('*.mxfp4')):
            match = re.fullmatch(r'layer-(\d+)-expert-(\d+)\.mxfp4', path.name)
            if not match: raise ValueError(f'Unrecognized original expert file: {path.name}')
            layer, expert = map(int, match.groups())
            if not 0 <= layer < 40 or not 0 <= expert < 384: raise ValueError('Raw expert ID outside original geometry')
            if (layer, expert) in populated: continue
            if path.stat().st_size != 18800640: raise ValueError('Wrong original expert file size')
            raw.append((layer, expert, path))
    for i, (layer, expert, path) in enumerate(raw):
        add(f'expert_raw.{i}.layer', layer); add(f'expert_raw.{i}.expert', expert); location(f'expert_raw.{i}.path', path)
    add('expert_raw_count', len(raw))
    writer.write(output)
    return {'manifest': str(output), 'expert_gguf_layers': len(ggufs), 'raw_experts': len(raw),
            'experts_complete_declared': expert_gguf is not None, 'revision': REVISION}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'reference-root', 'fixture-dir', 'supplemental-dir', 'boundary-fixture', 'engram-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expert-gguf', type=Path)
    a = p.parse_args()
    print(prepare(a.output, a.reference_root, a.fixture_dir, a.supplemental_dir, a.boundary_fixture, a.engram_dir, a.expert_gguf))


if __name__ == '__main__':
    main()

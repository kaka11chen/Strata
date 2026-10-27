"""Verify native token-to-logits captures against the pinned CPU boundaries.

The final residual is produced by the native quantized backbone. This verifies
its embedding and output head, not independent official full-backbone parity.
No model downloads or runtime Torch dependency are introduced.
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

from prepare_expert_fixture import Checkpoint
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION


def verify(model_dir: Path, reference: Path, capture: Path, generation: bool = False) -> dict:
    for path, expected in [(model_dir / 'config.json', CONFIG_HASH), (reference, SOURCE_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Not the pinned 1017 source: {path}')
    provenance = json.loads((model_dir / 'provenance.json').read_text())
    if provenance['model_revision'] != REVISION:
        raise ValueError('Wrong source revision')
    torch.set_num_threads(4)
    source = ast.parse(reference.read_bytes())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef)
               and node.name in {'ParallelEmbedding', 'RMSNorm', 'ParallelHead'}]
    identity = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'make_identity_pre_mix')
    block = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'Block')
    collapse = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == 'hc_pre')
    wrapper = ast.ClassDef(name='ReferenceBlock', bases=[], keywords=[], body=[collapse], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[*classes, identity, wrapper], type_ignores=[]))
    ns = {'torch': torch, 'nn': nn, 'F': F, 'world_size': 1, 'rank': 0}
    exec(compile(module, str(reference), 'exec'), ns)
    checkpoint, values, hashes = Checkpoint(model_dir), {}, {}
    for name, shape in [('embed.weight', [129280, 5120]), ('norm.weight', [5120]), ('head.weight', [129280, 5120])]:
        if checkpoint.dtype(name) != 'BF16' or checkpoint.shape(name) != shape:
            raise ValueError(f'Wrong original tensor: {name}')
        digest = hashlib.sha256(checkpoint.raw(name)).hexdigest()
        if digest != provenance['tensors'][name]['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        hashes[name] = digest
        values[name] = torch.from_numpy(checkpoint.array(name, np.uint16).copy()).view(torch.bfloat16)
        if not torch.isfinite(values[name]).all():
            raise ValueError(f'Nonfinite original tensor: {name}')
    embed = ns['ParallelEmbedding'].__new__(ns['ParallelEmbedding']); nn.Module.__init__(embed)
    embed.weight = nn.Parameter(values['embed.weight'], requires_grad=False)
    norm = ns['RMSNorm'](5120, 1e-20)
    norm.weight = nn.Parameter(values['norm.weight'], requires_grad=False)
    head = ns['ParallelHead'].__new__(ns['ParallelHead']); nn.Module.__init__(head)
    head.weight = nn.Parameter(values['head.weight'].float(), requires_grad=False)
    block = ns['ReferenceBlock']()
    capture_hashes = {}

    def load(case, name, dtype, shape):
        path = capture / f'case{case}.{name}'
        raw = path.read_bytes()
        if len(raw) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError(f'Wrong capture length: {path}')
        data = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        if not np.isfinite(data).all():
            raise ValueError(f'Nonfinite capture: {path}')
        capture_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        return torch.from_numpy(data)

    def compare(actual, expected):
        delta = actual.double() - expected.double()
        relative = (delta.norm() / expected.double().norm().clamp_min(1e-15)).item()
        maximum, peak = delta.abs().max().item(), expected.abs().max().item()
        return {'relative_l2': relative, 'max_abs': maximum, 'reference_peak': peak,
                'argmax_native': actual.argmax().item(), 'argmax_reference': expected.argmax().item(),
                'passed': relative <= 2e-4 and maximum <= 2e-4 * (1 + peak)
                          and actual.argmax().item() == expected.argmax().item()}

    rows = []
    with torch.inference_mode():
        for case, ids in enumerate(([0, 455], None if generation else [1805])):
            if ids is None:
                # The second forward must consume the first independent CPU
                # reference maximum, not a forced token supplied by the caller.
                ids = [rows[-1]['official_head']['argmax_reference']]
            n = len(ids)
            tokens = load(case, 'tokens.i32', '<i4', (1, n)).long()
            if tokens.flatten().tolist() != ids:
                raise ValueError('Unexpected fixed token sequence')
            initial = load(case, 'initial.f32', '<f4', (1, n, 4))
            embedded = load(case, 'embedding.f32', '<f4', (1, n, 4, 5120))
            residual = load(case, 'residual.f32', '<f4', (1, n, 4, 5120))
            pre = load(case, 'coefficients.f32', '<f4', (1, n, 4))
            actual = load(case, 'logits.f32', '<f4', (129280,))
            expected_embedding = embed(tokens).unsqueeze(2).repeat(1, 1, 4, 1)
            embedding_exact = torch.equal(embedded, expected_embedding.float())
            initial_exact = torch.equal(initial, ns['make_identity_pre_mix'](expected_embedding, 4).float())
            residual_bf16 = torch.equal(residual, residual.bfloat16().float())
            normalized = norm(block.hc_pre(residual.bfloat16(), pre))
            official = head(normalized).flatten()
            last = normalized[:, -1].double()
            fp64 = torch.cat([F.linear(last, head.weight[start:start + 4096].double()).flatten()
                              for start in range(0, 129280, 4096)]).float()
            rows.append({'case': case, 'tokens': ids, 'embedding_exact': embedding_exact,
                         'initial_exact': initial_exact, 'residual_bf16': residual_bf16,
                         'official_head': compare(actual, official), 'fp64_dot': compare(actual, fp64)})
    passed = all(row['embedding_exact'] and row['initial_exact'] and row['residual_bf16']
                 and row['official_head']['passed'] and row['fp64_dot']['passed'] for row in rows)
    return {'passed': passed, 'revision': REVISION, 'official_source_sha256': SOURCE_HASH,
            'source_hashes': hashes, 'capture_hashes': capture_hashes, 'cases': rows,
            'torch_oracle_version': torch.__version__, 'greedy_feedback_checked': generation,
            'greedy_ids': [row['official_head']['argmax_reference'] for row in rows],
            'scope': 'Pinned unmodified CPU embedding/initialization and hc_pre/RMSNorm/ParallelHead on actual native final states; FP64 output-dot controls. Native quantized backbone is not an independent official full-model oracle.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--reference-model', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--generation', action='store_true', help='Require the second capture to consume the first CPU-reference greedy token')
    args = parser.parse_args()
    report = verify(args.model_dir, args.reference_model, args.capture_dir, args.generation)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)

"""Pinned 1017 original BF16 embedding/head and independent CPU boundary references.

Uses unmodified ParallelEmbedding, Block.hc_pre, RMSNorm, ParallelHead and
make_identity_pre_mix from the hash-pinned source. No backbone, tokenizer,
sampling, GPU reference or runtime Torch dependency is introduced.
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


def prepare(model_dir: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
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
    checkpoint, writer, records, values = Checkpoint(model_dir), GGUFWriter(), {}, {}
    TYPE_IDS["BF16"] = 30
    writer.add('general.architecture', 'deepseek41-text-io-fixture')
    writer.add('strata.deepseek41.text_io_profile', '1017-bf16-hc4-last-logits')
    writer.add('strata.deepseek41.model_revision', REVISION)
    writer.add('strata.deepseek41.official_source_sha256', SOURCE_HASH)
    for original, name, shape in [('embed.weight', 'token_embd.weight', [129280, 5120]),
                                  ('norm.weight', 'output_norm.weight', [5120]),
                                  ('head.weight', 'output.weight', [129280, 5120])]:
        if checkpoint.dtype(original) != 'BF16' or checkpoint.shape(original) != shape:
            raise ValueError(f'Malformed original tensor: {original}')
        raw = checkpoint.raw(original).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != provenance['tensors'][original]['sha256']:
            raise ValueError(f'Source hash mismatch: {original}')
        writer.tensors.append(_Tensor(name, list(reversed(shape)), 'BF16', raw))
        records[original] = digest
        values[original] = torch.from_numpy(checkpoint.array(original, np.uint16).copy()).view(torch.bfloat16)
        if not torch.isfinite(values[original]).all():
            raise ValueError(f'Nonfinite original tensor: {original}')
    embed = ns['ParallelEmbedding'].__new__(ns['ParallelEmbedding']); nn.Module.__init__(embed)
    embed.weight = nn.Parameter(values['embed.weight'], requires_grad=False)
    norm = ns['RMSNorm'](5120, 1e-20)
    norm.weight = nn.Parameter(values['norm.weight'], requires_grad=False)
    head = ns['ParallelHead'].__new__(ns['ParallelHead']); nn.Module.__init__(head)
    head.weight = nn.Parameter(values['head.weight'].float(), requires_grad=False)
    block = ns['ReferenceBlock']()

    def add(name, value):
        value = value.detach().float().contiguous()
        writer.tensors.append(_Tensor(name, list(reversed(value.shape)), 'F32', value.numpy().tobytes()))

    ids = [torch.tensor([[0, 455, 129279, 455]]), torch.tensor([[2]]), torch.tensor([[1805, 20852]])]
    reference_stats = []
    with torch.inference_mode():
        for case, tokens in enumerate(ids):
            residual = embed(tokens).unsqueeze(2).repeat(1, 1, 4, 1)
            initial = ns['make_identity_pre_mix'](residual, 4)
            writer.add(f'embed.{case}.tokens', tokens.flatten().tolist(), 'array:u32')
            add(f'embed.{case}.residual', residual.flatten())
            add(f'embed.{case}.pre', initial.flatten())
        generator = torch.Generator().manual_seed(1017)
        cases = []
        for count, scale in [(3, 1.), (1, 0.), (2, 1e-12), (1, 16.), (1, 1.)]:
            residual = (torch.randn((1, count, 4, 5120), generator=generator) * scale).bfloat16()
            pre = torch.tensor([.125, .625, .25, .5]).reshape(1, 1, 4).repeat(1, count, 1)
            if count > 1:
                pre[0, 0] = torch.tensor([1., 0., 0., 0.])
            cases.append((residual, pre))
        cases[4] = (cases[4][0], torch.tensor([[[1., -1., .25, -.5]]]))
        residual = embed(ids[0]).unsqueeze(2).repeat(1, 1, 4, 1)
        cases.append((residual, ns['make_identity_pre_mix'](residual, 4)))
        for case, (residual, pre) in enumerate(cases):
            collapsed = block.hc_pre(residual, pre)
            normalized = norm(collapsed)
            logits = head(normalized).reshape(-1)
            last = normalized[:, -1].double()
            accurate = torch.cat([F.linear(last, head.weight[start:start + 4096].double()).reshape(-1)
                                  for start in range(0, 129280, 4096)]).float()
            for name, value in [('residual', residual), ('pre', pre), ('collapsed', collapsed),
                                ('normalized', normalized), ('logits', logits), ('logits_fp64', accurate)]:
                add(f'head.{case}.{name}', value.flatten())
            delta = (logits.double() - accurate.double())
            reference_stats.append({'tokens': residual.shape[1], 'fp32_vs_fp64_max_abs': delta.abs().max().item(),
                                    'fp32_vs_fp64_rel_l2': (delta.norm() / accurate.double().norm().clamp_min(1e-30)).item(),
                                    'argmax_fp32': logits.argmax().item(), 'argmax_fp64': accurate.argmax().item()})
    writer.add('strata.deepseek41.source_hashes', json.dumps(records, sort_keys=True))
    writer.add('strata.deepseek41.embedding_cases', len(ids), 'u32')
    writer.add('strata.deepseek41.head_cases', len(cases), 'u32')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output)
    return {'fixture': str(output), 'revision': REVISION, 'official_source_sha256': SOURCE_HASH,
            'source_hashes': records, 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'embedding_cases': len(ids),
            'head_cases': len(cases), 'head_reference_stats': reference_stats, 'vocab_size': 129280,
            'scope': 'Unmodified pinned CPU embedding, one-hot initialization, hc_pre, RMSNorm and FP32 ParallelHead; independent FP64 dot controls. No backbone or generation reference.',
            'torch_oracle_version': torch.__version__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--reference-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir, args.reference_model, args.output), indent=2))

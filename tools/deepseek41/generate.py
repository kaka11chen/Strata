"""Experimental raw completion for the guarded 1017 native generation fixture.

Tokenization is at the process boundary; C++ owns all forwards and greedy
sampling. This is batch-one greedy completion, without a chat template. Missing
original expert/Engram coverage is an error, never a substituted weight. The
currently validated prompt is ' The' with two new tokens, not arbitrary chat.
Requires the `tokenizers` package; the runtime entry does not import Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from tokenizers import Tokenizer

REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
TOKENIZER_SHA256 = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
CONFIG_SHA256 = '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'
VOCAB_SIZE, BOS_ID, EOS_ID, CONTEXT_CAPACITY = 129280, 0, 1, 5


def load_tokenizer(root: Path) -> Tokenizer:
    contents = {}
    for name, digest in [('tokenizer.json', TOKENIZER_SHA256), ('config.json', CONFIG_SHA256)]:
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Not the pinned DeepSeek-V4.1-Flash-1017 source: {name}')
        contents[name] = raw.decode('utf-8')
    tokenizer = Tokenizer.from_str(contents['tokenizer.json'])
    cfg = json.loads(contents['config.json'])
    if (tokenizer.get_vocab_size() != VOCAB_SIZE or cfg['bos_token_id'] != BOS_ID or
            cfg['eos_token_id'] != EOS_ID or cfg['text_config']['vocab_size'] != VOCAB_SIZE):
        raise ValueError('Pinned vocabulary or special token contract changed')
    return tokenizer


def encode_prompt(tokenizer: Tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    # The pinned postprocessor adds no BOS; specify it exactly once at this API
    # boundary. This raw-completion convention is not an official chat template.
    ids = [BOS_ID, *tokenizer.encode(prompt, add_special_tokens=False).ids]
    if (type(max_new_tokens) is not int or not 0 <= max_new_tokens <= CONTEXT_CAPACITY or
            len(ids) > CONTEXT_CAPACITY or len(ids) + max(0, max_new_tokens - 1) > CONTEXT_CAPACITY):
        raise ValueError('Prompt and generation budget exceed the guarded five-input-token fixture capacity')
    return ids


def decode_result(tokenizer: Tokenizer, stdout: str, prompt: str, ids: list[int], budget: int) -> dict:
    lines = [line.removeprefix('GENERATION_RESULT ') for line in stdout.splitlines()
             if line.startswith('GENERATION_RESULT ')]
    if len(lines) != 1:
        raise ValueError('Native runner did not return exactly one generation result')
    result = json.loads(lines[0])
    generated, eos = result['generated_ids'], result['stopped_on_eos']
    if (result['prompt_ids'] != ids or type(generated) is not list or
            any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in generated) or
            type(eos) is not bool or len(generated) > budget or EOS_ID in generated[:-1] or
            eos != (bool(generated) and generated[-1] == EOS_ID) or
            (not eos and len(generated) != budget) or
            result['evaluated_tokens'] != (len(ids) + len(generated) - 1 if generated else 0) or
            result['context_capacity'] != CONTEXT_CAPACITY):
        raise ValueError('Native generation result violates the bounded request contract')
    # Decode the sequence as a whole so UTF-8 bytes split across tokens survive.
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    return {**result, 'revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
            'prompt': prompt, 'completion': completion, 'text': prompt + completion,
            'finish_reason': 'eos' if eos else 'length', 'sampling': 'greedy',
            'scope': ('model manifest, raw completion, context5, no chat template' if 'model_manifest' in result
                      else 'guarded original-weight fixture, raw completion, no chat template')}


def decode_memory_plan(stdout: str) -> dict:
    lines = [line.removeprefix('MEMORY_PLAN ') for line in stdout.splitlines()
             if line.startswith('MEMORY_PLAN ')]
    if len(lines) != 1:
        raise ValueError('Native runner did not return exactly one memory plan')
    report = json.loads(lines[0])
    if report.get('profile') != '1017-full40-context5' or type(report.get('admitted')) is not bool:
        raise ValueError('Unexpected native memory admission result')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, help='Use the independent native manifest runner')
    for name in ('reference-root', 'fixture-dir', 'supplemental-dir', 'boundary-fixture', 'engram-dir'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=2)
    parser.add_argument('--gpu-budget-mib', type=int, default=20480,
                        help='Startup GPU allowance, including 2048 MiB CUDA/cuBLAS headroom')
    parser.add_argument('--host-budget-mib', type=int, default=65536,
                        help='Startup host allowance, including 4096 MiB system/library headroom')
    parser.add_argument('--expert-slots', type=int, default=2, help='Existing GPU expert slots per layer')
    parser.add_argument('--cpu-expert-cache-mib', type=int, default=0,
                        help='Global CPU expert cache allowance including metadata; 0 disables caching')
    parser.add_argument('--expert-gguf', type=Path,
                        help='Optional complete 1017 MXFP4 expert GGUF; dense and Engram fixture paths remain required')
    parser.add_argument('--plan', action='store_true', help='Show admission bounds and live availability without loading weights')
    args = parser.parse_args()
    if not (1 <= args.gpu_budget_mib <= 2147483647 and 1 <= args.host_budget_mib <= 2147483647 and
            1 <= args.expert_slots <= 384 and 0 <= args.cpu_expert_cache_mib <= 2147483647):
        parser.error('Memory allowances must be positive MiB, CPU cache nonnegative; expert slots must be 1..384')
    if args.model_manifest is not None:
        if any(value is not None for value in (args.reference_root, args.fixture_dir, args.supplemental_dir,
                                                args.boundary_fixture, args.engram_dir, args.expert_gguf)):
            parser.error('--model-manifest provides all model paths; do not combine it with legacy path options')
        # Reuse the existing header-only Python reader; no Torch or weight payload read.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from gguf_reader import GGUFFile
        path = args.model_manifest.resolve()
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Model manifest exceeds 8 MiB metadata limit')
        manifest = GGUFFile(path)
        metadata = manifest.metadata; key = 'strata.deepseek41.manifest.'
        if (manifest.tensors or metadata.get('general.architecture') != 'deepseek41-manifest' or
                metadata.get(key + 'version') != 1 or metadata.get(key + 'revision') != REVISION or
                metadata.get(key + 'profile') != '1017-full40-context5'):
            raise ValueError('Not the pinned 1017 model manifest schema/profile')
        reference = metadata.get(key + 'reference_root')
        if not isinstance(reference, str) or not reference or '\0' in reference:
            raise ValueError('Model manifest requires a tokenizer/config directory')
        tokenizer = load_tokenizer(path.parent / reference)
        ids = encode_prompt(tokenizer, args.prompt, args.max_new_tokens)
        command = [str(args.runner.resolve()), '--manifest', str(path),
                   '--gpu-budget-mib', str(args.gpu_budget_mib), '--host-budget-mib', str(args.host_budget_mib),
                   '--expert-slots', str(args.expert_slots), '--cpu-expert-cache-mib', str(args.cpu_expert_cache_mib),
                   '--max-new-tokens', str(args.max_new_tokens)]
        if args.plan: command.append('--plan')
        command += ['--tokens', *map(str, ids)]
    else:
        if any(value is None for value in (args.reference_root, args.fixture_dir, args.supplemental_dir,
                                            args.boundary_fixture, args.engram_dir)):
            parser.error('Provide --model-manifest or all legacy model paths')
        tokenizer = load_tokenizer(args.reference_root)
        ids = encode_prompt(tokenizer, args.prompt, args.max_new_tokens)
        command = [str(args.runner.resolve()), '--layers', '40', '--memory-plan' if args.plan else '--generate-budgeted',
                   '--cpu-expert-cache-mib', str(args.cpu_expert_cache_mib),
                   str(args.fixture_dir), str(args.supplemental_dir), str(args.boundary_fixture),
                   str(args.engram_dir), str(args.gpu_budget_mib), str(args.host_budget_mib),
                   str(args.expert_slots), str(args.max_new_tokens), *map(str, ids)]
        if args.expert_gguf is not None:
            command[4:4] = ['--expert-gguf', str(args.expert_gguf)]
    env = dict(os.environ); env.setdefault('STRATA_IO_THREADS', '2')
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if args.plan:
        if result.returncode and not any(line.startswith('MEMORY_PLAN ') for line in result.stdout.splitlines()):
            raise RuntimeError(f'Native memory inspection failed ({result.returncode}):\n{result.stdout[-4096:]}')
        report = decode_memory_plan(result.stdout)
        if result.returncode and report['admitted']:
            raise RuntimeError(f'Native memory inspection failed:\n{result.stdout[-4096:]}')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report['admitted'] else 1)
    if result.returncode:
        raise RuntimeError(f'Native generation failed ({result.returncode}):\n{result.stdout[-4096:]}')
    completion = decode_result(tokenizer, result.stdout, args.prompt, ids, args.max_new_tokens)
    plan = decode_memory_plan(result.stdout)
    if not plan['admitted'] or 'memory' not in completion:
        raise ValueError('Native completion omitted its admitted memory accounting')
    completion['memory_plan'] = plan
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

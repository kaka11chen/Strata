"""Bounded usability prompts through the validated native Strata engine.

Reuses a successful smoke_quantized.py result and its pack; no downloads, tuning
or new inference code. Optional pinned candidates can emit per-phase store
diagnostics. One resident engine, fresh KV per request, warm expert/OS
caches in the recorded order. Quality review is separate from protocol success.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from smoke_quantized import GpuBudget, UPSTREAM_REVISION, proc_memory, verified_files, write_json
from prepare_quant_sample import REPO, REVISION

ENGINE_SHA256 = '7d1eebfd7395f3b29d46948b7eb7d0c56654db51c6af1bef334713beb0257946'
TEXT_FLOW_ENGINE_SHA256 = 'a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96'
TEXT_MEMORY_ENGINE_SHA256 = 'd6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553'
CASES = [
    {'id': 'zh_fact', 'category': 'Chinese QA',
     'prompt': '中国的首都是哪里？请用一句话回答。', 'expected': '北京'},
    {'id': 'zh_explanation', 'category': 'Chinese explanation',
     'prompt': '用不超过50个汉字解释内存和显存的区别。',
     'expected': 'Distinguish system/CPU memory from GPU memory; coherent Chinese, at most 50 Han characters.'},
    {'id': 'arithmetic', 'category': 'Simple arithmetic',
     'prompt': '一本书18元，买3本付100元，应找回多少元？请写出算式和答案。',
     'expected': '100 - 18 * 3 = 46 元'},
    {'id': 'ordering', 'category': 'Simple reasoning',
     'prompt': '小明比小红高，小红比小刚高。三人中谁最高、谁最矮？用一句话回答。',
     'expected': '小明最高，小刚最矮'},
    {'id': 'python', 'category': 'Short code',
     'prompt': '写一个Python函数 sum_even(nums)，返回整数列表中所有偶数的和。只输出代码。',
     'expected': 'A valid sum_even function: [] -> 0, [1,2,3,4] -> 6, [-4,-1,0,2] -> -2.'},
]
LONG_CASES = [
    {'id': 'long_retrieval', 'category': 'Longer input and factual extraction', 'max_new_tokens': 128,
     'prompt': '请仅根据以下虚构仓库记录回答，不补充外部信息。所有数量均为件。\n'
               '青松仓：负责人林青，编号A17，周一入库24，周二入库18，周三出库9，周四入库7，周五出库5。\n'
               '白鹭仓：负责人陈白，编号B28，周一入库31，周二出库6，周三入库12，周四出库8，周五入库3。\n'
               '红枫仓：负责人赵红，编号C39，周一入库16，周二入库25，周三出库11，周四入库4，周五出库6。\n'
               '蓝海仓：负责人吴蓝，编号D40，周一入库28，周二出库7，周三入库14，周四出库9，周五入库2。\n'
               '各仓周初库存均为0，入库增加库存，出库减少库存。四个仓库独立计算，不能把其他仓的数加进来。\n'
               '只输出一个JSON对象，键恰好为 warehouse、manager、code、received、shipped、remaining。\n'
               '查询对象是红枫仓。warehouse填仓名，manager填负责人，code填编号；received为本周入库总数，'
               'shipped为本周出库总数，remaining为周五结束库存。不要输出分析过程。',
     'expected': {'warehouse': '红枫仓', 'manager': '赵红', 'code': 'C39', 'received': 45, 'shipped': 17, 'remaining': 28}},
    {'id': 'long_explanation', 'category': 'Sustained Chinese explanation', 'max_new_tokens': 256,
     'prompt': '给刚接触本地大模型的人写一份内存使用说明。请按1到6编号写六条，每条25到35个汉字，'
               '分别解释：CPU内存的用途、GPU显存的用途、为什么量化权重仍可能超过显存、'
               '内存缓存怎样减少SSD读取、为什么缓存变大也未必大幅提速、为什么要给系统保留可用内存。'
               '每条说清原因，用完整句子，避免口号，不要标题、开场白或总结。',
     'expected': 'Six coherent numbered explanations of the six requested memory topics; inspect completeness, relevance and requested per-item Han-character range.'},
    {'id': 'long_python', 'category': 'Longer code completion', 'max_new_tokens': 256,
     'prompt': '只输出Python代码，定义两个函数，不要导入库。sum_even(nums)返回整数列表中全部偶数的和，'
               '无偶数时返回0。summarize_even(nums)返回一个字典，键恰好是count、total、minimum、maximum、mean。'
               '只统计偶数：count是个数，total是总和，minimum和maximum是最小和最大值，mean是算术平均值。'
               '没有偶数时count和total为0，其他三个值为None。两个函数都不能修改输入列表。'
               '每个函数必须有一行简短的中文docstring，准确说明功能。',
     'expected': 'Two valid pure functions with docstrings; check empty, odd-only, negative, duplicate, zero and large-integer lists, including input preservation.'},
]
SHORT_BOUNDARY_CASES = [
    {**CASES[0], 'id': 'zh_fact_16', 'prompt': '请问，' + CASES[0]['prompt']},
    {**CASES[0], 'id': 'zh_fact_17', 'prompt': '请问一下，' + CASES[0]['prompt']},
]
# Resident-cache check: original three tasks, then the identical retrieval with fresh KV.
RESIDENT_CASES = LONG_CASES + [{**LONG_CASES[0], 'id': 'long_retrieval_repeat'}]
# Same task multiset in opposite orders; each repeat still starts with fresh KV.
RESIDENT_ORDERS = {
    name: list(pair) + [{**case, 'id': case['id'] + '_repeat'} for case in pair]
    for name, pair in (
        ('short-first', (CASES[0], LONG_CASES[0])),
        ('long-first', (LONG_CASES[0], CASES[0])),
        ('boundary16-first', (SHORT_BOUNDARY_CASES[0], LONG_CASES[0])),
        ('boundary17-first', (SHORT_BOUNDARY_CASES[1], LONG_CASES[0])),
        ('code-explain-repeat', (LONG_CASES[2], LONG_CASES[1])),
    )
}
RESIDENT_ORDERS.update({
    'short-repeat': [CASES[0], {**CASES[0], 'id': 'zh_fact_repeat'}],
    'long-then-short': [LONG_CASES[0], CASES[0]],
    'explain-code': [LONG_CASES[1], LONG_CASES[2]],
    'code-explain': [LONG_CASES[2], LONG_CASES[1]],
})
TEMPLATE = '<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>'


def prepare_case(tok, case, plain_user=False):
    cap = case.get('max_new_tokens', 64)
    if type(cap) is not int or not 1 <= cap <= 256 or not case['prompt'].strip():
        raise ValueError('Prompt must be nonempty and output limit must be 1..256')
    rendered = TEMPLATE.format(prompt=case['prompt'])
    kwargs = {}
    if plain_user:
        start = TEMPLATE.index('{prompt}')
        kwargs['plain'] = ((start, start + len(case['prompt'])),)
    ids = tok.encode(rendered, parse_special=True, **kwargs)
    if tok.decode(ids) != rendered or not 0 < len(ids) <= 1024 - cap:
        raise ValueError('Prompt and output limit exceed context 1024, or tokenization did not round-trip')
    return {**case, 'max_new_tokens': cap, 'rendered_prompt': rendered, 'prompt_ids': ids}


def io_bytes(pid):
    return int(next(line.split()[1] for line in Path(f'/proc/{pid}/io').read_text().splitlines()
                    if line.startswith('read_bytes:')))


def done_metrics(line, token_times, sent, prompt_tokens, max_new_tokens=64):
    parts = line.split()
    if len(parts) != 11 or parts[0] != 'DONE':
        raise ValueError('Malformed DONE: ' + line)
    count, prompt, prefill_ms, decode_ms = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
    if count != len(token_times) or not 0 < count <= max_new_tokens or prompt != prompt_tokens:
        raise ValueError('DONE token counts differ from the request or stream')
    if parts[5] not in ('stop', 'length') or int(parts[6]) or int(parts[7]) or int(parts[8]):
        raise ValueError('Cancelled, speculative or reused request is outside this baseline')
    if prefill_ms < 0 or decode_ms < 0:
        raise ValueError('Negative native timing')
    # With DSpark disabled, the first output is sampled from prefill. Only the
    # remaining outputs have decode steps, even though upstream STAT counts all.
    steps = count - 1
    return {'generated_tokens_including_eos': count, 'decode_steps': steps,
            'native_prefill_ms': prefill_ms, 'native_decode_ms': decode_ms,
            'decode_tokens_per_s': steps * 1000 / decode_ms if steps and decode_ms else None,
            'stream_tokens_per_s': steps / (token_times[-1] - token_times[0]) if steps else None,
            'first_token_s': token_times[0] - sent, 'finish_reason': parts[5], 'reused_tokens': 0}


def parse_store_line(line):
    fields = {}
    for item in line.split()[1:]:
        key, value = item.split('=', 1)
        if key in fields:
            raise ValueError('Duplicate STORE field: ' + key)
        fields[key] = value
    integer_fields = ('tokens', 'hits', 'misses', 'submitted_bytes', 'evictions', 'read_fails',
                      'prefetched', 'prefetch_used', 'prefetch_wasted', 'slots', 'slots_max')
    float_fields = ('wait_ms', 'cpu_lane_ms')
    if set(fields) != {'phase', *integer_fields, *float_fields} or fields['phase'] not in ('prefill', 'decode'):
        raise ValueError('Incomplete or unknown STORE fields/phase')
    for key in integer_fields:
        fields[key] = int(fields[key])
    for key in float_fields:
        fields[key] = float(fields[key])
    if any(not math.isfinite(fields[key]) or fields[key] < 0 for key in (*integer_fields, *float_fields)):
        raise ValueError('Non-finite or negative STORE counter')
    if fields['slots'] > fields['slots_max']:
        raise ValueError('STORE slots exceed their ceiling')
    lookups, tokens = fields['hits'] + fields['misses'], fields['tokens']
    fields['hit_ratio'] = fields['hits'] / lookups if lookups else None
    fields['submitted_mb_per_token'] = fields['submitted_bytes'] / 1e6 / tokens if tokens else None
    fields['wait_ms_per_token'] = fields['wait_ms'] / tokens if tokens else None
    return fields


def peak_memory(samples):
    keys = ('rss_bytes', 'hwm_bytes', 'swap_bytes', 'gpu_used_excluding_hold_bytes')
    return {**{key: max(s[key] for s in samples) for key in keys},
            'minimum_host_available_bytes': min(s['host_available_bytes'] for s in samples)}


def run_engine(cmd, env, out, budget, tok, report, timeout, on_token=None, host_floor_bytes=0,
               request_source=None, on_case_end=None, resource_diagnostics=False):
    start = time.monotonic()
    samples, lines = [], queue.Queue()
    phase = 'startup'
    progress = sys.stderr if on_token is not None else sys.stdout
    with (out / 'engine.stderr').open('w') as stderr, (out / 'engine.stdout').open('w') as stdout, \
            (out / 'events.jsonl').open('w') as events, (out / 'memory.jsonl').open('w') as memory:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                env=env, start_new_session=True)
        report['engine_pid'] = proc.pid

        def reader():
            for raw in iter(proc.stdout.readline, b''):
                lines.put((time.monotonic(), raw.decode('utf-8').rstrip('\n')))
            lines.put((time.monotonic(), None))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        def diagnose_resource_failure(reason, trigger):
            # The helper is bounded and opt-in; diagnostics must never replace a guard failure.
            try:
                from resource_diagnostics import capture_resource_failure
                report['resource_diagnostic'] = capture_resource_failure(proc.pid, out, trigger, reason)
            except Exception as error:
                report['resource_diagnostic'] = {'status': 'capture_failed',
                                                 'error': type(error).__name__ + ': ' + str(error)}

        def sample():
            status = proc_memory(Path(f'/proc/{proc.pid}/status'))
            free, _ = budget.memory()
            s = {'elapsed_s': time.monotonic() - start, 'phase': phase,
                 'rss_bytes': status.get('VmRSS', 0), 'hwm_bytes': status.get('VmHWM', 0),
                 'swap_bytes': status.get('VmSwap', 0),
                 'gpu_used_excluding_hold_bytes': budget.info['physical_bytes'] - free - budget.held,
                 'host_available_bytes': proc_memory(Path('/proc/meminfo'))['MemAvailable']}
            samples.append(s)
            memory.write(json.dumps(s) + '\n')
            memory.flush()
            if host_floor_bytes:
                if s['host_available_bytes'] < host_floor_bytes or s['swap_bytes']:
                    if resource_diagnostics:
                        diagnose_resource_failure('host_floor_or_engine_swap', s)
                    raise RuntimeError('Host memory floor reached or engine used swap; stopping')
                if s['gpu_used_excluding_hold_bytes'] > budget.info['target_bytes']:
                    if resource_diagnostics:
                        diagnose_resource_failure('gpu_capacity', s)
                    raise RuntimeError('Observed GPU use exceeded the capacity bound; stopping')

        def record_line(stamp, line):
            if line is None:
                raise RuntimeError('Engine exited before completing ' + phase)
            stdout.write(line + '\n')
            stdout.flush()
            events.write(json.dumps({'elapsed_s': stamp - start, 'phase': phase, 'line': line}) + '\n')
            events.flush()
            if line.startswith('ERR '):
                raise RuntimeError(line)
            return stamp, line

        def idle_tick():
            if not samples or time.monotonic() - start - samples[-1]['elapsed_s'] >= 1:
                sample()
            if proc.poll() is not None:
                raise RuntimeError('Engine exited while waiting for input: ' + str(proc.returncode))
            while True:
                try:
                    stamp, line = lines.get_nowait()
                except queue.Empty:
                    break
                record_line(stamp, line)
                if line.startswith(('READY ', 'T ', 'STAT ', 'STORE ', 'DONE ')):
                    raise ValueError('Unexpected native protocol output while waiting for input')

        def case_stream():
            nonlocal phase
            if request_source is None:
                yield from report['cases']
                return
            while True:
                phase = 'idle'
                report['session_state'] = 'waiting_for_input'
                write_json(out / 'result.json', report)
                raw = request_source(idle_tick)
                if raw is None:
                    return
                try:
                    case = prepare_case(tok, {**raw, 'id': 'text-%d' % (len(report['cases']) + 1)}, plain_user=True)
                except ValueError as error:
                    report.setdefault('input_rejections', []).append({'prompt_characters': len(raw['prompt']), 'error': str(error)})
                    write_json(out / 'result.json', report)
                    print('输入未提交：' + str(error), file=progress, flush=True)
                    continue
                report['cases'].append(case)
                yield case

        def next_line(deadline):
            while True:
                if not samples or time.monotonic() - start - samples[-1]['elapsed_s'] >= 1:
                    sample()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Engine timed out during ' + phase)
                try:
                    stamp, line = lines.get(timeout=min(1, remaining))
                except queue.Empty:
                    continue
                return record_line(stamp, line)

        try:
            deadline = start + timeout
            while True:
                stamp, line = next_line(deadline)
                if line.startswith('READY '):
                    if line.split()[1] != '1024':
                        raise ValueError('Engine reported unexpected context capacity')
                    report['startup_to_ready_s'] = stamp - start
                    break
            print('Engine ready in %.2f s' % report['startup_to_ready_s'], file=progress, flush=True)
            for case in case_stream():
                phase = case['id']
                if request_source is not None:
                    report['session_state'] = 'generating'
                sample()
                before_io = io_bytes(proc.pid)
                sent = time.monotonic()
                case['sent_elapsed_s'] = sent - start
                case['status'] = 'running'
                write_json(out / 'result.json', report)
                print('Request:', phase, 'prompt tokens:', len(case['prompt_ids']), file=progress, flush=True)
                max_new_tokens = case.get('max_new_tokens', 64)
                proc.stdin.write(('GEN %d temperature=0 spec=0 ' % max_new_tokens + ','.join(map(str, case['prompt_ids'])) + '\n').encode())
                proc.stdin.flush()
                token_times, ids, stats, store = [], [], [], {}
                while True:
                    stamp, line = next_line(sent + timeout)
                    if line.startswith('T '):
                        token = int(line[2:])
                        if len(ids) >= max_new_tokens or not 0 <= token < len(tok.tokens):
                            raise ValueError('Invalid token or too many output tokens')
                        ids.append(token)
                        token_times.append(stamp)
                        case['output_ids'] = ids
                        if on_token is not None:
                            on_token(tok, token)
                        if len(ids) == 1 or len(ids) % 8 == 0:
                            case['output_text_so_far'] = tok.decode(ids)
                            write_json(out / 'result.json', report)
                    elif line.startswith('STAT '):
                        stats.append(dict((k, float(v)) for k, v in (p.split('=', 1) for p in line.split()[1:])))
                    elif line.startswith('STORE '):
                        if env.get('DS4_STORE_STATS') != '1':
                            raise ValueError('STORE output appeared with diagnostics disabled')
                        counters = parse_store_line(line)
                        if counters['phase'] in store:
                            raise ValueError('Duplicate STORE phase')
                        store[counters['phase']] = counters
                    elif line.startswith('DONE '):
                        case.update(done_metrics(line, token_times, sent, len(case['prompt_ids']), max_new_tokens))
                        case['request_wall_s'] = stamp - sent
                        case['process_read_bytes'] = io_bytes(proc.pid) - before_io
                        break
                if env.get('DS4_STORE_STATS') == '1':
                    if set(store) != {'prefill', 'decode'}:
                        raise ValueError('Missing per-phase STORE diagnostics')
                    if store['prefill']['tokens'] != len(case['prompt_ids']) or store['decode']['tokens'] != case['decode_steps']:
                        raise ValueError('STORE token counts disagree with the measured phases')
                    for key in store:
                        wall_ms = case['native_' + key + '_ms']
                        store[key]['wait_over_phase_wall'] = store[key]['wait_ms'] / wall_ms if wall_ms else None
                    case['store_stats'] = store
                if not stats or 'disk_mb' not in stats[-1] or 'vram_hit' not in stats[-1]:
                    raise ValueError('Native disk/cache statistics missing')
                if (case['finish_reason'] == 'stop') != (ids[-1] == 1) or 1 in ids[:-1]:
                    raise ValueError('EOS and finish reason disagree')
                sample()
                case.update(status='measured', output_text=tok.decode(ids),
                            token_elapsed_s=[t - sent for t in token_times], native_stats=stats,
                            observed_memory=peak_memory([s for s in samples if s['phase'] == phase]))
                if on_token is not None:
                    on_token(tok, None)
                    case['completion_text'] = tok.decode([t for t in ids if t != 1])
                case.pop('output_text_so_far', None)
                write_json(out / 'result.json', report)
                print('%s: %d tokens, first %.2f s, decode %s tok/s, %s' % (
                    phase, len(ids), case['first_token_s'], case['decode_tokens_per_s'], case['finish_reason']), file=progress, flush=True)
                if on_case_end is not None:
                    on_case_end(case)
            phase = 'shutdown'
            proc.stdin.write(b'QUIT\n')
            proc.stdin.flush()
            proc.stdin.close()
            report['engine_returncode'] = proc.wait(timeout=30)
            if report['engine_returncode']:
                raise RuntimeError('Engine failed on shutdown')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            thread.join(timeout=1)
            proc.stdout.close()
            if not proc.stdin.closed:
                proc.stdin.close()
            report['engine_returncode'] = proc.returncode
            report['engine_wall_s'] = time.monotonic() - start
            if samples:
                report['observed_memory'] = peak_memory(samples)


def main(argv=None, *, request=None, on_token=None, expected_pack_sha256=None,
         request_source=None, on_case_end=None, text_profile=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('short', 'long', 'resident'), default='short', help='Five short tasks, three long tasks, or the long tasks plus repeated retrieval in one resident process')
    p.add_argument('--resident-order', choices=tuple(RESIDENT_ORDERS), help='Bounded resident fixtures and orders; requires resident suite and a pinned engine')
    p.add_argument('--first-request-only', action='store_true', help='Run only the first fixture in a fresh native process; requires --resident-order')
    p.add_argument('--smoke-result', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--exe', type=Path, help='Explicit candidate binary; requires --engine-sha256')
    p.add_argument('--engine-sha256', help='Expected SHA256 for --exe')
    p.add_argument('--store-stats', action='store_true', help='Require opt-in prefill/decode STORE diagnostics from the candidate')
    p.add_argument('--ssd-cache-gb', type=int, choices=(32, 36, 40), help='Override the native RAM-store ceiling in decimal GB;36 requires the pinned resource diagnostic or memory-experimental text profile')
    p.add_argument('--ssd-policy', choices=('lru', 'lfu'), help='Override the existing native RAM-store eviction policy for the bounded comparison')
    p.add_argument('--prefill-chunk', type=int, choices=(0, 256), help='Compare native windows with a bounded chunk size; enforces resource limits and emits flow diagnostics')
    p.add_argument('--settle-store', choices=('on', 'off'), help='Compare native post-prefill RAM preload; requires --prefill-chunk 256 and enables native timing logs')
    p.add_argument('--settle-ms', type=int, choices=(1000, 6000), help='Bounded native swap-loop budget; requires pinned controlled probes with normal selectors; RAM preload off is allowed only for the heat1 two-task experiment')
    p.add_argument('--native-verbose', action='store_true', help='Enable the same native timing logs for window/chunk comparisons without selecting post-prefill RAM preload')
    p.add_argument('--mixed-prefill', choices=('on', 'off'), help='Controlled first-flow/later-window schedule; requires resident suite, pinned engine, chunk256 and flow-store on')
    p.add_argument('--first-window', choices=('on', 'off'), help='Opt-in first eligible feed window guard; requires resident suite, pinned engine, chunk256, flow-store on and mixed-prefill off')
    p.add_argument('--flow-store', choices=('on', 'off'), help='Opt-in native streamed-expert handoff comparison; requires an explicitly pinned engine and --prefill-chunk 256')
    p.add_argument('--flow-store-min-heat', type=int, choices=(1, 4), help='Experimental free-slot retention threshold; requires a pinned controlled resident comparison and the complete flow configuration; native default remains 4')
    p.add_argument('--resource-diagnostics', action='store_true', help='Failure-only bounded resource metadata; requires the pinned heat1/preload-on/code-explain or code-explain-repeat diagnostic configuration')
    p.add_argument('--cuda-runtime', type=Path, default=Path('/usr/local/cuda-12.8/lib64/libcudart.so'))
    p.add_argument('--gpu-cap-mib', type=int, default=20480)
    p.add_argument('--timeout', type=int, default=600, help='Seconds for startup and for each request')
    args = p.parse_args(argv)
    text_mode = request is not None or request_source is not None
    resource_bounds = text_mode or args.prefill_chunk is not None or args.suite == 'resident'
    text_memory = text_profile == 'memory-experimental'
    text_flow = text_profile in ('flow-experimental', 'memory-experimental')
    retention_probe = args.flow_store_min_heat is not None
    retention_without_preload = args.flow_store_min_heat == 1 and args.settle_store == 'off'
    if args.ssd_cache_gb == 36 and not text_memory and (
            text_mode or text_profile is not None or args.suite != 'resident'
            or args.resident_order != 'code-explain-repeat' or not args.resource_diagnostics
            or args.flow_store_min_heat != 1 or args.settle_store != 'on'):
        p.error('--ssd-cache-gb 36 requires the pinned heat1/preload-on/code-explain-repeat resource diagnostic')
    if text_profile is not None and (
            not text_flow or (args.flow_store_min_heat != 1 if text_memory else retention_probe)
            or (text_memory and not args.resource_diagnostics)
            or (request is None) == (request_source is None) or not args.exe
            or args.engine_sha256 != (TEXT_MEMORY_ENGINE_SHA256 if text_memory else TEXT_FLOW_ENGINE_SHA256)
            or expected_pack_sha256 is None
            or args.ssd_cache_gb != (36 if text_memory else 40) or args.ssd_policy != 'lru' or args.gpu_cap_mib != 20480
            or not args.store_stats or args.resident_order is not None or args.first_request_only
            or (args.prefill_chunk, args.flow_store, args.mixed_prefill, args.first_window,
                args.settle_store, args.settle_ms) != (256, 'on', 'off', 'off', 'on', 1000)):
        p.error('memory-experimental requires one text source, its pinned engine/pack,36 GB LRU,heat1,resource diagnostics,20 GiB GPU and the complete measured prefill configuration' if text_memory else
                'flow-experimental requires one text source, its pinned engine/pack, 40 GB LRU, 20 GiB GPU, STORE diagnostics and the complete measured prefill configuration')
    if retention_probe and not text_memory and (
            text_mode or text_profile is not None or args.suite != 'resident' or not args.exe
            or args.resident_order not in ('explain-code', 'code-explain', 'code-explain-repeat') or args.first_request_only
            or args.ssd_cache_gb not in (36, 40) or args.ssd_policy != 'lru' or args.gpu_cap_mib != 20480
            or not args.store_stats
            or (args.settle_store != 'on' and not retention_without_preload)
            or (args.prefill_chunk, args.flow_store, args.mixed_prefill, args.first_window,
                args.settle_ms) != (256, 'on', 'off', 'off', 1000)):
        p.error('--flow-store-min-heat requires pinned --suite resident, --resident-order explain-code|code-explain (or diagnostic code-explain-repeat), STORE diagnostics, 40 GB LRU (36 only for the four-request diagnostic), 20 GiB GPU and the complete chunk256/flow-store/on, mixed/first-window/off, settle1000 and RAM preload on (off allowed only with heat1)')
    if args.resource_diagnostics and not text_memory and (not retention_probe or args.flow_store_min_heat != 1
                                      or args.settle_store != 'on' or args.resident_order not in ('code-explain', 'code-explain-repeat')):
        p.error('--resource-diagnostics requires the pinned heat1/preload-on/code-explain or code-explain-repeat configuration')
    if args.resident_order == 'code-explain-repeat' and not args.resource_diagnostics:
        p.error('--resident-order code-explain-repeat requires --resource-diagnostics')
    if args.resident_order in ('explain-code', 'code-explain', 'code-explain-repeat') and not retention_probe:
        p.error('--resident-order explain-code|code-explain requires --flow-store-min-heat')
    controlled_prefill = text_flow or (not text_mode and args.suite == 'resident')
    if bool(args.exe) != bool(args.engine_sha256):
        p.error('--exe and --engine-sha256 must be supplied together')
    if args.store_stats and not args.exe:
        p.error('--store-stats requires an explicitly pinned candidate binary')
    if args.settle_store is not None and args.prefill_chunk != 256:
        p.error('--settle-store requires --prefill-chunk 256')
    if args.flow_store is not None and (args.prefill_chunk != 256 or not args.exe):
        p.error('--flow-store requires --prefill-chunk 256 and an explicitly pinned --exe')
    if args.mixed_prefill is not None and (not controlled_prefill or not args.exe or args.prefill_chunk != 256 or args.flow_store != 'on'):
        p.error('--mixed-prefill requires --suite resident, a pinned --exe, --prefill-chunk 256 and --flow-store on')
    if args.first_window is not None and (not controlled_prefill or not args.exe or args.prefill_chunk != 256 or args.flow_store != 'on' or args.mixed_prefill != 'off'):
        p.error('--first-window requires --suite resident, a pinned --exe, --prefill-chunk 256, --flow-store on and --mixed-prefill off')
    if args.settle_ms is not None and (not controlled_prefill or not args.exe
                                      or (not text_flow and not retention_probe and args.resident_order not in ('short-first', 'long-first')) or args.prefill_chunk != 256
                                      or args.flow_store != 'on' or args.mixed_prefill != 'off' or args.first_window != 'off'
                                      or (args.settle_store != 'on' and not retention_without_preload)):
        p.error('--settle-ms requires pinned --suite resident, --resident-order short-first|long-first, --prefill-chunk 256, --flow-store on, --mixed-prefill off, --first-window off and RAM preload on (off allowed only in the heat1 two-task experiment)')
    if args.resident_order is not None and (text_mode or args.suite != 'resident' or not args.exe):
        p.error('--resident-order requires --suite resident, a pinned --exe and no text/interactive input')
    if args.first_request_only and args.resident_order is None:
        p.error('--first-request-only requires --resident-order')
    if args.timeout <= 0:
        p.error('--timeout must be positive')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'step': '7al' if text_memory else '7af' if retention_without_preload else '7ae' if retention_probe else '7ab' if text_flow else '7z' if args.settle_ms is not None and not args.first_request_only else '7y' if args.settle_ms is not None else '7w' if args.first_window is not None else '7v' if args.first_request_only else '7u' if args.resident_order is not None else '7t' if args.mixed_prefill is not None else '7s' if args.native_verbose else '7r' if args.suite == 'resident' else '7q' if args.flow_store is not None else '7p' if args.settle_store is not None else '7o' if args.prefill_chunk is not None else '7n' if args.ssd_policy is not None else '7m' if request_source is not None else '7l' if request is not None else '7k' if args.suite == 'long' else '7j' if args.ssd_cache_gb is not None else '7i' if args.exe else '7h', 'status': 'preflight', 'pid': os.getpid(),
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'baseline_result': str(args.smoke_result.resolve()), 'cases': [],
              'method': {'max_new_tokens': 256 if args.suite in ('long', 'resident') else 64, 'suite': args.suite, 'context': 1024, 'thinking': False, 'temperature': 0,
                         'fresh_kv_each_request': True, 'one_resident_engine': True, 'os_cache_dropped': False,
                         'ram_store_hit_ratio': None, 'ram_store_note': 'Not exposed by this pinned native serve protocol; STAT ram is a separate RAM-tier expert count, not store hit ratio.',
                         'decode_rate_note': '(generated tokens including EOS - 1) / native decode seconds; the first output comes from prefill. Native STAT tok_s includes that first output.',
                         'memory_note': '1-second samples plus request boundaries; HWM is process lifetime. GPU excludes the capacity hold but includes driver, monitor context and other users.'}}
    if args.suite == 'resident' and not text_mode:
        report['method'].update(request_count=4, repeated_case='long_retrieval_repeat', cache_state_note='Weights, routing profile and expert cache persist across requests; KV/prefix reuse is disabled. Compare first and later requests separately.')
    if args.resident_order is not None:
        ordered_cases = RESIDENT_ORDERS[args.resident_order]
        if args.first_request_only:
            ordered_cases = ordered_cases[:1]
        report['method'].pop('repeated_case')
        report['method'].update(resident_order=args.resident_order, request_count=len(ordered_cases),
                                max_new_tokens=max(c.get('max_new_tokens', 64) for c in ordered_cases),
                                repeated_cases=[c['id'] for c in ordered_cases if c['id'].endswith('_repeat')],
                                order_note='Identical short/long task multiset and output caps in opposite orders. Prompt lengths and routing content both differ; this is not an isolated length-threshold experiment.')
    if args.resident_order in ('short-repeat', 'long-then-short'):
        report['method']['order_note'] = 'Two independent requests in one resident model, with fresh KV, to inspect successful first-feed lifetime.'
    if retention_probe:
        report['method'].update(flow_store_min_heat=args.flow_store_min_heat,
                                flow_store_min_heat_note='Default4; only explicit1 lowers admission for already-routed, completed reads into free slots. Preload remains heat4; no eviction or extra read.',
                                order_note='Original explanation99/Python121 fixtures, cap256, in opposite two-request orders. Same native binary; fresh KV, resident expert/OS caches.')
    if retention_without_preload:
        report['method'].update(ram_preload_disabled=True,
                                ram_preload_note='Explicit DS4_SETTLE_STORE=0 disables post-prompt RAM preload; GPU swap-loop budget remains1000ms. An absent preload timer is not a measured zero.')
    if args.resource_diagnostics:
        report['method'].update(resource_diagnostics='guard_failure_only', resource_diagnostics_timeout_s=3)
    if args.resident_order == 'code-explain-repeat':
        report['method'].update(diagnostic_pair_repeated_once=True,
                                order_note='One resident diagnostic model: original code/explanation pair repeated once, cap256 each, fresh KV; expert/OS caches persist. Not a replacement performance sample.')
    if args.first_request_only:
        report['method'].update(request_count=1, first_request_only=True,
                                cache_state_note='First request in a new native process; no prior request/KV history. OS cache is not cleared.',
                                order_note='Only the first fixture of the selected order. No subsequent requests or mixed-schedule transition are measured.')
    if text_flow:
        report['method'].update(text_profile=text_profile, experimental_profile=True,
                                profile_note='Measured resident latency candidate; short answers may be slower. Existing native chooser and resource/input/cleanup bounds retained.')
        if text_memory:
            report['method'].update(profile_note='Measured36GB candidate provides extra host RAM headroom at a workload-dependent latency/read cost; existing resource guards remain enforced.',
                                    order_note='User independent questions with fresh KV; weights/expert/OS caches persist in resident mode. No fixed benchmark order is assumed.')
    if text_mode:
        report['method'].update(suite='interactive' if request_source is not None else 'text',
                                max_new_tokens=request_source.max_new_tokens if request_source is not None else request['max_new_tokens'])
    if resource_bounds:
        report['method'].update(host_available_floor_bytes=8 << 30, resource_bounds_enforced=True)
    if request_source is not None:
        report['session_state'] = 'starting'
    report['method']['store_stats_enabled'] = args.store_stats
    if args.store_stats:
        report['method'].pop('ram_store_hit_ratio')
        report['method']['ram_store_note'] = ('Per-request STORE phase snapshots: hit/(hit+miss) excludes prefetch-used/in-flight accesses; '
            'submitted_bytes counts payloads queued, not physical reads; wait_ms overlaps cpu_lane_ms and may overlap GPU work. '
            'Ratios are null for zero denominators. No synchronization is added at phase boundaries.')
    budget = None
    try:
        baseline = json.loads(args.smoke_result.read_text())
        if (baseline['status'], baseline['upstream_revision'], baseline['quant_revision'], baseline['engine_sha256'], baseline['pageable_embedding']) != (
                'smoke_passed_awaiting_review', UPSTREAM_REVISION, REVISION, ENGINE_SHA256, True):
            raise ValueError('Expected the validated 7g pageable-embedding smoke result')
        cmd = baseline['engine_command'].copy()
        if args.ssd_cache_gb is not None:
            cmd[cmd.index('--ssd-cache') + 1] = str(args.ssd_cache_gb)
        report['method']['ssd_cache_gb'] = float(cmd[cmd.index('--ssd-cache') + 1])
        if args.ssd_policy is not None:
            cmd.extend(['--ssd-policy', args.ssd_policy])
        report['method']['ssd_policy'] = cmd[cmd.index('--ssd-policy') + 1] if '--ssd-policy' in cmd else 'lru'
        if args.prefill_chunk is not None:
            cmd[cmd.index('--prefill-chunk') + 1] = str(args.prefill_chunk)
            report['method'].update(prefill_chunk=args.prefill_chunk, flow_stats_enabled=True,
                flow_stats_note='Native flow may bypass ExpertStore counters; zero STORE waits do not imply zero I/O. Compare physical process reads and native prompt-stream diagnostics.')
        if args.native_verbose or args.settle_store is not None or args.mixed_prefill is not None:
            cmd.append('--verbose')
            report['method']['native_timing_logs'] = True
        if args.settle_store is not None:
            report['method'].update(settle_store=args.settle_store, native_timing_logs=True,
                settle_store_note='Only post-prompt RAM preload is selected; native VRAM cache settling retains its default budget. Preload time is included in full prefill, and deferring reads can affect decode.')
        if args.settle_ms is not None:
            report['method'].update(settle_ms=args.settle_ms,
                settle_ms_note='Positive native swap-loop budget; RAM preload follows and its time is included in the settle log. Not a hard cap on total settling; placement changes can affect decode.',
                settle_store_note='RAM preload explicitly enabled after the selected positive swap-loop budget; actual preload membership/count may change.')
        if args.flow_store is not None:
            report['method'].update(flow_store=args.flow_store, flow_store_note='Default-off native copy of completed hot-expert spans into free RAM-store slots. Existing preload/eviction policy retained; not an additional cache.')
        if args.mixed_prefill is not None:
            report['method'].update(mixed_prefill=args.mixed_prefill, mixed_prefill_note='Default-off per-model native control: first successful eligible prefill uses flow, later eligible prefills use windows. This bounded suite has one eligible prefill per independent request; workspace and cache persist. Not an adaptive policy.')
        if args.first_window is not None:
            report['method'].update(first_window=args.first_window, first_window_limit_tokens=16,
                first_window_note='Default-off native per-model guard: first successful eligible feed uses windows when n<=native kWin16; larger first and later feeds delegate to the original selector. Failed/cancelled/bypassed feeds do not consume it. Existing chunk workspace is retained.')
        expected_sha256 = args.engine_sha256 or ENGINE_SHA256
        if args.exe:
            cmd[0] = str(args.exe.resolve())
        exe, pack, upstream = Path(cmd[0]), Path(cmd[cmd.index('--pack') + 1]), Path(baseline['pack_command'][1]).parents[2]
        if hashlib.sha256(exe.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Engine hash differs from the explicitly expected binary')
        manifest = json.loads((upstream.parent / 'step7f-upstream-source.json').read_text())
        if manifest['revision'] != UPSTREAM_REVISION or len(manifest['files']) != 1137:
            raise ValueError('Unexpected upstream source manifest')
        for item in manifest['files']:
            if hashlib.sha256((upstream / item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Upstream source changed: ' + item['path'])
        for flag in ('--ids', '--n'):
            at = cmd.index(flag)
            del cmd[at:at + 2]
        cmd.append('--serve')
        report.update(engine_command=cmd, engine_sha256=expected_sha256, upstream_revision=UPSTREAM_REVISION,
                      quant_revision=REVISION, source_files_verified_unchanged=len(manifest['files']),
                      pack_sha256={str(f.relative_to(pack)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted(pack.rglob('*')) if f.is_file()})
        if expected_pack_sha256 is not None and report['pack_sha256'] != expected_pack_sha256:
            raise ValueError('Model pack/tokenizer differs from the validated SHA256 pins')
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')
               and not k.startswith(('DS4_', 'STRATA_DS4_')) and k != 'LD_PRELOAD'}
        env.update(CUDA_VISIBLE_DEVICES='0', DS4_PAGEABLE_EMBED='1', STRATA_DS4_REUSE='0')
        if args.store_stats:
            env['DS4_STORE_STATS'] = '1'
        if args.prefill_chunk is not None:
            env['DS4_FLOW_STATS'] = '1'
        if args.settle_store is not None:
            env['DS4_SETTLE_STORE'] = '1' if args.settle_store == 'on' else '0'
        if args.settle_ms is not None:
            env['DS4_SETTLE_MS'] = str(args.settle_ms)
        if args.flow_store is not None:
            env['DS4_FLOW_STORE'] = '1' if args.flow_store == 'on' else '0'
        if args.mixed_prefill is not None:
            env['DS4_MIXED_PREFILL'] = '1' if args.mixed_prefill == 'on' else '0'
        if args.first_window is not None:
            env['DS4_FIRST_WINDOW'] = '1' if args.first_window == 'on' else '0'
        if retention_probe:
            env['DS4_FLOW_STORE_MIN_HEAT'] = str(args.flow_store_min_heat)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # The capacity hold uses the same physical device.
        report['environment_overrides'] = {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'DS4_PAGEABLE_EMBED', 'STRATA_DS4_REUSE', 'DS4_STORE_STATS', 'DS4_FLOW_STATS', 'DS4_SETTLE_STORE', 'DS4_SETTLE_MS', 'DS4_FLOW_STORE', 'DS4_MIXED_PREFILL', 'DS4_FIRST_WINDOW', 'DS4_FLOW_STORE_MIN_HEAT') if k in env}
        sys.path.insert(0, str(upstream / 'tools/dsv41'))
        import smoke
        tok = smoke.tokenizer(pack)
        cases = [] if request_source is not None else [request] if request is not None else (ordered_cases if args.resident_order is not None else RESIDENT_CASES) if args.suite == 'resident' else LONG_CASES if args.suite == 'long' else CASES
        report['cases'] = [prepare_case(tok, case, plain_user=request is not None) for case in cases]
        model = Path(baseline['model_dir'])
        pins = {item['name']: (item['bytes'], item['sha256']) for item in baseline['verified_files']}
        sys.path.insert(0, str(upstream))
        import setup
        if setup.HF_REVISIONS[REPO] != REVISION or any(setup.DSV41_ISTA_FILES[n] != pin for n, pin in pins.items()) or len(pins) != 5:
            raise ValueError('Weight pins differ from upstream')
        with (model / '.download.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            report['verified_files'] = verified_files(model, pins)
            report['integrity_source'] = 'Prior whole-file SHA256 record plus current sizes and markers under a shared lock; not a second full-file hash.'
            if resource_bounds and proc_memory(Path('/proc/meminfo'))['MemAvailable'] < (8 << 30):
                raise RuntimeError('At least 8 GiB host MemAvailable is required before startup')
            budget = GpuBudget(args.cuda_runtime, args.gpu_cap_mib)
            report['gpu_budget'] = budget.info
            report['status'] = 'running'
            write_json(out / 'result.json', report)
            resource_options = {'resource_diagnostics': True} if args.resource_diagnostics else {}
            run_engine(cmd, env, out, budget, tok, report, args.timeout,
                       on_token=on_token, host_floor_bytes=8 << 30 if resource_bounds else 0,
                       request_source=request_source, on_case_end=on_case_end, **resource_options)
        report['status'] = 'completed' if text_mode else 'measured_awaiting_text_review'
        return 0
    except BaseException as error:
        report.update(status='interrupted' if text_mode and isinstance(error, KeyboardInterrupt) else 'failed',
                      error=str(error) or type(error).__name__)
        raise
    finally:
        if budget is not None:
            budget.close()
        if request_source is not None:
            report['session_state'] = 'closed'
        report['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / 'result.json', report)


if __name__ == '__main__':
    sys.exit(main())

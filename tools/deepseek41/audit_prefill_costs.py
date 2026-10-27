#!/usr/bin/env python3
"""Audit saved single-GPU native benchmark traces without loading a model or modifying inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

NUMBER = r'\d+(?:\.\d+)?'
DECISION = re.compile(
    rf"^ds4: a prompt of (\d+) tokens through (the flow|windows): predicted ({NUMBER}) s, "
    rf"(the flow|windows) ({NUMBER}) s \((\d+)% of its CPU-lane experts' ({NUMBER}) GB in RAM\)$", re.M)
FIRST_WINDOW = re.compile(r'^ds4 first window: tokens=(\d+) limit=(\d+)$', re.M)
STREAM = re.compile(
    rf'^ds4 prompt stream, CUDA(\d+): (\d+) tokens in ({NUMBER}) ms; (\d+) experts streamed: '
    rf'(\d+) from the files \(({NUMBER}) GB, ({NUMBER}) GB/s: ({NUMBER}) GB from the disk unbuffered, '
    rf'({NUMBER}) GB through the page cache\), (\d+) from RAM \(({NUMBER}) GB\)'
    rf'(?:, (\d+) again from the layer cache \(layer-major, (\d+) chunks, ({NUMBER}) GB of RAM\))?; '
    rf'(\d+) likely ones queued early, (\d+) of them unpicked; the compute waited ({NUMBER}) ms for reads$', re.M)
SETTLE = re.compile(rf'^ds4: expert cache: (\d+) swaps after the prompt \(({NUMBER}) ms\)$', re.M)
PRELOAD = re.compile(rf'^ds4: RAM store: (\d+) hot experts read after the prompt \(({NUMBER}) ms\)$', re.M)
HANDOFF = re.compile(rf'^ds4 flow store: retained=(\d+) bytes=(\d+) present=(\d+) no_room=(\d+) early=(\d+) invalid=(\d+) copy_ms=({NUMBER})$', re.M)
FLOW_STORE_HEAT = re.compile(r'^ds4 flow store: experimental min_heat=(1)$', re.M)
REQUEST_END = re.compile(r'^maya ds4 serve: prompt (\d+) tokens = .*$', re.M)
CALIBRATION = re.compile(r"^ds4: prompts through the flow or windows by this PC's speeds:.*$", re.M)
TRACE_PREFIXES = ('ds4: a prompt of ', 'ds4 first window:', 'ds4 prompt stream,',
                  'ds4: expert cache:', 'ds4: RAM store: ', 'ds4 flow store:')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optional_record(pattern, text, name):
    matches = list(pattern.finditer(text))
    require(len(matches) <= 1, 'Multiple ' + name + ' records in one request; unsupported trace')
    return matches[0] if matches else None


def audit_trace(result, stderr):
    """Keep each native request boundary, including repeated prompts and missing optional timings."""
    require(result['status'] == 'measured_awaiting_text_review' and result['engine_returncode'] == 0,
            'Only completed successful benchmark results can be audited')
    endings = list(REQUEST_END.finditer(stderr))
    require(len(endings) == len(result['cases']), 'Native request boundaries do not match result cases')
    calibrations = CALIBRATION.findall(stderr)
    require(len(calibrations) <= 1, 'Multiple startup calibrations')
    rows, start = [], 0
    for index, (case, end) in enumerate(zip(result['cases'], endings)):
        block = stderr[start:end.start()]
        start = end.end()
        n = len(case['prompt_ids'])
        require(int(end[1]) == n, 'Native summary prompt length mismatch')
        decision = optional_record(DECISION, block, 'automatic decision')
        forced = optional_record(FIRST_WINDOW, block, 'first-window decision')
        stream = optional_record(STREAM, block, 'single-GPU stream')
        settle = optional_record(SETTLE, block, 'cache-settle')
        preload = optional_record(PRELOAD, block, 'RAM-preload')
        handoff = optional_record(HANDOFF, block, 'handoff')
        flow_heat = optional_record(FLOW_STORE_HEAT, block, 'flow retention heat')
        # Unrecognized diagnostics must not silently turn into missing/zero measurements.
        patterns = (DECISION, FIRST_WINDOW, STREAM, SETTLE, PRELOAD, HANDOFF, FLOW_STORE_HEAT)
        for line in block.splitlines():
            if line.startswith(TRACE_PREFIXES) and 'RAM store: up to ' not in line:
                require(any(p.fullmatch(line) for p in patterns), 'Unparsed diagnostic: ' + line)
        require(not (decision and forced), 'Conflicting path decisions')
        prediction = None
        if decision:
            require(int(decision[1]) == n and decision[2] != decision[4], 'Invalid decision token count or paths')
            prediction = {'selected_s': float(decision[3]), 'alternative_s': float(decision[5]),
                          'alternative_path': decision[4].removeprefix('the '),
                          'estimated_resident_pct': int(decision[6]), 'estimated_expert_GB': float(decision[7])}
            require(0 <= prediction['estimated_resident_pct'] <= 100, 'Invalid estimated resident percentage')
            kind, path = 'automatic', decision[2].removeprefix('the ')
        elif forced:
            require(int(forced[1]) == n and n <= int(forced[2]), 'Invalid forced-window boundary')
            kind, path = 'first_window', 'windows'
        else:
            require(not result['method'].get('prefill_chunk'), 'Missing decision for a chunk-enabled request')
            kind, path = 'window_profile', 'windows'
        require(bool(stream) == (path == 'flow'), 'Selected path and stream trace disagree')
        require(not (settle or preload or handoff or flow_heat) or bool(stream), 'Flow-only records on a window request')
        if decision or forced:
            require(len(calibrations) == 1, 'Missing native startup calibration')
        if flow_heat:
            require(result['method'].get('flow_store_min_heat') == 1, 'Flow retention heat differs from result method')
            require(bool(handoff), 'Flow retention heat without active handoff')
        elif path == 'flow' and result['method'].get('flow_store_min_heat') == 1:
            require(False, 'Missing experimental flow retention heat')
        values = [case[k] for k in ('native_prefill_ms', 'native_decode_ms', 'first_token_s', 'request_wall_s', 'process_read_bytes')]
        require(all(math.isfinite(v) and v >= 0 for v in values), 'Non-finite or negative benchmark measurement')
        row = {'request_index': index, 'case': case['id'], 'prompt_tokens': n, 'path': path, 'decision_kind': kind,
               'prediction': prediction, 'native_prefill_ms': case['native_prefill_ms'],
               'native_decode_ms': case['native_decode_ms'], 'first_token_s': case['first_token_s'],
               'request_wall_s': case['request_wall_s'], 'whole_request_process_read_bytes': case['process_read_bytes'],
               'prefill_physical_read_bytes': None, 'exact_prefill_cache_hit_pct': None,
               'output_ids': case['output_ids'], 'reused_tokens': case['reused_tokens'],
               'stream': None, 'cache_settle': None, 'RAM_preload': None, 'handoff': None,
               'prefill_minus_stream_and_settle_ms': None}
        if flow_heat:
            row['flow_store_min_heat'] = 1
        if stream:
            require(int(stream[2]) == n, 'Stream prompt length mismatch')
            s = {'gpu': int(stream[1]), 'wall_ms': float(stream[3]), 'experts': int(stream[4]),
                 'file_experts': int(stream[5]), 'file_GB': float(stream[6]), 'file_GB_per_s': float(stream[7]),
                 'direct_GB': float(stream[8]), 'buffered_GB': float(stream[9]),
                 'RAM_experts': int(stream[10]), 'RAM_GB': float(stream[11]),
                 'layer_cache_uses': int(stream[12] or 0), 'chunks': int(stream[13] or 1),
                 'layer_cache_GB': float(stream[14] or 0), 'speculative_experts': int(stream[15]),
                 'unpicked_speculative_experts': int(stream[16]), 'read_wait_ms': float(stream[17])}
            require(s['experts'] == s['file_experts'] + s['RAM_experts'] + s['layer_cache_uses'], 'Stream expert counts disagree')
            require(abs(s['file_GB'] - s['direct_GB'] - s['buffered_GB']) <= 0.151, 'Rounded IO payload totals disagree')
            require(s['read_wait_ms'] <= s['wall_ms'] + 1, 'Read wait exceeds single-GPU stream wall time')
            payload = s['direct_GB'] + s['buffered_GB']
            s['buffered_payload_share_pct'] = 100 * s['buffered_GB'] / payload if payload else None
            row['stream'] = s
            if settle:
                row['cache_settle'] = {'swaps': int(settle[1]), 'inclusive_ms': float(settle[2])}
                row['prefill_minus_stream_and_settle_ms'] = case['native_prefill_ms'] - s['wall_ms'] - float(settle[2])
            if preload:
                row['RAM_preload'] = {'experts': int(preload[1]), 'ms': float(preload[2])}
            if handoff:
                row['handoff'] = dict(zip(('retained', 'bytes', 'present', 'no_room', 'early', 'invalid', 'copy_ms'),
                                         [int(handoff[i]) for i in range(1, 7)] + [float(handoff[7])]))
            # A descriptive payload comparison only when reuse/speculation does not change its denominator.
            if prediction and s['chunks'] == 1 and s['speculative_experts'] == 0 and s['RAM_experts'] == 0 and s['file_GB']:
                row['estimated_expert_to_file_payload_ratio'] = prediction['estimated_expert_GB'] / s['file_GB']
        rows.append(row)
    require(not any(line.startswith(TRACE_PREFIXES) for line in stderr[start:].splitlines()), 'Unmatched trailing request diagnostics')
    return {'calibration_logs': calibrations, 'requests': rows}


LIMITS = [
    'Single-GPU saved traces only. This audit starts no engine and changes no scheduling or input evidence.',
    'Estimated residency samples CPU-lane experts before routing. Buffered payload classifies actual IO jobs later; they are different populations and times.',
    'Buffered payload can include fallback reads. It is not an exact cache-hit count; neither IO payload class equals whole-process physical read_bytes.',
    'Whole-request process reads include prefill, cache settling and decode. Per-prefill physical reads were not captured and remain null.',
    'Stream wall time already contains read waits and handoff work. The cache-settle log includes RAM preload; nested times must not be added again.',
    'Missing optional timing is unknown, not zero. File payload GB is rounded to one decimal; timings and predictions are also rounded.',
    'Prediction/file-payload ratios are descriptive, not fitted correction factors. Later model learning and multi-chunk reuse require separate treatment.',
]


def audit_result(path):
    path = Path(path)
    stderr_path = path.parent / 'engine.stderr'
    raw, stderr = path.read_bytes(), stderr_path.read_bytes()
    result = json.loads(raw)
    return {'result': str(path), 'engine_sha256': result['engine_sha256'],
            'input_sha256': {str(path): hashlib.sha256(raw).hexdigest(),
                             str(stderr_path): hashlib.sha256(stderr).hexdigest()},
            **audit_trace(result, stderr.decode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+', help='Saved result.json files with adjacent engine.stderr')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; existing files are never overwritten')
    args = parser.parse_args()
    require(len({p.resolve() for p in args.results}) == len(args.results), 'Duplicate result input')
    reports = [audit_result(p) for p in args.results]
    report = {'schema_version': 1, 'new_native_runs': 0, 'runs': reports,
              'request_count': sum(len(r['requests']) for r in reports), 'limits': LIMITS}
    with args.output.open('x') as out:
        out.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Audited %d saved processes / %d requests; no new native run.' % (len(reports), report['request_count']))


if __name__ == '__main__':
    main()

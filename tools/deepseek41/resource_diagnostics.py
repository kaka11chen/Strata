"""Bounded Linux resource metadata for an owned benchmark child after a guard failure.

No process-memory contents, policy writes or normal-path sampling. The parent
limits this metadata helper to3 seconds; partial snapshots remain explicit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

TIMEOUT_S = 3
SMAPS_LIMIT = 8 << 20
CGROUP_FILES = ('memory.current', 'memory.max', 'memory.high', 'memory.low',
                'memory.swap.current', 'memory.swap.max', 'memory.events',
                'memory.events.local', 'memory.pressure', 'memory.stat')
HEADER = re.compile(r'^[0-9a-f]+-[0-9a-f]+\s')


def read_bounded(path, limit=65536):
    try:
        with path.open('rb') as stream:
            data = stream.read(limit + 1)
        truncated = len(data) > limit
        data = data[:limit]
        if truncated:
            data = data[:data.rfind(b'\n') + 1] if b'\n' in data else b''
        return {'path': str(path), 'text': data.decode('utf-8', errors='replace'),
                'truncated': truncated, 'byte_limit': limit}
    except OSError as error:
        return {'path': str(path), 'error': type(error).__name__ + ': ' + str(error)}


def summarize_smaps(record, max_records=128):
    if 'error' in record:
        return {'error': record['error'], 'complete': False}
    swapped, current, total, count = [], None, 0, 0

    def finish():
        nonlocal total, count
        if current and current['kB'].get('Swap', 0):
            count += 1
            total += current['kB']['Swap']
            if len(swapped) < max_records:
                swapped.append(current)

    for line in record['text'].splitlines():
        if HEADER.match(line):
            finish()
            fields = line.split(maxsplit=5)
            current = {'header': line, 'mapping': fields[5] if len(fields) > 5 else '[anonymous]', 'kB': {}}
        elif current:
            fields = line.split()
            if len(fields) == 3 and fields[2] == 'kB' and fields[1].isdigit():
                if fields[0].rstrip(':') in ('Size', 'Rss', 'Pss', 'Anonymous', 'Private_Dirty', 'Private_Clean', 'Swap', 'SwapPss', 'Locked'):
                    current['kB'][fields[0].rstrip(':')] = int(fields[1])
    finish()
    return {'complete': not record['truncated'], 'byte_limit': record['byte_limit'],
            'swapped_mapping_count_in_read_prefix': count, 'swap_kB_in_read_prefix': total,
            'swapped_mappings': swapped, 'omitted_swapped_mapping_records': count - len(swapped),
            'limit': 'Sequential metadata read, not atomic with trigger/status/rollup; truncated prefixes are not full-process totals.'}


def cgroup_paths(text, root, max_levels=4):
    for line in text.splitlines():
        if line.startswith('0::'):
            path = PurePosixPath(line[3:])
            if not path.is_absolute() or '..' in path.parts:
                return []
            result = []
            while len(result) < max_levels:
                result.append(root.joinpath(*path.parts[1:]))
                if path == path.parent:
                    break
                path = path.parent
            return result
    return []


def collect_snapshot(pid, trigger, reason, directory, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    started = time.monotonic()
    report = {'pid': pid, 'reason': reason, 'trigger_sample': trigger, 'status': 'capturing',
              'started_unix_s': time.time(), 'sections': {},
              'limits': ['Metadata only; owned child remains unreaped during helper capture.',
                         'Files are read sequentially after the trigger, not an atomic snapshot.',
                         'Cgroup ancestors can include other processes. Missing files are unknown, not zero.']}
    target = directory / 'snapshot.json'

    def save():
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(target)

    save()
    process = proc_root / str(pid)
    for name, limit in [('status', 65536), ('stat', 65536), ('cgroup', 65536), ('smaps_rollup', 65536), ('maps', 262144)]:
        report['sections'][name] = read_bounded(process / name, limit)
        save()
    for name in ['meminfo', 'vmstat', 'pressure/memory', 'sys/vm/swappiness', 'sys/vm/zone_reclaim_mode']:
        report['sections']['host/' + name] = read_bounded(proc_root / name)
        save()
    membership = report['sections']['cgroup']
    paths = cgroup_paths(membership.get('text', ''), cgroup_root)
    report['cgroup_v2'] = {'paths': [str(p) for p in paths], 'files': [],
                           'status': 'available_paths' if paths else 'unavailable_or_unsupported'}
    for path in paths:
        for name in CGROUP_FILES:
            report['cgroup_v2']['files'].append(read_bounded(path / name, 16384))
        save()
    report['smaps_swap_summary'] = summarize_smaps(read_bounded(process / 'smaps', SMAPS_LIMIT))
    report.update(status='complete', capture_duration_s=time.monotonic() - started,
                  finished_unix_s=time.time())
    save()
    return report


def capture_resource_failure(pid, output, trigger, reason):
    """Best-effort diagnostic only; caller must still raise its original guard error."""
    directory = output / 'resource-diagnostic'
    directory.mkdir(exist_ok=False)
    target = directory / 'snapshot.json'
    started = time.monotonic()
    try:
        child = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).resolve()),
                                str(pid), str(directory)],
                               input=json.dumps({'trigger': trigger, 'reason': reason}),
                               text=True, capture_output=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {'status': 'timed_out', 'timeout_s': TIMEOUT_S,
                'snapshot_file': str(target) if target.exists() else None}
    info = {'helper_returncode': child.returncode, 'capture_duration_s': time.monotonic() - started,
            'snapshot_file': str(target) if target.exists() else None}
    if child.returncode or not target.exists():
        return {**info, 'status': 'capture_failed', 'error': child.stderr[-4096:]}
    data = json.loads(target.read_text())
    if data.get('pid') != pid or data.get('trigger_sample') != trigger or data.get('status') != 'complete':
        return {**info, 'status': 'capture_incomplete'}
    return {**info, 'status': 'captured', 'bytes': target.stat().st_size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__ == '__main__':
    request = json.loads(sys.stdin.read())
    collect_snapshot(int(sys.argv[1]), request['trigger'], request['reason'], Path(sys.argv[2]))

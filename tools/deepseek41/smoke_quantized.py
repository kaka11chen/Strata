"""One bounded 7g smoke using the pinned upstream packer, tokenizer and engine.

Linux/CUDA only. This never downloads weights or changes the engine. Completion
markers must come from download_quantized.py's whole-file SHA256 verification.
With --wait-for-download it waits for that process's lock before checking them.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from prepare_quant_sample import FILES, REPO, REVISION

UPSTREAM_REVISION = '7dd5cf18e68c639a8773e096c5c79f971ea2a9f5'
ENGINE_SHA256 = '30982a43ad4dd82975d0fc33b49d5c80e42deb846bff621aab7ba10d3cdfe9f3'


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def verified_files(model, pins):
    state = json.loads((model / 'download-state.json').read_text())
    if (state['repo'], state['revision'], state['status']) != (REPO, REVISION, 'verified'):
        raise ValueError('The pinned five-file download has not finished verification')
    expected = [{'name': n, 'bytes': s, 'sha256': h} for n, (s, h) in pins.items()]
    if sorted(state['verified'], key=lambda x: x['name']) != sorted(expected, key=lambda x: x['name']):
        raise ValueError('Download verification record does not match the five pinned files')
    for entry in expected:
        path = model / entry['name']
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Incomplete file: ' + path.name)
        if Path(str(path) + '.done').read_text().strip() != 'sha256 ' + entry['sha256']:
            raise ValueError('Missing or different full-file SHA256 marker: ' + path.name)
    return expected


class GpuBudget:
    """Hold physical VRAM above the target capacity; no inference implementation."""
    def __init__(self, runtime, cap_mib):
        self.lib = ctypes.CDLL(str(runtime))
        size_p = ctypes.POINTER(ctypes.c_size_t)
        self.lib.cudaMemGetInfo.argtypes = [size_p, size_p]
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.ptr = ctypes.c_void_p()
        self.check(self.lib.cudaSetDevice(0))
        self.check(self.lib.cudaFree(None))
        free, total = self.memory()
        physical = int(subprocess.check_output(
            ['nvidia-smi', '--id=0', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            text=True, timeout=10).strip()) << 20
        cap = cap_mib << 20
        if not 0 < cap <= physical:
            raise ValueError('GPU capacity must be positive and no larger than this physical device')
        # cudaMemGetInfo excludes driver-reserved VRAM. Use physical capacity
        # for the hold so a 22 GiB card really loses 2 GiB, not about 1.48 GiB.
        self.held = physical - cap
        if self.held:
            self.check(self.lib.cudaMalloc(ctypes.byref(self.ptr), self.held))
        after, _ = self.memory()
        if after > cap - (physical - total):
            self.close()
            raise RuntimeError('The GPU capacity reservation was not enforced')
        self.info = {'physical_bytes': physical, 'cuda_total_bytes': total, 'target_bytes': cap, 'held_bytes': self.held,
                     'free_before_hold': free, 'free_after_hold': after,
                     'note': 'The separate CUDA context and other GPU users also consume the remaining capacity.'}

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.cudaGetErrorString(code).decode())

    def memory(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)))
        return free.value, total.value

    def close(self):
        if self.ptr.value:
            self.check(self.lib.cudaFree(self.ptr))
            self.ptr = ctypes.c_void_p()


def proc_memory(path):
    fields = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] == 'kB':
                fields[parts[0].rstrip(':')] = int(parts[1]) * 1024
    except FileNotFoundError:
        pass
    return fields


def measured_run(cmd, out, env, budget, timeout):
    start = time.monotonic()
    peaks = {'rss_bytes': 0, 'hwm_bytes': 0, 'swap_bytes': 0, 'gpu_used_excluding_hold_bytes': 0}
    minimum_available = proc_memory(Path('/proc/meminfo'))['MemAvailable']
    timed_out = False
    with (out / 'engine.stdout').open('w') as stdout, (out / 'engine.stderr').open('w') as stderr, \
            (out / 'memory.jsonl').open('w') as samples:
        child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                status = proc_memory(Path('/proc') / str(child.pid) / 'status')
                host = proc_memory(Path('/proc/meminfo'))
                free, total = budget.memory()
                sample = {'elapsed_s': time.monotonic() - start,
                          'rss_bytes': status.get('VmRSS', 0), 'hwm_bytes': status.get('VmHWM', 0),
                          'swap_bytes': status.get('VmSwap', 0),
                          'gpu_used_excluding_hold_bytes': budget.info['physical_bytes'] - free - budget.held,
                          'host_available_bytes': host['MemAvailable']}
                for key in peaks:
                    peaks[key] = max(peaks[key], sample[key])
                minimum_available = min(minimum_available, sample['host_available_bytes'])
                samples.write(json.dumps(sample) + '\n')
                samples.flush()
                if sample['elapsed_s'] >= timeout:
                    timed_out = True
                    child.terminate()
                    break
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait()
    return {'pid': child.pid, 'returncode': child.returncode, 'timed_out': timed_out,
            'wall_s': time.monotonic() - start, 'observed_peaks': peaks,
            'minimum_host_available_bytes': minimum_available,
            'children_max_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
            'measurement_note': 'One-second samples can miss peaks. GPU usage includes driver reservations, the monitor context and other users; '
                                'children_max_rss includes the earlier pack/check subprocesses. RAM is observed, not hard capped.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream-dir', required=True, type=Path)
    ap.add_argument('--model-dir', required=True, type=Path)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--engine-sha256', default=ENGINE_SHA256, help='expected SHA256 of the selected build (default: unchanged 7f engine)')
    ap.add_argument('--pageable-embedding', action='store_true', help='enable the audited DS4_PAGEABLE_EMBED patch')
    ap.add_argument('--output-dir', required=True, type=Path, help='a new evidence directory')
    ap.add_argument('--cuda-runtime', type=Path, default=Path('/usr/local/cuda-12.8/lib64/libcudart.so'))
    ap.add_argument('--gpu-cap-mib', type=int, default=20480)
    ap.add_argument('--wait-for-download', action='store_true')
    ap.add_argument('--timeout', type=int, default=5400, help='engine startup plus generation seconds')
    args = ap.parse_args()
    if len(args.engine_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.engine_sha256):
        ap.error('--engine-sha256 must be a lowercase SHA256 digest')
    upstream, model, exe, out = (p.resolve() for p in (args.upstream_dir, args.model_dir, args.exe, args.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    report = {'step': '7g', 'pid': os.getpid(), 'status': 'preflight', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'upstream_revision': UPSTREAM_REVISION, 'quant_revision': REVISION, 'model_dir': str(model)}
    def save(status):
        report['status'] = status
        report['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / 'result.json', report)
        print(status, flush=True)
    budget = None
    try:
        save('preflight')
        manifest = json.loads((upstream.parent / 'step7f-upstream-source.json').read_text())
        if manifest['revision'] != UPSTREAM_REVISION or len(manifest['files']) != 1137:
            raise ValueError('Different upstream source manifest')
        for item in manifest['files']:
            if hashlib.sha256((upstream / item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Upstream source changed: ' + item['path'])
        if hashlib.sha256(exe.read_bytes()).hexdigest() != args.engine_sha256:
            raise ValueError('Engine differs from its explicitly expected SHA256')
        report['source_files_verified_unchanged'] = len(manifest['files'])
        report['engine_sha256'] = args.engine_sha256
        report['pageable_embedding'] = args.pageable_embedding
        # No inherited approximation / expert-skipping or proxy settings in this run.
        for key in list(os.environ):
            if key.lower().endswith('_proxy') or key.startswith(('DS4_', 'STRATA_DS4_')):
                os.environ.pop(key)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if args.pageable_embedding:
            os.environ['DS4_PAGEABLE_EMBED'] = '1'
        sys.path.insert(0, str(upstream))
        import setup
        if setup.HF_REVISIONS[REPO] != REVISION:
            raise ValueError('Different upstream quant revision')
        pins = {n: setup.DSV41_ISTA_FILES[n] for n in FILES}
        if {n: pin[0] for n, pin in pins.items()} != FILES:
            raise ValueError('Different upstream file sizes')
        # Keep a shared lock for packing and generation. A running download owns
        # the exclusive lock; a crash releases it and unverified state fails below.
        lock = (model / '.download.lock').open('a')
        deadline = time.monotonic() + 86400
        save('waiting_for_download' if args.wait_for_download else 'checking_download')
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not args.wait_for_download or time.monotonic() >= deadline:
                    raise RuntimeError('Download still holds its lock')
                time.sleep(10)
        report['verified_files'] = verified_files(model, pins)
        report['integrity_source'] = 'Fresh download whole-file SHA256 markers, checked under the download lock; not a second full hash pass.'
        extras = [p for pattern in ('dspark*', 'mmproj*', 'vision') for p in model.parent.glob(pattern)]
        if extras:
            raise ValueError('Optional sidecars would be auto-discovered: ' + str(extras))
        pack = out / 'pack'
        cmd = [sys.executable, str(upstream / 'tools/dsv41/pack_gguf.py'), '--model', str(model),
               '--shards', *[n for n in FILES if 'Engram' not in n],
               '--engram', str(model / 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'), '--out', str(pack)]
        report['pack_command'] = cmd
        save('packing')
        with (out / 'pack.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        report['pack_info_command'] = [str(exe), '--pack', str(pack), '--pack-info']
        with (out / 'pack-info.log').open('w') as log:
            subprocess.run(report['pack_info_command'], stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, timeout=120, check=True)
        sys.path.insert(0, str(upstream / 'tools/dsv41'))
        import smoke
        tok = smoke.tokenizer(pack)
        ids = tok.encode(smoke.PROMPT, parse_special=True)
        if tok.decode(ids) != smoke.PROMPT or not 0 < len(ids) < 1008:
            raise ValueError('Prompt tokenization did not round-trip within the context bound')
        (out / 'prompt-ids.txt').write_text(','.join(map(str, ids)))
        report['prompt'] = smoke.PROMPT
        report['prompt_ids'] = ids
        cmd = [str(exe), '--pack', str(pack), '--ids', '@' + str(out / 'prompt-ids.txt'),
               '--n', '16', '--max-context', '1024', '--dspark', 'none', '--vision', 'off',
               '--window', '16', '--prefill-chunk', '0', '--tp', '1', '--threads', 'off',
               '--gpus', '0', '--reserve-mib', '600', '--ssd-cache', '32', '--mmap-experts', 'on',
               '--cpu-threads', '18', '--expert-profile', 'none', '--temperature', '0']
        report['engine_command'] = cmd
        budget = GpuBudget(args.cuda_runtime, args.gpu_cap_mib)
        report['gpu_budget'] = budget.info
        save('running_short_generation')
        report['measurement'] = measured_run(cmd, out, dict(os.environ), budget, args.timeout)
        budget.close()
        budget = None
        if report['measurement']['returncode'] or report['measurement']['timed_out']:
            raise RuntimeError('Engine failed or timed out; see engine.stderr')
        line = (out / 'engine.stdout').read_text().strip().splitlines()[-1]
        generated = [int(n) for n in line.rstrip(',').split(',')]
        if not 0 < len(generated) <= 16 or any(n < 0 or n >= len(tok.tokens) for n in generated):
            raise ValueError('Invalid generated token IDs or token count')
        answer = tok.decode(generated)
        report.update(output_ids=generated, output_text=answer)
        good, reason = smoke.sensible(answer)
        report['text_check'] = {'passed': good, 'reason': reason}
        if not good:
            raise ValueError('Upstream text check failed: ' + reason)
        save('smoke_passed_awaiting_review')
        return 0
    except BaseException as error:
        report['error'] = str(error)
        save('failed')
        raise
    finally:
        if budget is not None:
            budget.close()


if __name__ == '__main__':
    sys.exit(main())

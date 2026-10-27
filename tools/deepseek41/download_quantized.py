"""Download the five pinned IQ1_M/Engram files through a mirror, without proxies.

Reuse the audited upstream Strata downloader for ranges, resume, retries and
whole-file SHA256. This command never builds, packs, installs or starts a model.
Linux only. The upstream source root is the unchanged 7f checkout.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
from urllib.parse import urlsplit
from prepare_quant_sample import FILES, REPO, REVISION

UPSTREAM_SETUP_SHA256 = '793925e603b324300c5411ee5af844f59c90324b5f4eb2955ffb786c85b93365'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mirror', default='https://hf-mirror.com')
    parser.add_argument('--connections', type=int, choices=range(1, 33), default=8)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    endpoint = args.mirror.rstrip('/')
    url = urlsplit(endpoint)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.path or url.query or url.fragment:
        parser.error('--mirror must be an HTTPS origin without credentials or a path')
    upstream = args.upstream_dir.resolve()
    if hashlib.sha256((upstream / 'setup.py').read_bytes()).hexdigest() != UPSTREAM_SETUP_SHA256:
        parser.error('upstream setup.py does not match the audited 7f revision')
    for key in list(os.environ):
        if key.lower().endswith('_proxy'):
            os.environ.pop(key)
    os.environ.update(HF_ENDPOINT=endpoint, STRATA_SOURCE='huggingface',
                      STRATA_ALLOW_UNPINNED='0', STRATA_DL_CONNECTIONS=str(args.connections))
    sys.path.insert(0, str(upstream))
    import setup
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    assert setup.HF_REVISIONS[REPO] == REVISION
    pins = {name: setup.DSV41_ISTA_FILES[name] for name in FILES}
    assert {name: pin[0] for name, pin in pins.items()} == FILES
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remaining = 0
    for name, (size, _) in pins.items():
        present = sum(p.stat().st_blocks * 512 for p in (output / name, output / (name + '.part')) if p.exists())
        remaining += max(0, size - present)
    free = shutil.disk_usage(output).free
    plan = {'repo': REPO, 'revision': REVISION, 'mirror': endpoint, 'proxy': False,
            'connections': args.connections, 'output_dir': str(output),
            'files': [{'name': n, 'bytes': s, 'sha256': h} for n, (s, h) in pins.items()],
            'total_bytes': sum(FILES.values()), 'remaining_disk_bytes_estimate': remaining,
            'free_disk_bytes': free, 'working_allowance_bytes': 6_000_000_000}
    print(json.dumps(plan, indent=2), flush=True)
    if free < remaining + plan['working_allowance_bytes']:
        parser.error('not enough free space for remaining files and 6 GB working allowance')
    if args.plan:
        return
    state_path = output / 'download-state.json'
    state = {**plan, 'pid': os.getpid(), 'status': 'downloading', 'verified': []}
    def save():
        state['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    try:
        for name, (size, sha) in sorted(pins.items(), key=lambda item: item[1][0]):
            state['current_file'] = name
            save()
            dst = output / name
            marker = output / (name + '.done')
            if marker.exists() and (not dst.exists() or dst.stat().st_size != size or marker.read_text().strip() != 'sha256 ' + sha):
                raise ValueError('Existing completion marker does not match the pinned file: ' + name)
            setup.download(setup.hf(REPO) + name, dst, name)
            if dst.stat().st_size != size or not marker.exists() or marker.read_text().strip() != 'sha256 ' + sha:
                raise ValueError('Download did not finish with the pinned SHA256: ' + name)
            state['verified'].append({'name': name, 'bytes': size, 'sha256': sha})
            save()
        state['status'] = 'verified'
        state.pop('current_file', None)
        save()
    except BaseException:
        state['status'] = 'interrupted_or_failed'
        save()
        raise


if __name__ == '__main__':
    main()

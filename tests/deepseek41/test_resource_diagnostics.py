"""Metadata-only controls plus fake-engine guard/cleanup tests; no model/GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import benchmark_quantized as bench
import resource_diagnostics as diag

SMAPS = ('1000-2000 rw-p 0000 00:00 0 [heap]\nSize: 4 kB\nRss: 0 kB\nSwap: 4 kB\n'
         '2000-3000 r--p 0000 00:01 1 /tmp/file with spaces\nSize: 4 kB\nSwap: 0 kB\n'
         '3000-5000 rw-p 0000 00:00 0\nSize: 8 kB\nSwap: 8 kB\n')


class MetadataChecks(unittest.TestCase):
    def test_bounded_read_and_missing_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'data'; p.write_text('first\nsecond\nthird\n')
            r = diag.read_bounded(p, 10)
            self.assertTrue(r['truncated']); self.assertEqual(r['text'], 'first\n')
            self.assertIn('error', diag.read_bounded(p.with_name('missing')))

    def test_swapped_mapping_summary_is_bounded_and_not_a_global_total(self):
        r = diag.summarize_smaps({'text': SMAPS, 'truncated': False, 'byte_limit': 1000}, max_records=1)
        self.assertTrue(r['complete']); self.assertEqual(r['swapped_mapping_count_in_read_prefix'], 2)
        self.assertEqual(r['swap_kB_in_read_prefix'], 12)
        self.assertEqual(r['swapped_mappings'][0]['mapping'], '[heap]')
        self.assertEqual(r['omitted_swapped_mapping_records'], 1)
        self.assertFalse(diag.summarize_smaps({'text': SMAPS, 'truncated': True, 'byte_limit': 1000})['complete'])
        self.assertFalse(diag.summarize_smaps({'error': 'denied'})['complete'])

    def test_cgroup_v2_ancestors_are_bounded_and_traversal_is_rejected(self):
        self.assertEqual(diag.cgroup_paths('0::/a/b/c/d/e\n', Path('/fixture')), [Path('/fixture/a/b/c/d/e'), Path('/fixture/a/b/c/d'), Path('/fixture/a/b/c'), Path('/fixture/a/b')])
        self.assertEqual(diag.cgroup_paths('0::/../../outside\n', Path('/fixture')), [])
        self.assertEqual(diag.cgroup_paths('3:memory:/a\n', Path('/fixture')), [])

    def test_fixture_snapshot_keeps_inputs_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); proc = root / 'proc'; child = proc / '42'; child.mkdir(parents=True)
            for name, text in {'status': 'Pid: 42\nVmSwap: 12 kB\n', 'stat': '42 (fake) S\n', 'cgroup': '0::/leaf\n', 'smaps': SMAPS}.items():
                (child / name).write_text(text)
            (proc / 'meminfo').write_text('MemAvailable: 32768 kB\n')
            group = root / 'cgroup/leaf'; group.mkdir(parents=True); (group / 'memory.events').write_text('oom 0\n')
            out = root / 'snapshot'; out.mkdir(); before = (child / 'smaps').read_bytes()
            r = diag.collect_snapshot(42, {'swap_bytes': 12288}, 'engine_swap', out, proc, root / 'cgroup')
            self.assertEqual(r['status'], 'complete'); self.assertEqual(r['trigger_sample']['swap_bytes'], 12288)
            self.assertEqual(r['smaps_swap_summary']['swap_kB_in_read_prefix'], 12)
            self.assertIn('error', r['sections']['maps'])
            self.assertTrue(any('error' in f for f in r['cgroup_v2']['files']))
            self.assertEqual((child / 'smaps').read_bytes(), before)
            self.assertEqual(json.loads((out / 'snapshot.json').read_text()), r)


class HelperChecks(unittest.TestCase):
    def test_timeout_keeps_original_trigger_path_and_does_not_succeed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(diag.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['helper'], 3)) as run:
            r = diag.capture_resource_failure(42, Path(temp), {'swap_bytes': 1}, 'test')
            self.assertEqual(r['status'], 'timed_out'); self.assertEqual(run.call_args.kwargs['timeout'], 3)
            self.assertEqual(json.loads(run.call_args.kwargs['input'])['trigger']['swap_bytes'], 1)

    def test_helper_exit_or_missing_snapshot_is_not_a_capture(self):
        for code in (0, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp, mock.patch.object(diag.subprocess, 'run', return_value=SimpleNamespace(returncode=code, stderr='failure')):
                self.assertEqual(diag.capture_resource_failure(42, Path(temp), {}, 'test')['status'], 'capture_failed')

    def test_live_metadata_helper_collects_only_current_test_process(self):
        with tempfile.TemporaryDirectory() as temp:
            r = diag.capture_resource_failure(os.getpid(), Path(temp), {'test_only': True}, 'metadata_API_smoke_not_a_model_failure')
            self.assertEqual(r['status'], 'captured', r)
            snapshot = json.loads(Path(r['snapshot_file']).read_text())
            self.assertEqual(snapshot['pid'], os.getpid())
            self.assertTrue(snapshot['smaps_swap_summary']['complete'], snapshot['smaps_swap_summary'])
            for name in ['status', 'maps', 'smaps_rollup']:
                self.assertNotIn('error', snapshot['sections'][name])
            self.assertIn('host/meminfo', snapshot['sections'])


class Budget:
    info = {'physical_bytes': 100, 'target_bytes': 80}
    held = 0
    def memory(self): return 90, 100


class GuardChecks(unittest.TestCase):
    def invoke(self, enabled, effect=None, gpu=False):
        report = {'cases': [{'id': 'guard', 'prompt_ids': [2, 3]}]}
        def memory(path):
            return {'MemAvailable': 32 << 30} if path.name == 'meminfo' else {'VmRSS': 1 << 20, 'VmHWM': 1 << 20, 'VmSwap': 0 if gpu else 4096}
        budget = Budget()
        if gpu: budget.memory = lambda: (0, 100)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(bench, 'proc_memory', side_effect=memory), mock.patch.object(diag, 'capture_resource_failure', side_effect=effect, return_value={'status': 'test_capture'}) as capture:
            message = 'Observed GPU use exceeded' if gpu else 'Host memory floor reached or engine used swap'
            with self.assertRaisesRegex(RuntimeError, message):
                bench.run_engine([sys.executable, '-u', '-c', 'import time;time.sleep(10)'], {}, Path(temp), budget, object(), report, 5, host_floor_bytes=8 << 30, resource_diagnostics=enabled)
            self.assertFalse(Path('/proc', str(report['engine_pid'])).exists())
            if enabled:
                capture.assert_called_once()
                self.assertEqual(capture.call_args.args[0], report['engine_pid'])
            else:
                capture.assert_not_called(); self.assertNotIn('resource_diagnostic', report)
        return report

    def test_disabled_diagnostics_preserve_existing_guard_and_cleanup(self):
        self.invoke(False)

    def test_swap_capture_happens_before_owned_child_is_terminated(self):
        def capture(pid, *args):
            self.assertTrue(Path('/proc', str(pid)).exists())
            return {'status': 'test_capture'}
        self.assertEqual(self.invoke(True, capture)['resource_diagnostic']['status'], 'test_capture')

    def test_capture_error_cannot_replace_guard_or_skip_cleanup(self):
        r = self.invoke(True, OSError('snapshot write denied'))
        self.assertEqual(r['resource_diagnostic']['status'], 'capture_failed')
        self.assertIn('snapshot write denied', r['resource_diagnostic']['error'])

    def test_gpu_guard_still_raises_after_capture(self):
        self.assertIn('resource_diagnostic', self.invoke(True, gpu=True))


class DiagnosticCliChecks(unittest.TestCase):
    def test_diagnostics_cannot_expand_the_fixed_reproduction_scope(self):
        base = ['--suite', 'resident', '--resident-order', 'code-explain', '--smoke-result', 'unused.json',
                '--exe', 'unused-engine', '--engine-sha256', 'a' * 64, '--store-stats', '--ssd-cache-gb', '40',
                '--ssd-policy', 'lru', '--prefill-chunk', '256', '--flow-store', 'on', '--mixed-prefill', 'off',
                '--first-window', 'off', '--settle-store', 'on', '--settle-ms', '1000', '--flow-store-min-heat', '1', '--resource-diagnostics']
        for flag, value in [('--flow-store-min-heat', '4'), ('--resident-order', 'explain-code'), ('--settle-store', 'off'), ('--gpu-cap-mib', '21000'), ('--settle-ms', '6000')]:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'must-not-exist'; args = base.copy()
                if flag in args: args[args.index(flag) + 1] = value
                else: args += [flag, value]
                with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(bench, 'GpuBudget') as gpu, mock.patch.object(bench, 'run_engine') as engine:
                    with self.assertRaises(SystemExit) as error: bench.main(args + ['--output-dir', str(output)])
                    self.assertEqual(error.exception.code, 2); gpu.assert_not_called(); engine.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

"""Experimental benchmark guard checks; no CUDA or model files required."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import benchmark_quantized as bench


class FlowStoreRetentionChecks(unittest.TestCase):
    def args(self, output):
        return ['--suite', 'resident', '--resident-order', 'explain-code',
                '--smoke-result', 'unused.json', '--output-dir', str(output),
                '--exe', 'unused-engine', '--engine-sha256', 'a' * 64,
                '--store-stats', '--ssd-cache-gb', '40', '--ssd-policy', 'lru',
                '--prefill-chunk', '256', '--flow-store', 'on', '--mixed-prefill', 'off',
                '--first-window', 'off', '--settle-store', 'on', '--settle-ms', '1000',
                '--flow-store-min-heat', '1']

    def rejected(self, args, output, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(bench, 'GpuBudget') as gpu, \
                mock.patch.object(bench, 'run_engine') as engine:
            with self.assertRaises(SystemExit) as error:
                bench.main(args, **kwargs)
            self.assertEqual(error.exception.code, 2)
            gpu.assert_not_called()
            engine.assert_not_called()
        self.assertFalse(output.exists())

    def test_incomplete_or_changed_configuration_fails_before_output_or_GPU(self):
        changes = {'--suite': 'long', '--resident-order': 'short-first',
                   '--ssd-cache-gb': '32', '--ssd-policy': 'lfu', '--gpu-cap-mib': '21000',
                   '--prefill-chunk': '0', '--flow-store': 'off', '--mixed-prefill': 'on',
                   '--first-window': 'on', '--settle-ms': '6000',
                   '--flow-store-min-heat': '2'}
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in changes.items():
                with self.subTest(flag=flag):
                    output = Path(temp) / flag[2:]
                    args = self.args(output)
                    if flag in args:
                        args[args.index(flag) + 1] = value
                    else:
                        args += [flag, value]
                    self.rejected(args, output)
            for flag in ('--exe', '--engine-sha256', '--store-stats', '--settle-store'):
                with self.subTest(missing=flag):
                    output = Path(temp) / ('missing-' + flag[2:])
                    args = self.args(output)
                    at = args.index(flag)
                    del args[at:at + (1 if flag == '--store-stats' else 2)]
                    self.rejected(args, output)

    def test_preload_off_cannot_bypass_fixed_heat1_resources_and_selectors(self):
        changes = {'--flow-store-min-heat': '4', '--settle-ms': '6000',
                   '--resident-order': 'short-first', '--flow-store': 'off',
                   '--prefill-chunk': '0', '--mixed-prefill': 'on', '--first-window': 'on'}
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in changes.items():
                with self.subTest(flag=flag):
                    output = Path(temp) / flag[2:]
                    args = self.args(output)
                    args[args.index('--settle-store') + 1] = 'off'
                    args[args.index(flag) + 1] = value
                    self.rejected(args, output)

    def test_public_text_and_first_only_cannot_select_retention_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            for label, extra, kwargs in (
                    ('text', [], {'request': {'prompt': 'hello', 'max_new_tokens': 64}}),
                    ('interactive', [], {'request_source': object()}),
                    ('public-profile', [], {'text_profile': 'flow-experimental', 'request': {}}),
                    ('first-only', ['--first-request-only'], {})):
                with self.subTest(label=label):
                    output = Path(temp) / label
                    self.rejected(self.args(output) + extra, output, **kwargs)

    def test_two_task_orders_require_explicit_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            for order in ('explain-code', 'code-explain'):
                output = Path(temp) / order
                args = self.args(output)
                args[args.index('--resident-order') + 1] = order
                del args[args.index('--flow-store-min-heat'):]
                self.rejected(args, output)

    def test_original_fixed_fixtures_and_caps_are_reused(self):
        self.assertEqual(bench.RESIDENT_ORDERS['explain-code'], bench.LONG_CASES[1:])
        self.assertEqual(bench.RESIDENT_ORDERS['code-explain'], bench.LONG_CASES[:0:-1])
        self.assertTrue(all(c['max_new_tokens'] == 256 for c in bench.RESIDENT_ORDERS['explain-code']))


if __name__ == '__main__':
    unittest.main()

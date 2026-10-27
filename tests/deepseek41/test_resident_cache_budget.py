"""36GB is a bounded four-request diagnostic, not a public profile."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import benchmark_quantized as bench


class ResidentCacheBudgetChecks(unittest.TestCase):
    def args(self, output):
        return ['--suite', 'resident', '--resident-order', 'code-explain-repeat',
                '--smoke-result', 'unused.json', '--output-dir', str(output),
                '--exe', 'unused-engine', '--engine-sha256', 'a' * 64, '--store-stats',
                '--ssd-cache-gb', '36', '--ssd-policy', 'lru', '--prefill-chunk', '256',
                '--flow-store', 'on', '--mixed-prefill', 'off', '--first-window', 'off',
                '--settle-store', 'on', '--settle-ms', '1000', '--flow-store-min-heat', '1',
                '--resource-diagnostics']

    def rejected(self, args, output, **options):
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(bench, 'GpuBudget') as gpu, mock.patch.object(bench, 'run_engine') as engine:
            with self.assertRaises(SystemExit) as error: bench.main(args, **options)
            self.assertEqual(error.exception.code, 2); gpu.assert_not_called(); engine.assert_not_called()
        self.assertFalse(output.exists())

    def test36_requires_complete_repeated_diagnostic_resources(self):
        changes = {'--suite': 'long', '--resident-order': 'code-explain', '--flow-store-min-heat': '4',
                   '--settle-store': 'off', '--settle-ms': '6000', '--ssd-policy': 'lfu',
                   '--gpu-cap-mib': '21000', '--prefill-chunk': '0', '--flow-store': 'off',
                   '--mixed-prefill': 'on', '--first-window': 'on'}
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in changes.items():
                with self.subTest(flag=flag):
                    output = Path(temp) / flag[2:]; args = self.args(output)
                    if flag in args: args[args.index(flag) + 1] = value
                    else: args += [flag, value]
                    self.rejected(args, output)
            for flag in ['--resource-diagnostics', '--store-stats', '--flow-store-min-heat', '--exe', '--engine-sha256', '--resident-order']:
                with self.subTest(missing=flag):
                    output = Path(temp) / ('missing-' + flag[2:]); args = self.args(output); at = args.index(flag)
                    del args[at:at + (1 if flag in ['--resource-diagnostics', '--store-stats'] else 2)]
                    self.rejected(args, output)
            output = Path(temp) / 'first-only'; self.rejected(self.args(output) + ['--first-request-only'], output)

    def test36_cannot_enable_public_or_interactive_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, options in [('text', {'request': {'prompt': 'hello', 'max_new_tokens': 64}}),
                                  ('interactive', {'request_source': object()}),
                                  ('profile', {'text_profile': 'flow-experimental', 'request': {}})]:
                output = Path(temp) / name; self.rejected(self.args(output), output, **options)

    def test36_cannot_be_used_as_ordinary_cache_override(self):
        with tempfile.TemporaryDirectory() as temp:
            for suite in ['short', 'long', 'resident']:
                output = Path(temp) / suite
                self.rejected(['--suite', suite, '--smoke-result', 'unused.json', '--output-dir', str(output), '--ssd-cache-gb', '36'], output)


if __name__ == '__main__':
    unittest.main()

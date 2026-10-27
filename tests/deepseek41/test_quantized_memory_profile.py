"""Explicit36GB public profile pins and shared text behavior, without GPU/model."""
import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import benchmark_quantized as bench
import run_quantized as entry
from test_quantized_text import ByteTokenizer


class MemoryProfileChecks(unittest.TestCase):
    def assert_profile(self, args, options):
        self.assertEqual(options['text_profile'], 'memory-experimental')
        self.assertEqual(options['expected_pack_sha256'], entry.PACK_SHA256)
        self.assertIn('build-deepseek-retention-sm75', args[args.index('--exe') + 1])
        for flag, value in [('--engine-sha256', 'd6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553'),
                            ('--ssd-cache-gb', '36'), ('--gpu-cap-mib', '20480'), ('--ssd-policy', 'lru'),
                            ('--prefill-chunk', '256'), ('--flow-store', 'on'), ('--mixed-prefill', 'off'),
                            ('--first-window', 'off'), ('--settle-store', 'on'), ('--settle-ms', '1000'),
                            ('--flow-store-min-heat', '1')]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--resource-diagnostics', args)
        self.assertIn('--store-stats', args)

    def test_single_prompt_file_and_stdin_share_pinned_streaming(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); question = root / 'question.txt'; question.write_text('你好😀', encoding='utf-8')
            for index, source in enumerate([['--prompt', '你好😀'], ['--prompt-file', str(question)], ['--prompt-file', '-']]):
                out = root / str(index); stdout, stderr = io.StringIO(), io.StringIO()
                def runtime(args, **options):
                    self.assert_profile(args, options)
                    self.assertEqual(options['request']['prompt'], '你好😀')
                    self.assertEqual(options['request']['max_new_tokens'], 64)
                    for token in (2, 3, 4, 5, 1, None): options['on_token'](ByteTokenizer(), token)
                    out.mkdir(); (out / 'result.json').write_text(json.dumps({'cases': [{'finish_reason': 'stop'}]})); return 0
                with mock.patch.object(bench, 'main', side_effect=runtime), mock.patch('sys.stdin', io.StringIO('你好😀')), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(entry.main(['--profile', 'memory-experimental', *source, '--max-new-tokens', '64', '--output-dir', str(out)]), 0)
                self.assertEqual(stdout.getvalue(), '中😀\n'); self.assertIn('36 GB RAM 缓存', stderr.getvalue())
                self.assertIn('部分任务更慢', stderr.getvalue()); self.assertNotIn('<EOS>', stdout.getvalue())

    def test_resident_source_and_answer_callbacks_are_reused(self):
        source = SimpleNamespace(max_new_tokens=256)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'result'; stdout, stderr = io.StringIO(), io.StringIO()
            def runtime(args, **options):
                self.assert_profile(args, options); self.assertIs(options['request_source'], source); self.assertNotIn('request', options)
                cases = [{'finish_reason': 'length'}, {'finish_reason': 'stop'}]
                for case in cases:
                    for token in (2, 3, 1, None): options['on_token'](ByteTokenizer(), token)
                    options['on_case_end'](case)
                out.mkdir(); (out / 'result.json').write_text(json.dumps({'cases': cases})); return 0
            with mock.patch.object(entry, 'LineInput', return_value=source) as reader, mock.patch.object(bench, 'main', side_effect=runtime), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(entry.main(['--profile', 'memory-experimental', '--interactive', '--max-new-tokens', '256', '--output-dir', str(out)]), 0)
            reader.assert_called_once_with(256); self.assertEqual(stdout.getvalue(), '中\n中\n')
            self.assertIn('回答可能未完成', stderr.getvalue()); self.assertIn('共完成 2 个问题', stderr.getvalue())

    def args(self, output):
        return ['--smoke-result', 'unused.json', '--output-dir', str(output), '--exe', 'unused-engine',
                '--engine-sha256', bench.TEXT_MEMORY_ENGINE_SHA256, '--store-stats', '--ssd-cache-gb', '36',
                '--ssd-policy', 'lru', '--gpu-cap-mib', '20480', '--prefill-chunk', '256', '--flow-store', 'on',
                '--mixed-prefill', 'off', '--first-window', 'off', '--settle-store', 'on', '--settle-ms', '1000',
                '--flow-store-min-heat', '1', '--resource-diagnostics']

    def rejected(self, args, output, **options):
        with mock.patch.object(bench, 'GpuBudget') as gpu, mock.patch.object(bench, 'run_engine') as engine, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: bench.main(args, **options)
            self.assertEqual(error.exception.code, 2); gpu.assert_not_called(); engine.assert_not_called()
        self.assertFalse(output.exists())

    def test_incomplete_resource_or_engine_configuration_fails_before_GPU(self):
        changes = {'--engine-sha256': '0' * 64, '--ssd-cache-gb': '40', '--ssd-policy': 'lfu',
                   '--gpu-cap-mib': '10240', '--prefill-chunk': '0', '--flow-store': 'off',
                   '--mixed-prefill': 'on', '--first-window': 'on', '--settle-store': 'off',
                   '--settle-ms': '6000', '--flow-store-min-heat': '4'}
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in changes.items():
                with self.subTest(flag=flag):
                    out = Path(temp) / flag[2:]; args = self.args(out); args[args.index(flag) + 1] = value
                    self.rejected(args, out, request={'prompt': '你好', 'max_new_tokens': 64}, text_profile='memory-experimental', expected_pack_sha256=entry.PACK_SHA256)
            for flag in ['--exe', '--engine-sha256', '--store-stats', '--resource-diagnostics', '--flow-store-min-heat']:
                with self.subTest(missing=flag):
                    out = Path(temp) / ('missing-' + flag[2:]); args = self.args(out); at = args.index(flag)
                    del args[at:at + (1 if flag in ['--store-stats', '--resource-diagnostics'] else 2)]
                    self.rejected(args, out, request={'prompt': '你好', 'max_new_tokens': 64}, text_profile='memory-experimental', expected_pack_sha256=entry.PACK_SHA256)

    def test_profile_cannot_bypass_text_source_or_old_profile_scope(self):
        request = {'prompt': '你好', 'max_new_tokens': 64}
        variants = [('no-source', {}, [], 'memory-experimental', entry.PACK_SHA256),
                    ('two-sources', {'request': request, 'request_source': object()}, [], 'memory-experimental', entry.PACK_SHA256),
                    ('no-pack', {'request': request}, [], 'memory-experimental', None),
                    ('benchmark-order', {'request': request}, ['--suite', 'resident', '--resident-order', 'code-explain-repeat'], 'memory-experimental', entry.PACK_SHA256),
                    ('first-only', {'request': request}, ['--first-request-only'], 'memory-experimental', entry.PACK_SHA256),
                    ('old-profile', {'request': request}, [], 'flow-experimental', entry.PACK_SHA256),
                    ('no-profile', {'request': request}, [], None, entry.PACK_SHA256)]
        with tempfile.TemporaryDirectory() as temp:
            for name, options, extra, profile, pack in variants:
                with self.subTest(name=name):
                    out = Path(temp) / name; self.rejected(self.args(out) + extra, out, **options, text_profile=profile, expected_pack_sha256=pack)

    def test_bad_public_input_caps_and_timeout_fail_before_runner(self):
        with mock.patch.object(bench, 'main') as runtime, contextlib.redirect_stderr(io.StringIO()):
            for args in [['--prompt', ' '], ['--prompt', '中', '--max-new-tokens', '0'], ['--prompt', '中', '--max-new-tokens', '257'], ['--prompt', '中', '--timeout', '0']]:
                with self.subTest(args=args), self.assertRaises(SystemExit) as error: entry.main(['--profile', 'memory-experimental', *args])
                self.assertEqual(error.exception.code, 2)
            runtime.assert_not_called()

    def test_runtime_error_and_interrupt_restore_signal_handler(self):
        original = signal.getsignal(signal.SIGTERM)
        for error, code in [(RuntimeError('controlled error'), 1), (KeyboardInterrupt(), 130)]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp, mock.patch.object(bench, 'main', side_effect=error) as runtime, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(entry.main(['--profile', 'memory-experimental', '--prompt', '你好', '--output-dir', str(Path(temp) / 'out')]), code)
                self.assertEqual(runtime.call_args.kwargs['text_profile'], 'memory-experimental')
                self.assertEqual(signal.getsignal(signal.SIGTERM), original)


if __name__ == '__main__':
    unittest.main()

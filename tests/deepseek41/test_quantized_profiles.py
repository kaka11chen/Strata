"""Explicit text profiles keep the shared runner's pins, streaming and admission bounds."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import benchmark_quantized as bench
import run_quantized as entry
from test_quantized_text import ByteTokenizer


class ProfileChecks(unittest.TestCase):
    def test_single_question_profile_selection_keeps_pins_and_streaming(self):
        for profile in (None, 'windows', 'flow-experimental'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / 'result'; stdout, stderr = io.StringIO(), io.StringIO()
                def runtime(args, **options):
                    experimental = profile == 'flow-experimental'
                    self.assertEqual(options['request']['prompt'], '你好')
                    self.assertEqual(options['request']['max_new_tokens'], 64)
                    self.assertEqual(options['expected_pack_sha256'], entry.PACK_SHA256)
                    self.assertEqual(args[args.index('--gpu-cap-mib') + 1], '20480')
                    self.assertEqual(args[args.index('--ssd-cache-gb') + 1], '40')
                    if experimental:
                        self.assertEqual(options['text_profile'], profile)
                        self.assertIn('build-deepseek-first-window-sm75', args[args.index('--exe') + 1])
                        self.assertEqual(args[args.index('--engine-sha256') + 1], 'a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96')
                        for flag, value in [('--prefill-chunk', '256'), ('--settle-ms', '1000'), ('--flow-store', 'on'), ('--settle-store', 'on'), ('--mixed-prefill', 'off'), ('--first-window', 'off'), ('--ssd-policy', 'lru')]:
                            self.assertEqual(args[args.index(flag) + 1], value)
                    else:
                        self.assertNotIn('text_profile', options)
                        self.assertIn(entry.ENGINE_SHA256, args)
                        self.assertNotIn('--prefill-chunk', args)
                        self.assertNotIn('--settle-ms', args)
                    for token in (6, 1, None): options['on_token'](ByteTokenizer(), token)
                    out.mkdir(); (out / 'result.json').write_text(json.dumps({'cases': [{'finish_reason': 'stop'}]}))
                    return 0
                args = ['--prompt', '你好', '--max-new-tokens', '64', '--output-dir', str(out)]
                if profile is not None: args += ['--profile', profile]
                with mock.patch.object(entry.benchmark_quantized, 'main', side_effect=runtime), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(entry.main(args), 0)
                self.assertEqual(stdout.getvalue(), 'A\n')
                self.assertEqual('使用实验配置' in stderr.getvalue(), profile == 'flow-experimental')

    def test_interactive_experimental_profile_reuses_source_and_finish_callback(self):
        source = object()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'result'; stdout, stderr = io.StringIO(), io.StringIO()
            def runtime(args, **options):
                self.assertIs(options['request_source'], source)
                self.assertNotIn('request', options)
                self.assertEqual(options['text_profile'], 'flow-experimental')
                self.assertEqual(options['expected_pack_sha256'], entry.PACK_SHA256)
                cases = [{'finish_reason': 'length'}, {'finish_reason': 'stop'}]
                for case in cases:
                    for token in (6, None): options['on_token'](ByteTokenizer(), token)
                    options['on_case_end'](case)
                out.mkdir(); (out / 'result.json').write_text(json.dumps({'cases': cases}))
                return 0
            with mock.patch.object(entry, 'LineInput', return_value=source) as reader, mock.patch.object(entry.benchmark_quantized, 'main', side_effect=runtime), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(entry.main(['--profile', 'flow-experimental', '--interactive', '--output-dir', str(out)]), 0)
            reader.assert_called_once_with(128)
            self.assertEqual(stdout.getvalue(), 'A\nA\n')
            self.assertIn('回答可能未完成', stderr.getvalue())
            self.assertIn('共完成 2 个问题', stderr.getvalue())

    def test_unknown_public_profile_fails_before_runner(self):
        with mock.patch.object(entry.benchmark_quantized, 'main') as runtime, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            entry.main(['--prompt', '你好', '--profile', 'unknown'])
        self.assertEqual(error.exception.code, 2)
        runtime.assert_not_called()

    def test_text_profile_rejects_incomplete_or_incompatible_configuration(self):
        base = ['--smoke-result', '/not-loaded/smoke.json', '--exe', '/not-loaded/engine', '--engine-sha256', bench.TEXT_FLOW_ENGINE_SHA256, '--store-stats', '--ssd-cache-gb', '40', '--ssd-policy', 'lru', '--gpu-cap-mib', '20480', '--prefill-chunk', '256', '--flow-store', 'on', '--mixed-prefill', 'off', '--first-window', 'off', '--settle-store', 'on', '--settle-ms', '1000']
        request = {'id': 'text', 'prompt': '你好', 'max_new_tokens': 64}
        cases = []
        for flag, value in [('--exe', None), ('--engine-sha256', '0' * 64), ('--store-stats', None), ('--ssd-cache-gb', '32'), ('--ssd-policy', 'lfu'), ('--gpu-cap-mib', '10240'), ('--prefill-chunk', '0'), ('--flow-store', 'off'), ('--mixed-prefill', 'on'), ('--first-window', 'on'), ('--settle-store', 'off'), ('--settle-ms', '6000')]:
            args = base.copy(); at = args.index(flag)
            if value is None: del args[at:at + (1 if flag == '--store-stats' else 2)]
            else: args[at + 1] = value
            cases.append((flag, args, request, None, 'flow-experimental', {}))
        cases += [('no_text', base, None, None, 'flow-experimental', {}), ('two_sources', base, request, object(), 'flow-experimental', {}), ('no_pack_pin', base, request, None, 'flow-experimental', None), ('unknown_internal_profile', base, request, None, 'unknown', {}), ('resident_order', base + ['--resident-order', 'short-first'], request, None, 'flow-experimental', {}), ('first_only', base + ['--first-request-only'], request, None, 'flow-experimental', {})]
        with tempfile.TemporaryDirectory() as temp:
            for index, (label, args, one, source, profile, pack) in enumerate(cases):
                out = Path(temp) / str(index)
                with self.subTest(label=label), mock.patch.object(bench, 'GpuBudget') as gpu, mock.patch.object(bench, 'run_engine') as engine, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    bench.main(args + ['--output-dir', str(out)], request=one, request_source=source, text_profile=profile, expected_pack_sha256=pack)
                self.assertEqual(error.exception.code, 2)
                gpu.assert_not_called(); engine.assert_not_called()
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

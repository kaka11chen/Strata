"""Resident input, continuation and cleanup without model files or CUDA."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import run_quantized
from benchmark_quantized import run_engine
from test_quantized_benchmark import Budget
from test_quantized_text import ByteTokenizer


class PromptTokenizer(ByteTokenizer):
    def encode(self, text, **kwargs):
        self.prompt = text
        self.encoded = [2] * 2000 if len(text) > 2000 else [2, 3]
        return self.encoded

    def decode(self, ids):
        return self.prompt if ids == self.encoded else super().decode(ids)


class SessionChecks(unittest.TestCase):
    def test_line_input_blank_exit_and_final_line_without_newline(self):
        for text in ('\n你好😀\r\n/exit\n', '你好😀'):
            read_fd, write_fd = os.pipe()
            try:
                os.write(write_fd, text.encode('utf-8')); os.close(write_fd)
                source = run_quantized.LineInput(128, read_fd)
                ticks = []
                with contextlib.redirect_stderr(io.StringIO()):
                    request = source(lambda: ticks.append(True))
                    self.assertEqual(request['prompt'], '你好😀')
                    self.assertIsNone(source(lambda: ticks.append(True)))
                self.assertGreater(len(ticks), 1)
            finally:
                os.close(read_fd)

    def test_input_utf8_can_span_reads(self):
        source = run_quantized.LineInput(128, 0)
        with mock.patch('run_quantized.select.select', return_value=([0], [], [])), \
             mock.patch('run_quantized.os.read', side_effect=[b'\xe4', b'\xb8\xad\n', b'']):
            self.assertEqual(source.read_line(lambda: None), '中')
            self.assertIsNone(source.read_line(lambda: None))

    def test_reply_stream_resets_newline_state(self):
        class NewlineTokenizer(ByteTokenizer):
            parts = {**ByteTokenizer.parts, 7: b'\n'}
        output = io.StringIO(); stream = run_quantized.TextStream(output)
        with contextlib.redirect_stderr(io.StringIO()):
            for ids in ((6, 7, 1, None), (1, None)):
                for token in ids: stream(NewlineTokenizer(), token)
                stream.finish_case({'finish_reason': 'stop'})
        self.assertEqual(output.getvalue(), 'A\n\n')

    SCRIPT = '''
import sys
print('READY 1024 stop', flush=True)
for cap, count, reason in ((1, 1, 'length'), (8, 2, 'stop')):
    assert sys.stdin.readline() == 'GEN %d temperature=0 spec=0 2,3\\n' % cap
    print('T 6', flush=True)
    if count == 2: print('T 1', flush=True)
    print('STAT tok_s=10 vram_hit=0.5 disk_mb=1', flush=True)
    print('DONE %d 2 1.0 %d %s 0 0 0 0 0' % (count, count-1, reason), flush=True)
assert sys.stdin.readline() == 'QUIT\\n'
'''

    def test_idle_excludes_timeout_and_rejected_or_truncated_input_can_continue(self):
        report = {'cases': []}; output = io.StringIO(); stream = run_quantized.TextStream(output)
        requests = iter([{'prompt': '先截断', 'max_new_tokens': 1},
                         {'prompt': '过长' * 2000, 'max_new_tokens': 8},
                         {'prompt': '再回答', 'max_new_tokens': 8}, None])
        first = True
        def source(tick):
            nonlocal first
            if first:
                first = False; until = time.monotonic() + 1.2
                while time.monotonic() < until:
                    tick(); time.sleep(.05)
            return next(requests)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            out = Path(temp)
            run_engine([sys.executable, '-u', '-c', self.SCRIPT], {}, out, Budget(), PromptTokenizer(), report, 1,
                       on_token=stream, request_source=source, on_case_end=stream.finish_case)
            protocol = (out / 'engine.stdout').read_text()
            memory = [json.loads(x) for x in (out / 'memory.jsonl').read_text().splitlines()]
        self.assertEqual(report['engine_returncode'], 0)
        self.assertEqual(protocol.count('READY '), 1)
        self.assertEqual([c['finish_reason'] for c in report['cases']], ['length', 'stop'])
        self.assertEqual([c['id'] for c in report['cases']], ['text-1', 'text-2'])
        self.assertTrue(all(c['reused_tokens'] == 0 for c in report['cases']))
        self.assertEqual(len(report['input_rejections']), 1)
        self.assertTrue(any(s['phase'] == 'idle' for s in memory))
        self.assertEqual(output.getvalue(), 'A\nA\n')
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())

    def test_native_exit_is_detected_while_waiting_for_input(self):
        report = {'cases': []}
        def source(tick):
            until = time.monotonic() + 2
            while time.monotonic() < until:
                tick(); time.sleep(.02)
            self.fail('Native exit was not detected during the bounded input wait')
        script = "import time;print('READY 1024 stop',flush=True);time.sleep(.1);raise SystemExit(7)"
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, 'waiting|idle'):
            run_engine([sys.executable, '-u', '-c', script], {}, Path(temp), Budget(), PromptTokenizer(), report, 1,
                       request_source=source)
        self.assertEqual(report['engine_returncode'], 7)
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())

    def test_interactive_entry_uses_shared_runtime_and_per_answer_finish(self):
        source = object()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'result'; stdout, stderr = io.StringIO(), io.StringIO()
            def run(args, *, request_source, on_case_end, on_token, expected_pack_sha256):
                self.assertIs(request_source, source)
                self.assertEqual(expected_pack_sha256, run_quantized.PACK_SHA256)
                cases = [{'finish_reason': 'length'}, {'finish_reason': 'stop'}]
                for case in cases:
                    for token in (6, None): on_token(ByteTokenizer(), token)
                    on_case_end(case)
                out.mkdir(); (out / 'result.json').write_text(json.dumps({'cases': cases}))
                return 0
            with mock.patch.object(run_quantized, 'LineInput', return_value=source), \
                 mock.patch.object(run_quantized.benchmark_quantized, 'main', side_effect=run), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = run_quantized.main(['--interactive', '--output-dir', str(out)])
            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), 'A\nA\n')
            self.assertIn('回答可能未完成', stderr.getvalue())
            self.assertIn('共完成 2 个问题', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()

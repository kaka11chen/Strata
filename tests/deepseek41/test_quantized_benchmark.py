"""Measurement checks without model files or CUDA; real GPU evidence is separate."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
from benchmark_quantized import done_metrics, parse_store_line, run_engine


class Budget:
    info = {'physical_bytes': 100}
    held = 20

    def memory(self):
        return 10, 90


class Tokenizer:
    tokens = list(range(8))

    def decode(self, ids):
        return ','.join(map(str, ids))


class BenchmarkChecks(unittest.TestCase):
    def test_first_token_is_not_a_decode_step(self):
        result = done_metrics('DONE 3 5 200.0 100.0 stop 0 0 0 0 0', [10.2, 10.25, 10.3], 10, 5)
        self.assertEqual(result['decode_steps'], 2)
        self.assertEqual(result['decode_tokens_per_s'], 20)
        self.assertAlmostEqual(result['stream_tokens_per_s'], 20)
        self.assertAlmostEqual(result['first_token_s'], .2)

    def test_done_respects_the_requested_output_cap(self):
        line = 'DONE 256 5 20.0 25500.0 length 0 0 0 0 0'
        times = [1 + i / 10 for i in range(256)]
        result = done_metrics(line, times, .9, 5, 256)
        self.assertEqual(result['decode_steps'], 255)
        self.assertEqual(result['decode_tokens_per_s'], 10)
        with self.assertRaises(ValueError):
            done_metrics(line, times, .9, 5)
        with self.assertRaises(ValueError):
            done_metrics(line, times, .9, 5, 128)

    def test_resident_requests_have_individual_caps_and_finish_reasons(self):
        script = '''
import sys
print('READY 1024 stop',flush=True)
for cap,reason in ((128,'stop'),(256,'length')):
    line=sys.stdin.readline()
    assert line=='GEN %d temperature=0 spec=0 2,3\\n'%cap,line
    ids=[4]*(cap-1)+[1 if reason=='stop' else 4]
    for token in ids: print('T '+str(token),flush=True)
    print('STAT tok_s=10 vram_hit=0.5 disk_mb=1',flush=True)
    print('DONE %d 2 20.0 %d %s 0 0 0 0 0'%(cap,(cap-1)*100,reason),flush=True)
assert sys.stdin.readline()=='QUIT\\n'
'''
        report = {'cases': [{'id': 'stop128', 'prompt_ids': [2, 3], 'max_new_tokens': 128},
                            {'id': 'length256', 'prompt_ids': [2, 3], 'max_new_tokens': 256}]}
        with tempfile.TemporaryDirectory() as temp:
            run_engine([sys.executable, '-u', '-c', script], {}, Path(temp), Budget(), Tokenizer(), report, 5)
        self.assertEqual([len(c['output_ids']) for c in report['cases']], [128, 256])
        self.assertEqual([c['finish_reason'] for c in report['cases']], ['stop', 'length'])
        self.assertEqual([c['decode_steps'] for c in report['cases']], [127, 255])
        self.assertTrue(all(c['decode_tokens_per_s'] == 10 for c in report['cases']))
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())

    def test_eos_from_prefill_has_no_decode_rate(self):
        result = done_metrics('DONE 1 5 10.0 0.0 stop 0 0 0 0 0', [2.01], 2, 5)
        self.assertIsNone(result['decode_tokens_per_s'])
        self.assertIsNone(result['stream_tokens_per_s'])

    def test_bad_counts_and_cancelled_or_reused_runs_fail(self):
        for line in ('DONE 2 5 10.0 1.0 stop 0 0 0 0 0',
                     'DONE 1 4 10.0 1.0 stop 0 0 0 0 0',
                     'DONE 1 5 10.0 1.0 cancel 0 0 0 0 0',
                     'DONE 1 5 10.0 1.0 stop 0 0 1 0 0',
                     'DONE 1 5 10.0 1.0 stop 1 2 0 0 0'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                done_metrics(line, [1.1], 1, 5)

    STORE = ('STORE phase=decode tokens=2 hits=6 misses=2 submitted_bytes=5000000 '
             'wait_ms=500.250 cpu_lane_ms=700.500 evictions=3 read_fails=1 '
             'prefetched=3 prefetch_used=2 prefetch_wasted=1 slots=9 slots_max=40')

    def test_store_uses_phase_counts_and_explicit_units(self):
        result = parse_store_line(self.STORE)
        self.assertEqual(result['hit_ratio'], .75)
        self.assertEqual(result['submitted_mb_per_token'], 2.5)
        self.assertEqual(result['wait_ms_per_token'], 250.125)
        self.assertEqual(result['slots'], 9)
        self.assertEqual(result['read_fails'], 1)

    def test_zero_lookup_and_token_denominators_are_unavailable(self):
        result = parse_store_line(self.STORE.replace('tokens=2', 'tokens=0').replace('hits=6', 'hits=0').replace('misses=2', 'misses=0'))
        for key in ('hit_ratio', 'submitted_mb_per_token', 'wait_ms_per_token'):
            self.assertIsNone(result[key])

    def test_invalid_store_counters_fail(self):
        for line in (self.STORE.replace('hits=6', 'hits=-1'), self.STORE.replace('wait_ms=500.250', 'wait_ms=nan'),
                     self.STORE.replace('slots=9', 'slots=41'), self.STORE + ' hits=1',
                     self.STORE.replace('phase=decode', 'phase=unknown'), self.STORE.replace('read_fails=1 ', '')):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_store_line(line)

    def test_per_phase_diagnostics_are_required_when_enabled(self):
        script = "import sys;print('READY 1024 stop',flush=True);sys.stdin.readline();print('T 1\\nSTAT disk_mb=0 vram_hit=1\\nDONE 1 2 1.0 0.0 stop 0 0 0 0 0',flush=True);sys.stdin.readline()"
        report = {'cases': [{'id': 'missing', 'prompt_ids': [2, 3]}]}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Missing per-phase'):
                run_engine([sys.executable, '-u', '-c', script], {'DS4_STORE_STATS': '1'}, Path(temp), Budget(), Tokenizer(), report, 5)
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())

    def test_split_stdout_lines_and_resident_requests(self):
        script = '''
import os,sys,time
print('INFO test=1',flush=True)
print('READY 1024 stop',flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'): break
    assert line.startswith('GEN 64 temperature=0 spec=0 2,3')
    sys.stdout.write('T ');sys.stdout.flush();time.sleep(.02)
    print('4',flush=True)
    time.sleep(.02)
    print('T 5',flush=True)
    print('T 1',flush=True)
    print('STAT tok_s=30.00 vram_hit=0.5 disk_mb=2.0',flush=True)
    if os.getenv('DS4_STORE_STATS') == '1':
        for phase in ('prefill', 'decode'):
            print('STORE phase='+phase+' tokens=2 hits=6 misses=2 submitted_bytes=5000000 wait_ms=50 cpu_lane_ms=70 evictions=0 read_fails=0 prefetched=0 prefetch_used=0 prefetch_wasted=0 slots=9 slots_max=40',flush=True)
    print('DONE 3 2 20.0 100.0 stop 0 0 0 0 0',flush=True)
'''
        report = {'cases': [{'id': 'a', 'prompt_ids': [2, 3]}, {'id': 'b', 'prompt_ids': [2, 3]}]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            run_engine([sys.executable, '-u', '-c', script], {}, out, Budget(), Tokenizer(), report, 5)
            self.assertEqual(report['engine_returncode'], 0)
            for case in report['cases']:
                self.assertEqual(case['output_ids'], [4, 5, 1])
                self.assertEqual(case['decode_tokens_per_s'], 20)
                self.assertGreater(case['first_token_s'], 0)
                self.assertEqual(case['native_stats'][-1]['tok_s'], 30)
                self.assertGreaterEqual(case['process_read_bytes'], 0)
            self.assertEqual(len([json.loads(s) for s in (out / 'events.jsonl').read_text().splitlines() if 'DONE' in s]), 2)
            self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())
            enabled = {'cases': [{'id': 'a', 'prompt_ids': [2, 3]}, {'id': 'b', 'prompt_ids': [2, 3]}]}
            run_engine([sys.executable, '-u', '-c', script], {'DS4_STORE_STATS': '1'}, out, Budget(), Tokenizer(), enabled, 5)
            for case in enabled['cases']:
                self.assertEqual(case['output_ids'], [4, 5, 1])
                self.assertEqual(set(case['store_stats']), {'prefill', 'decode'})
                self.assertEqual(case['store_stats']['decode']['hit_ratio'], .75)
                self.assertEqual(case['store_stats']['decode']['wait_over_phase_wall'], .5)
            self.assertFalse(Path(f"/proc/{enabled['engine_pid']}").exists())

    def test_timeout_reaps_the_engine(self):
        script = "import sys,time;print('READY 1024 stop',flush=True);sys.stdin.readline();time.sleep(30)"
        report = {'cases': [{'id': 'timeout', 'prompt_ids': [2, 3]}]}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TimeoutError):
                run_engine([sys.executable, '-u', '-c', script], {}, Path(temp), Budget(), Tokenizer(), report, 1)
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())
        self.assertNotEqual(report['engine_returncode'], 0)

    def test_early_exit_is_not_success(self):
        report = {'cases': []}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                run_engine([sys.executable, '-c', 'pass'], {}, Path(temp), Budget(), Tokenizer(), report, 5)
        self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())


if __name__ == '__main__':
    unittest.main()

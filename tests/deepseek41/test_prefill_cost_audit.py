"""Offline trace controls: phase boundaries, nested timings, incomplete data and immutable inputs."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import audit_prefill_costs as audit

CALIBRATION = "ds4: prompts through the flow or windows by this PC's speeds: DMA 6.4 GB/s, RAM to staging 14.3 GB/s, the disk 0.47 GB/s; the CPU lane 7.8 GB/s in RAM, 0.90 GB/s from the disk (345 ms)\n"
DECISION = "ds4: a prompt of 14 tokens through the flow: predicted 39.65 s, windows 20.70 s (9% of its CPU-lane experts' 20.2 GB in RAM)\n"
STREAM = 'ds4 prompt stream, CUDA0: 14 tokens in 18380 ms; 1349 experts streamed: 1349 from the files (10.0 GB, 0.54 GB/s: 8.8 GB from the disk unbuffered, 1.2 GB through the page cache), 0 from RAM (0.0 GB); 0 likely ones queued early, 0 of them unpicked; the compute waited 15612 ms for reads\n'
HANDOFF = 'ds4 flow store: retained=226 bytes=1701642240 present=0 no_room=0 early=0 invalid=0 copy_ms=2162.728\n'
PRELOAD = 'ds4: RAM store: 0 hot experts read after the prompt (0 ms)\n'
SETTLE = 'ds4: expert cache: 327 swaps after the prompt (6001 ms)\n'
END = 'maya ds4 serve: prompt 14 tokens = 0 reused + 14 read in 24401 ms, 6 generated\n'
BLOCK = DECISION + STREAM + HANDOFF + PRELOAD + SETTLE + END


def result():
    return {'status': 'measured_awaiting_text_review', 'engine_returncode': 0,
            'engine_sha256': 'fixture', 'method': {'prefill_chunk': 256},
            'cases': [{'id': 'short', 'prompt_ids': list(range(14)), 'native_prefill_ms': 24401.6,
                       'native_decode_ms': 7864.4, 'first_token_s': 24.403, 'request_wall_s': 32.267,
                       'process_read_bytes': 15008415744, 'output_ids': [17607, 2793, 3139, 6127, 320, 1],
                       'reused_tokens': 0}]}


class PrefillCostAuditTests(unittest.TestCase):
    def test_nested_times_and_io_populations_stay_distinct(self):
        row = audit.audit_trace(result(), CALIBRATION + BLOCK)['requests'][0]
        self.assertAlmostEqual(row['prefill_minus_stream_and_settle_ms'], 20.6)
        self.assertEqual(row['stream']['read_wait_ms'], 15612)
        self.assertEqual(row['handoff']['copy_ms'], 2162.728)
        self.assertEqual(row['stream']['buffered_payload_share_pct'], 12)
        self.assertAlmostEqual(row['estimated_expert_to_file_payload_ratio'], 2.02)
        self.assertEqual(row['whole_request_process_read_bytes'], 15008415744)
        self.assertIsNone(row['prefill_physical_read_bytes'])
        self.assertIsNone(row['exact_prefill_cache_hit_pct'])

    def test_same_prompt_in_two_requests_preserves_order(self):
        r = result(); second = copy.deepcopy(r['cases'][0]); second.update(id='repeat', native_prefill_ms=10998.9)
        r['cases'].append(second)
        block2 = BLOCK.replace('18380 ms', '8665 ms').replace('15612 ms', '6019 ms').replace('(6001 ms)', '(2306 ms)')
        rows = audit.audit_trace(r, CALIBRATION + BLOCK + block2)['requests']
        self.assertEqual([x['case'] for x in rows], ['short', 'repeat'])
        self.assertEqual([x['stream']['wall_ms'] for x in rows], [18380, 8665])
        self.assertAlmostEqual(rows[1]['prefill_minus_stream_and_settle_ms'], 27.9)

    def test_absent_optional_settle_is_unknown_not_zero(self):
        row = audit.audit_trace(result(), CALIBRATION + BLOCK.replace(SETTLE, ''))['requests'][0]
        self.assertIsNone(row['cache_settle'])
        self.assertIsNone(row['prefill_minus_stream_and_settle_ms'])

    def test_preload_is_inclusive_and_layer_cache_not_file_demand(self):
        r = result();r['cases'][0]['native_prefill_ms'] = 25401.6
        stream = STREAM.replace('1349 experts streamed:', '1449 experts streamed:').replace(
            '0 from RAM (0.0 GB);', '0 from RAM (0.0 GB), 100 again from the layer cache (layer-major, 2 chunks, 3.0 GB of RAM);')
        block = BLOCK.replace(STREAM, stream).replace(PRELOAD, PRELOAD.replace('0 hot', '63 hot').replace('(0 ms)', '(1000 ms)')).replace('(6001 ms)', '(7001 ms)')
        row = audit.audit_trace(r, CALIBRATION + block)['requests'][0]
        self.assertAlmostEqual(row['prefill_minus_stream_and_settle_ms'], 20.6)
        self.assertEqual(row['RAM_preload']['ms'], 1000)
        self.assertEqual(row['stream']['layer_cache_uses'], 100)
        self.assertNotIn('estimated_expert_to_file_payload_ratio', row)

    def test_forced_window_and_window_only_profile(self):
        row = audit.audit_trace(result(), CALIBRATION + 'ds4 first window: tokens=14 limit=16\n' + END)['requests'][0]
        self.assertEqual(row['decision_kind'], 'first_window')
        self.assertIsNone(row['stream'])
        self.assertIsNone(row['prediction'])
        r = result();r['method']['prefill_chunk'] = None
        self.assertEqual(audit.audit_trace(r, END)['requests'][0]['decision_kind'], 'window_profile')

    def test_incomplete_ambiguous_or_mismatched_trace_is_rejected(self):
        bad = [BLOCK.replace(END, ''), BLOCK + END, BLOCK.replace(DECISION, ''),
               BLOCK.replace(DECISION, DECISION * 2), BLOCK.replace('CUDA0: 14', 'CUDA0: 16'),
               BLOCK.replace('10.0 GB,', '13.0 GB,'), BLOCK.replace(STREAM, STREAM + STREAM.replace('CUDA0', 'CUDA1')),
               BLOCK.replace('1349 experts streamed:', '1350 experts streamed:'), BLOCK + DECISION,
               BLOCK.replace('compute waited 15612', 'compute waited 99999')]
        for trace in bad:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                audit.audit_trace(result(), CALIBRATION + trace)

    def test_missing_calibration_and_nonfinite_measurement_rejected(self):
        with self.assertRaises(ValueError):audit.audit_trace(result(), BLOCK)
        for value in (float('nan'), float('inf'), -1):
            r = result();r['cases'][0]['native_prefill_ms'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):audit.audit_trace(r, CALIBRATION + BLOCK)

    def test_experimental_retention_heat_is_explicit_and_inputs_stay_immutable(self):
        r = result(); r['method']['flow_store_min_heat'] = 1
        saved = copy.deepcopy(r)
        heat = 'ds4 flow store: experimental min_heat=1\n'
        row = audit.audit_trace(r, CALIBRATION + heat + BLOCK)['requests'][0]
        self.assertEqual(row['flow_store_min_heat'], 1)
        self.assertEqual(row['handoff']['retained'], 226)
        self.assertEqual(r, saved)
        self.assertNotIn('flow_store_min_heat', audit.audit_trace(result(), CALIBRATION + BLOCK)['requests'][0])

    def test_retention_heat_missing_duplicate_conflicting_or_inactive_is_rejected(self):
        heat = 'ds4 flow store: experimental min_heat=1\n'
        r = result(); r['method']['flow_store_min_heat'] = 1
        for block in (BLOCK, heat * 2 + BLOCK, heat + BLOCK.replace(HANDOFF, ''),
                      heat + 'ds4 first window: tokens=14 limit=16\n' + END):
            with self.subTest(block=block), self.assertRaises(ValueError):
                audit.audit_trace(r, CALIBRATION + block)
        for setting in (None, 4):
            wrong = result(); wrong['method']['flow_store_min_heat'] = setting
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                audit.audit_trace(wrong, CALIBRATION + heat + BLOCK)
        for value in ('0', '2', '4', 'nan', '1 extra'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.audit_trace(r, CALIBRATION + heat.replace('=1', '=' + value) + BLOCK)

    def test_cli_pins_inputs_and_refuses_duplicate_or_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);p = root/'result.json';err = root/'engine.stderr';out = root/'audit.json'
            p.write_text(json.dumps(result()));err.write_text(CALIBRATION + BLOCK)
            originals = {x: x.read_bytes() for x in (p, err)}
            argv = ['audit_prefill_costs.py', str(p), '--output', str(out)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):audit.main()
            report = json.loads(out.read_text());self.assertEqual(report['request_count'], 1)
            for path, raw in originals.items():
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(report['runs'][0]['input_sha256'][str(path)], hashlib.sha256(raw).hexdigest())
            saved = out.read_bytes()
            with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):audit.main()
            self.assertEqual(out.read_bytes(), saved)
            with patch.object(sys, 'argv', argv[:2] + [str(p)] + argv[2:]), self.assertRaises(ValueError):audit.main()


if __name__ == '__main__':
    unittest.main()

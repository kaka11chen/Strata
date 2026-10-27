"""Pinned tokenizer and process-result contract tests; no GPU or Torch needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'deepseek41'))
import generate


class GenerationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('build-deepseek-cuda/reference-cache')
        if not all((cls.root / name).is_file() for name in ('tokenizer.json', 'config.json')):
            raise unittest.SkipTest('Pinned 1017 tokenizer/config fixtures are not installed')
        cls.tokenizer = generate.load_tokenizer(cls.root)

    def test_pinned_encoding_and_whole_sequence_decoding(self):
        self.assertNotIn('torch', sys.modules)
        for prompt, body in [(' The', [455]), ('Hello', [19923]), ('你好', [30594]), ('', [])]:
            self.assertEqual(generate.encode_prompt(self.tokenizer, prompt, 2), [0, *body])
        text = '你好，🦙 café\n'
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=True), text)
        self.assertEqual(self.tokenizer.decode([1], skip_special_tokens=True), '')

    def test_capacity_rejected_before_native_process(self):
        for budget in (-1, 5, 6, True):
            with self.assertRaises(ValueError):
                generate.encode_prompt(self.tokenizer, ' The', budget)
        self.assertEqual(generate.encode_prompt(self.tokenizer, ' The', 4), [0, 455])
        with self.assertRaises(ValueError):
            generate.encode_prompt(self.tokenizer, ' The' * 8, 0)

    def test_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'tokenizer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pinned'):
                generate.load_tokenizer(root)
        # A correct tokenizer accompanied by a different config is also refused.
        original = Path.read_bytes
        def altered(path):
            return b'{}' if path.name == 'config.json' else original(path)
        with patch.object(Path, 'read_bytes', altered), self.assertRaisesRegex(ValueError, 'pinned'):
            generate.load_tokenizer(self.root)

    def result(self, generated, eos=False, budget=2, **changes):
        result = {'prompt_ids': [0, 455], 'generated_ids': generated, 'stopped_on_eos': eos,
                  'evaluated_tokens': 1 + len(generated) if generated else 0, 'context_capacity': 5}
        result.update(changes)
        return generate.decode_result(self.tokenizer, 'GENERATION_RESULT ' + json.dumps(result), ' The', [0, 455], budget)

    def test_length_eos_and_zero_budget(self):
        self.assertEqual(self.result([6520, 455])['text'], ' The distance The')
        eos = self.result([1], True)
        self.assertEqual((eos['finish_reason'], eos['completion']), ('eos', ''))
        self.assertEqual(self.result([], budget=0)['evaluated_tokens'], 0)

    def test_memory_plan_preserves_refusal_and_rejects_missing_results(self):
        report = {'profile': '1017-full40-context5', 'admitted': False, 'reason': 'GPU budget'}
        self.assertEqual(generate.decode_memory_plan('MEMORY_PLAN ' + json.dumps(report)), report)
        for stdout in ('', 'MEMORY_PLAN {}', 'MEMORY_PLAN {}\nMEMORY_PLAN {}'):
            with self.assertRaises(ValueError): generate.decode_memory_plan(stdout)

    def test_malformed_or_partial_result_is_not_completion(self):
        for ids, eos in [([6520], False), ([1, 455], False), ([6520, 1], False),
                         ([6520, 455], True), ([129280, 455], False), ([True, 455], False)]:
            with self.assertRaises(ValueError): self.result(ids, eos)
        for changes in ({'prompt_ids': [455]}, {'evaluated_tokens': 4}, {'context_capacity': 6}):
            with self.assertRaises(ValueError): self.result([6520, 455], **changes)
        with self.assertRaises(ValueError):
            generate.decode_result(self.tokenizer, '', ' The', [0, 455], 2)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])

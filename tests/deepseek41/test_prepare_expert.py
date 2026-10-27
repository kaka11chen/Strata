"""The fixture repack must preserve every FP4 code and E8M0 byte, without re-quantization."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

module_path = Path(__file__).resolve().parents[2] / "tools/deepseek41/prepare_expert_fixture.py"
spec = importlib.util.spec_from_file_location("prepare_expert_fixture", module_path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class RepackTests(unittest.TestCase):
    def test_known_nibbles_and_scale(self):
        # Sequence 0..15 twice; HF stores adjacent values in a byte, GGUF stores i and i+16.
        weight = np.array([[i | ((i + 1) << 4) for i in range(0, 16, 2)] * 2], dtype=np.uint8)
        scale = np.array([[127]], dtype=np.uint8)
        packed = prepare.repack_fp4(weight, scale)
        np.testing.assert_array_equal(packed[0, 0], [127] + [i | (i << 4) for i in range(16)])

    def test_all_finite_scales_and_multiple_rows(self):
        rng = np.random.default_rng(0)
        weight = rng.integers(0, 256, (255, 64), dtype=np.uint8)
        scale = np.repeat(np.arange(255, dtype=np.uint8)[:, None], 4, axis=1)
        packed = prepare.repack_fp4(weight, scale)
        low, high = packed[..., 1:] & 15, packed[..., 1:] >> 4
        codes = np.concatenate((low, high), axis=-1)
        restored = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(weight.shape)
        np.testing.assert_array_equal(restored, weight)
        np.testing.assert_array_equal(packed[..., 0], scale)
        self.assertEqual(packed.nbytes, weight.nbytes + scale.nbytes)

    def test_reject_nonfinite_scale(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prepare.repack_fp4(np.zeros((1,16), np.uint8), np.array([[255]], np.uint8))

    def test_reject_bad_shapes(self):
        for w, s in [(np.zeros((1,17),np.uint8),np.zeros((1,1),np.uint8)),
                     (np.zeros((2,32),np.uint8),np.zeros((1,2),np.uint8)),
                     (np.zeros((1,0),np.uint8),np.zeros((1,0),np.uint8))]:
            with self.subTest(shape=w.shape), self.assertRaises(ValueError):
                prepare.repack_fp4(w,s)

    def test_reject_wrong_dtype(self):
        with self.assertRaises(ValueError):
            prepare.repack_fp4(np.zeros((1,16),np.float32), np.zeros((1,1),np.uint8))


if __name__ == "__main__":
    unittest.main()

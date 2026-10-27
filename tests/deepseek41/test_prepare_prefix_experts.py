"""The original-byte fixture guard must agree with exact native scale conversion."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/deepseek41"))
from prepare_prefix_experts import validate_nvfp4_adapter


class AdapterGuardTests(unittest.TestCase):
    def test_all_scale_bytes_with_all_codes(self):
        weight = np.array([[i | ((15-i) << 4) for i in range(16)]], dtype=np.uint8)
        for scale in range(256):
            s = np.array([[scale]], dtype=np.uint8)
            with self.subTest(scale=scale):
                if 117 <= scale <= 135:
                    validate_nvfp4_adapter(weight, s)
                else:
                    with self.assertRaises(ValueError):
                        validate_nvfp4_adapter(weight, s)

    def test_all_signed_codes_at_both_nibbles(self):
        scale = np.array([[117]], dtype=np.uint8)
        for code in range(16):
            for shift in (0, 4):
                weight = np.full((1, 16), 0x88, dtype=np.uint8)
                weight[0, 7] = (int(weight[0, 7]) & (240 if shift == 0 else 15)) | (code << shift)
                before = weight.copy()
                with self.subTest(code=code, shift=shift):
                    validate_nvfp4_adapter(weight, scale)
                    np.testing.assert_array_equal(weight, before)
                    self.assertEqual(int(scale[0, 0]), 117)

    def test_only_tiny_blocks_need_halvable_codes(self):
        weight = np.full((2, 32), 0x31, dtype=np.uint8)
        scale = np.array([[118, 135], [127, 117]], dtype=np.uint8)
        weight[1, 16:] = 0xCA
        validate_nvfp4_adapter(weight, scale)
        weight[1, 31] = 0xB2
        with self.assertRaisesRegex(ValueError, "global"):
            validate_nvfp4_adapter(weight, scale)


    def test_inexact_tiny_codes_need_room_for_global_scaling(self):
        weight = np.full((1, 32), 0x31, dtype=np.uint8)
        scale = np.array([[117, 134]], dtype=np.uint8)
        validate_nvfp4_adapter(weight, scale)
        scale[0, 1] = 135
        with self.assertRaisesRegex(ValueError, "global"):
            validate_nvfp4_adapter(weight, scale)
        scale[0, 1] = 116
        with self.assertRaises(ValueError):
            validate_nvfp4_adapter(weight, scale)


if __name__ == "__main__":
    unittest.main()

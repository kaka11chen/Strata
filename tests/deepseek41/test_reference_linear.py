"""Independent arithmetic checks for the optional FP64 fixture oracle; no model/GPU needed."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/deepseek41"))
from prepare_shared_fixture import FP64Linear, ReferenceLinear, quantize
from prepare_layer2_attention_fixture import build_reference


class ReferenceLinearTests(unittest.TestCase):
    @staticmethod
    def linear(weights, mode="fp64"):
        linear = ReferenceLinear(linear_oracle=mode)
        linear.decoded = weights
        return linear

    def test_cancellation_preserves_small_dot(self):
        weights = np.zeros((2, 64), np.float32)
        weights[:, 0] = [1, -1]
        weights[:, 32] = [-1, 1]
        weights[:, 63] = [2**-25, -2**-25]
        x = torch.ones(1, 64, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(self.linear(weights)(x), torch.tensor([[2**-25, -2**-25]], dtype=torch.bfloat16)))
        # Retain the legacy block-32 behavior, including its cancellation loss.
        default = ReferenceLinear(); default.decoded = weights
        self.assertTrue(torch.equal(default(x), torch.zeros(1, 2, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(default(x), self.linear(weights, "block32")(x)))

    def test_bf16_midpoints_and_both_neighbors(self):
        weights = np.zeros((6, 96), np.float32)
        weights[:, 0] = 1
        weights[3:, 1] = 2**-7
        weights[:, 32] = 2**-8
        weights[:, 64] = [0, 2**-23, -2**-23, 0, 2**-23, -2**-23]
        expected = torch.tensor([[1, 1 + 2**-7, 1, 1 + 2**-6, 1 + 2**-6, 1 + 2**-7]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(self.linear(weights)(torch.ones(1, 96)), expected))

    def test_explicit_fp32_boundary_before_bf16(self):
        # A value less than half an FP32 ULP above the BF16 midpoint rounds
        # to that midpoint first, then to even BF16, as declared by this oracle.
        weights = np.zeros((1, 96), np.float32)
        weights[0, [0, 32, 64]] = [1, 2**-8, 2**-25]
        self.assertEqual(self.linear(weights)(torch.ones(1, 96)).item(), 1.)

    def test_matches_exact_rational_quantized_dots(self):
        generator = torch.Generator().manual_seed(1764)
        x = torch.randn(2, 2, 96, generator=generator).bfloat16()
        weights = torch.randn(3, 96, generator=generator).bfloat16().float().numpy()
        activation = quantize(x).reshape(-1, 96)
        self.assertFalse(np.array_equal(activation, x.float().numpy().reshape(-1, 96)))
        exact = np.empty((4, 3), np.float32)
        for token in range(4):
            for row in range(3):
                value = sum((Fraction(float(a)) * Fraction(float(w)) for a, w in zip(activation[token], weights[row])), Fraction())
                exact[token, row] = float(value)
        expected = torch.from_numpy(exact.reshape(2, 2, 3)).bfloat16()
        self.assertTrue(torch.equal(self.linear(weights)(x), expected))

    def test_row_tile_tail_batch_shape_and_input_preservation(self):
        weights = np.zeros((257, 64), np.float32)
        weights[:, 0] = 1; weights[:, 32] = -1
        residual = (np.arange(257) % 15 + 1).astype(np.float32) * np.float32(2**-25)
        weights[:, 63] = residual
        x = torch.tensor([1., 2., -1.]).reshape(3, 1, 1).expand(3, 1, 64).bfloat16()
        saved_x, saved_weights = x.clone(), weights.copy()
        result = self.linear(weights)(x)
        expected = torch.from_numpy(residual).reshape(1, 1, 257) * torch.tensor([1., 2., -1.]).reshape(3, 1, 1)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, expected.bfloat16()))
        self.assertTrue(torch.equal(x, saved_x))
        np.testing.assert_array_equal(weights, saved_weights)

    def test_dense_projection_keeps_input_and_output_precision(self):
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype), torch.inference_mode():
                linear = FP64Linear(96, 3, bias=False, dtype=dtype)
                linear.weight.zero_()
                linear.weight[0, 0] = 1
                linear.weight[1, [1, 33, 65]] = torch.tensor([1, -1, 2**-25], dtype=dtype)
                linear.weight[2, [2, 34]] = torch.tensor([1, 2**-9], dtype=dtype)
                x = torch.ones(2, 1, 96, dtype=dtype); x[..., 0] = 1.03125
                expected = torch.tensor([1.03125, 2**-25, 1 + 2**-9], dtype=dtype).expand(2, 1, 3)
                result = linear(x)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.equal(result, expected))

    def test_dense_projection_rejects_bias(self):
        with self.assertRaisesRegex(ValueError, "bias-free"):
            FP64Linear(32, 1)(torch.ones(32))

    def test_unknown_mode_rejected_before_loading_files(self):
        for mode in [None, "FP64", "float64"]:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Unsupported linear oracle"):
                ReferenceLinear(linear_oracle=mode)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Unsupported linear oracle"):
                build_reference(Path("missing"), Path("missing"), Path("missing"), linear_oracle=mode)


if __name__ == "__main__":
    unittest.main()

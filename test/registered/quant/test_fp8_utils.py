import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    inverse_transform_scale_ue8m0,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from deep_gemm.utils import per_token_cast_to_fp8
except ImportError:
    per_token_cast_to_fp8 = None

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-large")


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32_original = quant_weight_ue8m0(
                weight_bf16, weight_block_size=weight_block_size
            )
            mn = qweight.shape[-2]

            sf_packed_original = transform_scale_ue8m0(sf_fp32_original, mn=mn)
            sf_fp32_recreated = inverse_transform_scale_ue8m0(sf_packed_original, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32_recreated, mn=mn)

            assert torch.all(
                sf_packed_original == sf_packed_recreated
            ), f"{sf_packed_original=} {sf_packed_recreated}"
            assert torch.all(
                sf_fp32_original == sf_fp32_recreated
            ), f"{sf_fp32_original=} {sf_fp32_recreated}"


class TestPackedUe8m0ScaleShape(CustomTestCase):
    def test_group32_shape_matches_mega_contract(self):
        num_tokens = 17
        hidden = 7168
        x_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=(num_tokens, hidden),
            device="cpu",
            group_size=32,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(x_s.dtype, torch.int32)
        self.assertEqual(x_s.shape, (num_tokens, hidden // 128))

    def test_group128_shape_preserves_existing_packed_contract(self):
        num_tokens = 17
        hidden = 7168
        x_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=(num_tokens, hidden),
            device="cpu",
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(x_s.dtype, torch.int32)
        self.assertEqual(x_s.shape, (num_tokens, hidden // 512))

    @unittest.skipUnless(
        torch.cuda.is_available() and per_token_cast_to_fp8 is not None,
        "Requires CUDA and deep_gemm reference utilities.",
    )
    def test_group32_quant_matches_deep_gemm_reference(self):
        torch.manual_seed(0)
        x = torch.randn(7, 256, dtype=torch.bfloat16, device="cuda")
        x_q, x_s = sglang_per_token_group_quant_fp8(
            x,
            group_size=32,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        ref_q, ref_s = per_token_cast_to_fp8(
            x,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        self.assertEqual(x_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(x_s.dtype, torch.int32)
        self.assertTrue(torch.equal(x_q, ref_q))
        self.assertTrue(torch.equal(x_s, ref_s))


if __name__ == "__main__":
    unittest.main()

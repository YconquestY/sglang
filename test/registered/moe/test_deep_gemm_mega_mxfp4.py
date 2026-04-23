import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm_mega import (
    DeepGemmMegaMoeQuantInfo,
    ensure_deep_gemm_mega_runtime,
    fused_experts_none_to_deep_gemm_mega_mxfp4,
    prepare_deep_gemm_mega_weights,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod


class _DummyMegaLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_runner_config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            params_dtype=torch.bfloat16,
            top_k=2,
            gemm1_clamp_limit=None,
        )
        self.num_fused_shared_experts = 0
        self.hidden_size = 128
        self.intermediate_size_per_partition = 128
        self.num_experts = 8
        self.moe_ep_size = 4
        self.num_local_experts = 2
        self.top_k = 2
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 256, 64, dtype=torch.uint8), requires_grad=False
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.ones(2, 256, 4, dtype=torch.uint8), requires_grad=False
        )
        self.w13_weight_bias = torch.nn.Parameter(
            torch.zeros(2, 256, dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 128, 64, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.ones(2, 128, 4, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_bias = torch.nn.Parameter(
            torch.zeros(2, 128, dtype=torch.bfloat16), requires_grad=False
        )


class TestDeepGemmMegaMxfp4(unittest.TestCase):
    def test_prepare_weights_uses_expected_transforms(self):
        layer = _DummyMegaLayer()

        def _fake_transform_sf(sf, *, mn, k, recipe, num_groups, **_kwargs):
            self.assertEqual(recipe, (1, 32))
            self.assertEqual(num_groups, 2)
            self.assertIn((mn, k), {(256, 128), (128, 128)})
            return torch.ones(
                (*sf.shape[:-1], max(1, sf.shape[-1] // 4)), dtype=torch.int32
            )

        fake_deep_gemm = types.SimpleNamespace(
            transform_sf_into_required_layout=mock.Mock(side_effect=_fake_transform_sf)
        )

        with mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.transform_weights_for_mega_moe",
            side_effect=lambda l1, l2: (l1, l2),
        ) as transform_weights, mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ):
            prepare_deep_gemm_mega_weights(layer)

        self.assertEqual(fake_deep_gemm.transform_sf_into_required_layout.call_count, 2)
        self.assertEqual(transform_weights.call_count, 1)
        self.assertEqual(layer._deep_gemm_mega_l1_weights[0].dtype, torch.int8)
        self.assertEqual(layer._deep_gemm_mega_l2_weights[0].dtype, torch.int8)
        self.assertEqual(layer._deep_gemm_mega_l1_weights[1].dtype, torch.int32)
        self.assertEqual(layer._deep_gemm_mega_l2_weights[1].dtype, torch.int32)

    def test_process_weights_after_loading_reuses_cached_transform(self):
        layer = _DummyMegaLayer()
        def _fake_transform_sf(sf, *, mn, k, recipe, num_groups, **_kwargs):
            return torch.ones(
                (*sf.shape[:-1], max(1, sf.shape[-1] // 4)), dtype=torch.int32
            )

        fake_deep_gemm = types.SimpleNamespace(
            transform_sf_into_required_layout=mock.Mock(side_effect=_fake_transform_sf)
        )

        with mock.patch(
            "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
            return_value=MoeRunnerBackend.DEEP_GEMM_MEGA,
        ), mock.patch(
            "sglang.srt.layers.quantization.mxfp4.get_global_server_args",
            return_value=SimpleNamespace(flashinfer_mxfp4_moe_precision="default"),
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.transform_weights_for_mega_moe",
            side_effect=lambda l1, l2: (l1, l2),
        ) as transform_weights, mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ):
            method = Mxfp4MoEMethod(prefix="moe")
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)

        self.assertEqual(transform_weights.call_count, 1)

    def test_ensure_runtime_uses_ep_device_group(self):
        layer = _DummyMegaLayer()
        layer._deep_gemm_mega_l1_weights = (
            torch.zeros(2, 256, 64, dtype=torch.int8),
            torch.zeros(2, 256, 1, dtype=torch.int32),
        )
        layer._deep_gemm_mega_l2_weights = (
            torch.zeros(2, 128, 64, dtype=torch.int8),
            torch.zeros(2, 128, 1, dtype=torch.int32),
        )
        fake_symm_buffer = SimpleNamespace(num_max_tokens_per_rank=2048)

        with mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.prepare_deep_gemm_mega_weights"
        ) as prepare_weights, mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.get_moe_ep_group",
            return_value=SimpleNamespace(device_group="fake_pg"),
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.get_global_server_args",
            return_value=SimpleNamespace(cuda_graph_max_bs=256, chunked_prefill_size=1024),
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.get_symm_buffer_for_mega_moe",
            return_value=fake_symm_buffer,
        ) as get_buffer:
            runtime = ensure_deep_gemm_mega_runtime(layer)

        prepare_weights.assert_called_once_with(layer)
        get_buffer.assert_called_once()
        self.assertEqual(get_buffer.call_args.kwargs["group"], "fake_pg")
        self.assertEqual(runtime.symm_buffer, fake_symm_buffer)
        self.assertEqual(runtime.max_num_tokens_per_rank, 2048)
        self.assertEqual(runtime.transformed_l1_weights, layer._deep_gemm_mega_l1_weights)

    def test_rejects_mismatched_packed_activation_scales(self):
        num_tokens = 3
        hidden = 128
        dispatch_output = SimpleNamespace(
            hidden_states=torch.zeros(num_tokens, hidden, dtype=torch.bfloat16),
            topk_output=SimpleNamespace(
                topk_ids=torch.zeros(num_tokens, 2, dtype=torch.int32),
                topk_weights=torch.zeros(num_tokens, 2, dtype=torch.float32),
            ),
        )
        runtime = SimpleNamespace(
            max_num_tokens_per_rank=8,
            symm_buffer=SimpleNamespace(
                x=torch.zeros(8, hidden, dtype=torch.float32),
                x_sf=torch.zeros(8, 1, dtype=torch.int32),
                topk_idx=torch.zeros(8, 2, dtype=torch.int64),
                topk_weights=torch.zeros(8, 2, dtype=torch.float32),
            ),
            transformed_l1_weights=None,
            transformed_l2_weights=None,
        )
        quant_info = DeepGemmMegaMoeQuantInfo(runtime=runtime)
        runner_config = MoeRunnerConfig(activation="silu", is_gated=True)

        with mock.patch(
            "sglang.srt.layers.moe.topk.TopKOutputChecker.format_is_standard",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.sglang_per_token_group_quant_fp8",
            return_value=(
                torch.zeros(num_tokens, hidden, dtype=torch.float32),
                torch.zeros(num_tokens, 2, dtype=torch.int32),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "expected packed UE8M0 activation scales"
            ):
                fused_experts_none_to_deep_gemm_mega_mxfp4(
                    dispatch_output, quant_info, runner_config
                )

    def test_fused_op_copies_global_topk_and_calls_mega_kernel(self):
        num_tokens = 3
        hidden = 128
        topk_ids = torch.tensor([[7, 3], [5, 1], [6, 2]], dtype=torch.int32)
        topk_weights = torch.tensor(
            [[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]], dtype=torch.float32
        )
        dispatch_output = SimpleNamespace(
            hidden_states=torch.arange(
                num_tokens * hidden, dtype=torch.bfloat16
            ).reshape(num_tokens, hidden),
            topk_output=SimpleNamespace(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            ),
        )
        symm_buffer = SimpleNamespace(
            x=torch.zeros(8, hidden, dtype=torch.float32),
            x_sf=torch.zeros(8, 1, dtype=torch.int32),
            topk_idx=torch.zeros(8, 2, dtype=torch.int64),
            topk_weights=torch.zeros(8, 2, dtype=torch.float32),
        )
        runtime = SimpleNamespace(
            max_num_tokens_per_rank=8,
            symm_buffer=symm_buffer,
            transformed_l1_weights=("l1_w", "l1_sf"),
            transformed_l2_weights=("l2_w", "l2_sf"),
        )
        quant_info = DeepGemmMegaMoeQuantInfo(
            runtime=runtime, activation_clamp=7.0, fast_math=False
        )
        runner_config = MoeRunnerConfig(activation="silu", is_gated=True)
        x_q = torch.full((num_tokens, hidden), 2.0, dtype=torch.float32)
        x_s = torch.tensor([[11], [22], [33]], dtype=torch.int32)
        expected_output = torch.full(
            (num_tokens, hidden), 1.25, dtype=torch.bfloat16
        )

        def _fake_kernel(out, *_args, **_kwargs):
            out.copy_(expected_output)

        with mock.patch(
            "sglang.srt.layers.moe.topk.TopKOutputChecker.format_is_standard",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.sglang_per_token_group_quant_fp8",
            return_value=(x_q, x_s),
        ) as quantize, mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.deep_gemm_wrapper.fp8_fp4_mega_moe",
            side_effect=_fake_kernel,
        ) as mega_call:
            combine_input = fused_experts_none_to_deep_gemm_mega_mxfp4(
                dispatch_output, quant_info, runner_config
            )

        quantize.assert_called_once()
        self.assertTrue(torch.equal(symm_buffer.x[:num_tokens], x_q))
        self.assertTrue(torch.equal(symm_buffer.x_sf[:num_tokens], x_s))
        self.assertTrue(
            torch.equal(symm_buffer.topk_idx[:num_tokens], topk_ids.to(torch.int64))
        )
        self.assertTrue(
            torch.equal(symm_buffer.topk_weights[:num_tokens], topk_weights)
        )
        self.assertTrue(torch.equal(combine_input.hidden_states, expected_output))
        self.assertEqual(mega_call.call_args.args[1], runtime.transformed_l1_weights)
        self.assertEqual(mega_call.call_args.args[2], runtime.transformed_l2_weights)
        self.assertIs(mega_call.call_args.args[3], symm_buffer)
        self.assertEqual(mega_call.call_args.kwargs["recipe"], (1, 1, 32))
        self.assertEqual(mega_call.call_args.kwargs["activation"], "swiglu")
        self.assertEqual(mega_call.call_args.kwargs["activation_clamp"], 7.0)
        self.assertFalse(mega_call.call_args.kwargs["fast_math"])


if __name__ == "__main__":
    unittest.main()

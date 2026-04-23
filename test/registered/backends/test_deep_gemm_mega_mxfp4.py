import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner import deep_gemm_mega as _deep_gemm_mega  # noqa: F401
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES, ServerArgs


class TestDeepGemmMegaMxfp4(unittest.TestCase):
    def _make_server_args(self) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        args.quantization = "mxfp4"
        args.moe_runner_backend = "deep_gemm_mega"
        args.moe_a2a_backend = "none"
        args.tp_size = 4
        args.ep_size = 4
        args.disable_shared_experts_fusion = False
        return args

    def test_backend_choice_is_registered(self):
        self.assertIn("deep_gemm_mega", MOE_RUNNER_BACKEND_CHOICES)

    def test_runner_registers_mega_fused_func(self):
        config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=2,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
        )
        runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM_MEGA, config)
        self.assertIsNone(runner.runner_core)
        self.assertIsNotNone(runner.fused_func)

    def test_server_args_enable_mega_sets_shared_expert_flag(self):
        args = self._make_server_args()

        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ):
            args._handle_moe_kernel_config()

        self.assertTrue(args.disable_shared_experts_fusion)

    def test_server_args_rejects_wrong_quantization(self):
        args = self._make_server_args()
        args.quantization = "modelopt_fp4"
        with self.assertRaisesRegex(AssertionError, "supports only: 'mxfp4'"):
            args._handle_moe_kernel_config()

    def test_server_args_rejects_non_none_a2a_backend(self):
        args = self._make_server_args()
        args.moe_a2a_backend = "deepep"
        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ), self.assertRaisesRegex(AssertionError, "requires moe_a2a_backend='none'"):
            args._handle_moe_kernel_config()

    def test_server_args_rejects_ep_size_not_equal_tp_size(self):
        args = self._make_server_args()
        args.ep_size = 2
        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ), self.assertRaisesRegex(
            AssertionError, "requires ep_size == tp_size and ep_size > 1"
        ):
            args._handle_moe_kernel_config()

    def test_server_args_rejects_ep_size_one(self):
        args = self._make_server_args()
        args.ep_size = 1
        args.tp_size = 1
        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_MEGA_AVAILABLE",
            True,
        ), self.assertRaisesRegex(
            AssertionError, "requires ep_size == tp_size and ep_size > 1"
        ):
            args._handle_moe_kernel_config()

    def test_server_args_rejects_missing_mega_capability(self):
        args = self._make_server_args()
        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_MEGA_AVAILABLE",
            False,
        ), self.assertRaisesRegex(
            AssertionError, "requires Blackwell \\+ DeepGEMM"
        ):
            args._handle_moe_kernel_config()

    def test_nonserialized_checkpoint_is_rejected_for_mega(self):
        quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=False)
        layer = object.__new__(FusedMoE)
        with mock.patch(
            "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
            return_value=MoeRunnerBackend.DEEP_GEMM_MEGA,
        ), self.assertRaisesRegex(
            NotImplementedError, "offline-converted, serialized mxfp4 checkpoint"
        ):
            quant_config.get_quant_method(layer, prefix="moe")


if __name__ == "__main__":
    unittest.main()

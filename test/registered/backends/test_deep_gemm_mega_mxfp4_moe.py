"""Optional backend test for DeepGEMM Mega MoE with serialized MXFP4 weights.

Set ``SGLANG_DEEP_GEMM_MEGA_MOE_MODEL`` to an offline-converted MXFP4 MoE
checkpoint before running this test.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MEGA_MOE_MODEL_PATH = os.environ.get("SGLANG_DEEP_GEMM_MEGA_MOE_MODEL")
SERVER_LAUNCH_TIMEOUT = 1800
GSM8K_ACCURACY_THRESHOLD = 0.93


@unittest.skipUnless(
    MEGA_MOE_MODEL_PATH,
    "Set SGLANG_DEEP_GEMM_MEGA_MOE_MODEL to run the DeepGEMM Mega MoE backend test.",
)
class TestDeepGemmMegaMxfp4MoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MEGA_MOE_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--ep",
            "4",
            "--mem-fraction-static",
            "0.75",
            "--moe-runner-backend",
            "deep_gemm_mega",
            "--moe-a2a-backend",
            "none",
            "--quantization",
            "mxfp4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        env = dict(os.environ)
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()

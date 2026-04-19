# SGLang DeepGEMM Mega MoE Integration Plan

## Summary
- Add a new explicit MoE backend, `deep_gemm_mega`, for `mxfp4` only.
- Integrate it on the fused path: `FusedMoE -> StandardDispatcher -> Mxfp4MoEMethod.apply() -> MoeRunner.run() -> fused_experts_none_to_deep_gemm_mega_mxfp4() -> deep_gemm.fp8_fp4_mega_moe()`.
- Do not extend the current `moe_runner/deep_gemm.py` grouped-GEMM path and do not use `token_dispatcher/deepep.py`; Mega must own dispatch, GEMM1, SwiGLU, GEMM2, and combine inside the EP group.
- Do not support NVFP4 checkpoints in the serving path. If NVFP4 support is needed, convert the checkpoint offline to a Mega-compatible MXFP4 checkpoint before startup.

## Integration Points
- `python/sglang/srt/layers/moe/utils.py`: add `MoeRunnerBackend.DEEP_GEMM_MEGA` and `is_deep_gemm_mega()`.
- `python/sglang/srt/server_args.py`: add the backend choice and validate Mega-specific restrictions.
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`: keep `topk_ids` in global expert space for Mega.
- `python/sglang/srt/layers/moe/moe_runner/runner.py`: treat Mega as a fused-only backend (`runner_core = None`).
- `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`: expose Mega capability detection.
- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`: add thin wrappers for the Mega APIs.
- `python/sglang/srt/layers/quantization/mxfp4.py`: load native MXFP4 weights, build the DeepGEMM-specific layout cache, set up the runner, and add the Mega branch in `apply()`.
- New module `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`: fused op, runtime cache, and lazy symmetric-buffer creation.

## Files to Touch/Create
- Touch:
  - `python/sglang/srt/layers/moe/utils.py`
  - `python/sglang/srt/server_args.py`
  - `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
  - `python/sglang/srt/layers/moe/moe_runner/runner.py`
  - `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`
  - `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
  - `python/sglang/srt/layers/quantization/mxfp4.py`
- Create:
  - `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`
  - `test/registered/backends/test_deep_gemm_mega_mxfp4_moe.py`
  - `test/registered/moe/test_deep_gemm_mega_mxfp4.py`
- Optional manual coverage:
  - `test/manual/layers/moe/test_moe_runners_4gpu.py`

## Implementation Changes
- `MoeRunnerBackend.DEEP_GEMM_MEGA` / `is_deep_gemm_mega()`:
  - Add a new explicit backend. Do not change `auto`.
- `ServerArgs._handle_moe_kernel_config()`:
  - Enforce `quantization == "mxfp4"`.
  - Enforce `moe_a2a_backend == "none"`.
  - Enforce `ep_size == tp_size` for V1.
  - Enforce `ep_size > 1` for V1.
  - Auto-set `disable_shared_experts_fusion = True` and warn once.
- `StandardDispatcher.__init__()`:
  - Extend `skip_local_expert_mapping` to include `deep_gemm_mega`, so EP keeps global expert ids.
- `MoeRunner.__init__()`:
  - Add `is_deep_gemm_mega()` branch that sets `runner_core = None` and relies on a fused func.
- `deep_gemm_wrapper/configurer.py`:
  - Add `DEEPGEMM_MEGA_AVAILABLE = DEEPGEMM_BLACKWELL and hasattr(deep_gemm, "fp8_fp4_mega_moe")`.
- `deep_gemm_wrapper/entrypoint.py`:
  - Add wrappers `get_symm_buffer_for_mega_moe(...)`, `transform_weights_for_mega_moe(...)`, and `fp8_fp4_mega_moe(...)`.
  - V1 does not touch `compile_utils.py`; accept first-use JIT latency.
- `Mxfp4MoEMethod.create_moe_runner()`:
  - Import `sglang.srt.layers.moe.moe_runner.deep_gemm_mega` when Mega is selected so `@register_fused_func("none", "deep_gemm_mega")` is registered.
- `Mxfp4MoEMethod.process_weights_after_loading()`:
  - Add an early Mega branch before FlashInfer / Triton backend-specific swizzles.
  - Validate Mega constraints: gated MoE, activation `silu`, hidden/intermediate dims multiple of 128, `num_experts % moe_ep_size == 0`, Blackwell-only, and native MXFP4 checkpoint layout.
  - Build a dedicated DeepGEMM Mega cache from the native MXFP4 checkpoint tensors by layout repacking only.
  - Apply the DeepGEMM-required scale layout transform and `transform_weights_for_mega_moe()`.
  - Cache the transformed tensors on the layer as `_deep_gemm_mega_l1_weights` and `_deep_gemm_mega_l2_weights`.
  - If the checkpoint is NVFP4 or otherwise numerically incompatible, raise a clear error instructing users to preconvert offline before serving.
- New `deep_gemm_mega.py`:
  - `@dataclass DeepGemmMegaMoeRuntime`: store `symm_buffer`, `max_num_tokens`, `device_group`, and cached transformed weights.
  - `@dataclass DeepGemmMegaMoeQuantInfo(MoeQuantInfo)`: carry the runtime object.
  - `ensure_deep_gemm_mega_runtime(layer)`: lazily allocate the symmetric buffer with `get_moe_ep_group().device_group` and `get_symm_buffer_for_mega_moe(...)`; size it with `max(cuda_graph_max_bs or 512, chunked_prefill_size or 8192)`; create it under `torch.inference_mode(False)`; no dynamic grow in V1.
  - `fused_experts_none_to_deep_gemm_mega_mxfp4(...)`: quantize activations with `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)`, convert `topk_ids` to `int64` and `topk_weights` to `float32`, copy `x/x_sf/topk` into the symmetric buffer every forward, call `fp8_fp4_mega_moe(...)` with recipe `(1, 1, 32)` and activation `"swiglu"`, and return `StandardCombineInput(hidden_states=output)`.
- `Mxfp4MoEMethod.apply()`:
  - Add a Mega branch before the existing FlashInfer / Triton MXFP4 branches.
  - Call `ensure_deep_gemm_mega_runtime(layer)`, wrap it in `DeepGemmMegaMoeQuantInfo`, and dispatch through `self.runner.run(...)`.

## Test Plan
- Backend validation:
  - `deep_gemm_mega` is accepted by `--moe-runner-backend`.
  - Invalid combinations fail fast: wrong quantization, `moe_a2a_backend != "none"`, `ep_size != tp_size`, `ep_size <= 1`, shared-expert fusion enabled, or missing Mega capability.
  - NVFP4 / `modelopt_fp4` checkpoints are rejected with a clear offline-conversion error.
- Weight-prep coverage in `test/registered/moe/test_deep_gemm_mega_mxfp4.py`:
  - `Mxfp4MoEMethod.process_weights_after_loading()` creates `_deep_gemm_mega_l1_weights/_l2_weights` once per layer.
  - The Mega cache is built by layout repacking only; no dequantize / requantize path is exercised.
  - Cached weights have the expected shapes/dtypes, `w13` is interleaved, and scales match the DeepGEMM layout requirements.
- Fused backend functional coverage in `test/registered/moe/test_deep_gemm_mega_mxfp4.py`:
  - `fused_experts_none_to_deep_gemm_mega_mxfp4()` matches a direct `deep_gemm.fp8_fp4_mega_moe()` call on the same synthetic inputs/topk metadata.
  - Include an EP>1 case to verify global expert ids are preserved and consumed correctly.
- End-to-end 4-GPU coverage in `test/registered/backends/test_deep_gemm_mega_mxfp4_moe.py`:
  - Use a native or preconverted MXFP4 MoE checkpoint.
  - Config: `--tp 4 --ep 4 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization mxfp4`
  - Reuse the same accuracy / regression threshold style as the existing MoE backend tests.
- Optional manual smoke/perf:
  - Add a Mega config to `test/manual/layers/moe/test_moe_runners_4gpu.py`.

## Assumptions and Defaults
- V1 is fused-backend only; `moe_runner/deep_gemm.py` and `token_dispatcher/deepep.py` stay unchanged.
- V1 scope is `mxfp4` on Blackwell only, with BF16 output and standard-path dispatch only.
- V1 assumes a single NVLink domain; no cross-node EP support.
- V1 always uses runtime FP8 activation quantization, DeepGEMM recipe `(1, 1, 32)`, and maps SGLang `silu` to DeepGEMM `swiglu`.
- V1 does not add Mega prewarm support to `compile_utils.py`.
- V1 does not perform runtime or load-time NVFP4 -> MXFP4 transcoding inside `sglang`; any such conversion must happen offline before startup.

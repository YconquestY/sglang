# SGLang DeepGEMM Mega MoE Integration Plan

## Summary
- Add a new explicit MoE backend, `deep_gemm_mega`, for `modelopt_fp4` only.
- Integrate it on the fused path: `FusedMoE -> StandardDispatcher -> ModelOptNvFp4FusedMoEMethod.apply() -> MoeRunner.run() -> fused_experts_none_to_deep_gemm_mega_fp4() -> deep_gemm.fp8_fp4_mega_moe()`.
- Do not extend the current `moe_runner/deep_gemm.py` grouped-GEMM path and do not use `token_dispatcher/deepep.py`; Mega must own dispatch, GEMM1, SwiGLU, GEMM2, and combine inside the EP group.

## Integration Points
- `python/sglang/srt/layers/moe/utils.py`: add `MoeRunnerBackend.DEEP_GEMM_MEGA` and `is_deep_gemm_mega()`.
- `python/sglang/srt/server_args.py`: add the backend choice and validate Mega-specific restrictions.
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`: keep `topk_ids` in global expert space for Mega.
- `python/sglang/srt/layers/moe/moe_runner/runner.py`: treat Mega as a fused-only backend (`runner_core = None`).
- `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`: expose Mega capability detection.
- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`: add thin wrappers for the Mega APIs.
- `python/sglang/srt/layers/quantization/modelopt_quant.py`: load-time weight conversion, runtime cache setup, and Mega branch in `apply()`.
- New module `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`: fused op, runtime cache, and lazy symmetric-buffer creation.

## Files to Touch/Create
- Touch:
  - `python/sglang/srt/layers/moe/utils.py`
  - `python/sglang/srt/server_args.py`
  - `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
  - `python/sglang/srt/layers/moe/moe_runner/runner.py`
  - `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`
  - `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
  - `python/sglang/srt/layers/quantization/modelopt_quant.py`
- Create:
  - `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`
  - `test/registered/backends/test_deepseek_v3_fp4_deep_gemm_mega_moe.py`
  - `test/registered/moe/test_deep_gemm_mega_moe.py`
- Optional manual coverage:
  - `test/manual/layers/moe/test_moe_runners_4gpu.py`

## Implementation Changes
- `MoeRunnerBackend.DEEP_GEMM_MEGA` / `is_deep_gemm_mega()`:
  - Add a new explicit backend. Do not change `auto`.
- `ServerArgs._handle_moe_kernel_config()`:
  - Enforce `quantization == "modelopt_fp4"`.
  - Enforce `moe_a2a_backend == "none"`.
  - Enforce `ep_size in {1, tp_size}` for V1.
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
- `ModelOptNvFp4FusedMoEMethod.enable_deep_gemm_mega_moe`:
  - New property returning `get_moe_runner_backend().is_deep_gemm_mega()`.
- `ModelOptNvFp4FusedMoEMethod.create_moe_runner()`:
  - Import `sglang.srt.layers.moe.moe_runner.deep_gemm_mega` when Mega is selected so `@register_fused_func("none", "deep_gemm_mega")` is registered.
- `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading()`:
  - Add an early Mega branch before CUTLASS/CuteDSL swizzling.
  - Validate Mega constraints: gated MoE, activation `silu`, hidden/intermediate dims multiple of 128, `num_experts % moe_ep_size == 0`.
  - For each local expert, dequantize checkpoint NVFP4 weights with `deep_gemm.utils.math.cast_back_from_fp4(..., gran_k=16)`.
  - Requantize them into DeepGEMM FP4 format with `deep_gemm.utils.math.per_token_cast_to_fp4(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`.
  - Convert the new FP4 scales with `deep_gemm.transform_sf_into_required_layout(..., recipe=(1, 32), num_groups=num_local_experts)`.
  - Run `transform_weights_for_mega_moe()` to interleave L1 gate/up and transpose scales for UTCCP.
  - Cache the transformed tensors on the layer as `_deep_gemm_mega_l1_weights` and `_deep_gemm_mega_l2_weights`.
- New `deep_gemm_mega.py`:
  - `@dataclass DeepGemmMegaMoeRuntime`: store `symm_buffer`, `max_num_tokens`, `device_group`, and cached transformed weights.
  - `@dataclass DeepGemmMegaMoeQuantInfo(MoeQuantInfo)`: carry the runtime object.
  - `ensure_deep_gemm_mega_runtime(layer)`: lazily allocate the symmetric buffer with `get_moe_ep_group().device_group` and `get_symm_buffer_for_mega_moe(...)`; size it with `max(cuda_graph_max_bs or 512, chunked_prefill_size or 8192)`; create it under `torch.inference_mode(False)`; no dynamic grow in V1.
  - `fused_experts_none_to_deep_gemm_mega_fp4(...)`: quantize activations with `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)`, convert `topk_ids` to `int64` and `topk_weights` to `float32`, copy `x/x_sf/topk` into the symmetric buffer every forward, call `fp8_fp4_mega_moe(...)` with recipe `(1, 1, 32)` and activation `"swiglu"`, and return `StandardCombineInput(hidden_states=output)`.
- `ModelOptNvFp4FusedMoEMethod.apply()`:
  - Add a Mega branch before the existing Cutlass/CuteDSL branches.
  - Call `ensure_deep_gemm_mega_runtime(layer)`, wrap it in `DeepGemmMegaMoeQuantInfo`, and dispatch through `self.runner.run(...)`.

## Test Plan
- Backend validation:
  - `deep_gemm_mega` is accepted by `--moe-runner-backend`.
  - Invalid combinations fail fast: wrong quantization, `moe_a2a_backend != "none"`, unsupported EP layout, shared-expert fusion enabled, or missing Mega capability.
- Weight-prep coverage in `test/registered/moe/test_deep_gemm_mega_moe.py`:
  - The load-time conversion creates `_deep_gemm_mega_l1_weights/_l2_weights` once per layer.
  - Cached weights have the expected shapes/dtypes, `w13` is interleaved, and scales are `torch.int` in the DeepGEMM layout.
- Fused backend functional coverage in `test/registered/moe/test_deep_gemm_mega_moe.py`:
  - `fused_experts_none_to_deep_gemm_mega_fp4()` matches a direct `deep_gemm.fp8_fp4_mega_moe()` call on the same synthetic inputs/topk metadata.
  - Include an EP>1 case to verify global expert ids are preserved and consumed correctly.
- End-to-end 4-GPU coverage in `test/registered/backends/test_deepseek_v3_fp4_deep_gemm_mega_moe.py`:
  - Config A: `--tp 4 --ep 1 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization modelopt_fp4`
  - Config B: `--tp 4 --ep 4 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization modelopt_fp4`
  - Reuse the existing GSM8K threshold used by the CuteDSL FP4 backend test.
- Optional manual smoke/perf:
  - Add a Mega config to `test/manual/layers/moe/test_moe_runners_4gpu.py`.

## Assumptions and Defaults
- V1 is fused-backend only; `moe_runner/deep_gemm.py` and `token_dispatcher/deepep.py` stay unchanged.
- V1 scope is `modelopt_fp4` on Blackwell only, with BF16 output and standard-path dispatch only.
- V1 assumes a single NVLink domain; no cross-node EP support.
- V1 always uses runtime FP8 activation quantization, DeepGEMM recipe `(1, 1, 32)`, and maps SGLang `silu` to DeepGEMM `swiglu`.
- V1 does not add Mega prewarm support to `compile_utils.py`.

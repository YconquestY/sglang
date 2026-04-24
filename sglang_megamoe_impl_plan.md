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
- `python/sglang/srt/layers/quantization/fp8_kernel.py`: fix the shared UE8M0 activation-scale packing path so `group_size=32` produces the packed layout Mega actually expects.
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
  - `python/sglang/srt/layers/quantization/fp8_kernel.py`
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
- `fp8_kernel.py`:
  - Keep the Mega call site unchanged: `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)`.
  - Fix the shared `scale_ue8m0=True` output-shape/layout logic to derive packed scale width from `group_size`, not from a hardcoded `128`.
  - Preserve the existing column-major / TMA-aligned contract; for Mega the visible scale tensor shape must be `[num_tokens, hidden / 128]` because 4 group-32 scales are packed into each `torch.int32`.
  - Remove the implicit `group_size == 128` assumption from the fallback path so the helper contract is consistent even when the Triton fallback is exercised.
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
  - `fused_experts_none_to_deep_gemm_mega_mxfp4(...)`: quantize activations with `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)`, convert `topk_ids` to `int64` and `topk_weights` to `float32`, assert the packed activation scale tensor matches the symmetric-buffer view shape/dtype, copy `x/x_sf/topk` into the symmetric buffer every forward, call `fp8_fp4_mega_moe(...)` with recipe `(1, 1, 32)` and activation `"swiglu"`, and return `StandardCombineInput(hidden_states=output)`.
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
- Activation-quantization coverage:
  - `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)` returns `torch.float8_e4m3fn` activations plus `torch.int32` packed scales with visible shape `[num_tokens, hidden / 128]`.
  - The packed activation scales match DeepGEMM's `per_token_cast_to_fp8(..., gran_k=32, use_packed_ue8m0=True)` convention after layout normalization.
- Fused backend functional coverage in `test/registered/moe/test_deep_gemm_mega_mxfp4.py`:
  - `fused_experts_none_to_deep_gemm_mega_mxfp4()` matches a direct `deep_gemm.fp8_fp4_mega_moe()` call on the same synthetic inputs/topk metadata.
  - Include an EP>1 case to verify global expert ids are preserved and consumed correctly.
- End-to-end 4-GPU coverage in `test/registered/backends/test_deep_gemm_mega_mxfp4_moe.py`:
  - Use a native or preconverted MXFP4 MoE checkpoint.
  - Config: `--tp 4 --ep 4 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization mxfp4`
  - Reuse the same accuracy / regression threshold style as the existing MoE backend tests.
- Optional manual smoke/perf:
  - Add a Mega config to `test/manual/layers/moe/test_moe_runners_4gpu.py`.

## Testing Instructions
1. Reserve a 4-GPU Blackwell slice on the same NVLink domain of the GB200 NVL72 tray.
   - Do not start with all 72 GPUs; the current SGLang integration and checked-in backend tests target `tp=4, ep=4`.
2. Export the required environment variables.
   - `export CUDA_VISIBLE_DEVICES=0,1,2,3`
   - `export SGLANG_ENABLE_JIT_DEEPGEMM=1`
   - `export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`
   - `export SGLANG_DEEP_GEMM_MEGA_MOE_MODEL=/abs/path/to/offline_converted_serialized_mxfp4_moe_ckpt`
3. Verify the model format before running SGLang.
   - The checkpoint must already be an offline-converted, serialized MXFP4 MoE checkpoint.
   - Do not use NVFP4 / `modelopt_fp4`.
   - Do not rely on dynamic or in-serving MXFP4 quantization.
4. Sanity-check DeepGEMM Mega MoE by itself.
   - `cd /home/yy010/proj/deepgemm`
   - `python3 tests/test_mega_moe.py --num-processes 4 --num-max-tokens-per-rank 512 --hidden 7168 --intermediate-hidden 3072 --num-experts 384 --num-topk 6 --num-correctness-tests 1`
   - This validates symmetric memory, FP8 activation quantization, FP4 weight layout, and the fused Mega kernel outside SGLang.
5. Run the SGLang backend validation tests.
   - `cd /home/yy010/proj/sglang`
   - `python3 -m pytest test/registered/backends/test_deep_gemm_mega_mxfp4.py -q`
   - This checks runner registration, server-arg validation, and checkpoint-format rejection.
6. Run the activation-quantization regression test for the group-32 fix.
   - `python3 -m pytest test/registered/quant/test_fp8_utils.py -q -k PackedUe8m0ScaleShape`
   - This confirms that `sglang_per_token_group_quant_fp8(..., group_size=32, scale_ue8m0=True)` returns packed activation scales with visible shape `[num_tokens, hidden / 128]`.
7. Run the fused Mega MoE unit tests.
   - `python3 -m pytest test/registered/moe/test_deep_gemm_mega_mxfp4.py -q`
   - This covers weight preparation, runtime creation, packed-scale validation, top-k copying, and the fused DeepGEMM call path.
8. Run the manual 4-GPU MoE runner smoke test.
   - `python3 -m pytest test/manual/layers/moe/test_moe_runners_4gpu.py -q -s -k deep_gemm_mega_mxfp4`
   - This launches a real SGLang server with the Mega backend and runs a small MMLU smoke eval.
9. Run the registered end-to-end backend test.
   - `python3 -m pytest test/registered/backends/test_deep_gemm_mega_mxfp4_moe.py -q -s`
   - This launches SGLang with `--tp 4 --ep 4 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization mxfp4 --mem-fraction-static 0.75 --model-loader-extra-config '{"enable_multithread_load": true}'` and then runs GSM8K.
10. If a manual server bring-up is needed outside pytest, use the exact tested topology first.
   - `python3 -m sglang.launch_server --model-path "$SGLANG_DEEP_GEMM_MEGA_MOE_MODEL" --tp 4 --ep 4 --moe-runner-backend deep_gemm_mega --moe-a2a-backend none --quantization mxfp4 --mem-fraction-static 0.75 --model-loader-extra-config '{"enable_multithread_load": true}'`
11. Interpret common failures narrowly.
   - `supports only: 'mxfp4'` or `offline-converted, serialized mxfp4 checkpoint`: wrong checkpoint format.
   - `requires ep_size == tp_size and ep_size > 1`: wrong TP/EP topology.
   - `requires Blackwell + DeepGEMM`: missing hardware capability or missing DeepGEMM Mega support.
   - `expected packed UE8M0 activation scales`: activation quantization / scale packing path regressed.
   - `symmetric buffer is too small`: increase `chunked_prefill_size` or `cuda_graph_max_bs`.

## Assumptions and Defaults
- V1 is fused-backend only; `moe_runner/deep_gemm.py` and `token_dispatcher/deepep.py` stay unchanged.
- V1 scope is `mxfp4` on Blackwell only, with BF16 output and standard-path dispatch only.
- V1 assumes a single NVLink domain; no cross-node EP support.
- V1 always uses runtime FP8 activation quantization, DeepGEMM recipe `(1, 1, 32)`, and maps SGLang `silu` to DeepGEMM `swiglu`.
- V1 keeps the existing Mega call site shape in `deep_gemm_mega.py`; the activation-quantization fix is in the shared FP8 helper, not in a Mega-only custom quantizer.
- V1 does not add Mega prewarm support to `compile_utils.py`.
- V1 does not perform runtime or load-time NVFP4 -> MXFP4 transcoding inside `sglang`; any such conversion must happen offline before startup.

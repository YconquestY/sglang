# SGLang + DeepGEMM Mega MoE Research

This note is based on the local source trees on 2026-04-19:

- `deepgemm` HEAD = `7f2a703` (`[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes`)
- `sglang` local checkout in `/home/yy010/proj/sglang`

## Executive Summary

`sglang` does **not** use DeepGEMM today as an end-to-end MoE kernel. It currently uses DeepGEMM only as a **grouped GEMM backend** inside the MoE core, with explicit token reorder / scatter / combine logic around it.

DeepGEMM Mega MoE is a different abstraction boundary:

- it fuses **dispatch + GEMM1 + SwiGLU + GEMM2 + combine**
- it owns the **NVLink communication**
- it expects **raw token inputs + global top-k routing**
- it requires **FP8 activations + FP4 weights**, **symmetric memory**, and **Blackwell**

The most important numerical point is that DeepGEMM Mega MoE's FP4 contract is **not** SGLang's current `modelopt_fp4` contract:

- DeepGEMM Mega MoE uses FP4 E2M1 weights with **packed UE8M0 scales** and `gran_k = 32`
- `sglang`'s `modelopt_fp4` path is **NVFP4** with **FP8-E4M3 scales** and block size `16`

So a native `deep_gemm_mega + modelopt_fp4` design is wrong.

The correct native landing zone in `sglang` is:

- add a new runner backend, e.g. `deep_gemm_mega`
- use `MoeA2ABackend.NONE` + `StandardDispatcher`
- preserve **global** expert ids in the standard dispatch path
- register a fused op `@register_fused_func("none", "deep_gemm_mega")`
- integrate it through `python/sglang/srt/layers/quantization/mxfp4.py`
- keep the existing DeepEP / grouped-GEMM / FlashInfer paths as fallback

For NVFP4 checkpoints, this serving path should do **no** runtime or load-time transcoding. If NVFP4 support is needed, the checkpoint must be converted to a Mega-compatible MXFP4 form **offline before serving**.

## 1. How `sglang` Uses DeepGEMM Today

### 1.1 DeepGEMM is wrapped only as GEMM primitives

Current wrappers live in:

- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`

The wrapper exports only grouped/single GEMM entrypoints such as:

- `grouped_gemm_nt_f8f8bf16_masked`
- `grouped_gemm_nt_f8f8bf16_contig`
- `gemm_nt_f8f8bf16`

There is no Mega MoE wrapper there today.

### 1.2 The active DeepGEMM MoE path is FP8 grouped GEMM, not fused MoE

The current MoE runner is:

- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`

`DeepGemmRunnerCore` does:

1. pre-permute / reorder tokens into a DeepGEMM-friendly layout
2. `grouped_gemm_nt_f8f8bf16_*` for gate+up
3. explicit `silu_and_mul`
4. re-quantize activations to FP8
5. `grouped_gemm_nt_f8f8bf16_*` for down projection
6. explicit post-permute / gather / combine

So the current DeepGEMM integration is:

- **communication outside DeepGEMM**
- **routing outside DeepGEMM**
- **combine outside DeepGEMM**
- **two GEMMs inside DeepGEMM**

### 1.3 Current backend selection is tied to existing FP8 / non-Mega paths

Relevant logic is currently spread across:

- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/layers/quantization/mxfp4.py`

None of these paths currently expose Mega MoE as an end-to-end backend.

## 2. DeepGEMM Mega MoE Kernel Interface

### 2.1 Where Mega MoE landed

Mega MoE was introduced in:

- `deepgemm` commit `7f2a703`

Primary files:

- `deep_gemm/mega/__init__.py`
- `csrc/apis/mega.hpp`
- `tests/test_mega_moe.py`

### 2.2 Python-facing API

DeepGEMM exposes three top-level Mega MoE entrypoints:

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch=True,
    activation="swiglu",
)

transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    l1_weights,
    l2_weights,
)

deep_gemm.fp8_fp4_mega_moe(
    y,
    transformed_l1,
    transformed_l2,
    buffer,
    recipe=(1, 1, 32),
    activation="swiglu",
    activation_clamp=...,
    fast_math=True,
)
```

### 2.3 Required inputs and dtypes

From `deep_gemm/mega/__init__.py`, `csrc/apis/mega.hpp`, and `tests/test_mega_moe.py`:

- `y`: BF16 output, shape `[num_tokens, hidden]`
- `buffer.x`: FP8 E4M3FN activations, shape `[num_max_tokens_per_rank, hidden]`
- `buffer.x_sf`: packed UE8M0 activation scales, dtype `torch.int`, shape `[num_max_tokens_per_rank, hidden / 128]`
- `buffer.topk_idx`: global expert ids, dtype `int64`, shape `[num_max_tokens_per_rank, num_topk]`
- `buffer.topk_weights`: routing weights, dtype `float32`, shape `[num_max_tokens_per_rank, num_topk]`
- `l1_weights`: local expert FP4 packed weights + scale tensor
- `l2_weights`: local expert FP4 packed weights + scale tensor

DeepGEMM's own test fills the buffer per call:

```python
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)
```

### 2.4 Mega MoE's FP4 numerical contract is MXFP4-like, not NVFP4

DeepGEMM SM100 FP4 uses:

- FP4 E2M1 payloads
- packed **UE8M0** scales (`torch.int` on the Python side)
- `gran_k = 32`

This is visible in:

- `deep_gemm/utils/math.py`
- `tests/test_mega_moe.py`
- `csrc/apis/mega.hpp`
- `README.md`

So the native Mega contract is much closer to `sglang`'s `mxfp4` path than to `modelopt_fp4`.

By contrast, `sglang` `modelopt_fp4` is explicitly NVFP4:

- E2M1 weights
- FP8-E4M3 weight scales
- block size `16`

That mismatch is fundamental enough that `modelopt_fp4` should not be the native Mega integration target.

### 2.5 Weight transform requirements

`transform_weights_for_mega_moe()` is not cosmetic. It changes layout:

- L1 (`gate+up`) is interleaved in gate/up chunks of 8
- L1 scales are transposed into the UTCCP-required layout
- L2 scales are also transposed into the UTCCP-required layout

So `sglang` cannot just pass its current backend-specific MoE expert tensors directly.

### 2.6 Hard constraints

Mega MoE currently assumes:

- **Blackwell only**
  - `csrc/apis/mega.hpp` dispatches only when `arch_major == 10`
- **PyTorch symmetric memory**
  - `deep_gemm/mega/__init__.py` uses `torch.distributed._symmetric_memory`
- **single NVLink domain**
  - this is an intra-domain communication kernel, not a multi-node RDMA path
- **SwiGLU only**
  - `activation == "swiglu"`
- **recipe fixed to `(1, 1, 32)`**
- `hidden % 128 == 0`
- `intermediate_hidden % 128 == 0`
- `num_experts % num_ranks == 0`

The buffer allocator also aligns `num_max_tokens_per_rank` to the internal `block_m`.

### 2.7 Mega MoE is full MoE forward, not a GEMM primitive

DeepGEMM's README describes Mega MoE as fusing:

- EP dispatch
- linear 1
- SwiGLU
- linear 2
- EP combine

That is the key interface fact for `sglang`: Mega MoE wants to start from **raw local tokens + global routing**, not from an already-dispatched token buffer.

## 3. How `sglang` Implements MoE Forward and Quantization Today

### 3.1 High-level forward path

The generic MoE forward is in:

- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

`FusedMoE.forward_impl()` does:

1. `dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)`
2. `combine_input = self.run_moe_core(dispatch_output)`
3. `final_hidden_states = self.dispatcher.combine(combine_input)`
4. optional TP/EP reduce

So `sglang` is architected around:

- dispatcher
- moe core
- combiner

### 3.2 `StandardDispatcher` is the right no-op path

`StandardDispatcher` in:

- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`

normally just forwards:

- raw `hidden_states`
- `topk_output`

For some backends, it intentionally preserves **global expert ids** and lets the runner own EP internally. That pattern matches DeepGEMM Mega MoE much better than the current DeepEP dispatcher path.

### 3.3 `modelopt_quant.py` is not the universal MoE quantization entry point

The real abstraction is the shared quantization interface in:

- `python/sglang/srt/layers/quantization/base_config.py`

`FusedMoE` asks the selected quantization config for a quantization method, then calls:

- `create_weights()`
- `create_moe_runner()`
- `apply()`

Both of these are first-class implementations of that interface:

- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/layers/quantization/mxfp4.py`

So landing Mega in `mxfp4.py` is not a workaround. It is the native place if Mega's numerical contract matches MXFP4.

### 3.4 `modelopt_fp4` is NVFP4-only

`modelopt_quant.py` explicitly treats ModelOpt FP4 as NVFP4:

- supported ModelOpt formats are `FP8` and `NVFP4`
- NVFP4 weights use FP8-E4M3 scales
- NVFP4 block size is `16`

That makes `ModelOptNvFp4FusedMoEMethod` the wrong native integration point for Mega.

### 3.5 `mxfp4.py` is the better native fit

`mxfp4.py` already defines:

- `Mxfp4Config`
- `Mxfp4MoEMethod`

and integrates with `FusedMoE` through the same `create_weights()` / `create_moe_runner()` / `apply()` contract as every other quantization path.

The key fit is:

- it is already the MXFP4-specific MoE implementation
- it already owns MoE weight preparation and runner selection
- its weight/scaling contract is block-32 MXFP4-like, which is much closer to DeepGEMM Mega than NVFP4 is

## 4. What This Means for Integration

### 4.1 Do not integrate Mega MoE into `DeepGemmRunnerCore`

This is still the wrong place because `DeepGemmRunnerCore` assumes:

- dispatch already happened
- combine will happen later outside the runner
- the kernel interface is grouped GEMM, not full MoE

If you plug Mega MoE there, you either:

- duplicate communication, or
- bypass the intended overlap and reduce Mega MoE to a worse abstraction

### 4.2 The clean integration shape is a new EP-owning runner backend

Recommended shape:

- add `MoeRunnerBackend.DEEP_GEMM_MEGA`
- use it with `MoeA2ABackend.NONE`
- preserve global top-k ids in `StandardDispatcher`
- register `@register_fused_func("none", "deep_gemm_mega")`

Why this is the right fit:

- `dispatch_output` stays `StandardDispatchOutput`
- the fused op receives raw local tokens + global routing
- DeepGEMM Mega MoE owns dispatch/combine internally
- `StandardDispatcher.combine()` can remain a pass-through

This is exactly the same architectural pattern `sglang` already uses for backends that own EP internally.

### 4.3 The first native target should be `mxfp4`, not `modelopt_fp4`

Mega MoE is an **FP8xFP4** kernel, but not every FP4 flavor is equivalent.

The first native integration target should therefore be:

- `Mxfp4MoEMethod`
- in `python/sglang/srt/layers/quantization/mxfp4.py`

and explicitly **not**:

- `ModelOptNvFp4FusedMoEMethod`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`

Reason:

- `mxfp4.py` is already a first-class quantization path in `sglang`
- its numerical contract is much closer to DeepGEMM Mega's FP4 contract
- it already owns MoE weight preparation and `MoeRunner` integration

### 4.4 No serving-time transcoding

For this integration, `sglang` should do **no** numerical transcoding from NVFP4 to MXFP4 during serving:

- no runtime per-batch conversion
- no load-time dequantize / requantize inside `sglang`

If an NVFP4 checkpoint needs to be used with Mega MoE, the model must be converted **offline before startup** into a Mega-compatible MXFP4 checkpoint.

So the serving integration should either:

- accept native / preconverted MXFP4 checkpoints, or
- reject the checkpoint with a clear error

### 4.5 Add a DeepGEMM-specific transformed weight cache on the MXFP4 path

Do not try to reuse backend-specific swizzled weights from:

- FlashInfer MXFP4
- Triton MXFP4
- existing backend-specific layout transforms

DeepGEMM Mega MoE still needs its own transformed representation:

- local expert FP4 packed weights in the DeepGEMM-expected view
- DeepGEMM-specific scale layout
- L1 gate/up interleave expected by `transform_weights_for_mega_moe()`

So the implementation should maintain a dedicated DeepGEMM-Mega cache, e.g.:

- `layer.deepgemm_mega_l1_weight`
- `layer.deepgemm_mega_l2_weight`

This cache should be built from the native MXFP4 checkpoint tensors by **layout repacking only**, not by numerical dequantize / requantize.

### 4.6 Activation handling

`sglang` MoE config uses:

- `activation == "silu"`
- plus `is_gated == True`

DeepGEMM Mega MoE expects:

- `activation == "swiglu"`

These should be treated as the same effective mode for gated MoE in this integration.

The activation path should:

- quantize `hidden_states` to FP8 E4M3FN
- quantize activations with per-token **group-32** scaling
- emit packed UE8M0 scales compatible with DeepGEMM
- produce `x_sf` with visible shape `[num_tokens, hidden / 128]`, i.e. 4 group-32 scales packed into each `torch.int32`
- copy both into the symmetric buffer

The likely reusable helper on the `sglang` side is its existing FP8 quantization path used for DeepGEMM / Blackwell-compatible activations.

One important implementation detail for this helper reuse:

- the Mega path should still call `sglang_per_token_group_quant_fp8(group_size=32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)`
- but the shared UE8M0 packing path must derive its packed scale width from `group_size`, not from a hardcoded `128`
- for Mega, this means keeping the existing call shape while making the helper emit packed scales that match DeepGEMM's real activation contract instead of the old group-128 assumption used by the existing DeepGEMM FP8 GEMM path

### 4.7 Suggested control flow in `sglang`

Recommended fast path:

1. `FusedMoE.forward_impl()` uses `StandardDispatcher` because `moe_a2a_backend == none`.
2. `StandardDispatcher` does **not** remap expert ids to local ids for `deep_gemm_mega`.
3. `Mxfp4MoEMethod` creates `DeepGemmMegaMoeQuantInfo`.
4. `@register_fused_func("none", "deep_gemm_mega")`:
   - quantizes activations to FP8 + packed UE8M0 scale using the existing FP8 helper with `group_size=32`
   - lazily allocates / reuses a symmetric buffer
   - copies `x`, `x_sf`, `topk_ids`, `topk_weights` into the buffer
   - calls `deep_gemm.fp8_fp4_mega_moe(...)`
   - returns `StandardCombineInput(hidden_states=output)`
5. Existing `StandardDispatcher.combine()` returns the output as-is.
6. Existing TP/EP all-reduce logic remains unchanged.

## 5. Concrete `sglang` Touch Points

If I were implementing this revised design, I would start with these files:

- `python/sglang/srt/layers/moe/utils.py`
  - add `MoeRunnerBackend.DEEP_GEMM_MEGA`

- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
  - make `skip_local_expert_mapping` true for `deep_gemm_mega`

- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
  - expose capability checks and thin wrappers for:
    - `get_symm_buffer_for_mega_moe`
    - `transform_weights_for_mega_moe`
    - `fp8_fp4_mega_moe`

- `python/sglang/srt/layers/quantization/fp8_kernel.py`
  - make the shared `scale_ue8m0=True` path pack scales according to the requested `group_size`
  - preserve the current API and column-major/TMA-aligned contract so Mega can keep using `sglang_per_token_group_quant_fp8(...)` directly

- `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`
  - define `DeepGemmMegaMoeQuantInfo`
  - register `@register_fused_func("none", "deep_gemm_mega")`
  - add a local assertion that the quantized activation scale tensor matches the symmetric buffer view before copying

- `python/sglang/srt/layers/quantization/mxfp4.py`
  - create / cache transformed DeepGEMM-Mega weights
  - instantiate `MoeRunner(MoeRunnerBackend.DEEP_GEMM_MEGA, ...)`
  - package the quant info
  - reject non-native checkpoints instead of transcoding them

- optionally `python/sglang/srt/server_args.py`
  - expose a backend string if you want a first-class CLI knob

## 6. Things I Would Explicitly Not Do in V1

- Do not route Mega MoE through `DeepEPDispatcher`.
- Do not extend the current `DeepGemmRunnerCore` grouped-GEMM path.
- Do not enable it for `modelopt_fp4` / NVFP4.
- Do not do runtime transcoding.
- Do not do load-time transcoding inside `sglang`.
- Do not enable it for multi-node / RDMA EP.
- Do not enable it for non-Blackwell GPUs.
- Do not enable it for non-gated activations.
- Do not assume existing FlashInfer / Triton weight swizzles are reusable.
- Do not add a Mega-only ad hoc activation quantizer if the existing FP8 helper can be made group-size-correct with a narrow fix.

## 7. Practical Rollout Recommendation

I would implement V1 with the following guardrails:

- explicit backend opt-in: `--moe-runner-backend deep_gemm_mega`
- only for:
  - CUDA
  - Blackwell
  - `quantization == mxfp4`
  - `ep_size == tp_size`
  - `ep_size > 1`
  - `moe_a2a_backend == none`
  - single NVLink domain
  - gated SiLU / SwiGLU
- reject NVFP4 checkpoints unless they were preconverted offline before serving
- fallback to existing backend otherwise

This gives you a narrow, correct first integration that matches DeepGEMM Mega MoE's real interface instead of forcing it into the old grouped-GEMM design or into the wrong FP4 numerical format.

## 8. Bottom Line

The main conclusion is:

- **Mega MoE should be integrated as a new backend that owns EP internally**
- **not** as an extension of `sglang`'s current DeepGEMM grouped-GEMM runner
- **and not** as a native `modelopt_fp4` backend

In `sglang` terms, the right mental model is:

- architecturally closer to `flashinfer_cutedsl` / EP-owning fused backends
- numerically and kernel-wise driven by DeepGEMM's FP8xFP4 Mega MoE contract
- natively aligned with `sglang`'s `mxfp4` path, not its NVFP4 ModelOpt path
- operationally limited to single-node Blackwell NVLink domains

That is the cleanest way to make `sglang` consume the new DeepGEMM Mega MoE kernel without fighting the kernel's actual interface or introducing serving-time transcoding.

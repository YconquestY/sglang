# SGLang + DeepGEMM Mega MoE Research

This note is based on the local source trees on 2026-04-18:

- `deepgemm` HEAD = `7f2a703` (`[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes`)
- `sglang` local checkout in `/home/yy010/proj/sglang`

## Executive Summary

`sglang` does **not** use DeepGEMM today as an end-to-end MoE kernel. It currently uses DeepGEMM only as a **grouped GEMM backend** inside the MoE core, with explicit token reorder / scatter / combine logic around it.

DeepGEMM Mega MoE is a different abstraction boundary:

- it fuses **dispatch + GEMM1 + SwiGLU + GEMM2 + combine**
- it owns the **NVLink communication**
- it expects **raw token inputs + global top-k routing**
- it requires **FP8 activations + FP4 weights**, **symmetric memory**, and **Blackwell**

Because of that, the correct integration is **not** to extend `DeepGemmRunnerCore` in `sglang`. The correct integration is to add a **new MoE backend path** that lets DeepGEMM Mega MoE own the entire MoE forward inside the NVLink domain.

The cleanest `sglang` shape is:

- add a new runner backend, e.g. `deep_gemm_mega`
- use `MoeA2ABackend.NONE` + `StandardDispatcher`
- preserve **global** expert ids in the standard dispatch path
- register a fused op `@register_fused_func("none", "deep_gemm_mega")`
- call DeepGEMM Mega MoE directly from that fused op
- keep the existing DeepEP / grouped-GEMM / FlashInfer paths as fallback

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

### 1.3 Current backend selection is tied to FP8 MoE

Relevant logic is in:

- `python/sglang/srt/layers/quantization/fp8.py`

`Fp8MoEMethod.create_moe_runner()` auto-selects `MoeRunnerBackend.DEEP_GEMM` when:

- the MoE backend is `auto` or explicitly `deep_gemm`
- DeepGEMM is available
- the MoE A2A backend is `deepep`, `mooncake`, or `nixl`

This is a stale DeepGEMM use case relative to Mega MoE:

- current path is **FP8 MoE**
- Mega MoE is **FP8 activations + FP4 weights**

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

DeepGEMM’s own test fills the buffer per call:

```python
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)
```

### 2.4 Weight transform requirements

`transform_weights_for_mega_moe()` is not cosmetic. It changes layout:

- L1 (`gate+up`) is interleaved in gate/up chunks of 8
- L1 scales are transposed into the UTCCP-required layout
- L2 scales are also transposed into UTCCP-required layout

So `sglang` cannot just pass its current FP4 expert tensors directly.

### 2.5 Hard constraints

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

### 2.6 Mega MoE is full MoE forward, not a GEMM primitive

DeepGEMM’s README describes Mega MoE as fusing:

- EP dispatch
- linear 1
- SwiGLU
- linear 2
- EP combine

That is the key interface fact for `sglang`: Mega MoE wants to start from **raw local tokens + global routing**, not from an already-dispatched token buffer.

## 3. How `sglang` Implements MoE Forward Today

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

### 3.2 Expert-parallel dispatchers

Dispatcher creation is in:

- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:create_moe_dispatcher`

Important cases:

- `MoeA2ABackend.NONE` -> `StandardDispatcher`
- `DEEPEP` / `MOONCAKE` / `NIXL` -> DeepEP-class dispatcher

The DeepEP dispatcher returns either:

- `DeepEPNormalDispatchOutput`
- `DeepEPLLDispatchOutput`

from:

- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

### 3.3 Current DeepEP path already performs communication before the runner

For low-latency mode, `deepep.py` calls:

- `buffer.low_latency_dispatch(...)`
- later `buffer.low_latency_combine(...)`

So if `sglang` first enters DeepEP low-latency dispatch and then calls Mega MoE, communication has already happened and the Mega kernel cannot deliver its intended overlap.

### 3.4 Standard dispatcher is the right no-op path

`StandardDispatcher` in:

- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`

normally just forwards:

- raw `hidden_states`
- `topk_output`

For some backends, it intentionally preserves **global expert ids** and lets the runner own EP internally. Today this is already done for backends such as:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`
- `flashinfer_trtllm_routed`

That pattern matches DeepGEMM Mega MoE much better than the current DeepEP dispatcher path.

## 4. What This Means for Integration

## 4.1 Do not integrate Mega MoE into `DeepGemmRunnerCore`

This is the wrong place because `DeepGemmRunnerCore` assumes:

- dispatch already happened
- combine will happen later outside the runner
- the kernel interface is grouped GEMM, not full MoE

If you plug Mega MoE there, you either:

- duplicate communication, or
- bypass the intended overlap and reduce Mega MoE to a worse abstraction

## 4.2 The clean integration shape is a new EP-owning runner backend

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

## 4.3 The first target should be FP4 MoE, not current FP8 DeepGEMM MoE

Mega MoE is an **FP8xFP4** kernel.

Therefore the integration should target an FP4 MoE quantization path, not:

- `Fp8MoEMethod`
- `DeepGemmRunnerCore`

The closest existing control-plane landing zone in `sglang` is:

- `ModelOptNvFp4FusedMoEMethod`

Reason:

- it already owns FP4 expert weights
- it already plugs into `MoeRunner`
- it already supports backends that own EP internally via fused funcs

This is an inference from the code structure, but it is the cleanest existing fit.

## 4.4 Add a DeepGEMM-specific transformed weight copy

Do not try to reuse backend-specific swizzled weights from:

- FlashInfer TRTLLM
- FlashInfer CuteDSL
- existing blockscale-swizzled tensors

DeepGEMM Mega MoE needs its own transformed representation:

- original FP4 packed local expert weights
- DeepGEMM-specific scale layout
- L1 gate/up interleave expected by `transform_weights_for_mega_moe()`

So the implementation should maintain a dedicated DeepGEMM-Mega cache, e.g.:

- `layer.deepgemm_mega_l1_weight`
- `layer.deepgemm_mega_l2_weight`
- maybe lazily built on first use

## 4.5 Activation handling

`sglang` MoE config uses:

- `activation == "silu"`
- plus `is_gated == True`

DeepGEMM Mega MoE expects:

- `activation == "swiglu"`

These should be treated as the same effective mode for gated MoE in this integration.

The activation path should:

- quantize `hidden_states` to FP8 E4M3FN
- emit packed UE8M0 scales compatible with DeepGEMM
- copy both into the symmetric buffer

The likely reusable helper on the `sglang` side is its existing FP8 quantization path used for DeepGEMM / Blackwell-compatible activations.

## 4.6 Suggested control flow in `sglang`

Recommended fast path:

1. `FusedMoE.forward_impl()` uses `StandardDispatcher` because `moe_a2a_backend == none`.
2. `StandardDispatcher` does **not** remap expert ids to local ids for `deep_gemm_mega`.
3. `ModelOptNvFp4FusedMoEMethod` (or a dedicated DeepGEMM-Mega MoE method) creates `DeepGemmMegaMoeQuantInfo`.
4. `@register_fused_func("none", "deep_gemm_mega")`:
   - quantizes activations to FP8 + UE8M0 scale
   - lazily allocates / reuses a symmetric buffer
   - copies `x`, `x_sf`, `topk_ids`, `topk_weights` into the buffer
   - calls `deep_gemm.fp8_fp4_mega_moe(...)`
   - returns `StandardCombineInput(hidden_states=output)`
5. Existing `StandardDispatcher.combine()` returns the output as-is.
6. Existing TP/EP all-reduce logic remains unchanged.

## 5. Concrete `sglang` Touch Points

If I were implementing this, I would start with these files:

- `python/sglang/srt/layers/moe/utils.py`
  - add `MoeRunnerBackend.DEEP_GEMM_MEGA`

- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
  - make `skip_local_expert_mapping` true for `deep_gemm_mega`

- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
  - expose capability checks and thin wrappers for:
    - `get_symm_buffer_for_mega_moe`
    - `transform_weights_for_mega_moe`
    - `fp8_fp4_mega_moe`

- `python/sglang/srt/layers/moe/moe_runner/deep_gemm_mega.py`
  - define `DeepGemmMegaMoeQuantInfo`
  - register `@register_fused_func("none", "deep_gemm_mega")`

- `python/sglang/srt/layers/quantization/modelopt_quant.py`
  - create / cache transformed DeepGEMM-Mega weights
  - instantiate `MoeRunner(MoeRunnerBackend.DEEP_GEMM_MEGA, ...)`
  - package the quant info

- optionally `python/sglang/srt/server_args.py`
  - expose a backend string if you want a first-class CLI knob

## 6. Things I Would Explicitly Not Do in V1

- Do not route Mega MoE through `DeepEPDispatcher`.
- Do not extend the current `DeepGemmRunnerCore` grouped-GEMM path.
- Do not enable it for multi-node / RDMA EP.
- Do not enable it for non-Blackwell GPUs.
- Do not enable it for non-gated activations.
- Do not enable it for arbitrary quantization methods on day 1.
- Do not assume existing FlashInfer weight swizzles are reusable.

## 7. Practical Rollout Recommendation

I would implement V1 with the following guardrails:

- explicit backend opt-in: `--moe-runner-backend deep_gemm_mega`
- only for:
  - CUDA
  - Blackwell
  - EP world size > 1
  - single NVLink domain
  - FP4 MoE quantization path
  - gated SiLU / SwiGLU
- fallback to existing backend otherwise

This gives you a narrow, correct first integration that matches DeepGEMM Mega MoE’s real interface instead of forcing it into the old grouped-GEMM design.

## 8. Bottom Line

The main conclusion is:

- **Mega MoE should be integrated as a new backend that owns EP internally**
- **not** as an extension of `sglang`’s current DeepGEMM grouped-GEMM runner

In `sglang` terms, the right mental model is:

- architecturally closer to `flashinfer_cutedsl` / EP-owning fused backends
- numerically and kernel-wise driven by DeepGEMM’s FP8xFP4 Mega MoE contract
- operationally limited to single-node Blackwell NVLink domains

That is the cleanest way to make `sglang` consume the new DeepGEMM Mega MoE kernel without fighting the kernel’s actual interface.

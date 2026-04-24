# DeepGEMM MegaMoE and SGLang Integration Report

Generated: 2026-04-24

## Scope

This report compares the MegaMoE interface in the cloned upstream DeepGEMM repo at `../DeepGEMM` with the SGLang-side implementation on the `deepseek_v4` branch.

Sources inspected:

- DeepGEMM clone: `/sgl-workspace/DeepGEMM`, `main` at `7f2a703` (`[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes (#304)`).
- SGLang repo: `/sgl-workspace/sglang`, branch `deepseek_v4`.
- Key DeepGEMM files: `README.md`, `deep_gemm/mega/__init__.py`, `csrc/apis/mega.hpp`, `csrc/apis/layout.hpp`, `csrc/jit_kernels/heuristics/mega_moe.hpp`, `deep_gemm/utils/math.py`, `tests/test_mega_moe.py`.
- Key SGLang files: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`, `sgl-kernel/CMakeLists.txt`.

## Executive Summary

DeepGEMM MegaMoE is a fused SM100-oriented MoE kernel that combines EP dispatch, L1 FP8xFP4 GEMM, SwiGLU, L2 FP8xFP4 GEMM, and EP combine into one symmetric-memory communication/computation kernel. Its public API expects the caller to allocate a symmetric buffer, transform FP4 expert weights and UE8M0 scales into a specific layout, fill per-rank input/routing fields in that buffer, and call `deep_gemm.fp8_fp4_mega_moe`.

SGLang's `deepseek_v4` branch integrates this as a routed-expert fast path for DeepSeek V4 FP4 experts. The integration is real and deep: it builds MegaMoE-formatted expert weights at load time, caches DeepGEMM symmetric buffers by process group and shape, fills DeepGEMM's buffer either with a custom fused pre-dispatch JIT kernel or a slower fallback, and invokes `deep_gemm.fp8_fp4_mega_moe` with `recipe=(1, 1, 32)`.

The default is still off. `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` defaults to `False`, and the path is further gated by weight-preparation status, token cap, `nextn`, and hash-MoE constraints.

There is an important packaging mismatch in the current SGLang tree: `sgl-kernel/CMakeLists.txt` pins `sgl-project/DeepGEMM` at `35c4bc87713726d048f65275f6f1b551a4e7a6dc`, which predates upstream DeepGEMM's MegaMoE merge at `7f2a703`. A build using that pin will not contain the MegaMoE Python APIs SGLang calls. The branch therefore requires a newer DeepGEMM package/build that includes the MegaMoE release commit or an equivalent fork commit.

## DeepGEMM MegaMoE Interface

### Public API

DeepGEMM exports the MegaMoE interface from `deep_gemm/mega/__init__.py` and re-exports it from `deep_gemm/__init__.py`.

The three public pieces used by SGLang are:

- `get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, use_fp8_dispatch=True, activation="swiglu")`
- `transform_weights_for_mega_moe(l1_weights, l2_weights)`
- `fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, recipe=(1, 1, 32), activation="swiglu", activation_clamp=None, fast_math=True)`

The README describes the intended sequence:

1. Allocate a symmetric memory buffer with `get_symm_buffer_for_mega_moe`.
2. Transform FP4 weights and UE8M0 scale factors with `transform_weights_for_mega_moe`.
3. Fill `buffer.x`, `buffer.x_sf`, `buffer.topk_idx`, and `buffer.topk_weights` before every call.
4. Allocate BF16 output `y`.
5. Call `deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)`.

Reference: `../DeepGEMM/README.md:114`.

### Required Hardware and Runtime

DeepGEMM's general README lists SM90 or SM100 support for the library, but the MegaMoE C++ API dispatches only when `arch_major == 10`:

- `csrc/apis/mega.hpp` calls `sm100_fp8_fp4_mega_moe(...)` only for `arch_major == 10`.
- The `else` branch is `DG_HOST_UNREACHABLE("Unsupported architecture")`.

Reference: `../DeepGEMM/csrc/apis/mega.hpp:185`.

MegaMoE also depends on PyTorch symmetric memory:

- `deep_gemm/mega/__init__.py` imports `torch.distributed._symmetric_memory as symm_mem`.
- `SymmBuffer` allocates `symm_mem.empty(...)` and calls `symm_mem.rendezvous(...)`.
- README notes PyTorch >= 2.9 for the symmetric memory buffer.

References: `../DeepGEMM/deep_gemm/mega/__init__.py:8`, `../DeepGEMM/deep_gemm/mega/__init__.py:38`, `../DeepGEMM/README.md:120`.

### Symmetric Buffer Contract

`get_symm_buffer_for_mega_moe` aligns `num_max_tokens_per_rank` to DeepGEMM's MegaMoE `block_m` before creating the buffer:

- `block_m = _C.get_block_m_for_mega_moe(...)`
- `num_max_tokens_per_rank = align(num_max_tokens_per_rank, block_m)`

Reference: `../DeepGEMM/deep_gemm/mega/__init__.py:58`.

The current heuristic returns a fixed `block_m = 192`.

Reference: `../DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp:58`.

The C++ buffer layout validates:

- `num_experts % num_ranks == 0`.
- `hidden % 128 == 0`.
- `intermediate_hidden % 128 == 0`.
- Padded SF pool tokens are divisible by 4.

Reference: `../DeepGEMM/csrc/apis/mega.hpp:14`, `../DeepGEMM/csrc/apis/mega.hpp:78`.

The Python `SymmBuffer` exposes these views over one raw symmetric byte buffer:

| Field | Dtype | Shape / Layout |
|---|---:|---|
| `x` | `torch.float8_e4m3fn` | `[num_max_tokens_per_rank, hidden]` |
| `x_sf` | `torch.int` | `[num_max_tokens_per_rank, hidden / 128]`; each int packs 4 UE8M0 scale bytes, covering 4 groups of 32 K values |
| `topk_idx` | `torch.int64` | `[num_max_tokens_per_rank, num_topk]` |
| `topk_weights` | `torch.float32` | `[num_max_tokens_per_rank, num_topk]` |
| `l1_acts` | `torch.float8_e4m3fn` | `[num_max_pool_tokens, hidden]` |
| `l1_acts_sf` | `torch.int` | `[num_padded_sf_pool_tokens, hidden / 128]` with non-contiguous M-major stride `{1, num_padded_sf_pool_tokens}` |
| `l2_acts` | `torch.float8_e4m3fn` | `[num_max_pool_tokens, intermediate_hidden]` |
| `l2_acts_sf` | `torch.int` | `[num_padded_sf_pool_tokens, intermediate_hidden / 128]` with non-contiguous M-major stride `{1, num_padded_sf_pool_tokens}` |

Reference: `../DeepGEMM/csrc/apis/mega.hpp:82`.

The comment in `mega.hpp` is especially important: input `x_sf` is K-major, while intermediate activation scales `l1_acts_sf` and `l2_acts_sf` are M-major. SGLang's pre-dispatch kernel only fills the input fields; DeepGEMM fills the internal activation fields.

### Kernel Call Contract

The C++ API validates the following before launching:

- `recipe == (1, 1, 32)`.
- `activation == "swiglu"`.
- `activation_clamp >= 0` if present.
- L1/L2 weights are K-major packed FP4 grouped tensors.
- L1 shape logically represents `[num_experts_per_rank, 2 * intermediate_hidden, hidden]`.
- L2 shape logically represents `[num_experts_per_rank, hidden, intermediate_hidden]`.
- `num_tokens <= num_max_tokens_per_rank`.
- L1/L2 weights are contiguous.
- L1/L2 scale factors use UE8M0 packed int layout with `kGranMN = 1`, `kGranK = 32`.
- `num_experts == num_experts_per_rank * num_ranks`.

Reference: `../DeepGEMM/csrc/apis/mega.hpp:124`.

### Numerical Format

#### Activations

MegaMoE consumes input activations in FP8 E4M3FN with per-token grouped scales:

- `buffer.x`: `torch.float8_e4m3fn`
- `buffer.x_sf`: packed UE8M0 in `torch.int`
- Group size along K: `32`
- Packed scale shape: `hidden / 128` int32 values per token, because each int32 packs four 8-bit UE8M0 exponents and each exponent covers 32 K values.

DeepGEMM's utility `per_token_cast_to_fp8` shows the exact math used by its test path:

- Pad K to `gran_k`.
- Reshape to `[M, padded_K / gran_k, gran_k]`.
- `amax = abs(x).amax(group).clamp(1e-4)`.
- `sf = amax / 448.0`.
- If `use_ue8m0`, round scale up to a power-of-two UE8M0 exponent.
- Quantize with `(x / sf).to(torch.float8_e4m3fn)`.
- If packed, pack 4 UE8M0 exponents into one int32.

References: `../DeepGEMM/deep_gemm/utils/math.py:25`, `../DeepGEMM/tests/test_mega_moe.py:94`.

SGLang's fused pre-dispatch kernel implements the same activation-side contract for the MegaMoE buffer:

- Input `hidden_states` is BF16.
- It uses one CTA per token.
- It reduces absmax per 32-element group.
- It computes `raw_scale = absmax / FP8_E4M3_MAX`.
- It converts the scale to UE8M0.
- It writes FP8 E4M3 output into `buf.x`.
- It writes scale exponent bytes into `buf.x_sf` at byte offset `token_id * num_groups + group_id`.

References: `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:52`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:73`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:89`.

#### Weights

MegaMoE uses FP4 E2M1 expert weights packed into int8:

- Two FP4 values per int8.
- Scale group size along K: `32`.
- Scale format: packed UE8M0 in `torch.int`.
- Scale recipe: `(1, 32)` at weight-transform time and `(1, 1, 32)` for the final MegaMoE call.

DeepGEMM's utility quantizes FP4 this way:

- `sf = amax / 6.0`.
- FP4 value set is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` plus sign.
- It packs two 4-bit codes into one int8.
- `use_ue8m0=True` rounds scales to UE8M0.

Reference: `../DeepGEMM/deep_gemm/utils/math.py:72`, `../DeepGEMM/deep_gemm/utils/math.py:85`.

The MegaMoE test quantizes both L1 and L2 BF16 reference weights to FP4 with `gran_k=32`, then calls `transform_sf_into_required_layout(..., recipe=(1, 32), num_groups=...)`.

Reference: `../DeepGEMM/tests/test_mega_moe.py:97`.

#### Intermediate Activations

The fused kernel's logical work matches the DeepGEMM test baseline:

1. EP dispatch receives FP8 input and top-k routing.
2. L1 grouped GEMM runs `FP8 x FP4 -> BF16`, producing gate/up.
3. SwiGLU applies top-k weights and quantizes the activation to FP8 with UE8M0, `num_per_channels=32`.
4. L2 grouped GEMM runs `FP8 x FP4 -> BF16`.
5. EP combine returns BF16 output.

Reference: `../DeepGEMM/tests/test_mega_moe.py:112`.

### Weight Layout Transform

`transform_weights_for_mega_moe` does two format changes:

- L1: interleave gate and up weights in chunks of 8 rows, then transpose the scale factors for UTCCP.
- L2: leave packed weights as-is, but transpose scale factors for UTCCP.

Reference: `../DeepGEMM/deep_gemm/mega/__init__.py:77`, `../DeepGEMM/deep_gemm/mega/__init__.py:98`.

The L1 interleaving changes the first dimension of the output channel axis from `[gate all rows | up all rows]` to `[gate rows 0..7, up rows 0..7, gate rows 8..15, up rows 8..15, ...]`.

The UTCCP scale transpose requires:

- Scale dtype is `torch.int`.
- `mn % 128 == 0`.
- Reshape to `[num_groups, -1, 4, 32, packed_sf_k]`.
- Transpose the `4` and `32` axes.
- Flatten back.

Reference: `../DeepGEMM/deep_gemm/mega/__init__.py:89`.

### Scale Layout Conversion

`transform_sf_into_required_layout` handles FP32 scale input and converts it to SM100 packed UE8M0:

- For SM100, FP32 scale with `gran_k` 32 or 128 is broadcast if needed, converted to packed UE8M0, TMA-aligned, and MN-major.
- For SM100, already-packed int scale with `gran_mn == 1` and `gran_k` 32 or 128 is validated as TMA-aligned MN-major.

Reference: `../DeepGEMM/csrc/apis/layout.hpp:47`.

This matters for SGLang because the DeepSeek V4 FP4 checkpoint stores scale tensors that SGLang passes to `transform_sf_into_required_layout` before calling `transform_weights_for_mega_moe`.

## SGLang Integration

### Model Wiring

DeepSeek V4 does not implement a separate MoE module for MegaMoE. Its decoder layer instantiates `deepseek_v2.DeepseekV2MoE` with `is_deepseek_v4=True`.

Reference: `python/sglang/srt/models/deepseek_v4.py:945`.

The shared MoE `forward` checks `_should_use_mega_moe(hidden_states)` first. If true, it dispatches directly to `forward_mega_moe`.

Reference: `python/sglang/srt/models/deepseek_v2.py:622`.

### Load-Time Weight Preparation

SGLang builds MegaMoE weights during FP8/FP4 MoE post-load processing:

- If `self.is_fp4_expert`, the packed weight tensors are viewed as `torch.int8`.
- If `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` is true, it imports and calls `build_mega_moe_experts_weights(layer)`.
- It then returns early, bypassing the normal DeepGEMM scale conversion path.

Reference: `python/sglang/srt/layers/quantization/fp8.py:931`.

`build_mega_moe_experts_weights`:

- Imports DeepGEMM's `transform_sf_into_required_layout` and `transform_weights_for_mega_moe`.
- Imports private helpers `_interleave_l1_weights` and `_transpose_sf_for_utccp`.
- Converts `w13_weight_scale_inv` and `w2_weight_scale_inv` with `recipe=(1, 32)` and `num_groups`.
- Either calls upstream `transform_weights_for_mega_moe`, or manually interleaves/transposes in the memory-saving mode.
- Stores `experts.mega_l1_weights` and `experts.mega_l2_weights`.
- Marks `experts._mega_moe_weights_built = True`.

Reference: `python/sglang/srt/models/deepseek_v4.py:1977`.

The optional memory-saving mode is controlled by `SGLANG_OPT_FIX_MEGA_MOE_MEMORY`. In this mode SGLang shares the L1 interleaved weight buffer with the normal DeepEP path and keeps only the extra UTCCP-transposed scale for MegaMoE.

Reference: `python/sglang/srt/models/deepseek_v4.py:2014`.

The MXFP4 loader has a guard that returns immediately if MegaMoE weights have already been built, avoiding a later reshuffle that would invalidate the MegaMoE layout.

Reference: `python/sglang/srt/layers/quantization/mxfp4_deepseek.py:220`.

### Runtime Gating

MegaMoE is off by default:

- `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)`.
- `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)`.
- `SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH = EnvBool(True)`.
- `SGLANG_OPT_FIX_HASH_MEGA_MOE = EnvBool(False)`.
- `SGLANG_OPT_FIX_MEGA_MOE_MEMORY = EnvBool(False)`.

Reference: `python/sglang/srt/environ.py:508`.

`_should_use_mega_moe` requires:

- `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` is true.
- `self.experts._mega_moe_weights_built` exists and is true.
- `self.is_nextn` is false.
- Hash MoE is disabled unless `SGLANG_OPT_FIX_HASH_MEGA_MOE` is true.
- During CUDA graph capture, the path is allowed immediately after the above checks.
- Outside capture, `max_tokens_per_rank <= SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`.

Reference: `python/sglang/srt/models/deepseek_v2.py:1081`.

### Symmetric Buffer Use

SGLang wraps DeepGEMM buffer allocation in `_get_mega_moe_symm_buffer`:

- It keys a process-global cache by process-group identity, token cap, expert count, top-k, hidden, and intermediate hidden.
- It calls `deep_gemm.get_symm_buffer_for_mega_moe(...)` with `use_fp8_dispatch=True` and `activation="swiglu"`.
- It stores the returned `SymmBuffer` for reuse.

Reference: `python/sglang/srt/models/deepseek_v2.py:372`.

At runtime, `_run_mega_routed`:

- Gets the EP group from `get_moe_ep_group().device_group`.
- Uses `num_experts = self.experts.num_experts`.
- Computes `top_k = config.num_experts_per_tok + num_fused_shared_experts`.
- Uses `intermediate_size = config.moe_intermediate_size`.
- Uses the token cap env var as `num_max_tokens_per_rank`.
- Asserts local `num_tokens <= cap` before calling DeepGEMM.

Reference: `python/sglang/srt/models/deepseek_v2.py:1187`.

### Routing and Input Fill

If `num_tokens > 0`, SGLang computes:

- `router_logits = self.gate(hidden_states, forward_batch=forward_batch)`.
- Hash-MoE input IDs when `self.is_hash`.
- `topk_output = self.topk(...)`.
- `topk_ids = topk_output.topk_ids`.
- `topk_weights = topk_output.topk_weights`.

Reference: `python/sglang/srt/models/deepseek_v2.py:1165`.

Then SGLang fills the DeepGEMM buffer by one of two paths.

#### Default path: fused pre-dispatch kernel

Enabled by `SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH=True`.

The Python wrapper `mega_moe_pre_dispatch` JIT-loads `deepseek_v4/mega_moe_pre_dispatch.cuh`.

References: `python/sglang/jit_kernel/deepseek_v4.py:259`, `python/sglang/jit_kernel/deepseek_v4.py:835`.

SGLang calls it with `quant_group_size=32` and passes:

- BF16 hidden states.
- `topk_ids` as int32.
- `topk_weights` as float32.
- DeepGEMM `buf.x`.
- DeepGEMM `buf.x_sf`.
- DeepGEMM `buf.topk_idx`.
- DeepGEMM `buf.topk_weights`.

Reference: `python/sglang/srt/models/deepseek_v2.py:1211`.

The kernel does four jobs:

1. Quantize BF16 hidden states to `buf.x` as FP8 E4M3.
2. Compute per-token per-32-group UE8M0 scales and write them into DeepGEMM's packed `buf.x_sf` int32 layout.
3. Copy `topk_weights`.
4. Convert `topk_idx` from int32 input to int64 buffer storage.
5. Pad rows after `num_tokens` with `topk_idx=-1` and `topk_weights=0`.

References: `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:20`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:89`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:99`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:105`.

The kernel validates the exact shape/dtype contract:

- Input `x`: BF16 `[M, H]`.
- Input `topk_idx`: int32 `[M, K]`.
- Input `topk_weights`: float `[M, K]`.
- `buf_x`: FP8 `[P, H]`.
- `buf_x_sf`: int32 `[P, G/4]`, contiguous, where `G = hidden / group_size`.
- `buf_topk_idx`: int64 `[P, K]`.
- `buf_topk_weights`: float `[P, K]`.

It only supports `group_size` 32, 64, or 128 at compile time, but SGLang calls it with 32 for MegaMoE.

Reference: `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:123`.

#### Fallback path: separate quantize and copy

If fused pre-dispatch is disabled, SGLang calls `sglang_per_token_group_quant_fp8_ue8m0(hidden_states, group_size=32)`, then copies the result into `buf.x`, `buf.x_sf`, `buf.topk_idx`, and `buf.topk_weights`.

Reference: `python/sglang/srt/models/deepseek_v2.py:1232`.

### MegaMoE Invocation

After the buffer is filled, SGLang allocates BF16 output:

```python
y = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)
```

Then it calls:

```python
deep_gemm.fp8_fp4_mega_moe(
    y,
    self.experts.mega_l1_weights,
    self.experts.mega_l2_weights,
    buf,
    recipe=(1, 1, 32),
    activation="swiglu",
    activation_clamp=swiglu_limit,
    fast_math=True,
)
```

Reference: `python/sglang/srt/models/deepseek_v2.py:1245`.

If the top-k path did not already fuse the routed scaling factor, SGLang multiplies the BF16 output by `self.routed_scaling_factor`.

Reference: `python/sglang/srt/models/deepseek_v2.py:1262`.

### Shared Expert Handling

SGLang's MegaMoE path applies DeepGEMM MegaMoE to routed experts. Shared experts are handled separately:

- `forward_mega_moe` computes `shared_output = self._forward_shared_experts(hidden_states)`.
- It runs `_run_mega_routed(...)`.
- It adds `shared_output` to the routed result if present.

Reference: `python/sglang/srt/models/deepseek_v2.py:1128`.

This differs from the most literal reading of DeepGEMM README's "fuses EP dispatch, linear 1, SwiGLU, linear 2, and EP combine" because SGLang's surrounding model still owns shared experts and residual scaling policy.

### Relationship to Existing DeepGEMM MoE Runner

The MegaMoE path is separate from SGLang's existing `moe_runner/deep_gemm.py` path.

The existing DeepGEMM runner handles normal EP/MoE stages with separate dispatch, pre-permute, grouped GEMM, activation quantization, grouped GEMM, post-permute, and combine stages. MegaMoE bypasses that runner by returning early from `DeepseekV2MoE.forward` before `_enable_a2a_moe` handling and before the modular runner pipeline.

Reference: `python/sglang/srt/models/deepseek_v2.py:632`.

## Group Quantization Size

The effective MegaMoE group quantization size is 32 along K for both activation and weight scaling.

Evidence:

- DeepGEMM MegaMoE API requires `recipe == (1, 1, 32)`.
- DeepGEMM weight SF checks use `kGranMN=1`, `kGranK=32`.
- DeepGEMM test casts input activations with `per_token_cast_to_fp8(..., gran_k=32, use_packed_ue8m0=True)`.
- DeepGEMM test casts FP4 weights with `per_token_cast_to_fp4(..., gran_k=32)`.
- SGLang builds MegaMoE weight scales with `recipe=(1, 32)`.
- SGLang pre-dispatch fills activation scales with `quant_group_size=32`.
- SGLang calls `deep_gemm.fp8_fp4_mega_moe(..., recipe=(1, 1, 32))`.

References:

- `../DeepGEMM/csrc/apis/mega.hpp:141`
- `../DeepGEMM/csrc/apis/mega.hpp:164`
- `../DeepGEMM/tests/test_mega_moe.py:94`
- `../DeepGEMM/tests/test_mega_moe.py:104`
- `python/sglang/srt/models/deepseek_v4.py:1997`
- `python/sglang/srt/models/deepseek_v2.py:1230`
- `python/sglang/srt/models/deepseek_v2.py:1256`

## Build and Packaging Risk

The current SGLang source code expects MegaMoE APIs from DeepGEMM:

- `get_symm_buffer_for_mega_moe`
- `transform_weights_for_mega_moe`
- `fp8_fp4_mega_moe`
- private helpers under `deep_gemm.mega`

However, `sgl-kernel/CMakeLists.txt` currently fetches `https://github.com/sgl-project/DeepGEMM` at commit `35c4bc87713726d048f65275f6f1b551a4e7a6dc`.

Reference: `sgl-kernel/CMakeLists.txt:55`.

In the cloned upstream DeepGEMM repo, `35c4bc...` is an ancestor of `7f2a703...` and predates the MegaMoE merge by two commits. I also checked that `git ls-tree -r 35c4bc...` does not list MegaMoE files such as `csrc/apis/mega.hpp`, `deep_gemm/mega/__init__.py`, or `tests/test_mega_moe.py`.

Impact:

- If SGLang is built against that pinned DeepGEMM commit, the installed `deep_gemm` module will not provide the MegaMoE APIs used by this branch.
- Enabling `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` would likely fail during weight post-processing when importing `transform_weights_for_mega_moe`, or later when calling `get_symm_buffer_for_mega_moe` / `fp8_fp4_mega_moe`.
- To run this integration, the environment needs a DeepGEMM package built from `7f2a703` or another commit/fork containing the MegaMoE release.

## Operational Requirements and Flags

Minimum conditions for the SGLang path to work:

- Blackwell / SM100 GPU, because DeepGEMM MegaMoE C++ only dispatches arch major 10.
- PyTorch version with `torch.distributed._symmetric_memory` support.
- A `deep_gemm` Python package that includes the MegaMoE APIs from the latest release.
- DeepSeek V4 FP4 expert checkpoint path where `self.is_fp4_expert` is true.
- `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`.
- Token count per rank under `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`, unless running inside CUDA graph capture after setup.
- `num_experts % ep_world_size == 0`.
- `hidden % 128 == 0` and `moe_intermediate_size % 128 == 0`.
- `nextn` disabled for this path.
- Hash-MoE disabled unless `SGLANG_OPT_FIX_HASH_MEGA_MOE=1`.

Useful flags:

| Flag | Default | Meaning |
|---|---:|---|
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | `False` | Enables load-time MegaMoE weight build and runtime path selection. |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `1024` | Buffer capacity and non-capture runtime token cap. |
| `SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH` | `True` | Uses SGLang's fused BF16-to-FP8/topk-buffer-fill kernel. |
| `SGLANG_OPT_FIX_HASH_MEGA_MOE` | `False` | Allows MegaMoE for hash-MoE layers. |
| `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` | `False` | Uses shared transformed weight buffers to reduce extra memory. |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `True` | General DeepGEMM availability flag, but not sufficient by itself for MegaMoE. |

## Numerical Compatibility Notes

SGLang's fused pre-dispatch is designed to match DeepGEMM's expected activation format:

- BF16 input is grouped by 32 K elements.
- Scale is `ceil_ue8m0(absmax / FP8_E4M3_MAX)`.
- FP8 output is E4M3.
- Four UE8M0 scales are packed into each int32.
- Top-k indices are widened to int64 because DeepGEMM buffer layout expects int64.

The critical layout invariant is:

```text
num_groups = hidden / 32
buf.x_sf shape = [padded_max, num_groups / 4] as int32
byte address for scale token t, group g = t * num_groups + g
```

SGLang's kernel writes exactly this byte address into `reinterpret_cast<uint8_t*>(buf_x_sf)`, while the TensorMatcher verifies `buf_x_sf` as contiguous `[P, G/4]` int32.

Reference: `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh:89`.

The weight side depends on DeepGEMM's own transforms, so the SGLang-side numerical contract is mostly to call `transform_sf_into_required_layout` with `(1, 32)` before calling `transform_weights_for_mega_moe`.

## End-to-End Flow in SGLang

1. DeepSeek V4 layer creates `DeepseekV2MoE`.
2. FP4 expert weights load as packed tensors and scale tensors.
3. If `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`, `build_mega_moe_experts_weights` prepares `mega_l1_weights` and `mega_l2_weights`.
4. Runtime enters `DeepseekV2MoE.forward`.
5. `_should_use_mega_moe` checks feature flags, weight readiness, token cap, and exclusions.
6. `forward_mega_moe` computes shared expert output separately.
7. `_run_mega_routed` computes router logits and top-k.
8. SGLang gets or creates a DeepGEMM symmetric buffer.
9. SGLang fills DeepGEMM input/routing buffer fields, usually via `mega_moe_pre_dispatch`.
10. SGLang calls `deep_gemm.fp8_fp4_mega_moe`.
11. SGLang applies routed scaling if needed.
12. SGLang adds shared expert output if present.

## Verification Performed

This was source-level research only. I did not run an end-to-end MegaMoE execution because this workspace does not expose the required SM100 GPU/PyTorch symmetric-memory runtime in the current session.

Checks performed:

- Confirmed local DeepGEMM clone is at `7f2a703`.
- Confirmed SGLang branch is `deepseek_v4`.
- Read DeepGEMM public Python API, C++ API checks, buffer layout, scale transform code, test flow, and quantization helpers.
- Read SGLang model wiring, load-time weight build, runtime gate, pre-dispatch JIT wrapper/kernel, and DeepGEMM invocation.
- Checked SGLang's `sgl-kernel` DeepGEMM pin and verified the pinned upstream commit predates MegaMoE files.

## Conclusions

SGLang's branch integrates DeepGEMM MegaMoE as an opt-in DeepSeek V4 routed-expert fast path. The SGLang code aligns with DeepGEMM's latest MegaMoE API and numerical contract:

- FP8 E4M3 activations.
- FP4 E2M1 packed weights.
- UE8M0 packed scale factors.
- K-group quantization size 32.
- `recipe=(1, 1, 32)`.
- Symmetric memory buffer with DeepGEMM's specific `x`, `x_sf`, `topk_idx`, and `topk_weights` layout.
- SM100-only execution.

The main unresolved issue is packaging/build alignment: the SGLang Python code requires a MegaMoE-capable `deep_gemm`, but the current `sgl-kernel` CMake pin points at an older DeepGEMM commit without MegaMoE. This must be updated or overridden before the path can run from a normal SGLang build.

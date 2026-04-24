# DeepGEMM MegaMoE and SGLang Integration Report

Generated: 2026-04-24

## Scope

This report compares the MegaMoE interface in the cloned upstream DeepGEMM repo at `../deepgemm` with the SGLang-side implementation on the `deepseek_v4` branch.

Sources inspected:

- DeepGEMM clone: `/home/yuyue/proj/deepgemm`, `main` at `7f2a703` (`[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes (#304)`).
- SGLang repo: `/home/yuyue/proj/sglang`, branch `deepseek_v4`.
- SGLang day-0 DeepSeek V4 support delta: `0519b09..f5d03db`. `0519b09` is the merge-base/baseline commit and `f5d03db` is `origin/deepseek_v4` / `upstream/deepseek_v4`.
- Key DeepGEMM files: `README.md`, `deep_gemm/mega/__init__.py`, `csrc/apis/mega.hpp`, `csrc/apis/layout.hpp`, `csrc/jit_kernels/heuristics/mega_moe.hpp`, `deep_gemm/utils/math.py`, `tests/test_mega_moe.py`.
- Key SGLang files: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`, `sgl-kernel/CMakeLists.txt`.
- Key MNNVL-related SGLang files checked: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/disaggregation/mooncake/*`, and `docs/advanced_features/pd_disaggregation.md`.

## Executive Summary

DeepGEMM MegaMoE is a fused SM100-oriented MoE kernel that combines EP dispatch, L1 FP8xFP4 GEMM, SwiGLU, L2 FP8xFP4 GEMM, and EP combine into one symmetric-memory communication/computation kernel. Its public API expects the caller to allocate a symmetric buffer, transform FP4 expert weights and UE8M0 scales into a specific layout, fill per-rank input/routing fields in that buffer, and call `deep_gemm.fp8_fp4_mega_moe`.

SGLang's `deepseek_v4` branch integrates this as a routed-expert fast path for DeepSeek V4 FP4 experts. The integration is real and deep: it builds MegaMoE-formatted expert weights at load time, caches DeepGEMM symmetric buffers by process group and shape, fills DeepGEMM's buffer either with a custom fused pre-dispatch JIT kernel or a slower fallback, and invokes `deep_gemm.fp8_fp4_mega_moe` with `recipe=(1, 1, 32)`.

The default is still off. `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` defaults to `False`, and the path is further gated by weight-preparation status, token cap, `nextn`, and hash-MoE constraints.

The `0519b09..f5d03db` comparison shows that MegaMoE was not present in the baseline and is part of the DeepSeek V4 day-0 support delta. The delta adds the DeepSeek V4 model/config/loader, DeepSeek V4 JIT kernels, FP4 expert handling, DeepGEMM-compatible FP4/UE8M0 scale transforms, the MegaMoE pre-dispatch JIT kernel, and the runtime call into `deep_gemm.fp8_fp4_mega_moe`.

The current tree pins `sgl-project/DeepGEMM` at `54f99a8af537b3c6eb4819b69907ccbe2b600792`. The local upstream DeepGEMM clone used for this report is `deepseek-ai/DeepGEMM` at `7f2a703` and does not contain that fork object, so the SGLang code was checked against the public MegaMoE API in `../deepgemm` rather than against the exact pinned fork commit. The operational requirement remains: the installed `deep_gemm` package must include `get_symm_buffer_for_mega_moe`, `transform_weights_for_mega_moe`, and `fp8_fp4_mega_moe`.

MNNVL conclusion: this branch does not add a turnkey or validated MegaMoE MNNVL mode. SGLang's MegaMoE path uses DeepGEMM's PyTorch symmetric-memory buffer over the MoE EP process group, and DeepGEMM's kernel itself performs NVLink-style cross-rank operations. But SGLang does not add MNNVL/IMEX/NVLink-partition configuration, preflight, or feature flags for MegaMoE. Existing MNNVL hooks in DeepEP, FlashInfer, and Mooncake are adjacent paths, not the MegaMoE path.

## Day-0 DeepSeek V4 Support Delta

The requested comparison should be read as `0519b09..f5d03db`, because `0519b09` is the merge-base and `f5d03db` is the `deepseek_v4` support branch tip. The reverse diff, `f5d03db..0519b09`, mostly deletes DeepSeek V4 code.

`git diff --stat --find-renames 0519b09..f5d03db` reports:

- 156 changed files.
- 24,277 insertions.
- 422 deletions.

The broad day-0 DeepSeek V4 support added by that delta includes:

- DeepSeek V4 model/config/loader: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v4_nextn.py`, `python/sglang/srt/configs/deepseek_v4.py`, config backup JSONs, model registry changes, and DeepSeek V4 checkpoint name remapping.
- DeepSeek V4 OpenAI/chat handling: `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`, `python/sglang/srt/function_call/deepseekv4_detector.py`, and serving chat changes.
- DeepSeek V4 attention and memory: compressed attention modules, DeepSeek V4 radix attention backend, NSA/indexer helpers, DeepSeek V4 memory pools, SWA/radix cache adjustments, and model-runner/memory-profiler changes.
- DeepSeek V4 JIT kernels: top-k/hash-top-k, fused norm/rope, RMSNorm, compressed attention helpers, paged MQA metadata, store-cache helpers, HiSparse transfer, BF16xBF16-to-FP32 linear, custom all-reduce v2, and the MegaMoE pre-dispatch kernel.
- DeepSeek V4 MoE and quantization: FP4 expert weight allocation under the FP8 quantization method, DeepSeek-specific MXFP4 method, DeepGEMM MoE runner changes for SwiGLU clamp and FP8/UE8M0 activation scale layout, routed-expert capturer/top-k changes, and the MegaMoE runtime fast path.

MegaMoE-specific direct additions in that delta are:

- `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`, `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`, `SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH`, `SGLANG_OPT_FIX_HASH_MEGA_MOE`, and `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` in `python/sglang/srt/environ.py`.
- `_get_mega_moe_symm_buffer`, `_should_use_mega_moe`, `forward_mega_moe`, `_run_mega_routed`, and the `deep_gemm.fp8_fp4_mega_moe` call in `python/sglang/srt/models/deepseek_v2.py`.
- `build_mega_moe_experts_weights` in `python/sglang/srt/models/deepseek_v4.py`.
- `mega_moe_pre_dispatch` in `python/sglang/jit_kernel/deepseek_v4.py` and `python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`.
- FP4 expert load/post-load hooks in `python/sglang/srt/layers/quantization/fp8.py`, plus a guard in `python/sglang/srt/layers/quantization/mxfp4_deepseek.py` that returns early if MegaMoE weights were already built.

I also checked the baseline directly: `git grep` for `fp8_fp4_mega_moe`, `get_symm_buffer_for_mega_moe`, `transform_weights_for_mega_moe`, `build_mega_moe_experts_weights`, `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`, and `mega_moe_pre_dispatch` finds no matches in `0519b09` and finds the expected matches in `f5d03db`.

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

Reference: `../deepgemm/README.md:114`.

### Required Hardware and Runtime

DeepGEMM's general README lists SM90 or SM100 support for the library, but the MegaMoE C++ API dispatches only when `arch_major == 10`:

- `csrc/apis/mega.hpp` calls `sm100_fp8_fp4_mega_moe(...)` only for `arch_major == 10`.
- The `else` branch is `DG_HOST_UNREACHABLE("Unsupported architecture")`.

Reference: `../deepgemm/csrc/apis/mega.hpp:185`.

MegaMoE also depends on PyTorch symmetric memory:

- `deep_gemm/mega/__init__.py` imports `torch.distributed._symmetric_memory as symm_mem`.
- `SymmBuffer` allocates `symm_mem.empty(...)` and calls `symm_mem.rendezvous(...)`.
- README notes PyTorch >= 2.9 for the symmetric memory buffer.

References: `../deepgemm/deep_gemm/mega/__init__.py:8`, `../deepgemm/deep_gemm/mega/__init__.py:38`, `../deepgemm/README.md:120`.

### Symmetric Buffer Contract

`get_symm_buffer_for_mega_moe` aligns `num_max_tokens_per_rank` to DeepGEMM's MegaMoE `block_m` before creating the buffer:

- `block_m = _C.get_block_m_for_mega_moe(...)`
- `num_max_tokens_per_rank = align(num_max_tokens_per_rank, block_m)`

Reference: `../deepgemm/deep_gemm/mega/__init__.py:58`.

The current heuristic returns a fixed `block_m = 192`.

Reference: `../deepgemm/csrc/jit_kernels/heuristics/mega_moe.hpp:58`.

The C++ buffer layout validates:

- `num_experts % num_ranks == 0`.
- `hidden % 128 == 0`.
- `intermediate_hidden % 128 == 0`.
- Padded SF pool tokens are divisible by 4.

Reference: `../deepgemm/csrc/apis/mega.hpp:14`, `../deepgemm/csrc/apis/mega.hpp:78`.

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

Reference: `../deepgemm/csrc/apis/mega.hpp:82`.

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

Reference: `../deepgemm/csrc/apis/mega.hpp:124`.

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

References: `../deepgemm/deep_gemm/utils/math.py:25`, `../deepgemm/tests/test_mega_moe.py:94`.

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

Reference: `../deepgemm/deep_gemm/utils/math.py:72`, `../deepgemm/deep_gemm/utils/math.py:85`.

The MegaMoE test quantizes both L1 and L2 BF16 reference weights to FP4 with `gran_k=32`, then calls `transform_sf_into_required_layout(..., recipe=(1, 32), num_groups=...)`.

Reference: `../deepgemm/tests/test_mega_moe.py:97`.

#### Intermediate Activations

The fused kernel's logical work matches the DeepGEMM test baseline:

1. EP dispatch receives FP8 input and top-k routing.
2. L1 grouped GEMM runs `FP8 x FP4 -> BF16`, producing gate/up.
3. SwiGLU applies top-k weights and quantizes the activation to FP8 with UE8M0, `num_per_channels=32`.
4. L2 grouped GEMM runs `FP8 x FP4 -> BF16`.
5. EP combine returns BF16 output.

Reference: `../deepgemm/tests/test_mega_moe.py:112`.

### Weight Layout Transform

`transform_weights_for_mega_moe` does two format changes:

- L1: interleave gate and up weights in chunks of 8 rows, then transpose the scale factors for UTCCP.
- L2: leave packed weights as-is, but transpose scale factors for UTCCP.

Reference: `../deepgemm/deep_gemm/mega/__init__.py:77`, `../deepgemm/deep_gemm/mega/__init__.py:98`.

The L1 interleaving changes the first dimension of the output channel axis from `[gate all rows | up all rows]` to `[gate rows 0..7, up rows 0..7, gate rows 8..15, up rows 8..15, ...]`.

The UTCCP scale transpose requires:

- Scale dtype is `torch.int`.
- `mn % 128 == 0`.
- Reshape to `[num_groups, -1, 4, 32, packed_sf_k]`.
- Transpose the `4` and `32` axes.
- Flatten back.

Reference: `../deepgemm/deep_gemm/mega/__init__.py:89`.

### Scale Layout Conversion

`transform_sf_into_required_layout` handles FP32 scale input and converts it to SM100 packed UE8M0:

- For SM100, FP32 scale with `gran_k` 32 or 128 is broadcast if needed, converted to packed UE8M0, TMA-aligned, and MN-major.
- For SM100, already-packed int scale with `gran_mn == 1` and `gran_k` 32 or 128 is validated as TMA-aligned MN-major.

Reference: `../deepgemm/csrc/apis/layout.hpp:47`.

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

### Delta Cross-Check: FP4, MXFP4, and Normal DeepGEMM Interaction

The `0519b09..f5d03db` comparison shows a few MegaMoE-adjacent details that were not obvious from only reading the runtime call site:

- `Fp8Config.get_quant_method` keeps the normal FP8 MoE method, but when `SGLANG_DSV4_MODE=2604`, `SGLANG_DSV4_FP4_EXPERTS=1`, and the MoE runner backend is FlashInfer MXFP4, it wraps the FP8 method in `DeepSeekMxfp4MoEMethod`.
- `Fp8MoEMethod` itself understands the DeepSeek V4 FP4 expert checkpoint layout. It allocates packed int8 expert weights with `hidden_size // 2` / `intermediate_size // 2` K dimensions, and FP32 per-32-K scale tensors.
- During post-load, the FP4 expert path views weights as int8. If MegaMoE is enabled, it builds MegaMoE weights and returns immediately. If MegaMoE is not enabled but the normal DeepGEMM runner will be used, it converts the scale tensors into DeepGEMM's required UE8M0 layout at init time.
- `DeepSeekMxfp4MoEMethod.process_weights_after_loading` first delegates to the FP8 method. Therefore, when MegaMoE has built `mega_l1_weights` / `mega_l2_weights`, the MXFP4 method sees `_mega_moe_weights_built` and skips its own W1/W3 reorder and FlashInfer/TRT-LLM shuffle.
- `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` couples MegaMoE with the normal DeepGEMM MoE runner. In that mode, `DeepGemmRunnerCore` asserts MegaMoE, JIT EP activation, and SwiGLU-clamp fusion are enabled, then uses swizzle-aware activation quantization so the normal path can consume the shared interleaved L1 weight/scale layout.
- With `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=0`, the normal DeepGEMM path intentionally keeps a byte-equal fallback: BF16 `silu_and_mul`, then separate per-token FP8 group quantization.

References:

- `python/sglang/srt/layers/quantization/fp8.py:184`
- `python/sglang/srt/layers/quantization/fp8.py:618`
- `python/sglang/srt/layers/quantization/fp8.py:680`
- `python/sglang/srt/layers/quantization/fp8.py:746`
- `python/sglang/srt/layers/quantization/fp8.py:931`
- `python/sglang/srt/layers/quantization/fp8.py:943`
- `python/sglang/srt/layers/quantization/mxfp4_deepseek.py:223`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:119`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:188`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:218`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:700`

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

- `../deepgemm/csrc/apis/mega.hpp:141`
- `../deepgemm/csrc/apis/mega.hpp:164`
- `../deepgemm/tests/test_mega_moe.py:94`
- `../deepgemm/tests/test_mega_moe.py:104`
- `python/sglang/srt/models/deepseek_v4.py:1997`
- `python/sglang/srt/models/deepseek_v2.py:1230`
- `python/sglang/srt/models/deepseek_v2.py:1256`

## Build and Packaging Risk

The current SGLang source code expects MegaMoE APIs from DeepGEMM:

- `get_symm_buffer_for_mega_moe`
- `transform_weights_for_mega_moe`
- `fp8_fp4_mega_moe`
- private helpers under `deep_gemm.mega`

`sgl-kernel/CMakeLists.txt` currently fetches `https://github.com/sgl-project/DeepGEMM` at commit `54f99a8af537b3c6eb4819b69907ccbe2b600792`. Both `0519b09` and `f5d03db` use this same pin, so the DeepSeek V4 day-0 delta did not change the embedded DeepGEMM revision.

Reference: `sgl-kernel/CMakeLists.txt:55`.

The local DeepGEMM clone used for this report is `deepseek-ai/DeepGEMM` at `7f2a703`. It does not contain object `54f99a8...`, so I could not verify the exact fork pin from the local clone alone. I verified the SGLang integration against the public MegaMoE API in `../deepgemm` at `7f2a703`, which contains the MegaMoE files and APIs SGLang imports.

Impact:

- If the installed/pinned `deep_gemm` package lacks MegaMoE, enabling `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` will fail during weight post-processing when importing `transform_weights_for_mega_moe`, or later when calling `get_symm_buffer_for_mega_moe` / `fp8_fp4_mega_moe`.
- To run this integration, the environment needs a DeepGEMM package built from `7f2a703` or an equivalent `sgl-project/DeepGEMM` fork commit that contains MegaMoE.

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

## MNNVL / NVL72 Assessment

### Source-Level Conclusion

The `deepseek_v4` MegaMoE integration should be described as **MNNVL-capable by dependency, not MNNVL-supported as a SGLang feature**.

Evidence from the `0519b09..f5d03db` comparison:

- The day-0 delta does not add any MegaMoE-specific `MNNVL`, `mnnvl`, `IMEX`, `imex`, `MC_FORCE_MNNVL`, or NVLink-partition logic.
- `git diff -G'MNNVL|mnnvl|NVLink|nvlink|IMEX|imex|allow_mnnvl|MC_FORCE_MNNVL' --name-only 0519b09..f5d03db` returns no day-0 MNNVL changes.
- Existing MNNVL hooks in SGLang are present in both `0519b09` and `f5d03db`: DeepEP passes `allow_mnnvl=True`, FlashInfer creates `MnnvlConfig`, and Mooncake PD documents `MC_FORCE_MNNVL=True`. Those paths are not the MegaMoE path.
- The direct MegaMoE files do not reference MNNVL or IMEX. The MegaMoE path gets `get_moe_ep_group().device_group`, allocates a DeepGEMM symmetric buffer, fills it, and calls `deep_gemm.fp8_fp4_mega_moe`.

References:

- `python/sglang/srt/models/deepseek_v2.py:395`
- `python/sglang/srt/models/deepseek_v2.py:1187`
- `python/sglang/srt/models/deepseek_v2.py:1251`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:228`
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py:30`
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py:146`
- `docs/advanced_features/pd_disaggregation.md:129`

### What DeepGEMM Provides

DeepGEMM MegaMoE itself is the component that performs cross-rank NVLink-style communication:

- `deep_gemm/mega/__init__.py` allocates `torch.distributed._symmetric_memory` and calls `symm_mem.rendezvous(...)` on the provided process group.
- DeepGEMM's `SymBuffer` stores remote pointer offsets for up to 72 ranks, matching the GB200 NVL72 scale target.
- DeepGEMM's communication helper implements a cross-rank `nvlink_barrier` by mapping a local pointer into each remote rank's symmetric buffer and issuing system-scope remote atomics.
- The MegaMoE kernel uses those mapped pointers for dispatch pull and combine write-back.

References:

- `../deepgemm/deep_gemm/mega/__init__.py:8`
- `../deepgemm/deep_gemm/mega/__init__.py:38`
- `../deepgemm/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh:7`
- `../deepgemm/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh:34`
- `../deepgemm/deep_gemm/include/deep_gemm/comm/barrier.cuh:29`
- `../deepgemm/deep_gemm/include/deep_gemm/comm/barrier.cuh:48`

### What SGLang Does Not Provide

SGLang's MegaMoE integration does not currently:

- Select or force an MNNVL transport mode.
- Check that EP ranks are inside one NVLink partition.
- Check that `nvidia-imex` is running on every node.
- Check `nvidia-imex-ctl -N` domain health.
- Check that all participating nodes expose the same intended IMEX channel.
- Check fabric registration or NVLink link state before enabling MegaMoE.
- Fall back from MegaMoE to a non-MNNVL path if PyTorch symmetric-memory rendezvous succeeds locally but remote NVLink access later fails.

This is why the branch can be run on an already-correct GB200 NVL72/MNNVL setup, but it does not itself constitute validated MNNVL support.

### Operational Envelope for GB200 NVL72

For a single-tenant full NVL72 rack, the safe MegaMoE target is narrower than "generic multi-node":

- All MegaMoE EP ranks should be inside one NVLink partition.
- All participating compute nodes should be in one healthy IMEX domain.
- The job user should have access to the intended IMEX channel on every node; for a simple single-user deployment this is usually `channel0`.
- The rack can usually use the default whole-rack NVLink partition if no administrator-created user partitions are in play.

Practical preflight checks before calling this "supported" should include:

- `nv show sdn partition` and `nv show sdn partition 32766` on the NVSwitch control plane for the default partition case.
- `systemctl status nvidia-imex` and `/usr/bin/nvidia-imex --version` on every compute node.
- `nvidia-imex-ctl -N` for IMEX domain health.
- `cat /etc/nvidia-imex/nodes_config.cfg` to verify a consistent peer set.
- `ls -l /dev/nvidia-caps-imex-channels` to verify channel visibility.
- `nvidia-smi -q | grep Fabric -A 4` and `nvidia-smi nvlink --status` to verify fabric registration and active links.

Useful NVIDIA references:

- NVIDIA IMEX overview: `https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html`
- NVIDIA IMEX deployment models: `https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/deployment.html`
- NVIDIA IMEX channels: `https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html`
- NVIDIA MNNVL verification guide: `https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/verifying.html`
- NVIDIA Mission Control NVLink partition management: `https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.3.0/nvlink-partition-management.html`

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
- Compared `0519b09..f5d03db` and treated that additive diff as the DeepSeek V4 day-0 support delta.
- Confirmed MegaMoE symbols are absent from `0519b09` and present in `f5d03db`.
- Read DeepGEMM public Python API, C++ API checks, buffer layout, scale transform code, test flow, and quantization helpers.
- Read SGLang model wiring, load-time weight build, runtime gate, pre-dispatch JIT wrapper/kernel, and DeepGEMM invocation.
- Cross-checked the day-0 FP4/MXFP4/normal-DeepGEMM delta against the original report and added the missing interaction details.
- Checked SGLang's `sgl-kernel` DeepGEMM pin. The current pin is `sgl-project/DeepGEMM@54f99a8...`, which is not present in the local `deepseek-ai/DeepGEMM` clone, so exact fork-pin contents were not verified from local source.
- Searched the day-0 delta and direct MegaMoE files for MNNVL/IMEX-specific handling. No MegaMoE-specific MNNVL support code was found.

## Conclusions

SGLang's branch integrates DeepGEMM MegaMoE as an opt-in DeepSeek V4 routed-expert fast path. The SGLang code aligns with DeepGEMM's latest MegaMoE API and numerical contract:

- FP8 E4M3 activations.
- FP4 E2M1 packed weights.
- UE8M0 packed scale factors.
- K-group quantization size 32.
- `recipe=(1, 1, 32)`.
- Symmetric memory buffer with DeepGEMM's specific `x`, `x_sf`, `topk_idx`, and `topk_weights` layout.
- SM100-only execution.

The main build requirement is still packaging alignment: the SGLang Python code requires a MegaMoE-capable `deep_gemm`. The current `sgl-kernel` pin is a `sgl-project/DeepGEMM` fork commit (`54f99a8...`) that was not present in the local upstream clone used for this report, so the installed package must be checked or overridden to ensure it exposes the MegaMoE APIs.

For GB200 NVL72/MNNVL, this branch is not enough by itself to claim supported MegaMoE MNNVL. It can rely on DeepGEMM/PyTorch symmetric memory in a correctly configured MNNVL environment, but SGLang does not yet validate the NVLink partition, IMEX domain, IMEX channel, fabric health, or EP rank placement for MegaMoE.

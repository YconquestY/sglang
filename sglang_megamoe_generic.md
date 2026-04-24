# Generic DeepGEMM MegaMoE Backend Plan

## Current conclusion

The `deepseek_v4` branch does not currently provide a generic MegaMoE runner backend.
It has a DeepSeek-V4-specific path that is guarded by
`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` and implemented through model code in
`python/sglang/srt/models/deepseek_v2.py` plus DeepSeek-V4 weight conversion in
`python/sglang/srt/models/deepseek_v4.py`.

To support GLM-5.1, MiniMax M2.7, and Kimi K2.6 in the same style as
DeepSeek-V4, MegaMoE should be promoted into a first-class MoE runner backend:

```text
--moe-runner-backend deepgemm_mega_moe
--moe-a2a-backend none
```

`moe_a2a_backend=none` here does not mean single-rank only. It means SGLang
must not run its normal token dispatcher A2A path before the runner. DeepGEMM
MegaMoE owns its own EP communication through the symmetric-memory group.

The target integration should make the model forward path identical for
DeepSeek-V4, GLM, MiniMax, and Kimi:

```text
hidden_states
  -> model gate
  -> SGLang TopK
  -> FusedMoE / StandardDispatcher
  -> DeepGEMM MegaMoE fused runner
  -> optional model shared experts
  -> optional TP all-reduce
```

## Existing DeepSeek-V4 implementation

The current DeepSeek-V4 MegaMoE path has three parts:

1. A runtime detour in `DeepseekV2MoE.forward_mega_moe` and
   `DeepseekV2MoE._run_mega_routed`.
2. A DeepSeek-specific FP4 weight transform in
   `deepseek_v4.build_mega_moe_experts_weights`.
3. A quantization hook in `Fp8MoEMethod.process_weights_after_loading` that
   calls the DeepSeek-specific weight builder when the DeepSeek FP4 expert env
   flags are enabled.

The runtime already does the right high-level work:

- Run the model router and SGLang TopK.
- Preserve global expert ids.
- Build or reuse a DeepGEMM symmetric buffer from `get_moe_ep_group()`.
- Quantize activations to FP8.
- Call `deep_gemm.fp8_fp4_mega_moe(...)`.
- Apply `routed_scaling_factor` after the kernel if TopK did not fuse it.

The problem is placement. These pieces live under DeepSeek model files and are
triggered by env flags instead of the normal `moe_runner_backend` interface.

## Generic MoE integration point

The right integration point is the generic MoE stack:

- `python/sglang/srt/layers/moe/utils.py`
  - add a new `MoeRunnerBackend` value, for example
    `DEEPGEMM_MEGA_MOE = "deepgemm_mega_moe"`.
- `python/sglang/srt/server_args.py`
  - add `deepgemm_mega_moe` to `MOE_RUNNER_BACKEND_CHOICES`.
  - keep auto-selection conservative at first. Prefer explicit opt-in until
    all target model and checkpoint combinations are validated.
- `python/sglang/srt/layers/moe/moe_runner/runner.py`
  - treat MegaMoE as a fused runner function, like the FlashInfer fused paths.
  - no normal `runner_core` is needed initially.
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - continue using `FusedMoE` for model integration.
  - do not route MegaMoE through `DeepEPMoE`.
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
  - keep `topk_ids` as global expert ids for this backend.
  - skip the usual EP-local expert id remapping when
    `moe_runner_backend == deepgemm_mega_moe`.

The backend should reject incompatible settings early:

- `moe_a2a_backend` must be `none`.
- `moe_tp_size` should be `1` for the first implementation.
- `ep_num_redundant_experts` should be `0` for the first implementation.
- `num_experts % moe_ep_size == 0`.
- GPU capability must be SM100 or another explicitly validated DeepGEMM
  MegaMoE target.
- DeepGEMM MegaMoE Python APIs must be importable.
- PyTorch symmetric memory and the EP process group must be initialized.

## Proposed backend interface

Add a new module:

```text
python/sglang/srt/layers/moe/moe_runner/deepgemm_mega_moe.py
```

This module should own the runtime path that is currently embedded in
`DeepseekV2MoE._run_mega_routed`.

Proposed objects:

```python
@dataclass
class DeepGemmMegaMoeWeightBundle:
    mega_l1_weights: torch.Tensor
    mega_l2_weights: torch.Tensor
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation: str
    activation_clamp: float | None
    routed_scaling_factor: float | None


@dataclass
class DeepGemmMegaMoeRuntimeConfig:
    num_max_tokens_per_rank: int
    recipe: tuple[int, int, int] = (1, 1, 32)
```

Register a fused runner:

```python
@register_fused_func(
    a2a_backend=MoeA2ABackend.NONE,
    runner_backend=MoeRunnerBackend.DEEPGEMM_MEGA_MOE,
)
def deepgemm_mega_moe_fused_func(
    layer: FusedMoE,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    ...
```

The fused function should:

1. Read `hidden_states`, optional activation scales, and `topk_output` from
   `StandardDispatchOutput`.
2. Require `TopKOutput.format == STANDARD`.
3. Preserve global `topk_ids`.
4. Build or reuse a symmetric buffer for `get_moe_ep_group().device_group`.
5. Run the generic pre-dispatch quantization path.
6. Call `deep_gemm.fp8_fp4_mega_moe`.
7. Apply `routed_scaling_factor` if it was not fused into TopK.
8. Return a `StandardCombineInput` that the existing `StandardDispatcher`
   combine path can pass through.

This keeps the public model interface the same as every other FusedMoE backend:

```python
final_hidden_states = self.experts(hidden_states, topk_output)
```

## Generic weight conversion

Add a generic weight conversion helper instead of keeping the builder in
`deepseek_v4.py`:

```text
python/sglang/srt/layers/moe/moe_runner/deepgemm_mega_moe_weights.py
```

Responsibilities:

- Convert model-local MoE weights into DeepGEMM MegaMoE L1/L2 layouts.
- Normalize scale formats before calling DeepGEMM layout transforms.
- Store the resulting `DeepGemmMegaMoeWeightBundle` on the `FusedMoE` layer.
- Mark the source FP4 weights as transformed without mutating them for another
  backend.

Suggested helpers:

```python
def supports_deepgemm_mega_moe(layer: FusedMoE) -> tuple[bool, str | None]:
    ...


def build_deepgemm_mega_moe_weights(
    layer: FusedMoE,
    *,
    scale_format: Literal["deepseek_fp32", "mxfp4_e8m0"],
    gate_up_order: Literal["gate_up", "up_gate"] = "gate_up",
) -> DeepGemmMegaMoeWeightBundle:
    ...
```

DeepGEMM currently expects the MegaMoE weight transform flow used by
`../deepgemm/deep_gemm/mega/__init__.py`:

- `transform_sf_into_required_layout(...)`
- optional `_interleave_l1_weights(...)`
- optional `_transpose_sf_for_utccp(...)`
- `transform_weights_for_mega_moe(...)`
- `fp8_fp4_mega_moe(...)`

The generic SGLang helper should wrap these APIs once and hide model-specific
checkpoint details.

### DeepSeek-V4 FP4 adapter

DeepSeek-V4 already has the closest format:

- `w13_weight`
- `w13_weight_scale_inv`
- `w2_weight`
- `w2_weight_scale_inv`

The generic helper can lift the existing implementation nearly as-is from
`deepseek_v4.build_mega_moe_experts_weights`.

After the refactor, DeepSeek-V4 should no longer need a model-local MegaMoE
runtime path. Its existing env flag can remain as a compatibility alias that
sets or validates:

```text
moe_runner_backend = deepgemm_mega_moe
moe_a2a_backend = none
```

### True MXFP4 adapter

The generic `Mxfp4MoEMethod` stores weights and scales as:

- `w13_weight`: packed uint8 FP4
- `w13_weight_scale`: uint8 E8M0 scales
- `w2_weight`: packed uint8 FP4
- `w2_weight_scale`: uint8 E8M0 scales

This is the main path needed for GLM-5.1, MiniMax M2.7, and Kimi K2.6 if their
checkpoints use SGLang's static MXFP4 quantization method.

The weight builder should run before any backend-specific mutation in
`Mxfp4MoEMethod.process_weights_after_loading`. For
`deepgemm_mega_moe`, it should build the MegaMoE bundle and return before:

- FlashInfer swaps `w1` and `w3`.
- FlashInfer shuffles weights and scales into TRT-LLM layout.
- Triton fallback upcasts weights to BF16 and deletes FP4 tensors.

For scale conversion, start conservative:

1. Convert uint8 E8M0 scale bytes to FP32 scale tensors.
2. Pass those FP32 tensors through
   `transform_sf_into_required_layout(..., disable_ue8m0_cast=False)`.
3. Only switch to a direct packed-int scale path after exact layout tests prove
   it matches DeepGEMM expectations.

### ModelOpt and compressed NVFP4

Do not treat ModelOpt/Compressed NVFP4 as the same thing as true MXFP4.

`ModelOptNvFp4FusedMoEMethod` and
`CompressedTensorsW4A4Nvfp4MoEMethod` use packed FP4 weights, E4M3 scale
tensors, and extra per-tensor/global scale factors. DeepGEMM MegaMoE's current
path expects the DeepSeek/MXFP4-style scale recipe.

Support these later only after one of these is implemented and validated:

- a numerically exact conversion from ModelOpt/Compressed NVFP4 scales into the
  scale recipe expected by DeepGEMM MegaMoE; or
- a DeepGEMM MegaMoE kernel/API variant that directly accepts those scale
  semantics.

## Model-by-model plan

### DeepSeek-V4

DeepSeek-V4 should become the reference user of the generic backend.

Required changes:

- Move `build_mega_moe_experts_weights` out of `deepseek_v4.py` into the
  generic weight helper.
- Move `_run_mega_routed`, symmetric-buffer caching, and pre-dispatch logic out
  of `DeepseekV2MoE` into the new runner module.
- Keep shared experts outside MegaMoE unless they are quantized with the same
  supported FP4 format and pass all shape checks.
- Preserve the existing behavior that disables fused shared experts for
  DeepSeek-V4 FP4 routed experts when shared experts remain FP8.

Expected end state:

```python
final_hidden_states = self.experts(hidden_states, topk_output)
```

No DeepSeek-only MegaMoE forward branch should be needed.

### GLM-5.1

The GLM MoE family is already close to the target interface.
`Glm4MoeSparseMoeBlock` constructs experts through `get_moe_impl_class` and
calls:

```python
final_hidden_states = self.experts(hidden_states, topk_output)
```

It also passes `routing_method_type=RoutingMethodType.DeepSeekV3`, uses grouped
TopK, and can keep shared experts separate from the routed expert backend.

Required changes:

- No model-specific MegaMoE forward branch should be added.
- Ensure the GLM-5.1 checkpoint path resolves to `Mxfp4MoEMethod` for static
  MXFP4 experts.
- Let the generic `Mxfp4MoEMethod` MegaMoE hook build the weight bundle.
- Keep shared experts separate for the first implementation.
- Keep fused shared experts disabled under EP unless the shared expert is also
  in a supported FP4 format and the total expert count passes DeepGEMM shape
  checks.

Potential issue:

- If GLM-5.1 uses a model class other than `Glm4MoeSparseMoeBlock`, add only the
  missing metadata needed to call the same generic `FusedMoE` path.

### MiniMax M2.7

`MiniMaxM2MoE` is also close to the target interface. It constructs experts via
`get_moe_impl_class`, has no separate shared experts in the MoE block, computes
TopK through SGLang, and calls:

```python
final_hidden_states = self.experts(hidden_states, topk_output)
```

Required changes:

- No model-specific MegaMoE forward branch should be added.
- Validate that the M2.7 MXFP4 checkpoint uses `Mxfp4MoEMethod`.
- Let the generic MXFP4 adapter build the MegaMoE weight bundle.
- Preserve its existing router settings, including routing bias and
  renormalization, because MegaMoE should consume the already-computed
  `topk_ids` and `topk_weights`.

Potential issue:

- MiniMax uses `routed_scaling_factor=1.0`, so it is simpler than DeepSeek and
  GLM. Still, tests should cover routing bias because TopK must produce the same
  expert ids and weights as the existing backend.

### Kimi K2.6

`KimiMoE` also uses `get_moe_impl_class` and calls the generic expert layer.
It keeps shared experts separate and computes grouped TopK with correction bias.

Required changes:

- No model-specific MegaMoE forward branch should be added.
- Validate that Kimi K2.6 MXFP4 checkpoints use `Mxfp4MoEMethod`.
- Let the generic MXFP4 adapter build the MegaMoE weight bundle.
- Add explicit routing metadata to the `FusedMoE` constructor if needed. Today
  `KimiMoE` does not pass `routing_method_type`; for consistency with GLM and
  DeepSeek, pass `RoutingMethodType.DeepSeekV3` or add a Kimi-specific routing
  enum if its grouped-topk semantics differ.
- Keep shared experts outside MegaMoE for the first implementation.

Potential issue:

- Kimi's shared expert overlap and CUDA graph behavior should be tested with
  the fused MegaMoE runner. The shared expert itself should not be pulled into
  MegaMoE until quantization and divisibility constraints are proven.

## Dispatcher and TopK rules

MegaMoE should use SGLang's normal TopK implementation and consume
`TopKOutput` in `STANDARD` format.

Rules:

- Do not use FlashInfer-specific TopK output formats.
- Do not run DeepEP or Mooncake token dispatch before MegaMoE.
- Do not map global expert ids to local expert ids in `StandardDispatcher`.
- Do not fuse `routed_scaling_factor` into TopK unless the backend explicitly
  supports and tests that behavior.

This matches the existing DeepSeek-V4 path, which passes global expert ids into
`deep_gemm.fp8_fp4_mega_moe`.

## CLI and selection policy

Initial explicit opt-in:

```text
--moe-runner-backend deepgemm_mega_moe
--moe-a2a-backend none
```

The backend should fail fast with clear errors for unsupported quantization,
unsupported model shapes, or incompatible parallel settings.

Auto-selection should come later and should be allowlisted by:

- model architecture;
- quantization method;
- GPU capability;
- DeepGEMM version/API availability;
- EP configuration;
- successful validation coverage.

The legacy DeepSeek env flag can remain temporarily:

```text
SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
```

But it should become an alias for the same backend path, not a separate model
runtime.

## MNNVL / NVL72 implications

This plan does not require model-specific MNNVL code. MNNVL support belongs in
the DeepGEMM symmetric-memory runtime and process-group setup.

SGLang should do three things:

1. Pass the correct EP device group to DeepGEMM MegaMoE.
2. Avoid SGLang's own external MoE A2A dispatcher for this backend.
3. Add a startup preflight that fails clearly if symmetric memory cannot
   rendezvous across the configured EP group.

Assuming MNNVL and IMEX are correctly configured, the same
`deepgemm_mega_moe` backend should be the interface for one NVL72 or multiple
NVL72 compute trays. The remaining work is validation and preflight, not
different model integration.

## Validation plan

Unit-level tests:

- Compare DeepSeek-V4 outputs from the existing model-local MegaMoE weight
  builder against the new generic builder.
- Test MXFP4 E8M0 scale conversion with synthetic tensors.
- Verify that `StandardDispatcher` preserves global expert ids for
  `deepgemm_mega_moe`.
- Verify fail-fast errors for unsupported `moe_a2a_backend`, `moe_tp_size`,
  redundant experts, and unsupported quantization methods.

Runtime correctness tests:

- DeepSeek-V4 FP4: compare old MegaMoE path and generic backend.
- GLM-5.1 MXFP4: compare `deepgemm_mega_moe` against the best existing MXFP4
  backend on fixed prompts.
- MiniMax M2.7 MXFP4: same comparison, with routing bias enabled if applicable.
- Kimi K2.6 MXFP4: same comparison, including shared expert paths.
- Zero-token and padded-batch cases.
- CUDA graph capture with shared expert overlap for Kimi.

Distributed tests:

- EP size 2, 4, and 8 on one NVL72.
- Multi-NVL72 EP group after MNNVL/IMEX preflight passes.
- Token counts below, equal to, and above
  `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`.
- Failure test when symmetric-memory rendezvous is unavailable.

Performance tests:

- Compare against FlashInfer MXFP4 or the current best backend per model.
- Track pre-dispatch quantization time, kernel time, combine time, and memory
  overhead.
- Report separate numbers for single NVL72 and multi-NVL72 EP groups.

## Rollout phases

Phase 1: backend skeleton and DeepSeek refactor.

- Add `deepgemm_mega_moe` enum and CLI value.
- Add fused runner module.
- Move DeepSeek runtime and weight transform into generic modules.
- Keep behavior equivalent for DeepSeek-V4.

Phase 2: true MXFP4 adapter.

- Add `Mxfp4MoEMethod` support for `deepgemm_mega_moe`.
- Build MegaMoE weights before FlashInfer/Triton mutations.
- Enable explicit opt-in for GLM-5.1, MiniMax M2.7, and Kimi K2.6.

Phase 3: model validation and small metadata fixes.

- Add or correct routing metadata for Kimi if needed.
- Confirm GLM-5.1 model class and checkpoint quantization path.
- Confirm MiniMax M2.7 checkpoint quantization path.
- Add allowlist-based auto-selection only after correctness tests pass.

Phase 4: optional NVFP4 adapters.

- Investigate ModelOpt and compressed NVFP4 scale semantics.
- Add support only if conversion into DeepGEMM's expected scale recipe is exact
  enough for production inference, or DeepGEMM exposes a matching API.

Phase 5: MNNVL validation.

- Add symmetric-memory preflight.
- Validate multi-NVL72 EP groups.
- Document required launch and environment settings.

## Short answer for the three target models

GLM-5.1, MiniMax M2.7, and Kimi K2.6 should not need bespoke MegaMoE forward
paths if their routed experts are loaded through SGLang's static
`Mxfp4MoEMethod`.

The common work is:

- make MegaMoE a real `moe_runner_backend`;
- preserve global expert ids through the dispatcher;
- add a generic MXFP4-to-DeepGEMM weight adapter;
- keep shared experts outside MegaMoE initially;
- validate each model's router and checkpoint metadata.

After that, the model-facing interface can be the same as DeepSeek-V4:

```text
FusedMoE + TopKOutput(STANDARD) + deepgemm_mega_moe
```

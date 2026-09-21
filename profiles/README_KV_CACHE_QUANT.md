# `kv-cache-quant` profile for `vllm-position-parity` (prototype v0.1)

This is a **small analysis profile**, not a new harness and not a new evidence schema.
It reuses the existing `vllm-position-parity` canonical JSON files and `analyze.py`.

Drop:

```text
profiles/kv_cache_quant.py
```

into the existing VPP repository.

## Why this profile exists

KV-cache quantization should not be judged by bit-exact BF16-vs-quantized equality.
The useful questions are separated instead:

1. **Self-repeat stability** — is each arm reproducible on its own?
2. **Cross-precision drift** — how much do teacher-forced prompt logprobs and Top-1 predictions move?
3. **Completion trajectory** — where do modal generated token sequences first diverge?
4. **Cache observations** — were cached/cache-creation token counters present and what did they report?
5. **Environment comparability** — were hardware/runtime/parallelism variables held constant apart from KV dtype?
6. **Optional quality gates** — a project may turn measurements into CI PASS/FAIL using an explicit policy file.

The profile keeps the VPP principle: **measure first; classification is an optional downstream policy**.

## Suggested matrix

Collect matched reference/candidate evidence for selected cohorts rather than blindly tracing every possible long context:

```text
                       reference KV       quantized KV
short / prefix miss    R_short_miss       Q_short_miss
boundary / prefix miss R_bound_miss       Q_bound_miss
boundary / prefix hit  R_bound_hit        Q_bound_hit
long / selected stress R_long             Q_long
```

Repeat under eager vs graph-configured execution only when that axis matters to the backend under test.

The profile accepts any number of matched pairs:

```bash
python profiles/kv_cache_quant.py \
  --pair short_miss runs/bf16_short.json runs/fp8_short.json \
  --pair boundary_miss runs/bf16_boundary.json runs/fp8_boundary.json \
  --pair boundary_hit runs/bf16_boundary_hit.json runs/fp8_boundary_hit.json \
  --out runs/kv_quant
```

Outputs:

```text
runs/kv_quant/report.json
runs/kv_quant/summary.csv
```

Without a gate policy, `overall_gate_status` is `NOT_EVALUATED` and the process exits 0.

## Project-specific quality gates

For CI, pass a policy explicitly:

```bash
python profiles/kv_cache_quant.py \
  --pair short_miss runs/bf16_short.json runs/fp8_short.json \
  --drift-epsilon 0.05 \
  --quality-gates profiles/kv_cache_quant_gates.example.json \
  --out runs/kv_quant_ci
```

Example policy:

```json
{
  "policy_name": "EXAMPLE_ONLY_replace_with_project_calibrated_policy",
  "reference_repeat": {"min_top1_agreement_rate": 1.0},
  "candidate_repeat": {"min_top1_agreement_rate": 1.0},
  "cross_precision": {
    "min_modal_top1_agreement_rate": 0.99,
    "max_p99_abs_logprob_delta": 0.25,
    "drift_fraction": {"epsilon": 0.05, "max_fraction": 0.05}
  }
}
```

**These numbers are examples, not recommended universal thresholds.**
A real CI policy should be calibrated from the model + quantization format + attention backend + GPU architecture + execution mode being protected.

Gate exit semantics:

```text
PASS / NOT_EVALUATED / INCOMPLETE -> exit 0
FAIL                              -> exit 1
```

This makes the profile CI-usable without turning VPP itself into a universal quantization judge.

## Long-context cost

VPP v0.1 stores position-resolved `prompt_logprobs`, so asking vLLM for a 32K/64K prompt can be materially more expensive than analyzing the resulting JSON.

The v0.1 profile therefore does **not** pretend that post-capture downsampling saves collection cost. Instead it recommends selected cohorts around known boundaries:

```text
short baseline
just below boundary
just above boundary
one long stress point
```

Examples of boundaries worth choosing from the implementation under test include chunked-prefill sizes, sparse-attention/indexer budgets, block-size transitions, or backend-specific dispatch thresholds.

The profile records:

- prompt token count
- evidence JSON byte size
- analyzer runtime

Existing VPP v0.1 evidence does not record collector wall-clock time. This bundle therefore includes `OPTIONAL_collect_vllm_timing.patch`, a deliberately small optional patch that records engine-init, generation, normalization, and total pre-write wall time. The profile surfaces `collection_timing` when present. These timings are diagnostic overhead telemetry, not throughput benchmarks.

## Hardware/runtime telemetry

The profile surfaces the VPP evidence metadata and warns when matched arms differ in variables that normally should stay fixed:

- GPU name / compute capability / count
- CUDA runtime
- torch version
- vLLM version / commit
- TP / PP / DP
- weight/model quantization
- prefix-caching mode
- execution mode
- `max_num_batched_tokens`

`kv_cache_dtype` is intentionally treated as the primary comparison variable.

Unknown telemetry stays unknown; the profile does not infer missing hardware/backend facts.

For backend-specific data that VPP v0.1 cannot introspect reliably (for example an attention backend selected deep in runtime), attach it explicitly with existing VPP metadata:

```bash
--metadata attention_backend=FLASHINFER \
--metadata quant_scheme=fp8_e4m3 \
--metadata cohort=boundary_miss
```

## Metrics emitted

Per matched pair:

```text
reference_repeat.min_top1_agreement_rate
candidate_repeat.min_top1_agreement_rate
reference/candidate max & p99 forced-logprob self-spread
cross_precision.cross_modal_top1_agreement_rate
cross_precision mean/p50/p95/p99/max abs forced-logprob delta
optional fraction above a user-supplied drift epsilon
modal completion exact match / first token divergence
cache token observations
comparability warnings
hardware/runtime telemetry
optional quality-gate status
```

Completion identity is diagnostic across precision; once token IDs diverge, later generated-token differences are downstream evidence and should not be over-interpreted.

## Evidence discipline

The profile does not claim:

- BF16 and quantized KV must be bit-identical
- a universal logprob tolerance exists
- a token flip identifies root cause
- a synthetic fixture is implementation evidence

It is intended to provide a reusable **runtime evidence layer** for FP8/TurboQuant/KVarN/etc. experiments while preserving the existing VPP measurement philosophy.

# vllm-position-parity

Position-resolved behavioral regression measurement for vLLM `prompt_logprobs`.

> **native vLLM evidence → canonical JSON schema → position-resolved measurement report**

When debugging determinism, quantization, backend changes, or scheduling effects, end-to-end completion hashes and aggregate quality metrics can tell you that behavior changed without showing where along the prompt the difference first became observable.

`vllm-position-parity` adds that missing position dimension.

It is a small measurement tool built around native vLLM `prompt_logprobs`. It compares repeated runs, execution configurations, and scheduling modes while preserving the captured runtime metadata needed to interpret the result.

It measures what changed.

It does **not** label bugs, infer root causes, apply universal significance thresholds, or recommend optimizations.

> **Measure, don't classify.**

## What v0.1 measures

For repeated teacher-forced `prompt_logprobs` runs, it reports:

- Per-position forced-token logprob spread
- Per-position top-1 agreement rate
- Per-position pairwise top-k token-set Jaccard overlap
- First observed top-1 disagreement position
- First observed non-zero forced-token logprob spread position
- Cross-arm mean forced-token logprob delta
- Sequential vs. concurrent / batch-shape comparison

The analyzer has **no vLLM dependency**. `collect_vllm.py` is the thin adapter that normalizes native vLLM output into the canonical JSON schema used by `analyze.py`.

## Non-goals

v0.1 explicitly does not:

- Label any result as a bug or failure
- Infer root cause
- Define a universal "significant divergence" threshold
- Recommend quantization or layer settings
- Replace vLLM's existing correctness tests
- Claim that the first observed difference is causal

A position-level difference can be real while still belonging to a different execution path or subsystem than the one currently being investigated.

## Design principles

1. **Measure, don't classify.** Observations are reported; interpretation remains external.
2. **Backend-agnostic analyzer.** Core analysis consumes only canonical JSON.
3. **Metadata-first.** Reports preserve the captured execution context needed for reproducible comparisons.
4. **Privacy-conscious.** Raw prompt text is not stored by default. Token-level evidence may still contain sensitive information and should be handled accordingly.

## Three measurement axes

| Axis | Question | Typical comparison |
| --- | --- | --- |
| **Repeat Stability** | Is the same configuration reproducible run-to-run? | `main` × 8 repeats |
| **Cross-Arm Parity** | How do two configurations differ across prompt positions? | `BF16` vs. `FP8`, or `main` vs. `patch` |
| **Schedule Sensitivity** | Does batching or concurrency alter observed behavior? | `sequential` vs. `concurrent` |

These axes are reported separately rather than collapsed into one generic divergence score.

## Quick start

### 1. Run the synthetic fixtures — no vLLM required

```bash
python analyze.py fixtures/exact.json --out runs/exact
python analyze.py fixtures/noisy.json --out runs/noisy
python analyze.py fixtures/exact.json fixtures/injected_flip.json --out runs/compare
python analyze.py fixtures/exact.json fixtures/concurrent_only.json --out runs/schedule
```

The fixtures are synthetic mock data used only to verify analysis and reporting behavior. They are not benchmark results.

### 2. Collect a real vLLM run

```bash
python collect_vllm.py \
  --model /path/to/model \
  --prompt-file prompt.txt \
  --out runs/main_seq.json \
  --arm main \
  --execution-mode sequential \
  --repeats 8 \
  --prompt-logprobs 5 \
  --max-tokens 64 \
  --temperature 0 \
  --disable-prefix-caching
```

`sequential` performs one `generate()` call per repeat.

To exercise a batched/concurrent arm with the same prompt repeated in one offline `generate()` call:

```bash
python collect_vllm.py \
  --model /path/to/model \
  --prompt-file prompt.txt \
  --out runs/main_concurrent.json \
  --arm main_concurrent \
  --execution-mode concurrent \
  --repeats 8 \
  --prompt-logprobs 5 \
  --max-tokens 64 \
  --temperature 0 \
  --disable-prefix-caching
```

This is a useful batch-composition axis for regression experiments; it is not intended as a universal model of production concurrency.

### 3. Compare two runs

```bash
python analyze.py runs/reference.json runs/candidate.json --out runs/parity
```

Outputs:

```text
runs/parity/
├── report.json                 # Machine-readable measurements
├── key_positions.csv          # Positions with the largest observed differences
└── position_measurements.png  # Position-resolved logprob measurements
```

The JSON and CSV outputs use only the Python standard library. PNG generation uses `matplotlib` when available.

## Canonical JSON contract

The collector does not dump vLLM internal objects verbatim. It normalizes them into a small canonical evidence schema:

```text
vLLM native prompt_logprobs
          ↓
      thin adapter
          ↓
 canonical JSON evidence
          ↓
 backend-independent analyzer
```

The run record keeps explicit measurement metadata such as:

- arm / execution mode / repeat count
- model
- temperature / max tokens
- `prompt_logprobs_k`
- `logprobs_mode` when it can be resolved
- exact `LLM(...)` kwargs supplied to the collector
- best-effort resolved vLLM configuration
- environment and accelerator information when available

The schema keeps `logprobs_mode` optional. If the active vLLM configuration cannot be resolved safely, the collector records `null` rather than guessing.

## Passing vLLM runtime settings

`collect_vllm.py` deliberately does not hardcode runtime tuning choices. Extra `LLM(...)` keyword arguments can be passed through explicitly:

```bash
--llm-kwargs-json '{"kv_cache_dtype":"fp8","enforce_eager":true}'
```

You can also attach investigation-specific metadata:

```bash
--metadata issue=#54521 \
--metadata patch=#55122
```

Keep runtime settings identical between arms unless the setting itself is the variable being tested.

### Memory-conscious collection

vLLM may reserve a substantial KV-cache pool during engine initialization. For small parity experiments on memory-constrained GPUs, you can explicitly reduce the fraction of GPU memory reserved by the engine, for example:

```bash
--llm-kwargs-json '{"gpu_memory_utilization":0.5}'
```

`0.5` is only an example, **not a tool default**. Choose a value appropriate for the model, context length, and hardware, and keep it identical across arms unless memory configuration is itself under test.

## Why there is no default significance threshold

A universal threshold would mix model scale, precision level, context length, execution path, hardware, and self-repeat noise.

v0.1 therefore reports the observed distributions and boundaries without deciding whether they constitute a failure.

A downstream project can add an explicit policy — for example, several consecutive positions exceeding an empirically measured self-repeat envelope — without changing the measurement layer.

## Relationship to vLLM correctness tests

This tool is intended to complement, not replace, vLLM's existing correctness checks.

It uses the same native `prompt_logprobs` behavioral signal and adds position, repeat, cross-configuration, and scheduling dimensions around it. The vLLM-specific logic stays in the collector; the analyzer operates only on canonical evidence.

## Privacy note

Raw prompt text is not stored by default. The collector records a prompt SHA-256 and token-count metadata.

However, forced-token IDs and returned top-k token IDs can still reveal information about the input. Treat collected evidence as potentially sensitive when sharing reports publicly.

## Scope — v0.1

- One prompt per collected evidence file
- Native offline vLLM Python API collector
- Canonical JSON schema
- Repeat-stability, cross-arm, and schedule measurements
- Position-resolved PNG + key-position CSV + machine-readable JSON report
- No universal significance threshold
- No diagnosis or optimization layer
- No external framework branding


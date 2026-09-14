# vllm-position-parity

Position-resolved behavioral regression measurement for vLLM `prompt_logprobs`.

 **native vLLM evidence → canonical JSON schema → position-resolved measurement report**
 
*Currently focused on vLLM; design is framework-agnostic and can be extended to compatible inference runtimes.*

When debugging determinism, quantization, backend changes, or scheduling effects, end-to-end completion hashes and aggregate quality metrics can tell you that behavior changed without showing where along the prompt the difference first became observable.

`vllm-position-parity` adds that missing position dimension.

It is a small measurement tool built around native vLLM `prompt_logprobs`. It compares repeated runs, execution configurations, and scheduling modes while preserving the captured runtime metadata needed to interpret the result.

It measures what changed.

It does **not** label bugs, infer root causes, apply universal significance thresholds, or recommend optimizations.

**Measure, don't classify.**

### Quick Start: `bi_sp_probe` Adapter

`tools/adapt_bi_sp_probe.py` converts the batch-composition / repeat-stability
probe format used in `vllm-project/vllm#56370` into the draft canonical
`decode_sampled_logprob` evidence format.

The adapter is offline and standard-library only. It does not require vLLM,
CUDA, or a GPU.

It preserves the external probe as the source of execution evidence while
independently recomputing all comparison metrics from `runs.*`.

#### Convert a captured probe

```bash
python tools/adapt_bi_sp_probe.py \
  tests/fixtures/bi_sp_probe/tp4_sp_on_prefix_off_failing.json \
  --out examples/failing.canonical.json \
  --summary-only
```
Omit `--summary-only` to retain the normalized raw per-prompt traces in the canonical output.

The adapter explicitly records:
- `trace_domain = decode_sampled_logprob`
- repeat-stability vs batch-composition comparison axes
- prompt identity across `bs1_a`, `bsN_a`, `bs1_b`, and `bsN_b`
- resolved execution metadata when present
- sampled-token logprob mismatch positions
- same-token logprob drift
- different sampled-token positions
- token-sequence mismatch prompts

The source probe's `verdicts` are not used as measurement input.
They are retained only as a validation oracle. By default, adaptation fails if
the independently recomputed metrics do not match the recorded source verdicts.

**Compact report without raw traces**
```
python tools/adapt_bi_sp_probe.py \
  tests/fixtures/bi_sp_probe/tp4_sp_on_prefix_off_failing.json \
  --out /tmp/failing.summary.json \
  --summary-only
```
`--summary-only` omits the raw per-prompt traces from the output JSON; it does
not skip JSON output. A compact diagnostic summary is also printed to stdout.

For the included TP=4 / SP-on fixture:
```
trace_domain=decode_sampled_logprob arm=Q4_tp4_sp1_triton_noprefix
bs1_a_vs_bs1_b: compared=1536 lp_mismatch=0 same_token_drift=0 different_token_lp=0 token_id_diff=0 sequence_mismatch_prompts=0
bsN_a_vs_bsN_b: compared=1536 lp_mismatch=1197 same_token_drift=1090 different_token_lp=107 token_id_diff=107 sequence_mismatch_prompts=11
bs1_a_vs_bsN_a: compared=1536 lp_mismatch=1528 same_token_drift=1419 different_token_lp=109 token_id_diff=109 sequence_mismatch_prompts=12
bs1_b_vs_bsN_b: compared=1536 lp_mismatch=1527 same_token_drift=1425 different_token_lp=102 token_id_diff=102 sequence_mismatch_prompts=11
source_verdict_match=True
```
A useful distinction in this trace is that sampled-logprob drift becomes
observable before sampled-token divergence. For `bsN_a_vs_bsN_b`, 1,197
positions have different sampled-token logprobs; 1,090 of those still carry the
same sampled token ID, while 107 positions have different sampled token IDs.

This is a measurement distinction only. The adapter does not infer a root
cause, classify a run as buggy, or apply a universal significance threshold.

**Run the adapter regression tests**
```
python -m unittest discover \
  -s tests \
  -p 'test_adapt_bi_sp_probe.py'
```
For a clean batch-composition control, disable prefix caching unless prefix
caching itself is the variable under test. Resolved runtime fields are preserved
when present; missing fields are not inferred from notes or other runs.

### Real-world validation

`vllm-position-parity` has been used as the position-resolved measurement layer in real vLLM debugging and quantization validation scenarios:

- **Determinism isolation (#54521)**
  Used in a five-arm controlled experiment to isolate the root causes of non-deterministic greedy decoding on Qwen3.8-Flash-Next / GB10 (sm_121). The tool measured per-position disagreement across runs with individual fixes applied, confirming that no single patch removed the divergence, and that all four fixes together eliminated it completely.
  [54521#issuecomment-5600097794](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5600097794)

- **KV quantization parity (#54426)**
  Used to compare per-position logprob behavior across BF16 / FP8 / NVFP4 KV cache configurations on the same model and hardware. The position-resolved trace helped characterize the prefill regression pattern and separate kernel-level effects from aggregate end-to-end metrics.[54426#issuecomment-5602282777](https://github.com/vllm-project/vllm/issues/54426#issuecomment-5602282777)

All results above were produced and independently verified by community contributors on real production-grade hardware.

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

### Input and output paths

`analyze.py` takes evidence **JSON files** as positional inputs:

```bash
python analyze.py reference.json [candidate.json] --out report_directory/
```
- `reference` and optional `candidate` must be canonical evidence JSON files.
-  `--out` is an output directory, not an input file.
- To compare several arms, run pairwise analyses or inspect each arm's self-repeat report separately.

### Recording active fixes

When an experimental arm carries one or more patches or fixes, record them explicitly:

```bash
python collect_client.py \
  ... \
  --arm all4 \
  --active-fix "#55122" \
  --active-fix "moe_nonfused_finalize" \
  --active-fix "flashinfer_cachekey" \
  --active-fix "ple_semaphore_reset"
```
`active_fixes` is provenance only. The collector does not enable, validate, or interpret these fixes, and the analyzer does not infer causality from them.

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

## Future work
- **multi-arm matrix**

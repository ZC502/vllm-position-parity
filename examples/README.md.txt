# VPP × vLLM PR #47922 — Scheduler/KV Native-Simulator Pilot v0.1

A handoff-ready CPU-only pilot that applies the **VPP measurement grammar** to scheduler/KV-cache behavior:

```text
controlled arms
→ canonical evidence
→ self-repeat stability
→ cross-arm first observable divergence
→ Slack-ready matrix
```

This is not the original VPP prompt/decode-logprob analyzer. It reuses the same evidence discipline for a new trace domain: `scheduler_kv_request_boundary`.

## Default experiment

Hold the request token trace and caller-prescribed simulated output tokens fixed; change only virtual KV capacity.

```text
A_high_capacity = 16 GiB
B_low_capacity  =  2 GiB
3 fresh Python processes per arm
896-token seed prompt
12 unique 896-token churn prompts
same seed replayed at the end
2 prescribed output tokens/request
```

The probe reads existing request-boundary observables only: prefix-cache query/hit counters, KV-cache utilization, preemptions, running/waiting requests.

> `first observable divergence` does **not** mean first internal scheduler transition.

## Upstream basis

Targets vLLM PR **#47922 `[KV Cache] Native simulator`**. The runner records the **actual local git SHA** for the checked-out branch. Publish the SHA with any result.

The PR simulator is CPU-only and uses MRV2 + `triton-cpu`. Reuse the PR's current CPU-simulator environment/CI requirements rather than assuming an arbitrary Triton build.

## Files

```text
native_simulator_probe.py   one fresh-run collector → canonical JSON
run_matrix.sh               A0 A1 A2 B0 B1 B2 orchestration
summarize.py                zero-vLLM offline analyzer
CANONICAL_SCHEMA.md         evidence contract
selftest_summarize.py       synthetic analyzer self-test only
```

## Prepare vLLM

Use an existing vLLM development environment capable of running PR #47922. One convenient checkout route is:

```bash
git fetch origin pull/47922/head:pr-47922
git checkout pr-47922
```

Then install/update that checkout following vLLM contributor instructions and the PR's current `triton-cpu` requirement.

Before running the matrix:

```bash
export VLLM_REPO="$PWD"
export TRITON_CPU_BACKEND=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

`run_matrix.sh` never checks out, rebases, or mutates the vLLM repository.

## Inspect workload without vLLM

```bash
python native_simulator_probe.py \
  --arm preview \
  --repeat 0 \
  --kv-cache-memory-bytes 17179869184 \
  --out /tmp/unused.json \
  --print-workload
```

## Run

```bash
VLLM_REPO=/path/to/vllm-pr47922 ./run_matrix.sh
```

Outputs:

```text
results/<UTC timestamp>/
├── RUN_MANIFEST.json
├── evidence/
│   ├── A_high_capacity_r0.json
│   ├── A_high_capacity_r1.json
│   ├── A_high_capacity_r2.json
│   ├── B_low_capacity_r0.json
│   ├── B_low_capacity_r1.json
│   └── B_low_capacity_r2.json
├── logs/
├── SUMMARY.json
├── SUMMARY.md
├── SLACK_RESULT.md
└── SHA256SUMS.txt
```

## If 2 GiB cannot initialize

Treat that capture as **INCOMPLETE**. Do not quietly replace only failed repeats. Start a new complete capture, for example:

```bash
LOW_KV_BYTES=4294967296 \
OUT_DIR="$PWD/results/4g-pilot" \
VLLM_REPO=/path/to/vllm-pr47922 \
./run_matrix.sh
```

If a smoke run is used to choose workable parameters, freeze them and rerun **all six fresh repeats** before posting public numbers.

## If the low arm never loses the seed prefix

Create a new experiment with more churn and rerun both arms from scratch:

```bash
CHURN_COUNT=24 \
OUT_DIR="$PWD/results/churn24" \
VLLM_REPO=/path/to/vllm-pr47922 \
./run_matrix.sh
```

Do not mix different workload hashes.

## Canonical event shape

```json
{
  "event_index": 13,
  "label": "seed_replay",
  "prompt_sha256": "...",
  "simulated_output_token_ids_expected": [501, 502],
  "output_token_ids_observed": [501, 502],
  "control": {"output_token_control": "PASS"},
  "metrics": {
    "delta": {
      "prefix_cache_queries": 896,
      "prefix_cache_hits": 880,
      "num_preemptions": 0
    },
    "prefix_hit_ratio": 0.982
  }
}
```

Those numbers are a **schema example only**, not a measured #47922 result.

## Slack-ready output shape

`SLACK_RESULT.md` is generated automatically:

```text
Comparison   | A self | B self | Output ctrl | First observable div | Seed replay hit A/B | Preempt A/B
-------------+--------+--------+-------------+----------------------+---------------------+------------
16GiB ↔ 2GiB | PASS   | PASS   | PASS        | E13 seed_replay      | 0.982 / 0.000       | 0 / 0
```

Again, those values are illustrative only.

## Interpretation rules

1. **Self-repeat first.** If an arm is unstable, don't attribute its cross-arm residual to capacity alone.
2. **Output control first.** If caller-prescribed simulated token IDs are not reproduced, don't trust the comparison.
3. **No universal threshold.** Measure divergence; don't impose a universal “bad” prefix-hit delta.
4. **No root-cause classification.** A divergence says where observed behavior stopped agreeing, not why.
5. **Don't call request-boundary evidence an internal scheduler trace.**

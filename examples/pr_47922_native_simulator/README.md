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

## Experimental Controls and Operational Rules

This pilot applies the VPP differential-evidence pattern to a new trace domain:
scheduler / KV-cache behavior at observable request boundaries.

### Core controls

1. **Matched workload**
   - Both A/B arms must execute the exact same workload.
   - The default workload is:
     `1 seed + 12 churn requests + 1 seed replay = 14 events`.
   - The canonical evidence records a workload SHA-256, and the summarizer
     requires the workload hash to match across arms.

2. **Output-token control**
   - Every event verifies:

     ```text
     observed_output_tokens == prescribed_simulated_output_tokens
     ```

   - If this control fails, the scheduler/KV comparison should not be treated
     as valid evidence.

3. **Single experimental variable**
   - In the default matrix, the intended A/B difference is only:

     ```text
     A_high_capacity = 16 GiB virtual KV
     B_low_capacity  =  2 GiB virtual KV
     ```

   - Model, workload, simulator revision, request order, prefix-caching mode,
     and prescribed output tokens must remain matched.

---

## Operational Rules

### 1. Pin exact provenance

PR #47922 is still evolving, so results must be tied to the exact vLLM
revision that produced them.

`run_matrix.sh` records:

```text
git rev-parse HEAD
git branch
git working-tree status
vLLM version
model
KV capacity
workload hash
```
into the run manifest / canonical evidence.
The pilot also fixes these runtime controls:

```
TRITON_CPU_BACKEND=1
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

The latter is an experiment control used by this pilot; it should not be
interpreted as a universal requirement of simulated-forward mode.

### 2. Strict serial request-boundary snapshotting

The collector runs requests serially:
```
initial metric snapshot
    ↓
llm.generate(event_0)
    ↓
metric snapshot
    ↓
event_0 delta
    ↓
llm.generate(event_1)
    ↓
metric snapshot
    ↓
event_1 delta
    ↓
...
```
The previous request's `after` snapshot becomes the next request's `before`snapshot.

This prevents concurrent workload activity from being intentionally introducedinto the v0.1 experiment and keeps each metric delta associated with one observable request boundary.

Note that this is **request-boundary evidence**, not instrumentation of internal scheduler transitions.

### 3. Low-capacity fallback

The default low-capacity arm is:
```
2 GiB = 2147483648 bytes
```
If that arm cannot initialize or complete the workload, treat the capture as **INCOMPLETE**. Do not classify initialization failure as a scheduler-state divergence.

Run a new complete matrix at 4 GiB:
```
LOW_KV_BYTES=4294967296 \
OUT_DIR="$PWD/results/4g-pilot" \
VLLM_REPO=/path/to/vllm-pr47922 \
./run_matrix.sh
```
**Do not reuse successful 2 GiB runs together with new 4 GiB runs**. Once the capacity is changed, all repeats in both arms should be rerun as one new capture.



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

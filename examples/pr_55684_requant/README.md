# VPP five-arm runner for vLLM PR #55684

"**Status: experimental / PR-specific validation example.**

This example tracks the evolving vLLM PR #55684. It is not a stable VPP public API. Always record the exact vLLM revision used for a capture."

This bundle operationalizes the five-arm design:

- **A** — pre-PR baseline, native MXFP8, TP1
- **B** — exact PR revision, native MXFP8, TP1
- **C** — exact PR revision, MXFP8 → FP8 PTPC via `--quantization-config.linear fp8_per_channel`, TP1
- **D** — exact PR revision, native MXFP8, TP2
- **E** — exact PR revision, same re-quantization, TP2

The runner uses the PR's own serving interface rather than reimplementing the feature.

## Why these comparisons

- `A↔B`: did the PR alter the native/default path?
- `B↔C`: what behavioral residual appears when the intended re-quantization is enabled?
- `B↔D`: native MXFP8 TP sensitivity control
- `C↔E`: re-quantized FP8 PTPC TP sensitivity

## Prerequisites

You need:

1. a vLLM source checkout at the selected pre-PR baseline (`5893426b88...` by default),
2. a separate vLLM checkout at the exact PR #55684 revision to test,
3. a `vllm-position-parity` checkout containing `collect_client.py` (or `collect_client_reviewed.py`) and `analyze.py`,
4. dependencies installed so `vllm serve` can import from either source checkout via `PYTHONPATH`,
5. the ModelOpt MXFP8 test model available locally or downloadable.

## Run

```bash
export PRE_SRC=/path/to/vllm-pre-55684
export PR_SRC=/path/to/vllm-pr-55684
export VPP_ROOT=/path/to/vllm-position-parity

# RTX PRO 6000 example: A/B/C all use GPU0; D/E both use GPU0,1.
export GPU_TP1=0
export GPU_TP2=0,1

bash run_55684_vpp.sh
```

Optional knobs:

```bash
export MODEL=mmangkad/Qwen3-4B-Instruct-2507-MXFP8
export REPEATS=3
export GPU_MEMORY_UTILIZATION=0.60
export ATTENTION_BACKEND=FLASH_ATTN   # only if you intentionally want to pin it
export OUT_ROOT=/path/to/output
```

The script refuses to run if `PRE_SRC` does not start with the expected baseline prefix `5893426b88` unless you explicitly change:

```bash
export PRE_EXPECTED_PREFIX=<your selected baseline prefix>
```

The PR revision is *not* hardcoded because the PR can move. The exact `PR_SRC` HEAD is captured into provenance.

## Summarize

```bash
python summarize_55684.py \
  --run-root /path/to/output \
  --vpp-root "$VPP_ROOT"
```

This produces:

```text
SUMMARY.json
SUMMARY.md
PR_COMMENT.md
provenance.json
SHA256SUMS.txt
comparison_reports/
  A_vs_B.csv
  B_vs_C.csv
  B_vs_D.csv
  C_vs_E.csv
```

`PR_COMMENT.md` is intentionally conservative: it reports differential evidence, not a regression label or root-cause claim.

## Important environment note

The scripts do **not** run a GPU experiment inside ChatGPT. They are meant to be executed on your RTX PRO 6000 / AMD test host. ChatGPT's current execution container has no access to your 8×RTX PRO 6000 job environment or the model/checkpoints, so only syntax/static and synthetic analyzer tests can be performed here.

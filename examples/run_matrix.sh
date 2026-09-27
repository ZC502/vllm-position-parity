#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-openai/gpt-oss-120b}"
REPEATS="${REPEATS:-3}"
HIGH_KV_BYTES="${HIGH_KV_BYTES:-17179869184}"   # 16 GiB
LOW_KV_BYTES="${LOW_KV_BYTES:-2147483648}"      # 2 GiB
PROMPT_LEN="${PROMPT_LEN:-896}"
CHURN_COUNT="${CHURN_COUNT:-12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-2}"
VLLM_REPO="${VLLM_REPO:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$HERE/results/$STAMP}"
EVIDENCE_DIR="$OUT_DIR/evidence"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$EVIDENCE_DIR" "$LOG_DIR"

export TRITON_CPU_BACKEND="${TRITON_CPU_BACKEND:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

if [[ -n "$VLLM_REPO" ]]; then
  [[ -d "$VLLM_REPO" ]] || { echo "ERROR: VLLM_REPO does not exist: $VLLM_REPO" >&2; exit 2; }
  export PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[preflight] python: $(command -v python)"
echo "[preflight] model: $MODEL"
echo "[preflight] high KV: $HIGH_KV_BYTES bytes"
echo "[preflight] low KV:  $LOW_KV_BYTES bytes"
echo "[preflight] repeats: $REPEATS"
echo "[preflight] output: $OUT_DIR"

python - <<'PY'
import dataclasses
from vllm.engine.arg_utils import EngineArgs
names = {f.name for f in dataclasses.fields(EngineArgs)}
missing = {"simulate_forward", "kv_cache_memory_bytes"} - names
if missing:
    raise SystemExit("This vLLM environment does not expose PR #47922 simulator args: " + ", ".join(sorted(missing)))
print("[preflight] PR #47922 simulator args detected")
PY

GIT_HEAD=""; GIT_BRANCH=""; GIT_STATUS=""
if [[ -n "$VLLM_REPO" ]] && git -C "$VLLM_REPO" rev-parse HEAD >/dev/null 2>&1; then
  GIT_HEAD="$(git -C "$VLLM_REPO" rev-parse HEAD)"
  GIT_BRANCH="$(git -C "$VLLM_REPO" rev-parse --abbrev-ref HEAD)"
  GIT_STATUS="$(git -C "$VLLM_REPO" status --porcelain || true)"
fi

export MODEL REPEATS HIGH_KV_BYTES LOW_KV_BYTES PROMPT_LEN CHURN_COUNT MAX_MODEL_LEN OUTPUT_TOKENS VLLM_REPO GIT_HEAD GIT_BRANCH GIT_STATUS OUT_DIR
python - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["OUT_DIR"]) / "RUN_MANIFEST.json"
obj = {
  "schema_version": "vpp.scheduler-kv-run-manifest/0.1",
  "profile": "pr47922_native_simulator_kv_capacity",
  "model": os.environ["MODEL"],
  "repeats": int(os.environ["REPEATS"]),
  "arms": {
    "A_high_capacity": {"kv_cache_memory_bytes": int(os.environ["HIGH_KV_BYTES"])},
    "B_low_capacity": {"kv_cache_memory_bytes": int(os.environ["LOW_KV_BYTES"])},
  },
  "workload": {
    "prompt_len": int(os.environ["PROMPT_LEN"]),
    "churn_count": int(os.environ["CHURN_COUNT"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "output_tokens": int(os.environ["OUTPUT_TOKENS"]),
  },
  "source": {
    "vllm_repo": os.environ["VLLM_REPO"],
    "git_head": os.environ["GIT_HEAD"],
    "git_branch": os.environ["GIT_BRANCH"],
    "git_dirty": bool(os.environ["GIT_STATUS"]),
    "upstream_pr": 47922,
  },
  "controls": {
    "fresh_python_process_per_repeat": True,
    "triton_cpu_backend": os.environ.get("TRITON_CPU_BACKEND"),
    "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
  },
}
out.write_text(json.dumps(obj, indent=2) + "\n")
print("[manifest]", out)
PY

failures=0
run_one() {
  local arm="$1" repeat="$2" kv_bytes="$3"
  local out="$EVIDENCE_DIR/${arm}_r${repeat}.json"
  local log="$LOG_DIR/${arm}_r${repeat}.log"
  echo "[run] arm=$arm repeat=$repeat kv=$kv_bytes"
  set +e
  cmd=(python "$HERE/native_simulator_probe.py"
    --out "$out" --arm "$arm" --repeat "$repeat" --model "$MODEL"
    --kv-cache-memory-bytes "$kv_bytes" --max-model-len "$MAX_MODEL_LEN"
    --prompt-len "$PROMPT_LEN" --churn-count "$CHURN_COUNT" --output-tokens "$OUTPUT_TOKENS")
  if [[ -n "$VLLM_REPO" ]]; then cmd+=(--source-repo "$VLLM_REPO"); fi
  "${cmd[@]}" >"$log" 2>&1
  rc=$?
  set -e
  cat "$log"
  if [[ $rc -ne 0 ]]; then
    echo "[run] FAILED arm=$arm repeat=$repeat rc=$rc" >&2
    failures=$((failures + 1))
  fi
}

for ((r=0; r<REPEATS; r++)); do run_one "A_high_capacity" "$r" "$HIGH_KV_BYTES"; done
for ((r=0; r<REPEATS; r++)); do run_one "B_low_capacity" "$r" "$LOW_KV_BYTES"; done

set +e
python "$HERE/summarize.py" "$EVIDENCE_DIR" --out-dir "$OUT_DIR" --expected-repeats "$REPEATS"
summary_rc=$?
set -e

(
  cd "$OUT_DIR"
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)

echo
echo "[done] $OUT_DIR"
echo "[done] Slack-ready text: $OUT_DIR/SLACK_RESULT.md"
echo "[done] checksums:       $OUT_DIR/SHA256SUMS.txt"
if [[ $failures -ne 0 || $summary_rc -ne 0 ]]; then
  echo "[done] capture is INCOMPLETE (run failures=$failures, summary_rc=$summary_rc)" >&2
  exit 4
fi

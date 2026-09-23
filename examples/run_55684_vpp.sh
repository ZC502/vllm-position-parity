#!/usr/bin/env bash
set -Eeuo pipefail

# VPP five-arm differential matrix for vLLM PR #55684
#
# Arms:
#   A = pre-PR baseline, native MXFP8, TP1
#   B = PR head,          native MXFP8, TP1
#   C = PR head,          MXFP8 -> FP8 PTPC (fp8_per_channel), TP1
#   D = PR head,          native MXFP8, TP2
#   E = PR head,          MXFP8 -> FP8 PTPC (fp8_per_channel), TP2
#
# Intended comparisons:
#   A <-> B : PR default-path neutrality
#   B <-> C : intended re-quantization residual
#   B <-> D : native MXFP8 TP sensitivity
#   C <-> E : re-quantized FP8 PTPC TP sensitivity
#
# This runner uses vLLM's OpenAI-compatible server plus VPP collect_client.py.
# It intentionally runs A/B/C serially on the SAME GPU and D/E serially on
# the SAME GPU pair to avoid accelerator-set confounding.

die() { echo "[run_55684_vpp] ERROR: $*" >&2; exit 2; }
note() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

: "${PRE_SRC:?Set PRE_SRC=/path/to/vllm checkout at pre-PR baseline (default expected prefix 5893426b88)}"
: "${PR_SRC:?Set PR_SRC=/path/to/vllm checkout at the exact PR #55684 revision you want to test}"
: "${VPP_ROOT:?Set VPP_ROOT=/path/to/vllm-position-parity}"

MODEL="${MODEL:-mmangkad/Qwen3-4B-Instruct-2507-MXFP8}"
SERVED_MODEL="${SERVED_MODEL:-vpp-55684-qwen3-4b}"
OUT_ROOT="${OUT_ROOT:-$PWD/vpp-55684-$(date -u +%Y%m%d-%H%M%S)}"
REPEATS="${REPEATS:-3}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-5}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-0}"
PORT="${PORT:-8127}"
GPU_TP1="${GPU_TP1:-0}"
GPU_TP2="${GPU_TP2:-0,1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PRE_EXPECTED_PREFIX="${PRE_EXPECTED_PREFIX:-5893426b88}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
PROMPTS_JSON="${PROMPTS_JSON:-}"

[[ -f "$VPP_ROOT/collect_client.py" ]] || \
  [[ -f "$VPP_ROOT/collect_client_reviewed.py" ]] || \
  die "Need collect_client.py or collect_client_reviewed.py under VPP_ROOT=$VPP_ROOT"

if [[ -f "$VPP_ROOT/collect_client.py" ]]; then
  VPP_CLIENT="$VPP_ROOT/collect_client.py"
else
  VPP_CLIENT="$VPP_ROOT/collect_client_reviewed.py"
fi

for d in "$PRE_SRC" "$PR_SRC" "$VPP_ROOT"; do
  [[ -d "$d" ]] || die "not a directory: $d"
done

git -C "$PRE_SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "PRE_SRC is not a git checkout"
git -C "$PR_SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "PR_SRC is not a git checkout"
git -C "$VPP_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "VPP_ROOT is not a git checkout"

PRE_SHA="$(git -C "$PRE_SRC" rev-parse HEAD)"
PR_SHA="$(git -C "$PR_SRC" rev-parse HEAD)"
VPP_SHA="$(git -C "$VPP_ROOT" rev-parse HEAD)"
PRE_SUBJECT="$(git -C "$PRE_SRC" log -1 --format=%s)"
PR_SUBJECT="$(git -C "$PR_SRC" log -1 --format=%s)"
VPP_SUBJECT="$(git -C "$VPP_ROOT" log -1 --format=%s)"

if [[ "$PRE_SHA" != "$PRE_EXPECTED_PREFIX"* ]]; then
  die "PRE_SRC HEAD=$PRE_SHA does not start with expected baseline prefix $PRE_EXPECTED_PREFIX"
fi

mkdir -p "$OUT_ROOT"/{prompts,arms,meta}
RUN_LOG="$OUT_ROOT/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

note "OUT_ROOT=$OUT_ROOT"
note "PRE_SHA=$PRE_SHA"
note "PR_SHA=$PR_SHA"
note "VPP_SHA=$VPP_SHA"
note "MODEL=$MODEL"
note "GPU_TP1=$GPU_TP1 GPU_TP2=$GPU_TP2"

# Build a fixed short-prompt cohort if none was supplied. These are the same
# style of short deterministic prompts previously used in VPP/MRV2 captures.
if [[ -n "$PROMPTS_JSON" ]]; then
  [[ -f "$PROMPTS_JSON" ]] || die "PROMPTS_JSON does not exist: $PROMPTS_JSON"
  cp "$PROMPTS_JSON" "$OUT_ROOT/prompts.json"
else
  cat > "$OUT_ROOT/prompts.json" <<'JSON'
[
  "Compute 37 * 19. Give only the result.",
  "Write a Python function that returns the nth Fibonacci number iteratively.",
  "If a train travels 180 km in 2.5 hours, what is its average speed in km/h?",
  "Explain in two sentences why binary search requires sorted input.",
  "Continue the sequence: 2, 3, 5, 8, 13, 21,",
  "What is the derivative of x^3 + 2x^2 - 5x + 7?",
  "Return valid JSON with keys a, b, c whose values are 1, 2, 3.",
  "A box contains 4 red, 5 blue, and 6 green balls. How many balls are there?",
  "Implement a stable deduplication of a Python list while preserving order.",
  "Translate to French: The weather is clear today.",
  "Summarize the TCP three-way handshake in one short paragraph.",
  "Solve: 3x + 7 = 31.",
  "List the first eight prime numbers separated by commas.",
  "What does SQL LEFT JOIN preserve from the left table?",
  "Given [5, 1, 4, 2, 8], sort it in ascending order.",
  "A service receives requests in batches. Explain briefly why changing batch composition should not change the greedy token sequence for an otherwise identical deterministic request."
]
JSON
fi

python - "$OUT_ROOT/prompts.json" "$OUT_ROOT/prompts" <<'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
items = json.loads(src.read_text(encoding="utf-8"))
if not isinstance(items, list) or not items or not all(isinstance(x, str) for x in items):
    raise SystemExit("prompts.json must be a non-empty JSON array of strings")
dst.mkdir(parents=True, exist_ok=True)
for i, text in enumerate(items):
    (dst / f"p{i:02d}.txt").write_text(text, encoding="utf-8")
print(f"wrote {len(items)} prompts")
PY

# Capture hardware/software provenance before any model starts.
{
  echo "date_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname=$(hostname)"
  echo "python=$(python -V 2>&1 || true)"
  echo "pre_sha=$PRE_SHA"
  echo "pre_subject=$PRE_SUBJECT"
  echo "pr_sha=$PR_SHA"
  echo "pr_subject=$PR_SUBJECT"
  echo "vpp_sha=$VPP_SHA"
  echo "vpp_subject=$VPP_SUBJECT"
  echo "model=$MODEL"
  echo "served_model=$SERVED_MODEL"
  echo "gpu_tp1=$GPU_TP1"
  echo "gpu_tp2=$GPU_TP2"
  echo "repeats=$REPEATS"
  echo "prompt_logprobs=$PROMPT_LOGPROBS"
  echo "max_tokens=$MAX_TOKENS"
  echo "temperature=$TEMPERATURE"
  echo "seed=$SEED"
  echo "kv_cache_dtype=$KV_CACHE_DTYPE"
  echo "attention_backend=$ATTENTION_BACKEND"
} > "$OUT_ROOT/meta/run.env.txt"

(nvidia-smi -L || true) > "$OUT_ROOT/meta/nvidia-smi-L.txt" 2>&1
(nvidia-smi || true) > "$OUT_ROOT/meta/nvidia-smi.txt" 2>&1
(python - <<'PY'
try:
    import torch
    print("torch_version=" + str(torch.__version__))
    print("torch_cuda=" + str(torch.version.cuda))
except Exception as e:
    print("torch_import_error=" + repr(e))
PY
) > "$OUT_ROOT/meta/python-runtime.txt" 2>&1

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    note "stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      note "server did not exit after SIGTERM; sending SIGKILL"
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

wait_for_server() {
  local pid="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if python - "$PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
port = sys.argv[1]
for path in ("/health", "/v1/models"):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
            if 200 <= r.status < 300:
                raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_arm() {
  local arm="$1"
  local src="$2"
  local gpus="$3"
  local tp="$4"
  local requant="$5"  # "native" or "fp8_per_channel"
  local source_sha
  source_sha="$(git -C "$src" rev-parse HEAD)"

  local arm_dir="$OUT_ROOT/arms/$arm"
  mkdir -p "$arm_dir/evidence" "$arm_dir/client_logs"
  local server_log="$arm_dir/server.log"
  local cmd_file="$arm_dir/server_command.txt"

  local -a cmd=(
    vllm serve "$MODEL"
    --host 127.0.0.1
    --port "$PORT"
    --served-model-name "$SERVED_MODEL"
    --tensor-parallel-size "$tp"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --no-enable-prefix-caching
  )

  if [[ "$DTYPE" != "auto" ]]; then
    cmd+=(--dtype "$DTYPE")
  fi
  if [[ -n "$ATTENTION_BACKEND" ]]; then
    cmd+=(--attention-backend "$ATTENTION_BACKEND")
  fi
  if [[ "$requant" == "fp8_per_channel" ]]; then
    cmd+=(--quantization-config.linear fp8_per_channel)
  fi

  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$gpus" "$src" > "$cmd_file"
  printf '%q ' "${cmd[@]}" >> "$cmd_file"
  printf '\n' >> "$cmd_file"

  note "starting arm=$arm tp=$tp requant=$requant gpus=$gpus sha=$source_sha"
  (
    cd "$src"
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PYTHONPATH="$src${PYTHONPATH:+:$PYTHONPATH}"
    exec "${cmd[@]}"
  ) >"$server_log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server "$SERVER_PID"; then
    tail -200 "$server_log" >&2 || true
    die "server failed to become ready for arm=$arm"
  fi
  note "server ready arm=$arm pid=$SERVER_PID"

  local prompt_file label out_json client_log
  for prompt_file in "$OUT_ROOT"/prompts/p*.txt; do
    label="$(basename "$prompt_file" .txt)"
    out_json="$arm_dir/evidence/$label.json"
    client_log="$arm_dir/client_logs/$label.log"
    note "collect arm=$arm prompt=$label"

    python "$VPP_CLIENT" \
      --base-url "http://127.0.0.1:$PORT" \
      --model "$SERVED_MODEL" \
      --prompt-file "$prompt_file" \
      --out "$out_json" \
      --arm "$arm" \
      --execution-mode sequential \
      --repeats "$REPEATS" \
      --prompt-logprobs "$PROMPT_LOGPROBS" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --timeout "$REQUEST_TIMEOUT" \
      --endpoint-label "$arm" \
      --metadata "pr=55684" \
      --metadata "source_sha=$source_sha" \
      --metadata "tp=$tp" \
      --metadata "gpu_set=$gpus" \
      --metadata "model=$MODEL" \
      --metadata "quant_path=$requant" \
      --metadata "kv_cache_dtype=$KV_CACHE_DTYPE" \
      --metadata "attention_backend=${ATTENTION_BACKEND:-auto}" \
      >"$client_log" 2>&1
  done

  cleanup_server
  note "finished arm=$arm"
}

# Run serially to preserve same physical GPU / GPU pair across the relevant arms.
run_arm "A_pre_native_tp1" "$PRE_SRC" "$GPU_TP1" 1 native
run_arm "B_pr_native_tp1"  "$PR_SRC"  "$GPU_TP1" 1 native
run_arm "C_pr_requant_tp1" "$PR_SRC"  "$GPU_TP1" 1 fp8_per_channel
run_arm "D_pr_native_tp2"  "$PR_SRC"  "$GPU_TP2" 2 native
run_arm "E_pr_requant_tp2" "$PR_SRC"  "$GPU_TP2" 2 fp8_per_channel

python - "$OUT_ROOT" "$PRE_SHA" "$PR_SHA" "$VPP_SHA" "$MODEL" "$REPEATS" "$GPU_TP1" "$GPU_TP2" <<'PY'
import hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
manifest = {
    "schema": "vpp-pr55684-run-manifest/0.1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "VPP five-arm behavioral validation matrix for vLLM PR #55684",
    "pr": {
        "number": 55684,
        "title": "[Quantization] Layer re-quantization for linear layers through online quantization API (MXFP8 -> FP8 PTPC showcase)"
    },
    "commits": {
        "pre_pr_baseline": sys.argv[2],
        "pr_head": sys.argv[3],
        "vpp": sys.argv[4],
    },
    "model": sys.argv[5],
    "repeats": int(sys.argv[6]),
    "gpu_tp1": sys.argv[7],
    "gpu_tp2": sys.argv[8],
    "arms": {
        "A": "pre-PR native MXFP8 TP1",
        "B": "PR native MXFP8 TP1",
        "C": "PR MXFP8->FP8 PTPC fp8_per_channel TP1",
        "D": "PR native MXFP8 TP2",
        "E": "PR MXFP8->FP8 PTPC fp8_per_channel TP2",
    },
    "comparisons": {
        "A_vs_B": "PR default-path neutrality",
        "B_vs_C": "intended re-quantization residual",
        "B_vs_D": "native MXFP8 TP sensitivity",
        "C_vs_E": "re-quantized FP8 PTPC TP sensitivity",
    },
}
(out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("wrote", out / "run_manifest.json")
PY

note "captures complete"
note "next: python summarize_55684.py --run-root '$OUT_ROOT' --vpp-root '$VPP_ROOT'"

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# This profile intentionally reuses vllm-position-parity's canonical evidence
# loader/analyzer instead of inventing a second evidence schema.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import analyze as vpp_analyze
except Exception as exc:  # pragma: no cover - exercised by CLI error path
    raise SystemExit(
        "kv-cache-quant profile must live under the vllm-position-parity repo, "
        "with analyze.py at the repo root. Import failed: %r" % (exc,)
    )

PROFILE_SCHEMA = "vllm-position-parity/kv-cache-quant-profile/0.1"
PROFILE_VERSION = "0.1.0"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def stable_hash(obj: Any) -> str:
    raw = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_path(obj: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _completion_sequence(sample: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in sample.get("completion_token_ids", []) or [])


def modal_completion(run: dict[str, Any]) -> dict[str, Any]:
    seqs = [_completion_sequence(s) for s in run.get("samples", [])]
    if not seqs:
        return {"sequence": [], "count": 0, "fraction": None, "distinct_sequences": 0}
    counts = Counter(seqs)
    seq, count = counts.most_common(1)[0]
    return {
        "sequence": list(seq),
        "count": count,
        "fraction": count / len(seqs),
        "distinct_sequences": len(counts),
    }


def compare_modal_completions(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    a = list(ref.get("sequence", []))
    b = list(cand.get("sequence", []))
    upto = min(len(a), len(b))
    first = next((i for i in range(upto) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = upto
    same_positions = sum(1 for i in range(upto) if a[i] == b[i])
    denom = max(len(a), len(b))
    token_agreement = (same_positions / denom) if denom else None
    return {
        "exact_match": a == b,
        "reference_length": len(a),
        "candidate_length": len(b),
        "first_token_divergence_step": first,
        "aligned_token_agreement_rate": token_agreement,
    }


def self_repeat_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("positions", [])
    known_top1 = [float(r["top1_agreement_rate"]) for r in rows if r.get("top1_agreement_rate") is not None]
    spreads = [float(r["forced_logprob_spread"]) for r in rows if r.get("forced_logprob_spread") is not None]
    return {
        "repeat_count": get_path(summary, "summary", "repeat_count"),
        "position_count": len(rows),
        "min_top1_agreement_rate": min(known_top1) if known_top1 else None,
        "mean_top1_agreement_rate": statistics.fmean(known_top1) if known_top1 else None,
        "positions_with_top1_disagreement": get_path(summary, "summary", "positions_with_top1_disagreement"),
        "max_forced_logprob_spread": max(spreads) if spreads else None,
        "p99_forced_logprob_spread": quantile(spreads, 0.99),
    }


def cross_precision_summary(cross: dict[str, Any], drift_epsilon: float | None) -> dict[str, Any]:
    rows = cross.get("positions", [])
    modal_known = [r for r in rows if r.get("modal_top1_match") is not None]
    modal_matches = sum(1 for r in modal_known if r.get("modal_top1_match") is True)
    deltas = [float(r["abs_forced_logprob_mean_delta"]) for r in rows if r.get("abs_forced_logprob_mean_delta") is not None]

    out = {
        "common_position_count": len(rows),
        "cross_modal_top1_known_positions": len(modal_known),
        "cross_modal_top1_agreement_rate": (modal_matches / len(modal_known)) if modal_known else None,
        "positions_with_modal_top1_mismatch": sum(1 for r in modal_known if r.get("modal_top1_match") is False),
        "mean_abs_forced_logprob_delta": statistics.fmean(deltas) if deltas else None,
        "p50_abs_forced_logprob_delta": quantile(deltas, 0.50),
        "p95_abs_forced_logprob_delta": quantile(deltas, 0.95),
        "p99_abs_forced_logprob_delta": quantile(deltas, 0.99),
        "max_abs_forced_logprob_delta": max(deltas) if deltas else None,
        "drift_epsilon": drift_epsilon,
        "positions_over_drift_epsilon": None,
        "fraction_over_drift_epsilon": None,
    }
    if drift_epsilon is not None and deltas:
        n = sum(1 for d in deltas if d > drift_epsilon)
        out["positions_over_drift_epsilon"] = n
        out["fraction_over_drift_epsilon"] = n / len(deltas)
    return out


def cache_observation(run: dict[str, Any]) -> dict[str, Any]:
    cached = [s.get("num_cached_tokens") for s in run.get("samples", []) if s.get("num_cached_tokens") is not None]
    created = [s.get("num_cache_creation_tokens") for s in run.get("samples", []) if s.get("num_cache_creation_tokens") is not None]
    def stats(xs: list[Any]) -> dict[str, Any]:
        vals = [float(x) for x in xs]
        return {
            "observations": len(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "mean": statistics.fmean(vals) if vals else None,
        }
    return {"num_cached_tokens": stats(cached), "num_cache_creation_tokens": stats(created)}


def telemetry(run: dict[str, Any], evidence_path: Path) -> dict[str, Any]:
    env = _jsonable(run.get("environment", {}))
    resolved = _jsonable(run.get("resolved_vllm_config", {}))
    rr = run.get("run", {})
    meta = _jsonable(rr.get("metadata", {}))
    out = {
        "evidence_file": str(evidence_path),
        "evidence_bytes": evidence_path.stat().st_size,
        "environment": env,
        "resolved_vllm_config": resolved,
        "llm_kwargs": _jsonable(rr.get("llm_kwargs", {})),
        "run_metadata": meta,
        "collection_timing": _jsonable(run.get("collection_timing")),
        "execution_mode": rr.get("execution_mode"),
        "model": rr.get("model"),
        "temperature": rr.get("temperature"),
        "seed": rr.get("seed"),
        "prompt_logprobs_k": rr.get("prompt_logprobs_k"),
        "max_tokens": rr.get("max_tokens"),
        "prompt_token_count": get_path(run, "prompt", "token_count"),
    }
    fingerprint_basis = {
        "accelerators": get_path(run, "environment", "accelerators"),
        "cuda_version": get_path(run, "environment", "cuda_version"),
        "torch_version": get_path(run, "environment", "torch_version"),
        "vllm_version": get_path(run, "environment", "vllm_version"),
        "vllm_commit": get_path(run, "environment", "vllm_commit"),
        "tensor_parallel_size": get_path(run, "resolved_vllm_config", "tensor_parallel_size"),
        "pipeline_parallel_size": get_path(run, "resolved_vllm_config", "pipeline_parallel_size"),
        "data_parallel_size": get_path(run, "resolved_vllm_config", "data_parallel_size"),
        "quantization": get_path(run, "resolved_vllm_config", "quantization"),
        "kv_cache_dtype": get_path(run, "resolved_vllm_config", "kv_cache_dtype"),
    }
    out["hardware_runtime_fingerprint_sha256"] = stable_hash(fingerprint_basis)
    return out


def compare_field(warnings: list[str], ref: dict[str, Any], cand: dict[str, Any], label: str, *path: str) -> None:
    a = get_path(ref, *path)
    b = get_path(cand, *path)
    if a is not None and b is not None and a != b:
        warnings.append(f"{label} differs: reference={a!r}, candidate={b!r}")


def comparability_warnings(ref: dict[str, Any], cand: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    # kv_cache_dtype is the intended primary variable and is therefore not warned on.
    for label, path in [
        ("model", ("run", "model")),
        ("temperature", ("run", "temperature")),
        ("seed", ("run", "seed")),
        ("execution_mode", ("run", "execution_mode")),
        ("prompt_logprobs_k", ("run", "prompt_logprobs_k")),
        ("weight/model quantization", ("resolved_vllm_config", "quantization")),
        ("prefix caching", ("resolved_vllm_config", "enable_prefix_caching")),
        ("tensor_parallel_size", ("resolved_vllm_config", "tensor_parallel_size")),
        ("pipeline_parallel_size", ("resolved_vllm_config", "pipeline_parallel_size")),
        ("data_parallel_size", ("resolved_vllm_config", "data_parallel_size")),
        ("max_num_batched_tokens", ("resolved_vllm_config", "max_num_batched_tokens")),
        ("vLLM version", ("environment", "vllm_version")),
        ("vLLM commit", ("environment", "vllm_commit")),
        ("torch version", ("environment", "torch_version")),
        ("CUDA runtime", ("environment", "cuda_version")),
        ("accelerator set", ("environment", "accelerators")),
    ]:
        compare_field(warnings, ref, cand, label, *path)

    if get_path(ref, "prompt", "sha256") != get_path(cand, "prompt", "sha256"):
        warnings.append("prompt SHA-256 differs; VPP core analysis should reject this pair")
    return warnings


def load_gate_policy(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("quality-gate policy must be a JSON object")
    return data


def _gate(checks: list[dict[str, Any]], name: str, observed: float | None, op: str, threshold: float) -> None:
    if observed is None:
        checks.append({"name": name, "status": "UNAVAILABLE", "observed": None, "operator": op, "threshold": threshold})
        return
    if op == ">=":
        ok = observed >= threshold
    elif op == "<=":
        ok = observed <= threshold
    else:
        raise ValueError(f"unsupported gate operator {op}")
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "observed": observed, "operator": op, "threshold": threshold})


def evaluate_gates(pair: dict[str, Any], policy: dict[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return {
            "status": "NOT_EVALUATED",
            "policy_name": None,
            "checks": [],
            "note": "No quality-gate policy supplied. Measurements remain diagnostic only.",
        }

    checks: list[dict[str, Any]] = []
    ref = pair["reference_repeat"]
    cand = pair["candidate_repeat"]
    cross = pair["cross_precision"]

    refp = policy.get("reference_repeat", {})
    candp = policy.get("candidate_repeat", {})
    crossp = policy.get("cross_precision", {})

    if "min_top1_agreement_rate" in refp:
        _gate(checks, "reference_repeat.min_top1_agreement_rate", ref.get("min_top1_agreement_rate"), ">=", float(refp["min_top1_agreement_rate"]))
    if "min_top1_agreement_rate" in candp:
        _gate(checks, "candidate_repeat.min_top1_agreement_rate", cand.get("min_top1_agreement_rate"), ">=", float(candp["min_top1_agreement_rate"]))
    if "max_forced_logprob_spread" in refp:
        _gate(checks, "reference_repeat.max_forced_logprob_spread", ref.get("max_forced_logprob_spread"), "<=", float(refp["max_forced_logprob_spread"]))
    if "max_forced_logprob_spread" in candp:
        _gate(checks, "candidate_repeat.max_forced_logprob_spread", cand.get("max_forced_logprob_spread"), "<=", float(candp["max_forced_logprob_spread"]))

    if "min_modal_top1_agreement_rate" in crossp:
        _gate(checks, "cross_precision.min_modal_top1_agreement_rate", cross.get("cross_modal_top1_agreement_rate"), ">=", float(crossp["min_modal_top1_agreement_rate"]))
    if "max_p99_abs_logprob_delta" in crossp:
        _gate(checks, "cross_precision.max_p99_abs_logprob_delta", cross.get("p99_abs_forced_logprob_delta"), "<=", float(crossp["max_p99_abs_logprob_delta"]))
    if "max_abs_logprob_delta" in crossp:
        _gate(checks, "cross_precision.max_abs_logprob_delta", cross.get("max_abs_forced_logprob_delta"), "<=", float(crossp["max_abs_logprob_delta"]))

    drift_policy = crossp.get("drift_fraction")
    if isinstance(drift_policy, dict):
        epsilon = float(drift_policy["epsilon"])
        observed_epsilon = cross.get("drift_epsilon")
        if observed_epsilon is None or abs(float(observed_epsilon) - epsilon) > 1e-15:
            checks.append({
                "name": "cross_precision.drift_fraction",
                "status": "UNAVAILABLE",
                "observed": cross.get("fraction_over_drift_epsilon"),
                "operator": "<=",
                "threshold": float(drift_policy["max_fraction"]),
                "reason": f"profile was not evaluated with required epsilon={epsilon}",
            })
        else:
            _gate(checks, "cross_precision.drift_fraction", cross.get("fraction_over_drift_epsilon"), "<=", float(drift_policy["max_fraction"]))

    statuses = [c["status"] for c in checks]
    if any(s == "FAIL" for s in statuses):
        status = "FAIL"
    elif any(s == "UNAVAILABLE" for s in statuses):
        status = "INCOMPLETE"
    elif checks:
        status = "PASS"
    else:
        status = "NOT_EVALUATED"
    return {
        "status": status,
        "policy_name": policy.get("policy_name"),
        "checks": checks,
        "note": "Policy is user/project supplied; it is not a universal quantization-correctness threshold.",
    }


def analyze_pair(label: str, ref_path: Path, cand_path: Path, drift_epsilon: float | None, policy: dict[str, Any] | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    ref_run = vpp_analyze.load_run(ref_path)
    cand_run = vpp_analyze.load_run(cand_path)
    core = vpp_analyze.build_report(ref_run, cand_run)

    ref_modal = modal_completion(ref_run)
    cand_modal = modal_completion(cand_run)
    pair = {
        "label": label,
        "prompt_token_count": int(ref_run["prompt"]["token_count"]),
        "reference_kv_cache_dtype": get_path(ref_run, "resolved_vllm_config", "kv_cache_dtype"),
        "candidate_kv_cache_dtype": get_path(cand_run, "resolved_vllm_config", "kv_cache_dtype"),
        "reference_repeat": self_repeat_summary(core["reference"]),
        "candidate_repeat": self_repeat_summary(core["candidate"]),
        "cross_precision": cross_precision_summary(core["cross_arm"], drift_epsilon),
        "completion_modal": {
            "reference": ref_modal,
            "candidate": cand_modal,
            "cross": compare_modal_completions(ref_modal, cand_modal),
            "note": "Completion identity is diagnostic only across precision; post-divergence differences are downstream evidence.",
        },
        "cache_observation": {
            "reference": cache_observation(ref_run),
            "candidate": cache_observation(cand_run),
        },
        "telemetry": {
            "reference": telemetry(ref_run, ref_path),
            "candidate": telemetry(cand_run, cand_path),
        },
        "comparability_warnings": comparability_warnings(ref_run, cand_run),
        "analysis_runtime_seconds": None,
    }
    pair["quality_gate"] = evaluate_gates(pair, policy)
    pair["analysis_runtime_seconds"] = time.perf_counter() - t0
    return pair


def write_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "label", "prompt_token_count", "reference_kv_cache_dtype", "candidate_kv_cache_dtype",
        "reference_repeat_min_top1", "candidate_repeat_min_top1",
        "cross_modal_top1_agreement", "cross_logprob_p95", "cross_logprob_p99", "cross_logprob_max",
        "drift_epsilon", "fraction_over_drift_epsilon", "completion_exact_modal_match",
        "completion_first_divergence", "gate_status", "comparability_warning_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow({
                "label": p["label"],
                "prompt_token_count": p["prompt_token_count"],
                "reference_kv_cache_dtype": p["reference_kv_cache_dtype"],
                "candidate_kv_cache_dtype": p["candidate_kv_cache_dtype"],
                "reference_repeat_min_top1": p["reference_repeat"]["min_top1_agreement_rate"],
                "candidate_repeat_min_top1": p["candidate_repeat"]["min_top1_agreement_rate"],
                "cross_modal_top1_agreement": p["cross_precision"]["cross_modal_top1_agreement_rate"],
                "cross_logprob_p95": p["cross_precision"]["p95_abs_forced_logprob_delta"],
                "cross_logprob_p99": p["cross_precision"]["p99_abs_forced_logprob_delta"],
                "cross_logprob_max": p["cross_precision"]["max_abs_forced_logprob_delta"],
                "drift_epsilon": p["cross_precision"]["drift_epsilon"],
                "fraction_over_drift_epsilon": p["cross_precision"]["fraction_over_drift_epsilon"],
                "completion_exact_modal_match": p["completion_modal"]["cross"]["exact_match"],
                "completion_first_divergence": p["completion_modal"]["cross"]["first_token_divergence_step"],
                "gate_status": p["quality_gate"]["status"],
                "comparability_warning_count": len(p["comparability_warnings"]),
            })


def main() -> int:
    ap = argparse.ArgumentParser(
        description="KV-cache quantization profile for vllm-position-parity canonical evidence."
    )
    ap.add_argument(
        "--pair", nargs=3, action="append", metavar=("LABEL", "REFERENCE_JSON", "CANDIDATE_JSON"), required=True,
        help="Add one matched BF16/reference vs quantized-KV candidate evidence pair. May be repeated for short/boundary/long cohorts.",
    )
    ap.add_argument("--out", default="kv_cache_quant_report", help="Output directory")
    ap.add_argument(
        "--quality-gates", type=Path,
        help="Optional project-specific gate policy JSON. Without it the profile never emits PASS/FAIL.",
    )
    ap.add_argument(
        "--drift-epsilon", type=float,
        help="Optional absolute forced-logprob delta used only for the reported/gated drift fraction. No universal default is assumed.",
    )
    args = ap.parse_args()

    if args.drift_epsilon is not None and args.drift_epsilon < 0:
        ap.error("--drift-epsilon must be >= 0")

    try:
        policy = load_gate_policy(args.quality_gates)
        pairs = [
            analyze_pair(label, Path(ref), Path(cand), args.drift_epsilon, policy)
            for label, ref, cand in args.pair
        ]
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[kv-cache-quant] ERROR: {exc}", file=sys.stderr)
        return 2

    statuses = [p["quality_gate"]["status"] for p in pairs]
    if policy is None:
        overall = "NOT_EVALUATED"
    elif any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif any(s in {"INCOMPLETE", "NOT_EVALUATED"} for s in statuses):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    report = {
        "schema": PROFILE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "tool": "vllm-position-parity",
        "profile": "kv-cache-quant",
        "measurement_policy": {
            "cross_precision_bit_exact_required": False,
            "automatic_root_cause": False,
            "universal_significance_threshold": None,
            "quality_gate_source": "user_or_project_policy" if policy else None,
            "note": "Cross-precision drift is measured, not presumed erroneous. Self-repeat stability and user-supplied quality gates are reported separately.",
        },
        "quality_gate_policy": policy,
        "overall_gate_status": overall,
        "pairs": pairs,
        "long_context_cost_note": (
            "This profile reports evidence size, prompt length, and analyzer runtime. "
            "Existing VPP v0.1 evidence does not record collector wall-clock time, so capture overhead is unavailable unless supplied in run.metadata by the collector/wrapper. "
            "Position filtering after capture cannot reduce vLLM prompt_logprobs collection cost; use selected short/boundary/long prompt cohorts to control capture cost."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(out / "summary.csv", pairs)

    print(f"wrote {out / 'report.json'}")
    print(f"wrote {out / 'summary.csv'}")
    print(f"overall_gate_status={overall}")
    for p in pairs:
        cp = p["cross_precision"]
        print(
            f"[{p['label']}] tokens={p['prompt_token_count']} "
            f"top1={cp['cross_modal_top1_agreement_rate']} "
            f"p99_lp_delta={cp['p99_abs_forced_logprob_delta']} "
            f"gate={p['quality_gate']['status']} warnings={len(p['comparability_warnings'])}"
        )
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

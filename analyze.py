#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


def load_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise ValueError(f"{path}: evidence file does not exist")
    if path.is_dir():
        raise ValueError(
            f"{path}: expected an evidence JSON file, but received a directory. "
            "Usage: python analyze.py reference.json [candidate.json] "
            "--out REPORT_DIRECTORY"
        )
    if not path.is_file():
        raise ValueError(f"{path}: expected an evidence JSON file")

    try:
        with path.open("r", encoding="utf-8") as f:
            run = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from None
    except OSError as exc:
        raise ValueError(f"{path}: could not read evidence file: {exc}") from None

    validate_minimal(run, str(path))
    normalize_backward_compat(run)
    return run


def normalize_backward_compat(run: dict[str, Any]) -> None:
    """Normalize optional fields introduced after schema v0.1.

    v0.1/v0.1.1 evidence files do not contain run.active_fixes. Treat the
    missing field as an empty list so old evidence remains directly usable.
    """
    run["run"].setdefault("active_fixes", [])


def validate_minimal(run: dict[str, Any], source: str = "<memory>") -> None:
    if not isinstance(run, dict):
        raise ValueError(f"{source}: top-level JSON value must be an object")

    for key in ("schema_version", "run", "prompt", "samples"):
        if key not in run:
            raise ValueError(f"{source}: missing required key {key!r}")

    if str(run["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"{source}: unsupported schema_version {run['schema_version']!r}"
        )

    if not isinstance(run["run"], dict):
        raise ValueError(f"{source}: run must be an object")
    if not isinstance(run["prompt"], dict):
        raise ValueError(f"{source}: prompt must be an object")
    if not isinstance(run["samples"], list) or not run["samples"]:
        raise ValueError(f"{source}: samples must be a non-empty list")

    for key in ("sha256", "token_count"):
        if key not in run["prompt"]:
            raise ValueError(f"{source}: prompt.{key} is required")

    active_fixes = run["run"].get("active_fixes", [])
    if not isinstance(active_fixes, list) or not all(
        isinstance(item, str) for item in active_fixes
    ):
        raise ValueError(f"{source}: run.active_fixes must be an array of strings")


def _top1_id(row: dict[str, Any]) -> int | None:
    topk = [x for x in row.get("topk", []) if x.get("token_id") is not None and x.get("logprob") is not None]
    if not topk:
        return None
    return int(max(topk, key=lambda x: float(x["logprob"]))["token_id"])


def _topk_set(row: dict[str, Any]) -> set[int]:
    return {int(x["token_id"]) for x in row.get("topk", []) if x.get("token_id") is not None}


def _jaccard(a: set[int], b: set[int]) -> float | None:
    if not a and not b:
        return None
    u = a | b
    return len(a & b) / len(u) if u else None


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _index(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in sample["positions"]:
        p = int(row["position"])
        if p in out:
            raise ValueError(f"duplicate position {p} within one sample")
        out[p] = row
    return out


def _modal(values: list[int]) -> tuple[int | None, float | None, int]:
    if not values:
        return None, None, 0
    counts = Counter(values)
    token, count = counts.most_common(1)[0]
    return token, count / len(values), len(counts)


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    maps = [_index(s) for s in run["samples"]]
    positions = sorted(set().union(*(m.keys() for m in maps)))
    rows = []
    warnings = []

    for p in positions:
        observed = [m[p] for m in maps if p in m]
        forced_ids = [int(r["forced_token_id"]) for r in observed if r.get("forced_token_id") is not None]
        forced_id = forced_ids[0] if forced_ids else None
        if forced_ids and any(x != forced_id for x in forced_ids):
            warnings.append(f"position {p}: forced_token_id differs across repeats")
            forced_id = None

        lps = [float(r["forced_logprob"]) for r in observed if r.get("forced_logprob") is not None]
        if lps:
            lp_mean, lp_min, lp_max = statistics.fmean(lps), min(lps), max(lps)
            spread = lp_max - lp_min
        else:
            lp_mean = lp_min = lp_max = spread = None

        top1 = [x for x in (_top1_id(r) for r in observed) if x is not None]
        modal_top1, agreement, distinct = _modal(top1)
        sets = [_topk_set(r) for r in observed]
        js = [j for a, b in combinations(sets, 2) if (j := _jaccard(a, b)) is not None]

        rows.append({
            "position": p,
            "forced_token_id": forced_id,
            "observations": len(observed),
            "forced_logprob_mean": lp_mean,
            "forced_logprob_min": lp_min,
            "forced_logprob_max": lp_max,
            "forced_logprob_spread": spread,
            "modal_top1_token_id": modal_top1,
            "top1_agreement_rate": agreement,
            "top1_distinct_count": distinct,
            "pairwise_topk_jaccard_mean": _mean(js),
        })

    first_top1 = next((r["position"] for r in rows if r["top1_agreement_rate"] is not None and r["top1_agreement_rate"] < 1.0), None)
    first_spread = next((r["position"] for r in rows if r["forced_logprob_spread"] is not None and r["forced_logprob_spread"] > 0.0), None)
    spread_rows = [r for r in rows if r["forced_logprob_spread"] is not None]
    max_row = max(spread_rows, key=lambda r: r["forced_logprob_spread"]) if spread_rows else None

    return {
        "label": run["run"].get("arm") or run["run"].get("label") or "run",
        "execution_mode": run["run"].get("execution_mode"),
        "active_fixes": list(run["run"].get("active_fixes", [])),
        "summary": {
            "repeat_count": len(run["samples"]),
            "position_count": len(rows),
            "observed_first_top1_disagreement_position": first_top1,
            "observed_first_nonzero_forced_logprob_spread_position": first_spread,
            "positions_with_top1_disagreement": sum(1 for r in rows if r["top1_agreement_rate"] is not None and r["top1_agreement_rate"] < 1.0),
            "max_forced_logprob_spread": max_row["forced_logprob_spread"] if max_row else None,
            "max_forced_logprob_spread_position": max_row["position"] if max_row else None,
        },
        "positions": rows,
        "warnings": warnings,
    }


def compare_runs(ref_run, cand_run, ref_sum, cand_sum) -> dict[str, Any]:
    if ref_run["prompt"]["sha256"] != cand_run["prompt"]["sha256"]:
        raise ValueError("cannot compare runs: prompt SHA-256 differs")
    if int(ref_run["prompt"]["token_count"]) != int(cand_run["prompt"]["token_count"]):
        raise ValueError("cannot compare runs: prompt token_count differs")

    ref_maps = [_index(s) for s in ref_run["samples"]]
    cand_maps = [_index(s) for s in cand_run["samples"]]
    rr = {r["position"]: r for r in ref_sum["positions"]}
    cr = {r["position"]: r for r in cand_sum["positions"]}
    rows = []

    for p in sorted(set(rr) & set(cr)):
        a, b = rr[p], cr[p]
        delta = None
        if a["forced_logprob_mean"] is not None and b["forced_logprob_mean"] is not None:
            delta = b["forced_logprob_mean"] - a["forced_logprob_mean"]
        ref_pos = [m[p] for m in ref_maps if p in m]
        cand_pos = [m[p] for m in cand_maps if p in m]
        ref_top1 = [x for x in (_top1_id(r) for r in ref_pos) if x is not None]
        cand_top1 = [x for x in (_top1_id(r) for r in cand_pos) if x is not None]
        top1_pairwise = [1.0 if x == y else 0.0 for x, y in product(ref_top1, cand_top1)]
        topk_pairwise = [j for x, y in product([_topk_set(r) for r in ref_pos], [_topk_set(r) for r in cand_pos]) if (j := _jaccard(x, y)) is not None]
        modal_match = None
        if a["modal_top1_token_id"] is not None and b["modal_top1_token_id"] is not None:
            modal_match = a["modal_top1_token_id"] == b["modal_top1_token_id"]
        rows.append({
            "position": p,
            "reference_forced_logprob_mean": a["forced_logprob_mean"],
            "candidate_forced_logprob_mean": b["forced_logprob_mean"],
            "candidate_minus_reference_forced_logprob_mean": delta,
            "abs_forced_logprob_mean_delta": abs(delta) if delta is not None else None,
            "reference_modal_top1_token_id": a["modal_top1_token_id"],
            "candidate_modal_top1_token_id": b["modal_top1_token_id"],
            "modal_top1_match": modal_match,
            "cross_top1_pairwise_agreement_rate": _mean(top1_pairwise),
            "cross_topk_pairwise_jaccard_mean": _mean(topk_pairwise),
        })

    first_modal_mismatch = next((r["position"] for r in rows if r["modal_top1_match"] is False), None)
    delta_rows = [r for r in rows if r["abs_forced_logprob_mean_delta"] is not None]
    max_row = max(delta_rows, key=lambda r: r["abs_forced_logprob_mean_delta"]) if delta_rows else None
    return {
        "reference_label": ref_sum["label"],
        "candidate_label": cand_sum["label"],
        "reference_execution_mode": ref_sum["execution_mode"],
        "candidate_execution_mode": cand_sum["execution_mode"],
        "reference_active_fixes": list(ref_sum.get("active_fixes", [])),
        "candidate_active_fixes": list(cand_sum.get("active_fixes", [])),
        "summary": {
            "common_position_count": len(rows),
            "observed_first_cross_arm_modal_top1_mismatch_position": first_modal_mismatch,
            "positions_with_cross_arm_modal_top1_mismatch": sum(1 for r in rows if r["modal_top1_match"] is False),
            "max_abs_forced_logprob_mean_delta": max_row["abs_forced_logprob_mean_delta"] if max_row else None,
            "max_abs_forced_logprob_mean_delta_position": max_row["position"] if max_row else None,
        },
        "positions": rows,
    }


def build_report(ref_run, cand_run=None) -> dict[str, Any]:
    ref = summarize_run(ref_run)
    cand = summarize_run(cand_run) if cand_run else None
    cross = compare_runs(ref_run, cand_run, ref, cand) if cand_run else None
    return {
        "tool": "vllm-position-parity",
        "tool_version": "0.1.2",
        "measurement_policy": {
            "automatic_bug_label": False,
            "automatic_root_cause": False,
            "universal_significance_threshold": None,
            "note": "Observed measurements only; interpretation is external.",
        },
        "reference": ref,
        "candidate": cand,
        "cross_arm": cross,
    }


def write_key_positions_csv(path: Path, report: dict[str, Any], limit: int = 50) -> None:
    ref = {r["position"]: r for r in report["reference"]["positions"]}
    cand = {r["position"]: r for r in report["candidate"]["positions"]} if report["candidate"] else {}
    cross = {r["position"]: r for r in report["cross_arm"]["positions"]} if report["cross_arm"] else {}
    positions = sorted(set(ref) | set(cand))

    def score(p: int) -> float:
        vals = []
        for row in (ref.get(p), cand.get(p)):
            if row:
                vals.append(float(row.get("forced_logprob_spread") or 0.0))
                if row.get("top1_agreement_rate") is not None:
                    vals.append(1.0 - float(row["top1_agreement_rate"]))
                if row.get("pairwise_topk_jaccard_mean") is not None:
                    vals.append(1.0 - float(row["pairwise_topk_jaccard_mean"]))
        row = cross.get(p)
        if row:
            vals.append(float(row.get("abs_forced_logprob_mean_delta") or 0.0))
            if row.get("cross_top1_pairwise_agreement_rate") is not None:
                vals.append(1.0 - float(row["cross_top1_pairwise_agreement_rate"]))
            if row.get("cross_topk_pairwise_jaccard_mean") is not None:
                vals.append(1.0 - float(row["cross_topk_pairwise_jaccard_mean"]))
        return max(vals) if vals else 0.0

    selected = sorted(positions, key=lambda p: (-score(p), p))[:limit]
    fields = [
        "position", "reference_spread", "reference_top1_agreement", "reference_topk_jaccard",
        "candidate_spread", "candidate_top1_agreement", "candidate_topk_jaccard",
        "cross_abs_mean_logprob_delta", "cross_top1_pairwise_agreement",
        "cross_topk_pairwise_jaccard", "cross_modal_top1_match"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in selected:
            a, b, c = ref.get(p, {}), cand.get(p, {}), cross.get(p, {})
            w.writerow({
                "position": p,
                "reference_spread": a.get("forced_logprob_spread"),
                "reference_top1_agreement": a.get("top1_agreement_rate"),
                "reference_topk_jaccard": a.get("pairwise_topk_jaccard_mean"),
                "candidate_spread": b.get("forced_logprob_spread"),
                "candidate_top1_agreement": b.get("top1_agreement_rate"),
                "candidate_topk_jaccard": b.get("pairwise_topk_jaccard_mean"),
                "cross_abs_mean_logprob_delta": c.get("abs_forced_logprob_mean_delta"),
                "cross_top1_pairwise_agreement": c.get("cross_top1_pairwise_agreement_rate"),
                "cross_topk_pairwise_jaccard": c.get("cross_topk_pairwise_jaccard_mean"),
                "cross_modal_top1_match": c.get("modal_top1_match"),
            })


def write_plot(path: Path, report: dict[str, Any]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[vllm-position-parity] plot skipped: matplotlib unavailable ({exc})", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for summary in (report["reference"], report["candidate"]):
        if not summary:
            continue
        x = [r["position"] for r in summary["positions"] if r["forced_logprob_spread"] is not None]
        y = [r["forced_logprob_spread"] for r in summary["positions"] if r["forced_logprob_spread"] is not None]
        ax.plot(x, y, label=f"{summary['label']}: self-repeat spread")
    if report["cross_arm"]:
        x = [r["position"] for r in report["cross_arm"]["positions"] if r["abs_forced_logprob_mean_delta"] is not None]
        y = [r["abs_forced_logprob_mean_delta"] for r in report["cross_arm"]["positions"] if r["abs_forced_logprob_mean_delta"] is not None]
        ax.plot(x, y, label="cross-arm |mean forced-logprob delta|")
    ax.set_xlabel("Prompt position")
    ax.set_ylabel("Observed logprob difference")
    ax.set_title("Position-resolved inference measurements")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def validate_output_dir(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() and not out.is_dir():
        raise ValueError(
            f"{out}: --out must be a directory, but an existing file was provided"
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Position-resolved measurement for canonical vLLM evidence."
    )
    p.add_argument(
        "reference",
        help="Canonical evidence JSON file (not a directory)",
    )
    p.add_argument(
        "candidate",
        nargs="?",
        help="Optional candidate canonical evidence JSON file",
    )
    p.add_argument(
        "--out",
        default="parity_report",
        help="Output directory (created automatically if it does not exist)",
    )
    p.add_argument("--key-positions", type=int, default=50)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.key_positions < 1:
        p.error("--key-positions must be >= 1")

    try:
        ref = load_run(args.reference)
        cand = load_run(args.candidate) if args.candidate else None
        out = validate_output_dir(args.out)
        report = build_report(ref, cand)

        # Preserve automatic creation of a new output directory.
        out.mkdir(parents=True, exist_ok=True)

        with (out / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

        write_key_positions_csv(
            out / "key_positions.csv",
            report,
            args.key_positions,
        )
        if not args.no_plot:
            write_plot(out / "position_measurements.png", report)

    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"report: {out / 'report.json'}")
    print(f"key positions: {out / 'key_positions.csv'}")
    if not args.no_plot:
        print(
            f"plot: {out / 'position_measurements.png'} "
            "(if matplotlib is available)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

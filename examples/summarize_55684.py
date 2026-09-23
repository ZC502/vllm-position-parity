#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

COMPARISONS = [
    ("A_vs_B", "A_pre_native_tp1", "B_pr_native_tp1", "PR default-path neutrality"),
    ("B_vs_C", "B_pr_native_tp1", "C_pr_requant_tp1", "Intended MXFP8→FP8 PTPC re-quantization residual"),
    ("B_vs_D", "B_pr_native_tp1", "D_pr_native_tp2", "Native MXFP8 TP1↔TP2 sensitivity"),
    ("C_vs_E", "C_pr_requant_tp1", "E_pr_requant_tp2", "Re-quantized FP8 PTPC TP1↔TP2 sensitivity"),
]

def load_vpp_analyzer(vpp_root: Path):
    path = vpp_root / "analyze.py"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location("vpp_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def q(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    p = (len(xs)-1) * frac
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return xs[lo]
    w = p-lo
    return xs[lo]*(1-w)+xs[hi]*w

def modal_completion(run: dict[str, Any]) -> tuple[int, ...]:
    seqs = [tuple(int(x) for x in s.get("completion_token_ids", []) or []) for s in run["samples"]]
    if not seqs:
        return ()
    from collections import Counter
    return Counter(seqs).most_common(1)[0][0]

def first_divergence(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def summarize_prompt(analyze, ref_path: Path, cand_path: Path) -> dict[str, Any]:
    ref = analyze.load_run(ref_path)
    cand = analyze.load_run(cand_path)
    report = analyze.build_report(ref, cand)
    rr = report["reference"]["positions"]
    cr = report["candidate"]["positions"]
    xr = report["cross_arm"]["positions"]

    ref_top1 = [float(r["top1_agreement_rate"]) for r in rr if r.get("top1_agreement_rate") is not None]
    cand_top1 = [float(r["top1_agreement_rate"]) for r in cr if r.get("top1_agreement_rate") is not None]
    spreads = [
        float(r["forced_logprob_spread"])
        for r in rr + cr
        if r.get("forced_logprob_spread") is not None
    ]
    cross_known = [r for r in xr if r.get("modal_top1_match") is not None]
    cross_top1 = (
        sum(1 for r in cross_known if r["modal_top1_match"]) / len(cross_known)
        if cross_known else None
    )
    deltas = [float(r["abs_forced_logprob_mean_delta"]) for r in xr if r.get("abs_forced_logprob_mean_delta") is not None]
    a = modal_completion(ref)
    b = modal_completion(cand)
    return {
        "prompt": ref_path.stem,
        "tokens": int(ref["prompt"]["token_count"]),
        "ref_min_self_top1": min(ref_top1) if ref_top1 else None,
        "cand_min_self_top1": min(cand_top1) if cand_top1 else None,
        "max_self_spread": max(spreads) if spreads else None,
        "cross_top1": cross_top1,
        "p99_abs_logprob_delta": q(deltas, 0.99),
        "max_abs_logprob_delta": max(deltas) if deltas else None,
        "completion_exact": a == b,
        "first_completion_divergence": first_divergence(a, b),
        "reference_completion_len": len(a),
        "candidate_completion_len": len(b),
    }

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [r["cross_top1"] for r in rows if r["cross_top1"] is not None]
    p99s = [r["p99_abs_logprob_delta"] for r in rows if r["p99_abs_logprob_delta"] is not None]
    maxs = [r["max_abs_logprob_delta"] for r in rows if r["max_abs_logprob_delta"] is not None]
    return {
        "pairs": len(rows),
        "min_self_top1_ref": min(r["ref_min_self_top1"] for r in rows if r["ref_min_self_top1"] is not None),
        "min_self_top1_cand": min(r["cand_min_self_top1"] for r in rows if r["cand_min_self_top1"] is not None),
        "max_self_spread": max(r["max_self_spread"] for r in rows if r["max_self_spread"] is not None),
        "min_cross_top1": min(xs) if xs else None,
        "median_cross_top1": statistics.median(xs) if xs else None,
        "max_prompt_p99_abs_logprob_delta": max(p99s) if p99s else None,
        "max_abs_logprob_delta": max(maxs) if maxs else None,
        "completion_exact_count": sum(1 for r in rows if r["completion_exact"]),
        "first_divergence_steps": [
            r["first_completion_divergence"]
            for r in rows if r["first_completion_divergence"] is not None
        ],
    }

def fmt(v: Any, n=3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)

def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize VPP five-arm matrix for vLLM PR #55684")
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--vpp-root", required=True, type=Path)
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    analyze = load_vpp_analyzer(args.vpp_root.resolve())
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    reports_dir = run_root / "comparison_reports"
    reports_dir.mkdir(exist_ok=True)
    result: dict[str, Any] = {
        "schema": "vpp-pr55684-summary/0.1",
        "measurement_policy": {
            "automatic_bug_label": False,
            "automatic_root_cause": False,
            "universal_significance_threshold": None,
            "interpretation": "Differential behavioral evidence only.",
        },
        "manifest": manifest,
        "comparisons": {},
    }

    for key, ref_arm, cand_arm, purpose in COMPARISONS:
        ref_dir = run_root / "arms" / ref_arm / "evidence"
        cand_dir = run_root / "arms" / cand_arm / "evidence"
        ref_files = sorted(ref_dir.glob("p*.json"))
        cand_files = sorted(cand_dir.glob("p*.json"))
        ref_map = {p.name: p for p in ref_files}
        cand_map = {p.name: p for p in cand_files}
        names = sorted(set(ref_map) & set(cand_map))
        if not names:
            raise SystemExit(f"{key}: no overlapping evidence files")
        if set(ref_map) != set(cand_map):
            raise SystemExit(f"{key}: evidence file sets differ")

        rows = [summarize_prompt(analyze, ref_map[n], cand_map[n]) for n in names]
        agg = aggregate(rows)
        result["comparisons"][key] = {
            "purpose": purpose,
            "reference_arm": ref_arm,
            "candidate_arm": cand_arm,
            "aggregate": agg,
            "prompts": rows,
        }

        out_csv = reports_dir / f"{key}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Evidence checksums: canonical JSON + logs + commands + run metadata.
    evidence_manifest = []
    for p in sorted(run_root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"SUMMARY.json", "SUMMARY.md", "PR_COMMENT.md", "provenance.json", "SHA256SUMS.txt"}:
            continue
        rel = p.relative_to(run_root).as_posix()
        evidence_manifest.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    result["evidence_files"] = evidence_manifest

    (run_root / "SUMMARY.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with (run_root / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for item in evidence_manifest:
            f.write(f"{item['sha256']}  {item['path']}\n")

    prov = {
        "schema": "vpp-pr55684-provenance/0.1",
        "purpose": "Implementation-independent VPP differential matrix around vLLM PR #55684",
        "pr": manifest.get("pr"),
        "commits": manifest.get("commits"),
        "model": manifest.get("model"),
        "arms": manifest.get("arms"),
        "comparisons": manifest.get("comparisons"),
        "capture_contract": {
            "same_prompt_cohort": True,
            "same_sampling_contract": True,
            "A_B_C_same_physical_gpu_requested": True,
            "D_E_same_physical_gpu_pair_requested": True,
            "prefix_caching": False,
            "cross_precision_bit_exact_required": False,
        },
        "evidence_file_count": len(evidence_manifest),
        "sha256_manifest": "SHA256SUMS.txt",
    }
    (run_root / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Human summary table.
    headers = [
        ("A_vs_B", "pre-PR native ↔ PR native"),
        ("B_vs_C", "PR native ↔ PR FP8 PTPC"),
        ("B_vs_D", "PR native TP1 ↔ TP2"),
        ("C_vs_E", "PR FP8 PTPC TP1 ↔ TP2"),
    ]
    lines = [
        "# VPP × vLLM PR #55684 — five-arm behavioral matrix",
        "",
        "| comparison | self-repeat top1 ref/cand | max self spread | cross top1 min / med | max prompt-level p99 |Δlogprob| | max |Δ| | greedy exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in headers:
        a = result["comparisons"][key]["aggregate"]
        lines.append(
            f"| {label} | {fmt(a['min_self_top1_ref'])}/{fmt(a['min_self_top1_cand'])} "
            f"| {fmt(a['max_self_spread'],4)} "
            f"| {fmt(a['min_cross_top1'])} / {fmt(a['median_cross_top1'])} "
            f"| {fmt(a['max_prompt_p99_abs_logprob_delta'])} "
            f"| {fmt(a['max_abs_logprob_delta'])} "
            f"| {a['completion_exact_count']}/{a['pairs']} |"
        )

    ab = result["comparisons"]["A_vs_B"]["aggregate"]
    bc = result["comparisons"]["B_vs_C"]["aggregate"]
    bd = result["comparisons"]["B_vs_D"]["aggregate"]
    ce = result["comparisons"]["C_vs_E"]["aggregate"]

    lines += [
        "",
        "## Control-oriented reading",
        "",
        f"- **A↔B default-path control:** max prompt-level p99 |Δlogprob| = {fmt(ab['max_prompt_p99_abs_logprob_delta'])}; "
        f"greedy exact = {ab['completion_exact_count']}/{ab['pairs']}.",
        f"- **B↔C intended re-quantization residual:** max prompt-level p99 |Δlogprob| = {fmt(bc['max_prompt_p99_abs_logprob_delta'])}; "
        f"greedy exact = {bc['completion_exact_count']}/{bc['pairs']}.",
        f"- **Native MXFP8 TP residual (B↔D):** max prompt-level p99 |Δlogprob| = {fmt(bd['max_prompt_p99_abs_logprob_delta'])}; "
        f"greedy exact = {bd['completion_exact_count']}/{bd['pairs']}.",
        f"- **Re-quantized FP8 PTPC TP residual (C↔E):** max prompt-level p99 |Δlogprob| = {fmt(ce['max_prompt_p99_abs_logprob_delta'])}; "
        f"greedy exact = {ce['completion_exact_count']}/{ce['pairs']}.",
        "",
        "> Measurement only: this summary does not label a regression and does not infer root cause.",
    ]
    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Automatically generate a concise PR-ready comment.
    pr_sha = manifest.get("commits", {}).get("pr_head", "UNKNOWN")
    pre_sha = manifest.get("commits", {}).get("pre_pr_baseline", "UNKNOWN")
    vpp_sha = manifest.get("commits", {}).get("vpp", "UNKNOWN")
    comment = [
        "I ran an implementation-independent VPP differential matrix around #55684 to separate "
        "**PR default-path neutrality**, the intended **MXFP8→FP8 PTPC re-quantization residual**, "
        "and **TP sensitivity**.",
        "",
        f"Tested pre-PR baseline: `{pre_sha[:12]}`; PR revision: `{pr_sha[:12]}`; VPP: `{vpp_sha[:12]}`.",
        "",
        "| comparison | self-repeat top1 ref/cand | cross top1 min / med | max prompt-level p99 |Δlogprob| | exact greedy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in headers:
        a = result["comparisons"][key]["aggregate"]
        comment.append(
            f"| {label} | {fmt(a['min_self_top1_ref'])}/{fmt(a['min_self_top1_cand'])} "
            f"| {fmt(a['min_cross_top1'])} / {fmt(a['median_cross_top1'])} "
            f"| {fmt(a['max_prompt_p99_abs_logprob_delta'])} "
            f"| {a['completion_exact_count']}/{a['pairs']} |"
        )

    all_ab_exact = (
        ab["min_cross_top1"] == 1.0
        and ab["max_prompt_p99_abs_logprob_delta"] == 0.0
        and ab["completion_exact_count"] == ab["pairs"]
    )
    comment += ["", "**Control reading:**"]
    if all_ab_exact:
        comment.append(
            "- In this capture, the native MXFP8 path was exactly aligned between the selected "
            "pre-PR baseline and the PR revision at the captured prompt-logprob / greedy-output level."
        )
    else:
        comment.append(
            "- The pre-PR-native ↔ PR-native control itself shows measurable differences in this capture, "
            "so the re-quantization comparison should not be interpreted without that baseline."
        )

    native_p99 = bd["max_prompt_p99_abs_logprob_delta"]
    requant_p99 = ce["max_prompt_p99_abs_logprob_delta"]
    if native_p99 is not None and requant_p99 is not None:
        if requant_p99 <= native_p99:
            comment.append(
                "- The re-quantized TP1↔TP2 residual did not exceed the native-MXFP8 TP control on the "
                "max-prompt-p99 metric in this capture."
            )
        else:
            comment.append(
                "- The re-quantized TP1↔TP2 residual exceeded the native-MXFP8 TP control on the "
                "max-prompt-p99 metric in this capture; this is a measurement signal to isolate further, "
                "not a root-cause claim."
            )

    comment += [
        "",
        "All values above are behavioral measurements only — no universal tolerance or automatic regression label is applied.",
        "",
        "Artifacts: `SUMMARY.md`, `provenance.json`, per-comparison CSVs, canonical evidence, logs, and `SHA256SUMS.txt`.",
    ]
    (run_root / "PR_COMMENT.md").write_text("\n".join(comment) + "\n", encoding="utf-8")

    print((run_root / "SUMMARY.md").read_text(encoding="utf-8"))
    print(f"\nwrote {run_root/'SUMMARY.json'}")
    print(f"wrote {run_root/'SUMMARY.md'}")
    print(f"wrote {run_root/'PR_COMMENT.md'}")
    print(f"wrote {run_root/'provenance.json'}")
    print(f"wrote {run_root/'SHA256SUMS.txt'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

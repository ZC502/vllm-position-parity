#!/usr/bin/env python3
"""Offline analyzer for VPP scheduler/KV canonical evidence."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vpp.scheduler-kv-summary/0.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("evidence_dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expected-repeats", type=int, default=3)
    p.add_argument("--arm-a", default="A_high_capacity")
    p.add_argument("--arm-b", default="B_low_capacity")
    return p.parse_args()


def load_records(evidence_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(evidence_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("schema_version") == "vpp.scheduler-kv/0.1":
            obj["_path"] = str(path)
            out.append(obj)
    return out


def fmt_num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if math.isclose(x, round(x), abs_tol=1e-12):
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def fmt_ratio(x: float | None) -> str:
    return "NA" if x is None else f"{x:.3f}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "NA"
    gib = n / (1024**3)
    return f"{int(round(gib))}GiB" if abs(gib - round(gib)) < 1e-9 else f"{gib:.2f}GiB"


def event_signature(event: dict[str, Any], include_gauges: bool) -> tuple[Any, ...]:
    d = event.get("metrics", {}).get("delta", {})
    base = [
        event.get("event_index"),
        event.get("label"),
        event.get("prompt_sha256"),
        d.get("prefix_cache_queries"),
        d.get("prefix_cache_hits"),
        d.get("num_preemptions"),
        event.get("control", {}).get("output_token_control"),
    ]
    if include_gauges:
        after = event.get("metrics", {}).get("after", {})
        base.extend([
            after.get("kv_cache_usage_perc"),
            after.get("num_requests_running"),
            after.get("num_requests_waiting"),
        ])
    return tuple(base)


def repeat_stability(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    good = [r for r in records if r.get("status") in {"PASS", "INCOMPLETE"} and r.get("events")]
    if len(good) != expected:
        return {"status": "INCOMPLETE", "observed_repeats": len(good), "expected_repeats": expected, "first_repeat_divergence_event": None}
    good.sort(key=lambda r: r.get("repeat", -1))
    counts = {len(r["events"]) for r in good}
    if len(counts) != 1:
        return {"status": "FAIL", "observed_repeats": len(good), "expected_repeats": expected, "first_repeat_divergence_event": "event_count"}
    for idx in range(len(good[0]["events"])):
        sig0 = event_signature(good[0]["events"][idx], include_gauges=True)
        for r in good[1:]:
            if event_signature(r["events"][idx], include_gauges=True) != sig0:
                return {
                    "status": "FAIL",
                    "observed_repeats": len(good),
                    "expected_repeats": expected,
                    "first_repeat_divergence_event": {
                        "event_index": good[0]["events"][idx].get("event_index"),
                        "label": good[0]["events"][idx].get("label"),
                    },
                }
    return {"status": "PASS", "observed_repeats": len(good), "expected_repeats": expected, "first_repeat_divergence_event": None}


def output_control(records: list[dict[str, Any]], expected: int) -> str:
    if len(records) != expected:
        return "INCOMPLETE"
    vals = [r.get("control", {}).get("output_token_control") for r in records if r.get("status") != "ERROR"]
    if len(vals) != expected:
        return "INCOMPLETE"
    return "PASS" if all(v == "PASS" for v in vals) else "FAIL"


def metrics_control(records: list[dict[str, Any]], expected: int) -> str:
    if len(records) != expected:
        return "INCOMPLETE"
    vals = [r.get("control", {}).get("required_prefix_metrics") for r in records]
    return "PASS" if all(v == "PASS" for v in vals) else "INCOMPLETE"


def representative(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    good = [r for r in records if r.get("events")]
    return sorted(good, key=lambda r: r.get("repeat", 10**9))[0] if good else None


def first_cross_arm_divergence(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    ae, be = a.get("events", []), b.get("events", [])
    for left, right in zip(ae, be):
        # Exclude normalized KV usage cross-arm: capacity is intentionally different.
        if event_signature(left, include_gauges=False) != event_signature(right, include_gauges=False):
            ld = left.get("metrics", {}).get("delta", {})
            rd = right.get("metrics", {}).get("delta", {})
            return {
                "event_index": left.get("event_index"),
                "label": left.get("label"),
                "a_prefix_hits_delta": ld.get("prefix_cache_hits"),
                "b_prefix_hits_delta": rd.get("prefix_cache_hits"),
                "a_preemptions_delta": ld.get("num_preemptions"),
                "b_preemptions_delta": rd.get("num_preemptions"),
            }
    if len(ae) != len(be):
        return {"event_index": "event_count", "label": "event_count"}
    return None


def find_event(record: dict[str, Any], label: str) -> dict[str, Any] | None:
    return next((e for e in record.get("events", []) if e.get("label") == label), None)


def collect_event_ratios(records: list[dict[str, Any]], label: str) -> list[float]:
    out = []
    for r in records:
        e = find_event(r, label)
        if e is not None:
            v = e.get("metrics", {}).get("prefix_hit_ratio")
            if v is not None:
                out.append(float(v))
    return out


def collect_sum_counter(records: list[dict[str, Any]], name: str) -> list[float]:
    out = []
    for r in records:
        vals = [e.get("metrics", {}).get("delta", {}).get(name) for e in r.get("events", [])]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            out.append(sum(vals))
    return out


def collect_peak_gauge(records: list[dict[str, Any]], name: str) -> list[float]:
    out = []
    for r in records:
        vals = [e.get("metrics", {}).get("after", {}).get(name) for e in r.get("events", [])]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            out.append(max(vals))
    return out


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def arm_stats(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    final_ratios = collect_event_ratios(records, "seed_replay")
    hits = collect_sum_counter(records, "prefix_cache_hits")
    preempts = collect_sum_counter(records, "num_preemptions")
    peak = collect_peak_gauge(records, "kv_cache_usage_perc")
    return {
        "repeat_stability": repeat_stability(records, expected),
        "output_token_control": output_control(records, expected),
        "required_prefix_metrics": metrics_control(records, expected),
        "seed_replay_hit_ratio": {
            "median": median_or_none(final_ratios),
            "min": min(final_ratios) if final_ratios else None,
            "max": max(final_ratios) if final_ratios else None,
            "n": len(final_ratios),
        },
        "cumulative_prefix_hits": {"median": median_or_none(hits), "n": len(hits)},
        "cumulative_preemptions": {"median": median_or_none(preempts), "n": len(preempts)},
        "peak_kv_cache_usage_perc": {"median": median_or_none(peak), "n": len(peak)},
    }


def markdown_table(rows: list[list[str]]) -> str:
    sep = ["---"] * len(rows[0])
    return "\n".join("| " + " | ".join(r) + " |" for r in [rows[0], sep] + rows[1:])


def plain_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
        if idx == 0:
            out.append("-+-".join("-" * w for w in widths))
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    records = load_records(args.evidence_dir)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_arm.setdefault(str(r.get("arm")), []).append(r)
    for rs in by_arm.values():
        rs.sort(key=lambda r: r.get("repeat", -1))

    a_records, b_records = by_arm.get(args.arm_a, []), by_arm.get(args.arm_b, [])
    a_rep, b_rep = representative(a_records), representative(b_records)
    a_stats, b_stats = arm_stats(a_records, args.expected_repeats), arm_stats(b_records, args.expected_repeats)
    first_div = first_cross_arm_divergence(a_rep, b_rep)

    a_bytes = a_rep.get("runtime", {}).get("kv_cache_memory_bytes") if a_rep else None
    b_bytes = b_rep.get("runtime", {}).get("kv_cache_memory_bytes") if b_rep else None
    model = a_rep.get("runtime", {}).get("model") if a_rep else None
    sha_a = a_rep.get("environment", {}).get("git_head") if a_rep else None
    sha_b = b_rep.get("environment", {}).get("git_head") if b_rep else None
    wh_a = a_rep.get("workload", {}).get("sha256") if a_rep else None
    wh_b = b_rep.get("workload", {}).get("sha256") if b_rep else None
    workload_match = wh_a is not None and wh_a == wh_b
    git_head_match = sha_a is not None and sha_a == sha_b

    out_ctrl = "PASS" if a_stats["output_token_control"] == b_stats["output_token_control"] == "PASS" else (
        "FAIL" if "FAIL" in {a_stats["output_token_control"], b_stats["output_token_control"]} else "INCOMPLETE"
    )
    cross_status = "OBSERVED" if first_div is not None else ("NOT_OBSERVED" if a_records and b_records else "INCOMPLETE")
    first_text = "NONE" if first_div is None else f"E{first_div['event_index']} {first_div['label']}"
    ra = a_stats["seed_replay_hit_ratio"]["median"]
    rb = b_stats["seed_replay_hit_ratio"]["median"]
    pa = a_stats["cumulative_preemptions"]["median"]
    pb = b_stats["cumulative_preemptions"]["median"]

    rows = [[
        "Comparison", "A self", "B self", "Output ctrl", "First observable div", "Seed replay hit A/B", "Preempt A/B"
    ], [
        f"{fmt_bytes(a_bytes)} ↔ {fmt_bytes(b_bytes)}",
        a_stats["repeat_stability"]["status"],
        b_stats["repeat_stability"]["status"],
        out_ctrl,
        first_text,
        f"{fmt_ratio(ra)} / {fmt_ratio(rb)}",
        f"{fmt_num(pa)} / {fmt_num(pb)}",
    ]]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "comparison": {
            "arm_a": args.arm_a,
            "arm_b": args.arm_b,
            "kv_cache_memory_bytes": {"a": a_bytes, "b": b_bytes},
            "model": model,
            "expected_repeats": args.expected_repeats,
            "workload_match": workload_match,
            "git_head_match": git_head_match,
            "output_token_control": out_ctrl,
            "cross_arm_state_divergence": cross_status,
            "first_observable_cross_arm_divergence": first_div,
        },
        "arms": {args.arm_a: a_stats, args.arm_b: b_stats},
        "evidence_files": [r["_path"] for r in records],
        "interpretation_note": "First divergence is over request-boundary cache/preemption observables, not internal scheduler transition state.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# VPP Scheduler/KV Pilot Summary", "", markdown_table(rows), "", "## Controls", "",
        f"- Workload hash match: **{'PASS' if workload_match else 'FAIL'}**",
        f"- Git HEAD match across arms: **{'PASS' if git_head_match else 'FAIL'}**",
        f"- Output-token control: **{out_ctrl}**",
        f"- A required prefix metrics: **{a_stats['required_prefix_metrics']}**",
        f"- B required prefix metrics: **{b_stats['required_prefix_metrics']}**",
        "", "## Cross-arm observation", "",
        f"- State divergence at request-boundary observables: **{cross_status}**",
        f"- First observable divergence: **{first_text}**",
        f"- Final seed replay hit ratio, A median: `{fmt_ratio(ra)}`",
        f"- Final seed replay hit ratio, B median: `{fmt_ratio(rb)}`",
        "", "> This does not identify the first internal scheduler transition. It identifies the first request boundary where selected cache/preemption observables differ.",
        "", "No root cause is assigned by this report.", "",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    sha = sha_a or sha_b or "unknown"
    slack = [
        "Small downstream #47922 Native Simulator pilot:", "", "```text", plain_table(rows), "```", "",
        f"- model: `{model or 'unknown'}`",
        f"- simulator commit: `{sha}`",
        f"- repeats: `{args.expected_repeats}` fresh processes per arm",
        f"- workload hash match: `{'PASS' if workload_match else 'FAIL'}`",
        f"- output-token control: `{out_ctrl}`", "",
        "The comparison holds the token-ID workload and prescribed simulated outputs fixed and changes only the configured KV-cache capacity.", "",
        "“First observable divergence” is request-boundary evidence from prefix-cache / preemption metrics, not an internal scheduler-transition trace.",
    ]
    (args.out_dir / "SLACK_RESULT.md").write_text("\n".join(slack) + "\n", encoding="utf-8")

    print(plain_table(rows))
    print(f"\nWrote {args.out_dir / 'SUMMARY.md'}")
    print(f"Wrote {args.out_dir / 'SUMMARY.json'}")
    print(f"Wrote {args.out_dir / 'SLACK_RESULT.md'}")

    complete = (
        len(a_records) == args.expected_repeats
        and len(b_records) == args.expected_repeats
        and a_stats["required_prefix_metrics"] == "PASS"
        and b_stats["required_prefix_metrics"] == "PASS"
    )
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())

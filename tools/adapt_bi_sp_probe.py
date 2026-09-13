#!/usr/bin/env python3
"""Offline adapter for the vLLM #56370 ``bi_sp_probe.py`` output.

This adapter is intentionally stdlib-only and does not import vLLM.

It converts an external batch-composition probe JSON into a draft canonical
position-parity evidence shape for the trace domain
``decode_sampled_logprob``.  Crucially, it does *not* copy the probe's
``verdicts`` into its own measurements.  All comparisons are independently
recomputed from ``runs.*`` and the source verdicts are retained only as a
validation oracle.

The adapter also separates two facts that a single ``logprob_mismatch_tokens``
count conflates:

* same-token logprob drift: the sampled token id is unchanged, but its sampled
  logprob differs;
* different-token divergence: the sampled token id itself differs.

That distinction is descriptive only.  The adapter does not infer root cause,
apply significance thresholds, or label a run as buggy/unstable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ADAPTER_NAME = "adapt_bi_sp_probe"
ADAPTER_VERSION = "0.1.0"
ADAPTER_SCHEMA_VERSION = "0.2-draft.1"
TRACE_DOMAIN = "decode_sampled_logprob"

RUN_IDS = ("bs1_a", "bsN_a", "bs1_b", "bsN_b")
PAIR_SPECS = (
    ("bs1_a", "bs1_b", "repeat_stability", "isolated"),
    ("bsN_a", "bsN_b", "repeat_stability", "batched"),
    ("bs1_a", "bsN_a", "batch_composition", "replicate_a"),
    ("bs1_b", "bsN_b", "batch_composition", "replicate_b"),
)


class AdapterError(ValueError):
    """Raised for malformed or semantically incompatible probe evidence."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AdapterError(f"{path}: input file does not exist")
    if path.is_dir():
        raise AdapterError(f"{path}: expected a probe JSON file, received a directory")
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as exc:
        raise AdapterError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from None
    except OSError as exc:
        raise AdapterError(f"{path}: could not read input: {exc}") from None
    if not isinstance(obj, dict):
        raise AdapterError(f"{path}: top-level JSON value must be an object")
    return obj


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_prompt_record(
    raw: Any, *, run_id: str, prompt_idx: int
) -> tuple[list[int], list[float]]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise AdapterError(
            f"runs.{run_id}[{prompt_idx}]: expected [token_ids, sampled_logprobs]"
        )
    token_ids, logprobs = raw
    if not isinstance(token_ids, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in token_ids
    ):
        raise AdapterError(
            f"runs.{run_id}[{prompt_idx}][0]: token_ids must be an array of integers"
        )
    if not isinstance(logprobs, list) or not all(_is_number(x) for x in logprobs):
        raise AdapterError(
            f"runs.{run_id}[{prompt_idx}][1]: sampled_logprobs must be numeric"
        )
    if len(token_ids) != len(logprobs):
        raise AdapterError(
            f"runs.{run_id}[{prompt_idx}]: token_ids/logprobs length mismatch "
            f"({len(token_ids)} != {len(logprobs)})"
        )
    return list(token_ids), [float(x) for x in logprobs]


def _normalize_runs(doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_runs = doc.get("runs")
    if not isinstance(raw_runs, dict):
        raise AdapterError("missing or invalid top-level 'runs' object")

    missing = [run_id for run_id in RUN_IDS if run_id not in raw_runs]
    if missing:
        raise AdapterError(f"runs: missing required run(s): {', '.join(missing)}")

    prompt_counts: dict[str, int] = {}
    out: dict[str, list[dict[str, Any]]] = {}
    for run_id in RUN_IDS:
        raw = raw_runs[run_id]
        if not isinstance(raw, list):
            raise AdapterError(f"runs.{run_id}: expected an array of prompt traces")
        prompt_counts[run_id] = len(raw)
        traces: list[dict[str, Any]] = []
        for prompt_idx, record in enumerate(raw):
            token_ids, logprobs = _normalize_prompt_record(
                record, run_id=run_id, prompt_idx=prompt_idx
            )
            traces.append(
                {
                    "prompt_idx": prompt_idx,
                    "token_ids": token_ids,
                    "sampled_logprobs": logprobs,
                }
            )
        out[run_id] = traces

    if not prompt_counts or len(set(prompt_counts.values())) != 1:
        rendered = ", ".join(f"{k}={v}" for k, v in prompt_counts.items())
        raise AdapterError(
            "prompt identity cannot be preserved because run prompt counts differ: "
            + rendered
        )
    if next(iter(prompt_counts.values()), 0) == 0:
        raise AdapterError("runs contain zero prompts")
    return out


def _first_or_none(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    return items[0] if items else None


def _comparison(
    traces: dict[str, list[dict[str, Any]]],
    left_run: str,
    right_run: str,
    axis: str,
    scope: str,
    *,
    detail_limit: int = 12,
) -> dict[str, Any]:
    left = traces[left_run]
    right = traces[right_run]
    if len(left) != len(right):
        # _normalize_runs already prevents this, but keep the comparison self-contained.
        raise AdapterError(
            f"cannot compare {left_run} vs {right_run}: prompt counts differ"
        )

    compared_positions = 0
    token_sequence_mismatch_prompts = 0
    length_mismatch_prompts = 0
    sampled_logprob_mismatch_positions = 0
    same_token_logprob_mismatch_positions = 0
    different_token_logprob_mismatch_positions = 0
    different_sampled_token_positions = 0
    unpaired_positions = 0
    max_abs_delta = 0.0

    lp_mismatches: list[dict[str, Any]] = []
    same_token_lp_mismatches: list[dict[str, Any]] = []
    different_token_lp_mismatches: list[dict[str, Any]] = []
    token_id_disagreements: list[dict[str, Any]] = []

    # Source-compatible form of first_mismatches.  This is recomputed from raw
    # traces; it is never copied from the probe's source verdicts.
    source_style_first_mismatches: list[list[Any]] = []

    for prompt_idx, (a, b) in enumerate(zip(left, right)):
        ta = a["token_ids"]
        tb = b["token_ids"]
        la = a["sampled_logprobs"]
        lb = b["sampled_logprobs"]

        if ta != tb:
            token_sequence_mismatch_prompts += 1
        if len(ta) != len(tb):
            length_mismatch_prompts += 1
            unpaired_positions += abs(len(ta) - len(tb))

        # Match bi_sp_probe.py's verdict semantics exactly: compare the common
        # decode-step prefix using zip().
        for step, (lp_a, lp_b) in enumerate(zip(la, lb)):
            compared_positions += 1
            tok_a = ta[step]
            tok_b = tb[step]
            token_same = tok_a == tok_b

            if not token_same:
                different_sampled_token_positions += 1
                token_entry = {
                    "prompt_idx": prompt_idx,
                    "decode_step": step,
                    "token_id_a": tok_a,
                    "token_id_b": tok_b,
                    "sampled_logprob_a": lp_a,
                    "sampled_logprob_b": lp_b,
                    "abs_logprob_delta": abs(lp_a - lp_b),
                }
                if len(token_id_disagreements) < detail_limit:
                    token_id_disagreements.append(token_entry)

            if lp_a != lp_b:
                sampled_logprob_mismatch_positions += 1
                delta = abs(lp_a - lp_b)
                max_abs_delta = max(max_abs_delta, delta)
                entry = {
                    "prompt_idx": prompt_idx,
                    "decode_step": step,
                    "token_id_a": tok_a,
                    "token_id_b": tok_b,
                    "same_sampled_token": token_same,
                    "sampled_logprob_a": lp_a,
                    "sampled_logprob_b": lp_b,
                    "abs_logprob_delta": delta,
                }
                if len(lp_mismatches) < detail_limit:
                    lp_mismatches.append(entry)
                if len(source_style_first_mismatches) < detail_limit:
                    source_style_first_mismatches.append(
                        [prompt_idx, step, lp_a, lp_b, delta]
                    )
                if token_same:
                    same_token_logprob_mismatch_positions += 1
                    if len(same_token_lp_mismatches) < detail_limit:
                        same_token_lp_mismatches.append(entry)
                else:
                    different_token_logprob_mismatch_positions += 1
                    if len(different_token_lp_mismatches) < detail_limit:
                        different_token_lp_mismatches.append(entry)

    if (
        same_token_logprob_mismatch_positions
        + different_token_logprob_mismatch_positions
        != sampled_logprob_mismatch_positions
    ):
        raise AssertionError("internal mismatch partition invariant failed")

    return {
        "pair": f"{left_run}_vs_{right_run}",
        "comparison_axis": axis,
        "comparison_scope": scope,
        "left_run": left_run,
        "right_run": right_run,
        "summary": {
            "prompt_count": len(left),
            "compared_positions": compared_positions,
            "token_sequence_mismatch_prompts": token_sequence_mismatch_prompts,
            "length_mismatch_prompts": length_mismatch_prompts,
            "unpaired_positions": unpaired_positions,
            "sampled_logprob_mismatch_positions": sampled_logprob_mismatch_positions,
            "same_token_logprob_mismatch_positions": same_token_logprob_mismatch_positions,
            "different_token_logprob_mismatch_positions": different_token_logprob_mismatch_positions,
            "different_sampled_token_positions": different_sampled_token_positions,
            "max_abs_sampled_logprob_delta": max_abs_delta,
            "first_sampled_logprob_mismatch": _first_or_none(lp_mismatches),
            "first_same_token_logprob_mismatch": _first_or_none(
                same_token_lp_mismatches
            ),
            "first_different_token_logprob_mismatch": _first_or_none(
                different_token_lp_mismatches
            ),
            "first_sampled_token_id_disagreement": _first_or_none(
                token_id_disagreements
            ),
        },
        "examples": {
            "sampled_logprob_mismatches": lp_mismatches,
            "same_token_logprob_mismatches": same_token_lp_mismatches,
            "different_token_logprob_mismatches": different_token_lp_mismatches,
            "sampled_token_id_disagreements": token_id_disagreements,
        },
        "source_compatible_recomputed_verdict": {
            "pair": f"{left_run}_vs_{right_run}",
            "tokens": compared_positions,
            "token_seq_mismatch_prompts": token_sequence_mismatch_prompts,
            "logprob_mismatch_tokens": sampled_logprob_mismatch_positions,
            "max_abs_diff": max_abs_delta,
            "first_mismatches": source_style_first_mismatches,
        },
    }


def _float_equal(a: Any, b: Any) -> bool:
    if not _is_number(a) or not _is_number(b):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def _nested_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return _float_equal(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_nested_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_nested_equal(a[k], b[k]) for k in a)
    return a == b


def _validate_against_source_verdicts(
    source_verdicts: Any, comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    if source_verdicts is None:
        return {
            "source_verdicts_present": False,
            "recomputed_matches_source": None,
            "pairs": [],
            "errors": [],
        }
    if not isinstance(source_verdicts, list):
        raise AdapterError("top-level 'verdicts' must be an array when present")

    source_by_pair: dict[str, dict[str, Any]] = {}
    for i, verdict in enumerate(source_verdicts):
        if not isinstance(verdict, dict) or not isinstance(verdict.get("pair"), str):
            raise AdapterError(f"verdicts[{i}]: expected object with string 'pair'")
        source_by_pair[verdict["pair"]] = verdict

    pair_results: list[dict[str, Any]] = []
    errors: list[str] = []
    keys = (
        "pair",
        "tokens",
        "token_seq_mismatch_prompts",
        "logprob_mismatch_tokens",
        "max_abs_diff",
        "first_mismatches",
    )

    for comparison in comparisons:
        pair = comparison["pair"]
        recomputed = comparison["source_compatible_recomputed_verdict"]
        source = source_by_pair.get(pair)
        if source is None:
            msg = f"source verdict missing pair {pair}"
            errors.append(msg)
            pair_results.append({"pair": pair, "matches": False, "differences": [msg]})
            continue

        differences: list[str] = []
        for key in keys:
            if key not in source:
                differences.append(f"source missing field {key}")
                continue
            if not _nested_equal(recomputed[key], source[key]):
                differences.append(
                    f"{key}: recomputed={recomputed[key]!r} source={source[key]!r}"
                )
        matches = not differences
        if not matches:
            errors.extend(f"{pair}: {d}" for d in differences)
        pair_results.append(
            {"pair": pair, "matches": matches, "differences": differences}
        )

    expected_pairs = {c["pair"] for c in comparisons}
    extra_pairs = sorted(set(source_by_pair) - expected_pairs)
    for pair in extra_pairs:
        errors.append(f"source verdict contains unexpected pair {pair}")

    return {
        "source_verdicts_present": True,
        "recomputed_matches_source": not errors,
        "pairs": pair_results,
        "errors": errors,
    }


def _select_present(config: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: config[key] for key in keys if key in config}


def _derived_observations(traces: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = {run_id: len(traces[run_id]) for run_id in RUN_IDS}
    lengths: list[int] = []
    per_run_steps: dict[str, list[int]] = {}
    for run_id in RUN_IDS:
        steps = [len(t["token_ids"]) for t in traces[run_id]]
        per_run_steps[run_id] = steps
        lengths.extend(steps)
    return {
        "prompt_count_by_run": counts,
        "prompt_count": next(iter(counts.values())),
        "decode_steps_per_prompt_min": min(lengths),
        "decode_steps_per_prompt_max": max(lengths),
        "all_prompt_trace_lengths_equal": len(set(lengths)) == 1,
        "decode_steps_per_run": {
            run_id: sum(per_run_steps[run_id]) for run_id in RUN_IDS
        },
    }


def adapt_document(
    doc: dict[str, Any],
    *,
    source_name: str | None = None,
    include_traces: bool = True,
    detail_limit: int = 12,
    fail_on_source_verdict_mismatch: bool = True,
) -> dict[str, Any]:
    config = doc.get("config")
    if not isinstance(config, dict):
        raise AdapterError("missing or invalid top-level 'config' object")

    traces = _normalize_runs(doc)
    comparisons = [
        _comparison(
            traces,
            left,
            right,
            axis,
            scope,
            detail_limit=detail_limit,
        )
        for left, right, axis, scope in PAIR_SPECS
    ]

    validation = _validate_against_source_verdicts(doc.get("verdicts"), comparisons)
    if fail_on_source_verdict_mismatch and validation["recomputed_matches_source"] is False:
        details = "; ".join(validation["errors"][:4])
        if len(validation["errors"]) > 4:
            details += f"; ... ({len(validation['errors'])} total differences)"
        raise AdapterError(
            "independent recomputation does not match source verdicts: " + details
        )

    source: dict[str, Any] = {
        "format": "bi_sp_probe",
        "source_config": config,
        "source_verdicts": doc.get("verdicts"),
    }
    if source_name:
        source["source_name"] = source_name

    # Keep only facts that are explicitly present.  In particular, do not infer
    # missing cudagraph_mode / N / reverse fields from notes or from other runs.
    resolved_execution_config = _select_present(
        config,
        ("enable_sp", "sp_min_token_num", "prefix_caching", "cudagraph_mode"),
    )
    execution_context = _select_present(
        config,
        ("tp", "sp", "backend", "batch_invariant", "nccl_algo", "reverse", "N"),
    )

    out: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "tool": "vllm-position-parity",
        "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
        "measurement_policy": {
            "trace_domain": TRACE_DOMAIN,
            "automatic_bug_label": False,
            "automatic_root_cause": False,
            "universal_significance_threshold": None,
            "comparison_semantics": (
                "bitwise equality on sampled-token logprob; token-id divergence "
                "reported separately"
            ),
        },
        "source": source,
        "run": {
            "arm": config.get("cond", "bi_sp_probe"),
            "probe_order": ["bs1_a", "bsN_a", "bs1_b", "bsN_b"],
            "resolved_execution_config": resolved_execution_config,
            "execution_context": execution_context,
            "note": config.get("note"),
        },
        "derived_observations": _derived_observations(traces),
        "comparisons": comparisons,
        "validation": validation,
    }
    if include_traces:
        out["traces"] = traces
    return out


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        raise AdapterError(f"{path}: --out must be a file, not a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        raise AdapterError(f"{path}: could not write output: {exc}") from None


def _print_summary(adapted: dict[str, Any]) -> None:
    print(
        f"trace_domain={adapted['measurement_policy']['trace_domain']} "
        f"arm={adapted['run']['arm']}"
    )
    for c in adapted["comparisons"]:
        s = c["summary"]
        print(
            f"{c['pair']}: compared={s['compared_positions']} "
            f"lp_mismatch={s['sampled_logprob_mismatch_positions']} "
            f"same_token_drift={s['same_token_logprob_mismatch_positions']} "
            f"different_token_lp={s['different_token_logprob_mismatch_positions']} "
            f"token_id_diff={s['different_sampled_token_positions']} "
            f"sequence_mismatch_prompts={s['token_sequence_mismatch_prompts']}"
        )
    v = adapted["validation"]
    if v["source_verdicts_present"]:
        print(f"source_verdict_match={v['recomputed_matches_source']}")
    else:
        print("source_verdict_match=n/a (source verdicts absent)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Adapt vLLM #56370 bi_sp_probe JSON into draft decode-sampled-logprob "
            "position-parity evidence and independently recompute all verdicts."
        )
    )
    p.add_argument("input", type=Path, help="bi_sp_probe per-condition JSON file")
    p.add_argument("--out", type=Path, required=True, help="output canonical JSON file")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="omit raw traces from output (comparisons and source verdicts are retained)",
    )
    p.add_argument(
        "--detail-limit",
        type=int,
        default=12,
        help="maximum examples retained per mismatch category (default: 12)",
    )
    p.add_argument(
        "--allow-source-verdict-mismatch",
        action="store_true",
        help=(
            "write output even if independent recomputation differs from source verdicts; "
            "validation.errors will record the discrepancy"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.detail_limit < 0:
        print("error: --detail-limit must be >= 0", file=sys.stderr)
        return 2
    try:
        doc = _load_json(args.input)
        adapted = adapt_document(
            doc,
            source_name=args.input.name,
            include_traces=not args.summary_only,
            detail_limit=args.detail_limit,
            fail_on_source_verdict_mismatch=not args.allow_source_verdict_mismatch,
        )
        _write_json(args.out, adapted)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_summary(adapted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

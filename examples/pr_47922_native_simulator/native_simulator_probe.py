#!/usr/bin/env python3
"""
VPP-style request-boundary probe for vLLM PR #47922 Native Simulator.

Uses existing simulator/vLLM observables only.  "First divergence" in the
paired analyzer means the first divergence at an observable request boundary,
not the first internal scheduler transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vpp.scheduler-kv/0.1"
PROFILE = "pr47922_native_simulator_kv_capacity"

COUNTER_METRICS = {
    "prefix_cache_queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"),
    "prefix_cache_hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"),
    "num_preemptions": ("vllm:num_preemptions", "vllm:num_preemptions_total"),
    "prompt_tokens": ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
}
GAUGE_METRICS = {
    "kv_cache_usage_perc": ("vllm:kv_cache_usage_perc",),
    "num_requests_running": ("vllm:num_requests_running",),
    "num_requests_waiting": ("vllm:num_requests_waiting",),
}


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect one fresh-run scheduler/KV evidence record.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--repeat", required=True, type=int)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--kv-cache-memory-bytes", required=True, type=int)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--prompt-len", type=int, default=896)
    p.add_argument("--churn-count", type=int, default=12)
    p.add_argument("--output-tokens", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--source-repo", type=Path, default=None)
    p.add_argument("--pr-number", type=int, default=47922)
    p.add_argument("--print-workload", action="store_true")
    return p.parse_args()


def git_value(repo: Path | None, *args: str) -> str | None:
    if repo is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def make_prompt(event_id: int, prompt_len: int) -> list[int]:
    if prompt_len < 32:
        raise ValueError("--prompt-len must be >= 32")
    base = 1000 + event_id * 137
    span = 127
    return [base + ((j * 17) % span) for j in range(prompt_len)]


def make_output_tokens(event_id: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("--output-tokens must be >= 1")
    base = 501 + event_id * 7
    return [base + i for i in range(count)]


def build_workload(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed_prompt = make_prompt(0, args.prompt_len)
    workload = [{
        "event_index": 0,
        "label": "seed",
        "prompt_token_ids": seed_prompt,
        "simulated_output_token_ids": make_output_tokens(0, args.output_tokens),
    }]
    for i in range(1, args.churn_count + 1):
        workload.append({
            "event_index": i,
            "label": f"churn_{i:02d}",
            "prompt_token_ids": make_prompt(i, args.prompt_len),
            "simulated_output_token_ids": make_output_tokens(i, args.output_tokens),
        })
    workload.append({
        "event_index": args.churn_count + 1,
        "label": "seed_replay",
        "prompt_token_ids": seed_prompt,
        "simulated_output_token_ids": make_output_tokens(0, args.output_tokens),
    })
    for event in workload:
        event["prompt_sha256"] = sha256_json(event["prompt_token_ids"])
        event["prompt_tokens"] = len(event["prompt_token_ids"])
    return workload


def metric_name(metric: Any) -> str | None:
    name = getattr(metric, "name", None)
    return str(name) if name is not None else None


def metric_number(metric: Any) -> float | None:
    value = getattr(metric, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_metric_objects(llm: Any) -> list[Any]:
    try:
        llm.llm_engine.do_log_stats()
    except Exception:
        pass
    try:
        return list(llm.get_metrics())
    except Exception:
        return []


def read_metric(metrics: list[Any], aliases: tuple[str, ...]) -> float | None:
    values = []
    for metric in metrics:
        if metric_name(metric) in aliases:
            value = metric_number(metric)
            if value is not None:
                values.append(value)
    return float(sum(values)) if values else None


def snapshot_metrics(llm: Any) -> dict[str, float | None]:
    metrics = collect_metric_objects(llm)
    out: dict[str, float | None] = {}
    for key, aliases in COUNTER_METRICS.items():
        out[key] = read_metric(metrics, aliases)
    for key, aliases in GAUGE_METRICS.items():
        out[key] = read_metric(metrics, aliases)
    return out


def delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def extract_token_ids(outputs: Any) -> list[int] | None:
    try:
        first = outputs[0]
    except Exception:
        return None
    try:
        return [int(x) for x in first.outputs[0].token_ids]
    except Exception:
        pass
    try:
        candidate = first[0][0]
        if isinstance(candidate, (list, tuple)):
            return [int(x) for x in candidate]
    except Exception:
        pass
    return None


def environment_provenance(args: argparse.Namespace) -> dict[str, Any]:
    source_repo = args.source_repo.resolve() if args.source_repo else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_repo": str(source_repo) if source_repo else None,
        "git_head": git_value(source_repo, "rev-parse", "HEAD"),
        "git_branch": git_value(source_repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": git_value(source_repo, "status", "--porcelain"),
        "triton_cpu_backend": os.environ.get("TRITON_CPU_BACKEND"),
        "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
    }


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    workload = build_workload(args)
    if args.print_workload:
        print(json.dumps([{k: e[k] for k in (
            "event_index", "label", "prompt_tokens", "prompt_sha256", "simulated_output_token_ids"
        )} for e in workload], indent=2))
        return 0

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "source": {
            "collector": "native_simulator_inprocess",
            "upstream_pr": args.pr_number,
            "note": "Request-boundary scheduler/KV evidence; no internal scheduler transition instrumentation.",
        },
        "arm": args.arm,
        "repeat": args.repeat,
        "runtime": {
            "model": args.model,
            "simulate_forward": True,
            "enable_prefix_caching": True,
            "disable_hybrid_kv_cache_manager": False,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
        },
        "workload": {
            "id": "prefix_pressure_v0",
            "seed": args.seed,
            "prompt_len": args.prompt_len,
            "churn_count": args.churn_count,
            "output_tokens_per_request": args.output_tokens,
            "event_count": len(workload),
            "sha256": sha256_json([{
                "label": e["label"],
                "prompt_sha256": e["prompt_sha256"],
                "simulated_output_token_ids": e["simulated_output_token_ids"],
            } for e in workload]),
        },
        "environment": environment_provenance(args),
        "events": [],
        "control": {},
        "status": "RUNNING",
    }

    try:
        import vllm  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
        record["environment"]["vllm_version"] = getattr(vllm, "__version__", None)
        record["environment"]["vllm_module"] = getattr(vllm, "__file__", None)

        llm = LLM(
            model=args.model,
            simulate_forward=True,
            disable_hybrid_kv_cache_manager=False,
            enable_prefix_caching=True,
            disable_log_stats=False,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            max_model_len=args.max_model_len,
        )

        before = snapshot_metrics(llm)
        output_controls: list[bool] = []
        for event in workload:
            expected = list(event["simulated_output_token_ids"])
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=args.output_tokens,
                ignore_eos=True,
                detokenize=False,
                extra_args={"simulated_output_token_ids": expected},
            )
            t0 = time.perf_counter()
            outputs = llm.generate([event["prompt_token_ids"]], sampling_params=sampling)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            after = snapshot_metrics(llm)
            observed = extract_token_ids(outputs)
            output_ok = observed == expected
            output_controls.append(output_ok)

            event_record = {
                "event_index": event["event_index"],
                "label": event["label"],
                "prompt_tokens": event["prompt_tokens"],
                "prompt_sha256": event["prompt_sha256"],
                "simulated_output_token_ids_expected": expected,
                "output_token_ids_observed": observed,
                "control": {"output_token_control": "PASS" if output_ok else "FAIL"},
                "metrics": {
                    "before": before,
                    "after": after,
                    "delta": {
                        "prefix_cache_queries": delta(after.get("prefix_cache_queries"), before.get("prefix_cache_queries")),
                        "prefix_cache_hits": delta(after.get("prefix_cache_hits"), before.get("prefix_cache_hits")),
                        "num_preemptions": delta(after.get("num_preemptions"), before.get("num_preemptions")),
                        "prompt_tokens": delta(after.get("prompt_tokens"), before.get("prompt_tokens")),
                    },
                    "prefix_hit_ratio": None,
                },
                "elapsed_ms": elapsed_ms,
            }
            q = event_record["metrics"]["delta"]["prefix_cache_queries"]
            h = event_record["metrics"]["delta"]["prefix_cache_hits"]
            if q is not None and h is not None and q > 0:
                event_record["metrics"]["prefix_hit_ratio"] = h / q
            record["events"].append(event_record)
            before = after

        metrics_supported = all(
            e["metrics"]["delta"]["prefix_cache_queries"] is not None
            and e["metrics"]["delta"]["prefix_cache_hits"] is not None
            for e in record["events"]
        )
        record["control"] = {
            "output_token_control": "PASS" if all(output_controls) else "FAIL",
            "required_prefix_metrics": "PASS" if metrics_supported else "INCOMPLETE",
        }
        record["status"] = (
            "PASS" if record["control"]["output_token_control"] == "PASS"
            and record["control"]["required_prefix_metrics"] == "PASS" else "INCOMPLETE"
        )
        write_json_atomic(args.out, record)
        print(f"[{args.arm} repeat={args.repeat}] {record['status']} events={len(record['events'])} -> {args.out}")
        return 0 if record["status"] == "PASS" else 3
    except Exception as exc:
        record["status"] = "ERROR"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.out, record)
        print(f"[{args.arm} repeat={args.repeat}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

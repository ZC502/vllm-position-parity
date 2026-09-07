#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "0.1.0"
SCHEMA_VERSION = "0.1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ints(values: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in values).encode("ascii")).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit_near(path: str | Path) -> str | None:
    p = Path(path).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
            except Exception:
                return None
    return None


def parse_metadata(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--metadata expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty metadata key in {item!r}")
        out[key] = value
    return out


def logprob_value(obj: Any) -> float:
    if hasattr(obj, "logprob"):
        return float(obj.logprob)
    if isinstance(obj, dict) and "logprob" in obj:
        return float(obj["logprob"])
    return float(obj)


def normalize_one_result(result: Any, repeat_index: int) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    prompt_token_ids = [int(x) for x in result.prompt_token_ids]
    prompt_logprobs = result.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("result.prompt_logprobs is None; set prompt_logprobs=K")
    if len(prompt_logprobs) != len(prompt_token_ids):
        raise RuntimeError("prompt_logprobs length does not match prompt_token_ids")

    positions = []
    for pos, (forced_id, candidates) in enumerate(zip(prompt_token_ids, prompt_logprobs)):
        if candidates is None:
            positions.append({
                "position": pos,
                "forced_token_id": forced_id,
                "forced_logprob": None,
                "topk": [],
            })
            continue

        topk = []
        forced_logprob = None
        for token_id, lp_obj in candidates.items():
            token_id = int(token_id)
            lp = logprob_value(lp_obj)
            topk.append({"token_id": token_id, "logprob": lp})
            if token_id == forced_id:
                forced_logprob = lp
        topk.sort(key=lambda x: x["logprob"], reverse=True)
        if forced_logprob is None:
            warnings.append(
                f"repeat {repeat_index}, position {pos}: forced token {forced_id} "
                "was absent from returned prompt_logprobs"
            )
        positions.append({
            "position": pos,
            "forced_token_id": forced_id,
            "forced_logprob": forced_logprob,
            "topk": topk,
        })

    completion_ids = []
    if getattr(result, "outputs", None):
        completion_ids = [int(x) for x in getattr(result.outputs[0], "token_ids", [])]

    return {
        "repeat_index": repeat_index,
        "positions": positions,
        "completion_token_ids": completion_ids,
        "completion_sha256": sha256_ints(completion_ids),
        "num_cached_tokens": getattr(result, "num_cached_tokens", None),
        "num_cache_creation_tokens": getattr(result, "num_cache_creation_tokens", None),
    }, warnings


def collect_environment(vllm_module: Any) -> dict[str, Any]:
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "vllm_version": package_version("vllm"),
        "vllm_commit": git_commit_near(Path(vllm_module.__file__).parent),
        "torch_version": package_version("torch"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # Probe accelerator only after LLM construction.
    try:
        import torch
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = getattr(torch.version, "cuda", None)
            env["accelerators"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            env["accelerators"] = []
    except Exception as exc:
        env["accelerator_probe_error"] = repr(exc)
    return env




def best_effort_resolved_config(llm: Any) -> dict[str, Any]:
    """Read a small stable subset of resolved vLLM config when available.

    This is best-effort only: internal object paths may vary across vLLM versions,
    so failures are recorded rather than treated as collector errors.
    """
    out: dict[str, Any] = {}
    try:
        engine = getattr(llm, "llm_engine", None)
        cfg = getattr(engine, "vllm_config", None)
        if cfg is None:
            return {"available": False}
        model = getattr(cfg, "model_config", None)
        cache = getattr(cfg, "cache_config", None)
        parallel = getattr(cfg, "parallel_config", None)
        scheduler = getattr(cfg, "scheduler_config", None)

        def take(dst: str, obj: Any, attr: str):
            if obj is not None and hasattr(obj, attr):
                value = getattr(obj, attr)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[dst] = value
                else:
                    out[dst] = str(value)

        take("dtype", model, "dtype")
        take("quantization", model, "quantization")
        take("max_model_len", model, "max_model_len")
        take("enforce_eager", model, "enforce_eager")
        take("logprobs_mode", model, "logprobs_mode")
        take("kv_cache_dtype", cache, "cache_dtype")
        take("enable_prefix_caching", cache, "enable_prefix_caching")
        take("tensor_parallel_size", parallel, "tensor_parallel_size")
        take("pipeline_parallel_size", parallel, "pipeline_parallel_size")
        take("data_parallel_size", parallel, "data_parallel_size")
        take("max_num_batched_tokens", scheduler, "max_num_batched_tokens")
        out["available"] = True
    except Exception as exc:
        out = {"available": False, "probe_error": repr(exc)}
    return out

def main() -> int:
    p = argparse.ArgumentParser(description="Collect repeated vLLM prompt_logprobs into canonical JSON.")
    p.add_argument("--model", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="run")
    p.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="sequential")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--prompt-logprobs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int)
    pg = p.add_mutually_exclusive_group()
    pg.add_argument("--enable-prefix-caching", action="store_true")
    pg.add_argument("--disable-prefix-caching", action="store_true")
    p.add_argument("--llm-kwargs-json", default="{}")
    p.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.prompt_logprobs == 0 or args.prompt_logprobs < -1:
        p.error("--prompt-logprobs must be a positive integer or -1")
    if args.max_tokens < 1:
        p.error("--max-tokens must be >= 1")

    prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
    try:
        llm_kwargs = json.loads(args.llm_kwargs_json)
    except json.JSONDecodeError as exc:
        p.error(f"invalid --llm-kwargs-json: {exc}")
    if not isinstance(llm_kwargs, dict):
        p.error("--llm-kwargs-json must decode to an object")
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    elif args.disable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = False
    try:
        metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        p.error(str(exc))

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"vLLM import failed: {exc}", file=sys.stderr)
        return 2

    # Create LLM before querying CUDA from torch.
    llm = LLM(model=args.model, **llm_kwargs)
    sp_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_logprobs": args.prompt_logprobs,
    }
    if args.seed is not None:
        sp_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sp_kwargs)

    if args.execution_mode == "sequential":
        raw_results = []
        for _ in range(args.repeats):
            batch = llm.generate([prompt], sampling_params, use_tqdm=False)
            if len(batch) != 1:
                raise RuntimeError(f"expected one result, received {len(batch)}")
            raw_results.append(batch[0])
    else:
        raw_results = llm.generate([prompt] * args.repeats, sampling_params, use_tqdm=False)
        if len(raw_results) != args.repeats:
            raise RuntimeError(f"expected {args.repeats} results, received {len(raw_results)}")

    samples, warnings = [], []
    for i, result in enumerate(raw_results):
        sample, ws = normalize_one_result(result, i)
        samples.append(sample)
        warnings.extend(ws)

    prompt_token_ids = [int(x) for x in raw_results[0].prompt_token_ids]
    for i, result in enumerate(raw_results[1:], 1):
        if [int(x) for x in result.prompt_token_ids] != prompt_token_ids:
            raise RuntimeError(f"prompt tokenization differs at repeat {i}")

    resolved_vllm_config = best_effort_resolved_config(llm)
    logprobs_mode = resolved_vllm_config.get("logprobs_mode")
    if logprobs_mode is None and isinstance(llm_kwargs.get("logprobs_mode"), str):
        logprobs_mode = llm_kwargs["logprobs_mode"]

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "tool": "vllm-position-parity",
        "collector_version": TOOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"framework": "vllm", "collector": "offline_python_api"},
        "environment": collect_environment(vllm),
        "resolved_vllm_config": resolved_vllm_config,
        "run": {
            "arm": args.arm,
            "execution_mode": args.execution_mode,
            "repeats": args.repeats,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_logprobs_k": args.prompt_logprobs,
            "logprobs_mode": logprobs_mode,
            "seed": args.seed,
            "llm_kwargs": llm_kwargs,
            "metadata": metadata,
        },
        "prompt": {
            "sha256": sha256_text(prompt),
            "token_count": len(prompt_token_ids),
            "token_ids_sha256": sha256_ints(prompt_token_ids),
            "raw_text_stored": False,
        },
        "samples": samples,
        "collection_warnings": warnings,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"wrote {out}")
    print(f"arm={args.arm} mode={args.execution_mode} repeats={args.repeats} tokens={len(prompt_token_ids)}")
    if warnings:
        print(f"warnings={len(warnings)}; inspect collection_warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

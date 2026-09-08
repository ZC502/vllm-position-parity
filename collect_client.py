#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "0.1.1"
SCHEMA_VERSION = "0.1"
DEFAULT_TIMEOUT_S = 300.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ints(values: list[int]) -> str:
    payload = ",".join(str(x) for x in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def parse_metadata(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--metadata expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty metadata key in {item!r}")
        out[key] = value
    return out


def build_completions_url(base_url: str) -> str:
    """Accept host root, /v1, or the full /v1/completions URL."""
    base = base_url.rstrip("/")
    if base.endswith("/v1/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/completions"
    return base + "/v1/completions"


def _short_error_body(raw: bytes, limit: int = 2000) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


def fetch_completion(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """POST one request. Intentionally performs no automatic retry."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = _short_error_body(exc.read())
        detail = f": {body}" if body else ""
        raise RuntimeError(f"HTTP {exc.code}{detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection failed: {exc.reason}") from None
    except (TimeoutError, socket.timeout):
        raise RuntimeError(f"request timed out after {timeout_s:g}s") from None

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "server response was not valid UTF-8; expected a JSON response "
            "from vLLM /v1/completions"
        ) from exc

    try:
        result = json.loads(decoded)
    except json.JSONDecodeError as exc:
        preview = decoded[:500].replace("\n", "\\n")
        raise RuntimeError(
            "server response was not valid JSON; expected vLLM "
            f"/v1/completions output. Response prefix: {preview!r}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"server returned JSON type {type(result).__name__}; expected an object"
        )
    return result


def logprob_value(obj: Any) -> float:
    if isinstance(obj, dict):
        if "logprob" not in obj:
            raise ValueError("candidate object has no 'logprob' field")
        return float(obj["logprob"])
    return float(obj)


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "API response contains no non-empty 'choices' list. "
            "Confirm that the endpoint is vLLM's /v1/completions endpoint."
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("API response choices[0] is not a JSON object")
    return choice


def _extract_prompt_fields(
    response: dict[str, Any],
    choice: dict[str, Any],
    repeat_index: int,
) -> tuple[list[int], list[Any], str, list[str]]:
    """Extract vLLM prompt fields with narrowly scoped compatibility fallbacks.

    Current vLLM /v1/completions places prompt_token_ids and prompt_logprobs
    directly on CompletionResponseChoice. Two fallbacks are accepted only to
    make version mismatches diagnosable; they are recorded as warnings.
    """
    warnings: list[str] = []

    prompt_token_ids = choice.get("prompt_token_ids")
    prompt_logprobs = choice.get("prompt_logprobs")
    layout = "choice_fields"

    if prompt_token_ids is None and "prompt_token_ids" in response:
        prompt_token_ids = response.get("prompt_token_ids")
        layout = "response_fields"
        warnings.append(
            f"repeat {repeat_index}: using response-level prompt_token_ids "
            "compatibility fallback"
        )

    if prompt_logprobs is None and "prompt_logprobs" in response:
        prompt_logprobs = response.get("prompt_logprobs")
        layout = "response_fields"
        warnings.append(
            f"repeat {repeat_index}: using response-level prompt_logprobs "
            "compatibility fallback"
        )

    if prompt_logprobs is None:
        legacy_logprobs = choice.get("logprobs")
        if isinstance(legacy_logprobs, dict) and "prompt_logprobs" in legacy_logprobs:
            prompt_logprobs = legacy_logprobs.get("prompt_logprobs")
            layout = "legacy_nested_logprobs"
            warnings.append(
                f"repeat {repeat_index}: using legacy nested "
                "choice.logprobs.prompt_logprobs compatibility fallback"
            )

    if prompt_token_ids is None:
        raise RuntimeError(
            "API response is missing 'prompt_token_ids'. The collector sends "
            "return_token_ids=true; confirm that this vLLM server/version exposes "
            "prompt_token_ids on /v1/completions. "
            f"Observed choice keys: {sorted(choice.keys())}"
        )

    if prompt_logprobs is None:
        raise RuntimeError(
            "API response is missing 'prompt_logprobs'. The collector requests "
            "prompt_logprobs explicitly; confirm that this vLLM server/version "
            "supports prompt_logprobs on /v1/completions. "
            f"Observed choice keys: {sorted(choice.keys())}"
        )

    if not isinstance(prompt_token_ids, list):
        raise RuntimeError(
            "'prompt_token_ids' is not a list; incompatible response layout"
        )
    if not isinstance(prompt_logprobs, list):
        raise RuntimeError(
            "'prompt_logprobs' is not a list; incompatible response layout"
        )

    try:
        token_ids = [int(x) for x in prompt_token_ids]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "'prompt_token_ids' contains a value that cannot be converted to int"
        ) from exc

    if len(prompt_logprobs) != len(token_ids):
        raise RuntimeError(
            "prompt_logprobs length does not match prompt_token_ids length: "
            f"{len(prompt_logprobs)} vs {len(token_ids)}. "
            "Refusing to produce position-aligned evidence from an incompatible "
            "response."
        )

    return token_ids, prompt_logprobs, layout, warnings


def _parse_candidate_token_id(token_key: Any, lp_info: Any) -> int:
    try:
        return int(token_key)
    except (TypeError, ValueError):
        if isinstance(lp_info, dict) and "token_id" in lp_info:
            try:
                return int(lp_info["token_id"])
            except (TypeError, ValueError):
                pass
        raise RuntimeError(
            "could not recover an integer token_id from a prompt_logprobs "
            f"candidate key {token_key!r}; response layout may be incompatible "
            "with this collector"
        )


def normalize_client_result(
    response: dict[str, Any],
    repeat_index: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    choice = _first_choice(response)

    prompt_token_ids, prompt_logprobs, layout, layout_warnings = _extract_prompt_fields(
        response, choice, repeat_index
    )
    warnings.extend(layout_warnings)

    positions: list[dict[str, Any]] = []
    for pos, (forced_id, candidates) in enumerate(
        zip(prompt_token_ids, prompt_logprobs)
    ):
        # vLLM normally returns None for the first prompt token because there is
        # no previous prompt position from which to score it.
        if candidates is None:
            positions.append(
                {
                    "position": pos,
                    "forced_token_id": forced_id,
                    "forced_logprob": None,
                    "topk": [],
                }
            )
            continue

        if not isinstance(candidates, dict):
            raise RuntimeError(
                f"repeat {repeat_index}, position {pos}: prompt_logprobs entry "
                f"has type {type(candidates).__name__}, expected object or null"
            )

        topk: list[dict[str, Any]] = []
        forced_logprob: float | None = None

        for token_key, lp_info in candidates.items():
            token_id = _parse_candidate_token_id(token_key, lp_info)
            try:
                lp = logprob_value(lp_info)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"repeat {repeat_index}, position {pos}, token {token_id}: "
                    "could not parse logprob"
                ) from exc

            topk.append({"token_id": token_id, "logprob": lp})
            if token_id == forced_id:
                forced_logprob = lp

        topk.sort(key=lambda item: item["logprob"], reverse=True)

        if forced_logprob is None:
            warnings.append(
                f"repeat {repeat_index}, position {pos}: forced token {forced_id} "
                "was absent from returned prompt_logprobs"
            )

        positions.append(
            {
                "position": pos,
                "forced_token_id": forced_id,
                "forced_logprob": forced_logprob,
                "topk": topk,
            }
        )

    completion_token_ids = choice.get("token_ids")
    if completion_token_ids is None and "token_ids" in response:
        completion_token_ids = response.get("token_ids")
        warnings.append(
            f"repeat {repeat_index}: using response-level token_ids "
            "compatibility fallback"
        )

    sample: dict[str, Any] = {
        "repeat_index": repeat_index,
        "positions": positions,
    }

    if completion_token_ids is None:
        sample["completion_token_ids"] = []
        warnings.append(
            f"repeat {repeat_index}: completion token_ids missing despite "
            "return_token_ids=true; completion hash omitted"
        )
    else:
        if not isinstance(completion_token_ids, list):
            raise RuntimeError(
                "'token_ids' is not a list; incompatible completion response layout"
            )
        try:
            completion_ids = [int(x) for x in completion_token_ids]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "'token_ids' contains a value that cannot be converted to int"
            ) from exc
        sample["completion_token_ids"] = completion_ids
        sample["completion_sha256"] = sha256_ints(completion_ids)

    usage = response.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            sample["num_cached_tokens"] = details.get("cached_tokens")
            sample["num_cache_creation_tokens"] = details.get(
                "created_cache_tokens"
            )
        else:
            sample["num_cached_tokens"] = None
            sample["num_cache_creation_tokens"] = None
    else:
        sample["num_cached_tokens"] = None
        sample["num_cache_creation_tokens"] = None

    sample["response_layout"] = layout
    return sample, warnings


def _resolve_api_key(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    if args.api_key is not None:
        return args.api_key
    if args.api_key_env is not None:
        value = os.environ.get(args.api_key_env)
        if not value:
            parser.error(
                f"--api-key-env requested environment variable "
                f"{args.api_key_env!r}, but it is unset or empty"
            )
        return value
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Collect repeated vLLM prompt_logprobs from a running "
            "vLLM-compatible /v1/completions endpoint."
        )
    )
    p.add_argument(
        "--base-url",
        required=True,
        help=(
            "vLLM server root, /v1 URL, or full /v1/completions URL. "
            "The raw URL is not written to evidence."
        ),
    )
    p.add_argument("--model", required=True, help="Served model name")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="run")
    p.add_argument(
        "--execution-mode",
        choices=["sequential", "concurrent"],
        default="sequential",
    )
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--prompt-logprobs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=(
            f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_S:g}). "
            "Automatic retries are intentionally disabled."
        ),
    )
    p.add_argument(
        "--endpoint-label",
        default="default",
        help=(
            "Non-sensitive endpoint identifier stored in evidence instead of "
            "the raw URL."
        ),
    )
    p.add_argument(
        "--logprobs-mode",
        default=None,
        help=(
            "Optional metadata-only label for the server's logprobs_mode. "
            "The client does not infer this value."
        ),
    )
    p.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")

    auth = p.add_mutually_exclusive_group()
    auth.add_argument(
        "--api-key",
        help=(
            "Bearer API key. Never written to output or logs. "
            "For shell-history safety, prefer --api-key-env."
        ),
    )
    auth.add_argument(
        "--api-key-env",
        metavar="ENV_VAR",
        help=(
            "Read the Bearer API key from this explicitly named environment "
            "variable. The key and variable name are not written to evidence."
        ),
    )

    args = p.parse_args()

    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.prompt_logprobs == 0 or args.prompt_logprobs < -1:
        p.error("--prompt-logprobs must be a positive integer or -1")
    if args.max_tokens < 1:
        p.error("--max-tokens must be >= 1")
    if args.timeout <= 0:
        p.error("--timeout must be > 0")

    prompt = (
        args.prompt
        if args.prompt is not None
        else Path(args.prompt_file).read_text(encoding="utf-8")
    )

    try:
        metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        p.error(str(exc))

    api_key = _resolve_api_key(args, p)
    endpoint_url = build_completions_url(args.base_url)

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_logprobs": args.prompt_logprobs,
        "return_token_ids": True,
        "stream": False,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    request_concurrency = (
        args.repeats if args.execution_mode == "concurrent" else 1
    )
    print(
        "sending requests "
        f"(endpoint_label={args.endpoint_label!r}, "
        f"mode={args.execution_mode}, repeats={args.repeats}, "
        f"request_concurrency={request_concurrency}, "
        f"timeout={args.timeout:g}s, retries=0)"
    )

    raw_results: list[dict[str, Any] | None] = [None] * args.repeats

    if args.execution_mode == "sequential":
        for repeat_index in range(args.repeats):
            try:
                raw_results[repeat_index] = fetch_completion(
                    endpoint_url, headers, payload, args.timeout
                )
            except Exception as exc:
                print(
                    f"request {repeat_index} failed: {exc}",
                    file=sys.stderr,
                )
                return 1
    else:
        executor = ThreadPoolExecutor(max_workers=args.repeats)
        future_to_index = {
            executor.submit(
                fetch_completion,
                endpoint_url,
                headers,
                payload,
                args.timeout,
            ): repeat_index
            for repeat_index in range(args.repeats)
        }
        failed = False
        try:
            for future in as_completed(future_to_index):
                repeat_index = future_to_index[future]
                try:
                    raw_results[repeat_index] = future.result()
                except Exception as exc:
                    failed = True
                    print(
                        f"concurrent request {repeat_index} failed: {exc}",
                        file=sys.stderr,
                    )
                    for pending in future_to_index:
                        pending.cancel()
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if failed:
            return 1

    if any(result is None for result in raw_results):
        print(
            "collector did not receive all expected responses; no evidence written",
            file=sys.stderr,
        )
        return 1

    samples: list[dict[str, Any]] = []
    warnings: list[str] = []

    for repeat_index, response in enumerate(raw_results):
        assert response is not None
        try:
            sample, sample_warnings = normalize_client_result(
                response, repeat_index
            )
        except Exception as exc:
            print(
                f"response {repeat_index} is incompatible with the expected "
                f"vLLM prompt_logprobs layout: {exc}",
                file=sys.stderr,
            )
            return 1
        samples.append(sample)
        warnings.extend(sample_warnings)

    prompt_token_ids = [
        int(position["forced_token_id"]) for position in samples[0]["positions"]
    ]

    for repeat_index, sample in enumerate(samples[1:], start=1):
        current_ids = [
            int(position["forced_token_id"]) for position in sample["positions"]
        ]
        if current_ids != prompt_token_ids:
            print(
                f"prompt tokenization differs at repeat {repeat_index}; "
                "refusing to combine non-aligned requests into one evidence file",
                file=sys.stderr,
            )
            return 1

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "tool": "vllm-position-parity",
        "collector_version": TOOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "framework": "vllm",
            "collector": "openai_http",
            "api": "vllm_openai_compatible_completions",
        },
        "environment": {
            "scope": "collector_host",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        # A remote client cannot truthfully introspect the server's resolved
        # launch/runtime configuration.
        "resolved_vllm_config": None,
        "run": {
            "arm": args.arm,
            "execution_mode": args.execution_mode,
            "metadata_source": "client_observed_plus_user_supplied",
            "repeats": args.repeats,
            "request_concurrency": request_concurrency,
            "concurrency_semantics": (
                "client_requests_not_guaranteed_server_batch"
                if args.execution_mode == "concurrent"
                else "sequential_client_requests"
            ),
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_logprobs_k": args.prompt_logprobs,
            "logprobs_mode": args.logprobs_mode,
            "seed": args.seed,
            "llm_kwargs": {},
            "endpoint_label": args.endpoint_label,
            "endpoint_url_stored": False,
            "request_timeout_s": args.timeout,
            "automatic_retries": 0,
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
        json.dump(
            evidence,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(f"wrote {out}")
    print(
        f"arm={args.arm} mode={args.execution_mode} "
        f"repeats={args.repeats} tokens={len(prompt_token_ids)}"
    )
    if warnings:
        print(
            f"warnings={len(warnings)}; inspect collection_warnings in output"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

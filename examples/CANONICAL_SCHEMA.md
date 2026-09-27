# Canonical Evidence Contract — `vpp.scheduler-kv/0.1`

## Run-level fields

`schema_version`, `profile`, `source`, `arm`, `repeat`, `runtime`, `workload`, `environment`, `events[]`, `control`, `status`.

The run records the actual local vLLM git SHA when `VLLM_REPO` is supplied.

## Event-level fields

`event_index`, `label`, `prompt_tokens`, `prompt_sha256`, prescribed/observed token IDs, output-token control, metric snapshots, per-request deltas, prefix-hit ratio, and elapsed time.

Counters are differenced per request: `prefix_cache_queries`, `prefix_cache_hits`, `num_preemptions`, `prompt_tokens`.

Gauges are recorded after requests: `kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`.

`elapsed_ms` is provenance/debug information only. It is not part of the repeat/cross-arm signatures.

## Comparison semantics

- **self repeat stability**: deterministic request-boundary signature agrees across fresh-process repeats.
- **first observable cross-arm divergence**: first request where selected cache/preemption observables differ.
- **seed replay hit ratio**: cached-token hits / queried tokens when the initial seed prompt is replayed at the end.
- **output-token control**: observed token IDs equal the caller-prescribed simulated IDs.

The cross-arm signature intentionally excludes normalized KV utilization because KV capacity is the experimental variable.

No causal attribution, universal threshold, or bug label is produced.

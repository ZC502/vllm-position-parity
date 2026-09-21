from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("kvq", ROOT / "profiles" / "kv_cache_quant.py")
kvq = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kvq)


def run_obj(label: str, kv: str, top1_flip: bool = False, lp_shift: float = 0.0):
    positions = []
    forced_ids = [10, 11, 12, 13]
    for i, tid in enumerate(forced_ids):
        top = tid if not (top1_flip and i == 2) else 999
        positions.append({
            "position": i,
            "forced_token_id": tid,
            "forced_logprob": -0.1 * i + lp_shift,
            "topk": [
                {"token_id": top, "logprob": -0.1 * i + lp_shift},
                {"token_id": tid if top != tid else 777, "logprob": -1.0},
            ],
        })
    samples = []
    for r in range(2):
        samples.append({
            "repeat_index": r,
            "positions": positions,
            "completion_token_ids": [21, 22, 23],
            "completion_sha256": "x",
            "num_cached_tokens": 0,
            "num_cache_creation_tokens": 4,
        })
    return {
        "schema_version": "0.1",
        "tool": "vllm-position-parity",
        "collector_version": "0.1.2",
        "environment": {
            "vllm_version": "test",
            "vllm_commit": "abc",
            "torch_version": "test",
            "cuda_version": "13.0",
            "accelerators": [{"index": 0, "name": "GPU", "capability": [12, 0]}],
        },
        "resolved_vllm_config": {
            "kv_cache_dtype": kv,
            "quantization": None,
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_batched_tokens": 8192,
        },
        "run": {
            "arm": label,
            "execution_mode": "sequential",
            "repeats": 2,
            "model": "m",
            "temperature": 0.0,
            "max_tokens": 3,
            "prompt_logprobs_k": 2,
            "seed": 0,
            "llm_kwargs": {},
            "active_fixes": [],
            "metadata": {},
        },
        "prompt": {"sha256": "same", "token_count": 4, "raw_text_stored": False},
        "samples": samples,
        "collection_warnings": [],
    }


class TestKVProfile(unittest.TestCase):
    def write_run(self, td: Path, name: str, obj: dict) -> Path:
        p = td / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    def test_exact_cross_precision(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            a = self.write_run(td, "a.json", run_obj("bf16", "auto"))
            b = self.write_run(td, "b.json", run_obj("fp8", "fp8"))
            pair = kvq.analyze_pair("short", a, b, None, None)
            self.assertEqual(pair["cross_precision"]["cross_modal_top1_agreement_rate"], 1.0)
            self.assertEqual(pair["cross_precision"]["max_abs_forced_logprob_delta"], 0.0)
            self.assertEqual(pair["quality_gate"]["status"], "NOT_EVALUATED")

    def test_gate_can_fail(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            a = self.write_run(td, "a.json", run_obj("bf16", "auto"))
            b = self.write_run(td, "b.json", run_obj("fp8", "fp8", top1_flip=True, lp_shift=0.2))
            policy = {
                "policy_name": "test",
                "cross_precision": {
                    "min_modal_top1_agreement_rate": 0.99,
                    "drift_fraction": {"epsilon": 0.05, "max_fraction": 0.05},
                },
            }
            pair = kvq.analyze_pair("boundary", a, b, 0.05, policy)
            self.assertLess(pair["cross_precision"]["cross_modal_top1_agreement_rate"], 0.99)
            self.assertEqual(pair["quality_gate"]["status"], "FAIL")

    def test_hardware_mismatch_is_warning_not_fake_root_cause(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            ra = run_obj("bf16", "auto")
            rb = run_obj("fp8", "fp8")
            rb["environment"]["accelerators"][0]["capability"] = [9, 0]
            a = self.write_run(td, "a.json", ra)
            b = self.write_run(td, "b.json", rb)
            pair = kvq.analyze_pair("short", a, b, None, None)
            self.assertTrue(any("accelerator set differs" in x for x in pair["comparability_warnings"]))


if __name__ == "__main__":
    unittest.main()

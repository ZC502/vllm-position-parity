#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

spec = importlib.util.spec_from_file_location(
    "adapt_bi_sp_probe", ROOT / "tools" / "adapt_bi_sp_probe.py",
)

FIXTURES = ROOT / "tests" / "fixtures" / "bi_sp_probe"
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def load(name: str):
    with (FIXTURES / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def by_pair(adapted):
    return {c["pair"]: c for c in adapted["comparisons"]}


class AdapterTests(unittest.TestCase):
    def test_control_recomputes_all_zero(self):
        out = mod.adapt_document(
            load("tp4_sp_off_prefix_off_control.json"),
            source_name="control.json",
        )
        self.assertEqual(
            out["measurement_policy"]["trace_domain"], "decode_sampled_logprob"
        )
        self.assertTrue(out["validation"]["recomputed_matches_source"])
        self.assertEqual(out["derived_observations"]["prompt_count"], 64)
        self.assertEqual(out["derived_observations"]["decode_steps_per_prompt_min"], 24)
        self.assertEqual(out["derived_observations"]["decode_steps_per_prompt_max"], 24)
        for c in out["comparisons"]:
            s = c["summary"]
            self.assertEqual(s["compared_positions"], 1536)
            self.assertEqual(s["sampled_logprob_mismatch_positions"], 0)
            self.assertEqual(s["same_token_logprob_mismatch_positions"], 0)
            self.assertEqual(s["different_token_logprob_mismatch_positions"], 0)
            self.assertEqual(s["different_sampled_token_positions"], 0)
            self.assertEqual(s["token_sequence_mismatch_prompts"], 0)

    def test_failing_reproduces_source_and_splits_mismatch_semantics(self):
        out = mod.adapt_document(
            load("tp4_sp_on_prefix_off_failing.json"),
            source_name="failing.json",
        )
        self.assertTrue(out["validation"]["recomputed_matches_source"])
        pairs = by_pair(out)

        expected = {
            "bs1_a_vs_bs1_b": (0, 0, 0, 0, 0),
            # lp mismatch, same-token lp drift, different-token lp mismatch,
            # different token positions, token-sequence mismatch prompts
            "bsN_a_vs_bsN_b": (1197, 1090, 107, 107, 11),
            "bs1_a_vs_bsN_a": (1528, 1419, 109, 109, 12),
            "bs1_b_vs_bsN_b": (1527, 1425, 102, 102, 11),
        }
        for pair, values in expected.items():
            s = pairs[pair]["summary"]
            self.assertEqual(s["compared_positions"], 1536)
            self.assertEqual(s["sampled_logprob_mismatch_positions"], values[0])
            self.assertEqual(s["same_token_logprob_mismatch_positions"], values[1])
            self.assertEqual(s["different_token_logprob_mismatch_positions"], values[2])
            self.assertEqual(s["different_sampled_token_positions"], values[3])
            self.assertEqual(s["token_sequence_mismatch_prompts"], values[4])

        # The first observable logprob drift occurs while the sampled token is
        # still identical, which is precisely the semantic distinction this
        # adapter is intended to preserve.
        first = pairs["bsN_a_vs_bsN_b"]["summary"][
            "first_sampled_logprob_mismatch"
        ]
        self.assertEqual(first["prompt_idx"], 0)
        self.assertEqual(first["decode_step"], 0)
        self.assertTrue(first["same_sampled_token"])
        self.assertEqual(first["token_id_a"], first["token_id_b"])

    def test_missing_optional_resolved_fields_are_not_invented(self):
        out = mod.adapt_document(load("tp4_sp_on_prefix_off_failing.json"))
        resolved = out["run"]["resolved_execution_config"]
        self.assertEqual(resolved["enable_sp"], True)
        self.assertEqual(resolved["sp_min_token_num"], 1)
        self.assertEqual(resolved["prefix_caching"], False)
        self.assertNotIn("cudagraph_mode", resolved)
        self.assertNotIn("N", out["run"]["execution_context"])
        self.assertNotIn("reverse", out["run"]["execution_context"])

    def test_source_verdicts_are_validation_oracle_not_measurement_input(self):
        doc = load("tp4_sp_on_prefix_off_failing.json")
        doc["verdicts"][1]["logprob_mismatch_tokens"] = 999999
        with self.assertRaises(mod.AdapterError):
            mod.adapt_document(doc)
        out = mod.adapt_document(doc, fail_on_source_verdict_mismatch=False)
        self.assertFalse(out["validation"]["recomputed_matches_source"])
        self.assertEqual(
            by_pair(out)["bsN_a_vs_bsN_b"]["summary"][
                "sampled_logprob_mismatch_positions"
            ],
            1197,
        )


if __name__ == "__main__":
    unittest.main()

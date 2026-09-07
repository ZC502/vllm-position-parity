import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vpp_analyze", ROOT / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze)


class AnalyzeFixtureTests(unittest.TestCase):
    def load(self, name):
        return analyze.load_run(ROOT / "fixtures" / name)

    def test_exact_is_bit_stable(self):
        report = analyze.build_report(self.load("exact.json"))
        s = report["reference"]["summary"]
        self.assertIsNone(s["observed_first_top1_disagreement_position"])
        self.assertIsNone(s["observed_first_nonzero_forced_logprob_spread_position"])
        self.assertEqual(s["positions_with_top1_disagreement"], 0)
        self.assertEqual(s["max_forced_logprob_spread"], 0.0)

    def test_noisy_has_spread_without_top1_flip(self):
        report = analyze.build_report(self.load("noisy.json"))
        s = report["reference"]["summary"]
        self.assertIsNotNone(s["observed_first_nonzero_forced_logprob_spread_position"])
        self.assertIsNone(s["observed_first_top1_disagreement_position"])
        self.assertGreater(s["max_forced_logprob_spread"], 0.0)

    def test_injected_flip_begins_at_24(self):
        report = analyze.build_report(self.load("injected_flip.json"))
        s = report["reference"]["summary"]
        self.assertEqual(s["observed_first_top1_disagreement_position"], 24)
        self.assertGreater(s["positions_with_top1_disagreement"], 0)

    def test_cross_arm_detects_injected_modal_mismatch(self):
        report = analyze.build_report(self.load("exact.json"), self.load("injected_flip.json"))
        self.assertEqual(
            report["cross_arm"]["summary"]["observed_first_cross_arm_modal_top1_mismatch_position"],
            24,
        )

    def test_schedule_axis(self):
        report = analyze.build_report(self.load("exact.json"), self.load("concurrent_only.json"))
        self.assertEqual(report["reference"]["execution_mode"], "sequential")
        self.assertEqual(report["candidate"]["execution_mode"], "concurrent")
        self.assertIsNotNone(report["candidate"]["summary"]["observed_first_top1_disagreement_position"])

    def test_prompt_mismatch_rejected(self):
        a = self.load("exact.json")
        b = json.loads(json.dumps(self.load("injected_flip.json")))
        b["prompt"]["sha256"] = "different"
        with self.assertRaises(ValueError):
            analyze.build_report(a, b)


if __name__ == "__main__":
    unittest.main()

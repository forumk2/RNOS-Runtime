from __future__ import annotations

import unittest

from rnos.evaluation.domain_legitimacy import EXAMPLE_SCENARIOS, evaluate_claim


class DomainLegitimacyTests(unittest.TestCase):
    def test_multiple_domains_return_distinct_objections(self) -> None:
        scenario = EXAMPLE_SCENARIOS["retry_storm_detection"]
        result = evaluate_claim(scenario["claim"], scenario["context"])

        domain_results = result["domain_results"]
        self.assertGreaterEqual(len(domain_results), 4)

        objections = {entry["core_objection"] for entry in domain_results}
        self.assertGreater(len(objections), 1)

        verdicts = {entry["verdict"] for entry in domain_results}
        self.assertGreater(len(verdicts), 1)

    def test_aggregation_surfaces_non_trivial_gaps(self) -> None:
        scenario = EXAMPLE_SCENARIOS["policy_threshold_logic"]
        result = evaluate_claim(scenario["claim"], scenario["context"])

        aggregate = result["aggregate"]
        self.assertIn(aggregate["overall_verdict"], {"invalid", "unclear", "partially_valid"})
        self.assertGreaterEqual(len(aggregate["critical_gaps"]), 2)
        self.assertTrue(any("threshold" in gap.lower() or "plant" in gap.lower() for gap in aggregate["critical_gaps"]))

    def test_adversarial_mode_forces_a_strong_rejection(self) -> None:
        scenario = EXAMPLE_SCENARIOS["fanout_cascade"]
        context = dict(scenario["context"])
        context["adversarial_mode"] = True

        result = evaluate_claim(scenario["claim"], context)
        invalid_results = [entry for entry in result["domain_results"] if entry["verdict"] == "invalid"]

        self.assertTrue(invalid_results)
        self.assertTrue(
            any(entry["core_objection"].startswith("Strong rejection:") for entry in invalid_results)
        )


if __name__ == "__main__":
    unittest.main()

"""Cross-check published summaries against the saved individual measurements."""
import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RecordedTailChecks(unittest.TestCase):
    def test_published_running_means_and_checkpoints_match_raw_risks(self):
        result = ROOT / "labs" / "results"
        risks = defaultdict(list)
        with (result / "double_descent_tail_risks.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = int(row["n"]), int(row["d"])
                self.assertEqual(int(row["replicate"]), len(risks[key]) + 1)
                self.assertGreater(float(row["smallest_singular_value"]), 0)
                self.assertEqual(int(row["numerical_rank"]), min(key))
                risks[key].append(float(row["test_mse"]))
        self.assertEqual(set(risks), {(n, n + k) for n in [30, 60]
                                     for k in [-8, -2, -1, 0, 1, 2, 8]})
        self.assertTrue(all(len(values) == 1200 for values in risks.values()))

        totals = defaultdict(float)
        with (result / "double_descent_running_means.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                key, budget = (int(row["n"]), int(row["d"])), int(row["repetitions"])
                totals[key] += risks[key][budget - 1]
                self.assertAlmostEqual(float(row["running_mean_test_mse"]),
                                       totals[key] / budget, delta=1e-9)
        with (result / "double_descent_checkpoints.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                key, budget = (int(row["n"]), int(row["d"])), int(row["repetitions"])
                self.assertIn(budget, [40, 240, 1200])
                values = risks[key][:budget]
                self.assertAlmostEqual(float(row["mean"]), sum(values) / budget, delta=1e-9)
                self.assertEqual(float(row["maximum"]), max(values))
                # Infinite expectations must not receive a finite relative-error target.
                if abs(key[0] - key[1]) <= 1:
                    self.assertEqual(row["theoretical_expectation"], "inf")
                    self.assertEqual(row["relative_mean_error_if_finite"], "")

        metrics = json.loads((result / "metrics.json").read_text())
        for row in metrics["double_descent_tails"]["summaries"]:
            key = row["n"], row["d"]
            self.assertAlmostEqual(row["mean"], sum(risks[key]) / 1200, delta=1e-9)
            if abs(key[0] - key[1]) <= 1:
                self.assertEqual(row["expectation_status"], "infinite")
                self.assertIsNone(row["relative_mean_error"])
                self.assertIsNone(row["theoretical_expectation"])


if __name__ == "__main__":
    unittest.main()

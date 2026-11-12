"""Dependency-free checks for handoff validation utilities."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/videomae"))
from close_videomae_core_b_handoff import max_metric_difference, relative


class HandoffTests(unittest.TestCase):
    def test_nested_metrics(self):
        self.assertAlmostEqual(max_metric_difference({"a": [1, {"x": 0.3}]}, {"a": [1, {"x": 0.4}]}), 0.1)

    def test_missing_metric_rejected(self):
        with self.assertRaises(ValueError):
            max_metric_difference({"a": 1}, {})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            max_metric_difference(float("nan"), 0.0)

    def test_external_dependency_rejected(self):
        root = Path(__file__).resolve().parent
        with self.assertRaises(ValueError):
            relative(root.parent / "outside", root)


if __name__ == "__main__":
    unittest.main()

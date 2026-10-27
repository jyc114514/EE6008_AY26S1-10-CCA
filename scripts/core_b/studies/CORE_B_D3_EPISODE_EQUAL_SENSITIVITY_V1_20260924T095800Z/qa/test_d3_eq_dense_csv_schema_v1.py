"""Regression for mixed seed-level and pooled rows in dense CSV outputs."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from analyze_d3_eq_dense import write_csv  # noqa: E402


class DenseCsvSchemaTests(unittest.TestCase):
    def test_writer_uses_union_of_optional_fields(self) -> None:
        rows = [
            {"contrast": "seed_1", "estimate": -0.1, "final_test_access": 0},
            {
                "contrast": "three_seed_descriptive",
                "estimate": -0.2,
                "seed_mean": -0.2,
                "seed_sd": 0.03,
                "final_test_access": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_rows.csv"
            write_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["contrast", "estimate", "final_test_access", "seed_mean", "seed_sd"])
                decoded = list(reader)
        self.assertEqual(decoded[0]["seed_mean"], "")
        self.assertEqual(decoded[1]["seed_mean"], "-0.2")


if __name__ == "__main__":
    unittest.main()

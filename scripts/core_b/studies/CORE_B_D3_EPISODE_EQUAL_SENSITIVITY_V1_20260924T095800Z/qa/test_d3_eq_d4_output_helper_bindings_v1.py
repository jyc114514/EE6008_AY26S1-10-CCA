"""CPU regression for D4 rolling wrapper output-helper bindings."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import analyze_d3_eq_d4_rolling_v2 as d4  # noqa: E402


class D4OutputHelperBindingTests(unittest.TestCase):
    def test_writer_helpers_are_bound_from_the_locked_core(self) -> None:
        self.assertIs(d4.write_csv, d4.helpers.write_csv)
        self.assertIs(d4.atomic_json, d4.helpers.atomic_json)


if __name__ == "__main__":
    unittest.main()

"""CPU regression for the D4 rolling evaluator's duplicate-safe split binding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import analyze_d3_eq_d4_rolling_v2 as d4  # noqa: E402


class D4SplitManifestBindingTests(unittest.TestCase):
    def test_v2_uses_the_locked_duplicate_safe_split_manifest(self) -> None:
        expected = (
            d4.PROJECT_ROOT
            / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/"
              "splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
        )
        self.assertEqual(Path(d4.SPLIT_MANIFEST).resolve(), expected.resolve())
        self.assertTrue(expected.is_file())


if __name__ == "__main__":
    unittest.main()

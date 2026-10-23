"""CPU regression for injecting the shared V3 root into the dense worker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import d3_eq_dense_infer_entry as entry  # noqa: E402


class DenseInferenceEntryTests(unittest.TestCase):
    def test_dense_main_uses_injected_v3_root_in_its_own_globals(self) -> None:
        module = entry.load_dense_module()
        self.assertIs(module.main.__globals__, module.__dict__)
        self.assertEqual(Path(module.V3_ROOT).resolve(), Path(entry.V3_ROOT).resolve())


if __name__ == "__main__":
    unittest.main()

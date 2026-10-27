"""CPU regression for fixed D2_V3 event protocol/hash acceptance."""

from __future__ import annotations

import hashlib
import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from evaluate_d3_eq_d2v3_event_secondary import event_protocol_contract_matches  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EventProtocolContractTests(unittest.TestCase):
    def make_case(self, root: Path, d2_status: str = "locked_train_thresholds_ready"):
        protocol_path = root / "D2_V3_PROTOCOL_LOCKED.json"
        evaluator_path = root / "evaluate_d2_v3_panel.py"
        protocol_path.write_text("fixed protocol", encoding="utf-8")
        evaluator_path.write_text("fixed evaluator", encoding="utf-8")
        thresholds = {
            "Hand": {"theta_off": 0.05, "theta_on": 0.10, "train_q75": 0.10, "floor_applied": False},
            "Arm": {"theta_off": 0.01, "theta_on": 0.02, "train_q75": 0.02, "floor_applied": False},
        }
        algorithm = "source-group weighted empirical Q75"
        d2v3 = {
            "status": d2_status,
            "protocol_id": "CORE_B_D2_EVENT_MEASUREMENT_SENSITIVITY_V3",
            "final_test_access": 0,
            "smoothing_window_raw_steps": 8,
            "threshold_algorithm": algorithm,
            "thresholds": thresholds,
        }
        d3_lock = {
            "status": "LOCKED",
            "event_secondary": {
                "status": "secondary fixed-threshold diagnostic only",
                "protocol_sha256": digest(protocol_path),
                "evaluator_source_sha256": digest(evaluator_path),
                "dev_threshold_refit": False,
                "final_test_access": 0,
                "smoothing_window_raw_steps": 8,
                "threshold_algorithm": algorithm,
                "thresholds": copy.deepcopy(thresholds),
            },
        }
        return d3_lock, d2v3, protocol_path, evaluator_path

    def test_ready_threshold_protocol_passes_when_exact_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            self.assertTrue(event_protocol_contract_matches(*case))

    def test_generic_wrong_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory), d2_status="UNLOCKED")
            self.assertFalse(event_protocol_contract_matches(*case))

    def test_changed_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = list(self.make_case(Path(directory)))
            case[1]["thresholds"]["Hand"]["theta_on"] = 0.11
            self.assertFalse(event_protocol_contract_matches(*case))

    def test_changed_protocol_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = list(self.make_case(Path(directory)))
            case[2].write_text("changed protocol", encoding="utf-8")
            self.assertFalse(event_protocol_contract_matches(*case))


if __name__ == "__main__":
    unittest.main()

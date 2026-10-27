"""Regression tests for trainer RUN_STATUS/RUN_COMPLETE terminal encoding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from verify_d3_eq_training_grid import run_status_completion_matches  # noqa: E402


LOCK_HASH = "protocol-sha256"


def complete_record() -> dict[str, object]:
    return {
        "status": "PASS",
        "config": "TS00-D",
        "seed": 20261001,
        "horizon": 30,
        "epochs": 40,
        "optimizer_steps": 11120,
        "expected_optimizer_steps": 11120,
        "protocol_hash": LOCK_HASH,
        "checkpoint_last_sha256": "checkpoint-sha256",
        "init_hash": "init-sha256",
        "encoder_updated": False,
        "final_test_access": 0,
    }


class RunStatusReconciliationTests(unittest.TestCase):
    def test_actual_trainer_dict_merge_overwrite_is_accepted(self) -> None:
        complete = complete_record()
        # Mirrors d3_train_eq.py's final payload: complete.status overwrites
        # the preceding "complete" label because complete.status is "PASS".
        run_status = {"status": "complete", "completed_utc": "t", **complete}
        self.assertEqual(run_status["status"], "PASS")
        self.assertTrue(run_status_completion_matches(run_status, complete, LOCK_HASH))

    def test_intended_complete_label_is_accepted(self) -> None:
        complete = complete_record()
        run_status = {**complete, "status": "complete"}
        self.assertTrue(run_status_completion_matches(run_status, complete, LOCK_HASH))

    def test_mismatched_checkpoint_is_rejected(self) -> None:
        complete = complete_record()
        run_status = {"status": "complete", **complete}
        run_status["checkpoint_last_sha256"] = "different-checkpoint"
        self.assertFalse(run_status_completion_matches(run_status, complete, LOCK_HASH))

    def test_failed_status_is_rejected_even_with_pass_marker(self) -> None:
        complete = complete_record()
        run_status = {**complete, "status": "failed_or_interrupted"}
        self.assertFalse(run_status_completion_matches(run_status, complete, LOCK_HASH))

    def test_protocol_hash_mismatch_is_rejected(self) -> None:
        complete = complete_record()
        run_status = {"status": "PASS", **complete}
        self.assertFalse(run_status_completion_matches(run_status, complete, "other-protocol"))


if __name__ == "__main__":
    unittest.main()

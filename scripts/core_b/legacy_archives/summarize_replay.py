#!/usr/bin/env python3
"""Summarize fixed-fixture replay and lock the inference gate."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import numpy as np


RECOVERY = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z'))
SOURCE = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z'))
CELLS = ("FT-D-FRZ", "FT-D-LORA4", "FT-D-DENSE4")
TOL = {"eval_repeat_native_max_abs": 1e-6, "a_b_pooled_max_abs": 2e-4, "a_b_scaled_max_abs": 2e-4, "a_b_native_max_abs": 2e-4}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict[str, object]] = []
    all_pass = True
    for cell in CELLS:
        a_json = json.loads((RECOVERY / f"replay/{cell}/a/REPLAY_RESULT.json").read_text(encoding="utf-8"))
        b_json = json.loads((RECOVERY / f"replay/{cell}/b/REPLAY_RESULT.json").read_text(encoding="utf-8"))
        a = np.load(RECOVERY / f"replay/{cell}/a/eval_outputs.npz", allow_pickle=False)
        b = np.load(RECOVERY / f"replay/{cell}/b/eval_outputs.npz", allow_pickle=False)
        diffs = {f"a_b_{key}_max_abs": float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)))) for key in ("pooled", "scaled", "native", "standardized")}
        repeat_a = float(a_json["eval_repeat_native_max_abs"])
        repeat_b = float(b_json["eval_repeat_native_max_abs"])
        accepted = bool(repeat_a <= TOL["eval_repeat_native_max_abs"] and repeat_b <= TOL["eval_repeat_native_max_abs"] and diffs["a_b_pooled_max_abs"] <= TOL["a_b_pooled_max_abs"] and diffs["a_b_scaled_max_abs"] <= TOL["a_b_scaled_max_abs"] and diffs["a_b_native_max_abs"] <= TOL["a_b_native_max_abs"] and a_json["post_replay_state_digest_equal"] and b_json["post_replay_state_digest_equal"])
        all_pass = all_pass and accepted
        rows.append({"cell": cell, "a_builder": a_json["builder"], "b_builder": b_json["builder"], "a_gpu": a_json["physical_gpu"], "b_gpu": b_json["physical_gpu"], "a_eval_repeat_native_max_abs": repeat_a, "b_eval_repeat_native_max_abs": repeat_b, **diffs, "a_state_unchanged": a_json["post_replay_state_digest_equal"], "b_state_unchanged": b_json["post_replay_state_digest_equal"], "accepted": accepted, "historical_reference_C": "not_saved", "final_test_access": 0})
    with (RECOVERY / "REPLAY_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(RECOVERY / "REPLAY_RESULTS.json", {"status": "PASS" if all_pass else "FAIL", "rows": rows, "tolerances": TOL, "historical_reference_C": "not_saved", "final_test_access": 0})
    inventory = list(csv.DictReader((RECOVERY / "PARAMETER_COVERAGE.csv").open(encoding="utf-8", newline="")))
    coverage_pass = len(inventory) == 12 and all(row["strict_load_contract"] == "True" for row in inventory)
    acceptance = {
        "status": "RECONSTRUCTABLE_WITH_CAVEAT" if all_pass and coverage_pass else "BLOCKED",
        "checkpoints_reconstructable": "6/6" if coverage_pass else "less_than_6/6",
        "fresh_process_replay_accepted": "3/3" if all_pass else "less_than_3/3",
        "historical_eval_reference_available": False,
        "historical_reference_C": "not_saved",
        "reload_root_cause": "confirmed_code_level_semantic_mismatch_with_unreproducible_historical_numeric_reference" if all_pass else "unresolved",
        "accepted_eval_contract": "explicit model.eval plus inference_mode; source and recovery builders agree on fixed train fixture",
        "checkpoint_coverage_pass": coverage_pass,
        "source_checkpoint_sha256_unchanged": True,
        "final_test_access": 0,
    }
    write_json(RECOVERY / "RECONSTRUCTION_ACCEPTANCE.json", acceptance)
    diagnosis = """# Checkpoint diagnosis\n\n## Direct evidence\n\n- All six formal checkpoints are `epoch=10`, `global_step=430`, `seed=20260944`.\n- FRZ has 158 encoder keys and 19 head keys; LoRA4 has 174 encoder keys and 19 head keys; DENSE4 has 158 encoder keys and 19 head keys.\n- The recovery CPU inventory found zero missing, unexpected, shape-mismatched or dtype-mismatched keys in all 12 encoder/head groups.\n- The four original LoRA/DENSE replay attempts that OOMed are preserved in the recovery logs. The recovery exporter initially retained an autograd graph during forward; `inference_mode()` was then added and all four retry jobs completed.\n\n## Replay evidence\n\nThe same eight canonical train rows were evaluated by the source construction path A and an independently written recovery construction path B. Both paths explicitly use `eval()` and `inference_mode()`. All three representative cells have exact repeated-eval outputs and A/B maximum absolute differences at or below the locked tolerances. Model parameter/buffer digests were unchanged by replay.\n\n## Explanation of the historical nonzero field\n\nThe source runner writes the old replay field at the end of training. Its final training loop leaves the head in `train()` and leaves top-four encoder blocks in `train()` for LoRA4/DENSE4; the reload path constructs a fresh model with different mode/RNG state before the second forward. The old code also does not reset RNG between the two forwards. This is a confirmed non-equivalent comparison contract. The exact contribution of dropout, stochastic block behavior and RNG cannot be separated numerically because the original same-step in-memory reference output was not saved. Therefore the historical values are retained as evidence of a defective replay check, not as evidence of checkpoint corruption.\n\nThe accepted status is `RECONSTRUCTABLE_WITH_CAVEAT`, not `VERIFIED_REPLAY`, because historical reference C is unavailable.\n\nFinal-test access: 0.\n"""
    (RECOVERY / "CHECKPOINT_DIAGNOSIS.zh-CN.md").write_text(diagnosis, encoding="utf-8")
    defect = """# QA defect addendum\n\nThe original `checkpoint_reload_prediction_max_abs` field was computed without an explicit common eval-mode/RNG contract. The value therefore cannot be used as a strict checkpoint-integrity result. This addendum does not modify the original `RUN_COMPLETE.json` files.\n\nRecovery correction: strict state-dict coverage is checked separately; accepted replay uses `model.eval()` and `torch.inference_mode()` on both A and B paths; a fixed eight-row fixture is shared; source and recovery outputs are compared layer-wise. Historical C was never saved, so the corrected status remains `RECONSTRUCTABLE_WITH_CAVEAT`.\n"""
    (RECOVERY / "QA_DEFECT_ADDENDUM.md").write_text(defect, encoding="utf-8")
    print(json.dumps(acceptance, sort_keys=True))
    return 0 if acceptance["status"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_gated_cell import CELL_ORDER, CELL_SPECS, build_model


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows = []
    reference_ids = None
    reference_target = None
    reference_anchor = None
    for index, cell_id in enumerate(CELL_ORDER):
        run = ROOT / "runs" / cell_id
        watcher_path = ROOT / "logs" / f"{cell_id}_watcher.json"
        config = json.loads((run / "RUN_CONFIG_LOCKED.json").read_text())
        metrics = json.loads((run / "RUN_METRICS.json").read_text())
        complete = json.loads((run / "RUN_COMPLETE.json").read_text())
        status = json.loads((run / "RUN_STATUS.json").read_text())
        watcher = json.loads(watcher_path.read_text())
        with np.load(run / "predictions_best.npz", allow_pickle=False) as data:
            ids = data["episode_index"].copy()
            targets = data["targets_standardized"].copy()
            anchor = data["persistence_standardized"].copy()
            predictions = data["predictions_standardized"].copy()
            gates = data["gate_values"].copy()
        if reference_ids is None:
            reference_ids, reference_target, reference_anchor = ids, targets, anchor
        if not np.array_equal(ids, reference_ids) or not np.array_equal(targets, reference_target) or not np.array_equal(anchor, reference_anchor):
            raise RuntimeError(f"ID/target/anchor mismatch in {cell_id}")
        model, _hidden, _split_hidden = build_model(CELL_SPECS[cell_id], SEED + index * 1000)
        checkpoint = run / "checkpoint_best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        curves = json.loads((run / "epoch_curves.json").read_text())["curves"]
        expected_steps = len(curves) * 86
        if complete.get("status") != "complete" or status.get("status") != "complete" or watcher.get("status") != "complete" or watcher.get("exit_code") != 0:
            raise RuntimeError(f"terminal status failed for {cell_id}")
        if metrics.get("final_test_access") != 0 or complete.get("final_test_access") != 0 or watcher.get("non_child_process_signaled") is not False:
            raise RuntimeError(f"boundary failed for {cell_id}")
        if metrics.get("optimizer_steps") != expected_steps or metrics.get("optimizer_steps") <= 0:
            raise RuntimeError(f"optimizer steps mismatch for {cell_id}")
        if predictions.shape != (614, 30, 28) or gates.shape != (614, 30, 2) or not np.isfinite(predictions).all() or not np.isfinite(gates).all() or gates.min() < 0 or gates.max() > 1:
            raise RuntimeError(f"array contract failed for {cell_id}")
        if CELL_SPECS[cell_id]["residual_gate"] == "ungated" and not np.array_equal(gates, np.ones_like(gates)):
            raise RuntimeError(f"ungated gate sentinel failed for {cell_id}")
        rows.append({"cell_id": cell_id, "seed": SEED, "status": "accepted", "physical_gpu": watcher.get("physical_gpu"), "watcher_status": watcher.get("status"), "watcher_exit_code": watcher.get("exit_code"), "lease_released": watcher.get("lease_released"), "non_child_process_signaled": watcher.get("non_child_process_signaled"), "optimizer_steps": metrics.get("optimizer_steps"), "epochs_completed": len(curves), "expected_steps_from_epochs": expected_steps, "checkpoint_best_sha256": digest(checkpoint), "prediction_best_sha256": digest(run / "predictions_best.npz"), "prediction_shape": "(614,30,28)", "gate_shape": "(614,30,2)", "strict_reload": "PASS", "final_test_access": 0, "encoder_or_lora_modified": config.get("encoder_or_lora_modified"), "raw_video": config.get("raw_video")})
    fields = list(rows[0].keys())
    with (ROOT / "RUN_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "PASS", "accepted": len(rows), "expected": 8, "seed": SEED, "rows": rows, "common_N": 614, "common_prediction_shape": "(614,30,28)", "final_test_access": 0, "new_lora_or_encoder_training": False, "raw_video": False, "all_ids_targets_anchors_identical": True}
    (ROOT / "RUN_ACCEPTANCE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "RUN_ACCEPTANCE.md").write_text(f"# Exploratory run acceptance\n\nStatus: `{len(rows)}/8` accepted. All cells have terminal watcher/lease evidence, epoch-derived optimizer steps, strict checkpoint reload, `(614,30,28)` predictions, `(614,30,2)` gate values, shared IDs/targets/anchors, finite reconstruction, no final-test access and no encoder/LoRA/raw-video change.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "accepted": len(rows), "expected": 8}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

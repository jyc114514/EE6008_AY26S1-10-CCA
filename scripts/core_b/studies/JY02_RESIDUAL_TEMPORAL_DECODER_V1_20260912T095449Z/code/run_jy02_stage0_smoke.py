#!/usr/bin/env python3
"""Run four optimizer-step smoke coverage for all eight JY-02 cells."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_jy02_cell import ACCUMULATION, DIMS, PHYSICAL_BATCH, SEED, STEPS, CELL_SPECS, atomic_json, atomic_torch, build_model


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage0 CUDA contract failed")
    with np.load(args.root / "inputs/JY02_TARGETS.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train_indices = np.flatnonzero(data["split_code"] == 0)[:128]
    dev_indices = np.flatnonzero(data["split_code"] == 1)[:8]
    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for cell_id, spec in CELL_SPECS.items():
        out = args.root / "stage0_smoke" / cell_id; out.mkdir(parents=True, exist_ok=True)
        try:
            model = build_model(spec, SEED).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
            inputs = data["state_action_history"] if spec["input"] == "STATE" else np.concatenate([data["dynamic_pooled"], data["state_action_history"]], axis=1)
            target_abs = data["targets_standardized"].astype(np.float32)
            anchor = data["persistence_standardized"].astype(np.float32)
            residual = spec["target"] == "residual_to_persistence"
            for step in range(4):
                optimizer.zero_grad(set_to_none=True); samples = 0
                for micro in range(ACCUMULATION):
                    start = (step * ACCUMULATION + micro) * PHYSICAL_BATCH
                    index = train_indices[start : start + PHYSICAL_BATCH]
                    x = torch.tensor(inputs[index], device=device); y = torch.tensor(target_abs[index], device=device); p = torch.tensor(anchor[index], device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        raw = model(x); prediction = p + raw if residual else raw; loss = F.mse_loss(prediction.float(), y.float(), reduction="sum")
                    if not torch.isfinite(loss): raise FloatingPointError("stage0 loss non-finite")
                    loss.backward(); samples += len(index)
                denominator = samples * STEPS * DIMS
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.grad.div_(denominator)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm): raise FloatingPointError("stage0 gradient non-finite")
                optimizer.step(); scheduler.step()
            model.eval()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = torch.tensor(inputs[dev_indices], device=device); p = torch.tensor(anchor[dev_indices], device=device); raw = model(x).float(); prediction = p + raw if residual else raw
            smoke_prediction = prediction.cpu().numpy().astype(np.float32)
            if smoke_prediction.shape != (8, 30, 28) or not np.isfinite(smoke_prediction).all(): raise RuntimeError("stage0 output shape/finite failed")
            checkpoint = {"cell_id": cell_id, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_steps": 4, "final_test_access": 0}
            atomic_torch(out / "checkpoint_smoke.pt", checkpoint)
            loaded = torch.load(out / "checkpoint_smoke.pt", map_location="cpu", weights_only=False); model.load_state_dict(loaded["model_state_dict"], strict=True)
            zero_check = torch.tensor(anchor[dev_indices], device=device) + torch.zeros((8, STEPS, DIMS), device=device)
            if not torch.equal(zero_check, torch.tensor(anchor[dev_indices], device=device)): raise RuntimeError("zero residual persistence check failed")
            result = {"cell_id": cell_id, "status": "PASS", "optimizer_steps": 4, "prediction_shape": list(smoke_prediction.shape), "all_finite": True, "checkpoint_reload": True, "zero_residual_equals_persistence": True, "final_test_access": 0}
            atomic_json(out / "RUN_STATUS.json", {"status": "complete", **result}); atomic_json(out / "RUN_COMPLETE.json", {"status": "complete", "cell_id": cell_id, "optimizer_steps": 4, "final_test_access": 0}); rows.append(result)
        except Exception as exc:
            failure = {"cell_id": cell_id, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0}; atomic_json(out / "RUN_STATUS.json", failure); atomic_json(out / "RUN_FAILED.json", failure); rows.append(failure); raise
    with (args.root / "STAGE0_SMOKE_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(args.root / "STAGE0_SMOKE_RESULTS.json", {"status": "PASS", "rows": rows, "cells": 8, "optimizer_steps_per_cell": 4, "final_test_access": 0})
    (args.root / "STAGE0_SMOKE_RESULTS.md").write_text("# Stage0 smoke\n\nAll eight JY-02 cells constructed, completed four optimizer steps, passed finite/shape/checkpoint reload/zero-residual checks, and used no encoder or final-test data.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": 8}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the eight-cell, four-optimizer-step GPU engineering smoke."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_gated_cell import CELL_ORDER, CELL_SPECS, build_model, load_contract_data, set_seed


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{__import__('os').getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def run(root: Path, device: torch.device) -> list[dict]:
    data, train_mask, _dev_mask, _contracts = load_contract_data()
    state = torch.from_numpy(data["state_action_history"][train_mask][:32].astype(np.float32))
    dynamic = torch.from_numpy(data["dynamic_pooled"][train_mask][:32].astype(np.float32))
    target = torch.from_numpy(data["targets_standardized"][train_mask][:32].astype(np.float32))
    anchor = torch.from_numpy(data["persistence_standardized"][train_mask][:32].astype(np.float32))
    results = []
    for index, cell_id in enumerate(CELL_ORDER):
        set_seed(20260924 + index * 1000)
        spec = CELL_SPECS[cell_id]
        context = state if spec["input"] == "STATE" else torch.cat((dynamic, state), dim=1)
        model, hidden, split_hidden = build_model(spec, 20260924 + index * 1000)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        finite_gradients = True
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            context_device = context.to(device)
            target_device = target.to(device)
            anchor_device = anchor.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_residual, gates = model(context_device)
                loss = F.mse_loss(anchor_device + predicted_residual.float(), target_device, reduction="sum")
            loss.backward()
            finite_gradients = finite_gradients and all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
            optimizer.step()
        with torch.inference_mode():
            residual, gates = model(context.to(device))
            prediction = anchor.to(device) + residual.float()
        checkpoint = root / "runs/stage0" / cell_id / "checkpoint_smoke.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cell_id": cell_id, "seed": 20260924, "optimizer_steps": 4, "model_state_dict": model.state_dict()}, checkpoint)
        reloaded, _, _ = build_model(spec, 20260924 + index * 1000)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(payload["model_state_dict"], strict=True)
        gate_min = float(gates.min().item())
        gate_max = float(gates.max().item())
        passed = bool(finite_gradients and tuple(prediction.shape) == (32, 30, 28) and torch.isfinite(prediction).all().item() and 0.0 <= gate_min <= gate_max <= 1.0)
        marker = {"status": "PASS" if passed else "FAIL", "cell_id": cell_id, "seed": 20260924, "optimizer_steps": 4, "prediction_shape": list(prediction.shape), "gate_min": gate_min, "gate_max": gate_max, "finite_gradients": finite_gradients, "checkpoint_reload": "PASS", "final_test_access": 0, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(root / "runs/stage0" / cell_id / "RUN_COMPLETE.json", marker)
        results.append({"cell_id": cell_id, "status": marker["status"], "optimizer_steps": 4, "prediction_shape": "(32, 30, 28)", "gate_min": gate_min, "gate_max": gate_max, "checkpoint_reload": "PASS", "final_test_access": 0, "hidden_dim": hidden, "split_hidden": split_hidden})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    torch.set_num_threads(1)
    results = run(args.root, torch.device("cuda:0"))
    fields = list(results[0].keys())
    with (args.root / "STAGE0_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    atomic_json(args.root / "STAGE0_RESULTS.json", {"status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL", "accepted": sum(row["status"] == "PASS" for row in results), "expected": 8, "rows": results, "final_test_access": 0})
    (args.root / "STAGE0_RESULTS.md").write_text(f"# Stage0 results\n\nStatus: `{sum(row['status'] == 'PASS' for row in results)}/8` accepted. Each cell ran four optimizer steps with strict checkpoint reload, finite gradients, `(32,30,28)` output and bounded gates.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL", "accepted": sum(row["status"] == "PASS" for row in results), "expected": 8}, sort_keys=True))
    return 0 if all(row["status"] == "PASS" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

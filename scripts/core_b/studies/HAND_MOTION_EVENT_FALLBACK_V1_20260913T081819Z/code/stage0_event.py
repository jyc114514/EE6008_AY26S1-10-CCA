#!/usr/bin/env python3
"""Six-cell engineering smoke for the Hand event-gate matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from train_event_cell import (
    CELL_ORDER,
    CELL_SPECS,
    HAND,
    SEED,
    autocast,
    atomic_json,
    build_cell_model,
    event_labels,
    load_contract_data,
    set_seed,
)


def run(root: Path, device: torch.device) -> list[dict[str, object]]:
    data, train_mask, _dev_mask, contracts = load_contract_data()
    thresholds = contracts["thresholds"]
    target = data["targets_standardized"].astype("float32")
    persistence = data["persistence_standardized"].astype("float32")
    _motion, step_labels, window_labels = event_labels(target[train_mask], persistence[train_mask], thresholds)
    state = torch.from_numpy(data["state_action_history"][train_mask][:32].astype("float32"))
    dynamic = torch.from_numpy(data["dynamic_pooled"][train_mask][:32].astype("float32"))
    target_batch = torch.from_numpy(target[train_mask][:32])
    anchor_batch = torch.from_numpy(persistence[train_mask][:32])
    step_batch = torch.from_numpy(step_labels[:32])
    window_batch = torch.from_numpy(window_labels[:32])
    results: list[dict[str, object]] = []
    for index, cell_id in enumerate(CELL_ORDER):
        cell_dir = root / "runs" / "stage0" / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_seed = SEED + index * 1000
        set_seed(cell_seed)
        spec = CELL_SPECS[cell_id]
        context = state if spec["input"] == "STATE" else torch.cat((dynamic, state), dim=1)
        mode = str(spec["event_mode"])
        train_prevalence = float(window_labels.mean() if mode in {"U", "W"} else step_labels.mean())
        model, hidden_dim, parameter_count, added = build_cell_model(cell_id, cell_seed, train_prevalence)
        model = model.to(device)
        context_device = context.to(device)
        target_device = target_batch.to(device)
        anchor_device = anchor_batch.to(device)
        step_device = step_batch.to(device)
        window_device = window_batch.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        finite_gradients = True
        for _step in range(4):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                residual, gates, _q = model(context_device)
                prediction = anchor_device + residual
                rec_loss = F.mse_loss(prediction.float(), target_device.float(), reduction="mean")
                if mode == "W":
                    logits = model.event_logits(context_device.detach())
                    assert logits is not None
                    pos_weight = torch.tensor((1.0 - train_prevalence) / train_prevalence, device=device)
                    event_loss = F.binary_cross_entropy_with_logits(logits.float(), window_device.float(), pos_weight=pos_weight)
                elif mode == "T":
                    logits = model.event_logits(context_device.detach())
                    assert logits is not None
                    pos_weight = torch.tensor((1.0 - train_prevalence) / train_prevalence, device=device)
                    event_loss = F.binary_cross_entropy_with_logits(logits.float(), step_device.float(), pos_weight=pos_weight)
                else:
                    event_loss = torch.zeros((), device=device)
                loss = rec_loss + (0.1 if mode != "U" else 0.0) * event_loss
            loss.backward()
            finite_gradients = finite_gradients and all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
            optimizer.step()
        with torch.inference_mode():
            with autocast(device):
                residual, gates, event_probability = model(context_device)
            prediction = anchor_device + residual.float()
        checkpoint = cell_dir / "checkpoint_smoke.pt"
        torch.save({"cell_id": cell_id, "seed": cell_seed, "optimizer_steps": 4, "model_state_dict": model.state_dict()}, checkpoint)
        reloaded, _hidden, _params, _added = build_cell_model(cell_id, cell_seed, train_prevalence)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(payload["model_state_dict"], strict=True)
        gate_min = float(gates.min().item())
        gate_max = float(gates.max().item())
        passed = bool(finite_gradients and tuple(prediction.shape) == (32, 30, 28) and torch.isfinite(prediction).all().item() and 0.0 <= gate_min <= gate_max <= 1.0 and (event_probability is None or torch.isfinite(event_probability).all().item()))
        marker = {"status": "PASS" if passed else "FAIL", "cell_id": cell_id, "seed": cell_seed, "event_mode": mode, "optimizer_steps": 4, "prediction_shape": list(prediction.shape), "gate_min": gate_min, "gate_max": gate_max, "finite_gradients": finite_gradients, "checkpoint_reload": "PASS", "hidden_dim": hidden_dim, "parameter_count": parameter_count, "added_event_parameters": added, "final_test_access": 0, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(cell_dir / "RUN_COMPLETE.json", marker)
        results.append(marker)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage0 must run on a CUDA device")
    torch.set_num_threads(1)
    results = run(args.root, torch.device("cuda:0"))
    accepted = sum(row["status"] == "PASS" for row in results)
    atomic_json(args.root / "STAGE0_RESULTS.json", {"status": "PASS" if accepted == 6 else "FAIL", "accepted": accepted, "expected": 6, "rows": results, "final_test_access": 0})
    (args.root / "STAGE0_RESULTS.md").write_text(f"# Stage0 results\n\nStatus: `{accepted}/6` accepted. Each cell ran four optimizer steps, finite-gradient checks, strict checkpoint reload and `(32,30,28)` output validation.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if accepted == 6 else "FAIL", "accepted": accepted, "expected": 6}, sort_keys=True))
    return 0 if accepted == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())

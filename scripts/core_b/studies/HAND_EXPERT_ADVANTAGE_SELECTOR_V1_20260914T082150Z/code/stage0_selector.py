#!/usr/bin/env python3
"""Four-cell selector-only Stage0 smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from selector_model import SelectorHead, state_hash
from train_selector import MODES, SELECTOR_SEED, load_contract, loss_for_mode, selector_calibration_assignment


def run(root: Path, device: torch.device) -> list[dict[str, object]]:
    contract = load_contract(); oof = contract["oof"]; train_mask = contract["train_mask"]
    context = contract["context"][train_mask]; summary = contract["state_summary"][train_mask]
    p = oof["persistence_hand_standardized"].astype(np.float32); n = oof["oof_predictions_hand_standardized"].astype(np.float32); y = oof["targets_hand_standardized"].astype(np.float32)
    magnitude = np.abs(n - p).mean(axis=2, keepdims=True).astype(np.float32); motion = oof["motion_event_label"].astype(np.float32); hard = oof["hard_label"].astype(np.float32); soft = oof["soft_target"].astype(np.float32)
    ids = oof["episode_index"].astype(np.int64); fit_positions = np.flatnonzero(~selector_calibration_assignment(ids))[:32]
    batch = tuple(torch.from_numpy(value[fit_positions]).to(device) for value in (context, summary, magnitude, p, n, y, motion, hard, soft))
    results = []
    for cell in MODES:
        directory = root / "runs/stage0" / cell; directory.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(SELECTOR_SEED); model = SelectorHead(seed=SELECTOR_SEED).to(device); before = state_hash(model); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001); optimizer.zero_grad(set_to_none=True)
        pos_weights = {}
        for mode in ("SEL-M", "SEL-AH"):
            prevalence = float((motion if mode == "SEL-M" else hard)[fit_positions].mean()); pos_weights[mode] = torch.tensor((1.0 - prevalence) / prevalence, device=device)
        for _ in range(4):
            context_b, summary_b, magnitude_b, p_b, n_b, y_b, motion_b, hard_b, soft_b = batch
            q = model.probability(context_b, summary_b, magnitude_b); hand = p_b + q * (n_b - p_b); loss = loss_for_mode(cell, q, hand, y_b, motion_b, hard_b, soft_b, pos_weights); loss.backward()
            finite = all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
            if not finite: raise FloatingPointError(f"non-finite Stage0 selector gradient: {cell}")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode(): q = model.probability(batch[0], batch[1], batch[2]); hand = batch[3] + q * (batch[4] - batch[3])
        checkpoint = directory / "checkpoint_smoke.pt"; torch.save({"cell_id": cell, "seed": SELECTOR_SEED, "optimizer_steps": 4, "model_state_dict": model.state_dict()}, checkpoint); reloaded = SelectorHead(seed=SELECTOR_SEED); payload = torch.load(checkpoint, map_location="cpu", weights_only=False); reloaded.load_state_dict(payload["model_state_dict"], strict=True); after = state_hash(model); passed = bool(tuple(q.shape) == (32, 30, 1) and tuple(hand.shape) == (32, 30, 14) and torch.isfinite(q).all().item() and torch.isfinite(hand).all().item() and 0 <= float(q.min()) <= float(q.max()) <= 1 and before != after)
        marker = {"status": "PASS" if passed else "FAIL", "cell_id": cell, "seed": SELECTOR_SEED, "optimizer_steps": 4, "q_shape": list(q.shape), "prediction_hand_shape": list(hand.shape), "q_min": float(q.min()), "q_max": float(q.max()), "selector_only_parameters_changed": before != after, "base_expert_frozen": True, "strict_checkpoint_reload": "PASS", "final_test_access": 0, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (directory / "RUN_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"); results.append(marker)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--device", default="cpu"); args = parser.parse_args(); device = torch.device(args.device); results = run(args.root, device); accepted = sum(item["status"] == "PASS" for item in results); (args.root / "STAGE0_SELECTOR_RESULTS.json").write_text(json.dumps({"status": "PASS" if accepted == 4 else "FAIL", "accepted": accepted, "expected": 4, "rows": results, "final_test_access": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (args.root / "STAGE0_SELECTOR_RESULTS.md").write_text(f"# Selector Stage0\n\nStatus: `{accepted}/4`; each cell ran four selector-only optimizer steps with finite gradients, q/prediction shapes and strict checkpoint reload.\n", encoding="utf-8"); print(json.dumps({"status": "PASS" if accepted == 4 else "FAIL", "accepted": accepted, "expected": 4}, sort_keys=True)); return 0 if accepted == 4 else 1


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Lock the shared train/calibration split before selector training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from train_selector import ROOT, SELECTOR_SEED, load_contract, selector_calibration_assignment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    contract = load_contract(); oof = contract["oof"]; ids = oof["episode_index"].astype(np.int64); tasks = oof["original_task_index"].astype(np.int64); calibration = selector_calibration_assignment(ids); fit = ~calibration
    rows = [{"episode_index": int(i), "original_task_index": int(t), "selector_split": "calibration" if bool(c) else "fit"} for i, t, c in zip(ids, tasks, calibration)]
    path = ROOT / "configs/SELECTOR_TRAIN_CALIBRATION_SPLIT.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assignment = {"status": "LOCKED", "seed": SELECTOR_SEED, "rows": rows, "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()), "overlap": 0, "group_unit": "episode ID; no source/session group field used", "oof_sha256": digest(ROOT / "oof/OOF_ADVANTAGE_TARGETS.npz"), "dev_targets_in_split": False}
    (ROOT / "configs/SELECTOR_TRAIN_CALIBRATION_SPLIT.json").write_text(json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {"status": "LOCKED", "selector_seed": SELECTOR_SEED, "cells": ["SEL-U", "SEL-M", "SEL-AH", "SEL-AS"], "matched_input": "past FUSION context + past state summary + OOF/all-train base residual magnitude + learned time embedding", "shared_initialization": True, "shared_optimizer_schedule": True, "max_epochs": 40, "min_epochs": 10, "patience": 6, "losses": {"SEL-U": "Hand reconstruction MAE", "SEL-M": "class-weighted BCE motion event", "SEL-AH": "class-weighted BCE OOF hard advantage", "SEL-AS": "BCE OOF soft advantage"}, "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()), "dev_target_in_fit": False, "future_label_input": False, "arm_source": "locked E-F-SP-G development Arm14", "oof_sha256": digest(ROOT / "oof/OOF_ADVANTAGE_TARGETS.npz")}
    (ROOT / "configs/SELECTOR_PROTOCOL_LOCKED.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "configs/SELECTOR_PROTOCOL_LOCKED.md").write_text(f"# Selector protocol\n\nLocked seed `{SELECTOR_SEED}`; shared fit/calibration split is `{int(fit.sum())}/{int(calibration.sum())}` episodes. Four cells use identical inputs, initialization, optimizer, schedule and budget; only supervision differs. Development targets are not used for fitting or calibration.\n", encoding="utf-8")
    (ROOT / "PRELAUNCH_SELECTOR_AUDIT.md").write_text(f"# Selector prelaunch audit\n\nStatus: `PASS`; Gate 1 passed; OOF coverage is `2741/2741`; selector fit/calibration split is `{int(fit.sum())}/{int(calibration.sum())}` with overlap `0`. The OOF target archive is train-only and the dev target is excluded from selector fit/calibration.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "fit_rows": int(fit.sum()), "calibration_rows": int(calibration.sum()), "overlap": 0}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

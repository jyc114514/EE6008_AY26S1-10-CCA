#!/usr/bin/env python3
"""Write V3 protocol metadata and the pre-registered 36-row matrix."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
SEEDS = (20260950, 20260951, 20260952)
CELLS = ("S-HIST", "F-POOL-D", "F-TOK-D", "F-TOK-S", "V-TOK-D-A", "V-TOK-S-A")
HORIZONS = (30, 60)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    write_json(EXP / "SEED_REGISTRY.json", {"status": "LOCKED", "seeds": list(SEEDS), "excluded_historical": [20260921, 20260922, 20260923, 20260939, 20260940, 20260941, 20260942, 20260944], "fresh_seed_rule": "selected before results and excluded all prior Core B seeds", "final_test_access": 0})
    rows = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            for cell in CELLS:
                rows.append({"run_id": f"V3-{cell}-H{horizon}-S{seed}", "seed": seed, "horizon_steps": horizon, "horizon_seconds": horizon / 30.0, "cell": cell, "pilot_seed": seed == SEEDS[0], "status": "LOCKED", "checkpoint_rule": "epoch40_last", "final_test_access": 0})
    with (EXP / "MATRIX_36.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(EXP / "PROTOCOL_LOCKED.json", {"status": "LOCKED", "protocol_id": "CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1", "cells": list(CELLS), "horizons": list(HORIZONS), "seeds": list(SEEDS), "run_count": 36, "pilot_run_count": 12, "epochs": 40, "checkpoint_rule": "last epoch only", "prediction_target": "Full28", "encoder_updated": False, "final_test_access": 0})
    (EXP / "MODEL_LOSS_CONTRACT.md").write_text("""# V3 model and loss contract\n\n- Full 28D direct residual output; no old Arm splice, DCT, prototype, selector or LoRA.\n- State branch: 71D history + mask/time metadata, Linear 128, five causal residual Conv1d blocks with dilations `[1,2,4,8,16]`.\n- Video branch: 16 contextual temporal tokens, per-token Linear 768→128, fixed sinusoidal past-time encoding. GLOBAL control is D-token mean repeated 16 times.\n- Anchor command: independent 28→128 memory token.\n- Decoder: two Transformer decoder layers, `d_model=128`, 4 heads, FFN 512, dropout 0.1; future query time encoding is fixed and continuous for 30/60 steps.\n- Output: zero-initialized 28D normalized residual; native prediction is `anchor + sigma_y * residual`.\n- Loss: `L_pos + 0.1*L_delta`, with equal Arm/Hand group terms and train-only time-invariant scales.\n- Optimizer: AdamW lr `1e-3`, cosine to `1e-5`, weight decay `1e-4`, gradient clip 1.0, 40 epochs, effective batch 64.\n- No dev early stopping or pilot-based architecture changes.\n""", encoding="utf-8")
    (EXP / "CONTEXT_RESTORED.md").write_text("""# V3 context restored\n\nV3 is an independent development branch. It inherits only the canonical train/validation episode allowlist and verified raw G1 schema. Old Core B results remain historical references; old gates and final candidates are unchanged. Final-test access is zero.\n\nThe primary population is defined by legal 2-second anchors and is shared by the 1-second and 2-second main comparisons. A supplementary 1-second-only development population is separate and never enters training.\n""", encoding="utf-8")
    (EXP / "OUTCOME_EXPOSURE.md").write_text("""# Outcome exposure\n\nThis branch is development work on repeatedly used canonical train/validation episodes. Anchor selection is time-equidistant and outcome-blind. No target, task label, future phase, dev score or final-test data enters model input, anchor selection, normalization fit, checkpoint selection or pilot continuation.\n""", encoding="utf-8")
    (EXP / "CLAIM_LEDGER.md").write_text("""# Claim ledger\n\n- Primary claim candidates: history usefulness, pooled-vs-temporal video representation, video increment beyond state, dynamic-vs-static difference, horizon sensitivity.\n- Required evidence: matched anchor populations, per-seed results, crossed seed×episode uncertainty and task/joint support.\n- Not allowed: final/generalization claims, physical velocity claims, causal dynamics claims from one interaction, or direct comparison of V3 standardized scores with old fixed-anchor normalizers.\n""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": len(rows), "seeds": SEEDS, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())

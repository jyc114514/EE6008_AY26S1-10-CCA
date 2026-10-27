#!/usr/bin/env python3
"""Lock train-only prototype loss scales before any prototype training."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
MODE_ROOT = ROOT / "mode_audit"
sys.path.insert(0, str(ROOT / "code"))
from model import ConditionalPrototypeDecoder, dct_basis, loss_components  # type: ignore  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main() -> int:
    data = load_npz(INPUT)
    train = data["split_code"] == 0
    oof = load_npz(DCT_ROOT / "oof/HD-D1-L1_OOF.npz")
    ids = oof["episode_index"].astype(np.int64)
    if not np.array_equal(ids, data["episode_index"][train]):
        raise RuntimeError("prototype calibration OOF IDs failed")
    assignment_rows: list[dict[str, str]] = []
    with (MODE_ROOT / "CLUSTER_ASSIGNMENTS.csv").open(encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    if len(assignment_rows) != len(ids) or [int(row["episode_index"]) for row in assignment_rows] != ids.tolist():
        raise RuntimeError("mode assignment order failed")
    labels = np.asarray([int(row["K2"]) for row in assignment_rows], dtype=np.int64)
    centers_payload = load_npz(MODE_ROOT / "PROTOTYPE_CENTERS.npz")
    centers_whitened = centers_payload["centers_K2"].astype(np.float64)
    if centers_whitened.shape == (2, 16 * 14):
        centers_whitened = centers_whitened.reshape(2, 16, 14)
    if centers_whitened.shape != (2, 16, 14):
        raise RuntimeError(f"K2 center shape {centers_whitened.shape} != (2,16,14)")
    coefficient_mean = centers_payload["coefficient_mean"].astype(np.float64)
    coefficient_scale = centers_payload["coefficient_scale"].astype(np.float64)
    centers_native = centers_whitened * coefficient_scale[None, :, :] + coefficient_mean[None, :, :]
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float32)[:, :14]
    scale = np.asarray(normalizer["safe_std"], dtype=np.float32)[:, :14]
    loss_contract = json.loads((DCT_ROOT / "TEMPORAL_LOSS_CONTRACT.json").read_text(encoding="utf-8"))
    safe_scale = np.asarray(loss_contract["safe_scale_j"], dtype=np.float32)
    lambda_fd = float(loss_contract["lambda_fd"])
    p_native = oof["persistence_hand_native"].astype(np.float32)
    target_native = oof["targets_hand_native"].astype(np.float32)
    target_std = oof["targets_hand_standardized"].astype(np.float32)
    basis = dct_basis(30).numpy()
    target_coeff = np.einsum("kt,etj->ekj", basis[:16], target_native - p_native).astype(np.float32)
    calibration_mask = np.asarray([int(hashlib.sha256(f"HD-DUAL-LOSS-CALIBRATION:{int(episode)}".encode()).hexdigest()[:8], 16) % 4 == 0 for episode in ids])
    if int(calibration_mask.sum()) != 632:
        raise RuntimeError(f"prototype loss calibration rows {int(calibration_mask.sum())} != 632")
    torch.manual_seed(20260938)
    centers_tensor = torch.from_numpy(centers_native.astype(np.float32))
    model = ConditionalPrototypeDecoder(centers=centers_tensor, seed=20260938)
    with torch.no_grad():
        probabilities, coefficients, residual = model(torch.from_numpy(np.concatenate((data["dynamic_pooled"][train], data["state_action_history"][train]), axis=1).astype(np.float32)))
        p_native_t = torch.from_numpy(p_native)
        mean_t = torch.from_numpy(mean)
        scale_t = torch.from_numpy(scale)
        native_components = p_native_t.unsqueeze(1) + residual
        components_std = (native_components - mean_t.unsqueeze(0).unsqueeze(0)) / scale_t.unsqueeze(0).unsqueeze(0)
        soft_std = (probabilities.unsqueeze(-1).unsqueeze(-1) * components_std).sum(dim=1)
        target_std_t = torch.from_numpy(target_std)
        target_native_t = torch.from_numpy(target_native)
        safe_t = torch.from_numpy(safe_scale)
        lambda_fd_t = torch.tensor(lambda_fd)
        soft_native = soft_std * scale_t.unsqueeze(0) + mean_t.unsqueeze(0)
        position_rows = torch.abs(soft_std - target_std_t).mean(dim=(1, 2))
        fd_rows = torch.abs((soft_native[:, 1:] - soft_native[:, :-1]) - (target_native_t[:, 1:] - target_native_t[:, :-1]))
        fd_rows = (fd_rows / safe_t).mean(dim=(1, 2))
        dual = position_rows + lambda_fd_t * fd_rows
        selected_coefficients = coefficients[torch.arange(len(labels)), torch.from_numpy(labels)]
        refine_loss = F.smooth_l1_loss(selected_coefficients, torch.from_numpy(target_coeff))
    med_dual = float(torch.median(dual[torch.from_numpy(calibration_mask)]))
    # Use per-row CE for the scale contract.
    per_row_mode = -torch.log(probabilities[torch.arange(len(labels)), torch.from_numpy(labels)].clamp_min(1e-8))
    med_mode = float(torch.median(per_row_mode[torch.from_numpy(calibration_mask)]))
    per_row_refine = F.smooth_l1_loss(selected_coefficients, torch.from_numpy(target_coeff), reduction="none").mean(dim=(1, 2))
    med_refine = float(torch.median(per_row_refine[torch.from_numpy(calibration_mask)]))
    if not all(np.isfinite([med_dual, med_mode, med_refine])) or min(med_dual, med_mode, med_refine) <= 0:
        raise RuntimeError("prototype loss scale median invalid")
    payload = {"status": "LOCKED", "mode_count": 2, "seed": 20260938, "calibration_rule": "sha256(HD-DUAL-LOSS-CALIBRATION:<episode_id>)[:8] mod 4 == 0", "calibration_rows": 632, "median_dual_loss": med_dual, "median_mode_loss": med_mode, "median_refine_loss": med_refine, "lambda_mode": med_dual / med_mode, "lambda_refine": med_dual / med_refine, "dct_K": 16, "centers_sha256": digest(MODE_ROOT / "PROTOTYPE_CENTERS.npz"), "input_sha256": digest(INPUT), "normalizer_sha256": digest(NORMALIZER), "dev_used": False, "final_test_access": 0}
    write_json(ROOT / "configs/PROTOTYPE_LOSS_SCALE_LOCKED.json", payload)
    (ROOT / "configs/PROTOTYPE_LOSS_SCALE_LOCKED.md").write_text("# Prototype loss scale lock\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

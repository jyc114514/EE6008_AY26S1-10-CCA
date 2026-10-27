#!/usr/bin/env python3
"""Fresh-process CPU replay and independent scoring of Core A linear heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


SEEDS = (6008, 7, 42)
SPLITS = ("train", "validation")
TOLERANCE_ATOL = 1e-6
TOLERANCE_RTOL = 1e-6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_digest(state: dict[str, object]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score_logits(labels: np.ndarray, logits: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels), 120) or labels.ndim != 1:
        raise AssertionError(f"Core A score schema mismatch: {labels.shape}, {logits.shape}")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= 120)):
        raise AssertionError("Core A labels/logits are outside the finite 120-class contract")

    max_logit = logits.max(axis=1, keepdims=True)
    log_partition = max_logit[:, 0] + np.log(np.exp(logits - max_logit).sum(axis=1))
    log_prob = logits - log_partition[:, None]
    probabilities = np.exp(log_prob)
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.bincount(labels * 120 + predictions, minlength=120 * 120).reshape(120, 120)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    denom = true_support + predicted_support
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(120, dtype=np.float64), where=denom > 0)
    correct = predictions == labels
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for b in range(15):
        in_bin = (confidence > b / 15.0) & (confidence <= (b + 1) / 15.0)
        if in_bin.any():
            ece += float(in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean()))
    metrics = {
        "top1": float(correct.mean()),
        "top5": float(np.any(top5 == labels[:, None], axis=1).mean()),
        "macro_f1_supported": float(f1[denom > 0].mean()),
        "macro_f1_all120_zero_for_unsupported": float(f1.mean()),
        "supported_classes": int((denom > 0).sum()),
        "nll": float(-log_prob[np.arange(len(labels)), labels].mean()),
        "brier_multiclass_sum120": float(np.square(probabilities - np.eye(120)[labels]).sum(axis=1).mean()),
        "ece15_equal_example_weight_bins_(lo,hi]": float(ece),
        "examples": int(len(labels)),
    }
    return metrics, confusion, probabilities.astype(np.float32)


def replay_one(project: Path, audit: Path, feature_root: Path, probe_root: Path, seed: int) -> dict[str, object]:
    import torch

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    config_path = probe_root / "CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_manifest_path = feature_root / "FEATURE_MANIFEST.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if config["device"] != "cpu" or config["seeds"] != list(SEEDS):
        raise AssertionError("Core A replay must retain the as-run CPU device and seed list")
    if digest(feature_manifest_path) != config["feature_manifest_sha256"]:
        raise AssertionError("Core A feature manifest differs from the probe config")
    if digest(Path(feature_manifest["manifest"])) != config["classification_manifest_sha256"]:
        raise AssertionError("Core A classification manifest hash mismatch")

    checkpoint_path = probe_root / f"seed{seed}/checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["seed"] != seed or checkpoint["config_sha256"] != digest(config_path):
        raise AssertionError("Core A checkpoint identity does not match its cell/config")

    model = torch.nn.Linear(384, 120, bias=True).cpu().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    before = tensor_state_digest(model.state_dict())
    metric_records: list[dict[str, object]] = []
    support_records: list[dict[str, object]] = []
    replay_records: list[dict[str, object]] = []
    pred_dir = audit / "metrics/core_a"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        feature_path = feature_root / f"videomae_{split}_pooled.pt"
        if digest(feature_path) != config["feature_sha256"][split]:
            raise AssertionError(f"Core A {split} feature file hash mismatch")
        bundle = torch.load(feature_path, map_location="cpu", weights_only=True)
        x, y, ids = bundle["features"], bundle["labels"], bundle["episode_indices"]
        expected_n = 2757 if split == "train" else 617
        if x.shape != (expected_n, 384) or y.shape != (expected_n,) or ids.shape != (expected_n,):
            raise AssertionError(f"Core A {split} feature schema mismatch")
        if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
            raise AssertionError(f"Core A {split} feature dtype/finite check failed")

        logits_parts = []
        with torch.inference_mode():
            for start in range(0, expected_n, 256):
                logits_parts.append(model(x[start : start + 256]).cpu())
        logits = torch.cat(logits_parts).numpy()
        labels = y.cpu().numpy().astype(np.int64, copy=False)
        episode_ids = ids.cpu().numpy().astype(np.int64, copy=False)
        metrics, confusion, probabilities = score_logits(labels, logits)

        saved_path = probe_root / f"seed{seed}/predictions_{split}.npz"
        with np.load(saved_path, allow_pickle=False) as saved:
            if set(saved.files) != {"episode_indices", "labels", "logits", "predictions"}:
                raise AssertionError(f"Core A saved {split} prediction keys changed")
            if not np.array_equal(saved["episode_indices"], episode_ids):
                raise AssertionError(f"Core A {split} prediction IDs/order mismatch")
            if not np.array_equal(saved["labels"], labels):
                raise AssertionError(f"Core A {split} prediction label order mismatch")
            old_logits = saved["logits"]
            if old_logits.shape != logits.shape or not np.isfinite(old_logits).all():
                raise AssertionError(f"Core A saved {split} logits schema/finite check failed")
            diff = np.abs(old_logits.astype(np.float64) - logits.astype(np.float64))
            max_abs = float(diff.max(initial=0.0))
            mean_abs = float(diff.mean())
            pred_equal = bool(np.array_equal(saved["predictions"], logits.argmax(axis=1)))
        tolerance_pass = bool(np.allclose(old_logits, logits, atol=TOLERANCE_ATOL, rtol=TOLERANCE_RTOL))
        if not tolerance_pass or not pred_equal:
            raise AssertionError(f"Core A checkpoint replay mismatch for seed={seed} split={split}: max_abs={max_abs}")

        output = pred_dir / f"seed{seed}_{split}_per_id.npz"
        np.savez_compressed(output, episode_indices=episode_ids, labels=labels, logits=logits,
                            probabilities=probabilities, predictions=logits.argmax(axis=1).astype(np.int64))
        confusion_path = pred_dir / f"seed{seed}_{split}_confusion.npz"
        np.savez_compressed(confusion_path, confusion=confusion, true_support=confusion.sum(axis=1),
                            predicted_support=confusion.sum(axis=0))
        metric_records.append({"seed": seed, "split": split, **metrics})
        replay_records.append({
            "task": "CoreA", "cell": f"seed{seed}", "split": split,
            "status": "PASS", "checkpoint_sha256": digest(checkpoint_path),
            "prediction_sha256": digest(saved_path), "input_sha256": digest(feature_path),
            "backend": "cpu", "physical_gpu": "NA", "batch_size": 256,
            "dtype": "float32", "tolerance_atol": TOLERANCE_ATOL,
            "tolerance_rtol": TOLERANCE_RTOL, "tolerance_basis": "provisional; same CPU backend/features/dtype; no prior committed replay tolerance found",
            "max_abs": max_abs, "mean_abs": mean_abs, "ids_shape_finite_match": True,
            "predictions_match": pred_equal, "parameters_before_sha256": before,
            "parameters_after_sha256": tensor_state_digest(model.state_dict()),
        })
        for cls in range(120):
            row = confusion[cls]
            tp = int(row[cls])
            actual = int(confusion[cls].sum())
            predicted = int(confusion[:, cls].sum())
            denom = actual + predicted
            support_records.append({"seed": seed, "split": split, "class_id": cls,
                                    "true_support": actual, "predicted_support": predicted,
                                    "tp": tp, "fp": predicted - tp, "fn": actual - tp,
                                    "f1": (2 * tp / denom) if denom else 0.0,
                                    "included_in_supported_macro_f1": int(denom > 0)})

    after = tensor_state_digest(model.state_dict())
    if before != after:
        raise AssertionError("Core A inference mutated model parameters/buffers")
    record = {"seed": seed, "checkpoint_sha256": digest(checkpoint_path),
              "best_epoch": checkpoint["epoch"], "parameters_before_sha256": before,
              "parameters_after_sha256": after, "metrics": metric_records,
              "replay": replay_records, "support": support_records,
              "final_test_access": 0, "device": "cpu", "optimizer_created": False,
              "backward_called": False}
    (pred_dir / f"seed{seed}_replay.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_CORE_A_REPLAY", "seed": seed, "best_epoch": checkpoint["epoch"],
                      "splits": [{"split": r["split"], "max_abs": r["max_abs"],
                                  "top1": m["top1"], "macro_f1_supported": m["macro_f1_supported"]}
                                 for r, m in zip(replay_records, metric_records)],
                      "final_test_access": 0}, allow_nan=False))
    return record


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    project, audit, feature_root, probe_root = [p.resolve() for p in
                                                  (args.project_root, args.audit_root, args.feature_root, args.probe_root)]
    records = []
    if args.seed is not None:
        records.append(replay_one(project, audit, feature_root, probe_root, args.seed))
    else:
        # Each seed is launched in its own fresh CPU process, matching the requested replay boundary.
        for seed in SEEDS:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--project-root", str(project), "--audit-root", str(audit),
                       "--feature-root", str(feature_root), "--probe-root", str(probe_root),
                       "--seed", str(seed)]
            subprocess.run(command, check=True)
            records.append(json.loads((audit / f"metrics/core_a/seed{seed}_replay.json").read_text(encoding="utf-8")))
        metrics = [row for record in records for row in record["metrics"]]
        replay = [row for record in records for row in record["replay"]]
        support = [row for record in records for row in record["support"]]
        write_csv(audit / "CORE_A_PER_SEED_METRICS.csv", metrics)
        write_csv(audit / "CORE_A_CLASS_SUPPORT.csv", support)
        existing = (audit / "CHECKPOINT_REPLAY.csv")
        prior = list(csv.DictReader(existing.open(newline="", encoding="utf-8"))) if existing.is_file() else []
        prior = [row for row in prior if row.get("task") != "CoreA"]
        write_csv(existing, prior + replay)
        means = {}
        for split in SPLITS:
            means[split] = {}
            for metric in ("top1", "top5", "macro_f1_supported", "nll", "brier_multiclass_sum120", "ece15_equal_example_weight_bins_(lo,hi]"):
                values = np.asarray([float(r[metric]) for r in metrics if r["split"] == split])
                means[split][metric] = {"mean": float(values.mean()), "sample_sd_ddof1": float(values.std(ddof=1))}
        (audit / "metrics/core_a/SEED_SUMMARY.json").write_text(
            json.dumps({"status": "PASS_CORE_A_3_OF_3", "means": means,
                        "macro_f1_rule": "120-class confusion; mean over classes with true+predicted support > 0; all-120 zero-fill also reported",
                        "top5_rule": "any true class in logits top 5", "brier_rule": "per-example sum over 120 class probabilities squared error, then mean",
                        "ece_rule": "15 equal-width confidence bins (lower, upper], equal example weights",
                        "selection": "validation Macro-F1, earliest strict maximum; no held-out data",
                        "final_test_access": 0}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS_CORE_A_3_OF_3", "summary": means,
                          "final_test_access": 0}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

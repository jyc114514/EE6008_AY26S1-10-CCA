from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from v7_common import (
    NORMALIZER,
    ROOT,
    SEEDS,
    TASKS,
    TRAIN_NPZ,
    atomic_json,
    load_safe_prediction,
    load_train_prefix,
    load_train_y_after_teacher_seal,
    receiver_rows,
    sha_file,
    write_csv,
    write_npy,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def row_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["anchor_id"]),
        int(row["episode_id"]),
        int(row["task_id"]),
        int(row["anchor_frame"]),
        str(row["source_cluster"]),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty teacher receiver row file: {path}")
    if [int(row["row"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"teacher receiver row indices are not contiguous: {path}")
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate teacher receiver identity: {path}")
    return rows


def atomic_safetensor(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    save_file({"prediction_native": np.ascontiguousarray(values, dtype=np.float64)}, str(temporary))
    temporary.replace(path)


def verify_train_archive_without_y() -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    data = load_train_prefix()
    if "target_native_h30" in data:
        raise RuntimeError("prefix reader unexpectedly materialized train Y before teacher-bank seal")
    if len(data["anchor_id"]) != 1280:
        raise RuntimeError(f"accepted outer-train receiver count mismatch: {len(data['anchor_id'])}")
    return data, receiver_rows(data)


def source_bundle(seed: int, role: str) -> tuple[Path, Path, dict[str, Any]]:
    index_path = ROOT / "coordination/TEACHER_JOB_INDEX.json"
    if not index_path.is_file():
        raise FileNotFoundError("teacher replay job index is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    matches = [entry for entry in index.get("entries", []) if int(entry.get("seed", -1)) == seed and entry.get("teacher_role") == role and entry.get("status") == "PASS"]
    if len(matches) != 1:
        raise RuntimeError(f"teacher replay index has {len(matches)} accepted entries for {role}/{seed}")
    selected = matches[0]
    prediction = Path(selected["prediction_path"])
    rows_path = Path(selected["prediction_rows_path"])
    receipt_path = Path(selected["completion_path"])
    if not all(p.is_file() for p in (prediction, rows_path, receipt_path)):
        raise FileNotFoundError(f"teacher train bundle incomplete: {prediction.parent}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("future_target_values_read") != 0:
        raise RuntimeError(f"teacher source receipt is not accepted or read train Y: {receipt_path}")
    if sha_file(prediction) != receipt.get("prediction_sha256") or sha_file(rows_path) != receipt.get("prediction_rows_sha256"):
        raise RuntimeError(f"teacher bundle receipt digest mismatch: {output}")
    return prediction, rows_path, receipt


def explicit_join(
    canonical_rows: list[dict[str, Any]],
    source_rows: list[dict[str, str]],
    prediction: np.ndarray,
    source_label: str,
) -> np.ndarray:
    if len(source_rows) != len(prediction) or len(source_rows) != 1280:
        raise RuntimeError(f"{source_label}: row/prediction count mismatch")
    mapping: dict[tuple[str, int, int, int, str], np.ndarray] = {}
    for row, value in zip(source_rows, prediction):
        key = row_key(row)
        if key in mapping:
            raise RuntimeError(f"{source_label}: duplicate receiver identity {key}")
        mapping[key] = value
    canonical_keys = [row_key(row) for row in canonical_rows]
    if set(mapping) != set(canonical_keys):
        raise RuntimeError(f"{source_label}: receiver ID/episode/task/anchor join mismatch")
    return np.stack([mapping[key] for key in canonical_keys]).astype(np.float64, copy=False)


def build() -> None:
    protocol_dir = ROOT / "protocol"
    seal_path = protocol_dir / "TEACHER_BANK_SEAL.json"
    targets_lock_path = protocol_dir / "TARGET_LOCK.json"
    if seal_path.exists() or targets_lock_path.exists():
        raise FileExistsError("teacher/target seals already exist; preserve and inspect before rerun")
    train_data, canonical_rows = verify_train_archive_without_y()
    id_hash = __import__("hashlib").sha256(
        "\n".join(str(row["anchor_id"]) for row in canonical_rows).encode("utf-8")
    ).hexdigest()

    source_predictions: dict[tuple[int, str], np.ndarray] = {}
    source_entries: list[dict[str, Any]] = []
    t_predictions: dict[int, np.ndarray] = {}
    t_entries: list[dict[str, Any]] = []

    # Only accepted train-prefix replay outputs are touched before the seal.
    # This loop does not open target_native_h30 or any dev target archive.
    for seed in SEEDS:
        aligned: dict[str, np.ndarray] = {}
        for role in ("S", "V"):
            pred_path, rows_path, receipt = source_bundle(seed, role)
            pred = load_safe_prediction(pred_path)
            src_rows = read_rows(rows_path)
            aligned[role] = explicit_join(canonical_rows, src_rows, pred, f"{role}_{seed}")
            source_predictions[(seed, role)] = aligned[role]
            source_entries.append({
                "seed": seed,
                "teacher_role": role,
                "model_role": "G-S" if role == "S" else "G-V",
                "prediction_path": str(pred_path),
                "prediction_sha256": sha_file(pred_path),
                "prediction_rows_path": str(rows_path),
                "prediction_rows_sha256": sha_file(rows_path),
                "completion_path": str(pred_path.parent / "PREDICTION_COMPLETE.json"),
                "completion_sha256": sha_file(pred_path.parent / "PREDICTION_COMPLETE.json"),
                "source_model_path": receipt["source_model_path"],
                "source_model_sha256": receipt["source_model_sha256"],
                "source_model_manifest_path": receipt["source_model_manifest_path"],
                "source_model_manifest_sha256": receipt["source_model_manifest_sha256"],
                "receiver_rows": 1280,
                "ordered_receiver_ids_sha256": id_hash,
                "prediction_dtype_after_join": "float64",
                "outer_train_prediction": True,
                "in_sample_not_OOF": True,
            })
        if aligned["S"].shape != (1280, 30, 28) or aligned["V"].shape != (1280, 30, 28):
            raise RuntimeError(f"teacher source shape mismatch for seed {seed}")
        t_native = 0.5 * aligned["S"] + 0.5 * aligned["V"]
        if not np.isfinite(t_native).all():
            raise RuntimeError(f"nonfinite T_i teacher average for seed {seed}")
        t_path = ROOT / "teacher_predictions_train" / f"seed{seed}" / "T_NATIVE.safetensors"
        atomic_safetensor(t_path, t_native)
        rows_path = ROOT / "teacher_predictions_train" / f"seed{seed}" / "T_NATIVE_ROWS.csv"
        write_csv(rows_path, list(canonical_rows[0].keys()), canonical_rows)
        t_predictions[seed] = t_native
        t_entries.append({
            "seed": seed,
            "teacher_role": "T",
            "formula": "0.5*S_i+0.5*V_i in native float64 after explicit ID join",
            "prediction_path": str(t_path),
            "prediction_sha256": sha_file(t_path),
            "prediction_rows_path": str(rows_path),
            "prediction_rows_sha256": sha_file(rows_path),
            "receiver_rows": 1280,
            "ordered_receiver_ids_sha256": id_hash,
            "prediction_dtype": "float64",
            "source_seeds": [seed],
            "outer_train_prediction": True,
            "in_sample_not_OOF": True,
        })

    # The nonempty teacher bank is sealed before the first access to outer-train Y.
    teacher_seal = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "study": "post_V6_single_fusion_head_supervised_soft_target_development",
        "teacher_roles": ["S_i", "V_i", "T_i"],
        "source_seed_count": len(SEEDS),
        "source_prediction_bundles_expected": 6,
        "source_prediction_bundles_accepted": len(source_entries),
        "derived_T_bundles_expected": 3,
        "derived_T_bundles_accepted": len(t_entries),
        "entries": source_entries + t_entries,
        "ordered_receiver_ids_sha256": id_hash,
        "receiver_rows": 1280,
        "train_archive_sha256": sha_file(TRAIN_NPZ),
        "future_target_values_read_before_seal": 0,
        "dev_target_archive_opened": False,
        "teacher_parameters_updated": 0,
        "student_training_started": False,
        "final_heldout_payload_access": 0,
        "old_root_writes": 0,
    }
    if len(teacher_seal["entries"]) != 9:
        raise RuntimeError("teacher bank is empty or incomplete")
    atomic_json(seal_path, teacher_seal)

    # Only after the seal above is durable may the train outcome Y be read.
    y_native = load_train_y_after_teacher_seal().astype(np.float64, copy=False)
    write_npy(ROOT / "training_targets/Y_NATIVE.npy", y_native)
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    safe_fd = np.asarray(normalizer["safe_fd_scale"], dtype=np.float64)
    if y_native.shape != (1280, 30, 28) or safe_fd.shape != (28,):
        raise RuntimeError("outer-train target or safe-FD shape mismatch")

    target_entries: list[dict[str, Any]] = []
    for seed in SEEDS:
        s_native = source_predictions[(seed, "S")]
        v_native = source_predictions[(seed, "V")]
        t_native = t_predictions[seed]
        targets = {
            "GT_RETRAIN": y_native,
            "SELF_SOFT": 0.5 * y_native + 0.5 * v_native,
            "MIX_SOFT": 0.5 * y_native + 0.5 * t_native,
        }
        mix_minus_self = targets["MIX_SOFT"] - targets["SELF_SOFT"]
        expected_difference = 0.25 * (s_native - v_native)
        if not np.allclose(mix_minus_self, expected_difference, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"linear target identity failed for seed {seed}")
        if any(value.shape != (1280, 30, 28) or not np.isfinite(value).all() for value in targets.values()):
            raise RuntimeError(f"target shape/finite failure for seed {seed}")

        target_dir = ROOT / "training_targets" / f"seed{seed}"
        for program, values in targets.items():
            target_path = target_dir / f"{program}.npy"
            write_npy(target_path, values.astype(np.float64, copy=False))
            target_entries.append({
                "seed": seed,
                "program": program,
                "target_path": str(target_path),
                "target_sha256": sha_file(target_path),
                "shape": list(values.shape),
                "dtype": "float64",
                "source_role": {
                    "GT_RETRAIN": "Y",
                    "SELF_SOFT": "0.5*Y+0.5*V_i",
                    "MIX_SOFT": "0.5*Y+0.5*T_i",
                }[program],
                "teacher_bank_seal_sha256": sha_file(seal_path),
            })

    y_path = ROOT / "training_targets/Y_NATIVE.npy"
    target_lock = {
        "schema_version": 1,
        "status": "LOCKED",
        "created_utc": utc_now(),
        "train_archive_path": str(TRAIN_NPZ),
        "train_archive_sha256": sha_file(TRAIN_NPZ),
        "Y_native_path": str(y_path),
        "Y_native_sha256": sha_file(y_path),
        "Y_native_read_after_teacher_seal": True,
        "teacher_bank_seal_path": str(seal_path),
        "teacher_bank_seal_sha256": sha_file(seal_path),
        "receiver_rows": 1280,
        "ordered_receiver_ids_sha256": id_hash,
        "shape": [1280, 30, 28],
        "dtype": "float64",
        "native_formula": "Y, S_i, V_i and T_i are joined in native coordinates before any student target is formed",
        "targets": [
            "Z_GT=Y",
            "Z_SELF=0.5*Y+0.5*stopgrad(V_i)",
            "Z_MIX=0.5*Y+0.5*stopgrad(T_i)=0.5*Y+0.25*S_i+0.25*V_i",
        ],
        "soft_target_ratio": 0.5,
        "target_cache_role": "training loss only; never a model input",
        "normalizer_refit": False,
        "dev_target_values_read": 0,
        "final_heldout_payload_access": 0,
        "target_entries": target_entries,
    }
    atomic_json(targets_lock_path, target_lock)

    signal_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        s_native = source_predictions[(seed, "S")]
        v_native = source_predictions[(seed, "V")]
        t_native = t_predictions[seed]
        for task in TASKS:
            mask = np.asarray(train_data["task_id"], dtype=np.int64) == task
            for name, difference in (
                ("S_MINUS_V", s_native - v_native),
                ("T_MINUS_V", t_native - v_native),
            ):
                internal_fd = np.diff(difference[mask], axis=1)
                signal_rows.append({
                    "seed": seed,
                    "task_id": task,
                    "comparison": name,
                    "windows": int(mask.sum()),
                    "native_position_mae": float(np.abs(difference[mask]).mean()),
                    "native_position_rms": float(np.sqrt(np.square(difference[mask]).mean())),
                    "native_internal_fd_mae": float(np.abs(internal_fd).mean()),
                    "safe_scaled_internal_fd_l1": float((np.abs(internal_fd) / safe_fd[None, None, :]).mean()),
                    "axes": 28,
                    "horizon": 30,
                    "future_internal_differences": 29,
                    "source_support": "accepted outer-train episodes; in-sample predictions",
                })
    write_csv(
        ROOT / "results/TRAIN_TARGET_SIGNAL.csv",
        ["seed", "task_id", "comparison", "windows", "native_position_mae", "native_position_rms",
         "native_internal_fd_mae", "safe_scaled_internal_fd_l1", "axes", "horizon",
         "future_internal_differences", "source_support"],
        signal_rows,
    )
    print(json.dumps({
        "status": "PASS",
        "teacher_seal_sha256": sha_file(seal_path),
        "target_lock_sha256": sha_file(targets_lock_path),
        "target_count": len(target_entries),
        "train_target_read_after_seal": True,
        "dev_target_values_read": 0,
    }, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", required=True)
    args = parser.parse_args()
    if args.build:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

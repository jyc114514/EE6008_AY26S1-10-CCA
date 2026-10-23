#!/usr/bin/env python3
"""Run the frozen V9 Core B matrix with the explicit CURRENT_PROFILE_V2 grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import core_b_predict_v2 as core_b

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V1_MATRIX = V1_OUT / "protocol/FINAL_MODEL_MATRIX.csv"
CURRENT_LOCK = OUT / "core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
DEV_PREFIX = RUN / "replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
FINAL_PREFIX = RUN / "replay/core_b/FINAL_PREFIXES_CURRENT_PROFILE_V2.npz"
DEV_FEATURE_MANIFEST = OUT / "core_b/CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json"
FINAL_FEATURE_MANIFEST = OUT / "core_b/CURRENT_PROFILE_V2_FINAL_FEATURE_MANIFEST.json"
FINAL_FREEZE = OUT / "core_b/CORE_B_V2_FINAL_FREEZE.json"
FINAL_ACCESS = OUT / "protocol/FINAL_ACCESS_OPENED.json"
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("A0", "AD", "V0", "VD", "S_OPT")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_prediction(path: Path, ids: np.ndarray, values: np.ndarray) -> dict:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction artifact: {path}")
    # Derived outputs use the same strict two-key, hash-checked archive schema.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, canonical_ids=np.asarray(ids).astype("U"), prediction_native=np.asarray(values, dtype="<f8", order="C"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"canonical_ids", "prediction_native"}:
            raise ValueError("derived prediction archive has unexpected keys")
        got_ids = archive["canonical_ids"].astype(str)
        got = np.asarray(archive["prediction_native"], dtype=np.float64)
    if not np.array_equal(got_ids, np.asarray(ids).astype(str)) or not np.array_equal(got, np.asarray(values, dtype=np.float64)):
        raise ValueError("derived prediction failed reload identity/value check")
    return {
        "file_sha256": sha(path),
        "canonical_ids_sha256": hashlib.sha256("\n".join(got_ids.tolist()).encode()).hexdigest(),
        "prediction_native_sha256": hashlib.sha256(np.asarray(got, dtype="<f8", order="C").tobytes()).hexdigest(),
        "shape": list(got.shape), "dtype": "float64",
    }


def matrix_rows() -> list[dict[str, str]]:
    rows = list(csv.DictReader(V1_MATRIX.open(encoding="utf-8-sig", newline="")))
    selected = [r for r in rows if r["core"] == "B" and r["program"] in PROGRAMS and r["status"].startswith("accepted_source_bound")]
    if len(selected) != 15 or {int(r["seed"]) for r in selected} != set(SEEDS):
        raise ValueError(f"accepted V9/S_OPT matrix must contain 15 cells, found {len(selected)}")
    return sorted(selected, key=lambda r: (PROGRAMS.index(r["program"]), int(r["seed"])))


def accepted_inference_profile(program: str, seed: int) -> str:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if program == "S_OPT":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        return "accepted_v2_inference_flags_full_context"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return "accepted_v9_seed_everything_profile_full_context"


def run(split: str, expected_lock_sha256: str, expected_final_freeze_sha256: str | None = None,
        expected_final_access_sha256: str | None = None, dry_run: bool = False) -> dict:
    if split not in {"dev", "final_test"}:
        raise ValueError("split must be dev or final_test")
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical != "3" or visible != "3":
        raise RuntimeError("Core B V2 runner is restricted to physical GPU3, mapped as logical cuda:0")

    if sha(CURRENT_LOCK) != expected_lock_sha256:
        raise ValueError("candidate current-profile lock SHA mismatch")
    lock = json.loads(CURRENT_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_CURRENT_PROFILE_DEV_FEATURES" or lock.get("route") != "R2":
        raise ValueError("CURRENT_PROFILE_V2 is not locked before dev feature extraction")
    final_auth = None
    if split == "final_test":
        if not expected_final_freeze_sha256 or not expected_final_access_sha256:
            raise ValueError("final_test requires the pre-access freeze and V2 access receipt hashes")
        if sha(FINAL_FREEZE) != expected_final_freeze_sha256:
            raise ValueError("V2 final freeze hash mismatch")
        freeze = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS":
            raise ValueError("V2 final freeze is not sealed before payload access")
        final_auth = core_b.verify_final_access_receipt(FINAL_ACCESS, expected_final_access_sha256)
        if freeze.get("profile_lock_sha256") != expected_lock_sha256:
            raise ValueError("final freeze differs from CURRENT_PROFILE_V2 lock")

    prefix_path = DEV_PREFIX if split == "dev" else FINAL_PREFIX
    feature_manifest_path = DEV_FEATURE_MANIFEST if split == "dev" else FINAL_FEATURE_MANIFEST
    if not prefix_path.is_file() or not feature_manifest_path.is_file():
        raise FileNotFoundError(f"missing {split} prefix or CURRENT_PROFILE_V2 feature manifest")
    expected_prefix_sha = sha(prefix_path)
    expected_feature_manifest_sha = sha(feature_manifest_path)
    with np.load(prefix_path, allow_pickle=False) as prefix:
        if "canonical_id" not in prefix.files:
            raise ValueError("Core B prefix is missing canonical identities")
        forbidden = [key for key in prefix.files if any(term in key.lower() for term in ("target", "future", "label", "ground_truth"))]
        if forbidden:
            raise ValueError(f"prefix archive contains forbidden keys: {forbidden}")
        canonical_ids = prefix["canonical_id"].astype(str)
        anchors = prefix["anchor_id"].astype(str)
        episodes = prefix["episode_id"].astype(np.int64)
        tasks = prefix["task_id"].astype(np.int64)
        anchor_native = prefix["anchor_native"].copy()
    n = len(canonical_ids)
    expected_rows = 185 if split == "dev" else 162
    if n != expected_rows or len(set(canonical_ids.tolist())) != n or len(set(anchors.tolist())) != n:
        raise ValueError(f"{split} prefix identities do not match the locked {expected_rows}-row population")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("status") != "PASS" or feature_manifest.get("profile_lock_sha256") != expected_lock_sha256:
        raise ValueError("CURRENT_PROFILE_V2 feature manifest does not match the frozen lock")
    if feature_manifest.get("split") != split:
        raise ValueError("feature manifest split does not match prediction request")
    if dry_run:
        return {"status": "DRY_RUN_PASS", "split": split, "rows": n,
                "candidate_lock_sha256": expected_lock_sha256,
                "feature_manifest_sha256": expected_feature_manifest_sha,
                "prefix_sha256": expected_prefix_sha,
                "model_cells": 15, "derived_mixes": 6, "persistence": 1}

    if not torch.cuda.is_available() or int(torch.cuda.current_device()) != 0:
        raise RuntimeError("physical GPU3 is not available as logical cuda:0 after watcher launch")
    out_dir = RUN / "replay/core_b" / ("current_profile_dev" if split == "dev" else "current_profile_final") / "predictions"
    status_path = out_dir.parent / "PREDICTION_RUN_STATUS.json"
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1, "status": "RUNNING", "started_utc": utc(), "split": split,
        "candidate_profile_lock_sha256": expected_lock_sha256,
        "final_freeze_sha256": expected_final_freeze_sha256,
        "final_access_receipt_sha256": expected_final_access_sha256,
        "feature_manifest_path": str(feature_manifest_path),
        "feature_manifest_sha256": expected_feature_manifest_sha,
        "prefix_path": str(prefix_path), "prefix_sha256": expected_prefix_sha,
        "rows": n, "physical_gpu": 3, "logical_gpu": "cuda:0",
        "new_training_runs": 0, "future_targets_materialized": False,
        "final_payload_read": split == "final_test",
        "prediction_records": [],
    }
    atomic_json(status_path, record)
    grid = core_b.CurrentProfileGrid(feature_manifest, feature_manifest_path)
    model_rows = matrix_rows()
    model_cells = lock.get("predictor_contract", {}).get("model_cells")
    if not isinstance(model_cells, dict):
        raise ValueError("candidate lock is missing predictor_contract.model_cells")
    learned_paths: dict[tuple[str, int], Path] = {}
    try:
        for row in model_rows:
            program, seed, mode = row["program"], int(row["seed"]), row["mode"]
            if row["checkpoint_sha256"] != model_cells[program][str(seed)]["checkpoint_sha256"]:
                raise ValueError(f"model checkpoint hash differs from candidate lock: {program}/{seed}")
            video = mode == "TS11-D"
            cell = core_b.ModelCell(
                program_id=program, seed=seed, video_mode=mode,
                model_path=Path(row["checkpoint_path"]), expected_model_sha256=row["checkpoint_sha256"],
                model_manifest_path=Path(row["model_manifest_path"]),
                expected_model_manifest_sha256=row["model_manifest_sha256"],
                normalizer_path=Path(row["normalizer_path"]),
                expected_normalizer_sha256=row["normalizer_sha256"],
                prefix_path=prefix_path, expected_prefix_sha256=expected_prefix_sha,
                feature_cache_manifest_path=feature_manifest_path if video else None,
                expected_feature_cache_manifest_sha256=expected_feature_manifest_sha if video else None,
            )
            inference_profile = accepted_inference_profile(program, seed)
            output = out_dir / program / f"seed{seed}.npz"
            started = time.perf_counter()
            bundle, artifact = core_b.predict_cell(
                cell, canonical_ids=canonical_ids, split=split,
                feature_reader=grid if video else None,
                final_access_opened_path=FINAL_ACCESS if split == "final_test" else None,
                expected_final_access_opened_sha256=expected_final_access_sha256 if split == "final_test" else None,
                output_path=output, device=torch.device("cuda:0"), batch_size=64,
            )
            if artifact is None:
                raise RuntimeError("accepted V2 predictor did not write its prediction bundle")
            learned_paths[(program, seed)] = output
            record["prediction_records"].append({
                "core": "B", "program": program, "seed": seed, "path": str(output),
                "sha256": artifact["file_sha256"], "canonical_ids_sha256": artifact["canonical_ids_sha256"],
                "prediction_native_sha256": artifact["prediction_native_sha256"], "rows": n,
                "shape": artifact["shape"], "dtype": artifact["dtype"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "model_manifest_sha256": row["model_manifest_sha256"],
                "normalizer_sha256": row["normalizer_sha256"],
                "feature_profile_id": "CURRENT_PROFILE_V2" if video else "not_applicable_state_only",
                "feature_manifest_sha256": expected_feature_manifest_sha if video else None,
                "predictor_inference_profile": inference_profile,
                "predictor_batch_size": 64, "forward_rows": n,
                "elapsed_seconds": time.perf_counter() - started,
            })
            atomic_json(status_path, record)

        persistence = core_b.make_persistence_bundle(canonical_ids, anchor_native)
        persist_path = out_dir / "PERSISTENCE/reference.npz"
        persist_artifact = core_b.write_prediction_bundle(persist_path, persistence)
        record["prediction_records"].append({
            "core": "B", "program": "PERSISTENCE", "seed": None, "path": str(persist_path),
            "sha256": persist_artifact["file_sha256"], "canonical_ids_sha256": persist_artifact["canonical_ids_sha256"],
            "prediction_native_sha256": persist_artifact["prediction_native_sha256"],
            "rows": n, "shape": persist_artifact["shape"], "dtype": persist_artifact["dtype"],
            "forward_rows": 0,
        })
        for seed in SEEDS:
            for mix, video_program in (("MIX0", "V0"), ("MIXD", "VD")):
                state_path = learned_paths[("S_OPT", seed)]
                video_path = learned_paths[(video_program, seed)]
                with np.load(state_path, allow_pickle=False) as state, np.load(video_path, allow_pickle=False) as video:
                    state_ids = state["canonical_ids"].astype(str)
                    video_ids = video["canonical_ids"].astype(str)
                    if not np.array_equal(state_ids, video_ids) or not np.array_equal(state_ids, canonical_ids):
                        raise ValueError(f"{mix}/{seed} source prediction IDs differ")
                    mixed = 0.5 * np.asarray(state["prediction_native"], dtype=np.float64) + 0.5 * np.asarray(video["prediction_native"], dtype=np.float64)
                if mixed.shape != (n, 30, 28) or not np.isfinite(mixed).all():
                    raise ValueError(f"{mix}/{seed} is invalid")
                mix_path = out_dir / mix / f"seed{seed}.npz"
                artifact = atomic_prediction(mix_path, canonical_ids, mixed)
                record["prediction_records"].append({
                    "core": "B", "program": mix, "seed": seed, "path": str(mix_path),
                    "sha256": artifact["file_sha256"], "canonical_ids_sha256": artifact["canonical_ids_sha256"],
                    "prediction_native_sha256": artifact["prediction_native_sha256"], "rows": n,
                    "shape": artifact["shape"], "dtype": artifact["dtype"],
                    "coefficient": 0.5, "source_program": "S_OPT", "video_program": video_program,
                    "source_prediction_sha256": sha(state_path), "video_prediction_sha256": sha(video_path),
                    "arithmetic": "native float64 after canonical ID order equality check", "forward_rows": 0,
                })
                atomic_json(status_path, record)
        if len(record["prediction_records"]) != 22:
            raise ValueError(f"expected 22 V2 prediction artifacts, got {len(record['prediction_records'])}")
        record["status"] = "PREDICTIONS_COMPLETE_PENDING_SEAL"
        record["completed_utc"] = utc()
        record["neural_prediction_cells"] = 15
        record["derived_mixture_cells"] = 6
        record["persistence_reference_count"] = 1
        record["future_targets_materialized"] = False
        record["final_payload_read"] = split == "final_test"
        atomic_json(status_path, record)
        receipt_path = out_dir.parent / "PREDICTION_RUN_RECEIPT.json"
        atomic_json(receipt_path, record)
        return record
    except Exception as exc:
        record.update({"status": "FAILED", "failed_utc": utc(), "error_type": type(exc).__name__, "error": str(exc)})
        atomic_json(status_path, record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "final_test"), required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-final-freeze-sha256")
    parser.add_argument("--expected-final-access-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.split, args.expected_lock_sha256, args.expected_final_freeze_sha256,
                  args.expected_final_access_sha256, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

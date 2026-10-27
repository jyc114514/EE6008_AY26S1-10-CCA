"""Replay the frozen Core A/B model matrix on accepted train/dev inputs only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import core_a_predict as core_a
import core_b_predict as core_b


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
CORE_A_MATRIX = OUT_ROOT / "protocol/FINAL_MODEL_MATRIX.csv"
CORE_B_ROOT = PROJECT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
CORE_B_Q0_MANIFEST = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
NORMALIZER_SHA = "22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d"
Q0_SHA = "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee"
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("A0", "AD", "V0", "VD", "S_OPT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_npz(path: Path, *, canonical_ids: np.ndarray, prediction_native: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    ids = np.asarray(canonical_ids).astype("U", copy=False)
    prediction = np.ascontiguousarray(prediction_native, dtype="<f8")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, canonical_ids=ids, prediction_native=prediction)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "canonical_ids_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "prediction_native_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
        "shape": list(prediction.shape),
        "dtype": "float64",
    }


def read_matrix() -> list[dict[str, str]]:
    with CORE_A_MATRIX.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _core_a_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if row["core"] == "A" and row["program"] in {"A_DYNAMIC", "A_STATIC"}]
    if len(selected) != 6 or {int(row["seed"]) for row in selected} != {20260825, 20260826, 20260827}:
        raise ValueError("Core A dev replay matrix must contain six fixed Batch-48 cells")
    return sorted(selected, key=lambda row: (row["program"], int(row["seed"])))


def _core_b_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if row["core"] == "B" and row["program"] in PROGRAMS and row["status"].startswith("accepted_source_bound")]
    if len(selected) != 15:
        raise ValueError(f"Core B dev replay requires 12 V9 and 3 S_OPT checkpoints, found {len(selected)}")
    return sorted(selected, key=lambda row: (PROGRAMS.index(row["program"]), int(row["seed"])))



def set_accepted_inference_profile(program: str, seed: int) -> str:
    """Apply the fixed original V9/V2 inference flags before model creation."""
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if program == "S_OPT":
        # Exact V2 runner profile: cudnn deterministic/benchmark are left at
        # their fresh-process defaults, which are explicitly restored here.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        return "accepted_v2_inference_flags"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return "accepted_v9_seed_everything_profile"

def run(attempt: str, *, reuse_core_a_from: str | None = None) -> dict[str, Any]:
    physical_gpu = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    if physical_gpu != "3":
        raise RuntimeError(f"this replay was authorized for physical GPU3; watcher selected {physical_gpu!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the watcher-authorized dev replay")
    device = torch.device("cuda:0")
    attempt_root = RUN_ROOT / "replay" / f"dev_full_{attempt}" / "artifacts"
    if attempt_root.exists():
        raise FileExistsError(f"refusing to overwrite dev replay artifacts: {attempt_root}")
    attempt_root.mkdir(parents=True)
    status_path = attempt_root / "DEV_REPLAY_STATUS.json"
    matrix_rows = read_matrix()
    core_a_rows = _core_a_rows(matrix_rows)
    core_b_rows = _core_b_rows(matrix_rows)
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "EE6008_FINAL_TEST_V1_20261019T041642Z",
        "attempt": attempt,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "split": "dev",
        "physical_gpu": int(physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "logical_device": str(device),
        "new_training_runs": 0,
        "final_payload_read": False,
        "prediction_records": [],
    }
    atomic_json(status_path, record)
    try:
        if reuse_core_a_from is not None:
            old_root = RUN_ROOT / "replay" / f"dev_full_{reuse_core_a_from}" / "artifacts"
            old_status = json.loads((old_root / "DEV_REPLAY_STATUS.json").read_text(encoding="utf-8"))
            old_records = {
                (str(item["program"]), int(item["seed"])): item
                for item in old_status.get("prediction_records", []) if item.get("core") == "A"
            }
            if len(old_records) != 6:
                raise ValueError("Core A reuse source does not contain all six completed dev cells")
            for model_row in core_a_rows:
                key = (model_row["program"], int(model_row["seed"]))
                source_record = old_records.get(key)
                if source_record is None or source_record["checkpoint_sha256"] != model_row["checkpoint_sha256"]:
                    raise ValueError(f"Core A reuse source model identity differs: {key}")
                source_path = Path(source_record["path"]).resolve()
                if not source_path.is_relative_to(old_root.resolve()) or sha256_file(source_path) != source_record["sha256"]:
                    raise ValueError(f"Core A reuse source output is not intact: {source_path}")
                source_sidecar = source_path.with_suffix(".json")
                sidecar_value = json.loads(source_sidecar.read_text(encoding="utf-8"))
                if sidecar_value.get("output_sha256") != source_record["sha256"]:
                    raise ValueError(f"Core A reuse source sidecar differs: {source_sidecar}")
                destination = attempt_root / "core_a" / key[0] / f"seed{key[1]}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.with_suffix(".json").exists():
                    raise FileExistsError(destination)
                shutil.copy2(source_path, destination)
                if sha256_file(destination) != source_record["sha256"]:
                    raise ValueError(f"Core A reused output copy hash mismatch: {destination}")
                sidecar_value["output_path"] = str(destination)
                sidecar_value["reused_from_attempt"] = reuse_core_a_from
                sidecar_value["reused_from_path"] = str(source_path)
                atomic_json(destination.with_suffix(".json"), sidecar_value)
                reused_record = dict(source_record)
                reused_record.update({
                    "path": str(destination), "reused": True,
                    "reused_from_attempt": reuse_core_a_from,
                    "reused_from_path": str(source_path),
                    "forward_rows_this_attempt": 0,
                })
                record["prediction_records"].append(reused_record)
                atomic_json(status_path, record)
            record["core_a_reused_from_attempt"] = reuse_core_a_from
        else:
            for row in core_a_rows:
                condition = "dynamic" if row["program"] == "A_DYNAMIC" else "static"
                manifest = RUN_ROOT / "replay/core_a" / f"{condition}_dev_manifest.csv"
                seed = int(row["seed"])
                output = attempt_root / "core_a" / row["program"] / f"seed{seed}.npz"
                started = time.perf_counter()
                sidecar = core_a.predict_manifest(
                    manifest_path=manifest,
                    checkpoint_path=Path(row["checkpoint_path"]),
                    checkpoint_sha256=row["checkpoint_sha256"],
                    run_id=f"EE6008_FINAL_TEST_V1_20261019T041642Z_{row['program']}_{seed}_DEV_REPLAY",
                    output_path=output,
                    device="cuda",
                    expected_rows=617,
                    output_root=attempt_root,
                )
                record["prediction_records"].append({
                    "core": "A", "program": row["program"], "seed": seed,
                    "path": str(output), "sha256": sidecar["output_sha256"],
                    "rows": 617, "shape": sidecar["logits_shape"], "dtype": sidecar["logits_dtype"],
                    "checkpoint_sha256": sidecar["checkpoint_sha256"],
                    "feature_manifest_sha256": sidecar["feature_manifest_sha256"],
                    "episode_index_sha256": sidecar["episode_index_sha256"],
                    "canonical_identity_map": str(RUN_ROOT / "replay/core_a/core_a_dev_identity.csv"),
                    "elapsed_seconds": time.perf_counter() - started,
                    "forward_rows": 617,
                    "forward_rows_this_attempt": 617,
                })
                atomic_json(status_path, record)

        cb_sources = core_b.load_accepted_sources()
        q0_manifest = json.loads(CORE_B_Q0_MANIFEST.read_text(encoding="utf-8"))
        if sha256_file(CORE_B_Q0_MANIFEST) != Q0_SHA:
            raise ValueError("accepted Q0 cache manifest hash changed")
        q0 = cb_sources.q0_grid_class(q0_manifest)
        prefix_path = RUN_ROOT / "replay/core_b/DEV_PREFIXES_NO_LABELS.npz"
        prefix_sha = sha256_file(prefix_path)
        with np.load(prefix_path, allow_pickle=False) as archive:
            canonical_ids = archive["canonical_id"].astype(str)
            anchor_native = archive["anchor_native"].copy()
        if len(canonical_ids) != 185 or len(set(canonical_ids.tolist())) != 185:
            raise ValueError("Core B canonical dev prefix IDs are not the accepted 185-row population")
        core_b_prediction_paths: dict[tuple[str, int], Path] = {}
        for row in core_b_rows:
            program = row["program"]
            seed = int(row["seed"])
            mode = row["mode"]
            video = mode == "TS11-D"
            model_cell = core_b.ModelCell(
                program_id=program,
                seed=seed,
                video_mode=mode,
                model_path=Path(row["checkpoint_path"]),
                expected_model_sha256=row["checkpoint_sha256"],
                model_manifest_path=Path(row["model_manifest_path"]),
                expected_model_manifest_sha256=row["model_manifest_sha256"],
                normalizer_path=Path(row["normalizer_path"]),
                expected_normalizer_sha256=row["normalizer_sha256"],
                prefix_path=prefix_path,
                expected_prefix_sha256=prefix_sha,
                feature_cache_manifest_path=CORE_B_Q0_MANIFEST if video else None,
                expected_feature_cache_manifest_sha256=Q0_SHA if video else None,
            )
            output = attempt_root / "core_b" / program / f"seed{seed}.npz"
            inference_profile = set_accepted_inference_profile(program, seed)
            started = time.perf_counter()
            bundle, artifact = core_b.predict_cell(
                model_cell,
                canonical_ids=canonical_ids,
                split="dev",
                feature_reader=q0 if video else None,
                output_path=output,
                device=device,
                batch_size=64,
            )
            if artifact is None:
                raise AssertionError("Core B dev prediction did not write its safe output")
            core_b_prediction_paths[(program, seed)] = output
            record["prediction_records"].append({
                "core": "B", "program": program, "seed": seed,
                "path": str(output), "sha256": artifact["file_sha256"],
                "canonical_ids_sha256": artifact["canonical_ids_sha256"],
                "prediction_native_sha256": artifact["prediction_native_sha256"],
                "rows": 185, "shape": artifact["shape"], "dtype": artifact["dtype"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "model_manifest_sha256": row["model_manifest_sha256"],
                "normalizer_sha256": row["normalizer_sha256"],
                "feature_cache_manifest_sha256": Q0_SHA if video else None,
                "prefix_sha256": prefix_sha,
                "inference_profile": inference_profile,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "elapsed_seconds": time.perf_counter() - started,
                "forward_rows": 185,
            })
            atomic_json(status_path, record)

        persistence = core_b.make_persistence_bundle(canonical_ids, anchor_native)
        persistence_path = attempt_root / "core_b/PERSISTENCE/reference.npz"
        persistence_artifact = core_b.write_prediction_bundle(persistence_path, persistence)
        aliases = core_b.make_persistence_aliases(persistence)
        persistence_alias_path = attempt_root / "core_b/PERSISTENCE/ALIASES.json"
        atomic_json(persistence_alias_path, {"aliases": [alias.__dict__ for alias in aliases], "reference_path": str(persistence_path), "reference_sha256": persistence_artifact["file_sha256"]})
        record["prediction_records"].append({
            "core": "B", "program": "PERSISTENCE", "seed": None,
            "path": str(persistence_path), "sha256": persistence_artifact["file_sha256"],
            "canonical_ids_sha256": persistence_artifact["canonical_ids_sha256"],
            "prediction_native_sha256": persistence_artifact["prediction_native_sha256"],
            "rows": 185, "shape": persistence_artifact["shape"], "dtype": persistence_artifact["dtype"],
            "forward_rows": 0,
        })

        for seed in SEEDS:
            for mix, video_program in (("MIX0", "V0"), ("MIXD", "VD")):
                source_path = core_b_prediction_paths[("S_OPT", seed)]
                video_path = core_b_prediction_paths[(video_program, seed)]
                with np.load(source_path, allow_pickle=False) as source, np.load(video_path, allow_pickle=False) as video:
                    source_ids = source["canonical_ids"].astype(str)
                    video_ids = video["canonical_ids"].astype(str)
                    if not np.array_equal(source_ids, video_ids):
                        raise ValueError(f"{mix}/{seed} source and video canonical IDs differ")
                    mixed = 0.5 * np.asarray(source["prediction_native"], dtype=np.float64) + 0.5 * np.asarray(video["prediction_native"], dtype=np.float64)
                if mixed.shape != (185, 30, 28) or not np.isfinite(mixed).all():
                    raise ValueError(f"{mix}/{seed} has invalid shape or values")
                mix_path = attempt_root / "core_b" / mix / f"seed{seed}.npz"
                artifact = atomic_npz(mix_path, canonical_ids=source_ids, prediction_native=mixed)
                atomic_json(mix_path.with_suffix(".json"), {
                    "schema_version": 1, "status": "PASS", "program": mix, "seed": seed,
                    "coefficient": 0.5, "source_program": "S_OPT", "video_program": video_program,
                    "source_prediction_sha256": sha256_file(source_path),
                    "video_prediction_sha256": sha256_file(video_path),
                    "canonical_ids_sha256": artifact["canonical_ids_sha256"],
                    "prediction_native_sha256": artifact["prediction_native_sha256"],
                    "output_sha256": artifact["file_sha256"], "shape": artifact["shape"],
                    "dtype": artifact["dtype"], "arithmetic": "native float64 after exact canonical ID order join",
                })
                record["prediction_records"].append({
                    "core": "B", "program": mix, "seed": seed,
                    "path": str(mix_path), "sha256": artifact["file_sha256"],
                    "canonical_ids_sha256": artifact["canonical_ids_sha256"],
                    "prediction_native_sha256": artifact["prediction_native_sha256"],
                    "rows": 185, "shape": artifact["shape"], "dtype": artifact["dtype"],
                    "forward_rows": 0,
                })

        if len(record["prediction_records"]) != 28:
            raise ValueError(f"dev replay bundle count mismatch: {len(record['prediction_records'])}")
        record["status"] = "PREDICTIONS_COMPLETE_PENDING_SEAL"
        record["completed_utc"] = utc_now()
        record["core_a_forward_rows"] = 6 * 617
        record["core_a_forward_rows_this_attempt"] = 0 if reuse_core_a_from else 6 * 617
        record["core_a_forward_rows_reused"] = 6 * 617 if reuse_core_a_from else 0
        record["core_b_forward_rows"] = 15 * 185
        record["core_b_forward_rows_this_attempt"] = 15 * 185
        record["neural_prediction_cells"] = 21
        record["derived_mixture_cells"] = 6
        record["persistence_references"] = 1
        record["new_training_runs"] = 0
        record["final_payload_read"] = False
        atomic_json(attempt_root / "DEV_REPLAY_RUN_RECEIPT.json", record)
        atomic_json(status_path, record)
        print(json.dumps({"status": record["status"], "attempt_root": str(attempt_root), "prediction_records": len(record["prediction_records"]), "physical_gpu": 3}, sort_keys=True), flush=True)
        return record
    except Exception as exc:
        record["status"] = "FAILED"
        record["failed_utc"] = utc_now()
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        atomic_json(status_path, record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=("attempt_01", "attempt_02", "attempt_03", "attempt_04", "attempt_05", "attempt_06"), required=True)
    parser.add_argument("--reuse-core-a-from", choices=("attempt_03", "attempt_04", "attempt_05"))
    args = parser.parse_args()
    run(args.attempt, reuse_core_a_from=args.reuse_core_a_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

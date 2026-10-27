#!/usr/bin/env python3
"""Extract fixed, label-free Core B features for CURRENT_PROFILE_V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
V3 = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
HISTORICAL_D3 = V3 / "code/d3_extract_grid.py"
PROBE_JSON = OUT / "core_b/CORE_B_GPU_DIFFERENTIAL_PROBES_ATTEMPT_02.json"
PROBE_ARCHIVE = RUN / "core_b/gpu_probes/CURRENT_PROFILE_BRIDGE_FEATURES_ATTEMPT_02.npz"
CPU_AUDIT = OUT / "core_b/CORE_B_TRAIN_BRIDGE_CPU_AUDIT.json"
BRIDGE_CSV = V1_RUN / "replay/core_b/core_b_bridge_manifest.csv"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
VJEPA_SOURCE = PROJECT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
CHECKPOINT = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
CURRENT_LOCK = OUT / "core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
INPUT_MANIFEST = OUT / "core_b/DEV_CURRENT_PROFILE_INPUTS.csv"
PREFIX_PATH = RUN / "replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
OUTPUT_DIR = RUN / "replay/core_b/current_profile_dev"
EXPECTED_PHYSICAL_GPU = 3
GRID_SHAPE = (16, 4, 768)
FIXED_TRAIN_IDS = ("e001726_a01_f000149", "e001769_a00_f000119")
INPUT_FIELDS = (
    "dataset_role", "canonical_id", "source_key", "literal_source_key",
    "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index",
    "video_path", "parquet_path",
)
FORBIDDEN = ("target", "targets", "future", "label", "ground_truth", "teacher")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def causal_frame_indices(anchor_frame: int) -> np.ndarray:
    anchor = int(anchor_frame)
    if anchor < 119:
        raise ValueError(f"anchor cannot provide the fixed 120-row observation window: {anchor}")
    indices = np.floor(np.linspace(anchor - 119, anchor, 32)).astype(np.int64)
    if indices.shape != (32,) or len(np.unique(indices)) != 32 or int(indices[-1]) != anchor or int(indices.max()) > anchor:
        raise ValueError(f"current-profile frame schedule exceeds the observation anchor: {anchor}")
    return indices


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import approved extractor source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def read_input_manifest() -> list[dict[str, str]]:
    with INPUT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != INPUT_FIELDS:
            raise ValueError(f"dev feature manifest schema mismatch: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 185:
        raise ValueError(f"CURRENT_PROFILE_V2 expects 185 fixed dev rows, got {len(rows)}")
    ids = [row["anchor_id"] for row in rows]
    canonical = [row["canonical_id"] for row in rows]
    if len(set(ids)) != len(ids) or len(set(canonical)) != len(canonical):
        raise ValueError("dev input manifest contains duplicate anchor/canonical IDs")
    if any(row["dataset_role"] != "dev" for row in rows):
        raise ValueError("dev extractor refuses any non-dev input row")
    if any(row["task_id"] not in {"119", "94", "90", "141", "230"} for row in rows):
        raise ValueError("dev extractor input contains a task outside the frozen five-task population")
    if any(any(token in key.lower() for token in FORBIDDEN) for row in rows for key in row):
        raise ValueError("dev input manifest exposes a target/label/future field")
    return rows


def check_profile_lock(expected_sha: str) -> tuple[dict, str]:
    actual = sha_file(CURRENT_LOCK)
    if actual != expected_sha:
        raise ValueError("candidate lock hash differs from the pre-registered GPU job")
    lock = json.loads(CURRENT_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_CURRENT_PROFILE_DEV_FEATURES":
        raise ValueError("candidate profile lock is not frozen")
    if lock.get("route") != "R2" or lock.get("profile_id") != "CURRENT_PROFILE_V2":
        raise ValueError("candidate profile lock does not authorize CURRENT_PROFILE_V2")
    if lock.get("code_sha256", {}).get("dev_extractor") != sha_file(Path(__file__)):
        raise ValueError("extractor source changed after profile lock")
    if lock.get("inputs", {}).get("dev_manifest_sha256") != sha_file(INPUT_MANIFEST):
        raise ValueError("dev input manifest differs from candidate profile lock")
    if lock.get("inputs", {}).get("dev_prefix_sha256") != sha_file(PREFIX_PATH):
        raise ValueError("dev prefix archive differs from candidate profile lock")
    return lock, actual


def runtime_flags() -> dict[str, object]:
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def read_frames(d3, video: Path, wanted: np.ndarray) -> np.ndarray:
    mapping = d3.read_frames(video, wanted)
    return np.stack([mapping[int(index)] for index in wanted], axis=0)


def encode_clip(encoder, tensor: torch.Tensor, d3) -> tuple[np.ndarray, str, str]:
    if tuple(tensor.shape) != (3, 32, 384, 384) or tensor.dtype != torch.float32:
        raise ValueError(f"preprocess output contract mismatch: {tuple(tensor.shape)} {tensor.dtype}")
    device_tensor = tensor.unsqueeze(0).to("cuda:0")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), nullcontext():
        raw = encoder(device_tensor)[0]
    if tuple(raw.shape) != (9216, 768):
        raise ValueError(f"encoder raw-token contract mismatch: {tuple(raw.shape)}")
    raw32 = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    pooled, _ = d3.grid_pool(raw)
    if pooled.shape != GRID_SHAPE or not np.isfinite(raw32).all() or not np.isfinite(pooled).all():
        raise FloatingPointError("current-profile encoder returned nonfinite or wrong-shape output")
    return pooled, hash_array(raw32), hash_array(tensor.numpy())


def candidate_repeat_check(d3, preprocess, encoder, device: torch.device) -> dict:
    audit = json.loads(CPU_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_CPU_METADATA_FRAME_STORAGE_PREPROCESS_AUDIT":
        raise ValueError("fixed train bridge CPU audit is not PASS")
    audit_rows = {row["anchor_id"]: row for row in audit["rows"]}
    if set(audit_rows) != set(FIXED_TRAIN_IDS):
        raise ValueError("fixed train bridge IDs differ from candidate repeatability lock")
    bridge_rows = list(csv.DictReader(BRIDGE_CSV.open(encoding="utf-8-sig", newline="")))
    by_id = {r["anchor_id"]: r for r in bridge_rows if r["dataset_role"] == "train" and r["anchor_id"] in FIXED_TRAIN_IDS}
    if set(by_id) != set(FIXED_TRAIN_IDS):
        raise ValueError("fixed train bridge rows are missing")
    with np.load(PROBE_ARCHIVE, allow_pickle=False) as probe:
        expected = probe["P5_CURRENT_PROFILE_V2_REPEAT_0"].copy()
        repeat = probe["P6_CURRENT_PROFILE_V2_REPEAT_1"].copy()
    if not np.array_equal(expected, repeat):
        raise ValueError("pre-score P5/P6 candidate profile repeatability failed")
    got = []
    records = []
    for anchor_id in FIXED_TRAIN_IDS:
        row = by_id[anchor_id]
        anchor = int(row["anchor_frame"])
        wanted = causal_frame_indices(anchor)
        video = (RAW_ROOT / row["video_path"]).resolve()
        if not video.is_relative_to(RAW_ROOT.resolve()):
            raise ValueError(f"train bridge path escapes fixed raw root: {anchor_id}")
        frames = read_frames(d3, video, wanted)
        dynamic = preprocess(frames, resolution=384).to(device)
        static_raw = np.repeat(frames[-1:, ...], 32, axis=0)
        static = preprocess(static_raw, resolution=384).to(device)
        for mode, tensor in (("dynamic", dynamic), ("static", static)):
            pooled, raw_sha, prep_sha = encode_clip(encoder, tensor, d3)
            got.append(pooled)
            records.append({"anchor_id": anchor_id, "mode": mode, "raw_tokens_sha256": raw_sha, "preprocessed_sha256": prep_sha})
    got_array = np.stack(got).astype(np.float32, copy=False)
    if not np.array_equal(got_array, expected):
        delta = np.abs(got_array.astype(np.float64) - expected.astype(np.float64))
        raise ValueError(f"current extractor differs from locked P5 train bridge: max_abs={float(delta.max())}")
    return {
        "status": "PASS_BITWISE_MATCH_TO_P5_AND_P6",
        "fixed_train_anchor_ids": list(FIXED_TRAIN_IDS),
        "rows": len(got),
        "candidate_repeat_sha256": hash_array(expected),
        "extractor_match_sha256": hash_array(got_array),
        "records": records,
    }


def run(expected_lock_sha256: str, dry_run: bool = False) -> dict:
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # These guards precede all CUDA runtime/device queries.
    if physical != str(EXPECTED_PHYSICAL_GPU):
        raise RuntimeError(f"watcher did not select authorized physical GPU3: {physical!r}")
    if visible != "3":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must map physical GPU3, got {visible!r}")
    lock, lock_sha = check_profile_lock(expected_lock_sha256)
    rows = read_input_manifest()
    if sha_file(PROBE_JSON) != lock["source_evidence"]["gpu_probe_result_sha256"]:
        raise ValueError("pre-score GPU probe result changed after candidate profile lock")
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    if probe.get("final_payload_read") is not False or probe.get("future_targets_materialized") is not False:
        raise ValueError("train-only probe evidence reports prohibited data access")
    if dry_run:
        planned = []
        for row in rows:
            anchor = int(row["anchor_frame"])
            wanted = causal_frame_indices(anchor)
            if wanted[-1] != anchor or int(wanted.max()) > anchor or len(np.unique(wanted)) != 32:
                raise ValueError(f"invalid causal frame schedule: {row['anchor_id']}")
            planned.append({"anchor_id": row["anchor_id"], "first_frame": int(wanted[0]), "last_frame": int(wanted[-1])})
        return {"status": "DRY_RUN_PASS", "rows": len(rows), "frame_cutoff_pass": True,
                "first_anchor": rows[0]["anchor_id"], "last_anchor": rows[-1]["anchor_id"],
                "lock_sha256": lock_sha, "video_rows": planned}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after physical GPU3 guard")
    if int(torch.cuda.current_device()) != 0:
        raise RuntimeError("physical GPU3 must appear as logical cuda:0")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("physical GPU3 device identity differs from preflight")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    flags = runtime_flags()
    expected_flags = lock.get("encoder_profile", {}).get("numeric_profile", {}).get("required_runtime_flags")
    if not isinstance(expected_flags, dict):
        raise ValueError("candidate lock is missing encoder_profile.numeric_profile.required_runtime_flags")
    for key, value in expected_flags.items():
        if flags.get(key) != value:
            raise RuntimeError(f"runtime flag mismatch for {key}: expected {value!r}, got {flags.get(key)!r}")

    sys.dont_write_bytecode = True
    d3 = load_module(HISTORICAL_D3, "v2_current_profile_d3_source")
    if sha_file(HISTORICAL_D3) != lock["sources"]["historical_d3_extractor_sha256"]:
        raise RuntimeError("historical D3 extraction source changed after candidate lock")
    preprocess, load_base = d3.import_stack()
    preprocess_file = Path(sys.modules[preprocess.__module__].__file__).resolve()
    if sha_file(preprocess_file) != lock["sources"]["preprocess_module_sha256"]:
        raise RuntimeError("V-JEPA preprocessing module changed after candidate lock")
    if sha_file(CHECKPOINT) != lock["sources"]["encoder_checkpoint_sha256"]:
        raise RuntimeError("V-JEPA checkpoint changed after candidate lock")
    encoder, encoder_info = load_base(source_root=VJEPA_SOURCE, checkpoint=CHECKPOINT,
                                     device=torch.device("cuda:0"), num_frames=32, resolution=384)
    encoder.eval()
    if encoder.training:
        raise RuntimeError("encoder did not enter eval mode")
    if encoder_info.get("embed_dim") != 768 or encoder_info.get("num_frames") != 32 or encoder_info.get("resolution") != 384:
        raise RuntimeError(f"encoder contract differs from lock: {encoder_info}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    status_path = OUTPUT_DIR / "EXTRACTION_STATUS.json"
    atomic_json(status_path, {
        "schema_version": 1, "status": "RUNNING", "started_utc": utc(),
        "profile_lock_sha256": lock_sha, "dev_rows": len(rows),
        "physical_gpu": 3, "logical_device": "cuda:0",
        "encoder_profile": lock["encoder_profile"], "runtime_flags": flags,
        "future_targets_materialized": False, "final_payload_read": False,
    })
    try:
        repeat_receipt = candidate_repeat_check(d3, preprocess, encoder, torch.device("cpu"))
        grids = np.empty((len(rows), 2, *GRID_SHAPE), dtype=np.float32)
        row_records = []
        video_sha_cache: dict[str, str] = {}
        started = time.perf_counter()
        for i, row in enumerate(rows):
            anchor = int(row["anchor_frame"])
            wanted = causal_frame_indices(anchor)
            if wanted[-1] != anchor or int(wanted.max()) > anchor or len(np.unique(wanted)) != 32:
                raise ValueError(f"invalid causal frame schedule: {row['anchor_id']}")
            video = (RAW_ROOT / row["video_path"]).resolve()
            if not video.is_relative_to(RAW_ROOT.resolve()) or not video.is_file():
                raise FileNotFoundError(f"dev video is absent or escapes fixed raw root: {row['anchor_id']}")
            video_key = str(video)
            if video_key not in video_sha_cache:
                video_sha_cache[video_key] = sha_file(video)
            video_sha = video_sha_cache[video_key]
            frames = read_frames(d3, video, wanted)
            static_frames = np.repeat(frames[-1:, ...], 32, axis=0)
            rgb_sha = hash_array(frames)
            dynamic_tensor = preprocess(frames, resolution=384)
            static_tensor = preprocess(static_frames, resolution=384)
            dynamic_grid, dynamic_tokens_sha, dynamic_prep_sha = encode_clip(encoder, dynamic_tensor, d3)
            static_grid, static_tokens_sha, static_prep_sha = encode_clip(encoder, static_tensor, d3)
            grids[i, 0] = dynamic_grid
            grids[i, 1] = static_grid
            row_records.append({
                "order": i, "canonical_id": row["canonical_id"], "anchor_id": row["anchor_id"],
                "episode_id": int(row["episode_id"]), "task_id": int(row["task_id"]),
                "anchor_frame": anchor, "cache_index": int(row["cache_index"]),
                "video_path": row["video_path"], "video_sha256": video_sha,
                "frame_indices": wanted.tolist(), "frame_indices_sha256": hash_array(wanted.astype("<i8")),
                "rgb_frames_sha256": rgb_sha,
                "preprocessed_dynamic_sha256": dynamic_prep_sha,
                "preprocessed_static_sha256": static_prep_sha,
                "raw_tokens_dynamic_sha256": dynamic_tokens_sha,
                "raw_tokens_static_sha256": static_tokens_sha,
                "dynamic_grid_sha256": hash_array(dynamic_grid),
                "static_grid_sha256": hash_array(static_grid),
            })
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                atomic_json(status_path, {
                    "schema_version": 1, "status": "RUNNING", "started_utc": json.loads(status_path.read_text())["started_utc"],
                    "updated_utc": utc(), "completed_rows": i + 1, "expected_rows": len(rows),
                    "last_anchor_id": row["anchor_id"], "profile_lock_sha256": lock_sha,
                    "physical_gpu": 3, "logical_device": "cuda:0", "runtime_flags": flags,
                    "repeatability": repeat_receipt, "future_targets_materialized": False, "final_payload_read": False,
                })

        archive_path = OUTPUT_DIR / "DEV_CURRENT_PROFILE_V2_FEATURES.npz"
        if archive_path.exists():
            raise FileExistsError(archive_path)
        temp = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.part")
        with temp.open("wb") as stream:
            np.savez_compressed(
                stream, canonical_ids=np.asarray([r["canonical_id"] for r in rows], dtype="U768"),
                anchor_ids=np.asarray([r["anchor_id"] for r in rows], dtype="U64"),
                episode_ids=np.asarray([int(r["episode_id"]) for r in rows], dtype="<i8"),
                task_ids=np.asarray([int(r["task_id"]) for r in rows], dtype="<i8"),
                cache_indices=np.asarray([int(r["cache_index"]) for r in rows], dtype="<i8"),
                grid_features=grids,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, archive_path)
        manifest_path = OUT / "core_b/CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest = {
            "schema_version": 1, "status": "PASS",
            "profile_id": "CURRENT_PROFILE_V2", "profile_lock_sha256": lock_sha,
            "split": "dev", "rows": len(rows), "feature_shape": list(grids.shape),
            "feature_dtype": "float32", "feature_modes": ["dynamic", "static"],
            "canonical_id_order_sha256": hashlib.sha256("\n".join(r["canonical_id"] for r in rows).encode()).hexdigest(),
            "anchor_id_order_sha256": hashlib.sha256("\n".join(r["anchor_id"] for r in rows).encode()).hexdigest(),
            "feature_archive_path": str(archive_path), "feature_archive_sha256": sha_file(archive_path),
            "dev_input_manifest_path": str(INPUT_MANIFEST), "dev_input_manifest_sha256": sha_file(INPUT_MANIFEST),
            "dev_prefix_path": str(PREFIX_PATH), "dev_prefix_sha256": sha_file(PREFIX_PATH),
            "extractor_path": str(Path(__file__).resolve()), "extractor_sha256": sha_file(Path(__file__)),
            "historical_q0_cache_manifest_sha256": lock["sources"]["q0_cache_manifest"]["sha256"],
            "old_q0_bridge_status": "FAIL_UNRESOLVED",
            "row_records": row_records, "repeatability_check": repeat_receipt,
            "runtime_flags": flags, "encoder_info": encoder_info,
            "started_elapsed_seconds": time.perf_counter() - started,
            "future_targets_materialized": False, "final_payload_read": False,
            "completed_utc": utc(),
        }
        atomic_json(manifest_path, manifest)
        status = {
            "schema_version": 1, "status": "COMPLETED_FEATURES_PENDING_TECHNICAL_ACCEPTANCE",
            "started_utc": json.loads(status_path.read_text())["started_utc"],
            "completed_utc": utc(), "profile_lock_sha256": lock_sha,
            "dev_rows": len(rows), "completed_rows": len(rows), "feature_archive_sha256": manifest["feature_archive_sha256"],
            "feature_manifest_path": str(manifest_path), "feature_manifest_sha256": sha_file(manifest_path),
            "repeatability": repeat_receipt, "physical_gpu": 3, "logical_device": "cuda:0",
            "future_targets_materialized": False, "final_payload_read": False,
        }
        atomic_json(status_path, status)
        return status
    except Exception as exc:
        failure = json.loads(status_path.read_text(encoding="utf-8"))
        failure.update({"status": "FAILED", "failed_utc": utc(),
                        "error_type": type(exc).__name__, "error": str(exc)})
        atomic_json(status_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.expected_lock_sha256, args.dry_run), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

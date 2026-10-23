#!/usr/bin/env python3
"""Run at most six controlled train-only D3/Q0 numerical diagnostics on physical GPU3."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
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
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
D3_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
D3_SOURCE = D3_ROOT / "code/d3_extract_grid.py"
CACHE_MANIFEST = D3_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
BRIDGE = V1_RUN / "replay/core_b/core_b_bridge_manifest.csv"
CPU_AUDIT = OUT / "core_b/CORE_B_TRAIN_BRIDGE_CPU_AUDIT.json"
PROBE_DIR = RUN / "core_b/gpu_probes"
EXPECTED_PHYSICAL_GPU = 3
FIXED_IDS = ("e001726_a01_f000149", "e001769_a00_f000119")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + str(os.getpid()) + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def profile_flags() -> dict[str, object]:
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


def diff_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(reference, dtype=np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"comparison arrays invalid: {a.shape} vs {b.shape}")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = float(np.linalg.norm(b.astype(np.float64).ravel()))
    return {
        "shape": list(a.shape),
        "max_abs_difference": float(delta.max(initial=0.0)),
        "mean_abs_difference": float(delta.mean()) if delta.size else 0.0,
        "p95_abs_difference": float(np.quantile(delta, 0.95)) if delta.size else 0.0,
        "p99_abs_difference": float(np.quantile(delta, 0.99)) if delta.size else 0.0,
        "relative_l2_difference": float(np.linalg.norm(delta.ravel()) / max(denom, 1e-30)),
        "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-physical-gpu", type=int, default=3)
    parser.add_argument("--attempt-suffix", type=str, default="01")
    args = parser.parse_args()
    if not args.attempt_suffix.isdigit():
        raise ValueError("attempt suffix must contain digits only")
    ATTEMPT = PROBE_DIR / f"CORE_B_GPU_PROBE_ATTEMPT_{args.attempt_suffix}.json"
    RESULT = OUT / "core_b" / f"CORE_B_GPU_DIFFERENTIAL_PROBES_ATTEMPT_{args.attempt_suffix}.json"
    FEATURES = PROBE_DIR / f"CURRENT_PROFILE_BRIDGE_FEATURES_ATTEMPT_{args.attempt_suffix}.npz"
    physical_gpu = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # This guard precedes every torch.cuda API and model/device construction.
    if physical_gpu != str(args.expected_physical_gpu) or args.expected_physical_gpu != EXPECTED_PHYSICAL_GPU:
        raise RuntimeError(f"watcher selected physical GPU {physical_gpu!r}; V2 authorizes only physical GPU 3")
    if visible != "3":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be physical GPU 3, got {visible!r}")
    if ATTEMPT.exists() or RESULT.exists() or FEATURES.exists():
        raise FileExistsError("V2 GPU probe attempt/output already exists; refusing duplicate run")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    attempt = {"schema_version": 1, "attempt_id": f"CORE_B_GPU_PROBE_ATTEMPT_{args.attempt_suffix}", "status": "STARTED",
        "started_utc": utc(), "physical_gpu": 3, "cuda_visible_devices": visible,
        "logical_device_expected": "cuda:0", "scope": "two fixed train rows; no dev outcomes, final payload, or targets",
        "new_training_runs": 0, "probe_configurations_allowed": 6, "probe_configurations_completed": 0,
        "completed_probe_ids": []}
    atomic_json(ATTEMPT, attempt)
    def mark_probe_completed(probe_id):
        attempt["status"] = "RUNNING_PROBES"
        attempt["completed_probe_ids"].append(probe_id)
        attempt["probe_configurations_completed"] = len(attempt["completed_probe_ids"])
        attempt["last_probe_completed_utc"] = utc()
        atomic_json(ATTEMPT, attempt)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable after the watcher selected physical GPU 3")
        logical_index = int(torch.cuda.current_device())
        if logical_index != 0:
            raise RuntimeError(f"expected process-local cuda:0 after physical GPU 3 masking, got cuda:{logical_index}")
        gpu_name = torch.cuda.get_device_name(logical_index)

        cpu_audit = json.loads(CPU_AUDIT.read_text(encoding="utf-8"))
        if cpu_audit.get("status") != "PASS_CPU_METADATA_FRAME_STORAGE_PREPROCESS_AUDIT":
            raise RuntimeError("two-row CPU audit has not passed")
        audit_rows = {row["anchor_id"]: row for row in cpu_audit["rows"]}
        if set(audit_rows) != set(FIXED_IDS):
            raise RuntimeError("CPU audit fixed IDs differ from the GPU probe IDs")
        bridge = list(csv.DictReader(BRIDGE.open(encoding="utf-8-sig", newline="")))
        by_id = {row["anchor_id"]: row for row in bridge if row["dataset_role"] == "train" and row["anchor_id"] in FIXED_IDS}
        if set(by_id) != set(FIXED_IDS):
            raise RuntimeError("both frozen train bridge IDs must exist")

        d3 = load_module(D3_SOURCE, "v2_d3_gpu_probe_source")
        preprocess, load_base = d3.import_stack()
        clips_module = sys.modules[preprocess.__module__]
        if sha_file(D3_SOURCE) != cpu_audit["source_hashes"]["d3_extractor_script_sha256"]:
            raise RuntimeError("historical D3 source hash changed after the CPU audit")
        if sha_file(Path(clips_module.__file__).resolve()) != cpu_audit["source_hashes"]["preprocess_module_sha256"]:
            raise RuntimeError("preprocessing source hash changed after the CPU audit")
        manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
        q0mod = load_module(PROJECT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/code/prepare_v2_data.py", "v2_q0_for_gpu_probe")
        q0_reader = q0mod.Q0ShardedGrid(manifest)

        # Prepare exactly four clips in fixed order: row1 dynamic/static, row2 dynamic/static.
        clips = []
        q0_rows = []
        frame_records = []
        for anchor_id in FIXED_IDS:
            row = by_id[anchor_id]
            arow = audit_rows[anchor_id]
            video = (d3.RAW_ROOT / row["video_path"]).resolve()
            if not video.is_relative_to(d3.RAW_ROOT.resolve()):
                raise RuntimeError("train video path escaped the fixed raw root")
            wanted = np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64)
            frame_map = d3.read_frames(video, wanted)
            frames = np.stack([frame_map[int(i)] for i in wanted], axis=0)
            static = np.repeat(frames[-1:, ...], 32, axis=0)
            for mode, raw in (("dynamic", frames), ("static", static)):
                tensor = preprocess(raw, resolution=384).detach().cpu().contiguous()
                if tuple(tensor.shape) != (3, 32, 384, 384) or tensor.dtype != torch.float32:
                    raise RuntimeError(f"preprocessed tensor contract changed for {anchor_id}/{mode}")
                expected_prep = next(item for item in arow["preprocessed"] if item["mode"] == mode)
                got_prep_sha = hash_array(tensor.numpy())
                if got_prep_sha != expected_prep["sha256"]:
                    raise RuntimeError(f"preprocessed tensor hash differs from CPU audit for {anchor_id}/{mode}")
                clips.append({"anchor_id": anchor_id, "mode": mode, "tensor": tensor, "preprocessed_sha256": got_prep_sha})
            q0_rows.append(q0_reader.row(int(row["cache_index"]), anchor_id, str(row["episode_id"])))
            frame_records.append({"anchor_id": anchor_id, "video_sha256": sha_file(video),
                "frame_indices": wanted.tolist(), "frame_indices_sha256": hash_array(wanted.astype("<i8")),
                "rgb_frames_sha256": hash_array(frames), "preprocessed_dynamic_sha256": arow["preprocessed"][0]["sha256"],
                "preprocessed_static_sha256": arow["preprocessed"][1]["sha256"],
                "q0_row_sha256": hash_array(q0_rows[-1])})

        encoder, encoder_info = load_base(source_root=d3.VJEPA_SOURCE, checkpoint=d3.CHECKPOINT,
            device=torch.device("cuda:0"), num_frames=32, resolution=384)
        encoder.eval()
        if encoder.training:
            raise RuntimeError("encoder did not enter eval mode")
        baseline_flags = profile_flags()

        def run_batch(indices, *, amp, batch_label, math_sdp=False, tf32_off=False):
            all_pool, all_raw_float32, all_raw_dtype = [], [], []
            for start in range(0, len(indices), batch_label):
                selected = indices[start:start + batch_label]
                batch = torch.stack([clips[i]["tensor"] for i in selected], dim=0).to("cuda:0")
                if math_sdp:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    sdp_context = sdpa_kernel(SDPBackend.MATH)
                else:
                    sdp_context = nullcontext()
                if tf32_off:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else nullcontext()
                with torch.inference_mode(), sdp_context, autocast_context:
                    tokens_batch = encoder(batch)
                if tokens_batch.ndim != 3 or tokens_batch.shape[0] != len(selected):
                    raise RuntimeError(f"raw token output batch mismatch: {tuple(tokens_batch.shape)}")
                for offset, token in enumerate(tokens_batch):
                    raw_dtype = str(token.dtype)
                    raw32 = token.detach().float().cpu().contiguous().numpy()
                    pooled, bridge_error = d3.grid_pool(token)
                    if not np.isfinite(raw32).all() or not np.isfinite(pooled).all():
                        raise FloatingPointError("nonfinite raw token or grid output")
                    all_pool.append(pooled)
                    all_raw_float32.append(raw32)
                    all_raw_dtype.append(raw_dtype)
                del batch, tokens_batch
            return np.stack(all_pool), all_raw_float32, all_raw_dtype

        variants = {}
        labels = [0, 1, 2, 3]
        # P1: source-faithful D3 numeric path (BF16 autocast, batch 1, environment defaults).
        legacy, legacy_raw, legacy_dtype = run_batch(labels, amp=True, batch_label=1)
        mark_probe_completed("P1_D3_BF16_BATCH1_DEFAULT")
        variants["P1_D3_BF16_BATCH1_DEFAULT"] = legacy
        # P2: one factor from P1 changes: autocast disabled; same batch and backend defaults.
        fp32, fp32_raw, fp32_dtype = run_batch(labels, amp=False, batch_label=1)
        mark_probe_completed("P2_FP32_BATCH1_DEFAULT")
        variants["P2_FP32_BATCH1_DEFAULT"] = fp32
        # P3: one factor from P2 changes: SDPA backend forced to math.
        fp32_math, fp32_math_raw, fp32_math_dtype = run_batch(labels, amp=False, batch_label=1, math_sdp=True)
        mark_probe_completed("P3_FP32_BATCH1_MATH_SDPA")
        variants["P3_FP32_BATCH1_MATH_SDPA"] = fp32_math
        # P4: one factor from P2 changes: both TF32 flags are disabled as one TF32 policy.
        fp32_tf32off, fp32_tf32off_raw, fp32_tf32off_dtype = run_batch(labels, amp=False, batch_label=1, tf32_off=True)
        mark_probe_completed("P4_FP32_BATCH1_TF32_OFF")
        variants["P4_FP32_BATCH1_TF32_OFF"] = fp32_tf32off
        # P5/P6: fixed candidate profile, repeated twice for a pre-score stability gate.
        candidate0, candidate0_raw, candidate0_dtype = run_batch(labels, amp=False, batch_label=1, math_sdp=True, tf32_off=True)
        mark_probe_completed("P5_CURRENT_PROFILE_V2_REPEAT_0")
        candidate1, candidate1_raw, candidate1_dtype = run_batch(labels, amp=False, batch_label=1, math_sdp=True, tf32_off=True)
        mark_probe_completed("P6_CURRENT_PROFILE_V2_REPEAT_1")
        variants["P5_CURRENT_PROFILE_V2_REPEAT_0"] = candidate0
        variants["P6_CURRENT_PROFILE_V2_REPEAT_1"] = candidate1

        raw_by_variant = {
            "P1_D3_BF16_BATCH1_DEFAULT": legacy_raw,
            "P2_FP32_BATCH1_DEFAULT": fp32_raw,
            "P3_FP32_BATCH1_MATH_SDPA": fp32_math_raw,
            "P4_FP32_BATCH1_TF32_OFF": fp32_tf32off_raw,
            "P5_CURRENT_PROFILE_V2_REPEAT_0": candidate0_raw,
            "P6_CURRENT_PROFILE_V2_REPEAT_1": candidate1_raw,
        }
        dtype_by_variant = {
            "P1_D3_BF16_BATCH1_DEFAULT": legacy_dtype,
            "P2_FP32_BATCH1_DEFAULT": fp32_dtype,
            "P3_FP32_BATCH1_MATH_SDPA": fp32_math_dtype,
            "P4_FP32_BATCH1_TF32_OFF": fp32_tf32off_dtype,
            "P5_CURRENT_PROFILE_V2_REPEAT_0": candidate0_dtype,
            "P6_CURRENT_PROFILE_V2_REPEAT_1": candidate1_dtype,
        }
        q0_stack = np.stack([np.asarray(row) for row in q0_rows])
        q0_modes = np.stack([q0_stack[0], q0_stack[1]]).reshape(4, 16, 4, 768)
        baseline_raw_stack = legacy_raw
        comparison = {}
        for name, pooled in variants.items():
            comparison[name] = {
                "pooled_vs_q0_by_input": [diff_stats(pooled[i], q0_modes[i]) for i in range(4)],
                "pooled_vs_legacy_d3_by_input": [diff_stats(pooled[i], legacy[i]) for i in range(4)],
                "raw_token_dtype_by_input": dtype_by_variant[name],
                "raw_token_shape_by_input": [list(x.shape) for x in raw_by_variant[name]],
                "raw_token_sha256_by_input": [hash_array(x) for x in raw_by_variant[name]],
                "raw_token_vs_legacy_by_input": [diff_stats(raw_by_variant[name][i], baseline_raw_stack[i]) for i in range(4)],
                "pooled_sha256_by_input": [hash_array(pooled[i]) for i in range(4)],
                "all_finite": bool(np.isfinite(pooled).all()),
            }
        profile_repeatability = {
            "raw_tokens": [diff_stats(candidate1_raw[i], candidate0_raw[i]) for i in range(4)],
            "pooled_grid": [diff_stats(candidate1[i], candidate0[i]) for i in range(4)],
            "repeated_profile_allclose_1e-5": all(np.allclose(candidate0[i], candidate1[i], atol=1e-5, rtol=1e-5) for i in range(4)),
        }
        feature_payload = {name: arr.astype(np.float32, copy=False) for name, arr in variants.items()}
        feature_temp = FEATURES.with_name("." + FEATURES.name + "." + str(os.getpid()) + ".part")
        with feature_temp.open("wb") as f:
            np.savez_compressed(f, **feature_payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(feature_temp, FEATURES)
        final_flags = profile_flags()
        report = {
            "schema_version": 1,
            "status": "COMPLETED_TRAIN_ONLY_CONTROLLED_GPU3_PROBES",
            "created_utc": utc(),
            "protocol_id": OUT.name,
            "physical_gpu": 3,
            "logical_device": "cuda:0",
            "cuda_visible_devices": visible,
            "gpu_name": gpu_name,
            "scope": "two fixed train rows, dynamic/static; no dev outcomes, final payload, or targets",
            "new_training_runs": 0,
            "source_hashes": {
                "historical_d3_extractor_sha256": sha_file(D3_SOURCE),
                "preprocess_module_sha256": sha_file(Path(clips_module.__file__).resolve()),
                "checkpoint_sha256": sha_file(d3.CHECKPOINT),
                "q0_manifest_sha256": sha_file(CACHE_MANIFEST),
                "accepted_q0_reader_sha256": sha_file(PROJECT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/code/prepare_v2_data.py"),
                "gpu_probe_script_sha256": sha_file(Path(__file__)),
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torch_cuda_build": str(torch.version.cuda),
                "encoder_info": encoder_info,
                "encoder_eval": True,
                "baseline_backend_flags": baseline_flags,
                "flags_after_probe_set": final_flags,
                "new_profile_attention_api": "torch.nn.attention.sdpa_kernel(SDPBackend.MATH)",
            },
            "rows": frame_records,
            "probe_design": [
                {"probe_id": "P1", "config": "BF16 autocast, batch 1, source runtime backend defaults", "factor": "reference; historical source recipe replayed at current GPU3/runtime"},
                {"probe_id": "P2", "config": "autocast disabled, batch 1, backend defaults", "factor": "precision/autocast only versus P1"},
                {"probe_id": "P3", "config": "FP32, batch 1, math SDPA", "factor": "attention backend only versus P2"},
                {"probe_id": "P4", "config": "FP32, batch 1, TF32 disabled", "factor": "TF32 policy only versus P2"},
                {"probe_id": "P5", "config": "FP32, batch 1, math SDPA, TF32 disabled", "factor": "fixed candidate-profile repeatability pass 1; composite profile, not a causal contrast"},
                {"probe_id": "P6", "config": "same as P5", "factor": "fixed candidate-profile repeatability pass 2; not a causal contrast"},
            ],
            "comparisons": comparison,
            "candidate_profile_repeatability": profile_repeatability,
            "candidate_profile_v2": {
                "precision": "FP32, autocast disabled",
                "batch_size": 1,
                "tf32": "matmul and cuDNN TF32 disabled",
                "attention": "SDPA math backend forced where the encoder uses SDPA",
                "cudnn_benchmark_and_deterministic": "record actual runtime values; do not alter in candidate profile",
                "deterministic_algorithms": "record actual runtime value; do not alter in candidate profile",
                "reason": "A fully specified, full-precision inference profile with fixed batch/backend policy; selected for source clarity and repeatability, never for similarity or dev performance.",
            },
            "feature_archive_path": str(FEATURES),
            "feature_archive_sha256": sha_file(FEATURES),
            "final_payload_read": False,
            "future_targets_materialized": False,
        }
        atomic_json(RESULT, report)
        attempt.update({"status": "COMPLETED", "completed_utc": utc(), "probe_configurations_completed": 6,
            "result_path": str(RESULT), "result_sha256": sha_file(RESULT), "feature_archive_path": str(FEATURES),
            "feature_archive_sha256": sha_file(FEATURES), "cuda_device_logical": logical_index})
        atomic_json(ATTEMPT, attempt)
        print(json.dumps({"status": report["status"], "physical_gpu": 3, "gpu": gpu_name,
            "candidate_repeatable": profile_repeatability["repeated_profile_allclose_1e-5"],
            "result": str(RESULT)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        attempt.update({"status": "FAILED", "failed_utc": utc(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc()})
        atomic_json(ATTEMPT, attempt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

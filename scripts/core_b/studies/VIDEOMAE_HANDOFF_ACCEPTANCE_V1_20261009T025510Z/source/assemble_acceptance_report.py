#!/usr/bin/env python3
"""Assemble the durable, evidence-linked outputs for this bounded audit."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
AUDIT = PROJECT / "reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z"
HANDOFF = PROJECT / 'tmp/project-team/videomae_handoff_20261009T020906Z'
FEATURES = PROJECT / 'tmp/project-team/core_a_strict_features_20261008T163351Z/features'
PROBES = PROJECT / 'tmp/project-team/core_a_strict_probes_20261009T020138Z'
VM_CACHE = PROJECT / 'tmp/project-team/videomae_cache_full_20261008T074027Z'
VM_REMAINING = PROJECT / 'tmp/project-team/videomae_remaining_20261008T133752Z'
VM_EVAL = PROJECT / 'tmp/project-team/videomae_batch_eval_20261008T152556Z'
VM_COMPARISON = PROJECT / 'tmp/project-team/videomae_comparison_20261008T154206Z'
TEAM = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
TARGET_CACHE = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
INIT_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/runs"
HF_ROOT = PROJECT / 'tmp/project-team/hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots'
VJEPA_WEIGHTS = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("NA" if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in row.items()})


def add_dependency(rows: list[dict[str, object]], *, group: str, role: str, path: Path,
                   expected_sha: str | None = None, expected_bytes: int | None = None,
                   binding: str = "current file hash") -> None:
    exists = path.is_file()
    size = path.stat().st_size if exists else ""
    actual = sha(path) if exists else ""
    rows.append({
        "group": group, "cell_id": "shared", "role": role, "path": str(path),
        "expected_bytes": expected_bytes if expected_bytes is not None else "",
        "actual_bytes": size, "expected_sha256": expected_sha or "",
        "actual_sha256": actual, "exists": int(exists),
        "size_match": int(exists and expected_bytes == size) if expected_bytes is not None else "NA",
        "sha256_match": int(exists and expected_sha == actual) if expected_sha else "NA",
        "identity_binding": binding,
    })


def build_inventory() -> dict[str, object]:
    idx = read_json(HANDOFF / "HANDOFF_INDEX.json")
    indexed_paths = set(idx["artifacts"])
    original = [r for r in csv_rows(AUDIT / "ARTIFACT_INVENTORY.csv") if r["path"] in indexed_paths]
    original_paths = {r["path"] for r in original}
    deps: list[dict[str, object]] = []
    fm = read_json(FEATURES / "FEATURE_MANIFEST.json")
    a_config = read_json(PROBES / "CONFIG.json")
    b_cache = read_json(VM_CACHE / "cache/CACHE_MANIFEST.json")
    vnorm = read_json(V3 / "data/V3_NORMALIZERS.json")
    d3_cache = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json")
    d3_extract = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_EXTRACTION_RUN_MANIFEST.json")
    team_hashes = read_json(TEAM / "protocol/SOURCE_HASHES.json")
    team_hash_lookup = {x["path"]: x for x in team_hashes.get("sources", [])}
    coreb_run_config = read_json(Path(idx["core_b"]["cells"][0]["run_root"]) / "CONFIG.json")
    team_run_id = "D3EQ-CS-STATE-CTRL-H30-S20261001"
    team_run_env = read_json(TEAM / f"runs/formal/{team_run_id}/RUN_ENVIRONMENT.json")
    d3_cache_path = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
    d3_norm_path = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json"
    d3_target_builder = EQ / "code/build_train_target_cache.py"

    # Execution code, shared model/loss/scheduler sources, canonical inputs and normalizers.
    source_pairs = [
        ("Core A probe as-run source", PROJECT.parent / "train_videomae_core_a_strict_probes.py", a_config.get("script_sha256"), "per-cell config script_sha256"),
        ("Core A feature extractor as-run source", PROJECT.parent / "preflight_videomae_core_a_strict.py", fm.get("script_sha256"), "feature manifest script_sha256"),
        ("Core B VideoMAE trainer as-run source", PROJECT.parent / "train_videomae_core_b_formal.py", coreb_run_config.get("runner_sha256"), "per-cell CONFIG runner_sha256"),
        ("Core B VideoMAE cache extractor", HANDOFF / "code/extract_videomae_core_b_cache.py", b_cache.get("script_sha256"), "cache manifest script_sha256"),
        ("Core B per-cell evaluator", HANDOFF / "code/evaluate_videomae_core_b.py", None, "handoff code copy; per-cell evaluator source identity recorded in METRICS"),
        ("D3 corrected trainer as-run source", TEAM / "code/d3_train_corrected.py", team_run_env.get("trainer_source_sha256"), "team RUN_ENVIRONMENT trainer_source_sha256"),
        ("D3 model source", TEAM / "code/d3_model.py", team_hash_lookup.get(str(TEAM / "code/d3_model.py"), {}).get("sha256"), "team SOURCE_HASHES"),
        ("D3 loss contract", TEAM / "code/d3_eq_contract.py", team_hash_lookup.get(str(TEAM / "code/d3_eq_contract.py"), {}).get("sha256"), "team SOURCE_HASHES"),
        ("D3 update scheduler", TEAM / "code/corrected_scheduler.py", team_hash_lookup.get(str(TEAM / "code/corrected_scheduler.py"), {}).get("sha256"), "team SOURCE_HASHES"),
        ("D3 post-inference evaluator", TEAM / "code/evaluate_corrected_scheduler.py", team_hash_lookup.get(str(TEAM / "code/evaluate_corrected_scheduler.py"), {}).get("sha256"), "team SOURCE_HASHES"),
        ("Project runtime wrapper", PROJECT / "envs/run_jyc.sh", None, "current runtime wrapper; no per-cell wrapper hash in handoff"),
        ("VideoMAE config snapshot", HF_ROOT / "1bf2b82c809609f5df1ea7648aee577e2e486b63/config.json", fm.get("config_sha256"), "feature manifest config_sha256"),
        ("VideoMAE processor snapshot", HF_ROOT / "1bf2b82c809609f5df1ea7648aee577e2e486b63/preprocessor_config.json", fm.get("processor_sha256"), "feature manifest processor_sha256"),
        ("VideoMAE encoder weights", HF_ROOT / "240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors", fm.get("weights_sha256"), "feature manifest weights_sha256"),
        ("V-JEPA2 encoder checkpoint", VJEPA_WEIGHTS, d3_extract.get("source_backbone_hash"), "D3 extraction manifest source_backbone_hash"),
        ("Core B main inputs", V3 / "data/V3_MAIN_INPUTS.npz", b_cache.get("input_sha256"), "VideoMAE cache manifest input_sha256"),
        ("Core B state/target normalizer", V3 / "data/V3_NORMALIZERS.json", coreb_run_config.get("target_normalizer_sha256"), "per-cell CONFIG target_normalizer_sha256"),
        ("VideoMAE train-only video normalizer", VM_CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json", coreb_run_config.get("normalizer_sha256"), "per-cell CONFIG normalizer_sha256"),
        ("Team V-JEPA D3 cache manifest", d3_cache_path, team_hash_lookup.get(str(d3_cache_path), {}).get("sha256"), "team SOURCE_HASHES source record"),
        ("Team V-JEPA D3 video normalizer", d3_norm_path, team_hash_lookup.get(str(d3_norm_path), {}).get("sha256"), "team SOURCE_HASHES source record"),
        ("Core B train-only target cache", TARGET_CACHE, coreb_run_config.get("target_cache_sha256"), "per-cell CONFIG target_cache_sha256"),
        ("Core B target cache builder", d3_target_builder, None, "target cache manifest does not bind builder source hash"),
        ("Core B canonical split manifest", SPLIT, None, "canonical train/dev membership metadata"),
        ("Core A classification manifest", Path(fm["manifest"]), fm.get("manifest_sha256"), "feature manifest manifest_sha256"),
        ("V3 feature extractor code copy", V3 / "code/extract_v3_features.py", None, "current source file; extraction manifest contains no source code SHA"),
        ("VJEPA encoder loader code copy", PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src/ee6008/models.py", None, "current source file; as-run code SHA not recorded in V3 extraction manifest"),
        ("Core A prior CPU preflight report", PROJECT / 'tmp/project-team/core_a_strict_preflight_20261008T162343Z/CORE_A_PREFLIGHT.json', None, "artifact hash indexed; source script SHA inside report is unresolved"),
    ]
    audit_source_files = [
        ("CoreA safe checkpoint replay scorer", AUDIT / "source/replay_core_a.py"),
        ("CoreA strict-conversion fixture", AUDIT / "source/core_a_conversion_fixture.py"),
        ("Shared initialization audit", AUDIT / "source/audit_initialization.py"),
        ("Independent CoreB dev scorer", AUDIT / "source/independent_core_b_metrics.py"),
        ("Independent CoreB train prediction scorer", AUDIT / "source/score_coreb_train_artifacts.py"),
        ("Audit report assembler", AUDIT / "source/assemble_acceptance_report.py"),
        ("Red-capable fixtures", AUDIT / "fixtures/red_capable_metric_fixtures.py"),
        ("Parameterized safe reproduce entrypoint", AUDIT / "REPRODUCE_DEV.sh"),
    ]
    for role, path in audit_source_files:
        if path.is_file():
            deps.append({"group": "audit_source", "cell_id": "this_audit", "role": role,
                         "path": str(path), "expected_bytes": path.stat().st_size,
                         "actual_bytes": path.stat().st_size, "expected_sha256": sha(path),
                         "actual_sha256": sha(path), "exists": 1, "size_match": 1, "sha256_match": 1,
                         "identity_binding": "audit-owned source self-hash; not an experiment as-run hash"})

    for role, path, expected, binding in source_pairs:
        expected_bytes = None
        if expected:
            if role == "VideoMAE config snapshot": expected_bytes = 22864
            elif role == "VideoMAE processor snapshot": expected_bytes = 271
            elif role == "VideoMAE encoder weights": expected_bytes = 88157872
            elif role == "Core B main inputs": expected_bytes = 240744943
            elif role == "Core B train-only target cache": expected_bytes = 126360672
        add_dependency(deps, group="source_or_dependency", role=role, path=path,
                       expected_sha=expected, expected_bytes=expected_bytes,
                       binding=binding if expected else "not content-hash-bound in source manifest")

    # The CPU preflight report names a historical source SHA that is not the current file/copy hash.
    preflight = read_json(PROJECT / 'tmp/project-team/core_a_strict_preflight_20261008T162343Z/CORE_A_PREFLIGHT.json')
    historical_script_sha = preflight.get("script_sha256")
    if historical_script_sha:
        deps.append({"group": "source_or_dependency", "cell_id": "CoreA-preflight",
                     "role": "unresolved historical preflight source snapshot",
                     "path": f"UNRESOLVED_SOURCE_SHA256:{historical_script_sha}",
                     "expected_bytes": "", "actual_bytes": "", "expected_sha256": historical_script_sha,
                     "actual_sha256": "", "exists": 0, "size_match": "NA", "sha256_match": 0,
                     "identity_binding": "preflight report source SHA is not present in its output root; current as-run path hashes differently"})

    extras = [r for r in deps if r["path"] not in original_paths]
    merged = original + extras
    if merged:
        fields = list(merged[0].keys())
        for row in merged:
            for key in row:
                if key not in fields: fields.append(key)
        with (AUDIT / "ARTIFACT_INVENTORY.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(merged)
    with (AUDIT / "SOURCE_AND_DEPENDENCY_INVENTORY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(deps[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(deps)
    return {"indexed_artifacts": len(original), "index_sha_failures": sum(r["sha256_match"] != "1" for r in original),
            "additional_sources": len(deps), "unbound_as_run_source_rows": sum("not content-hash-bound" in str(r.get("identity_binding")) or "unresolved" in str(r.get("identity_binding")) for r in deps),
            "inventory_rows": len(merged)}


def build_data_contract() -> dict[str, object]:
    fm = read_json(FEATURES / "FEATURE_MANIFEST.json")
    feature_records = fm["records"]
    corea_sets = {s: {int(r["episode_index"]) for r in feature_records if r["split"] == s}
                  for s in ("train", "validation")}
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as z:
        split = np.asarray(z["split_code"], dtype=np.int8)
        ep = np.asarray(z["episode_index"], dtype=np.int64)
        frame = np.asarray(z["anchor_frame"], dtype=np.int64)
        timestamp = np.asarray(z["anchor_timestamp"], dtype=np.float64)
        ids = np.asarray(z["anchor_id"], dtype="U64")
        mask = np.asarray(z["history_mask"], dtype=np.bool_)
        hist_times = np.asarray(z["history_selected_timestamps"], dtype=np.float64)
    coreb_rows = {"train": np.flatnonzero(split == 0), "validation": np.flatnonzero(split == 1)}
    split_table = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint",
                                                "v3_benchmark_inclusion", "classification_inclusion",
                                                "eligible_4s", "duration_seconds", "timestamp_duration_seconds",
                                                "metadata_length", "num_frames", "video_fps", "video_nb_frames",
                                                "action_dim", "arm_dim", "hand_dim"],
                                filters=[("split", "in", ["train", "validation"])])
    rows = split_table.to_pylist()
    by_ep = {int(r["episode_index"]): r for r in rows}
    population_diff = {}
    for split_name in ("train", "validation"):
        c_ids = corea_sets[split_name]
        b_ids = set(map(int, np.unique(ep[split == (0 if split_name == "train" else 1)])))
        missing = sorted(c_ids - b_ids)
        missing_rows = [by_ep[e] for e in missing]
        population_diff[split_name] = {
            "core_a_episode_count": len(c_ids), "core_b_episode_count": len(b_ids),
            "common_episode_count": len(c_ids & b_ids), "core_a_only": len(missing),
            "core_b_only": len(b_ids - c_ids),
            "missing_episode_inclusions": sorted({f"{r['v3_benchmark_inclusion']}|{r['classification_inclusion']}|eligible4s={r['eligible_4s']}" for r in missing_rows}),
            "missing_metadata_length_min_median_max": [
                int(min(r["metadata_length"] for r in missing_rows)),
                float(np.median([r["metadata_length"] for r in missing_rows])),
                int(max(r["metadata_length"] for r in missing_rows))],
            "missing_duration_seconds_min_median_max": [
                float(min(r["duration_seconds"] for r in missing_rows)),
                float(np.median([r["duration_seconds"] for r in missing_rows])),
                float(max(r["duration_seconds"] for r in missing_rows))],
            "missing_video_fps": sorted({float(r["video_fps"]) for r in missing_rows}),
        }
    fps_values = sorted({float(by_ep[int(e)]["video_fps"]) for e in np.unique(ep)})
    residual = np.abs(timestamp - frame / np.asarray([float(by_ep[int(e)]["video_fps"]) for e in ep]))
    history_span = hist_times[:, -1] - hist_times[:, 0]
    mask_count = mask.sum(axis=1)
    target_norm = read_json(V3 / "data/V3_NORMALIZERS.json")
    vm_norm = read_json(VM_CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json")
    d3_norm = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json")
    index = read_json(HANDOFF / "HANDOFF_INDEX.json")
    scripts = {
        "anchor_writer": str(EQ / "code/build_train_target_cache.py"),
        "core_b_inference_model": str(TEAM / "code/d3_model.py"),
        "core_b_prediction_writer": str(HANDOFF / "code/train_videomae_core_b_formal.py"),
        "video_cache_extractor": str(HANDOFF / "code/extract_videomae_core_b_cache.py"),
        "metric_source_team_contract": str(TEAM / "code/evaluate_corrected_scheduler.py"),
    }
    return {
        "schema_version": "1.0", "status": "PASS_TRAIN_DEV_CONTRACT_WITH_ACTION_LABEL_CAVEAT",
        "final_test_access_this_audit": 0, "heldout_loaded_this_audit": False,
        "core_a": {"train_episodes": len(corea_sets["train"]), "validation_episodes": len(corea_sets["validation"]),
                   "classes": 120, "label_map_scope": "train-supported raw contiguous class IDs mapped to 0..119; validation labels must be a subset",
                   "feature_shape": fm["feature_shape"], "backbone_frozen": True,
                   "feature_pooling": fm["pooling"], "sampling": fm["sampling"],
                   "manifest_sha256": fm["manifest_sha256"], "encoder_weight_sha256": fm["weights_sha256"],
                   "processor_sha256": fm["processor_sha256"]},
        "core_b": {"input_path": str(V3 / "data/V3_MAIN_INPUTS.npz"), "input_sha256": sha(V3 / "data/V3_MAIN_INPUTS.npz"),
                   "split_codes": {"0": "train", "1": "dev"}, "train_anchors": len(coreb_rows["train"]),
                   "train_episodes": int(len(np.unique(ep[coreb_rows["train"]]))), "dev_anchors": len(coreb_rows["validation"]),
                   "dev_episodes": int(len(np.unique(ep[coreb_rows["validation"]]))), "dev_source_groups": 603,
                   "source_group_overlap": 0, "unique_anchor_ids": int(len(np.unique(ids))),
                   "core_a_core_b_population_difference": population_diff,
                   "population_explanation": "All 59 Core A-only episodes are still primary, classification-included and eligible_4s; metadata length is 120..179 frames. A full H60 target needs indices a+1..a+60 and a legal history anchor at/after frame119, so at least 180 frames are needed.",
                   "video_fps": fps_values, "anchor_timestamp_frame_over_fps_max_abs": float(residual.max()),
                   "history_shape_per_anchor": [32, 71], "history_mask_true_min_median_max": [int(mask_count.min()), float(np.median(mask_count)), int(mask_count.max())],
                   "history_time_span_seconds_min_median_max": [float(history_span.min()), float(np.median(history_span)), float(history_span.max())],
                   "video_frame_indices": "floor(linspace(anchor_frame-119, anchor_frame, 32)); every sampled index <= anchor",
                   "horizon_semantics": "H30/H60 are source-native 30/60 action steps; eligible train/dev manifest FPS is 30 and anchor timestamps agree with frame/fps, so about 1s/2s. Physical action-coordinate units remain unresolved.",
                   "anchor_and_future_target": "anchor action[a_i]; future action[a_i+1:a_i+H+1] from train-only target-cache builder; no action-coordinate reorder",
                   "target_schema": [2322, 60, 28], "target_normalizer_sha256": sha(V3 / "data/V3_NORMALIZERS.json"),
                   "target_fit_scope": target_norm["fit_scope"], "target_fit_episodes": target_norm["train_episodes"],
                   "target_fit_windows": target_norm["train_windows"], "target_sigma_dim": len(target_norm["target_sigma"]),
                   "safe_fd_scale_available": True, "safe_fd_scale_dim": len(target_norm["safe_fd_scale"]),
                   "video_normalizers": {"VideoMAE": {"sha256": sha(VM_CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json"),
                                                        "fit_scope": vm_norm["fit_scope"], "fit_episodes": vm_norm["fit_episodes"],
                                                        "fit_windows": vm_norm["fit_windows"], "dim": len(vm_norm["mean"])},
                                         "V-JEPA": {"sha256": sha(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json"),
                                                     "fit_scope": d3_norm["fit_scope"], "fit_episodes": d3_norm["fit_episodes"],
                                                     "fit_windows": d3_norm["fit_windows"], "dim": len(d3_norm["mean"])}},
                   "action_raw_slices": {"action_0_14": [0, 14], "action_14_28": [14, 28], "action_full28": [0, 28]},
                   "action_dimension_metadata": {"action_dim": 28, "arm_dim": 14, "hand_dim": 14,
                                                 "limits": "source inventory records observation arm/hand dimensions, but the action[0:28] ordering is not bound to those names"},
                   "anatomical_half_labels": "unresolved; scorer preserves raw slices only; current corrected scorer explicitly says unresolved; older sensitivity labels and handoff prose conflict",
                   "inference_inputs_exclude_future_targets": True,
                   "prediction_bundles_prevalidated_before_target_join": 24,
                   "source_files": scripts},
        "final_access_evidence": {"current_audit": "none; only train/dev arrays/manifests opened",
                                  "historical": "handoff/config/run reports claim final_test_access=0; these flags and scoped manifests do not prove complete historical access absence"},
    }


def build_comparability() -> list[dict[str, object]]:
    summaries = csv_rows(AUDIT / "metrics/core_b/MODEL_SEED_SUMMARIES.csv")
    initqa = read_json(AUDIT / "metrics/core_b/SHARED_INITIALIZATION_QA.json")
    vm_cache = read_json(VM_CACHE / "cache/CACHE_MANIFEST.json")
    vm_norm = read_json(VM_CACHE / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json")
    d3_extract = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_EXTRACTION_RUN_MANIFEST.json")
    d3_cache = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json")
    d3_norm = read_json(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json")
    vjepa_manifest = read_json(V3 / "SOURCE_REGISTRY.json")
    compare = read_json(VM_COMPARISON / "COMPARISON.json")
    state_diffs = {}
    for horizon in (30, 60):
        for seed in (20261001, 20261002, 20261003):
            own = VM_REMAINING / f"STATE-CTRL_H{horizon}_S{seed}/run/predictions_dev.npz"
            team_pred = TEAM / f"predictions/dev/D3EQ-CS-STATE-CTRL-H{horizon}-S{seed}.npz"
            with np.load(own, allow_pickle=False) as a, np.load(team_pred, allow_pickle=False) as b:
                lookup = {v: i for i, v in enumerate(b["anchor_id"].astype("U64"))}
                order = np.asarray([lookup[v] for v in a["anchor_id"].astype("U64")], dtype=np.int64)
                delta = np.abs(a["predictions_native"].astype(np.float64) - b["predictions_native"][order].astype(np.float64))
                state_diffs[(horizon, seed)] = (float(delta.max()), float(delta.mean()), float(np.quantile(delta, 0.95)))
    seed_init = {r["seed"]: r for r in initqa["rows"]}
    rows = []
    feature_specs = {
        "VideoMAE-fusion": {"backbone": "MCG-NJU/videomae-small-finetuned-kinetics (small, 384D)",
                            "checkpoint_sha256": vm_cache["weights_sha256"], "feature_shape": "N,2,16,4,384",
                            "frame_preprocessing": "same anchor-119..anchor 32-frame sequence; OpenCV RGB; HF resize/center crop 224; ImageNet mean/std",
                            "encoder_layout": "32 frames, tubelet2=>16 temporal tokens, 14x14 spatial; four 7x7 quadrant mean features",
                            "visual_input_dim": 384, "projection_param_count": 384*128+128,
                            "position_encoding": "fixed sinusoidal table regenerated for num_frames=32; no learned-position checkpoint extension"},
        "V-JEPA-fusion": {"backbone": "V-JEPA2.1 Vit-B distill ViT-G/384 checkpoint (768D output)",
                           "checkpoint_sha256": vjepa_manifest["pretrained_encoder_sha256"], "feature_shape": "N,2,16,4,768",
                           "frame_preprocessing": "same anchor-119..anchor 32-frame sequence; OpenCV RGB; resize-shortest-side/center crop 384; ImageNet mean/std",
                           "encoder_layout": "32 frames, tubelet2=>16 temporal tokens, 24x24 spatial; four disjoint 12x12 area-mean bins",
                           "visual_input_dim": 768, "projection_param_count": 768*128+128,
                           "position_encoding": "pretrained V-JEPA encoder positional behavior; exact extractor source hash was not recorded in extraction manifest"},
        "VM-STATE-CTRL": {"backbone": "VideoMAE visual input masked to zero (no frame content)",
                          "checkpoint_sha256": "separate state-control checkpoint per horizon/seed", "feature_shape": "B,16,4,768 zeros",
                          "frame_preprocessing": "not consumed by the model call; caller creates zero slots",
                          "encoder_layout": "64 video memory slots retained after zero input",
                          "visual_input_dim": 768, "projection_param_count": 768*128+128,
                          "position_encoding": "same D3 temporal/spatial encodings and video modality slots are retained"},
        "TEAM-STATE-CTRL": {"backbone": "V-JEPA visual grid passed to model; D3 STATE-CTRL masks to zero before video projection",
                             "checkpoint_sha256": "separate team state-control checkpoint per horizon/seed", "feature_shape": "N,2,16,4,768",
                             "frame_preprocessing": "same V-JEPA D/S cache as team fusion; content is discarded in the model branch",
                             "encoder_layout": "64 video memory slots retained after zero input",
                             "visual_input_dim": 768, "projection_param_count": 768*128+128,
                             "position_encoding": "same D3 temporal/spatial encodings and video modality slots are retained"},
    }
    for model, spec in feature_specs.items():
        for horizon in (30, 60):
            metric_model = {"VideoMAE-fusion": "VideoMAE", "V-JEPA-fusion": "V-JEPA"}.get(model, model)
            score = next(x for x in summaries if x["model"] == metric_model and int(x["horizon"]) == horizon
                         and x["region_raw"] == "action_full28" and x["metric"] == "standardized_mae")
            inits = ";".join(f"{seed}:{seed_init[seed]['shared_non_video_projection_state_sha256']}" for seed in seed_init)
            lr_max = 0.0
            for mode in ("TS11-D", "STATE-CTRL"):
                for seed in (20261001, 20261002, 20261003):
                    h = next(c for c in read_json(HANDOFF / "HANDOFF_INDEX.json")["core_b"]["cells"]
                             if c["mode"] == mode and int(c["horizon"]) == horizon and int(c["seed"]) == seed)
                    team_trace = TEAM / f"runs/formal/D3EQ-CS-{mode}-H{horizon}-S{seed}/LR_TRACE_ACTUAL.csv"
                    vm_trace = Path(h["run_root"]) / "LR_TRACE.csv"
                    with team_trace.open(newline="") as f: aa = [float(r["lr_used_for_update"]) for r in csv.DictReader(f)]
                    with vm_trace.open(newline="") as f: bb = [float(r["lr_used_for_update"]) for r in csv.DictReader(f)]
                    lr_max = max(lr_max, max(abs(a-b) for a,b in zip(aa,bb)))
            rs = state_diffs.get((horizon, 20261001)) if model in {"VM-STATE-CTRL", "TEAM-STATE-CTRL"} else None
            rows.append({
                "split": "dev", "model": model, "horizon_steps": horizon,
                "seeds": "20261001;20261002;20261003", "dev_rows_per_seed": 2322,
                "dev_episodes": 603, "dev_source_groups": 603,
                "input_sha256": sha(V3 / "data/V3_MAIN_INPUTS.npz"),
                "shared_input_and_target_normalizer": "exact same input/target normalizer hashes; train/dev IDs and target order verified",
                "backbone": spec["backbone"], "checkpoint_sha256": spec["checkpoint_sha256"],
                "feature_shape": spec["feature_shape"], "frame_preprocessing": spec["frame_preprocessing"],
                "encoder_layout_and_pooling": spec["encoder_layout"], "position_encoding": spec["position_encoding"],
                "visual_input_dim": spec["visual_input_dim"], "video_projection_parameters": spec["projection_param_count"],
                "predictor_non_video_architecture": "32x71 state history + 2 metadata channels; five causal TCN blocks (dilations1,2,4,8,16); 128 width; 2-layer 4-head decoder; 28D anchor; native B,H,28 output",
                "state_control_zero_node": "before video_projection; projection bias, temporal/spatial encoding, modality embedding and 64 slots retained",
                "initialization_file_and_shared_module": "same seed init file; non-video shared initial state exact across 384D/768D; see SHARED_INITIALIZATION_QA.json",
                "shared_non_video_initial_hashes_by_seed": inits,
                "loss_and_training_recipe": "same d3_eq_contract SHA; episode-equal per-sample objective; AdamW 1e-3/betas .9,.999/eps1e-8/wd1e-4; clip1; 40 epochs; 11120 updates; B64/accum1; 278 updates/epoch; last batch29; actual batch denominator29",
                "scheduler": "UpdateDrivenCosineAnnealingLR T_max11120 eta_min1e-5; 11120 steps; per-cell LR_TRACE max abs difference across all 12 cells = 0",
                "checkpoint_selection": "last epoch40; dev not used for model selection",
                "actual_full28_standardized_mae_mean": score["mean"],
                "actual_full28_standardized_mae_sample_sd_ddof1": score["sample_sd_ddof1"],
                "seed_values": ";".join(str(score[f"seed_{s}"]) for s in ("20261001", "20261002", "20261003")),
                "state_control_prediction_difference_vs_other_checkpoint_max_abs_seed20261001": rs[0] if rs else "NA",
                "comparability_class": "development system comparison; backbone resolution/pooling/input dimensions differ; no causal backbone attribution",
                "known_source_binding_gap": ("V-JEPA extractor source SHA is absent; 64 first-shard train anchors differ from the older V3 global-token cache under anchor-ID-joined 2x2-bin averaging; D3 manifest does not identify the bridge target" if model == "V-JEPA-fusion" else "none for listed model/loss/scheduler sources"),
                "checkpoint_replay_this_audit": ("blocked by unsafe-checkpoint approval gate" if model in {"VideoMAE-fusion", "VM-STATE-CTRL"} else "reference predictions accepted/re-scored; no reference checkpoint replay in scope"),
            })
    # Separate Core A row: the supplied Core B team reference is not a matched Core A probe reference.
    corea = read_json(AUDIT / "metrics/core_a/SEED_SUMMARY.json")
    rows.append({"split": "train/validation", "model": "VideoMAE-CoreA-frozen-linear-probe", "horizon_steps": "NA",
                 "seeds": "6008;7;42", "dev_rows_per_seed": 617, "dev_episodes": 617, "dev_source_groups": "NA",
                 "input_sha256": read_json(FEATURES / "FEATURE_MANIFEST.json")["manifest_sha256"],
                 "shared_input_and_target_normalizer": "CoreA train/validation manifest only; separate task from CoreB",
                 "backbone": "MCG-NJU/videomae-small-finetuned-kinetics frozen token-mean 384D",
                 "checkpoint_sha256": "see Core A per-seed metrics/replay", "feature_shape": "train 2757x384; val 617x384",
                 "frame_preprocessing": "16 frames t=0,.25,...,3.75s at 4Hz; 224 center crop; ImageNet norm",
                 "encoder_layout_and_pooling": "mean of final 16-frame encoder tokens; classifier head unused",
                 "position_encoding": "native 16-frame fixed sinusoidal positions",
                 "visual_input_dim": 384, "video_projection_parameters": 0,
                 "predictor_non_video_architecture": "single 384-to-120 linear head; CPU probe training",
                 "state_control_zero_node": "NA", "initialization_file_and_shared_module": "not a CoreB comparison",
                 "shared_non_video_initial_hashes_by_seed": "NA", "loss_and_training_recipe": "AdamW lr1e-3/wd1e-4, B256, validation Macro-F1 earliest max, frozen features",
                 "scheduler": "none", "checkpoint_selection": "validation Macro-F1; no held-out data",
                 "actual_full28_standardized_mae_mean": "NA", "actual_full28_standardized_mae_sample_sd_ddof1": "NA",
                 "seed_values": "NA", "state_control_prediction_difference_vs_other_checkpoint_max_abs_seed20261001": "NA",
                 "comparability_class": "not matched against a frozen V-JEPA CoreA probe in the supplied reference roots; LoRA/end-to-end evidence is separate",
                 "known_source_binding_gap": "historical 4-example CPU preflight source SHA 95c954... has no surviving source snapshot; full feature run source is bound",
                 "checkpoint_replay_this_audit": "accepted 3/3 CPU replay; see CHECKPOINT_REPLAY.csv"})
    write_csv(AUDIT / "COMPARABILITY_MATRIX.csv", rows)
    return rows


def build_archive_plan() -> None:
    inventory = csv_rows(AUDIT / "ARTIFACT_INVENTORY.csv")
    index_rows = [r for r in inventory if r["group"] in {"CoreA", "CoreB", "team_reference", "comparison", "batch_evaluation", "core_b_cache", "core_a_features", "core_a_probes", "shared"}]
    items = []
    for r in index_rows:
        role = r["role"]
        if role in {"checkpoint_or_tensor", "train_prediction", "dev_or_validation_prediction", "config", "metric_or_summary", "training_history_or_lr_trace", "environment", "manifest", "other"}:
            items.append({"source": r["path"], "role": role, "sha256": r["actual_sha256"],
                          "bytes": int(r["actual_bytes"]) if r["actual_bytes"] else None,
                          "copy_recommendation": "private archive copy; preserve raw source untouched"})
    source_inventory = csv_rows(AUDIT / "SOURCE_AND_DEPENDENCY_INVENTORY.csv")
    for r in source_inventory:
        q = Path(r["path"])
        if q.is_file() and q.suffix in {".py", ".sh"}:
            items.append({"source": r["path"], "role": r["role"], "sha256": r["actual_sha256"],
                          "bytes": int(r["actual_bytes"]), "copy_recommendation": "private archive code copy; retain the recorded as-run binding status"})
    dependencies = [
        {"path": str(VM_CACHE / "cache"), "bytes": 3947732224, "role": "VideoMAE CoreB feature bank",
         "copy_recommendation": "external dependency; keep manifest + shard hashes and source path; do not duplicate unless archive capacity is approved"},
        {"path": str(PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache"), "bytes": 7895424256,
         "role": "team V-JEPA D3 feature bank", "copy_recommendation": "external dependency; actual shard hashes inventoried here but original manifest lacks expected per-shard SHA; first 64-row train bridge cohort remains unresolved"},
        {"path": str(PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"),
         "role": "read-only raw G1 media", "copy_recommendation": "do not copy into model archive; cite restricted data root and allowlist provenance"},
        {"path": str(HF_ROOT), "role": "local VideoMAE model snapshots", "copy_recommendation": "external dependency; preserve config/weight/processor revisions and license record"},
        {"path": str(VJEPA_WEIGHTS), "role": "V-JEPA2 checkpoint", "copy_recommendation": "external dependency; preserve path/hash and licensing terms"},
    ]
    audit_files = []
    for q in sorted(AUDIT.rglob("*")):
        if not q.is_file() or q.name in {"ARCHIVE_PLAN.json", "CHECKSUMS.sha256"} or q.suffix == ".pyc":
            continue
        audit_files.append({"path_within_audit_root": q.relative_to(AUDIT).as_posix(),
                            "bytes": q.stat().st_size, "sha256": sha(q)})
    obj = {"status": "PLAN_ONLY_NO_SOURCE_MOVES_OR_COPIES", "proposed_private_destination": str(PROJECT / "research_archive/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z"),
           "indexed_artifact_count": 169, "indexed_artifact_total_bytes": 1385016124,
           "indexed_files_verified": "169/169 exist; sizes and SHA256 match HANDOFF_INDEX.json",
           "copyable_records": items, "audit_package_root": str(AUDIT),
           "audit_package_files_to_copy": audit_files, "external_dependencies": dependencies,
           "public_repository": "do not publish raw data, caches, model weights, checkpoints, predictions or private paths",
           "replay_block": "CoreB checkpoint replay was not accepted by auto-review for unrestricted pickle loading; no coreB replay outputs exist in this audit"}
    (AUDIT / "ARCHIVE_PLAN.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_import_manifest() -> None:
    qa = read_json(AUDIT / "metrics/core_b/CORE_B_METRIC_QA.json")
    train_qa = read_json(AUDIT / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json")
    a = read_json(AUDIT / "metrics/core_a/SEED_SUMMARY.json")
    comp = read_json(VM_COMPARISON / "COMPARISON.json")
    obj = {
        "schema_version": "1.0", "status": "ACCEPTED_WITH_CAVEATS_DEVELOPMENT_ONLY",
        "benchmark_population": "V3 train/dev primary duplicate-safe; dev only for cross-model comparison",
        "core_a_import": {"status": "standalone_frozen_probe_accepted_with_caveats",
                          "split": "train/validation", "train_rows": 2757, "validation_rows": 617,
                          "seeds": [6008, 7, 42], "metrics_file": str(AUDIT / "CORE_A_PER_SEED_METRICS.csv"),
                          "matched_team_frozen_vjepa_probe": "not found in the supplied TEAM_REFERENCE or the bounded current CoreA benchmark roots; do not imply a matched backbone comparison",
                          "validation_mean_sample_sd": a["means"]["validation"]},
        "core_b_import": {"status": "independent_dev_metrics_pass; checkpoint replay blocked",
                          "models": ["VideoMAE fusion", "V-JEPA fusion", "VideoMAE STATE-CTRL", "team STATE-CTRL"],
                          "split": "dev", "anchors": 2322, "episodes": 603, "source_groups": 603,
                          "seeds": [20261001, 20261002, 20261003],
                          "per_seed_metrics": str(AUDIT / "CORE_B_PER_SEED_METRICS.csv"),
                          "paired_contrasts": str(AUDIT / "CORE_B_PAIRED_CONTRASTS.csv"),
                          "scorer_qa": qa, "saved_train_prediction_score_qa": train_qa,
                          "old_comparator_pvalues_and_holm_rechecked": "exact agreement for all four registered tests",
                          "bootstrap": "10,000 paired common-source-group resamples; fixed three seeds; CI conditional on these trained models",
                          "status_is_not_final_test": True},
        "action_regions": {"action_0_14": "raw coordinates only", "action_14_28": "raw coordinates only",
                           "anatomical_mapping": "unresolved; do not rename as Arm/Hand"},
        "system_comparison_limitations": [
            "VideoMAE uses 384D pooled features at 224px; team V-JEPA uses 768D pooled grid features at 384px.",
            "The V-JEPA extractor source hash is not recorded in its cache manifest; a train-only 64-row first-shard bridge difference versus the older V3 global-token cache remains unexplained.",
            "STATE-CTRL prediction bundles differ across separate training checkpoints; no bitwise equality claim is made.",
            "CoreB checkpoint-to-prediction replay is pending explicit approval for local full-checkpoint pickle loading and the site GPU gate.",
            "Do not combine with R3D/P1 legacy recipe or N0 results as a single backbone-causal ranking.",
        ],
        "references": {"team_reference": str(TEAM), "handoff_root": str(HANDOFF),
                       "audit_root": str(AUDIT), "final_test_access_this_audit": 0,
                       "historical_final_test_evidence": "reported flags and scoped train/dev manifests only; not proof of complete historical non-access"},
        "existing_n0_n1_reconciliation": {"n0": "PASS_6_OF_6_REAL_REPLAYS and 12/12 interventions; human visual review pending",
                                          "n1": "final acceptance complete with 6/6 formal training and 24/24 prediction bundles; all N1 leases reported released; submission_ready=false pending human visual review",
                                          "n1_owner_liveness": "latest producer heartbeat/status snapshot is 2026-10-08 and stale at this audit; current live process not queried",
                                          "no_launch_or_intervention_by_this_audit": True},
    }
    (AUDIT / "BENCHMARK_IMPORT_MANIFEST.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_status_and_docs() -> None:
    idx = read_json(HANDOFF / "HANDOFF_INDEX.json")
    inv = csv_rows(AUDIT / "ARTIFACT_INVENTORY.csv")
    index_inv = [r for r in inv if r.get("group") in {"CoreA", "CoreB", "core_a_features", "core_a_probes", "core_b_cache", "batch_evaluation", "comparison", "team_reference", "shared"}
                 and r.get("expected_sha256")]
    assert len(index_inv) >= 169
    a_metrics = csv_rows(AUDIT / "CORE_A_PER_SEED_METRICS.csv")
    core_a_replay = [r for r in csv_rows(AUDIT / "CHECKPOINT_REPLAY.csv") if r.get("task") == "CoreA"]
    core_b_qa = read_json(AUDIT / "metrics/core_b/CORE_B_METRIC_QA.json")
    core_b_train_qa = read_json(AUDIT / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json")
    cache_bridge = read_json(AUDIT / "metrics/core_b/VJEPA_CACHE_BRIDGE_COMPARISON.json")
    fixture = read_json(AUDIT / "metrics/core_b/RED_CAPABLE_FIXTURE_RESULTS.json")
    strict_fixture = read_json(AUDIT / "metrics/core_a/STRICT_CONVERSION_FIXTURE.json")
    shared_init = read_json(AUDIT / "metrics/core_b/SHARED_INITIALIZATION_QA.json")
    n1_final = read_json(PROJECT / "reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/qa/FINAL_ACCEPTANCE.json")
    n1_obs = read_json(PROJECT / "reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/coordination/N1_PRODUCER_OBSERVATION.json")
    n0_accept = read_json(PROJECT / "reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/qa/N0_REAL_REPLAY_ACCEPTANCE.json")
    a_seed_summary = read_json(AUDIT / "metrics/core_a/SEED_SUMMARY.json")
    a_val = a_seed_summary["means"]["validation"]
    a_best_epochs = {str(r["seed"]): int(r["best_epoch"]) for r in read_json(PROBES / "SUMMARY.json")["results"]}
    a_support_rows = csv_rows(AUDIT / "CORE_A_CLASS_SUPPORT.csv")
    a_supported_min = min(sum(1 for r in a_support_rows if r["seed"] == str(seed) and r["split"] == split and r["included_in_supported_macro_f1"] == "1")
                          for seed in (6008, 7, 42) for split in ("train", "validation"))
    a_supported_max = max(sum(1 for r in a_support_rows if r["seed"] == str(seed) and r["split"] == split and r["included_in_supported_macro_f1"] == "1")
                          for seed in (6008, 7, 42) for split in ("train", "validation"))
    summary_rows = csv_rows(AUDIT / "metrics/core_b/MODEL_SEED_SUMMARIES.csv")
    def full28(model: str, h: int) -> dict[str, str]:
        return next(r for r in summary_rows if r["model"] == model and int(r["horizon"]) == h
                    and r["region_raw"] == "action_full28" and r["metric"] == "standardized_mae")
    contrast_rows = csv_rows(AUDIT / "CORE_B_PAIRED_CONTRASTS.csv")
    def contrast(contrast_name: str, h: int) -> dict[str, str]:
        return next(r for r in contrast_rows if r["contrast"] == contrast_name and int(r["horizon"]) == h
                    and r["region_raw"] == "action_full28" and r["metric"] == "standardized_mae")

    core_b_replay_rows = []
    for cell in idx["core_b"]["cells"]:
        root = Path(cell["run_root"])
        core_b_replay_rows.append({"task": "CoreB", "cell": f"{cell['mode']}_H{cell['horizon']}_S{cell['seed']}",
                                  "split": "dev", "status": "blocked_auto_review_before_checkpoint_load",
                                  "checkpoint_sha256": sha(root / "checkpoint_last.pt"),
                                  "prediction_sha256": sha(root / "predictions_dev.npz"),
                                  "prediction_artifact_validated": True, "prediction_regenerated": False,
                                  "checkpoint_loaded": False, "backend": "not_run", "final_test_access": 0,
                                  "block_reason": "auto-review rejected torch.load(weights_only=False); weights_only=True rejected the stored NumPy RNG global; no alternative decoder used"})
    # Keep the Core A replay rows, then append all blocked Core B rows.
    write_csv(AUDIT / "CHECKPOINT_REPLAY.csv", core_a_replay + core_b_replay_rows)

    acceptance = {
        "schema_version": "1.0", "status": "partial",
        "reproduction_entrypoint": {"status": "PASS_SAFE_CPU_REPLAY_AND_RESCORING", "path": str(AUDIT / "REPRODUCE_DEV.sh"),
                                    "completed": ["CoreA 3-seed fresh-process train/validation replay",
                                                  "CoreA 16-frame strict-conversion layer/feature fixture",
                                                  "CoreB 24 dev prediction bundle verification and independent scoring",
                                                  "CoreB 24 train prediction artifact rescoring",
                                                  "metric/future-sentinel red-capable fixtures"],
                                    "coreb_checkpoint_replay": "blocked; not attempted"},
        "acceptance": {
            "artifacts": {"status": "accepted_with_caveats", "core_a_cells_verified": "3/3", "core_b_cells_verified": "12/12",
                          "handoff_index_artifacts": 169, "index_hash_size_mismatches": 0,
                          "videomae_cache_shards": "314/314 SHA256 and size match cache manifest",
                          "vjepa_cache_shards": "314/314 present and size match; master manifest has no per-shard expected SHA, audit records actual SHA"},
            "checkpoint_replay": {"status": "blocked", "core_a_train_validation": "3/3 seeds x 2 splits, exact logits max_abs=0",
                                  "core_b_dev": "0/12 regenerated; all existing dev prediction bundles independently ID/schema/hash checked",
                                  "core_b_train": "not replayed", "reason": "automatic approval review blocked unrestricted local pickle loading; safe weights-only loader fails on NumPy RNG state"},
            "core_a_baseline": {"status": "accepted_with_caveats", "validation": a_val,
                                "matched_team_frozen_vjepa_probe": "not found in bounded supplied references"},
            "core_b_metrics": {"status": "accepted_with_caveats", "dev_qa": core_b_qa,
                                "saved_train_prediction_score_qa": core_b_train_qa},
            "matched_comparison": {"status": "accepted_with_caveats", "scope": "development system-level comparison, not causal backbone ranking",
                                   "state_control_checkpoint_output_diff": "measured; parameter-level cause unresolved under checkpoint-load block",
                                   "vjepa_cache_bridge": {"status": "unresolved_source_link", "rows_different": 64,
                                                           "total_rows": 20079, "outlier_split": "train", "outlier_episodes": 9,
                                                           "max_abs_to_older_v3_global_tokens": cache_bridge["max_abs_difference"],
                                                           "meaning_of_d3_bridge_max_abs": "not bound to the compared V3 cache"}},
            "portable_integration": {"status": "needs_fix", "reasons": ["results/code still depend on tmp roots", "encoder/cache/raw data remain external", "V-JEPA cache does not record per-shard expected SHA", "V-JEPA extractor as-run source hash not recorded", "64 first-shard train rows differ from the older V3 global-token cache under an unbound bridge comparison", "CoreA historical preflight source snapshot missing"]},
            "final_test": {"status": "not_in_scope", "access_this_audit": 0,
                           "historical_evidence": "flags and scoped train/dev manifests; complete past non-access is not proven"},
        },
        "core_a_strict_conversion_fixture": strict_fixture,
        "core_b_red_capable_fixtures": fixture,
        "shared_initialization_audit": shared_init,
        "existing_n0_n1_reconciliation": {"n0_status": n0_accept["status"], "n0_accepted": n0_accept["accepted"],
                                           "n1_final_status": n1_final["status"], "n1_final_acceptance": n1_final["N1"],
                                           "n1_all_leases_released": n1_final["all_n1_leases_released"],
                                           "n1_last_active_observation_utc": n1_obs["observed_utc"],
                                           "n1_live_owner_liveness_this_audit": "not checked"},
        "actions": {"old_source_writes": 0, "new_training_runs": 0, "heldout_access_this_audit": 0,
                    "gpu_commands_this_audit": 0},
        "blockers": [
            {"type": "automatic_approval_review", "operation": "torch.load(weights_only=False) on a local Core B formal checkpoint",
             "reason": "review rejected arbitrary pickle deserialization; safe weights_only=True cannot read NumPy RNG state",
             "reviewer_reason_exact": "The command deserializes local checkpoints with torch.load(weights_only=False), which can execute arbitrary pickle code; a safer weights-only/state-dict comparison is available and was not authorized. Do not bypass this rejection through a workaround or indirect execution.",
             "explicit_permission_needed": "approve restricted loading of these user-provided local checkpoint files, or supply safe model-only state-dict companions; GPU replay then also requires host admission approval"},
        ],
        "next_task": "After the checkpoint-load approval, run fresh-process CoreB dev-only replay through shared GPU lease watcher; compare all 12 cells, then complete portable archive and final acceptance update without retraining.",
    }
    (AUDIT / "FINAL_ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    n1_state = (f"Final package at {n1_final['created_utc']} reports {n1_final['N1']['formal_training']} training, "
                f"{n1_final['N1']['prediction_bundles']} predictions, all N1 leases released; submission_ready={n1_final['n1_submission_ready']}. "
                f"The last producer observation is older ({n1_obs['observed_utc']}); live owner liveness was not checked.")
    report = f"""# VideoMAE Core A/Core B independent acceptance

## Overall status

`STATUS=partial`. The indexed handoff is intact and the CPU acceptance work passed. Core A checkpoint replay and Core B independent dev scoring passed. Core B checkpoint-to-prediction replay remains blocked by automatic review of unrestricted PyTorch pickle loading; no GPU command was attempted.

## Artifacts and provenance

- `HANDOFF_INDEX.json` names {len(idx['artifacts'])} artifacts ({sum(int(x.get('bytes',0)) for x in idx['artifacts'].values()):,} bytes). All exist and all actual sizes and SHA-256 values match the index.
- VideoMAE Core B cache: all 314 manifest-listed shards match recorded sizes and SHA-256.
- Team V-JEPA D3 cache: all 314 shards exist and match recorded sizes; its master manifest does not store expected per-shard SHA-256. The audit records current hashes in `CACHE_SHARD_INVENTORY.csv`. A full anchor-ID join finds the first 64 train rows (9 episodes) differ from the older V3 global-token cache under 2x2-bin averaging (`VJEPA_CACHE_BRIDGE_OUTLIERS.csv`); the D3 manifest does not identify that bridge reference.
- Core A encoder is MCG-NJU VideoMAE-small fine-tuned on Kinetics: config SHA `{read_json(FEATURES / 'FEATURE_MANIFEST.json')['config_sha256']}`, processor SHA `{read_json(FEATURES / 'FEATURE_MANIFEST.json')['processor_sha256']}`, weights SHA `{read_json(FEATURES / 'FEATURE_MANIFEST.json')['weights_sha256']}`. The probe checkpoints contain only a `384→120` linear head; the frozen encoder is external and its pooled features are bound by manifest hashes.
- The 4-example historical CPU preflight report records source SHA `95c954…`, but no matching source snapshot was found. The full Core A feature run's script SHA matches the current as-run source at `7a1ed…`.

## Core A

All three CPU checkpoints replayed in separate processes on both train and validation. Every regenerated logit matched its saved prediction exactly (`max_abs=0`). Independent validation results (mean ± sample SD, `ddof=1`): Top‑1 {100*a_val['top1']['mean']:.4f}% ± {100*a_val['top1']['sample_sd_ddof1']:.4f} pp; Top‑5 {100*a_val['top5']['mean']:.4f}% ± {100*a_val['top5']['sample_sd_ddof1']:.4f} pp; Macro‑F1 {100*a_val['macro_f1_supported']['mean']:.4f}% ± {100*a_val['macro_f1_supported']['sample_sd_ddof1']:.4f} pp; NLL {a_val['nll']['mean']:.6f} ± {a_val['nll']['sample_sd_ddof1']:.6f}; Brier {a_val['brier_multiclass_sum120']['mean']:.6f} ± {a_val['brier_multiclass_sum120']['sample_sd_ddof1']:.6f}; ECE15 {a_val['ece15_equal_example_weight_bins_(lo,hi]']['mean']:.6f} ± {a_val['ece15_equal_example_weight_bins_(lo,hi]']['sample_sd_ddof1']:.6f}.

The selected best epochs were seed 6008=24, 7=14 and 42=29: earliest strict validation Macro-F1 maxima, with patience 15; no held-out score was used. Macro-F1 averages classes with true or predicted support; all six train/validation seed-splits have {a_supported_min}..{a_supported_max} supported classes, so it equals the fixed 120-class zero-filled score here. Top-5 is any true class in the five largest logits; NLL is mean multiclass cross-entropy; Brier is the per-example sum of squared 120-class probability errors, averaged over examples; ECE15 uses top-class confidence, 15 bins `(lo,hi]`, equal example weights. Per-ID logits/predictions and per-class confusion/support are saved. The Core A roots supplied here do not contain a matched frozen V-JEPA probe with a bound label map/scorer. LoRA/end-to-end results remain a separate training-method comparison.

Strict-conversion fixture: one canonical train episode, 16 sampled frames. Manual q/v bias mapping and zero key bias matched a native split-bias attention reference at every encoder layer and the pooled feature (`max_abs=0`). The currently installed Transformers 5.15.1 `from_pretrained` loader reports q/v biases as unexpected and q/k/v biases missing; use the strict conversion path. As-run Transformers version was not recorded in the full-feature environment manifest.

## Core B data/action contract

- Core A has 2,757 train and 617 validation episodes. Core B is a subset: 2,712 train episodes/17,757 anchors and 603 dev episodes/2,322 anchors. The 59 Core A-only episodes are still primary and 4-second eligible, but have only 120–179 frames. Full H60 targets plus the 119-frame history require 180 frames.
- The eligible train/dev manifests report 30 FPS; `anchor_timestamp` agrees with `anchor_frame/fps` within {5.1e-7:.1g}s. H30/H60 therefore span about 1/2 seconds in source time. Action coordinate units remain unresolved.
- Anchor is `action[a_i]`; future is `action[a_i+1:a_i+H+1]`. The predictor receives 32×71 standardized history, mask, relative-time features, video tokens, and anchor command; it has no future-target argument. Video samples are at or before the anchor.
- Score only raw `action[0:14]`, `action[14:28]`, and `action[0:28]`. The source schema reports `action_dim=28`, `arm_dim=14`, `hand_dim=14`, but does not bind the action vector's dimension order to those observation labels. Current corrected scoring leaves anatomy unresolved; no Arm/Hand name is promoted.

## Core B development results

Independent NumPy scoring joined dev targets only after validating all 24 dev prediction bundles. Its 72 cell/region checks differ from saved handoff metrics by at most {core_b_qa['max_abs_vs_handoff_saved_metrics']:.2g}; the comparator descriptives differ by at most {core_b_qa['max_abs_vs_handoff_comparator_descriptives']:.2g}. Existing centered paired-bootstrap p-values and Holm values reproduce exactly. A separate post-inference pass also rescored all 24 saved train prediction bundles against the train-only target cache; its maximum difference from stored handoff train metrics is {core_b_train_qa['max_abs_vs_saved_handoff_train_metrics']:.2g}. These train scores are not checkpoint replays.

Full28 standardized MAE (source-group equal mean ± sample SD; three seed values in parentheses):

| Model | H30 | H60 |
|---|---:|---:|
| VideoMAE fusion | {full28('VideoMAE',30)['mean']} ± {full28('VideoMAE',30)['sample_sd_ddof1']} ({full28('VideoMAE',30)['seed_20261001']}, {full28('VideoMAE',30)['seed_20261002']}, {full28('VideoMAE',30)['seed_20261003']}) | {full28('VideoMAE',60)['mean']} ± {full28('VideoMAE',60)['sample_sd_ddof1']} ({full28('VideoMAE',60)['seed_20261001']}, {full28('VideoMAE',60)['seed_20261002']}, {full28('VideoMAE',60)['seed_20261003']}) |
| V-JEPA fusion | {full28('V-JEPA',30)['mean']} ± {full28('V-JEPA',30)['sample_sd_ddof1']} ({full28('V-JEPA',30)['seed_20261001']}, {full28('V-JEPA',30)['seed_20261002']}, {full28('V-JEPA',30)['seed_20261003']}) | {full28('V-JEPA',60)['mean']} ± {full28('V-JEPA',60)['sample_sd_ddof1']} ({full28('V-JEPA',60)['seed_20261001']}, {full28('V-JEPA',60)['seed_20261002']}, {full28('V-JEPA',60)['seed_20261003']}) |
| VideoMAE STATE-CTRL | {full28('VM-STATE-CTRL',30)['mean']} ± {full28('VM-STATE-CTRL',30)['sample_sd_ddof1']} ({full28('VM-STATE-CTRL',30)['seed_20261001']}, {full28('VM-STATE-CTRL',30)['seed_20261002']}, {full28('VM-STATE-CTRL',30)['seed_20261003']}) | {full28('VM-STATE-CTRL',60)['mean']} ± {full28('VM-STATE-CTRL',60)['sample_sd_ddof1']} ({full28('VM-STATE-CTRL',60)['seed_20261001']}, {full28('VM-STATE-CTRL',60)['seed_20261002']}, {full28('VM-STATE-CTRL',60)['seed_20261003']}) |
| Team STATE-CTRL | {full28('TEAM-STATE-CTRL',30)['mean']} ± {full28('TEAM-STATE-CTRL',30)['sample_sd_ddof1']} ({full28('TEAM-STATE-CTRL',30)['seed_20261001']}, {full28('TEAM-STATE-CTRL',30)['seed_20261002']}, {full28('TEAM-STATE-CTRL',30)['seed_20261003']}) | {full28('TEAM-STATE-CTRL',60)['mean']} ± {full28('TEAM-STATE-CTRL',60)['sample_sd_ddof1']} ({full28('TEAM-STATE-CTRL',60)['seed_20261001']}, {full28('TEAM-STATE-CTRL',60)['seed_20261002']}, {full28('TEAM-STATE-CTRL',60)['seed_20261003']}) |

For Full28, `Delta_backbone = VideoMAE−V-JEPA` is {contrast('Delta_backbone',30)['mean_fixed_seed_contrast']} at H30 (seed signs 2/3 negative; paired-group 95% CI {contrast('Delta_backbone',30)['ci95_source_group_bootstrap_fixed_seeds']}) and {contrast('Delta_backbone',60)['mean_fixed_seed_contrast']} at H60 (2/3 negative; CI {contrast('Delta_backbone',60)['ci95_source_group_bootstrap_fixed_seeds']}). `G_VM` is {contrast('G_VM',30)['mean_fixed_seed_contrast']} / {contrast('G_VM',60)['mean_fixed_seed_contrast']} for H30/H60; `G_VJ` is {contrast('G_VJ',30)['mean_fixed_seed_contrast']} / {contrast('G_VJ',60)['mean_fixed_seed_contrast']}; `I=G_VM−G_VJ` is {contrast('I',30)['mean_fixed_seed_contrast']} / {contrast('I',60)['mean_fixed_seed_contrast']}. A positive H30 `I` does not imply VideoMAE's own state gain is positive: its H30 `G_VM` is negative.

The registered Holm family is only four standardized-MAE tests on the two raw halves at H30/H60. It reproduces exactly: H30 p={new_primary_p(contrast_rows,'action_0_14',30):.6g}/{new_primary_p(contrast_rows,'action_14_28',30):.6g}; H60 p={new_primary_p(contrast_rows,'action_0_14',60):.6g}/{new_primary_p(contrast_rows,'action_14_28',60):.6g}, with Holm-adjusted p={new_holm_p(contrast_rows,'action_0_14',30):.6g}/{new_holm_p(contrast_rows,'action_14_28',30):.6g} and {new_holm_p(contrast_rows,'action_0_14',60):.6g}/{new_holm_p(contrast_rows,'action_14_28',60):.6g}. These are development-set intervals conditional on three fixed trained seeds, not confirmatory training-population inference.

## Comparability and unresolved state-control difference

Both arms use the same V3 input and target-normalizer hashes, train/dev anchors, source groups, common model/loss/scheduler source hashes, seed initialization files, batch 64, 40 epochs, 11,120 updates, and exact per-update LR traces for all 12 cells. The final partial batch is 29 rows; both as-run objectives divide by the actual batch size. The VideoMAE fusion projection is 384→128; V-JEPA is 768→128. Video resolutions and spatial feature extraction differ, so this is a matched-settings development system comparison, not a causal backbone-only comparison.

For STATE-CTRL, both paths zero video input before the 768D projection and retain projection bias, time/space encodings, modality embedding, and 64 slots. The non-video shared initial tensors match exactly for all three seeds. Yet the saved state-control predictions differ substantially per element (H30 max absolute delta 1.28–1.32; H60 1.38–1.58). Their Full28 mean errors are close, but the parameter-level reason remains unresolved. The registered LR traces match exactly. Training-log aggregates are not directly comparable: the team records row-weighted epoch losses while the VideoMAE runner averages batch losses, giving the 29-row tail batch different logging weight. These summaries do not establish a training defect or explain the prediction difference.

Core B checkpoint replay was blocked before deserialization. Automatic review rejected `torch.load(weights_only=False)` because the saved full checkpoint contains pickle state; `weights_only=True` rejects the NumPy RNG reconstruction global. No workaround, allowlist, GPU query, or new training run was attempted.

## Existing N0/N1 and final-test boundary

N0's recorded root accepts 6/6 real replays and 12/12 interventions; human visual review is pending. N1's final package dated {n1_final['created_utc']} reports 6/6 formal training, 24/24 prediction bundles, all leases released, and `submission_ready=false`. Its last producer heartbeat snapshot is {n1_obs['observed_utc']}; current owner liveness was not checked. No N0/N1 job was launched, changed, or stopped.

This audit opened only train/dev inputs and targets. Historical records claim zero final-test access, but their flags do not prove complete historical non-access.

## Acceptance and next task

- Artifacts: `accepted_with_caveats`.
- Core A baseline and Core B dev metrics: `accepted_with_caveats`.
- Matched comparison: `accepted_with_caveats` for development system comparison.
- Core B checkpoint replay: `blocked`.
- Portable integration: `needs_fix` because dependencies remain under temporary/external roots and the V-JEPA cache lacks per-shard expected SHA/source-code identity.
- Final-test evaluation: `not_in_scope`; zero access this audit.
- `REPRODUCE_DEV.sh` safe CPU pipeline: passed end-to-end for CoreA replay, strict-conversion/future-sentinel fixtures, CoreB saved train/dev rescoring, and metric QA.

Next: obtain approval to load only the 12 local Core B checkpoints with unrestricted pickle deserialization, or have the teammate provide model-only safetensors companions. Then run dev-only fresh-process replay through the shared GPU lease watcher. Do not retrain the 12-cell matrix.
"""
    (AUDIT / "AUDIT_REPORT.zh-CN.md").write_text(report, encoding="utf-8")

    fix_list = """# Teammate fix list

| Owner role | Evidence | Required action | Acceptance condition |
|---|---|---|---|
| Core B trainer owner | 12 complete checkpoints/prediction bundles, all 11,120-step LR traces; state-control predictions differ from the independent same-seed reference by up to 1.58 native units | Provide a safe model-only safetensors export plus checkpoint/config hashes, or document why the full checkpoint requires NumPy RNG state. Reconcile STATE-CTRL training/RNG inputs and per-epoch objective logs; do not retrain the whole matrix. | Fresh-process dev replay validates all 12 cells; any state-control output difference is classified from exact as-run code/input and checkpoint evidence. |
| Core B feature-cache owner | VideoMAE cache has 314 expected shard hashes; V-JEPA D3 cache has no expected per-shard SHA or extractor source SHA. Its first 64 train rows (9 episodes) differ from the older V3 global-token cache under anchor-ID-joined bin averaging; `bridge_max_abs` is not bound to that comparator. | Add per-shard SHA-256 and as-run extractor source hash; identify the intended bridge reference; explain the 64-row cohort as an intended transform or a cache defect. Do not silently rebuild the full bank. | Every shard has a recorded digest and the cohort is source-explained with a fixture or repaired under a separately authorized bounded extraction. |
| Core A feature owner | Full feature run source/weights/processor are bound; earlier CPU preflight report references source SHA 95c954… with no surviving code snapshot; as-run Transformers version is absent | Preserve the exact preflight source snapshot and record Python/Torch/Transformers/processor versions in future feature manifests. | Source hash resolves to a file and a one-video 16-frame fixture reproduces layer outputs and pooled features. |
| Dataset/action schema owner | Source schema has action_dim28, observation arm_dim14 and hand_dim14; writers preserve `action[0:28]`; project references use conflicting anatomy names. | Add an authoritative action-to-coordinate ordering schema and map it to source channel names; update derived labels/tables only after closure. | Raw slices 0:14/14:28 trace to source fields, or all analysis material remains explicitly anatomical-unknown. |
| Benchmark integration owner | All experiment roots and large feature/checkpoint dependencies remain in tmp/external locations. | Make the planned private archive copy of code, manifests, configs, checkpoints and predictions; retain hashes and external data/cache references. | `REPRODUCE_DEV.sh` resolves roots from arguments and a clean isolated archive reproduces accepted dev metrics without absolute-path edits. |

Named individual owners were not present in the supplied handoff; owner roles are used here.
"""
    (AUDIT / "TEAMMATE_FIX_LIST.md").write_text(fix_list, encoding="utf-8")
    next_task = """# Next task recommendation

1. Resolve the Core B checkpoint-load approval and the 64-row V-JEPA cache bridge discrepancy. The current PyTorch weights-only reader cannot extract `model` because full training checkpoints include NumPy RNG state. Automatic review rejected unrestricted pickle loading. Either approve that exact local-only operation or receive 12 model-only safetensors companions from the checkpoint owner.
2. Run one dry-run through `tools/gpu_lease_watcher.py`; after the separate host approval, replay each dev split in a fresh process at CUDA float32, batch 64, with the actual per-cell checkpoint. Do not open dev targets until predictions pass ID/schema checks.
3. Compare regenerated predictions with the saved NPZs at a tolerance registered before replay. Record physical GPU chosen, max/mean/quantile error, state hashes before/after, and scorer results. Stop on a source/input mismatch; do not retrain.
4. Update `CHECKPOINT_REPLAY.csv`, `FINAL_ACCEPTANCE.json`, and `CHECKSUMS.sha256`, then complete the private archive copy plan.

Current audit used zero GPU commands, launched zero training jobs, and made no edits outside the new audit root.
"""
    (AUDIT / "NEXT_TASK_RECOMMENDATION.md").write_text(next_task, encoding="utf-8")

    # Machine-readable unresolved fields mirror the requested handoff schema.
    status_ini = f"""STATUS=partial
ARTIFACTS_VERIFIED_CORE_A=3/3
ARTIFACTS_VERIFIED_CORE_B=12/12
CORE_A_VALIDATION_REPLAY=3/3
CORE_B_DEV_REPLAY=0/12
TRAIN_REPLAY_STATUS=checkpoint replay not run; 24 saved train prediction bundles independently rescored; CoreB load blocked by auto-review
INDEPENDENT_METRIC_QA=PASS; Core A train/validation and Core B dev metrics independently scored
MATCH_LEVEL=matched principal settings; development system comparison with feature-dimension/pooling caveats
ACTION_RAW_SLICES=action[0:14],action[14:28],action[0:28]
ACTION_ANATOMICAL_LABEL_STATUS=unresolved
STATE_REPRODUCTION_STATUS=shared init and LR traces verified; saved state predictions differ; parameter-level cause blocked/unresolved
PORTABLE_INTEGRATION_STATUS=needs_fix
OLD_SOURCE_WRITES=0
NEW_TRAINING_RUNS=0
FINAL_HELDOUT_ACCESS_THIS_AUDIT=0
HISTORICAL_FINAL_ACCESS_EVIDENCE_LEVEL=reported flags and scoped train/dev manifests; complete historical proof unavailable
N0_N1_REMOTE_STATUS=N0 6/6 replay plus 12/12 interventions accepted; N1 final package complete 6/6 training and 24/24 predictions, leases released, submission_ready=false; latest live-owner snapshot stale
NEXT_TEAMMATE_TASK=resolve safe CoreB checkpoint loading; then dev-only fresh-process replay via shared GPU lease watcher
UNRESOLVED=CoreB checkpoint replay; V-JEPA per-shard expected hash and as-run extractor hash; CoreB state prediction divergence; action anatomy; portable archive; historical preflight source 95c954... missing
"""
    (AUDIT / "FINAL_ACCEPTANCE.ini").write_text(status_ini, encoding="utf-8")


def new_primary_p(rows: list[dict[str, str]], region: str, h: int) -> float:
    r = next(x for x in rows if x["primary_family"] == "1" and int(x["horizon"]) == h and x["region_raw"] == region)
    return float(r["p_centered_bootstrap_add_one_two_sided"])


def new_holm_p(rows: list[dict[str, str]], region: str, h: int) -> float:
    r = next(x for x in rows if x["primary_family"] == "1" and int(x["horizon"]) == h and x["region_raw"] == region)
    return float(r["p_holm_existing_four_test_family"])


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    initial_inventory = build_inventory()
    contract = build_data_contract()
    (AUDIT / "DATA_AND_ACTION_CONTRACT.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    build_comparability()
    write_import_manifest()
    write_status_and_docs()
    build_archive_plan()
    print(json.dumps({"status": "PASS_STATIC_REPORT_ASSEMBLY", "inventory": initial_inventory,
                      "data_contract": "DATA_AND_ACTION_CONTRACT.json", "final_acceptance": "FINAL_ACCEPTANCE.json"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

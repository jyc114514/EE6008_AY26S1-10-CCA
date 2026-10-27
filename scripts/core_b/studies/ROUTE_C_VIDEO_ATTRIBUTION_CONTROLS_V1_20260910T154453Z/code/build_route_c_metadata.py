#!/usr/bin/env python3
"""Create Route C source, checkpoint and donor-pairing manifests."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from apply_fusion_feature_interventions import cross_task_donors, pairing_summary, same_task_donors


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
B23 = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
FORMAL = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"
INPUT = B23 / "inputs/B2_INPUTS_0125.npz"
DATA_MANIFEST = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.csv"
FEATURE_NORM = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"
TARGET_NORM = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
BASE_CHECKPOINT = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
FRAME_MAP = B23 / "inputs/B2_FRAME_MAPS.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_inventory() -> list[dict[str, object]]:
    rows = json.loads((FORMAL / "CORE_B_LOGICAL_CELL_INVENTORY.json").read_text())["rows"]
    return [row for row in rows if row.get("stage") in {"B1", "B2", "B3"} and str(row.get("status", "")).startswith("accepted")]


def source_row(rows: list[dict[str, object]], cell: str) -> dict[str, object]:
    return next(row for row in rows if row["cell"] == cell)


def run_root_for(cell: str) -> Path:
    if cell == "B3-FUS-D-F": return B23 / "runs/B2-FUS-0125"
    if cell == "B3-V-D-F": return B23 / "runs/B2-V-0125"
    if cell == "B3-FUS-D-L": return B23 / "runs/corrected/B3-FUS-D-L"
    if cell == "B3-V-D-L": return B23 / "runs/corrected/B3-V-D-L"
    raise KeyError(cell)


def main() -> None:
    ROOT.joinpath("inputs").mkdir(parents=True, exist_ok=True)
    accepted = read_inventory()
    model_source = {"B3-FUS-D-F": "B2-FUS-0125", "B3-FUS-D-L": "B3-FUS-D-L", "B3-V-D-F": "B2-V-0125", "B3-V-D-L": "B3-V-D-L"}
    rows: list[dict[str, object]] = []
    base_hash = sha256(BASE_CHECKPOINT)
    feature_hash = sha256(FEATURE_NORM)
    target_hash = sha256(TARGET_NORM)
    input_hash = sha256(INPUT)
    manifest_hash = sha256(DATA_MANIFEST)
    frame_hash = sha256(FRAME_MAP)
    for model, source_cell in model_source.items():
        source = source_row(accepted, source_cell)
        rr = run_root_for(model)
        config_path = rr / "RUN_CONFIG_LOCKED.json"
        metrics_path = rr / "RUN_METRICS.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        rows.append({"model": model, "source_cell": source_cell, "branch": "fusion" if "FUS" in model else "video_only", "adaptation": "lora" if model.endswith("-L") else "frozen", "checkpoint_path": str(rr / "checkpoint_best.pt"), "checkpoint_sha256": sha256(rr / "checkpoint_best.pt"), "config_path": str(config_path), "config_sha256": sha256(config_path), "canonical_prediction_path": source["prediction_path"], "canonical_prediction_sha256": sha256(Path(source["prediction_path"])), "selected_epoch": metrics.get("best_epoch", 19), "base_checkpoint_path": str(BASE_CHECKPOINT), "base_checkpoint_sha256": base_hash, "feature_normalization_path": str(FEATURE_NORM), "feature_normalization_sha256": feature_hash, "target_normalization_path": str(TARGET_NORM), "target_normalization_sha256": target_hash, "input_archive": str(INPUT), "input_archive_sha256": input_hash, "data_manifest": str(DATA_MANIFEST), "data_manifest_sha256": manifest_hash, "frame_map": str(FRAME_MAP), "frame_map_sha256": frame_hash, "final_test_access": 0, "canonical_inference": "unmerged_base_plus_adapter" if model.endswith("-L") else "frozen_base_plus_head"})
    with (ROOT / "ROUTE_C_CHECKPOINT_REGISTRY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (ROOT / "ROUTE_C_CHECKPOINT_REGISTRY.json").write_text(json.dumps({"rows": rows, "count": len(rows), "final_test_access": 0}, indent=2) + "\n")

    with np.load(INPUT, allow_pickle=False) as archive:
        ids_all = np.asarray(archive["episode_index"], dtype=np.int64); split = np.asarray(archive["split_code"]); tasks_all = np.asarray(archive["original_task_index"], dtype=np.int64)
    dev = split == 1
    ids = np.sort(ids_all[dev]); task_lookup = {int(episode): int(task) for episode, task in zip(ids_all[dev], tasks_all[dev])}; tasks = np.asarray([task_lookup[int(episode)] for episode in ids], dtype=np.int64)
    same = same_task_donors(ids, tasks, seed=20260911); cross = cross_task_donors(ids, tasks, seed=20260912)
    pairing_rows = []
    for episode, task in zip(ids, tasks):
        pairing_rows.append({"recipient_episode": int(episode), "recipient_task": int(task), "same_task_donor": same.get(int(episode), ""), "cross_task_donor": cross.get(int(episode), ""), "same_task_available": int(episode) in same, "cross_task_available": int(episode) in cross, "same_task_seed": 20260911, "cross_task_seed": 20260912, "donor_future_target_used": False})
    with (ROOT / "ROUTE_C_PAIRING_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairing_rows[0])); writer.writeheader(); writer.writerows(pairing_rows)
    task_map = {int(episode): int(task) for episode, task in zip(ids, tasks)}
    (ROOT / "ROUTE_C_PAIRING_MANIFEST.json").write_text(json.dumps({"same_task": pairing_summary(same, task_map), "cross_task": pairing_summary(cross, task_map), "same_task_seed": 20260911, "cross_task_seed": 20260912, "final_test_access": 0}, indent=2) + "\n")

    # A fixed 16-episode smoke set: seeded, target/task-only, balanced across strata.
    v2_manifest = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_STRATA_MANIFEST.csv"
    strata_rows = list(csv.DictReader(v2_manifest.open(newline="")))
    rng = np.random.default_rng(20260913); chosen = []
    for name, count in (("low", 5), ("mid", 5), ("high", 6)):
        candidates = [row for row in strata_rows if row["motion_stratum"] == name]; candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        used = set()
        for row in candidates:
            if row["task_index"] not in used:
                chosen.append({"episode_id": int(row["episode_id"]), "task_index": int(row["task_index"]), "motion_stratum": name, "selection_seed": 20260913}); used.add(row["task_index"])
            if sum(item["motion_stratum"] == name for item in chosen) == count: break
    with (ROOT / "ROUTE_C_STAGE0_EPISODES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0])); writer.writeheader(); writer.writerows(chosen)
    (ROOT / "ROUTE_C_RESOURCE_TIMELINE.csv").write_text("timestamp_utc,phase,status,physical_gpu,uuid,utilization_percent,memory_used_mib,memory_total_mib,raw_video_io_token\n" + f"2026-09-10T15:44:53Z,preflight,waiting,,,,,,unacquired\n")
    print(json.dumps({"checkpoint_count": len(rows), "pairing_N": len(pairing_rows), "same_task_N": len(same), "cross_task_N": len(cross), "stage0_N": len(chosen), "status": "metadata_ready"}))


if __name__ == "__main__":
    main()

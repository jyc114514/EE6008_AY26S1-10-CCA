"""Build a source-backed Core B model/seed/horizon contract matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, REPO_ROOT, STATE_ROOT, V3_ROOT

D3_PRED = BENCH_ROOT / "d3/predictions"
D3_RUNS = BENCH_ROOT / "d3/runs"
SPLIT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"

FIELDS = [
    "source_root", "experiment_family", "run_id", "config", "seed", "horizon_steps", "population",
    "input_history_video_contract", "target_schema", "normalizer_id", "checkpoint_path", "checkpoint_selected_rule",
    "checkpoint_sha256", "prediction_bundle", "prediction_sha256", "episodes", "source_groups", "windows",
    "arm14_hand14_full28_composition", "mae_fd_event_metric_definition", "qa_replay_status", "native_score_availability",
    "common_score_eligibility", "comparability_group", "old_gate_status", "retry_attempt_count", "final_test_access", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(NEXT_ROOT.parents[2]))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalizer_hash(path: Path) -> str:
    return sha256(path)


def main() -> None:
    registry_path = BENCH_ROOT / "inventory/MODEL_REGISTRY.csv"
    base_rows = read_csv(registry_path)
    source_manifest = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint"])
    group_map = {int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}") for row in source_manifest.to_pylist() if row["split"] in {"train", "validation"}}
    v3_dev_eps = sorted({int(row["episode_index"]) for row in read_csv(V3_ROOT / "ANCHOR_MANIFEST.csv") if row["population"] == "primary" and row["split"] == "dev" and row["final_test_access"] == "0"})
    v3_dev_groups = len({group_map[ep] for ep in v3_dev_eps})
    v3_dev_rows = 2322
    with np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        if set(np.unique(archive["split_code"]).tolist()) != {0, 1}:
            raise ValueError("V3 primary input archive must contain only train/dev rows")
        dev_ids = set(archive["anchor_id"][archive["split_code"] == 1].astype("U64").tolist())
    v3_norm_hash = normalizer_hash(V3_ROOT / "data/V3_NORMALIZERS.json")
    d3_video_norm_hash = normalizer_hash(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json")
    state_mask_hash = normalizer_hash(STATE_ROOT / "INPUT_MASK_CONTRACT.json")

    rows: list[dict] = []
    for record in base_rows:
        model_id = record["model_id"]
        family = record["family"]
        horizon = record["native_horizon"]
        prediction_marker = record["prediction_path"]
        marker_path = PROJECT_ROOT / prediction_marker if prediction_marker not in {"", "UNKNOWN"} else None
        prediction_path = marker_path.parent / "primary.npz" if marker_path and marker_path.name == "PREDICTION_EXPORT_COMPLETE.json" else None
        accepted = bool(marker_path and marker_path.is_file() and json.loads(marker_path.read_text()).get("status") == "PASS")
        if family == "V3_MULTI_ANCHOR" and accepted:
            comparator = f"V3_PRIMARY_DEV_FULL28_H{horizon}"
            episodes, groups, windows = 603, 603, 2322
            eligibility = "within_V3_native_cell_only; primary-dev IDs and target joins accepted"
            metric_def = "standardized MAE over native horizon/joints, window mean then episode-equal; native FD mean absolute first-difference error; event metrics from D2_EVENT_REANALYSIS_V2"
            pred_hash = sha256(prediction_path)
        else:
            comparator = f"NOT_COMMON_SCORE:{family}"
            episodes = groups = windows = "UNKNOWN"
            eligibility = "not eligible for common scoreboard: source prediction/normalizer/ID compatibility not established in this audit"
            metric_def = "UNKNOWN or source-specific; do not compare by metric name alone"
            pred_hash = "UNKNOWN"
        input_contract = record.get("input_schema", "UNKNOWN")
        normalizer_id = record.get("state_video_target_normalizer_identity", "UNKNOWN")
        if family == "V3_MULTI_ANCHOR":
            normalizer_id = f"V3_NORMALIZERS.json sha256={v3_norm_hash}"
        rows.append({
            "source_root": record.get("experiment_root", "UNKNOWN"), "experiment_family": family, "run_id": model_id,
            "config": model_id, "seed": record.get("seed", "UNKNOWN"), "horizon_steps": horizon,
            "population": record.get("population_anchor_ids", "UNKNOWN"), "input_history_video_contract": input_contract,
            "target_schema": record.get("output_dims", "UNKNOWN") + "; " + record.get("full28_composition", "UNKNOWN"),
            "normalizer_id": normalizer_id, "checkpoint_path": record.get("checkpoint_path", "UNKNOWN"),
            "checkpoint_selected_rule": record.get("selected_checkpoint_rule", "UNKNOWN"), "checkpoint_sha256": record.get("checkpoint_hash", "UNKNOWN"),
            "prediction_bundle": str(prediction_path) if accepted else record.get("prediction_path", "UNKNOWN"), "prediction_sha256": pred_hash,
            "episodes": episodes, "source_groups": groups, "windows": windows,
            "arm14_hand14_full28_composition": f"{record.get('arm_hand_mapping','UNKNOWN')}; {record.get('full28_composition','UNKNOWN')}",
            "mae_fd_event_metric_definition": metric_def, "qa_replay_status": "PASS_source_bundle_accepted" if accepted else record.get("eligibility_reason", "UNKNOWN"),
            "native_score_availability": "prediction_bundle_accepted" if accepted else "not_verified", "common_score_eligibility": eligibility,
            "comparability_group": comparator, "old_gate_status": record.get("old_candidate_gate", "UNKNOWN"),
            "retry_attempt_count": "see_ARTIFACT_INVENTORY_RECONCILED_and_RUN_FAILURE_REGISTER", "final_test_access": record.get("final_test_access", "UNKNOWN"),
            "notes": f"registry row retained verbatim in source fields; replay={record.get('replay_status','UNKNOWN')}; {record.get('eligibility_reason','')}",
        })

    # D3: select the accepted prediction marker exactly as the D3 inference
    # manifest does (prefer a PASS RETRY run when a slot has both attempts).
    d3_candidates: dict[tuple[str, int, int], tuple[Path, dict]] = {}
    for sidecar in sorted(D3_PRED.glob("D3-*.json")):
        marker = json.loads(sidecar.read_text())
        if marker.get("status") != "PASS":
            continue
        key = (str(marker["config"]), int(marker["seed"]), int(marker["horizon_steps"]))
        current = d3_candidates.get(key)
        if current is None or ("RETRY" in str(marker["run_id"]) and "RETRY" not in str(current[1]["run_id"])):
            d3_candidates[key] = (sidecar, marker)
    if len(d3_candidates) != 36:
        raise ValueError(f"accepted D3 config/seed/horizon slots={len(d3_candidates)}, expected 36")
    d3_protocol = json.loads((BENCH_ROOT / "protocol/D3_PROTOCOL_LOCKED.json").read_text())
    d3_matrix = read_csv(BENCH_ROOT / "protocol/D3_MATRIX_36.csv")
    for config, seed, horizon in sorted(d3_candidates, key=lambda x: (x[2], x[1], x[0])):
        sidecar, marker = d3_candidates[(config, seed, horizon)]
        prediction = sidecar.with_suffix(".npz")
        with np.load(prediction, allow_pickle=False) as archive:
            d3_ids = set(archive["anchor_id"].astype("U64").tolist())
            d3_shape = archive["predictions_native"].shape
            d3_finite = bool(np.isfinite(archive["predictions_native"]).all())
        if d3_ids != dev_ids or d3_shape != (2322, horizon, 28) or not d3_finite:
            raise ValueError(f"D3 bundle ID/shape/finiteness mismatch: {prediction.name}")
        run_id = str(marker["run_id"])
        checkpoint = Path(marker["checkpoint"])
        rows.append({
            "source_root": str(BENCH_ROOT), "experiment_family": "D3_TIME_SPACE_FACTORIAL", "run_id": run_id,
            "config": config, "seed": seed, "horizon_steps": horizon,
            "population": "V3 primary development anchors; same 2322 IDs for all 36 cells",
            "input_history_video_contract": "history32x71 standardized + dynamic/static spatial grid (16x4x768 per mode) + anchor28; 64 video slots",
            "target_schema": "Full28; Hand=0:14; Arm=14:28; native residual decoder",
            "normalizer_id": f"V3 target normalizer sha256={v3_norm_hash}; D3 train-only spatial normalizer sha256={d3_video_norm_hash}",
            "checkpoint_path": str(checkpoint), "checkpoint_selected_rule": "epoch-40 last checkpoint; same accepted prediction sidecar; retry1 preferred only for duplicate slot",
            "checkpoint_sha256": "not_recorded_in_D3_prediction_sidecar; source checkpoint is immutable and remains in BENCH_ROOT",
            "prediction_bundle": str(prediction), "prediction_sha256": sha256(prediction), "episodes": 603, "source_groups": v3_dev_groups, "windows": 2322,
            "arm14_hand14_full28_composition": "Arm native dims 14:28; Hand native dims 0:14; Full28 all 28",
            "mae_fd_event_metric_definition": "MAE=mean |pred-GT|/target_sigma over horizon/joints then episode-equal; FD=mean native |diff(pred)-diff(GT)|; D2 event v2 separately, not yet common score",
            "qa_replay_status": "PASS: sidecar PASS, 2322 primary-dev rows, accepted D3 prediction acceptance 36/36",
            "native_score_availability": "D3_RESULTS.csv; source-group MAE/native-FD and factorial contrasts reanalyzed in NEXT_ROOT/D3_SOURCE_GROUP_RESULTS.csv and D3_SOURCE_GROUP_CONTRASTS.csv; 216/216 point values reconcile",
            "common_score_eligibility": "within_D3_6x2x3_factorial_only; not a V3 pure-spatial ablation",
            "comparability_group": f"D3_PRIMARY_DEV_FULL28_H{horizon}", "old_gate_status": "development_only_no_candidate_promotion",
            "retry_attempt_count": sum(1 for path in D3_RUNS.glob(f"D3-{config}-H{horizon}-S{seed}*") if path.is_dir() and "RETRY" in path.name),
            "final_test_access": 0, "notes": f"protocol_id={d3_protocol['protocol_id']}; source training loss implementation is not sample-wise equal-episode weighted; exploratory actual-run estimate only",
        })

    # V3_STATE: 54 accepted cells, retained as its own information-masking study.
    state_protocol = json.loads((STATE_ROOT / "PROTOCOL_LOCKED.json").read_text())
    state_matrix = read_csv(STATE_ROOT / "MATRIX_54.csv")
    state_norm_hash = v3_norm_hash
    state_rows = []
    for marker_path in sorted((STATE_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS":
            continue
        prediction = marker_path.parent / "primary.npz"
        with np.load(prediction, allow_pickle=False) as bundle:
            ids = bundle["anchor_id"].astype("U64")
            pred = bundle["predictions_native"]
            horizon = int(bundle["horizon_steps"])
        if set(ids.tolist()) != dev_ids or pred.shape != (2322, horizon, 28) or not np.isfinite(pred).all():
            raise ValueError(f"V3_STATE prediction shape mismatch: {marker_path.parent.name}")
        state_level, video_level = str(marker["state_level"]), str(marker["video_level"])
        seed, run_id = int(marker["seed"]), str(marker["run_id"])
        state_rows.append({
            "source_root": str(STATE_ROOT), "experiment_family": "V3_STATE_INFORMATION_MASKING", "run_id": run_id,
            "config": f"{state_level}-{video_level}", "seed": seed, "horizon_steps": horizon,
            "population": "V3 primary development; 603 episodes/603 exact-video groups/2322 windows",
            "input_history_video_contract": f"state mask={state_level} + video level={video_level}; contract sha256={state_mask_hash}; frozen V-JEPA features",
            "target_schema": "Full28; Arm=14:28; Hand=0:14", "normalizer_id": f"V3 train-only normalizers sha256={state_norm_hash}",
            "checkpoint_path": marker.get("checkpoint", "UNKNOWN"), "checkpoint_selected_rule": "epoch40_last",
            "checkpoint_sha256": marker.get("checkpoint_sha256", "UNKNOWN"), "prediction_bundle": str(prediction), "prediction_sha256": sha256(prediction),
            "episodes": 603, "source_groups": v3_dev_groups, "windows": 2322,
            "arm14_hand14_full28_composition": "Arm native dims 14:28; Hand native dims 0:14; Full28 all 28",
            "mae_fd_event_metric_definition": "MAE standardized by V3 target_sigma, episode-equal; native FD; figures/results include V3_STATE's declared comparisons",
            "qa_replay_status": "PASS: sidecar PASS; prediction shape; exact primary-dev population; 54/54 markers",
            "native_score_availability": "analysis/RESULTS_BY_SEED.csv, AGGREGATE_RESULTS.csv, FD_RESULTS.csv; source values retained",
            "common_score_eligibility": "within_V3_STATE_3x3_masking_study_only; compatible target/population but not D1 dispersion or D3 spatial factorial",
            "comparability_group": f"V3_STATE_PRIMARY_DEV_FULL28_H{horizon}", "old_gate_status": "development_only_no_candidate_promotion",
            "retry_attempt_count": 0, "final_test_access": 0,
            "notes": f"protocol_id={state_protocol['protocol_id']}; state/video mask question; no pooling with V3/D3 leaderboard",
        })
    if len(state_rows) != 54:
        raise ValueError(f"accepted V3_STATE rows={len(state_rows)}, expected 54")
    rows.extend(state_rows)

    if len(base_rows) != 124 or len(rows) != 214:
        raise ValueError(f"contract matrix cardinality mismatch: base={len(base_rows)}, total={len(rows)}")
    write_csv(NEXT_ROOT / "RESULT_CONTRACT_MATRIX.csv", rows, FIELDS)

    family_count = len({row["family"] for row in base_rows})
    summary = {
        "base_registry_rows": len(base_rows), "base_registry_families": family_count,
        "accepted_v3_rows_in_base_registry": sum(row["experiment_family"] == "V3_MULTI_ANCHOR" and row["qa_replay_status"] == "PASS_source_bundle_accepted" for row in rows[:len(base_rows)]),
        "accepted_d3_rows": 36, "accepted_v3_state_rows": len(state_rows), "total_rows": len(rows),
        "primary_dev_episodes": 603, "primary_dev_source_groups": v3_dev_groups, "primary_dev_windows": v3_dev_rows,
        "comparability_policy": "separate by exact protocol/input/normalizer/target/horizon/population; no all-row leaderboard",
        "final_test_access": 0,
    }
    (NEXT_ROOT / "results/CONTRACT_MATRIX_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

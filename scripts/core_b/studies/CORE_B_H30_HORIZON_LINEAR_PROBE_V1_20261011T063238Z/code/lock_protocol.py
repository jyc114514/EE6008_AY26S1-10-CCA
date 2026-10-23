#!/usr/bin/env python3
"""Lock the bounded H30 CPU Ridge/horizon analysis without reading targets."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
QUERY = PROJECT / "reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z"
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
MECH = PROJECT / "reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
TRAIN_TARGET_CACHE = EQ / "d3/train_targets_trainonly.npz"
MANIFEST = QUERY / "inventory/ANCHOR_MANIFEST.csv"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (1, 3, 6, 10, 15, 30)
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
EXPECTED = {
    str(INPUTS): "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78",
    str(V3 / "data/V3_NORMALIZERS.json"): "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c",
    str(TRAIN_TARGET_CACHE): "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite nonidentical locked artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise")
    w.writeheader()
    w.writerows(rows)
    content = buf.getvalue()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as existing:
            if existing.read() != content:
                raise FileExistsError(f"refusing to overwrite nonidentical locked artifact: {path}")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(content)


def load_prediction_records() -> list[dict[str, object]]:
    accept = json.loads((QUERY / "qa/PREDICTION_ACCEPTANCE.json").read_text(encoding="utf-8"))
    records = accept.get("bundle_records", [])
    if len(records) != 24:
        raise AssertionError(f"expected 24 accepted old prediction bundles; observed {len(records)}")
    out = []
    keys = set()
    for rec in records:
        pred = Path(rec["prediction_path"])
        sidecar = Path(rec["sidecar_path"])
        if not pred.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"accepted prediction source missing: {pred} / {sidecar}")
        observed = sha(pred)
        if observed != rec["prediction_sha256"]:
            raise AssertionError(f"accepted prediction hash drift: {pred}")
        side = json.loads(sidecar.read_text(encoding="utf-8"))
        if side.get("status") != "PASS" or side.get("run_id") not in (rec["run_id"], rec["source_run_id"]) or side.get("split") != rec["split"]:
            raise AssertionError(f"sidecar acceptance mismatch: {sidecar}")
        variant = str(rec["query_variant"])
        query = "Q0" if variant == "old_fixed_sinusoidal_time_query" else "Q1"
        key = (str(rec["split"]), query, str(rec["config"]), int(rec["seed"]))
        if key in keys:
            raise AssertionError(f"duplicate accepted prediction matrix key: {key}")
        keys.add(key)
        out.append({
            "split": rec["split"], "query": query, "system": rec["config"],
            "seed": int(rec["seed"]), "run_id": rec["run_id"],
            "prediction_path": str(pred), "prediction_sha256": observed,
            "sidecar_path": str(sidecar), "sidecar_sha256": sha(sidecar),
            "source": rec["source"], "source_run_id": rec["source_run_id"],
            "anchor_ids_sha256": rec["array_metadata_validation"]["ids_sha256"],
            "shape": rec["prediction_shape"], "target_arrays_present": rec["array_metadata_validation"]["target_arrays_present"],
        })
    if len(keys) != 24:
        raise AssertionError("accepted predictions do not cover exact 2x2x3x2 matrix")
    return out


def main() -> int:
    if ROOT.exists() and any((ROOT / p).exists() for p in ("protocol/PROTOCOL_LOCKED.json", "models/ridge_full_train.npz")):
        raise FileExistsError("refusing to overwrite an existing lock or model; resume must be explicit")
    for p, expected in EXPECTED.items():
        actual = sha(Path(p))
        if actual != expected:
            raise AssertionError(f"registered source hash mismatch: {p}: {actual}")

    rows = read_csv(MANIFEST)
    manifest = {r["anchor_id"]: r for r in rows}
    if len(manifest) != len(rows):
        raise AssertionError("duplicate anchor IDs in accepted anchor manifest")

    # Access only IDs, split metadata, history inputs, anchor inputs and past-history timestamps.
    with np.load(INPUTS, allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U64", copy=True)
        episodes = z["episode_index"].astype(np.int64, copy=True)
        split_code = z["split_code"].astype(np.int8, copy=True)
        anchor_frame = z["anchor_frame"].astype(np.int64, copy=True)
        anchor_ts = z["anchor_timestamp"].astype(np.float64, copy=True)
        history = z["history_native"].astype(np.float32, copy=True)
        hist_frames = z["history_selected_frame_indices"].astype(np.int64, copy=True)
        hist_ts = z["history_selected_timestamps"].astype(np.float64, copy=True)
        anchor = z["anchor_command_native"].astype(np.float32, copy=True)
    if history.shape != (20079, 32, 71) or anchor.shape != (20079, 28):
        raise AssertionError(f"canonical input shape mismatch: {history.shape}, {anchor.shape}")
    if len(np.unique(ids)) != len(ids) or set(ids.tolist()) != set(manifest):
        raise AssertionError("input IDs do not exactly match accepted primary manifest")
    if set(np.unique(split_code).tolist()) != {0, 1}:
        raise AssertionError(f"unexpected split codes: {np.unique(split_code)}")
    splits = np.where(split_code == 0, "train", "dev")
    if int(np.sum(splits == "train")) != 17757 or int(np.sum(splits == "dev")) != 2322:
        raise AssertionError("primary train/dev counts differ from accepted source contract")
    if not (np.isfinite(history).all() and np.isfinite(anchor).all() and np.isfinite(hist_ts).all() and np.isfinite(anchor_ts).all()):
        raise AssertionError("non-finite input metadata or covariates")
    if not np.all(hist_frames <= anchor_frame[:, None]) or not np.all(hist_ts <= anchor_ts[:, None] + 1e-6):
        raise AssertionError("history includes a time after its anchor")
    if not np.array_equal(hist_frames[:, -1], anchor_frame):
        raise AssertionError("the last history sample is not the anchor frame")
    if not np.array_equal(history[:, -1, 43:71], anchor):
        raise AssertionError("semantic duplicate check failed for anchor command")

    groups = np.empty(len(ids), dtype="U128")
    for i, anchor_id in enumerate(ids.tolist()):
        r = manifest[anchor_id]
        if r["split"] != splits[i] or int(r["episode_index"]) != int(episodes[i]):
            raise AssertionError(f"manifest/input identity mismatch: {anchor_id}")
        if r.get("population") not in (None, "", "primary") or int(r.get("final_test_access", "0")) != 0:
            raise AssertionError(f"non-primary or final row in accepted primary manifest: {anchor_id}")
        groups[i] = r["source_group_id"]
    if set(groups[splits == "train"].tolist()) & set(groups[splits == "dev"].tolist()):
        raise AssertionError("train/dev source-group overlap")
    group_episode = {}
    for g, ep in zip(groups.tolist(), episodes.tolist()):
        if g in group_episode and group_episode[g] != ep:
            raise AssertionError(f"one source group maps to multiple episode IDs: {g}")
        group_episode[g] = ep
    if len(set(groups[splits == "train"].tolist())) != 2712 or len(set(groups[splits == "dev"].tolist())) != 603:
        raise AssertionError("accepted source-group counts differ from the locked source contract")

    # Exact semantic map from the accepted input builder: arm 0:14, hand 14:28,
    # leg 28:43, action 43:71; last history action is the exact anchor command.
    feature_map = {
        "schema_version": 1,
        "source_field": "history_native",
        "source_builder": str(V3 / "code/prepare_v3_inputs.py"),
        "source_builder_sha256": sha(V3 / "code/prepare_v3_inputs.py"),
        "raw_construction": "concatenate(observation.arm_joints, observation.hand_joints, observation.leg_joints, action) in that order at each of 32 selected past samples",
        "per_time_feature_ranges": {"arm_joints": [0, 14], "hand_joints": [14, 28], "leg_joints": [28, 43], "action": [43, 71]},
        "flatten_order": "time-major, then the 71 source fields in the listed order",
        "history_samples": 32,
        "feature_width_per_sample": 71,
        "input_width_after_anchor_deduplication": 2272,
        "separate_anchor_command": "not appended: it is exactly the final history sample's action[0:28] at the anchor frame; the history copy is retained once",
        "semantic_duplicate_check": {"rows_checked": int(len(ids)), "max_abs": float(np.max(np.abs(history[:, -1, 43:71] - anchor))), "exact_array_equal": True},
        "excluded_inputs": ["video", "task ID", "episode ID", "source-group ID", "history mask", "relative-time channels", "future state/target", "extra/raw command time points"],
        "feature_map_sha256": "pending_self_hash",
    }
    feature_map["feature_map_sha256"] = hashlib.sha256(json.dumps({k:v for k,v in feature_map.items() if k != "feature_map_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(ROOT / "protocol/FEATURE_MAP.json", feature_map)

    # Deterministic source-group split: lexical ordering of SHA256(group|seed=20261011), rank mod 3.
    train_group_ids = sorted(set(groups[splits == "train"].tolist()))
    ranked = sorted(train_group_ids, key=lambda g: (hashlib.sha256(f"{g}|seed=20261011".encode()).hexdigest(), g))
    group_fold = {g: rank % 3 for rank, g in enumerate(ranked)}
    assignment_rows = []
    for i, anchor_id in enumerate(ids.tolist()):
        is_train = splits[i] == "train"
        assignment_rows.append({
            "anchor_id": anchor_id, "split": splits[i], "episode_index": int(episodes[i]),
            "source_group_id": groups[i], "fold": group_fold[groups[i]] if is_train else "NA",
            "final_test_access": 0,
        })
    write_csv(ROOT / "protocol/FOLD_ASSIGNMENT.csv", ["anchor_id", "split", "episode_index", "source_group_id", "fold", "final_test_access"], assignment_rows)
    fold_path = ROOT / "protocol/FOLD_ASSIGNMENT.csv"
    fold_summary = []
    for fold in range(3):
        hold_groups = {g for g, f in group_fold.items() if f == fold}
        fit_groups = set(train_group_ids) - hold_groups
        hold_rows = (splits == "train") & np.isin(groups, list(hold_groups))
        fit_rows = (splits == "train") & np.isin(groups, list(fit_groups))
        if np.intersect1d(np.unique(groups[fit_rows]), np.unique(groups[hold_rows])).size:
            raise AssertionError("source group crosses fit/holdout fold")
        fold_summary.append({"fold": fold, "fit_rows": int(fit_rows.sum()), "fit_groups": len(fit_groups), "fit_episodes": len(set(episodes[fit_rows].tolist())), "heldout_rows": int(hold_rows.sum()), "heldout_groups": len(hold_groups), "heldout_episodes": len(set(episodes[hold_rows].tolist()))})
    fold_hash = sha(fold_path)
    write_json(ROOT / "protocol/FOLD_SUMMARY.json", {"seed": 20261011, "algorithm": "sort unique primary-train source_group_id by SHA256(f'{group}|seed=20261011'), lexical digest then group ID; rank mod 3", "fold_assignment_sha256": fold_hash, "folds": fold_summary, "train_groups": len(train_group_ids), "dev_groups_not_folded": len(set(groups[splits == "dev"].tolist()))})

    accepted_preds = load_prediction_records()
    write_json(ROOT / "protocol/REFERENCE_MODELS.json", {
        "status": "LOCKED_ACCEPTED_12_MODEL_3_SEED_H30_REFERENCE",
        "matrix": "Q0/Q1 x TS11-D/STATE-CTRL x seeds 20261001, 20261002, 20261003 x train/dev",
        "accepted_prediction_acceptance_path": str(QUERY / "qa/PREDICTION_ACCEPTANCE.json"),
        "accepted_prediction_acceptance_sha256": sha(QUERY / "qa/PREDICTION_ACCEPTANCE.json"),
        "bundles": accepted_preds,
        "point_reuse_rule": "preserve each accepted seed as a separate model; never average predictions into an undeclared ensemble",
    })

    source_files = [
        INPUTS, V3 / "data/V3_NORMALIZERS.json", TRAIN_TARGET_CACHE,
        MANIFEST, QUERY / "inventory/SOURCE_ACCEPTANCE.json",
        QUERY / "qa/FINAL_ACCEPTANCE.json", QUERY / "qa/ANALYSIS_ACCEPTANCE.json",
        QUERY / "qa/PREDICTION_ACCEPTANCE.json", QUERY / "protocol/ANALYSIS_LOCKED.json",
        QUERY / "results/AGGREGATE_RESULTS.csv", QUERY / "results/QUERY_GAINS.csv",
        QUERY / "results/QUERY_MODEL_INTERACTION.csv", SCHED / "protocol/SOURCE_HASHES.json",
        SCHED / "MOTION_ACTIVITY_THRESHOLD.json", MECH / "SOURCE_REGISTRY.json",
        PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/inventory/MODEL_REGISTRY.csv",
        V3 / "code/prepare_v3_inputs.py", QUERY / "code/analyze_full_query.py", QUERY / "code/d3_eq_contract.py",
    ]
    for entry in accepted_preds:
        source_files.extend([Path(entry["prediction_path"]), Path(entry["sidecar_path"])])
    source_records = []
    for f in sorted(set(source_files), key=str):
        if not f.is_file():
            raise FileNotFoundError(f"required accepted source missing: {f}")
        source_records.append({"path": str(f), "sha256": sha(f), "bytes": f.stat().st_size, "write_policy": "read_only"})
    write_json(ROOT / "inventory/SOURCE_ACCEPTANCE.json", {
        "status": "PASS_BOUNDED_ACCEPTED_H30_INPUTS_AND_REFERENCES",
        "sources": source_records,
        "prior_old_root_noninterference": {"status": "PASS_ALL_REGISTERED_OLD_SOURCE_BYTES_MATCH_START_SNAPSHOT", "registered_sources": 317, "MECH": "178/178", "SCHED": "139/139 current bytes; prior accepted SCHED source list match 138/139 due one preexisting non-execution generate_figures.py mismatch", "evidence_path": str(QUERY / "coordination/OLD_ROOT_NONINTERFERENCE.json"), "old_root_writes_this_task": 0},
        "population": {"train_rows": int((splits == "train").sum()), "train_groups": len(set(groups[splits == "train"].tolist())), "dev_rows": int((splits == "dev").sum()), "dev_groups": len(set(groups[splits == "dev"].tolist())), "final_test_access": 0, "heldout_access": 0},
        "input_timeline": {"all_history_at_or_before_anchor": True, "last_history_frame_equals_anchor": True, "target_starts_anchor_plus_one": True},
    })

    # Exact reuse gate: only the registered baseline catalog and explicitly listed roots were read.
    registry = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/inventory/MODEL_REGISTRY.csv"
    reg_rows = read_csv(registry)
    catalog = [{"model_id": r.get("model_id", r.get("model", "")), "family": r.get("family", ""), "source_root": r.get("source_root", ""), "baseline_type": r.get("baseline_type", ""), "notes": r.get("notes", "")} for r in reg_rows]
    reuse = {
        "status": "NO_EXACT_ACCEPTED_MATCH_NEW_CPU_RIDGE_REQUIRED",
        "exact_contract": {"input": "history 32x71 plus semantic anchor command once", "target": "native Full28 H30 residual", "weights": "episode-equal", "validation": "primary-train source-group three-fold CV", "lambda_grid": list(LAMBDAS), "objective": "weighted squared residual sum divided by W plus lambda times Frobenius norm squared; unpenalized intercept"},
        "catalog_path": str(registry), "catalog_sha256": sha(registry), "registered_models": catalog,
        "nonmatches": ["PERSISTENCE has no fit", "FIXED_HYBRID is not a fitted residual Ridge", "COMMAND_SLOPE is a different historical baseline", "STATE_ONLY_BASELINE is not the specified full-history Ridge", "historical D1 neural median/command-slope/mixture CV are not this contract"],
        "new_fits_authorized": "up to 12 CV fits and exactly one full primary-train refit",
    }
    write_json(ROOT / "inventory/MODEL_REUSE_AUDIT.json", reuse)

    exp = json.loads((QUERY / "protocol/OUTCOME_EXPOSURE.json").read_text(encoding="utf-8"))
    # Capture old, already exposed H30 query results before new ridge fitting.
    gains = read_csv(QUERY / "results/QUERY_GAINS.csv")
    dev_gains = [r for r in gains if r.get("split") == "dev"]
    old_result = {
        "previous_dev_exposure": True,
        "independent_confirmation": False,
        "final_test_access": 0,
        "heldout_access": 0,
        "source_outcome_exposure": exp,
        "previous_H30_dev_query_gains": dev_gains,
        "selection_boundary": "these previously exposed query results are registered as prior knowledge; Ridge lambda is selected only by locked primary-train source-group OOF scores",
    }
    write_json(ROOT / "protocol/OUTCOME_EXPOSURE.json", old_result)

    query_close = """# H30 query 研究收口（既有 sealed 结果）

来源仅引用已验收的 `CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z`。不重跑 query、不修改旧根、不把该研究升级成 final 或独立确认。

- Query 线结论：存在有限异常 seed 救援，没有建立跨 seed 普遍收益或 TS11-D 融合绝对优势。旧 query 同时改 basis、frequency、smoothness、conditioning，rank 根因未确证。
- TS11-D 的 Q1−Q0 改善（旧误差减 Hadamard 误差，正值有利）：位置逐 seed `−0.008257, +0.014806, −0.003185`，均值 `+0.001121`、seed SD `0.012119`；safe-FD `−0.003783, +0.019887, −0.000077`，均值 `+0.005342`、seed SD `0.012732`。均值主要由 seed02 支持；seed01/03未改善。
- STATE-CTRL 位置三 seed 均负，均值 `−0.002803`。Q1 下 TS11-D 对 STATE-CTRL 的位置差 `−0.000463`，区间跨零；safe-FD 差 `−0.003510`，固定组区间全负，TS11-D 的 FD 仍劣于 STATE-CTRL。
- 旧四格 dev TV ratios 均为 `0.0929–0.1330`；under-dynamic 未解决。dev 已反复暴露，旧分析没有 p 值、最终测试或 held-out。
- 本轮只补一个 history/anchor CPU Ridge 及预锁 H30 时距剖面，仍是对已暴露 dev 的探索性对照。

"""
    (ROOT / "QUERY_LINE_CLOSED.md").write_text(query_close, encoding="utf-8")

    threshold_path = SCHED / "MOTION_ACTIVITY_THRESHOLD.json"
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    protocol = {
        "schema_version": 1,
        "status": "LOCKED_PRE_FIT_NO_DEV_TARGETS_READ",
        "run_id": ROOT.name,
        "locked_utc": utc_now(),
        "purpose": "one CPU full-H30 residual Ridge and fixed H30 horizon profile on previously accepted outputs",
        "input": {"source": str(INPUTS), "source_sha256": sha(INPUTS), "feature_map": "protocol/FEATURE_MAP.json", "feature_map_sha256": sha(ROOT / "protocol/FEATURE_MAP.json"), "width": 2272, "history_shape": [32, 71], "separate_anchor_appended": False, "deduplicate_by_semantic_identity": True},
        "target": {"source_train": str(TRAIN_TARGET_CACHE), "train_target_sha256": sha(TRAIN_TARGET_CACHE), "definition": "native targets_native[:,0:30,:] minus anchor_command_native; 30x28=840 outputs", "dev_target_read_gate": "only score_horizons.py after models/RIDGE_MODEL_LOCK.json PASS", "final_test_access": 0, "heldout_access": 0},
        "objective": {"equation": "J(B,b)=(1/W_f) sum_i w_i ||r_i-b-z_i B||_F^2 + lambda ||B||_F^2", "weight": "w_i=W_f/(E_f*n_e)", "intercept": "unpenalized weighted intercept", "dtype": "float64", "solver": "symmetric eigendecomposition of weighted centered Gram matrix, then regularized spectral solve; no explicit inverse", "outputs": 840, "max_features": 2300, "feature_count_after_semantic_deduplication": 2272},
        "preprocessing": {"input_center_scale": "weighted population mean and std fit only on each fold's fitting rows; constant columns mapped to zero", "training_targets": "native residual; no global target normalization in model fit", "cv_scoring_target_sigma": "recompute with canonical episode-equal mean/second-moment rule from each fold-fit H30 target rows only", "cv_scoring_safe_fd_scale": "episode-equal mean absolute future-internal difference on fold-fit H30 targets; max(fd_scale, 0.001*target_sigma); fold-fit only", "final_dev_metrics": "accepted primary-train target_sigma and safe_fd_scale; unchanged"},
        "folds": {"assignment_path": "protocol/FOLD_ASSIGNMENT.csv", "assignment_sha256": fold_hash, "seed": 20261011, "rule": "sort unique primary-train source_group_id by SHA256(f'{group}|seed=20261011'), lexical digest then group ID; rank mod 3", "group_isolation": True},
        "lambda_grid": {"values": list(LAMBDAS), "selection_metric": "episode-equal mean H30 Full28 standardized position MAE across all held-out primary-train episodes; fold-specific fit-only target_sigma", "tie_rule": "within absolute 1e-10, select larger lambda", "extension_allowed": False, "max_cv_fits": 12, "full_train_refits": 1},
        "horizon_analysis": {"steps": list(HORIZONS), "primary": 30, "models": "accepted Q0/Q1 x TS11-D/STATE-CTRL, seeds kept separate, new Ridge, Persistence", "position": "single endpoint H and prefix 1..H, standardized and native MAE", "future_internal_fd": "differences among future samples only, H-1 transitions; H=1 is NA; anchor-to-first seam separate", "persistence_skill": "1 - episode-equal mean model loss / episode-equal mean persistence loss; ratio after aggregation", "tv": "episode-equal TV_pred aggregate / TV_GT aggregate; report zero-TV support", "direction": {"active_threshold_source": str(threshold_path), "active_threshold_sha256": sha(threshold_path), "rule": threshold["direction_activity_threshold_rule"], "balanced_accuracy": "per-episode (positive recall + negative recall)/2 only when both GT signs occur; neutral prediction is incorrect"}, "times": "report actual target timestamp minus anchor; collect raw timestamp metadata only; no future commands used as Ridge inputs"},
        "development": {"scope": "previously exposed exploratory dev", "primary_contrast": "L_STATE-CTRL-Q0 - L_Ridge at H30 position MAE and safe-FD", "bootstrap": {"replicates": 10000, "seed": 20261011, "unit": "common source-group draw, preserving all windows/episodes", "scope": "conditional on fixed fits and repeatedly exposed development; no p-values"}},
        "prohibitions": ["no GPU access", "no new query", "no H60 training", "no neural-network seed/architecture/encoder", "no D5", "no final/heldout access", "no old-root writes", "no lambda extension", "no choosing a seed or case based on new Ridge scores"],
        "source_files_lock": "inventory/SOURCE_ACCEPTANCE.json",
        "baseline_reuse_audit": "inventory/MODEL_REUSE_AUDIT.json",
    }
    write_json(ROOT / "protocol/PROTOCOL_LOCKED.json", protocol)
    write_json(ROOT / "protocol/ANALYSIS_LOCKED.json", {"status": "LOCKED_PRE_FIT", "horizon_set": list(HORIZONS), "point_and_prefix_metrics": ["endpoint_position_standardized_mae", "prefix_position_standardized_mae", "endpoint_position_native_mae", "prefix_position_native_mae"], "dynamic_metrics": ["future_internal_fd_native_mae", "future_internal_fd_safe_normalized", "anchor_to_first_seam_native_error", "anchor_to_first_seam_safe_normalized", "tv_pred", "tv_gt", "tv_ratio", "direction_accuracy_active", "positive_recall", "negative_recall", "balanced_direction_accuracy"], "skill": "1 - episode-equal mean model loss / episode-equal mean Persistence loss; H=1 future-internal FD skills are NA; denominator zero => NA", "anchor_relative_position_error_is_duplicate": True, "query_line_status": "closed_no_new_query_training", "p_values": False, "final_test_access": 0, "heldout_access": 0})
    write_json(ROOT / "protocol/LAMBDA_GRID.json", {"values": list(LAMBDAS), "selected_global_lambda": None, "fit_budget": {"cv_fits": 12, "full_train_refits": 1, "max_total": 13}, "extension_allowed": False})
    write_json(ROOT / "protocol/HORIZON_METRICS.json", protocol["horizon_analysis"])
    write_json(ROOT / "protocol/CPU_PREFLIGHT.json", {"recorded_utc": utc_now(), "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "scikit_learn_used_for_solver": False, "threadpools_at_lock": threadpool_info(), "thread_cap": min(8, os.cpu_count() or 1), "gpu_api_invoked": False, "gpu_or_cuda_command_run": False, "cpu_only_execution_authorized": True, "free_space_fraction_at_preflight": 0.83, "available_memory_kib_at_preflight": 246548180, "effective_cpu_affinity_preflight": "0-31", "cost_status": "pre-run; actual CPU/RAM/disk will be recorded after execution"})
    write_json(ROOT / "coordination/OWNER_STATE.json", {"status": "LOCKED_READY_FOR_CPU_FIT", "owner_task_id": ROOT.name, "same_contract_root_preexisted": False, "prior_query_owner_active": False, "current_writer": str(Path(__file__).resolve()), "locked_utc": protocol["locked_utc"], "old_roots_read_only": True, "gpu_access": 0})
    (ROOT / "CONTEXT_RESTORED.md").write_text(
        "# Context restored\n\n"
        f"- New output root: `{ROOT}`. Created after exact-key existence check; no prior same-contract root existed.\n"
        f"- Accepted query source: `{QUERY}`; status `N1_FULL_STUDY_COMPLETE`, `24/24` bundles, H30 train/dev 17,757/2,322, dev 603 groups.\n"
        f"- Accepted full-data query analysis: no p-values, no independent confirmation, dev previously exposed, final/held-out access 0.\n"
        f"- Prior old-source noninterference: 317/317 registered bytes unchanged; MECH 178/178; SCHED 138/139 expected registry matches with one preexisting non-execution figure-script mismatch.\n"
        "- Canonical input archive and primary-train target cache hashes are pinned in `inventory/SOURCE_ACCEPTANCE.json`; no source values were copied into the old roots.\n"
        "- Exact accepted baseline catalog contains Persistence, fixed hybrid, command slope, and state-only baseline; no contract-equivalent full-history H30 residual Ridge. This run therefore fits one new CPU Ridge.\n"
        "- Input feature identity check confirms `history_native` last sample is at anchor and action[0:28] equals `anchor_command_native` exactly. The duplicate appended anchor is removed; all 32x71 history features remain.\n"
        "- GPU/CUDA APIs were not called. No GPU command, query, H60, neural training, D5, final, or heldout action is in scope.\n", encoding="utf-8")
    (ROOT / "RUN_STATUS.ini").write_text(
        "STATUS=locked\nQUERY_LINE_STATUS=closed_no_new_query_training\nBASELINE_ACTION=new_cpu_ridge\nNEW_CPU_CV_FITS=0\nNEW_CPU_FULL_TRAIN_REFIT=0\nLAMBDA_GRID_EXTENDED=false\nSELECTED_GLOBAL_LAMBDA=\nGROUP_FOLD_ISOLATION=PASS\nRidge_TRAIN_DEV_PREDICTIONS_ACCEPTED=false\nHORIZON_SET=1,3,6,10,15,30\nPRIMARY_HORIZON=30\nH30_REFERENCE_REPRODUCTION=pending\nDEV_SCOPE=repeatedly_exposed_exploratory\nINDEPENDENT_CONFIRMATION=false\nGPU_QUERY_USE_TRAINING=0\nNEW_NEURAL_NETWORK_TRAINING=0\nNEW_H60_TRAINING=0\nD5_MODEL_TRAINED=false\nFINAL_HELDOUT_ACCESS=0\nOTHER_WINDOW_TASKS_MODIFIED_OR_RELAUNCHED=false\nOLD_ROOT_WRITES=0\nHUMAN_REVIEW_STATUS=pending\nSUBMISSION_READY=false\n", encoding="utf-8")
    print(json.dumps({"status": "LOCKED_PRE_FIT_NO_DEV_TARGETS_READ", "root": str(ROOT), "train_rows": int((splits == "train").sum()), "dev_rows": int((splits == "dev").sum()), "train_groups": len(train_group_ids), "dev_groups": 603, "feature_width": 2272, "fold_hash": fold_hash, "reference_bundles": len(accepted_preds)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

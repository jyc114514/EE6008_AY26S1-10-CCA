#!/usr/bin/env python3
"""Score sealed train/dev predictions with V2 and V1 scorers; never runs inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT_REL = Path("repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz")
NORMALIZER_REL = INPUT_REL.parent / "V3_NORMALIZERS.json"
TARGET_REL = Path("reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz")
SPLIT_REL = Path("repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet")
TEAM_EVAL_REL = Path("reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/evaluate_corrected_scheduler.py")
REGION_NAMES = ("action_0_14", "action_14_28", "action_full28")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    temp = path.with_name("." + path.name + ".part")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"anchor_id", "episode_index", "predictions_native"}
        if not required.issubset(z.files):
            raise AssertionError(f"{path} missing prediction arrays")
        return {k: z[k].copy() for k in required}


def number_or_none(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def compare_scalar(old, new) -> tuple[object, object, object, object, str]:
    old, new = number_or_none(old), number_or_none(new)
    if old is None and new is None:
        return None, None, None, None, "both_NA"
    if old is None or new is None:
        return old, new, None, None, "NA_support_mismatch"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        delta = float(new) - float(old)
        abs_delta = abs(delta)
        relative = None if float(old) == 0.0 else delta / abs(float(old))
        return old, new, abs_delta, relative, "compared"
    return old, new, None, None, "metadata"


def field_semantics(field: str) -> tuple[str, str]:
    low = field.lower()
    if field in {"n_windows", "n_episodes", "n_source_groups"} or "count" in low or low.endswith("_windows") or low.endswith("_groups"):
        return "count/support", "exact"
    if "standardized" in low:
        return "standardized; train-only target_sigma", "window_to_episode_to_source_group_equal"
    if "native" in low or "anchor" in low or "endpoint" in low or "seam" in low:
        return "native action coordinate; physical units unresolved", "window_to_episode_to_source_group_equal"
    if "tv_ratio" in low or "ratio" in low:
        return "dimensionless ratio", "statistic-specific"
    if "direction" in low:
        return "fraction", "active-step support"
    if field.endswith("_mean") or field.endswith("_p25") or field.endswith("_p50") or field.endswith("_p75"):
        return "metric statistic", "statistic-specific"
    return "metric/count metadata", "statistic-specific"


def panel_rows(cell: str, split: str, region: str, model: str, panel: dict) -> dict[str, object]:
    return panel



def append_panel_diffs(rows: list[dict], *, cell: str, mode: str, horizon: int, seed: int,
                       split: str, region: str, model: str, comparison: str,
                       old_panel: dict, new_panel: dict, n_windows: int,
                       n_episodes: int, n_groups: int) -> None:
    for field in sorted(set(old_panel) | set(new_panel)):
        a0, b0 = old_panel.get(field), new_panel.get(field)
        a, b, absd, reld, state = compare_scalar(a0, b0)
        support = {
            "windows": n_windows, "episodes": n_episodes, "source_groups": n_groups,
            "direction_active_support": {"old": old_panel.get("active_step_count_total"),
                                         "new": new_panel.get("active_step_count_total")},
            "metric_support": {"old": old_panel.get("n_source_groups"),
                               "new": new_panel.get("n_source_groups")}
        }
        unit, aggregation = field_semantics(field)
        rows.append({
            "cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
            "split": split, "region": region, "model": model,
            "metric": field, "statistic": field, "unit_scale": unit, "aggregation": aggregation,
            "backend_comparison": comparison, "old": a, "new": b,
            "abs_delta": absd, "relative_delta": reld,
            "support": json.dumps(support, sort_keys=True), "status": state
        })

def score_panel(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray, sigma: np.ndarray,
                safe_fd: np.ndarray, threshold: np.ndarray, episodes: np.ndarray,
                groups: np.ndarray, team_eval, independent) -> tuple[dict[str, float], dict[str, object], dict[str, object]]:
    team_metrics = team_eval.window_metrics(pred, target, anchor, sigma, safe_fd, threshold)
    ind_metrics = independent.window_metrics(pred, target, anchor, sigma, safe_fd, threshold)
    if set(team_metrics) != set(ind_metrics):
        raise AssertionError(f"scorer metric key mismatch: team={sorted(team_metrics)} independent={sorted(ind_metrics)}")
    reducer = team_eval.GroupReducer(episodes, groups)
    selected = np.ones(len(pred), dtype=bool)
    panel = team_eval.summarize_window_metrics(team_metrics, reducer, selected)
    continuous_diffs = []
    discrete_exact = True
    na_positions_exact = True
    group_diffs = []
    for name in sorted(team_metrics):
        a, b = np.asarray(team_metrics[name]), np.asarray(ind_metrics[name])
        if a.shape != b.shape:
            raise AssertionError(f"scorer per-window shape mismatch for {name}")
        if np.issubdtype(a.dtype, np.integer) or np.issubdtype(b.dtype, np.integer):
            if not np.array_equal(a, b):
                discrete_exact = False
        else:
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                na_positions_exact = False
            finite = np.isfinite(a) & np.isfinite(b)
            if np.any(finite):
                continuous_diffs.append(float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))))
        ind_agg, ind_groups = independent.grouped_scores(b, episodes, groups)
        panel_key = {"tv_pred": "tv_pred_mean", "tv_gt": "tv_gt_mean",
                     "tv_ratio_window": "tv_ratio_group_mean_distribution_mean"}.get(name, name)
        if panel_key in panel and "source_group_equal" in ind_agg:
            team_value = panel[panel_key]
            if team_value is None:
                if math.isfinite(float(ind_agg["source_group_equal"])):
                    group_diffs.append(float("inf"))
            elif math.isfinite(float(ind_agg["source_group_equal"])):
                group_diffs.append(abs(float(team_value) - float(ind_agg["source_group_equal"])))
        if name == "tv_ratio_window" and ind_groups:
            vals = np.asarray(list(ind_groups.values()), dtype=np.float64)
            for q, suffix in ((0.25, "p25"), (0.50, "p50"), (0.75, "p75")):
                key = f"tv_ratio_group_{suffix}"
                if key in panel and abs(float(panel[key]) - float(np.quantile(vals, q))) > 1e-10:
                    group_diffs.append(abs(float(panel[key]) - float(np.quantile(vals, q))))
        if name == "tv_ratio_window":
            finite = b[np.isfinite(b)]
            for q, suffix in ((0.25, "p25"), (0.50, "p50"), (0.75, "p75")):
                key = f"tv_ratio_window_{suffix}"
                if key in panel and len(finite):
                    group_diffs.append(abs(float(panel[key]) - float(np.quantile(finite, q))))
    max_per_window = max(continuous_diffs, default=0.0)
    max_group = max(group_diffs, default=0.0)
    qa = {
        "continuous_per_window_max_abs": max_per_window,
        "continuous_group_summary_max_abs": max_group,
        "discrete_counts_exact": discrete_exact,
        "na_positions_exact": na_positions_exact,
        "metric_keys_exact": True,
        "continuous_gate_abs": 1e-10,
        "status": "PASS" if max_per_window <= 1e-10 and max_group <= 1e-10 and discrete_exact and na_positions_exact else "FAIL"
    }
    return panel, qa, {"team": team_metrics, "independent": ind_metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--v1-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    args = ap.parse_args()
    project, v2, v1, audit = [p.resolve() for p in (args.project_root, args.v2_root, args.v1_root, args.audit_root)]
    seal_path = audit / "PREDICTION_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("PREDICTION_SEAL.json is required before reading any targets")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "PREDICTIONS_SEALED_BEFORE_TARGET_JOIN" or seal.get("prediction_count") != 24 or seal.get("targets_loaded") is not False:
        raise RuntimeError("prediction seal is incomplete")
    for rec in seal["prediction_hashes"]:
        if digest(Path(rec["path"])) != rec["sha256"]:
            raise RuntimeError(f"sealed prediction changed before scoring: {rec['path']}")

    sys.dont_write_bytecode = True
    scorer_path = audit / "source/independent_core_b_metrics.py"
    fixtures_path = audit / "source/v1_red_capable_metric_fixtures.py"
    scorer_sha = digest(scorer_path)
    expected_scorer_sha = "64d69975cb4427622238e3976cbc1dcbf6697e6653f7dbf3177a2a765dddda0b"
    if scorer_sha != expected_scorer_sha:
        raise AssertionError("copied V1 independent scorer hash mismatch")
    spec = importlib.util.spec_from_file_location("audit_v1_independent_core_b_metrics", scorer_path)
    independent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = independent
    spec.loader.exec_module(independent)

    team_path = v2 / TEAM_EVAL_REL
    spec2 = importlib.util.spec_from_file_location("audit_team_core_b_evaluator", team_path)
    team_eval = importlib.util.module_from_spec(spec2)
    sys.modules[spec2.name] = team_eval
    spec2.loader.exec_module(team_eval)

    input_path, normalizer_path = v2 / INPUT_REL, v2 / NORMALIZER_REL
    target_path, split_path = v2 / TARGET_REL, v2 / SPLIT_REL
    normalizers = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if normalizers.get("status") != "LOCKED" or normalizers.get("fit_scope") != "primary train windows only; episode equal" or normalizers.get("final_test_access") != 0:
        raise AssertionError("target normalizer lock/scope mismatch")
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    safe_fd = np.asarray(normalizers["safe_fd_scale"], dtype=np.float32)
    if sigma.shape != (28,) or safe_fd.shape != (28,) or not np.isfinite(sigma).all() or not np.isfinite(safe_fd).all() or np.any(sigma <= 0) or np.any(safe_fd <= 0):
        raise AssertionError("target_sigma or safe_fd_scale is not an accepted finite train-only scale")
    groups_map = independent.episode_group_map(split_path)

    with np.load(input_path, allow_pickle=False) as source:
        # This scorer is after the prediction seal; target arrays are first opened here.
        split_codes = source["split_code"].astype(np.int8, copy=True)
        ids = source["anchor_id"].astype("U96", copy=True)
        episodes = source["episode_index"].astype(np.int64, copy=True)
        anchors_all = source["anchor_command_native"].astype(np.float32, copy=True)
        dev_targets_all = source["targets_native"].astype(np.float32, copy=True)
    rows_by_split = {"train": np.flatnonzero(split_codes == 0), "dev": np.flatnonzero(split_codes == 1)}
    with np.load(target_path, allow_pickle=False) as targets:
        train_ids = targets["anchor_id"].astype("U96", copy=True)
        train_eps = targets["episode_index"].astype(np.int64, copy=True)
        train_anchor = targets["anchor_command_native"].astype(np.float32, copy=True)
        train_targets60 = targets["targets_native_60"].astype(np.float32, copy=True)
    if (len(rows_by_split["train"]), len(np.unique(episodes[rows_by_split["train"]])) ) != (17757, 2712):
        raise AssertionError("train anchor/episode support mismatch")
    if (len(rows_by_split["dev"]), len(np.unique(episodes[rows_by_split["dev"]])) ) != (2322, 603):
        raise AssertionError("dev anchor/episode support mismatch")
    if not np.array_equal(train_ids, ids[rows_by_split["train"]]) or not np.array_equal(train_eps, episodes[rows_by_split["train"]]):
        raise AssertionError("train-only target cache does not align to canonical train IDs")
    if train_targets60.shape != (17757, 60, 28) or not np.isfinite(train_targets60).all():
        raise AssertionError("train-only target cache target schema mismatch")
    if dev_targets_all.shape != (20079, 60, 28) or not np.isfinite(dev_targets_all[rows_by_split["dev"]]).all():
        raise AssertionError("dev targets schema mismatch")
    threshold, fallback = independent.activity_threshold(train_targets60)
    team_threshold, team_fallback = team_eval.activity_threshold(train_targets60)
    if fallback != team_fallback or not np.array_equal(threshold, team_threshold):
        raise AssertionError("train-only activity threshold differs across scorer implementations")
    train_episodes = np.unique(episodes[rows_by_split["train"]])
    dev_episodes = np.unique(episodes[rows_by_split["dev"]])
    train_groups = {groups_map[int(ep)][1] for ep in train_episodes}
    dev_groups = {groups_map[int(ep)][1] for ep in dev_episodes}
    if any(groups_map[int(ep)][0] != "train" for ep in train_episodes) or any(groups_map[int(ep)][0] != "dev" for ep in dev_episodes):
        raise AssertionError("episode split mapping mismatch")
    if train_groups & dev_groups or (len(train_groups), len(dev_groups)) != (2712, 603):
        raise AssertionError("source group split support mismatch")
    groups_by_split = {
        "train": np.asarray([groups_map[int(ep)][1] for ep in episodes[rows_by_split["train"]]], dtype="U128"),
        "dev": np.asarray([groups_map[int(ep)][1] for ep in episodes[rows_by_split["dev"]]], dtype="U128")
    }
    targets_by_split = {
        "train": train_targets60,
        "dev": dev_targets_all[rows_by_split["dev"]]
    }
    anchors_by_split = {"train": train_anchor, "dev": anchors_all[rows_by_split["dev"]]}

    expected_rows = list(csv.DictReader((audit / "EXPECTED_CELLS.csv").open(newline="", encoding="utf-8")))
    archive = json.loads((v2 / "ARCHIVE.json").read_text(encoding="utf-8"))
    archive_cells = {c["name"]: c for c in archive["cells"]}
    cpu_root = v2 / "replay/cpu_zero_20261013T065954Z"
    independent_qa_rows = []
    metric_diff_rows = []
    full_panels = []
    score_errors = []

    for expected in expected_rows:
        cell, mode, horizon, seed = expected["cell"], expected["mode"], int(expected["horizon"]), int(expected["seed"])
        archive_cell = archive_cells[cell]
        old_run = v2 / archive_cell["run"]
        old_metrics_path = v2 / archive_cell["metrics"]
        old_saved_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        model_label = "VideoMAE" if mode == "TS11-D" else "STATE-CTRL"
        new_panels = {"train": {}, "dev": {}}
        rescored_old_panels = {"train": {}, "dev": {}}
        cpu_panels = {"train": {}, "dev": {}}
        for split in ("train", "dev"):
            n = len(rows_by_split[split])
            canonical_rows = rows_by_split[split]
            canonical_ids = ids[canonical_rows]
            canonical_eps = episodes[canonical_rows]
            canonical_groups = groups_by_split[split]
            region_outputs = {}
            old_region_outputs = {}
            cpu_region_outputs = {}
            new_path = audit / "replay/gpu" / cell / f"predictions_{split}.npz"
            old_path = old_run / f"predictions_{split}.npz"
            cpu_path = cpu_root / cell / f"predictions_{split}.npz"
            new_bundle, old_bundle, cpu_bundle = [load_bundle(p) for p in (new_path, old_path, cpu_path)]
            for label, bundle in (("new", new_bundle), ("old", old_bundle), ("cpu", cpu_bundle)):
                if not np.array_equal(bundle["anchor_id"].astype("U96"), canonical_ids) or not np.array_equal(bundle["episode_index"].astype(np.int64), canonical_eps):
                    raise AssertionError(f"{label} prediction IDs/order mismatch: {cell}/{split}")
                if bundle["predictions_native"].shape != (n, horizon, 28) or bundle["predictions_native"].dtype != np.float32 or not np.isfinite(bundle["predictions_native"]).all():
                    raise AssertionError(f"{label} prediction schema/finite mismatch: {cell}/{split}")
            target = targets_by_split[split][:, :horizon, :]
            anchor_values = anchors_by_split[split]
            persistence = np.repeat(anchor_values[:, None, :], horizon, axis=1).astype(np.float32)
            saved_cpu_metrics = json.loads((cpu_root / cell / "evaluation/METRICS.json").read_text(encoding="utf-8"))["panels"][split]
            for region in REGION_NAMES:
                sl = independent.REGIONS[region]
                region_outputs[region] = {}
                old_region_outputs[region] = {}
                cpu_region_outputs[region] = {}
                for model_name, new_pred, old_pred in (
                    (model_label, new_bundle["predictions_native"], old_bundle["predictions_native"]),
                    ("PERSISTENCE", persistence, persistence)
                ):
                    new_panel, qa, _ = score_panel(
                        new_pred[:, :, sl], target[:, :, sl], anchor_values[:, sl],
                        sigma[sl], safe_fd[sl], threshold[sl], canonical_eps, canonical_groups,
                        team_eval, independent)
                    old_panel, old_qa, _ = score_panel(
                        old_pred[:, :, sl], target[:, :, sl], anchor_values[:, sl],
                        sigma[sl], safe_fd[sl], threshold[sl], canonical_eps, canonical_groups,
                        team_eval, independent)
                    cpu_panel = saved_cpu_metrics[region][model_name]
                    region_outputs[region][model_name] = new_panel
                    old_region_outputs[region][model_name] = old_panel
                    cpu_region_outputs[region][model_name] = cpu_panel
                    qa_record = {
                        "cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
                        "split": split, "region": region, "model": model_name,
                        "continuous_per_window_max_abs": qa["continuous_per_window_max_abs"],
                        "continuous_group_summary_max_abs": qa["continuous_group_summary_max_abs"],
                        "discrete_counts_exact": qa["discrete_counts_exact"], "na_positions_exact": qa["na_positions_exact"],
                        "status": qa["status"], "scorer_sha256": scorer_sha,
                        "team_evaluator_sha256": digest(team_path), "targets_joined_after_prediction_seal": True,
                        "final_test_access": 0
                    }
                    independent_qa_rows.append(qa_record)
                    if qa["status"] != "PASS" or old_qa["status"] != "PASS":
                        score_errors.append(f"scorer disagreement: {cell}/{split}/{region}/{model_name}")
                    saved_old_panel = old_saved_metrics["panels"][split][region][model_name]
                    if set(saved_old_panel) != set(old_panel):
                        score_errors.append(f"saved old panel fields differ from locked evaluator: {cell}/{split}/{region}/{model_name}")
                    for field in sorted(set(saved_old_panel) & set(old_panel)):
                        a, b, absd, reld, state = compare_scalar(saved_old_panel[field], old_panel[field])
                        if state not in {"compared", "both_NA"} or (absd is not None and absd > 1e-10):
                            score_errors.append(f"old stored metric not reproduced: {cell}/{split}/{region}/{model_name}/{field}")
                    supports = {
                        "n_windows": n, "n_episodes": int(len(np.unique(canonical_eps))),
                        "n_source_groups": int(len(np.unique(canonical_groups)))
                    }
                    append_panel_diffs(metric_diff_rows, cell=cell, mode=mode, horizon=horizon, seed=seed,
                                       split=split, region=region, model=model_name,
                                       comparison="stored_historical_GPU_vs_rescored_historical_GPU",
                                       old_panel=saved_old_panel, new_panel=old_panel,
                                       n_windows=n, n_episodes=supports["n_episodes"], n_groups=supports["n_source_groups"])
                    append_panel_diffs(metric_diff_rows, cell=cell, mode=mode, horizon=horizon, seed=seed,
                                       split=split, region=region, model=model_name,
                                       comparison="historical_GPU_vs_new_GPU",
                                       old_panel=saved_old_panel, new_panel=new_panel,
                                       n_windows=n, n_episodes=supports["n_episodes"], n_groups=supports["n_source_groups"])
                    append_panel_diffs(metric_diff_rows, cell=cell, mode=mode, horizon=horizon, seed=seed,
                                       split=split, region=region, model=model_name,
                                       comparison="historical_GPU_vs_historical_CPU",
                                       old_panel=saved_old_panel, new_panel=cpu_panel,
                                       n_windows=n, n_episodes=supports["n_episodes"], n_groups=supports["n_source_groups"])
            new_panels[split] = region_outputs
            rescored_old_panels[split] = old_region_outputs
            cpu_panels[split] = cpu_region_outputs
        metric_report = {
            "status": "PASS_POST_SEAL_SCORING" if not score_errors else "SCORING_WITH_FINDINGS",
            "cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
            "panels": new_panels, "aggregation": "window -> episode -> source_group equal",
            "scorer_sha256": scorer_sha, "team_evaluator_sha256": digest(team_path),
            "input_sha256": digest(input_path), "target_normalizer_sha256": digest(normalizer_path),
            "train_target_cache_sha256": digest(v2 / TARGET_REL), "safe_fd_scale_source": "V3_NORMALIZERS.json; primary train only",
            "activity_threshold_fit_scope": "primary train targets only; joint-wise q05 positive H60 increments, floor 1e-6",
            "train_rows": 17757, "train_episodes": 2712, "train_source_groups": 2712,
            "dev_rows": 2322, "dev_episodes": 603, "dev_source_groups": 603,
            "source_group_overlap": 0, "targets_joined_after_prediction_seal": True,
            "paired_bootstrap_rerun": False, "final_test_access": 0
        }
        write_json(audit / "replay/gpu" / cell / "evaluation/METRICS.json", metric_report)

    diff_fields = ["cell", "mode", "horizon", "seed", "split", "region", "model", "metric", "statistic", "unit_scale",
                   "aggregation", "backend_comparison", "old", "new", "abs_delta", "relative_delta", "support", "status"]
    csv_write(audit / "METRIC_REPLAY_DIFFS.csv", metric_diff_rows, diff_fields)
    csv_write(audit / "INDEPENDENT_SCORER_QA.csv", independent_qa_rows)
    max_independent = max((max(float(r["continuous_per_window_max_abs"]), float(r["continuous_group_summary_max_abs"]))
                           for r in independent_qa_rows), default=0.0)
    independent_status = "PASS" if independent_qa_rows and all(r["status"] == "PASS" for r in independent_qa_rows) and max_independent <= 1e-10 else "FAIL"
    write_json(audit / "INDEPENDENT_SCORER_QA.json", {
        "status": independent_status, "scorer": "copied V1 independent_core_b_metrics.py",
        "scorer_sha256": scorer_sha, "team_evaluator_sha256": digest(team_path),
        "cell_split_region_comparisons": len(independent_qa_rows),
        "continuous_metric_gate_abs": 1e-10,
        "maximum_continuous_difference": max_independent,
        "discrete_counts_support_keys_and_NA_positions_exact": all(
            r["discrete_counts_exact"] and r["na_positions_exact"] for r in independent_qa_rows),
        "fixtures_source": str(fixtures_path), "fixtures_sha256": digest(fixtures_path),
        "new_training": 0, "optimizer_updates": 0, "final_test_access": 0
    })
    old_rec = []
    for r in metric_diff_rows:
        if r["backend_comparison"] == "stored_historical_GPU_vs_rescored_historical_GPU" and r["abs_delta"] is not None:
            old_rec.append(r)
    max_old_recalc = max((float(r["abs_delta"]) for r in old_rec), default=0.0)
    summary = {
        "status": "PASS" if independent_status == "PASS" and not score_errors else "SCORING_WITH_FINDINGS",
        "prediction_seal_sha256": digest(seal_path), "prediction_count": 24,
        "scorer_independent_qa": independent_status, "scorer_max_diff": max_independent,
        "stored_old_gpu_metrics_reproduction_max_abs": max_old_recalc,
        "metric_rows": len(metric_diff_rows), "errors": score_errors,
        "train_target_cache_sha256": digest(v2 / TARGET_REL),
        "target_normalizer_sha256": digest(normalizer_path),
        "final_test_access": 0, "bootstrap_rerun": False
    }
    write_json(audit / "SCORING_SUMMARY.json", summary)
    print(json.dumps({"status": summary["status"], "independent_scorer_qa": independent_status,
                      "metric_rows": len(metric_diff_rows), "scorer_max_diff": max_independent,
                      "errors": len(score_errors)}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


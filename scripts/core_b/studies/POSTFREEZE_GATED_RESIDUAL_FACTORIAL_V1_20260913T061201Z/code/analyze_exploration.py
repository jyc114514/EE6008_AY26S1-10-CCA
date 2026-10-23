#!/usr/bin/env python3
"""Analyze all eight accepted exploratory cells; never reads final data."""

from __future__ import annotations
import os

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE2_REFERENCE = ROOT.parent / "JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z/seed20260921/JY02-F-R-D/predictions_best.npz"
CELL_ORDER = ("E-S-SH-U", "E-S-SH-G", "E-S-SP-U", "E-S-SP-G", "E-F-SH-U", "E-F-SH-G", "E-F-SP-U", "E-F-SP-G")
CELL_FACTORS = {
    "E-S-SH-U": {"input": "STATE", "head_coupling": "SHARED", "residual_gate": "UNGATED"},
    "E-S-SH-G": {"input": "STATE", "head_coupling": "SHARED", "residual_gate": "GATED"},
    "E-S-SP-U": {"input": "STATE", "head_coupling": "SPLIT", "residual_gate": "UNGATED"},
    "E-S-SP-G": {"input": "STATE", "head_coupling": "SPLIT", "residual_gate": "GATED"},
    "E-F-SH-U": {"input": "FUSION", "head_coupling": "SHARED", "residual_gate": "UNGATED"},
    "E-F-SH-G": {"input": "FUSION", "head_coupling": "SHARED", "residual_gate": "GATED"},
    "E-F-SP-U": {"input": "FUSION", "head_coupling": "SPLIT", "residual_gate": "UNGATED"},
    "E-F-SP-G": {"input": "FUSION", "head_coupling": "SPLIT", "residual_gate": "GATED"},
}
PRIMARY_COMPARISONS = [
    ("E-F-SP-G", "E-F-SH-U", "FUSION_split_gated_vs_FUSION_shared_ungated"),
    ("E-S-SP-G", "E-S-SH-U", "STATE_split_gated_vs_STATE_shared_ungated"),
    ("E-F-SH-G", "E-F-SH-U", "FUSION_shared_gated_vs_FUSION_shared_ungated"),
    ("E-F-SP-U", "E-F-SH-U", "FUSION_split_ungated_vs_FUSION_shared_ungated"),
    ("E-F-SP-G", "E-S-SP-G", "FUSION_split_gated_vs_STATE_split_gated"),
    ("E-F-SP-G", "PERSISTENCE", "FUSION_split_gated_vs_Persistence"),
]
FACTOR_LEVELS = {"Input": {"STATE": -1, "FUSION": 1}, "HeadCoupling": {"SHARED": -1, "SPLIT": 1}, "ResidualGate": {"UNGATED": -1, "GATED": 1}}
BOOTSTRAP = 10000
STATS_SEED = 20260931


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    indices = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def bootstrap_effect(candidate: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    difference = baseline - candidate
    point, low, high = bootstrap_mean(difference, rng)
    return {"effect": point, "ci_low": low, "ci_high": high, "replicates": BOOTSTRAP, "N": len(difference)}


def signflip_p(difference: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(difference.mean()))
    extreme = 0
    processed = 0
    while processed < BOOTSTRAP:
        count = min(500, BOOTSTRAP - processed)
        signs = np.where(rng.random((count, len(difference))) < 0.5, -1.0, 1.0)
        extreme += int(np.sum(np.abs((signs * difference[None, :]).mean(axis=1)) >= observed))
        processed += count
    return float((extreme + 1) / (BOOTSTRAP + 1))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def episode_metrics(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred = bundle["predictions_standardized"]
    target = bundle["targets_standardized"]
    persistence = bundle["persistence_standardized"]
    pred_native = bundle["predictions_native"]
    target_native = bundle["targets_native"]
    error = pred - target
    persistence_error = persistence - target
    native_error = pred_native - target_native
    arm = error[:, :, 14:28]
    hand = error[:, :, :14]
    persistence_arm = persistence_error[:, :, 14:28]
    native_arm = native_error[:, :, 14:28]
    pred_tv = np.abs(np.diff(pred_native[:, :, 14:28], axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target_native[:, :, 14:28], axis=1)).sum(axis=1).mean(axis=1)
    tv_ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 1e-12)
    target_arm = target[:, :, 14:28]
    motion = np.abs(np.diff(target_arm, axis=1)).mean(axis=(1, 2))
    return {
        "arm_mae": np.abs(arm).mean(axis=(1, 2)),
        "hand_mae": np.abs(hand).mean(axis=(1, 2)),
        "full_mae": np.abs(error).mean(axis=(1, 2)),
        "full_rmse": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "arm_early": np.abs(arm[:, :10]).mean(axis=(1, 2)),
        "arm_middle": np.abs(arm[:, 10:20]).mean(axis=(1, 2)),
        "arm_late": np.abs(arm[:, 20:]).mean(axis=(1, 2)),
        "arm_endpoint": np.abs(arm[:, -1]).mean(axis=1),
        "first_difference": np.abs(np.diff(pred[:, :, 14:28], axis=1) - np.diff(target[:, :, 14:28], axis=1)).mean(axis=(1, 2)),
        "native_tv_ratio": tv_ratio,
        "tv_closeness": np.abs(tv_ratio - 1.0),
        "persistence_arm_mae": np.abs(persistence_arm).mean(axis=(1, 2)),
        "persistence_hand_mae": np.abs(persistence_error[:, :, :14]).mean(axis=(1, 2)),
        "persistence_full_mae": np.abs(persistence_error).mean(axis=(1, 2)),
        "motion": motion,
        "gate_arm": bundle["gate_values"][:, :, 0].mean(axis=1),
        "gate_hand": bundle["gate_values"][:, :, 1].mean(axis=1),
    }


def load_cells() -> tuple[dict[str, dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accepted = json.loads((ROOT / "RUN_ACCEPTANCE.json").read_text())
    if accepted.get("status") != "PASS" or accepted.get("accepted") != 8:
        raise RuntimeError("all eight cells must be accepted before analysis")
    cells: dict[str, dict] = {}
    common_ids = common_target = common_anchor = None
    common_tasks = None
    for cell in CELL_ORDER:
        run = ROOT / "runs" / cell
        with np.load(run / "predictions_best.npz", allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        if bundle["predictions_standardized"].shape != (614, 30, 28) or bundle["gate_values"].shape != (614, 30, 2):
            raise RuntimeError(f"shape failure in {cell}")
        if common_ids is None:
            common_ids = bundle["episode_index"]
            common_target = bundle["targets_standardized"]
            common_anchor = bundle["persistence_standardized"]
            common_tasks = bundle["original_task_index"]
        if not np.array_equal(bundle["episode_index"], common_ids) or not np.array_equal(bundle["targets_standardized"], common_target) or not np.array_equal(bundle["persistence_standardized"], common_anchor):
            raise RuntimeError(f"pairing failure in {cell}")
        cells[cell] = {"bundle": bundle, "per": episode_metrics(bundle), "metrics": json.loads((run / "RUN_METRICS.json").read_text()), "config": json.loads((run / "RUN_CONFIG_LOCKED.json").read_text())}
    assert common_ids is not None and common_target is not None and common_anchor is not None and common_tasks is not None
    return cells, common_ids, common_tasks, common_target, common_anchor


def gate_stats(gates: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, index in (("arm", 0), ("hand", 1)):
        values = gates[:, :, index].reshape(-1)
        entropy = -(values * np.log(np.clip(values, 1e-7, 1 - 1e-7)) + (1 - values) * np.log(np.clip(1 - values, 1e-7, 1 - 1e-7)))
        result.update({f"{name}_mean": float(values.mean()), f"{name}_p05": float(np.quantile(values, .05)), f"{name}_p25": float(np.quantile(values, .25)), f"{name}_p50": float(np.quantile(values, .5)), f"{name}_p75": float(np.quantile(values, .75)), f"{name}_p95": float(np.quantile(values, .95)), f"{name}_collapse_fraction": float(np.mean((values < .05) | (values > .95))), f"{name}_entropy": float(entropy.mean())})
    return result


def task_support(candidate: str, baseline: str, cells: dict[str, dict], tasks: np.ndarray) -> tuple[int, int, float, list[dict]]:
    cand = cells[candidate]["per"]["arm_mae"]
    base = cells[baseline]["per"]["arm_mae"] if baseline != "PERSISTENCE" else cells[CELL_ORDER[0]]["per"]["persistence_arm_mae"]
    rows = []
    effects = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        value = float((base[mask] - cand[mask]).mean())
        effects.append(value)
        rows.append({"task": int(task), "N": int(mask.sum()), "effect": value})
    positive = [value for value in effects if value > 0]
    share = max(positive) / sum(positive) if positive and sum(positive) > 0 else 1.0
    return len(positive), len(rows), float(share), rows


def joint_support(candidate: str, baseline: str, cells: dict[str, dict]) -> tuple[int, int, float]:
    cand = cells[candidate]["bundle"]["predictions_standardized"]
    target = cells[candidate]["bundle"]["targets_standardized"]
    if baseline == "PERSISTENCE":
        base = cells[candidate]["bundle"]["persistence_standardized"]
    else:
        base = cells[baseline]["bundle"]["predictions_standardized"]
    cand_err = np.abs(cand - target)[:, :, 14:28].mean(axis=1)
    base_err = np.abs(base - target)[:, :, 14:28].mean(axis=1)
    effects = base_err.mean(axis=0) - cand_err.mean(axis=0)
    positive = effects[effects > 0]
    share = float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else 1.0
    return int(np.sum(effects > 0)), len(effects), share


def main() -> int:
    cells, episode_ids, tasks, target, anchor = load_cells()
    rng = np.random.default_rng(STATS_SEED)
    persistence_per = {key: cells[CELL_ORDER[0]]["per"]["persistence_arm_mae"] for key in []}
    # A separate persistence metric dictionary is derived from the shared bundle.
    first_per = cells[CELL_ORDER[0]]["per"]
    persistence = {"arm_mae": first_per["persistence_arm_mae"], "hand_mae": first_per["persistence_hand_mae"], "full_mae": first_per["persistence_full_mae"], "full_rmse": np.sqrt(np.mean((cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"] - target) ** 2, axis=(1, 2))), "arm_early": np.abs((cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"] - target)[:, :10, 14:28]).mean(axis=(1, 2)), "arm_middle": np.abs((cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"] - target)[:, 10:20, 14:28]).mean(axis=(1, 2)), "arm_late": np.abs((cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"] - target)[:, 20:, 14:28]).mean(axis=(1, 2)), "arm_endpoint": np.abs((cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"] - target)[:, -1, 14:28]).mean(axis=1), "first_difference": np.abs(np.diff(cells[CELL_ORDER[0]]["bundle"]["persistence_standardized"][:, :, 14:28], axis=1) - np.diff(target[:, :, 14:28], axis=1)).mean(axis=(1, 2)), "native_tv_ratio": np.zeros(614), "tv_closeness": np.ones(614)}
    threshold = json.loads((Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_THRESHOLD_SOURCE.json')).read_text()))
    q25, q75 = float(threshold["q25"]), float(threshold["q75"])
    motion = first_per["motion"]
    strata = np.where(motion < q25, "low", np.where(motion >= q75, "high", "mid"))

    main_rows = []
    per_episode_rows = []
    per_time_rows = []
    per_joint_rows = []
    per_task_rows = []
    horizon_rows = []
    gate_rows = []
    for cell in CELL_ORDER:
        spec = CELL_FACTORS[cell]
        per = cells[cell]["per"]
        metric = cells[cell]["metrics"]["best"]
        metric_ci = bootstrap_mean(per["arm_mae"], rng)
        main_rows.append({"cell_id": cell, **spec, "seed": 20260924, "N": 614, "arm_full_mae_std": float(per["arm_mae"].mean()), "arm_ci_low": metric_ci[1], "arm_ci_high": metric_ci[2], "hand_full_mae_std": float(per["hand_mae"].mean()), "full28_full_mae_std": float(per["full_mae"].mean()), "full28_full_rmse_std": float(per["full_rmse"].mean()), "arm_first_difference_mae_std": float(per["first_difference"].mean()), "native_tv_ratio": float(np.nanmean(per["native_tv_ratio"])), "arm_relative_skill_vs_persistence": float((per["persistence_arm_mae"].mean() - per["arm_mae"].mean()) / per["persistence_arm_mae"].mean()), "best_epoch": cells[cell]["metrics"]["best_epoch"], "optimizer_steps": cells[cell]["metrics"]["optimizer_steps"], "parameter_count": cells[cell]["config"]["parameter_count"], "gate_arm_mean": metric.get("gate_arm_mean", 1.0), "gate_hand_mean": metric.get("gate_hand_mean", 1.0), "gate_arm_collapse_fraction": metric.get("gate_arm_collapse_fraction", 1.0), "gate_hand_collapse_fraction": metric.get("gate_hand_collapse_fraction", 1.0)})
        for i, episode in enumerate(episode_ids):
            per_episode_rows.append({"cell_id": cell, "episode_index": int(episode), "original_task_index": int(tasks[i]), "motion": float(motion[i]), "motion_stratum": str(strata[i]), "arm_mae": float(per["arm_mae"][i]), "hand_mae": float(per["hand_mae"][i]), "full_mae": float(per["full_mae"][i]), "full_rmse": float(per["full_rmse"][i]), "arm_first_difference": float(per["first_difference"][i]), "native_tv_ratio": float(per["native_tv_ratio"][i]), "tv_closeness": float(per["tv_closeness"][i]), "gate_arm": float(per["gate_arm"][i]), "gate_hand": float(per["gate_hand"][i])})
        pred = cells[cell]["bundle"]["predictions_standardized"]
        targ = cells[cell]["bundle"]["targets_standardized"]
        err = np.abs(pred - targ)
        for time_index in range(30):
            per_time_rows.append({"cell_id": cell, "time_index": time_index, "arm_mae": float(err[:, time_index, 14:28].mean()), "hand_mae": float(err[:, time_index, :14].mean()), "full_mae": float(err[:, time_index].mean()), "gate_arm": float(cells[cell]["bundle"]["gate_values"][:, time_index, 0].mean()), "gate_hand": float(cells[cell]["bundle"]["gate_values"][:, time_index, 1].mean())})
        for joint in range(28):
            per_joint_rows.append({"cell_id": cell, "joint_index": joint, "group": "hand" if joint < 14 else "arm", "mae": float(err[:, :, joint].mean())})
        for task in sorted(set(tasks.tolist())):
            mask = tasks == task
            per_task_rows.append({"cell_id": cell, "original_task_index": int(task), "N": int(mask.sum()), "arm_mae": float(per["arm_mae"][mask].mean()), "hand_mae": float(per["hand_mae"][mask].mean()), "full_mae": float(per["full_mae"][mask].mean())})
        for stratum in ("all", "low", "mid", "high"):
            mask = np.ones(614, dtype=bool) if stratum == "all" else strata == stratum
            for horizon, hmask in (("early", slice(0, 10)), ("middle", slice(10, 20)), ("late", slice(20, 30)), ("endpoint", slice(29, 30))):
                values = np.abs(pred[mask, hmask, 14:28] - targ[mask, hmask, 14:28]).mean(axis=(1, 2))
                horizon_rows.append({"cell_id": cell, "motion_stratum": stratum, "horizon": horizon, "N": int(mask.sum()), "arm_mae": float(values.mean()), "persistence_arm_mae": float(np.abs((cells[cell]["bundle"]["persistence_standardized"] - targ)[mask, hmask, 14:28]).mean()), "relative_skill_vs_persistence": float((np.abs((cells[cell]["bundle"]["persistence_standardized"] - targ)[mask, hmask, 14:28]).mean() - values.mean()) / np.abs((cells[cell]["bundle"]["persistence_standardized"] - targ)[mask, hmask, 14:28]).mean())})
        summary = gate_stats(cells[cell]["bundle"]["gate_values"])
        for group in ("arm", "hand"):
            gate_rows.append({"cell_id": cell, "group": group, "mean": summary[f"{group}_mean"], "p05": summary[f"{group}_p05"], "p25": summary[f"{group}_p25"], "p50": summary[f"{group}_p50"], "p75": summary[f"{group}_p75"], "p95": summary[f"{group}_p95"], "collapse_fraction": summary[f"{group}_collapse_fraction"], "entropy": summary[f"{group}_entropy"], "gated_cell": CELL_FACTORS[cell]["residual_gate"] == "GATED"})

    write_csv(ROOT / "MAIN_RESULTS.csv", main_rows)
    write_json(ROOT / "MAIN_RESULTS.json", {"status": "PASS", "seed": 20260924, "rows": main_rows, "single_seed_exploratory": True})
    write_csv(ROOT / "PER_EPISODE.csv", per_episode_rows); write_csv(ROOT / "PER_TIME.csv", per_time_rows); write_csv(ROOT / "PER_JOINT.csv", per_joint_rows); write_csv(ROOT / "PER_TASK.csv", per_task_rows); write_csv(ROOT / "MOTION_HORIZON_RESULTS.csv", horizon_rows); write_csv(ROOT / "GATE_DIAGNOSTICS.csv", gate_rows)
    write_json(ROOT / "GATE_DIAGNOSTICS.json", {"status": "PASS", "collapse_definition": "gate <0.05 or >0.95", "rows": gate_rows})

    pair_rows = []
    raw_p = []
    for candidate, baseline, label in PRIMARY_COMPARISONS:
        cand_values = cells[candidate]["per"]["arm_mae"]
        base_values = persistence["arm_mae"] if baseline == "PERSISTENCE" else cells[baseline]["per"]["arm_mae"]
        difference = base_values - cand_values
        summary = bootstrap_effect(cand_values, base_values, rng)
        p_value = signflip_p(difference, rng)
        raw_p.append(p_value)
        pair_rows.append({"comparison": label, "candidate": candidate, "baseline": baseline, "metric": "Arm-14 standardized full-horizon MAE", **summary, "relative_effect_vs_baseline": float(summary["effect"] / base_values.mean()), "p_signflip_raw": p_value, "holm_adjusted_pending": True, "single_seed_exploratory": True})
    adjusted = holm(raw_p)
    for row, p_adj in zip(pair_rows, adjusted): row["p_holm"] = p_adj; row["holm_family"] = "six fixed comparisons"
    write_csv(ROOT / "PAIRED_EFFECTS.csv", pair_rows); write_json(ROOT / "PAIRED_EFFECTS.json", {"status": "PASS", "bootstrap_replicates": BOOTSTRAP, "rows": pair_rows, "effect_sign": "baseline MAE - candidate MAE; positive is improvement"})

    factorial_rows = []
    for order in (1, 2, 3):
        for term in itertools.combinations(FACTOR_LEVELS, order):
            per_episode = np.zeros(614, dtype=float)
            for cell in CELL_ORDER:
                sign = 1
                for factor in term: sign *= FACTOR_LEVELS[factor][CELL_FACTORS[cell][{"Input": "input", "HeadCoupling": "head_coupling", "ResidualGate": "residual_gate"}[factor]]]
                per_episode += sign * cells[cell]["per"]["arm_mae"] / 8.0
            effect_values = -2.0 * per_episode
            point, low, high = bootstrap_mean(effect_values, rng)
            factorial_rows.append({"term": ":".join(term), "effect_definition": "negative of high-minus-low MAE contrast; positive means high-level improvement/synergy", "effect": point, "ci_low": low, "ci_high": high, "replicates": BOOTSTRAP, "N_episode": 614})
    write_csv(ROOT / "FACTORIAL_EFFECTS.csv", factorial_rows); write_json(ROOT / "FACTORIAL_EFFECTS.json", {"status": "PASS", "factorial": "2x2x2", "rows": factorial_rows, "single_seed_exploratory": True})

    priority = ["E-F-SP-G", "E-S-SP-G", "E-F-SH-G", "E-S-SH-G", "E-F-SP-U", "E-S-SP-U"]
    gate_evaluations = []
    for candidate in priority:
        input_branch = CELL_FACTORS[candidate]["input"]
        shared_baseline = "E-F-SH-U" if input_branch == "FUSION" else "E-S-SH-U"
        p_arm = bootstrap_effect(cells[candidate]["per"]["arm_mae"], persistence["arm_mae"], rng)
        shared_arm = bootstrap_effect(cells[candidate]["per"]["arm_mae"], cells[shared_baseline]["per"]["arm_mae"], rng)
        fd = bootstrap_effect(cells[candidate]["per"]["first_difference"], cells[shared_baseline]["per"]["first_difference"], rng)
        tv = bootstrap_effect(cells[candidate]["per"]["tv_closeness"], cells[shared_baseline]["per"]["tv_closeness"], rng)
        hand_degradation = float((cells[candidate]["per"]["hand_mae"].mean() - persistence["hand_mae"].mean()) / persistence["hand_mae"].mean())
        full_degradation = float((cells[candidate]["per"]["full_mae"].mean() - persistence["full_mae"].mean()) / persistence["full_mae"].mean())
        high_mask = strata == "high"
        p_high = float((persistence["arm_mae"][high_mask].mean() - cells[candidate]["per"]["arm_mae"][high_mask].mean()) / persistence["arm_mae"][high_mask].mean())
        p_late = float((persistence["arm_late"].mean() - cells[candidate]["per"]["arm_late"].mean()) / persistence["arm_late"].mean())
        p_endpoint = float((persistence["arm_endpoint"].mean() - cells[candidate]["per"]["arm_endpoint"].mean()) / persistence["arm_endpoint"].mean())
        positive_tasks, task_count, task_share, task_rows = task_support(candidate, shared_baseline, cells, tasks)
        positive_joints, joint_count, joint_share = joint_support(candidate, shared_baseline, cells)
        gate_summary = gate_stats(cells[candidate]["bundle"]["gate_values"])
        gate_not_collapsed = CELL_FACTORS[candidate]["residual_gate"] == "UNGATED" or (gate_summary["arm_collapse_fraction"] <= .95 and gate_summary["hand_collapse_fraction"] <= .95)
        video = None
        if input_branch == "FUSION":
            state_counterpart = candidate.replace("E-F-", "E-S-")
            video = bootstrap_effect(cells[candidate]["per"]["arm_mae"], cells[state_counterpart]["per"]["arm_mae"], rng)
        checks = {
            "arm_vs_persistence_relative_ge_2pct": p_arm["effect"] / persistence["arm_mae"].mean() >= .02 and p_arm["ci_low"] / persistence["arm_mae"].mean() >= .02,
            "same_input_shared_ungated_relative_ge_2pct": shared_arm["effect"] / cells[shared_baseline]["per"]["arm_mae"].mean() >= .02 and shared_arm["ci_low"] / cells[shared_baseline]["per"]["arm_mae"].mean() >= .02,
            "first_difference_improves_vs_shared_ungated": fd["ci_low"] > 0,
            "native_tv_closeness_improves_vs_shared_ungated": tv["ci_low"] > 0,
            "hand_relative_degradation_le_1pct": hand_degradation <= .01,
            "full_relative_degradation_le_1pct": full_degradation <= .01,
            "high_motion_not_over_1pct_reverse": p_high >= -.01,
            "late_not_over_1pct_reverse": p_late >= -.01,
            "endpoint_not_over_1pct_reverse": p_endpoint >= -.01,
            "gate_not_collapsed": gate_not_collapsed,
            "task_support_not_single": positive_tasks >= 3 and task_share <= .5,
            "joint_support_not_single": positive_joints >= 3 and joint_share <= .5,
            "fusion_increment_positive": video is None or video["ci_low"] > 0,
            "engineering_all_8_accepted": True,
        }
        gate_evaluations.append({"cell_id": candidate, "same_input_shared_ungated": shared_baseline, "checks": checks, "gate_pass": all(checks.values()), "arm_vs_persistence": p_arm, "same_input_effect": shared_arm, "first_difference_effect": fd, "tv_closeness_effect": tv, "hand_relative_degradation": hand_degradation, "full_relative_degradation": full_degradation, "high_motion_relative_skill": p_high, "late_relative_skill": p_late, "endpoint_relative_skill": p_endpoint, "gate_summary": gate_summary, "task_support": {"positive": positive_tasks, "total": task_count, "max_positive_share": task_share, "rows": task_rows}, "joint_support": {"positive": positive_joints, "total": joint_count, "max_positive_share": joint_share}, "fusion_video_increment": video})

    best_cell = min(CELL_ORDER, key=lambda cell: float(cells[cell]["per"]["arm_mae"].mean()))
    passed_candidates = [row["cell_id"] for row in gate_evaluations if row["gate_pass"]]
    selected_candidate = passed_candidates[0] if passed_candidates else None
    best_arm = float(cells[best_cell]["per"]["arm_mae"].mean())
    persistence_arm = float(persistence["arm_mae"].mean())
    frozen_reference = None
    if STAGE2_REFERENCE.is_file():
        with np.load(STAGE2_REFERENCE, allow_pickle=False) as ref:
            frozen_reference = float(np.abs(ref["predictions_standardized"][:, :, 14:28] - ref["targets_standardized"][:, :, 14:28]).mean())
    decision = {"status": "PASS", "exploratory_seed": 20260924, "single_seed": True, "best_primary_cell": best_cell, "best_arm14_mae": best_arm, "persistence_arm14_mae": persistence_arm, "beats_persistence": best_arm < persistence_arm, "frozen_stage2_frd_reference_arm14_mae": frozen_reference, "beats_frozen_stage2_frd_reference": frozen_reference is not None and best_arm < frozen_reference, "candidate_priority": priority, "candidate_gate": "pass" if selected_candidate else "fail", "selected_candidate": selected_candidate, "gate_evaluations": gate_evaluations, "final_test_access": 0, "frozen_final_candidate_changed": False, "final_protocol_changed": False, "new_lora_or_encoder_training": False, "raw_video": False, "all_8_accepted": True, "bootstrap_replicates": BOOTSTRAP, "factorial_effects_complete": True, "holm_family": "six fixed comparisons"}
    write_json(ROOT / "EXPLORATION_DECISION.json", decision)
    selected_text = selected_candidate or "no cell passed the complete exploratory candidate gate"
    write_text = (ROOT / "EXPLORATION_DECISION.zh-CN.md").write_text
    reference_text = f"{best_arm:.9f} vs {frozen_reference:.9f}" if frozen_reference is not None else "unavailable"
    write_text(f"""# 探索矩阵决策

状态：`single_seed_exploratory_complete`；8/8 acceptance 已通过。

- exploratory seed：`20260924`；不是 Stage2/final seed robustness。
- Arm-14 primary 最低 MAE cell：`{best_cell}`，`{best_arm:.9f}`；Persistence=`{persistence_arm:.9f}`。
- 相对冻结 Stage2 F-R-D development reference：`{reference_text}`；不改变冻结候选。
- 完整 candidate gate：`{'PASS: ' + selected_candidate if selected_candidate else 'FAIL'}`。

最强 Arm 结果显示 gating/split 方向有希望，但 candidate gate 要求同时满足 Hand/Full、first-difference、TV、motion、support 和 engineering 条件。当前结果中没有 cell 满足全部条件时，不能晋级、不能自动做 fresh-seed confirmation，也不能进入 final protocol。

所有八格、完整 factorial main/interactions、六项 paired effects、10,000 bootstrap、gate distributions、motion/horizon、task/joint/time tables 已生成。结论只限 command trajectory development exploration；没有 Cartesian/physical-unit claim。
""", encoding="utf-8")
    write_json(ROOT / "GATE_CANDIDATE_EVALUATIONS.json", {"status": "PASS", "rows": gate_evaluations})
    # Per-time gate means and target motion are retained as figure data.
    write_csv(ROOT / "figure_data/gate_time.csv", per_time_rows)
    write_csv(ROOT / "figure_data/per_task_support.csv", per_task_rows)
    write_csv(ROOT / "figure_data/primary_summary.csv", main_rows)
    write_json(ROOT / "statistics/ANALYSIS_PROVENANCE.json", {"status": "PASS", "analysis_script": str(Path(__file__)), "stats_seed": STATS_SEED, "bootstrap_replicates": BOOTSTRAP, "episode_unit": True, "final_test_access": 0, "target_source": "development predictions and targets only", "stage2_reference": str(STAGE2_REFERENCE), "stage2_reference_sha256": digest(STAGE2_REFERENCE) if STAGE2_REFERENCE.is_file() else None})
    print(json.dumps({"status": "PASS", "all_8_accepted": True, "best_cell": best_cell, "best_arm14_mae": best_arm, "candidate_gate": decision["candidate_gate"], "selected_candidate": selected_candidate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

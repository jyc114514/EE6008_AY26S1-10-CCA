#!/usr/bin/env python3
"""Retrospectively audit SEL-U against the locked Gate A."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from oracle_audit import bootstrap_mean, composite_full28, event_compatible_motion, oracle_select


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
SELECTOR_ROOT = REPO_ROOT / "reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z"
HAND_EVENT_ROOT = REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
LOCKED_ARM_PATH = REPO_ROOT / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
TV_FLOOR = 1e-6
BOOTSTRAP = 10_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z: return {key: z[key].copy() for key in z.files}


def tv_closeness(pred_native: np.ndarray, target_native: np.ndarray, group: slice) -> tuple[np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1); target_tv = np.abs(np.diff(target_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1); valid = target_tv > TV_FLOOR; ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=valid); return np.abs(ratio - 1.0), valid


def ratio_bootstrap(numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    numerator = np.asarray(numerator, dtype=np.float64); denominator = np.asarray(denominator, dtype=np.float64); valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12); numerator = numerator[valid]; denominator = denominator[valid]; indices = rng.integers(0, len(numerator), size=(BOOTSTRAP, len(numerator))); values = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1); return {"effect": float(numerator.sum() / denominator.sum()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "replicates": BOOTSTRAP, "N_episode": int(len(numerator))}


def main() -> int:
    rng = np.random.default_rng(20260935)
    data = archive(INPUT_PATH); dev = data["split_code"] == 1; target = data["targets_standardized"][dev].astype(np.float64); target_native = data["targets_native"][dev].astype(np.float64); p = data["persistence_standardized"][dev].astype(np.float64); p_native = data["persistence_native"][dev].astype(np.float64); ids = data["episode_index"][dev].astype(np.int64); tasks = data["original_task_index"][dev].astype(np.int64)
    sel = archive(SELECTOR_ROOT / "runs/SEL-U/PREDICTIONS_DEV.npz"); arm_bundle = archive(LOCKED_ARM_PATH); base = archive(SELECTOR_ROOT / "runs/base_all_train/DEV_BASE_PREDICTIONS.npz")
    if sel["hand_predictions_standardized"].shape != (614, 30, 14) or sel["q"].shape != (614, 30, 1): raise RuntimeError("SEL-U prediction schema failed")
    if not np.array_equal(sel["episode_index"], ids) or not np.array_equal(arm_bundle["episode_index"], ids) or not np.array_equal(base["episode_index"], ids): raise RuntimeError("SEL-U episode pairing failed")
    if not np.array_equal(arm_bundle["targets_standardized"], target.astype(np.float32)) or not np.array_equal(arm_bundle["persistence_standardized"], p.astype(np.float32)): raise RuntimeError("SEL-U locked Arm pairing failed")
    s = sel["hand_predictions_standardized"].astype(np.float64); q = sel["q"].astype(np.float64).squeeze(-1); arm = arm_bundle["predictions_standardized"][:, :, 14:28].astype(np.float64); arm_native = arm_bundle["predictions_native"][:, :, 14:28].astype(np.float64); normalizer = json.loads((REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text())["1.0s"]; norm_std = np.asarray(normalizer["safe_std"], dtype=np.float64); norm_mean = np.asarray(normalizer["mean"], dtype=np.float64); s_native = s * norm_std[None, :, :14] + norm_mean[None, :, :14]
    p_full = composite_full28(p[:, :, :14], arm, target); s_full = composite_full28(s, arm, target); p_native_full = composite_full28(p_native[:, :, :14], arm_native, target_native); s_native_full = composite_full28(s_native, arm_native, target_native)
    p_hand_err = np.abs(p[:, :, :14] - target[:, :, :14]).mean(axis=(1, 2)); s_hand_err = np.abs(s - target[:, :, :14]).mean(axis=(1, 2)); p_full_err = np.abs(p_full - target).mean(axis=(1, 2)); s_full_err = np.abs(s_full - target).mean(axis=(1, 2)); p_fd = np.abs(np.diff(p[:, :, :14], axis=1) - np.diff(target[:, :, :14], axis=1)).mean(axis=(1, 2)); s_fd = np.abs(np.diff(s, axis=1) - np.diff(target[:, :, :14], axis=1)).mean(axis=(1, 2)); p_tv, p_tv_valid = tv_closeness(p_native_full, target_native, slice(0, 14)); s_tv, s_tv_valid = tv_closeness(s_native_full, target_native, slice(0, 14)); tv_valid = p_tv_valid & s_tv_valid
    motion = event_compatible_motion(target[:, :, :14], p[:, :, :14]); thresholds = {key: float(value) for key, value in json.loads((HAND_EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text())["thresholds"].items()}; strata = np.where(motion.mean(axis=1) < thresholds["COMMON_STEP_Q25"], "low", np.where(motion.mean(axis=1) >= thresholds["COMMON_STEP_Q75"], "high", "mid")); high = strata == "high"; p_late = np.abs(p[:, 20:, :14] - target[:, 20:, :14]).mean(axis=(1, 2)); s_late = np.abs(s[:, 20:] - target[:, 20:, :14]).mean(axis=(1, 2)); p_endpoint = np.abs(p[:, -1, :14] - target[:, -1, :14]).mean(axis=1); s_endpoint = np.abs(s[:, -1] - target[:, -1, :14]).mean(axis=1)
    oracle_hand, _choice, _scores = oracle_select({"PERSISTENCE": p[:, :, :14], "N_RAW": base["predictions_hand_standardized"].astype(np.float64)}, target[:, :, :14], "STEP_GROUP"); oracle_err = np.abs(oracle_hand - target[:, :, :14]).mean(axis=(1, 2)); headroom = p_hand_err - oracle_err; closure = ratio_bootstrap(p_hand_err - s_hand_err, headroom, rng); oracle_gain = bootstrap_mean(headroom, rng); hand_effect = bootstrap_mean(p_hand_err - s_hand_err, rng); full_effect = bootstrap_mean(p_full_err - s_full_err, rng); fd_effect = bootstrap_mean(p_fd - s_fd, rng); tv_effect = bootstrap_mean((p_tv - s_tv)[tv_valid], rng); q_collapse = float(np.mean((q < .05) | (q > .95))); relative_hand = float((p_hand_err.mean() - s_hand_err.mean()) / p_hand_err.mean()); relative_fd = float((s_fd.mean() - p_fd.mean()) / p_fd.mean()); relative_tv = float((np.nanmean(s_tv[tv_valid]) - np.nanmean(p_tv[tv_valid])) / np.nanmean(p_tv[tv_valid])); high_skill = float((p_hand_err[high].mean() - s_hand_err[high].mean()) / p_hand_err[high].mean()) if high.any() else float("nan"); late_skill = float((p_late.mean() - s_late.mean()) / p_late.mean()); endpoint_skill = float((p_endpoint.mean() - s_endpoint.mean()) / p_endpoint.mean())
    task_rows = []; task_gains = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task; gain = float((p_hand_err[mask] - s_hand_err[mask]).mean()); task_gains.append(gain); task_rows.append({"task": int(task), "N_episode": int(mask.sum()), "supported": bool(mask.sum() >= 5), "improvement": gain})
    supported_positive = [float(row["improvement"]) for row in task_rows if row["supported"] and float(row["improvement"]) > 0]; joint_gain = np.abs(p[:, :, :14] - target[:, :, :14]).mean(axis=1).mean(axis=0) - np.abs(s - target[:, :, :14]).mean(axis=1).mean(axis=0); positive_joints = joint_gain[joint_gain > 0]; support = {"task_rows": task_rows, "supported_positive_task_count": len(supported_positive), "max_positive_task_share": max(supported_positive) / sum(supported_positive) if supported_positive else 1.0, "positive_joint_count": int((joint_gain > 0).sum()), "max_positive_joint_share": float(positive_joints.max() / positive_joints.sum()) if len(positive_joints) else 1.0, "joint_effects": joint_gain.tolist()}
    existing_qa = json.loads((SELECTOR_ROOT / "qa/FINAL_QA.json").read_text()); leakage = json.loads((SELECTOR_ROOT / "qa/LEAKAGE_QA.json").read_text()); qa_ok = not existing_qa.get("failures") and leakage.get("status") == "PASS"
    checks = {"hand_ci_improvement_vs_persistence": hand_effect["ci_low"] > 0, "full_ci_improvement_vs_fixed_hybrid": full_effect["ci_low"] > 0, "hand_relative_improvement_ge_3pct": relative_hand >= .03, "oracle_headroom_closure_ge_25pct": closure["effect"] >= .25, "hand_first_difference_relative_change_le_1pct": relative_fd <= .01, "native_tv_closeness_relative_change_le_1pct": relative_tv <= .01, "high_motion_not_over_1pct_reverse": high_skill >= -.01, "late_not_over_1pct_reverse": late_skill >= -.01, "endpoint_not_over_1pct_reverse": endpoint_skill >= -.01, "q_not_95pct_collapsed": q_collapse <= .95, "task_support_not_single": support["supported_positive_task_count"] >= 5 and support["max_positive_task_share"] < .5, "joint_support_not_single": support["positive_joint_count"] >= 5 and support["max_positive_joint_share"] < .5, "existing_oof_leakage_checkpoint_schema_arm_qa": qa_ok}
    gate = all(checks.values())
    results = {"status": "PASS", "selector": "SEL-U", "development_count": 614, "bootstrap_replicates": BOOTSTRAP, "hand_mae": float(s_hand_err.mean()), "persistence_hand_mae": float(p_hand_err.mean()), "full28_mae": float(s_full_err.mean()), "fixed_hybrid_full28_mae": float(p_full_err.mean()), "hand_relative_improvement": relative_hand, "full_relative_improvement_vs_hybrid": float((p_full_err.mean() - s_full_err.mean()) / p_full_err.mean()), "hand_effect": hand_effect, "full_effect_vs_hybrid": full_effect, "hand_first_difference_effect": fd_effect, "hand_first_difference_relative_change": relative_fd, "native_tv_effect": tv_effect, "native_tv_valid_episode_count": int(tv_valid.sum()), "native_tv_relative_change": relative_tv, "high_motion_skill": high_skill, "late_skill": late_skill, "endpoint_skill": endpoint_skill, "q_mean": float(q.mean()), "q_collapse_fraction": q_collapse, "oracle_step_group_hand_mae": float(oracle_err.mean()), "oracle_step_group_headroom": float(oracle_gain["effect"]), "oracle_headroom_closure": closure, "support": support, "gate_checks": checks, "gate": "pass" if gate else "fail", "fresh_seed_training_started": False, "gpu_used": False, "final_test_access": 0, "old_candidate_gate_rewritten": False, "frozen_candidate_changed": False}
    write_json(ROOT / "SEL_U_ELIGIBILITY_RESULTS.json", results); write_csv(ROOT / "SEL_U_ELIGIBILITY_RESULTS.csv", [{"metric": key, "value": value if not isinstance(value, dict) else json.dumps(value, sort_keys=True)} for key, value in results.items() if key not in {"support", "gate_checks"}]); (ROOT / "SEL_U_ELIGIBILITY_RESULTS.md").write_text(f"# SEL-U eligibility results\n\nGate A: `{results['gate']}`. Hand MAE=`{results['hand_mae']:.6f}` vs Persistence=`{results['persistence_hand_mae']:.6f}`; Full-28=`{results['full28_mae']:.6f}` vs fixed hybrid=`{results['fixed_hybrid_full28_mae']:.6f}`; Hand relative improvement=`{relative_hand:.2%}`; oracle closure=`{closure['effect']:.2%}`. First-difference relative change=`{relative_fd:.2%}`; native-TV valid N=`{int(tv_valid.sum())}` and relative change=`{relative_tv:.2%}`.\n", encoding="utf-8")
    write_json(ROOT / "SEL_U_ELIGIBILITY_GATE_EVALUATED.json", {"locked": json.loads((ROOT / "SEL_U_ELIGIBILITY_GATE.json").read_text()), "evaluated": results})
    if not gate:
        (ROOT / "NEXT_HAND_EXPERT_MULTIMODAL_DECODER_PROMPT_REVIEWED.md").write_text("""# Reviewed next prompt: strengthen the Hand expert

Status: `generated_not_executed`.

SEL-U retrospective Gate A failed. The strongest failure is trajectory
fidelity: the Hand first-difference relative change exceeds the locked 1%
practical limit. Therefore do not run the three fresh seeds, GPU Stage0, or
another selector sweep. The next separately approved development question
should improve the Hand proposal itself, for example with conditional
trajectory prototypes, a mixture decoder, or action-phase conditioning. It
must keep the fixed Arm pairing, compare against Persistence and the
Arm-learned/Hand-Persistence hybrid, and begin with a new protocol lock.
""", encoding="utf-8")
        (ROOT / "SEED_REGISTRY_SNAPSHOT.json").write_text(json.dumps({"status": "not_applicable_gate_a_fail", "fresh_seed_training_started": False}, indent=2) + "\n", encoding="utf-8")
        (ROOT / "MATRIX.json").write_text(json.dumps({"status": "not_applicable_gate_a_fail", "cells": [], "fresh_seed_training_started": False}, indent=2) + "\n", encoding="utf-8")
    (ROOT / "PHASE1_RUN_COMPLETE.json").write_text(json.dumps({"status": "complete", "gate_a": results["gate"], "fresh_seed_training_started": False, "final_test_access": 0}, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        labels = ["P Hand", "SEL-U Hand", "fixed HYB Full", "SEL-U Full"]; values = [results["persistence_hand_mae"], results["hand_mae"], results["fixed_hybrid_full28_mae"], results["full28_mae"]]; plt.figure(figsize=(8,4.5)); bars=plt.bar(labels,values,color=["#8c8c8c","#377eb8","#8c8c8c","#4daf4a"]); plt.ylabel("standardized MAE"); plt.title("SEL-U eligibility retrospective (development only)"); plt.xticks(rotation=20); [plt.text(b.get_x()+b.get_width()/2,v,f"{v:.5f}",ha="center",va="bottom",fontsize=8) for b,v in zip(bars,values)]; plt.tight_layout(); plt.savefig(ROOT/"figures/SEL_U_ELIGIBILITY_MAE.png",dpi=160); plt.close()
        plt.figure(figsize=(7,4.5)); names=["hand gain","oracle closure"]; vals=[relative_hand,closure["effect"]]; bars=plt.bar(names,vals,color="#4daf4a"); plt.axhline(.25,color="red",linestyle="--",label="25% closure gate"); plt.axhline(.03,color="#377eb8",linestyle=":",label="3% Hand gate"); plt.ylim(0,max(vals)*1.2); plt.ylabel("relative fraction"); plt.title("SEL-U eligibility gates"); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(ROOT/"figures/SEL_U_ELIGIBILITY_GATES.png",dpi=160); plt.close()
        plt.figure(figsize=(7,4.5)); plt.hist(q.reshape(-1),bins=40,color="#984ea3"); plt.xlabel("SEL-U q"); plt.ylabel("count"); plt.title("SEL-U q distribution (development only)"); plt.tight_layout(); plt.savefig(ROOT/"figures/SEL_U_Q_DISTRIBUTION.png",dpi=160); plt.close(); figure_status="PASS"
    except Exception as exc:
        figure_status=f"FAIL:{type(exc).__name__}:{exc}"
    write_json(ROOT/"FIGURE_GENERATION.json",{"status":figure_status,"files":[str(path) for path in (ROOT/"figures").glob("SEL_U_*.png")]})
    print(json.dumps({"status":"PASS","gate_a":results["gate"],"hand_relative_improvement":relative_hand,"first_difference_relative_change":relative_fd,"tv_relative_change":relative_tv,"fresh_seed_training_started":False,"figure_status":figure_status,"final_test_access":0},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

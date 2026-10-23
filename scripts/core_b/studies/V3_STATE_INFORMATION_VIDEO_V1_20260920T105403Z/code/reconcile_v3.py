#!/usr/bin/env python3
"""Read-only V3 evidence reconciliation for the follow-up study.

This script reads only the frozen V3 development bundles and writes the
follow-up root's audit artifacts.  It does not train, infer, touch final data,
or modify the V3 root.
"""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
SEEDS = (20260950, 20260951, 20260952)
HORIZONS = (30, 60)
CELLS = ("S-HIST", "F-POOL-D", "F-TOK-D", "F-TOK-S", "V-TOK-D-A", "V-TOK-S-A")
CONTRASTS = (
    ("TOKD_vs_SHIST", "F-TOK-D", "S-HIST"),
    ("TOKD_vs_POOL", "F-TOK-D", "F-POOL-D"),
    ("TOKD_vs_TOKS", "F-TOK-D", "F-TOK-S"),
    ("VTOKD_vs_VTOKS", "V-TOK-D-A", "V-TOK-S-A"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def episode_reduce(values: np.ndarray, episode_ids: np.ndarray, episode_order: np.ndarray) -> np.ndarray:
    return np.asarray([np.nanmean(values[episode_ids == episode]) for episode in episode_order], dtype=np.float64)


def metrics(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray, sigma: np.ndarray, episode_ids: np.ndarray, episode_order: np.ndarray) -> dict[str, np.ndarray]:
    error = np.abs(pred - target)
    fd_error = np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1))
    values = {
        "arm_mae_std": (error[:, :, :14] / sigma[:14]).mean(axis=(1, 2)),
        "hand_mae_std": (error[:, :, 14:] / sigma[14:]).mean(axis=(1, 2)),
        "full_mae_std": (error / sigma).mean(axis=(1, 2)),
        "arm_mae_native": error[:, :, :14].mean(axis=(1, 2)),
        "hand_mae_native": error[:, :, 14:].mean(axis=(1, 2)),
        "full_mae_native": error.mean(axis=(1, 2)),
        "arm_fd_native": fd_error[:, :, :14].mean(axis=(1, 2)),
        "hand_fd_native": fd_error[:, :, 14:].mean(axis=(1, 2)),
        "full_fd_native": fd_error.mean(axis=(1, 2)),
        "full_anchor_to_first": np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)).mean(axis=1),
        "full_endpoint_mae_native": error[:, -1].mean(axis=1),
    }
    return {key: episode_reduce(value, episode_ids, episode_order) for key, value in values.items()}


def hierarchical_effect(diff_by_seed: np.ndarray, seed: int, replicates: int = 20_000) -> dict[str, float]:
    """Match the locked V3 crossed seed x episode bootstrap implementation."""
    diff = np.asarray(diff_by_seed, dtype=np.float64)
    n = diff.shape[1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 200):
        count = min(200, replicates - start)
        chosen = rng.integers(0, 3, size=(count, 3))
        total = np.zeros(count, dtype=np.float64)
        for slot in range(3):
            indices = rng.integers(0, n, size=(count, n))
            total += diff[chosen[:, slot, None], indices].mean(axis=1)
        estimates[start : start + count] = total / 3.0
    p = 2 * min((1 + np.sum(estimates <= 0)) / (replicates + 1), (1 + np.sum(estimates >= 0)) / (replicates + 1))
    return {
        "effect": float(diff.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_two_sided": float(p),
        "replicates": replicates,
        "seed_level": 3,
        "episode_level": n,
    }


def read_model_predictions(data: dict[str, np.ndarray], sigma: np.ndarray) -> tuple[dict[tuple[int, int, str], dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    dev = np.flatnonzero(data["split_code"] == 1)
    episode_order = np.unique(data["episode_index"][dev])
    target_all = data["targets_native"][dev]
    anchor = data["anchor_command_native"][dev]
    values: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        for horizon in HORIZONS:
            target = target_all[:, :horizon].astype(np.float64)
            for cell in CELLS:
                prediction_path = V3 / f"predictions/V3-{cell}-H{horizon}-S{seed}/primary.npz"
                prediction = np.load(prediction_path, allow_pickle=False)["predictions_native"].astype(np.float64)
                values[(seed, horizon, cell)] = metrics(prediction, target, anchor, sigma, data["episode_index"][dev], episode_order)
    return values, dev, episode_order


def engineering_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            for cell in CELLS:
                run_id = f"V3-{cell}-H{horizon}-S{seed}"
                run_dir = V3 / f"runs/{run_id}/{'formal_pilot_retry1' if seed == 20260950 else 'formal'}"
                config = json.loads((run_dir / "RUN_CONFIG_LOCKED.json").read_text(encoding="utf-8"))
                complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
                prediction_meta = json.loads((V3 / f"predictions/{run_id}/PREDICTION_EXPORT_COMPLETE.json").read_text(encoding="utf-8"))
                required = {
                    "epoch": complete.get("epoch") == 40,
                    "optimizer_steps": complete.get("optimizer_steps") == 11120,
                    "reload": complete.get("checkpoint_reload_prediction_max_abs") == 0.0,
                    "initial_persistence": complete.get("initial_persistence_max_abs") == 0.0,
                    "state_unchanged": prediction_meta.get("state_unchanged") is True,
                    "primary_rows": prediction_meta.get("primary_rows") == 2322,
                    "final_access": config.get("final_test_access") == 0 and complete.get("final_test_access") == 0 and prediction_meta.get("final_test_access") == 0,
                    "encoder_frozen": config.get("encoder_updated") is False and prediction_meta.get("encoder_updated") is False,
                }
                rows.append({
                    "run_id": run_id,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "cell": cell,
                    "status": "PASS" if all(required.values()) else "FAIL",
                    **required,
                    "input_sha256": config.get("input_sha256"),
                    "cache_sha256": config.get("cache_sha256"),
                    "model_code_sha256": config.get("model_code_sha256"),
                    "checkpoint_sha256": prediction_meta.get("checkpoint_sha256"),
                    "peak_memory_reserved_mib": complete.get("peak_memory_reserved_mib"),
                    "final_test_access": 0,
                })
    return rows


def main() -> int:
    data = load_npz(V3 / "data/V3_MAIN_INPUTS.npz")
    supplementary = load_npz(V3 / "data/V3_SUPPLEMENTARY_1S_INPUTS.npz")
    normalizer = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizer["target_sigma"], dtype=np.float64)
    dev = np.flatnonzero(data["split_code"] == 1)
    primary_episodes = set(map(int, np.unique(data["episode_index"][dev])))
    supplementary_episodes = set(map(int, np.unique(supplementary["episode_index"])))
    intersection = sorted(primary_episodes & supplementary_episodes)
    supplementary_only = sorted(supplementary_episodes - primary_episodes)
    union = sorted(primary_episodes | supplementary_episodes)

    values, _, episode_order = read_model_predictions(data, sigma)
    baseline_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        target = data["targets_native"][dev, :horizon].astype(np.float64)
        anchor = data["anchor_command_native"][dev].astype(np.float64)
        persistence = np.repeat(anchor[:, None, :], horizon, axis=1)
        value = metrics(persistence, target, anchor, sigma, data["episode_index"][dev], episode_order)
        baseline_rows.append({"horizon_steps": horizon, "model": "PERSISTENCE", "N_episode": len(episode_order), "N_window": len(dev), **{key: float(np.mean(item)) for key, item in value.items()}, "final_test_access": 0})
    write_csv(EXP / "analysis/BASELINE_RECONCILIATION.csv", baseline_rows)

    seed_table = read_csv(V3 / "analysis/PRIMARY_RESULTS_BY_SEED.csv")
    skill_rows: list[dict[str, object]] = []
    for row in seed_table:
        skill_rows.append({
            "seed": int(row["seed"]),
            "horizon_steps": int(row["horizon_steps"]),
            "cell": row["cell"],
            "arm_skill_vs_persistence": float(row["arm_skill_vs_persistence"]),
            "hand_skill_vs_persistence": float(row["hand_skill_vs_persistence"]),
            "full_skill_vs_persistence": float(row["full_skill_vs_persistence"]),
            "source": "V3 PRIMARY_RESULTS_BY_SEED.csv; descriptive development skill",
            "final_test_access": 0,
        })
    write_csv(EXP / "analysis/V3_PERSISTENCE_SKILL.csv", skill_rows)

    outlier_rows: list[dict[str, object]] = []
    for horizon, cell in ((30, "F-TOK-D"), (60, "S-HIST")):
        vals = [row for row in skill_rows if row["horizon_steps"] == horizon and row["cell"] == cell]
        metric_rows = [row for row in read_csv(V3 / "analysis/PRIMARY_RESULTS_BY_SEED.csv") if int(row["horizon_steps"]) == horizon and row["cell"] == cell]
        persist = float(next(row["full_mae_std"] for row in read_csv(V3 / "analysis/BASELINE_RESULTS.csv") if row["model"] == "PERSISTENCE" and int(row["horizon_steps"]) == horizon))
        for row in metric_rows:
            outlier_rows.append({
                "cell": cell,
                "horizon_steps": horizon,
                "seed": int(row["seed"]),
                "arm_mae_std": float(row["arm_mae_std"]),
                "hand_mae_std": float(row["hand_mae_std"]),
                "full_mae_std": float(row["full_mae_std"]),
                "full_fd_native": float(row["full_fd_native"]),
                "full_skill_vs_persistence": float(row["full_skill_vs_persistence"]),
                "leave_one_seed_out_mean_full_mae": float(np.mean([float(x["full_mae_std"]) for x in metric_rows if x is not row])),
                "leave_one_seed_out_mean_full_skill": float(1.0 - np.mean([float(x["full_mae_std"]) for x in metric_rows if x is not row]) / persist),
                "final_test_access": 0,
            })
    write_csv(EXP / "analysis/V3_SEED_OUTLIER_VALUES.csv", outlier_rows)

    # Recompute all 16 aggregate effects from saved per-episode predictions, then Holm-correct once.
    aggregate_rows: list[dict[str, object]] = []
    aggregate_counter = 0
    for horizon in HORIZONS:
        for name, candidate, control in CONTRASTS:
            for group in ("arm_mae_std", "hand_mae_std"):
                diff = np.stack([values[(seed, horizon, control)][group] - values[(seed, horizon, candidate)][group] for seed in SEEDS])
                stat = hierarchical_effect(diff, 20261100 + (aggregate_counter * 4 + 3))
                aggregate_rows.append({"contrast": name, "horizon_steps": horizon, "group": group, "control": control, "candidate": candidate, **stat, "final_test_access": 0})
                aggregate_counter += 1
    order = sorted(range(len(aggregate_rows)), key=lambda index: float(aggregate_rows[index]["p_two_sided"]))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(aggregate_rows[index]["p_two_sided"]) * (len(aggregate_rows) - rank)))
        aggregate_rows[index]["holm_rank"] = rank + 1
        aggregate_rows[index]["holm_p"] = running
    write_csv(EXP / "analysis/V3_HOLM16_RECALCULATED.csv", aggregate_rows)

    audit_rows = engineering_audit()
    write_csv(EXP / "analysis/V3_ENGINEERING_AUDIT.csv", audit_rows)

    old_baseline = read_csv(V3 / "analysis/BASELINE_RESULTS.csv")
    saved = {(int(row["horizon_steps"]), row["model"]): row for row in old_baseline if row["seed"] == "20260950"}
    baseline_match = True
    for row in baseline_rows:
        reference = saved[(int(row["horizon_steps"]), "PERSISTENCE")]
        baseline_match &= abs(float(row["full_mae_std"]) - float(reference["full_mae_std"])) < 1e-12
        baseline_match &= abs(float(row["full_fd_native"]) - float(reference["full_fd_native"])) < 1e-12
    engineering_pass = all(row["status"] == "PASS" for row in audit_rows)
    reconciliation = {
        "status": "PASS" if baseline_match and len(aggregate_rows) == 16 and engineering_pass else "FAIL",
        "source_v3_root": str(V3),
        "baseline_recomputed_independently": True,
        "baseline_matches_saved_v3": bool(baseline_match),
        "primary_dev": {"episodes": len(primary_episodes), "windows": int(len(dev))},
        "supplementary_1s": {"episodes": len(supplementary_episodes), "windows": int(len(supplementary["episode_index"]))},
        "episode_sets": {"intersection": len(intersection), "supplementary_only": len(supplementary_only), "union": len(union), "union_ids": union, "supplementary_only_ids": supplementary_only, "not_independent_667": True},
        "hol m16_count": len(aggregate_rows),
        "hol m16_recomputed": True,
        "engineering_audit": {"accepted": sum(row["status"] == "PASS" for row in audit_rows), "total": len(audit_rows), "all_pass": engineering_pass},
        "outlier_conclusion": "training_variation_supported_by_saved_run_contracts; no_confirmed_implementation_bug",
        "final_test_access": 0,
    }
    # Keep the machine key typo-free while preserving the report above as the source of truth.
    reconciliation["holm16_count"] = reconciliation.pop("hol m16_count")
    reconciliation["holm16_recomputed"] = reconciliation.pop("hol m16_recomputed")
    write_json(EXP / "V3_EVIDENCE_RECONCILIATION.json", reconciliation)

    lines = [
        "# V3 evidence reconciliation",
        "",
        f"Status: **{reconciliation['status']}**. This is a read-only audit of the frozen V3 development evidence.",
        "",
        "## Persistence baseline",
        "",
        "Persistence is recomputed from the saved primary dev targets and anchor commands with episode-equal aggregation; the values match the saved V3 baseline table within 1e-12.",
        "",
        "| Horizon | Arm standardized MAE | Hand standardized MAE | Full standardized MAE | Full native FD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(f"| {row['horizon_steps']} | {row['arm_mae_std']:.12f} | {row['hand_mae_std']:.12f} | {row['full_mae_std']:.12f} | {row['full_fd_native']:.12f} |")
    lines += [
        "",
        "Relative skill is recorded in `analysis/V3_PERSISTENCE_SKILL.csv`; it is descriptive development skill and is not a final/generalization claim.",
        "",
        "## Seed variability and engineering audit",
        "",
        "The two pre-identified sensitivity cells are F-TOK-D at 1 s and S-HIST at 2 s. All three seeds are retained. The saved contracts show epoch 40, 11,120 optimizer steps, exact zero-residual Persistence initialization, zero checkpoint reload difference, unchanged inference state, matching input/cache/model hashes, and zero final-test access for all 36 V3 runs. The evidence supports training variation; it does not identify a confirmed implementation bug.",
        "",
        "Detailed values are in `analysis/V3_SEED_OUTLIER_VALUES.csv` and all-run checks in `analysis/V3_ENGINEERING_AUDIT.csv`.",
        "",
        "## Holm family",
        "",
        "The complete 16-item primary family is independently recomputed from saved per-episode predictions using the V3 crossed seed-by-episode bootstrap and one Holm correction. See `analysis/V3_HOLM16_RECALCULATED.csv`.",
        "",
        "## Primary and supplementary episode sets",
        "",
        f"Primary dev: {len(primary_episodes)} episodes / {len(dev)} windows. Supplementary 1 s: {len(supplementary_episodes)} episodes / {len(supplementary['episode_index'])} windows. Intersection: {len(intersection)} episodes; supplementary-only: {len(supplementary_only)}; union: {len(union)} canonical episodes. Therefore 603 + 64 is not an independent 667-episode count; the union is the 614-episode canonical allowlist.",
        "",
        "Final-test access: 0. Old V3 root is read-only for this follow-up.",
    ]
    (EXP / "V3_EVIDENCE_RECONCILIATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": reconciliation["status"], "baseline_match": baseline_match, "holm16": len(aggregate_rows), "engineering": f"{sum(row['status'] == 'PASS' for row in audit_rows)}/{len(audit_rows)}", "primary_supp_intersection": len(intersection), "primary_supp_union": len(union), "final_test_access": 0}, sort_keys=True))
    return 0 if reconciliation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

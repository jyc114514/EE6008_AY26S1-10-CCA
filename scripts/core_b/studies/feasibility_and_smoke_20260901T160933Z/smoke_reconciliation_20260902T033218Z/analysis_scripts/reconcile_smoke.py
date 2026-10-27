"""Recompute sealed Core B smoke comparisons without modifying prior artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

LANES = [
    ("video_only", "1p0s"),
    ("state_action_history_only", "1p0s"),
    ("video_only", "0p5s"),
    ("state_action_history_only", "0p5s"),
]
HORIZONS = {"1p0s": ("1.0s", 30), "0p5s": ("0.5s", 15)}
TRAIN_COUNT = 256
DEV_COUNT = 128
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_SEED = 20260902


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_delta(delta: np.ndarray, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_COUNT, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "bootstrap_count": BOOTSTRAP_COUNT,
        "seed": seed,
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_positive_fraction": float(np.mean(samples > 0)),
        "bootstrap_negative_fraction": float(np.mean(samples < 0)),
    }


def classify(delta: dict[str, object]) -> str:
    mean = float(delta["mean_delta"])
    lower = float(delta["ci95_lower"])
    upper = float(delta["ci95_upper"])
    if lower > 0:
        return "SIGNIFICANTLY_BETTER_THAN_PERSISTENCE"
    if mean > 0:
        return "DIRECTIONALLY_BETTER_THAN_PERSISTENCE"
    if upper < 0:
        return "WORSE_THAN_PERSISTENCE"
    return "NO_CLEAR_DIFFERENCE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    smoke_root = args.smoke_root
    output_root = args.output_root
    existing_outputs = [path for path in output_root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.resolve() != Path(__file__).resolve()] if output_root.exists() else []
    if existing_outputs:
        raise RuntimeError(f"refusing to overwrite reconciliation root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "analysis_scripts").mkdir(exist_ok=True)
    source_norm_path = smoke_root.parent / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    target_norm_all = json.loads(source_norm_path.read_text(encoding="utf-8"))
    sample_manifest = json.loads((smoke_root / "NATIVE_SMOKE_SAMPLE_MANIFEST.json").read_text(encoding="utf-8"))
    if sample_manifest.get("final_test_accessed") != 0 or sample_manifest.get("task_labels_used_as_model_input") is not False:
        raise RuntimeError("smoke sample manifest boundary failed")
    rows = []
    marker_checks = []
    predictions_can_be_read = True
    for lane, horizon in LANES + [("gated_fusion", "1p0s")]:
        run_dir = smoke_root / "runs" / f"{lane}_{horizon}"
        marker_path = run_dir / "RUN_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("exit_code") != 0 or marker.get("final_test_accessed") != 0:
            predictions_can_be_read = False
            marker_checks.append({"lane": lane, "horizon": horizon, "status": "FAIL"})
            continue
        hashes_ok = all((run_dir / name).is_file() and sha256_file(run_dir / name) == expected for name, expected in marker.get("artifact_sha256", {}).items())
        config_ok = sha256_file(smoke_root / "smoke_configs" / f"{lane}_{horizon}_LOCKED.json") == marker.get("config_sha256")
        marker_checks.append({"lane": lane, "horizon": horizon, "status": "pass" if hashes_ok and config_ok else "FAIL", "artifact_hashes": hashes_ok, "config_hash": config_ok})
        if not hashes_ok or not config_ok:
            predictions_can_be_read = False
    if not predictions_can_be_read or any(item["status"] != "pass" for item in marker_checks):
        raise RuntimeError("sealed marker/hash gate failed; predictions were not read")

    for index, (lane, horizon) in enumerate(LANES + [("gated_fusion", "1p0s")]):
        run_dir = smoke_root / "runs" / f"{lane}_{horizon}"
        with np.load(run_dir / "predictions_best.npz", allow_pickle=False) as prediction_archive:
            prediction = {name: prediction_archive[name].copy() for name in prediction_archive.files}
        with np.load(smoke_root / f"NATIVE_SMOKE_INPUTS_{horizon}.npz", allow_pickle=False) as input_archive:
            inputs = {name: input_archive[name].copy() for name in input_archive.files}
        if not np.array_equal(prediction["episode_index"], inputs["episode_index"][TRAIN_COUNT:]) or not np.array_equal(prediction["split_code"], np.ones(DEV_COUNT, dtype=np.int8)):
            raise RuntimeError(f"dev ID/denominator mismatch for {lane}_{horizon}")
        horizon_name, steps = HORIZONS[horizon]
        norm = target_norm_all[horizon_name]
        mean = np.asarray(norm["mean"], dtype=np.float32)
        safe_std = np.asarray(norm["safe_std"], dtype=np.float32)
        target_std = prediction["targets_standardized"]
        pred_std = prediction["predictions_standardized"]
        target_native = target_std * safe_std + mean
        pred_native = pred_std * safe_std + mean
        persistence_native = np.repeat(inputs["state_action_history"][TRAIN_COUNT:, -28:][:, None, :], steps, axis=1)
        persistence_std = (persistence_native - mean) / safe_std
        learned_std_episode = np.abs(pred_std - target_std).mean(axis=(1, 2))
        persistence_std_episode = np.abs(persistence_std - target_std).mean(axis=(1, 2))
        learned_native_episode = np.abs(pred_native - target_native).mean(axis=(1, 2))
        persistence_native_episode = np.abs(persistence_native - target_native).mean(axis=(1, 2))
        standardized_delta = persistence_std_episode - learned_std_episode
        native_delta = persistence_native_episode - learned_native_episode
        std_bootstrap = bootstrap_delta(standardized_delta, BOOTSTRAP_SEED + index)
        native_bootstrap = bootstrap_delta(native_delta, BOOTSTRAP_SEED + 100 + index)
        stored_native = prediction["predictions_native"]
        if not np.allclose(stored_native, pred_native, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"stored native prediction mismatch for {lane}_{horizon}")
        rows.append({
            "lane": lane,
            "horizon": horizon,
            "dev_episodes": DEV_COUNT,
            "standardized_learned_mae": float(learned_std_episode.mean()),
            "standardized_persistence_mae": float(persistence_std_episode.mean()),
            "standardized_delta_persistence_minus_learned": std_bootstrap["mean_delta"],
            "standardized_ci95_lower": std_bootstrap["ci95_lower"],
            "standardized_ci95_upper": std_bootstrap["ci95_upper"],
            "standardized_classification": classify(std_bootstrap),
            "standardized_wins": int(np.sum(standardized_delta > 1e-12)),
            "standardized_losses": int(np.sum(standardized_delta < -1e-12)),
            "standardized_ties": int(np.sum(np.abs(standardized_delta) <= 1e-12)),
            "native_learned_mae": float(learned_native_episode.mean()),
            "native_persistence_mae": float(persistence_native_episode.mean()),
            "native_delta_persistence_minus_learned": native_bootstrap["mean_delta"],
            "native_ci95_lower": native_bootstrap["ci95_lower"],
            "native_ci95_upper": native_bootstrap["ci95_upper"],
            "native_classification": classify(native_bootstrap),
            "native_wins": int(np.sum(native_delta > 1e-12)),
            "native_losses": int(np.sum(native_delta < -1e-12)),
            "native_ties": int(np.sum(np.abs(native_delta) <= 1e-12)),
            "bootstrap_count": BOOTSTRAP_COUNT,
            "final_test_accessed": 0,
        })

    primary = [row for row in rows if row["horizon"] == "1p0s" and row["lane"] in {"state_action_history_only", "gated_fusion"}]
    point_gate = any(float(row["standardized_delta_persistence_minus_learned"]) > 0 for row in primary)
    strict_gate = any(row["standardized_classification"] in {"SIGNIFICANTLY_BETTER_THAN_PERSISTENCE", "DIRECTIONALLY_BETTER_THAN_PERSISTENCE"} for row in primary)
    gate = {
        "metric_reconciliation_complete": True,
        "primary_standardized_point_gate": point_gate,
        "primary_standardized_directional_gate": strict_gate,
        "full_baseline_authorized": point_gate,
        "reason": "at least one primary state/fusion lane must have lower standardized MAE than persistence" if not point_gate else "primary state/fusion standardized MAE point estimate beats persistence",
        "final_test_accessed": 0,
        "old_artifact_modified": False,
    }
    write_json(output_root / "CORE_B_SMOKE_RECONCILIATION.json", {"status": "pass", "marker_checks": marker_checks, "rows": rows, "gate": gate, "delta_definition": "persistence episode MAE minus learned episode MAE; positive means learned is lower-error", "normalization_source": str(source_norm_path), "normalization_sha256": sha256_file(source_norm_path), "final_test_accessed": 0})
    write_csv(output_root / "CORE_B_SMOKE_RECONCILIATION.csv", rows)
    write_json(output_root / "PAIRED_BOOTSTRAP_RESULTS.json", {"status": "pass", "bootstrap_count": BOOTSTRAP_COUNT, "rows": rows, "resampling_unit": "episode", "final_test_accessed": 0})
    write_csv(output_root / "PAIRED_BOOTSTRAP_RESULTS.csv", rows)
    (output_root / "OLD_CONTRADICTORY_STATEMENT_CORRECTION.md").write_text(
        "# Correction of the prior Core B smoke statement\n\n"
        "The prior sentence claiming that persistence was better than every learned lane was not safe to retain without specifying the metric space. This version independently recomputes standardized and source-native command-value MAE on the same 128 dev episodes, with the same train-only normalization and no final-test access. The two metric spaces may have different directions because standardized MAE weights dimensions/timesteps by inverse train-only scale. The canonical correction is the row-level result in `CORE_B_SMOKE_RECONCILIATION.csv`; positive delta means learned MAE is lower than persistence.\n\n"
        f"Primary standardized gate passed: `{str(point_gate).lower()}`. Full baseline authorization from this reconciliation: `{str(point_gate).lower()}`.\n",
        encoding="utf-8",
    )
    (output_root / "CORE_B_SMOKE_RECONCILIATION.md").write_text(
        "# Core B smoke reconciliation\n\n"
        "Status: `pass`. All sealed run markers and artifact hashes passed before dev predictions were read. The comparison uses the same 128 dev episodes, same denominators, train-only target normalization, and 10,000 episode bootstrap resamples per metric/lane. Delta is defined as `persistence MAE - learned MAE`; positive means learned is better.\n\n"
        + "| lane | horizon | standardized learned | standardized persistence | standardized delta | standardized 95% CI | native learned | native persistence | native delta | native 95% CI |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(f"| {row['lane']} | {row['horizon']} | {row['standardized_learned_mae']:.6f} | {row['standardized_persistence_mae']:.6f} | {row['standardized_delta_persistence_minus_learned']:.6f} | [{row['standardized_ci95_lower']:.6f}, {row['standardized_ci95_upper']:.6f}] | {row['native_learned_mae']:.6f} | {row['native_persistence_mae']:.6f} | {row['native_delta_persistence_minus_learned']:.6f} | [{row['native_ci95_lower']:.6f}, {row['native_ci95_upper']:.6f}] |" for row in rows)
        + "\n\nThe prior blanket statement is corrected: its validity depends on metric space. The current primary standardized gate is `"
        + str(point_gate).lower()
        + "`; no full baseline is started unless this gate is true. Physical units remain unresolved; native values are not labeled radians/degrees. Final-test targets accessed: `0`.\n",
        encoding="utf-8",
    )
    (output_root / "RECONCILIATION_STATUS.json").write_text(json.dumps({"status": "pass", "prediction_read_after_seal_gate": True, "full_baseline_authorized": point_gate, "final_test_accessed": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output_root)}")
    (output_root / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "runs": len(rows), "bootstrap_count": BOOTSTRAP_COUNT, "full_baseline_authorized": point_gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

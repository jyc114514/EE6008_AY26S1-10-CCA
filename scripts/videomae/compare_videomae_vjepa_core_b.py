"""CPU development-set paired comparison using the locked team scoring code."""

import argparse
import importlib.util
import json
from pathlib import Path

from evaluate_videomae_core_b import digest, require


from path_config import scratch_root
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    import numpy as np

    project, batch, output = [p.resolve() for p in (args.project_root, args.batch_root, args.output_root)]
    scratch = scratch_root(project)
    require(batch.is_relative_to(scratch) and output.is_relative_to(scratch) and output != scratch,
            "Paths are outside the configured scratch root")
    require(not output.exists() and args.bootstrap_draws >= 1000, "Fresh output and at least 1000 draws required")
    require(read(batch / "BATCH_STATUS.json")["status"] == "PASS_12_CELL_AUDIT_AND_EVALUATION", "Batch incomplete")
    task = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
    evaluator_path = task / "code/evaluate_corrected_scheduler.py"
    require(digest(evaluator_path) == "086fb3304022937f6e3e3667c7c2c73d77164c44bd65c42751b96a249e619a83", "Evaluator changed")
    spec = importlib.util.spec_from_file_location("paired_team_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data"
    inputs = v3 / "V3_MAIN_INPUTS.npz"
    require(digest(inputs) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input changed")
    with np.load(inputs, allow_pickle=False) as source:
        require(set(source["split_code"].tolist()) == {0, 1}, "Unexpected split")
        rows = np.flatnonzero(source["split_code"] == 1)
        ids, episodes, anchors = [source[k][rows].copy() for k in ("anchor_id", "episode_index", "anchor_command_native")]
        truth = source["targets_native"][rows].astype(np.float32)
        metadata = {k: source[k][rows].copy() for k in ("original_task_index", "anchor_frame", "anchor_timestamp")}
    require(len(rows) == 2322 and len(np.unique(episodes)) == 603, "Dev population mismatch")
    norm_path = v3 / "V3_NORMALIZERS.json"
    norm_hash = digest(norm_path)
    require(norm_hash == "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c", "Target scale changed")
    norm = read(norm_path)
    sigma, fd = [np.asarray(norm[k], dtype=np.float32) for k in ("target_sigma", "safe_fd_scale")]
    targets = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz"
    require(digest(targets) == "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1", "Train targets changed")
    with np.load(targets, allow_pickle=False) as source:
        threshold, _ = evaluator.activity_threshold(source["targets_native_60"])

    output.mkdir(parents=True, exist_ok=False)
    write(output / "COMPARISON_CONFIG.json", {
        "analysis_scope": "post-hoc primary-dev comparison, not confirmatory preregistration",
        "primary_family": "VideoMAE minus V-JEPA standardized MAE: H30/H60 x arm/hand (4 tests)",
        "bootstrap": "paired source-group percentile CI; fixed three-seed mean; no seed resampling",
        "draws": args.bootstrap_draws, "rng_seed": 20261008,
        "p_value": "two-sided centered paired bootstrap, add-one correction; primary-family Holm",
        "script_sha256": digest(Path(__file__)), "evaluator_sha256": digest(evaluator_path),
        "input_sha256": digest(inputs), "target_normalizer_sha256": norm_hash,
        "batch_status_sha256": digest(batch / "BATCH_STATUS.json"), "final_test_access": 0})
    metrics = ("standardized_mae", "native_future_internal_fd")
    seeds = (20261001, 20261002, 20261003)
    panels, provenance, checks, group_order = {}, [], [], None
    protocols = set()
    input_hash, evaluator_hash = digest(inputs), digest(evaluator_path)
    split_path = project / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
    split_hash = digest(split_path)
    for horizon in (30, 60):
        for seed in seeds:
            for mode, own_label, team_label in (("TS11-D", "VideoMAE", "V-JEPA"),
                                                ("STATE-CTRL", "STATE-CTRL", "TEAM-STATE-CTRL")):
                base = batch / f"{mode}_H{horizon}_S{seed}/evaluation"
                report_path = base / "METRICS.json"
                report = read(report_path)
                require(report["status"] == "PASS_POST_INFERENCE_EVALUATION" and report["final_test_access"] == 0,
                        "Invalid local evaluation")
                require((report["mode"], report["horizon"], report["seed"]) == (mode, horizon, seed), "Cell mismatch")
                require(report["input_sha256"] == input_hash and report["target_normalizer_sha256"] == norm_hash
                        and report["evaluator_sha256"] == evaluator_hash, "Scoring provenance mismatch")
                require(split_hash == report["split_manifest_sha256"], "Grouping manifest changed")
                name = f"D3EQ-CS-{mode}-H{horizon}-S{seed}"
                prediction_path = task / f"predictions/dev/{name}.npz"
                side_path = prediction_path.with_suffix(".json")
                side = read(side_path)
                require(side["status"] == "PASS" and side["final_test_access"] == 0 and side["rows"] == len(rows), "Team inference invalid")
                require((side["config"], side["horizon_steps"], side["seed"], side["split"]) == (mode, horizon, seed, "dev"), "Team cell mismatch")
                require(side["target_normalizer_sha256"] == norm_hash and digest(prediction_path) == side["prediction_sha256"], "Team hash mismatch")
                protocols.add(side["study_protocol_sha256"])
                with np.load(prediction_path, allow_pickle=False) as source:
                    team_ids = source["anchor_id"].tolist()
                    require(len(set(team_ids)) == len(ids) and set(team_ids) == set(ids.tolist()), "Team anchor set mismatch")
                    lookup = {value: i for i, value in enumerate(team_ids)}
                    order = np.asarray([lookup[value] for value in ids.tolist()])
                    require(np.array_equal(source["episode_index"][order], episodes), "Team episode mismatch")
                    for key, values in metadata.items():
                        require(np.array_equal(source[key][order], values), f"Team {key} mismatch")
                    for key, value in (("config", mode), ("seed", seed), ("split", "dev"), ("horizon_steps", horizon)):
                        require(source[key].item() == value, f"Team embedded {key} mismatch")
                    prediction = source["predictions_native"][order].copy()
                require(prediction.shape == (2322, horizon, 28) and prediction.dtype == np.float32
                        and np.isfinite(prediction).all(), "Team prediction schema invalid")
                provenance.append({"cell": name, "prediction_sha256": digest(prediction_path),
                                   "sidecar_sha256": digest(side_path), "own_metrics_sha256": digest(report_path)})
                for region, sl in evaluator.REGIONS.items():
                    window_path = base / f"dev_{region}_{own_label}_window_metrics.npz"
                    with np.load(window_path, allow_pickle=False) as source:
                        require(np.array_equal(source["anchor_id"], ids) and np.array_equal(source["episode_index"], episodes), "Own window join mismatch")
                        groups = source["source_group"].copy()
                        own = {metric: source[metric].copy() for metric in metrics}
                    require(len(np.unique(groups)) == 603, "Source-group population mismatch")
                    reducer = evaluator.GroupReducer(episodes, groups)
                    team = evaluator.window_metrics(prediction[:, :, sl], truth[:, :horizon, sl], anchors[:, sl], sigma[sl], fd[sl], threshold[sl])
                    provenance.append({"window_file": str(window_path), "sha256": digest(window_path)})
                    for metric in metrics:
                        for label, values in ((own_label, own[metric]), (team_label, team[metric])):
                            mean, by_group = reducer.mean(values, return_groups=True)
                            current_order = sorted(by_group)
                            if group_order is None:
                                group_order = current_order
                            require(current_order == group_order, "Unpaired source groups")
                            vector = np.asarray([by_group[key] for key in group_order], dtype=np.float64)
                            require(np.isfinite(vector).all(), "Nonfinite group scores")
                            if label == own_label:
                                require(abs(mean - report["panels"]["dev"][region][label][metric]) < 1e-8, "Saved metrics mismatch")
                            panels[(horizon, seed, region, metric, label)] = vector
                print(json.dumps({"scored": name}), flush=True)
    require(len(protocols) == 1, "Mixed team protocols")
    # All seeds share each resampled group index, preserving paired observations.
    rng = np.random.default_rng(20261008)
    draws = rng.integers(0, len(group_order), size=(args.bootstrap_draws, len(group_order)))
    contrasts = (("VideoMAE_minus_VJEPA", "VideoMAE", "V-JEPA"),
                 ("VideoMAE_minus_STATE", "VideoMAE", "STATE-CTRL"),
                 ("VJEPA_minus_TEAM_STATE", "V-JEPA", "TEAM-STATE-CTRL"),
                 ("STATE_reproduction_difference", "STATE-CTRL", "TEAM-STATE-CTRL"))
    results, descriptive, arrays = [], [], {"source_group": np.asarray(group_order)}
    for horizon in (30, 60):
        for region in evaluator.REGIONS:
            for metric in metrics:
                for label in ("VideoMAE", "V-JEPA", "STATE-CTRL", "TEAM-STATE-CTRL"):
                    values = np.asarray([panels[(horizon, seed, region, metric, label)].mean() for seed in seeds])
                    descriptive.append({"horizon": horizon, "region": region, "metric": metric, "model": label,
                                        "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "seed_values": values.tolist()})
                for contrast, left, right in contrasts:
                    delta = np.asarray([panels[(horizon, seed, region, metric, left)] - panels[(horizon, seed, region, metric, right)] for seed in seeds])
                    average = delta.mean(axis=0)
                    observed = float(average.mean())
                    samples = average[draws].mean(axis=1)
                    p = float((1 + np.count_nonzero(np.abs(samples - observed) >= abs(observed))) / (args.bootstrap_draws + 1))
                    primary = contrast == "VideoMAE_minus_VJEPA" and metric == "standardized_mae" and region != "action_full28"
                    results.append({"horizon": horizon, "region": region, "metric": metric, "contrast": contrast,
                                    "mean_difference": observed, "seed_differences": delta.mean(axis=1).tolist(),
                                    "ci95_conditional_fixed_seeds": np.quantile(samples, [0.025, 0.975]).tolist(),
                                    "p_centered_bootstrap": p, "primary_family": primary, "negative_favors_left": True})
                    arrays[f"H{horizon}_{region}_{metric}_{contrast}"] = delta
                    if contrast == "STATE_reproduction_difference":
                        checks.append({"horizon": horizon, "region": region, "metric": metric,
                                       "max_abs_group_score_difference": float(np.abs(delta).max()),
                                       "within_1e_6": bool(np.abs(delta).max() <= 1e-6)})
    primary = sorted([row for row in results if row["primary_family"]], key=lambda row: row["p_centered_bootstrap"])
    require(len(primary) == 4, "Primary test family mismatch")
    running = 0.0
    for rank, row in enumerate(primary):
        running = max(running, min(1.0, (len(primary) - rank) * row["p_centered_bootstrap"]))
        row["p_holm_primary_family"] = running
    np.savez_compressed(output / "PAIRED_GROUP_DIFFERENCES.npz", **arrays)
    report = {"status": "PASS_DEV_PAIRED_COMPARISON", "descriptive": descriptive, "comparisons": results,
              "state_reproduction_checks": checks, "state_reproduction_within_1e_6": all(c["within_1e_6"] for c in checks),
              "team_protocol_sha256": next(iter(protocols)), "provenance": provenance,
              "paired_groups": len(group_order), "seeds": list(seeds), "gpu_used": False, "final_test_access": 0,
              "limitations": ["Post-hoc development-set analysis; not final-test evidence.",
                              "Bootstrap conditions on three fixed trained seeds, not seed-population uncertainty.",
                              "VideoMAE uses 384-dimensional features and extended 32-frame sinusoidal positions; V-JEPA uses 768 dimensions.",
                              "State controls retain the original 768-dimensional projection; compare reproduction before attribution.",
                              "Only four arm/hand MAE backbone comparisons receive Holm adjustment; other tests are exploratory."]}
    write(output / "COMPARISON.json", report)
    print(json.dumps({"status": report["status"], "paired_groups": len(group_order),
                      "state_reproduction_within_1e_6": report["state_reproduction_within_1e_6"],
                      "full28_mae": [r for r in descriptive if r["region"] == "action_full28" and r["metric"] == "standardized_mae"],
                      "report": str(output / "COMPARISON.json"), "gpu_used": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()

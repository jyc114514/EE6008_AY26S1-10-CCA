"""CPU post-inference scoring using the pinned team evaluator functions."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    import math
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", action="store_true", help="Score within a verified portable archive")
    args = parser.parse_args()
    import numpy as np
    import pyarrow.parquet as pq

    project, run, output = args.project_root.resolve(), args.run_root.resolve(), args.output_root.resolve()
    scratch = (project / 'tmp/project-team').resolve()
    if args.archive:
        require((project / "ARCHIVE.json").is_file() and run.is_relative_to(project / "replay")
                and output.is_relative_to(project / "replay"), "Archive replay paths required")
    else:
        require(run.is_relative_to(scratch) and output != scratch and output.is_relative_to(scratch), "Personal paths required")
    require(not output.exists(), "Refusing to overwrite evaluation")
    status, config = [json.loads((run / name).read_text()) for name in ("RUN_STATUS.json", "CONFIG.json")]
    require(status["status"] == "PASS_TRAIN_AND_EXPORT" and status["epochs"] == 40 and status["optimizer_steps"] == 11120, "Completed formal run required")
    require(not config["resume_check"] and status["final_test_access"] == 0, "Run scope mismatch")
    require(digest(run / "checkpoint_last.pt") == status["checkpoint_sha256"], "Checkpoint hash mismatch")
    evaluator_path = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/evaluate_corrected_scheduler.py"
    require(digest(evaluator_path) == "086fb3304022937f6e3e3667c7c2c73d77164c44bd65c42751b96a249e619a83", "Team evaluator changed")
    spec = importlib.util.spec_from_file_location("team_core_b_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
    input_path = v3 / "data/V3_MAIN_INPUTS.npz"
    require(digest(input_path) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input identity mismatch")
    with np.load(input_path, allow_pickle=False) as source:
        codes = source["split_code"]
        require(set(codes.tolist()) == {0, 1}, "Unexpected split")
        ids, episodes, anchors = [source[k].copy() for k in ("anchor_id", "episode_index", "anchor_command_native")]
        train_rows, dev_rows = [np.flatnonzero(codes == code) for code in (0, 1)]
        # This is the post-inference evaluator, never used for checkpoint selection.
        dev_truth = source["targets_native"][dev_rows].astype(np.float32)
    require(len(train_rows) == 17757 and len(dev_rows) == 2322, "Population mismatch")
    target_path = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz"
    require(digest(target_path) == config["target_cache_sha256"] == "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1", "Train target identity mismatch")
    with np.load(target_path, allow_pickle=False) as source:
        require(np.array_equal(source["anchor_id"], ids[train_rows]) and np.array_equal(source["episode_index"], episodes[train_rows]), "Train target join mismatch")
        require(np.max(np.abs(source["anchor_command_native"] - anchors[train_rows])) <= 1e-6, "Train anchor join mismatch")
        train_truth = source["targets_native_60"].astype(np.float32)
    require(train_truth.shape == (17757, 60, 28) and dev_truth.shape == (2322, 60, 28), "Truth schema mismatch")
    require(np.isfinite(train_truth).all() and np.isfinite(dev_truth).all(), "Nonfinite truth")
    normalizer_path = v3 / "data/V3_NORMALIZERS.json"
    require(digest(normalizer_path) == config["target_normalizer_sha256"], "Target normalizer identity mismatch")
    norm = json.loads(normalizer_path.read_text())
    sigma, fd = [np.asarray(norm[k], dtype=np.float32) for k in ("target_sigma", "safe_fd_scale")]
    require(sigma.shape == fd.shape == (28,) and np.isfinite(sigma).all() and np.isfinite(fd).all() and (sigma > 0).all() and (fd > 0).all(), "Invalid scales")
    threshold, fallback = evaluator.activity_threshold(train_truth)
    split_path = project / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
    table = pq.read_table(split_path, columns=["episode_index", "split", "video_sha256_fingerprint"],
                          filters=[("split", "in", ["train", "dev", "validation", "val"])])
    mapping = {}
    for row in table.to_pylist():
        ep = int(row["episode_index"])
        split = "dev" if row["split"] in {"validation", "val"} else row["split"]
        fingerprint = row["video_sha256_fingerprint"]
        require(fingerprint is not None and str(fingerprint).strip(), "Missing fingerprint")
        pair = (split, str(fingerprint))
        require(ep not in mapping or mapping[ep] == pair, "Conflicting group mapping")
        mapping[ep] = pair
    panels = {}
    group_sets = []
    details = []
    horizon = config["horizon"]
    require(horizon in {30, 60}, "Invalid horizon")
    for name, rows, truth, expected_episodes in (("train", train_rows, train_truth, 2712), ("dev", dev_rows, dev_truth, 603)):
        require(len(np.unique(episodes[rows])) == expected_episodes, "Episode count mismatch")
        require(all(mapping[int(ep)][0] == name for ep in episodes[rows]), "Split mapping mismatch")
        groups = np.asarray([mapping[int(ep)][1] for ep in episodes[rows]], dtype="U128")
        group_sets.append(set(groups.tolist()))
        reducer = evaluator.GroupReducer(episodes[rows], groups)
        path = run / f"predictions_{name}.npz"
        require(digest(path) == status[f"{name}_prediction_sha256"], "Prediction hash mismatch")
        with np.load(path, allow_pickle=False) as source:
            require(np.array_equal(source["anchor_id"], ids[rows]) and np.array_equal(source["episode_index"], episodes[rows]), "Prediction alignment mismatch")
            prediction = source["predictions_native"].copy()
        require(prediction.shape == (len(rows), horizon, 28) and prediction.dtype == np.float32 and np.isfinite(prediction).all(), "Prediction schema mismatch")
        persistence = np.repeat(anchors[rows, None, :], horizon, axis=1).astype(np.float32)
        panel = {}
        for region, sl in evaluator.REGIONS.items():
            scores = {}
            for model_name, pred in (("VideoMAE" if config["mode"] == "TS11-D" else "STATE-CTRL", prediction), ("PERSISTENCE", persistence)):
                metrics = evaluator.window_metrics(pred[:, :, sl], truth[:, :horizon, sl], anchors[rows, sl], sigma[sl], fd[sl], threshold[sl])
                scores[model_name] = evaluator.summarize_window_metrics(metrics, reducer, np.ones(len(rows), dtype=bool))
                details.append((name, region, model_name, rows, groups, metrics))
            panel[region] = scores
        panels[name] = panel
    require(not group_sets[0] & group_sets[1], "Source-group leakage")
    output.mkdir(parents=True, exist_ok=False)
    for name, region, model_name, rows, groups, metrics in details:
        np.savez_compressed(output / f"{name}_{region}_{model_name}_window_metrics.npz",
                            anchor_id=ids[rows], episode_index=episodes[rows], source_group=groups, **metrics)
    report = {"status": "PASS_POST_INFERENCE_EVALUATION", "mode": config["mode"], "horizon": horizon,
              "seed": config["seed"], "panels": panels, "aggregation": "window -> episode -> source-group equal",
              "evaluator_sha256": digest(evaluator_path), "prediction_hashes": {k: status[k] for k in ("train_prediction_sha256", "dev_prediction_sha256")},
              "input_sha256": digest(input_path), "target_normalizer_sha256": digest(normalizer_path),
              "split_manifest_sha256": digest(split_path), "activity_threshold_train_only": True,
              "activity_threshold_fallback_joints": fallback, "gpu_used": False,
              "dev_targets_loaded": True, "final_test_access": 0, "vjepa_comparison_completed": False,
              "bootstrap_completed": False, "checkpoint_selection": "fixed epoch40; no dev-based selection"}
    (output / "METRICS.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    brief = {"status": report["status"], "mode": config["mode"], "horizon": horizon, "seed": config["seed"], "dev": {}}
    for region, scores in panels["dev"].items():
        brief["dev"][region] = {model: {k: values[k] for k in ("standardized_mae", "native_future_internal_fd", "n_source_groups")} for model, values in scores.items()}
    brief.update({"metrics_file": str(output / "METRICS.json"), "gpu_used": False, "final_test_access": 0})
    print(json.dumps(clean(brief), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

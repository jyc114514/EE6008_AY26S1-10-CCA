"""Collect bounded, read-only state-control evidence without using CUDA or targets."""

import argparse
import json
from pathlib import Path
import importlib.util

from evaluate_videomae_core_b import digest, require


from path_config import scratch_root
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-replay", action="store_true", help="Replay accepted state heads on CPU only")
    args = parser.parse_args()
    project, queue, comparison, output = [p.resolve() for p in
                                         (args.project_root, args.queue_root, args.comparison, args.output)]
    scratch = scratch_root(project)
    require(all(p.is_relative_to(scratch) for p in (queue, comparison, output))
            and output != scratch and not output.exists(), "Fresh personal output required")
    require(read(queue / "QUEUE_STATUS.json")["status"] == "PASS_11_REMAINING_TRAIN_EXPORT", "Queue incomplete")
    report = read(comparison)
    require(report["status"] == "PASS_DEV_PAIRED_COMPARISON" and report["final_test_access"] == 0, "Comparison invalid")
    task = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
    sources = {}
    for name in ("d3_train_corrected.py", "corrected_scheduler.py", "d3_eq_contract.py", "d3_model.py", "d3_corrected_infer.py"):
        path = task / "code" / name
        sources[name] = {"sha256": digest(path), "source": path.read_text(encoding="utf-8")}
    require(sources["d3_model.py"]["sha256"] == "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589", "Model changed")
    protocol_path = task / "PROTOCOL_LOCKED.json"
    require(digest(protocol_path) == report["team_protocol_sha256"], "Team protocol changed since comparison")
    cells = []
    team_runs = {}
    environment_paths = list((task / "runs").glob("*/RUN_ENVIRONMENT.json"))
    environment_paths += list((task / "runs/formal").glob("*/RUN_ENVIRONMENT.json"))
    for path in environment_paths:
        environment = read(path)
        if environment.get("config") != "STATE-CTRL" or environment.get("epochs") != 40:
            continue
        key = (environment.get("horizon"), environment.get("seed"))
        if key not in {(h, s) for h in (30, 60) for s in (20261001, 20261002, 20261003)}:
            continue
        require(key not in team_runs, "Duplicate team state run; resolve manually")
        complete_path = path.parent / "RUN_COMPLETE.json"
        require(complete_path.is_file(), "Team state run incomplete")
        complete = read(complete_path)
        team_runs[key] = {"path": str(path.parent), "environment": environment, "completion": complete}
    replay_inputs = None
    if args.cpu_replay:
        import numpy as np
        import torch
        torch.set_num_threads(8)
        model_path = task / "code/d3_model.py"
        spec = importlib.util.spec_from_file_location("state_replay_model", model_path)
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data"
        input_path = v3 / "V3_MAIN_INPUTS.npz"
        require(digest(input_path) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Inputs changed")
        with np.load(input_path, allow_pickle=False) as archive:
            dev_rows = np.flatnonzero(archive["split_code"] == 1)
            replay_inputs = {k: archive[k][dev_rows].astype(np.float32) for k in
                             ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")}
            replay_ids = archive["anchor_id"][dev_rows].copy()
            train_anchors = archive["anchor_command_native"][archive["split_code"] == 0].copy()
        target_path = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz"
        with np.load(target_path, allow_pickle=False) as archive:
            anchor_error = float(np.abs(train_anchors.astype(np.float64) - archive["anchor_command_native"]).max())
        sigma = torch.tensor(read(v3 / "V3_NORMALIZERS.json")["target_sigma"], dtype=torch.float32)
    for cell in read(queue / "QUEUE.json")["cells"]:
        run = Path(cell["job"]).resolve().parent / "run"
        require(run.is_relative_to(scratch), "Run outside personal scope")
        config = read(run / "CONFIG.json")
        if config["mode"] != "STATE-CTRL":
            continue
        horizon, seed = config["horizon"], config["seed"]
        side_path = task / f"predictions/dev/D3EQ-CS-STATE-CTRL-H{horizon}-S{seed}.json"
        side = read(side_path)
        own_status = read(run / "RUN_STATUS.json")
        require(own_status["status"] == "PASS_TRAIN_AND_EXPORT" and own_status["final_test_access"] == 0
                and side["final_test_access"] == 0, "Invalid run scope")
        # Only prediction arrays are read; no target archives or checkpoints are opened.
        import numpy as np
        own_path = run / "predictions_dev.npz"
        team_path = side_path.with_suffix(".npz")
        require(digest(own_path) == own_status["dev_prediction_sha256"]
                and digest(team_path) == side["prediction_sha256"], "Prediction hash mismatch")
        with np.load(own_path, allow_pickle=False) as own, np.load(team_path, allow_pickle=False) as team:
            own_ids, team_ids = own["anchor_id"].tolist(), team["anchor_id"].tolist()
            require(len(set(team_ids)) == len(team_ids) and set(own_ids) == set(team_ids), "Anchor mismatch")
            lookup = {value: index for index, value in enumerate(team_ids)}
            order = [lookup[value] for value in own_ids]
            require(np.array_equal(own["episode_index"], team["episode_index"][order]), "Episode mismatch")
            delta = own["predictions_native"].astype(np.float64) - team["predictions_native"][order]
            require(delta.shape == (2322, horizon, 28) and np.isfinite(delta).all(), "Prediction mismatch")
            prediction_difference = {"max_abs": float(np.abs(delta).max()),
                                     "mean_abs": float(np.abs(delta).mean()),
                                     "exactly_equal": bool(np.count_nonzero(delta) == 0)}
        team_run = team_runs.get((horizon, seed))
        if team_run is not None:
            require(team_run["completion"]["checkpoint_last_sha256"] == side["checkpoint_sha256"],
                    "Team training/inference checkpoint mismatch")
        replay = None
        if args.cpu_replay:
            require(team_run is not None, "Missing team training metadata; cannot replay")
            own_checkpoint = run / "checkpoint_last.pt"
            team_checkpoint = Path(team_run["path"]) / "checkpoint_last.pt"
            require(digest(own_checkpoint) == own_status["checkpoint_sha256"]
                    and digest(team_checkpoint) == side["checkpoint_sha256"], "Checkpoint hash mismatch")
            # These are hash-verified project-generated checkpoints, never arbitrary attachments.
            own_payload = torch.load(own_checkpoint, map_location="cpu", weights_only=False)
            team_payload = torch.load(team_checkpoint, map_location="cpu", weights_only=False)
            own_state, team_state = own_payload["model"], team_payload["model"]
            require(set(own_state) == set(team_state), "Model parameter schema mismatch")
            parameter_differences = {k: float((own_state[k] - team_state[k]).abs().max()) for k in own_state}
            model = models.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode="STATE-CTRL")
            predictions = {}
            for label, state_dict in (("own", own_state), ("team", team_state)):
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(replay_ids), 64):
                        values = [torch.from_numpy(replay_inputs[k][start:start + 64]) for k in
                                  ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")]
                        zeros = torch.zeros(len(values[0]), 16, 4, 768)
                        residual = model(values[0], values[1], values[2], zeros, zeros, values[3])
                        chunks.append((values[3][:, None, :] + sigma[None, None, :] * residual).numpy())
                predictions[label] = np.concatenate(chunks)
            with np.load(own_path, allow_pickle=False) as archive:
                require(np.array_equal(archive["anchor_id"], replay_ids), "Replay ID mismatch")
                own_export = archive["predictions_native"].copy()
            with np.load(team_path, allow_pickle=False) as archive:
                team_export = archive["predictions_native"][order].copy()
            replay = {"parameter_max_abs_differences": parameter_differences,
                      "all_parameters_equal": all(v == 0 for v in parameter_differences.values()),
                      "own_cpu_vs_own_gpu_export_max_abs": float(np.abs(predictions["own"] - own_export).max()),
                      "team_cpu_vs_team_saved_max_abs": float(np.abs(predictions["team"] - team_export).max()),
                      "own_cpu_vs_team_cpu_mean_abs": float(np.abs(predictions["own"] - predictions["team"]).mean()),
                      "train_anchor_max_abs_difference": anchor_error}
        cells.append({"horizon": horizon, "seed": seed, "own_config": config,
                      "own_environment": read(run / "ENVIRONMENT.json"), "team_inference_metadata": side,
                      "team_training": team_run,
                      "cpu_replay": replay,
                      "own_run_status": own_status, "prediction_difference": prediction_difference})
    require(len(cells) == 6, "Expected six state-control cells")
    evidence = {"status": "STATE_REPRODUCTION_EVIDENCE_COLLECTED", "sources": sources,
                "team_protocol": read(protocol_path), "cells": cells,
                "group_metric_checks": report["state_reproduction_checks"],
                "comparison_sha256": digest(comparison), "gpu_used": False,
                "targets_loaded": False, "final_test_access": 0,
                "root_cause_identified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(output),
                      "cells": [{"horizon": c["horizon"], "seed": c["seed"],
                                 **c["prediction_difference"],
                                 "team_training_found": c["team_training"] is not None,
                                 "parameters_equal": c["cpu_replay"]["all_parameters_equal"] if c["cpu_replay"] else None} for c in cells],
                      "gpu_used": False, "targets_loaded": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()

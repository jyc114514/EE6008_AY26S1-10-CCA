"""Audit and score the fixed 12-cell panel; descriptive statistics, not significance."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project, queue_root, output = args.project_root.resolve(), args.queue_root.resolve(), args.output_root.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(output != scratch and output.is_relative_to(scratch), "Personal output required")
    require(not output.exists() or args.resume, "Existing output requires --resume")
    require(not args.resume or output.is_dir(), "Resume directory missing")
    queue_path = queue_root / "QUEUE.json"
    queue = json.loads(queue_path.read_text())
    require(json.loads((queue_root / "QUEUE_STATUS.json").read_text())["status"] == "PASS_11_REMAINING_TRAIN_EXPORT", "Training queue incomplete")
    scripts = Path(__file__).resolve().parent
    audit, evaluator = scripts / "audit_videomae_core_b_predictions.py", scripts / "evaluate_videomae_core_b.py"
    require(audit.is_file() and evaluator.is_file(), "Upload audit and evaluation scripts beside this script")
    identity = {"queue_sha256": digest(queue_path), "audit_sha256": digest(audit),
                "evaluator_sha256": digest(evaluator), "batch_sha256": digest(Path(__file__)),
                "project_root": str(project)}
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        require(json.loads((output / "BATCH_CONFIG.json").read_text()) == identity, "Resume configuration changed")
    else:
        write(output / "BATCH_CONFIG.json", identity)
    runs = [Path(queue["reference_job_root"]) / "run"] + [Path(c["job"]).parent / "run" for c in queue["cells"]]
    expected = {(m, h, s) for m in ("TS11-D", "STATE-CTRL") for h in (30, 60) for s in (20261001, 20261002, 20261003)}
    found, records = set(), {}
    for number, run in enumerate(runs, 1):
        require(run.resolve().is_relative_to(scratch), "Run outside personal scope")
        config = json.loads((run / "CONFIG.json").read_text())
        cell = (config["mode"], config["horizon"], config["seed"])
        require(cell in expected and cell not in found, "Unexpected/duplicate cell")
        found.add(cell)
        name = f"{cell[0]}_H{cell[1]}_S{cell[2]}"
        base = output / name
        base.mkdir(exist_ok=args.resume)
        with (base / "AUDIT.log").open("w", encoding="utf-8") as handle:
            subprocess.run([sys.executable, str(audit), "--project-root", str(project), "--run-root", str(run)],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        audit_result = json.loads((base / "AUDIT.log").read_text())
        require(audit_result["status"] == "PASS_PREDICTION_EXPORT_AUDIT", "Audit failed")
        metrics_path = base / "evaluation/METRICS.json"
        if not metrics_path.exists():
            require(not metrics_path.parent.exists(), "Partial evaluation detected; use a fresh batch directory")
            with (base / "EVALUATION.log").open("w", encoding="utf-8") as handle:
                subprocess.run([sys.executable, str(evaluator), "--project-root", str(project),
                                "--run-root", str(run), "--output-root", str(metrics_path.parent)],
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads(metrics_path.read_text())
        require(result["status"] == "PASS_POST_INFERENCE_EVALUATION" and result["final_test_access"] == 0, "Evaluation failed")
        require((result["mode"], result["horizon"], result["seed"]) == cell, "Metrics cell mismatch")
        status = json.loads((run / "RUN_STATUS.json").read_text())
        require(all(result["prediction_hashes"][key] == status[key] for key in ("train_prediction_sha256", "dev_prediction_sha256")), "Stale evaluation")
        records[cell] = result
        write(output / "BATCH_STATUS.json", {"status": "running", "completed_cells": number, "total_cells": 12, "last_cell": name})
        print(json.dumps({"completed_cells": number, "total_cells": 12, "cell": name}), flush=True)
    require(found == expected, "Incomplete matrix")
    summaries, differences = [], []
    for split in ("train", "dev"):
        for horizon in (30, 60):
            for region in ("action_0_14", "action_14_28", "action_full28"):
                for metric in ("standardized_mae", "native_future_internal_fd", "native_mae", "standardized_rmse"):
                    vectors = {}
                    for mode, label in (("TS11-D", "VideoMAE"), ("STATE-CTRL", "STATE-CTRL")):
                        values = [records[(mode, horizon, seed)]["panels"][split][region][label][metric]
                                  for seed in (20261001, 20261002, 20261003)]
                        vectors[mode] = values
                        summaries.append({"split": split, "horizon": horizon, "region": region, "metric": metric,
                                          "mode": mode, "n_seeds": 3, "mean": statistics.mean(values),
                                          "sample_sd": statistics.stdev(values), "seed_values": values})
                    delta = [d - s for d, s in zip(vectors["TS11-D"], vectors["STATE-CTRL"])]
                    differences.append({"split": split, "horizon": horizon, "region": region, "metric": metric,
                                        "contrast": "VideoMAE_dynamic_minus_STATE_CTRL", "negative_favors_dynamic": True,
                                        "mean_difference": statistics.mean(delta), "sample_sd": statistics.stdev(delta),
                                        "same_seed_differences": delta, "significance_tested": False})
    write(output / "SEED_SUMMARY.json", {"status": "PASS_12_CELL_AUDIT_AND_EVALUATION", "summaries": summaries,
                                        "paired_seed_differences": differences, "sd_definition": "sample SD, ddof=1",
                                        "vjepa_comparison_completed": False, "bootstrap_completed": False,
                                        "core_a_completed": False, "final_test_access": 0})
    write(output / "BATCH_STATUS.json", {"status": "PASS_12_CELL_AUDIT_AND_EVALUATION", "cells": 12,
                                        "summary_file": str(output / "SEED_SUMMARY.json"), "gpu_used": False,
                                        "final_test_access": 0})
    print((output / "BATCH_STATUS.json").read_text())


if __name__ == "__main__":
    main()

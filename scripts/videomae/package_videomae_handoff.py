"""Audit server-resident VideoMAE artifacts and write a non-destructive handoff index."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from evaluate_videomae_core_b import digest, require


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--core-a-root", type=Path, required=True)
    parser.add_argument("--core-b-queue", type=Path, required=True)
    parser.add_argument("--core-b-batch", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    a, queue, batch, comparison, output = [p.resolve() for p in
        (args.core_a_root, args.core_b_queue, args.core_b_batch, args.comparison, args.output_root)]
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(all(p.is_relative_to(scratch) and p != scratch for p in (a, queue, batch, comparison, output))
            and not output.exists(), "Existing personal artifacts and fresh output required")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    a_summary = read(a / "SUMMARY.json")
    require(a_summary["status"] == "PASS_CORE_A_STRICT_THREE_SEED_PROBES" and a_summary["final_test_access"] == 0, "Core A incomplete")
    require(read(batch / "BATCH_STATUS.json")["status"] == "PASS_12_CELL_AUDIT_AND_EVALUATION", "Core B evaluation incomplete")
    require(read(queue / "QUEUE_STATUS.json")["status"] == "PASS_11_REMAINING_TRAIN_EXPORT", "Core B training incomplete")
    comp = read(comparison)
    require(comp["status"] == "PASS_DEV_PAIRED_COMPARISON" and comp["final_test_access"] == 0, "Comparison incomplete")
    scripts = Path(__file__).resolve().parent
    audit_script = scripts / "audit_videomae_core_b_predictions.py"
    require(audit_script.is_file(), "Upload Core B prediction auditor alongside this script")
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit").mkdir()
    artifacts = {}

    def index(path, expected=None):
        path = path.resolve()
        require(path.is_file() and path.is_relative_to(project), "Artifact outside project or missing")
        sha = digest(path)
        require(expected is None or sha == expected, f"Artifact hash mismatch: {path}")
        artifacts[str(path)] = {"sha256": sha, "bytes": path.stat().st_size}
        return str(path)

    index(a / "SUMMARY.json")
    index(a / "CONFIG.json")
    a_config = read(a / "CONFIG.json")
    a_features = scratch / "core_a_strict_features_20261008T163351Z/features"
    index(a_features / "FEATURE_MANIFEST.json", a_config["feature_manifest_sha256"])
    for split in ("train", "validation"):
        index(a_features / f"videomae_{split}_pooled.pt", a_config["feature_sha256"][split])
    b_cache = scratch / "videomae_cache_full_20261008T074027Z"
    index(b_cache / "cache/CACHE_MANIFEST.json")
    index(b_cache / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json")
    a_cells = []
    for seed in (6008, 7, 42):
        root = a / f"seed{seed}"
        record = read(root / "METRICS.json")
        require(record["seed"] == seed and record["final_test_access"] == 0, "Core A seed mismatch")
        require(record == next(r for r in a_summary["results"] if r["seed"] == seed), "Core A stale summary")
        checkpoint = root / "checkpoint_best.pt"
        index(checkpoint, record["checkpoint_sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["seed"] == seed and payload["epoch"] == record["best_epoch"]
                and payload["config_sha256"] == digest(a / "CONFIG.json"), "Core A checkpoint binding mismatch")
        model = torch.nn.Linear(384, 120)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for split, count in (("train", 2757), ("validation", 617)):
            path = root / f"predictions_{split}.npz"
            index(path, record["prediction_hashes"][split])
            with np.load(path, allow_pickle=False) as source:
                ids, labels, logits, pred = [source[k].copy() for k in ("episode_indices", "labels", "logits", "predictions")]
            require(ids.shape == labels.shape == pred.shape == (count,) and len(np.unique(ids)) == count
                    and logits.shape == (count, 120) and np.isfinite(logits).all(), "Core A prediction schema mismatch")
            require(np.array_equal(logits.argmax(1), pred) and labels.min() >= 0 and labels.max() < 120, "Core A prediction mismatch")
            top1 = float(np.mean(pred == labels))
            require(abs(top1 - record["metrics"][split]["top1"]) < 1e-7, "Core A Top1 mismatch")
        index(root / "HISTORY.json")
        index(root / "METRICS.json")
        a_cells.append({"seed": seed, "root": str(root), "validation": record["metrics"]["validation"]})
    for metric, summary in a_summary["validation_summary"].items():
        values = [cell["validation"][metric] for cell in a_cells]
        require(abs(statistics.mean(values) - summary["mean"]) < 1e-12
                and abs(statistics.stdev(values) - summary["sample_sd"]) < 1e-12, "Core A seed reduction mismatch")
    q = read(queue / "QUEUE.json")
    runs = [Path(q["reference_job_root"]) / "run"] + [Path(c["job"]).parent / "run" for c in q["cells"]]
    expected = {(m, h, s) for m in ("TS11-D", "STATE-CTRL") for h in (30, 60) for s in (20261001, 20261002, 20261003)}
    found, b_cells = set(), []
    for number, run in enumerate(runs, 1):
        config, status = read(run / "CONFIG.json"), read(run / "RUN_STATUS.json")
        cell = config["mode"], config["horizon"], config["seed"]
        require(cell in expected and cell not in found, "Unexpected Core B cell")
        found.add(cell)
        name = f"{cell[0]}_H{cell[1]}_S{cell[2]}"
        log = output / "audit" / f"{name}.json"
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run([sys.executable, str(audit_script), "--project-root", str(project), "--run-root", str(run)],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        require(read(log)["status"] == "PASS_PREDICTION_EXPORT_AUDIT", "Core B audit failed")
        metrics_path = batch / name / "evaluation/METRICS.json"
        metrics = read(metrics_path)
        require((metrics["mode"], metrics["horizon"], metrics["seed"]) == cell
                and metrics["final_test_access"] == 0, "Core B metrics mismatch")
        require(all(metrics["prediction_hashes"][f"{split}_prediction_sha256"] == status[f"{split}_prediction_sha256"]
                    for split in ("train", "dev")), "Core B stale metrics")
        for file in ("CONFIG.json", "RUN_STATUS.json", "ENVIRONMENT.json", "checkpoint_last.pt", "predictions_train.npz", "predictions_dev.npz"):
            index(run / file)
        index(metrics_path)
        b_cells.append({"mode": cell[0], "horizon": cell[1], "seed": cell[2], "run_root": str(run), "metrics": str(metrics_path)})
        print(json.dumps({"audited_core_b_cells": number, "total": 12}), flush=True)
    require(found == expected, "Incomplete Core B matrix")
    index(comparison)
    for item in comp["provenance"]:
        if "window_file" in item:
            index(Path(item["window_file"]), item["sha256"])
        if "cell" in item:
            team_prediction = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/predictions/dev" / (item["cell"] + ".npz")
            index(team_prediction, item["prediction_sha256"])
            index(team_prediction.with_suffix(".json"), item["sidecar_sha256"])
    index(batch / "SEED_SUMMARY.json")
    index(queue / "QUEUE.json")
    code_names = ("package_videomae_handoff.py", "evaluate_videomae_core_b.py", "audit_videomae_core_b_predictions.py",
                  "train_videomae_core_a_strict_probes.py", "preflight_videomae_core_a_strict.py",
                  "train_videomae_core_b_formal.py", "extract_videomae_core_b_cache.py",
                  "fit_videomae_core_b_normalizer.py", "batch_evaluate_videomae_core_b.py",
                  "compare_videomae_vjepa_core_b.py")
    (output / "code").mkdir()
    for name in code_names:
        require((scripts / name).is_file(), f"Upload runnable source before packaging: {name}")
        shutil.copy2(scripts / name, output / "code" / name)
    caveats = ["Server-resident handoff index, not a portable copy of all large artifacts; preserve referenced directories.",
               "Core A is a frozen linear probe with validation Macro-F1 selection; no final-test score and no matched LoRA comparison.",
               "Core B is an independent rerun under matching principal settings, not bitwise reproduction.",
               "Short GPU probes show tiny repeat-run differences; full-training discrepancy is not fully explained.",
               "Backbone comparisons are post-hoc dev analyses, conditional on three fixed seeds; 384 versus 768 feature capacity differs.",
               "Both tasks still require team acceptance of comparison settings and any separately authorized final-test procedure."]
    handoff = {"status": "PASS_VIDEOMAE_HANDOFF_ARTIFACT_AUDIT", "core_a": {"cells": a_cells, "validation_summary": a_summary["validation_summary"]},
               "core_b": {"cells": b_cells, "comparison": str(comparison), "seed_summary": str(batch / "SEED_SUMMARY.json")},
               "artifacts": artifacts, "limitations": caveats, "gpu_used": False, "final_test_access": 0,
               "team_acceptance_pending": True, "benchmark_final_test_complete": False}
    write(output / "HANDOFF_INDEX.json", handoff)
    lines = ["VideoMAE Core A + Core B server-resident handoff", "", "Artifact audit passed: Core A 3 seeds; Core B 12 cells.",
             "Full paths and SHA256 hashes: HANDOFF_INDEX.json", "Runnable sources: code/ (requires the existing project environment, team modules and caches).", "", "Core A validation means:"]
    lines += [f"{metric}: {values['mean']:.6f} +/- {values['sample_sd']:.6f} (sample SD)" for metric, values in a_summary["validation_summary"].items()]
    lines += ["", "Limitations:"] + caveats
    (output / "HANDOFF_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(output / "AUDIT_STATUS.json", {"status": handoff["status"], "core_a_cells": 3, "core_b_cells": 12,
                                         "handoff_index": str(output / "HANDOFF_INDEX.json"), "final_test_access": 0})
    print((output / "AUDIT_STATUS.json").read_text())


if __name__ == "__main__":
    main()

"""Final, paired Core A comparison on the fixed 602-episode/120-task set.

This is a scoring-only program. It does not import models, tune hyperparameters,
or modify any source input. All nine predictions must pass preflight before
the frozen final truth is read. Historical baseline comparisons also include
the changed execution environment and cannot isolate input choice alone.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

SEEDS = (20260825, 20260826, 20260827)
PROGRAMS = ("A_SECOND4_CYCLE", "A_STATIC_FIRST_FRAME", "A_DYNAMIC")
METRICS = ("top1_accuracy", "top5_accuracy", "macro_f1_120")
BOOTSTRAP_SEED = 2026101901
BOOTSTRAP_DRAWS = 5000
_legacy_prefix = os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
OLD_ROOT = _legacy_prefix + "/" if _legacy_prefix else ""
V1 = "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V2 = "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
SPLIT = "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
QUALIFICATION = (
    "Same frozen 602 episodes and 120 classes; historical baseline comparison includes "
    "execution-environment differences, so any difference cannot be attributed exclusively "
    "to the input modification. Intervals condition on the nine fitted checkpoints and "
    "exclude training-seed uncertainty. This is an already-exposed same-source final partition, "
    "not a new blind or session-independent test."
)


def longpath(value):
    p = Path(value).resolve()
    return Path("\\\\?\\" + str(p)) if os.name == "nt" and not str(p).startswith("\\\\?\\") else p


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    actual = sha(path)
    check(actual == expected, f"SHA mismatch for {path}: {actual} != {expected}")
    return actual


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    check(bool(rows), "Refusing an empty result table")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def relocated(root, remote):
    check(bool(OLD_ROOT), "Set EE6008_LEGACY_SOURCE_PREFIX to the exact absolute prefix recorded in the authorized source manifests")
    check(str(remote).startswith(OLD_ROOT), "Unexpected original remote path")
    rel = str(remote)[len(OLD_ROOT):]
    check(".." not in Path(rel).parts, "Unsafe relative source path")
    return root / rel


def stable_rank(logits):
    x = np.asarray(logits, dtype=np.float64)
    check(x.ndim == 2 and np.isfinite(x).all(), "Logits must be finite rank two")
    return np.argsort(-x, axis=1, kind="stable")


def metric_vector(y_index, predicted_index, top5_correct, class_count):
    """Fixed class set with zero F1 when both truth/prediction support are zero."""
    y = np.asarray(y_index, dtype=np.int64)
    pred = np.asarray(predicted_index, dtype=np.int64)
    top5 = np.asarray(top5_correct, dtype=bool)
    check(y.ndim == 1 and len(y) and pred.shape == y.shape and top5.shape == y.shape, "Metric row dimensions differ")
    check(np.all((y >= 0) & (y < class_count)) and np.all((pred >= 0) & (pred < class_count)), "Metric label outside fixed set")
    support = np.bincount(y, minlength=class_count)
    predicted = np.bincount(pred, minlength=class_count)
    tp = np.bincount(y[y == pred], minlength=class_count)
    denominator = support + predicted
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros(class_count), where=denominator != 0)
    return np.asarray([np.mean(y == pred), np.mean(top5), np.mean(f1)], dtype=np.float64)


def metrics_and_details(y_tasks, logits, columns):
    columns = np.asarray(columns, dtype=np.int64)
    check(len(set(columns.tolist())) == len(columns), "Class map repeats a task")
    label_index = {int(t): i for i, t in enumerate(columns)}
    y = np.asarray([label_index[int(t)] for t in y_tasks], dtype=np.int64)
    rank = stable_rank(logits)
    pred = rank[:, 0]
    top5_correct = np.any(rank[:, :min(5, len(columns))] == y[:, None], axis=1)
    vector = metric_vector(y, pred, top5_correct, len(columns))
    support = np.bincount(y, minlength=len(columns))
    predicted = np.bincount(pred, minlength=len(columns))
    tp = np.bincount(y[y == pred], minlength=len(columns))
    rows = []
    for j, task in enumerate(columns):
        s, p, t = int(support[j]), int(predicted[j]), int(tp[j])
        rows.append({"task_id": int(task), "support": s, "predicted_count": p, "tp": t,
                     "fp": p - t, "fn": s - t,
                     "precision": t / p if p else 0.0, "recall": t / s if s else 0.0,
                     "f1": 2 * t / (s + p) if s + p else 0.0,
                     "top5_recall": float(top5_correct[y == j].mean()) if s else 0.0})
    return vector, rows, {"y": y, "pred": pred, "top5": top5_correct}


def bootstrap_metrics(prepared, row_draws, class_count):
    """One shared episode draw is applied to every model/seed cell."""
    draws = np.asarray(row_draws, dtype=np.int64)
    check(draws.ndim == 2 and draws.shape[1] == len(next(iter(prepared.values()))["y"]), "Invalid bootstrap draw dimensions")
    result = {key: np.empty((len(draws), len(METRICS)), dtype=np.float64) for key in prepared}
    for b, indices in enumerate(draws):
        for key, data in prepared.items():
            result[key][b] = metric_vector(data["y"][indices], data["pred"][indices], data["top5"][indices], class_count)
    return result


def verify_prediction_arrays(episodes, logits, expected_episodes, columns):
    episodes = np.asarray(episodes, dtype=np.int64)
    check(episodes.shape == (602,) and len(set(episodes.tolist())) == 602, "Prediction episode count/uniqueness differs")
    check(set(episodes.tolist()) == set(expected_episodes.tolist()), "Prediction episode population differs")
    check(np.asarray(logits).shape == (602, 120) and np.isfinite(logits).all(), "Prediction logits differ from [602,120]")
    order = np.argsort(episodes, kind="stable")
    check(np.array_equal(episodes[order], expected_episodes), "Prediction episode order cannot align")
    check(np.asarray(columns).shape == (120,) and len(set(map(int, columns))) == 120, "Invalid 120-column map")
    return np.asarray(logits)[order].astype(np.float64)


def preflight(root, new_root):
    identity_path = root / V1 / "protocol/CORE_A_FINAL_IDENTITY.csv"
    identity = sorted(read_csv(identity_path), key=lambda r: int(r["episode_id"]))
    episodes = np.asarray([int(r["episode_id"]) for r in identity], dtype="<i8")
    check(len(identity) == 602 and len(set(episodes.tolist())) == 602, "Original final identity set invalid")
    check(all(r["split"] == "final_test" for r in identity), "Non-final source identity")
    canonical_sha = hashlib.sha256("\n".join(r["canonical_id"] for r in identity).encode()).hexdigest()
    episode_sha = hashlib.sha256(episodes.tobytes()).hexdigest()
    map_path = root / V2 / "core_a/CORRECTED_CLASS_MAP_V2.json"
    class_map = read_json(map_path)
    classes = sorted(class_map["classes"], key=lambda r: int(r["column_index"]))
    check([int(r["column_index"]) for r in classes] == list(range(120)), "Corrected class map is not dense")
    columns = np.asarray([int(r["canonical_task_id"]) for r in classes], dtype=np.int64)
    check(len(set(columns.tolist())) == 120, "Corrected class map repeats tasks")
    manifest_path = root / V1 / "predictions/CORE_A_PREDICTION_MANIFEST.csv"
    seal = read_json(root / V1 / "predictions/CORE_A_PREDICTION_SEAL.json")
    verify(manifest_path, seal["prediction_manifest_sha256"])
    manifest = read_csv(manifest_path)
    check(len(manifest) == 6 and int(seal["prediction_count"]) == 6, "Original 2x3 prediction matrix absent")
    logits_by_cell, inputs = {}, []
    for row in manifest:
        program = {"A_DYNAMIC": "A_DYNAMIC", "A_STATIC": "A_STATIC_FIRST_FRAME"}[row["program"]]
        seed = int(row["seed"])
        check(seed in SEEDS and (program, seed) not in logits_by_cell, "Unexpected original cell")
        path, sidepath = relocated(root, row["prediction_path"]), relocated(root, row["sidecar_path"])
        verify(path, row["prediction_sha256"]); verify(sidepath, row["sidecar_sha256"])
        side = read_json(sidepath)
        check(row["canonical_ids_sha256"] == canonical_sha, "Original canonical order SHA mismatch")
        check(side["episode_index_sha256"] == episode_sha, "Original episode order SHA mismatch")
        check(side["checkpoint_sha256"] == row["checkpoint_sha256"], "Original checkpoint identity mismatch")
        with np.load(str(path), allow_pickle=False) as z:
            check(set(z.files) == {"episode_index", "logits"}, "Unexpected original prediction members")
            original_e, original_l = z["episode_index"].copy(), z["logits"].copy()
        check(np.array_equal(original_e, episodes), "Original sealed prediction order differs")
        check(hashlib.sha256(np.ascontiguousarray(original_l, dtype="<f8").tobytes()).hexdigest() == side["logits_sha256"], "Original logits array SHA mismatch")
        logits_by_cell[program, seed] = verify_prediction_arrays(original_e, original_l, episodes, columns)
        inputs.append({"program": program, "seed": seed, "path": str(path), "sha256": sha(path), "receipt_sha256": sha(sidepath)})
    for seed in SEEDS:
        path = new_root / "predictions" / f"seed_{seed}" / "final_test.npz"
        receipt_path = path.with_suffix(".receipt.json")
        receipt = read_json(receipt_path)
        verify(path, receipt["npz_sha256"])
        check(receipt["truth_labels_used_for_prediction"] is False and int(receipt["row_count"]) == 602,
              "New inference receipt does not attest label-free 602-row predictions")
        with np.load(str(path), allow_pickle=False) as z:
            check({"episode_index", "logits", "canonical_task_ids"}.issubset(z.files), "New prediction schema missing keys")
            check(not any(k in z.files for k in ("labels", "y_true", "targets", "true_task_ids")), "New inference bundle includes truth")
            e, logits, mapping = (z[k].copy() for k in ("episode_index", "logits", "canonical_task_ids"))
            if "predicted_task_id" in z.files:
                check(np.array_equal(z["predicted_task_id"], mapping[np.argmax(logits, axis=1)]), "New saved top1 differs from stable argmax")
        check(np.array_equal(mapping, columns), "New output-column map differs from corrected training map")
        logits_by_cell["A_SECOND4_CYCLE", seed] = verify_prediction_arrays(e, logits, episodes, mapping)
        inputs.append({"program": "A_SECOND4_CYCLE", "seed": seed, "path": str(path), "sha256": sha(path), "receipt_sha256": sha(receipt_path), "checkpoint_sha256": receipt["checkpoint_sha256"]})
    check(set(logits_by_cell) == {(p, s) for p in PROGRAMS for s in SEEDS}, "Nine-cell matrix incomplete")
    return identity, episodes, classes, columns, logits_by_cell, inputs


def make_cluster_draws(root, identity, episodes):
    path = root / V1 / "protocol/CORE_A_FINAL_BOOTSTRAP_CLUSTER_MAP.csv"
    rows = read_csv(path)
    check(len(rows) == 602 and len({r["canonical_id"] for r in rows}) == 602, "Original bootstrap cluster map count/uniqueness invalid")
    by_cid = {r["canonical_id"]: r for r in rows}
    check(set(by_cid) == {r["canonical_id"] for r in identity}, "Original cluster map canonical set differs")
    groups = {}
    for i, record in enumerate(identity):
        item = by_cid[record["canonical_id"]]
        check(int(item["episode_index"]) == int(episodes[i]), "Original cluster episode identity differs")
        groups.setdefault(item["cluster_id_sha256"], []).append(i)
    check(len(groups) == 602 and all(len(v) == 1 for v in groups.values()), "Original cluster is not one distinct episode per group")
    group_order = sorted(groups)
    group_to_row = np.asarray([groups[g][0] for g in group_order], dtype=np.int64)
    generated = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 602, size=(BOOTSTRAP_DRAWS, 602), dtype=np.int16)
    draw_paths = list(root.rglob("A-FULL602_BOOTSTRAP_DRAWS.npy"))
    check(len(draw_paths) <= 1, "Multiple original draw files; supply one unambiguous source mirror")
    original_draw_sha, original_manifest_sha = None, None
    draw_source = "regenerated exactly from original RNG, dtype, and sorted cluster order"
    if draw_paths:
        original_path = draw_paths[0]
        original = np.load(str(original_path), allow_pickle=False)
        check(original.dtype == np.int16 and np.array_equal(original, generated), "Original bootstrap draws differ from exact regeneration")
        original_draw_sha = sha(original_path)
        manifest_path = root / V2 / "core_a/BOOTSTRAP_DRAW_MANIFEST.json"
        if manifest_path.is_file():
            m = read_json(manifest_path)["estimands"]["A-FULL602"]
            verify(original_path, m["sha256"])
            check(m["shape"] == [5000, 602] and int(m["seed"]) == BOOTSTRAP_SEED, "Original draw manifest contract differs")
            original_manifest_sha = sha(manifest_path)
        draw_source = "original draw values loaded and proven identical to exact regeneration"
    receipt = {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "dtype": "int16",
               "cluster_count": 602, "cluster_map_path": str(path), "cluster_map_sha256": sha(path),
               "group_order": "lexicographic original cluster_id_sha256", "source": draw_source,
               "original_draw_file_sha256": original_draw_sha, "original_manifest_sha256": original_manifest_sha,
               "same_draws_across_all_nine_cells": True,
               "interval": "95% percentile episode/source-cluster bootstrap conditional on fixed fitted models; no seed resampling"}
    return generated, group_to_row[generated], receipt


def run(args):
    check(bool(OLD_ROOT), "Set EE6008_LEGACY_SOURCE_PREFIX before reading external assets")
    root, new_root, output = longpath(args.source_root), longpath(args.new_run_root), longpath(args.output)
    identity, episodes, classes, columns, logits, input_receipts = preflight(root, new_root)
    group_draws, row_draws, draw_receipt = make_cluster_draws(root, identity, episodes)
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS_NO_FINAL_TRUTH_READ", "cells": len(logits), "rows": 602, "classes": 120, "bootstrap": draw_receipt}))
        return
    check(not output.exists(), "Use a fresh output directory; existing scoring results are not overwritten")
    output.mkdir(parents=True)
    input_record = {"status": "NINE_PREDICTION_CELLS_VERIFIED_BEFORE_TRUTH_READ", "inputs": input_receipts,
                    "new_run_root": str(new_root), "source_root": str(root), "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "SEALED_SCORING_INPUTS.json", input_record)
    # All fitted-model outputs are present and verified before this final truth read.
    import pyarrow.parquet as pq
    split_path = root / SPLIT
    truth_rows = pq.read_table(str(split_path), columns=["episode_index", "original_task_index", "split", "v3_benchmark_inclusion", "split_assignment_key"], filters=[("episode_index", "in", episodes.tolist())]).to_pylist()
    check(len(truth_rows) == 602 and len({int(r["episode_index"]) for r in truth_rows}) == 602, "Frozen final truth count/uniqueness invalid")
    truth_by_episode = {int(r["episode_index"]): r for r in truth_rows}
    check(set(truth_by_episode) == set(episodes.tolist()), "Final truth identity differs")
    for record in identity:
        item = truth_by_episode[int(record["episode_id"])]
        check(item["split"] == "final_test" and item["v3_benchmark_inclusion"] == "primary", "Truth row outside original primary final population")
        check(item["split_assignment_key"] == record["source_key"], "Truth source-group identity differs")
    truth = np.asarray([int(truth_by_episode[int(e)]["original_task_index"]) for e in episodes], dtype=np.int64)
    check(set(truth.tolist()) == set(columns.tolist()), "Final true 120 classes differ from fixed class map")
    names = {int(c["canonical_task_id"]): c["canonical_task_name"] for c in classes}
    point, per_class, prepared, per_seed = {}, [], {}, []
    for program in PROGRAMS:
        for seed in SEEDS:
            key = program, seed
            point[key], details, prepared[key] = metrics_and_details(truth, logits[key], columns)
            per_seed.append({"program": program, "seed": seed, "episodes": 602, "fixed_classes": 120,
                             **{m: float(point[key][j]) for j, m in enumerate(METRICS)}})
            per_class.extend({"program": program, "seed": seed, "task_name": names[r["task_id"]], **r} for r in details)
    boot = bootstrap_metrics(prepared, row_draws, 120)
    summary, seed_ci, contrast_seed, contrast_summary = [], [], [], []
    for program in PROGRAMS:
        points = np.stack([point[program, s] for s in SEEDS])
        boot_mean = np.stack([boot[program, s] for s in SEEDS]).mean(axis=0)
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(boot_mean[:, j], [0.025, 0.975])
            summary.append({"program": program, "metric": metric, "episodes": 602, "classes": 120, "seeds": 3,
                            "mean": float(points[:, j].mean()), "sample_sd": float(points[:, j].std(ddof=1)),
                            "mean_percent": float(100 * points[:, j].mean()), "sample_sd_percentage_points": float(100 * points[:, j].std(ddof=1)),
                            "conditional_ci95_low": float(lo), "conditional_ci95_high": float(hi),
                            "conditional_ci95_low_percent": float(lo * 100), "conditional_ci95_high_percent": float(hi * 100)})
            for seed in SEEDS:
                low_s, high_s = np.quantile(boot[program, seed][:, j], [0.025, 0.975])
                seed_ci.append({"program": program, "seed": seed, "metric": metric, "value": float(point[program, seed][j]),
                                "conditional_ci95_low": float(low_s), "conditional_ci95_high": float(high_s)})
    for baseline in ("A_STATIC_FIRST_FRAME", "A_DYNAMIC"):
        paired = np.stack([point["A_SECOND4_CYCLE", s] - point[baseline, s] for s in SEEDS])
        paired_boot = np.stack([boot["A_SECOND4_CYCLE", s] - boot[baseline, s] for s in SEEDS])
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(paired_boot[:, :, j].mean(axis=0), [0.025, 0.975])
            contrast_summary.append({"contrast": "A_SECOND4_CYCLE_minus_" + baseline, "metric": metric,
                                     "mean_delta_percentage_points": float(100 * paired[:, j].mean()),
                                     "paired_seed_sample_sd_percentage_points": float(100 * paired[:, j].std(ddof=1)),
                                     "conditional_ci95_low_percentage_points": float(100 * lo),
                                     "conditional_ci95_high_percentage_points": float(100 * hi),
                                     "direction": "positive favors fourth-second cycle", "episodes": 602, "seeds": 3})
            for k, seed in enumerate(SEEDS):
                low_s, high_s = np.quantile(paired_boot[k, :, j], [0.025, 0.975])
                contrast_seed.append({"contrast": "A_SECOND4_CYCLE_minus_" + baseline, "seed": seed, "metric": metric,
                                      "delta_percentage_points": float(100 * paired[k, j]),
                                      "conditional_ci95_low_percentage_points": float(100 * low_s),
                                      "conditional_ci95_high_percentage_points": float(100 * high_s)})
    class_summary = []
    for program in PROGRAMS:
        for task in columns:
            rows = [r for r in per_class if r["program"] == program and r["task_id"] == int(task)]
            check(len(rows) == 3 and len({r["support"] for r in rows}) == 1, "Per-class fixed support differs across seeds")
            record = {"program": program, "task_id": int(task), "task_name": names[int(task)], "support": rows[0]["support"], "seeds": 3}
            for metric in ("f1", "recall", "precision", "top5_recall"):
                values = np.asarray([r[metric] for r in rows])
                record[metric + "_mean"] = float(values.mean()); record[metric + "_sample_sd"] = float(values.std(ddof=1))
            class_summary.append(record)
    tables = {"METRICS_PER_SEED.csv": per_seed, "METRICS_MEAN_SD_CI.csv": summary,
              "METRICS_PER_SEED_CI.csv": seed_ci, "PAIRED_DELTA_PER_SEED.csv": contrast_seed,
              "PAIRED_DELTA_MEAN_CI.csv": contrast_summary, "PER_CLASS_PER_SEED.csv": per_class,
              "PER_CLASS_MEAN_SD.csv": class_summary}
    for name, rows in tables.items():
        write_csv(output / name, rows)
    np.save(str(output / "PAIRED_BOOTSTRAP_GROUP_DRAWS.npy"), group_draws, allow_pickle=False)
    if draw_receipt["original_draw_file_sha256"]:
        verify(output / "PAIRED_BOOTSTRAP_GROUP_DRAWS.npy", draw_receipt["original_draw_file_sha256"])
    missing_counts = Counter()
    for draw in row_draws:
        present = set(truth[draw].tolist())
        missing_counts.update(int(c) for c in columns if int(c) not in present)
    receipt = {"status": "PASS", "scoring_only": True, "model_or_hyperparameter_changes": 0,
               "fixed_population": {"episodes": 602, "classes": 120, "seeds": list(SEEDS)},
               "ranking_ties": "lower original output column first", "macro_f1": "fixed 120 labels, zero_division=0 in every resample",
               "split_truth_path": str(split_path), "split_truth_sha256": sha(split_path),
               "inputs_manifest_sha256": sha(output / "SEALED_SCORING_INPUTS.json"), "bootstrap": draw_receipt,
               "bootstrap_draws_sha256": sha(output / "PAIRED_BOOTSTRAP_GROUP_DRAWS.npy"),
               "bootstrap_zero_support_draw_count_by_task": {str(t): missing_counts[int(t)] for t in columns},
               "qualification": QUALIFICATION,
               "table_sha256": {name: sha(output / name) for name in tables}, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "ANALYSIS_RECEIPT.json", receipt)
    print(json.dumps({"status": "PASS", "output": str(output), "tables": len(tables), "rows": 602, "cells": 9, "bootstrap_draws": 5000}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--new-run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())

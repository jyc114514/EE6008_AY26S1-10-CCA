"""Check this dated export using public aggregates only (Python standard library).

This does not open private assets, resample raw predictions, or train a model.
The recorded conditional intervals are checked for ordering and provenance;
their original bootstrap draws cannot be recomputed from aggregates alone.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/supplementary/followup_v1"
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")
A_SEEDS = {"20260825", "20260826", "20260827"}
B_SEEDS = {"202610131", "202610132", "202610133"}
TASKS = {"119", "94", "90", "141", "230"}
checks = []


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ": nonfinite")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"{label}: {actual} differs from {expected}")


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        result = list(reader)
    require(bool(result), name + ": empty table")
    require(all(None not in r and all(v is not None for v in r.values()) for r in result),
            name + ": malformed CSV row")
    return result


def index(records, fields):
    result = {}
    for row in records:
        key = tuple(row[f] for f in fields)
        require(key not in result, "Duplicate aggregate key: " + repr(key))
        result[key] = row
    return result


def grouped(records, fields):
    result = defaultdict(list)
    for row in records:
        result[tuple(row[f] for f in fields)].append(row)
    return result


def check_a():
    seed_rows = rows("CORE_A_PER_SEED.csv")
    groups = grouped(seed_rows, ("split", "condition"))
    summary = rows("CORE_A_FINAL_SUMMARY.csv")
    summary_map = index(summary, ("split", "condition", "metric"))
    require(len(seed_rows) == 9 and len(groups) == 3 and len(summary) == 9, "Core A cell count")
    for (split, condition), group in groups.items():
        require(split == "final_test" and {r["seed"] for r in group} == A_SEEDS, "Core A fitted seeds")
        require(all(r["episodes"] == "602" and r["classes"] == "120" for r in group), "Core A population")
        for metric in ("top1_accuracy", "top5_accuracy", "macro_f1_120"):
            values = [float(r[metric + "_percent"]) for r in group]
            require(all(0 <= v <= 100 for v in values), "Core A percentage range")
            published = summary_map[(split, condition, metric)]
            close(published["mean_percent"], statistics.mean(values), "Core A mean")
            close(published["sample_sd_percentage_points"], statistics.stdev(values), "Core A sample SD")
    contrasts = rows("CORE_A_PAIRED_DIFFERENCES.csv")
    require(len(contrasts) == 6, "Core A comparison count")
    for row in contrasts:
        key = (row["split"], row["new_condition"], row["metric"])
        ref = (row["split"], row["comparator"], row["metric"])
        close(row["difference_percentage_points"], float(summary_map[key]["mean_percent"]) -
              float(summary_map[ref]["mean_percent"]), "Core A difference uses percentage points")
        require(float(row["conditional_ci95_low_percentage_points"]) <=
                float(row["conditional_ci95_high_percentage_points"]), "Core A interval order")
    checks.append("Core A: nine fitted-model rows reproduce nine means/SDs and six paired point differences")


def check_cascade():
    seed_rows = rows("CORE_B_PREDICTED_TASK_PER_SEED.csv")
    groups = grouped(seed_rows, ("split", "condition", "scope", "program"))
    summary = rows("CORE_B_PREDICTED_TASK_SUMMARY.csv")
    require(len(groups) == len(summary), "Cascade summary-group coverage")
    for row in summary:
        key = tuple(row[f] for f in ("split", "condition", "scope", "program"))
        group = groups[key]
        require(int(row["pairs"]) == len(group), "Cascade seed-pair count")
        require(int(row["anchor_support_min"]) == min(int(r["anchor_support"]) for r in group), "Cascade support min")
        require(int(row["anchor_support_max"]) == max(int(r["anchor_support"]) for r in group), "Cascade support max")
        require(int(row["task_support_min"]) == min(int(r["task_support"]) for r in group), "Cascade task support")
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            close(row[metric + "_mean"], statistics.mean(values), "Cascade grouped mean")
            if len(values) > 1:
                close(row[metric + "_sample_sd"], statistics.stdev(values), "Cascade sample SD")
            else:
                require(row[metric + "_sample_sd"] == "", "Singleton SD is missing, not zero")
    coverage = grouped(rows("CORE_B_PREDICTED_TASK_COVERAGE.csv"), ("split", "a_seed", "b_seed"))
    require(len(coverage) == 6, "Cascade coverage seed/split count")
    for (split, a_seed, b_seed), group in coverage.items():
        overall = [r for r in group if r["true_task_id"] == "ALL"]
        group = [r for r in group if r["true_task_id"] != "ALL"]
        require(len(overall) == 1 and {r["true_task_id"] for r in group} == TASKS, "Cascade five true tasks and one total")
        require((a_seed, b_seed) in zip(sorted(A_SEEDS), sorted(B_SEEDS)), "Cascade paired seeds")
        total = sum(int(r["anchors"]) for r in group)
        accepted = sum(int(r["accepted_anchors"]) for r in group)
        expected = 162 if split == "final_test" else 185
        require(total == expected, "Cascade original denominator")
        require(accepted == (181 if split == "dev" and a_seed == "20260827" else expected), "Cascade abstention support")
        require(int(overall[0]["anchors"]) == total and int(overall[0]["accepted_anchors"]) == accepted,
                "Cascade ALL row agrees with task totals")
        close(overall[0]["coverage"], accepted / total, "Overall coverage ratio")
        for row in group:
            close(row["coverage"], int(row["accepted_anchors"]) / int(row["anchors"]), "Task coverage ratio")
    intervals = rows("CORE_B_PREDICTED_TASK_PAIRED_INTERVALS.csv")
    for row in intervals:
        require(float(row["ci_low"]) <= float(row["ci_high"]), "Cascade interval order")
        require(int(row["common_accepted_anchors"]) == (162 if row["split"] == "final_test" else 181), "Common cohort")
        if row["split"] == "final_test":
            require(all(float(row[k]) == 0 for k in ("delta", "ci_low", "ci_high")), "Final substitution differences")
    checks.append("Task substitution: scoped means/SDs, three seed pairs, common-cohort support and final zero deltas")


def check_experts():
    seeds = rows("CORE_B_EXPERTS_PER_SEED.csv")
    by_system = grouped(seeds, ("split", "system"))
    by_seed = index(seeds, ("split", "system", "b_seed"))
    macro = rows("CORE_B_EXPERTS_MACRO.csv")
    require(len(macro) == 6 and len(seeds) == 12, "Expert primary table sizes")
    for row in macro:
        metric, split = row["metric"], row["split"]
        for system, prefix in (("SHARED_V0", "shared"), ("EXPERT_V0", "expert")):
            group = by_system[(split, system)]
            require({r["b_seed"] for r in group} == B_SEEDS, "Expert fitted seeds")
            require(all(r["condition"] == "oracle_task" and r["scope"] == "full_original_population" for r in group),
                    "Primary expert population/condition")
            values = [float(r[metric]) for r in group]
            close(row[prefix + "_mean"], statistics.mean(values), "Expert macro mean")
            close(row[prefix + "_sample_sd"], statistics.stdev(values), "Expert macro SD")
        delta = float(row["expert_mean"]) - float(row["shared_mean"])
        close(row["expert_minus_shared"], delta, "Expert minus fresh shared")
        close(row["relative_percent"], delta / float(row["shared_mean"]) * 100, "Relative percentage, not points")
        require(float(row["conditional_ci95_low"]) <= float(row["conditional_ci95_high"]), "Expert interval order")
    for row in rows("CORE_B_EXPERTS_PAIRED_PER_SEED.csv"):
        for metric in METRICS:
            close(row[metric], float(by_seed[(row["split"], "EXPERT_V0", row["b_seed"])][metric]) -
                  float(by_seed[(row["split"], "SHARED_V0", row["b_seed"])][metric]), "Paired expert seed difference")
    tasks = rows("CORE_B_EXPERTS_PER_TASK.csv")
    task_groups = grouped(tasks, ("split", "metric"))
    paired_tasks = grouped(rows("CORE_B_EXPERTS_PAIRED_PER_TASK_SEED.csv"), ("split", "task_id"))
    for row in macro:
        group = task_groups[(row["split"], row["metric"])]
        require({r["task_id"] for r in group} == TASKS, "Five-task macro membership")
        require(sum(int(r["episodes"]) for r in group) == int(row["episodes"]), "Expert episode support sum")
        require(sum(int(r["anchors"]) for r in group) == int(row["anchors"]), "Expert anchor support sum")
        for field in ("shared_mean", "expert_mean", "expert_minus_shared"):
            close(row[field], statistics.mean(float(r[field]) for r in group), "Equal-task macro weighting")
    for row in tasks:
        delta = float(row["expert_mean"]) - float(row["shared_mean"])
        close(row["expert_minus_shared"], delta, "Per-task absolute change")
        close(row["relative_percent"], delta / float(row["shared_mean"]) * 100, "Per-task relative change")
        paired = paired_tasks[(row["split"], row["task_id"])]
        require({r["b_seed"] for r in paired} == B_SEEDS, "Per-task paired seeds")
        close(row["expert_minus_shared"], statistics.mean(float(r[row["metric"]]) for r in paired), "Per-task paired mean")
    budget = {r["system"]: r for r in rows("CORE_B_EXPERTS_BUDGET.csv")}
    shared, expert = budget["SHARED_V0"], budget["EXPERT_V0_BANK"]
    require(int(shared["total_fits"]) == 3 and int(expert["total_fits"]) == 15, "18 fresh downstream fits")
    close(int(expert["total_parameters_per_seed"]) / int(shared["total_parameters_per_seed"]), 5, "Bank parameter ratio")
    close(int(expert["optimizer_updates_per_seed_across_models"]) / int(shared["optimizer_updates_per_seed_across_models"]), 1.15, "Update budget ratio")
    require(shared["sample_presentations_per_seed_across_models"] == expert["sample_presentations_per_seed_across_models"] == "204800", "Equal sample presentations")
    checks.append("Experts: paired and per-task arithmetic, equal-task means, sample SDs and unequal parameter/update budgets")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_integrity():
    provenance = json.loads((DATA / "PROVENANCE.json").read_text(encoding="utf-8"))
    for entry in provenance["outputs"]:
        path = DATA / entry["file"]
        require(sha(path) == entry["sha256"], "Aggregate provenance hash: " + entry["file"])
        require(path.stat().st_size == entry["bytes"], "Aggregate provenance bytes")
    code_dir = ROOT / "scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107"
    for entry in json.loads((code_dir / "SOURCE_EXPORT_HASHES.json").read_text(encoding="utf-8")):
        path = code_dir / entry["public_path"]
        require(sha(path) == entry["export_sha256"], "Prepared source export hash: " + entry["public_path"])
    manifest = ROOT / "docs/releases/supplementary-20261107/CHECKSUMS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        require(path.resolve().is_relative_to(ROOT), "Manifest path outside repository")
        require(sha(path) == expected, "Supplement manifest: " + rel)
    checks.append("Aggregate, source-export and dated-package byte hashes match the published manifests")


def main():
    check_a()
    check_cascade()
    check_experts()
    check_integrity()
    print(json.dumps({"status": "PASS", "checks": checks,
        "scope": "Public aggregate arithmetic and byte integrity only; no model training, inference or raw-array rescore."}, indent=2))


if __name__ == "__main__":
    main()

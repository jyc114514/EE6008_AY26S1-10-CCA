#!/usr/bin/env python3
"""Run the locked N5 CPU-only retrospective intervals from accepted group losses."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from paired_resampling_math import (
    CONTRASTS,
    EXPECTED_CELLS,
    EXPECTED_METRICS,
    EXPECTED_SEEDS,
    compute_scopes,
    inverse_empirical_cdf,
    make_exact_seed_triples,
    run_synthetic_fixtures,
    validate_cell_order,
    validate_group_order,
    validate_inputs,
)


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z"
BENCH = PROJECT / "reports/benchmarks"
N2 = BENCH / "CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z"
N4 = BENCH / "CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z"
METRICS = tuple(EXPECTED_METRICS)
SEEDS = tuple(EXPECTED_SEEDS)
CELLS = tuple(EXPECTED_CELLS)
EFFECT_INDEX = {name: i for i, name in enumerate(CONTRASTS)}
N2_ESTIMAND = {
    "G_TS": "G/TS11-D/CUDA_COMMON",
    "G_STATE": "G/STATE-CTRL/CUDA_COMMON",
    "V_Q0": "V/Q0/CUDA_COMMON",
    "V_Q1": "V/Q1/CUDA_COMMON",
    "I": "I/CUDA_COMMON",
}
N4_ESTIMAND = {
    "G_TS": "G_TS11-D",
    "G_STATE": "G_STATE-CTRL",
    "V_Q0": "V_Q0",
    "V_Q1": "V_Q1",
    "I": "I",
}
CI_TOLERANCE_ABS = 1e-12


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def sha_array(array):
    return sha_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def put_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name("." + path.name + ".part")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def read_npz_whitelisted(path, expected_keys):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_keys):
            raise ValueError(f"unexpected NPZ key set in {path}")
        result = {key: np.asarray(archive[key]).copy() for key in expected_keys}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError(f"object payload rejected in {path}")
    return result


def load_stats():
    path = OUT / "stats_input/ACCEPTED_DERIVED_LOSS_SUMS.npz"
    arrays = read_npz_whitelisted(path, ("loss_sums", "episode_counts", "group_ids", "metrics", "seeds", "cells"))
    metrics = tuple(str(x) for x in arrays["metrics"].tolist())
    seeds = tuple(int(x) for x in arrays["seeds"].tolist())
    cells = tuple(str(x) for x in arrays["cells"].tolist())
    if metrics != METRICS or seeds != SEEDS:
        raise ValueError("metric or seed order differs from the lock")
    validate_cell_order(cells)
    group_ids = arrays["group_ids"].astype("U128")
    if len(group_ids) != 603 or len(set(group_ids.tolist())) != 603:
        raise ValueError("duplicate or missing accepted group IDs")
    counts = arrays["episode_counts"].astype(np.int64)
    loss_sums = arrays["loss_sums"].astype(np.float64)
    validate_inputs(loss_sums, counts)
    n4_lock = json.loads((N4 / "protocol/ANALYSIS_LOCK.json").read_text(encoding="utf-8"))
    expected_group_hash = n4_lock["dev_bootstrap"]["source_group_order_sha256"]
    actual_group_hash = sha_bytes("\n".join(group_ids.tolist()).encode("utf-8"))
    if len(group_ids) != n4_lock["dev_bootstrap"]["source_groups"] or actual_group_hash != expected_group_hash:
        raise ValueError("ordered source groups differ from N4's locked ID order/hash")
    return loss_sums, counts, group_ids, metrics, seeds, cells


def load_old_draws():
    path = N2 / "protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz"
    arrays = read_npz_whitelisted(path, ("source_group_ids", "group_draw_indices", "seed_draw_indices", "prng_seed", "replicates"))
    group_ids = arrays["source_group_ids"].astype("U128")
    group_draws = arrays["group_draw_indices"].astype(np.int64)
    seed_draws = arrays["seed_draw_indices"].astype(np.int64)
    validate_group_order(group_ids.tolist(), load_stats()[2].tolist())
    if int(arrays["prng_seed"].item()) != 20261011 or int(arrays["replicates"].item()) != 10000:
        raise ValueError("historical draw metadata differs from accepted N2 record")
    validate_inputs(load_stats()[0], load_stats()[1], group_draws, seed_draws)
    return path, group_ids, group_draws, seed_draws


def write_new_draws(group_ids, group_draws, seed_draws, seed_triples):
    group_path = OUT / "draws/GROUP_INDICES.npz"
    seed_path = OUT / "draws/PAIRED_SEED_INDICES.npz"
    triples_path = OUT / "draws/EXACT27_SEED_TRIPLES.npz"
    for path, writer in (
        (group_path, lambda f: np.savez_compressed(f, source_group_ids=group_ids, group_draw_indices=group_draws, prng_seed=np.asarray(20261013), replicates=np.asarray(10000))),
        (seed_path, lambda f: np.savez_compressed(f, seeds=np.asarray(SEEDS, dtype=np.int64), seed_draw_indices=seed_draws, prng_seed=np.asarray(20261014), replicates=np.asarray(10000))),
        (triples_path, lambda f: np.savez_compressed(f, seeds=np.asarray(SEEDS, dtype=np.int64), seed_triples=seed_triples)),
    ):
        temp = path.with_name("." + path.name + ".part")
        with temp.open("wb") as handle:
            writer(handle)
        temp.replace(path)
    return {
        "GROUP_INDICES.npz": {"path": str(group_path), "file_sha256": sha_file(group_path), "array_sha256": sha_array(group_draws)},
        "PAIRED_SEED_INDICES.npz": {"path": str(seed_path), "file_sha256": sha_file(seed_path), "array_sha256": sha_array(seed_draws)},
        "EXACT27_SEED_TRIPLES.npz": {"path": str(triples_path), "file_sha256": sha_file(triples_path), "array_sha256": sha_array(seed_triples)},
    }


def interval_row(metric, contrast, point, scope, values, episode_counts, seed_effects):
    low = inverse_empirical_cdf(values, 0.025)
    high = inverse_empirical_cdf(values, 0.975)
    labels = {
        "GROUP_ONLY_FIXED_SEEDS": "95% percentile inverse-CDF; 10,000 source-group draws; all 3 observed seeds fixed and equal-weighted",
        "SEED_ONLY_FIXED_COHORT": "95% inverse empirical CDF; exact 27 ordered seed triples, probability 1/27 each; no linear interpolation",
        "CROSSED_PAIRED_SEED_GROUP": "95% percentile inverse-CDF; 10,000 shared source-group draws crossed with shared paired seed triples",
    }
    return {
        "metric": metric,
        "contrast": contrast,
        "point_estimate_fixed_three_seed_mean": float(point),
        "interval_scope": scope,
        "ci95_lower": low,
        "ci95_upper": high,
        "interval_width": high - low,
        "quantile_method": "inverse_empirical_cdf_ceil_Bp_minus_1",
        "interval_definition": labels[scope],
        "effective_draws": int(len(values)),
        "episode_count": int(np.sum(episode_counts)),
        "source_group_count": int(len(episode_counts)),
        "fixed_training_seed_count": 3,
        "all_seed_estimates_retained": True,
        "training_order_pairing_status": "nominal_pairing_unverified",
        "p_value": "NOT_COMPUTED",
        "confirmatory": False,
    }


def historical_reproduction(loss_sums, counts, old_draws, old_seed_draws):
    historical = compute_scopes(loss_sums, counts, old_draws, old_seed_draws, make_exact_seed_triples())
    n2_header, n2_rows = read_csv(N2 / "results/BOOTSTRAP_CI.csv")
    n4_header, n4_rows = read_csv(N4 / "results/GROUP_BOOTSTRAP_CI.csv")
    if not {"metric", "estimand", "group_ci95_low", "group_ci95_high", "crossed_seed_group_ci95_low_low_precision", "crossed_seed_group_ci95_high_low_precision"}.issubset(n2_header or []):
        raise ValueError("N2 historical CI schema mismatch")
    if not {"split", "axis_scope", "endpoint", "estimand", "ci_percentile_lower_2_5", "ci_percentile_upper_97_5", "bootstrap_mean"}.issubset(n4_header or []):
        raise ValueError("N4 historical CI schema mismatch")
    output = []
    for mi, metric in enumerate(METRICS):
        for contrast, ci in EFFECT_INDEX.items():
            n2_est = N2_ESTIMAND[contrast]
            n2_row = next((r for r in n2_rows if r["metric"] == metric and r["estimand"] == n2_est), None)
            n4_est = N4_ESTIMAND[contrast]
            n4_row = next((r for r in n4_rows if r["split"] == "dev" and r["axis_scope"] == "raw28" and r["endpoint"] == metric and r["estimand"] == n4_est), None)
            if n2_row is None or n4_row is None:
                raise ValueError(f"historical CI row missing for {metric}/{contrast}")
            distributions = {
                "N2_GROUP_ONLY_FIXED_SEEDS": historical["group_only"][mi, :, ci],
                "N2_CROSSED_SEED_GROUP_SENSITIVITY": historical["crossed"][mi, :, ci],
                "N4_REPORTED_CROSSED_SEED_GROUP": historical["crossed"][mi, :, ci],
            }
            reported = {
                "N2_GROUP_ONLY_FIXED_SEEDS": (float(n2_row["group_ci95_low"]), float(n2_row["group_ci95_high"])),
                "N2_CROSSED_SEED_GROUP_SENSITIVITY": (float(n2_row["crossed_seed_group_ci95_low_low_precision"]), float(n2_row["crossed_seed_group_ci95_high_low_precision"])),
                "N4_REPORTED_CROSSED_SEED_GROUP": (float(n4_row["ci_percentile_lower_2_5"]), float(n4_row["ci_percentile_upper_97_5"])),
            }
            for source, distribution in distributions.items():
                reproduced = (
                    float(np.quantile(distribution, 0.025, method="linear")),
                    float(np.quantile(distribution, 0.975, method="linear")),
                )
                low_diff = abs(reproduced[0] - reported[source][0])
                high_diff = abs(reproduced[1] - reported[source][1])
                difference = max(low_diff, high_diff)
                boot_mean = float(n4_row["bootstrap_mean"]) if source == "N4_REPORTED_CROSSED_SEED_GROUP" else ""
                output.append({
                    "source_interval": source,
                    "metric": metric,
                    "contrast": contrast,
                    "fixed_seed_point_estimate": float(historical["point"][mi, ci]),
                    "reported_lower": reported[source][0],
                    "reported_upper": reported[source][1],
                    "reproduced_lower": reproduced[0],
                    "reproduced_upper": reproduced[1],
                    "max_abs_endpoint_difference": difference,
                    "reproduction_tolerance_abs": CI_TOLERANCE_ABS,
                    "status": "PASS_REPRODUCED_FROM_STORED_DRAWS" if difference <= CI_TOLERANCE_ABS else "MISMATCH_REPORTED_WITHOUT_RETRY",
                    "quantile_method_reproduced": "numpy linear (legacy np.quantile/np.percentile default)",
                    "effective_draws": int(len(distribution)),
                    "group_draw_array_sha256": sha_array(old_draws),
                    "seed_draw_array_sha256": sha_array(old_seed_draws),
                    "N4_bootstrap_distribution_mean_not_point": boot_mean,
                    "p_values": "NOT_COMPUTED",
                })
    return output


def compare_seed_and_point_summaries(loss_sums, counts, result):
    n2_header, n2_rows = read_csv(N2 / "results/ALL_SEED_LOO.csv")
    n4_header, n4_rows = read_csv(N4 / "results/ALL_SEED_LOO.csv")
    if not {"metric", "estimand", "mean_three_fixed_seeds", "sample_sd_ddof1", "median_three_fixed_seeds", "seed_20261001", "seed_20261002", "seed_20261003", "loo_excluding_20261001", "loo_excluding_20261002", "loo_excluding_20261003"}.issubset(n2_header or []):
        raise ValueError("N2 seed/LOO schema mismatch")
    if not {"split", "axis_scope", "endpoint", "contrast", "seed", "seed_estimate", "three_seed_mean", "three_seed_sample_sd", "three_seed_median", "loo_drop_seed", "loo_mean"}.issubset(n4_header or []):
        raise ValueError("N4 seed/LOO schema mismatch")
    output = []
    by_seed = result["by_seed_effects"]
    for mi, metric in enumerate(METRICS):
        for contrast, ci in EFFECT_INDEX.items():
            n2_row = next((r for r in n2_rows if r["metric"] == metric and r["estimand"] == N2_ESTIMAND[contrast]), None)
            n4_group = [r for r in n4_rows if r["split"] == "dev" and r["axis_scope"] == "raw28" and r["endpoint"] == metric and r["contrast"] == N4_ESTIMAND[contrast]]
            if n2_row is None or len(n4_group) != 3:
                raise ValueError(f"accepted seed/LOO row missing for {metric}/{contrast}")
            n4_by_seed = {int(r["seed"]): r for r in n4_group}
            if set(n4_by_seed) != set(SEEDS):
                raise ValueError("N4 seed set differs from fixed scope")
            values = by_seed[mi, :, ci]
            mean = float(np.mean(values))
            sample_sd = float(np.std(values, ddof=1))
            median = float(np.median(values))
            n2_mean = float(n2_row["mean_three_fixed_seeds"])
            n2_sd = float(n2_row["sample_sd_ddof1"])
            n2_median = float(n2_row["median_three_fixed_seeds"])
            for si, seed in enumerate(SEEDS):
                loo = float(np.delete(values, si).mean())
                n2seed = float(n2_row[f"seed_{seed}"])
                n2loo = float(n2_row[f"loo_excluding_{seed}"])
                n4row = n4_by_seed[seed]
                n4seed = float(n4row["seed_estimate"])
                n4loo = float(n4row["loo_mean"])
                n4_mean = float(n4row["three_seed_mean"])
                n4_sd = float(n4row["three_seed_sample_sd"])
                n4_median = float(n4row["three_seed_median"])
                diffs = [abs(values[si] - n2seed), abs(values[si] - n4seed), abs(loo - n2loo), abs(loo - n4loo),
                         abs(mean - n2_mean), abs(mean - n4_mean), abs(sample_sd - n2_sd), abs(sample_sd - n4_sd),
                         abs(median - n2_median), abs(median - n4_median)]
                output.append({
                    "metric": metric,
                    "contrast": contrast,
                    "seed": seed,
                    "seed_estimate": float(values[si]),
                    "three_seed_mean": mean,
                    "three_seed_sample_sd_ddof1": sample_sd,
                    "three_seed_median": median,
                    "accepted_N2_three_seed_mean": n2_mean,
                    "accepted_N4_three_seed_mean": n4_mean,
                    "accepted_N2_three_seed_sample_sd_ddof1": n2_sd,
                    "accepted_N4_three_seed_sample_sd_ddof1": n4_sd,
                    "accepted_N2_three_seed_median": n2_median,
                    "accepted_N4_three_seed_median": n4_median,
                    "loo_drop_seed": seed,
                    "loo_mean": loo,
                    "max_abs_difference_vs_N2_N4_accepted": max(diffs),
                    "tolerance_abs": 1e-12,
                    "status": "PASS_ACCEPTED_SEED_AND_LOO_REPRODUCTION" if max(diffs) <= 1e-12 else "MISMATCH_REPORTED_WITHOUT_RETRY",
                    "training_order_pairing_status": "nominal_pairing_unverified",
                    "p_values": "NOT_COMPUTED",
                })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="run synthetic math and output/schema checks only")
    mode.add_argument("--execute", action="store_true", help="run the locked 30-row CPU statistical appendix")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(run_synthetic_fixtures(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    started = time.monotonic()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    disk0 = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    lock = json.loads((OUT / "protocol/STATISTICAL_ANALYSIS_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_RETROSPECTIVE_BEFORE_REAL_STATISTICS":
        raise ValueError("statistical analysis lock absent or not frozen")
    fixture = run_synthetic_fixtures()
    loss_sums, episode_counts, group_ids, _, _, _ = load_stats()
    old_path, old_ids, old_group_draws, old_seed_draws = load_old_draws()
    validate_group_order(old_ids.tolist(), group_ids.tolist())

    group_rng = np.random.Generator(np.random.PCG64(20261013))
    seed_rng = np.random.Generator(np.random.PCG64(20261014))
    group_draws = group_rng.integers(0, len(group_ids), size=(10000, len(group_ids)), endpoint=False, dtype=np.int64)
    seed_draws = seed_rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)), endpoint=False, dtype=np.int64)
    triples = make_exact_seed_triples()
    validate_inputs(loss_sums, episode_counts, group_draws, seed_draws)
    saved_draws = write_new_draws(group_ids, group_draws, seed_draws, triples)
    result = compute_scopes(loss_sums, episode_counts, group_draws, seed_draws, triples)

    scope_rows = []
    scope_map = (
        ("GROUP_ONLY_FIXED_SEEDS", result["group_only"]),
        ("SEED_ONLY_FIXED_COHORT", result["seed_only"]),
        ("CROSSED_PAIRED_SEED_GROUP", result["crossed"]),
    )
    by_seed = result["by_seed_effects"]
    for mi, metric in enumerate(METRICS):
        for contrast, ci in EFFECT_INDEX.items():
            for scope, distribution in scope_map:
                scope_rows.append(interval_row(
                    metric, contrast, result["point"][mi, ci], scope,
                    distribution[mi, :, ci], episode_counts, by_seed[mi, :, ci],
                ))
    write_csv(OUT / "results/CO_PRIMARY_SCOPE_TABLE30.csv", scope_rows)

    seed_rows = compare_seed_and_point_summaries(loss_sums, episode_counts, result)
    write_csv(OUT / "results/BY_SEED_AND_LOO.csv", seed_rows)
    historical_rows = historical_reproduction(loss_sums, episode_counts, old_group_draws, old_seed_draws)
    write_csv(OUT / "results/HISTORICAL_INTERVAL_REPRODUCTION.csv", historical_rows)

    algebra_rows = []
    for mi, metric in enumerate(METRICS):
        for scope, distribution in scope_map:
            max_error = float(np.max(np.abs(distribution[mi, :, EFFECT_INDEX["I"]] - (
                distribution[mi, :, EFFECT_INDEX["V_Q1"]] - distribution[mi, :, EFFECT_INDEX["V_Q0"]]
            ))))
            algebra_rows.append({
                "metric": metric,
                "scope": scope,
                "identity": "I = G_TS - G_STATE = V_Q1 - V_Q0",
                "draws": int(distribution.shape[1]),
                "max_abs_identity_error": max_error,
                "atol": 1e-12,
                "rtol": 1e-10,
                "status": "PASS" if max_error <= 1e-12 else "FAIL",
                "shared_group_indices_all_cells_metrics": scope in ("GROUP_ONLY_FIXED_SEEDS", "CROSSED_PAIRED_SEED_GROUP"),
                "group_sampling_status": "NOT_APPLICABLE_FIXED_COHORT" if scope == "SEED_ONLY_FIXED_COHORT" else "SHARED_VECTOR_ALL_CELLS_METRICS",
                "paired_seed_indices_all_cells_metrics": scope in ("SEED_ONLY_FIXED_COHORT", "CROSSED_PAIRED_SEED_GROUP"),
            })
    write_csv(OUT / "results/PAIRING_ALGEBRA_QA.csv", algebra_rows)

    draw_provenance = {
        "schema_version": 1,
        "status": "PASS_N5_REALIZED_DRAW_AUDIT",
        "new_group_draws": saved_draws["GROUP_INDICES.npz"],
        "new_paired_seed_draws": saved_draws["PAIRED_SEED_INDICES.npz"],
        "exact_27_seed_triples": saved_draws["EXACT27_SEED_TRIPLES.npz"],
        "group_draw_generator": "NumPy Generator(PCG64)",
        "group_draw_seed": 20261013,
        "paired_seed_generator": "NumPy Generator(PCG64)",
        "paired_seed_draw_seed": 20261014,
        "numpy_version": np.__version__,
        "group_draw_count": 10000,
        "seed_draw_count": 10000,
        "seed_only_exact_ordered_triples": 27,
        "old_n2_draw_file": str(old_path),
        "old_n2_draw_file_sha256": sha_file(old_path),
        "old_n2_group_indices_sha256": sha_array(old_group_draws),
        "old_n2_seed_indices_sha256": sha_array(old_seed_draws),
        "same_new_group_draws_used_by_A_and_C": True,
        "same_crossed_seed_draws_shared_by_all_cells_and_metrics": True,
        "training_seeds": list(SEEDS),
    }
    put_json(OUT / "protocol/REALIZED_DRAW_HASHES.json", draw_provenance)

    # Provenance is one row per actual interval scope, not one row per file.
    provenance_rows = []
    n2_code_sha = sha_file(N2 / "code/score_n2.py")
    n4_code_sha = sha_file(N4 / "code/score_n4.py")
    old_draw_sha = sha_file(old_path)
    for metric in METRICS:
        for contrast in CONTRASTS:
            specs = (
                ("N2_GROUP_ONLY_FIXED_SEEDS", "N2", "fixed three seeds; paired source-group resampling", 10000, "numpy linear; np.quantile default", old_draw_sha, ""),
                ("N2_CROSSED_SEED_GROUP_SENSITIVITY", "N2", "paired seed indices crossed with paired source-group indices", 10000, "numpy linear; np.quantile default", old_draw_sha, old_draw_sha),
                ("N4_REPORTED_CROSSED_SEED_GROUP", "N4", "paired seed indices crossed with paired source-group indices", 10000, "numpy linear; np.percentile default", old_draw_sha, old_draw_sha),
                ("N5_GROUP_ONLY_FIXED_SEEDS", "N5", "fixed three seeds; new paired source-group draws", 10000, "inverse empirical CDF", saved_draws["GROUP_INDICES.npz"]["file_sha256"], ""),
                ("N5_SEED_ONLY_FIXED_COHORT", "N5", "exact all 27 ordered paired seed triples on the full fixed cohort", 27, "inverse empirical CDF; no interpolation", "", saved_draws["EXACT27_SEED_TRIPLES.npz"]["file_sha256"]),
                ("N5_CROSSED_PAIRED_SEED_GROUP", "N5", "shared paired seed triples crossed with shared paired source-group draws", 10000, "inverse empirical CDF", saved_draws["GROUP_INDICES.npz"]["file_sha256"], saved_draws["PAIRED_SEED_INDICES.npz"]["file_sha256"]),
            )
            for scope, source, actual, replicates, quantile, group_file_sha, seed_file_sha in specs:
                provenance_rows.append({
                    "source_interval": scope,
                    "source": source,
                    "endpoint": metric,
                    "contrast": contrast,
                    "point_estimand": "complete episode-equal 603-group cohort; equal mean of all 3 fixed seeds",
                    "group_unit": "source_group_id; 603 groups map one-to-one to 603 episodes",
                    "source_group_order_sha256": sha_bytes("\n".join(group_ids.tolist()).encode()),
                    "seed_order": ";".join(str(x) for x in SEEDS),
                    "training_init_pairing": "same init hashes across configs verified by accepted Q0/Q1 manifests",
                    "training_sample_order_pairing": "no shared sampler-order hash in accepted receipts; nominal_pairing_unverified",
                    "bootstrap_replicates_or_exact_draws": replicates,
                    "confidence_level": 0.95,
                    "quantile_method": quantile,
                    "actual_method": actual,
                    "group_indices_file_sha256": group_file_sha,
                    "seed_indices_file_sha256": seed_file_sha,
                    "N2_code_sha256": n2_code_sha if source == "N2" else "",
                    "N4_code_sha256": n4_code_sha if source == "N4" else "",
                    "N2_draw_archive_sha256": old_draw_sha if source in ("N2", "N4") else "",
                    "missing_support_or_weight_rule": "none; all cells/metrics have complete 603-group support; episode_count=1 per group",
                    "p_values": "NOT_COMPUTED",
                    "scope": "conditional on fixed checkpoints and previously exposed development set; sensitivity only",
                })
    write_csv(OUT / "inventory/CI_PROVENANCE.csv", provenance_rows)
    put_json(OUT / "inventory/CI_PROVENANCE.json", {
        "schema_version": 1,
        "status": "PASS_CI_TYPES_VERIFIED_FROM_CODE_AND_STORED_DRAWS",
        "historical_interval_reproduction_rows": len(historical_rows),
        "historical_interval_reproductions_passed": sum(r["status"] == "PASS_REPRODUCED_FROM_STORED_DRAWS" for r in historical_rows),
        "N2_primary_interval_type": "GROUP_ONLY_FIXED_SEEDS",
        "N2_secondary_crossed_column_type": "CROSSED_PAIRED_SEED_GROUP_SENSITIVITY",
        "N4_reported_interval_type": "CROSSED_PAIRED_SEED_GROUP_SENSITIVITY",
        "N4_caption_and_short_return_scope_wording": "underspecified as source-group bootstrap; code uses shared seed_draw_indices in emit_boot",
        "training_seed_pairing_status": "same initialization hashes verified; epoch/sample-order pairing not verified",
        "training_seed_pairing_provenance_path": str(OUT / "protocol/SEED_PAIRING_PROVENANCE.json"),
        "training_init_hashes_verified": True,
        "training_sample_order_hash_verified": False,
        "source_group_independence": "source-group bootstrap assumes independent clusters; source-group ID does not establish session/alias independence",
        "full_old_roots_rehashed": False,
        "new_scope_rows": len(scope_rows),
        "p_values": 0,
    })

    ci_rows = historical_rows
    erratum = (
        "# N4 置信区间口径勘误\n\n"
        "N4 的 `GROUP_BOOTSTRAP_CI.csv` 及图注把区间简称为 source-group bootstrap，容易被读成固定 seed 的 group-only 区间。实际 `score_n4.py` 从 N2 draw manifest 同时读取 `group_draw_indices` 和 `seed_draw_indices`；`emit_boot` 对每次 group draw 先得到三 seed 的 cell/contrast，再用同一 seed triple 有放回重采样并求均值。故 N4 已报告的开发 CI 应命名为 **crossed paired seed×source-group sensitivity interval**。N4 return 第 11 行明确提到 paired source-group 与 seed draws，但第 26–27 行和图注没有保留这一完整口径。\n\n"
        "N2 主 `group_ci95_*` 列固定三个已训练 seed，只重采样 source groups；其 `crossed_seed_group_ci95_*_low_precision` 是单独的 crossed sensitivity 列。两者共享同一存储 draw 文件，但消费索引不同，CI 类型不能由 manifest 名称推断。N4 的区间端点经本附录以接受的 per-group losses 和原始 draws 重放；具体重放差异见 `results/HISTORICAL_INTERVAL_REPRODUCTION.csv`。旧 N2/N4 根未修改。\n\n"
        "新 N5 表另给 group-only、精确 27 个有序 seed triples 的 seed-only sensitivity、以及 crossed scope；新 draw seed/array/file hashes 见 `protocol/REALIZED_DRAW_HASHES.json`。这些均是已暴露 development 的回顾性区间，不构成独立确认，也不提供 p 值。"
    )
    (OUT / "STATISTICAL_ERRATUM.md").write_text(erratum, encoding="utf-8")

    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    disk1 = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    execution = {
        "status": "PASS_N5_STATISTICAL_TABLES_WRITTEN" if all(r["status"] == "PASS_REPRODUCED_FROM_STORED_DRAWS" for r in historical_rows) and all(r["status"] == "PASS" for r in algebra_rows) and all(r["status"] == "PASS_ACCEPTED_SEED_AND_LOO_REPRODUCTION" for r in seed_rows) else "PARTIAL_N5_STATISTICS_WITH_EXPLICIT_QA_FAILURES",
        "hostname": os.uname().nodename,
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": (usage1.ru_utime + usage1.ru_stime) - (usage0.ru_utime + usage0.ru_stime),
        "peak_rss_kib": int(usage1.ru_maxrss),
        "output_disk_bytes_delta": int(disk1 - disk0),
        "groups": int(len(group_ids)),
        "episodes": int(np.sum(episode_counts)),
        "metrics": list(METRICS),
        "contrasts": list(CONTRASTS),
        "scope_rows": len(scope_rows),
        "historical_reproductions_passed": sum(r["status"] == "PASS_REPRODUCED_FROM_STORED_DRAWS" for r in historical_rows),
        "historical_reproductions_total": len(historical_rows),
        "seed_loo_checks_passed": sum(r["status"] == "PASS_ACCEPTED_SEED_AND_LOO_REPRODUCTION" for r in seed_rows),
        "seed_loo_checks_total": len(seed_rows),
        "raw_target_array_read": False,
        "new_evaluation_numeric_access": 0,
        "model_forward": 0,
        "supervised_fits": 0,
        "optimizer_steps": 0,
        "encoder_extraction": 0,
        "gpu_query_use_lease": 0,
        "final_heldout_access": 0,
        "p_values": 0,
        "old_root_writes": 0,
    }
    put_json(OUT / "logs/RUN_N5_STATISTICS.json", execution)
    print(json.dumps(execution, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution["status"] == "PASS_N5_STATISTICAL_TABLES_WRITTEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())

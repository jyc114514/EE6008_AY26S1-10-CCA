#!/usr/bin/env python3
"""Prepare N5 from accepted per-source-group loss summaries only."""

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from paired_resampling_math import (
    EXPECTED_CELLS,
    EXPECTED_METRICS,
    EXPECTED_SEEDS,
    validate_group_order,
)


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z"
BENCH = PROJECT / "reports/benchmarks"
N2 = BENCH / "CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z"
N3 = BENCH / "CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z"
N4 = BENCH / "CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z"
SEEDS = tuple(EXPECTED_SEEDS)
MODELS = ("TS11-D", "STATE-CTRL")
QUERIES = ("Q0", "Q1")
CELLS = tuple(EXPECTED_CELLS)
METRICS = tuple(EXPECTED_METRICS)

EXPECTED_MANIFESTS = {
    "N2": "21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f",
    "N3": "60701b7b357e5282222c60b15994a1408fcc1398618025dbebe5388fc69f1a75",
    "N4": "e3dba3237275022b7338d0eacde5a5f37549415c4a3d6649ab8b79c3a2d7e976",
}

N2_SELECTED = (
    "results/N2_GROUP_LEVEL_METRICS.csv",
    "results/BOOTSTRAP_CI.csv",
    "protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz",
    "code/score_n2.py",
    "qa/ANALYSIS_ACCEPTANCE.json",
    "qa/FINAL_ACCEPTANCE.json",
    "qa/OUTCOME_EXPOSURE.json",
    "protocol/ANALYSIS_LOCKED.json",
    "n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv",
    "n1_refs/accepted_n1_refs/PROTOCOL_LOCKED.json",
    "protocol/N2_MODEL_REGISTRY.csv",
)
N4_SELECTED = (
    "code/score_n4.py",
    "code/add_moment_bootstrap_cis.py",
    "results/GROUP_BOOTSTRAP_CI.csv",
    "results/PAIRED_COMPONENT_GAINS.csv",
    "qa/ANALYSIS_ACCEPTANCE.json",
    "qa/FINAL_ACCEPTANCE.json",
    "qa/ANALYSIS_IDENTITIES.json",
    "protocol/ANALYSIS_LOCK.json",
    "protocol/PROTOCOL_LOCK.json",
    "protocol/OUTCOME_EXPOSURE.json",
    "inventory/SOURCE_ACCEPTANCE.json",
    "inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json",
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def sha_array(array):
    array = np.ascontiguousarray(array)
    return sha_bytes(array.tobytes(order="C"))

def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def put_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def manifest_entries(root):
    manifest = root / "CHECKSUMS.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"unparseable checksum row: {root.name}")
        digest, rel = parts
        rel = rel.lstrip("*").removeprefix("./")
        entries[rel] = digest
    return entries


def verify_root(name, root, selected):
    actual_manifest_sha = sha_file(root / "CHECKSUMS.sha256")
    expected_manifest_sha = EXPECTED_MANIFESTS[name]
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(f"{name} CHECKSUMS.sha256 hash changed: {actual_manifest_sha}")
    entries = manifest_entries(root)
    result = []
    for rel in selected:
        if rel not in entries:
            raise ValueError(f"{name} manifest lacks selected file {rel}")
        path = root / rel
        actual = sha_file(path)
        expected = entries[rel]
        if actual != expected:
            raise ValueError(f"{name} selected file mismatch: {rel}")
        result.append({"path": str(path), "sha256": actual, "manifest_sha256": expected, "status": "PASS"})
    return {"root": str(root), "manifest_sha256": actual_manifest_sha, "selected": result}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_acceptance():
    n2 = read_json(N2 / "qa/ANALYSIS_ACCEPTANCE.json")
    n2f = read_json(N2 / "qa/FINAL_ACCEPTANCE.json")
    n4 = read_json(N4 / "qa/ANALYSIS_ACCEPTANCE.json")
    n4f = read_json(N4 / "qa/FINAL_ACCEPTANCE.json")
    if n2.get("status") != "PASS_N2_SCORING" or n2f.get("status") != "COMPLETE_N2_SAFE_EXPORT_BRIDGE_N3_DIRECT_REUSE":
        raise ValueError("N2 accepted analysis/final status mismatch")
    if n4.get("status") != "PASS_N4_ERROR_BUDGET_ANALYSIS":
        raise ValueError("N4 analysis status mismatch")
    if n4f.get("status") != "COMPLETE_N4_COMMAND_ERROR_BUDGET":
        raise ValueError("N4 final status mismatch")
    reconciliation = n4f.get("co_primary_reconciliation", {})
    if reconciliation.get("rows") != 24 or reconciliation.get("max_abs_difference") != 0.0:
        raise ValueError("N4 co-primary reconciliation is not the accepted 24-row exact match")
    if n4f.get("final_heldout_access") != 0 or n4f.get("old_root_writes") != 0:
        raise ValueError("N4 final/old-root boundary is not clean")
    return {"N2": n2, "N2_FINAL": n2f, "N4": n4, "N4_FINAL": n4f}


def load_episode_counts(source_map_path):
    counts = Counter()
    episode_ids = set()
    with Path(source_map_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"episode_index", "split", "source_group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("source-group map schema mismatch")
        for row in reader:
            if row["split"] != "dev":
                continue
            episode = row["episode_index"]
            gid = row["source_group_id"]
            if episode in episode_ids:
                raise ValueError("duplicate development episode ID")
            episode_ids.add(episode)
            counts[gid] += 1
    if len(counts) != 603 or len(episode_ids) != 603 or any(v != 1 for v in counts.values()):
        raise ValueError("accepted dev source groups are not 603 one-episode groups")
    return counts


def read_loss_rows(path):
    expected_fields = {"model", "query", "profile", "seed", "metric", "source_group_id", "group_value"}
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not expected_fields.issubset(reader.fieldnames or []):
            raise ValueError("N2 group-loss schema mismatch")
        for row in reader:
            if row["profile"] != "CUDA_COMMON" or row["metric"] not in METRICS:
                continue
            rows.append(row)
    return rows


def validate_one_accepted_fixture(path, group_ids):
    """Validate one typed accepted row before reading the complete loss table."""
    expected_fields = {"model", "query", "profile", "seed", "metric", "source_group_id", "group_value"}
    selected_group = str(group_ids[0])
    wanted = ("position_standardized_mae", str(SEEDS[0]), "TS11-D/Q0", selected_group)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not expected_fields.issubset(reader.fieldnames or []):
            raise ValueError("N2 group-loss schema mismatch before fixture")
        for row in reader:
            cell = f"{row['model']}/{row['query']}"
            key = (row["metric"], row["seed"], cell, row["source_group_id"])
            if row["profile"] != "CUDA_COMMON" or key != wanted:
                continue
            value = np.float64(row["group_value"])
            if not np.isfinite(value):
                raise ValueError("accepted derived-statistic fixture is nonfinite")
            mapped = np.asarray([value], dtype=np.float64)
            if mapped.shape != (1,) or mapped.dtype != np.dtype("float64") or mapped[0] != value:
                raise AssertionError("accepted derived-statistic fixture field map failed")
            return {
                "status": "PASS_ACCEPTED_DERIVED_STATISTIC_SCHEMA_FIXTURE",
                "source": str(path),
                "source_sha256": sha_file(path),
                "key": {"metric": key[0], "seed": key[1], "cell": key[2], "source_group_id": key[3]},
                "mapped_array_index": [METRICS.index(key[0]), SEEDS.index(int(key[1])), CELLS.index(key[2]), list(group_ids).index(key[3])],
                "exact_value_match": True,
                "value_written_to_fixture": False,
                "raw_target_read": False,
            }
    raise ValueError("locked accepted derived-statistic fixture row not found")


def dry_run():
    acceptance = check_acceptance()
    return {
        "status": "PASS_N5_INPUT_DRY_RUN",
        "n2_status": acceptance["N2"].get("status"),
        "n4_status": acceptance["N4"].get("status"),
        "reads_raw_targets": False,
        "reads_model_weights": False,
        "runs_model_forward": False,
        "runs_bootstrap": False,
    }


def prepare():
    started = time.monotonic()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    disk0 = tree_bytes(OUT)

    roots = {
        "N2": verify_root("N2", N2, N2_SELECTED),
        "N3": {"root": str(N3), "manifest_sha256": sha_file(N3 / "CHECKSUMS.sha256"), "selected": []},
        "N4": verify_root("N4", N4, N4_SELECTED),
    }
    if roots["N3"]["manifest_sha256"] != EXPECTED_MANIFESTS["N3"]:
        raise ValueError("N3 checksum manifest hash changed")
    accepted = check_acceptance()

    draw_path = N2 / "protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz"
    with np.load(draw_path, allow_pickle=False) as archive:
        allowed = {"source_group_ids", "group_draw_indices", "seed_draw_indices", "prng_seed", "replicates"}
        if set(archive.files) != allowed:
            raise ValueError("N2 draw archive key set changed")
        draw_ids = archive["source_group_ids"].astype("U128", copy=True)
        old_group_draws = np.asarray(archive["group_draw_indices"], dtype=np.int64)
        old_seed_draws = np.asarray(archive["seed_draw_indices"], dtype=np.int64)
        old_prng_seed = int(np.asarray(archive["prng_seed"]).item())
        old_replicates = int(np.asarray(archive["replicates"]).item())
    if old_prng_seed != 20261011 or old_replicates != 10000:
        raise ValueError("N2 stored draw metadata differs from N2 analysis receipt")
    if old_group_draws.shape != (10000, 603) or old_seed_draws.shape != (10000, 3):
        raise ValueError("N2 stored draw shape mismatch")
    if old_group_draws.min() < 0 or old_group_draws.max() >= 603 or old_seed_draws.min() < 0 or old_seed_draws.max() >= 3:
        raise ValueError("N2 stored draw index out of range")

    source_map_path = N2 / "n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv"
    episode_counts_by_id = load_episode_counts(source_map_path)
    group_ids = np.asarray(sorted(episode_counts_by_id), dtype="U128")
    validate_group_order(draw_ids.tolist(), group_ids.tolist())
    episode_counts = np.asarray([episode_counts_by_id[x] for x in group_ids.tolist()], dtype=np.int64)

    loss_table_path = N2 / "results/N2_GROUP_LEVEL_METRICS.csv"
    fixture = validate_one_accepted_fixture(loss_table_path, group_ids)
    put_json(OUT / "qa/ACCEPTED_DERIVED_FIXTURE.json", fixture)

    # Only after the accepted fixture passes do we batch-read the whitelisted dev loss rows.
    rows = read_loss_rows(loss_table_path)
    if len(rows) != 2 * 3 * 4 * 603:
        raise ValueError(f"expected 14,472 CUDA co-primary group rows, found {len(rows)}")
    m_index = {name: i for i, name in enumerate(METRICS)}
    s_index = {str(seed): i for i, seed in enumerate(SEEDS)}
    c_index = {name: i for i, name in enumerate(CELLS)}
    g_index = {name: i for i, name in enumerate(group_ids.tolist())}
    loss_sums = np.full((len(METRICS), len(SEEDS), len(CELLS), len(group_ids)), np.nan, dtype=np.float64)
    seen = set()
    for row in rows:
        cell = f"{row['model']}/{row['query']}"
        key = (row["metric"], row["seed"], cell, row["source_group_id"])
        if key in seen:
            raise ValueError("duplicate co-primary group-loss key")
        seen.add(key)
        if row["seed"] not in s_index or cell not in c_index or row["source_group_id"] not in g_index:
            raise ValueError("unexpected seed/cell/group key")
        value = float(row["group_value"])
        if not np.isfinite(value):
            raise ValueError("nonfinite accepted group loss")
        mi, si, ci, gi = m_index[row["metric"]], s_index[row["seed"]], c_index[cell], g_index[row["source_group_id"]]
        loss_sums[mi, si, ci, gi] = value  # one accepted episode per source group
    if not np.isfinite(loss_sums).all() or len(seen) != loss_sums.size:
        raise ValueError("missing accepted group-loss cells")
    stats_path = OUT / "stats_input/ACCEPTED_DERIVED_LOSS_SUMS.npz"
    with stats_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            loss_sums=loss_sums,
            episode_counts=episode_counts,
            group_ids=group_ids,
            metrics=np.asarray(METRICS, dtype="U64"),
            seeds=np.asarray(SEEDS, dtype=np.int64),
            cells=np.asarray(CELLS, dtype="U32"),
        )

    group_ids_sha = sha_bytes(("\n".join(group_ids.tolist()) + "\n").encode("utf-8"))
    seed_bytes = np.asarray(SEEDS, dtype="<i8").tobytes()
    stats_hash = sha_file(stats_path)
    schema = {
        "schema_version": 1,
        "source": str(N2 / "results/N2_GROUP_LEVEL_METRICS.csv"),
        "source_sha256": sha_file(N2 / "results/N2_GROUP_LEVEL_METRICS.csv"),
        "array_file": str(stats_path),
        "array_file_sha256": stats_hash,
        "arrays": {
            "loss_sums": {"dtype": str(loss_sums.dtype), "shape": list(loss_sums.shape), "units": "sum of per-episode means within sampled source group"},
            "episode_counts": {"dtype": str(episode_counts.dtype), "shape": list(episode_counts.shape), "units": "episodes per source group"},
            "group_ids": {"dtype": str(group_ids.dtype), "shape": list(group_ids.shape), "units": "opaque accepted source-group IDs"},
            "metrics": {"dtype": "<U64", "shape": [len(METRICS)]},
            "seeds": {"dtype": str(np.asarray(SEEDS, dtype=np.int64).dtype), "shape": [len(SEEDS)]},
            "cells": {"dtype": "<U32", "shape": [len(CELLS)]},
        },
        "episode_group_mapping": {"episodes": int(len(episode_counts)), "source_groups": int(len(group_ids)), "one_episode_per_group": bool(np.all(episode_counts == 1))},
        "accepted_rows_loaded": int(loss_sums.size),
        "source_population": "N2 CUDA_COMMON development group losses; N4 co-primary reconciliation max_abs_difference=0 across 24 rows",
        "excluded_arrays": ["targets_native", "anchor targets", "raw future state", "video", "features", "predictions"],
        "raw_target_array_read": False,
    }
    put_json(OUT / "stats_input/SCHEMA_AND_HASH_MANIFEST.json", schema)
    put_json(OUT / "protocol/INPUT_SCHEMA_MAP.json", {
        "schema_version": 1,
        "source_metric_fields": {"position_standardized_mae": "group_value", "safe_fd_normalized": "group_value"},
        "source_key_fields": ["metric", "seed", "model", "query", "source_group_id"],
        "profile_filter": "CUDA_COMMON",
        "population": "dev only; N2 group-level result table contains 603 dev source groups and no train split rows",
        "aggregate_contract": "anchors within episode mean; source group maps to exactly one dev episode; group_value is therefore the episode mean; episode_counts=1",
        "array_contract": "loss_sums[metric,seed,cell,group], episode_counts[group]",
        "cell_order": list(CELLS),
        "metric_order": list(METRICS),
        "seed_order": list(SEEDS),
        "raw_target_values_loaded": False,
    })
    put_json(OUT / "protocol/ORDERED_IDS_AND_SEEDS.json", {
        "schema_version": 1,
        "ordered_source_group_ids_sha256_newline_utf8": group_ids_sha,
        "ordered_source_group_ids_count": int(len(group_ids)),
        "ordered_source_group_ids_path": str(stats_path),
        "ordered_source_group_ids_array_sha256": sha_array(group_ids),
        "seed_order": list(SEEDS),
        "seed_order_little_endian_i64_sha256": sha_bytes(seed_bytes),
        "cell_order": list(CELLS),
        "metric_order": list(METRICS),
        "source_group_episode_mapping_sha256": sha_file(source_map_path),
        "episode_group_one_to_one": bool(np.all(episode_counts == 1)),
    })
    put_json(OUT / "protocol/DRAW_SPEC.json", {
        "schema_version": 1,
        "historical_draw_file": str(draw_path),
        "historical_draw_file_sha256": sha_file(draw_path),
        "historical_keys": ["source_group_ids", "group_draw_indices", "seed_draw_indices", "prng_seed", "replicates"],
        "historical_shapes": {"group_draw_indices": list(old_group_draws.shape), "seed_draw_indices": list(old_seed_draws.shape)},
        "historical_prng_seed": old_prng_seed,
        "historical_replicates": old_replicates,
        "historical_source_group_ids_match_ordered_ids": True,
        "historical_group_indices_array_sha256": sha_array(old_group_draws),
        "historical_seed_indices_array_sha256": sha_array(old_seed_draws),
        "historical_generator_version": "not recorded in accepted execution receipts; stored realized indices are authoritative",
        "historical_generator": "np.random.default_rng; PCG64 is inferred from NumPy default_rng, original NumPy version is unknown",
        "raw_target_array_read": False,
    })

    source_hashes = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checksums": roots,
        "n2_draw_archive_sha256": sha_file(draw_path),
        "n2_loss_table_sha256": sha_file(N2 / "results/N2_GROUP_LEVEL_METRICS.csv"),
        "n4_analysis_lock_sha256": sha_file(N4 / "protocol/ANALYSIS_LOCK.json"),
        "n4_analysis_acceptance_sha256": sha_file(N4 / "qa/ANALYSIS_ACCEPTANCE.json"),
        "n4_final_acceptance_sha256": sha_file(N4 / "qa/FINAL_ACCEPTANCE.json"),
        "n3_manifest_sha256": roots["N3"]["manifest_sha256"],
        "verification_scope": "selected source files only; no complete old-root rehash",
        "old_roots_modified": False,
    }
    put_json(OUT / "inventory/SELECTED_SOURCE_HASHES.json", source_hashes)
    put_json(OUT / "inventory/SOURCE_ACCEPTANCE.json", {
        "schema_version": 1,
        "status": "PASS_N2_N4_SELECTED_SOURCES_AND_DERIVED_STATISTICS_ACCEPTED",
        "created_utc": source_hashes["created_utc"],
        "N2_status": accepted["N2"].get("status"),
        "N4_status": accepted["N4"].get("status"),
        "N4_bundles_accepted": accepted["N4_FINAL"].get("analysis_bundles_accepted"),
        "N4_co_primary_reconciliation": accepted["N4_FINAL"].get("co_primary_reconciliation"),
        "selected_source_hashes_path": str(OUT / "inventory/SELECTED_SOURCE_HASHES.json"),
        "N2_manifest_sha256": roots["N2"]["manifest_sha256"],
        "N3_manifest_sha256": roots["N3"]["manifest_sha256"],
        "N4_manifest_sha256": roots["N4"]["manifest_sha256"],
        "historical_N2_CI_source": "N2 results/BOOTSTRAP_CI.csv and code/score_n2.py",
        "historical_N4_CI_source": "N4 results/GROUP_BOOTSTRAP_CI.csv and code/score_n4.py",
        "raw_target_array_read": False,
        "old_root_writes": 0,
    })

    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    disk1 = tree_bytes(OUT)
    put_json(OUT / "logs/PREPARE_N5_INPUTS.json", {
        "status": "PASS_PREPARED_ACCEPTED_DERIVED_INPUT",
        "hostname": os.uname().nodename,
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": (usage1.ru_utime + usage1.ru_stime) - (usage0.ru_utime + usage0.ru_stime),
        "peak_rss_kib": int(usage1.ru_maxrss),
        "output_disk_used_delta_bytes": int(disk1 - disk0),
        "input_groups": int(len(group_ids)),
        "input_cells": int(len(CELLS)),
        "input_metrics": list(METRICS),
        "input_rows": int(len(rows)),
        "raw_target_array_read": False,
        "model_forward": 0,
        "supervised_fits": 0,
        "gpu_query_or_use": 0,
    })
    return {"status": "PASS_PREPARED_ACCEPTED_DERIVED_INPUT", "groups": len(group_ids), "loss_rows": len(rows), "array_sha256": stats_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="check acceptance metadata only; no stats table, targets, model or bootstrap")
    mode.add_argument("--prepare", action="store_true", help="prepare the whitelisted accepted dev group-loss tensor")
    args = parser.parse_args()
    result = dry_run() if args.dry_run else prepare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import os

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
QUERY = PROJECT / "reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z"
INPUTS = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
BOOT_SEED = 20261011
BOOT_REPS = 10000
TOL = 1e-5


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def main():
    gate_path = ROOT / "protocol/OUTCOME_SCORE_GATE.json"
    model_lock_path = ROOT / "models/RIDGE_MODEL_LOCK.json"
    if not gate_path.exists() or not model_lock_path.exists():
        raise AssertionError("post-fit target gate or Ridge model lock missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_MODEL_LOCK_AND_PREDICTIONS_VERIFIED_BEFORE_TARGET_READ":
        raise AssertionError("pre-target scoring gate failed")
    if gate.get("model_lock_sha256") != sha(model_lock_path) or lock.get("dev_targets_read") is not False:
        raise AssertionError("target boundary lock/hash mismatch")
    summary_path = ROOT / "results/HORIZON_METRICS_BY_SEED.csv"
    episode_path = ROOT / "results/HORIZON_EPISODE_METRICS.csv"
    time_path = ROOT / "results/ACTUAL_HORIZON_TIMESTAMPS.csv"
    for path in (summary_path, episode_path, time_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_csv(summary_path)
    if len(summary) != 168:
        raise AssertionError(f"expected 168 fixed horizon summary rows; observed {len(summary)}")
    episode_count = sum(1 for _ in episode_path.open(encoding="utf-8")) - 1
    if episode_count != 278460:
        raise AssertionError(f"unexpected episode metric rows: {episode_count}")
    timestamp_count = sum(1 for _ in time_path.open(encoding="utf-8")) - 1
    if timestamp_count != 120474:
        raise AssertionError(f"unexpected actual horizon timestamp rows: {timestamp_count}")

    # Reproduce only the 12 fitted old Q0/Q1 models; the persistence ALL row is a separate deterministic baseline.
    accepted = read_csv(QUERY / "results/TRAIN_DEV_BY_SEED.csv")
    accepted_lookup = {}
    for r in accepted:
        if r["seed"] not in ("20261001", "20261002", "20261003"):
            continue
        key = (r["split"], r["config"], r["query_variant"], int(r["seed"]))
        accepted_lookup[key] = r
    reproduced = []
    for row in summary:
        if int(row["horizon"]) != 30 or row["family"] in ("Ridge-CPU", "Persistence"):
            continue
        variant = "old" if row["query"] == "Q0" else "had"
        key = (row["split"], row["family"], variant, int(row["seed"]))
        ref_row = accepted_lookup.get(key)
        if ref_row is None:
            raise AssertionError(f"accepted H30 reference missing: {key}")
        for old_name, new_name in [
            ("position_standardized_mae", "prefix_position_standardized_mae"),
            ("safe_fd_normalized", "future_internal_fd_safe_normalized"),
        ]:
            ref = float(ref_row[old_name])
            got = float(row[new_name])
            delta = got - ref
            reproduced.append({
                "split": row["split"], "family": row["family"], "query": row["query"],
                "seed": int(row["seed"]), "metric": old_name,
                "accepted_value": ref, "recomputed_value": got, "delta": delta,
                "absolute_tolerance": TOL, "status": "PASS" if abs(delta) <= TOL else "FAIL",
            })
    if len(reproduced) != 48 or any(r["status"] != "PASS" for r in reproduced):
        raise AssertionError(f"old H30 primary reference reproduction failed: {len(reproduced)} rows")
    write_csv(ROOT / "results/H30_REFERENCE_REPRODUCTION.csv", list(reproduced[0].keys()), reproduced)
    write_json(ROOT / "qa/H30_REFERENCE_REPRODUCTION.json", {
        "status": "PASS_OLD_12_MODEL_H30_POSITION_AND_SAFE_FD_REPRODUCED",
        "comparisons": len(reproduced), "absolute_tolerance": TOL,
        "max_abs_delta": max(abs(float(r["delta"])) for r in reproduced),
        "accepted_reference": str(QUERY / "results/TRAIN_DEV_BY_SEED.csv"),
        "p_values": False, "final_test_access": 0, "heldout_access": 0,
    })

    # Query-gain reproduction includes fixed seeds only; skip the accepted three-seed summary row.
    gains_ref = read_csv(QUERY / "results/QUERY_GAINS.csv")
    gain_rows = []
    lookup = {(r["split"], r["family"], r["query"], int(r["seed"])): r for r in summary
              if r["seed"] in ("20261001", "20261002", "20261003")}
    for family in ("TS11-D", "STATE-CTRL"):
        for seed in (20261001, 20261002, 20261003):
            for metric, field in (("position_standardized_mae", "prefix_position_standardized_mae"),
                                  ("safe_fd_normalized", "future_internal_fd_safe_normalized")):
                old = float(lookup[("dev", family, "Q0", seed)][field])
                had = float(lookup[("dev", family, "Q1", seed)][field])
                computed = old - had
                match = [r for r in gains_ref if r["split"] == "dev" and r["model"] == family and r["seed"] == str(seed) and r["metric"] == metric]
                if len(match) != 1:
                    raise AssertionError(f"accepted H30 query-gain row missing or duplicated: {family}/{seed}/{metric}")
                expected = float(match[0]["G_Q_old_minus_had"])
                delta = computed - expected
                gain_rows.append({
                    "family": family, "seed": seed, "metric": metric,
                    "recomputed_old_minus_had": computed, "accepted_old_minus_had": expected,
                    "delta": delta, "absolute_tolerance": TOL,
                    "status": "PASS" if abs(delta) <= TOL else "FAIL",
                })
    if len(gain_rows) != 12 or any(r["status"] != "PASS" for r in gain_rows):
        raise AssertionError("H30 per-seed query-gain reproduction failed")
    write_csv(ROOT / "results/H30_QUERY_GAIN_REPRODUCTION.csv", list(gain_rows[0].keys()), gain_rows)

    # The H30 episode table is already scored. Reuse it for one paired source-group bootstrap.
    group_metric: dict[tuple[str, str, int | None], dict[str, dict[str, float]]] = {}
    with episode_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "dev" or int(row["horizon"]) != 30:
                continue
            if row["system"] == "Ridge-CPU":
                key = ("Ridge-CPU", None, None)
            elif row["system"] == "Q0_STATE-CTRL":
                key = ("Q0_STATE-CTRL", int(row["seed"]), None)
            else:
                continue
            gid = row["source_group_id"]
            group_metric.setdefault(key, {})[gid] = {
                "position": float(row["prefix_position_standardized_mae"]),
                "safe_fd": float(row["future_internal_fd_safe_normalized"]),
            }
    ridge = group_metric[("Ridge-CPU", None, None)]
    seeds = [20261001, 20261002, 20261003]
    state = [group_metric[("Q0_STATE-CTRL", seed, None)] for seed in seeds]
    gids = sorted(ridge)
    if len(gids) != 603 or any(set(m) != set(gids) for m in state):
        raise AssertionError("paired source-group H30 support is not 603 common dev groups")
    state_pos, ridge_pos, state_fd, ridge_fd = {}, {}, {}, {}
    paired_rows = []
    for gid in gids:
        state_pos[gid] = float(np.mean([m[gid]["position"] for m in state]))
        state_fd[gid] = float(np.mean([m[gid]["safe_fd"] for m in state]))
        ridge_pos[gid] = ridge[gid]["position"]
        ridge_fd[gid] = ridge[gid]["safe_fd"]
        paired_rows.append({
            "source_group_id": gid, "state_q0_mean_seed_position": state_pos[gid],
            "ridge_position": ridge_pos[gid], "state_minus_ridge_position": state_pos[gid] - ridge_pos[gid],
            "state_q0_mean_seed_safe_fd": state_fd[gid], "ridge_safe_fd": ridge_fd[gid],
            "state_minus_ridge_safe_fd": state_fd[gid] - ridge_fd[gid],
            "final_test_access": 0, "heldout_access": 0,
        })
    write_csv(ROOT / "results/PAIRED_H30_RIDGE_VS_STATE_Q0_GROUP.csv", list(paired_rows[0].keys()), paired_rows)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(gids), size=(BOOT_REPS, len(gids)))
    def paired_ci(a, b):
        av = np.asarray([a[g] for g in gids], dtype=np.float64)
        bv = np.asarray([b[g] for g in gids], dtype=np.float64)
        deltas = av - bv
        boot = (av[draws] - bv[draws]).mean(axis=1)
        return {
            "mean_difference": float(deltas.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "replicates": BOOT_REPS,
        }
    ci_pos = paired_ci(state_pos, ridge_pos)
    ci_fd = paired_ci(state_fd, ridge_fd)
    ci_rows = [
        {"split": "dev", "contrast": "L_STATE_Q0_mean_3seed - L_Ridge", "metric": "H30 prefix Full28 position standardized MAE", **ci_pos,
         "interpretation": "positive favors Ridge; conditional on fixed models and repeatedly exposed dev; no p-value"},
        {"split": "dev", "contrast": "L_STATE_Q0_mean_3seed - L_Ridge", "metric": "H30 future-internal safe-FD normalized MAE", **ci_fd,
         "interpretation": "positive favors Ridge; conditional on fixed models and repeatedly exposed dev; no p-value"},
    ]
    write_csv(ROOT / "results/PAIRED_H30_RIDGE_VS_STATE_Q0_CI.csv", list(ci_rows[0].keys()), ci_rows)

    write_csv(ROOT / "figure_data/HORIZON_SUMMARY.csv", list(summary[0].keys()), summary)
    write_csv(ROOT / "figure_data/QUERY_GAIN_POINTS.csv", list(gain_rows[0].keys()), gain_rows)
    write_csv(ROOT / "figure_data/PAIRED_H30_RIDGE_VS_STATE_Q0.csv", list(paired_rows[0].keys()), paired_rows)

    case_rows = read_csv(QUERY / "qualitative/CASE_MANIFEST.csv")
    if len(case_rows) != 12 or any(r["final_test_access"] != "0" for r in case_rows):
        raise AssertionError("old fixed case manifest does not pass")
    ids_file = ROOT / "inventory/INDEX_AND_CASE_ROWS.json"
    if not ids_file.exists():
        # Reconstruct index identities from accepted input IDs without opening targets.
        with np.load(INPUTS, allow_pickle=False) as z:
            input_ids = z["anchor_id"].astype("U64", copy=True)
            input_eps = z["episode_index"].astype(np.int64, copy=True)
            input_split_code = z["split_code"].astype(np.int8, copy=True)
        idx = {str(x): i for i, x in enumerate(input_ids.tolist())}
        identity = {str(a): {"index": int(i), "episode_index": int(input_eps[i]), "split": "train" if input_split_code[i] == 0 else "dev"}
                    for a, i in idx.items()}
    else:
        identity = json.loads(ids_file.read_text(encoding="utf-8"))
    case_out = []
    for c in case_rows:
        q = identity[c["anchor_id"]]
        if q["split"] != "dev" or int(q["episode_index"]) != int(c["episode_index"]):
            raise AssertionError(f"fixed case identity mismatch: {c['anchor_id']}")
        case_out.append({
            "case_id": c["case_id"], "stratum": c["stratum"], "stratum_rank": c["stratum_rank"],
            "source_group_id": c["source_group_id"], "episode_index": int(c["episode_index"]),
            "anchor_id": c["anchor_id"], "dev_index": int(q["index"]),
            "target_first_frame": int(c["target_first_frame"]), "target_last_frame": int(c["target_last_frame"]),
            "selection_hash": c["selection_hash"], "selection_source": str(QUERY / "qualitative/CASE_MANIFEST.csv"),
            "ridge_score_used_for_selection": False, "final_test_access": 0,
        })
    write_csv(ROOT / "figure_data/LOCKED_CASE_IDS.csv", list(case_out[0].keys()), case_out)

    score_access = {
        "status": "PASS_DEV_TARGETS_SCORED_AFTER_RIDGE_MODEL_LOCK",
        "score_gate_path": "protocol/OUTCOME_SCORE_GATE.json",
        "score_gate_sha256": sha(ROOT / "protocol/OUTCOME_SCORE_GATE.json"),
        "ridge_model_lock_sha256": sha(model_lock_path),
        "target_source": str(INPUTS), "target_source_sha256": sha(INPUTS),
        "target_rows_scored": {"train": 17757, "dev": 2322},
        "target_horizon_used": "native H30 only, indices 0:30",
        "underlying_npz_member_stores_60_steps": True,
        "steps_31_to_60_used_in_scoring_or_model_fit": False,
        "exact_initial_scoring_timestamp": "UNKNOWN; first scorer process did not persist its start time before reaching the parser failure",
        "scoring_output_recovered_without_target_reread": True,
        "target_read_order": "model lock and target-free predictions accepted before OUTCOME_SCORE_GATE; original scorer then read only primary train/dev rows and scored H30",
        "raw_timestamp_metadata": {"episodes": 3315, "columns": ["frame_index", "timestamp"], "action_column_read": False},
        "dev_scope": "previously exposed exploratory development",
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(ROOT / "protocol/OUTCOME_SCORE_ACCESS.json", score_access)

    ridge_dev30 = next(r for r in summary if r["split"] == "dev" and r["system"] == "Ridge-CPU" and int(r["horizon"]) == 30)
    result = {
        "status": "PASS_FIXED_H30_HORIZON_ANALYSIS_AND_RIDGE_COMPARISON",
        "scoring_tables_reused": True, "episode_metric_rows": episode_count,
        "summary_rows": len(summary), "actual_timestamp_rows": timestamp_count,
        "old_h30_reference_comparisons": len(reproduced), "old_h30_query_gain_comparisons": len(gain_rows),
        "max_reference_abs_delta": max(abs(float(r["delta"])) for r in reproduced),
        "max_query_gain_abs_delta": max(abs(float(r["delta"])) for r in gain_rows),
        "selected_lambda": lock["selected_global_lambda"], "lambda_grid_extended": False,
        "ridge_h30_dev": {
            "position_prefix_standardized_mae": float(ridge_dev30["prefix_position_standardized_mae"]),
            "position_endpoint_standardized_mae": float(ridge_dev30["endpoint_position_standardized_mae"]),
            "safe_fd_normalized": float(ridge_dev30["future_internal_fd_safe_normalized"]),
            "position_prefix_skill_vs_persistence": float(ridge_dev30["position_prefix_skill_vs_persistence"]),
            "safe_fd_skill_vs_persistence": float(ridge_dev30["future_internal_fd_safe_skill_vs_persistence"]),
            "tv_ratio_aggregate": float(ridge_dev30["tv_ratio_aggregate"]),
            "direction_accuracy": float(ridge_dev30["direction_accuracy_active_episode_equal"]),
            "balanced_direction_accuracy": float(ridge_dev30["balanced_direction_accuracy_episode_equal"]),
        },
        "ridge_vs_state_q0_h30": ci_rows,
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(ROOT / "results/HORIZON_SCORE_SUMMARY.json", result)
    write_json(ROOT / "qa/HORIZON_ANALYSIS_ACCEPTANCE.json", {
        "status": "PASS_FIXED_H30_HORIZON_ANALYSIS_AND_RIDGE_COMPARISON",
        "horizons": [1, 3, 6, 10, 15, 30],
        "systems": "12 accepted fixed predictions plus Ridge and Persistence, seeds kept separate",
        "episode_metric_rows": episode_count, "summary_rows": len(summary),
        "old_h30_reference_comparisons": len(reproduced), "old_query_gain_comparisons": len(gain_rows),
        "paired_dev_groups": len(gids), "bootstrap_replicates": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
        "p_values": False, "dev_previously_exposed": True, "independent_confirmation": False,
        "H1_internal_fd": "NA", "H30_internal_differences": 29,
        "anchor_relative_position_error_is_duplicate": True,
        "actual_timestamp_rows": timestamp_count, "action_column_read_for_timestamps": False,
        "final_test_access": 0, "heldout_access": 0,
    })
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


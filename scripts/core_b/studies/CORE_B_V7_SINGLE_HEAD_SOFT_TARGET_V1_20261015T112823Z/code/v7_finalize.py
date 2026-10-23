from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v7_common import ROOT, SEEDS, TASKS, TRAIN_NPZ, V4, atomic_json, load_train_prefix, sha_file, write_csv


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_tables() -> dict[str, list[dict[str, str]]]:
    names = (
        "PER_WINDOW_METRICS.csv", "PRIMARY_TASK_SEED.csv", "PAIRED_EFFECTS.csv",
        "PAIRED_GROUP_CI.csv", "SEED_BLOCK_LOO.csv", "TASK_LOO.csv",
        "TEACHER_RETENTION.csv", "TRAIN_TARGET_SIGNAL.csv",
        "TRAIN_FIT_AND_IMITATION.csv", "TRAINING_CURVES.csv",
        "RESOURCE_LEDGER.csv", "INFERENCE_PROFILE.csv",
    )
    tables = {}
    for name in names:
        path = ROOT / "results" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[name] = read_csv(path)
    return tables


def absolute_macro(primary: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    methods = ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT", "S", "V", "T", "BIAS_STATE", "BIAS_VIDEO", "PERSISTENCE")
    for method in methods:
        rows = [r for r in primary if r["method"] == method and r["task_id"] == "macro"]
        if not rows:
            continue
        metrics: dict[str, Any] = {}
        for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1", "native_mae"):
            values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
            metrics[metric] = {
                "per_seed": {str(r["seed"]): float(r[metric]) for r in rows},
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "n_seed_instances": len(values),
            }
        out[method] = metrics
    return out


def decisions(effects: list[dict[str, str]], ci: list[dict[str, str]]) -> dict[str, Any]:
    out = {}
    for contrast in ("MIX_VS_GT", "MIX_VS_SELF"):
        ci_macro = {r["metric"]: r for r in ci if r["contrast"] == contrast and r["scope"] == "macro"}
        seed_rows = [r for r in effects if r["contrast"] == contrast and r["scope"] == "macro"]
        metrics = {}
        for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1"):
            metric_rows = [r for r in seed_rows if r["metric"] == metric]
            row = ci_macro[metric]
            seed_gains = np.asarray([float(r["gain"]) for r in metric_rows], dtype=np.float64)
            metrics[metric] = {
                "point_gain": float(row["point_gain"]),
                "sample_sd_across_seeds": float(seed_gains.std(ddof=1)),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "per_seed_gain": {str(r["seed"]): float(r["gain"]) for r in metric_rows},
                "positive_seed_count": sum(float(r["gain"]) > 0 for r in metric_rows),
            }
        pos = metrics["position_mae"]
        fd = metrics["safe_fd_l1"]
        out[contrast] = {
            "metrics": metrics,
            "POINT_JOINT_GAIN": pos["point_gain"] > 0 and fd["point_gain"] > 0,
            "CONDITIONAL_CI_JOINT_GAIN": pos["ci95_low"] > 0 and fd["ci95_low"] > 0,
        }
    return out


def build_figures(tables: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    ci = tables["PAIRED_GROUP_CI.csv"]
    primary = [
        row for row in ci
        if row["scope"] == "macro" and row["contrast"] in {"MIX_VS_GT", "MIX_VS_SELF"}
        and row["metric"] in {"position_mae", "safe_fd_l1"}
    ]
    if len(primary) != 4:
        raise RuntimeError(f"expected four primary macro CI records; got {len(primary)}")
    write_csv(ROOT / "figure_data/PRIMARY_DUAL_METRIC_GAINS.csv", list(primary[0]), primary)

    colors = {"MIX_VS_GT": "#276FBF", "MIX_VS_SELF": "#D17A00"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for contrast in ("MIX_VS_GT", "MIX_VS_SELF"):
        pos = next(r for r in primary if r["contrast"] == contrast and r["metric"] == "position_mae")
        fd = next(r for r in primary if r["contrast"] == contrast and r["metric"] == "safe_fd_l1")
        x, y = float(pos["point_gain"]), float(fd["point_gain"])
        xerr = np.array([[x-float(pos["ci95_low"])], [float(pos["ci95_high"])-x]])
        yerr = np.array([[y-float(fd["ci95_low"])], [float(fd["ci95_high"])-y]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4, color=colors[contrast], label=contrast.replace("_", " "))
        ax.annotate(contrast.replace("MIX_VS_", "MIX vs "), (x, y), xytext=(7, 6), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Position MAE gain (reference loss minus MIX_SOFT)")
    ax.set_ylabel("Safe-FD gain (reference loss minus MIX_SOFT)")
    ax.set_title("V7 primary development contrasts\n95% paired source-group percentile intervals")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig1 = ROOT / "figures/PRIMARY_DUAL_METRIC_GAINS.png"
    fig.savefig(fig1, dpi=180)
    plt.close(fig)

    seed_rows = [
        r for r in tables["PAIRED_EFFECTS.csv"]
        if r["scope"] == "task" and r["metric"] == "position_mae"
        and r["contrast"] in {"MIX_VS_GT", "MIX_VS_SELF"}
    ]
    write_csv(ROOT / "figure_data/POSITION_GAIN_TASK_SEED.csv", list(seed_rows[0]), seed_rows)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True)
    limit = max(abs(float(r["gain"])) for r in seed_rows)
    for ax, contrast in zip(axes, ("MIX_VS_GT", "MIX_VS_SELF")):
        matrix = np.zeros((len(TASKS), len(SEEDS)), dtype=np.float64)
        for ti, task in enumerate(TASKS):
            for si, seed in enumerate(SEEDS):
                row = next(r for r in seed_rows if r["contrast"] == contrast and int(r["task_id"]) == task and int(r["seed"]) == seed)
                matrix[ti, si] = float(row["gain"])
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(TASKS)), labels=[str(t) for t in TASKS])
        ax.set_ylabel("Task ID")
        ax.set_title(contrast.replace("_", " "))
        for ti in range(len(TASKS)):
            for si in range(len(SEEDS)):
                ax.text(si, ti, f"{matrix[ti, si]:+.5f}", ha="center", va="center", fontsize=8)
    axes[-1].set_xticks(range(len(SEEDS)), labels=[str(s) for s in SEEDS])
    axes[-1].set_xlabel("Paired source seed")
    fig.colorbar(im, ax=axes, label="Position MAE gain")
    fig2 = ROOT / "figures/POSITION_GAIN_TASK_SEED.png"
    fig.savefig(fig2, dpi=180)
    plt.close(fig)

    retention = tables["TEACHER_RETENTION.csv"]
    valid = [r for r in retention if r["retention_R"] != "NA"]
    write_csv(ROOT / "figure_data/TEACHER_RETENTION.csv", list(retention[0]), retention)
    labels = [str(r["seed"]) if r["scope"] == "seed" else "overall" for r in valid]
    values = [float(r["retention_R"]) for r in valid]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors_ret = ["#567D46"] * max(0, len(values)-1) + ["#365A27"]
    ax.bar(np.arange(len(values)), values, color=colors_ret)
    ax.axhline(1.0, color="#8B1E3F", linestyle="--", linewidth=1.2, label="R=1 point reference")
    ax.set_xticks(np.arange(len(values)), labels=labels, rotation=20)
    ax.set_ylabel("MAE retention ratio R")
    ax.set_title("MIX_SOFT retention of fixed MIX50 MAE gain")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig3 = ROOT / "figures/TEACHER_RETENTION.png"
    fig.savefig(fig3, dpi=180)
    plt.close(fig)

    captions = {
        "schema_version": 1,
        "figures": [
            {
                "path": str(fig1),
                "caption": "Paired five-task macro position-MAE and safe-FD gains for MIX_SOFT versus the synchronous GT_RETRAIN and SELF_SOFT controls. Positive gains favor MIX_SOFT. Intervals are unadjusted exploratory 95% source-group bootstrap intervals conditional on three paired source instances.",
                "data_path": str(ROOT / "figure_data/PRIMARY_DUAL_METRIC_GAINS.csv"),
            },
            {
                "path": str(fig2),
                "caption": "Per-task and per-seed position-MAE gains for MIX_SOFT against GT_RETRAIN and SELF_SOFT. Cell values are rounded to five decimals; positive values favor MIX_SOFT and the CSV keeps full precision.",
                "data_path": str(ROOT / "figure_data/POSITION_GAIN_TASK_SEED.csv"),
            },
            {
                "path": str(fig3),
                "caption": "Per-seed and equal-seed overall descriptive point estimate of the fraction of fixed MIX50 position-MAE gain retained by MIX_SOFT. This ratio has no confidence interval and is not a non-inferiority claim.",
                "data_path": str(ROOT / "figure_data/TEACHER_RETENTION.csv"),
            },
        ],
    }
    atomic_json(ROOT / "captions/CAPTIONS.json", captions)
    qa = {
        "schema_version": 1,
        "status": "BUILT_PENDING_VISUAL_REVIEW",
        "created_utc": utc_now(),
        "figures": [{"path": str(p), "sha256": sha_file(p), "exists": p.is_file()} for p in (fig1, fig2, fig3)],
        "figure_data_source": "V7 accepted result tables only; no target archive access",
        "target_archive_opened": False,
    }
    atomic_json(ROOT / "qa/FIGURE_BUILD_QA.json", qa)
    (ROOT / "qa/FIGURE_VISUAL_REVIEW.md").write_text(
        "# Figure visual review\n\nStatus: PENDING\n\nManual visual inspection of the three generated figures is pending.\n",
        encoding="utf-8",
    )
    return qa


def gpu_resource_summary() -> dict[str, Any]:
    status_dir = ROOT / "coordination/status"
    statuses = [read_json(path) for path in sorted(status_dir.glob("*.json"))]
    if not statuses:
        raise RuntimeError("V7 watcher status directory is empty")
    intervals = []
    devices = set()
    resource_rows = read_csv(ROOT / "results/RESOURCE_LEDGER.csv")
    peak_estimate = 0
    peak_free_margin = None
    peak_reserved = 0.0
    peak_allocated = 0.0
    elapsed = 0.0
    for item in statuses:
        if item.get("status") != "complete" or item.get("lease_released") is not True:
            raise RuntimeError(f"watcher status is incomplete or lease remains: {item.get('job_id')}")
        if item.get("external_process_identity_recorded") is not False or item.get("non_child_process_signaled") is not False:
            raise RuntimeError(f"watcher safety receipt is invalid: {item.get('job_id')}")
        devices.add(int(item["physical_gpu"]))
        peak_estimate = max(peak_estimate, int(item.get("measured_peak_mib", 0)))
        if item.get("final_recheck"):
            snap = item["final_recheck"]
            margin = int(snap["free_memory_mib"]) - int(item.get("required_free_memory_mib", 0))
            peak_free_margin = margin if peak_free_margin is None else min(peak_free_margin, margin)
        start = datetime.fromisoformat(str(item["lease_acquired_utc"]).replace("Z", "+00:00")).timestamp()
        end = datetime.fromisoformat(str(item["lease_released_utc"]).replace("Z", "+00:00")).timestamp()
        intervals.append((start, 1))
        intervals.append((end, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(intervals, key=lambda x: (x[0], x[1])):
        active += delta
        max_active = max(max_active, active)
    peak_reserved = max([float(row["peak_reserved_mib"]) for row in resource_rows if row.get("peak_reserved_mib") not in {"", "None"}] or [0.0])
    peak_allocated = max([float(row["peak_allocated_mib"]) for row in resource_rows if row.get("peak_allocated_mib") not in {"", "None"}] or [0.0])
    elapsed = sum([float(row["elapsed_seconds"]) for row in resource_rows if row.get("elapsed_seconds") not in {"", "None"}])
    result = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "watcher_jobs_completed": len(statuses),
        "all_leases_released": True,
        "physical_gpus_used": sorted(devices),
        "visible_gpu_pool": "dynamic_all_visible",
        "max_project_children_cap": 2,
        "max_observed_project_children": max_active,
        "min_final_free_memory_minus_admission_mib": peak_free_margin,
        "max_registered_peak_mib": peak_estimate,
        "max_child_peak_allocated_mib": peak_allocated,
        "max_child_peak_reserved_mib": peak_reserved,
        "sum_gpu_child_elapsed_seconds": elapsed,
        "external_process_identity_reads": 0,
        "non_child_signals": 0,
        "status_paths": [str(path) for path in sorted(status_dir.glob("*.json"))],
    }
    atomic_json(ROOT / "results/GPU_RESOURCE_SUMMARY.json", result)
    return result


def old_root_readonly_audit() -> dict[str, Any]:
    source_map = read_json(ROOT / "protocol/SOURCE_MAP.json")
    checked: list[dict[str, Any]] = []
    def verify(path_value: str, expected: str, role: str) -> None:
        path = Path(path_value)
        actual = sha_file(path)
        if actual != expected:
            raise RuntimeError(f"old source changed since V7 lock: {role} {path}")
        checked.append({"path": str(path), "sha256": actual, "role": role, "status": "PASS"})
    for row in source_map["models"]:
        verify(row["model_path"], row["model_sha256"], f"accepted_teacher_model_{row['role']}_{row['seed']}")
        verify(row["model_manifest_path"], row["model_manifest_sha256"], f"accepted_teacher_manifest_{row['role']}_{row['seed']}")
        verify(row["init_path"], row["init_sha256"], f"accepted_common_INIT_{row['seed']}")
    for row in source_map["evaluation_reference_bundles"]:
        verify(row["prediction_path"], row["prediction_sha256"], f"evaluation_reference_{row['role']}_{row['seed']}")
        verify(row["rows_path"], row["rows_sha256"], f"evaluation_reference_rows_{row['role']}_{row['seed']}")
        verify(row["seal_path"], row["seal_sha256"], f"accepted_source_seal_{row['source_root']}")
    evidence = source_map["acceptance_evidence"]
    for key in ("V2_prediction_seal", "V3_prediction_seal", "V4_source_map", "V5_prediction_seal", "V5_final_receipt", "V6_final_receipt", "V4_common_init_receipt", "V4_recipe_lock"):
        item = evidence[key]
        verify(item["path"], item["sha256"], f"locked_acceptance_evidence_{key}")
    watcher_policy = read_json(V4 / "inventory/WATCHER_POLICY.json")
    watcher_path = Path(watcher_policy["source_path"])
    verify(str(watcher_path), watcher_policy["source_sha256"], "shared_GPU_watcher")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "checked_old_source_files": len(checked),
        "checks": checked,
        "V5_manifest_receipt_status": evidence["V5_final_receipt"]["status"],
        "V6_manifest_receipt_status": evidence["V6_final_receipt"]["status"],
        "old_root_writes": 0,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "inventory/OLD_ROOT_READONLY_AUDIT.json", audit)
    return audit


def write_provenance() -> dict[str, Any]:
    config_files = sorted(path for path in (ROOT / "configs").rglob("*.json") if path.is_file())
    config_records = [{"path": str(path), "relative_path": path.relative_to(ROOT).as_posix(), "sha256": sha_file(path)} for path in config_files]
    atomic_json(ROOT / "inventory/CONFIG_PROVENANCE.json", {
        "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
        "configuration_count": len(config_records), "configurations": config_records,
        "final_heldout_payload_access": 0,
    })
    v4 = ROOT.parent.parent.parent / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
    repo = ROOT.parent.parent.parent / "repo/ee6008-humanoid-vjepa"
    reused = [
        v4 / "code/v2_model.py", v4 / "code/locked_source/D3_model.py",
        v4 / "code/locked_source/d3_eq_contract.py", v4 / "code/v2_scheduler.py",
        v4 / "code/prepare_v2_data.py", v4 / "code/v4_infer.py",
        repo / "tools/gpu_lease_watcher.py",
    ]
    new_code = sorted(path for path in (ROOT / "code").rglob("*") if path.is_file())
    records = [{"path": str(path), "sha256": sha_file(path), "role": "V7_new_or_backup_source"} for path in new_code]
    records.extend({"path": str(path), "sha256": sha_file(path), "role": "read_only_reused_source"} for path in reused)
    atomic_json(ROOT / "inventory/CODE_PROVENANCE.json", {
        "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
        "new_v7_source_files": len(new_code), "read_only_reused_source_files": len(reused),
        "files": records, "old_root_writes": 0,
    })
    # This records the installed framework build without asking CUDA/NVML for device state.
    import torch
    watcher_statuses = [read_json(path) for path in sorted((ROOT / "coordination/status").glob("*.json"))]
    gpu_names = sorted({str(snapshot.get("name")) for row in watcher_statuses for snapshot in row.get("visible_snapshots", [])})
    selected = sorted({int(row["physical_gpu"]) for row in watcher_statuses if str(row.get("physical_gpu", "")).isdigit()})
    resource_rows = read_csv(ROOT / "results/RESOURCE_LEDGER.csv")
    visible_devices = sorted({str(row["cuda_visible_devices"]) for row in resource_rows if row.get("cuda_visible_devices") not in {"", "None"}})
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except OSError:
        meminfo = {}
    cpu_affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    mem_max_path = Path("/sys/fs/cgroup/memory.max")
    train_prefix = load_train_prefix()
    train_prefix_bytes = int(sum(value.nbytes for value in train_prefix.values()))
    y_path = ROOT / "training_targets/Y_NATIVE.npy"
    target_paths = [Path(row["target_path"]) for row in read_json(ROOT / "protocol/TARGET_LOCK.json")["target_entries"]]
    teacher_entries = read_json(ROOT / "protocol/TEACHER_BANK_SEAL.json")["entries"]
    source_teacher_bytes = sum(Path(row["prediction_path"]).stat().st_size for row in teacher_entries)
    reference_seed = min(int(row["seed"]) for row in teacher_entries if row.get("seed") is not None)
    teacher_v = next(row for row in teacher_entries if int(row.get("seed", -1)) == reference_seed and row.get("teacher_role") == "V")
    teacher_t = next(row for row in teacher_entries if int(row.get("seed", -1)) == reference_seed and row.get("teacher_role") == "T")
    per_fit_vt_bytes = Path(teacher_v["prediction_path"]).stat().st_size + Path(teacher_t["prediction_path"]).stat().st_size
    per_fit_target = next(Path(row["target_path"]) for row in read_json(ROOT / "protocol/TARGET_LOCK.json")["target_entries"] if int(row["seed"]) == reference_seed and row["program"] == "MIX_SOFT")
    model_manifest = read_json(ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json")["entries"][0]["model_manifest_path"]
    model_data = read_json(Path(model_manifest))
    parameter_bytes = int(model_data["parameter_count"]) * 4
    feature_batch_bytes = 64 * 2 * 16 * 4 * 768 * 4
    cpu_materialized_arrays_bytes = train_prefix_bytes + y_path.stat().st_size + per_fit_target.stat().st_size + per_fit_vt_bytes
    runtime = {
        "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
        "host": os.environ.get("EE6008_HOST_LABEL", "shared-host"), "user": 'project-team', "project_root": str(ROOT.parent.parent.parent),
        "python_executable": sys.executable, "python_version": sys.version.split()[0],
        "platform": platform.platform(), "torch_version": str(torch.__version__),
        "torch_cuda_build_version": str(torch.version.cuda),
        "nvidia_driver_version": "NOT_QUERIED_IN_FINALIZER; see escalated host gate and watcher receipts",
        "visible_cpu_count": os.cpu_count(), "process_cpu_affinity_count": cpu_affinity,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
        "cgroup_cpu_max": cpu_max_path.read_text().strip() if cpu_max_path.is_file() else "UNKNOWN_OR_NOT_MOUNTED",
        "cgroup_memory_max": mem_max_path.read_text().strip() if mem_max_path.is_file() else "UNKNOWN_OR_NOT_MOUNTED",
        "system_memory_snapshot_bytes": meminfo,
        "host_process_peak_rss_measured": False,
        "input_payload_estimate": {
            "train_prefix_materialized_bytes": train_prefix_bytes,
            "Y_native_array_bytes": y_path.stat().st_size,
            "one_native_float64_training_target_bytes": target_paths[0].stat().st_size if target_paths else None,
            "full_six_source_plus_three_T_teacher_bank_bytes": source_teacher_bytes,
            "one_seed_V_plus_T_native_prediction_bytes": per_fit_vt_bytes,
            "one_Q0_feature_batch64_bytes": feature_batch_bytes,
            "student_parameter_bytes_float32": parameter_bytes,
            "one_fit_materialized_prefix_Y_Z_V_T_payload_bytes_excluding_python_runtime_and_mmap_page_cache": cpu_materialized_arrays_bytes,
            "note": "payload sizes are not peak process RSS; runtime/allocator/mmap-page cache were not instrumented",
        },
        "V4_trusted_gpu_peak_allocated_mib": read_json(V4 / "results/GPU_RESOURCE_SUMMARY.json")["max_measured_allocated_mib"],
        "V4_trusted_gpu_peak_reserved_mib": read_json(V4 / "results/GPU_RESOURCE_SUMMARY.json")["max_measured_reserved_mib"],
        "gpu_models_seen_by_watcher": gpu_names, "physical_gpu_indices_used": selected,
        "cuda_visible_devices_values": visible_devices,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "inventory/RUNTIME_ENV.json", runtime)
    return {"configuration_count": len(config_records), "new_v7_source_files": len(new_code), "reused_source_files": len(reused), "runtime_env_path": str(ROOT / "inventory/RUNTIME_ENV.json")}


def build() -> None:
    if read_json(ROOT / "qa/SCORER_SEAL.json").get("status") != "PASS":
        raise RuntimeError("V7 scorer seal is not PASS")
    tables = load_tables()
    old_root_audit = old_root_readonly_audit()
    provenance = write_provenance()
    if len(tables["TRAINING_CURVES.csv"]) != 9 * 160:
        raise RuntimeError("V7 training curves do not cover 9 x 160 epochs")
    if len(tables["INFERENCE_PROFILE.csv"]) != 5:
        raise RuntimeError("V7 timing table does not contain five cached-input paths")
    figure_qa = build_figures(tables)
    gpu_resources = gpu_resource_summary()
    failure_paths = sorted(list((ROOT / "coordination").glob("GPU_PIPELINE_FAILURE*.json")) + list((ROOT / "qa").glob("TARGET_BLIND_REPLAY_QA_ATTEMPT_*.json")) + list((ROOT / "qa").glob("PREDICTION_PREFLIGHT_ATTEMPT_*.json")) + list((ROOT / "qa").glob("FINALIZATION_ATTEMPT_*.json")))
    preserved_failures = [{"path": str(path), **{k: read_json(path).get(k) for k in ("status", "error_type", "error", "created_utc", "failed_utc")}} for path in failure_paths]
    decisions_summary = decisions(tables["PAIRED_EFFECTS.csv"], tables["PAIRED_GROUP_CI.csv"])
    absolute = absolute_macro(tables["PRIMARY_TASK_SEED.csv"])
    fit_index = read_json(ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json")
    pred_seal = read_json(ROOT / "qa/PREDICTION_SEAL.json")
    scorer = read_json(ROOT / "qa/SCORER_SEAL.json")
    vi_reproduction = read_json(ROOT / "qa/GT_RETRAIN_VI_REPRODUCTION.json")
    if vi_reproduction.get("status") != "PASS" or len(vi_reproduction.get("entries", [])) != 3:
        raise RuntimeError("GT_RETRAIN versus accepted V_i byte-identity audit is not PASS")
    cpu_qa = read_json(ROOT / "qa/CPU_PREFLIGHT.json")
    replay_qa = read_json(ROOT / "qa/TARGET_BLIND_REPLAY_QA.json")
    stage0 = read_json(ROOT / "qa/STAGE0.json")
    teacher_seal = read_json(ROOT / "protocol/TEACHER_BANK_SEAL.json")
    target_lock = read_json(ROOT / "protocol/TARGET_LOCK.json")
    if any(value.get("status") != expected for value, expected in ((cpu_qa, "PASS"), (replay_qa, "PASS"), (stage0, "PASS"), (teacher_seal, "PASS"), (target_lock, "LOCKED"), (fit_index, "PASS"), (pred_seal, "PASS"), (scorer, "PASS"))):
        raise RuntimeError("a required V7 teacher/student acceptance receipt is not in its accepted state")
    summary = {
        "schema_version": 1,
        "status": "results_complete_pending_visual_review_and_checksum",
        "created_utc": utc_now(),
        "study": "post_V6_single_fusion_head_supervised_soft_target_development",
        "new_head_fits_accepted": fit_index["accepted_fit_count"],
        "student_dev_prediction_jobs_accepted": pred_seal["accepted_student_jobs"],
        "formal_optimizer_updates": fit_index["total_accepted_optimizer_updates"],
        "GT_RETRAIN_byte_identical_to_accepted_V_i_all_seeds": True,
        "optimizer_updates_per_fit": 3200,
        "programs": ["GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"],
        "seeds": list(SEEDS),
        "primary_decisions": decisions_summary,
        "absolute_macro_losses": absolute,
        "teacher_retention": tables["TEACHER_RETENTION.csv"],
        "paired_group_ci_macro": [
            row for row in tables["PAIRED_GROUP_CI.csv"]
            if row["scope"] == "macro" and row["metric"] in {"position_mae", "safe_fd_l1", "seam_safe_fd_l1"}
        ],
        "student_dev_window_loss_evaluations": scorer["student_dev_window_loss_evaluations"],
        "dev_target_archive_loads": scorer["dev_target_archive_load_operations"],
        "final_heldout_payload_access": 0,
        "independent_confirmation": False,
        "checksum_receipt_path": str(ROOT / "qa/FINAL_CHECKSUM_VERIFICATION.json"),
        "checksum_receipt_status": "post-manifest read-back receipt not written yet",
        "figure_build_status": figure_qa["status"],
        "gpu_resource_summary": gpu_resources,
        "provenance_summary": provenance,
        "old_root_readonly_audit": {"status": old_root_audit["status"], "checked_files": old_root_audit["checked_old_source_files"]},
        "preserved_failed_attempts": preserved_failures,
        "limitations": [
            "Development-only evidence conditional on three paired source instances and the exposed development groups.",
            "Outer-train teacher predictions are in-sample supervision, not out-of-fold or independent labels.",
            "This tests fixed supervised soft-target assistance for one fusion head; it does not establish video-unique information or that averaging is irreplaceable.",
            "Joint-action coordinate errors are not physical-world accuracy claims.",
            "Host process peak RSS was not instrumented; the runtime receipt records visible CPU/memory and materialized array payload sizes separately.",
            "Final/heldout payload access remained zero.",
        ],
    }
    atomic_json(ROOT / "results/RESULTS_SUMMARY.json", summary)

    lines = [
        "# EE6008 Core B V7: fixed teacher soft targets for one fusion head",
        "",
        "## Execution",
        "",
        f"- Accepted new head fits: {fit_index['accepted_fit_count']}/9; formal optimizer updates: {fit_index['total_accepted_optimizer_updates']}.",
        f"- Student dev prediction jobs: {pred_seal['accepted_student_jobs']}/9; student primary window-loss evaluations: {scorer['student_dev_window_loss_evaluations']}.",
        "- Six frozen S/V outer-train teacher bundles were replayed; three fixed T_i predictions were averaged in native float64 before the Y train read.",
        "- The synchronous GT_RETRAIN control reproduced the accepted V_i checkpoint byte-identically for all three seeds.",
        "- Train teacher outputs are in-sample supervision, not OOF. The student forward uses one TS11-D head and reads no teacher output or dev GT.",
        "- Final/heldout payload access: 0.",
        f"- Preserved earlier technical attempts: {len(preserved_failures)} (details in the final acceptance record).",
        "",
        "## Absolute macro losses",
        "",
        "| Method | Position MAE | Safe-FD L1 | Seam safe-FD L1 |",
        "|---|---:|---:|---:|",
    ]
    for method in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT", "S", "V", "T", "BIAS_STATE", "BIAS_VIDEO", "PERSISTENCE"):
        if method not in absolute:
            continue
        m = absolute[method]
        def fmt_sd(metric: str) -> str:
            sd = m[metric]["sample_sd"]
            return "shared" if sd is None else f"{m[metric]['mean']:.6f} ± {sd:.6f}"
        lines.append(
            f"| {method} | {fmt_sd('position_mae')} | {fmt_sd('safe_fd_l1')} | {fmt_sd('seam_safe_fd_l1')} |"
        )
    lines += [
        "",
        "Positive gain means reference loss minus candidate loss; positive values favor MIX_SOFT.",
        "",
    ]
    for contrast in ("MIX_VS_GT", "MIX_VS_SELF"):
        decision = decisions_summary[contrast]
        pos = decision["metrics"]["position_mae"]
        fd = decision["metrics"]["safe_fd_l1"]
        lines.append(
            f"- {contrast}: MAE gain {pos['point_gain']:+.6f} ± {pos['sample_sd_across_seeds']:.6f} "
            f"(95% CI [{pos['ci95_low']:+.6f}, {pos['ci95_high']:+.6f}]); "
            f"safe-FD gain {fd['point_gain']:+.6f} ± {fd['sample_sd_across_seeds']:.6f} "
            f"(95% CI [{fd['ci95_low']:+.6f}, {fd['ci95_high']:+.6f}]); "
            f"positive seeds MAE {pos['positive_seed_count']}/3, FD {fd['positive_seed_count']}/3; "
            f"POINT_JOINT_GAIN={decision['POINT_JOINT_GAIN']}; "
            f"CONDITIONAL_CI_JOINT_GAIN={decision['CONDITIONAL_CI_JOINT_GAIN']}."
        )
    lines += ["", "### Other paired references", "", "| Contrast | Position MAE gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |", "|---|---:|---:|---:|"]
    ci_map = {(row["contrast"], row["metric"]): row for row in tables["PAIRED_GROUP_CI.csv"] if row["scope"] == "macro"}
    for contrast in ("SELF_VS_GT", "MIX_VS_OLD_V", "MIX_VS_TEACHER", "MIX_VS_BIAS_STATE", "MIX_VS_BIAS_VIDEO"):
        cells = []
        for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1"):
            row = ci_map[(contrast, metric)]
            cells.append(f"{float(row['point_gain']):+.6f} [{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]")
        lines.append(f"| {contrast} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "The intervals are unadjusted exploratory source-group bootstrap intervals (5,000 draws, seed 2026101507), conditional on the three paired source instances and exposed dev groups. They are not p-values, equivalence intervals, or independent confirmation.", "", "## MAE teacher-gain retention", ""]
    for row in tables["TEACHER_RETENTION.csv"]:
        lines.append(
            f"- {row['scope']} {row['seed']}: R={row['retention_R']}; "
            f"numerator={float(row['numerator_LV_minus_student']):.6f}; "
            f"denominator={float(row['denominator_LV_minus_LT']):.6f}; "
            f"epsilon={float(row['epsilon_num']):.3e}."
        )
    lines += [
        "",
        "R is a descriptive MAE point estimate with no ratio CI. R=1 means the MIX_SOFT point loss reaches the teacher's MAE gain; it does not prove ensemble equivalence.",
        "",
        "## Evidence files",
        "",
        f"- Training target perturbations: {ROOT / 'results/TRAIN_TARGET_SIGNAL.csv'}",
        f"- Last-epoch train fit and imitation diagnostics: {ROOT / 'results/TRAIN_FIT_AND_IMITATION.csv'}",
        f"- Full training curves: {ROOT / 'results/TRAINING_CURVES.csv'}",
        f"- Task/seed rows: {ROOT / 'results/PRIMARY_TASK_SEED.csv'}",
        f"- Paired effects and intervals: {ROOT / 'results/PAIRED_EFFECTS.csv'} and {ROOT / 'results/PAIRED_GROUP_CI.csv'}",
        f"- Seed/task LOO: {ROOT / 'results/SEED_BLOCK_LOO.csv'} and {ROOT / 'results/TASK_LOO.csv'}",
        f"- Cached-input timing: {ROOT / 'results/INFERENCE_PROFILE.csv'}",
        f"- Resource/exposure ledgers: {ROOT / 'results/RESOURCE_LEDGER.csv'} and {ROOT / 'inventory/OUTCOME_EXPOSURE.json'}; CPU/RAM metadata: {ROOT / 'inventory/RUNTIME_ENV.json'}.",
        "- Host process peak RSS was not instrumented; system availability and materialized array payload sizes are recorded separately.",
        f"- V7 root: {ROOT}",
        f"- Final checksum receipt: {ROOT / 'qa/FINAL_CHECKSUM_VERIFICATION.json'}",
        "",
        "The conclusion is bounded to the fixed 0.5 soft-target ratio, this fusion-head architecture, recipe, and three paired source instances. It does not establish that video-only information caused a gain, that averaging is irreplaceable, or that development results transfer to an independent test set.",
    ]
    (ROOT / "RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(ROOT / "RETURN_TO_CHATGPT.json", {
        "schema_version": 1,
        "status": "results_complete_pending_visual_review_and_checksum",
        "study": "post_V6_single_fusion_head_supervised_soft_target_development",
        "new_head_fits_accepted": fit_index["accepted_fit_count"],
        "student_dev_prediction_jobs": pred_seal["accepted_student_jobs"],
        "formal_optimizer_updates": fit_index["total_accepted_optimizer_updates"],
        "primary_decisions": decisions_summary,
        "absolute_macro_losses": absolute,
        "teacher_retention": tables["TEACHER_RETENTION.csv"],
        "checksum_receipt_path": str(ROOT / "qa/FINAL_CHECKSUM_VERIFICATION.json"),
        "final_heldout_payload_access": 0,
        "independent_confirmation": False,
    })
    atomic_json(ROOT / "qa/FINAL_ACCEPTANCE.json", {
        "schema_version": 1,
        "status": "results_complete_pending_visual_review_and_checksum",
        "created_utc": utc_now(),
        "accepted_new_head_fits": 9,
        "expected_new_head_fits": 9,
        "accepted_student_dev_prediction_jobs": 9,
        "expected_student_dev_prediction_jobs": 9,
        "formal_epochs_per_fit": 160,
        "optimizer_updates_per_fit": 3200,
        "final_checkpoint_rule": "last epoch 160",
        "CPU_PREFLIGHT": "PASS",
        "STAGE0": "PASS",
        "TEACHER_BANK_SEAL": "PASS",
        "TARGET_LOCK": "LOCKED",
        "FIT_ACCEPTANCE_INDEX": "PASS",
        "GT_RETRAIN_VI_REPRODUCTION": "PASS",
        "PREDICTION_SEAL": "PASS",
        "PREDICTION_PREFLIGHT": read_json(ROOT / "qa/PREDICTION_PREFLIGHT.json")["status"],
        "SCORER_SEAL": "PASS",
        "reference_score_reconciliation": "PASS",
        "figure_build": figure_qa["status"],
        "figure_visual_review": "PENDING",
        "preserved_failed_attempts": preserved_failures,
        "checksum_receipt_path": "qa/FINAL_CHECKSUM_VERIFICATION.json",
        "checksum_receipt_status": "post-manifest receipt not written yet",
        "final_heldout_payload_access": 0,
        "old_root_writes": 0,
        "independent_confirmation": False,
    })
    fresh_reloads = [row.get("fresh_reload_receipt_path") for row in fit_index["entries"]]
    if len(fresh_reloads) != 9 or any(not Path(path).is_file() for path in fresh_reloads):
        raise RuntimeError("fresh-process safe reload receipts are incomplete")
    atomic_json(ROOT / "qa/TEACHER_STUDENT_QA.json", {
        "schema_version": 1,
        "status": "PASS",
        "cpu_preflight_path": str(ROOT / "qa/CPU_PREFLIGHT.json"),
        "cpu_preflight_sha256": sha_file(ROOT / "qa/CPU_PREFLIGHT.json"),
        "target_blind_replay_qa_path": str(ROOT / "qa/TARGET_BLIND_REPLAY_QA.json"),
        "target_blind_replay_qa_sha256": sha_file(ROOT / "qa/TARGET_BLIND_REPLAY_QA.json"),
        "target_blind_eval_replay": replay_qa["status"],
        "preserved_failed_attempts": preserved_failures,
        "stage0_path": str(ROOT / "qa/STAGE0.json"),
        "stage0_status": stage0["status"],
        "teacher_bank_seal_sha256": sha_file(ROOT / "protocol/TEACHER_BANK_SEAL.json"),
        "target_lock_sha256": sha_file(ROOT / "protocol/TARGET_LOCK.json"),
        "accepted_student_fits": fit_index["accepted_fit_count"],
        "GT_RETRAIN_byte_identical_to_accepted_V_i_all_seeds": True,
        "fresh_process_model_reload_receipts": 9,
        "prediction_seal_status": pred_seal["status"],
        "scorer_seal_status": scorer["status"],
        "prediction_preflight_path": str(ROOT / "qa/PREDICTION_PREFLIGHT.json"),
        "prediction_preflight_sha256": sha_file(ROOT / "qa/PREDICTION_PREFLIGHT.json"),
        "prediction_preflight_status": read_json(ROOT / "qa/PREDICTION_PREFLIGHT.json")["status"],
        "dev_target_access_receipt_path": str(ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json"),
        "dev_target_access_receipt_status": read_json(ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json")["status"],
        "dev_target_archive_load_operations": read_json(ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json")["archive_load_operations"],
        "final_heldout_payload_access": 0,
    })
    print(json.dumps({"status": "BUILT", "figures": len(figure_qa["figures"]), "results_summary": str(ROOT / "results/RESULTS_SUMMARY.json")}, sort_keys=True))


def seal() -> None:
    visual = read_json(ROOT / "qa/FIGURE_VISUAL_REVIEW.json")
    if visual.get("status") != "PASS":
        raise RuntimeError("figure visual review must be PASS before checksum sealing")
    acceptance_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
    acceptance = read_json(acceptance_path)
    if acceptance.get("status") != "results_complete_pending_visual_review_and_checksum":
        raise RuntimeError("unexpected final acceptance state before checksum seal")
    acceptance["status"] = "complete"
    acceptance["figure_visual_review"] = "PASS"
    acceptance["checksum_receipt_status"] = "post-manifest read-back receipt; see excluded receipt file"
    acceptance["completed_utc"] = utc_now()
    figure_build_path = ROOT / "qa/FIGURE_BUILD_QA.json"
    figure_build = read_json(figure_build_path)
    figure_build["status"] = "PASS"
    figure_build["visual_review_status"] = "PASS_ASSISTANT"
    figure_build["external_human_visual_review_status"] = "pending"
    atomic_json(figure_build_path, figure_build)
    atomic_json(acceptance_path, acceptance)
    for rel in ("RETURN_TO_CHATGPT.json", "results/RESULTS_SUMMARY.json"):
        path = ROOT / rel
        value = read_json(path)
        value["status"] = "complete"
        value["checksum_receipt_status"] = "post-manifest read-back receipt; see excluded receipt file"
        atomic_json(path, value)
    return_md = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
    content = return_md.read_text(encoding="utf-8")
    content += "\n\nExecution status: complete. The checksum read-back receipt is written after the manifest at qa/FINAL_CHECKSUM_VERIFICATION.json.\n"
    return_md.write_text(content, encoding="utf-8")
    owner_lock_path = ROOT / "coordination/OWNER.lock"
    owner_lock = read_json(owner_lock_path)
    owner_lock["owner_lock_acquired"] = False
    owner_lock["released_utc"] = utc_now()
    atomic_json(owner_lock_path, owner_lock)
    state_path = ROOT / "coordination/OWNER_STATE.json"
    state = read_json(state_path)
    state.pop("last_error", None)
    state.pop("failed_utc", None)
    state.update({
        "status": "complete",
        "phase": "checksum_seal",
        "owner_lock_acquired": False,
        "active_jobs": 0,
        "active_gpu_jobs": 0,
        "new_head_fits_accepted": 9,
        "student_dev_predictions_accepted": 9,
        "final_heldout_payload_access": 0,
        "completed_utc": utc_now(),
        "updated_utc": utc_now(),
    })
    atomic_json(state_path, state)

    manifest_path = ROOT / "CHECKSUMS.sha256"
    receipt_path = ROOT / "qa/FINAL_CHECKSUM_VERIFICATION.json"
    excluded = {"CHECKSUMS.sha256", "qa/FINAL_CHECKSUM_VERIFICATION.json"}
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded
    )
    manifest_lines = [f"{sha_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    manifest_sha = sha_file(manifest_path)
    verified = 0
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha_file(path)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
        else:
            verified += 1
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "manifest_path": "CHECKSUMS.sha256",
        "manifest_sha256": manifest_sha,
        "manifest_entry_count": len(files),
        "verified_entry_count": verified,
        "missing": missing,
        "mismatches": mismatches,
        "excluded_from_manifest": [
            {"path": "CHECKSUMS.sha256", "reason": "manifest cannot hash itself"},
            {"path": "qa/FINAL_CHECKSUM_VERIFICATION.json", "reason": "post-manifest read-back receipt"},
        ],
        "status": "PASS" if not missing and not mismatches and verified == len(files) else "FAIL",
    }
    atomic_json(receipt_path, receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"final checksum verification failed: {receipt}")
    print(json.dumps({
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "manifest_entries": len(files),
        "verified_entries": verified,
        "receipt_path": str(receipt_path),
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.build == args.seal:
        parser.error("choose exactly one of --build or --seal")
    if args.build:
        build()
    else:
        seal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

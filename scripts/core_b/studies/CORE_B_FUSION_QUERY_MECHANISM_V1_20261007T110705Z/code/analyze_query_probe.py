from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
MATRIX = ROOT / "protocol/MATRIX_4.csv"
MATCHED = ROOT / "protocol/MATCHED_BASELINE_4.csv"
PROTOCOL = ROOT / "protocol/PROTOCOL_LOCKED.json"
ACCEPT = ROOT / "protocol/B_FORMAL_ACCEPTANCE.json"
COLORS = {"TS11-D": "#0072B2", "STATE-CTRL": "#009E73"}
OLD_COLOR = "#0072B2"
NEW_COLOR = "#D55E00"
TARGET_COLOR = "#000000"
ANCHOR_COLOR = "#666666"
FIT_LABEL = "TRAIN-ONLY FIT PROBE — targets used in fitting"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def number(value: Any) -> float:
    if value is None or str(value) in ("", "NA", "nan", "NaN"):
        return float("nan")
    return float(value)

def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    vals = [number(x.get(field)) for x in rows]
    vals = [x for x in vals if math.isfinite(x)]
    return fmean(vals) if vals else float("nan")

def wcag_luminance(hex_color: str) -> float:
    c = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    rgb = [v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4 for v in c]
    return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]

def contrast_white(hex_color: str) -> float:
    lum = wcag_luminance(hex_color)
    return 1.05 / (lum + 0.05)

def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    formal_accept = json.loads(ACCEPT.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_BEFORE_FORMAL":
        raise AssertionError("formal analysis requires the locked formal protocol")
    if formal_accept.get("status") != "PASS" or formal_accept.get("accepted_cells") != 4:
        raise AssertionError("four formal cells are not accepted")
    matrix = load_csv(MATRIX)
    matched = load_csv(MATCHED)
    if len(matrix) != 4 or len(matched) != 4:
        raise AssertionError("analysis requires exactly four new and four matched old cells")

    new_sample_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    old_sample_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    new_curve_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    old_curve_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    predictions: dict[tuple[str, str, str], dict[str, Any]] = {}
    run_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []

    baseline_by_new = {x["new_run_id"]: x for x in matched}
    for cell in matrix:
        config, panel, run_id = cell["config"], cell["panel"], cell["run_id"]
        pair = baseline_by_new[run_id]
        old_id = pair["old_run_id"]
        old_dir = LEARN / "runs/formal" / old_id
        new_dir = ROOT / "runs/formal" / run_id
        old_run = json.loads((old_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        new_run = json.loads((new_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        old_reload = json.loads((old_dir / "FRESH_PROCESS_RELOAD.json").read_text(encoding="utf-8")) if (old_dir / "FRESH_PROCESS_RELOAD.json").exists() else None
        new_reload = json.loads((new_dir / "FRESH_PROCESS_RELOAD.json").read_text(encoding="utf-8"))
        if old_run.get("status") != "PASS" or new_run.get("status") != "PASS":
            raise AssertionError(f"unaccepted cell: {run_id}")
        if new_run.get("optimizer_updates") != 2000 or new_run.get("dev_targets_opened") is not False or new_run.get("final_or_heldout_access") != 0:
            raise AssertionError(f"new run scope/update mismatch: {run_id}")
        if new_run.get("parameter_count") != 890780 or new_run.get("candidate_trainable_parameter_count") != 0:
            raise AssertionError(f"new run changed parameter count: {run_id}")
        if not new_reload.get("prediction_bitwise_equal"):
            raise AssertionError(f"new checkpoint fresh-process reload failed: {run_id}")
        old_samples = load_csv(old_dir / "SAMPLE_METRICS.csv")
        new_samples = load_csv(new_dir / "SAMPLE_METRICS.csv")
        old_final = [x for x in old_samples if int(x["step"]) == 2000]
        new_final = [x for x in new_samples if int(x["step"]) == 2000]
        if len(old_final) != len(new_final) or len(new_final) != int(cell["panel"] == "P1") + 8 * int(cell["panel"] == "P8"):
            raise AssertionError(f"per-sample final metrics count mismatch: {run_id}")
        old_map = {x["anchor_id"]: x for x in old_final}
        new_map = {x["anchor_id"]: x for x in new_final}
        if set(old_map) != set(new_map):
            raise AssertionError(f"paired sample IDs differ: {run_id}")
        old_npz = np.load(old_dir / "PREDICTIONS_FINAL.npz", allow_pickle=False)
        new_npz = np.load(new_dir / "PREDICTIONS_FINAL.npz", allow_pickle=False)
        old_ids = old_npz["anchor_id"].astype("U96")
        new_ids = new_npz["anchor_id"].astype("U96")
        if old_ids.tolist() != new_ids.tolist():
            raise AssertionError(f"old/new prediction ID order differs: {run_id}")
        if not np.array_equal(old_npz["target_native"], new_npz["target_native"]):
            raise AssertionError(f"old/new train target values differ: {run_id}")
        if not np.array_equal(old_npz["anchor_native"], new_npz["anchor_native"]):
            raise AssertionError(f"old/new anchor values differ: {run_id}")
        old_curve = load_csv(old_dir / "METRICS.csv")
        new_curve = load_csv(new_dir / "METRICS.csv")
        key = (config, panel)
        old_sample_by_key[key] = old_samples
        new_sample_by_key[key] = new_samples
        old_curve_by_key[key] = old_curve
        new_curve_by_key[key] = new_curve
        predictions[(config, panel, "old")] = {
            "ids": old_ids, "target": old_npz["target_native"].astype(np.float32),
            "anchor": old_npz["anchor_native"].astype(np.float32),
            "prediction": old_npz["prediction_native"].astype(np.float32),
        }
        predictions[(config, panel, "new")] = {
            "ids": new_ids, "target": new_npz["target_native"].astype(np.float32),
            "anchor": new_npz["anchor_native"].astype(np.float32),
            "prediction": new_npz["prediction_native"].astype(np.float32),
        }
        run_records[(config, panel, "old")] = old_run
        run_records[(config, panel, "new")] = new_run
        for anchor_id in sorted(old_map):
            o, n = old_map[anchor_id], new_map[anchor_id]
            paired_rows.append({
                "config": config, "panel": panel, "anchor_id": anchor_id,
                "source_group_id": o["source_group_id"],
                "old_run_id": old_id, "new_run_id": run_id,
                "old_position_error": number(o["position_error_normalized"]),
                "new_position_error": number(n["position_error_normalized"]),
                "G_Q_position_old_minus_new": number(o["position_error_normalized"]) - number(n["position_error_normalized"]),
                "old_fd_error": number(o["fd_error_safe_scale_normalized"]),
                "new_fd_error": number(n["fd_error_safe_scale_normalized"]),
                "G_Q_fd_old_minus_new": number(o["fd_error_safe_scale_normalized"]) - number(n["fd_error_safe_scale_normalized"]),
                "old_position_gain_vs_persistence": number(o["position_gain_vs_persistence"]),
                "new_position_gain_vs_persistence": number(n["position_gain_vs_persistence"]),
                "old_fd_gain_vs_persistence": number(o["fd_gain_vs_persistence"]),
                "new_fd_gain_vs_persistence": number(n["fd_gain_vs_persistence"]),
                "old_tv_ratio": number(o["tv_ratio"]), "new_tv_ratio": number(n["tv_ratio"]),
                "old_direction_accuracy_active": number(o["direction_accuracy_active"]),
                "new_direction_accuracy_active": number(n["direction_accuracy_active"]),
                "old_within_time_std_native": number(o["within_sample_time_std_native"]),
                "new_within_time_std_native": number(n["within_sample_time_std_native"]),
                "old_strong_fit": o["strong_all_metrics_pass"],
                "new_strong_fit": n["strong_all_metrics_pass"],
                "target_used_in_fitting": True,
            })
        old_steps = {int(x["step"]): x for x in old_curve}
        new_steps = {int(x["step"]): x for x in new_curve}
        if set(old_steps) != set(new_steps):
            raise AssertionError(f"learning curve update grid differs: {run_id}")
        for step in sorted(old_steps):
            o, n = old_steps[step], new_steps[step]
            curve_rows.append({
                "config": config, "panel": panel, "step": step,
                "old_position_error": number(o["position_error_normalized_mean"]),
                "new_position_error": number(n["position_error_normalized_mean"]),
                "G_Q_position_old_minus_new": number(o["position_error_normalized_mean"]) - number(n["position_error_normalized_mean"]),
                "old_fd_error": number(o["fd_error_safe_scale_normalized_mean"]),
                "new_fd_error": number(n["fd_error_safe_scale_normalized_mean"]),
                "G_Q_fd_old_minus_new": number(o["fd_error_safe_scale_normalized_mean"]) - number(n["fd_error_safe_scale_normalized_mean"]),
                "old_position_gain_vs_persistence": number(o["position_gain_vs_persistence_mean"]),
                "new_position_gain_vs_persistence": number(n["position_gain_vs_persistence_mean"]),
                "old_fd_gain_vs_persistence": number(o["fd_gain_vs_persistence_mean"]),
                "new_fd_gain_vs_persistence": number(n["fd_gain_vs_persistence_mean"]),
                "old_tv_ratio": number(o["tv_ratio_mean"]), "new_tv_ratio": number(n["tv_ratio_mean"]),
                "old_direction_accuracy_active": number(o["direction_accuracy_active_mean"]),
                "new_direction_accuracy_active": number(n["direction_accuracy_active_mean"]),
                "old_within_time_std_native": number(o["within_sample_time_std_native_mean"]),
                "new_within_time_std_native": number(n["within_sample_time_std_native_mean"]),
            })
        old_strong = sum(x["strong_all_metrics_pass"] == "True" for x in old_final)
        new_strong = sum(x["strong_all_metrics_pass"] == "True" for x in new_final)
        summary_rows.append({
            "config": config, "panel": panel, "n_source_groups": len(old_final),
            "old_position_error_mean": mean_field(old_final, "position_error_normalized"),
            "new_position_error_mean": mean_field(new_final, "position_error_normalized"),
            "G_Q_position_old_minus_new": mean_field(old_final, "position_error_normalized") - mean_field(new_final, "position_error_normalized"),
            "old_fd_error_mean": mean_field(old_final, "fd_error_safe_scale_normalized"),
            "new_fd_error_mean": mean_field(new_final, "fd_error_safe_scale_normalized"),
            "G_Q_fd_old_minus_new": mean_field(old_final, "fd_error_safe_scale_normalized") - mean_field(new_final, "fd_error_safe_scale_normalized"),
            "old_position_gain_mean": mean_field(old_final, "position_gain_vs_persistence"),
            "new_position_gain_mean": mean_field(new_final, "position_gain_vs_persistence"),
            "old_fd_gain_mean": mean_field(old_final, "fd_gain_vs_persistence"),
            "new_fd_gain_mean": mean_field(new_final, "fd_gain_vs_persistence"),
            "old_tv_ratio_mean": mean_field(old_final, "tv_ratio"),
            "new_tv_ratio_mean": mean_field(new_final, "tv_ratio"),
            "old_direction_accuracy_mean": mean_field(old_final, "direction_accuracy_active"),
            "new_direction_accuracy_mean": mean_field(new_final, "direction_accuracy_active"),
            "old_time_std_native_mean": mean_field(old_final, "within_sample_time_std_native"),
            "new_time_std_native_mean": mean_field(new_final, "within_sample_time_std_native"),
            "old_strong_fit_count": old_strong, "new_strong_fit_count": new_strong,
            "new_all_samples_strong_fit": new_strong == len(new_final),
            "inference_scope": "train-only fit probe; targets used in fitting",
            "no_ci_no_p": True,
        })

    # First 100-step record satisfying the predeclared per-sample diagnostic gates.
    for variant, mapping in (("old_query", old_sample_by_key), ("orthogonal_query", new_sample_by_key)):
        for (config, panel), samples in mapping.items():
            by_id: dict[str, list[dict[str, str]]] = {}
            for row in samples:
                by_id.setdefault(row["anchor_id"], []).append(row)
            for anchor_id, series in by_id.items():
                series.sort(key=lambda x: int(x["step"]))
                first = {"position": None, "fd": None, "tv": None, "all": None}
                for row in series:
                    step = int(row["step"])
                    pos_ok = number(row["position_gain_vs_persistence"]) >= 0.90
                    fd_ok = number(row["fd_gain_vs_persistence"]) >= 0.80
                    tv = number(row["tv_ratio"])
                    tv_ok = math.isfinite(tv) and 0.8 <= tv <= 1.2
                    if pos_ok and first["position"] is None: first["position"] = step
                    if fd_ok and first["fd"] is None: first["fd"] = step
                    if tv_ok and first["tv"] is None: first["tv"] = step
                    if pos_ok and fd_ok and tv_ok and first["all"] is None: first["all"] = step
                crossing_rows.append({
                    "variant": variant, "config": config, "panel": panel, "anchor_id": anchor_id,
                    "first_position_gain_ge_0p90": first["position"] if first["position"] is not None else ">2000",
                    "first_fd_gain_ge_0p80": first["fd"] if first["fd"] is not None else ">2000",
                    "first_tv_ratio_in_0p8_to_1p2": first["tv"] if first["tv"] is not None else ">2000",
                    "first_all_three_strong_gates": first["all"] if first["all"] is not None else ">2000",
                    "resolution_updates": 100, "best_checkpoint_selection": False,
                })

    interaction_rows = []
    for panel in ("P1", "P8"):
        ts = next(x for x in summary_rows if x["config"] == "TS11-D" and x["panel"] == panel)
        st = next(x for x in summary_rows if x["config"] == "STATE-CTRL" and x["panel"] == panel)
        interaction_rows.append({
            "panel": panel, "nested_panel_warning": "P1 is nested inside P8; do not treat as independent repetition",
            "G_Q_position_TS11D": ts["G_Q_position_old_minus_new"],
            "G_Q_position_STATE_CTRL": st["G_Q_position_old_minus_new"],
            "G_Q_interaction_position_TS11D_minus_STATE_CTRL": ts["G_Q_position_old_minus_new"] - st["G_Q_position_old_minus_new"],
            "G_Q_fd_TS11D": ts["G_Q_fd_old_minus_new"],
            "G_Q_fd_STATE_CTRL": st["G_Q_fd_old_minus_new"],
            "G_Q_interaction_fd_TS11D_minus_STATE_CTRL": ts["G_Q_fd_old_minus_new"] - st["G_Q_fd_old_minus_new"],
            "descriptive_only": True, "statistical_test": "none",
        })

    RESULTS = ROOT / "results"
    write_csv(RESULTS / "QUERY_GEOMETRY_PER_SAMPLE.csv", paired_rows)
    write_csv(RESULTS / "QUERY_GEOMETRY_CURVE_COMPARISON.csv", curve_rows)
    write_csv(RESULTS / "QUERY_GEOMETRY_RUN_SUMMARY.csv", summary_rows)
    write_csv(RESULTS / "QUERY_GEOMETRY_THRESHOLD_CROSSINGS.csv", crossing_rows)
    write_csv(RESULTS / "QUERY_GEOMETRY_INTERACTIONS.csv", interaction_rows)

    # Metrics over the complete 100-update evaluation grid; no best-step selection.
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8}):
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), sharex=True, sharey="col", layout="constrained")
        series_styles = [
            ("TS11-D", "old_query", OLD_COLOR, "--", "o"),
            ("TS11-D", "orthogonal_query", OLD_COLOR, "-", "s"),
            ("STATE-CTRL", "old_query", "#009E73", "--", "o"),
            ("STATE-CTRL", "orthogonal_query", "#009E73", "-", "s"),
        ]
        metrics = [
            ("position_gain_vs_persistence", "Position gain vs Persistence", 0.90),
            ("fd_gain_vs_persistence", "FD gain vs Persistence", 0.80),
            ("tv_ratio", "Future-internal TV ratio", None),
        ]
        for ri, panel in enumerate(("P1", "P8")):
            for ci, (metric, title, threshold) in enumerate(metrics):
                ax = axes[ri, ci]
                for config, variant, color, linestyle, marker in series_styles:
                    rows = [x for x in curve_rows if x["config"] == config and x["panel"] == panel]
                    col = ("old_" if variant == "old_query" else "new_") + metric
                    ax.plot([int(x["step"]) for x in rows], [number(x[col]) for x in rows],
                            color=color, linestyle=linestyle, marker=marker, markevery=3,
                            linewidth=1.7, markersize=3.0, label=f"{config} {variant.replace('_',' ')}")
                if threshold is not None:
                    ax.axhline(threshold, color="#666666", linestyle=":", linewidth=1.0)
                if metric == "tv_ratio":
                    ax.axhspan(0.8, 1.2, color="#999999", alpha=0.12, zorder=0)
                    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0)
                ax.set_title(f"{panel}: {title}")
                ax.set_xlabel("Optimizer updates")
                if ci == 0:
                    ax.set_ylabel("Gain (normalized error reduction)")
                elif ci == 1:
                    ax.set_ylabel("Gain (normalized error reduction)")
                else:
                    ax.set_ylabel("TV(prediction) / TV(target)")
                ax.grid(axis="y", color="#dddddd", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(FIT_LABEL + "\n100-update train-only curves; no best-checkpoint selection", fontsize=12)
        fig.savefig(ROOT / "figures/QUERY_GEOMETRY_METRIC_CURVES.png", dpi=200, facecolor="white")
        plt.close(fig)

        # Train-target trajectory figures, P1 and P8 are shown separately and share scale within panel.
        for config in ("TS11-D", "STATE-CTRL"):
            for action_index in (0, 14):
                fig = plt.figure(figsize=(12.5, 8.2), layout="constrained")
                outer = fig.add_gridspec(2, 1, height_ratios=[1, 3])
                ax1 = fig.add_subplot(outer[0, 0])
                ax1.set_title("P1, one locked train source group")
                p1_old = predictions[(config, "P1", "old")]
                p1_new = predictions[(config, "P1", "new")]
                target = p1_new["target"][0, :, action_index]
                old_pred = p1_old["prediction"][0, :, action_index]
                new_pred = p1_new["prediction"][0, :, action_index]
                anchor = p1_new["anchor"][0, action_index]
                x = np.arange(1, 31)
                ax1.plot(x, target, color=TARGET_COLOR, linestyle="-", linewidth=2.0, label="train target")
                ax1.plot(x, old_pred, color=OLD_COLOR, linestyle="--", linewidth=1.7, label="old sinusoidal query")
                ax1.plot(x, new_pred, color=NEW_COLOR, linestyle="-.", linewidth=1.7, label="orthogonal query")
                ax1.axhline(anchor, color=ANCHOR_COLOR, linestyle=":", linewidth=1.2, label="last-command anchor (separate)")
                ax1.set_xlabel("Future step (30 Hz; 1 step = 1/30 s)")
                ax1.set_ylabel(f"Native value, action index {action_index}")
                ax1.grid(axis="y", color="#dddddd", linewidth=0.6)
                ax1.spines[["top", "right"]].set_visible(False)
                p8_grid = outer[1, 0].subgridspec(2, 4)
                axes8 = [fig.add_subplot(p8_grid[i // 4, i % 4], sharey=None) for i in range(8)]
                p8_old = predictions[(config, "P8", "old")]
                p8_new = predictions[(config, "P8", "new")]
                all_y = np.concatenate([
                    p8_old["target"][:, :, action_index].ravel(),
                    p8_old["prediction"][:, :, action_index].ravel(),
                    p8_new["prediction"][:, :, action_index].ravel(),
                ])
                lo, hi = float(np.min(all_y)), float(np.max(all_y))
                margin = max((hi - lo) * 0.05, 1e-6)
                for i, ax in enumerate(axes8):
                    target = p8_new["target"][i, :, action_index]
                    old_pred = p8_old["prediction"][i, :, action_index]
                    new_pred = p8_new["prediction"][i, :, action_index]
                    anchor = p8_new["anchor"][i, action_index]
                    aid = str(p8_new["ids"][i])
                    ax.plot(x, target, color=TARGET_COLOR, linestyle="-", linewidth=1.5)
                    ax.plot(x, old_pred, color=OLD_COLOR, linestyle="--", linewidth=1.3)
                    ax.plot(x, new_pred, color=NEW_COLOR, linestyle="-.", linewidth=1.3)
                    ax.axhline(anchor, color=ANCHOR_COLOR, linestyle=":", linewidth=0.9)
                    ax.set_title(aid, fontsize=7)
                    ax.set_ylim(lo - margin, hi + margin)
                    ax.set_xticks([1, 10, 20, 30])
                    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
                    ax.spines[["top", "right"]].set_visible(False)
                    if i // 4 == 1:
                        ax.set_xlabel("Future step")
                    if i % 4 == 0:
                        ax.set_ylabel("Native value")
                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
                fig.suptitle(f"{FIT_LABEL}\n{config} | H30 | FULL_LOSS | action index {action_index}; all P1/P8 targets used in fitting", fontsize=12)
                fig.savefig(ROOT / f"figures/QUERY_GEOMETRY_{config}_ACTION_{action_index}_P1_P8.png", dpi=200, facecolor="white")
                plt.close(fig)

        p8_rows = [x for x in paired_rows if x["panel"] == "P8"]
        ids = sorted({x["anchor_id"] for x in p8_rows})
        configs = ["TS11-D", "STATE-CTRL"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), layout="constrained")
        for ax, key, title in [
            (axes[0], "G_Q_position_old_minus_new", "Position G_Q: old error − orthogonal error"),
            (axes[1], "G_Q_fd_old_minus_new", "FD G_Q: old error − orthogonal error"),
        ]:
            gq_values = np.array([[number(next(x[key] for x in p8_rows if x["config"] == config and x["anchor_id"] == aid))
                                for config in configs] for aid in ids], dtype=float)
            bound = max(float(np.max(np.abs(gq_values))), 1e-9)
            image = ax.imshow(gq_values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), aspect="auto")
            ax.set_xticks(range(len(configs)), configs)
            ax.set_yticks(range(len(ids)), [f"sample {i+1}" for i in range(len(ids))])
            ax.set_title(title)
            ax.set_xlabel("Model")
            ax.set_ylabel("Locked P8 train sample")
            for i in range(gq_values.shape[0]):
                for j in range(gq_values.shape[1]):
                    ax.text(j, i, f"{gq_values[i,j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
            fig.colorbar(image, ax=ax, label="Positive means lower error with orthogonal query")
        fig.suptitle(FIT_LABEL + "\nP8 per-sample paired differences; targets used in fitting", fontsize=12)
        fig.savefig(ROOT / "figures/QUERY_GEOMETRY_P8_PER_SAMPLE_GQ.png", dpi=200, facecolor="white")
        plt.close(fig)

    # Summary interactions are descriptive only; P1 is nested in P8.
    pos_both = all(x["G_Q_position_old_minus_new"] > 0 for x in summary_rows)
    fd_both = all(x["G_Q_fd_old_minus_new"] > 0 for x in summary_rows)
    tv_all_in_band = all(0.8 <= x["new_tv_ratio_mean"] <= 1.2 for x in summary_rows)
    all_fit = all(x["new_all_samples_strong_fit"] for x in summary_rows)
    if pos_both and fd_both and tv_all_in_band and all_fit:
        interpretation = "Both models improved train position and FD fit with TV in band across P1/P8. This supports query-geometry/optimization sensitivity in this diagnostic setting, not improved video representation or generalization."
    elif pos_both and not fd_both:
        interpretation = "Position fit improved, but FD gains were not consistent across the four cells; do not declare a multi-metric query improvement."
    else:
        interpretation = "The continuous fit gains are mixed or incomplete; stop this query route and do not add scale/frequency/LR/seed variants."
    report = [
        "# Query geometry train-fit probe",
        "",
        f"Status: {formal_accept['status']} ({formal_accept['accepted_cells']}/4 formal cells, 2,000 updates each).",
        "",
        "All values below are train-only fit diagnostics. The P1/P8 targets were used in fitting; P1 is nested in P8. There are no confidence intervals, p-values, or seed-stability claims.",
        "",
        "## Paired update-2000 results",
        "",
        "| Model | Panel | n | Old position | New position | G_Q position | Old FD | New FD | G_Q FD | Old TV | New TV | Strong old | Strong new |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary_rows:
        report.append(
            f"| {x['config']} | {x['panel']} | {x['n_source_groups']} | "
            f"{x['old_position_error_mean']:.6f} | {x['new_position_error_mean']:.6f} | {x['G_Q_position_old_minus_new']:+.6f} | "
            f"{x['old_fd_error_mean']:.6f} | {x['new_fd_error_mean']:.6f} | {x['G_Q_fd_old_minus_new']:+.6f} | "
            f"{x['old_tv_ratio_mean']:.3f} | {x['new_tv_ratio_mean']:.3f} | "
            f"{x['old_strong_fit_count']}/{x['n_source_groups']} | {x['new_strong_fit_count']}/{x['n_source_groups']} |"
        )
    report += [
        "",
        "G_Q is the same-panel old-query error minus orthogonal-query error; positive favors the orthogonal query for that fitted panel. The model interaction is G_Q(TS11-D) minus G_Q(STATE-CTRL), and is descriptive only.",
        "",
        "## Model interaction",
        "",
        "| Panel | G_Q position TS11-D | G_Q position STATE-CTRL | Difference | G_Q FD TS11-D | G_Q FD STATE-CTRL | Difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in interaction_rows:
        report.append(f"| {x['panel']} | {x['G_Q_position_TS11D']:+.6f} | {x['G_Q_position_STATE_CTRL']:+.6f} | {x['G_Q_interaction_position_TS11D_minus_STATE_CTRL']:+.6f} | {x['G_Q_fd_TS11D']:+.6f} | {x['G_Q_fd_STATE_CTRL']:+.6f} | {x['G_Q_interaction_fd_TS11D_minus_STATE_CTRL']:+.6f} |")
    report += [
        "",
        "## Interpretation and boundary",
        "",
        interpretation,
        "",
        "The single Hadamard contrast changes time distance, spectrum, smoothness, and conditioning together. RMS and parameter-count parity do not isolate rank as the cause. The table and figures cannot establish development performance, forecasting generalization, seed stability, or a benchmark winner.",
        "",
        "The predeclared per-sample position ≥0.90 gain, FD ≥0.80 gain, and TV ratio 0.8–1.2 gates are reported with the first observed 100-update crossing; rows that never cross are marked >2000. No best checkpoint is selected.",
        "",
        "All four formal cells were freshly reloaded in another process and their train-panel predictions reproduced bitwise. The model parameter count remains 890,780; the candidate query is a fixed buffer and adds zero trainable parameters.",
        "",
        "Raw action axes use numeric indices only. Physical units and anatomical mappings are unresolved.",
        "",
        "## Data and figures",
        "",
        "- Paired per-sample metrics: results/QUERY_GEOMETRY_PER_SAMPLE.csv",
        "- Complete 100-update old/new curves: results/QUERY_GEOMETRY_CURVE_COMPARISON.csv",
        "- First-threshold crossings: results/QUERY_GEOMETRY_THRESHOLD_CROSSINGS.csv",
        "- P1/P8 trajectories by raw action index: figures/QUERY_GEOMETRY_TS11-D_ACTION_0_P1_P8.png, figures/QUERY_GEOMETRY_TS11-D_ACTION_14_P1_P8.png, figures/QUERY_GEOMETRY_STATE-CTRL_ACTION_0_P1_P8.png, figures/QUERY_GEOMETRY_STATE-CTRL_ACTION_14_P1_P8.png",
        "- Per-sample P8 paired errors: figures/QUERY_GEOMETRY_P8_PER_SAMPLE_GQ.png",
        "- Metric curves: figures/QUERY_GEOMETRY_METRIC_CURVES.png",
        "",
        "Figure captions and alt descriptions are in reports/FIGURE_CAPTIONS_AND_ALT.zh-CN.md. Every figure is labeled as a train-only fit probe.",
        "",
        "## Next work",
        "",
        "No additional query-scale, frequency, LR, or seed sweep is recommended from these four single-seed cells. A separate full-data validation proposal, if retained, must lock its target population, multi-seed design, independent dev scoring, primary metrics, and held-out exclusion before training; this report does not execute that proposal.",
        "",
    ]
    (ROOT / "reports/QUERY_GEOMETRY_TRAIN_FIT_REPORT.zh-CN.md").write_text("\n".join(report), encoding="utf-8")

    caps = [
        {"file":"figures/QUERY_GEOMETRY_METRIC_CURVES.png","caption":"Train-only update curves for old sinusoidal and RMS-matched orthogonal future-query encodings across TS11-D and STATE-CTRL, shown separately for P1 and P8. Position and safe-FD gains are relative to per-sample Persistence; the future-internal TV ratio is shown with the locked 0.8–1.2 band. Targets were used in fitting. No confidence interval or p-value is shown.","alt":"Three aligned metric columns show old-query dashed curves and orthogonal-query solid curves for TS11-D and STATE-CTRL, with P1 and P8 in separate rows over 2,000 updates. The panels report position gain, FD gain, and future TV ratio. Every curve is fitted to the train targets."},
        {"file":"figures/QUERY_GEOMETRY_TS11-D_ACTION_0_P1_P8.png","caption":"TS11-D, H30, raw action index 0. P1 and P8 compare the train target, old-query prediction, orthogonal-query prediction, and last-command anchor reference. Targets were used in fitting.","alt":"One P1 plot and eight P8 small multiples show native future values at raw action index 0 for TS11-D. Each compares the train target with old sinusoidal and orthogonal-query fits; a separate horizontal reference marks the last-command anchor. These are train-only fits."},
        {"file":"figures/QUERY_GEOMETRY_TS11-D_ACTION_14_P1_P8.png","caption":"TS11-D, H30, raw action index 14. P1 and P8 compare the train target, old-query prediction, orthogonal-query prediction, and last-command anchor reference. Targets were used in fitting.","alt":"One P1 plot and eight P8 small multiples show native future values at raw action index 14 for TS11-D, comparing train targets with old-query and orthogonal-query fitted trajectories and a separate anchor reference."},
        {"file":"figures/QUERY_GEOMETRY_STATE-CTRL_ACTION_0_P1_P8.png","caption":"STATE-CTRL, H30, raw action index 0. P1 and P8 compare the train target, old-query prediction, orthogonal-query prediction, and last-command anchor reference. Targets were used in fitting.","alt":"One P1 plot and eight P8 small multiples show native future values at raw action index 0 for STATE-CTRL, comparing train targets with old-query and orthogonal-query fitted trajectories and a separate anchor reference."},
        {"file":"figures/QUERY_GEOMETRY_STATE-CTRL_ACTION_14_P1_P8.png","caption":"STATE-CTRL, H30, raw action index 14. P1 and P8 compare the train target, old-query prediction, orthogonal-query prediction, and last-command anchor reference. Targets were used in fitting.","alt":"One P1 plot and eight P8 small multiples show native future values at raw action index 14 for STATE-CTRL, comparing train targets with old-query and orthogonal-query fitted trajectories and a separate anchor reference."},
        {"file":"figures/QUERY_GEOMETRY_P8_PER_SAMPLE_GQ.png","caption":"P8 per-sample old-query error minus orthogonal-query error for position and FD. Positive values favor the orthogonal query on the fitted training samples. This is a descriptive train-only display with no interval or significance test.","alt":"Two diverging heatmaps show the position and FD paired error differences for eight locked P8 train samples in TS11-D and STATE-CTRL. Positive cells mean lower error under the orthogonal query. All targets were used in fitting."},
    ]
    (ROOT/"reports/FIGURE_CAPTIONS_AND_ALT.zh-CN.md").write_text(
        "# Figure captions and alt descriptions\n\n" +
        "\n\n".join(f"## {x['file']}\n\nCaption: {x['caption']}\n\nAlt: {x['alt']}" for x in caps) +
        "\n\nPalette uses blue and green for model families, blue dashed versus orange dash-dot for old versus orthogonal query, plus marker and line-style redundancy. No uncertainty intervals are shown because the diagnostic is one seed on targets used in fitting. Publisher requirements were not specified; these are provisional general-purpose PNGs.\n",
        encoding="utf-8",
    )
    # Minimal truthful figure metadata and palette checks; no publisher compliance assertion.
    figures = []
    for path in sorted((ROOT/"figures").glob("*.png")):
        with Image.open(path) as image:
            figures.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size,
                            "width_px":image.width,"height_px":image.height,"mode":image.mode,
                            "dpi":image.info.get("dpi"),"format":image.format})
    palette = [OLD_COLOR, NEW_COLOR, COLORS["STATE-CTRL"], TARGET_COLOR, ANCHOR_COLOR]
    qa = {
        "status":"PASS_FIGURE_METADATA_AND_PALETTE_SCREEN",
        "matplotlib_version":matplotlib.__version__,
        "figures":figures,"contrast_ratio_vs_white":{c:contrast_white(c) for c in palette},
        "redundant_encoding":"query variant uses line style and marker as well as color; target and anchor use different colors/styles",
        "target_used_in_fitting_disclosed":True,"uncertainty_intervals_added":False,
        "publisher_compliance":"not assessed; no target journal or submission phase was specified",
        "manual_visual_review":"pending",
    }
    (ROOT/"protocol/FIGURE_QA.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2,sort_keys=True)+chr(10),encoding="utf-8")
    analysis = {
        "schema_version":1,"status":"PASS_PAIRED_TRAIN_ONLY_ANALYSIS",
        "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "protocol_sha256":sha(PROTOCOL),"formal_acceptance_sha256":sha(ACCEPT),
        "matrix_sha256":sha(MATRIX),"matched_baseline_sha256":sha(MATCHED),
        "formal_run_ids":[x["run_id"] for x in matrix],
        "new_cells_accepted":len(matrix),"optimizer_updates_per_cell":2000,
        "paired_sample_rows":len(paired_rows),"curve_rows":len(curve_rows),
        "gq_interactions":interaction_rows,"interpretation":interpretation,
        "independent_confirmation":False,"new_dev_model_selection":False,
        "new_full_data_benchmark_training":False,"p_values":False,"confidence_intervals":False,
        "dev_targets_opened":False,"final_or_heldout_access":0,"gpu_queries_use_lease":0,
        "query_table_sha256_h30":protocol["query_table_sha256_h30"],
        "query_intervention":"fixed H30 Hadamard future-query table only",
        "new_benchmark_candidate":False,
        "figures":[x["path"] for x in figures],
    }
    (ROOT/"protocol/POST_RUN_ANALYSIS_ACCEPTANCE.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2,sort_keys=True)+chr(10),encoding="utf-8")
    print(json.dumps({"status":analysis["status"],"cells":analysis["new_cells_accepted"],
                      "paired_sample_rows":analysis["paired_sample_rows"],"curve_rows":analysis["curve_rows"],
                      "gq_interactions":interaction_rows,"interpretation":interpretation,
                      "figures":analysis["figures"],"palette_qa":qa["status"]},ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Audit eight locked train-only tiny-fits, produce summary tables and figures."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/formal"
OUT_RESULTS = ROOT / "results"
OUT_FIGURES = ROOT / "figures"
MATRIX = ROOT / "protocol/B_MATRIX_8.csv"
PANEL_LOCK = ROOT / "protocol/B_PANEL_SELECTION_LOCK.json"
PROTOCOL = ROOT / "protocol/B_PROTOCOL_LOCKED.json"
EXECUTION = ROOT / "protocol/B_EXECUTION_LOCK.json"
RESOURCE_GATE = ROOT / "protocol/B_RESOURCE_GATE.json"
DIAG_LABEL = "TRAIN-ONLY MEMORIZATION DIAGNOSTIC — targets used in fitting"
COLORS = {"TS11-D": "#0072B2", "STATE-CTRL": "#D55E00", "TARGET": "#111111", "ANCHOR": "#777777"}
LOSSES = {"FULL_LOSS": {"linestyle": "-", "marker": "o"}, "POS_ONLY": {"linestyle": "--", "marker": "s"}}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def load_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matrix = list(csv.DictReader(MATRIX.open(newline="", encoding="utf-8")))
    panel_lock = read_json(PANEL_LOCK)
    expected_ids = {str(x["anchor_id"]) for x in panel_lock["rows"] if x["panel"] == "P8"}
    final_runs: list[dict[str, Any]] = []
    final_samples: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    expected_protocol_hash = sha_file(PROTOCOL)
    expected_panel_hash = sha_file(PANEL_LOCK)
    for row in matrix:
        run_id = row["run_id"]
        run_root = RUNS / run_id
        complete = read_json(run_root / "RUN_COMPLETE.json")
        if (complete.get("status") != "PASS" or int(complete.get("optimizer_updates", -1)) != 2000
                or complete.get("protocol_sha256") != expected_protocol_hash
                or complete.get("panel_lock_sha256") != expected_panel_hash
                or complete.get("init_state_file_sha256") != panel_lock["init"]["file_sha256"]
                or complete.get("dev_targets_opened") is not False
                or int(complete.get("final_or_heldout_access", -1)) != 0):
            raise AssertionError(f"run completion/source/scope acceptance failed: {run_id}")
        final_path = run_root / "checkpoint_final.pt"
        pred_path = run_root / "PREDICTIONS_FINAL.npz"
        if sha_file(final_path) != complete["final_checkpoint_sha256"] or sha_file(pred_path) != complete["prediction_sha256"]:
            raise AssertionError(f"final artifact hash mismatch: {run_id}")
        metrics = list(csv.DictReader((run_root / "METRICS.csv").open(newline="", encoding="utf-8")))
        samples = list(csv.DictReader((run_root / "SAMPLE_METRICS.csv").open(newline="", encoding="utf-8")))
        if len(metrics) != 20 or [int(x["step"]) for x in metrics] != list(range(100, 2001, 100)):
            raise AssertionError(f"evaluation curve is incomplete or not fixed every100: {run_id}")
        last_samples = [x for x in samples if int(x["step"]) == 2000]
        expected_ids_this = {str(x["anchor_id"]) for x in panel_lock["rows"] if x["panel"] == row["panel"]}
        if len(last_samples) != (1 if row["panel"] == "P1" else 8) or {x["anchor_id"] for x in last_samples} != expected_ids_this:
            raise AssertionError(f"final sample metrics do not cover exact locked panel: {run_id}")
        run_fit = all(as_bool(x["strong_all_metrics_pass"]) for x in last_samples)
        if run_fit != bool(complete["strong_training_fit"]):
            raise AssertionError(f"run-level strong-fit label disagrees with per-sample rows: {run_id}")
        mean_pos = float(np.mean([float(x["position_error_normalized"]) for x in last_samples]))
        mean_pos_persist = float(np.mean([float(x["persistence_position_error_normalized"]) for x in last_samples]))
        mean_fd = float(np.mean([float(x["fd_error_safe_scale_normalized"]) for x in last_samples]))
        mean_fd_persist = float(np.mean([float(x["persistence_fd_error_safe_scale_normalized"]) for x in last_samples]))
        mean_tv = float(np.mean([float(x["tv_ratio"]) for x in last_samples]))
        mean_direction = np.asarray([float(x["direction_accuracy_active"]) for x in last_samples if x["direction_accuracy_active"] != "NA"])
        summary = {
            "run_id": run_id, "config": row["config"], "loss": row["loss"], "panel": row["panel"],
            "seed": int(row["seed"]), "updates": int(complete["optimizer_updates"]),
            "elapsed_seconds": float(complete["elapsed_seconds"]),
            "mean_update_seconds": float(complete["mean_update_seconds"]),
            "panel_rows": len(last_samples),
            "position_error_normalized_mean": mean_pos,
            "persistence_position_error_normalized_mean": mean_pos_persist,
            "position_gain_vs_persistence_mean": 1.0 - mean_pos / mean_pos_persist if mean_pos_persist else "NA",
            "fd_error_safe_scale_normalized_mean": mean_fd,
            "persistence_fd_error_safe_scale_normalized_mean": mean_fd_persist,
            "fd_gain_vs_persistence_mean": 1.0 - mean_fd / mean_fd_persist if mean_fd_persist else "NA",
            "prediction_tv_over_target_tv_mean": mean_tv,
            "direction_accuracy_active_mean": float(mean_direction.mean()) if mean_direction.size else "NA",
            "direction_active_steps_total": int(sum(int(x["direction_active_steps"]) for x in last_samples)),
            "within_sample_time_std_native_mean": float(np.mean([float(x["within_sample_time_std_native"]) for x in last_samples])),
            "cross_sample_time_mean_offset_std_native": float(metrics[-1]["cross_sample_time_mean_offset_std_native"]),
            "per_sample_strong_fit_count": sum(as_bool(x["strong_all_metrics_pass"]) for x in last_samples),
            "strong_training_fit": run_fit,
            "final_checkpoint_sha256": complete["final_checkpoint_sha256"],
            "prediction_sha256": complete["prediction_sha256"],
            "new_benchmark_candidate": False,
        }
        final_runs.append(summary)
        for sample in last_samples:
            final_samples.append({
                **{k: summary[k] for k in ("run_id", "config", "loss", "panel", "seed")},
                **{k: sample[k] for k in sample if k not in {"run_id", "step"}},
                "strong_all_metrics_pass": as_bool(sample["strong_all_metrics_pass"]),
            })
        for metric_row in metrics:
            curves.append({
                **{k: row[k] for k in ("run_id", "config", "loss", "panel", "seed")},
                **metric_row,
            })
        with np.load(pred_path, allow_pickle=False) as z:
            ids = z["anchor_id"].astype("U96")
            if ids.tolist() != [x["anchor_id"] for x in last_samples]:
                raise AssertionError(f"final prediction array IDs/order differ from metrics: {run_id}")
            if row["panel"] == "P8" and set(ids.tolist()) != expected_ids:
                raise AssertionError(f"P8 final prediction IDs differ from locked panel: {run_id}")
            predictions[run_id] = {
                "ids": ids, "target": z["target_native"].astype(np.float32),
                "anchor": z["anchor_native"].astype(np.float32),
                "prediction": z["prediction_native"].astype(np.float32),
            }
    return final_runs, final_samples, {"curves": curves, "predictions": predictions, "panel_lock": panel_lock, "matrix": matrix}


def series_key(config: str, loss: str, panel: str) -> str:
    return f"TINYFIT-{config}-H30-{loss}-{panel}-S20261002"


def trajectory_figures(predictions: dict[str, dict[str, Any]], panel_lock: dict[str, Any]) -> list[dict[str, Any]]:
    figures = []
    h = np.arange(1, 31)
    p1_runs = [series_key(config, loss, "P1") for config in ("TS11-D", "STATE-CTRL")
               for loss in ("FULL_LOSS", "POS_ONLY")]
    # One P1 figure, with raw coordinates 0 and 14, and shared scales within each column.
    p1_values = {0: [], 14: []}
    for run_id in p1_runs:
        item = predictions[run_id]
        for coord in (0, 14):
            p1_values[coord].extend(item["target"][0, :, coord].tolist())
            p1_values[coord].extend(item["prediction"][0, :, coord].tolist())
            p1_values[coord].append(float(item["anchor"][0, coord]))
    ylims = {}
    for coord, vals in p1_values.items():
        low, high = min(vals), max(vals)
        margin = max((high-low)*0.06, 1e-6)
        ylims[coord] = (low-margin, high+margin)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, layout="constrained")
    for row_i, run_id in enumerate(p1_runs):
        item = predictions[run_id]
        config = "TS11-D" if "TS11-D" in run_id else "STATE-CTRL"
        loss = "FULL_LOSS" if "FULL_LOSS" in run_id else "POS_ONLY"
        style = LOSSES[loss]
        for col_i, coord in enumerate((0, 14)):
            ax = axes[row_i, col_i]
            ax.scatter([0], [float(item["anchor"][0, coord])], marker="x", s=28, color=COLORS["ANCHOR"], label="anchor (separate seam)")
            ax.plot(h, item["target"][0, :, coord], color=COLORS["TARGET"], linestyle=":", linewidth=1.7, label="future target (used in fitting)")
            ax.plot(h, item["prediction"][0, :, coord], color=COLORS[config], linestyle=style["linestyle"],
                    marker=style["marker"], markevery=5, linewidth=1.6, label=f"{config} / {loss}")
            ax.set_ylim(*ylims[coord])
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{config} · {loss} · raw coordinate {coord}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel("Native command value (unit unspecified)")
            if row_i == 3:
                ax.set_xlabel("Future query index (anchor at 0 is not joined)")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"P1 · 1 train sample · {DIAG_LABEL}", fontsize=13)
    path = OUT_FIGURES / "TINYFIT_P1_TRAJECTORIES.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    figures.append({"path": str(path.relative_to(ROOT)), "caption": "P1 final future trajectories for the four model/loss cells; raw coordinate indices 0 and 14, with the anchor shown separately at index 0 and not connected to future steps.",
                    "alt_text": "Eight panels compare the fitted 30-step train target with final TS11-D or STATE-CTRL predictions under FULL_LOSS or POS_ONLY for the same single P1 training sample. Both the target and the prediction are training data; the anchor is shown separately, with no seam line."})

    # P8 trajectories by model at raw coordinate 0; all eight sample axes share y limits per figure.
    p8_rows = [x for x in panel_lock["rows"] if x["panel"] == "P8"]
    for config in ("TS11-D", "STATE-CTRL"):
        run_full = series_key(config, "FULL_LOSS", "P8")
        run_pos = series_key(config, "POS_ONLY", "P8")
        full, pos = predictions[run_full], predictions[run_pos]
        values = [float(v) for item in (full, pos) for key in ("target", "prediction") for v in item[key][:, :, 0].ravel()]
        low, high = min(values), max(values); margin = max((high-low)*0.06, 1e-6)
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), sharex=True, sharey=True, layout="constrained")
        for i, ax in enumerate(axes.flat):
            anchor = float(full["anchor"][i, 0])
            ax.scatter([0], [anchor], marker="x", s=24, color=COLORS["ANCHOR"], label="anchor (separate seam)")
            ax.plot(h, full["target"][i, :, 0], color=COLORS["TARGET"], linestyle=":", linewidth=1.5, label="target (used in fitting)")
            ax.plot(h, full["prediction"][i, :, 0], color=COLORS[config], linestyle="-", marker="o", markevery=7,
                    linewidth=1.3, label="FULL_LOSS")
            ax.plot(h, pos["prediction"][i, :, 0], color=COLORS[config], linestyle="--", marker="s", markevery=7,
                    linewidth=1.3, label="POS_ONLY")
            ax.set_ylim(low-margin, high+margin); ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"P8 {i+1}: {p8_rows[i]['anchor_id']}", fontsize=8)
            if i // 4 == 1: ax.set_xlabel("Future query index")
            if i % 4 == 0: ax.set_ylabel("Native coordinate 0 (unit unspecified)")
            if i == 0: ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"P8 · {config} · 8 train samples · {DIAG_LABEL}", fontsize=13)
        path = OUT_FIGURES / f"TINYFIT_P8_{config.replace('-', '_')}_TRAJECTORIES.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        figures.append({"path": str(path.relative_to(ROOT)),
                        "caption": f"P8 final train trajectories for {config}, raw coordinate 0. Each panel is one locked training sample; target, FULL_LOSS, and POS_ONLY are shown at update 2000.",
                        "alt_text": f"Eight small multiples for {config}. For each of the eight fitted train samples, the 30-step target is compared with the final FULL_LOSS and POS_ONLY predictions. The anchor is a separate point at index zero; targets were used during fitting."})
    return figures


def metric_figures(curves: list[dict[str, Any]], final_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, layout="constrained")
    metrics = [
        ("position_gain_vs_persistence_mean", "Position error reduction vs Persistence", 0.90),
        ("fd_gain_vs_persistence_mean", "FD error reduction vs Persistence", 0.80),
        ("tv_ratio_mean", "Prediction / target TV ratio", None),
    ]
    all_values: dict[str, list[float]] = {key: [] for key, _, _ in metrics}
    for row in curves:
        for key, _, _ in metrics:
            all_values[key].append(float(row[key]))
    for col, (metric, label, threshold) in enumerate(metrics):
        finite = np.asarray([x for x in all_values[metric] if np.isfinite(x)], dtype=float)
        lower = min(-0.15, float(finite.min()) - 0.08) if finite.size else -0.15
        upper = max(1.15, float(finite.max()) + 0.08) if finite.size else 1.25
        if metric == "tv_ratio_mean":
            lower = max(0.0, min(lower, float(finite.min()) - 0.08))
            upper = max(1.3, float(finite.max()) + 0.08)
        for row_i, panel in enumerate(("P1", "P8")):
            ax = axes[row_i, col]
            panel_rows = [x for x in curves if x["panel"] == panel]
            for config in ("TS11-D", "STATE-CTRL"):
                for loss in ("FULL_LOSS", "POS_ONLY"):
                    series = sorted([x for x in panel_rows if x["config"] == config and x["loss"] == loss], key=lambda x: int(x["step"]))
                    spec = LOSSES[loss]
                    ax.plot([int(x["step"]) for x in series], [float(x[metric]) for x in series],
                            color=COLORS[config], linestyle=spec["linestyle"], marker=spec["marker"],
                            markevery=4, linewidth=1.5, markersize=3.5,
                            label=f"{config} · {loss}")
            if threshold is not None:
                ax.axhline(threshold, color="#333333", linestyle=":", linewidth=1.2, label="predeclared threshold")
            else:
                ax.axhline(0.8, color="#555555", linestyle=":", linewidth=1.0, label="TV lower bound")
                ax.axhline(1.2, color="#555555", linestyle="-.", linewidth=1.0, label="TV upper bound")
            ax.set_ylim(lower, upper); ax.grid(alpha=0.2)
            ax.set_title(f"{panel}: {label}", fontsize=9)
            if col == 0: ax.set_ylabel("Fraction / ratio")
            if row_i == 1: ax.set_xlabel("Optimizer updates")
            if row_i == 0 and col == 0: ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Fixed-panel learning curves · all targets used in fitting · {DIAG_LABEL}", fontsize=13)
    path = OUT_FIGURES / "TINYFIT_METRIC_CURVES.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    figures.append({"path": str(path.relative_to(ROOT)),
                    "caption": "Position and FD error reduction relative to per-sample Persistence and future-internal TV ratio over 2000 updates. Lines are separate model/loss cells; P1 and P8 are shown in separate rows. Horizontal lines are the predeclared diagnostic thresholds.",
                    "alt_text": "A 2 by 3 grid of train-only learning curves for P1 and P8. The three metrics are normalized position gain, safe-FD gain, and prediction-to-target total-variation ratio. Solid versus dashed lines distinguish FULL_LOSS and POS_ONLY; blue versus orange distinguishes TS11-D and STATE-CTRL. Targets were used in fitting."})

    p8 = [x for x in curves if x["panel"] == "P8" and int(x["step"]) == 2000]
    ids = sorted({x["anchor_id"] for x in p8}, key=lambda aid: next(i for i,x in enumerate(read_json(PANEL_LOCK)["rows"]) if x["panel"]=="P8" and x["anchor_id"]==aid))
    columns = [(c,l) for c in ("TS11-D","STATE-CTRL") for l in ("FULL_LOSS","POS_ONLY")]
    gains = {metric: np.full((len(ids),len(columns)),np.nan) for metric in ("position_gain_vs_persistence_mean","fd_gain_vs_persistence_mean")}
    for j,(config,loss) in enumerate(columns):
        run_id=series_key(config,loss,"P8")
        samples=[x for x in final_samples if x["run_id"]==run_id]
        for i,aid in enumerate(ids):
            r=next(x for x in samples if x["anchor_id"]==aid)
            for metric in gains:gains[metric][i,j]=float(r[metric])
    fig, axes = plt.subplots(1,2,figsize=(10,5.2),layout="constrained")
    for ax,(metric,title) in zip(axes,(("position_gain_vs_persistence_mean","Position gain vs Persistence"),("fd_gain_vs_persistence_mean","FD gain vs Persistence"))):
        mat=gains[metric]; mn=min(-0.2,float(np.nanmin(mat))); mx=max(1.0,float(np.nanmax(mat)))
        norm=TwoSlopeNorm(vmin=mn,vcenter=0.0,vmax=mx)
        im=ax.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto",interpolation="nearest")
        ax.set_title(title);ax.set_xticks(range(len(columns)),[f"{c}\n{l}" for c,l in columns],rotation=0,fontsize=8)
        ax.set_yticks(range(len(ids)),[f"{i+1}. {aid}" for i,aid in enumerate(ids)],fontsize=7)
        ax.set_xlabel("Locked model / loss cell");ax.set_ylabel("Locked train sample")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]): ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=7)
        fig.colorbar(im,ax=ax,label="Gain fraction (0 = Persistence error)")
    fig.suptitle(f"P8 per-sample final gains · {DIAG_LABEL}",fontsize=12)
    path=OUT_FIGURES/"TINYFIT_P8_PER_SAMPLE_GAINS.png";fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
    figures.append({"path":str(path.relative_to(ROOT)),"caption":"Final update-2000 per-sample position and FD gains versus Persistence for all four P8 model/loss cells. Each panel uses a diverging scale centered at zero gain; all displayed targets were used in fitting.","alt_text":"Two heatmaps show eight locked P8 training samples by four model/loss cells. Cell labels are per-sample reductions in normalized position error and safe-FD error relative to Persistence; zero means no improvement. This is a memorization diagnostic, not held-out performance."})
    return figures


def main() -> int:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True);OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    runs,samples,extras=load_runs();curves=extras["curves"];predictions=extras["predictions"];panel_lock=extras["panel_lock"]
    write_csv(OUT_RESULTS/"TINYFIT_FINAL_RUN_SUMMARY.csv",runs)
    write_csv(OUT_RESULTS/"TINYFIT_FINAL_SAMPLE_METRICS.csv",samples)
    write_csv(OUT_RESULTS/"TINYFIT_LEARNING_CURVES.csv",curves)
    p1_runs=[x for x in runs if x["panel"]=="P1"]
    p1_trigger=all(not x["strong_training_fit"] for x in p1_runs)
    p8_runs=[x for x in runs if x["panel"]=="P8"]
    # The protocol’s positive-control trigger starts with the smallest panel; P8 is considered only if P1 triggers.
    if p1_trigger:
        free_path="required_P1"
    elif all(not x["strong_training_fit"] for x in p8_runs):
        free_path="required_P8"
    else:
        free_path="not_required"
    total_updates=sum(int(x["updates"]) for x in runs)
    total_seconds=sum(float(x["elapsed_seconds"]) for x in runs)
    report={
        "schema_version":1,"status":"PHASE_B_8_OF_8_COMPLETE_TRAIN_ONLY",
        "matrix_sha256":sha_file(MATRIX),"protocol_sha256":sha_file(PROTOCOL),"panel_lock_sha256":sha_file(PANEL_LOCK),
        "formal_runs":len(runs),"optimizer_updates":total_updates,"cpu_wall_seconds_formal":total_seconds,
        "device":"cpu","torch_threads":1,"gpu_hours":0,"gpu_operations":0,
        "p1_strong_fit_cells":[x["run_id"] for x in p1_runs if x["strong_training_fit"]],
        "p8_strong_fit_cells":[x["run_id"] for x in p8_runs if x["strong_training_fit"]],
        "free_path_decision":free_path,"free_path_runs":0,
        "all_runs_from_original_init":all(x["final_checkpoint_sha256"] for x in runs),
        "all_runs_final_update_2000":all(x["updates"]==2000 for x in runs),
        "all_runs_development_or_final_access_zero":all(x["new_benchmark_candidate"] is False for x in runs),
        "strong_fit_definition":"Every panel sample must individually pass all three frozen conditions: position error <=10% of its Persistence error; safe-FD error <=20% of its Persistence error; prediction/target TV ratio in [0.8,1.2].",
        "P1_samples":1,"P8_samples":8,"P8_source_groups":8,"P8_distinct_tasks":len(set(int(x["original_task_index"]) for x in panel_lock["rows"] if x["panel"]=="P8")),
        "p8_exact_input_collision_exclusions":panel_lock["candidate_support"]["exact_input_collision_rows_excluded"],
        "sample_level_rows":len(samples),"fit_curve_rows":len(curves),
        "figures":[],
        "scope":{"targets_used_in_fitting":True,"dev_targets_opened":False,"dev_model_selection":False,"final_or_heldout_access":0,"new_full_data_benchmark_candidate":False,"D5_training":False},
        "source_hashes":{"panel_lock":sha_file(PANEL_LOCK),"protocol":sha_file(PROTOCOL),"matrix":sha_file(MATRIX),"phase_a_report":sha_file(ROOT/"A_FORENSIC_REPORT.json")},
    }
    figures=trajectory_figures(predictions,panel_lock)+metric_figures(curves,samples)
    report["figures"]=figures
    write_json(ROOT/"B_FINAL_RESULTS.json",report)
    # Result prose is deliberately descriptive and does not promote this one-seed fitted panel to generalization evidence.
    lines=[
        "# Core B：近静态取证与 train-only tiny-fit 结果",
        "",
        "状态：Phase A 与 8/8 个 Phase B tiny-fit 完成。所有训练样本的未来目标均参与了对应 run 的优化；这些结果只回答固定 train 轨迹在指定诊断条件下是否可拟合，不是 dev/test 预测，不是新 benchmark 候选，也不支持泛化结论。",
        "",
        "## 结论摘要",
        "",
        f"- 正式执行 {len(runs)}/8 格、{total_updates} 个 optimizer updates；设备 CPU、单线程；实际 fit 用时 {total_seconds/60:.1f} 分钟。D5、dev、final/held-out 均未访问。",
        f"- P1 上满足逐样本 strong-fit 三条件的格子：{', '.join(x['config']+' / '+x['loss'] for x in p1_runs if x['strong_training_fit']) or '无'}。",
        f"- P8 上满足逐样本 strong-fit 三条件的格子：{', '.join(x['config']+' / '+x['loss'] for x in p8_runs if x['strong_training_fit']) or '无'}。所有每样本值和 100-step 曲线均已保留。",
        f"- Free-path 触发判定：{free_path}。P1 四个 head 格中存在通过格时，不满足“两个模型×两个 loss 在最小失败 panel 全失败”的条件，因此未运行正对照。",
        "- 结合 Phase A：异常 TS11-D seed 的静态现象不是 anchor-copy；query 向量唯一但高度相似、FP32 有效秩较低。P1/P8 的拟合差异提供的是损失/融合分支/样本干扰线索，不能单独定位根因。",
        "",
        "## 八格汇总（末步=2000）",
        "",
        "| 模型 | Loss | Panel | 位置误差降低 | FD误差降低 | TV ratio | 逐样本强拟合 | 通过数 |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for x in runs:
        lines.append(f"| {x['config']} | {x['loss']} | {x['panel']} | {x['position_gain_vs_persistence_mean']:.3f} | {x['fd_gain_vs_persistence_mean']:.3f} | {x['prediction_tv_over_target_tv_mean']:.3f} | {'是' if x['strong_training_fit'] else '否'} | {x['per_sample_strong_fit_count']}/{x['panel_rows']} |")
    lines += [
        "",
        "## 解释与边界",
        "",
        "P1 只含一条高运动 train 轨迹。若该格通过，只能说明给定固定学习率、dropout-off、weight_decay=0 和初始化下，模型可以记忆该条训练轨迹。P8 是固定的 8 条高运动 train 轨迹，各自仍是拟合目标；它不是 held-out panel。",
        "",
        "强拟合标签要求每个样本同时通过三项阈值，避免 panel 均值掩盖单样本失败。位置增益和 FD 增益分别相对每条轨迹自己的 Persistence；TV ratio 采用 safe-FD 归一化的 future-internal 差分，不包括 anchor-to-first seam。方向准确率仅对冻结阈值定义的 active train differences 计算，raw action 坐标未赋予 Hand/Arm 语义。",
        "",
        "只运行了一个 seed（20261002），而且每个 panel 被反复优化 2000 次；指标曲线不是独立重复，也不应作置信区间或稳定性结论。没有进行 LR/seed/query sweep，也没有根据 fit 结果换样本。",
        "",
        "## 图表与数据",
        "",
    ]
    for f in figures:
        lines.append(f"- `{f['path']}` — {f['caption']} Alt text: {f['alt_text']}")
    lines += [
        "",
        "逐样本末步指标见 `results/TINYFIT_FINAL_SAMPLE_METRICS.csv`；全部 100-step 曲线见 `results/TINYFIT_LEARNING_CURVES.csv`；每个 run 的 `PREDICTIONS_FINAL.npz`、`checkpoint_final.pt`、checkpoint 日志及 `RUN_COMPLETE.json` 均在对应 `runs/formal/` 目录。",
        "",
        "## 下一步建议",
        "",
        "停止新增训练格。当前证据不支持唯一归因：Phase A 的 query conditioning 风险、TS11-D 与 STATE-CTRL 的差异、FULL/POS 的目标敏感性和 P1→P8 的样本干扰都可能贡献。先做一个只读代码路径审计，确认 TS11-D 动态视频 token 到 decoder 的实际梯度/尺度与 STATE-CTRL 对照一致性；若没有明确、可证伪的实现问题，不再追加优化器、query scale 或 seed sweep，转向独立评价前的数据/目标/指标冻结清单。",
        "",
    ]
    (ROOT/"B_FINAL_REPORT.zh-CN.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":report["status"],"runs":len(runs),"updates":total_updates,"strong_p1":[x["run_id"] for x in p1_runs if x["strong_training_fit"]],"strong_p8":[x["run_id"] for x in p8_runs if x["strong_training_fit"]],"free_path":free_path,"report_sha256":sha_file(ROOT/"B_FINAL_REPORT.zh-CN.md"),"figure_count":len(figures)},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

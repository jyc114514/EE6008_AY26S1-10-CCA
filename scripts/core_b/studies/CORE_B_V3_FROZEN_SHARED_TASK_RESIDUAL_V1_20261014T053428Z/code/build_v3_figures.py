#!/usr/bin/env python3
"""Build V3 development figures from sealed predictions and existing V2 case data."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
SCORE = ROOT / "results/scoring/attempt_02"
SEEDS = (202610131, 202610132, 202610133)
TASKS = (119, 94, 90, 141, 230)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty figure-data table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir, datadir = ROOT / "figures", ROOT / "figure_data"
    figdir.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True)
    if not (SCORE / "DEV_TARGET_SCORING_QA.json").is_file():
        raise RuntimeError("figures require completed post-seal scoring")

    metric_rows = read_csv(SCORE / "METHOD_METRICS_BY_TASK_SEED.csv")
    summary_rows = read_csv(SCORE / "SUMMARY_STATISTICS.csv")
    contrast_rows = read_csv(SCORE / "CONTRASTS_BY_TASK_SEED.csv")
    rgb_rows = read_csv(SCORE / "RGB_PROBE_RESULTS.csv")
    by_metric: dict[tuple[str, int, int, str], float] = {
        (r["method"], int(r["seed"]), int(r["task_id"]), r["metric"]): float(r["value"])
        for r in metric_rows
    }
    metric_def = ("position_mae", "safe_fd_l1")
    labels = {
        "PERSISTENCE": "Persistence", "BASE_STATE": "Base state", "BIAS_STATE": "Bias state",
        "SHARED_RES_STATE": "Shared residual state", "TASK_RES_STATE": "Task residual state",
        "BASE_VIDEO": "Base video", "BIAS_VIDEO": "Bias video",
        "SHARED_RES_VIDEO": "Shared residual video", "TASK_RES_VIDEO": "Task residual video",
    }
    colors = {
        "PERSISTENCE": "#7d7d7d", "BASE_STATE": "#2673a6", "BIAS_STATE": "#54a0c5",
        "SHARED_RES_STATE": "#df8f2d", "TASK_RES_STATE": "#a45d07",
        "BASE_VIDEO": "#538b42", "BIAS_VIDEO": "#84ad6d",
        "SHARED_RES_VIDEO": "#a34a77", "TASK_RES_VIDEO": "#6e254f",
    }

    # Main five-task dual-metric chart: accepted bases, both residuals, and persistence.
    methods_main = ["BASE_STATE", "SHARED_RES_STATE", "TASK_RES_STATE",
                    "BASE_VIDEO", "SHARED_RES_VIDEO", "TASK_RES_VIDEO", "PERSISTENCE"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    x = np.arange(len(TASKS)); width = 0.105
    for ax, metric, title in zip(axes, metric_def, ("Position MAE (canonical scale)", "Safe-FD L1 (29 future differences)")):
        for j, method in enumerate(methods_main):
            means, sds = [], []
            for task in TASKS:
                vals = [by_metric[(method, seed, task, metric)] for seed in SEEDS]
                means.append(float(np.mean(vals))); sds.append(float(np.std(vals, ddof=1)))
            offset = (j - (len(methods_main)-1)/2) * width
            ax.errorbar(x + offset, means, yerr=sds, fmt="o", capsize=2, lw=1.1,
                        color=colors[method], label=labels[method])
        ax.set_xticks(x, [str(t) for t in TASKS]); ax.set_xlabel("Task ID")
        ax.set_ylabel(title); ax.set_title(title); ax.grid(axis="y", alpha=.22)
    axes[1].legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    fig.suptitle("V3 frozen-base outputs on the five-task development set\nmean across three seeds; error bars are sample SD")
    fig.savefig(figdir / "MAIN_TASK_METRICS.png", dpi=190); plt.close(fig)

    # Task/seed paired contrasts, displayed as five-task macro per seed.
    desired = ["G_R_shared_STATE", "G_R_task_STATE", "G_BIAS_STATE", "T_R_STATE",
               "R_vs_BIAS_shared_STATE", "R_vs_BIAS_task_STATE",
               "G_R_shared_VIDEO", "G_R_task_VIDEO", "G_BIAS_VIDEO", "T_R_VIDEO",
               "R_vs_BIAS_shared_VIDEO", "R_vs_BIAS_task_VIDEO"]
    contrast_macro = []
    for name in desired:
        for seed in SEEDS:
            for metric in metric_def:
                vals = [float(r["value"]) for r in contrast_rows
                        if r["contrast"] == name and int(r["seed"]) == seed and r["metric"] == metric]
                if len(vals) != len(TASKS): raise RuntimeError(f"incomplete macro contrast {name}/{seed}/{metric}")
                contrast_macro.append({"contrast": name, "seed": seed, "metric": metric, "five_task_macro": float(np.mean(vals))})
    write_csv(datadir / "V3_FIVE_TASK_SEED_CONTRASTS.csv", contrast_macro)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    seed_colors = {SEEDS[0]: "#2673a6", SEEDS[1]: "#df8f2d", SEEDS[2]: "#538b42"}
    for ax, metric, title in zip(axes, metric_def, ("Position improvement", "Safe-FD improvement")):
        xx = np.arange(len(desired)); spread = 0.075
        for si, seed in enumerate(SEEDS):
            vals = [next(r["five_task_macro"] for r in contrast_macro if r["contrast"] == n and r["seed"] == seed and r["metric"] == metric) for n in desired]
            ax.scatter(xx + (si-1)*spread, vals, color=seed_colors[seed], s=22, label=str(seed), zorder=3)
        ax.axhline(0, color="black", lw=.8); ax.set_xticks(xx, desired, rotation=70, ha="right")
        ax.set_ylabel("L(left) − L(right); positive favors right")
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].legend(title="Seed", frameon=False)
    fig.suptitle("V3 paired contrasts by seed\npositive values mean the named right-hand method has lower error")
    fig.savefig(figdir / "SEED_MACRO_CONTRASTS.png", dpi=190); plt.close(fig)

    # Bias versus learned residuals at five-task macro level.
    macro = {(r["name"], r["metric"]): (float(r["mean_across_seeds"]), float(r["sample_sd_across_seeds"]))
             for r in summary_rows if r["scope"] == "five_task_macro" and r["result_type"] == "method"}
    method_bias = ["BIAS_STATE", "SHARED_RES_STATE", "TASK_RES_STATE", "BIAS_VIDEO", "SHARED_RES_VIDEO", "TASK_RES_VIDEO"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    for ax, metric, title in zip(axes, metric_def, ("Position MAE", "Safe-FD L1")):
        xx = np.arange(len(method_bias))
        means = [macro[(m, metric)][0] for m in method_bias]
        sds = [macro[(m, metric)][1] for m in method_bias]
        ax.bar(xx, means, yerr=sds, capsize=3, color=[colors[m] for m in method_bias])
        ax.set_xticks(xx, [labels[m] for m in method_bias], rotation=45, ha="right")
        ax.set_title(title); ax.set_ylabel("Five-task macro error"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Train-only task-axis bias versus frozen-base residual readouts\nmean across seeds; error bars are sample SD")
    fig.savefig(figdir / "BIAS_VS_RESIDUAL.png", dpi=190); plt.close(fig)

    # Screening curves: fit objective by epoch and inner-tune canonical J at checkpoints.
    curve_rows: list[dict[str, str]] = []
    for program in ("SHARED_RES_STATE", "TASK_RES_STATE", "SHARED_RES_VIDEO", "TASK_RES_VIDEO"):
        for lam in (0, 1):
            f = ROOT / f"model_only/screening/{program}/lambda{lam}_s202610130/attempt_01/EPOCH_METRICS.csv"
            if not f.is_file(): raise RuntimeError(f"screening curve source missing: {f}")
            for row in read_csv(f): curve_rows.append({**row, "source_path": str(f)})
    write_csv(datadir / "V3_SCREENING_TRAIN_TUNE_CURVES.csv", curve_rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True, constrained_layout=True)
    for ax, program in zip(axes.flat, ("SHARED_RES_STATE", "TASK_RES_STATE", "SHARED_RES_VIDEO", "TASK_RES_VIDEO")):
        rows = [r for r in curve_rows if r["program"] == program]
        for lam, color in ((0, "#2673a6"), (1, "#df8f2d")):
            lr = sorted((r for r in rows if float(r["lambda"]) == lam), key=lambda r: int(r["epoch"]))
            epochs = np.asarray([int(r["epoch"]) for r in lr])
            fit = np.asarray([float(r["fit_objective_weighted_mean"]) for r in lr])
            ax.plot(epochs, fit, color=color, lw=1.4, label=f"λ={lam} fit objective")
            keep = [(int(r["epoch"]), float(r["tune_J"])) for r in lr if r["tune_J"] not in {"NA", ""}]
            if keep:
                tx, ty = zip(*keep); ax.plot(tx, ty, color=color, ls="--", marker="o", ms=4, label=f"λ={lam} inner-tune J")
        ax.axvline(20, color="#777", ls=":", lw=.7); ax.axvline(40, color="#777", ls=":", lw=.7); ax.axvline(80, color="#777", ls=":", lw=.7)
        ax.set_title(program); ax.set_xlabel("Epoch"); ax.set_ylabel("Recorded objective / canonical J"); ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("Screening fits used inner-fit for updates and inner-tune only at locked checkpoints")
    fig.savefig(figdir / "SCREENING_TRAIN_TUNE_CURVES.png", dpi=190); plt.close(fig)

    # RGB probe h/output sensitivity and metric changes, without cross-checkpoint hidden-state comparisons.
    write_csv(datadir / "V3_RGB_PROBE_RESULTS.csv", rgb_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    conds = ("ZERO_NORMALIZED_BLOCK", "SAME_TASK_DONOR")
    gv = [r for r in rgb_rows if r["model_id"] == "G-V"]
    xx = np.arange(len(TASKS)); width=.36
    for j, cond in enumerate(conds):
        sub = [r for r in gv if r["condition"] == cond]
        vals_h = [float(np.mean([float(r["h_relative_l2_change"]) for r in sub if int(r["task_id"]) == task])) for task in TASKS]
        vals_z = [float(np.mean([float(r["base_residual_mean_abs_change"]) for r in sub if int(r["task_id"]) == task])) for task in TASKS]
        axes[0].bar(xx + (j-.5)*width, vals_h, width, label=cond.replace("_", " "))
        axes[1].bar(xx + (j-.5)*width, vals_z, width, label=cond.replace("_", " "))
    axes[0].set_ylabel("Relative L2 change in h"); axes[0].set_title("G-V hidden representation sensitivity")
    axes[1].set_ylabel("Mean absolute change in base output"); axes[1].set_title("G-V projected output sensitivity")
    for ax in axes:
        ax.set_xticks(xx, [str(task) for task in TASKS], rotation=0); ax.set_xlabel("Task ID; bars are condition means over 3 seeds")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed-checkpoint RGB input perturbations; same-checkpoint h comparisons only")
    fig.savefig(figdir / "RGB_H_AND_OUTPUT_SENSITIVITY.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    for ax, metric, title in zip(axes, metric_def, ("Δ Position MAE", "Δ Safe-FD L1")):
        for j, cond in enumerate(conds):
            sub = [r for r in rgb_rows if r["model_id"] == "G-V" and r["condition"] == cond]
            means=[]; sds=[]
            for task in TASKS:
                values=[float(r[f"{metric}_condition_minus_matched"]) for r in sub if int(r["task_id"])==task]
                means.append(float(np.mean(values)));sds.append(float(np.std(values,ddof=1)))
            ax.errorbar(np.arange(len(TASKS))+(j-.5)*.12,means,yerr=sds,fmt="o",capsize=2,label=cond.replace("_"," "))
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(np.arange(len(TASKS)),[str(t) for t in TASKS]);ax.set_xlabel("Task ID")
        ax.set_ylabel("Condition error − matched error");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle("RGB perturbation changes in scored G-V errors; positive means the perturbation worsened error")
    fig.savefig(figdir / "RGB_ERROR_CHANGES.png", dpi=190); plt.close(fig)
    if os.environ.get("V3_RGB_FIGURE_ONLY") == "1":
        print(json.dumps({"status":"PASS","rgb_figures_rebuilt":2,"scorer_only_target_archive_opened":False,
                          "output_dir":str(figdir)},indent=2))
        return

    # Trajectory examples reuse the exact first V2-selected exposed-dev case per task.
    case_rows = read_csv(V2 / "figure_data/TRAJECTORY_CASES.csv")
    selected = {}
    for task in TASKS:
        candidates = sorted((r for r in case_rows if int(r["task_id"]) == task and r["role"] == "EXPOSED_DEV"), key=lambda r: int(r["case_order"]))
        if not candidates: raise RuntimeError(f"V2 has no accepted dev trajectory case for task {task}")
        selected[task] = candidates[0]
    selected_ids = {r["anchor_id"] for r in selected.values()}
    # This is an existing V2 derived plot table, not the scorer-only target archive.
    v2_plot = V2 / "figure_data/TRAJECTORY_PLOT_DATA.csv"
    chosen: dict[tuple[str,int,int,str], float] = {}
    case_plot_rows: dict[tuple[str,int], list[dict[str,str]]] = defaultdict(list)
    with v2_plot.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["anchor_id"] not in selected_ids or int(r["action_axis_index"]) not in {0,14}:
                continue
            case_plot_rows[(r["anchor_id"], int(r["seed"]))].append(r)
            if int(r["seed"]) != SEEDS[0]:
                continue
            if r["series_type"] == "target":
                chosen[(r["anchor_id"], int(r["action_axis_index"]), int(r["horizon_step"]), "GT")] = float(r["prediction_or_target_value"])
            elif r["model_id"] == "Persistence" and r["series_type"] == "prediction":
                chosen[(r["anchor_id"], int(r["action_axis_index"]), int(r["horizon_step"]), "Persistence")] = float(r["prediction_or_target_value"])

    seal = json.loads((ROOT / "protocol/PREDICTION_BATCH_SEAL.json").read_text())
    pred_series: dict[tuple[str,int,int,str], np.ndarray] = {}
    for entry in seal["new_prediction_entries"]:
        if entry["method_kind"] not in {"bias", "readout"}: continue
        cfg = json.loads(Path(entry["job_config_path"]).read_text())
        rows = read_csv(Path(entry["prediction_rows_path"]))
        tensors = load_file(entry["prediction_path"], device="cpu")
        pred = tensors["prediction_native"].numpy()
        rowix = {r["anchor_id"]: i for i, r in enumerate(rows)}
        for aid in selected_ids:
            if aid in rowix:
                pred_series[(aid,int(entry["seed"]),int(cfg["action_axis_index"]) if "action_axis_index" in cfg else -1,entry["method_id"])] = pred[rowix[aid]]
        # Axis is selected at plot time; retain full H30x28 native predictions in this local map.
        for aid in selected_ids:
            if aid in rowix:
                pred_series[(aid,int(entry["seed"]),-2,entry["method_id"])] = pred[rowix[aid]]

    traj_data: list[dict] = []
    for modality, base_model in (("STATE", "G-S"), ("VIDEO", "G-V")):
        for task in TASKS:
            aid = selected[task]["anchor_id"]
            for seed in SEEDS:
                old_rows = case_plot_rows.get((aid, seed), [])
                for axis in (0,14):
                    series_values: dict[str,np.ndarray] = {}
                    for name in ("GT", "Persistence"):
                        series_values[name] = np.asarray([chosen[(aid,axis,h,name)] for h in range(1,31)],dtype=np.float64)
                    for model,name in ((base_model,"BASE"),):
                        source = {(int(r["action_axis_index"]),int(r["horizon_step"])):float(r["prediction_or_target_value"])
                                  for r in old_rows if r["model_id"]==model and r["series_type"]=="prediction"}
                        series_values[name] = np.asarray([source[(axis,h)] for h in range(1,31)],dtype=np.float64)
                    suffix = "STATE" if modality == "STATE" else "VIDEO"
                    for method,name in ((f"BIAS_{suffix}","BIAS"),(f"SHARED_RES_{suffix}","SHARED-R"),(f"TASK_RES_{suffix}","TASK-R")):
                        full = pred_series[(aid,seed,-2,method)]
                        series_values[name] = np.asarray(full[:,axis],dtype=np.float64)
                    for name,values in series_values.items():
                        for h,value in enumerate(values,1):
                            traj_data.append({"task_id":task,"episode_id":selected[task]["episode_id"],"anchor_id":aid,
                                              "seed":seed,"modality":modality,"action_axis_index":axis,"horizon_step":h,
                                              "series":name,"value_native":float(value),"case_source":"V2 accepted TRAJECTORY_CASES"})
    write_csv(datadir / "TRAJECTORY_EXAMPLES_V3.csv", traj_data)
    series_order = ("GT","Persistence","BASE","BIAS","SHARED-R","TASK-R")
    series_colors = {"GT":"#111111","Persistence":"#888888","BASE":"#2673a6","BIAS":"#54a0c5",
                     "SHARED-R":"#df8f2d","TASK-R":"#a34a77"}
    for modality in ("STATE","VIDEO"):
        for axis in (0,14):
            fig, axs = plt.subplots(len(TASKS),len(SEEDS),figsize=(15.5,15),sharex=True,constrained_layout=True)
            for ti,task in enumerate(TASKS):
                aid=selected[task]["anchor_id"]
                for si,seed in enumerate(SEEDS):
                    ax=axs[ti,si]
                    subset=[r for r in traj_data if r["task_id"]==task and r["seed"]==seed and r["modality"]==modality and r["action_axis_index"]==axis]
                    for series in series_order:
                        sr=sorted((r for r in subset if r["series"]==series),key=lambda x:int(x["horizon_step"]))
                        ax.plot([int(x["horizon_step"]) for x in sr],[float(x["value_native"]) for x in sr],
                                color=series_colors[series],lw=1.35 if series in {"GT","TASK-R"} else 1.0,
                                ls="--" if series=="Persistence" else "-",label=series)
                    if si==0: ax.set_ylabel(f"Task {task}\naxis {axis}\nvalue (native)")
                    if ti==0: ax.set_title(f"Seed {seed}")
                    if ti==len(TASKS)-1: ax.set_xlabel("Future step")
                    ax.grid(alpha=.16)
                    if ti==0 and si==0: ax.legend(frameon=False,fontsize=7,ncol=2)
                    if si==2: ax.text(.99,.04,aid,transform=ax.transAxes,ha="right",va="bottom",fontsize=6,color="#555")
            fig.suptitle(f"V3 selected dev trajectories: {modality.lower()} base, action axis {axis}\nGT/persistence/base/bias/shared/task residual; V2 case IDs")
            fig.savefig(figdir/f"TRAJECTORY_{modality}_AXIS{axis}.png",dpi=160);plt.close(fig)

    captions = """# V3 figure captions and data notes\n\n"""
    captions += "- `MAIN_TASK_METRICS.png`: Per-task position MAE and safe-FD L1 for persistence, frozen state/video bases, and both residual readouts. Points are three-seed means; bars are sample SD. Lower is better.\n"
    captions += "- `SEED_MACRO_CONTRASTS.png`: Five-task paired contrasts by seed. Positive values favor the named right-hand method. No significance tests are shown.\n"
    captions += "- `BIAS_VS_RESIDUAL.png`: Five-task macro absolute errors for train-only bias and residual methods; three-seed mean and sample SD.\n"
    captions += "- `SCREENING_TRAIN_TUNE_CURVES.png`: Recorded inner-fit weighted objective and inner-tune canonical J by epoch and lambda for each readout program; dashed tune series only appear at measured checkpoints.\n"
    captions += "- `RGB_H_AND_OUTPUT_SENSITIVITY.png`: G-V hidden and base-output changes under fixed-checkpoint RGB interventions. Hidden states are compared only within the same checkpoint.\n"
    captions += "- `RGB_ERROR_CHANGES.png`: G-V position and safe-FD error change relative to each checkpoint's matched prediction; positive means perturbation worsened error.\n"
    captions += "- `TRAJECTORY_{STATE,VIDEO}_AXIS{0,14}.png`: Three seeds across columns and five tasks down rows, using the first V2-selected exposed-dev case per task. Target traces are reused from the existing V2 derived `TRAJECTORY_PLOT_DATA.csv`; no scorer-only target archive is opened by this figure builder.\n"
    captions += "\nAll plot data are saved under `figure_data/`. Human review is pending; generated images were produced at this stage for visual inspection.\n"
    (figdir / "CAPTIONS_AND_DESCRIPTIONS.md").write_text(captions,encoding="utf-8")
    (ROOT/"qa/FIGURE_BUILD_QA.json").write_text(json.dumps({"schema_version":1,"status":"PASS","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
      "figures":[p.name for p in sorted(figdir.glob("*.png"))],"trajectory_case_ids":{str(t):selected[t]["anchor_id"] for t in TASKS},
      "trajectory_axes":[0,14],"trajectory_seeds":list(SEEDS),"trajectory_target_source":"existing V2 derived figure_data/TRAJECTORY_PLOT_DATA.csv",
      "scorer_only_target_archive_opened_by_figure_builder":False,"reused_existing_v2_target_trace_points":len(selected_ids)*2*30,
      "rgb_probe_views":len(rgb_rows),"final_heldout_payload_access":0},ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","figures":len(list(figdir.glob('*.png'))),"trajectory_plot_rows":len(traj_data),
                      "case_ids":{str(t):selected[t]["anchor_id"] for t in TASKS},"output":str(figdir)},indent=2))


if __name__ == "__main__":
    main()

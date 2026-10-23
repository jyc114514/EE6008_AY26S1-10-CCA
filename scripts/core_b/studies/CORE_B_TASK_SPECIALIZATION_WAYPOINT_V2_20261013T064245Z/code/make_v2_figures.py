#!/usr/bin/env python3
"""Create inspectable V2 train/tune, contrast, persistence and trajectory figures."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def task_names() -> dict[int, str]:
    payload = json.loads((ROOT / "protocol/TASKS_V2.json").read_text(encoding="utf-8"))
    return {int(row["task_id"]): str(row["original_task_name"]) for row in payload["tasks"]}


def plot_train_tune() -> None:
    rows = read_csv(ROOT / "screening/TRAIN_TUNE_CURVES.csv")
    model_ids = ["G-S", "G-V"] + [f"S-{mode}-{task}" for task in TASK_IDS for mode in ("S", "V")]
    lr_styles = {
        5e-4: {"color": "#0072B2", "linestyle": "-", "marker": "o"},
        1e-3: {"color": "#D55E00", "linestyle": "--", "marker": "s"},
        2e-3: {"color": "#009E73", "linestyle": ":", "marker": "^"},
    }
    fig, axes = plt.subplots(4, 6, figsize=(19, 13), sharex="col")
    for model_index, model_id in enumerate(model_ids):
        row = model_index // 3
        pair_col = (model_index % 3) * 2
        train_ax = axes[row, pair_col]
        tune_ax = axes[row, pair_col + 1]
        for lr in (5e-4, 1e-3, 2e-3):
            selected = [item for item in rows if item["model_id"] == model_id and abs(float(item["lr"]) - lr) < 1e-12]
            style = lr_styles[lr]
            train_points = sorted(
                (int(item["epoch"]), float(item["train_total"]))
                for item in selected if item.get("train_total", "") not in ("", "NA")
            )
            if train_points:
                x_train, y_train = zip(*train_points)
                train_ax.plot(x_train, y_train, linewidth=1.25, label=f"LR {lr:g}", **style)
            tune_points = sorted(
                (int(item["epoch"]), float(item["tune_j"]))
                for item in selected if item.get("tune_j", "") not in ("", "NA")
            )
            if tune_points:
                x_tune, y_tune = zip(*tune_points)
                tune_ax.plot(x_tune, y_tune, linewidth=1.6, markevery=1, label=f"LR {lr:g}", **style)
        train_ax.set_title(f"{model_id}: train objective")
        tune_ax.set_title(f"{model_id}: tune J")
        for ax in (train_ax, tune_ax):
            ax.grid(alpha=0.22)
            ax.set_xticks([40, 80, 120, 160])
        train_ax.set_ylabel("weighted train loss")
        tune_ax.set_ylabel("inner-tune J")
    for ax in axes[-1, :]:
        ax.set_xlabel("training epoch")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Screening training objective and inner-tune J; tune scored at epochs 40, 80, 160", y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(ROOT / "figures/TRAIN_TUNE_CURVES.png", dpi=170)
    plt.close(fig)


def plot_contrasts() -> None:
    rows = read_csv(ROOT / "results/ALL_SEED_D_V_I.csv")
    support = {
        int(row["task_id"]): int(row["dev_clusters_after_overlap_exclusion"])
        for row in read_csv(ROOT / "inventory/DEV_SUPPORT_V2.csv")
    }
    contrasts = ("D_S", "D_V", "V_G", "V_S", "I")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")
    hatches = ("/", "\\", "x", ".", "-")
    seed_markers = {SEEDS[0]: "o", SEEDS[1]: "s", SEEDS[2]: "^"}
    for metric in ("position_mae", "safe_fd_l1"):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        x = np.arange(len(TASK_IDS))
        width = 0.14
        for ci, (contrast, color, hatch) in enumerate(zip(contrasts, colors, hatches)):
            means = []
            sds = []
            per_task_seed = []
            for task in TASK_IDS:
                values = {
                    int(row["seed"]): float(row["value"])
                    for row in rows
                    if int(row["task_id"]) == task and row["metric"] == metric
                    and row["contrast"] == contrast and row["value"] != "NA"
                }
                ordered = [values[seed] for seed in SEEDS if seed in values]
                means.append(float(np.mean(ordered)) if ordered else np.nan)
                sds.append(float(np.std(ordered, ddof=1)) if len(ordered) > 1 else 0.0)
                per_task_seed.append(values)
            centers = x + (ci - 2) * width
            ax.bar(
                centers, means, width=width, yerr=sds, capsize=2.5,
                color=color, edgecolor="black", linewidth=0.55, hatch=hatch,
                alpha=0.72, label=contrast, zorder=2,
            )
            for task_index, values in enumerate(per_task_seed):
                for seed_index, seed in enumerate(SEEDS):
                    if seed in values:
                        jitter = (seed_index - 1) * width * 0.18
                        ax.scatter(
                            centers[task_index] + jitter, values[seed], marker=seed_markers[seed],
                            s=22, color="black", edgecolors="white", linewidths=0.35, zorder=3,
                        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, [f"{task}\nsource groups n={support.get(task, 0)}" for task in TASK_IDS])
        ax.set_xlabel("task ID and exposed-dev source-group support")
        ax.set_ylabel(f"paired change in {metric} (positive = benefit)")
        ax.set_title(f"Task and RGB contrasts: {metric}; bars mean ± sample SD, symbols are the three seeds")
        ax.grid(axis="y", alpha=0.22, zorder=0)
        ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
        fig.text(
            0.5, 0.018,
            f"Seed markers: circle {SEEDS[0]}, square {SEEDS[1]}, triangle {SEEDS[2]}; bars show mean ± sample SD.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(ROOT / f"figures/CONTRASTS_{metric.upper()}.png", dpi=170)
        plt.close(fig)


def load_trajectory_inputs():
    with np.load(ROOT / "data/DEV_PREFIXES_NO_LABELS.npz", allow_pickle=False) as archive:
        prefix = {key: archive[key].copy() for key in (
            "anchor_id", "episode_id", "task_id", "source_cluster", "anchor_frame", "anchor_native",
        )}
    with np.load(ROOT / "data/DEV_TARGETS_SCORER_ONLY.npz", allow_pickle=False) as archive:
        target = {key: archive[key].copy() for key in ("anchor_id", "target_native_h30")}
    if not np.array_equal(prefix["anchor_id"], target["anchor_id"]):
        raise RuntimeError("trajectory dev target/prefix anchor order mismatch")
    target_by_anchor = {str(anchor): target["target_native_h30"][i] for i, anchor in enumerate(target["anchor_id"].tolist())}
    prefix_index = {str(anchor): i for i, anchor in enumerate(prefix["anchor_id"].tolist())}
    model_pred: dict[tuple[str, int, str], np.ndarray] = {}
    queue = json.loads((ROOT / "coordination/queues/FORMAL_DEV.json").read_text())
    for raw in queue["jobs"]:
        watcher = json.loads(Path(raw).read_text())
        cfg = json.loads(Path(watcher["command"][-1]).read_text())
        out = Path(cfg["output_root"])
        complete = out / "PREDICTION_COMPLETE.json"
        if not complete.exists():
            continue
        ids = read_csv(out / "PREDICTION_ROWS.csv")
        tensors = load_file(str(out / "PREDICTIONS.safetensors"), device="cpu")
        pred = tensors["prediction_native"].numpy()
        for j, row in enumerate(ids):
            model_pred[(str(cfg["model_id"]), int(cfg["seed"]), str(row["anchor_id"]))] = pred[j]
    return prefix, prefix_index, target_by_anchor, model_pred


def plot_trajectories() -> None:
    prefix, prefix_index, target_by_anchor, predictions = load_trajectory_inputs()
    names = task_names()
    cases_rows: list[dict] = []
    plot_rows: list[dict] = []
    for task in TASK_IDS:
        task_indices = [i for i, value in enumerate(prefix["task_id"].tolist()) if int(value) == task]
        episodes: dict[int, list[int]] = defaultdict(list)
        cluster_by_episode: dict[int, str] = {}
        for i in task_indices:
            episode = int(prefix["episode_id"][i])
            episodes[episode].append(i)
            cluster_by_episode[episode] = str(prefix["source_cluster"][i])
        ordered = sorted(
            episodes,
            key=lambda ep: (hashlib.sha256(str(ep).encode("utf-8")).hexdigest(), ep),
        )[:2]
        models = ["G-S", "G-V", f"S-S-{task}", f"S-V-{task}"]
        model_styles = {
            "G-S": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
            "G-V": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
            f"S-S-{task}": {"color": "#D55E00", "linestyle": "-", "marker": "o"},
            f"S-V-{task}": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
        }
        if task == 119:
            models += ["B-AUX_ONLY", "B-GOAL_COND"]
            model_styles.update({
                "B-AUX_ONLY": {"color": "#009E73", "linestyle": "-.", "marker": "D"},
                "B-GOAL_COND": {"color": "#CC79A7", "linestyle": ":", "marker": "^"},
            })
        if not ordered:
            cases_rows.append({"task_id": task, "task_name": names[task], "case_order": "", "episode_id": "",
                               "source_cluster_hash": "", "anchor_id": "", "role": "NO_DEV"})
            continue
        fig, axes = plt.subplots(2 * len(ordered), len(SEEDS), figsize=(18, 4.2 * len(ordered)), squeeze=False, sharex=True)
        for case_idx, episode in enumerate(ordered):
            candidates = sorted(episodes[episode], key=lambda i: str(prefix["anchor_id"][i]))
            i = candidates[0]
            anchor_id = str(prefix["anchor_id"][i])
            target = target_by_anchor[anchor_id]
            source = cluster_by_episode[episode]
            cases_rows.append({
                "task_id": task, "task_name": names[task], "case_order": case_idx + 1,
                "episode_id": episode,
                "canonical_episode_id_sha256": hashlib.sha256(str(episode).encode("utf-8")).hexdigest(),
                "source_cluster_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "anchor_id": anchor_id, "anchor_frame": int(prefix["anchor_frame"][i]),
                "role": "EXPOSED_DEV", "support_note": "development-only trajectory",
            })
            for axis_row, action_axis in enumerate((0, 14)):
                ax_row = 2 * case_idx + axis_row
                for col, seed in enumerate(SEEDS):
                    ax = axes[ax_row, col]
                    step = np.arange(1, 31)
                    ax.plot(step, target[:, action_axis], color="#000000", linewidth=2.3, linestyle="-", label="observed dev")
                    anchor_value = float(prefix["anchor_native"][i, action_axis])
                    persistence = np.full(30, anchor_value)
                    ax.plot(step, persistence, color="#777777", linestyle=(0, (2, 1)), linewidth=1.35, label="Persistence")
                    for horizon, y in enumerate(persistence, start=1):
                        plot_rows.append({
                            "task_id": task, "episode_id": episode, "anchor_id": anchor_id,
                            "seed": seed, "model_id": "Persistence", "action_axis_index": action_axis,
                            "horizon_step": horizon, "prediction_or_target_value": float(y),
                            "series_type": "prediction",
                        })
                    for model in models:
                        value = predictions.get((model, seed, anchor_id))
                        if value is not None:
                            ax.plot(step, value[:, action_axis], linewidth=1.25, alpha=0.92, markevery=5, label=model, **model_styles[model])
                            for horizon, y in enumerate(value[:, action_axis], start=1):
                                plot_rows.append({
                                    "task_id": task, "episode_id": episode, "anchor_id": anchor_id,
                                    "seed": seed, "model_id": model, "action_axis_index": action_axis,
                                    "horizon_step": horizon, "prediction_or_target_value": float(y),
                                    "series_type": "prediction",
                                })
                    for horizon, y in enumerate(target[:, action_axis], start=1):
                        plot_rows.append({
                            "task_id": task, "episode_id": episode, "anchor_id": anchor_id,
                            "seed": seed, "model_id": "observed_dev", "action_axis_index": action_axis,
                            "horizon_step": horizon, "prediction_or_target_value": float(y),
                            "series_type": "target",
                        })
                    ax.set_title(f"case {case_idx + 1} · seed {seed} · axis {action_axis}")
                    ax.set_xlabel("future step")
                    ax.set_ylabel("numeric command value")
                    ax.grid(alpha=0.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.015))
        dev_support = next(row for row in read_csv(ROOT / "inventory/DEV_SUPPORT_V2.csv") if int(row["task_id"]) == task)
        fig.suptitle(
            f"Task {task}: {names[task]} — dev trajectories (n={dev_support['dev_episodes_after_overlap_exclusion']} episodes, one anchor per case)",
            y=0.995, fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(ROOT / f"figures/TRAJECTORY_TASK_{task}.png", dpi=160)
        plt.close(fig)
    write_csv(ROOT / "figure_data/TRAJECTORY_CASES.csv", list(cases_rows[0]) if cases_rows else ["task_id", "role"], cases_rows)
    write_csv(ROOT / "figure_data/TRAJECTORY_PLOT_DATA.csv", list(plot_rows[0]) if plot_rows else ["task_id", "role"], plot_rows)


def plot_waypoint() -> None:
    rows = read_csv(ROOT / "results/WAYPOINT.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("position_mae", ("B_GAIN", "AUX_VS_SV", "GOAL_VS_SV")),
        ("safe_fd_l1", ("B_GAIN", "AUX_VS_SV", "GOAL_VS_SV")),
        ("goal_endpoint_mae_normalized", ("B_GOAL_HEAD_GAIN",)),
    ]
    for ax, (metric, contrasts) in zip(axes, specs):
        contrast_styles = {
            "B_GAIN": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
            "AUX_VS_SV": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
            "GOAL_VS_SV": {"color": "#009E73", "linestyle": ":", "marker": "^"},
            "B_GOAL_HEAD_GAIN": {"color": "#CC79A7", "linestyle": "-.", "marker": "D"},
        }
        for contrast in contrasts:
            values = []
            for seed in SEEDS:
                found = [float(row["value"]) for row in rows
                         if row["metric"] == metric and row["contrast"] == contrast
                         and int(row["seed"]) == seed and row["value"] != "NA"]
                values.append(found[0] if found else np.nan)
            ax.plot([str(seed) for seed in SEEDS], values, linewidth=1.7, label=contrast, **contrast_styles[contrast])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("formal seed")
        ax.set_ylabel("positive = benefit")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    support_row = next(row for row in read_csv(ROOT / "inventory/DEV_SUPPORT_V2.csv") if int(row["task_id"]) == 119)
    fig.suptitle(
        f"Task 119 waypoint: action contrasts and goal endpoint error (dev source groups n={support_row['dev_clusters_after_overlap_exclusion']})"
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures/WAYPOINT_TASK_119.png", dpi=170)
    plt.close(fig)
    endpoint_rows = read_csv(ROOT / "results/WAYPOINT_ENDPOINT.csv")
    if endpoint_rows:
        write_csv(ROOT / "figure_data/WAYPOINT_ENDPOINT_PLOT_DATA.csv", list(endpoint_rows[0]), endpoint_rows)


def main() -> None:
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "figure_data").mkdir(parents=True, exist_ok=True)
    plot_train_tune()
    plot_contrasts()
    plot_trajectories()
    plot_waypoint()
    figure_files = sorted(path for path in (ROOT / "figures").glob("*.png") if path.stat().st_size > 0)
    data_files = sorted(path for path in (ROOT / "figure_data").glob("*.csv") if path.stat().st_size > 0)
    if not figure_files or not data_files:
        raise RuntimeError("V2 figure or figure-data outputs are empty")
    from PIL import Image
    image_metadata = []
    for path in figure_files:
        with Image.open(path) as image:
            image_metadata.append({
                "path": str(path), "format": image.format, "mode": image.mode,
                "pixel_size": list(image.size), "dpi": image.info.get("dpi"),
                "file_bytes": path.stat().st_size,
            })
            image.verify()
    if any(row["format"] != "PNG" or min(row["pixel_size"]) < 400 for row in image_metadata):
        raise RuntimeError("one or more V2 PNG figures failed raster metadata QA")
    palette_audit_path = ROOT / "qa/PALETTE_AUDIT.json"
    palette_audit = json.loads(palette_audit_path.read_text(encoding="utf-8"))
    if palette_audit.get("status") != "PASS" or palette_audit.get("all_palette_colors_pass_background_graphical_contrast") is not True:
        raise RuntimeError("V2 palette audit is missing or failed background-contrast screening")
    qa = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "figure_count": len(figure_files),
        "figure_paths": [str(path) for path in figure_files],
        "figure_data_count": len(data_files),
        "figure_data_paths": [str(path) for path in data_files],
        "image_metadata": image_metadata,
        "matplotlib_version": matplotlib.__version__,
        "palette": "Okabe-Ito on white with line-style, marker and hatch redundancy",
        "palette_audit_sha256": hashlib.sha256(palette_audit_path.read_bytes()).hexdigest(),
        "palette_grayscale_review_count": palette_audit["pairwise_grayscale_review_count"],
        "unit_note": "native command axes remain unidentified numeric values",
        "seed_spread": "individual formal seed points; sample SD shown on contrast bars",
        "train_tune_curves_included": (ROOT / "figures/TRAIN_TUNE_CURVES.png").is_file(),
        "prediction_target_persistence_trajectories_included": any(path.name.startswith("TRAJECTORY_TASK_") for path in figure_files),
        "human_review_status": "pending",
        "final_heldout_payload_access": 0,
    }
    (ROOT / "qa/FIGURE_BUILD_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    descriptions = [
        "TRAIN_TUNE_CURVES.png: For each of 12 programs, paired panels show weighted training objective each epoch and episode-equal inner-tune J at epochs 40, 80, and 160; lower is preferred.",
        "CONTRASTS_POSITION_MAE.png and CONTRASTS_SAFE_FD_L1.png: Five paired task contrasts; bars show mean across three seeds with sample SD, black marker shapes show individual seeds, and zero is no change.",
        "TRAJECTORY_TASK_*.png: Up to two dev episodes chosen by canonical episode-ID SHA-256 per task; each panel plots observed target, Persistence, and formal-seed predictions for action indices 0 and 14. Units are unidentified numeric command values.",
        "WAYPOINT_TASK_119.png: Per-seed action and predicted-goal endpoint gains for task 119; positive is a lower error for the named intervention.",
    ]
    (ROOT / "figures/CAPTIONS_AND_DESCRIPTIONS.md").write_text("\n".join(descriptions) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "figures": sorted(path.name for path in (ROOT / "figures").glob("*.png")),
        "figure_data": sorted(path.name for path in (ROOT / "figure_data").glob("*.csv")),
        "human_review_status": "pending",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

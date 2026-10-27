#!/usr/bin/env python3
"""Create locked-seed scheduler effects, amplitude, and fixed-anchor figures."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
FIG_DIR = TASK / "figures"
STYLE = FIG_DIR / "assets/publication.mplstyle"
sys.path.insert(0, str(FIG_DIR / "assets"))
from color_palettes import OKABE_ITO_ON_WHITE  # noqa: E402

CONFIGS = ("TS11-D", "STATE-CTRL")
SEEDS = (20261001, 20261002, 20261003)
COLORS = {"TS11-D": OKABE_ITO_ON_WHITE[0], "STATE-CTRL": OKABE_ITO_ON_WHITE[1]}
MARKERS = {"TS11-D": "o", "STATE-CTRL": "s"}
OLD_ID = "old_as_run"
NEW_ID = "corrected_scheduler"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def verify_prediction_acceptance(matrix: list[dict[str, str]], protocol_sha: str) -> None:
    qa = json.loads((TASK / "EVALUATION_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or int(qa.get("train_prediction_bundles", -1)) != 12:
        raise AssertionError("train/dev scorer has not accepted the complete 12-cell prediction grid")
    if int(qa.get("dev_prediction_bundles_inferred", -1)) != 12:
        raise AssertionError("dev prediction grid is not 12/12 accepted")
    for row in matrix:
        complete_path = TASK / "runs/formal" / row["run_id"] / "RUN_COMPLETE.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS" or complete.get("study_protocol_hash") != protocol_sha:
            raise AssertionError(f"unaccepted formal checkpoint for figure source: {row['run_id']}")
        for split in ("train", "dev"):
            side = TASK / "predictions" / split / f"{row['run_id']}.json"
            record = json.loads(side.read_text(encoding="utf-8"))
            prediction = side.with_suffix(".npz")
            if (record.get("status") != "PASS"
                or record.get("checkpoint_sha256") != complete.get("checkpoint_last_sha256")
                or record.get("prediction_sha256") != sha_file(prediction)
                or record.get("study_protocol_sha256") != protocol_sha
                or record.get("target_arrays_loaded") is not False):
                raise AssertionError(f"figure source bundle is not accepted: {row['run_id']} {split}")


def plot_scheduler_gain() -> tuple[plt.Figure, str, str]:
    source = TASK / "analysis/SCHEDULE_GAINS_SUMMARY.csv"
    data = [row for row in rows(source)
            if row["split"] == "dev" and row["region_raw"] == "action_full28"
            and row["family"] == "primary"]
    expected = {(cfg, h, metric) for cfg in CONFIGS for h in (30, 60)
                for metric in ("standardized_mae", "native_future_internal_fd")}
    seen = {(row["config"], int(row["horizon_steps"]), row["metric"]) for row in data}
    if seen != expected:
        raise AssertionError(f"primary dev effect rows mismatch: {len(data)}")
    labels = {
        "standardized_mae": "Standardized MAE (dimensionless)",
        "native_future_internal_fd": "Native future-internal FD (source units)",
    }
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
        for row_index, metric in enumerate(("standardized_mae", "native_future_internal_fd")):
            for col_index, horizon in enumerate((30, 60)):
                ax = axes[row_index, col_index]
                panel = [row for row in data if row["metric"] == metric
                         and int(row["horizon_steps"]) == horizon]
                y_positions = {"STATE-CTRL": 0.0, "TS11-D": 1.0}
                all_limits = [0.0]
                for row in panel:
                    config = row["config"]
                    y = y_positions[config]
                    estimate = float(row["estimate"])
                    low = float(row["ci95_low"])
                    high = float(row["ci95_high"])
                    color = COLORS[config]
                    ax.errorbar(
                        estimate, y, xerr=[[estimate - low], [high - estimate]],
                        fmt=MARKERS[config], color=color, ecolor=color,
                        capsize=3, markersize=5, linewidth=1.2, zorder=3,
                    )
                    seed_points = json.loads(row["seed_points"])
                    offsets = (-0.16, 0.0, 0.16)
                    for offset, seed in zip(offsets, SEEDS):
                        ax.scatter(
                            float(seed_points[str(seed)]), y + offset, marker="x",
                            color=color, s=20, linewidths=0.9, zorder=4,
                        )
                    all_limits.extend([low, high])
                span = max(max(abs(value) for value in all_limits), 1e-8)
                ax.set_xlim(-1.15 * span, 1.15 * span)
                ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
                ax.set_yticks([0.0, 1.0], ["STATE-CTRL", "TS11-D"])
                ax.set_title(f"H{horizon} · {labels[metric]}")

                ax.grid(axis="x", color="0.85", linewidth=0.5)
        fig.supxlabel("Schedule gain (old - corrected loss; positive improves)", fontsize=8)
        handles = [
            Line2D([0], [0], marker=MARKERS[cfg], color=COLORS[cfg], linestyle="none", label=cfg)
            for cfg in CONFIGS
        ]
        handles.append(Line2D([0], [0], marker="x", color="0.25", linestyle="none", label="seed point"))
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    return fig, (
        "Points show the fixed three-seed mean of paired old-as-run minus corrected-scheduler loss; "
        "whiskers are 95% source-group bootstrap CIs conditional on these trained models. "
        "Crosses show the three seed-specific group means. Positive values indicate lower corrected-scheduler loss."
    ), (
        "Four-panel forest plot for Full28 standardized MAE and native future-internal FD at H30 and H60. "
        "Separate rows show STATE-CTRL and TS11-D; zero lines mark no schedule gain. "
        "Seed points and group-bootstrap intervals are shown separately."
    )


def plot_video_specific_gain() -> tuple[plt.Figure, str, str]:
    source = TASK / "analysis/VIDEO_SPECIFIC_GAIN_SUMMARY.csv"
    data = [row for row in rows(source)
            if row["split"] == "dev" and row["region_raw"] == "action_full28"
            and row["family"] == "secondary"]
    expected = {(h, metric) for h in (30, 60)
                for metric in ("standardized_mae", "native_future_internal_fd")}
    if {(int(row["horizon_steps"]), row["metric"]) for row in data} != expected:
        raise AssertionError("video-specific secondary family must contain exactly four dev estimates")
    labels = {
        "standardized_mae": "Standardized MAE (dimensionless)",
        "native_future_internal_fd": "Native future-internal FD (source units)",
    }
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), layout="constrained")
        for row_index, metric in enumerate(("standardized_mae", "native_future_internal_fd")):
            for col_index, horizon in enumerate((30, 60)):
                ax = axes[row_index, col_index]
                row = next(item for item in data
                           if int(item["horizon_steps"]) == horizon and item["metric"] == metric)
                estimate = float(row["estimate"])
                low, high = float(row["ci95_low"]), float(row["ci95_high"])
                ax.errorbar(
                    estimate, 0.0, xerr=[[estimate - low], [high - estimate]],
                    fmt="D", color=OKABE_ITO_ON_WHITE[2], ecolor=OKABE_ITO_ON_WHITE[2],
                    capsize=3, markersize=5, linewidth=1.2,
                )
                seed_points = json.loads(row["seed_points"])
                for offset, seed in zip((-0.14, 0.0, 0.14), SEEDS):
                    ax.scatter(float(seed_points[str(seed)]), offset, marker="x",
                               color=OKABE_ITO_ON_WHITE[4], s=20, linewidths=0.9)
                span = max(abs(low), abs(high), 1e-8)
                ax.set_xlim(-1.15 * span, 1.15 * span)
                ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
                ax.set_yticks([0.0], ["TS11-D − STATE-CTRL"])
                ax.set_title(f"H{horizon} · {labels[metric]}")

                ax.grid(axis="x", color="0.85", linewidth=0.5)
        fig.supxlabel("Video-specific gain difference (positive favors TS11-D)", fontsize=8)
        fig.suptitle("Video-specific optimization sensitivity (secondary family)", fontsize=11)
    return fig, (
        "Each estimate is (TS11-D schedule gain) minus (STATE-CTRL schedule gain), "
        "with gain defined as old loss minus corrected-scheduler loss. "
        "Whiskers are 95% source-group bootstrap CIs, conditional on three fixed seeds; "
        "the four rows form the separately Holm-adjusted secondary family."
    ), (
        "Four-panel forest plot of video-specific scheduler gain differences for standardized MAE and native FD at H30 and H60. "
        "Zero means equal schedule gain for TS11-D and STATE-CTRL; positive values favor a larger gain for TS11-D."
    )


def plot_tv_train_dev() -> tuple[plt.Figure, str, str]:
    source = TASK / "analysis/MODEL_METRICS_BY_SEED.csv"
    data = [row for row in rows(source) if row["region_raw"] == "action_full28"]
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True, layout="constrained")
        for row_index, horizon in enumerate((30, 60)):
            for col_index, split in enumerate(("train", "dev")):
                ax = axes[row_index, col_index]
                panel = [row for row in data if row["split"] == split and int(row["horizon_steps"]) == horizon]
                xloc = {
                    ("TS11-D", "old_as_run"): 0.0,
                    ("TS11-D", "corrected_scheduler"): 0.55,
                    ("STATE-CTRL", "old_as_run"): 1.75,
                    ("STATE-CTRL", "corrected_scheduler"): 2.30,
                }
                for row in panel:
                    config, variant = row["config"], row["variant"]
                    denom = float(row["tv_gt_mean"])
                    ratio = float(row["tv_pred_mean"]) / denom if denom > 0 else float("nan")
                    if not np.isfinite(ratio):
                        continue
                    x = xloc[(config, variant)]
                    seed_offset = {20261001: -0.07, 20261002: 0.0, 20261003: 0.07}[int(row["seed"])]
                    marker = "o" if variant == "old_as_run" else "^"
                    face = "none" if variant == "old_as_run" else COLORS[config]
                    ax.scatter(x + seed_offset, ratio, marker=marker, s=27,
                               edgecolor=COLORS[config], facecolor=face, linewidth=1.0, zorder=3)
                means = {}
                for key, x in xloc.items():
                    vals = []
                    for row in panel:
                        if (row["config"], row["variant"]) == key:
                            denom = float(row["tv_gt_mean"])
                            if denom > 0:
                                vals.append(float(row["tv_pred_mean"]) / denom)
                    means[key] = float(np.mean(vals)) if vals else float("nan")
                    if vals:
                        ax.scatter(x, means[key], marker="D", s=38, color=COLORS[key[0]],
                                   edgecolor="black", linewidth=0.5, zorder=4)
                ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9, label="TVpred = TVGT")
                ax.set_ylim(0.0, 1.05)
                ax.set_xticks([0.0, 0.55, 1.75, 2.30],
                              ["TS11-D\nold", "TS11-D\ncorrected", "STATE-CTRL\nold", "STATE-CTRL\ncorrected"])
                ax.set_title(f"{split.title()} · H{horizon}")
                ax.set_ylabel("Group-equal TVpred / TVGT")
                ax.grid(axis="y", color="0.88", linewidth=0.5)
        legend = [
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="0.2",
                   linestyle="none", label="seed point (old as-run)"),
            Line2D([0], [0], marker="^", markerfacecolor="0.2", markeredgecolor="0.2",
                   linestyle="none", label="seed point (corrected)"),
            Line2D([0], [0], marker="D", color="0.2", linestyle="none", label="three-seed mean"),
            Line2D([0], [0], color="0.35", linestyle="--", label="ratio = 1"),
        ]
        fig.legend(handles=legend, loc="outside upper center", ncol=4, frameon=False)
    return fig, (
        "Each circle/triangle is one fixed-seed source-group-equal TV ratio; diamonds are descriptive means of three seeds. "
        "The dashed line is TVpred/TVGT=1. Read amplitude jointly with MAE, FD, and direction; "
        "a ratio closer to one alone does not establish motion fidelity."
    ), (
        "Four-panel train/dev by H30/H60 plot of group-equal predicted total variation divided by ground-truth total variation. "
        "Three seed points and their descriptive mean are shown for old-as-run and corrected-scheduler TS11-D and STATE-CTRL. "
        "The common y-axis runs from zero to one, with a dashed target at one."
    )


def read_bundle(path: Path, sidecar_path: Path, *, config: str, seed: int, horizon: int,
                checkpoint_sha: str, study_sha: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if side.get("status") != "PASS" or side.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError(f"trajectory bundle checkpoint/status mismatch: {path.name}")
    if side.get("prediction_sha256") != sha_file(path):
        raise AssertionError(f"trajectory bundle hash mismatch: {path.name}")
    if study_sha is not None and side.get("study_protocol_sha256") != study_sha:
        raise AssertionError(f"trajectory task protocol mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as bundle:
        ids = bundle["anchor_id"].astype("U96", copy=True)
        pred = bundle["predictions_native"].astype(np.float32, copy=True)
        if (str(bundle["config"].item()), int(bundle["seed"].item()), int(bundle["horizon_steps"])) != (config, seed, horizon):
            raise AssertionError(f"trajectory bundle identity mismatch: {path.name}")
    return pred, {"ids": ids, "sidecar": side}


def verify_trajectory_bundles(matrix: list[dict[str, str]], study_sha: str) -> None:
    for row in matrix:
        complete = json.loads((TASK / "runs/formal" / row["run_id"] / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        if complete.get("status") != "PASS" or complete.get("study_protocol_hash") != study_sha:
            raise AssertionError(f"trajectory source checkpoint unaccepted: {row['run_id']}")
        for split in ("train", "dev"):
            sidecar = TASK / "predictions" / split / f"{row['run_id']}.json"
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("status") != "PASS" or data.get("target_arrays_loaded") is not False:
                raise AssertionError(f"trajectory inference is not target-blind/accepted: {row['run_id']} {split}")
            if data.get("study_protocol_sha256") != study_sha:
                raise AssertionError(f"trajectory inference protocol mismatch: {row['run_id']} {split}")


def plot_fixed_trajectories(matrix: list[dict[str, str]], study_sha: str) -> tuple[plt.Figure, str, str, dict[str, Any]]:
    selection_path = TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    representatives = list(selection["representatives"])
    target_run = next(row for row in matrix
                      if row["config"] == "TS11-D" and int(row["seed"]) == 20261001
                      and int(row["horizon_steps"]) == 60)
    old_id = target_run["matched_old_run_id"]
    new_id = target_run["run_id"]
    old_run = EQ / "d3/runs" / old_id
    old_complete = json.loads((old_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    new_run = TASK / "runs/formal" / new_id
    new_complete = json.loads((new_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    old_path = EQ / "d3/predictions" / f"{old_id}.npz"
    old_side = old_path.with_suffix(".json")
    new_path = TASK / "predictions/dev" / f"{new_id}.npz"
    new_side = new_path.with_suffix(".json")

    # Fixed failure illustration is selected by old-as-run Full28 dev sMAE only;
    # corrected predictions are deliberately not loaded until after this choice.
    old_pred, old_bundle = read_bundle(
        old_path, old_side, config="TS11-D", seed=20261001, horizon=60,
        checkpoint_sha=old_complete["checkpoint_last_sha256"],
    )
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        split = archive["split_code"].astype(np.int8, copy=True)
        dev_idx = np.flatnonzero(split == 1).astype(np.int64)
        ids = archive["anchor_id"][dev_idx].astype("U96", copy=True)
        anchors = archive["anchor_command_native"][dev_idx].astype(np.float32, copy=True)
        targets = archive["targets_native"][dev_idx].astype(np.float32, copy=True)
    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    lookup_old = {value: index for index, value in enumerate(old_bundle["ids"].tolist())}
    old_order = np.asarray([lookup_old[value] for value in ids.tolist()], dtype=np.int64)
    old_pred = old_pred[old_order]
    losses = np.abs(old_pred - targets) / sigma[None, None, :]
    per_anchor_loss = losses.mean(axis=(1, 2))
    failure_index = min(range(len(ids)), key=lambda i: (-float(per_anchor_loss[i]), str(ids[i])))
    failure_id = str(ids[failure_index])
    failure_record = {
        "status": "LOCKED_SELECTION_APPLIED",
        "selection_rule": "post-acceptance illustrative case: maximum old-as-run TS11-D/H60/S20261001 Full28 standardized dev MAE; tie by anchor_id",
        "selection_old_run_id": old_id,
        "anchor_id": failure_id,
        "old_as_run_standardized_mae": float(per_anchor_loss[failure_index]),
        "new_checkpoint_predictions_read_for_selection": False,
        "selection_is_inferential": False,
        "final_test_access": 0,
    }
    write_json(TASK / "analysis/FAILURE_TRAJECTORY_SELECTION.json", failure_record)

    new_pred, new_bundle = read_bundle(
        new_path, new_side, config="TS11-D", seed=20261001, horizon=60,
        checkpoint_sha=new_complete["checkpoint_last_sha256"], study_sha=study_sha,
    )
    new_lookup = {value: index for index, value in enumerate(new_bundle["ids"].tolist())}
    new_order = np.asarray([new_lookup[value] for value in ids.tolist()], dtype=np.int64)
    new_pred = new_pred[new_order]

    ids_selected: list[dict[str, Any]] = []
    selected_by_id = {}
    for representative in representatives:
        selected_by_id[str(representative["anchor_id"])] = {
            "anchor_id": str(representative["anchor_id"]),
            "selection_role": "outcome_blind_locked_representative",
        }
    if failure_id in selected_by_id:
        selected_by_id[failure_id]["selection_role"] += "+old_as_run_failure_illustration"
    else:
        selected_by_id[failure_id] = {
            "anchor_id": failure_id,
            "selection_role": "old_as_run_failure_illustration_preselected_without_new_scores",
        }
    id_order = {value: index for index, value in enumerate(ids.tolist())}
    for value, detail in selected_by_id.items():
        if value not in id_order:
            raise AssertionError(f"fixed trajectory ID absent from primary dev: {value}")
        detail["index"] = id_order[value]
        ids_selected.append(detail)
    ids_selected.sort(key=lambda item: str(item["anchor_id"]))

    # Load STATE-CTRL old/new once for the same locked IDs.
    state_row = next(row for row in matrix
                     if row["config"] == "STATE-CTRL" and int(row["seed"]) == 20261001
                     and int(row["horizon_steps"]) == 60)
    state_old_id = state_row["matched_old_run_id"]
    state_new_id = state_row["run_id"]
    state_old_run = EQ / "d3/runs" / state_old_id
    state_old_complete = json.loads((state_old_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    state_new_run = TASK / "runs/formal" / state_new_id
    state_new_complete = json.loads((state_new_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    state_old_path = EQ / "d3/predictions" / f"{state_old_id}.npz"
    state_old_pred, state_old_bundle = read_bundle(
        state_old_path, state_old_path.with_suffix(".json"), config="STATE-CTRL",
        seed=20261001, horizon=60, checkpoint_sha=state_old_complete["checkpoint_last_sha256"],
    )
    state_new_path = TASK / "predictions/dev" / f"{state_new_id}.npz"
    state_new_pred, state_new_bundle = read_bundle(
        state_new_path, state_new_path.with_suffix(".json"), config="STATE-CTRL",
        seed=20261001, horizon=60, checkpoint_sha=state_new_complete["checkpoint_last_sha256"],
        study_sha=study_sha,
    )
    state_old_lookup = {value: index for index, value in enumerate(state_old_bundle["ids"].tolist())}
    state_new_lookup = {value: index for index, value in enumerate(state_new_bundle["ids"].tolist())}
    state_old_pred = state_old_pred[np.asarray([state_old_lookup[value] for value in ids.tolist()])]
    state_new_pred = state_new_pred[np.asarray([state_new_lookup[value] for value in ids.tolist()])]

    series_by_sample = []
    for selection_detail in ids_selected:
        index = int(selection_detail["index"])
        anchor = anchors[index]
        truth = targets[index]
        series_by_sample.append({
            "selection": selection_detail,
            "anchor": anchor,
            "truth": truth,
            "old_ts11d": old_pred[index],
            "new_ts11d": new_pred[index],
            "old_state": state_old_pred[index],
            "new_state": state_new_pred[index],
        })
    dimensions = (0, 14)
    minima = {dimension: float("inf") for dimension in dimensions}
    maxima = {dimension: float("-inf") for dimension in dimensions}
    for sample in series_by_sample:
        for dimension in dimensions:
            values = [float(sample["anchor"][dimension])]
            for key in ("truth", "old_ts11d", "new_ts11d", "old_state", "new_state"):
                values.extend(np.asarray(sample[key])[:, dimension].astype(float).tolist())
            minima[dimension] = min(minima[dimension], min(values))
            maxima[dimension] = max(maxima[dimension], max(values))

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(len(series_by_sample), 2, figsize=(8.0, 2.05 * len(series_by_sample)),
                                 sharex=True, layout="constrained", squeeze=False)
        time_axis = np.arange(61)
        for row_index, sample in enumerate(series_by_sample):
            selection = sample["selection"]
            for col_index, dimension in enumerate(dimensions):
                ax = axes[row_index, col_index]
                anchor_value = float(sample["anchor"][dimension])
                ax.plot(time_axis, np.r_[anchor_value, sample["truth"][:, dimension]],
                        color="black", linestyle="-", linewidth=1.8, label="Ground truth")
                ax.plot(time_axis, np.r_[anchor_value, sample["old_ts11d"][:, dimension]],
                        color=COLORS["TS11-D"], linestyle="--", linewidth=1.0, label="TS11-D old")
                ax.plot(time_axis, np.r_[anchor_value, sample["new_ts11d"][:, dimension]],
                        color=COLORS["TS11-D"], linestyle="-", linewidth=1.4, label="TS11-D corrected")
                ax.plot(time_axis, np.r_[anchor_value, sample["old_state"][:, dimension]],
                        color=COLORS["STATE-CTRL"], linestyle="--", linewidth=1.0, label="STATE-CTRL old")
                ax.plot(time_axis, np.r_[anchor_value, sample["new_state"][:, dimension]],
                        color=COLORS["STATE-CTRL"], linestyle="-", linewidth=1.4, label="STATE-CTRL corrected")
                ax.axhline(anchor_value, color="0.45", linestyle=":", linewidth=0.9, label="Persistence")
                low, high = minima[dimension], maxima[dimension]
                pad = max((high - low) * 0.05, 1e-6)
                ax.set_ylim(low - pad, high + pad)
                ax.set_xlim(0, 60)
                ax.set_xlabel("Future step (30 Hz)" if row_index == len(series_by_sample) - 1 else "")
                ax.set_ylabel(f"action[{dimension}] (source units)")
                role = selection["selection_role"]
                if col_index == 0:
                    role_label = "prelocked representative" if role.startswith("outcome_blind") else "old-model failure illustration"
                    ax.set_title(f"{selection['anchor_id']} · {role_label}", fontsize=7)
                elif row_index == 0:
                    ax.set_title("action[14]", fontsize=8)
                ax.grid(axis="y", color="0.88", linewidth=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fixed_ids = ", ".join(str(row["anchor_id"]) for row in representatives)
    return fig, (
        "Native source-unit trajectories for prelocked dev representatives (" + fixed_ids + ") plus an illustrative case "
        "selected only from old TS11-D/H60/S20261001 before reading corrected predictions. "
        "The illustrative failure ID is " + failure_id + "; it is not inferential. Raw action[0] and action[14] have no anatomical labels."
    ), (
        "Grid of future action trajectories for fixed dev anchors, with separate columns for raw action[0] and action[14]. "
        "Each panel shows ground truth, persistence, old-as-run TS11-D and STATE-CTRL, and corrected-scheduler TS11-D and STATE-CTRL."
    ), {"fixed_representatives": [row["anchor_id"] for row in representatives], "failure_case": failure_record}


def export_figure(fig: plt.Figure, stem: str, caption: str, alt_text: str,
                  source_paths: list[Path]) -> dict[str, Any]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, format="pdf", facecolor="white")
    fig.savefig(png, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(png) as image_file:
        image_file.verify()
    image = Image.open(png)
    alpha_extrema = image.getchannel("A").getextrema() if "A" in image.getbands() else (255, 255)
    png_opaque = alpha_extrema == (255, 255)
    record = {
        "pdf": str(pdf), "pdf_sha256": sha_file(pdf), "pdf_bytes": pdf.stat().st_size,
        "png": str(png), "png_sha256": sha_file(png), "png_bytes": png.stat().st_size,
        "png_pixel_size": list(image.size), "png_mode": image.mode,
        "png_alpha_extrema": list(alpha_extrema), "png_opaque": png_opaque,
        "figure_size_inches": [float(x) for x in fig.get_size_inches()],
        "caption": caption, "alt_text": alt_text,
        "source_hashes": {str(path): sha_file(path) for path in source_paths},
        "publisher_compliance": "not asserted; no journal/submission target specified",
    }
    image.close()
    return record


def verify_model_metrics_by_seed(matrix: list[dict[str, str]]) -> dict[str, Any]:
    path = TASK / "analysis/MODEL_METRICS_BY_SEED.csv"
    metric_rows = rows(path)
    expected_run_ids = {}
    for row in matrix:
        key = (row["config"], int(row["seed"]), int(row["horizon_steps"]))
        expected_run_ids[(OLD_ID, *key)] = row["matched_old_run_id"]
        expected_run_ids[(NEW_ID, *key)] = row["run_id"]
    seen = set()
    for row in metric_rows:
        key = (row["split"], row["variant"], row["config"], int(row["seed"]),
               int(row["horizon_steps"]), row["region_raw"])
        if key in seen:
            raise AssertionError(f"duplicate model-metric seed cell: {key}")
        seen.add(key)
        expected = expected_run_ids.get((row["variant"], row["config"],
                                         int(row["seed"]), int(row["horizon_steps"])))
        if row["run_id"] != expected:
            raise AssertionError(f"model-metric run ID does not match locked matrix: {key}")
    metric_qa_path = TASK / "analysis/MODEL_METRICS_QA.json"
    metric_qa = json.loads(metric_qa_path.read_text(encoding="utf-8"))
    if metric_qa.get("status") != "PASS" or len(metric_rows) != 144:
        raise AssertionError("deduplicated model-metric table is not accepted at 144 rows")
    return {"rows": len(metric_rows), "sha256": sha_file(path),
            "qa_sha256": sha_file(metric_qa_path)}


def main() -> int:
    protocol = json.loads((TASK / "PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED":
        raise AssertionError("figure generation requires a locked analysis protocol")
    study_sha = sha_file(TASK / "PROTOCOL_LOCKED.json")
    matrix = list(csv.DictReader((TASK / "protocol/MATRIX_12.csv").open(newline="", encoding="utf-8")))
    metric_table = verify_model_metrics_by_seed(matrix)
    verify_prediction_acceptance(matrix, study_sha)

    figure_records = {}
    sources = [
        TASK / "analysis/SCHEDULE_GAINS_SUMMARY.csv",
        TASK / "analysis/VIDEO_SPECIFIC_GAIN_SUMMARY.csv",
        TASK / "analysis/MODEL_METRICS_BY_SEED.csv",
        TASK / "analysis/MODEL_METRICS_QA.json",
        TASK / "analysis/ANALYSIS_QA.json",
        TASK / "FIGURE_PALETTE_AUDIT.json",
        TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json",
    ]
    fig, caption, alt = plot_scheduler_gain()
    figure_records["schedule_gains"] = export_figure(
        fig, "FIG_CORRECTED_SCHEDULER_GAINS_V1", caption, alt,
        [TASK / "analysis/SCHEDULE_GAINS_SUMMARY.csv"],
    )
    fig, caption, alt = plot_video_specific_gain()
    figure_records["video_specific_gain"] = export_figure(
        fig, "FIG_VIDEO_SPECIFIC_SCHEDULER_GAIN_V1", caption, alt,
        [TASK / "analysis/VIDEO_SPECIFIC_GAIN_SUMMARY.csv"],
    )
    fig, caption, alt = plot_tv_train_dev()
    figure_records["tv_train_dev"] = export_figure(
        fig, "FIG_TV_AMPLITUDE_TRAIN_DEV_V1", caption, alt,
        [TASK / "analysis/MODEL_METRICS_BY_SEED.csv"],
    )
    fig, caption, alt, selection = plot_fixed_trajectories(matrix, study_sha)
    figure_records["fixed_trajectories"] = export_figure(
        fig, "FIG_FIXED_OLD_NEW_GT_TRAJECTORIES_V1", caption, alt,
        [TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json",
         TASK / "analysis/FAILURE_TRAJECTORY_SELECTION.json"],
    )
    manifest = {
        "status": "PASS",
        "study_protocol_sha256": study_sha,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "figures": figure_records,
        "trajectory_selection": selection,
        "figure_intended_destination": "internal scientific report",
        "generator_sha256": sha_file(Path(__file__).resolve()),
        "model_metrics_table": metric_table,
        "palette_audit_sha256": sha_file(TASK / "FIGURE_PALETTE_AUDIT.json"),
        "palette_screen_note": "Background contrast screen passes; four of ten pairwise grayscale separations flag for heuristic review. Redundant labels/markers/linestyles are used; no accessibility certification is claimed.",
        "publisher_compliance": "not asserted",
        "agent_visual_review": "pending",
        "human_visual_review": "pending",
        "final_test_access": 0,
    }
    write_json(FIG_DIR / "FIGURE_MANIFEST.json", manifest)
    caption_md = [
        "# Figures and accessibility text",
        "",
        "No publisher or manuscript target was specified; these are internal scientific report figures, not a compliance claim.",
        "",
    ]
    for key, record in figure_records.items():
        caption_md.extend([f"## {key}", "", f"Caption: {record['caption']}", "", f"Alt text: {record['alt_text']}", ""])
    (TASK / "FIGURE_CAPTIONS_AND_ALT.zh-CN.md").write_text("\n".join(caption_md) + "\n", encoding="utf-8")
    all_pngs_nonempty = all(row["png_bytes"] > 0 for row in figure_records.values())
    all_pdfs_nonempty = all(row["pdf_bytes"] > 0 for row in figure_records.values())
    all_pngs_opaque = all(row["png_opaque"] for row in figure_records.values())
    qa = {
        "status": "PASS" if len(figure_records) == 4 and all_pngs_nonempty and all_pdfs_nonempty and all_pngs_opaque else "NEEDS_REVIEW",
        "figures": len(figure_records),
        "all_pngs_rgb": all(row["png_mode"] == "RGB" for row in figure_records.values()),
        "all_pngs_opaque": all_pngs_opaque,
        "all_pngs_nonempty": all_pngs_nonempty,
        "all_pdfs_nonempty": all_pdfs_nonempty,
        "model_metrics_seed_rows": metric_table["rows"],
        "model_metrics_table_sha256": metric_table["sha256"],
        "palette_audit_sha256": sha_file(TASK / "FIGURE_PALETTE_AUDIT.json"),
        "agent_visual_review": "pending",
        "human_visual_review": "pending",
        "publisher_compliance": "not asserted",
        "final_test_access": 0,
    }
    write_json(TASK / "FIGURE_QA.json", qa)
    print(json.dumps(qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render the eight Core B harvest figures from independent CSV tables."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT = Path(os.environ.get("CORE_B_ANALYSIS_OUT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z')))
FIGURES = OUT / "figures"
DATA = OUT / "figure_data"
SCRIPT = OUT / "code/make_core_b_figures.py"
COLORS = {"frozen": "#486581", "lora": "#d64550", "dynamic": "#2f855a", "static": "#805ad5", "control": "#9aa5b1", "proxy": "#9c6644", "true": "#2f855a", "model": "#d97706", "persistence": "#7b2cbf", "target": "#102a43"}
STEPS = 30


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("png", "svg", "pdf"):
        path = FIGURES / f"{stem}.{ext}"
        fig.savefig(path, dpi=240, bbox_inches="tight" if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def fig01():
    rows = [r for r in read_csv("CORE_B_UNIFIED_RESULTS.csv") if r["stage"] in {"B1", "CONTROL"}]
    order = ["ZERO-ACTION", "TRAIN-GLOBAL-MEAN", "TRAIN-PER-TIME-MEAN", "PERSISTENCE", "B-STATE", "B-D-P-F", "B-S-P-F", "B-D-A-F", "B-S-A-F", "B-D-FUS-P-F", "B-S-FUS-P-F"]
    rows.sort(key=lambda r: order.index(r["cell"]) if r["cell"] in order else 99)
    write_csv(DATA / "fig01_b1_baseline_comparison.csv", rows)
    labels = [r["cell"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), layout="constrained")
    for ax, metric, title in zip(axes, ("std_arm_mae", "std_hand_mae", "std_full28_mae"), ("Arm14 standardized MAE", "Hand14 standardized MAE", "Full28 standardized MAE")):
        values = np.array([f(r[metric]) for r in rows])
        colors = [COLORS["control"] if r["stage"] == "CONTROL" else COLORS["dynamic"] if r["video_input"] == "dynamic" else COLORS["static"] if r["video_input"] == "static" else COLORS["target"] for r in rows]
        ax.bar(x, values, color=colors, edgecolor="white")
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Lower is better")
        ax.set_title(title)
        style(ax)
    axes[2].legend(handles=[Patch(facecolor=COLORS["control"], label="controls"), Patch(facecolor=COLORS["dynamic"], label="dynamic"), Patch(facecolor=COLORS["static"], label="static"), Patch(facecolor=COLORS["target"], label="fusion/state")], frameon=False, fontsize=8)
    fig.suptitle("Core B B1 canonical baselines and controls", fontsize=15, fontweight="bold")
    return save(fig, "fig01_b1_baseline_comparison"), "Canonical B1 learned lanes and train-only/persistence controls across arm14, hand14 and full28 standardized trajectory MAE.", "B1 各 canonical learned lane 与 zero、train mean、persistence 控制在 arm14、hand14、full28 standardized MAE 上的比较；数值越低越好。", "fig01_b1_baseline_comparison.csv"


def fig02():
    rows = [r for r in read_csv("CORE_B_PAIRWISE_COMPARISONS.csv") if r["stage"] == "B1" and r["family"] in {"B1_learned_vs_persistence", "B1_dynamic_static", "B1_pooled_attentive", "B1_fusion_vs_state"}]
    write_csv(DATA / "fig02_b1_effects.csv", rows)
    rows.sort(key=lambda r: f(r["improvement_point"]))
    labels = [r["comparison"] for r in rows]
    y = np.arange(len(rows))
    value = np.array([f(r["improvement_point"]) for r in rows])
    low = np.array([f(r["ci95_lower"]) for r in rows])
    high = np.array([f(r["ci95_upper"]) for r in rows])
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    ax.axvline(0, color="#334e68", linewidth=1)
    ax.errorbar(value, y, xerr=[value - low, high - value], fmt="o", color=COLORS["model"], ecolor="#334e68", capsize=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Reference MAE − candidate MAE (positive = candidate improves)")
    ax.set_title("Core B B1 paired episode effects")
    style(ax)
    return save(fig, "fig02_b1_effects_forest"), "Paired episode bootstrap effects for B1 dynamic/static, pooled/attentive, fusion/state and persistence comparisons.", "B1 配对 episode effect forest plot；横轴为 reference MAE 减 candidate MAE，正值表示 candidate 更低误差。", "fig02_b1_effects.csv"


def fig03():
    rows = [r for r in read_csv("B2_RESULTS_INDEPENDENT.csv") if r["stage"] == "B2"]
    write_csv(DATA / "fig03_true_b2_sampling.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    for ax, metric, title in zip(axes, ("std_arm_mae", "std_full28_mae"), ("Arm14 MAE", "Full28 MAE")):
        for branch, color, marker in (("video_only", COLORS["dynamic"], "o"), ("fusion", COLORS["static"], "s")):
            sub = sorted([r for r in rows if r["branch"] == branch], key=lambda r: f(r["interval_seconds"]))
            x = np.array([f(r["interval_seconds"]) for r in sub])
            values = np.array([f(r[metric]) for r in sub])
            ax.plot(x, values, marker=marker, linewidth=2, color=color, label=branch)
        ax.axhline(min(f(r[metric]) for r in rows), color="#9aa5b1", linestyle="--", linewidth=1, label="best formal point")
        ax.set_xlabel("Nominal interval (s); frames 32/16/8")
        ax.set_ylabel("Standardized MAE")
        ax.set_title(title)
        style(ax)
        ax.legend(frameon=False)
    fig.suptitle("True-pixel B2 observation strategy", fontsize=15, fontweight="bold")
    return save(fig, "fig03_true_b2_sampling"), "True-pixel B2 video-only and fusion performance across dense and sparse observation strategies.", "真正像素级 B2 中 video-only 与 fusion 在 0.125、0.25、0.5 秒采样间隔下的 arm14 和 full28 MAE。", "fig03_true_b2_sampling.csv"


def fig04():
    rows = [r for r in read_csv("B2_PROXY_RESULTS.csv")]
    formal = [r for r in read_csv("B2_RESULTS_INDEPENDENT.csv") if r["stage"] == "B2" and r["interval_seconds"] in {"0.25", "0.5"}]
    out = []
    for r in rows:
        out.append({"source": "proxy", "cell": r["cell"], "branch": r["branch"], "interval_seconds": r["interval_seconds"], "std_arm_mae": r["std_arm_mae"], "std_full28_mae": r["std_full28_mae"]})
    for r in formal:
        out.append({"source": "true_pixel", "cell": r["cell"], "branch": r["branch"], "interval_seconds": r["interval_seconds"], "std_arm_mae": r["std_arm_mae"], "std_full28_mae": r["std_full28_mae"]})
    write_csv(DATA / "fig04_proxy_true_b2.csv", out)
    labels = ["V-0.25", "FUS-0.25", "V-0.5", "FUS-0.5"]
    pairs = [("PROXY-B2-V-0250", "B2-V-0250"), ("PROXY-B2-FUS-0250", "B2-FUS-0250"), ("PROXY-B2-V-0500", "B2-V-0500"), ("PROXY-B2-FUS-0500", "B2-FUS-0500")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    for ax, metric, title in zip(axes, ("std_arm_mae", "std_full28_mae"), ("Arm14 MAE", "Full28 MAE")):
        x = np.arange(len(pairs))
        width = 0.36
        proxy_vals = [f(next(r[metric] for r in out if r["cell"] == a and r["source"] == "proxy")) for a, _ in pairs]
        true_vals = [f(next(r[metric] for r in out if r["cell"] == b and r["source"] == "true_pixel")) for _, b in pairs]
        ax.bar(x - width / 2, proxy_vals, width, color=COLORS["proxy"], label="proxy")
        ax.bar(x + width / 2, true_vals, width, color=COLORS["true"], label="true pixel")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Standardized MAE")
        ax.set_title(title)
        style(ax)
        ax.legend(frameon=False)
    fig.suptitle("B2 proxy versus true-pixel evidence", fontsize=15, fontweight="bold")
    return save(fig, "fig04_proxy_true_b2"), "Auxiliary contextual-token B2 proxy versus formal true-pixel B2 sparse conditions; proxy is not used for selection.", "旧 contextual-token proxy 与正式 true-pixel B2 sparse 条件的 arm14/full28 对照；proxy 不参与正式 selection。", "fig04_proxy_true_b2.csv"


def fig05():
    rows = [r for r in read_csv("B3_RESULTS_INDEPENDENT.csv") if r["stage"] == "B3"]
    order = ["B3-V-D-F", "B3-V-D-L", "B3-V-S-F", "B3-V-S-L", "B3-FUS-D-F", "B3-FUS-D-L", "B3-FUS-S-F", "B3-FUS-S-L"]
    rows.sort(key=lambda r: order.index(r["cell"]))
    write_csv(DATA / "fig05_b3_factorial.csv", rows)
    x = np.arange(len(rows))
    values = np.array([f(r["std_arm_mae"]) for r in rows])
    colors = [COLORS["lora"] if r["adaptation"] == "lora" else COLORS["frozen"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5.6), layout="constrained")
    bars = ax.bar(x, values, color=colors, edgecolor="white")
    for bar, row in zip(bars, rows):
        if row["video_input"] == "static":
            bar.set_hatch("//")
    ax.set_xticks(x, [r["cell"] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("Standardized arm14 MAE")
    ax.set_title("Corrected B3 2×2×2 factorial matrix")
    ax.legend(handles=[Patch(facecolor=COLORS["frozen"], label="frozen"), Patch(facecolor=COLORS["lora"], label="LoRA"), Patch(facecolor="white", edgecolor="#555", hatch="//", label="static")], frameon=False, ncol=3)
    style(ax)
    return save(fig, "fig05_b3_factorial"), "Corrected B3 eight-cell factorial results for branch, dynamic/static input and frozen/LoRA adaptation.", "修正后的 B3 八格 factorial；颜色区分 frozen/LoRA，斜线表示 static，纵轴为 standardized arm14 MAE。", "fig05_b3_factorial.csv"


def fig06():
    rows = read_csv("CORE_B_FACTORIAL_EFFECTS.csv")
    out = []
    for row in rows:
        for effect in ("G_D", "G_S", "interaction"):
            out.append({"branch": row["branch"], "effect": effect, "point": f(row[f"{effect}_point"]), "low": f(row[f"{effect}_ci95_lower"]), "high": f(row[f"{effect}_ci95_upper"])})
    write_csv(DATA / "fig06_b3_lora_interaction.csv", out)
    labels = ["G_D dynamic", "G_S static", "interaction"]
    effects = ["G_D", "G_S", "interaction"]
    x = np.arange(len(effects))
    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    for offset, branch, color in [(-0.18, "video_only", COLORS["dynamic"]), (0.18, "fusion", COLORS["static"])]:
        sub = {r["effect"]: r for r in out if r["branch"] == branch}
        vals = np.array([f(sub[e]["point"]) for e in effects])
        low = np.array([f(sub[e]["low"]) for e in effects])
        high = np.array([f(sub[e]["high"]) for e in effects])
        ax.errorbar(x + offset, vals, yerr=[vals - low, high - vals], fmt="o", color=color, capsize=4, linewidth=2, label=branch)
    ax.axhline(0, color="#334e68", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE improvement (standardized arm14)")
    ax.set_title("B3 LoRA gains and dynamic-specific interaction")
    ax.legend(frameon=False)
    style(ax)
    return save(fig, "fig06_b3_lora_interaction"), "Corrected B3 dynamic LoRA gain, static LoRA gain and dynamic-minus-static interaction with paired episode bootstrap intervals.", "B3 dynamic LoRA 增益、static LoRA 增益及 dynamic-specific interaction；正值表示误差下降。", "fig06_b3_lora_interaction.csv"


def fig07():
    rows = read_csv("TRAJECTORY_REPRESENTATIVE_EPISODES.csv")
    write_csv(DATA / "fig07_trajectory_representatives.csv", rows)
    episodes = sorted({r["episode_index"] for r in rows})
    fig, axes = plt.subplots(len(episodes), 1, figsize=(11, max(5, 3.0 * len(episodes))), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, episode in zip(axes, episodes):
        sub = [r for r in rows if r["episode_index"] == episode]
        times = sorted({int(r["time_step"]) for r in sub})
        target = [np.mean([f(r["target"]) for r in sub if int(r["time_step"]) == t]) for t in times]
        persistence = [np.mean([f(r["persistence"]) for r in sub if int(r["time_step"]) == t]) for t in times]
        model = [np.mean([f(r["model"]) for r in sub if int(r["time_step"]) == t]) for t in times]
        ax.plot(times, target, color=COLORS["target"], linewidth=2, label="target")
        ax.plot(times, persistence, color=COLORS["persistence"], linestyle="--", linewidth=1.7, label="persistence")
        ax.plot(times, model, color=COLORS["model"], linewidth=1.7, label="B3-FUS-D-L")
        ax.set_ylabel(f"ep {episode}\narm mean")
        style(ax)
    axes[-1].set_xlabel("Future time step")
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Prediction versus target and persistence on predeclared episodes", fontsize=15, fontweight="bold")
    return save(fig, "fig07_trajectory_representatives"), "Same-episode arm14 mean trajectories for target, persistence and the predeclared B3-FUS-D-L model at persistence-error quantiles 0.25, 0.50 and 0.75.", "预先按 persistence arm14 误差分位数选择的 episode 中，target、persistence 与 B3-FUS-D-L 的 arm14 平均轨迹对比。", "fig07_trajectory_representatives.csv"


def fig08():
    rows = read_csv("CORE_B_PER_JOINT_TIME_METRICS.csv")
    selected = [r for r in rows if ((r["stage"] == "B3" and r["cell"] == "B3-FUS-D-L") or (r["stage"] == "CONTROL" and r["cell"] == "PERSISTENCE")) and r["space"] == "standardized"]
    write_csv(DATA / "fig08_error_time_heatmap.csv", selected)
    def matrix(cell: str) -> np.ndarray:
        out = np.zeros((STEPS, 28))
        for row in selected:
            if row["cell"] == cell:
                out[int(row["time_step"]), int(row["joint"])] = f(row["mae"])
        return out
    model = matrix("B3-FUS-D-L")
    persistence_rows = [r for r in rows if r["stage"] == "CONTROL" and r["cell"] == "PERSISTENCE" and r["space"] == "standardized"]
    if persistence_rows:
        p = np.zeros((STEPS, 28))
        for row in persistence_rows:
            p[int(row["time_step"]), int(row["joint"])] = f(row["mae"])
    else:
        p = model.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    time = np.arange(STEPS)
    axes[0].plot(time, model[:, 14:].mean(axis=1), color=COLORS["model"], label="B3-FUS-D-L")
    axes[0].plot(time, p[:, 14:].mean(axis=1), color=COLORS["persistence"], linestyle="--", label="persistence")
    axes[0].set_xlabel("Future time step")
    axes[0].set_ylabel("Arm14 MAE")
    axes[0].set_title("Error over time")
    axes[0].legend(frameon=False)
    style(axes[0])
    im = axes[1].imshow(model[:, 14:].T, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_xlabel("Future time step")
    axes[1].set_ylabel("Arm joint")
    axes[1].set_title("B3-FUS-D-L joint×time error")
    fig.colorbar(im, ax=axes[1], label="Standardized MAE")
    fig.suptitle("B3 temporal and joint error diagnostics", fontsize=15, fontweight="bold")
    return save(fig, "fig08_error_time_heatmap"), "Temporal error curve and joint-by-time standardized MAE heatmap for the designated B3 fusion dynamic LoRA model versus persistence.", "B3 fusion dynamic LoRA 的随时间误差与 arm joint×time 热图，并与 persistence 的时间曲线对照。", "fig08_error_time_heatmap.csv"


def main() -> int:
    builders = [
        ("fig01_b1_baseline_comparison", fig01),
        ("fig02_b1_effects_forest", fig02),
        ("fig03_true_b2_sampling", fig03),
        ("fig04_proxy_true_b2", fig04),
        ("fig05_b3_factorial", fig05),
        ("fig06_b3_lora_interaction", fig06),
        ("fig07_trajectory_representatives", fig07),
        ("fig08_error_time_heatmap", fig08),
    ]
    manifest = []
    for figure_id, builder in builders:
        outputs, caption, alt_text, data_csv = builder()
        manifest.append({"figure_id": figure_id, "file_stem": figure_id, "caption_en": caption, "alt_text_zh": alt_text, "data_csv": str(DATA / data_csv), "plot_script": str(SCRIPT), "png": outputs["png"], "svg": outputs["svg"], "pdf": outputs["pdf"], "visual_qa_status": "pending_view_image"})
    write_csv(OUT / "FIGURE_MANIFEST.csv", manifest)
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": len(manifest), "directory": str(FIGURES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

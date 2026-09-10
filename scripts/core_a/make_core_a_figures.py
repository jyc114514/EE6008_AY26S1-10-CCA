#!/usr/bin/env python3
"""Create the six Core A Full-120 figures from locked analysis tables.

The script is CPU-only and only reads the analysis CSVs produced by
run_core_a_unified_analysis.py.  Every figure is saved as PNG, SVG and PDF;
the exact plotting inputs and bilingual figure metadata are saved alongside
the rendered files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results/core_a"
FIGURES = OUT / "figures"
FIGURE_DATA = OUT / "plot_data"
SCRIPT = Path("scripts/core_a/make_core_a_figures.py")

COLORS = {
    "frozen": "#486581",
    "lora": "#d64550",
    "dynamic": "#2f855a",
    "static": "#805ad5",
    "historical": "#9aa5b1",
    "current": "#d97706",
    "state": "#334e68",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("png", "svg", "pdf"):
        path = FIGURES / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"} if ext == "png" else {}
        fig.savefig(path, dpi=240, **kwargs)
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig01() -> tuple[dict[str, str], str, str, str]:
    rows = read_csv("BASELINE_PANORAMA.csv")
    keep_methods = {
        "chance",
        "majority",
        "state_only",
        "r3d18_kinetics400_v1",
        "static_first_frame_vjepa2_1_b",
        "vjepa2_1_b_full_existing_cache",
        "v jepa_full_dynamic_attentive_selected_best".replace(" ", ""),
        "vjepa_full_dynamic_attentive_selected_best",
        "vjepa_full_static_attentive_selected_best",
        "nonparametric_dynamic_cosine_knn1",
        "nonparametric_static_cosine_knn1",
        "D-P-F",
        "D-P-L",
        "D-A-F",
        "D-A-L",
        "S-P-F",
        "S-P-L",
        "S-A-F",
        "S-A-L",
        "STATE-CTRL",
    }
    selected = [r for r in rows if r["name"] in keep_methods]
    # Sort by Top-1 for a compact, readable panorama.
    selected.sort(key=lambda r: f(r["top1"]))
    write_csv(FIGURE_DATA / "fig01_baseline_panorama.csv", selected)
    labels = []
    for r in selected:
        label = r["name"]
        label = label.replace("vjepa2_1_b_full_existing_cache", "pooled dynamic")
        label = label.replace("static_first_frame_vjepa2_1_b", "pooled static")
        label = label.replace("vjepa_full_dynamic_attentive_selected_best", "attentive dynamic")
        label = label.replace("vjepa_full_static_attentive_selected_best", "attentive static")
        label = label.replace("r3d18_kinetics400_v1", "R3D-18")
        label = label.replace("nonparametric_dynamic_cosine_knn1", "dynamic 1-NN")
        label = label.replace("nonparametric_static_cosine_knn1", "static 1-NN")
        labels.append(label)
    y = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("top1", "macro_f1"), ("Top-1 accuracy", "Macro-F1")):
        values = np.array([f(r[metric]) * 100 for r in selected])
        colors = [COLORS["state"] if r["name"] in {"state_only", "STATE-CTRL"} else COLORS["current"] if r["source_group"].startswith("full120") else COLORS["historical"] for r in selected]
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
        style_axes(ax)
        for yi, value in zip(y, values):
            ax.text(min(value + 0.7, 99.5), yi, f"{value:.1f}", va="center", fontsize=8)
    axes[0].set_yticks(y, labels, fontsize=8)
    from matplotlib.patches import Patch

    axes[1].legend(
        handles=[
            Patch(facecolor=COLORS["historical"], label="historical reference"),
            Patch(facecolor=COLORS["current"], label="current Full-120"),
            Patch(facecolor=COLORS["state"], label="state reference"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    fig.suptitle("Core A baseline panorama: historical references and current Full-120 Stage1", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig01_baseline_panorama"), "Historical Core A reference methods versus current Full-120 Stage1 on Top-1 and macro-F1.", "历史基线与当前 Full-120 Stage1 的 Top-1 和 macro-F1 全景比较；橙色条为当前视频条件，灰色为历史参考，深蓝为 state reference。", "fig01_baseline_panorama.csv"


def fig02() -> tuple[dict[str, str], str, str, str]:
    rows = read_csv("BATCH_SIZE_EFFECTS.csv")
    rows = [r for r in rows if r["variant"] in {"dynamic", "static", "static_repeated_first_frame"}]
    normalized = []
    for r in rows:
        normalized.append({**r, "variant_plot": "static" if r["variant"] != "dynamic" else "dynamic"})
    write_csv(FIGURE_DATA / "fig02_batch4_32_48.csv", normalized)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    for ax, metric, title in zip(axes, ("top1_mean", "macro_f1_mean"), ("Top-1 accuracy", "Macro-F1")):
        for variant, color, marker in (("dynamic", COLORS["dynamic"], "o"), ("static", COLORS["static"], "s")):
            sub = sorted([r for r in normalized if r["variant_plot"] == variant], key=lambda r: f(r["batch_size"]))
            x = np.array([f(r["batch_size"]) for r in sub])
            value = np.array([f(r[metric]) * 100 for r in sub])
            sd = np.array([f(r["top1_sample_std" if metric == "top1_mean" else "macro_f1_sample_std"]) * 100 for r in sub])
            ax.errorbar(x, value, yerr=sd, color=color, marker=marker, linewidth=2, capsize=3, label=variant)
        ax.set_xticks([4, 32, 48])
        ax.set_xlabel("Nominal batch size")
        ax.set_ylabel("Percent")
        ax.set_title(title)
        style_axes(ax)
        ax.legend(frameon=False)
    fig.suptitle("Batch-size sensitivity: dynamic versus static", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig02_batch4_32_48_dynamic_static"), "Batch-4, Batch-32 and Batch-48 dynamic/static development aggregates with existing seed-level sample SD.", "动态与静态条件在 Batch4、32、48 的 Top-1 和 macro-F1 变化；误差线表示 seed-level sample SD。", "fig02_batch4_32_48.csv"


def fig03() -> tuple[dict[str, str], str, str, str]:
    rows = [r for r in read_csv("CORE_A_UNIFIED_RESULTS.csv") if r["input"] != "state_only"]
    order = ["D-P-F", "D-P-L", "S-P-F", "S-P-L", "D-A-F", "D-A-L", "S-A-F", "S-A-L"]
    rows.sort(key=lambda r: order.index(r["cell"]))
    write_csv(FIGURE_DATA / "fig03_full120_lora_factorial.csv", rows)
    labels = [r["cell"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for ax, metric, title in zip(axes, ("top1", "macro_f1"), ("Top-1 accuracy", "Macro-F1")):
        values = np.array([f(r[metric]) * 100 for r in rows])
        colors = [COLORS["lora"] if r["adaptation"] == "lora" else COLORS["frozen"] for r in rows]
        bars = ax.bar(x, values, width=0.72, color=colors, edgecolor="white")
        for bar, r in zip(bars, rows):
            if r["input"] == "static":
                bar.set_hatch("//")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(75, 100)
        ax.set_ylabel("Percent")
        ax.set_title(title)
        style_axes(ax)
        for xi, val in zip(x, values):
            ax.text(xi, val + 0.35, f"{val:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)
    from matplotlib.patches import Patch

    axes[1].legend(handles=[Patch(facecolor=COLORS["frozen"], label="frozen"), Patch(facecolor=COLORS["lora"], label="LoRA"), Patch(facecolor="white", edgecolor="#555", hatch="//", label="static")], frameon=False, ncol=3, loc="lower right")
    fig.suptitle("Full-120 Stage1 factorial screen: eight video cells", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig03_full120_lora_factorial"), "Eight-cell Full-120 Stage1 factorial results. Color encodes frozen versus LoRA and hatching encodes static input.", "Full-120 八个视频条件的 Top-1 与 macro-F1；颜色区分 frozen/LoRA，斜线表示 static 输入。", "fig03_full120_lora_factorial.csv"


def fig04() -> tuple[dict[str, str], str, str, str]:
    rows = read_csv("CORE_A_FACTORIAL_EFFECTS.csv")
    out = []
    for r in rows:
        for effect in ("dynamic_lora_gain", "static_lora_gain", "interaction"):
            out.append({
                "head": r["head"],
                "effect": effect,
                "top1_pp": f(r[f"{effect}_top1_pp"]) if effect != "interaction" else f(r["dynamic_minus_static_interaction_top1_pp"]),
                "top1_ci_low_pp": f(r[f"{effect}_top1_ci95_low_pp"]) if effect != "interaction" else f(r["dynamic_minus_static_interaction_top1_ci95_low_pp"]),
                "top1_ci_high_pp": f(r[f"{effect}_top1_ci95_high_pp"]) if effect != "interaction" else f(r["dynamic_minus_static_interaction_top1_ci95_high_pp"]),
                "macro_f1_pp": f(r[f"{effect}_macro_f1_pp"]) if effect != "interaction" else f(r["dynamic_minus_static_interaction_macro_f1_pp"]),
            })
    write_csv(FIGURE_DATA / "fig04_lora_gain_interaction.csv", out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), layout="constrained")
    effects = ["dynamic_lora_gain", "static_lora_gain", "interaction"]
    labels = ["dynamic LoRA gain", "static LoRA gain", "dynamic−static interaction"]
    colors = [COLORS["dynamic"], COLORS["static"], COLORS["current"]]
    x = np.arange(3)
    for ax, metric, title in zip(axes, ("top1_pp", "macro_f1_pp"), ("Top-1 effect (percentage points)", "Macro-F1 effect (percentage points)")):
        for j, head in enumerate(("pooled", "attentive")):
            vals = [next(r[metric] for r in out if r["head"] == head and r["effect"] == e) for e in effects]
            offset = -0.19 if j == 0 else 0.19
            bars = ax.bar(x + offset, vals, width=0.36, color=colors, alpha=0.9 if j == 0 else 0.58, edgecolor="white", label=head)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + (0.18 if val >= 0 else -0.34), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, rotation=90)
        ax.axhline(0, color="#334e68", linewidth=1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Percentage points")
        ax.set_title(title)
        all_values = [r[metric] for r in out]
        ax.set_ylim(min(0.0, min(all_values)) - 0.5, max(0.0, max(all_values)) + 0.5)
        style_axes(ax)
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("LoRA gain and dynamic-specific interaction", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig04_lora_gain_interaction"), "Matched LoRA gains in dynamic and static inputs, with the dynamic-specific interaction for pooled and attentive heads.", "pooled 与 attentive head 的 dynamic/static LoRA 增益及 dynamic-specific interaction；interaction 定义为动态增益减静态增益。", "fig04_lora_gain_interaction.csv"


def fig05() -> tuple[dict[str, str], str, str, str]:
    rare = [r for r in read_csv("CORE_A_RARE_CLASS_SUMMARY.csv") if r["support_bucket"] in {"support_1", "support_5_plus"} and r["input"] != "state_only"]
    shortcut = [r for r in read_csv("CORE_A_SHORTCUT_DIAGNOSTICS.csv") if r["comparison"] == "dynamic_vs_static" and r["scope"] == "per_class"]
    data_rows = []
    for r in rare:
        data_rows.append({"record_type": "rare_bucket", "group": r["cell"], "bucket": r["support_bucket"], "value": r["mean_recall"], "class_id": "", "head": r["head"], "adaptation": r["adaptation"]})
    for r in shortcut:
        data_rows.append({"record_type": "dynamic_static_class_recall_delta", "group": f"{r['cell']}−{r['reference']}", "bucket": "", "value": r["recall_delta"], "class_id": r["class_id"], "head": "", "adaptation": ""})
    write_csv(FIGURE_DATA / "fig05_rare_shortcut.csv", data_rows)
    cells = ["D-P-F", "D-P-L", "S-P-F", "S-P-L", "D-A-F", "D-A-L", "S-A-F", "S-A-L"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    x = np.arange(len(cells))
    for bucket, offset, color in (("support_1", -0.18, COLORS["current"]), ("support_5_plus", 0.18, COLORS["historical"])):
        vals = [f(next(r["mean_recall"] for r in rare if r["cell"] == c and r["support_bucket"] == bucket)) for c in cells]
        axes[0].bar(x + offset, np.array(vals) * 100, width=0.34, color=color, label="support=1" if bucket == "support_1" else "support≥5")
    axes[0].set_xticks(x, cells, rotation=35, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Mean class recall (%)")
    axes[0].set_title("Rare/support-stratified recall")
    axes[0].legend(frameon=False)
    style_axes(axes[0])
    comparisons = ["D-P-F−S-P-F", "D-P-L−S-P-L", "D-A-F−S-A-F", "D-A-L−S-A-L"]
    for i, comp in enumerate(comparisons):
        vals = np.array([f(r["value"]) for r in data_rows if r["record_type"] == "dynamic_static_class_recall_delta" and r["group"] == comp])
        # Deterministic jitter is only for legibility; it does not imply a CI.
        jitter = ((np.arange(len(vals)) * 37) % 101) / 1000 - 0.05
        axes[1].scatter(np.full(len(vals), i) + jitter, vals, s=13, alpha=0.58, color=COLORS["dynamic"] if "P" in comp else COLORS["current"], edgecolors="none")
        axes[1].plot([i - 0.16, i + 0.16], [np.mean(vals), np.mean(vals)], color="#102a43", linewidth=2)
    axes[1].axhline(0, color="#334e68", linewidth=1)
    axes[1].set_xticks(np.arange(len(comparisons)), comparisons, rotation=25, ha="right")
    axes[1].set_ylabel("Per-class recall delta")
    axes[1].set_title("Dynamic − static per-class recall")
    style_axes(axes[1])
    fig.suptitle("Rare-class and shortcut diagnostics", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig05_rare_shortcut"), "Support-stratified recall and per-class dynamic-minus-static recall diagnostics for shortcut and rare-class checks.", "左图比较 support=1 与 support≥5 类的平均 recall；右图展示四个 matched 条件中每一类的 dynamic−static recall，横线表示均值。", "fig05_rare_shortcut.csv"


def fig06() -> tuple[dict[str, str], str, str, str]:
    rows = read_csv("CALIBRATION_DATA.csv")
    cells = ["D-P-F", "D-P-L", "D-A-F", "D-A-L", "S-P-F", "S-P-L", "S-A-F", "S-A-L", "STATE-CTRL"]
    write_csv(FIGURE_DATA / "fig06_calibration.csv", rows)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for ax, cell in zip(axes.flat, cells):
        sub = [r for r in rows if r["cell"] == cell and f(r["count"]) > 0]
        x = np.array([f(r["mean_confidence"]) for r in sub])
        y = np.array([f(r["accuracy"]) for r in sub])
        ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1)
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.5, color=COLORS["state"] if cell == "STATE-CTRL" else COLORS["current"] if cell.endswith("L") else COLORS["historical"])
        ece = next(f(r["ece_contribution"]) for r in rows if r["cell"] == cell) if sub else float("nan")
        # Sum contributions is more meaningful than the first bin; calculate
        # from the loaded rows without adding a second source table.
        ece = sum(f(r["ece_contribution"]) for r in rows if r["cell"] == cell)
        ax.set_title(f"{cell}\nECE={ece:.3f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    fig.suptitle("Calibration reliability diagrams (15 equal-width bins)", fontsize=15, fontweight="bold")
    return save_figure(fig, "fig06_calibration"), "Reliability diagrams for all nine accepted current Stage1 cells using 15 equal-width confidence bins.", "九个已验收 Stage1 条件的 15-bin reliability diagram；虚线表示理想校准，标题显示 ECE。", "fig06_calibration.csv"


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    specs = [
        ("fig01_baseline_panorama", fig01),
        ("fig02_batch4_32_48_dynamic_static", fig02),
        ("fig03_full120_lora_factorial", fig03),
        ("fig04_lora_gain_interaction", fig04),
        ("fig05_rare_shortcut", fig05),
        ("fig06_calibration", fig06),
    ]
    manifest = []
    for figure_id, builder in specs:
        _outputs, caption, alt_text, data_file = builder()
        manifest.append({
            "figure_id": figure_id,
            "file_stem": figure_id,
            "caption_en": caption,
            "alt_text_zh": alt_text,
            "data_csv": str(Path("plot_data") / data_file),
            "plot_script": str(SCRIPT),
            "png": str(Path("figures") / f"{figure_id}.png"),
            "svg": str(Path("figures") / f"{figure_id}.svg"),
            "pdf": str(Path("figures") / f"{figure_id}.pdf"),
            "visual_qa_status": "pending_view_image",
        })
    with (OUT / "figure_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": len(manifest), "directory": str(FIGURES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

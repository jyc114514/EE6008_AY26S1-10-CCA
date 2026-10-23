#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results/PAIRED_GROUP_CI.csv"
rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
contrasts = ("NUMCTX_DROP_VIDEO", "NUMCTX_DROP_STATE", "INTERACTION", "MIX_GAIN")
labels = {
    "NUMCTX_DROP_VIDEO": "Video numeric-context drop",
    "NUMCTX_DROP_STATE": "State numeric-context drop",
    "INTERACTION": "Video × drop interaction",
    "MIX_GAIN": "MIX0 to MIXD",
}
metrics = (
    ("position_mae", "Position MAE"),
    ("safe_fd_l1", "Safe-FD L1"),
    ("seam_safe_fd_l1", "Seam Safe-FD L1"),
)
colors = {
    "NUMCTX_DROP_VIDEO": "#235789",
    "NUMCTX_DROP_STATE": "#4B7F52",
    "INTERACTION": "#D17B0F",
    "MIX_GAIN": "#7B4B94",
}
lookup = {(r["contrast"], r["metric"]): r for r in rows}
fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.1), sharey=True)
for ax, (metric, title) in zip(axes, metrics):
    for index, contrast in enumerate(contrasts):
        row = lookup[(contrast, metric)]
        point = float(row["point_estimate"])
        low = float(row["ci_2_5"])
        high = float(row["ci_97_5"])
        ax.errorbar(
            point, index,
            xerr=[[point - low], [high - point]],
            fmt="o", markersize=5, capsize=3, linewidth=1.5,
            color=colors[contrast],
        )
    ax.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_yticks(range(len(contrasts)), [labels[c] for c in contrasts], fontsize=8)
    ax.grid(axis="x", alpha=0.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Gain (reference − candidate)")
axes[0].invert_yaxis()
fig.suptitle("V9 full-context contrasts: corrected source-group bootstrap", fontsize=12, y=0.98)
fig.text(0.5, 0.01, "Positive gain favors candidate. 95% percentile CI; 5,000 draws; seed 2026101709.", ha="center", fontsize=8)
fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.92))
out = ROOT / "figures/CORRECTED_CI_PRIMARY_CONTRASTS.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
(ROOT / "figures/CAPTIONS.zh-CN.md").write_text(
    "# 修正后的 paired contrast 区间图\n\n"
    "此图用 V9 已封存的 per-window score rows 按 prompt 锁定 seed 2026101709 重算。"
    "四项完整数值上下文 contrast 的 point estimate 与 95% source-group bootstrap CI 均为 reference error − candidate error；"
    "正值有利于 candidate。训练模型、dev predictions 与 dev labels 均未重算或重读。\n",
    encoding="utf-8",
)
print(out)

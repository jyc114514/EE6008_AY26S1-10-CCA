#!/usr/bin/env python3
"""Render the single N2/N4 historical versus N5 three-scope CI figure."""
import os

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z'))
METRICS = ("position_standardized_mae", "safe_fd_normalized")
CONTRASTS = ("G_TS", "G_STATE", "V_Q0", "V_Q1", "I")
SCOPES = ("GROUP_ONLY_FIXED_SEEDS", "SEED_ONLY_FIXED_COHORT", "CROSSED_PAIRED_SEED_GROUP")
SCOPE_LABELS = {
    "GROUP_ONLY_FIXED_SEEDS": "group-only",
    "SEED_ONLY_FIXED_COHORT": "seed-only",
    "CROSSED_PAIRED_SEED_GROUP": "crossed",
}
COLORS = {
    "GROUP_ONLY_FIXED_SEEDS": "#0072B2",
    "SEED_ONLY_FIXED_COHORT": "#CC79A7",
    "CROSSED_PAIRED_SEED_GROUP": "#009E73",
}
OLD_TYPES = {
    "N2_GROUP_ONLY_FIXED_SEEDS": ("N2 old group-only", "#333333", "--", "o"),
    "N2_CROSSED_SEED_GROUP_SENSITIVITY": ("N2 old crossed sensitivity", "#777777", ":", "s"),
    "N4_REPORTED_CROSSED_SEED_GROUP": ("N4 reported crossed", "#D55E00", "--", "^"),
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interval(ax, lo, hi, y, color, marker, linestyle="-", linewidth=2.0, markersize=4.5, open_marker=False, label=None):
    ax.plot([lo, hi], [y, y], color=color, linestyle=linestyle, linewidth=linewidth, zorder=2, label=label)
    face = "white" if open_marker else color
    ax.plot([lo, hi], [y, y], linestyle="None", marker=marker, markersize=markersize,
            markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.0, zorder=3)


def main():
    new_rows = read_rows(ROOT / "results/CO_PRIMARY_SCOPE_TABLE30.csv")
    old_rows = read_rows(ROOT / "results/HISTORICAL_INTERVAL_REPRODUCTION.csv")
    new = {(r["metric"], r["contrast"], r["interval_scope"]): r for r in new_rows}
    old = {(r["metric"], r["contrast"], r["source_interval"]): r for r in old_rows}
    if len(new_rows) != 30 or len(old_rows) != 30:
        raise ValueError("expected 30 new scope rows and 30 historical reproduction rows")

    figure_data = []
    for metric in METRICS:
        for contrast in CONTRASTS:
            for scope in SCOPES:
                r = new[(metric, contrast, scope)]
                figure_data.append({
                    "metric": metric, "contrast": contrast, "interval_type": scope,
                    "lower": r["ci95_lower"], "upper": r["ci95_upper"],
                    "point_estimate": r["point_estimate_fixed_three_seed_mean"],
                    "source": "N5",
                    "quantile_method": r["quantile_method"],
                    "effective_draws": r["effective_draws"],
                })
            for historic in OLD_TYPES:
                r = old[(metric, contrast, historic)]
                figure_data.append({
                    "metric": metric, "contrast": contrast, "interval_type": historic,
                    "lower": r["reported_lower"], "upper": r["reported_upper"],
                    "point_estimate": r["fixed_seed_point_estimate"],
                    "source": historic.split("_")[0],
                    "quantile_method": r["quantile_method_reproduced"],
                    "effective_draws": r["effective_draws"],
                })
    data_path = ROOT / "figure_data/N5_CI_SCOPE_COMPARISON_DATA.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_data[0]))
        writer.writeheader()
        writer.writerows(figure_data)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 9.0), sharey=True, constrained_layout=False)
    offsets = {"N2_GROUP_ONLY_FIXED_SEEDS": 0.20, "N2_CROSSED_SEED_GROUP_SENSITIVITY": 0.20, "N4_REPORTED_CROSSED_SEED_GROUP": 0.38}
    for ax, metric in zip(axes, METRICS):
        for ci, contrast in enumerate(CONTRASTS):
            for si, scope in enumerate(SCOPES):
                y = ci * 3.0 + si
                row = new[(metric, contrast, scope)]
                interval(ax, float(row["ci95_lower"]), float(row["ci95_upper"]), y,
                         COLORS[scope], {SCOPES[0]: "o", SCOPES[1]: "s", SCOPES[2]: "^"}[scope],
                         linewidth=2.2, markersize=4.2)
                ax.plot(float(row["point_estimate_fixed_three_seed_mean"]), y, marker="D", color="#111111",
                        markersize=4.0, linestyle="None", zorder=5)
                if scope == "GROUP_ONLY_FIXED_SEEDS":
                    historic_names = ("N2_GROUP_ONLY_FIXED_SEEDS",)
                elif scope == "CROSSED_PAIRED_SEED_GROUP":
                    historic_names = ("N2_CROSSED_SEED_GROUP_SENSITIVITY", "N4_REPORTED_CROSSED_SEED_GROUP")
                else:
                    historic_names = ()
                for historic in historic_names:
                    old_row = old[(metric, contrast, historic)]
                    label, color, ls, marker = OLD_TYPES[historic]
                    interval(ax, float(old_row["reported_lower"]), float(old_row["reported_upper"]),
                             y + offsets[historic], color, marker, linestyle=ls, linewidth=1.4,
                             markersize=3.6, open_marker=True)

        ax.axvline(0.0, color="#555555", linewidth=1.0, alpha=0.8)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_yticks([ci * 3.0 + si for ci in range(len(CONTRASTS)) for si in range(3)])
        ax.set_yticklabels([f"{contrast} · {SCOPE_LABELS[scope]}" for contrast in CONTRASTS for scope in SCOPES], fontsize=8.4)
        ax.invert_yaxis()
        ax.set_xlabel("Error-difference estimate (positive G favors Q1; positive V favors TS11-D)")
        ax.set_title("Position standardized MAE" if metric == "position_standardized_mae" else "Safe-FD normalized")
        ax.margins(y=0.06)

    axes[0].set_ylabel("Contrast and interval scope")
    legend_handles = [
        Line2D([0], [0], color=COLORS["GROUP_ONLY_FIXED_SEEDS"], marker="o", lw=2.2, label="N5 group-only"),
        Line2D([0], [0], color=COLORS["SEED_ONLY_FIXED_COHORT"], marker="s", lw=2.2, label="N5 seed-only (27 exact triples)"),
        Line2D([0], [0], color=COLORS["CROSSED_PAIRED_SEED_GROUP"], marker="^", lw=2.2, label="N5 crossed"),
        Line2D([0], [0], color=OLD_TYPES["N2_GROUP_ONLY_FIXED_SEEDS"][1], marker="o", markerfacecolor="white", ls="--", lw=1.4, label="N2 reported group-only"),
        Line2D([0], [0], color=OLD_TYPES["N2_CROSSED_SEED_GROUP_SENSITIVITY"][1], marker="s", markerfacecolor="white", ls=":", lw=1.4, label="N2 crossed sensitivity"),
        Line2D([0], [0], color=OLD_TYPES["N4_REPORTED_CROSSED_SEED_GROUP"][1], marker="^", markerfacecolor="white", ls="--", lw=1.4, label="N4 reported crossed"),
        Line2D([0], [0], color="#111111", marker="D", lw=0, label="Fixed 3-seed point"),
    ]
    fig.suptitle("N2/N4 historical and N5 interval scopes · exposed development set · 95% intervals", fontsize=13, y=0.99)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.19, right=0.98, wspace=0.17)

    png = ROOT / "figures/N5_CI_SCOPE_COMPARISON.png"
    pdf = ROOT / "figures/N5_CI_SCOPE_COMPARISON.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "# N5_CI_SCOPE_COMPARISON\n\n"
        "Two panels compare the accepted N2/N4 historical intervals with the N5 retrospective three-scope intervals for all five G/V/I contrasts and both co-primary error metrics. Each solid interval uses the N5 inverse empirical CDF; seed-only enumerates all 27 ordered triples with probability 1/27. The open N2 marker shows its fixed-three-seed group-only interval; the open N2 crossed-sensitivity and N4 reported intervals use the historical stored group and seed draws and the original linear percentile rule. Diamonds mark the complete-cohort mean across the same three fixed seed estimates.\n\n"
        "Signs: positive G favors Q1; positive V favors TS11-D; I=G_TS−G_STATE=V_Q1−V_Q0. All intervals are conditional on fixed checkpoints and previously exposed development data. They are sensitivity intervals, not independent confirmation or p-values. N4's caption abbreviated the scope as source-group bootstrap; its implementation also resampled seed indices. Raw scales and physical units remain unverified. Underlying values and scope methods are in `figure_data/N5_CI_SCOPE_COMPARISON_DATA.csv`."
    )
    (ROOT / "captions/N5_CI_SCOPE_COMPARISON.md").write_text(caption, encoding="utf-8")
    print(f"PASS_N5_SINGLE_FIGURE png={png} pdf={pdf} data_rows={len(figure_data)}")


if __name__ == "__main__":
    main()

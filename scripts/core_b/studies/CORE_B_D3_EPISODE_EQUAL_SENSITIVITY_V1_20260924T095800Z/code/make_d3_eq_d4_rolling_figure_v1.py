"""Create provisional D4 rolling paired-effect figures from locked CSV output."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EQ_ROOT = Path(__file__).resolve().parents[1]
OUT = EQ_ROOT / "figures"
RESULTS = EQ_ROOT / "d3/analysis/D3_EQ_D4_ROLLING_PAIRED_EFFECTS.csv"
PROTOCOL = EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json"
METRICS = (
    ("delta_U", "Update utility ΔU", "ΔU (standardized MAE; + favors TS11-D)"),
    ("delta_L_new", "Updated loss ΔL_new", "ΔL_new (standardized MAE; − favors TS11-D)"),
)
BODIES = ("Hand", "Arm")
HORIZONS = (30, 60)
SEEDS = (20261001, 20261002, 20261003)
COLORS = {"Hand": "#0072B2", "Arm": "#D55E00"}
MARKERS = {20261001: "o", 20261002: "s", 20261003: "^"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    d3_lock = json.loads((EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_INPUT_ONLY" or protocol.get("final_test_access") != 0:
        raise AssertionError("D4 figures require the locked-input-only, final-test-denied protocol")
    if d3_lock.get("status") != "LOCKED" or d3_lock.get("final_test_access") != 0:
        raise AssertionError("D3-EQ parent protocol must remain locked and final-test denied")
    if d3_lock.get("d4_rolling_protocol_sha256") != sha256_file(PROTOCOL):
        raise AssertionError("D3-EQ parent lock does not bind the current D4 input-only protocol")
    rows = [
        row for row in csv.DictReader(RESULTS.open(newline="", encoding="utf-8"))
        if row["contrast"] == "TS11D_vs_STATE"
        and row["body"] in BODIES
        and row["metric"] in {metric for metric, _, _ in METRICS}
    ]
    expected = len(HORIZONS) * len(BODIES) * len(SEEDS) * len(METRICS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} paired D4 rows, found {len(rows)}")
    if any(int(row["pair_count"]) != 3600 or int(row["source_group_count"]) != 118 for row in rows):
        raise AssertionError("D4 figure rows do not share the locked pair/source-group support")

    interval_extent = max(
        max(abs(float(row["ci95_low"])), abs(float(row["ci95_high"])))
        for row in rows
    )
    x_limit = max(interval_extent * 1.08, 0.0005)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), layout="constrained", sharex=True)
    seed_labels = {seed: f"S{index + 1}" for index, seed in enumerate(SEEDS)}

    for metric_index, (metric, metric_title, x_label) in enumerate(METRICS):
        for horizon_index, horizon in enumerate(HORIZONS):
            ax = axes[metric_index, horizon_index]
            subset = [
                row for row in rows
                if row["metric"] == metric and int(row["horizon_steps"]) == horizon
            ]
            y_values = []
            y_labels = []
            y_position = 0
            for body in BODIES:
                for seed in SEEDS:
                    row = next(
                        item for item in subset
                        if item["body"] == body and int(item["seed"]) == seed
                    )
                    point = float(row["estimate"])
                    low = float(row["ci95_low"])
                    high = float(row["ci95_high"])
                    ax.errorbar(
                        point,
                        y_position,
                        xerr=[[point - low], [high - point]],
                        fmt=MARKERS[seed],
                        color=COLORS[body],
                        ecolor=COLORS[body],
                        capsize=2.5,
                        markersize=5.5,
                        elinewidth=1.2,
                        linewidth=0,
                    )
                    y_values.append(y_position)
                    y_labels.append(f"{body} · {seed_labels[seed]}")
                    y_position += 1
            ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            ax.set_xlim(-x_limit, x_limit)
            ax.set_yticks(y_values, y_labels)
            ax.invert_yaxis()
            ax.set_title(f"{metric_title}; H{horizon} (3,600 pairs; 118 groups)", fontsize=10)
            ax.set_xlabel(x_label)
            ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.75)
            ax.tick_params(axis="y", labelsize=10)

    fig.suptitle("D4 rolling paired contrast: TS11-D minus STATE-CTRL", fontsize=13)

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf"):
        path = OUT / f"FIG_D3_EQ_D4_ROLLING_TS11D_VS_STATE_V4.{extension}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing figure: {path.name}")
        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches=None,
            facecolor="white",
        )
        outputs.append({
            "path": path.relative_to(EQ_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "format": extension,
            "width_inches": 12.0,
            "height_inches": 8.4,
            "dpi": 300 if extension == "png" else "vector",
        })
    plt.close(fig)

    manifest = {
        "status": "GENERATED_PROVISIONAL_GENERAL_FIGURE_V4",
        "supersedes": "FIG_D3_EQ_D4_ROLLING_TS11D_VS_STATE_V3",
        "supersedes_reason": "vector PDF now embeds TrueType Type 42 fonts; layout and numeric rows/intervals are unchanged",
        "pdf_fonttype": 42,
        "ps_fonttype": 42,
        "audience": "research report; no journal/submission phase specified",
        "publisher_compliance": "not claimed",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_csv": RESULTS.relative_to(EQ_ROOT).as_posix(),
        "source_csv_sha256": sha256_file(RESULTS),
        "d4_protocol_sha256": sha256_file(PROTOCOL),
        "estimand": "TS11-D minus STATE-CTRL on identical adjacent-anchor pairs and common absolute future frames",
        "metrics": {
            "delta_U": "U=L_old-L_new; positive favors TS11-D",
            "delta_L_new": "updated standardized error; negative favors TS11-D",
        },
        "uncertainty": "95% paired source-group bootstrap intervals, separately by seed; not a pooled three-seed interval",
        "support": {"pairs_per_horizon": 3600, "source_groups": 118, "seeds": list(SEEDS)},
        "palette": {"Hand": COLORS["Hand"], "Arm": COLORS["Arm"]},
        "redundant_encoding": {"seed_markers": {str(seed): MARKERS[seed] for seed in SEEDS}, "zero_reference": "dashed vertical line"},
        "x_axis_limit_symmetric": x_limit,
        "alt_text": "Four-panel forest plot of TS11-D minus STATE-CTRL D4 rolling effects for Hand and Arm at 30- and 60-step horizons. Each seed is a separate marker; horizontal whiskers are 95% source-group bootstrap intervals. A vertical zero line marks no contrast. Positive delta-U favors fusion; negative delta-new-loss favors fusion.",
        "final_test_access": 0,
        "outputs": outputs,
    }
    manifest_path = OUT / "D3_EQ_D4_ROLLING_FIGURE_MANIFEST_V4.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing figure manifest: {manifest_path.name}")
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "outputs": len(outputs), "figure_manifest": manifest_path.name, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

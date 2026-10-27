"""Provisional, outcome-blind selected trajectory and paired-effect figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
FIG = EQ_ROOT / "figures"
COLORS = {
    "GT": "#000000",
    "D3-EQ TS11-D": "#0072B2",
    "D3-EQ STATE-CTRL": "#D55E00",
    "OLD-D3 TS11-D": "#009E73",
    "Persistence": "#666666",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path, anchor_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        ids = data["anchor_id"].astype("U64")
        matches = np.flatnonzero(ids == anchor_id)
        if len(matches) != 1:
            raise AssertionError(f"fixed trajectory anchor missing/duplicated in {path.name}")
        return data["predictions_native"][int(matches[0])].astype(np.float32, copy=True)


def plot_trajectories() -> list[dict[str, Any]]:
    selection = json.loads((EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json").read_text(encoding="utf-8"))
    if selection.get("prediction_outcomes_opened") is not False or len(selection.get("representatives", [])) != 3:
        raise AssertionError("trajectory IDs were not locked before outcomes")
    archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    dev = np.flatnonzero(archive["split_code"] == 1)
    dev_ids = archive["anchor_id"][dev].astype("U64")
    id_index = {str(value): int(index) for value, index in zip(dev_ids, dev)}
    output_records = []
    FIG.mkdir(parents=True, exist_ok=True)
    for horizon in (30, 60):
        fig, axes = plt.subplots(3, 2, figsize=(12.4, 8.4), layout="constrained", sharex=True)
        for row_index, selected in enumerate(selection["representatives"]):
            anchor_id = selected["anchor_id"]
            if anchor_id not in id_index:
                raise AssertionError(f"locked plot anchor is not a primary-dev row: {anchor_id}")
            source_index = id_index[anchor_id]
            anchor = archive["anchor_command_native"][source_index].astype(np.float32)
            target = archive["targets_native"][source_index, :horizon].astype(np.float32)
            persistence = np.broadcast_to(anchor[None, :], (horizon, 28))
            run_ids = {
                "D3-EQ TS11-D": f"D3EQ-TS11-D-H{horizon}-S20261001",
                "D3-EQ STATE-CTRL": f"D3EQ-STATE-CTRL-H{horizon}-S20261001",
                "OLD-D3 TS11-D": f"D3-TS11-D-H{horizon}-S20261001-FORMAL",
            }
            predictions = {
                name: load_prediction((EQ_ROOT / "d3/predictions" / f"{run_id}.npz") if name.startswith("D3-EQ") else (BENCH_ROOT / "d3/predictions" / f"{run_id}.npz"), anchor_id)
                for name, run_id in run_ids.items()
            }
            seconds = np.arange(1, horizon + 1, dtype=np.float64) / 30.0
            for column, (body, joint_index) in enumerate((("Hand", 0), ("Arm", 14))):
                ax = axes[row_index, column]
                ax.plot(seconds, target[:, joint_index], color=COLORS["GT"], linestyle="-", linewidth=2.1, label="Ground truth")
                ax.plot(seconds, predictions["D3-EQ TS11-D"][:, joint_index], color=COLORS["D3-EQ TS11-D"], linestyle="-", linewidth=1.7, label="D3-EQ TS11-D")
                ax.plot(seconds, predictions["D3-EQ STATE-CTRL"][:, joint_index], color=COLORS["D3-EQ STATE-CTRL"], linestyle="--", linewidth=1.6, label="D3-EQ STATE-CTRL")
                ax.plot(seconds, predictions["OLD-D3 TS11-D"][:, joint_index], color=COLORS["OLD-D3 TS11-D"], linestyle="-.", linewidth=1.5, label="OLD-D3 TS11-D")
                ax.plot(seconds, persistence[:, joint_index], color=COLORS["Persistence"], linestyle=":", linewidth=1.4, label="Persistence")
                ax.set_title(f"{body} joint {joint_index}; task {selected['original_task_index']}; ID {anchor_id[:12]}")
                ax.set_ylabel("Command (native dataset units)")
                ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
                if row_index == 2:
                    ax.set_xlabel("Time after anchor (s)")
                if row_index == 0 and column == 0:
                    ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
        fig.suptitle(f"D3-EQ fixed-horizon trajectories — H{horizon} ({horizon/30:.1f}s); selected before outcomes", fontsize=13)
        for extension in ("png", "pdf"):
            path = FIG / f"FIG_D3_EQ_TRAJECTORIES_H{horizon}_V2.{extension}"
            fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches=None, facecolor="white")
            output_records.append({"path": str(path.relative_to(EQ_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size, "format": extension, "horizon_steps": horizon})
        plt.close(fig)
    return output_records


def plot_primary_effects() -> list[dict[str, Any]]:
    rows = list(csv.DictReader((EQ_ROOT / "d3/analysis/D3_EQ_PRIMARY_POOLED.csv").open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row["metric"] == "standardized_mae" and row["body"] in ("Hand", "Arm")]
    if len(rows) != 16:
        raise AssertionError(f"primary effect figure requires 16 planned rows, found {len(rows)}")
    contrast_order = ["TS11D_vs_TS10D", "TS11D_vs_TS01D", "TS11D_vs_TS11S", "TS11D_vs_STATE"]
    marker = {name: shape for name, shape in zip(contrast_order, ("o", "s", "^", "D"))}
    color = {"Hand": "#0072B2", "Arm": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), layout="constrained", sharex=False)
    for row_index, body in enumerate(("Hand", "Arm")):
        for col_index, horizon in enumerate((30, 60)):
            ax = axes[row_index, col_index]
            subset = [row for row in rows if row["body"] == body and int(row["horizon_steps"]) == horizon]
            y = np.arange(len(contrast_order), dtype=np.float64)
            for index, contrast in enumerate(contrast_order):
                row = next(item for item in subset if item["contrast"] == contrast)
                point = float(row["estimate_three_seed_mean"])
                low = float(row["ci95_low_descriptive"])
                high = float(row["ci95_high_descriptive"])
                ax.errorbar(point, y[index], xerr=[[point-low], [high-point]], fmt=marker[contrast], color=color[body], ecolor=color[body], capsize=3, markersize=6, linewidth=1.4)
            ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
            ax.set_yticks(y, contrast_order)
            ax.invert_yaxis()
            ax.set_title(f"{body}; H{horizon}")
            ax.set_xlabel("D3-EQ candidate − control standardized MAE (lower is better)")
            ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    fig.suptitle("D3-EQ primary paired effects; 95% descriptive source-group bootstrap intervals", fontsize=13)
    output_records = []
    for extension in ("png", "pdf"):
        path = FIG / f"FIG_D3_EQ_PRIMARY_EFFECTS_V2.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches=None, facecolor="white")
        output_records.append({"path": str(path.relative_to(EQ_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size, "format": extension})
    plt.close(fig)
    return output_records


def main() -> int:
    protocol = json.loads((EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED":
        raise AssertionError("figures require a locked D3-EQ protocol")
    records = plot_trajectories() + plot_primary_effects()
    manifest = {
        "status": "GENERATED_PROVISIONAL_GENERAL_FIGURES_V2",
        "supersedes": "D3_EQ_FIGURE_MANIFEST.json",
        "supersedes_reason": "V2 uses embedded Type 42 TrueType fonts and adds explicit alt-text descriptions; trajectories and statistics are unchanged.",
        "audience": "research report; exact journal/article/submission phase not specified",
        "publisher_compliance": "not claimed; verify live guidance if a target journal is selected",
        "matplotlib_version": matplotlib.__version__,
        "trajectory_selection_sha256": sha256_file(EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json"),
        "source_group_bootstrap": "10,000 paired source-group resamples; three-seed pooled intervals are descriptive",
        "axes": "native command units; prediction time is seconds after anchor; no outcome-selected sample",
        "alt_text": {
            "primary_effects": "Forest plot of 16 planned TS11-D-minus-control Hand and Arm standardized-MAE contrasts at H30 and H60. Points are three-seed descriptive source-group-bootstrap estimates; whiskers are 95% descriptive intervals; the zero line marks no contrast. Negative values favor TS11-D.",
            "trajectories": "Six-panel native-command trajectories for each of three outcome-blind selected primary-development anchors at H30 or H60, comparing ground truth, D3-EQ TS11-D, D3-EQ state control, historical D3 TS11-D, and persistence."
        },
        "uncertainty": "95% paired source-group bootstrap intervals; seeds are shown as separate results and are not treated as independent resampling units",
        "analysis_population": "primary development panel only; final/held-out access 0",
        "pdf_fonttype": 42,
        "ps_fonttype": 42,
        "records": records,
        "final_test_access": 0,
    }
    atomic_path = FIG / "D3_EQ_FIGURE_MANIFEST_V2.json"
    atomic_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figures": len(records), "matplotlib_version": matplotlib.__version__, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from common import BENCH_ROOT, atomic_write_json, read_csv


def save(fig, stem):
    fig.savefig(BENCH_ROOT / "figures" / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(BENCH_ROOT / "figures" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = read_csv(BENCH_ROOT / "d3/analysis/D3_RESULTS.csv")
    configs = ["TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, horizon in zip(axes, (30, 60)):
        values=[]; errors=[]
        for config in configs:
            data=[float(row["estimate"]) for row in rows if row["config"]==config and int(row["horizon_steps"])==horizon and row["body"]=="Full28" and row["metric"]=="mae"]
            values.append(float(np.mean(data))); errors.append(float(np.std(data,ddof=1)) if len(data)>1 else 0.0)
        ax.bar(configs, values, yerr=errors, capsize=3, color=["#94a3b8","#60a5fa","#fbbf24","#2563eb","#7c3aed","#64748b"])
        ax.set_title(f"D3 Full28 MAE, H={horizon}")
        ax.set_ylabel("episode-equal standardized MAE")
        ax.tick_params(axis='x', rotation=35)
    save(fig, "FIG_D3_FACTOR_MATRIX_FULL28")
    contrasts = read_csv(BENCH_ROOT / "d3/analysis/D3_CONTRASTS.csv")
    rows2=[row for row in contrasts if row["contrast"]=="factorial_interaction_TS11_TS10_TS01_TS00" and row["body"]=="Full28"]
    labels=[f"H{row['horizon_steps']} S{int(row['seed'])%100}" for row in rows2]
    vals=[float(row['estimate']) for row in rows2]
    fig, ax=plt.subplots(figsize=(8,3.8)); ax.axhline(0,color='black',lw=.8); ax.bar(labels,vals,color='#0f766e'); ax.set_ylabel("(L11-L10)-(L01-L00)"); ax.set_title("D3 factorial interaction; negative favors joint time×space"); ax.tick_params(axis='x',rotation=35); save(fig,"FIG_D3_FACTORIAL_INTERACTION")
    manifest_path=BENCH_ROOT / "figures/FIGURE_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"status":"PASS","figures":[]}
    manifest.setdefault("figures",[]).extend([{"figure_id":"FIG_D3_FACTOR_MATRIX_FULL28","status":"PASS","caption":"D3 Full28 factor matrix by horizon and seed; new trained spatial-grid matrix.","population":"primary development","uncertainty":"seed SD"},{"figure_id":"FIG_D3_FACTORIAL_INTERACTION","status":"PASS","caption":"D3 factorial interaction with 10,000 cluster bootstrap underlying rows.","population":"primary development","uncertainty":"cluster bootstrap CI in D3_CONTRASTS.csv"}])
    atomic_write_json(manifest_path,manifest)
    print(json.dumps({"status":"PASS","figures_added":2}))
    return 0


if __name__ == "__main__": raise SystemExit(main())

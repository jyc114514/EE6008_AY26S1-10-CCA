"""Merge optional D3 per-anchor dense-panel scores and contrasts into the handoff."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "02_SCIENTIFIC_RESULTS.csv"
CLAIMS = ROOT / "02_CLAIM_LEDGER.csv"
REPORT = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
LONG_FIELDS = [
    "study", "analysis", "model", "candidate", "control", "seed", "horizon_steps",
    "body", "metric", "stratum", "event_bin", "estimate", "ci95_low", "ci95_high",
    "seed_sd_n3", "support_episodes", "support_source_groups", "support_windows_or_pairs",
    "opportunities", "hits", "misses", "false_alarms", "exposure_hours",
    "timing_abs_error_seconds", "bootstrap_replicates", "confidence_interval_type",
    "status", "final_test_access", "source_file", "notes",
]


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def long_row(**values):
    row = {key: "" for key in LONG_FIELDS}
    row.update(values)
    row["final_test_access"] = 0
    return row


def main():
    summary = json.loads((ROOT / "d3_extension/results/D3_DENSE_PANEL_SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE_POSTHOC_D3_DENSE_PANEL_ANALYSIS" or int(summary.get("accepted_cells", -1)) != 36:
        raise SystemExit("D3 dense panel analysis is not 36/36 complete")
    cell_rows = csv_rows(ROOT / "d3_extension/results/D3_DENSE_PANEL_RESULTS.csv")
    contrast_rows = csv_rows(ROOT / "d3_extension/results/D3_DENSE_PANEL_CONTRASTS.csv")
    if len(cell_rows) != 216 or len(contrast_rows) != 252:
        raise SystemExit("D3 dense panel output cardinality mismatch")
    existing = csv_rows(RESULTS)
    if any(row["study"] == "D3_dense_panel" for row in existing):
        raise SystemExit("D3 dense-panel results are already merged; refusing duplicate append")
    new = []
    for row in cell_rows:
        new.append(long_row(
            study="D3_dense_panel", analysis="per_seed_cell_score", model=row["config"],
            seed=row["seed"], horizon_steps=row["horizon_steps"], body=row["body"],
            metric=row["metric"], estimate=row["estimate"], ci95_low=row["ci95_low"], ci95_high=row["ci95_high"],
            support_episodes=row["episodes"], support_source_groups=row["source_groups"],
            support_windows_or_pairs=row["queries"], bootstrap_replicates=row["bootstrap_replicates"],
            confidence_interval_type="exact video-hash source-group bootstrap per fixed D3 training seed",
            status="posthoc_task_balanced_dense_development_extension",
            source_file="d3_extension/results/D3_DENSE_PANEL_RESULTS.csv",
            notes="Same six D3 configurations/three seeds/H30-H60; actual D3 training-weighting deviation remains.",
        ))
    for row in contrast_rows:
        new.append(long_row(
            study="D3_dense_panel", analysis=row["contrast"], model="", candidate=row["candidate"],
            control=row["control"], seed=row["seed"], horizon_steps=row["horizon_steps"],
            body=row["body"], metric=row["metric"], stratum=row["interpretation"],
            estimate=row["estimate"], ci95_low=row["ci95_low"], ci95_high=row["ci95_high"],
            support_episodes=row["episodes"], support_source_groups=row["source_groups"],
            support_windows_or_pairs=row["queries"], bootstrap_replicates=row["bootstrap_replicates"],
            confidence_interval_type="paired source-group bootstrap, fixed D3 training seed",
            status="posthoc_task_balanced_dense_development_extension",
            source_file="d3_extension/results/D3_DENSE_PANEL_CONTRASTS.csv",
            notes=row["estimand"],
        ))
    with RESULTS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(new)

    existing_claims = csv_rows(CLAIMS)
    if not any(row["claim_id"] == "C11" for row in existing_claims):
        with CLAIMS.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow([
                "C11", "D3 dense-panel cell and factorial effects remain seed-sensitive and separate from the primary D3 population.",
                "The same six D3 configurations were scored on 3,718 task-balanced dense queries across 118 source groups; spatial/temporal contrasts and CIs are in the dense-panel result tables.",
                "This is a matched inference extension, not a new training sweep or a natural-prevalence estimate.",
                "The D3 training objective implementation differs from the locked sample-wise equal-episode weighting; only three training seeds are available.",
                "post-hoc descriptive development extension",
                "36/36 bundles; 118 source groups; 10,000 source-group bootstrap replicates per seed",
                "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json; d3_extension/results/D3_DENSE_PANEL_CONTRASTS.csv",
                "0",
            ])

    contrasts = [row for row in contrast_rows if row["metric"] == "mae" and row["body"] in ("Arm", "Hand")
                 and row["contrast"] in ("TS11_vs_TS10", "TS11_vs_TS01")]
    lines = [
        "### D3 per-anchor dense-panel complement\n",
        "The task-balanced dense panel provides a separate 118-source-group check of D3's conditional spatial/temporal effects. "
        "It uses the same accepted D3 checkpoints, no retraining, 3,718 dense queries, and 10,000 source-group bootstrap draws per seed. "
        "Negative candidate-minus-control standardized MAE favors TS11-D; these remain post-hoc because the original D3 outcomes were exposed and the training objective deviated from its lock.\n",
        "| Contrast | H | Body | seed mean ± SD; each seed estimate [95% CI] |\n|---|---:|---|---|\n",
    ]
    for contrast in ("TS11_vs_TS10", "TS11_vs_TS01"):
        for h in ("30", "60"):
            for body in ("Arm", "Hand"):
                rr = [r for r in contrasts if r["contrast"] == contrast and r["horizon_steps"] == h and r["body"] == body]
                rr.sort(key=lambda row: row["seed"])
                points = [float(r["estimate"]) for r in rr]
                detail = "; ".join(
                    f"{r['seed'][-2:]} {float(r['estimate']):.6f} [{float(r['ci95_low']):.6f},{float(r['ci95_high']):.6f}]"
                    for r in rr
                )
                lines.append(
                    f"| {contrast} | {h} | {body} | {statistics.mean(points):.6f} ± {statistics.stdev(points):.6f}; {detail} |\n"
                )
    lines.append(
        "\nThe dense-panel effects remain separate from primary-dev estimates and are not pooled with V3 or V3_STATE. "
        "Full Arm/Hand/Full28 MAE/native-FD cell scores, four matched contrasts, and factorial interactions are in the unified CSV and the two dense-panel source tables.\n\n"
    )
    text = REPORT.read_text(encoding="utf-8")
    if "### D3 per-anchor dense-panel complement" in text:
        raise SystemExit("dense-panel report subsection already exists")
    marker = "V3_STATE dense inference accepted"
    if marker not in text:
        raise SystemExit("optional rolling section insertion anchor is missing")
    REPORT.write_text(text.replace(marker, "".join(lines) + marker), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "appended_long_rows": len(new),
        "D3_dense_panel_results": len(cell_rows),
        "D3_dense_panel_contrasts": len(contrast_rows),
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

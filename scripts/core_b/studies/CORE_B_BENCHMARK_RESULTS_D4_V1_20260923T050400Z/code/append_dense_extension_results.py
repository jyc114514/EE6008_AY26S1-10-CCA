"""Add the optional D3 and V3_STATE dense rolling extensions to the handoff tables."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "02_SCIENTIFIC_RESULTS.csv"
CLAIMS = ROOT / "02_CLAIM_LEDGER.csv"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return ""
    return value


def long_row(**values):
    row = {key: "" for key in LONG_FIELDS}
    row.update(values)
    row["final_test_access"] = 0
    return row


LONG_FIELDS = [
    "study", "analysis", "model", "candidate", "control", "seed", "horizon_steps",
    "body", "metric", "stratum", "event_bin", "estimate", "ci95_low", "ci95_high",
    "seed_sd_n3", "support_episodes", "support_source_groups", "support_windows_or_pairs",
    "opportunities", "hits", "misses", "false_alarms", "exposure_hours",
    "timing_abs_error_seconds", "bootstrap_replicates", "confidence_interval_type",
    "status", "final_test_access", "source_file", "notes",
]


def append_long_rows():
    d3d4_summary = read_json(ROOT / "d4/d3_extension/D3_D4_SUMMARY.json")
    state_summary = read_json(ROOT / "d4/v3_state_dense/D4_SUMMARY.json")
    new_rows = []
    for r in read_csv(ROOT / "d4/d3_extension/D3_D4_PRIMARY_RESULTS_BY_SEED.csv"):
        body, metric = r["body"], r["metric"]
        new_rows.append(long_row(
            study="D3_D4_dense", analysis="per_seed_model_pair_metric", model=r["model"],
            seed=r["seed"], horizon_steps=r["horizon_steps"], body=body, metric=metric,
            estimate=clean(r["estimate"]), ci95_low=clean(r["ci95_low"]), ci95_high=clean(r["ci95_high"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            support_windows_or_pairs=r["pairs"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="source-group bootstrap over matched dense rolling pairs",
            status="posthoc_dense_development_extension", source_file="d4/d3_extension/D3_D4_PRIMARY_RESULTS_BY_SEED.csv",
            notes="Same absolute target timestamps; the D3 training objective deviation remains.",
        ))
    for r in read_csv(ROOT / "d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv"):
        new_rows.append(long_row(
            study="D3_D4_dense", analysis=r["contrast"], candidate=r["candidate"], control=r["control"],
            seed=r["seed"], horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            stratum=r["estimand"], estimate=clean(r["estimate"]), ci95_low=clean(r["ci95_low"]),
            ci95_high=clean(r["ci95_high"]), support_episodes=r["episodes"],
            support_source_groups=r["source_groups"], support_windows_or_pairs=r["pairs"],
            bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="paired source-group bootstrap; identical pair IDs and target frames",
            status="posthoc_dense_development_extension", source_file="d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv",
            notes=r["interpretation"],
        ))
    for r in read_csv(ROOT / "d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv"):
        new_rows.append(long_row(
            study="D3_D4_dense", analysis="cross_seed_descriptive", candidate=r["contrast"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            estimate=clean(r["estimate_equal_seed_group_mean"]), ci95_low=clean(r["ci95_descriptive_low"]),
            ci95_high=clean(r["ci95_descriptive_high"]), seed_sd_n3=clean(r["seed_mean_sd_n3"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            support_windows_or_pairs=r["pairs_per_seed"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type=r["interval_type"], status="posthoc_dense_descriptive",
            source_file="d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv",
            notes=f"seed means={r['seed_20261001']};{r['seed_20261002']};{r['seed_20261003']}",
        ))

    for r in read_csv(ROOT / "d4/v3_state_dense/D4_PAIRED_EFFECTS.csv"):
        new_rows.append(long_row(
            study="V3_STATE_D4_dense", analysis=r["contrast"], model=r["state_level"],
            candidate=r["candidate"], control=r["control"], seed=r["seed"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            estimate=clean(r["estimate"]), ci95_low=clean(r["ci95_low"]), ci95_high=clean(r["ci95_high"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            support_windows_or_pairs=r["pairs"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="source-group bootstrap; matched within-state video control",
            status="posthoc_dense_diagnostic", source_file="d4/v3_state_dense/D4_PAIRED_EFFECTS.csv",
            notes=f"separate V3_STATE mask study; state_level={r['state_level']}; no pooling with V3/D3",
        ))
    for r in read_csv(ROOT / "d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv"):
        new_rows.append(long_row(
            study="V3_STATE_D4_dense", analysis="cross_seed_descriptive", model=r["state_level"],
            candidate=r["contrast"], horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            estimate=clean(r["estimate_equal_seed_group_mean"]), ci95_low=clean(r["ci95_descriptive_low"]),
            ci95_high=clean(r["ci95_descriptive_high"]), seed_sd_n3=clean(r["seed_mean_sd_n3"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            support_windows_or_pairs=r["pairs_per_seed"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type=r["interval_type"], status="posthoc_dense_descriptive",
            source_file="d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv",
            notes=f"seed means={r['seed_20260953']};{r['seed_20260954']};{r['seed_20260955']}; separate state-mask study",
        ))

    existing = read_csv(LONG)
    if any(r["study"] in {"D3_D4_dense", "V3_STATE_D4_dense"} for r in existing):
        raise ValueError("optional dense extension rows already exist; refusing duplicate append")
    with LONG.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows({k: clean(v) for k, v in row.items()} for row in new_rows)
    return d3d4_summary, state_summary, len(new_rows)


def append_claims(d3d4_summary, state_summary):
    existing = read_csv(CLAIMS)
    ids = {row["claim_id"] for row in existing}
    additions = []
    if "C9" not in ids:
        additions.append([
            "C9", "Optional D3 dense rolling-update evidence remains a separate descriptive extension.",
            f"Accepted D3 dense cells={d3d4_summary['accepted_d3_dense_bundles']}/36; pairs={d3d4_summary['pairs_per_horizon']}/horizon; D3 D4 summaries compare spatial, temporal, and dynamic/static controls on the identical 118 source groups.",
            "Interpret only within the D3 actual-run factorial; do not call it a V3 spatial causal effect.",
            "D3 trainer objective is nonconformant, dense panel is task-balanced, and prior per-anchor development outcomes were exposed.",
            "posthoc development extension; no training; no final access",
            "3,718 dense queries; 118 source groups; exact D4 common target frames",
            "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json; d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv",
            "0",
        ])
    if "C10" not in ids:
        additions.append([
            "C10", "V3_STATE dense rolling effects remain conditional on the study's state mask.",
            f"Accepted V3_STATE dense bundles={state_summary['accepted_bundles']}/54; D4 contrasts compare D against N and S only within each FULL/OBS/ANCHOR mask.",
            "These are mechanism diagnostics on the same task-balanced panel, not independent confirmation or a cross-study leaderboard.",
            "Mask differences include optimization/fusion effects; small Hand/event supports and post-hoc development exposure remain.",
            "posthoc descriptive extension; no p-values or Holm adjustment",
            "3,718 queries; 118 episodes/source groups; 3 seeds",
            "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json; d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv",
            "0",
        ])
    if additions:
        with CLAIMS.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(additions)


def update_report(d3d4_summary, state_summary):
    d3 = read_csv(ROOT / "d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv")
    state = read_csv(ROOT / "d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv")
    lines = [
        "## Optional dense rolling extensions (separate studies)\n",
        f"D3 dense inference accepted {d3d4_summary['accepted_d3_dense_bundles']}/36 on the fixed 3,718-query panel; "
        f"D3 rolling analysis uses {d3d4_summary['pairs_per_horizon']} matched pairs/horizon across {d3d4_summary['source_groups']} source groups. "
        "D3 remains an actual-run exploratory analysis because its training objective deviated from the lock.\n",
        "| Study/contrast | H | Body | ΔU cross-seed mean [descriptive 95% CI] | seed SD |\n|---|---:|---|---|---:|\n",
    ]
    for r in d3:
        if r["metric"] == "delta_U" and r["body"] in ("Arm", "Hand"):
            lines.append(
                f"| D3 {r['contrast']} | {r['horizon_steps']} | {r['body']} | "
                f"{float(r['estimate_equal_seed_group_mean']):.6f} "
                f"[{float(r['ci95_descriptive_low']):.6f},{float(r['ci95_descriptive_high']):.6f}] | "
                f"{float(r['seed_mean_sd_n3']):.6f} |\n"
            )
    lines.append(
        f"\nV3_STATE dense inference accepted {state_summary['accepted_bundles']}/54 on the same fixed panel. "
        "Only within-mask D-vs-N and D-vs-S comparisons are included; these are post-hoc descriptive diagnostics, "
        "not a pooled extension of V3/D3 and not a new Holm family. Full per-seed ΔU/ΔL_new/ΔFD_new, support, and intervals "
        "are in d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv. "
        "D3 per-seed effects are in d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv; cross-seed intervals are descriptive only.\n"
    )
    path = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
    text = path.read_text(encoding="utf-8")
    if "## Optional dense rolling extensions (separate studies)" in text:
        raise ValueError("dense extension report section already exists; refusing duplicate insertion")
    marker = "## 解释边界"
    if marker not in text:
        raise ValueError("scientific report is missing its interpretation-boundary section")
    path.write_text(text.replace(marker, "".join(lines) + "\n" + marker), encoding="utf-8")

    d5 = ROOT / "d5/D5_GO_NO_GO.md"
    d5_text = d5.read_text(encoding="utf-8")
    if "## Optional dense rolling extension" not in d5_text:
        d5_text += (
            "\n\n## Optional dense rolling extension\n\n"
            f"D3 dense rolling analysis accepted {d3d4_summary['accepted_d3_dense_bundles']}/36 bundles; "
            "V3_STATE dense rolling analysis accepted 54/54. Both use the locked 118-source-group dense panel and "
            "same absolute target frames. These post-hoc diagnostics are kept within their own model/input contracts; "
            "they do not repair the D3 trainer's objective deviation or establish a probabilistic forecast need. "
            "The D5 decision remains redesign; no training was performed.\n"
        )
        d5.write_text(d5_text, encoding="utf-8")


def update_sources_and_status(d3d4_summary, state_summary):
    paths = [
        ROOT / "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json",
        ROOT / "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json",
        ROOT / "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json",
        ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        ROOT / "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        ROOT / "d4/d3_extension/D3_D4_SUMMARY.json",
        ROOT / "d4/v3_state_dense/D4_SUMMARY.json",
    ]
    hashes_path = ROOT / "SOURCE_HASHES.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for path in paths:
        if path.is_file():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path)] = h
    hashes_path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    d3d4_summary, state_summary, appended = append_long_rows()
    append_claims(d3d4_summary, state_summary)
    update_report(d3d4_summary, state_summary)
    update_sources_and_status(d3d4_summary, state_summary)
    record = {
        "status": "PASS",
        "d3_d4_extension": d3d4_summary["status"],
        "v3_state_d4_extension": state_summary["status"],
        "unified_result_rows_appended": appended,
        "final_test_access": 0,
        "training": "none",
    }
    (ROOT / "qa/DENSE_EXTENSION_REPORT_ACCEPTANCE.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

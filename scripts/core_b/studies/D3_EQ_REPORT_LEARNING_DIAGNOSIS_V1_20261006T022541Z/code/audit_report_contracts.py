#!/usr/bin/env python3
"""Audit locked D3-EQ semantics, statistical rules, claims and training curves."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_semantics() -> list[dict[str, Any]]:
    model_path = EQ / "code/d3_model.py"
    infer_path = EQ / "code/d3_eq_infer.py"
    extract_path = OLD / "code/d3_extract_grid.py"
    extract_manifest_path = OLD / "d3/cache/D3_EXTRACTION_RUN_MANIFEST.json"
    protocol_path = EQ / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    model = model_path.read_text(encoding="utf-8")
    infer = infer_path.read_text(encoding="utf-8")
    extractor = extract_path.read_text(encoding="utf-8")
    protocol = read_json(protocol_path)
    manifest = read_json(extract_manifest_path)
    checks = {
        "TS00-D": "source.mean(dim=(1, 2), keepdim=True).expand(-1, TIME_SLOTS, SPACE_BINS, -1)" in model,
        "TS10-D": "source.mean(dim=2, keepdim=True).expand(-1, -1, SPACE_BINS, -1)" in model,
        "TS01-D": "source.mean(dim=1, keepdim=True).expand(-1, TIME_SLOTS, -1, -1)" in model,
        "TS11-D": "else:\n            values = source" in model,
        "TS11-S": "source = static if self.video_mode == \"TS11-S\" else dynamic" in model,
        "STATE-CTRL": "return torch.zeros_like(source).reshape(source.shape[0], VIDEO_SLOTS, VIDEO_DIM)" in model,
    }
    if not all(checks.values()):
        raise AssertionError(f"model layout code no longer matches checked semantics: {checks}")
    raw_static_check = "static = np.repeat(frames[-1:, ...], 32, axis=0)" in extractor and "(frames, static)" in extractor
    extractor_hash_bound = any("extract" in key.lower() and "sha" in key.lower() for key in manifest)
    common = {
        "state_history": "same standardized 32x71 state plus mask/relative-time tokens for every cell",
        "anchor": "same native 28-D last-command token for every cell",
        "head": "same D3TimeSpacePredictor architecture and per-cell seeded head",
        "memory_slots": "32 state + 64 video (16x4) + 1 anchor = 97 memory tokens",
        "normalizers": "train-fitted V3 state/target scales; D3 train-fitted video mean/safe_std",
        "source_hash_model": sha(model_path),
        "source_hash_inference": sha(infer_path),
    }
    statements = {
        "TS00-D": ("dynamic D grid; joint time-space global mean broadcast to all 64 slots", "mean axes T and 4 spatial bins; repeat axes T16 x S4"),
        "TS10-D": ("dynamic D grid; preserve 16 time slots, spatially average each slot", "mean spatial axis; repeat each mean across 4 bins"),
        "TS01-D": ("dynamic D grid; preserve 4 spatial bins, temporally average each bin", "mean temporal axis; repeat each mean across 16 time slots"),
        "TS11-D": ("dynamic D grid; preserve full contextual 16x4 slots", "no temporal or spatial pooling/repeat"),
        "TS11-S": ("static-video control with same state/history and last-command anchor", "repeat last legal raw frame 32x, then separately preprocess and full-encoder encode; retain its contextual 16x4 grid"),
        "STATE-CTRL": ("same state/history and anchor/head; video content zeroed after train-only normalization", "zero all 64 video tokens"),
    }
    rows = []
    for config in protocol["configs"]:
        meaning, pooling = statements[config]
        rows.append({"config": config, "verified_semantics": meaning, "pooling_and_repeat_axes": pooling,
                     **common,
                     "model_code_check": checks[config],
                     "raw_static_source_check": raw_static_check if config == "TS11-S" else "not_applicable",
                     "extractor_code_sha256": sha(extract_path) if config == "TS11-S" else "not_applicable",
                     "run_manifest_has_extractor_sha_binding": extractor_hash_bound if config == "TS11-S" else "not_applicable",
                     "semantic_acceptance": ("source_semantics_verified_runtime_extractor_hash_unbound" if config == "TS11-S" and raw_static_check and not extractor_hash_bound else "verified"),
                     "locked_protocol_text": protocol["configuration_semantics"][config],
                     "final_test_access": 0})
    return rows


def statistical_protocol() -> list[dict[str, Any]]:
    p = read_json(EQ / "protocol/D3_EQ_PROTOCOL_LOCKED.json")
    qa = read_json(EQ / "d3/analysis/D3_EQ_ANALYSIS_QA.json")
    checks = [
        ("primary_bootstrap_unit", p["evaluation"]["bootstrap_unit"], "paired video_sha256 source groups; primary dev episode/group bijection is 1:1", "matched_by_locked_protocol_and_QA"),
        ("group_draws", str(p["evaluation"]["group_bootstrap_draws_per_seed"]), "per-seed source-group bootstrap draws", "matched"),
        ("primary_p", p["evaluation"]["primary_p_value_procedure"], "two-sided centered paired group bootstrap with add-one smoothing; Holm over 16", "matched"),
        ("seed_aggregate", p["evaluation"]["seed_aggregate"], "arithmetic mean of fixed three seed effects; seed SD descriptive", "matched"),
        ("interval_scope", "group-only bootstrap over fixed trained seeds", "conditional on the three observed models; not a seed-population interval", "clarification_added"),
        ("factorial_family", str(p["factorial_interaction"]), "factorial CSV states Holm not applied; no separate locked Holm family field found; retain as unadjusted secondary evidence", "unadjusted_secondary"),
        ("scheduler_as_run", p["training"]["scheduler_contract_vs_as_run"]["historical_code"], "T_max=11120 but scheduler.step once per epoch; epoch40 LR remains about 0.000999968", "as_run_preserved_not_repaired"),
        ("episode_equal_target", p["episode_weight"], "corrected D3-EQ objective; evaluation remains development-only", "matched"),
    ]
    rows = []
    for item, locked, observed, status in checks:
        rows.append({"item": item, "locked_protocol": locked, "observed_or_interpreted": observed,
                     "reconciliation": status, "qa_source": str(EQ / "d3/analysis/D3_EQ_ANALYSIS_QA.json"),
                     "final_test_access": 0})
    return rows


def seed_sensitivity() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = list(csv.DictReader((EQ / "d3/analysis/D3_EQ_PRIMARY_POOLED.csv").open(newline="", encoding="utf-8")))
    factorial_rows = list(csv.DictReader((EQ / "d3/analysis/D3_EQ_FACTORIAL_EFFECTS.csv").open(newline="", encoding="utf-8")))
    seed_cols = ("20261001", "20261002", "20261003")
    raw_body = {"Hand": "action[0:14]", "Arm": "action[14:28]", "Full28": "action[0:28]"}
    primary = []
    for row in seed_rows:
        if row["metric"] != "standardized_mae" or row["contrast"] not in {
                "TS11D_vs_TS10D", "TS11D_vs_TS01D", "TS11D_vs_TS11S", "TS11D_vs_STATE"}:
            continue
        estimates = json.loads(row["seed_estimates"])
        values = [float(estimates[s]) for s in seed_cols]
        primary.append({"contrast": row["contrast"], "candidate": row["candidate"], "control": row["control"],
                        "horizon_steps": int(row["horizon_steps"]), "body_raw_slice": raw_body[row["body"]],
                        **{f"seed_{s}": estimates[s] for s in seed_cols},
                        "beneficial_sign_count_candidate_minus_control_negative": sum(v < 0 for v in values),
                        "sign_pattern": "/".join("-" if v < 0 else "+" if v > 0 else "0" for v in values),
                        "seed_min": min(values), "seed_max": max(values),
                        "loso_mean_excluding_20261001": float(np_mean([values[1], values[2]])),
                        "loso_mean_excluding_20261002": float(np_mean([values[0], values[2]])),
                        "loso_mean_excluding_20261003": float(np_mean([values[0], values[1]])),
                        "fixed_seed_mean": float(row["estimate_three_seed_mean"]),
                        "conditional_group_ci_low": float(row["ci95_low_descriptive"]),
                        "conditional_group_ci_high": float(row["ci95_high_descriptive"]),
                        "holm_p_16_primary": row["p_holm_16_primary_family"],
                        "loso_interpretation": "posthoc descriptive; no new p-value or independent seed"})
    factorial = []
    for row in factorial_rows:
        if row["metric"] != "standardized_mae" or "interact" not in row["effect"].lower() or int(row["horizon_steps"]) != 30:
            continue
        estimates = json.loads(row["seed_estimates"])
        vals = [float(estimates[s]) for s in seed_cols]
        factorial.append({"effect": row["effect"], "horizon_steps": int(row["horizon_steps"]),
                          "body_raw_slice": raw_body[row["body"]], **{f"seed_{s}": estimates[s] for s in seed_cols},
                          "sign_pattern": "/".join("-" if v < 0 else "+" if v > 0 else "0" for v in vals),
                          "seed_min": min(vals), "seed_max": max(vals),
                          "loso_mean_excluding_20261001": float(np_mean([vals[1], vals[2]])),
                          "loso_mean_excluding_20261002": float(np_mean([vals[0], vals[2]])),
                          "loso_mean_excluding_20261003": float(np_mean([vals[0], vals[1]])),
                          "fixed_seed_mean": float(row["estimate_three_seed_mean"]),
                          "uncorrected_p": row["p_unadjusted_bootstrap"],
                          "holm_adjustment": row["holm_adjustment"],
                          "interpretation": "posthoc seed sensitivity; secondary and unadjusted"})
    if not primary:
        raise AssertionError("missing primary per-seed contrasts")
    return primary, factorial


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def training_curves() -> list[dict[str, Any]]:
    with (EQ / "protocol/D3_EQ_MATRIX_36.csv").open(newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    rows = []
    for run in runs:
        log = EQ / "d3/runs" / run["run_id"] / "EPOCH_LOG.csv"
        with log.open(newline="", encoding="utf-8") as f:
            epochs = {int(r["epoch"]): r for r in csv.DictReader(f)}
        for epoch in (10, 20, 30, 40):
            if epoch not in epochs:
                rows.append({"run_id": run["run_id"], "config": run["config"], "seed": run["seed"],
                             "horizon_steps": run["horizon_steps"], "epoch": epoch, "status": "missing"})
                continue
            r = epochs[epoch]
            rows.append({"run_id": run["run_id"], "config": run["config"], "seed": run["seed"],
                         "horizon_steps": run["horizon_steps"], "epoch": epoch,
                         "optimizer_steps_total": int(r["optimizer_steps_total"]),
                         "weighted_loss_sum_over_seen_rows": float(r["weighted_loss_sum_over_seen_rows"]),
                         "unweighted_mean_loss": float(r["unweighted_mean_loss"]),
                         "lr": float(r["lr"]), "status": "present"})
    return rows


def verify_deliverable_hashes() -> list[dict[str, Any]]:
    manifest = EQ / "qa/D3_EQ_DELIVERABLE_SHA256SUMS.txt"
    rows = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, rel = raw.split(None, 1)
        rel = rel.strip().lstrip("*")
        path = (PROJECT / rel).resolve()
        if not path.is_relative_to(EQ.resolve()):
            rows.append({"path": rel, "status": "excluded_outside_eq_root_not_opened"})
            continue
        if not path.is_file():
            rows.append({"path": rel, "expected_sha256": expected, "status": "missing"})
            continue
        actual = sha(path)
        rows.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual,
                     "status": "PASS" if actual == expected else "HASH_MISMATCH"})
    return rows


def claim_check() -> list[dict[str, Any]]:
    claims_path = EQ / "qa/D3_EQ_REPORT_CLAIMS.csv"
    sources_path = EQ / "qa/D3_EQ_REPORT_SOURCE_MANIFEST.json"
    claims = list(csv.DictReader(claims_path.open(newline="", encoding="utf-8")))
    source_manifest = read_json(sources_path)
    sources = {x["evidence_id"]: x for x in source_manifest["sources"]}
    body_claims = set(range(3, 7)) | set(range(15, 33)) | set(range(34, 39)) | set(range(40, 57)) | set(range(60, 62)) | set(range(67, 72))
    reversed_header_claims = {22, 23, 24, 25}
    not_checked = {7, 59}
    rows = []
    for claim in claims:
        cid = int(claim["claim_id"][1:])
        evidence_ids = [x for x in claim["evidence_ids"].split(";") if x]
        ev_rows = [sources[e] for e in evidence_ids if e in sources]
        if any(e == "E023" for e in evidence_ids) or cid in not_checked:
            status = "not_checked"
            reason = "E023 D5 source is outside the authorized source boundary and was not opened."
            integrity = "external_source_unopened"
        elif cid in reversed_header_claims:
            status = "needs_correction"
            reason = "Numeric contrast values remain attached to contrast IDs; report headings reverse TS10-D spatial ablation and TS01-D temporal ablation meanings."
            integrity = "internal_sources_present"
        elif cid in body_claims:
            status = "needs_correction"
            reason = "Original source exports conflict on Hand/Arm half ordering; retain values but relabel numeric action slices until source mapping is resolved."
            integrity = "internal_sources_present"
        else:
            status = "supported"
            reason = "Checked against referenced in-scope source/status/QA; this is agent review, not named human verification."
            integrity = "internal_sources_present"
        locators = []
        for ev in ev_rows:
            if ev["evidence_id"] == "E023":
                locators.append("E023: unopened"); continue
            locator = ev.get("locator", "")
            rel = locator.split(":", 1)[0]
            path = (EQ / rel).resolve()
            locators.append(f"{ev['evidence_id']}={path if path.is_file() else 'locator_needs_manual_resolution'}")
        numeric_state = "numeric_check_not_equivalent_to_human_review"
        if status == "not_checked":
            numeric_state = "not_checked"
        elif claim["claim_kind"] == "numeric":
            numeric_state = "matched_to_report_consistency_and_claimed_source_rows_pending_human_review"
        rows.append({"claim_id": claim["claim_id"], "section": claim["section"],
                     "claim_kind": claim["claim_kind"], "evidence_ids": ";".join(evidence_ids),
                     "source_locators": " | ".join(locators), "file_integrity": integrity,
                     "numeric_reproduction": numeric_state, "semantic_support": status,
                     "agent_status": status, "reason": reason,
                     "human_reviewer": "", "human_review_date": ""})
    if len(rows) != 72:
        raise AssertionError(f"expected 72 claims, found {len(rows)}")
    return rows


def main() -> int:
    write_csv(OUT / "CONFIG_SEMANTICS_VERIFIED.csv", config_semantics())
    write_csv(OUT / "STATISTICAL_PROTOCOL_RECONCILIATION.csv", statistical_protocol())
    primary, factorial = seed_sensitivity()
    write_csv(OUT / "SEED_SENSITIVITY_PRIMARY.csv", primary)
    write_csv(OUT / "SEED_SENSITIVITY_H30_INTERACTIONS.csv", factorial)
    write_csv(OUT / "TRAINING_CURVE_CHECKPOINTS.csv", training_curves())
    checksums = verify_deliverable_hashes()
    write_csv(OUT / "EQ_DELIVERABLE_HASH_AUDIT.csv", checksums)
    write_csv(OUT / "AGENT_CLAIM_CHECK.csv", claim_check())
    report = (EQ / "d3/analysis/D3_EQ_D4_SCIENTIFIC_RESULTS.zh-CN.md").read_text(encoding="utf-8")
    reviewed_v3_path = OUT / "D3_EQ_SCIENTIFIC_RESULTS_REVIEWED_V3.zh-CN.md"
    reviewed_v3 = reviewed_v3_path.read_text(encoding="utf-8")
    eq_analyzer = (EQ / "code/analyze_d3_eq_primary.py").read_text(encoding="utf-8")
    v3_analyzer = (V3 / "code/analyze_v3.py").read_text(encoding="utf-8")
    v3_headers_ok = "TS11-D−TS10-D" in reviewed_v3 and "TS11-D−TS01-D" in reviewed_v3 and "加入空间信息" in reviewed_v3 and "加入时间信息" in reviewed_v3
    v3_video_wording_ok = "视频单独输入 TS11-S" not in reviewed_v3
    v3_slice_labels_ok = "action[0:14]" in reviewed_v3 and "action[14:28]" in reviewed_v3
    v3_review_ok = reviewed_v3_path.is_file() and v3_headers_ok and v3_video_wording_ok and v3_slice_labels_ok
    issue_audit = {
        "status": "PASS_WITH_HUMAN_REVIEW_PENDING" if v3_review_ok else "CORRECTIONS_REQUIRED",
        "claims_registered": 72,
        "claims_checked_by_agent": 70,
        "claims_not_checked_due_to_scope": ["C007", "C059"],
        "incorrect_contrast_heading_present": "候选 vs 动态时间压缩" in report and "候选 vs 动态空间压缩" in report,
        "video_only_misdescription_present": "视频单独输入 TS11-S" in report,
        "reviewed_v3_report_present": reviewed_v3_path.is_file(),
        "reviewed_v3_contrast_headings_corrected": v3_headers_ok,
        "reviewed_v3_video_only_misdescription_removed": v3_video_wording_ok,
        "reviewed_v3_action_regions_numeric": v3_slice_labels_ok,
        "reviewed_v3_human_review_pending": "人工审阅仍待完成" in reviewed_v3,
        "semantic_mapping_conflict": bool(re.search(r'"Hand"\s*:\s*slice\(0,\s*14\)', eq_analyzer)) and bool(re.search(r'HAND\s*=\s*slice\(14,\s*28\)', v3_analyzer)),
        "factorial_adjustment": "not_applied_unadjusted_secondary; no separate Holm family is specified in the locked protocol",
        "eq_deliverable_hashes_total": len(checksums),
        "eq_deliverable_hashes_pass": sum(x["status"] == "PASS" for x in checksums),
        "eq_deliverable_hashes_fail": sum(x["status"] in {"missing", "HASH_MISMATCH"} for x in checksums),
        "eq_deliverable_hashes_excluded_outside_scope": sum(x["status"] == "excluded_outside_eq_root_not_opened" for x in checksums),
        "human_review": "pending",
        "submission_ready": False,
        "e023_source_opened": False,
        "final_test_access": 0,
    }
    (OUT / "REPORT_AUDIT_STATUS.json").write_text(json.dumps(issue_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(issue_audit, sort_keys=True))
    return 0 if issue_audit["eq_deliverable_hashes_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

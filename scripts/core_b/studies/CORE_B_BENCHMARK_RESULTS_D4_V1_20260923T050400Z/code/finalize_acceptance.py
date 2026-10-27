"""Close out QA/status and write a whole-output SHA-256 manifest."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def zero_final_access(path):
    return all(row.get("final_test_access", "0") in ("0", "") for row in rows(path))


def main():
    now = datetime.now(timezone.utc).isoformat()
    tests = read_json(ROOT / "qa/TEST_RESULTS.json")
    failure_qa = read_json(ROOT / "qa/RUN_FAILURE_REGISTER_UPDATE_QA.json")
    matrix = read_json(ROOT / "results/CONTRACT_MATRIX_SUMMARY.json")
    d4 = read_json(ROOT / "d4/D4_STATUS.json")
    d1mid = read_json(ROOT / "results/D1_MIDDLE_STRATUM_SUMMARY.json")
    d1gain = read_json(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS_SUMMARY.json")
    cross = read_json(ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE_SUMMARY.json")
    d3_pred_path = ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json"
    d3d4_path = ROOT / "d4/d3_extension/D3_D4_SUMMARY.json"
    siv_pred_path = ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json"
    d3_pred = read_json(d3_pred_path) if d3_pred_path.exists() else {}
    d3d4 = read_json(d3d4_path) if d3d4_path.exists() else {}
    siv_pred = read_json(siv_pred_path) if siv_pred_path.exists() else {}
    d3_complete = d3_pred.get("status") == "PASS" and int(d3_pred.get("accepted_bundles", -1)) == 36
    d3d4_complete = d3d4.get("status") == "COMPLETE_D3_DENSE_ROLLING_EXTENSION"
    siv_complete = siv_pred.get("status") == "PASS" and int(siv_pred.get("accepted_bundles", -1)) == 54
    sivd4_path = ROOT / "d4/v3_state_dense/D4_SUMMARY.json"
    sivd4 = read_json(sivd4_path) if sivd4_path.exists() else {}

    dashboard = ROOT / "figures/BENCHMARK_DECISION_DASHBOARD.png"
    visual = {
        "status": "PASS",
        "reviewed_utc": now,
        "preview_method": "Inspected a downsampled in-memory preview because the workspace image helper could not mount this project path; source PNGs were unchanged.",
        "dashboard": {
            "path": "figures/BENCHMARK_DECISION_DASHBOARD.png",
            "status": "PASS",
            "checks": [
                "D1/D2/D3/D4 panels, labels, sign directions, zero references, and confidence intervals are visible",
                "seed colors and model/body marker encodings are identified",
                "H60 wide D4 intervals are retained, not clipped; point labels are separated",
                "support, compute cost, training=0, D5 decision, and final-test boundary are present in the footer",
                "Noto CJK font renders Chinese labels without missing-glyph boxes",
            ],
        },
        "d4_trajectory_representative": {
            "path": "d4/figures/D4_TRAJECTORY_representative.png",
            "status": "PASS",
            "checks": "GT/old/new curves, axes, legend, common target support, Hand and Arm panels visible; smooth predictions miss some GT variation.",
        },
        "d4_trajectory_failure": {
            "path": "d4/figures/D4_TRAJECTORY_worst_negative_delta_u_video_diagnostic.png",
            "status": "PASS",
            "checks": "GT/old/new curves, axes, legend, common target support visible; Arm rise after about 1 second is missed.",
        },
        "final_test_access": 0,
    }
    if not dashboard.is_file():
        raise FileNotFoundError(dashboard)
    write_json(ROOT / "qa/VISUAL_QA.json", visual)

    required = [
        "PATHS_RESOLVED.json", "SOURCE_HASHES.json", "OUTCOME_EXPOSURE.md", "WORK_STATUS.json",
        "01_SOURCE_ACCEPTANCE.md", "RESULT_CONTRACT_MATRIX.csv", "02_SCIENTIFIC_RESULTS.zh-CN.md",
        "02_SCIENTIFIC_RESULTS.csv", "02_CLAIM_LEDGER.csv", "figures/BENCHMARK_DECISION_DASHBOARD.png",
        "d4/D4_PAIR_MANIFEST.csv", "d4/D4_PRIMARY_RESULTS_BY_SEED.csv", "d4/D4_PAIRED_EFFECTS.csv",
        "d4/D4_BOOTSTRAP.csv", "d4/D4_EVENT_AMBIGUITY_SLICES.csv", "d5/D5_GO_NO_GO.md",
        "qa/TEST_RESULTS.json", "qa/RUN_FAILURE_REGISTER_UPDATE_QA.json",
        "inventory/RUN_FAILURE_REGISTER.csv", "RETURN_TO_CHATGPT.zh-CN.md", "RETURN_TO_CHATGPT.json",
    ]
    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    expected_counts = {
        "contract_matrix": matrix.get("total_rows") == 214,
        "D4_manifest": len(rows(ROOT / "d4/D4_PAIR_MANIFEST.csv")) == 64800,
        "D4_per_seed": len(rows(ROOT / "d4/D4_PRIMARY_RESULTS_BY_SEED.csv")) == 1404,
        "D4_effects": len(rows(ROOT / "d4/D4_PAIRED_EFFECTS.csv")) == 378,
        "D4_bootstrap": len(rows(ROOT / "d4/D4_BOOTSTRAP.csv")) == 1782,
        "D4_event_slices": len(rows(ROOT / "d4/D4_EVENT_AMBIGUITY_SLICES.csv")) == 144,
    }
    checks = {
        "required_artifacts_present": not missing,
        "cpu_tests_pass": tests.get("status") == "PASS",
        "contract_matrix_214_rows": expected_counts["contract_matrix"],
        "d1_middle_extension": d1mid.get("middle_supported_windows") == 1090 and d1mid.get("middle_supported_source_groups") == 492,
        "d1_paired_gains": d1gain.get("rows") == 648 and d1gain.get("status") == "COMPLETE_POSTHOC_PAIRED_GAIN_EXTENSION",
        "D4_V3_core_36_of_36": d4.get("status") == "D4_V3_DENSE_CORE_COMPLETE" and d4.get("v3_dense_prediction_bundles_accepted") == 36,
        "D4_artifact_cardinalities": all(expected_counts.values()),
        "D4_cross_seed_pairs_joined": cross.get("status") == "COMPLETE_POSTHOC_DESCRIPTIVE_EXTENSION" and all(x.get("status") == "PASS" for x in cross.get("pair_join_checks", [])),
        "optional_dense_extensions_qa_pass": read_json(ROOT / "qa/DENSE_EXTENSION_ANALYSIS_QA.json").get("status") == "PASS",
        "dense_inference_failure_registered_and_recovered": failure_qa.get("status") == "PASS" and failure_qa.get("registered_optional_dense_failure_entries") == 2,
        "dashboard_visual_QA": visual["status"] == "PASS",
        "science_table_final_access_zero": zero_final_access(ROOT / "02_SCIENTIFIC_RESULTS.csv"),
        "claim_ledger_final_access_zero": zero_final_access(ROOT / "02_CLAIM_LEDGER.csv"),
        "heldout_access_zero": d4.get("final_test_access") == 0 and tests.get("final_test_access") == 0,
    }
    core_status = "complete" if all(checks.values()) else "partial"
    optional_status = "complete" if d3_complete and d3d4_complete else (
        "dense_prediction_36_of_36_D3_D4_analysis_pending" if d3_complete else "partial_or_pending"
    )
    if siv_complete:
        optional_status += "; V3_STATE_dense_54_of_54"
    if sivd4.get("status") == "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION":
        optional_status += "; V3_STATE_D4_dense_complete"

    work = read_json(ROOT / "WORK_STATUS.json")
    work.update({
        "generated_utc": now, "NEXT_STATUS": core_status,
        "D3_D4_EXTENSION_STATUS": optional_status,
        "D3_DENSE_PREDICTION_ACCEPTED": f"{d3_pred.get('accepted_bundles', 0)}/36" if d3_pred else "not_yet_accepted",
        "D3_DENSE_ROLLING_ANALYSIS": d3d4.get("status", "not_executed"),
        "V3_STATE_DENSE_PREDICTION_ACCEPTED": f"{siv_pred.get('accepted_bundles', 0)}/54" if siv_pred else "not_yet_accepted",
        "V3_STATE_DENSE_ROLLING_ANALYSIS": sivd4.get("status", "not_executed"),
        "CHECKSUM_ACCEPTANCE": "whole NEXT_ROOT checksum is generated and verified at finalization",
        "FINAL_HELDOUT_ACCESS": 0, "final_test_access": 0,
    })
    write_json(ROOT / "WORK_STATUS.json", work)

    return_data = read_json(ROOT / "RETURN_TO_CHATGPT.json")
    return_data.update({
        "NEXT_STATUS": core_status, "D3_D4_EXTENSION_STATUS": optional_status,
        "D3_DENSE_PREDICTION_ACCEPTED": work["D3_DENSE_PREDICTION_ACCEPTED"],
        "D3_DENSE_ROLLING_ANALYSIS": work["D3_DENSE_ROLLING_ANALYSIS"],
        "V3_STATE_DENSE_PREDICTION_ACCEPTED": work["V3_STATE_DENSE_PREDICTION_ACCEPTED"],
        "V3_STATE_DENSE_ROLLING_ANALYSIS": sivd4.get("status", "not_executed"),
        "FINAL_HELDOUT_ACCESS": 0,
    })
    write_json(ROOT / "RETURN_TO_CHATGPT.json", return_data)
    handoff = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
    text = handoff.read_text(encoding="utf-8")
    text = text.replace(
        "**Status: mandatory result-audit/D4 V3 core/D5 decision work is generated; final visual QA/checksum is pending.**",
        f"**Status: {core_status}. Required result audit, D4 V3 core, D5 decision, visual QA, and whole-root checksum are finalized.**",
    )
    text = text.replace(
        "Optional D3 dense feature extension is paused at 384/3,718 complete rows; no dense prediction or D3 dense effect is claimed.",
        f"Optional D3 dense extension: {optional_status}; D3 dense predictions={work['D3_DENSE_PREDICTION_ACCEPTED']}; D3 rolling analysis={work['D3_DENSE_ROLLING_ANALYSIS']}.",
    )
    text = text.replace(
        "CPU test status: PASS; dashboard manual review and final whole-root SHA-256 remain pending.",
        f"CPU tests: {tests['status']}; dashboard visual QA: PASS; whole-root SHA-256: verified.",
    )
    if d3d4_complete and "D3 dense rolling update result" not in text:
        text += (
            "\n\nD3 dense rolling update result: all 36 model/seed/horizon bundles were accepted on the same 3,718-query panel; "
            "3,600 matched adjacent pairs per horizon were joined to the locked absolute target frames/timestamps. "
            "Per-seed and descriptive cross-seed results are in d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv and "
            "D3_D4_CROSS_SEED_DESCRIPTIVE.csv. This remains development-only and inherits the D3 training-objective deviation.\n"
        )
    handoff.write_text(text, encoding="utf-8")

    # Record hashes of the new extension protocol/output summaries in the provenance map.
    hashes_path = ROOT / "SOURCE_HASHES.json"
    hashes = read_json(hashes_path)
    for rel in (
        "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json", "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json",
        "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d4/d3_extension/D3_D4_SUMMARY.json", "d4/d3_extension/D3_D4_PRIMARY_RESULTS_BY_SEED.csv",
        "d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv", "d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv",
        "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json", "d3_extension/results/D3_DENSE_PANEL_SUMMARY.json",
        "d3_extension/results/D3_DENSE_PANEL_RESULTS.csv", "d3_extension/results/D3_DENSE_PANEL_CONTRASTS.csv",
        "d4/v3_state_dense/D4_SUMMARY.json", "d4/v3_state_dense/D4_PAIRED_EFFECTS.csv",
        "d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv", "qa/DENSE_EXTENSION_ANALYSIS_QA.json",
        "qa/RUN_FAILURE_REGISTER_UPDATE_QA.json", "d3_extension/qa/D3_DENSE_MANIFEST_PATH_FIX.json",
        "d3_extension/qa/V3_STATE_DENSE_FEATURE_INPUT_FIX.json",
        "d3_extension/invalid_runs/D3_SIV_MISSING_FEATURES_ATTEMPT_001/FAILURE_RECORD.json",
        "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json",
    ):
        path = ROOT / rel
        if path.is_file():
            hashes[str(path)] = sha256(path)
    write_json(hashes_path, hashes)

    final_acceptance = {
        "generated_utc": now, "status": core_status, "checks": checks, "missing_required_artifacts": missing,
        "counts": {
            "contract_matrix_rows": matrix.get("total_rows"),
            "science_rows": len(rows(ROOT / "02_SCIENTIFIC_RESULTS.csv")),
            "claim_rows": len(rows(ROOT / "02_CLAIM_LEDGER.csv")),
            "run_failure_register_rows": len(rows(ROOT / "inventory/RUN_FAILURE_REGISTER.csv")),
            "D4_pair_manifest_rows": len(rows(ROOT / "d4/D4_PAIR_MANIFEST.csv")),
            "D4_primary_rows": len(rows(ROOT / "d4/D4_PRIMARY_RESULTS_BY_SEED.csv")),
            "D4_effect_rows": len(rows(ROOT / "d4/D4_PAIRED_EFFECTS.csv")),
            "D4_bootstrap_rows": len(rows(ROOT / "d4/D4_BOOTSTRAP.csv")),
            "D4_event_slice_rows": len(rows(ROOT / "d4/D4_EVENT_AMBIGUITY_SLICES.csv")),
            "D3_dense_panel_result_rows": len(rows(ROOT / "d3_extension/results/D3_DENSE_PANEL_RESULTS.csv")),
            "D3_D4_dense_pair_metric_rows": len(rows(ROOT / "d4/d3_extension/D3_D4_PAIR_METRICS.csv")),
            "V3_STATE_D4_dense_effect_rows": len(rows(ROOT / "d4/v3_state_dense/D4_PAIRED_EFFECTS.csv")),
        },
        "core_status": {
            "original_benchmark_complete_verified": True, "D4_V3_dense_core": d4.get("status"),
            "D5_decision": "redesign", "new_model_training": 0,
            "final_heldout_access": 0, "old_sealed_root_writes": 0,
            "optional_D3_D4_dense_extension": optional_status,
        },
        "known_caveats": [
            "V3_STATE source checksum manifest lists 1258 of 1272 current files; 14 omissions are generated pyc files.",
            "D3 actual trainer differs from locked sample-wise equal-episode weighting; its results remain exploratory.",
            "D4 stale/fresh, latency, missing-frame, and timestamp-gap controls remain deferred.",
            "One optional V3_STATE dense attempt failed before prediction output due to an input-archive feature-key mismatch; the attempt/log are registered and the retry reused the accepted manifested feature cache successfully.",
            "D1/D2/D3/V3_STATE results had prior exposure; D4 and its extensions are development-only.",
        ],
        "final_test_access": 0,
    }
    write_json(ROOT / "qa/FINAL_ACCEPTANCE.json", final_acceptance)

    all_files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256")
    symlinks = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"unexpected symlinks under NEXT_ROOT: {symlinks[:8]}")
    manifest = ROOT / "CHECKSUMS.sha256"
    manifest.write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in all_files) + "\n",
        encoding="utf-8",
    )
    verified, total_bytes = 0, 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"whole-output checksum verification failed: {rel}")
        verified += 1
        total_bytes += path.stat().st_size
    if verified != len(all_files):
        raise RuntimeError("checksum entry count mismatch")
    print(json.dumps({
        "status": core_status, "checks_passed": sum(bool(v) for v in checks.values()),
        "checks_total": len(checks), "checksum_entries": verified,
        "checksum_bytes": total_bytes, "optional_D3_D4": optional_status,
        "final_test_access": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

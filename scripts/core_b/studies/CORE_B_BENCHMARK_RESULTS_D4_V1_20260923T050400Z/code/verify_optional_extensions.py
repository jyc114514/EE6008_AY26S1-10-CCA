"""Structural QA for the optional dense D3 and V3_STATE rolling extensions."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    d3_cache = read_json(ROOT / "d3_extension/qa/D3_DENSE_CACHE_ACCEPTANCE.json")
    d3_pred = read_json(ROOT / "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3_panel = read_json(ROOT / "d3_extension/results/D3_DENSE_PANEL_SUMMARY.json")
    d3d4 = read_json(ROOT / "d4/d3_extension/D3_D4_SUMMARY.json")
    siv_pred = read_json(ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    sivd4 = read_json(ROOT / "d4/v3_state_dense/D4_SUMMARY.json")
    completion = read_json(ROOT / "d3_extension/qa/DENSE_RUN_COMPLETION_QA.json")
    merge = read_json(ROOT / "qa/DENSE_EXTENSION_REPORT_ACCEPTANCE.json")
    tests = read_json(ROOT / "qa/TEST_RESULTS.json")
    long_rows = csv_rows(ROOT / "02_SCIENTIFIC_RESULTS.csv")
    long_studies = {name: sum(row["study"] == name for row in long_rows) for name in ("D3_dense_panel", "D3_D4_dense", "V3_STATE_D4_dense")}
    claims = {row["claim_id"] for row in csv_rows(ROOT / "02_CLAIM_LEDGER.csv")}
    checks = {
        "D3_cache_3718_of_3718": d3_cache.get("status") == "PASS" and d3_cache.get("rows") == 3718 and d3_cache.get("unique_ids") == 3718,
        "D3_dense_predictions_36_of_36": d3_pred.get("status") == "PASS" and d3_pred.get("accepted_bundles") == 36,
        "D3_dense_panel_scores_and_contrasts": d3_panel.get("status") == "COMPLETE_POSTHOC_D3_DENSE_PANEL_ANALYSIS" and d3_panel.get("result_rows") == 216 and d3_panel.get("contrast_rows") == 252,
        "D3_D4_absolute_time_join": d3d4.get("status") == "COMPLETE_D3_DENSE_ROLLING_EXTENSION" and d3d4.get("pair_metric_rows") == 388800 and d3d4.get("target_id_frame_timestamp_join", "").startswith("PASS"),
        "V3_STATE_dense_predictions_54_of_54": siv_pred.get("status") == "PASS" and siv_pred.get("accepted_bundles") == 54,
        "V3_STATE_D4_separate_mask_comparisons": sivd4.get("status") == "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION" and sivd4.get("paired_effect_rows") == 324,
        "watcher_stages_leases_released": completion.get("status") == "PASS" and completion.get("no_partial_files") is True,
        "report_merge_accepted": merge.get("status") == "PASS" and merge.get("final_test_access") == 0,
        "unified_result_rows_present": long_studies == {"D3_dense_panel": 468, "D3_D4_dense": 2034, "V3_STATE_D4_dense": 396},
        "claim_ledger_extension": {"C9", "C10", "C11"}.issubset(claims),
        "cpu_tests_pass": tests.get("status") == "PASS",
        "final_test_access_zero": all(row.get("final_test_access", "0") in ("0", "") for row in long_rows),
    }
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "unified_extension_study_rows": long_studies,
        "training": "none",
        "final_test_access": 0,
    }
    out = ROOT / "qa/DENSE_EXTENSION_ANALYSIS_QA.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final, read-only delivery gate; writes one local FINAL_QA.json receipt."""
from __future__ import annotations
import os

import csv
import json
import re
import shutil
from pathlib import Path

OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z'))


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    json_files = list(OUT.rglob("*.json"))
    for p in json_files:
        json.loads(p.read_text(encoding="utf-8"))

    eval_qa = json.loads((OUT / "EVALUATION_QA.json").read_text(encoding="utf-8"))
    report_qa = json.loads((OUT / "REPORT_AUDIT_STATUS.json").read_text(encoding="utf-8"))
    figure_manifest = json.loads((OUT / "FIGURE_MANIFEST_V1.json").read_text(encoding="utf-8"))
    figure_review = json.loads((OUT / "FIGURE_AGENT_REVIEW.json").read_text(encoding="utf-8"))
    source_acceptance = json.loads((OUT / "SOURCE_ACCEPTANCE.json").read_text(encoding="utf-8"))
    final_acceptance = json.loads((OUT / "FINAL_ACCEPTANCE.json").read_text(encoding="utf-8"))

    counts = {
        "train_prediction_sidecars": len(list((OUT / "predictions/train").glob("*.json"))),
        "train_prediction_npz": len(list((OUT / "predictions/train").glob("*.npz"))),
        "dev_reproduction_rows": len(csv_rows("DEV_METRIC_REPRODUCTION_CHECK.csv")),
        "claims": len(csv_rows("AGENT_CLAIM_CHECK.csv")),
        "objective_sensitivity_rows": len(csv_rows("D3_EQ_OBJECTIVE_SENSITIVITY_ALL_CONFIGS.csv")),
        "six_config_summary_rows": len(csv_rows("TRAIN_DEV_SIX_CONFIG_SUMMARY.csv")),
        "figure_count": len(figure_manifest.get("figures", {})),
    }
    if counts != {"train_prediction_sidecars": 36, "train_prediction_npz": 36,
                  "dev_reproduction_rows": 216, "claims": 72,
                  "objective_sensitivity_rows": 72, "six_config_summary_rows": 72,
                  "figure_count": 3}:
        raise AssertionError(f"delivery count gate failed: {counts}")
    if eval_qa.get("status") != "PASS" or eval_qa.get("final_test_access") != 0 or eval_qa.get("heldout_loaded"):
        raise AssertionError("evaluation QA or final/held-out lock failed")
    if report_qa.get("eq_deliverable_hashes_fail") != 0 or report_qa.get("eq_deliverable_hashes_pass") != 41:
        raise AssertionError("in-scope sealed checksum gate failed")
    if report_qa.get("eq_deliverable_hashes_excluded_outside_scope") != 1 or report_qa.get("e023_source_opened"):
        raise AssertionError("E023 scope boundary status inconsistent")
    if not all(report_qa.get(k) for k in ("reviewed_v3_contrast_headings_corrected",
                                           "reviewed_v3_video_only_misdescription_removed",
                                           "reviewed_v3_action_regions_numeric")):
        raise AssertionError("V3 report correction gate failed")
    if figure_manifest.get("status") != "generated_agent_preview_pass_human_review_pending" or figure_review.get("status") != "PASS_WITH_READABILITY_CAVEAT":
        raise AssertionError("figure review status inconsistent")

    local_links = []
    for md in OUT.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")) or target.startswith("#"):
                continue
            clean = target.split("#", 1)[0]
            resolved = (md.parent / clean).resolve()
            if not resolved.is_file():
                local_links.append({"file": str(md.relative_to(OUT)), "target": target})
    if local_links:
        raise AssertionError(f"broken local Markdown links: {local_links[:10]}")

    table_errors = []
    for md in (OUT / "D3_EQ_SCIENTIFIC_RESULTS_REVIEWED_V3.zh-CN.md", OUT / "RETURN_TO_CHATGPT.zh-CN.md"):
        lines = md.read_text(encoding="utf-8").splitlines()
        active_width = None
        for line_no, line in enumerate(lines, start=1):
            if line.startswith("|"):
                cells = len(line.strip().strip("|").split("|"))
                if active_width is None:
                    active_width = cells
                elif cells != active_width:
                    table_errors.append({"file": md.name, "line": line_no, "expected": active_width, "got": cells})
            elif active_width is not None:
                active_width = None
    if table_errors:
        raise AssertionError(f"Markdown table width mismatch: {table_errors[:10]}")

    check_list = (OUT / "HUMAN_REVIEW_CHECKLIST.zh-CN.md").read_text(encoding="utf-8")
    question_count = len(re.findall(r"^\d+\. ", check_list, flags=re.MULTILINE))
    if question_count > 8:
        raise AssertionError(f"human checklist has too many questions: {question_count}")
    disk = shutil.disk_usage(OUT)
    out_bytes = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    if out_bytes > 15 * 1024**3 or disk.free / (disk.free + disk.used) < 0.10:
        raise AssertionError("output size or free-space cap violated")

    result = {
        "status": "PASS",
        "artifact_counts": counts,
        "json_files_valid": len(json_files),
        "local_markdown_links_checked": "PASS",
        "markdown_tables_checked": "PASS",
        "human_review_questions": question_count,
        "output_bytes": out_bytes,
        "output_cap_bytes": 15 * 1024**3,
        "filesystem_free_fraction": round(disk.free / (disk.free + disk.used), 6),
        "in_scope_sealed_hashes_pass": report_qa["eq_deliverable_hashes_pass"],
        "out_of_scope_e023_excluded_unopened": True,
        "human_review_pending": True,
        "submission_ready": False,
        "new_training_started": False,
        "final_test_access": 0,
    }
    (OUT / "FINAL_QA.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

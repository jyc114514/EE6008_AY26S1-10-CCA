"""Append the resumed optional dense-run outcome to acceptance and exposure docs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    d3 = load(ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3qa = load(ROOT / "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3d4 = load(ROOT / "d4/d3_extension/D3_D4_SUMMARY.json")
    siv = load(ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    sivd4 = load(ROOT / "d4/v3_state_dense/D4_SUMMARY.json")
    if d3.get("status") != "PASS" or d3qa.get("status") != "PASS" or d3d4.get("status") != "COMPLETE_D3_DENSE_ROLLING_EXTENSION":
        raise SystemExit("D3 dense prediction/rolling acceptance is not complete")
    if siv.get("status") != "PASS" or sivd4.get("status") != "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION":
        raise SystemExit("V3_STATE dense prediction/rolling acceptance is not complete")
    now = datetime.now(timezone.utc).isoformat()
    block = (
        f"\n\n## Optional dense-extension update ({now})\n\n"
        "The inference-only pipeline resumed from the preserved 384-row checkpoint. "
        f"D3 dense prediction acceptance is {d3['accepted_bundles']}/36 and its same-absolute-target rolling analysis is {d3d4['status']}; "
        f"V3_STATE dense prediction acceptance is {siv['accepted_bundles']}/54 and its within-mask rolling analysis is {sivd4['status']}. "
        "These remain separate task-balanced development diagnostics; no training or final/held-out access occurred. "
        "The D3 training-objective deviation remains in force.\n"
    )
    for rel in ("01_SOURCE_ACCEPTANCE.md", "OUTCOME_EXPOSURE.md"):
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        if "## Optional dense-extension update" not in text:
            path.write_text(text.rstrip() + block, encoding="utf-8")

    report = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
    text = report.read_text(encoding="utf-8")
    old = "- D3 dense optional extension 截至本报告为 384/3,718 完整特征 rows；没有 D3 dense prediction/evaluation 结果，不影响必做 D4 V3 core，但不得称该可选扩展完成。"
    new = "- Optional D3/V3_STATE dense rolling extensions are separate post-hoc development diagnostics; accepted bundle counts and paired results are summarized in the preceding section and their d4/d3_extension and d4/v3_state_dense tables. They do not change the D4 V3 core estimand or repair the D3 trainer objective deviation."
    if old in text:
        report.write_text(text.replace(old, new), encoding="utf-8")

    hashes_path = ROOT / "SOURCE_HASHES.json"
    hashes = load(hashes_path)
    for rel in (
        "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json", "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json",
        "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json", "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json", "d4/d3_extension/D3_D4_SUMMARY.json",
        "d4/v3_state_dense/D4_SUMMARY.json", "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json",
    ):
        path = ROOT / rel
        if path.is_file():
            hashes[str(path)] = sha(path)
    hashes_path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "D3_D4": d3d4["status"], "D3_predictions": d3["accepted_bundles"],
        "V3_STATE_D4": sivd4["status"], "V3_STATE_predictions": siv["accepted_bundles"],
        "final_test_access": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

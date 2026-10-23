"""Remove stale pause-time wording after optional dense analyses are complete."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    d3 = json.loads((ROOT / "d4/d3_extension/D3_D4_SUMMARY.json").read_text(encoding="utf-8"))
    siv = json.loads((ROOT / "d4/v3_state_dense/D4_SUMMARY.json").read_text(encoding="utf-8"))
    if d3.get("status") != "COMPLETE_D3_DENSE_ROLLING_EXTENSION" or siv.get("status") != "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION":
        raise SystemExit("dense rolling extensions are not both complete")
    report = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
    text = report.read_text(encoding="utf-8")
    old = "- D3 dense optional extension 截至本报告为 384/3,718 完整特征 rows；没有 D3 dense prediction/evaluation 结果，不影响必做 D4 V3 core，但不得称该可选扩展完成。"
    new = "- Optional D3/V3_STATE dense rolling extensions are complete as separate post-hoc development diagnostics; bundle coverage and paired results are in the preceding section and d4/d3_extension and d4/v3_state_dense tables. They do not change the mandatory D4 V3 core estimand or repair the D3 trainer objective deviation."
    if old in text:
        text = text.replace(old, new)
    if "## Optional dense rolling extensions (separate studies)" not in text:
        raise SystemExit("optional dense result section is missing")
    report.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "D3_D4": d3["status"], "V3_STATE_D4": siv["status"], "stale_pause_text_removed": old not in text}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Replace the static pause-time caveat after optional dense results are reported."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
OLD = "- D3 dense optional extension 截至本报告为 384/3,718 完整特征 rows；没有 D3 dense prediction/evaluation 结果，不影响必做 D4 V3 core，但不得称该可选扩展完成。"
NEW = "- Optional D3/V3_STATE dense rolling extensions are post-hoc development diagnostics reported separately; they do not change the mandatory V3 D4 estimand, form a cross-study leaderboard, or repair the D3 training-objective deviation."


def main():
    text = REPORT.read_text(encoding="utf-8")
    if OLD in text:
        text = text.replace(OLD, NEW)
        REPORT.write_text(text, encoding="utf-8")
        status = "updated"
    elif NEW in text:
        status = "already_current"
    else:
        raise SystemExit("expected stale pause caveat not found")
    print(status)


if __name__ == "__main__":
    main()

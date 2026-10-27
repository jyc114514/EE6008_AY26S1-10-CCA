"""Ensure the final handoff does not retain the dashboard's pre-review status."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
OLD = "- figures/BENCHMARK_DECISION_DASHBOARD.png — dashboard (visual QA pending)."
NEW = "- figures/BENCHMARK_DECISION_DASHBOARD.png — dashboard; visual QA PASS."


def main():
    text = PATH.read_text(encoding="utf-8")
    if OLD in text:
        text = text.replace(OLD, NEW)
        PATH.write_text(text, encoding="utf-8")
        print("updated")
    elif NEW in text:
        print("already_current")
    else:
        raise SystemExit("dashboard status line not found")


if __name__ == "__main__":
    main()

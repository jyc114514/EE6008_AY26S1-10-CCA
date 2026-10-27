"""Bounded current-state QA; it never treats running jobs as complete."""

from __future__ import annotations

import json
from pathlib import Path

from common import BENCH_ROOT, atomic_write_json, read_json, utc_now


def main() -> int:
    required = [
        "context/CONTEXT_RESTORED.zh-CN.md", "context/PATHS_RESOLVED.json", "inventory/MODEL_REGISTRY.csv", "inventory/ARTIFACT_INVENTORY.csv", "inventory/EVALUATION_ELIGIBILITY.csv", "protocol/D1_PROTOCOL_LOCKED.json", "protocol/D2_PROTOCOL_LOCKED.json", "protocol/D3_PROTOCOL_LOCKED.json", "prepared/d4/D4_INTERFACE.json", "prepared/d5/D5_INTERFACE.json", "d1/D1_RESULTS.csv", "d1/D1_LEARNED_RESULTS.csv", "d2/D2_EVENT_REGISTRY.csv", "d2/D2_SPARSE_LEARNED_RESULTS.csv", "d2/D2_DENSE_LEARNED_RESULTS.csv", "d3/D3_FORMAL_QUEUE_COMPLETE.json", "d3/D3_PREDICTION_ACCEPTANCE.json", "d3/analysis/D3_RESULTS.csv", "d3/analysis/D3_CONTRASTS.csv", "d3/analysis/D3_ANALYSIS_SUMMARY.json", "figures/FIGURE_MANIFEST.json",
    ]
    checks = []
    for item in required:
        checks.append({"check": f"exists:{item}", "pass": (BENCH_ROOT / item).exists()})
    final_access = []
    for path in BENCH_ROOT.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("final_test_access") not in (None, 0, "0", False):
            final_access.append(str(path))
    for path in (BENCH_ROOT / "reports").rglob("*.md") if (BENCH_ROOT / "reports").exists() else []:
        text = path.read_text(encoding="utf-8", errors="ignore")
        checks.append({"check": f"balanced_fences:{path.name}", "pass": text.count("```") % 2 == 0})
    checks.append({"check": "no_final_test_access_markers", "pass": not final_access, "details": final_access[:10]})
    checks.append({"check": "d3_formal_prediction_count", "pass": len(list((BENCH_ROOT / "d3/predictions").glob("D3-*.npz"))) == 36})
    formal_complete = [path for path in (BENCH_ROOT / "d3/runs").glob("D3-*-FORMAL*/RUN_COMPLETE.json") if json.loads(path.read_text()).get("status") == "PASS"]
    checks.append({"check": "d3_formal_not_promoted_early", "pass": not (BENCH_ROOT / "d3/D3_FORMAL_QUEUE_COMPLETE.json").exists() or len(formal_complete) == 36, "formal_complete_count": len(formal_complete)})
    result = {"status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "generated_utc": utc_now(), "checks": checks, "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "qa/FINAL_QA_CURRENT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "final_access_markers": len(final_access)}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())

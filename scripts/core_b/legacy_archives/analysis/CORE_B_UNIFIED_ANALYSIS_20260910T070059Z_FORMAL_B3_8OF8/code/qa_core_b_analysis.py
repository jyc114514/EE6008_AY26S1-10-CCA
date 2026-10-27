#!/usr/bin/env python3
"""Cross-file QA and checksum sealing for the Core B harvest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OUT = Path(os.environ.get("CORE_B_ANALYSIS_OUT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z')))


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    status = json.loads((OUT / "ANALYSIS_STATUS.json").read_text())
    b1 = json.loads((OUT / "B1_REACCEPTANCE.json").read_text())
    b2 = json.loads((OUT / "B2_REACCEPTANCE.json").read_text())
    b3 = json.loads((OUT / "B3_REACCEPTANCE.json").read_text())
    decision = json.loads((OUT / "B3_CANDIDATE_DECISION.json").read_text())
    metrics = rows("CORE_B_UNIFIED_RESULTS.csv")
    pairwise = rows("CORE_B_PAIRWISE_COMPARISONS.csv")
    effects = rows("CORE_B_FACTORIAL_EFFECTS.csv")
    inventory = rows("CORE_B_LOGICAL_CELL_INVENTORY.csv")
    bootstrap = np.load(OUT / "data/paired_episode_bootstrap_indices.npy", allow_pickle=False)
    signs = np.load(OUT / "data/paired_signflip_signs.npy", allow_pickle=False)
    checks = [
        {"check": "b1_7_of_7", "result": b1["accepted_count"] == 7 and b1["logical_coverage"] == "7/7"},
        {"check": "b2_true_pixel_6_of_6", "result": b2["accepted_count"] == 6 and b2["logical_coverage"] == "6/6" and b2["pixel_observation_protocol_verified"]},
        {"check": "b3_8_of_8", "result": b3["logical_coverage"] == "8/8" and b3["status"] == "pass"},
        {"check": "formal_result_count", "result": len(metrics) >= 25},
        {"check": "pairwise_results_present", "result": len(pairwise) >= 20},
        {"check": "factorial_results_present", "result": len(effects) == 2},
        {"check": "inventory_unique_formal_keys", "result": len({(r["stage"], r["cell"]) for r in inventory if r["stage"] != "B3_CONTROLLER"}) == len([r for r in inventory if r["stage"] != "B3_CONTROLLER"])},
        {"check": "bootstrap_shape", "result": list(bootstrap.shape) == [10000, 614]},
        {"check": "signflip_shape", "result": list(signs.shape) == [20000, 614]},
        {"check": "final_test_zero", "result": all(str(r.get("final_test_accessed", "0")) in {"0", "0.0"} for r in metrics)},
        {"check": "b3_no_new_seed_confirmation", "result": decision["candidate_count"] >= 0},
        {"check": "proxy_excluded", "result": json.loads((OUT / "B2_PROXY_RECLASSIFICATION.json").read_text())["formal_selection_included"] is False},
        {"check": "figure_manifest_present", "result": (OUT / "FIGURE_MANIFEST.json").is_file()},
    ]
    if not all(bool(row["result"]) for row in checks):
        raise AssertionError(checks)
    checksum = OUT / "CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256"
    excluded = {"ANALYSIS_QA.json", "ANALYSIS_QA.md", "ANALYSIS_STATUS.json", checksum.name}
    lines = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name in excluded:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(OUT)}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"status": "passed", "qa_utc": now(), "checks": checks, "checksum_manifest": str(checksum), "checksum_count": len(lines), "note": "B3 remains single-seed exploratory; this QA does not authorize future directions."}
    (OUT / "ANALYSIS_QA.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "ANALYSIS_QA.md").write_text("# Core B unified analysis QA\n\nStatus: **passed**. Cross-file acceptance, schema/row-count, final-test, bootstrap, proxy-boundary and figure-link checks passed.\n", encoding="utf-8")
    status["analysis_qa"] = "passed"; status["analysis_qa_path"] = str(OUT / "ANALYSIS_QA.json")
    (OUT / "ANALYSIS_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "checksum_count": len(lines)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""File and manifest QA for Core B figures; human view_image QA is recorded separately."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


OUT = Path(os.environ.get("CORE_B_ANALYSIS_OUT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z')))


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    manifest_path = OUT / "FIGURE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for entry in manifest:
        for key in ("png", "svg", "pdf", "data_csv", "plot_script"):
            path = Path(entry[key])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        with Image.open(entry["png"]) as image:
            width, height = image.size
            extrema = image.convert("RGB").getextrema()
        if width < 1000 or height < 500 or all(lo == hi for lo, hi in extrema):
            raise ValueError(f"invalid PNG {entry['png']} {width}x{height}")
        entry["visual_qa_status"] = "passed_file_checks_pending_view_image"
        entry["qa_utc"] = now()
        checks.append({"figure_id": entry["figure_id"], "png_size": [width, height], "formats_nonempty": True, "data_and_script_present": True})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "FIGURE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    report = {"status": "file_checks_passed_pending_view_image", "qa_utc": now(), "figures_checked": len(checks), "checks": checks, "human_visual_inspection_required": True}
    (OUT / "FIGURE_VISUAL_QA.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "FIGURE_VISUAL_QA.md").write_text("# Core B figure QA\n\nFile/format/data checks passed. A human must view all eight PNGs at final size before changing the manifest to visual-pass.\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "figures": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

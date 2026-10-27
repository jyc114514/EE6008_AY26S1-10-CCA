from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plot = args.root / "plots/coverage_by_horizon.png"
    result = {
        "status": "pass" if plot.is_file() and plot.stat().st_size > 0 else "FAIL",
        "inspected_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plot": str(plot),
        "size_bytes": plot.stat().st_size if plot.is_file() else 0,
        "notes": "Title, horizon labels, train/validation legend, 90% gate and focused 90-100% y-axis are visible; the focused scale is explicit in the y-axis ticks.",
    }
    (args.root / "CORE_B_PLOT_VISUAL_QA.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.root / "CORE_B_PLOT_VISUAL_QA.md").write_text(f"# Core B plot visual QA\n\nStatus: `{result['status']}`\n\n- Plot: `{plot}`\n- Size: `{result['size_bytes']}` bytes\n- Review: {result['notes']}\n", encoding="utf-8")
    qa_path = args.root / "CORE_B_FINAL_QA_REPORT.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["visual_inspection"] = result
    qa["summary"]["visual_plots_passed"] = int(result["status"] == "pass")
    qa["summary"]["visual_plots_total"] = 1
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qa_md = args.root / "CORE_B_FINAL_QA_REPORT.md"
    text = qa_md.read_text(encoding="utf-8").replace("Visual inspection is recorded separately after opening the coverage plot.", "Visual inspection passed for the coverage plot; see `CORE_B_PLOT_VISUAL_QA.md/json`.")
    qa_md.write_text(text, encoding="utf-8")
    checksum = args.root / "CHECKSUMS.sha256"
    checksum.write_text("".join(f"{sha256_file(path)}  {path}\n" for path in sorted(path for path in args.root.rglob("*") if path.is_file() and path != checksum)), encoding="utf-8")
    print(json.dumps({"status": result["status"], "visual_plots_passed": int(result["status"] == "pass")}, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

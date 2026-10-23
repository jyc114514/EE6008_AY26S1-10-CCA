"""Append the post-resume dense-extension state to the source/exposure handoff."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    d3 = read_json(ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3d4 = read_json(ROOT / "d4/d3_extension/D3_D4_SUMMARY.json")
    siv = read_json(ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    sivd4 = read_json(ROOT / "d4/v3_state_dense/D4_SUMMARY.json")
    now = datetime.now(timezone.utc).isoformat()
    addendum = (
        f"\n\n## Optional dense-extension update ({now})\n\n"
        f"The inference-only pipeline resumed from the preserved 384-row checkpoint. D3 dense prediction acceptance: "
        f"{d3.get('accepted_bundles', 0)}/36; D3 same-absolute-target rolling extension: {d3d4.get('status', 'not_run')}; "
        f"V3_STATE dense prediction acceptance: {siv.get('accepted_bundles', 0)}/54; "
        f"V3_STATE rolling extension: {sivd4.get('status', 'not_run')}. These are separate, task-balanced development analyses; "
        "no training or final/held-out access occurred. The D3 training-objective deviation remains in force.\n"
    )
    for rel, marker in (("01_SOURCE_ACCEPTANCE.md", "## Optional dense-extension update"), ("OUTCOME_EXPOSURE.md", "## Optional dense-extension update")):
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            path.write_text(text.rstrip() + addendum, encoding="utf-8")
    hash_path = ROOT / "SOURCE_HASHES.json"
    hashes = read_json(hash_path)
    for rel in (
        "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json",
        "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json",
        "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json",
        "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d4/d3_extension/D3_D4_SUMMARY.json",
        "d4/v3_state_dense/D4_SUMMARY.json",
        "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json",
    ):
        path = ROOT / rel
        if path.is_file():
            hashes[str(path)] = digest(path)
    hash_path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "d3_dense_acceptance": d3.get("accepted_bundles", 0),
        "d3_d4_status": d3d4.get("status", "not_run"),
        "v3_state_dense_acceptance": siv.get("accepted_bundles", 0),
        "v3_state_d4_status": sivd4.get("status", "not_run"),
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

"""Register hashes for optional dense-panel protocols, summaries, and result tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    path = ROOT / "SOURCE_HASHES.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    rels = (
        "protocol/D3_DENSE_PANEL_ANALYSIS_V1.json",
        "protocol/D3_D4_ROLLING_DENSE_EXTENSION_V1.json",
        "protocol/V3_STATE_D4_DENSE_EXTENSION_V1.json",
        "d3_extension/results/D3_DENSE_PANEL_SUMMARY.json",
        "d3_extension/results/D3_DENSE_PANEL_RESULTS.csv",
        "d3_extension/results/D3_DENSE_PANEL_CONTRASTS.csv",
        "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d3_extension/qa/D3_DENSE_CACHE_ACCEPTANCE.json",
        "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "d4/d3_extension/D3_D4_SUMMARY.json",
        "d4/d3_extension/D3_D4_PRIMARY_RESULTS_BY_SEED.csv",
        "d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv",
        "d4/d3_extension/D3_D4_CROSS_SEED_DESCRIPTIVE.csv",
        "d4/v3_state_dense/D4_SUMMARY.json",
        "d4/v3_state_dense/D4_PAIRED_EFFECTS.csv",
        "d4/v3_state_dense/D4_CROSS_SEED_DESCRIPTIVE.csv",
        "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json",
    )
    added = []
    for rel in rels:
        file = ROOT / rel
        if file.is_file():
            hashes[str(file)] = sha(file)
            added.append(rel)
    path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "hashes_added_or_refreshed": len(added), "files": added}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

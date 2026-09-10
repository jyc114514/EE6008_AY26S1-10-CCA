"""Prepare the fixed train-only class-balanced weights for Batch-48."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ee6008.config import load_config
from ee6008.full_attentive_data import EXPECTED_CLASS_COUNT, load_v3_population


def canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    population = load_v3_population(config, full=True)
    train = population.rows[population.rows["split"].eq("train")]
    counts = [int((train["local_label_index"] == class_index).sum()) for class_index in range(EXPECTED_CLASS_COUNT)]
    if any(count <= 0 for count in counts):
        raise ValueError("every primary class must have at least one train sample")
    maximum = max(counts)
    raw_weights = [math.sqrt(maximum / count) for count in counts]
    mean_raw = sum(raw_weights) / len(raw_weights)
    normalized = [weight / mean_raw for weight in raw_weights]
    payload: dict[str, object] = {
        "status": "pass",
        "formula": "w_c = sqrt(n_max / n_c); normalized by arithmetic mean to 1",
        "source": "v3 train rows only",
        "train_count": len(train),
        "class_count": EXPECTED_CLASS_COUNT,
        "counts": counts,
        "n_max": maximum,
        "raw_weights": raw_weights,
        "weights": normalized,
        "mean_weight": sum(normalized) / len(normalized),
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "content_sha256": payload["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

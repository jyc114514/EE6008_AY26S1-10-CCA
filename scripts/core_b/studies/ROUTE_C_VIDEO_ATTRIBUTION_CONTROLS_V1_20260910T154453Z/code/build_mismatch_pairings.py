#!/usr/bin/env python3
"""Build Route C donor pairings from a supplied accepted development manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from apply_fusion_feature_interventions import cross_task_donors, pairing_summary, same_task_donors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--same-task-seed", type=int, default=20260911)
    parser.add_argument("--cross-task-seed", type=int, default=20260912)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    rows = [row for row in rows if row.get("split") in {"dev", "development", "development_validation", "validation"}]
    ids = np.asarray(sorted(int(row["episode_index"]) for row in rows), dtype=np.int64)
    task_map = {int(row["episode_index"]): int(row["original_task_index"]) for row in rows}
    tasks = np.asarray([task_map[int(value)] for value in ids], dtype=np.int64)
    same = same_task_donors(ids, tasks, seed=args.same_task_seed)
    cross = cross_task_donors(ids, tasks, seed=args.cross_task_seed)
    payload = {"same_task": pairing_summary(same, task_map), "cross_task": pairing_summary(cross, task_map), "same_task_seed": args.same_task_seed, "cross_task_seed": args.cross_task_seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

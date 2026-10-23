#!/usr/bin/env python3
"""Freeze grouped within-task paired bootstrap draws before Core B dev scoring."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
DEV_PREFIX = RUN / "replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
FINAL_POP = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol/CORE_B_FINAL_POPULATION_METADATA.csv"
TASKS = (119, 94, 90, 141, 230)
DRAWS = 5000
SEED = 2026101902


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def make_draws(split: str, clusters_by_task: dict[int, list[str]]) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(SEED)
    arrays = {}
    for task in TASKS:
        clusters = sorted(clusters_by_task[task])
        if not clusters:
            raise ValueError(f"task {task} has no source clusters in {split}")
        arrays[f"task_{task}"] = rng.integers(0, len(clusters), size=(DRAWS, len(clusters)), dtype=np.int32)
    metadata = {
        "split": split,
        "draws": DRAWS,
        "seed": SEED,
        "rng": "numpy.default_rng(PCG64)",
        "resampling_unit": "episode/source_cluster sampled with replacement within each fixed task",
        "task_order": list(TASKS),
        "clusters_by_task": {str(t): sorted(clusters_by_task[t]) for t in TASKS},
        "draw_arrays": {f"task_{t}": {"shape": list(arrays[f"task_{t}"].shape), "dtype": str(arrays[f"task_{t}"].dtype)} for t in TASKS},
        "metric_reduction": "anchor mean within episode; equal episode mean within task; equal macro mean across five fixed tasks",
        "conditions_share_draws": True,
        "scores_or_targets_read": False,
    }
    return arrays, metadata


def write(split: str, clusters_by_task: dict[int, list[str]], row_count: int) -> dict:
    arrays, meta = make_draws(split, clusters_by_task)
    prefix = OUT / "core_b" / f"CORE_B_{split.upper()}_BOOTSTRAP_DRAWS_V2"
    npz_path, json_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite locked bootstrap draws for {split}")
    temp = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.part")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, npz_path)
    meta.update({
        "schema_version": 1,
        "status": "LOCKED_BEFORE_CURRENT_PROFILE_DEV_SCORE" if split == "dev" else "LOCKED_BEFORE_FINAL_SCORE",
        "row_count": row_count,
        "episode_support_by_task": {str(t): len(clusters_by_task[t]) for t in TASKS},
        "draw_npz_path": str(npz_path),
        "draw_npz_sha256": sha(npz_path),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    atomic_json(json_path, meta)
    meta["draw_json_sha256"] = sha(json_path)
    # Write the digest back to a receipt sidecar; JSON self-hash is intentionally external.
    receipt_path = prefix.with_name(prefix.name + "_RECEIPT.json")
    atomic_json(receipt_path, {
        "status": meta["status"], "split": split, "draw_npz_sha256": sha(npz_path),
        "draw_metadata_sha256": sha(json_path), "draw_rows": DRAWS,
        "task_episode_support": meta["episode_support_by_task"],
        "scores_or_targets_read": False,
    })
    return {"split": split, "draw_npz_path": str(npz_path), "draw_npz_sha256": sha(npz_path),
            "draw_metadata_path": str(json_path), "draw_metadata_sha256": sha(json_path),
            "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "episode_support_by_task": meta["episode_support_by_task"]}


def main() -> int:
    with np.load(DEV_PREFIX, allow_pickle=False) as dev:
        if any(any(term in key.lower() for term in ("target", "future", "label")) for key in dev.files):
            raise ValueError("dev prefix contains an outcome/target key")
        dev_tasks = dev["task_id"].astype(np.int64)
        dev_groups = dev["source_cluster"].astype(str)
    dev_clusters = {t: sorted(set(dev_groups[dev_tasks == t].tolist())) for t in TASKS}

    final_rows = list(csv.DictReader(FINAL_POP.open(encoding="utf-8-sig", newline="")))
    if len(final_rows) != 162 or any(row["split"] != "final_test" for row in final_rows):
        raise ValueError("fixed final population metadata differs from the locked 162-anchor set")
    final_clusters = {t: sorted({row["source_group_key"] for row in final_rows if int(row["task_id"]) == t}) for t in TASKS}

    dev_result = write("dev", dev_clusters, len(dev_tasks))
    final_result = write("final", final_clusters, len(final_rows))
    print(json.dumps({"status": "PASS", "seed": SEED, "draws": DRAWS,
                      "dev": dev_result, "final": final_result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

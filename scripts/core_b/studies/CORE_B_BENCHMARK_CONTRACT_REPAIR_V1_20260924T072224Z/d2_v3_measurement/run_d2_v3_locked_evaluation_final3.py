"""Run D2_V3 with the D1-verified exact-video SHA map and locked matcher."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_d2_v3_panel as evaluator


ROOT = Path(__file__).resolve().parent
D1_SOURCE_GROUP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/results/D1_SOURCE_GROUP_QUERY_TABLE.csv'))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


matcher = json.loads((ROOT / "D2_V3_MATCHER_LOCKED.json").read_text())
runner_lock = json.loads((ROOT / "D2_V3_RUNNER_LOCKED_ATTEMPT_003.json").read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("attempt-three runner not locked before evaluation")
if runner_lock.get("runner_code_sha256") != sha256_file(Path(__file__)):
    raise RuntimeError("attempt-three runner source changed after lock")
if runner_lock.get("matcher_protocol_sha256") != sha256_file(ROOT / "D2_V3_MATCHER_LOCKED.json"):
    raise RuntimeError("matcher lock changed after attempt-three runner lock")
if runner_lock.get("dev_source_group_map_sha256") != sha256_file(D1_SOURCE_GROUP):
    raise RuntimeError("D1 exact-video source-group map changed after lock")
if matcher.get("evaluator_code_sha256") != sha256_file(ROOT / "evaluate_d2_v3_panel.py"):
    raise RuntimeError("core evaluator changed after matcher lock")
if matcher.get("threshold_protocol_sha256") != sha256_file(ROOT / "D2_V3_PROTOCOL_LOCKED.json"):
    raise RuntimeError("train threshold lock changed after matcher lock")


def dev_only_source_groups(split: str, episode_ids: set[int]):
    if split != "dev":
        raise AssertionError("the D2_V3 dev evaluator must not request another split")
    output = {}
    with D1_SOURCE_GROUP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_index"])
            if episode in episode_ids:
                value = {"task": int(row["task"]), "group": str(row["source_group_id"])}
                if episode in output and output[episode] != value:
                    raise AssertionError(f"conflicting exact-video group for episode {episode}")
                output[episode] = value
    missing = episode_ids - set(output)
    if missing or len(output) != 603:
        raise AssertionError(f"D1 exact-video group lookup is incomplete: missing={len(missing)}")
    return output


evaluator.load_split_groups = dev_only_source_groups
evaluator.SIGMA = np.asarray(
    json.loads((evaluator.V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64
)
old_v2 = json.loads((evaluator.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
evaluator.D2_V2_ON = {"Hand": float(old_v2["theta_on"][0]), "Arm": float(old_v2["theta_on"][1])}
evaluator.D2_V2_OFF = {"Hand": float(old_v2["theta_off"][0]), "Arm": float(old_v2["theta_off"][1])}

_write_csv = evaluator.write_csv


def write_union_csv(path: Path, rows, fields=None):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields or [])
    return _write_csv(path, rows, columns)


evaluator.write_csv = write_union_csv
evaluator.main()

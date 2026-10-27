"""Run the frozen D2_V3 evaluator with locked identity and import bindings."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import evaluate_d2_v3_panel as evaluator


ROOT = Path(__file__).resolve().parent
GROUP_MAP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/results/D1_SOURCE_GROUP_QUERY_TABLE.csv'))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


matcher = json.loads((ROOT / "D2_V3_MATCHER_LOCKED.json").read_text())
runner_lock = json.loads((ROOT / "D2_V3_RUNNER_LOCKED_ATTEMPT_004.json").read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("attempt-four runner is not locked")
if runner_lock.get("runner_code_sha256") != sha256_file(Path(__file__)):
    raise RuntimeError("runner source changed after attempt-four lock")
if runner_lock.get("matcher_protocol_sha256") != sha256_file(ROOT / "D2_V3_MATCHER_LOCKED.json"):
    raise RuntimeError("matcher protocol changed after lock")
if runner_lock.get("dev_source_group_map_sha256") != sha256_file(GROUP_MAP):
    raise RuntimeError("dev source-group map changed after lock")
if matcher.get("evaluator_code_sha256") != sha256_file(ROOT / "evaluate_d2_v3_panel.py"):
    raise RuntimeError("core event matcher changed after lock")
if matcher.get("threshold_protocol_sha256") != sha256_file(ROOT / "D2_V3_PROTOCOL_LOCKED.json"):
    raise RuntimeError("train threshold protocol changed after lock")


def dev_only_source_groups(split: str, episode_ids: set[int]):
    if split != "dev":
        raise AssertionError("only V3 primary-dev source groups may be loaded here")
    output = {}
    with GROUP_MAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_index"])
            if episode in episode_ids:
                current = {"task": int(row["task"]), "group": str(row["source_group_id"])}
                if episode in output and output[episode] != current:
                    raise AssertionError(f"conflicting source-group mapping for {episode}")
                output[episode] = current
    if episode_ids - set(output) or len(output) != 603:
        raise AssertionError("exact video SHA map must reconcile all 603 primary-dev episodes")
    return output


evaluator.Counter = Counter
evaluator.load_split_groups = dev_only_source_groups
evaluator.SIGMA = np.asarray(json.loads((evaluator.V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
old_v2 = json.loads((evaluator.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
evaluator.D2_V2_ON = {"Hand": float(old_v2["theta_on"][0]), "Arm": float(old_v2["theta_on"][1])}
evaluator.D2_V2_OFF = {"Hand": float(old_v2["theta_off"][0]), "Arm": float(old_v2["theta_off"][1])}
_write_csv = evaluator.write_csv


def write_union_csv(path: Path, rows, fields=None):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields or [])
    return _write_csv(path, rows, columns)


evaluator.write_csv = write_union_csv
evaluator.main()

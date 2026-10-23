"""Run the frozen D2_V3 evaluator with its explicit, hashed runtime fixes."""

from __future__ import annotations
import os

import ast
import csv
import hashlib
import json
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "evaluate_d2_v3_panel.py"
LOCK = ROOT / "D2_V3_MATCHER_LOCKED_ATTEMPT_002.json"
RUNNER_LOCK = ROOT / "D2_V3_RUNNER_LOCKED_ATTEMPT_005.json"
GROUP_MAP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/results/D1_SOURCE_GROUP_QUERY_TABLE.csv'))
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transformed_source(source: str) -> str:
    changes = (
        ("from collections import defaultdict", "from collections import Counter, defaultdict"),
        ('known_key = f"{definition.lower()}_known_{body}"', 'known_key = f"{definition.lower().replace(\'d2_\', \'\')}_known_{body}"'),
        ('armed_key = f"{definition.lower()}_armed_{body}"', 'armed_key = f"{definition.lower().replace(\'d2_\', \'\')}_armed_{body}"'),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise AssertionError(f"expected one locked compatibility site: {old}")
        source = source.replace(old, new, 1)
    return source


matcher = json.loads(LOCK.read_text())
runner_lock = json.loads(RUNNER_LOCK.read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("attempt-five runner not locked")
if runner_lock.get("runner_code_sha256") != sha256_file(Path(__file__)):
    raise RuntimeError("attempt-five runner source changed after lock")
if runner_lock.get("matcher_protocol_sha256") != sha256_file(LOCK):
    raise RuntimeError("attempt-five matcher lock changed after runner lock")
if runner_lock.get("dev_source_group_map_sha256") != sha256_file(GROUP_MAP):
    raise RuntimeError("attempt-five dev group map changed after lock")
if runner_lock.get("core_evaluator_sha256") != sha256_file(CORE):
    raise RuntimeError("core evaluator source changed after lock")
if matcher.get("evaluator_code_sha256") != sha256_file(CORE):
    raise RuntimeError("matcher lock core-source hash mismatch")
source = transformed_source(CORE.read_text(encoding="utf-8"))
ast.parse(source)
runtime_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
if runtime_sha != matcher.get("runtime_transformed_source_sha256") or runtime_sha != runner_lock.get("runtime_transformed_source_sha256"):
    raise RuntimeError("runtime transformed evaluator hash mismatch")
if matcher.get("threshold_protocol_sha256") != sha256_file(ROOT / "D2_V3_PROTOCOL_LOCKED.json"):
    raise RuntimeError("train threshold lock changed")

module = types.ModuleType("d2_v3_evaluator_locked")
module.__file__ = str(CORE)
module.__dict__["__name__"] = "d2_v3_evaluator_locked"
exec(compile(source, str(CORE), "exec"), module.__dict__)


def dev_only_source_groups(split: str, episode_ids: set[int]):
    if split != "dev":
        raise AssertionError("attempt-five runner may load only primary-dev source groups")
    output = {}
    with GROUP_MAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_index"])
            if episode in episode_ids:
                current = {"task": int(row["task"]), "group": str(row["source_group_id"])}
                if episode in output and output[episode] != current:
                    raise AssertionError(f"conflicting source-group identity for dev episode {episode}")
                output[episode] = current
    if episode_ids - set(output) or len(output) != 603:
        raise AssertionError("source-group map must cover the 603 primary-dev episodes exactly")
    return output


module.LOCK_PATH = LOCK
module.load_split_groups = dev_only_source_groups
module.SIGMA = np.asarray(json.loads((module.V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
legacy = json.loads((module.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
module.D2_V2_ON = {"Hand": float(legacy["theta_on"][0]), "Arm": float(legacy["theta_on"][1])}
module.D2_V2_OFF = {"Hand": float(legacy["theta_off"][0]), "Arm": float(legacy["theta_off"][1])}
write_csv = module.write_csv


def write_union_csv(path: Path, rows, fields=None):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields or [])
    return write_csv(path, rows, columns)


module.write_csv = write_union_csv
module.main()

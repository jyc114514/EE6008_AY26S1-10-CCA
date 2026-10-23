"""Execute D2_V3 after attempt-six code/hash validation and train locks."""

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
MATCHER = ROOT / "D2_V3_MATCHER_LOCKED_ATTEMPT_003.json"
RUNNER_LOCK = ROOT / "D2_V3_RUNNER_LOCKED_ATTEMPT_006.json"
GROUP_MAP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/results/D1_SOURCE_GROUP_QUERY_TABLE.csv'))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transformed(source: str) -> str:
    replacements = (
        ("from collections import defaultdict", "from collections import Counter, defaultdict"),
        ('known_key = f"{definition.lower()}_known_{body}"', 'known_key = f"{definition.lower().replace(\'d2_\', \'\')}_known_{body}"'),
        ('armed_key = f"{definition.lower()}_armed_{body}"', 'armed_key = f"{definition.lower().replace(\'d2_\', \'\')}_armed_{body}"'),
        ("all_events = panel_events[group]", "all_events = panel_truth_by_body_group[(group, body)]"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"locked compatibility source site changed: {old}")
        source = source.replace(old, new, 1)
    return source


matcher = json.loads(MATCHER.read_text())
runner_lock = json.loads(RUNNER_LOCK.read_text())
if runner_lock.get("status") != "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION":
    raise RuntimeError("attempt-six runner not locked")
if runner_lock.get("runner_code_sha256") != file_hash(Path(__file__)):
    raise RuntimeError("attempt-six runner hash mismatch")
if runner_lock.get("matcher_protocol_sha256") != file_hash(MATCHER):
    raise RuntimeError("attempt-six matcher lock hash mismatch")
if runner_lock.get("dev_source_group_map_sha256") != file_hash(GROUP_MAP):
    raise RuntimeError("dev source group map hash mismatch")
if matcher.get("evaluator_code_sha256") != file_hash(CORE):
    raise RuntimeError("core evaluator source hash mismatch")
if matcher.get("threshold_protocol_sha256") != file_hash(ROOT / "D2_V3_PROTOCOL_LOCKED.json"):
    raise RuntimeError("train threshold protocol hash mismatch")
source = transformed(CORE.read_text(encoding="utf-8"))
ast.parse(source)
runtime_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
if runtime_hash != matcher.get("runtime_transformed_source_sha256") or runtime_hash != runner_lock.get("runtime_transformed_source_sha256"):
    raise RuntimeError("transformed evaluator hash mismatch")

module = types.ModuleType("d2_v3_evaluator_attempt6")
module.__file__ = str(CORE)
exec(compile(source, str(CORE), "exec"), module.__dict__)


def dev_only_source_groups(split: str, episode_ids: set[int]):
    if split != "dev":
        raise AssertionError("D2_V3 evaluator only permits primary-dev source groups")
    output = {}
    with GROUP_MAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_index"])
            if episode in episode_ids:
                current = {"task": int(row["task"]), "group": str(row["source_group_id"])}
                if episode in output and output[episode] != current:
                    raise AssertionError(f"conflicting exact-video group for {episode}")
                output[episode] = current
    if episode_ids - set(output) or len(output) != 603:
        raise AssertionError("D1 exact video group map must cover 603 primary dev episodes")
    return output


module.LOCK_PATH = MATCHER
module.load_split_groups = dev_only_source_groups
module.SIGMA = np.asarray(json.loads((module.V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
old_v2 = json.loads((module.RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
module.D2_V2_ON = {"Hand": float(old_v2["theta_on"][0]), "Arm": float(old_v2["theta_on"][1])}
module.D2_V2_OFF = {"Hand": float(old_v2["theta_off"][0]), "Arm": float(old_v2["theta_off"][1])}
original_write_csv = module.write_csv


def write_union_csv(path: Path, rows, fields=None):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields or [])
    return original_write_csv(path, rows, columns)


module.write_csv = write_union_csv
module.main()

#!/usr/bin/env python3
"""Record the metric-free Core A scorer correction as freeze revision 4."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parents[2] / "outputs" / 'project-team' / ROOT.name
FREEZE_PATH = ROOT / "core_a/FREEZE_RECEIPT.json"
LOG_PATH = ROOT / "core_a/FREEZE_REVISION_LOG.json"
AMENDMENT_PATH = ROOT / "core_a/PROTOCOL_AMENDMENT_V2.json"
QA_PATH = ROOT / "qa/SEMANTIC_SCORER_AND_CAUSAL_TESTS.json"
ATTEMPT01_PATH = ROOT / "core_a/CORE_A_SCORE_ATTEMPT_01.json"
ATTEMPT02_PATH = ROOT / "core_a/CORE_A_SCORE_ATTEMPT_02.json"
EXPOSURE_PATH = ROOT / "core_a/SCORE_EXPOSURE.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, value) -> None:
    temp = path.with_name("." + path.name + "." + str(os.getpid()) + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


if ATTEMPT02_PATH.exists():
    raise SystemExit("attempt 02 already exists; refusing to revise after a score launch")
if read(ATTEMPT01_PATH).get("status") != "FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC":
    raise SystemExit("attempt 01 does not match the documented metric-free failure")
if read(ATTEMPT01_PATH).get("metric_values_computed") != 0:
    raise SystemExit("attempt 01 reports a metric; revision is not eligible")
if read(EXPOSURE_PATH).get("metrics_computed") is not False:
    raise SystemExit("exposure record reports a metric; revision is not eligible")
if read(FREEZE_PATH).get("revision") != 3 or read(FREEZE_PATH).get("status") != "LOCKED_BEFORE_V2_SCORE":
    raise SystemExit("expected active metric-free freeze revision 3")
for rel in (
    "core_a/FULL602_PER_SEED.csv",
    "core_a/FULL602_SUMMARY.csv",
    "core_a/PAIRED_CONTRASTS_AND_CI.csv",
    "core_a/BOOTSTRAP_DRAW_MANIFEST.json",
):
    if (ROOT / rel).exists():
        raise SystemExit(f"score output already exists: {rel}")
if (RUN / "core_a/A-FULL602_BOOTSTRAP_DRAWS.npy").exists():
    raise SystemExit("bootstrap draw artifact already exists")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
paths = [FREEZE_PATH, LOG_PATH, AMENDMENT_PATH, QA_PATH]
backups = []
for path in paths:
    backup = path.with_name(path.name + ".pre-revision-4-" + stamp + ".bak")
    if backup.exists():
        raise SystemExit(f"backup already exists: {backup}")
    shutil.copy2(path, backup)
    backups.append(str(backup))

now = utc()
scorer_path = ROOT / "code/score_core_a_v2.py"
test_path = ROOT / "code/test_core_a_v2_scorer.py"
scorer_sha = sha(scorer_path)
test_sha = sha(test_path)
script_sha = sha(Path(__file__))
attempt01 = read(ATTEMPT01_PATH)
exposure = read(EXPOSURE_PATH)

amendment = read(AMENDMENT_PATH)
revision = {
    "revision": 4,
    "updated_utc": now,
    "trigger": "Final code-path review found the bootstrap draw manifest was assigned to itself, so JSON serialization would fail after metric computation; the post-label read but pre-metric attempt boundary was still intact.",
    "fix": "Collect one JSON-serializable draw-manifest entry per estimand; preserve common-class per-seed rows when class sets differ; record Macro-F1 labels by estimand; separate explicit score-label value counts from label-bearing metadata row counts.",
    "scoring_code_sha256": scorer_sha,
    "test_file_sha256": test_sha,
    "revision_script_sha256": script_sha,
    "pytest_tests_passed": 7,
    "prediction_preflight": "PASS",
    "attempt01_labels_materialized": 602,
    "attempt01_metrics_computed": 0,
    "attempt02_started": False,
    "scientific_contract_changed": False,
    "population_changed": False,
    "metric_changed": False,
    "prediction_or_checkpoint_changed": False,
    "tie_rule_changed": False,
}
amendment.setdefault("engineering_revision_history", []).append(revision)
amendment["engineering_revision"] = revision
amendment["scoring_code_sha256"] = scorer_sha
amendment["metrics_computed"] = False
amendment.setdefault("score_exposure", {})["first_v2_label_read_utc"] = attempt01["started_utc"]
amendment["score_exposure"]["first_v2_metric_computed_utc"] = None
amendment["failed_metric_free_attempts"] = [{
    "attempt_id": "CORE_A_SCORE_ATTEMPT_01",
    "started_utc": attempt01["started_utc"],
    "labels_materialized": 602,
    "metrics_computed": 0,
    "failure": attempt01["failure"],
}]
write_atomic(AMENDMENT_PATH, amendment)

qa = read(QA_PATH)
qa["created_utc"] = now
qa["final_label_manifest_read_for_scoring"] = True
qa["metric_values_computed"] = False
qa["core_a_scorer_tests"] = {
    "command": "envs/jyc/bin/python -m pytest -q code/test_core_a_v2_scorer.py",
    "details": [
        "stable top-k tie rule",
        "unmapped true class behavior and coverage identity",
        "common ranking retains all original output columns",
        "fixed-label zero-support macro-F1 against sklearn",
        "C_union confusion distinct from F1 denominator",
        "identical model/final sets make common estimand non-distinct",
        "bootstrap manifest is separate per estimand and JSON serializable",
    ],
    "status": "PASS",
    "test_file_sha256": test_sha,
    "tests": 7,
}
qa["core_a_scorer_code_sha256"] = scorer_sha
qa.setdefault("failed_attempts", []).append({
    "kind": "core_a_score_attempt_01",
    "symptom": "Freeze schema accessor raised KeyError C_final after materializing the same frozen 602 labels.",
    "resolution": "Preserved the failed attempt, added an attempt-02 receipt path, corrected frozen field access, and reran metric-free tests before relocking.",
    "labels_materialized": 602,
    "metric_values_computed": 0,
})
write_atomic(QA_PATH, qa)

freeze = read(FREEZE_PATH)
prior_freeze_sha = sha(FREEZE_PATH)
freeze["revision"] = 4
freeze["created_utc"] = now
freeze["supersedes_freeze_receipt_sha256"] = prior_freeze_sha
freeze["revision_reason"] = "After attempt 01 read the exact 602 frozen labels and failed before metrics, final code-path review found a self-referential bootstrap manifest assignment. Revision 4 fixes manifest aggregation and regression coverage only; no V2 metric had been computed before this lock."
freeze["code_sha256"]["score_core_a_v2.py"] = scorer_sha
freeze["code_sha256"]["test_core_a_v2_scorer.py"] = test_sha
freeze["code_sha256"]["revise_core_a_freeze_after_manifest_fix.py"] = script_sha
freeze["qa_test_status"] = {
    "focused_tests": 7,
    "pytest_status": "PASS",
    "prediction_preflight": "PASS",
    "metric_values_computed": False,
}
freeze["first_v2_label_read_utc"] = attempt01["started_utc"]
freeze["first_v2_metric_computed_utc"] = None
freeze["metrics_computed"] = False
freeze["protocol_amendment_sha256"] = sha(AMENDMENT_PATH)
freeze["v2_metric_free_failed_attempt"] = {
    "attempt_path": str(ATTEMPT01_PATH),
    "attempt_status": attempt01["status"],
    "labels_materialized": 602,
    "metric_values_computed": 0,
    "freeze_revision_at_attempt": attempt01["freeze_revision"],
    "freeze_sha256_at_attempt": attempt01["freeze_sha256"],
}
write_atomic(FREEZE_PATH, freeze)
new_freeze_sha = sha(FREEZE_PATH)

log = read(LOG_PATH)
log["revision"] = 4
log["current_revision"] = 4
log["current_freeze_sha256"] = new_freeze_sha
log["label_manifest_read_for_scoring_before_revision"] = True
log["metrics_computed_before_revision"] = False
log["reason"] = revision["trigger"]
log["backups"] = list(log.get("backups", [])) + backups
log.setdefault("revisions", []).append({
    "revision": 4,
    "created_utc": now,
    "status": freeze["status"],
    "freeze_sha256": new_freeze_sha,
    "supersedes_freeze_sha256": prior_freeze_sha,
    "after_metric_free_label_read": True,
    "labels_materialized": 602,
    "metrics_computed": 0,
    "attempt02_started": False,
    "reason": revision["trigger"],
    "scorer_sha256": scorer_sha,
    "test_sha256": test_sha,
})
write_atomic(LOG_PATH, log)
print(json.dumps({
    "status": "LOCKED_BEFORE_V2_SCORE",
    "revision": 4,
    "freeze_sha256": new_freeze_sha,
    "scorer_sha256": scorer_sha,
    "test_sha256": test_sha,
    "attempt01_labels": 602,
    "metrics_before_revision": 0,
    "attempt02_started": False,
    "backups": backups,
}, ensure_ascii=False, indent=2))

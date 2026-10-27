"""Reconcile the live Core B/V3 evidence into the benchmark root."""

from __future__ import annotations

import json
import os
from pathlib import Path

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, atomic_write_json, read_json, rel, sha256_file, utc_now, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
HISTORY = REPO_ROOT / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
A_ROOT = REPO_ROOT / "reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z"
B_ROOT = REPO_ROOT / "reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z"
RECOVERY = PROJECT_ROOT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
AUDIT = REPO_ROOT / "reports/core_b/PROTOTYPE_DIAGNOSIS_ROUTER_STUDY_V1_20260917T060236Z"
FREEZE = REPO_ROOT / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z"
V2 = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"


KNOWN = {
    "project_root": PROJECT_ROOT,
    "repo_root": REPO_ROOT,
    "v3_root": V3,
    "history_root": HISTORY,
    "history_replication_a_root": A_ROOT,
    "adaptation_b_root": B_ROOT,
    "recovery_root": RECOVERY,
    "router_audit_root": AUDIT,
    "freeze_root": FREEZE,
    "v2_root": V2,
    "python_bin": PROJECT_ROOT / "envs/jyc/bin/python",
    "gpu_watcher": REPO_ROOT / "tools/gpu_lease_watcher.py",
    "gpu_policy": REPO_ROOT / "docs/operations/GPU_WATCHER_POLICY.md",
    "source_vjepa": PROJECT_ROOT / "repo/third_party/vjepa2",
    "checkpoint": PROJECT_ROOT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
}


def tree_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": rel(path),
        "exists": True,
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "top_level": sorted(item.name for item in path.iterdir())[:40],
    }


def v3_training_status() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in sorted((V3 / "runs").iterdir() if (V3 / "runs").exists() else []):
        if not run.is_dir():
            continue
        candidates = []
        for variant in ("formal", "formal_pilot_retry1"):
            root = run / variant
            complete = root / "RUN_COMPLETE.json"
            status = root / "RUN_STATUS.json"
            if complete.exists():
                try:
                    data = read_json(complete)
                except Exception as exc:  # pragma: no cover - evidence preservation
                    data = {"status": "JSON_ERROR", "error": str(exc)}
                candidates.append({"variant": variant, "kind": "complete", "data": data})
            elif status.exists():
                try:
                    data = read_json(status)
                except Exception as exc:  # pragma: no cover
                    data = {"status": "JSON_ERROR", "error": str(exc)}
                candidates.append({"variant": variant, "kind": "status", "data": data})
        accepted = next((item for item in candidates if item["kind"] == "complete" and item["data"].get("status") == "PASS"), None)
        running = next((item for item in candidates if item["data"].get("status") == "running"), None)
        failed = [item for item in candidates if item["data"].get("status") == "failed"]
        if accepted:
            state = "training_accepted"
            source = accepted
        elif running:
            state = "training_running"
            source = running
        elif failed:
            state = "training_failed_or_incomplete"
            source = failed[-1]
        elif candidates:
            state = "training_status_unknown"
            source = candidates[-1]
        else:
            state = "missing"
            source = {}
        data = source.get("data", {})
        rows.append({
            "run_id": run.name,
            "state": state,
            "variant": source.get("variant", ""),
            "status": data.get("status", ""),
            "seed": data.get("seed", ""),
            "optimizer_steps": data.get("optimizer_steps", data.get("last_optimizer_steps", "")),
            "last_completed_epoch": data.get("last_completed_epoch", ""),
            "final_test_access": data.get("final_test_access", 0),
        })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"rows": rows, "counts": counts, "expected_matrix": 36}


def prediction_status() -> dict[str, object]:
    prediction_files = [item for item in (V3 / "predictions").rglob("*") if item.is_file()] if (V3 / "predictions").exists() else []
    recovery = {}
    if (RECOVERY / "PREDICTION_ACCEPTANCE.json").exists():
        recovery = read_json(RECOVERY / "PREDICTION_ACCEPTANCE.json")
    return {
        "v3_prediction_file_count": len(prediction_files),
        "v3_prediction_paths": [rel(item) for item in prediction_files[:20]],
        "v3_predictions_accepted": "0/36" if not prediction_files else "unknown_until_bundle_acceptance",
        "recovery_prediction_acceptance": recovery.get("accepted", "UNKNOWN"),
        "recovery_final_test_access": recovery.get("final_test_access", "UNKNOWN"),
    }


def main() -> int:
    BENCH_ROOT.joinpath("context").mkdir(parents=True, exist_ok=True)
    resolved: dict[str, object] = {"generated_utc": utc_now(), "cwd": os.getcwd(), "paths": {}}
    for key, path in KNOWN.items():
        item: dict[str, object] = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        if path.exists() and path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
            item["sha256"] = sha256_file(path)
            item["bytes"] = path.stat().st_size
        resolved["paths"][key] = item
    resolved["git_surface"] = {
        "repo_root_has_dot_git": (REPO_ROOT / ".git").exists(),
        "evidence_limit": "git command unavailable in current shell; no commit/branch claim made",
    }
    atomic_write_json(BENCH_ROOT / "context/PATHS_RESOLVED.json", resolved)

    train = v3_training_status()
    prediction = prediction_status()
    v3_meta = {
        "generated_utc": utc_now(),
        "protocol": read_json(V3 / "PROTOCOL_LOCKED.json") if (V3 / "PROTOCOL_LOCKED.json").exists() else {},
        "resource_estimate_actual": read_json(V3 / "RESOURCE_ESTIMATE_ACTUAL.json") if (V3 / "RESOURCE_ESTIMATE_ACTUAL.json").exists() else {},
        "cache_manifest": read_json(V3 / "cache/CACHE_MANIFEST.json") if (V3 / "cache/CACHE_MANIFEST.json").exists() else {},
        "training": train,
        "prediction": prediction,
        "data": tree_summary(V3 / "data"),
        "final_test_access": 0,
    }
    atomic_write_json(BENCH_ROOT / "context/V3_STATUS_SNAPSHOT.json", v3_meta)

    source_rows = []
    for key, path in KNOWN.items():
        if key.endswith("_root") or key in {"v3_root", "history_root", "recovery_root"}:
            source_rows.append({**tree_summary(path), "source_key": key})
    write_csv(BENCH_ROOT / "context/SOURCE_ROOT_SUMMARY.csv", source_rows, ["source_key", "path", "exists", "file_count", "bytes", "top_level"])

    accepted = train["counts"].get("training_accepted", 0)
    running = train["counts"].get("training_running", 0)
    conflicts = [
        "Project AGENTS.md describes repo/ee6008-humanoid-vjepa as an empty placeholder, but live files exist under src/tests/tools/reports; direct filesystem evidence wins and the repo has no .git.",
        f"Pasted prior status said remaining formal seeds=24; live V3 status is {accepted}/36 training accepted and {running}/36 still running.",
        "V3 accepted temporal cache reports 20143 rows including supplementary coverage, while V3_MAIN_INPUTS.npz currently contains 20079 rows; these are different scopes, not silently merged.",
        "V3 predictions directory currently has no sealed prediction bundle, so training completion is not prediction completion.",
        "GPU2 lease file records an old PID that is no longer present; it is recorded as a stale lease signal and is not deleted by this benchmark.",
    ]
    (BENCH_ROOT / "context/CONTEXT_CONFLICTS.md").write_text("# Context conflicts and unresolved evidence\n\n" + "\n".join(f"- {item}" for item in conflicts) + "\n", encoding="utf-8")

    context = f"""# Core B context restored\n\nGenerated: `{utc_now()}`\n\n## Scope\n\nThis root is an independent development benchmark. It reads historical Core B artifacts and the V3 train/development contract, writes only under this benchmark root, never reads final/held-out inputs or targets, and does not alter the original V3 producer. New results are `exploratory_development`.\n\n## Live V3 state\n\n- Protocol: `{v3_meta['protocol'].get('protocol_id', 'UNKNOWN')}`, 6 cells × 2 horizons × 3 seeds = 36 locked rows; encoder update `{v3_meta['protocol'].get('encoder_updated', 'UNKNOWN')}`; final-test access `0`.\n- Training status: `{accepted}/36` accepted with a PASS completion marker; `{running}/36` running at snapshot; failed/retry snapshots are retained separately. See `context/V3_STATUS_SNAPSHOT.json`.\n- Accepted temporal cache: `{v3_meta['cache_manifest'].get('row_count', 'UNKNOWN')}` rows, shape `{v3_meta['cache_manifest'].get('shape', 'UNKNOWN')}`, status `{v3_meta['cache_manifest'].get('status', 'UNKNOWN')}`.\n- Sealed V3 prediction bundles: `{prediction['v3_predictions_accepted']}` at this snapshot. A training marker is not promoted to a prediction marker.\n\n## Historical evidence recovered\n\n- Recovery branch: `{prediction['recovery_prediction_acceptance']}` development predictions; fixed six-cell checkpoint replay is reconstructable with caveat; candidate gate remains FAIL; no new training and final-test access `0`. Source: `{rel(RECOVERY / 'FINAL_STATUS.ini')}`, `{rel(RECOVERY / 'PREDICTION_ACCEPTANCE.json')}`.\n- V3 data contract: primary train/dev arrays and train-only normalizers are under `{rel(V3 / 'data')}`; the existing model has 16 temporal tokens × 768 channels and Full-28 native command output. Source: `{rel(V3 / 'MODEL_LOSS_CONTRACT.md')}`, `{rel(V3 / 'data/V3_NORMALIZERS.json')}`.\n- Existing history, replication, adaptation, router, freeze, and V2 roots were found and registered. They are not collapsed into one leaderboard; incompatible lineage remains explicitly distinct.\n\n## Unknowns and boundaries\n\n- V3 predictions are pending; D1/D2 model contrasts that require V3 predictions remain awaiting upstream.\n- The dense spatial D3 cache cannot be reconstructed from the existing temporal mean cache; it requires a fresh frozen-encoder forward after the shared GPU slots free.\n- Physical units/URDF/FK/Cartesian semantics remain unresolved; all metrics stay in source-native command coordinates.\n- The project has no `.git` at the active V3 repo root and the shell lacks `git`; no commit or branch claim is made.\n\n## Evidence map\n\n- Resolved paths and hashes: `context/PATHS_RESOLVED.json`\n- V3 reconciliation: `context/V3_STATUS_SNAPSHOT.json`\n- Conflicts: `context/CONTEXT_CONFLICTS.md`\n- Graphify code map: `{rel(REPO_ROOT / 'graphify-out/GRAPH_REPORT.md')}`\n"""
    (BENCH_ROOT / "context/CONTEXT_RESTORED.zh-CN.md").write_text(context, encoding="utf-8")

    state = {
        "benchmark_status": "awaiting_upstream" if running else "partial",
        "phase": "CONTEXT_RESTORED",
        "updated_utc": utc_now(),
        "context_restored": True,
        "inventory_t0": "not_started",
        "v3_training_accepted": f"{accepted}/36",
        "v3_predictions_accepted": prediction["v3_predictions_accepted"],
        "d1_status": "not_started",
        "d2_status": "not_started",
        "d3_status": "awaiting_shared_gpu_slots",
        "d4_status": "not_started",
        "d5_status": "not_started",
        "final_test_access": 0,
        "production_root_writes": 0,
        "unresolved_dependencies": ["V3 producer must finish the two running rows and seal predictions before full common/extended evaluation"],
    }
    atomic_write_json(BENCH_ROOT / "BENCHMARK_STATE.json", state)
    (BENCH_ROOT / "JOB_LEDGER.jsonl").write_text(json.dumps({"utc": utc_now(), "job_id": "CONTEXT-RESTORE", "status": "complete", "output_root": str(BENCH_ROOT), "gpu_hours": 0, "final_test_access": 0}, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(BENCH_ROOT / "DEPENDENCIES.csv", [{"dependency_id": "V3_UPSTREAM", "status": "running" if running else "sealed", "accepted": f"{accepted}/36", "predictions": prediction["v3_predictions_accepted"], "source": rel(V3)}], ["dependency_id", "status", "accepted", "predictions", "source"])
    (BENCH_ROOT / "NEXT_ACTIONS.md").write_text("# Next actions\n\n1. Complete CPU inventory and lock D1/D2/D3 protocols.\n2. Run D1 train-bank/nearest-neighbor CPU benchmark and D2 event registry/sparse baseline.\n3. Prepare D4/D5 interfaces and synthetic tests.\n4. Wait for the two live V3 children before requesting a benchmark GPU lease.\n5. Reconcile V3 prediction bundles, then run dense/model extensions.\n", encoding="utf-8")
    print(json.dumps({"status": state["benchmark_status"], "v3_training_accepted": f"{accepted}/36", "v3_training_running": running, "v3_predictions": prediction["v3_predictions_accepted"], "bench_root": str(BENCH_ROOT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

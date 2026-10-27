"""Build a broad Core B source/artifact/model inventory without touching sources."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, read_csv, read_json, rel, sha256_file, utc_now, write_csv


CORE_B = REPO_ROOT / "reports/core_b"
ARCHIVE_B = PROJECT_ROOT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
ARCHIVE_HARVEST = PROJECT_ROOT / "research_archive/CORE_B_ADAPTATION_HARVEST_20260919T053154Z"
FORMAL_ARCHIVE = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b"
V3 = CORE_B / "CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def all_roots() -> list[Path]:
    roots = sorted(item for item in CORE_B.iterdir() if item.is_dir())
    for path in (ARCHIVE_B, ARCHIVE_HARVEST, FORMAL_ARCHIVE / "benchmark"):
        if path.exists():
            roots.append(path)
    return roots


def artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if "prediction" in name or "forecast" in name or "pred" in name:
        return "prediction_or_forecast"
    if "checkpoint" in name or path.suffix in {".pt", ".pth", ".ckpt"}:
        return "checkpoint_or_weight"
    if "cache" in name or "/cache/" in str(path).lower() or path.suffix in {".npy", ".npz"}:
        return "feature_or_cache"
    if any(token in name for token in ("fail", "error", "retry", "invalid", "oom", "stderr", "stdout", "log")):
        return "failure_or_execution_log"
    if name.endswith((".json", ".csv", ".md", ".ini", ".yaml", ".yml")):
        return "contract_or_metadata"
    return "other"


def root_summary(root: Path) -> dict[str, object]:
    files = [item for item in root.rglob("*") if item.is_file()]
    return {
        "source_root": str(root),
        "source_root_relative": rel(root),
        "exists": root.exists(),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "checkpoint_count": sum(artifact_kind(item) == "checkpoint_or_weight" for item in files),
        "prediction_count": sum(artifact_kind(item) == "prediction_or_forecast" for item in files),
        "cache_count": sum(artifact_kind(item) == "feature_or_cache" for item in files),
        "failure_record_count": sum(artifact_kind(item) == "failure_or_execution_log" for item in files),
        "final_test_access": 0,
    }


MODEL_FIELDS = [
    "model_id", "family", "experiment_root", "scientific_role", "seed", "training_status",
    "checkpoint_path", "checkpoint_hash", "selected_checkpoint_rule", "backbone_adapter_identity",
    "code_config_hash", "trained_episodes_manifest_hash", "input_schema", "history_span",
    "observation_anchor_rule", "video_preprocess_feature_identity", "native_horizon", "output_dims",
    "arm_hand_mapping", "full28_composition", "state_video_target_normalizer_identity",
    "prediction_path", "population_anchor_ids", "native_score_source", "replay_status",
    "old_candidate_gate", "current_owner", "eligibility_reason",
]


def row_base(model_id: str, family: str, root: Path, role: str) -> dict[str, object]:
    return {field: "UNKNOWN" for field in MODEL_FIELDS} | {
        "model_id": model_id, "family": family, "experiment_root": rel(root),
        "scientific_role": role, "current_owner": "benchmark_inventory", "final_test_access": 0,
    }


def parse_v3(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for item in rows:
        row = row_base(item.get("run_id", "UNKNOWN"), "V3_MULTI_ANCHOR", V3, "formal_v3_model")
        run_id = item.get("run_id", "")
        run_root = V3 / "runs" / run_id
        variant = "formal_pilot_retry1" if (run_root / "formal_pilot_retry1/RUN_COMPLETE.json").exists() else "formal"
        checkpoint = run_root / variant / "checkpoint_last.pt"
        status = "training_accepted" if (run_root / variant / "RUN_COMPLETE.json").exists() else "training_status_unknown"
        prediction_marker = V3 / "predictions" / run_id / "PREDICTION_EXPORT_COMPLETE.json"
        prediction_ok = prediction_marker.exists() and read_json(prediction_marker).get("status") == "PASS"
        row.update({
            "seed": item.get("seed", "UNKNOWN"), "native_horizon": item.get("horizon_steps", "UNKNOWN"),
            "training_status": status, "checkpoint_path": rel(checkpoint) if checkpoint.exists() else "missing",
            "checkpoint_hash": sha256_file(checkpoint) if checkpoint.exists() else "missing",
            "selected_checkpoint_rule": "epoch40_last", "backbone_adapter_identity": "frozen V-JEPA2.1-B; encoder_updated=false",
            "input_schema": "history32x71 + video16x768 + anchor28", "output_dims": "Full28",
            "arm_hand_mapping": "Arm14=14:28; Hand14=0:14 pending direct schema audit",
            "full28_composition": "joint direct residual", "state_video_target_normalizer_identity": "V3 train-only normalizers",
            "prediction_path": rel(prediction_marker) if prediction_ok else "missing: V3 predictions directory has no sealed bundle",
            "population_anchor_ids": "V3 primary train/dev manifest", "replay_status": "replayable_checkpoint if sealed",
            "old_candidate_gate": "not an old candidate gate", "eligibility_reason": "sealed prediction bundle accepted" if prediction_ok else ("awaiting prediction bundle" if status == "training_accepted" else "training status not accepted"),
        })
        output.append(row)
    return output


def parse_generic_matrix(root: Path, matrix: Path, family: str) -> list[dict[str, object]]:
    output = []
    try:
        rows = read_csv(matrix)
    except Exception:
        return output
    for index, item in enumerate(rows):
        model_id = item.get("run_id") or item.get("cell") or item.get("model") or f"{family}:{index:03d}"
        role = "historical_matrix_cell"
        row = row_base(model_id, family, root, role)
        row.update({
            "seed": item.get("seed", "UNKNOWN"), "training_status": item.get("status", item.get("training_status", "UNKNOWN")),
            "native_horizon": item.get("horizon_steps", item.get("horizon", "UNKNOWN")),
            "input_schema": "; ".join(f"{k}={v}" for k, v in item.items() if k in {"state_input", "video_input", "video_mode", "adaptation"}),
            "output_dims": "Full28 or family-specific; see source matrix", "population_anchor_ids": "source matrix population",
            "prediction_path": "source root; exact sealed bundle requires marker audit",
            "eligibility_reason": "historical cell retained; common scoring only if source prediction/normalizer/IDs pass",
        })
        output.append(row)
    return output


def add_baselines() -> list[dict[str, object]]:
    rows = []
    for model_id, role in (("PERSISTENCE", "no_training_baseline"), ("FIXED_HYBRID", "legal_hybrid_baseline"), ("COMMAND_SLOPE", "historical_baseline"), ("STATE_ONLY_BASELINE", "no_video_baseline")):
        row = row_base(model_id, "BASELINES", V3, role)
        row.update({"training_status": "not_applicable", "prediction_path": "constructed only after locked evaluator", "eligibility_reason": "baseline; no checkpoint ranking"})
        rows.append(row)
    return rows


def main() -> int:
    roots = all_roots()
    write_csv(BENCH_ROOT / "inventory/SOURCE_REGISTRY.csv", [root_summary(root) for root in roots], ["source_root", "source_root_relative", "exists", "file_count", "bytes", "checkpoint_count", "prediction_count", "cache_count", "failure_record_count", "final_test_access"])

    artifact_rows = []
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            size = path.stat().st_size
            artifact_rows.append({
                "artifact_path": rel(path), "source_root": rel(root), "artifact_kind": artifact_kind(path),
                "bytes": size, "sha256": sha256_file(path) if size <= 1024 * 1024 and path.suffix in {".json", ".csv", ".md", ".ini", ".yaml", ".yml"} else "not_hashed_or_large_file",
                "snapshot_or_failure_marker": int(any(token in path.name.lower() for token in ("retry", "fail", "invalid", "oom", "smoke", "stage0"))),
                "final_test_access": 0,
            })
    write_csv(BENCH_ROOT / "inventory/ARTIFACT_INVENTORY.csv", artifact_rows, ["artifact_path", "source_root", "artifact_kind", "bytes", "sha256", "snapshot_or_failure_marker", "final_test_access"])

    model_rows: list[dict[str, object]] = []
    if (V3 / "MATRIX_36.csv").exists():
        model_rows.extend(parse_v3(read_csv(V3 / "MATRIX_36.csv")))
    for root, filename, family in [
        (CORE_B / "HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z", "MATRIX.csv", "HISTORY_VIDEO_2X2"),
        (CORE_B / "ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z", "B_MATRIX.csv", "ENCODER_ADAPTATION_B"),
        (CORE_B / "HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z", "MATRIX.csv", "HAND_DCT_DUAL_LOSS"),
        (CORE_B / "HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z", "MATRIX.csv", "HAND_EXPERT_SELECTOR"),
        (CORE_B / "HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z", "MATRIX_RESULTS.csv", "HAND_EVENT_FALLBACK"),
        (CORE_B / "HAND_TEMPORAL_FIDELITY_V1_20260914T152845Z", "MATRIX.csv", "HAND_TEMPORAL_FIDELITY"),
        (CORE_B / "JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z", "JY02_MATRIX.csv", "JY02_RESIDUAL"),
        (CORE_B / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z", "MATRIX.csv", "GATED_RESIDUAL"),
        (CORE_B / "ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z", "ROUTE_C_MATRIX.csv", "ROUTE_C"),
        (CORE_B / "SEL_U_PROSPECTIVE_CONFIRMATION_V1_20260914T142514Z", "MATRIX.csv", "SEL_U"),
        (CORE_B / "B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z", "B2_B3_MATRIX.csv", "FORMAL_B2_B3"),
    ]:
        path = root / filename
        if path.exists():
            model_rows.extend(parse_generic_matrix(root, path, family))
    if (ARCHIVE_B / "PREDICTION_ACCEPTANCE.json").exists():
        acceptance = read_json(ARCHIVE_B / "PREDICTION_ACCEPTANCE.json")
        for item in acceptance.get("rows", []):
            row = row_base(item.get("cell", "UNKNOWN"), "RECOVERY_B", ARCHIVE_B, "recovered_development_model")
            row.update({"seed": "20260944", "training_status": "checkpoint_reconstructable", "checkpoint_path": "recovery source checkpoint", "checkpoint_hash": item.get("checkpoint_sha256", "UNKNOWN"), "prediction_path": "recovery sealed development bundle", "population_anchor_ids": "canonical dev 614", "native_horizon": "30", "output_dims": "Hand14 + Full28", "replay_status": "fresh_process replay accepted for representative FRZ/LORA/DENSE", "old_candidate_gate": "FAIL", "eligibility_reason": "accepted development prediction; no final promotion"})
            model_rows.append(row)
    model_rows.extend(add_baselines())

    # Ensure unique IDs without collapsing duplicate snapshots into one method.
    seen: dict[str, int] = {}
    for row in model_rows:
        key = str(row["model_id"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            row["model_id"] = f"{key}__{seen[key]}"
    write_csv(BENCH_ROOT / "inventory/MODEL_REGISTRY.csv", model_rows, MODEL_FIELDS + ["final_test_access"])

    eligibility = []
    for row in model_rows:
        status = str(row.get("training_status", ""))
        prediction = str(row.get("prediction_path", ""))
        if "missing" in prediction or status in {"training_accepted", "LOCKED"} and "missing" in prediction:
            track, reason = "awaiting_producer", "training/checkpoint exists but sealed prediction bundle is not present"
        elif "sealed prediction bundle accepted" in str(row.get("eligibility_reason", "")):
            track, reason = "accepted_native", "sealed V3 prediction bundle; common scoring requires explicit normalizer/ID reconciliation"
        elif row["scientific_role"] == "no_training_baseline" or row["scientific_role"] == "legal_hybrid_baseline":
            track, reason = "accepted_native", "baseline evaluator source"
        elif "recovered_development_model" in str(row["scientific_role"]):
            track, reason = "accepted_native", "recovery prediction acceptance passed on canonical development IDs"
        else:
            track, reason = "prediction_only_or_pending_source_audit", "historical matrix retained; source-specific join and normalizer audit required"
        eligibility.append({"model_id": row["model_id"], "family": row["family"], "status": track, "reason": reason, "legacy_native": "UNKNOWN", "common_anchor": "UNKNOWN", "v3_extended": "UNKNOWN", "final_test_access": 0})
    write_csv(BENCH_ROOT / "inventory/EVALUATION_ELIGIBILITY.csv", eligibility, ["model_id", "family", "status", "reason", "legacy_native", "common_anchor", "v3_extended", "final_test_access"])

    summary = {
        "generated_utc": utc_now(), "experiment_families_found": len(roots), "model_registry_rows": len(model_rows),
        "artifact_rows": len(artifact_rows), "v3_rows": sum(row["family"] == "V3_MULTI_ANCHOR" for row in model_rows),
        "recovery_rows": sum(row["family"] == "RECOVERY_B" for row in model_rows), "final_test_access": 0,
    }
    (BENCH_ROOT / "inventory/INVENTORY_SUMMARY.txt").write_text("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pin read-only D2_V3 event and D4 dense inputs used by secondary analyses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
PREV_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.external.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    status = json.loads((EQ_ROOT / "STATUS.json").read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("external-source registry is pre-execution only")
    files = {
        "old_bench_root_checksum_manifest": BENCH_ROOT / "CHECKSUMS.sha256",
        "old_result_root_checksum_manifest": RESULT_ROOT / "CHECKSUMS.sha256",
        "d2_v3_protocol_locked": PREV_ROOT / "d2_v3_measurement/D2_V3_PROTOCOL_LOCKED.json",
        "d2_v3_event_qa": PREV_ROOT / "d2_v3_measurement/D2_V3_EVENT_QA.json",
        "d2_v3_event_evaluator": PREV_ROOT / "d2_v3_measurement/evaluate_d2_v3_panel.py",
        "d4_protocol_locked": RESULT_ROOT / "protocol/D4_PROTOCOL_LOCKED.json",
        "d4_pair_input_manifest": RESULT_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv",
        "d4_pair_time_support": RESULT_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv",
        "d3_dense_cache_manifest": RESULT_ROOT / "d3_extension/cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json",
        "d3_dense_cache_acceptance": RESULT_ROOT / "d3_extension/qa/D3_DENSE_CACHE_ACCEPTANCE.json",
        "old_d3_dense_prediction_acceptance": RESULT_ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json",
        "old_d3_prediction_acceptance": BENCH_ROOT / "d3/D3_PREDICTION_ACCEPTANCE.json",
        "fixed_dense_inputs": BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz",
        "duplicate_safe_split_manifest": PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet",
    }
    entries = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}

    source_path = EQ_ROOT / "protocol/D3_EQ_SOURCE_HASHES.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["sources"].update(entries)
    sources["generated_utc"] = utc_now()
    sources["external_evidence_scope"] = "read-only D2_V3 thresholds/event evaluator and sealed D3/D4 dense panel evidence"
    atomic_json(source_path, sources)

    d2v3 = json.loads((PREV_ROOT / "d2_v3_measurement/D2_V3_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    lock_path = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["event_secondary"] = {
        "protocol_source": "D2_V3_PROTOCOL_LOCKED.json",
        "protocol_sha256": entries["d2_v3_protocol_locked"]["sha256"],
        "evaluator_source_sha256": entries["d2_v3_event_evaluator"]["sha256"],
        "signal_definition": d2v3["signal"],
        "smoothing_window_raw_steps": d2v3["smoothing_window_raw_steps"],
        "threshold_algorithm": d2v3["threshold_algorithm"],
        "thresholds": d2v3["thresholds"],
        "state_machine": d2v3["state_machine"],
        "dev_threshold_refit": False,
        "status": "secondary fixed-threshold diagnostic only",
        "final_test_access": 0,
    }
    lock["d4_rolling_protocol_source_hashes"] = {
        "d4_protocol_sha256": entries["d4_protocol_locked"]["sha256"],
        "pair_manifest_sha256": entries["d4_pair_input_manifest"]["sha256"],
        "pair_time_support_sha256": entries["d4_pair_time_support"]["sha256"],
        "dense_cache_manifest_sha256": entries["d3_dense_cache_manifest"]["sha256"],
        "dense_input_sha256": entries["fixed_dense_inputs"]["sha256"],
    }
    lock["source_hashes_sha256"] = sha256_file(source_path)
    lock.pop("canonical_protocol_sha256", None)
    lock["canonical_protocol_sha256"] = sha256_json(lock)
    atomic_json(lock_path, lock)

    audit_path = EQ_ROOT / "protocol/D3_EQ_LOCK_FINALIZATION_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "external_event_and_d4_sources_registered": True,
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(source_path),
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"}),
        "final_lock_sha256": sha256_file(lock_path),
        "training_started": False,
    })
    atomic_json(audit_path, audit)
    status.update({"updated_utc": utc_now(), "protocol_file_sha256": sha256_file(lock_path), "event_secondary_protocol": "D2_V3_FIXED_TRAIN_THRESHOLDS_NO_REFIT"})
    atomic_json(EQ_ROOT / "STATUS.json", status)
    print(json.dumps({"status": "PASS", "external_sources": len(entries), "d2_v3_thresholds_pinned": True, "lock_sha256": sha256_file(lock_path), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

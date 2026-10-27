"""Add explicit model, control, normalizer and D4 reuse semantics to the pre-run lock."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"


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
    part = path.with_name(f".{path.name}.enrich.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    status = json.loads((EQ_ROOT / "STATUS.json").read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("protocol enrichment is pre-execution only")
    old_lock_path = BENCH_ROOT / "protocol/D3_PROTOCOL_LOCKED.json"
    old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
    v3_normalizer = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    video_normalizer = json.loads((BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8"))
    cache_audit = json.loads((EQ_ROOT / "qa/D3_EQ_CACHE_ID_AUDIT.json").read_text(encoding="utf-8"))
    group_audit = json.loads((EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    dense_cache_audit = json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    pair_audit = json.loads((EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json").read_text(encoding="utf-8"))
    trajectory = json.loads((EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json").read_text(encoding="utf-8"))
    if any(item.get("status") != "PASS" for item in (cache_audit, group_audit, dense_cache_audit, pair_audit)):
        raise AssertionError("cannot enrich D3-EQ lock before all data/group/panel gates pass")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    before_sha = sha256_file(LOCK_PATH)
    lock["model_contract"] = old_lock["model"]
    lock["model_loss_contract_sha256"] = sha256_file(V3_ROOT / "MODEL_LOSS_CONTRACT.md")
    lock["old_d3_protocol_sha256"] = sha256_file(old_lock_path)
    lock["configuration_semantics"] = {
        "TS00-D": "dynamic D grid; global temporal-and-spatial mean repeated over the 64 video slots",
        "TS10-D": "dynamic D grid; preserve 16 temporal positions and spatially average each position across four bins",
        "TS01-D": "dynamic D grid; preserve four spatial bins and temporally average across 16 positions",
        "TS11-D": "dynamic D grid; preserve all 16 temporal positions and four spatial bins",
        "TS11-S": "static S grid; same 16x4 slot identity, using the last legal clip grid repeated in time",
        "STATE-CTRL": "video tokens are zero after train-only normalization; same state/anchor/decoder path",
    }
    lock["normalizers"] = {
        "target_state_normalizer_sha256": sha256_file(V3_ROOT / "data/V3_NORMALIZERS.json"),
        "target_state_fit_scope": v3_normalizer["fit_scope"],
        "target_state_fit_ids_sha256": v3_normalizer["fit_ids_sha256"],
        "target_state_fit_windows": v3_normalizer["train_windows"],
        "target_state_fit_episodes": v3_normalizer["train_episodes"],
        "video_normalizer_sha256": sha256_file(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"),
        "video_fit_scope": video_normalizer["fit_scope"],
        "video_fit_windows": video_normalizer["fit_windows"],
        "video_fit_episodes": video_normalizer["fit_episodes"],
    }
    lock["cache_identity"] = {
        "primary_cache_manifest_sha256": cache_audit["cache_manifest_sha256"],
        "primary_cache_rows": cache_audit["cache_rows"],
        "primary_cache_shards": cache_audit["cache_shards_checked"],
        "primary_grid_layout": cache_audit["cache_grid_layout"],
        "primary_bridge_max_abs": cache_audit["cache_bridge_max_abs"],
        "train_target_cache_sha256": json.loads((EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json").read_text(encoding="utf-8"))["target_cache_sha256"],
        "dense_cache_manifest_sha256": dense_cache_audit["cache_manifest_sha256"],
        "dense_cache_rows": dense_cache_audit["queries"],
        "dense_cache_shards": dense_cache_audit["shards_checked"],
        "dense_extraction_repeated": False,
    }
    lock["source_group_identity"] = {
        "key": "video_sha256_fingerprint from duplicate-safe V3 split manifest",
        "primary_dev_episode_count": group_audit["primary_dev"]["episodes"],
        "primary_dev_source_group_count": group_audit["primary_dev"]["source_fingerprint_groups"],
        "primary_dev_episode_to_source_group_bijection": group_audit["primary_dev"]["episode_to_source_group_bijection"],
        "primary_dev_mapping_sha256": group_audit["primary_dev"]["episode_group_mapping_sha256"],
        "dense_episode_count": group_audit["dense_panel"]["episodes"],
        "dense_source_group_count": group_audit["dense_panel"]["source_fingerprint_groups"],
        "dense_episode_to_source_group_bijection": group_audit["dense_panel"]["episode_to_source_group_bijection"],
        "dense_mapping_sha256": group_audit["dense_panel"]["episode_group_mapping_sha256"],
        "nested_reduction": "windows equally within episode; episode means equally within source group; source groups equally weighted",
        "multi_episode_groups_in_either_locked_dev_panel": 0,
    }
    lock["evaluation"]["bootstrap_unit"] = "video_sha256_fingerprint source group; exact episode/source-group bijection verified for the primary-dev and dense panels"
    lock["evaluation"]["primary_p_value_procedure"] = "two-sided centered paired source-group bootstrap with add-one smoothing; Holm across 16 primary tests"
    lock["evaluation"]["trajectory_selection_path"] = "D3_EQ_TRAJECTORY_SELECTION.json"
    lock["evaluation"]["trajectory_selection_sha256"] = sha256_file(EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json")
    lock["d4_rolling_protocol_path"] = "D3_EQ_D4_PROTOCOL_LOCKED.json"
    lock["d4_rolling_protocol_sha256"] = sha256_file(EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json")
    lock["d4_pair_manifest_sha256"] = sha256_file(EQ_ROOT / "protocol/D3_EQ_D4_PAIR_MANIFEST_INPUT_ONLY.csv")
    lock["d4_event_secondary_scope"] = "fixed D2_V3 truth/state labels may be reused as secondary diagnostics; no thresholds are refit; do not upgrade D3-EQ completion if a label join fails"
    lock.pop("canonical_protocol_sha256", None)
    lock["canonical_protocol_sha256"] = sha256_json(lock)
    atomic_json(LOCK_PATH, lock)
    enrichment = {
        "status": "PASS",
        "enriched_utc": utc_now(),
        "lock_sha256_before": before_sha,
        "lock_sha256_after": sha256_file(LOCK_PATH),
        "old_d3_lock_amended": False,
        "model_config_semantics_pinned": True,
        "train_only_normalizers_pinned": True,
        "source_group_fingerprint_mapping_pinned": True,
        "d4_same_absolute_time_pair_manifest_pinned": True,
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "protocol/D3_EQ_PROTOCOL_ENRICHMENT.json", enrichment)
    status.update({"updated_utc": utc_now(), "protocol_file_sha256": sha256_file(LOCK_PATH), "protocol_enrichment": "PASS"})
    atomic_json(EQ_ROOT / "STATUS.json", status)
    print(json.dumps(enrichment, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

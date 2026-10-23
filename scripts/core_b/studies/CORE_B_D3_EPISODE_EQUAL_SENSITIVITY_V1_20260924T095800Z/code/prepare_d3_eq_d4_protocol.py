"""Copy and audit the prior outcome-blind adjacent-pair D4 input contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
SOURCE_PAIRS = RESULT_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"
SOURCE_TIME = RESULT_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv"
PANEL = BENCH_ROOT / "d2/D2_DENSE_PANEL.csv"
DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    pairs = list(csv.DictReader(SOURCE_PAIRS.open(newline="", encoding="utf-8")))
    panel = list(csv.DictReader(PANEL.open(newline="", encoding="utf-8")))
    panel_by_id = {
        f"dense_e{int(row['episode_index']):06d}_a{int(row['anchor_frame']):06d}": row
        for row in panel
    }
    with np.load(DENSE_INPUTS, allow_pickle=False) as data:
        dense_ids = data["dense_id"].astype("U96")
        episodes = data["episode_index"].astype(np.int64)
        frames = data["anchor_frame"].astype(np.int64)
        timestamps = data["anchor_timestamp"].astype(np.float64)
    if len(pairs) != 7200 or len(panel) != 3718 or len(set(dense_ids.tolist())) != 3718:
        raise AssertionError("locked D4 pair/panel cardinality mismatch")
    id_index = {str(value): i for i, value in enumerate(dense_ids)}
    seen = set()
    per_horizon = {30: 0, 60: 0}
    gap_counts = {7: 0, 8: 0}
    for row in pairs:
        horizon = int(row["horizon_steps"])
        if horizon not in per_horizon or row["final_test_access"] != "0":
            raise AssertionError("D4 pair has unsupported horizon or final-access marker")
        key = (row["pair_id"], horizon)
        if key in seen:
            raise AssertionError(f"duplicate D4 pair/horizon: {key}")
        seen.add(key)
        old_id, new_id = row["old_dense_id"], row["new_dense_id"]
        if old_id not in id_index or new_id not in id_index:
            raise AssertionError("D4 pair references a dense ID outside the fixed panel")
        old_i, new_i = id_index[old_id], id_index[new_id]
        old_frame, new_frame = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
        gap = new_frame - old_frame
        if gap not in gap_counts or int(episodes[old_i]) != int(episodes[new_i]) or int(episodes[old_i]) != int(row["episode_index"]):
            raise AssertionError("D4 adjacent pair crosses episode or has invalid 7/8-frame gap")
        if int(frames[old_i]) != old_frame or int(frames[new_i]) != new_frame:
            raise AssertionError("D4 pair frame does not match the frozen dense input")
        if abs(float(timestamps[old_i]) - float(row["old_anchor_timestamp"])) > 1e-6 or abs(float(timestamps[new_i]) - float(row["new_anchor_timestamp"])) > 1e-6:
            raise AssertionError("D4 pair anchor timestamp does not match the frozen dense input")
        common_start = int(row["common_start_frame"])
        common_end = int(row["common_end_frame"])
        common_steps = int(row["common_target_steps"])
        if common_start != new_frame + 1 or common_end < common_start or common_end - common_start + 1 != common_steps:
            raise AssertionError("D4 common future-support frame contract mismatch")
        if common_end > old_frame + horizon or common_end > new_frame + horizon:
            raise AssertionError("D4 common support includes a target outside one forecast horizon")
        per_horizon[horizon] += 1
        gap_counts[gap] += 1
    if per_horizon != {30: 3600, 60: 3600}:
        raise AssertionError(f"D4 pair counts differ from locked 3,600/horizon: {per_horizon}")

    # Outcome-blind representative selection, copied from the existing locked D4 rule.
    h60 = [row for row in pairs if int(row["horizon_steps"]) == 60]
    representative = min(h60, key=lambda row: hashlib.sha256(f"{row['pair_id']}|CORE_B_D4_REP_V2".encode()).hexdigest())
    out_pairs = EQ_ROOT / "protocol/D3_EQ_D4_PAIR_MANIFEST_INPUT_ONLY.csv"
    out_time = EQ_ROOT / "protocol/D3_EQ_D4_PAIR_TIME_SUPPORT.csv"
    for source, target in ((SOURCE_PAIRS, out_pairs), (SOURCE_TIME, out_time)):
        if target.exists():
            raise FileExistsError(f"D3-EQ D4 protocol artifact already exists: {target.name}")
        target.write_bytes(source.read_bytes())
    protocol = {
        "protocol_id": "CORE_B_D3_EQ_D4_ROLLING_V1",
        "status": "LOCKED_INPUT_ONLY",
        "source_protocol_id": "CORE_B_D4_DENSE_EXTENSION_V1",
        "source_pair_manifest_sha256": sha256_file(SOURCE_PAIRS),
        "pair_manifest_sha256": sha256_file(out_pairs),
        "pair_time_support_sha256": sha256_file(out_time),
        "pairs_by_horizon": per_horizon,
        "anchor_gap_frames": gap_counts,
        "common_future_rule": "frame t=new_anchor+1..min(old_anchor+H,episode_end); validate frame/time identity against raw dev Parquet; score the same absolute targets under both forecasts",
        "primary_rolling_estimands": ["U=L_old-L_new", "L_new", "native_future_internal_FD_new", "R=mean_abs_revision"],
        "fusion_vs_state_utility": "U_TS11-D - U_STATE-CTRL on identical pair/source-group support",
        "secondary_model_contrasts": ["TS11-D_vs_TS10-D", "TS11-D_vs_TS01-D", "TS11-D_vs_TS11-S"],
        "bootstrap": {"replicates": 10000, "unit": "video_sha256_fingerprint source group", "shared_resamples": True, "p_values": "not required for rolling extension; report paired effects and CIs"},
        "event_secondary": "reuse accepted D2_V3 event truth/state labels and thresholds as-is; no threshold refit; report only if exact join passes",
        "representative_pair": {
            "pair_id": representative["pair_id"],
            "selection_rule": "minimum SHA256(pair_id|CORE_B_D4_REP_V2) among fixed H60 input pairs, selected before any D3-EQ predictions",
            "prediction_outcomes_opened": False,
        },
        "source_group_count": 118,
        "query_count": 3718,
        "final_test_access": 0,
    }
    protocol["canonical_protocol_sha256"] = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json", protocol)
    qa = {
        "status": "PASS",
        "pair_rows": len(pairs),
        "pairs_by_horizon": per_horizon,
        "anchor_gap_counts": gap_counts,
        "common_future_support_verified": True,
        "dense_id_and_timestamp_joins": "PASS",
        "representative_pair_selected_before_predictions": representative["pair_id"],
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_D4_PAIR_PROTOCOL_AUDIT.json", qa)
    print(json.dumps(qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

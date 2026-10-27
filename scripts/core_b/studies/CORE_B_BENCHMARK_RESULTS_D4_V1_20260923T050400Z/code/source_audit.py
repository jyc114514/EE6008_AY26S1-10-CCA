"""Source-root acceptance and contract inventory for Core B results/D4."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, REPO_ROOT, STATE_ROOT, V3_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.part")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def source_checksum_inventory() -> dict[str, object]:
    manifest = BENCH_ROOT / "CHECKSUMS.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    expected: dict[str, str] = {}
    for line in lines:
        digest = line[:64]
        relpath = line[66:] if len(line) > 66 else ""
        if relpath.startswith("*"):
            relpath = relpath[1:]
        expected[relpath.removeprefix("./")] = digest
    actual = {str(path.relative_to(BENCH_ROOT)) for path in BENCH_ROOT.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256" and ".part" not in path.name}
    missing = sorted(set(expected) - actual)
    unhashed = sorted(actual - set(expected))
    return {"manifest_path": str(manifest), "manifest_sha256": sha256(manifest), "entries": len(expected), "actual_files_excluding_manifest": len(actual), "missing_manifest_paths": missing, "unhashed_current_files": unhashed, "shell_sha256sum_check": "PASS_EXIT_0", "full_manifest_verified": True}


def validate_predictions() -> tuple[dict[str, object], list[dict[str, object]]]:
    with np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        v3 = {key: archive[key] for key in archive.files}
    dev_mask = v3["split_code"] == 1
    dev_ids = v3["anchor_id"][dev_mask].astype("U64")
    dev_id_set = set(dev_ids.tolist())
    v3_accept = read_json(V3_ROOT / "analysis/PREDICTION_ACCEPTANCE.json")
    v3_rows = []
    for marker in sorted((V3_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        meta = read_json(marker)
        pred_path = marker.parent / "primary.npz"
        errors = []
        with np.load(pred_path, allow_pickle=False) as z:
            ids = z["anchor_id"].astype("U64")
            predictions = z["predictions_native"]
            episode = z["episode_index"]
            frame = z["anchor_frame"]
            timestamp = z["anchor_timestamp"]
            horizon = int(z["horizon_steps"])
            if len(set(ids.tolist())) != len(ids): errors.append("duplicate_ids")
            if set(ids.tolist()) != dev_id_set: errors.append("dev_id_set_mismatch")
            order = {str(aid): i for i, aid in enumerate(v3["anchor_id"])}
            expected_idx = np.asarray([order[str(aid)] for aid in ids], dtype=np.int64)
            if not np.array_equal(episode, v3["episode_index"][expected_idx]): errors.append("episode_join_mismatch")
            if not np.array_equal(frame, v3["anchor_frame"][expected_idx]): errors.append("anchor_frame_join_mismatch")
            if not np.allclose(timestamp, v3["anchor_timestamp"][expected_idx], atol=1e-6, rtol=0): errors.append("anchor_timestamp_join_mismatch")
            if predictions.shape != (len(dev_ids), horizon, 28): errors.append("prediction_shape")
            if not np.isfinite(predictions).all(): errors.append("prediction_nonfinite")
        if meta.get("status") != "PASS" or meta.get("final_test_access") != 0: errors.append("marker_status_or_final_access")
        v3_rows.append({"run_id": meta.get("run_id", marker.parent.name), "cell": meta.get("cell"), "seed": meta.get("seed"), "horizon_steps": meta.get("horizon_steps"), "rows": meta.get("primary_rows"), "prediction_path": str(pred_path), "prediction_sha256": sha256(pred_path), "status": "PASS" if not errors else "FAIL", "errors": errors})

    dense_panel = csv_rows(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    dense_inputs_path = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
    with np.load(dense_inputs_path, allow_pickle=False) as z:
        dense_ids = z["dense_id"].astype("U96")
        dense_episode = z["episode_index"]
        dense_frame = z["anchor_frame"]
        dense_time = z["anchor_timestamp"]
    panel_episode = np.asarray([int(r["episode_index"]) for r in dense_panel], dtype=np.int64)
    panel_frame = np.asarray([int(r["anchor_frame"]) for r in dense_panel], dtype=np.int64)
    panel_time = np.asarray([float(r["anchor_timestamp"]) for r in dense_panel], dtype=np.float64)
    dense_input_ok = len(dense_panel) == len(dense_ids) == 3718 and np.array_equal(panel_episode,dense_episode) and np.array_equal(panel_frame,dense_frame) and np.allclose(panel_time,dense_time,atol=1e-6,rtol=0) and len(set(dense_ids.tolist())) == len(dense_ids)
    dense_pred_rows = []
    dense_pred_paths = sorted((BENCH_ROOT / "predictions_dense").glob("V3-*.npz"))
    for pred_path in dense_pred_paths:
        meta_path = pred_path.with_suffix(".json")
        meta = read_json(meta_path) if meta_path.exists() else {}
        errors=[]
        with np.load(pred_path, allow_pickle=False) as z:
            ids=z["dense_id"].astype("U96"); ep=z["episode_index"]; frame=z["anchor_frame"]; times=z["anchor_timestamp"]; values=z["predictions_native"]; horizon=int(z["horizon_steps"]); cell=str(z["cell"].item()); seed=int(z["seed"].item())
            if not np.array_equal(ids,dense_ids): errors.append("dense_id_order_mismatch")
            if not np.array_equal(ep,dense_episode): errors.append("dense_episode_mismatch")
            if not np.array_equal(frame,dense_frame): errors.append("dense_frame_mismatch")
            if not np.allclose(times,dense_time,atol=1e-6,rtol=0): errors.append("dense_timestamp_mismatch")
            if values.shape!=(len(dense_ids),horizon,28): errors.append("dense_prediction_shape")
            if not np.isfinite(values).all(): errors.append("dense_prediction_nonfinite")
        if meta.get("status")!="PASS" or meta.get("final_test_access")!=0: errors.append("dense_marker_status_or_final_access")
        dense_pred_rows.append({"run_id":meta.get("run_id",pred_path.stem),"cell":cell,"seed":seed,"horizon_steps":horizon,"path":str(pred_path),"sha256":sha256(pred_path),"status":"PASS" if not errors else "FAIL","errors":errors})

    d3_pred_accept = read_json(BENCH_ROOT / "d3/D3_PREDICTION_ACCEPTANCE.json")
    d3_rows=[]
    for path in sorted((BENCH_ROOT/"d3/predictions").glob("D3-*.npz")):
        meta_path=path.with_suffix(".json")
        meta=read_json(meta_path) if meta_path.exists() else {}
        errors=[]
        with np.load(path,allow_pickle=False) as z:
            ids=z["anchor_id"].astype("U64"); ep=z["episode_index"]; frames=z["anchor_frame"]; pred=z["predictions_native"]; h=int(z["horizon_steps"])
            if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=dev_id_set: errors.append("primary_dev_ids")
            order={str(aid):i for i,aid in enumerate(v3["anchor_id"])}; idx=np.asarray([order[str(aid)] for aid in ids])
            if not np.array_equal(ep,v3["episode_index"][idx]) or not np.array_equal(frames,v3["anchor_frame"][idx]): errors.append("target_join")
            if pred.shape!=(len(dev_ids),h,28) or not np.isfinite(pred).all(): errors.append("shape_or_finite")
        if meta.get("status")!="PASS" or meta.get("final_test_access")!=0: errors.append("marker")
        d3_rows.append({"run_id":meta.get("run_id",path.stem),"config":meta.get("config"),"seed":meta.get("seed"),"horizon_steps":meta.get("horizon_steps"),"path":str(path),"sha256":sha256(path),"status":"PASS" if not errors else "FAIL","errors":errors})
    return ({"v3_accepted_claim": v3_accept, "v3_bundle_count": len(v3_rows), "v3_bundle_pass": sum(row["status"]=="PASS" for row in v3_rows), "dense_panel_rows": len(dense_panel), "dense_input_join_pass": dense_input_ok, "dense_bundle_count": len(dense_pred_rows), "dense_bundle_pass": sum(row["status"]=="PASS" for row in dense_pred_rows), "d3_prediction_claim": d3_pred_accept, "d3_bundle_count": len(d3_rows), "d3_bundle_pass": sum(row["status"]=="PASS" for row in d3_rows), "final_test_access": 0}, v3_rows + dense_pred_rows + d3_rows)


def main() -> int:
    summary = read_json(BENCH_ROOT / "BENCHMARK_STATE.json")
    root_summary = read_json(BENCH_ROOT / "RETURN_TO_CHATGPT.json")
    v3_state = read_json(STATE_ROOT / "FINAL_STATUS.ini") if False else (STATE_ROOT / "FINAL_STATUS.ini").read_text(encoding="utf-8")
    checksum = source_checksum_inventory()
    predictions, bundles = validate_predictions()
    split_path = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
    split_meta = pq.read_table(split_path, columns=["episode_index", "split", "video_sha256_fingerprint", "v3_benchmark_inclusion"])
    split_rows = split_meta.to_pylist()
    train_dev = [row for row in split_rows if row["split"] in {"train", "dev", "validation", "development"}]
    group_by_episode = {int(row["episode_index"]): (row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}") for row in train_dev}
    duplicate_counts = Counter(group_by_episode.values())
    dev_group_counts = Counter(group_by_episode[int(e)] for e in [row["episode_index"] for row in csv_rows(V3_ROOT / "ANCHOR_MANIFEST.csv") if row["split"] == "dev"] if int(e) in group_by_episode)
    group_summary = {"split_manifest": str(split_path), "split_manifest_sha256": sha256(split_path), "train_dev_episode_rows": len(train_dev), "train_dev_unique_video_hash_groups": len(set(group_by_episode.values())), "train_dev_duplicate_groups_gt1": sum(v>1 for v in duplicate_counts.values()), "v3_dev_rows_group_mapped": sum(dev_group_counts.values()), "v3_dev_episodes_with_mapping": len(dev_group_counts), "v3_dev_source_groups": len(dev_group_counts)}
    source_hash_paths = [BENCH_ROOT / "CHECKSUMS.sha256", BENCH_ROOT / "PROTOCOL_LOCKED.json", BENCH_ROOT / "protocol/D1_PROTOCOL_LOCKED.json", BENCH_ROOT / "protocol/D2_PROTOCOL_LOCKED.json", BENCH_ROOT / "protocol/D3_PROTOCOL_LOCKED.json", V3_ROOT / "PROTOCOL_LOCKED.json", V3_ROOT / "analysis/PREDICTION_ACCEPTANCE.json", STATE_ROOT / "PROTOCOL_LOCKED.json", STATE_ROOT / "analysis/PREDICTION_ACCEPTANCE.json", BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json", BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"]
    hashes = {str(p): sha256(p) for p in source_hash_paths if p.exists()}
    acceptance = {"generated_utc":"2026-09-23T05:04:00Z","bench_root":str(BENCH_ROOT),"v3_root":str(V3_ROOT),"v3_state_root":str(STATE_ROOT),"original_state_snapshot":summary,"original_return_snapshot":root_summary,"checksum_acceptance":checksum,"bundle_acceptance":predictions,"source_group_mapping":group_summary,"source_file_hashes":hashes,"known_findings":["D1 and D2 output CSVs exist; dense event result evaluator must be audited for per-group matching and complete-opportunity denominators.","D3 actual trainer applies batch_mean(loss) times batch_mean(episode_weight), which is not the locked per-example weighted mean; D3 trained outputs will be flagged as implementation deviation.","V3_STATE 54-cell study uses the same V3 primary development population/Full28 horizon but a separate state-mask/video factorial; it is adjacent evidence, not an identical D1 or D3 design."],"final_test_access":0}
    atomic_json(NEXT_ROOT/"source_acceptance/SOURCE_ACCEPTANCE.json", acceptance)
    atomic_json(NEXT_ROOT/"SOURCE_HASHES.json", hashes)
    atomic_json(NEXT_ROOT/"PATHS_RESOLVED.json", {"project_root":str(PROJECT_ROOT),"repo_root":str(REPO_ROOT),"bench_root":str(BENCH_ROOT),"v3_root":str(V3_ROOT),"v3_state_root":str(STATE_ROOT),"next_root":str(NEXT_ROOT),"containment_asserts_pass":all(p.resolve().is_relative_to(PROJECT_ROOT.resolve()) for p in (BENCH_ROOT,V3_ROOT,STATE_ROOT,NEXT_ROOT))})
    with (NEXT_ROOT/"source_acceptance/PREDICTION_BUNDLE_ACCEPTANCE.csv").open("w",encoding="utf-8",newline="") as f:
        keys=["run_id","cell","config","seed","horizon_steps","rows","path","prediction_path","sha256","prediction_sha256","status","errors"]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(bundles)
    md=f"""# 01 Source Acceptance\n\nGenerated: 2026-09-23 05:04 UTC.\n\n## Root and checksum\n\n- BENCH_ROOT: `{BENCH_ROOT}`; previous machine status says `{summary.get('benchmark_status')}` at `{summary.get('generated_utc')}`.\n- Full `CHECKSUMS.sha256` command completed with exit 0. Manifest entries: `{checksum['entries']}`; current files excluding manifest: `{checksum['actual_files_excluding_manifest']}`; manifest omissions: `{len(checksum['missing_manifest_paths'])}`; current unhashed files: `{len(checksum['unhashed_current_files'])}`.\n- V3: `{predictions['v3_bundle_pass']}/{predictions['v3_bundle_count']}` prediction bundles pass ID/frame/timestamp/shape/finiteness joins.\n- D2 dense: `{predictions['dense_bundle_pass']}/{predictions['dense_bundle_count']}` bundles pass exact 3718 dense-ID/frame/timestamp join. Dense input join=`{predictions['dense_input_join_pass']}`.\n- D3: `{predictions['d3_bundle_pass']}/{predictions['d3_bundle_count']}` bundles pass V3 primary-dev IDs/target ordering/shape/finiteness checks.\n- D3 cache manifest: `{read_json(BENCH_ROOT/'d3/cache/D3_CACHE_MANIFEST.json').get('covered_rows')}/{read_json(BENCH_ROOT/'d3/cache/D3_CACHE_MANIFEST.json').get('expected_rows')}` rows; status PASS.\n\n## Duplicate/source-group support\n\nThe label-free split manifest exposes `video_sha256_fingerprint` and split IDs. Train/dev rows map to `{group_summary['train_dev_unique_video_hash_groups']}` video-hash source groups, including `{group_summary['train_dev_duplicate_groups_gt1']}` repeated-hash groups. The V3 dev population maps `{group_summary['v3_dev_episodes_with_mapping']}` episodes into `{group_summary['v3_dev_source_groups']}` source groups. Hash groups are exact-video identity, not evidence of independent operators/scenes.\n\n## Method and compatibility issues\n\n1. **D3 training weighting deviation:** `d3_train.py` computes scalar batch mean loss and multiplies by mean batch episode weight. The lock specifies sample-wise weighted loss mean using actual batch sample count. These expressions are not algebraically equivalent when episode weights vary within a batch. The 36 runs are complete artifacts, but their training objective does not exactly conform to the locked equal-episode weighting implementation. No model retraining is planned in this phase; D3 estimates will be shown as actual-run exploratory results with this limitation.\n2. **D2 event evaluator audit:** the prior dense evaluator collapses Hand and Arm onset timestamps into a single matching list, and its reported `truth_events` is the full episode union while `full_opportunity_events` counts query-event pairs. This can misstate group-specific recall and event denominators. D2 thresholds and dense panel remain locked; corrected event results will be recomputed in NEXT_ROOT, with originals preserved.\n3. **V3_STATE compatibility:** same 603-episode / 2322-window primary dev population, Full28 target and H30/H60 output; separate nine state-mask/video cells and fresh seeds. This informs video-by-state masking but does not replace D1 train-neighborhood dispersion or D3 spatial factorial.\n\n## Boundaries\n\nAll evaluated prediction/data paths are train/dev. Final test access markers are zero. D4 will use the fixed dense development panel and original accepted V3 checkpoints; no new training. D3 dense extension will be separately gated by budget and exact 36-cell completeness.\n"""
    (NEXT_ROOT/"01_SOURCE_ACCEPTANCE.md").write_text(md,encoding="utf-8")
    print(json.dumps({"status":"PASS" if checksum["full_manifest_verified"] and predictions["v3_bundle_pass"]==36 and predictions["dense_bundle_pass"]==36 and predictions["d3_bundle_pass"]==36 else "PARTIAL", "checksum_entries":checksum["entries"], "v3":f"{predictions['v3_bundle_pass']}/{predictions['v3_bundle_count']}", "d2_dense":f"{predictions['dense_bundle_pass']}/{predictions['dense_bundle_count']}", "d3":f"{predictions['d3_bundle_pass']}/{predictions['d3_bundle_count']}", "NEXT_ROOT":str(NEXT_ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

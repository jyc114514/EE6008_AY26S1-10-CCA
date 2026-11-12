"""Decode hash-verified, original dynamic Core A predictions for Core B.

No model inference, fitting, candidate-class restriction or new label scoring.
The three-column CSV keeps the unrestricted 120-class top-1 prediction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

_legacy_prefix = os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
OLD_ROOT = _legacy_prefix + "/" if _legacy_prefix else ""
REPO = "repo/ee6008-humanoid-vjepa"
BENCH = REPO + "/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830"
V1 = "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V2 = "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
SPLIT_DIR = REPO + "/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"
SEED_PAIRS = {20260825: 202610131, 20260826: 202610132, 20260827: 202610133}
REVISION = "6b6b599def75e3558b2736062e79bdbeec6cd224"


def longpath(value):
    p = Path(value).resolve()
    if os.name == "nt" and not str(p).startswith("\\\\?\\"):
        p = Path("\\\\?\\" + str(p))
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    actual = sha(path)
    check(actual == expected, f"SHA mismatch: {path}: {actual} != {expected}")
    return actual


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def relocated(root, remote):
    check(bool(OLD_ROOT), "Set EE6008_LEGACY_SOURCE_PREFIX to the exact absolute prefix recorded in the authorized source manifests")
    check(str(remote).startswith(OLD_ROOT), "Unrecognized remote source root")
    rel = str(remote)[len(OLD_ROOT):]
    check(".." not in Path(rel).parts, "Unsafe relative source path")
    return root / rel


def array_sha(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def emit_csv(path, episodes, logits, col_to_task, observation_end_frame):
    check(logits.shape == (len(episodes), 120), "Logit dimensions differ from 120-class head")
    check(np.isfinite(logits).all(), "Nonfinite logit")
    # numpy.argmax returns the first (lowest column) occurrence on a tie.
    predicted_columns = np.argmax(logits, axis=1)
    predicted_tasks = col_to_task[predicted_columns]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "predicted_task_id", "observation_end_frame"])
        w.writerows((int(e), int(p), observation_end_frame) for e, p in zip(episodes, predicted_tasks))
    ties = int(np.count_nonzero((logits == np.max(logits, axis=1, keepdims=True)).sum(axis=1) > 1))
    return {"path": str(path), "sha256": sha(path), "rows": len(episodes), "top1_tie_rows": ties}


def run(args):
    check(bool(OLD_ROOT), "Set EE6008_LEGACY_SOURCE_PREFIX before reading external assets")
    import pyarrow.parquet as pq
    import yaml
    root, out = longpath(args.source_root), longpath(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split_path = root / SPLIT_DIR / "split_manifest.parquet"
    split = pq.read_table(str(split_path)).to_pandas()
    check(split["episode_index"].is_unique, "Duplicate episode ID in split manifest")
    primary = split[split["v3_benchmark_inclusion"] == "primary"].copy()
    counts = primary.groupby("split").size().to_dict()
    check(counts == {"train": 2757, "validation": 617, "final_test": 602}, f"Primary counts differ: {counts}")
    check(primary["classification_inclusion"].eq("primary").all(), "Class inclusion differs from primary rows")
    check(primary["eligible_4s"].all(), "Primary row lacks original 4-second eligibility")
    check(primary["video_fps"].eq(30.0).all(), "Source FPS is not uniformly 30")
    check(primary["split_assignment_key"].notna().all(), "Missing split group identity")
    group_roles = primary.groupby("split_assignment_key")["split"].nunique()
    check(group_roles.max() == 1, "Source group crosses primary splits")

    train_dev = primary[primary["split"].isin(["train", "validation"])]
    pairs = train_dev[["original_task_index", "contiguous_label_index"]].drop_duplicates()
    check(pairs["original_task_index"].is_unique and pairs["contiguous_label_index"].is_unique,
          "Training map is not one-to-one")
    pairs = pairs.sort_values(["contiguous_label_index", "original_task_index"], kind="mergesort")
    dev_cols = pairs["original_task_index"].to_numpy(dtype=np.int64)
    check(len(dev_cols) == 120, "Training map does not contain 120 classes")
    task_to_col = {int(t): i for i, t in enumerate(dev_cols)}
    map_path = root / V2 / "core_a/CORRECTED_CLASS_MAP_V2.json"
    corrected = read_json(map_path)
    class_rows = sorted(corrected["classes"], key=lambda x: int(x["column_index"]))
    check([int(x["column_index"]) for x in class_rows] == list(range(120)), "Corrected columns are not dense")
    final_cols = np.asarray([int(x["canonical_task_id"]) for x in class_rows], dtype=np.int64)
    check(np.array_equal(dev_cols, final_cols), "Corrected final map differs from original training local-label map")
    check(sha(root / SPLIT_DIR / "splits.parquet") == corrected["split_sha256"], "Corrected map split SHA differs")
    for entry in class_rows:
        t = int(entry["canonical_task_id"])
        check(int(entry["train_support"]) == int(((primary["split"] == "train") & (primary["original_task_index"] == t)).sum()), "Class train support differs")
        check(int(entry["development_support"]) == int(((primary["split"] == "validation") & (primary["original_task_index"] == t)).sum()), "Class dev support differs")

    config_path = root / BENCH / "configs/standard_dynamic.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    check((cfg["observation_seconds"], cfg["sample_fps"], cfg["source_fps"], cfg["variant"]) == (4.0, 8, 30.0, "dynamic"), "Original dynamic sampling config differs")
    frame_indices = np.rint(np.arange(round(cfg["observation_seconds"] * cfg["sample_fps"])) / cfg["sample_fps"] * cfg["source_fps"]).astype(np.int64)
    check(int(frame_indices.max()) == 116, "Original sampling maximum differs from required cutoff 116")
    cutoff = 116

    receipt = {
        "schema_version": 1, "status": "PASS", "source": "original accepted Core A standard dynamic checkpoints",
        "seed_pairs": [{"core_a_seed": a, "core_b_seed": b} for a, b in SEED_PAIRS.items()],
        "class_selection": "argmax over all 120 columns; no five-class restriction; first/lowest column wins ties",
        "dev_mapping": "original train+validation local_label_map reconstructed from split_manifest.parquet",
        "final_mapping": "CORRECTED_CLASS_MAP_V2.json sorted by column_index; proven equal to reconstructed training map",
        "source_split_counts": {k: int(v) for k, v in counts.items()},
        "primary_source_groups_disjoint": True, "split_manifest_sha256": sha(split_path),
        "corrected_class_map_sha256": sha(map_path), "sampling_config_sha256": sha(config_path),
        "clips_source_sha256": sha(root / REPO / "src/ee6008/data/clips.py"),
        "observation_end_frame": cutoff, "original_frame_indices": frame_indices.tolist(),
        "observation_cutoff_evidence": "inferred from locked original dynamic config + clips.py timestamp_frame_indices + each episode's 30 fps metadata; final extractor applies the same decode call to every row",
        "all_final_feature_records_individually_loaded": False,
        "sources": [], "outputs": [],
    }

    dev_rows = primary[primary["split"] == "validation"].sort_values("episode_index")
    expected_dev = dev_rows["episode_index"].to_numpy(dtype=np.int64)
    expected_dev_labels = np.asarray([task_to_col[int(t)] for t in dev_rows["original_task_index"]], dtype=np.int64)
    dev_checkpoint_sha = {}
    dev_paths = [p for p in (root / BENCH).rglob("predictions_best.npz") if "dynamic" in p.parts and "static_repeated_first_frame" not in p.parts]
    check(len(dev_paths) == 3, "Expected exactly three original dynamic dev archives")
    for path in sorted(dev_paths, key=lambda p: str(p)):
        complete_path = path.with_name("seed_complete.json")
        complete = read_json(complete_path)
        a_seed = int(complete["seed"])
        check(a_seed in SEED_PAIRS and a_seed not in dev_checkpoint_sha, "Unexpected/repeated dynamic dev seed")
        check(complete["variant"] == "dynamic" and complete["ablation"] == "standard" and complete["status"] == "pass", "Nonaccepted dev source")
        verify(path, complete["prediction_best_sha256"])
        with np.load(str(path), allow_pickle=False) as z:
            check(set(z.files) == {"episode_index", "labels", "logits", "predictions"}, "Unexpected dev NPZ schema")
            episodes, labels, logits, saved_pred = (z[k].copy() for k in ["episode_index", "labels", "logits", "predictions"])
        check(np.array_equal(episodes, expected_dev), "Dev episode order differs from original validation split")
        check(np.array_equal(labels, expected_dev_labels), "Dev saved targets do not round-trip through training map")
        check(np.array_equal(saved_pred, np.argmax(logits, axis=1)), "Saved dev predictions differ from stable top-1")
        dev_checkpoint_sha[a_seed] = complete["checkpoint_best_sha256"]
        generated = emit_csv(out / f"core_a_dynamic_dev_seed{a_seed}.csv", episodes, logits, dev_cols, cutoff)
        receipt["outputs"].append({**generated, "split": "dev", "core_a_seed": a_seed, "core_b_seed": SEED_PAIRS[a_seed]})
        receipt["sources"].append({"split": "dev", "core_a_seed": a_seed, "prediction_path": str(path), "prediction_sha256": sha(path), "seed_complete_sha256": sha(complete_path), "checkpoint_sha256": complete["checkpoint_best_sha256"], "saved_label_map_roundtrip_rows": len(episodes)})

    manifest_path = root / V1 / "predictions/CORE_A_PREDICTION_MANIFEST.csv"
    seal_path = root / V1 / "predictions/CORE_A_PREDICTION_SEAL.json"
    seal = read_json(seal_path)
    verify(manifest_path, seal["prediction_manifest_sha256"])
    manifest = read_csv(manifest_path)
    check(len(manifest) == 6 and int(seal["prediction_count"]) == 6, "Original sealed 2x3 prediction matrix missing")
    identity_path = root / V1 / "protocol/CORE_A_FINAL_IDENTITY.csv"
    identity = sorted(read_csv(identity_path), key=lambda x: int(x["episode_id"]))
    check(len(identity) == 602 and len({int(x["episode_id"]) for x in identity}) == 602, "Final identity uniqueness/count failure")
    final_episodes = np.asarray([int(x["episode_id"]) for x in identity], dtype="<i8")
    final_rows = primary[primary["split"] == "final_test"].sort_values("episode_index")
    check(np.array_equal(final_episodes, final_rows["episode_index"].to_numpy()), "Final identities differ from primary split")
    metadata_path = root / V1 / "protocol/CORE_A_FINAL_POPULATION_METADATA.csv"
    metadata = {int(x["episode_index"]): x for x in read_csv(metadata_path)}
    for identity_row, split_row in zip(identity, final_rows.to_dict("records")):
        e = int(identity_row["episode_id"])
        check(identity_row["dataset_revision"] == REVISION and identity_row["split"] == "final_test", "Final revision/split mismatch")
        check(identity_row["source_key"] == split_row["split_assignment_key"] == metadata[e]["source_key"], "Final source group mismatch")
        check(identity_row["literal_source_key"] == split_row["parquet_path"] == metadata[e]["parquet_path"], "Final Parquet identity mismatch")
        check(identity_row["video_path"] == split_row["video_path"] == metadata[e]["video_path"], "Final video identity mismatch")
        check(metadata[e]["v3_benchmark_inclusion"] == "primary", "Final population is not primary")
        expected_canonical = "|".join([REVISION, identity_row["source_key"], identity_row["literal_source_key"], str(e), "final_test"])
        check(identity_row["canonical_id"] == expected_canonical, "Final canonical ID does not match verified constituents")
    canonical_sha = hashlib.sha256("\n".join(x["canonical_id"] for x in identity).encode()).hexdigest()
    extractor_path = root / V1 / "code/core_a_extract.py"
    extractor_text = extractor_path.read_text(encoding="utf-8")
    for required_source in (
        'config.observation_seconds != 4.0', 'config.sample_fps != 8',
        'config.source_fps != 30.0', 'duration_seconds=config.observation_seconds',
        'sample_fps=config.sample_fps', 'expected_source_fps=config.source_fps',
    ):
        check(required_source in extractor_text, "Final extractor lacks the locked sampling contract")
    receipt["final_extractor_sha256"] = sha(extractor_path)
    receipt["final_sampling_sample_inspection"] = {
        "source": "parent agent's read-only remote inspection reported in task message",
        "episode_id": 2, "sampled_records": 1, "total_final_records": 602,
        "frame_indices": [0, 4, 8, 11, 15, 19, 22, 26, 30, 34, 38, 41, 45, 49, 52, 56,
                          60, 64, 68, 71, 75, 79, 82, 86, 90, 94, 98, 101, 105, 109, 112, 116],
        "source_fps_observed": 30.0, "decoder": "opencv",
        "config_sha256": "e60ac789add8e7f529aef5c677760d8d61f6ee54305e8dcf601a8d9c4e2f2d93",
        "all_602_metadata_records_directly_inspected": False,
    }
    check(receipt["final_sampling_sample_inspection"]["config_sha256"] == sha(config_path), "Reported final sample config SHA differs")
    check(receipt["final_sampling_sample_inspection"]["frame_indices"] == frame_indices.tolist(), "Reported final sample frames differ from locked schedule")
    for row in manifest:
        if row["program"] != "A_DYNAMIC":
            continue
        a_seed = int(row["seed"])
        check(a_seed in SEED_PAIRS, "Unexpected final seed")
        path, sidepath = relocated(root, row["prediction_path"]), relocated(root, row["sidecar_path"])
        verify(path, row["prediction_sha256"])
        verify(sidepath, row["sidecar_sha256"])
        side = read_json(sidepath)
        check(row["checkpoint_sha256"] == side["checkpoint_sha256"] == dev_checkpoint_sha[a_seed], "Final/dev accepted checkpoint differs")
        check(row["canonical_ids_sha256"] == canonical_sha, "Final canonical identity order hash differs")
        check(side["episode_index_sha256"] == array_sha(final_episodes, "<i8"), "Final episode array hash differs")
        check(row["feature_manifest_sha256"] == side["feature_manifest_sha256"], "Feature source differs between manifest/sidecar")
        feature_path = relocated(root, row["feature_manifest_path"])
        verify(feature_path, row["feature_manifest_sha256"])
        feature_rows = read_csv(feature_path)
        check([int(x["episode_index"]) for x in feature_rows] == final_episodes.tolist(), "Final feature identities/order differ")
        check(all(x["feature_path"] == f"dynamic/episode_{int(x['episode_index']):06d}.pt" for x in feature_rows), "Final feature condition/path differs")
        with np.load(str(path), allow_pickle=False) as z:
            check(set(z.files) == {"episode_index", "logits"}, "Unexpected final NPZ schema")
            episodes, logits = z["episode_index"].copy(), z["logits"].copy()
        check(np.array_equal(episodes, final_episodes), "Final NPZ row order differs from identity manifest")
        check(side["logits_sha256"] == array_sha(logits, "<f8"), "Final logit array hash differs")
        generated = emit_csv(out / f"core_a_dynamic_final_test_seed{a_seed}.csv", episodes, logits, final_cols, cutoff)
        receipt["outputs"].append({**generated, "split": "final_test", "core_a_seed": a_seed, "core_b_seed": SEED_PAIRS[a_seed]})
        receipt["sources"].append({"split": "final_test", "core_a_seed": a_seed, "prediction_path": str(path), "prediction_sha256": sha(path), "sidecar_sha256": sha(sidepath), "checkpoint_sha256": row["checkpoint_sha256"], "feature_manifest_sha256": side["feature_manifest_sha256"], "identity_and_source_group_verified_rows": len(episodes)})
    check(len(receipt["outputs"]) == 6, "Expected six decoded CSV outputs")
    receipt["final_prediction_manifest_sha256"] = sha(manifest_path)
    receipt["final_prediction_seal_sha256"] = sha(seal_path)
    receipt["final_identity_sha256"] = sha(identity_path)
    receipt["final_population_metadata_sha256"] = sha(metadata_path)
    (out / "DECODE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "COLUMN_TO_CANONICAL_TASK.json").write_text(json.dumps({str(i): int(t) for i, t in enumerate(final_cols)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "outputs": [{"name": Path(x["path"]).name, "rows": x["rows"], "sha256": x["sha256"]} for x in receipt["outputs"]]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())

#!/usr/bin/env python3
"""CPU semantic/causal tests for the V2 Core B profile, seal and target reader."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
V1 = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
sys.path.insert(0, str(V2 / "code"))
sys.path.insert(0, str(V1 / "code"))

import core_b_predict_v2 as cb
import extract_core_b_current_profile_final_v2 as final_extractor
import extract_core_b_current_profile_v2 as dev_extractor
import score_core_b_v2 as scorer


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(list(ids)).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeQ0:
    def __init__(self, manifest):
        self.manifest = manifest

    def row(self, cache_index, anchor_id, episode_id):
        return np.zeros((2, 16, 4, 768), dtype=np.float32)


class CoreBV2ContractTests(unittest.TestCase):
    def test_anchor_frame_schedule_stops_at_anchor(self):
        with (V2 / "core_b/DEV_CURRENT_PROFILE_INPUTS.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (V2 / "core_b/FINAL_CURRENT_PROFILE_INPUTS.csv").open(encoding="utf-8-sig", newline="") as stream:
            final_rows = list(csv.DictReader(stream))
        for module, manifest_rows in ((dev_extractor, rows), (final_extractor, final_rows)):
            self.assertEqual(len(manifest_rows), 185 if module is dev_extractor else 162)
            for row in manifest_rows:
                anchor = int(row["anchor_frame"])
                selected = module.causal_frame_indices(anchor)
                self.assertEqual(selected.shape, (32,))
                self.assertEqual(len(np.unique(selected)), 32)
                self.assertEqual(int(selected[0]), anchor - 119)
                self.assertEqual(int(selected[-1]), anchor)
                self.assertLessEqual(int(selected.max()), anchor)
                self.assertGreaterEqual(int(selected.min()), 0)

    def _make_grid(self, root: Path, split: str = "dev"):
        lock_path = root / "candidate_lock.json"
        lock_path.write_text("{\"status\":\"synthetic\"}\n", encoding="utf-8")
        lock_sha = sha(lock_path)
        ids = np.asarray([f"canonical-{i}" for i in range(2)], dtype="U64")
        anchors = np.asarray(["a0", "a1"], dtype="U64")
        episodes = np.asarray([10, 11], dtype="<i8")
        tasks = np.asarray([119, 94], dtype="<i8")
        indices = np.asarray([41, 42], dtype="<i8")
        features = np.zeros((2, 2, 16, 4, 768), dtype=np.float32)
        features[0, 0] = 1.0
        features[0, 1] = 2.0
        features[1, 0] = 3.0
        features[1, 1] = 4.0
        archive_path = root / "features.npz"
        np.savez_compressed(archive_path, canonical_ids=ids, anchor_ids=anchors,
                            episode_ids=episodes, task_ids=tasks, cache_indices=indices,
                            grid_features=features)
        manifest = {
            "status": "PASS", "profile_id": "CURRENT_PROFILE_V2", "split": split,
            "profile_lock_sha256": lock_sha, "rows": 2,
            "feature_archive_path": str(archive_path), "feature_archive_sha256": sha(archive_path),
            "canonical_id_order_sha256": ids_sha(ids.tolist()), "anchor_id_order_sha256": ids_sha(anchors.tolist()),
            "shards": [],
        }
        manifest_path = root / ("CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json" if split == "dev" else "CURRENT_PROFILE_V2_FINAL_FEATURE_MANIFEST.json")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return lock_path, lock_sha, manifest_path, manifest, features

    def test_current_profile_reader_checks_cache_index_anchor_episode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_path, lock_sha, manifest_path, manifest, expected = self._make_grid(root)
            with patch.object(cb, "CURRENT_PROFILE_LOCK_PATH", lock_path):
                reader = cb.CurrentProfileGrid(manifest, manifest_path)
            np.testing.assert_array_equal(reader.row(41, "a0", "10"), expected[0])
            with self.assertRaises(cb.PredictorContractError):
                reader.row(41, "wrong-anchor", "10")
            with self.assertRaises(cb.PredictorContractError):
                reader.row(41, "a0", "999")

    def test_v2_dev_allows_only_current_profile_or_original_q0(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_path, lock_sha, manifest_path, manifest, _ = self._make_grid(root)
            source = cb.AcceptedSources(object, lambda *args: None, FakeQ0, lambda model: "state")
            with patch.object(cb, "CURRENT_PROFILE_LOCK_PATH", lock_path), patch.object(cb, "CURRENT_PROFILE_DEV_MANIFEST", manifest_path):
                reader = cb.CurrentProfileGrid(manifest, manifest_path)
                bound = cb.bind_feature_reader(reader, manifest_path=manifest_path,
                    expected_manifest_sha256=sha(manifest_path), split="dev", source=source)
                self.assertIs(bound, reader)
                with self.assertRaises(cb.PredictorContractError):
                    cb.bind_feature_reader(reader, manifest_path=manifest_path,
                        expected_manifest_sha256=sha(manifest_path), split="train", source=source)

            q0_root = root / "q0"
            q0_root.mkdir()
            v9_root = q0_root / "CORE_B_V9_CONTEXT_DROPOUT_V1"
            v9_root.mkdir()
            q0_path = q0_root / "CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
            q0_path.parent.mkdir(parents=True)
            q0_manifest = {"status": "PASS", "final_test_access": 0, "shards": []}
            q0_path.write_text(json.dumps(q0_manifest) + "\n", encoding="utf-8")
            q0_reader = FakeQ0(q0_manifest)
            with patch.object(cb, "ACCEPTED_V9_ROOT", v9_root), patch.object(cb, "ACCEPTED_Q0_CACHE_SHA256", sha(q0_path)):
                bound_q0 = cb.bind_feature_reader(q0_reader, manifest_path=q0_path,
                    expected_manifest_sha256=sha(q0_path), split="dev", source=source)
            self.assertIs(bound_q0, q0_reader)

    def test_final_current_profile_reader_requires_access_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_path, lock_sha, manifest_path, manifest, _ = self._make_grid(root, split="final_test")
            source = cb.AcceptedSources(object, lambda *args: None, FakeQ0, lambda model: "state")
            with patch.object(cb, "CURRENT_PROFILE_LOCK_PATH", lock_path), patch.object(cb, "CURRENT_PROFILE_FINAL_MANIFEST", manifest_path):
                reader = cb.CurrentProfileGrid(manifest, manifest_path)
                with self.assertRaisesRegex(cb.PredictorContractError, "require parent V2 FINAL_ACCESS_OPENED"):
                    cb.bind_feature_reader(reader, manifest_path=manifest_path,
                        expected_manifest_sha256=sha(manifest_path), split="final_test", source=source)

    def test_score_metric_direction_and_task_macro_oracle(self):
        n = 5
        tasks = np.asarray([119, 94, 90, 141, 230], dtype=np.int64)
        episodes = np.arange(100, 105, dtype=np.int64)
        groups = np.asarray([f"g{x}" for x in episodes], dtype="U16")
        anchors = np.asarray([f"a{i}" for i in range(n)], dtype="U16")
        canonical = np.asarray([f"c{i}" for i in range(n)], dtype="U16")
        anchor = np.zeros((n, 28), dtype=np.float64)
        prefix = {"canonical_id": canonical, "anchor_id": anchors, "episode_id": episodes,
                  "task_id": tasks, "source_cluster": groups, "anchor_native": anchor}
        target = np.zeros((n, 30, 28), dtype=np.float64)
        values = {"A0": 0.50, "AD": 0.45, "V0": 0.25, "VD": 0.30,
                  "S_OPT": 0.40, "MIX0": 0.325, "MIXD": 0.35, "PERSISTENCE": 0.0}
        bundles = {}
        for program, value in values.items():
            seeds = (None,) if program == "PERSISTENCE" else (202610131, 202610132, 202610133)
            for seed in seeds:
                bundles[(program, seed)] = np.full((n, 30, 28), value, dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            draws = {f"task_{task}": np.zeros((5000, 1), dtype=np.int32) for task in tasks.tolist()}
            draw_path = Path(td) / "draws.npz"
            np.savez_compressed(draw_path, **draws)
            metric_scale = {"target_sigma": np.ones(28), "safe_fd_scale": np.ones(28)}
            scored = scorer.score_all(prefix, target, bundles, metric_scale, draw_path)
        rows = {(r["contrast"], r["metric"], r.get("grain")): r for r in scored["contrast"]}
        a0_v0 = rows[("A0_minus_V0", "position_mae", "three_seed_task_macro")]
        persist_v0 = rows[("PERSISTENCE_minus_V0", "position_mae", "three_seed_task_macro")]
        self.assertAlmostEqual(a0_v0["reference_minus_candidate_mean"], 0.25, places=12)
        self.assertAlmostEqual(persist_v0["reference_minus_candidate_mean"], -0.25, places=12)
        boot = {(r["contrast"], r["metric"]): r for r in scored["bootstrap"]}
        self.assertAlmostEqual(boot[("A0_minus_V0", "position_mae")]["ci_low"], 0.25, places=12)
        self.assertAlmostEqual(boot[("A0_minus_V0", "position_mae")]["ci_high"], 0.25, places=12)
        self.assertTrue(all(r["safe_fd_l1"] == 0.0 for r in scored["per_anchor"]))

    def test_final_prefix_broker_reads_only_history_through_each_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            input_rows = []
            pop_rows = []
            task_layout = [(119, 12, 4), (94, 12, 4), (90, 8, 4), (90, 1, 2), (141, 5, 4), (230, 3, 4)]
            episode = 0
            for task, episode_count, per_episode in task_layout:
                for _ in range(episode_count):
                    episode += 1
                    relative = f"chunk/episode_{episode:04d}.parquet"
                    parquet = raw / relative
                    parquet.parent.mkdir(parents=True, exist_ok=True)
                    max_anchor = [119, 179, 239, 299][per_episode - 1]
                    frame_index = np.arange(max_anchor + 31, dtype=np.int64)
                    timestamp = frame_index.astype(np.float64) / 30.0
                    action = np.stack([frame_index + 300 + j for j in range(28)], axis=1).astype(np.float32)
                    arm = np.stack([frame_index + j for j in range(14)], axis=1).astype(np.float32)
                    hand = np.stack([frame_index + 100 + j for j in range(14)], axis=1).astype(np.float32)
                    leg = np.stack([frame_index + 200 + j for j in range(15)], axis=1).astype(np.float32)
                    pq.write_table(pa.table({
                        "frame_index": frame_index, "timestamp": timestamp, "action": pa.array(action.tolist()),
                        "observation.arm_joints": pa.array(arm.tolist()),
                        "observation.hand_joints": pa.array(hand.tolist()),
                        "observation.leg_joints": pa.array(leg.tolist()),
                    }), parquet)
                    cluster = f"group-{episode}"
                    for k, anchor_frame in enumerate([119, 179, 239, 299][:per_episode]):
                        anchor_id = f"e{episode:04d}_a{k}_f{anchor_frame:06d}"
                        source_key = f"source-{episode}"
                        canonical = f"rev|{source_key}|{relative}|{episode}|{task}|{anchor_id}|final_test"
                        input_rows.append({
                            "dataset_role": "final_test", "canonical_id": canonical,
                            "source_key": source_key, "literal_source_key": relative,
                            "episode_id": str(episode), "task_id": str(task), "anchor_id": anchor_id,
                            "anchor_frame": str(anchor_frame), "cache_index": str(len(input_rows)),
                            "video_path": f"videos/episode_{episode:04d}.mp4", "parquet_path": relative,
                        })
                        pop_rows.append({
                            "dataset_role": "final_test", "source_key": source_key,
                            "literal_source_key": relative, "source_group_key": cluster,
                            "episode_id": str(episode), "task_id": str(task), "anchor_id": anchor_id,
                            "anchor_frame": str(anchor_frame),
                            "target_h30_row_start_inclusive": str(anchor_frame + 1),
                            "target_h30_row_end_inclusive": str(anchor_frame + 30),
                        })
            self.assertEqual(len(input_rows), 162)
            inputs_path, pop_path = root / "final_inputs.csv", root / "final_population.csv"
            for path, rows in ((inputs_path, input_rows), (pop_path, pop_rows)):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            lock = {
                "inputs": {"final_input_manifest_sha256": sha(inputs_path)},
                "final_inputs": {"final_population_metadata_sha256": sha(pop_path)},
            }
            builder = __import__("prepare_core_b_final_prefix_v2")
            prefix_out = root / "outputs/FINAL_PREFIXES_CURRENT_PROFILE_V2.npz"
            receipt_out = root / "reports/FINAL_PREFIX_EXTRACTION_RECEIPT.json"
            with patch.object(builder, "verify_gates", return_value=(lock, {"status": "FROZEN_BEFORE_FINAL_ACCESS"})), \
                 patch.object(builder, "INPUTS", inputs_path), patch.object(builder, "POP", pop_path), \
                 patch.object(builder, "RAW_ROOT", raw), patch.object(builder, "OUT_PREFIX", prefix_out), \
                 patch.object(builder, "OUT_RECEIPT", receipt_out), patch.object(sys, "argv", [
                     "prepare_core_b_final_prefix_v2.py", "--expected-lock-sha256", "a" * 64,
                     "--expected-final-freeze-sha256", "b" * 64, "--expected-final-access-sha256", "c" * 64,
                 ]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(builder.main(), 0)
            with np.load(prefix_out, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {
                    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster", "cache_index",
                    "anchor_frame", "history_native", "history_mask", "relative_delta", "anchor_native",
                    "canonical_id", "dataset_revision", "source_key", "literal_source_key", "split",
                })
                self.assertEqual(archive["history_native"].shape, (162, 32, 71))
                self.assertEqual(archive["anchor_native"].shape, (162, 28))
                np.testing.assert_array_equal(archive["split"], np.full(162, "final_test"))
                for i in (0, 1, 47, 96, 161):
                    anchor_frame = int(archive["anchor_frame"][i])
                    self.assertEqual(float(archive["relative_delta"][i, -1]), 0.0)
                    self.assertLessEqual(anchor_frame, int(archive["anchor_frame"][i]))
                    self.assertAlmostEqual(float(archive["anchor_native"][i, 0]), anchor_frame + 300.0)
                    self.assertAlmostEqual(float(archive["history_native"][i, -1, 0]), anchor_frame)
            receipt = json.loads(receipt_out.read_text())
            self.assertEqual(receipt["target_h30_rows_materialized"], 0)
            self.assertFalse(receipt["future_target_array_materialized"])
            self.assertTrue(receipt["final_payload_access"])


    def test_final_target_reader_selects_exact_anchor_plus_one_to_thirty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            raw.mkdir()
            episodes = []
            tasks = []
            # Five task episode counts total 41; final anchor support is fixed at 162.
            layout = [(119, 12, 4), (94, 12, 4), (90, 8, 4), (90, 1, 2), (141, 5, 4), (230, 3, 4)]
            anchors_rows = []
            prefix_rows = []
            episode = 0
            for task, episode_count, per_episode in layout:
                for _ in range(episode_count):
                    episode += 1
                    parquet_rel = f"chunk/episode_{episode:04d}.parquet"
                    parquet_path = raw / parquet_rel
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    max_anchor = [119, 179, 239, 299][per_episode - 1]
                    frame_index = np.arange(max_anchor + 31, dtype=np.int64)
                    timestamp = frame_index.astype(np.float64) / 30.0
                    action = np.stack([frame_index + j for j in range(28)], axis=1).astype(np.float32)
                    pq.write_table(pa.table({"frame_index": frame_index, "timestamp": timestamp, "action": pa.array(action.tolist())}), parquet_path)
                    cluster = f"episode-group-{episode}"
                    for k, anchor_frame in enumerate([119, 179, 239, 299][:per_episode]):
                        anchor_id = f"e{episode:04d}_a{k}_f{anchor_frame:06d}"
                        row = {"dataset_revision": "rev", "split": "final_test", "dataset_role": "final_test",
                               "source_key": f"source-{episode}", "literal_source_key": parquet_rel,
                               "source_group_key": cluster, "episode_id": str(episode), "task_id": str(task),
                               "anchor_id": anchor_id, "anchor_frame": str(anchor_frame),
                               "history_frame_start": str(anchor_frame - 119), "history_frame_end": str(anchor_frame),
                               "target_h30_row_start_inclusive": str(anchor_frame + 1),
                               "target_h30_row_end_inclusive": str(anchor_frame + 30)}
                        anchors_rows.append(row)
                        prefix_rows.append(row)
                        episodes.append(episode); tasks.append(task)
            self.assertEqual(len(anchors_rows), 162)
            pop_path = root / "population.csv"
            pop_fields = list(anchors_rows[0])
            with pop_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pop_fields); w.writeheader(); w.writerows(anchors_rows)
            map_path = root / "video_map.csv"
            map_path.write_text("fixed metadata only\n", encoding="utf-8")
            ids = np.asarray([r["anchor_id"] for r in prefix_rows], dtype="U64")
            ep = np.asarray(episodes, dtype=np.int64)
            task_ids = np.asarray(tasks, dtype=np.int64)
            cluster = np.asarray([r["source_group_key"] for r in prefix_rows], dtype="U64")
            parquet_paths = np.asarray([r["literal_source_key"] for r in prefix_rows], dtype="U256")
            anchor_frames = np.asarray([int(r["anchor_frame"]) for r in prefix_rows], dtype=np.int64)
            prefix = {"anchor_id": ids, "episode_id": ep, "task_id": task_ids, "source_cluster": cluster,
                      "parquet_path": parquet_paths, "anchor_frame": anchor_frames,
                      "anchor_native": np.zeros((162, 28), dtype=np.float64)}
            lock = {"final_inputs": {"final_population_metadata_sha256": sha(pop_path),
                                      "final_video_map_sha256": sha(map_path)}}
            latch_path = root / "exposure.json"
            latch = {"target_rows_materialized_current": 0, "target_values_materialized_current": 0,
                     "raw_parquet_action_rows_read_unique": 0, "raw_parquet_files_read": 0}
            with patch.object(scorer, "RAW_ROOT", raw), patch.object(scorer, "FINAL_POP", pop_path), patch.object(scorer, "FINAL_VIDEO_MAP", map_path), patch.object(scorer, "RUN", root):
                got = scorer.read_final_targets(prefix, lock, latch, latch_path)
            self.assertEqual(got.shape, (162, 30, 28))
            for i, anchor_frame in enumerate(anchor_frames.tolist()):
                np.testing.assert_array_equal(got[i, :, 0], np.arange(anchor_frame + 1, anchor_frame + 31))
                np.testing.assert_array_equal(got[i, :, 1], np.arange(anchor_frame + 2, anchor_frame + 32))
            self.assertEqual(latch["target_rows_materialized_current"], 162 * 30)
            self.assertEqual(latch["target_values_materialized_current"], 162 * 30 * 28)
            self.assertEqual(latch["final_payload_reads"], 41)


if __name__ == "__main__":
    unittest.main(verbosity=2)

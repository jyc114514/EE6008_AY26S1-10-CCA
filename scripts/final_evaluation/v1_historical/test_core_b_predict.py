from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import core_b_predict as cb


SOURCE_ROOT = cb.ACCEPTED_V9_CODE_ROOT


class ZeroResidualModel(torch.nn.Module):
    def __init__(self, emit_nan: bool = False) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(28))
        self.emit_nan = emit_nan
        self.seen_prefix_keys: set[str] | None = None

    def forward(self, prefix_inputs, task_id):
        self.seen_prefix_keys = set(prefix_inputs)
        batch = int(prefix_inputs["state"].shape[0])
        output = self.offset.view(1, 1, 28).expand(batch, 30, 28)
        if self.emit_nan:
            output = output * torch.tensor(float("nan"))
        return SimpleNamespace(action_residual=output)


def make_prefix(n_rows: int = 2, *, with_video_metadata: bool = True) -> tuple[dict[str, np.ndarray], list[str]]:
    task_ids = np.asarray([119, 94][:n_rows], dtype=np.int64)
    task_ranks = np.asarray([cb.TASK_IDS.index(int(task)) for task in task_ids], dtype=np.int64)
    episode_ids = np.arange(101, 101 + n_rows, dtype=np.int64)
    anchor_ids = np.asarray([f"anchor-{i}" for i in range(n_rows)], dtype="U16")
    source_keys = np.asarray([f"source-{i}" for i in range(n_rows)], dtype="U16")
    literal_keys = np.asarray([f"parquet/{i}.parquet" for i in range(n_rows)], dtype="U32")
    split = np.repeat("dev", n_rows).astype("U8")
    revision = np.repeat("revision-abc", n_rows).astype("U16")
    canonical_ids = [
        f"{revision[i]}|{source_keys[i]}|{literal_keys[i]}|{episode_ids[i]}|{task_ids[i]}|{anchor_ids[i]}|{split[i]}"
        for i in range(n_rows)
    ]
    data: dict[str, np.ndarray] = {
        "anchor_id": anchor_ids,
        "episode_id": episode_ids,
        "task_id": task_ids,
        "task_rank": task_ranks,
        "source_cluster": source_keys.copy(),
        "cache_index": np.arange(n_rows, dtype=np.int64),
        "anchor_frame": np.full(n_rows, 119, dtype=np.int64),
        "history_native": np.arange(n_rows * 32 * 71, dtype=np.float32).reshape(n_rows, 32, 71) / 1000.0,
        "history_mask": np.ones((n_rows, 32), dtype=np.float32),
        "relative_delta": np.zeros((n_rows, 32), dtype=np.float32),
        "anchor_native": np.arange(n_rows * 28, dtype=np.float32).reshape(n_rows, 28) / 10.0,
        "canonical_id": np.asarray(canonical_ids, dtype="U256"),
        "dataset_revision": revision,
        "source_key": source_keys,
        "literal_source_key": literal_keys,
        "split": split,
    }
    if not with_video_metadata:
        for key in ("dataset_revision", "source_key", "literal_source_key", "split"):
            data.pop(key)
    return data, canonical_ids


def make_normalizer(*, with_video: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "state_mean": np.zeros(71, dtype=np.float32).tolist(),
        "state_safe_std": np.ones(71, dtype=np.float32).tolist(),
        "relative_delta_mean": 0.0,
        "relative_delta_safe_std": 1.0,
        "target_sigma": np.ones(28, dtype=np.float32).tolist(),
    }
    if with_video:
        value["video_mean"] = np.zeros(768, dtype=np.float32).tolist()
        value["video_safe_std"] = np.ones(768, dtype=np.float32).tolist()
    return value


class CoreBPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Imports only pinned source code; it does not access CUDA or model/data payloads.
        cls.sources = cb.load_accepted_sources(SOURCE_ROOT)

    def make_predictor(self, *, mode: str = "STATE-CTRL", emit_nan: bool = False) -> tuple[cb.CoreBPredictor, ZeroResidualModel]:
        program = "A0" if mode == "STATE-CTRL" else "V0"
        model = ZeroResidualModel(emit_nan=emit_nan).eval()
        state_hash = cb.module_state_sha256(model)
        predictor = cb.CoreBPredictor(
            model,
            program_id=program,
            video_mode=mode,
            native_scale=np.ones(28, dtype=np.float32),
            expected_state_sha256=state_hash,
        )
        return predictor, model

    def test_exact_accepted_sources_and_task_order_are_bound(self) -> None:
        self.assertEqual(self.sources.model_class.__name__, "V2TaskConditionedPredictor")
        self.assertEqual(self.sources.model_input.__module__, "v9_common")
        self.assertEqual(self.sources.q0_grid_class.__name__, "Q0ShardedGrid")
        self.assertEqual(cb.TASK_IDS, (119, 94, 90, 141, 230))

    def test_predictor_api_has_no_future_target_parameter(self) -> None:
        parameters = tuple(inspect.signature(cb.CoreBPredictor.__call__).parameters)
        self.assertEqual(parameters, ("self", "prefix_inputs", "task_id"))
        predictor, _ = self.make_predictor()
        with self.assertRaises(TypeError):
            predictor({}, torch.zeros(0, dtype=torch.long), np.zeros((1, 30, 28)))

    def test_prefix_archive_rejects_target_keys_and_bad_hash(self) -> None:
        data, ids = make_prefix()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DEV_PREFIXES_NO_LABELS.npz"
            np.savez(path, **data, target_native_h30=np.zeros((2, 30, 28), dtype=np.float32))
            digest = cb.sha256_file(path)
            with self.assertRaises(cb.PredictorContractError):
                cb.load_prefix_npz(path, expected_sha256=digest, canonical_ids=ids, split="dev")
            np.savez(path, **data)
            with self.assertRaises(cb.PredictorContractError):
                cb.load_prefix_npz(path, expected_sha256="0" * 64, canonical_ids=ids, split="dev")

    def test_prefix_shape_order_finite_and_canonical_binding_checks(self) -> None:
        data, ids = make_prefix()
        validated = cb.validate_prefix_data(data, canonical_ids=ids, split="dev")
        self.assertEqual(validated["history_native"].shape, (2, 32, 71))
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_prefix_data(data, canonical_ids=list(reversed(ids)), split="dev")
        wrong_rank = dict(data)
        wrong_rank["task_rank"] = np.asarray([4, 1], dtype=np.int64)
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_prefix_data(wrong_rank, canonical_ids=ids, split="dev")
        wrong_shape = dict(data)
        wrong_shape["anchor_native"] = np.zeros((2, 27), dtype=np.float32)
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_prefix_data(wrong_shape, canonical_ids=ids, split="dev")
        nonfinite = dict(data)
        nonfinite["history_native"] = data["history_native"].copy()
        nonfinite["history_native"][0, 0, 0] = np.nan
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_prefix_data(nonfinite, canonical_ids=ids, split="dev")
        wrong_key = dict(data)
        wrong_key["labels"] = np.zeros(2, dtype=np.int64)
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_prefix_data(wrong_key, canonical_ids=ids, split="dev")

    def test_runtime_job_rejects_target_label_keys_and_paths(self) -> None:
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_runtime_job({"target_archive_path": "/tmp/future.npz"})
        with self.assertRaises(cb.PredictorContractError):
            cb.validate_runtime_job({"prefix_path": "/tmp/targets/DEV_PREFIXES_NO_LABELS.npz"})
        cb.validate_runtime_job({"prefix_path": "/tmp/DEV_PREFIXES_NO_LABELS.npz", "seed": 202610131})

    def test_model_manifest_identity_checks_fixed_checkpoint_seed_and_mode(self) -> None:
        key = ("A0", 202610131)
        manifest_sha, model_sha, state_hash = cb.PINNED_MODEL_IDENTITIES[key]
        del manifest_sha
        manifest = {
            "status": "PASS",
            "program_id": "A0",
            "seed": 202610131,
            "video_mode": "STATE-CTRL",
            "numeric_context_drop_probability": 0.0,
            "model_file": "/tmp/model.safetensors",
            "model_file_sha256": model_sha,
            "normalizer_sha256": cb.ACCEPTED_NORMALIZER_SHA256,
            "final_state_hash": cb.PINNED_MODEL_IDENTITIES[key][2],
            "teacher_target_used": False,
            "final_heldout_payload_access": 0,
        }
        self.assertEqual(
            cb.validate_model_manifest(
                manifest,
                program_id="A0",
                seed=202610131,
                video_mode="STATE-CTRL",
                model_sha256=model_sha,
                normalizer_sha256=cb.ACCEPTED_NORMALIZER_SHA256,
            ),
            state_hash,
        )
        for kwargs in (
            {"seed": 202610132},
            {"video_mode": "TS11-D"},
            {"model_sha256": "0" * 64},
        ):
            args = {
                "program_id": "A0",
                "seed": 202610131,
                "video_mode": "STATE-CTRL",
                "model_sha256": model_sha,
                "normalizer_sha256": cb.ACCEPTED_NORMALIZER_SHA256,
            }
            args.update(kwargs)
            with self.assertRaises(cb.PredictorContractError):
                cb.validate_model_manifest(manifest, **args)

    def test_S_OPT_uses_the_accepted_five_task_G_S_identity(self) -> None:
        key = ("S_OPT", 202610131)
        _, model_sha, state_hash = cb.PINNED_MODEL_IDENTITIES[key]
        manifest = {
            "status": "PASS",
            "model_id": "G-S",
            "seed": 202610131,
            "video_mode": "STATE-CTRL",
            "waypoint_mode": None,
            "model_file": "/tmp/accepted_s_opt.safetensors",
            "model_file_sha256": model_sha,
            "normalizer_sha256": cb.ACCEPTED_NORMALIZER_SHA256,
            "final_state_hash": state_hash,
            "task_ids": list(cb.TASK_IDS),
            "final_heldout_payload_access": 0,
            "teacher_target_used": False,
        }
        self.assertEqual(
            cb.validate_model_manifest(
                manifest,
                program_id="S_OPT",
                seed=202610131,
                video_mode="STATE-CTRL",
                model_sha256=model_sha,
                normalizer_sha256=cb.ACCEPTED_NORMALIZER_SHA256,
            ),
            state_hash,
        )

    def test_final_feature_reader_requires_exact_parent_access_receipt(self) -> None:
        class ParentFinalReader:
            def __init__(self, manifest):
                self.manifest = manifest

            def row(self, cache_index, anchor_id, episode_id):
                return np.zeros((2, 16, 4, 768), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "protocol"
            protocol.mkdir()
            receipt_path = protocol / "FINAL_ACCESS_OPENED.json"
            receipt = {
                "status": "OPENED_BY_USER_AUTHORIZED_TASK",
                "protocol_id": "synthetic-protocol",
                "authorization_source": "this user request",
                "opened_utc": "2026-10-19T00:00:00Z",
                "study_lock_sha256": "a" * 64,
                "freeze_receipt_sha256": "b" * 64,
                "core_b_access_authorized": True,
                "first_final_payload_access_utc": "2026-10-19T00:00:01Z",
                "access_stage": "synthetic test only",
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            receipt_sha = cb.sha256_file(receipt_path)
            manifest = {
                "status": "PASS",
                "final_test_access": 1,
                "final_access_opened_sha256": receipt_sha,
                "logical_shape": [1, 2, 16, 4, 768],
                "shards": [],
            }
            manifest_path = root / "FINAL_FEATURE_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reader = ParentFinalReader(manifest)
            bound = cb.bind_feature_reader(
                reader,
                manifest_path=manifest_path,
                expected_manifest_sha256=cb.sha256_file(manifest_path),
                split="final_test",
                source=self.sources,
                final_access_opened_path=receipt_path,
                expected_final_access_opened_sha256=receipt_sha,
            )
            self.assertIs(bound, reader)
            # Every final entrypoint refuses before touching model/prefix inputs.
            with self.assertRaises(cb.PredictorContractError):
                cb.load_prefix_npz(
                    root / "missing_FINAL_PREFIXES_NO_LABELS.npz",
                    expected_sha256="0" * 64,
                    canonical_ids=[],
                    split="final_test",
                )
            predictor, _ = self.make_predictor()
            with self.assertRaises(cb.PredictorContractError):
                cb.predict_rows(
                    {}, canonical_ids=[], split="final_test", predictor=predictor,
                    normalizer=make_normalizer(), model_input_fn=self.sources.model_input,
                    device="cpu",
                )
            dummy_cell = cb.ModelCell(
                "A0", 202610131, "STATE-CTRL", root / "missing_model.safetensors", "0" * 64,
                root / "missing_manifest.json", "0" * 64, root / "missing_normalizer.json",
                cb.ACCEPTED_NORMALIZER_SHA256, root / "missing_FINAL_PREFIXES_NO_LABELS.npz", "0" * 64,
            )
            with self.assertRaises(cb.PredictorContractError):
                cb.predict_cell(dummy_cell, canonical_ids=[], split="final_test")
            authorization = cb.verify_final_access_receipt(receipt_path, receipt_sha)
            closed_branch_receipt = dict(receipt, core_b_access_authorized=False)
            receipt_path.write_text(json.dumps(closed_branch_receipt), encoding="utf-8")
            with self.assertRaises(cb.PredictorContractError):
                cb.verify_final_access_receipt(receipt_path, cb.sha256_file(receipt_path))
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaises(cb.PredictorContractError):
                cb.bind_feature_reader(
                    reader,
                    manifest_path=manifest_path,
                    expected_manifest_sha256=cb.sha256_file(manifest_path),
                    split="final_test",
                    source=self.sources,
                )
            with self.assertRaises(cb.PredictorContractError):
                cb.load_prefix_npz(
                    root / "missing_DEV_PREFIXES_NO_LABELS.npz",
                    expected_sha256="0" * 64,
                    canonical_ids=[],
                    split="dev",
                    final_access_authorization=authorization,
                )
            bad_receipt = dict(receipt, status="PROPOSED")
            receipt_path.write_text(json.dumps(bad_receipt), encoding="utf-8")
            with self.assertRaises(cb.PredictorContractError):
                cb.bind_feature_reader(
                    reader,
                    manifest_path=manifest_path,
                    expected_manifest_sha256=cb.sha256_file(manifest_path),
                    split="final_test",
                    source=self.sources,
                    final_access_opened_path=receipt_path,
                    expected_final_access_opened_sha256=cb.sha256_file(receipt_path),
                )

    def test_file_hash_verification_rejects_manifest_normalizer_prefix_or_cache_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role in ("model", "model manifest", "normalizer", "prefix", "feature cache manifest"):
                file_path = root / f"{role.replace(' ', '_')}.bin"
                file_path.write_bytes(role.encode())
                expected = hashlib.sha256(role.encode()).hexdigest()
                self.assertEqual(cb.verify_file_sha256(file_path, expected, role), expected)
                with self.assertRaises(cb.PredictorContractError):
                    cb.verify_file_sha256(file_path, "f" * 64, role)

    def test_core_predictor_returns_ordered_float64_and_preserves_model_and_input(self) -> None:
        data, ids = make_prefix()
        before = {key: value.copy() for key, value in data.items()}
        predictor, model = self.make_predictor()
        prediction = cb.predict_rows(
            data,
            canonical_ids=ids,
            split="dev",
            predictor=predictor,
            normalizer=make_normalizer(),
            model_input_fn=self.sources.model_input,
            device="cpu",
            batch_size=1,
        )
        self.assertEqual(prediction.shape, (2, 30, 28))
        self.assertEqual(prediction.dtype, np.float64)
        self.assertTrue(np.isfinite(prediction).all())
        expected = np.repeat(data["anchor_native"][:, None, :], 30, axis=1)
        np.testing.assert_array_equal(prediction, expected)
        self.assertEqual(model.seen_prefix_keys, {"state", "history_mask", "relative_delta", "anchor_command"})
        self.assertEqual(cb.module_state_sha256(model), predictor._expected_state_sha256)
        for key in before:
            np.testing.assert_array_equal(data[key], before[key])

    def test_prefix_target_field_never_reaches_model_and_nonfinite_output_fails(self) -> None:
        data, ids = make_prefix()
        predictor, model = self.make_predictor()
        with self.assertRaises(cb.PredictorContractError):
            cb.predict_rows(
                {**data, "future_Y": np.zeros((2, 30, 28), dtype=np.float32)},
                canonical_ids=ids,
                split="dev",
                predictor=predictor,
                normalizer=make_normalizer(),
                model_input_fn=self.sources.model_input,
                device="cpu",
            )
        nan_predictor, _ = self.make_predictor(emit_nan=True)
        with self.assertRaises(cb.PredictorContractError):
            cb.predict_rows(
                data,
                canonical_ids=ids,
                split="dev",
                predictor=nan_predictor,
                normalizer=make_normalizer(),
                model_input_fn=self.sources.model_input,
                device="cpu",
            )
        self.assertIsNotNone(model)

    def test_q0_reader_uses_synthetic_shard_and_rejects_final_role(self) -> None:
        data, ids = make_prefix(n_rows=1, with_video_metadata=False)
        data["cache_index"] = np.asarray([0], dtype=np.int64)
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        root = Path(temp_context.name)
        array_path = root / "grid_000000_000001.npy"
        meta_path = root / "grid_000000_000001.json"
        np.save(array_path, np.zeros((1, 2, 16, 4, 768), dtype=np.float32), allow_pickle=False)
        meta_path.write_text(json.dumps({
            "status": "PASS",
            "final_test_access": 0,
            "start_row": 0,
            "end_row": 1,
            "anchor_ids": [str(data["anchor_id"][0])],
            "episode_index": [int(data["episode_id"][0])],
        }), encoding="utf-8")
        manifest = {
            "status": "PASS",
            "final_test_access": 0,
            "logical_shape": [1, 2, 16, 4, 768],
            "shards": [{
                "start_row": 0, "end_row": 1,
                "array": str(array_path), "meta": str(meta_path),
            }],
        }
        grid = self.sources.q0_grid_class(manifest)
        predictor, model = self.make_predictor(mode="TS11-D")
        prediction = cb.predict_rows(
            data,
            canonical_ids=ids,
            split="dev",
            predictor=predictor,
            normalizer=make_normalizer(with_video=True),
            model_input_fn=self.sources.model_input,
            feature_reader=grid,
            device="cpu",
        )
        self.assertEqual(prediction.shape, (1, 30, 28))
        self.assertTrue({"dynamic", "static"}.issubset(model.seen_prefix_keys or set()))
        with self.assertRaises(RuntimeError):
            self.sources.q0_grid_class({**manifest, "final_test_access": 1})

    def test_bundle_shape_finite_order_and_safe_write(self) -> None:
        data, ids = make_prefix()
        prediction = np.zeros((2, 30, 28), dtype=np.float64)
        bundle = cb.build_prediction_bundle(ids, prediction, program_id="V0", seed=202610131)
        self.assertEqual(bundle.canonical_ids.tolist(), ids)
        self.assertEqual(bundle.prediction_native.dtype, np.float64)
        with self.assertRaises(cb.PredictorContractError):
            cb.build_prediction_bundle(ids, np.zeros((2, 29, 28)), program_id="V0", seed=202610131)
        with self.assertRaises(cb.PredictorContractError):
            cb.build_prediction_bundle(ids, np.full((2, 30, 28), np.nan), program_id="V0", seed=202610131)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "PREDICTIONS.npz"
            receipt = cb.write_prediction_bundle(output, bundle)
            self.assertEqual(receipt["shape"], [2, 30, 28])
            self.assertEqual(receipt["dtype"], "float64")
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"canonical_ids", "prediction_native"})
                self.assertEqual(archive["canonical_ids"].astype(str).tolist(), ids)
            with self.assertRaises(cb.PredictorContractError):
                cb.write_prediction_bundle(output, bundle)

    def test_persistence_is_one_reference_with_seed_aliases(self) -> None:
        data, ids = make_prefix()
        bundle = cb.make_persistence_bundle(ids, data["anchor_native"])
        expected = np.repeat(data["anchor_native"][:, None, :], 30, axis=1)
        np.testing.assert_array_equal(bundle.prediction_native, expected)
        aliases = cb.make_persistence_aliases(bundle)
        self.assertEqual(tuple(alias.seed for alias in aliases), cb.SEEDS)
        self.assertEqual({alias.prediction_native_sha256 for alias in aliases}, {bundle.prediction_native_sha256})
        self.assertEqual({alias.alias_of for alias in aliases}, {bundle.bundle_sha256})
        self.assertIsNone(bundle.seed)


class CoreBPathSafetyRegressionTests(unittest.TestCase):
    def test_only_exact_accepted_normalizer_path_is_allowlisted(self) -> None:
        expected = cb.ACCEPTED_NORMALIZER_PATH.resolve()
        self.assertEqual(cb._safe_path(expected, "artifact"), expected)
        with self.assertRaises(cb.PredictorContractError):
            cb._safe_path(expected.parent / "target_archive.npz", "artifact")


if __name__ == "__main__":
    unittest.main(verbosity=2)


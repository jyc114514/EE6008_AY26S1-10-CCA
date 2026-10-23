from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import core_a_extract as extract
import core_a_predict as predict
import numpy as np
import pytest
import torch
from torch import Tensor, nn


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, predict.EXPECTED_CLASS_COUNT)

    def forward(self, tokens: Tensor) -> Tensor:
        assert tokens.ndim == 3
        assert tokens.shape[1:] == predict.EXPECTED_TOKEN_SHAPE
        return self.linear(tokens.mean(dim=1))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _feature_file(path: Path, episode_index: int, tokens: Tensor) -> None:
    torch.save(
        {
            "episode_index": episode_index,
            "tokens": tokens,
            "pooled": torch.zeros(3),
            # The predictor intentionally never reads metadata as model input.
            "metadata": {"label": "synthetic-only", "target": [123]},
        },
        path,
    )


def test_manifest_is_exactly_two_columns_ordered_and_unique(tmp_path: Path) -> None:
    one = tmp_path / "one.pt"
    two = tmp_path / "two.pt"
    _feature_file(one, 7, torch.zeros(predict.EXPECTED_TOKEN_SHAPE))
    _feature_file(two, 9, torch.zeros(predict.EXPECTED_TOKEN_SHAPE))
    manifest = tmp_path / "features.csv"
    _write_csv(
        manifest,
        ["episode_index", "feature_path"],
        [["7", str(one)], ["9", str(two)]],
    )
    rows = predict.read_feature_manifest(manifest)
    assert [row.episode_index for row in rows] == [7, 9]

    duplicate = tmp_path / "duplicate.csv"
    _write_csv(
        duplicate,
        ["episode_index", "feature_path"],
        [["7", str(one)], ["7", str(two)]],
    )
    with pytest.raises(ValueError, match="duplicate"):
        predict.read_feature_manifest(duplicate)

    reordered = tmp_path / "reordered.csv"
    _write_csv(
        reordered,
        ["episode_index", "feature_path"],
        [["9", str(two)], ["7", str(one)]],
    )
    assert [row.episode_index for row in predict.read_feature_manifest(reordered)] == [9, 7]


def test_manifest_rejects_label_target_and_label_map_columns(tmp_path: Path) -> None:
    feature = tmp_path / "one.pt"
    _feature_file(feature, 2, torch.zeros(predict.EXPECTED_TOKEN_SHAPE))
    for forbidden in ("label", "target", "label_map"):
        manifest = tmp_path / f"{forbidden}.csv"
        _write_csv(
            manifest,
            ["episode_index", "feature_path", forbidden],
            [["2", str(feature), "not-used"]],
        )
        with pytest.raises(ValueError, match="exactly"):
            predict.read_feature_manifest(manifest)


def test_feature_record_id_shape_and_finiteness_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predict, "EXPECTED_TOKEN_SHAPE", (2, 3))
    path = tmp_path / "record.pt"
    _feature_file(path, 10, torch.zeros(2, 3))
    tokens, digest = predict.load_feature_tokens(predict.FeatureRow(10, path))
    assert tuple(tokens.shape) == (2, 3)
    assert digest == predict.sha256_file(path)

    with pytest.raises(ValueError, match="ID mismatch"):
        predict.load_feature_tokens(predict.FeatureRow(11, path))

    _feature_file(path, 10, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="expected tokens"):
        predict.load_feature_tokens(predict.FeatureRow(10, path))

    _feature_file(path, 10, torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]))
    with pytest.raises(FloatingPointError, match="non-finite"):
        predict.load_feature_tokens(predict.FeatureRow(10, path))


def test_checkpoint_hash_and_strict_keys_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Linear(3, 2)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    actual_sha = predict.sha256_file(checkpoint)

    original_load = torch.load
    calls: list[bool] = []

    def counted_load(*args: object, **kwargs: object) -> object:
        calls.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        predict.load_checkpoint_state(checkpoint, "0" * 64)
    assert calls == []

    state, observed = predict.load_checkpoint_state(checkpoint, actual_sha)
    assert observed == actual_sha
    predict.strict_load_model(nn.Linear(3, 2), state)
    with pytest.raises(RuntimeError):
        predict.strict_load_model(nn.Linear(3, 3), state)

    missing = {"weight": torch.zeros(2, 3)}
    with pytest.raises(RuntimeError):
        predict.strict_load_model(nn.Linear(3, 2), missing)
    unexpected = {**state, "extra.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError):
        predict.strict_load_model(nn.Linear(3, 2), unexpected)


def test_cpu_prediction_writes_float64_logits_and_verified_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predict, "EXPECTED_TOKEN_SHAPE", (2, 3))
    checkpoint_model = TinyClassifier()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": checkpoint_model.state_dict()}, checkpoint)
    checkpoint_sha = predict.sha256_file(checkpoint)

    feature_paths: list[Path] = []
    for episode_index in (3, 8):
        path = tmp_path / f"episode_{episode_index}.pt"
        values = torch.full((2, 3), float(episode_index), dtype=torch.float16)
        _feature_file(path, episode_index, values)
        feature_paths.append(path)
    manifest = tmp_path / "input.csv"
    _write_csv(
        manifest,
        ["episode_index", "feature_path"],
        [["3", str(feature_paths[0])], ["8", str(feature_paths[1])]],
    )
    output = tmp_path / "predictions" / "core_a.npz"
    report = predict.predict_manifest(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        run_id="cpu-fixture-1",
        output_path=output,
        device="cpu",
        expected_rows=2,
        model_factory=TinyClassifier,
        output_root=tmp_path,
    )
    with np.load(output, allow_pickle=False) as arrays:
        ids = arrays["episode_index"]
        logits = arrays["logits"]
    assert ids.tolist() == [3, 8]
    assert logits.shape == (2, predict.EXPECTED_CLASS_COUNT)
    assert logits.dtype == np.dtype("<f8")
    assert np.isfinite(logits).all()
    assert report["episode_index_sha256"] == hashlib.sha256(
        np.ascontiguousarray(ids, dtype="<i8").tobytes()
    ).hexdigest()
    assert report["logits_sha256"] == hashlib.sha256(
        np.ascontiguousarray(logits, dtype="<f8").tobytes()
    ).hexdigest()
    assert report["output_sha256"] == predict.sha256_file(output)
    assert output.with_suffix(".json").is_file()
    assert json.loads(output.with_suffix(".json").read_text())["logits_dtype"] == "float64"


def test_extract_manifest_never_accepts_label_fields_or_duplicate_ids(
    tmp_path: Path
) -> None:
    g1_root = tmp_path / "g1"
    g1_root.mkdir()
    video_a = g1_root / "a.mp4"
    video_b = g1_root / "b.mp4"
    video_a.write_bytes(b"fixture")
    video_b.write_bytes(b"fixture")
    manifest = tmp_path / "episodes.csv"
    _write_csv(
        manifest,
        ["episode_index", "video_path"],
        [["1", "a.mp4"], ["2", "b.mp4"]],
    )
    rows = extract.read_episode_manifest(manifest, g1_root=g1_root)
    assert [row.episode_index for row in rows] == [1, 2]

    labeled = tmp_path / "labeled.csv"
    _write_csv(
        labeled,
        ["episode_index", "video_path", "label"],
        [["1", "a.mp4", "x"]],
    )
    with pytest.raises(ValueError, match="exactly"):
        extract.read_episode_manifest(labeled, g1_root=g1_root)

    duplicate = tmp_path / "duplicate.csv"
    _write_csv(
        duplicate,
        ["episode_index", "video_path"],
        [["1", "a.mp4"], ["1", "b.mp4"]],
    )
    with pytest.raises(ValueError, match="duplicate"):
        extract.read_episode_manifest(duplicate, g1_root=g1_root)


def test_final_extraction_requires_parent_receipt_before_manifest_read(
    tmp_path: Path
) -> None:
    output_root = tmp_path / "isolated"
    protocol = output_root / "protocol"
    manifest = protocol / "CORE_A_FINAL_VIDEO_INDEX.csv"
    with pytest.raises(FileNotFoundError, match="FINAL_ACCESS_OPENED"):
        extract.authorize_manifest(
            mode="final",
            manifest_path=manifest,
            receipt_path=None,
            freeze_receipt_path=protocol / "FREEZE_RECEIPT.json",
            output_root=output_root,
        )


def test_final_receipt_binds_locked_population_and_freeze_hashes(tmp_path: Path) -> None:
    output_root = tmp_path / "isolated"
    protocol = output_root / "protocol"
    protocol.mkdir(parents=True)
    manifest = protocol / "CORE_A_FINAL_VIDEO_INDEX.csv"
    manifest.write_text("episode_index,video_path\n12,v.mp4\n", encoding="utf-8")
    input_sha = extract.sha256_file(manifest)
    protocol_id = "EE6008_FINAL_TEST_CPU_FIXTURE"
    lock_path = protocol / "FINAL_STUDY_LOCK.json"
    lock_path.write_text(
        json.dumps(
            {
                "protocol_id": protocol_id,
                "label_free_final_index_sha256": input_sha,
                "final_access_opened_receipt_required": True,
                "selection_status": "locked_by_task_protocol",
                "engineering_code_hashes_status": "bound_to_freeze_receipt",
            }
        ),
        encoding="utf-8",
    )
    freeze_path = protocol / "FREEZE_RECEIPT.json"
    freeze_path.write_text(
        json.dumps({
            "protocol_id": protocol_id,
            "status": "FROZEN_BEFORE_FINAL_ACCESS",
            "study_lock_sha256": extract.sha256_file(lock_path),
            "input_sha256": {"core_a_final_video_index": input_sha},
        }),
        encoding="utf-8",
    )
    receipt_path = protocol / "FINAL_ACCESS_OPENED.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "OPENED_BY_USER_AUTHORIZED_TASK",
                "protocol_id": protocol_id,
                "core_a_access_authorized": True,
                "core_b_access_authorized": False,
                "authorization_source": "synthetic fixture",
                "opened_utc": "2026-10-19T00:00:00Z",
                "study_lock_sha256": extract.sha256_file(lock_path),
                "freeze_receipt_sha256": extract.sha256_file(freeze_path),
                "access_stage": "final_feature_extraction",
            }
        ),
        encoding="utf-8",
    )
    assert (
        extract.authorize_manifest(
            mode="final",
            manifest_path=manifest,
            receipt_path=receipt_path,
            freeze_receipt_path=freeze_path,
            output_root=output_root,
        )
        == input_sha
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["study_lock_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PermissionError, match="study_lock_sha256"):
        extract.authorize_manifest(
            mode="final",
            manifest_path=manifest,
            receipt_path=receipt_path,
            freeze_receipt_path=freeze_path,
            output_root=output_root,
        )


def test_repeated_first_frame_condition_is_exact() -> None:
    clip = torch.arange(3 * 32 * 2 * 2, dtype=torch.float32).reshape(3, 32, 2, 2)
    dynamic = extract.make_condition_clip(clip, "dynamic")
    static = extract.make_condition_clip(clip, "static_repeated_first_frame")
    assert dynamic.data_ptr() == clip.data_ptr()
    assert torch.equal(static[:, 0], static[:, 1:][:, 0])
    assert torch.equal(static, static[:, :1].expand_as(static))



def test_cpu_extract_writes_aligned_dynamic_and_static_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extract, "EXPECTED_FRAME_COUNT", 2)
    monkeypatch.setattr(extract, "EXPECTED_RESOLUTION", 2)
    monkeypatch.setattr(extract, "EXPECTED_TOKEN_SHAPE", (1, 3))
    monkeypatch.setattr(extract, "EXPECTED_POOLED_SHAPE", (3,))
    monkeypatch.setattr(predict, "EXPECTED_TOKEN_SHAPE", (1, 3))
    config = SimpleNamespace(
        observation_seconds=4.0,
        sample_fps=8,
        source_fps=30.0,
        input_resolution=2,
        frame_count=2,
        model_name="synthetic",
        checkpoint_key="ema_encoder",
        dataset_revision="synthetic",
    )
    hashes = {
        "input_manifest_sha256": "c" * 64,
        "config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "source_commit": "d" * 40,
    }
    clip = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2)

    class Sample:
        frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        frame_indices = np.asarray([0, 4])
        timestamps_seconds = np.asarray([0.0, 0.125])
        source_fps = 30.0
        decoder = "synthetic"

    def decode_fn(*args: object, **kwargs: object) -> Sample:
        return Sample()

    def preprocess_fn(frames: np.ndarray, resolution: int) -> Tensor:
        assert resolution == 2
        return clip.clone()

    def encode_fn(model: nn.Module, value: Tensor) -> tuple[Tensor, Tensor]:
        mean = value.mean()
        tokens = torch.full((1, 3), mean, dtype=torch.float32)
        return tokens, tokens[0]

    receipt = extract.extract_rows(
        rows=[extract.EpisodeRow(4, tmp_path / "4.mp4")],
        config=config,
        hashes=hashes,
        run_id="cpu-extract-1",
        output_dir=tmp_path / "features",
        model=nn.Identity(),
        device=torch.device("cpu"),
        decode_fn=decode_fn,
        preprocess_fn=preprocess_fn,
        encode_fn=encode_fn,
        mode="dev",
    )
    assert receipt["status"] == "FEATURE_EXTRACTION_COMPLETE"
    assert receipt["mode"] == "dev"
    for condition in extract.CONDITIONS:
        manifest = tmp_path / "features" / f"feature_manifest_{condition}.csv"
        rows = predict.read_feature_manifest(manifest)
        assert [row.episode_index for row in rows] == [4]
        tokens, _ = predict.load_feature_tokens(rows[0])
        assert tuple(tokens.shape) == (1, 3)
    dynamic = torch.load(
        tmp_path / "features" / "dynamic" / "episode_fixture_000004.pt",
        map_location="cpu",
        weights_only=True,
    )["tokens"]
    static = torch.load(
        tmp_path / "features" / "static_repeated_first_frame" / "episode_fixture_000004.pt",
        map_location="cpu",
        weights_only=True,
    )["tokens"]
    assert not torch.equal(dynamic, static)

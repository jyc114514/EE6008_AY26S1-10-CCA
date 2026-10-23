from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from sklearn.metrics import f1_score
from torch import nn

import core_a_extract as extract
import score_final_core_a as score
import seal_final_core_a as seal
import run_final_core_a as runner


def test_first_final_access_callback_runs_before_decoder_and_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "EXPECTED_FRAME_COUNT", 2)
    monkeypatch.setattr(extract, "EXPECTED_RESOLUTION", 2)
    monkeypatch.setattr(extract, "EXPECTED_TOKEN_SHAPE", (2, 3))
    monkeypatch.setattr(extract, "EXPECTED_POOLED_SHAPE", (3,))
    events: list[str] = []
    timestamp = "2026-10-19T06:00:00.123456Z"
    sample = SimpleNamespace(
        frames=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.125], dtype=np.float64),
        source_fps=8.0,
        decoder="synthetic",
    )

    def decode(_path: Path, **_kwargs):
        events.append("decode")
        return sample

    def preprocess(_frames: np.ndarray, _resolution: int) -> torch.Tensor:
        return torch.zeros((3, 2, 2, 2), dtype=torch.float32)

    def encode(_model: nn.Module, _clip: torch.Tensor):
        return torch.zeros((2, 3), dtype=torch.float32), torch.zeros((3,), dtype=torch.float32)

    def mark_access() -> str:
        events.append("access_timestamp")
        return timestamp

    config = SimpleNamespace(
        observation_seconds=0.25,
        sample_fps=8,
        source_fps=8.0,
        input_resolution=2,
        frame_count=2,
        model_name="synthetic",
        checkpoint_key="synthetic",
        dataset_revision="synthetic",
    )
    result = extract.extract_rows(
        rows=[extract.EpisodeRow(1, tmp_path / "video.mp4")],
        config=config,
        hashes={"input_manifest_sha256": "a" * 64, "config_sha256": "b" * 64, "checkpoint_sha256": "c" * 64, "source_commit": "d" * 40},
        run_id="fixture-final",
        output_dir=tmp_path / "features",
        model=nn.Identity(),
        device=torch.device("cpu"),
        decode_fn=decode,
        preprocess_fn=preprocess,
        encode_fn=encode,
        mode="final",
        first_payload_access_callback=mark_access,
    )
    assert events == ["access_timestamp", "decode"]
    assert result["first_final_payload_access_utc"] == timestamp
    assert result["row_count"] == 1
    assert result["video_decoder_invocations"] == 1
    assert result["decoder_invocations_by_backend"] == {"synthetic": 1}
    assert result["selected_source_frames_returned"] == 2
    assert result["opencv_sequential_frame_reads"] == 0
    assert (tmp_path / "features/EXTRACTION_RECEIPT.json").is_file()


def test_locked_label_reader_filters_to_exact_requested_final_ids(tmp_path: Path) -> None:
    path = tmp_path / "labels.parquet"
    table = pa.table({
        "episode_index": [10, 11, 12, 13],
        "split": ["final_test", "final_test", "train", "final_test"],
        "contiguous_label_index": [1, 2, 3, 4],
    })
    pq.write_table(table, path)
    rows = score.read_locked_labels(path, np.asarray([11, 13], dtype=np.int64))
    assert {(row["episode_index"], row["contiguous_label_index"]) for row in rows} == {(11, 2), (13, 4)}
    assert len(rows) == 2


def test_fixed_class_numpy_macro_f1_matches_sklearn_oracle() -> None:
    labels = np.asarray([0, 1, 2, 2, 2], dtype=np.int64)
    predicted = np.asarray([0, 2, 2, 1, 0], dtype=np.int64)
    classes = np.arange(120, dtype=np.int64)
    expected = f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)
    assert score._macro_f1_fixed_classes(labels, predicted, classes) == pytest.approx(expected)


def test_bootstrap_is_reproducible_and_pairs_the_same_cluster_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "BOOTSTRAP_DRAWS", 50)
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    preds = {}
    for seed in score.SEEDS:
        preds[("A_DYNAMIC", seed)] = np.asarray([0, 1, 2, 1, 1, 2], dtype=np.int64)
        preds[("A_STATIC", seed)] = np.asarray([0, 2, 2, 0, 1, 0], dtype=np.int64)
    clusters = [f"cluster-{i}" for i in range(6)]
    one = score._bootstrap_contrast(labels, preds, clusters)
    two = score._bootstrap_contrast(labels, preds, clusters)
    assert one == two
    assert set(one) == {"top1_accuracy", "macro_f1"}


def test_final_score_fails_closed_without_prediction_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "OUT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        score.verify_seal()
    path = tmp_path / "predictions/CORE_A_PREDICTION_SEAL.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "NOT_SEALED", "prediction_count": 0}), encoding="utf-8")
    with pytest.raises(PermissionError, match="not sealed"):
        score.verify_seal()

def test_seal_binds_both_feature_manifests_to_extraction_receipt(tmp_path: Path) -> None:
    dynamic = tmp_path / "feature_manifest_dynamic.csv"
    static = tmp_path / "feature_manifest_static_repeated_first_frame.csv"
    dynamic.write_text("episode_index,feature_path\n1,dynamic.pt\n", encoding="utf-8")
    static.write_text("episode_index,feature_path\n1,static.pt\n", encoding="utf-8")
    receipt = {
        "feature_manifest_sha256": {
            "dynamic": seal.sha256_file(dynamic),
            "static_repeated_first_frame": seal.sha256_file(static),
        }
    }
    validated = seal.verify_feature_manifests(receipt, tmp_path)
    assert validated["A_DYNAMIC"][0] == "dynamic"
    assert validated["A_STATIC"][0] == "static_repeated_first_frame"
    receipt["feature_manifest_sha256"]["dynamic"] = "0" * 64
    with pytest.raises(ValueError, match="feature manifest differs"):
        seal.verify_feature_manifests(receipt, tmp_path)


def test_runner_allows_only_watcher_created_log_in_attempt_root(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    runner.verify_watcher_output_root(empty)
    empty.mkdir()
    (empty / "watcher_child.log").write_text("watcher metadata only\n", encoding="utf-8")
    runner.verify_watcher_output_root(empty)
    (empty / "unrelated.bin").write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="unexpected data"):
        runner.verify_watcher_output_root(empty)


def test_runner_requires_project_runtime_library_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="must include the project runtime library"):
        runner.verify_runtime_environment()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(runner.PROJECT / "envs/jyc/lib"))
    runner.verify_runtime_environment()

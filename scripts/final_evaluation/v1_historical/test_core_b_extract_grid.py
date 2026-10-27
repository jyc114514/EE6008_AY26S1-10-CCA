from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import core_b_extract_grid as extract


FIELDS = (
    "dataset_role", "canonical_id", "source_key", "literal_source_key",
    "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index",
    "video_path", "parquet_path",
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(role: str, i: int = 0) -> dict[str, str]:
    return {
        "dataset_role": role,
        "canonical_id": f"revision|source-{i}|data/episode_{i}.parquet|{i}|119|a{i}|{role}",
        "source_key": f"source-{i}",
        "literal_source_key": f"data/episode_{i}.parquet",
        "episode_id": str(i),
        "task_id": "119",
        "anchor_id": f"a{i}",
        "anchor_frame": "119",
        "cache_index": str(i),
        "video_path": f"videos/episode_{i}.mp4",
        "parquet_path": f"data/episode_{i}.parquet",
    }


def test_wanted_frames_are_fixed_and_never_after_anchor() -> None:
    frames = extract.wanted_frame_indices(179)
    assert frames.shape == (32,)
    assert int(frames[0]) == 60
    assert int(frames[-1]) == 179
    assert np.all(frames <= 179)
    np.testing.assert_array_equal(frames, np.floor(np.linspace(60, 179, 32)).astype(np.int64))


def test_video_decode_stops_exactly_at_registered_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic")
    state = {"calls": 0}

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.open = True

        def isOpened(self) -> bool:
            return self.open

        def read(self):
            value = state["calls"]
            state["calls"] += 1
            return True, np.full((2, 2, 3), value % 255, dtype=np.uint8)

        def release(self) -> None:
            self.open = False

    monkeypatch.setattr(extract.cv2, "VideoCapture", FakeCapture)
    wanted = np.asarray([0, 4, 9], dtype=np.int64)
    frames = extract.read_frames_through(video, wanted, max_frame=9)
    assert set(frames) == {0, 4, 9}
    assert state["calls"] == 10


def test_manifest_rejects_wrong_role_and_duplicate_canonical_id(tmp_path: Path) -> None:
    final_path = tmp_path / "final.csv"
    _write_rows(final_path, [_row("final_test")])
    with pytest.raises(ValueError, match="non-train/dev"):
        extract.read_rows(final_path, mode="bridge")

    duplicate_path = tmp_path / "duplicate.csv"
    first = _row("dev", 1)
    second = _row("dev", 2)
    second["canonical_id"] = first["canonical_id"]
    _write_rows(duplicate_path, [first, second])
    with pytest.raises(ValueError, match="duplicate canonical"):
        extract.read_rows(duplicate_path, mode="bridge")


def test_final_receipt_path_gate_runs_before_manifest_open(tmp_path: Path) -> None:
    nonexistent_manifest = tmp_path / "final-video-map.csv"
    with pytest.raises(ValueError, match="canonical FREEZE_RECEIPT"):
        extract._verify_receipts(
            freeze_receipt_path=tmp_path / "not-a-freeze.json",
            final_access_receipt_path=tmp_path / "not-opened.json",
            input_manifest_path=nonexistent_manifest,
        )
    assert not nonexistent_manifest.exists()

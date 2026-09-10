"""Read-only inventory and integrity audit for the local G1 snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EPISODE_RE = re.compile(r"episode_(\d+)\.(parquet|mp4)$")
VECTOR_COLUMNS = {
    "observation.arm_joints": "arm_dim",
    "observation.hand_joints": "hand_dim",
    "observation.leg_joints": "leg_dim",
    "action": "action_dim",
}
REQUIRED_COLUMNS = {
    "observation.arm_joints",
    "observation.hand_joints",
    "observation.leg_joints",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def suffix_class(value: Any) -> str:
    text = str(value or "").casefold()
    if "_g1" in text:
        return "_g1"
    if "_h1" in text:
        return "_h1"
    return "none"


def parse_episode_index(path: Path) -> int | None:
    match = EPISODE_RE.search(path.name)
    return int(match.group(1)) if match else None


def ffprobe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result: dict[str, Any] = {
        "ffprobe_exit": completed.returncode,
        "ffprobe_seconds": round(time.monotonic() - started, 4),
    }
    if completed.returncode != 0:
        result["ffprobe_error"] = completed.stderr.strip()[:1000]
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result["ffprobe_error"] = f"invalid ffprobe JSON: {exc}"
        return result
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    result.update(
        {
            "video_codec": stream.get("codec_name"),
            "video_codec_type": stream.get("codec_type"),
            "video_width": stream.get("width"),
            "video_height": stream.get("height"),
            "video_pix_fmt": stream.get("pix_fmt"),
            "video_avg_frame_rate": stream.get("avg_frame_rate"),
            "video_r_frame_rate": stream.get("r_frame_rate"),
            "video_nb_frames": stream.get("nb_frames") or stream.get("nb_read_frames"),
            "video_duration_seconds": _float_or_none(
                stream.get("duration") or fmt.get("duration")
            ),
            "video_size_bytes": _int_or_none(fmt.get("size")),
        }
    )
    return result


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vector_stats(values: Iterable[Any]) -> tuple[int | None, int, int, bool, bool]:
    rows = list(values)
    dims = sorted({len(row) for row in rows if row is not None})
    dimension = dims[0] if len(dims) == 1 else None
    nonfinite = 0
    all_zero = 0
    for row in rows:
        if row is None:
            continue
        array = np.asarray(row, dtype=np.float64)
        nonfinite += int(np.count_nonzero(~np.isfinite(array)))
        if array.size and bool(np.all(array == 0)):
            all_zero += 1
    return (
        dimension,
        len(dims),
        nonfinite,
        all_zero == len(rows),
        bool(dims and len(dims) == 1),
    )


def _rational_to_float(value: Any) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        numerator, denominator = str(value).split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return _float_or_none(value)


def _schema_signature(schema: pa.Schema) -> str:
    return "\n".join(f"{field.name}:{field.type}" for field in schema)


def _expected_paths(root: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    parquet: dict[int, Path] = {}
    videos: dict[int, Path] = {}
    for path in (root / "data").rglob("*.parquet"):
        index = parse_episode_index(path)
        if index is not None:
            parquet[index] = path
    for path in (root / "videos").rglob("*.mp4"):
        index = parse_episode_index(path)
        if index is not None:
            videos[index] = path
    return parquet, videos


def audit_g1(
    root: Path,
    ffprobe: str = "ffprobe",
    manifest_path: Path | None = None,
    limit: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Audit all or the first ``limit`` episodes without changing raw data."""

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(root / "meta/tasks.jsonl")
    episodes = load_jsonl(root / "meta/episodes.jsonl")
    episode_stats = load_jsonl(root / "meta/episodes_stats.jsonl")
    task_by_index = {int(record["task_index"]): record for record in tasks}
    episode_by_index = {int(record["episode_index"]): record for record in episodes}
    parquet_by_index, video_by_index = _expected_paths(root)

    all_episode_indices = sorted(
        set(episode_by_index) | set(parquet_by_index) | set(video_by_index)
    )
    selected_indices = all_episode_indices[:limit] if limit else all_episode_indices
    canonical: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    schema_signatures: Counter[str] = Counter()
    parquet_task_indices: Counter[int] = Counter()
    episode_task_indices: Counter[int] = Counter()
    task_valid_4s: Counter[int] = Counter()
    task_episode_counts: Counter[int] = Counter()
    category_episode_counts: Counter[str] = Counter()
    nonfinite_counts: Counter[str] = Counter()
    zero_row_counts: Counter[str] = Counter()
    video_probe_failures = 0
    first_schema: str | None = None
    first_columns: list[str] = []

    for position, episode_index in enumerate(selected_indices, 1):
        episode = episode_by_index.get(episode_index, {})
        raw_tasks = episode.get("tasks") or []
        raw_task_indices = [int(value) for value in raw_tasks if value is not None]
        for task_index in raw_task_indices:
            episode_task_indices[task_index] += 1
        original_task_index = (
            raw_task_indices[0] if len(raw_task_indices) == 1 else None
        )
        # task index 0 is a valid key; using ``or -1`` would silently erase
        # the first task's labels from every episode and selection manifest.
        task = task_by_index.get(
            original_task_index if original_task_index is not None else -1,
            {},
        )
        task_name = task.get("task", "")
        category = task.get("category", "")
        parquet_path = parquet_by_index.get(episode_index)
        video_path = video_by_index.get(episode_index)
        reasons: list[str] = []
        row_count: int | None = None
        timestamp_min = timestamp_max = timestamp_step_median = None
        parquet_episode_values: list[int] = []
        parquet_task_values: list[int] = []
        action_dim = arm_dim = hand_dim = leg_dim = None
        action_dims = arm_dims = hand_dims = leg_dims = None
        schema_ok = False
        parquet_read_ok = False
        video_info: dict[str, Any] = {}

        if parquet_path is None:
            reasons.append("missing_parquet")
        else:
            try:
                parquet_file = pq.ParquetFile(parquet_path)
                schema = parquet_file.schema_arrow
                schema_signature = _schema_signature(schema)
                schema_signatures[schema_signature] += 1
                if first_schema is None:
                    first_schema = schema_signature
                    first_columns = list(schema.names)
                schema_ok = (
                    set(schema.names) == REQUIRED_COLUMNS
                    and schema_signature == first_schema
                )
                if set(schema.names) != REQUIRED_COLUMNS:
                    reasons.append("schema_columns_mismatch")
                elif first_schema is not None and schema_signature != first_schema:
                    reasons.append("schema_type_mismatch")
                table = parquet_file.read()
                parquet_read_ok = True
                row_count = table.num_rows
                data = table.to_pydict()
                parquet_episode_values = [
                    int(x) for x in data.get("episode_index", []) if x is not None
                ]
                parquet_task_values = [
                    int(x) for x in data.get("task_index", []) if x is not None
                ]
                parquet_task_indices.update(parquet_task_values)
                if not row_count:
                    reasons.append("empty_parquet")
                if set(parquet_episode_values) != {episode_index}:
                    reasons.append("parquet_episode_index_mismatch")
                if parquet_task_values and len(set(parquet_task_values)) > 1:
                    reasons.append("multiple_task_indices_in_episode")
                if parquet_task_values and original_task_index not in set(
                    parquet_task_values
                ):
                    reasons.append("parquet_task_index_mapping_mismatch")
                timestamps = np.asarray(data.get("timestamp", []), dtype=np.float64)
                frame_indices = np.asarray(data.get("frame_index", []), dtype=np.int64)
                if timestamps.size:
                    timestamp_min = float(np.min(timestamps))
                    timestamp_max = float(np.max(timestamps))
                    diffs = np.diff(timestamps)
                    positive_diffs = diffs[diffs > 0]
                    timestamp_step_median = (
                        float(np.median(positive_diffs))
                        if positive_diffs.size
                        else None
                    )
                    if not np.all(np.isfinite(timestamps)):
                        reasons.append("timestamp_nonfinite")
                        nonfinite_counts["timestamp"] += int(
                            np.count_nonzero(~np.isfinite(timestamps))
                        )
                    if diffs.size and not np.all(diffs >= -1e-6):
                        reasons.append("timestamp_not_monotonic")
                    if timestamp_min < -1e-6:
                        reasons.append("timestamp_before_zero")
                if frame_indices.size and not np.all(np.diff(frame_indices) >= 0):
                    reasons.append("frame_index_not_monotonic")
                for column, output_name in VECTOR_COLUMNS.items():
                    dimension, dim_count, nonfinite, all_zero, _stable = _vector_stats(
                        data.get(column, [])
                    )
                    if output_name == "action_dim":
                        action_dim, action_dims = dimension, dim_count
                    elif output_name == "arm_dim":
                        arm_dim, arm_dims = dimension, dim_count
                    elif output_name == "hand_dim":
                        hand_dim, hand_dims = dimension, dim_count
                    else:
                        leg_dim, leg_dims = dimension, dim_count
                    if nonfinite:
                        nonfinite_counts[column] += nonfinite
                        reasons.append(f"{column}_nonfinite")
                    if all_zero:
                        zero_row_counts[column] += 1
            except (OSError, RuntimeError, ValueError, KeyError, pa.ArrowException) as exc:
                reasons.append(f"parquet_read_error:{type(exc).__name__}:{exc}")

        if video_path is None:
            reasons.append("missing_video")
        else:
            video_info = ffprobe_video(video_path, ffprobe)
            if video_info.get("ffprobe_exit") != 0:
                video_probe_failures += 1
                reasons.append("ffprobe_failed")
            if (
                video_info.get("video_codec")
                and video_info.get("video_codec") != "h264"
            ):
                reasons.append("video_codec_not_h264")

        metadata_length = _int_or_none(episode.get("length"))
        if (
            row_count is not None
            and metadata_length is not None
            and row_count != metadata_length
        ):
            reasons.append("parquet_metadata_length_mismatch")
        duration_seconds = video_info.get("video_duration_seconds")
        timestamp_duration = (
            (timestamp_max + 1.0 / float(info.get("fps", 30)))
            if timestamp_max is not None
            else None
        )
        if duration_seconds is None:
            duration_seconds = timestamp_duration
        if (
            duration_seconds is not None
            and timestamp_duration is not None
            and abs(duration_seconds - timestamp_duration) > 1.0
        ):
            reasons.append("video_parquet_duration_gap_gt_1s")
        for task_index in raw_task_indices:
            task_episode_counts[task_index] += 1
            category_episode_counts[
                str(task_by_index.get(task_index, {}).get("category", ""))
            ] += 1
        eligibility = {
            f"eligible_{seconds}s": bool(
                duration_seconds is not None and duration_seconds + 1e-6 >= seconds
            )
            for seconds in (1, 2, 4, 8)
        }
        if eligibility["eligible_4s"] and not reasons:
            for task_index in raw_task_indices:
                task_valid_4s[task_index] += 1
        status = "valid" if not reasons else "invalid"
        record = {
            "episode_index": episode_index,
            "original_task_index": original_task_index,
            "episode_task_indices": ",".join(str(x) for x in raw_task_indices),
            "task_name_raw": task_name,
            "task_name_normalized": normalize_label(task_name),
            "category_raw": category,
            "category_normalized": normalize_label(category),
            "task_name_suffix_class": suffix_class(task_name),
            "robot_type_raw": episode.get("robot_type", info.get("robot_type", "")),
            "video_path": str(video_path.relative_to(root)) if video_path else "",
            "parquet_path": str(parquet_path.relative_to(root)) if parquet_path else "",
            "num_frames": row_count,
            "metadata_length": metadata_length,
            "duration_seconds": duration_seconds,
            "timestamp_duration_seconds": timestamp_duration,
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "timestamp_step_median": timestamp_step_median,
            "action_dim": action_dim,
            "action_dim_variants": action_dims,
            "arm_dim": arm_dim,
            "arm_dim_variants": arm_dims,
            "hand_dim": hand_dim,
            "hand_dim_variants": hand_dims,
            "leg_dim": leg_dim,
            "leg_dim_variants": leg_dims,
            "parquet_read_ok": parquet_read_ok,
            "schema_ok": schema_ok,
            "video_size_bytes": video_info.get("video_size_bytes"),
            "video_codec": video_info.get("video_codec"),
            "video_width": video_info.get("video_width"),
            "video_height": video_info.get("video_height"),
            "video_pix_fmt": video_info.get("video_pix_fmt"),
            "video_fps": _rational_to_float(video_info.get("video_avg_frame_rate")),
            "video_nb_frames": _int_or_none(video_info.get("video_nb_frames")),
            "ffprobe_exit": video_info.get("ffprobe_exit"),
            **eligibility,
            "validation_status": status,
            "invalid_reason": ";".join(sorted(set(reasons))),
        }
        canonical.append(record)
        if status == "invalid":
            invalid.append(record)
        if progress_every and position % progress_every == 0:
            print(f"AUDIT_PROGRESS {position}/{len(selected_indices)}", flush=True)

    task_inventory: list[dict[str, Any]] = []
    for task_index in sorted(task_by_index):
        task = task_by_index[task_index]
        task_inventory.append(
            {
                "original_task_index": task_index,
                "task_name_raw": task.get("task", ""),
                "task_name_normalized": normalize_label(task.get("task", "")),
                "category_raw": task.get("category", ""),
                "category_normalized": normalize_label(task.get("category", "")),
                "task_name_suffix_class": suffix_class(task.get("task", "")),
                "episode_count_metadata": task_episode_counts.get(task_index, 0),
                "parquet_row_task_count": parquet_task_indices.get(task_index, 0),
                "valid_4s_episode_count": task_valid_4s.get(task_index, 0),
            }
        )
    labels = Counter(item["task_name_normalized"] for item in task_inventory)
    categories = Counter(item["category_raw"] for item in task_inventory)
    episode_unique_tasks = sorted(episode_task_indices)
    parquet_unique_tasks = sorted(parquet_task_indices)
    expected_task_indices = sorted(task_by_index)
    gaps = (
        sorted(
            set(range(min(expected_task_indices), max(expected_task_indices) + 1))
            - set(expected_task_indices)
        )
        if expected_task_indices
        else []
    )
    manifest_summary: dict[str, Any] = {}
    if manifest_path and manifest_path.exists():
        asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_summary = {
            "path": str(manifest_path),
            "repo_id": asset_manifest.get("repo_id"),
            "revision": asset_manifest.get("revision"),
            "file_count": asset_manifest.get("file_count"),
            "total_bytes": asset_manifest.get("total_bytes"),
        }
    summary = {
        "generated_utc": utc_now(),
        "g1_root": str(root),
        "info": {
            "robot_type": info.get("robot_type"),
            "total_tasks": info.get("total_tasks"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "total_videos": info.get("total_videos"),
            "features": info.get("features"),
        },
        "counts": {
            "tasks_jsonl_rows": len(tasks),
            "episodes_jsonl_rows": len(episodes),
            "episodes_stats_jsonl_rows": len(episode_stats),
            "selected_episode_count": len(selected_indices),
            "parquet_file_count": len(parquet_by_index),
            "video_file_count": len(video_by_index),
            "invalid_episode_count": len(invalid),
            "video_probe_failures": video_probe_failures,
        },
        "task_index_audit": {
            "tasks_min": min(expected_task_indices) if expected_task_indices else None,
            "tasks_max": max(expected_task_indices) if expected_task_indices else None,
            "tasks_gaps": gaps,
            "episode_unique_min": min(episode_unique_tasks)
            if episode_unique_tasks
            else None,
            "episode_unique_max": max(episode_unique_tasks)
            if episode_unique_tasks
            else None,
            "episode_unique_count": len(episode_unique_tasks),
            "parquet_unique_min": min(parquet_unique_tasks)
            if parquet_unique_tasks
            else None,
            "parquet_unique_max": max(parquet_unique_tasks)
            if parquet_unique_tasks
            else None,
            "parquet_unique_count": len(parquet_unique_tasks),
            "episode_only": sorted(
                set(episode_unique_tasks) - set(parquet_unique_tasks)
            ),
            "parquet_only": sorted(
                set(parquet_unique_tasks) - set(episode_unique_tasks)
            ),
        },
        "task_label_duplicates": {
            key: value for key, value in labels.items() if value > 1
        },
        "category_task_counts": dict(sorted(categories.items())),
        "category_episode_counts": dict(sorted(category_episode_counts.items())),
        "task_name_suffix_counts": dict(
            Counter(item["task_name_suffix_class"] for item in task_inventory)
        ),
        "vector_nonfinite_counts": dict(nonfinite_counts),
        "all_zero_rows_or_fields": dict(zero_row_counts),
        "schema_signatures": dict(schema_signatures),
        "first_schema_columns": first_columns,
        "asset_manifest": manifest_summary,
        "core_a_fields_present": sorted(REQUIRED_COLUMNS),
        "full_corpus_status": "pass"
        if not invalid and len(selected_indices) == len(all_episode_indices)
        else "review",
        "episodes": canonical,
        "tasks": task_inventory,
        "invalid_episodes": invalid,
    }
    return summary


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_inventory_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")

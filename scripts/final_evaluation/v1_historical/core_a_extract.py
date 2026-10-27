"""Label-free V-JEPA 2.1-B prefix feature extraction for Core A.

The only episode input accepted is an ordered two-column CSV containing
``episode_index,video_path``. The extractor reads no split, target, or label
manifest. Final-mode extraction is gated by the parent-written
``protocol/FINAL_ACCESS_OPENED.json`` and ``protocol/FREEZE_RECEIPT.json``;
the access receipt must match the study lock and its label-free input hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

EXPECTED_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_DATASET_REVISION = "6b6b599def75e3558b2736062e79bdbeec6cd224"
EXPECTED_INPUT_COLUMNS = ("episode_index", "video_path")
EXPECTED_FRAME_COUNT = 32
EXPECTED_RESOLUTION = 384
EXPECTED_TOKEN_SHAPE = (9216, 768)
EXPECTED_POOLED_SHAPE = (768,)
CONDITIONS = ("dynamic", "static_repeated_first_frame")

_CODE_ROOT = Path(__file__).resolve().parent
OUT_ROOT = _CODE_ROOT.parent
PROJECT_ROOT = OUT_ROOT.parents[2]
LOCAL_SRC_ROOT = PROJECT_ROOT / "repo" / "ee6008-humanoid-vjepa" / "src"


@dataclass(frozen=True)
class EpisodeRow:
    episode_index: int
    video_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    return normalized


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("run_id must use 1-128 ASCII letters, digits, '.', '_' or '-'")
    return run_id


def _parse_episode_index(value: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"invalid episode_index: {value!r}")
    return int(text)


def _within(root: Path, candidate: Path, *, name: str) -> Path:
    root = Path(root).resolve()
    candidate = Path(candidate).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"{name} must be a child of the isolated output root {root}")
    return candidate


def read_episode_manifest(path: Path, *, g1_root: Path) -> list[EpisodeRow]:
    """Read an exact, ordered, label-free CSV of episode IDs and video paths."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("episode manifest must be CSV")
    g1_root = Path(g1_root).resolve()
    rows: list[EpisodeRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_INPUT_COLUMNS:
            raise ValueError("episode manifest header must be exactly: episode_index,video_path")
        for line_number, item in enumerate(reader, start=2):
            if None in item or any(item.get(key) is None for key in EXPECTED_INPUT_COLUMNS):
                raise ValueError(f"malformed episode manifest row at line {line_number}")
            episode_index = _parse_episode_index(item["episode_index"])
            video_text = item["video_path"].strip()
            if not video_text:
                raise ValueError(f"empty video_path at line {line_number}")
            video_path = Path(video_text)
            if not video_path.is_absolute():
                video_path = g1_root / video_path
            video_path = video_path.resolve()
            if video_path == g1_root or not video_path.is_relative_to(g1_root):
                raise ValueError(f"video_path escapes the configured G1 root: {video_path}")
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            rows.append(EpisodeRow(episode_index, video_path))
    if not rows:
        raise ValueError("episode manifest is empty")
    ids = [row.episode_index for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("episode manifest contains duplicate episode_index values")
    return rows


def _git_dir(source_root: Path) -> Path:
    marker = source_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        line = marker.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir:"):
            raise ValueError(f"invalid .git pointer at {marker}")
        pointed = Path(line.partition(":")[2].strip())
        return pointed if pointed.is_absolute() else (source_root / pointed).resolve()
    raise FileNotFoundError(f"source Git metadata is absent: {marker}")


def read_source_head(source_root: Path) -> str:
    """Read the checked-out Git HEAD without invoking Git or using the network."""
    git_dir = _git_dir(Path(source_root).resolve())
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", head):
        return head.lower()
    if not head.startswith("ref: "):
        raise ValueError(f"unrecognized Git HEAD value: {head!r}")
    ref_name = head[5:]
    ref_path = git_dir / ref_name
    if ref_path.is_file():
        value = ref_path.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", value):
            return value.lower()
        raise ValueError(f"invalid Git ref value: {ref_name}")
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="ascii").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref_name:
                if re.fullmatch(r"[0-9a-fA-F]{40}", parts[0]):
                    return parts[0].lower()
                break
    raise FileNotFoundError(f"cannot resolve Git ref {ref_name} under {git_dir}")


def _load_core_config_module() -> Any:
    source = str(LOCAL_SRC_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from ee6008.config import load_config

    return load_config


def validate_pinned_config(
    config_path: Path,
    *,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
) -> tuple[Any, dict[str, str]]:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    expected_config = _validate_sha256(
        expected_config_sha256, field="config_sha256"
    )
    expected_checkpoint = _validate_sha256(
        expected_checkpoint_sha256, field="checkpoint_sha256"
    )
    actual_config = sha256_file(config_path)
    if actual_config != expected_config:
        raise ValueError(
            f"config SHA-256 mismatch: expected {expected_config}, observed {actual_config}"
        )
    config = _load_core_config_module()(config_path)
    project_root = Path(config.project_root).resolve()
    if project_root != PROJECT_ROOT.resolve():
        raise ValueError(f"config project_root is not this isolated project: {project_root}")
    expected_source_root = (PROJECT_ROOT / "repo" / "third_party" / "vjepa2").resolve()
    source_root = Path(config.vjepa_source_root).resolve()
    if source_root != expected_source_root:
        raise ValueError(f"config source root is not the pinned local source: {source_root}")
    expected_g1_root = (
        PROJECT_ROOT
        / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    ).resolve()
    expected_checkpoint_path = (
        PROJECT_ROOT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
    ).resolve()
    expected_feature_root = (PROJECT_ROOT / "features/g1/vjepa2").resolve()
    if Path(config.g1_root).resolve() != expected_g1_root:
        raise ValueError("config G1 root differs from the pinned dataset location")
    if Path(config.checkpoint).resolve() != expected_checkpoint_path:
        raise ValueError("config checkpoint differs from the pinned local V-JEPA checkpoint")
    if Path(config.feature_root).resolve() != expected_feature_root:
        raise ValueError("config feature root differs from the pinned local feature location")
    if config.dataset_revision != EXPECTED_DATASET_REVISION:
        raise ValueError("config dataset revision differs from the pinned G1 revision")
    observed_head = read_source_head(source_root)
    expected_head = str(config.source_commit).lower()
    if expected_head != EXPECTED_SOURCE_COMMIT or observed_head != expected_head:
        raise ValueError(
            "V-JEPA source commit mismatch: "
            f"config={expected_head}, observed_HEAD={observed_head}, "
            f"required={EXPECTED_SOURCE_COMMIT}"
        )
    if (
        config.observation_seconds != 4.0
        or config.sample_fps != 8
        or config.source_fps != 30.0
        or config.frame_count != EXPECTED_FRAME_COUNT
        or config.input_resolution != EXPECTED_RESOLUTION
        or config.model_name != "vjepa2_1_vit_base_384"
        or config.checkpoint_key != "ema_encoder"
    ):
        raise ValueError("Core A V-JEPA extraction config differs from the pinned contract")
    if bool(config.allow_final_test_label_metrics):
        raise ValueError("config enables final-test label metrics")
    checkpoint_path = Path(config.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_checkpoint = sha256_file(checkpoint_path)
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint}, observed {actual_checkpoint}"
        )
    hashes = {
        "config_sha256": actual_config,
        "checkpoint_sha256": actual_checkpoint,
        "source_commit": observed_head,
    }
    return config, hashes


def authorize_manifest(
    *,
    mode: str,
    manifest_path: Path,
    receipt_path: Path | None,
    freeze_receipt_path: Path | None,
    output_root: Path = OUT_ROOT,
) -> str:
    """Require the parent receipt and frozen two-column video index for final use."""
    if mode not in {"dev", "final"}:
        raise ValueError("mode must be 'dev' or 'final'")
    manifest_path = Path(manifest_path).resolve()
    if mode == "dev":
        if receipt_path is not None or freeze_receipt_path is not None:
            raise ValueError("final-access and freeze receipts are not accepted in dev mode")
        return sha256_file(manifest_path)

    output_root = Path(output_root).resolve()
    protocol_dir = output_root / "protocol"
    expected_manifest = (protocol_dir / "CORE_A_FINAL_VIDEO_INDEX.csv").resolve()
    expected_access_receipt = (protocol_dir / "FINAL_ACCESS_OPENED.json").resolve()
    lock_path = (protocol_dir / "FINAL_STUDY_LOCK.json").resolve()
    if manifest_path != expected_manifest:
        raise PermissionError("final mode only accepts protocol/CORE_A_FINAL_VIDEO_INDEX.csv")
    if receipt_path is None:
        receipt_path = expected_access_receipt
    receipt_path = Path(receipt_path).resolve()
    if receipt_path != expected_access_receipt:
        raise PermissionError("final receipt must be protocol/FINAL_ACCESS_OPENED.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    expected_freeze_receipt = (protocol_dir / "FREEZE_RECEIPT.json").resolve()
    if freeze_receipt_path is None:
        raise PermissionError("final mode requires protocol/FREEZE_RECEIPT.json")
    freeze_receipt_path = Path(freeze_receipt_path).resolve()
    if freeze_receipt_path != expected_freeze_receipt:
        raise PermissionError("freeze receipt must be protocol/FREEZE_RECEIPT.json")
    if not freeze_receipt_path.is_file():
        raise FileNotFoundError(freeze_receipt_path)

    # Control files may be read before final video access. The video index
    # itself remains unopened until the receipt/hash checks below have passed.
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (receipt, lock, freeze)):
        raise TypeError("final-access receipt, freeze receipt and study lock must be JSON mappings")
    if receipt.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK":
        raise PermissionError("final-access receipt status is not user-authorized")
    if lock.get("selection_status") != "locked_by_task_protocol":
        raise PermissionError("study lock selection_status is not locked")
    engineering_status = lock.get("engineering_code_hashes_status")
    if not isinstance(engineering_status, str) or not engineering_status.strip() or "pending" in engineering_status.lower():
        raise PermissionError("study lock does not record completed train/dev code freeze")
    protocol_id = lock.get("protocol_id")
    if not protocol_id or receipt.get("protocol_id") != protocol_id or freeze.get("protocol_id") != protocol_id:
        raise PermissionError("protocol_id does not match across final receipts")
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS":
        raise PermissionError("freeze receipt is not a completed pre-access freeze")
    if receipt.get("access_stage") != "final_feature_extraction":
        raise PermissionError("final-access receipt has the wrong access_stage")
    if receipt.get("core_a_access_authorized") is not True:
        raise PermissionError("final-access receipt does not authorize Core A")
    for field in ("authorization_source", "opened_utc"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise PermissionError(f"final-access receipt is missing {field}")
    first_access = receipt.get("first_final_payload_access_utc")
    if first_access is not None and (not isinstance(first_access, str) or not first_access.strip()):
        raise PermissionError("final-access receipt has an invalid first_final_payload_access_utc")

    observed_lock_sha = sha256_file(lock_path)
    observed_freeze_sha = sha256_file(freeze_receipt_path)
    if receipt.get("study_lock_sha256") != observed_lock_sha:
        raise PermissionError("final-access receipt study_lock_sha256 does not match")
    if receipt.get("freeze_receipt_sha256") != observed_freeze_sha:
        raise PermissionError("final-access receipt freeze_receipt_sha256 does not match")
    if freeze.get("study_lock_sha256") != observed_lock_sha:
        raise PermissionError("freeze receipt does not bind the current study lock")
    input_hashes = freeze.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise PermissionError("freeze receipt lacks input_sha256 bindings")
    expected_manifest_sha = _validate_sha256(
        str(input_hashes.get("core_a_final_video_index", "")),
        field="FREEZE_RECEIPT.input_sha256.core_a_final_video_index",
    )
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise PermissionError("Core A final video index differs from the frozen label-free input")
    return actual_manifest_sha


def make_condition_clip(clip: Tensor, condition: str) -> Tensor:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported Core A condition: {condition}")
    if clip.ndim != 4 or tuple(clip.shape[:2]) != (3, EXPECTED_FRAME_COUNT):
        raise ValueError("clip must have shape 3,32,H,W")
    if condition == "dynamic":
        return clip
    result = clip.clone()
    result[:, 1:] = result[:, :1]
    return result


def _cache_config_hash(config: Any, hashes: dict[str, str], condition: str) -> str:
    contract = {
        "model_name": config.model_name,
        "checkpoint_key": config.checkpoint_key,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "source_commit": hashes["source_commit"],
        "config_sha256": hashes["config_sha256"],
        "observation_seconds": config.observation_seconds,
        "sample_fps": config.sample_fps,
        "source_fps": config.source_fps,
        "resolution": config.input_resolution,
        "frame_count": config.frame_count,
        "condition": condition,
        "stored_token_dtype": "float16",
        "stored_pooled_dtype": "float32",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".part", dir=path.parent,
        delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".part",
        dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("episode_index", "feature_path")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
        suffix=".part", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _encode_clip(
    model: nn.Module,
    clip: Tensor,
    *,
    device: torch.device,
    encode_fn: Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    clip = clip.to(device, non_blocking=(device.type == "cuda"))
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        tokens, pooled = encode_fn(model, clip)
    if not isinstance(tokens, Tensor) or tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        actual = tuple(tokens.shape) if isinstance(tokens, Tensor) else None
        raise ValueError(f"encoder token shape must be {EXPECTED_TOKEN_SHAPE}, got {actual}")
    if not isinstance(pooled, Tensor) or tuple(pooled.shape) != EXPECTED_POOLED_SHAPE:
        actual = tuple(pooled.shape) if isinstance(pooled, Tensor) else None
        raise ValueError(f"encoder pooled shape must be {EXPECTED_POOLED_SHAPE}, got {actual}")
    if not bool(torch.isfinite(tokens).all()) or not bool(torch.isfinite(pooled).all()):
        raise FloatingPointError("encoder emitted non-finite features")
    return tokens.detach().to(device="cpu", dtype=torch.float16), pooled.detach().to(
        device="cpu", dtype=torch.float32
    )


@torch.inference_mode()
def extract_rows(
    *,
    rows: Sequence[EpisodeRow],
    config: Any,
    hashes: dict[str, str],
    run_id: str,
    output_dir: Path,
    model: nn.Module,
    device: torch.device,
    decode_fn: Callable[..., Any],
    preprocess_fn: Callable[..., Tensor],
    encode_fn: Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]],
    mode: str,
    first_payload_access_callback: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Extract both conditions; dependency injection supports CPU-only tests."""
    if not rows:
        raise ValueError("cannot extract an empty episode manifest")
    output_dir = Path(output_dir).resolve()
    condition_rows = {condition: [] for condition in CONDITIONS}
    feature_hashes: dict[str, list[str]] = {condition: [] for condition in CONDITIONS}
    decoder_counts: dict[str, int] = {}
    selected_source_frames_returned = 0
    opencv_sequential_frame_reads = 0
    first_final_payload_access_utc: str | None = None
    for row in rows:
        if mode == "final" and first_final_payload_access_utc is None:
            if first_payload_access_callback is None:
                raise PermissionError("final extraction requires a first-payload-access recorder")
            first_final_payload_access_utc = first_payload_access_callback()
        sample = decode_fn(
            row.video_path,
            duration_seconds=config.observation_seconds,
            sample_fps=config.sample_fps,
            expected_source_fps=config.source_fps,
        )
        decoder_name = str(sample.decoder).lower()
        decoder_counts[decoder_name] = decoder_counts.get(decoder_name, 0) + 1
        selected_source_frames_returned += int(len(sample.frame_indices))
        if decoder_name == "opencv":
            opencv_sequential_frame_reads += int(sample.frame_indices[-1]) + 1
        clip = preprocess_fn(sample.frames, config.input_resolution)
        if tuple(clip.shape) != (3, EXPECTED_FRAME_COUNT, EXPECTED_RESOLUTION, EXPECTED_RESOLUTION):
            raise ValueError(f"preprocessor returned unexpected clip shape: {tuple(clip.shape)}")
        for condition in CONDITIONS:
            condition_clip = make_condition_clip(clip, condition)
            tokens, pooled = _encode_clip(
                model, condition_clip, device=device, encode_fn=encode_fn
            )
            contract_hash = _cache_config_hash(config, hashes, condition)
            feature_path = (
                output_dir / condition / f"episode_{row.episode_index:06d}.pt"
            )
            feature_record = {
                "config_hash": contract_hash,
                "episode_index": row.episode_index,
                "pooled": pooled,
                "tokens": tokens,
                "metadata": {
                    "run_id": run_id,
                    "input_mode": condition,
                    "frame_indices": sample.frame_indices.tolist(),
                    "timestamps_seconds": sample.timestamps_seconds.tolist(),
                    "source_fps_observed": float(sample.source_fps),
                    "decoder": str(sample.decoder),
                    "clip_shape_cthw": list(clip.shape),
                    "encoder_token_shape": list(EXPECTED_TOKEN_SHAPE),
                    "stored_token_shape": list(EXPECTED_TOKEN_SHAPE),
                    "pooled_shape": list(EXPECTED_POOLED_SHAPE),
                    "dataset_revision": str(config.dataset_revision),
                    "source_commit": hashes["source_commit"],
                    "config_sha256": hashes["config_sha256"],
                    "checkpoint_sha256": hashes["checkpoint_sha256"],
                },
            }
            _atomic_torch_save(feature_path, feature_record)
            relative_path = feature_path.relative_to(output_dir).as_posix()
            feature_hash = sha256_file(feature_path)
            condition_rows[condition].append(
                {
                    "episode_index": row.episode_index,
                    "feature_path": relative_path,
                }
            )
            feature_hashes[condition].append(feature_hash)

    manifest_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        manifest_path = output_dir / f"feature_manifest_{condition}.csv"
        _atomic_write_csv(manifest_path, condition_rows[condition])
        manifest_hashes[condition] = sha256_file(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": "FEATURE_EXTRACTION_COMPLETE",
        "mode": mode,
        "run_id": run_id,
        "row_count": len(rows),
        "video_decoder_invocations": len(rows),
        "decoder_invocations_by_backend": decoder_counts,
        "selected_source_frames_returned": selected_source_frames_returned,
        "opencv_sequential_frame_reads": opencv_sequential_frame_reads,
        "decord_internal_decoded_frame_count": "not exposed by the decoder API",
        "episode_index_sha256": hashlib.sha256(
            np.asarray([row.episode_index for row in rows], dtype="<i8").tobytes()
        ).hexdigest(),
        "input_manifest_sha256": hashes["input_manifest_sha256"],
        "config_sha256": hashes["config_sha256"],
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "source_commit": hashes["source_commit"],
        "conditions": list(CONDITIONS),
        "feature_manifest_sha256": manifest_hashes,
        "feature_record_hashes": feature_hashes,
        "device": str(device),
        "torch_version": torch.__version__,
        "first_final_payload_access_utc": first_final_payload_access_utc,
        "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The pin is verified from Git HEAD; the Git CLI is unavailable here,
        # so worktree dirtiness is intentionally left unknown.
        "source_worktree_clean": "not_verified",
    }
    _atomic_write_json(output_dir / "EXTRACTION_RECEIPT.json", receipt)
    return receipt


def _build_real_model(config: Any, device: torch.device) -> tuple[nn.Module, Any, Any, Any]:
    source = str(LOCAL_SRC_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.models import encode_clip, load_vjepa2_1_base

    model, _ = load_vjepa2_1_base(
        source_root=config.vjepa_source_root,
        checkpoint=config.checkpoint,
        device=device,
        num_frames=config.frame_count,
        resolution=config.input_resolution,
    )
    return model, decode_prefix, preprocess_vjepa, encode_clip


def run_extraction(
    *,
    manifest_path: Path,
    config_path: Path,
    config_sha256: str,
    checkpoint_sha256: str,
    run_id: str,
    output_dir: Path,
    mode: str,
    final_access_receipt: Path | None,
    freeze_receipt: Path | None,
    device_name: str,
    expected_rows: int | None = None,
    output_root: Path = OUT_ROOT,
    protocol_root: Path = OUT_ROOT,
    model_bundle: tuple[nn.Module, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    run_id = _validate_run_id(run_id)
    if mode not in {"dev", "final"}:
        raise ValueError("mode must be 'dev' or 'final'")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    output_root = Path(output_root).resolve()
    output = _within(output_root, Path(output_dir), name="output_dir")
    if output.exists():
        raise FileExistsError(f"run output directory already exists: {output}")

    # Verify the final gate before reading, hashing, or parsing episode rows.
    config, hashes = validate_pinned_config(
        config_path,
        expected_config_sha256=config_sha256,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    manifest_path = Path(manifest_path).resolve()
    hashes["input_manifest_sha256"] = authorize_manifest(
        mode=mode,
        manifest_path=manifest_path,
        receipt_path=final_access_receipt,
        freeze_receipt_path=freeze_receipt,
        output_root=protocol_root,
    )
    first_payload_access_callback: Callable[[], str] | None = None
    if mode == "final":
        if final_access_receipt is None:
            raise PermissionError("final extraction requires the canonical final access receipt")
        access_path = Path(final_access_receipt).resolve()
        authorization_receipt_sha256 = sha256_file(access_path)

        def mark_first_payload_access() -> str:
            current = json.loads(access_path.read_text(encoding="utf-8"))
            existing = current.get("first_final_payload_access_utc")
            if existing:
                if not isinstance(existing, str):
                    raise PermissionError("existing first final payload timestamp is invalid")
                return existing
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            current["first_final_payload_access_utc"] = timestamp
            current["first_final_payload_access_event"] = "immediately_before_core_a_video_decoder_invocation"
            current["authorization_receipt_sha256_before_first_payload"] = authorization_receipt_sha256
            _atomic_write_json(access_path, current)
            return timestamp

        first_payload_access_callback = mark_first_payload_access
    rows = read_episode_manifest(manifest_path, g1_root=config.g1_root)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} manifest rows, got {len(rows)}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model, decode_fn, preprocess_fn, encode_fn = model_bundle or _build_real_model(
        config, device
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output.mkdir(parents=True, exist_ok=False)
    return extract_rows(
        rows=rows,
        config=config,
        hashes=hashes,
        run_id=run_id,
        output_dir=output,
        model=model,
        device=device,
        decode_fn=decode_fn,
        preprocess_fn=preprocess_fn,
        encode_fn=encode_fn,
        mode=mode,
        first_payload_access_callback=first_payload_access_callback,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z'),
    )
    parser.add_argument("--protocol-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=("dev", "final"), required=True)
    parser.add_argument("--final-access-receipt", type=Path)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args(argv)
    result = run_extraction(
        manifest_path=args.manifest,
        config_path=args.config,
        config_sha256=args.config_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        run_id=args.run_id,
        output_dir=args.output_dir,
        output_root=args.output_root,
        protocol_root=args.protocol_root,
        mode=args.mode,
        final_access_receipt=args.final_access_receipt,
        freeze_receipt=args.freeze_receipt,
        device_name=args.device,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

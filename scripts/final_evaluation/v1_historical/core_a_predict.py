"""Inference-only Core A Batch-48 predictor.

The predictor consumes only an ordered ``episode_index,feature_path`` view and
an accepted classifier checkpoint. It never opens label, target, or label-map
fields. The checkpoint class order is preserved as the model's output-column
order (0 through 119).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

EXPECTED_TOKEN_SHAPE = (9216, 768)
EXPECTED_CLASS_COUNT = 120
EXPECTED_NUM_HEADS = 16
EXPECTED_DEPTH = 4
INFERENCE_BATCH_SIZE = 48
MANIFEST_COLUMNS = ("episode_index", "feature_path")
SIDECAR_SUFFIX = ".json"

_CODE_ROOT = Path(__file__).resolve().parent
OUT_ROOT = _CODE_ROOT.parent
PROJECT_ROOT = OUT_ROOT.parents[2]
VJEPA_SOURCE_ROOT = PROJECT_ROOT / "repo" / "third_party" / "vjepa2"


@dataclass(frozen=True)
class FeatureRow:
    episode_index: int
    feature_path: Path


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


def _parse_episode_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("episode_index must be a non-negative integer")
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"invalid episode_index: {value!r}")
    return int(text)


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("run_id must use 1-128 ASCII letters, digits, '.', '_' or '-'")
    return run_id


def _within(root: Path, candidate: Path, *, name: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"{name} must be a child of the isolated output root {root}")
    return candidate


def read_feature_manifest(path: Path) -> list[FeatureRow]:
    """Read only episode IDs and feature paths, preserving the supplied row order.

    Parquet manifests are projected through the two-column whitelist. CSV
    manifests must contain exactly those two columns, in that order.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, columns=list(MANIFEST_COLUMNS))
        rows_raw = frame.loc[:, list(MANIFEST_COLUMNS)].itertuples(
            index=False, name=None
        )
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                raise ValueError(
                    "CSV manifest header must be exactly: episode_index,feature_path"
                )
            raw_rows = []
            for line_number, item in enumerate(reader, start=2):
                if None in item or any(item.get(key) is None for key in MANIFEST_COLUMNS):
                    raise ValueError(f"malformed CSV manifest row at line {line_number}")
                raw_rows.append((item["episode_index"], item["feature_path"]))
            rows_raw = iter(raw_rows)
    else:
        raise ValueError("feature manifest must be CSV or Parquet")

    rows: list[FeatureRow] = []
    for episode_value, feature_value in rows_raw:
        episode_index = _parse_episode_index(episode_value)
        feature_text = str(feature_value).strip()
        if not feature_text:
            raise ValueError(f"empty feature_path for episode_index={episode_index}")
        feature_path = Path(feature_text)
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        feature_path = feature_path.resolve()
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        rows.append(FeatureRow(episode_index, feature_path))

    if not rows:
        raise ValueError("feature manifest is empty")
    ids = [row.episode_index for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("feature manifest contains duplicate episode_index values")
    return rows


def load_feature_tokens(row: FeatureRow) -> tuple[Tensor, str]:
    """Load a single feature record safely and return only its token tensor."""
    digest = sha256_file(row.feature_path)
    record = torch.load(
        row.feature_path, map_location="cpu", weights_only=True
    )
    if not isinstance(record, dict):
        raise TypeError(f"feature record is not a mapping: {row.feature_path}")
    stored_id = record.get("episode_index")
    if isinstance(stored_id, bool) or not isinstance(stored_id, (int, np.integer)):
        raise TypeError(f"feature record has invalid episode_index: {row.feature_path}")
    if int(stored_id) != row.episode_index:
        raise ValueError(
            f"feature ID mismatch: manifest={row.episode_index}, "
            f"record={int(stored_id)} ({row.feature_path})"
        )
    tokens = record.get("tokens")
    if not isinstance(tokens, Tensor) or tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        actual_shape = tuple(tokens.shape) if isinstance(tokens, Tensor) else None
        raise ValueError(
            f"expected tokens with shape {EXPECTED_TOKEN_SHAPE}, got {actual_shape} "
            f"for {row.feature_path}"
        )
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError(f"non-finite feature tokens: {row.feature_path}")
    # Other cache fields, including descriptive metadata, are deliberately not
    # read or passed to the classifier.
    return tokens.float(), digest


def build_attentive_classifier() -> nn.Module:
    """Construct the accepted 768-d, 16-head, depth-4, 120-logit head."""
    source_root = VJEPA_SOURCE_ROOT.resolve()
    module_path = source_root / "src" / "models" / "attentive_pooler.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"pinned V-JEPA source is missing: {module_path}")
    source_text = str(source_root)
    if source_text not in os.sys.path:
        os.sys.path.insert(0, source_text)
    from src.models.attentive_pooler import AttentiveClassifier

    return AttentiveClassifier(
        embed_dim=EXPECTED_TOKEN_SHAPE[1],
        num_heads=EXPECTED_NUM_HEADS,
        depth=EXPECTED_DEPTH,
        num_classes=EXPECTED_CLASS_COUNT,
        use_activation_checkpointing=True,
    )


def load_checkpoint_state(
    checkpoint_path: Path, expected_sha256: str
) -> tuple[dict[str, Tensor], str]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    expected = _validate_sha256(expected_sha256, field="checkpoint_sha256")
    observed = sha256_file(checkpoint_path)
    if observed != expected:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload is not a mapping")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty model_state_dict")
    if any(not isinstance(value, Tensor) for value in state.values()):
        raise TypeError("model_state_dict contains a non-tensor value")
    return state, observed


def strict_load_model(
    model: nn.Module, state: dict[str, Tensor]
) -> nn.Module:
    """Load checkpoint weights with exact missing/unexpected-key rejection."""
    model.load_state_dict(state, strict=True)
    return model.eval()


def _load_batch(rows: Sequence[FeatureRow]) -> tuple[Tensor, list[str]]:
    tensors: list[Tensor] = []
    hashes: list[str] = []
    for row in rows:
        tokens, digest = load_feature_tokens(row)
        tensors.append(tokens)
        hashes.append(digest)
    return torch.stack(tensors, dim=0), hashes


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def infer_logits(
    rows: Sequence[FeatureRow], model: nn.Module, device: torch.device
) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("cannot infer an empty feature sequence")
    model = model.to(device).eval()
    output_batches: list[Tensor] = []
    feature_hashes: list[str] = []
    for start in range(0, len(rows), INFERENCE_BATCH_SIZE):
        batch_rows = rows[start : start + INFERENCE_BATCH_SIZE]
        tokens, hashes = _load_batch(batch_rows)
        feature_hashes.extend(hashes)
        tokens = tokens.to(device, non_blocking=(device.type == "cuda"))
        with _autocast_context(device):
            logits = model(tokens)
        if not isinstance(logits, Tensor) or tuple(logits.shape) != (
            len(batch_rows),
            EXPECTED_CLASS_COUNT,
        ):
            actual_shape = tuple(logits.shape) if isinstance(logits, Tensor) else None
            raise ValueError(
                "classifier output shape mismatch: "
                f"expected {(len(batch_rows), EXPECTED_CLASS_COUNT)}, got {actual_shape}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("classifier emitted non-finite logits")
        output_batches.append(logits.to(dtype=torch.float64, device="cpu"))
    logits_array = torch.cat(output_batches, dim=0).numpy().astype("<f8", copy=False)
    if logits_array.shape != (len(rows), EXPECTED_CLASS_COUNT):
        raise AssertionError("internal logits row count mismatch")
    if not np.isfinite(logits_array).all():
        raise FloatingPointError("float64 output logits contain NaN or Inf")
    return logits_array, feature_hashes


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _write_npz(path: Path, *, episode_index: np.ndarray, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".part", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            episode_index=np.ascontiguousarray(episode_index, dtype="<i8"),
            logits=np.ascontiguousarray(logits, dtype="<f8"),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".part",
        dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def predict_manifest(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    run_id: str,
    output_path: Path,
    device: str = "cpu",
    expected_rows: int | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    output_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    run_id = _validate_run_id(run_id)
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = Path(output_root).resolve()
    output = _within(output_root, Path(output_path), name="output_path")
    if output.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz")
    sidecar = output.with_suffix(SIDECAR_SUFFIX)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"output or sidecar already exists: {output}")

    manifest_path = Path(manifest_path).resolve()
    rows = read_feature_manifest(manifest_path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} feature rows, got {len(rows)}")
    manifest_sha256 = sha256_file(manifest_path)
    state, actual_checkpoint_sha = load_checkpoint_state(
        checkpoint_path, checkpoint_sha256
    )
    model = (model_factory or build_attentive_classifier)()
    strict_load_model(model, state)
    logits, feature_hashes = infer_logits(rows, model, torch.device(device))
    episode_ids = np.asarray([row.episode_index for row in rows], dtype="<i8")
    _write_npz(output, episode_index=episode_ids, logits=logits)
    output_sha = sha256_file(output)
    class_indices = np.arange(EXPECTED_CLASS_COUNT, dtype="<i8")
    sidecar_value = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": actual_checkpoint_sha,
        "feature_manifest_path": str(manifest_path),
        "feature_manifest_sha256": manifest_sha256,
        "feature_rows": len(rows),
        "feature_record_sha256_by_episode": [
            {"episode_index": row.episode_index, "sha256": digest}
            for row, digest in zip(rows, feature_hashes, strict=True)
        ],
        "output_path": str(output),
        "output_sha256": output_sha,
        "episode_index_sha256": _array_sha256(episode_ids, "<i8"),
        "logits_sha256": _array_sha256(logits, "<f8"),
        "class_index_order_sha256": _array_sha256(class_indices, "<i8"),
        "logits_shape": list(logits.shape),
        "logits_dtype": "float64",
        "logit_column_contract": (
            "columns 0..119 preserve the accepted Batch-48 classifier output order"
        ),
        "device": device,
        "torch_version": torch.__version__,
    }
    _write_json_atomic(sidecar, sidecar_value)
    return sidecar_value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args(argv)
    result = predict_manifest(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        run_id=args.run_id,
        output_path=args.output,
        device=args.device,
        expected_rows=args.expected_rows,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

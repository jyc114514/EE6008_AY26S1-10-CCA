"""Inference-only, prefix-bounded Core B V9 adapter.

This module deliberately has no scorer and no training interface. Its model
call accepts only V9 prefix tensors and task IDs. Future trajectories are not
loaded from an input archive or passed to the model.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


TASK_IDS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
SPLITS = frozenset({"train", "dev", "final_test"})
PROGRAM_SPECS: dict[str, tuple[str, float]] = {
    "A0": ("STATE-CTRL", 0.0),
    "AD": ("STATE-CTRL", 0.5),
    "V0": ("TS11-D", 0.0),
    "VD": ("TS11-D", 0.5),
    "S_OPT": ("STATE-CTRL", 0.0),
}

PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ACCEPTED_V9_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
V2_OUT_ROOT = PROJECT_ROOT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
V2_PROTOCOL_ID = "EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
CURRENT_PROFILE_LOCK_PATH = V2_OUT_ROOT / "core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
CURRENT_PROFILE_DEV_MANIFEST = V2_OUT_ROOT / "core_b/CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json"
CURRENT_PROFILE_FINAL_MANIFEST = V2_OUT_ROOT / "core_b/CURRENT_PROFILE_V2_FINAL_FEATURE_MANIFEST.json"
V2_FINAL_FREEZE_PATH = V2_OUT_ROOT / "core_b/CORE_B_V2_FINAL_FREEZE.json"
V2_DEV_ACCEPTANCE_PATH = V2_OUT_ROOT / "core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json"
ACCEPTED_V9_CODE_ROOT = ACCEPTED_V9_ROOT / "code"
ACCEPTED_Q0_CACHE_SHA256 = "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee"
ACCEPTED_NORMALIZER_SHA256 = "22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d"
ACCEPTED_NORMALIZER_PATH = PROJECT_ROOT / "reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/normalizers/NORMALIZER_FORMAL.json"
ACCEPTED_CODE_LOCK_SHA256 = "7bc231365cf207d0ba3ece24d588ecf72cf4816a5e46de409c75621dc8d429c5"
_FINAL_AUTH_SENTINEL = object()

PINNED_SOURCE_HASHES: dict[str, str] = {
    "locked_source/D3_model.py": "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589",
    "locked_source/d3_eq_contract.py": "7359c956bf405092dade2e49e72f236759099330357652469c610d422c770b1c",
    "prepare_v2_data.py": "64126bdf992037bb3455b4a78392269e7ecd9e62ae01a6b425ef6b090341e15d",
    "v2_model.py": "6e1e967c89afeeb6e676e9265a893fed54f66682f54af3bf04cf3d41766c8c24",
    "v9_common.py": "f023fa7af55f681d1308b0165dbcb4b1fe75091fcf3f9ed1df59d456d1a12615",
}

# (program, seed) -> (accepted MODEL_MANIFEST SHA-256, model SHA-256,
# accepted in-memory state hash). These are read from the accepted V9/V2
# manifests and are checked in addition to the caller's frozen job hashes.
PINNED_MODEL_IDENTITIES: dict[tuple[str, int], tuple[str, str, str]] = {
    ("A0", 202610131): ("5ead28e3a775544b4a6ecd1c6ce6635349955273caa3f49f26031db5800ed0f2", "5208423d943f89b4642d8ff44905ba0593d699a1db8e662a515f83aaa9e16ae4", "e951420f98b562dd9e11e9fc424113e472edc80a9051da2dea3f2fd8c91604ce"),
    ("A0", 202610132): ("72a70ebab7ea859ab57fe39741add0f154420dd021d6275d05b0c228669c5c56", "b9cd77b04b50da1e5b472400cdb0b41eeabf11f18edf4ca464ba23263d3ee5bc", "0eb72a72f265e78e45ae1383dfe033c45e7eb80d054125209b6ab6141b3e60cc"),
    ("A0", 202610133): ("5efaa8344b3d96ed029ffe747ba88f9cfcea3d1c81a769c48bbca8da45e23044", "8d1d4a8420a723b8b1c8ed85aad62b26a73b7f11a86014892dd3d92ab6ea8497", "bcda4100d3b25e8ae64173d70a7aeeb2af31b92f609e912145c97b2aa05bdcea"),
    ("AD", 202610131): ("7407bdcf320c8fb8e0128bdef65681aefbbc9d15c05734ed03863b4314b9571a", "16f3e68679f942648a85bf0cce6505bf549670943287328bff2b5dce767a7068", "7d4a1932e1297a0bd55b07fc89a801fcf6935a5271aa1c0eae0e13bb1936a129"),
    ("AD", 202610132): ("1fdd46693e8cf5709152a4b4c6c51c66868dcf86727246bf86f5dc169588ed2a", "7b793de39933bc1de400b07c931258e38c6845f175720e35314a911d94f473cf", "77cef2c44ac700d97de53f8c219c9cd39df1a7a135e0b05fd190978b1be6cdd9"),
    ("AD", 202610133): ("c85b744bc77ca5550bcf04e7168fe0ca98a73983da3312b2565a2a6854817d50", "76dc124b9252296236ac42a84a7ea4a74a78fa2d77e81c1b21efa47d482eaf96", "dd13ae56a4dbd7652759e76aafea7898a2d20cb6b8d5300b311abcee99f762e9"),
    ("V0", 202610131): ("af5ec09a24e2211b458cc56b242aa7031900795a44afb118c7a5fe217f36bd05", "f86c3c752e0f4d7752303dcff8e9f20912c56cd244ce68de3552cabd0c4db2b8", "4e8735734f10d9f21a3102ccf9943ba693010099bfb44a9ce8f1a8b610c56fb4"),
    ("V0", 202610132): ("bbbd0bc7e378c0355c7a66f94ac59fa14f4bb9b9d0f7fddcb0cff058974287b1", "14fce627c0ed3a1b47d23e06d924da203caf88bc86fb5e1675c4a6ebe05238ff", "a4f1c0971c6d682d6cd02e337ed68b99f8ad2a3c69f9ae9d8ada44347ecf5b98"),
    ("V0", 202610133): ("dd9fecc4eca815c4c75bda2712aea770c999b750780d8db73d04040a344b23a9", "f52e32a6c1f0dd2b00573eb1078be03d5f27c021c0456505ad8294a0fe24cdaa", "7096b32a23cffef07dae223b71670e025dafece23bad9dfa4e8d05d9ee1129b8"),
    ("VD", 202610131): ("11bb9b0cd16ded880c5d613fcc47c23f26209cb4276be8ef5ecb39a81fa7e8b1", "aac73d96d997d2691a49db7e2f082a2e2f6c66880ddfde23037b89978bbc1888", "d21486848b4ff5205891b58ed7f81289e89ccabfac6fb7088272986482de286f"),
    ("VD", 202610132): ("8fc59ede9480bcd1091187edd3d6260e8cb23c30b89fdf547bbc4c5220b86c79", "6b8fcf627f462bf68ce804d30e0bebcca114c4e322103873874d641af8ce09d0", "966f7e01c8fc47604795b71ca42cce428822bf5597b1e7879bddca9eaf10cd8d"),
    ("VD", 202610133): ("3ff7efca01b8754b0f1e7e507140f74980b5c2ce7d6ca71f567c7ace2857387e", "e502b945f4c5a162b26d17464ec3f63ac51fc1bdb4e5987258f46bf78315444d", "58fa91f58e3c3aeaee6ae306c9e1657308d71a14c99cdd5359d64a5f24587b3b"),
    ("S_OPT", 202610131): ("85a81f20e2f3a45c96845e7e6cc883903731fb01155563183e9219fea8c06310", "20bed18629f4aa2cb67629198fb01634fae507f580e9f46b5ce23dc686d4cb0b", "c418a1c114d6750978fb433830663bde38e85260a5aa1b1002a9d5d6746e8144"),
    ("S_OPT", 202610132): ("df620ef85ad02a093c518eb8008a220766482cc18085c96d2c6780fb683f78c7", "8ea09d15e05a271ed7b20c7d5321249709bb265559e66b02f699da2393def9f0", "35123cbf3eedce64e1423bb8ef3bfeb2d922c51811fdb86c57008cad91c8ad2d"),
    ("S_OPT", 202610133): ("b3c355e499b82ff706f6e422a5c2624babb442f08fe5baf466b4f9c1549f3181", "3d53232aea572c81a1580aa06a1a5662d646c0049957b036f628a8bbb92366d7", "de944816d89b442913c49c0ed34ab82e19cc941a8a194086d9960a9481f869b5"),
}

PREFIX_REQUIRED_KEYS = frozenset({
    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
    "cache_index", "anchor_frame", "history_native", "history_mask",
    "relative_delta", "anchor_native",
})
PREFIX_OPTIONAL_KEYS = frozenset({
    "canonical_id", "dataset_revision", "source_key", "literal_source_key", "split",
})
PREFIX_KEYS = PREFIX_REQUIRED_KEYS | PREFIX_OPTIONAL_KEYS
MODEL_INPUT_KEYS = {
    "STATE-CTRL": frozenset({"state", "history_mask", "relative_delta", "anchor_command"}),
    "TS11-D": frozenset({"state", "history_mask", "relative_delta", "anchor_command", "dynamic", "static"}),
}
_FORBIDDEN_KEY = re.compile(r"(^|[_./-])(target|targets|label|labels|future|ground_truth|teacher)([_./-]|$)", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PredictorContractError(RuntimeError):
    """Raised when a predictor input or identity breaks the frozen contract."""


@dataclass(frozen=True)
class ModelCell:
    program_id: str
    seed: int
    video_mode: str
    model_path: Path
    expected_model_sha256: str
    model_manifest_path: Path
    expected_model_manifest_sha256: str
    normalizer_path: Path
    expected_normalizer_sha256: str
    prefix_path: Path
    expected_prefix_sha256: str
    feature_cache_manifest_path: Path | None = None
    expected_feature_cache_manifest_sha256: str | None = None


@dataclass(frozen=True)
class AcceptedSources:
    model_class: type
    model_input: Callable[..., Any]
    q0_grid_class: type
    state_hash: Callable[[torch.nn.Module], str]


@dataclass(frozen=True)
class PredictionBundle:
    program_id: str
    seed: int | None
    canonical_ids: np.ndarray
    prediction_native: np.ndarray
    canonical_ids_sha256: str
    prediction_native_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class PersistenceAlias:
    program_id: str
    seed: int
    canonical_ids_sha256: str
    prediction_native_sha256: str
    shape: tuple[int, int, int]
    alias_of: str


@dataclass(frozen=True)
class FinalAccessAuthorization:
    path: Path
    sha256: str
    _sentinel: object


class CurrentProfileGrid:
    """Hash-bound V2 feature rows keyed by accepted cache index and anchor identity."""

    def __init__(self, manifest: Mapping[str, Any], manifest_path: Path):
        self.manifest = dict(manifest)
        self.manifest_path = Path(manifest_path).resolve()
        if self.manifest.get("status") != "PASS" or self.manifest.get("profile_id") != "CURRENT_PROFILE_V2":
            raise PredictorContractError("current-profile feature manifest is not accepted")
        if self.manifest.get("split") not in {"dev", "final_test"}:
            raise PredictorContractError("current-profile feature manifest split is unsupported")
        lock_sha = sha256_file(CURRENT_PROFILE_LOCK_PATH)
        if self.manifest.get("profile_lock_sha256") != lock_sha:
            raise PredictorContractError("current-profile feature manifest does not bind the frozen candidate lock")
        archive_path = _safe_path(str(self.manifest.get("feature_archive_path", "")), "artifact")
        expected_archive_sha = str(self.manifest.get("feature_archive_sha256", ""))
        verify_file_sha256(archive_path, expected_archive_sha, "current-profile feature archive")
        with np.load(archive_path, allow_pickle=False) as archive:
            expected_keys = {"canonical_ids", "anchor_ids", "episode_ids", "task_ids", "cache_indices", "grid_features"}
            if set(archive.files) != expected_keys:
                raise PredictorContractError("current-profile feature archive key set mismatch")
            canonical = archive["canonical_ids"].astype(str)
            anchors = archive["anchor_ids"].astype(str)
            episodes = archive["episode_ids"].astype(np.int64)
            tasks = archive["task_ids"].astype(np.int64)
            indices = archive["cache_indices"].astype(np.int64)
            features = np.asarray(archive["grid_features"])
        n = int(self.manifest.get("rows", -1))
        if n <= 0 or canonical.shape != (n,) or anchors.shape != (n,) or episodes.shape != (n,) or tasks.shape != (n,) or indices.shape != (n,):
            raise PredictorContractError("current-profile feature identity vector shape mismatch")
        if features.shape != (n, 2, 16, 4, 768) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise PredictorContractError("current-profile feature tensor shape/dtype/finite check failed")
        if len(set(canonical.tolist())) != n or len(set(anchors.tolist())) != n or len(set(indices.tolist())) != n:
            raise PredictorContractError("current-profile feature archive contains duplicate identities")
        ids_sha = hashlib.sha256("\n".join(canonical.tolist()).encode()).hexdigest()
        anchors_sha = hashlib.sha256("\n".join(anchors.tolist()).encode()).hexdigest()
        if ids_sha != self.manifest.get("canonical_id_order_sha256") or anchors_sha != self.manifest.get("anchor_id_order_sha256"):
            raise PredictorContractError("current-profile feature archive order differs from its manifest")
        self._by_cache = {
            int(index): (str(anchor), int(episode), np.array(features[i], dtype=np.float32, order="C", copy=True))
            for i, (index, anchor, episode) in enumerate(zip(indices.tolist(), anchors.tolist(), episodes.tolist(), strict=True))
        }
        self.profile_lock_sha256 = lock_sha

    def row(self, cache_index: int, anchor_id: str, episode_id: str) -> np.ndarray:
        key = int(cache_index)
        if key not in self._by_cache:
            raise PredictorContractError(f"current-profile feature cache index is absent: {key}")
        stored_anchor, stored_episode, feature = self._by_cache[key]
        if stored_anchor != str(anchor_id) or stored_episode != int(episode_id):
            raise PredictorContractError(f"current-profile feature row identity mismatch at cache index {key}")
        return feature


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise PredictorContractError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PredictorContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _safe_path(path: Path | str, role: str) -> Path:
    resolved = Path(path).resolve()
    # The accepted train-derived normalizer lives under a historical study
    # directory whose name contains TARGET_DISTILLATION. Allow only this exact
    # hash-pinned artifact; verify_file_sha256 checks its expected digest next.
    if role == "artifact" and resolved == ACCEPTED_NORMALIZER_PATH.resolve():
        return resolved
    parts = resolved.parts
    for index, part in enumerate(parts):
        is_prefix_basename = (
            role == "prefix"
            and index == len(parts) - 1
            and part.upper().endswith("PREFIXES_NO_LABELS.NPZ")
        )
        if is_prefix_basename:
            continue
        if _FORBIDDEN_KEY.search(part):
            raise PredictorContractError(f"{role} path contains a forbidden data term: {path}")
    return resolved


def verify_file_sha256(path: Path, expected_sha256: str, role: str) -> str:
    _safe_path(path, "prefix" if role == "prefix" else "artifact")
    expected = _validate_sha(expected_sha256, f"expected {role} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise PredictorContractError(f"{role} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _assert_safe_mapping_keys(value: Mapping[str, Any], context: str) -> None:
    for key, nested in value.items():
        key_text = str(key)
        if _FORBIDDEN_KEY.search(key_text):
            raise PredictorContractError(f"{context} contains a forbidden key: {key_text}")
        if isinstance(nested, Mapping):
            _assert_safe_mapping_keys(nested, context)
        elif isinstance(nested, (list, tuple)):
            for item in nested:
                if isinstance(item, Mapping):
                    _assert_safe_mapping_keys(item, context)


def validate_runtime_job(job: Mapping[str, Any]) -> None:
    """Reject target/label keys and any unsafe path embedded in a runtime job."""
    if not isinstance(job, Mapping):
        raise PredictorContractError("runtime job must be a mapping")
    _assert_safe_mapping_keys(job, "runtime job")

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for nested_key, nested in value.items():
                walk(nested, str(nested_key))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                walk(nested, key)
        elif isinstance(value, str) and (value.startswith("/") or ".npz" in value.lower() or ".json" in value.lower()):
            role = "prefix" if "prefix" in key.lower() else "artifact"
            _safe_path(value, role)

    walk(job)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorContractError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PredictorContractError(f"JSON contract must be an object: {path}")
    return value


def verify_final_access_receipt(path: Path, expected_sha256: str) -> FinalAccessAuthorization:
    """Verify the parent-written receipt required before any final_test input read."""
    receipt_path = Path(path)
    if receipt_path.name != "FINAL_ACCESS_OPENED.json" or receipt_path.parent.name != "protocol":
        raise PredictorContractError("final access receipt must be protocol/FINAL_ACCESS_OPENED.json")
    digest = verify_file_sha256(receipt_path, expected_sha256, "final access receipt")
    receipt = _read_json(receipt_path)
    required = {
        "status", "protocol_id", "authorization_source", "opened_utc",
        "study_lock_sha256", "freeze_receipt_sha256",
        "first_final_payload_access_utc", "access_stage", "core_b_access_authorized",
    }
    missing = required - set(receipt)
    if missing:
        raise PredictorContractError(f"final access receipt is incomplete: {sorted(missing)}")
    if receipt.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK":
        raise PredictorContractError("final access receipt is not OPENED_BY_USER_AUTHORIZED_TASK")
    if receipt.get("core_b_access_authorized") is not True:
        raise PredictorContractError("final access receipt does not authorize Core B")
    if "user" not in str(receipt.get("authorization_source", "")).lower():
        raise PredictorContractError("final access receipt does not cite user authorization")
    for field in ("protocol_id", "authorization_source", "opened_utc", "first_final_payload_access_utc", "access_stage"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise PredictorContractError(f"final access receipt field {field} must be nonempty text")
    for field in ("study_lock_sha256", "freeze_receipt_sha256"):
        _validate_sha(receipt[field], f"final access receipt {field}")
    expected_receipt_path = (V2_OUT_ROOT / "protocol/FINAL_ACCESS_OPENED.json").resolve()
    if receipt_path.resolve() != expected_receipt_path:
        raise PredictorContractError("V2 final access receipt must be the canonical protocol receipt")
    if receipt.get("protocol_id") != V2_PROTOCOL_ID:
        raise PredictorContractError("V2 final access protocol ID mismatch")
    freeze_sha = sha256_file(V2_FINAL_FREEZE_PATH)
    if receipt.get("study_lock_sha256") != freeze_sha or receipt.get("freeze_receipt_sha256") != freeze_sha:
        raise PredictorContractError("V2 final access receipt does not bind the pre-access final freeze")
    if receipt.get("core_b_profile_lock_sha256") != sha256_file(CURRENT_PROFILE_LOCK_PATH):
        raise PredictorContractError("V2 final access receipt does not bind CURRENT_PROFILE_V2")
    if receipt.get("current_profile_dev_acceptance_sha256") != sha256_file(V2_DEV_ACCEPTANCE_PATH):
        raise PredictorContractError("V2 final access receipt does not bind current-profile dev acceptance")
    return FinalAccessAuthorization(receipt_path, digest, _FINAL_AUTH_SENTINEL)


def _require_final_access_authorization(
    split: str,
    authorization: FinalAccessAuthorization | None,
) -> None:
    if split == "final_test":
        if not isinstance(authorization, FinalAccessAuthorization) or authorization._sentinel is not _FINAL_AUTH_SENTINEL:
            raise PredictorContractError("final_test inference requires a verified parent access receipt")
        verify_final_access_receipt(authorization.path, authorization.sha256)
    elif authorization is not None:
        raise PredictorContractError("final access receipt cannot be attached to a non-final split")


def load_accepted_sources(source_code_root: Path = ACCEPTED_V9_CODE_ROOT) -> AcceptedSources:
    """Verify and import the exact V9 model/input/cache source closure."""
    source_code_root = Path(source_code_root).resolve()
    if source_code_root != ACCEPTED_V9_CODE_ROOT.resolve():
        raise PredictorContractError(f"V9 source root is not the accepted source: {source_code_root}")
    code_lock_path = source_code_root.parent / "protocol" / "CODE_LOCK.json"
    if sha256_file(code_lock_path) != ACCEPTED_CODE_LOCK_SHA256:
        raise PredictorContractError("accepted V9 CODE_LOCK.json hash changed")
    code_lock = _read_json(code_lock_path)
    if code_lock.get("status") != "LOCKED_BEFORE_STAGE0_AND_FORMAL_FITS":
        raise PredictorContractError("accepted V9 code lock is not in its locked state")
    for relative_path, expected in PINNED_SOURCE_HASHES.items():
        file_path = source_code_root / relative_path
        actual = sha256_file(file_path)
        if actual != expected:
            raise PredictorContractError(f"accepted V9 source hash mismatch for {relative_path}")
        lock_entry = code_lock.get("files", {}).get(f"code/{relative_path}")
        if not isinstance(lock_entry, dict) or lock_entry.get("sha256") != expected:
            raise PredictorContractError(f"CODE_LOCK does not bind {relative_path}")

    # Imports must not write pyc files into the sealed historical source tree.
    sys.dont_write_bytecode = True
    locked_source = source_code_root / "locked_source"
    for directory in (str(locked_source), str(source_code_root)):
        if directory not in sys.path:
            sys.path.insert(0, directory)

    expected_module_paths = {
        "v9_common": source_code_root / "v9_common.py",
        "prepare_v2_data": source_code_root / "prepare_v2_data.py",
        "v2_model": source_code_root / "v2_model.py",
        "D3_model": locked_source / "D3_model.py",
        "d3_eq_contract": locked_source / "d3_eq_contract.py",
    }
    for module_name, expected_path in expected_module_paths.items():
        loaded = sys.modules.get(module_name)
        loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
        if loaded_path is not None and Path(loaded_path).resolve() != expected_path.resolve():
            raise PredictorContractError(f"conflicting module already loaded: {module_name} from {loaded_path}")

    common = importlib.import_module("v9_common")
    data_module = importlib.import_module("prepare_v2_data")
    model_module = importlib.import_module("v2_model")
    d3_module = importlib.import_module("D3_model")
    eq_module = importlib.import_module("d3_eq_contract")
    modules = {
        "v9_common": common,
        "prepare_v2_data": data_module,
        "v2_model": model_module,
        "D3_model": d3_module,
        "d3_eq_contract": eq_module,
    }
    for module_name, module in modules.items():
        if Path(module.__file__).resolve() != expected_module_paths[module_name].resolve():
            raise PredictorContractError(f"import resolved outside accepted V9 source: {module_name}")
    return AcceptedSources(
        model_class=model_module.V2TaskConditionedPredictor,
        model_input=common.model_input,
        q0_grid_class=data_module.Q0ShardedGrid,
        state_hash=model_module.state_hash,
    )


def validate_model_manifest(
    manifest: Mapping[str, Any],
    *,
    program_id: str,
    seed: int,
    video_mode: str,
    model_sha256: str,
    normalizer_sha256: str,
) -> str:
    """Validate accepted model/seed/mode/normalizer identity and return state hash."""
    key = (program_id, int(seed))
    if key not in PINNED_MODEL_IDENTITIES:
        raise PredictorContractError(f"unsupported Core B model cell: {program_id}/{seed}")
    pinned_manifest_sha, pinned_model_sha, pinned_state_hash = PINNED_MODEL_IDENTITIES[key]
    if _validate_sha(model_sha256, "model SHA-256") != pinned_model_sha:
        raise PredictorContractError(f"model SHA does not match accepted {program_id}/{seed} checkpoint")
    if manifest.get("status") != "PASS":
        raise PredictorContractError("model manifest is not accepted")
    if int(manifest.get("seed", -1)) != int(seed):
        raise PredictorContractError("model manifest seed mismatch")
    expected_mode, expected_dropout = PROGRAM_SPECS[program_id]
    if video_mode != expected_mode or manifest.get("video_mode") != expected_mode:
        raise PredictorContractError("model/video mode mismatch")
    if manifest.get("model_file_sha256") != pinned_model_sha:
        raise PredictorContractError("model manifest weight hash mismatch")
    model_file = manifest.get("model_file")
    if not isinstance(model_file, str) or not model_file.strip():
        raise PredictorContractError("model manifest does not identify a checkpoint file")
    if manifest.get("normalizer_sha256") != ACCEPTED_NORMALIZER_SHA256:
        raise PredictorContractError("model manifest normalizer identity mismatch")
    if _validate_sha(normalizer_sha256, "normalizer SHA-256") != ACCEPTED_NORMALIZER_SHA256:
        raise PredictorContractError("normalizer is not the accepted V8/V9 normalizer")
    if int(manifest.get("final_heldout_payload_access", 0)) != 0:
        raise PredictorContractError("accepted model manifest reports prior heldout payload access")
    if program_id == "S_OPT":
        if manifest.get("model_id") != "G-S" or manifest.get("waypoint_mode") is not None:
            raise PredictorContractError("S_OPT must bind to the accepted five-task G-S model")
        manifest_dropout = 0.0
    else:
        if manifest.get("program_id") != program_id:
            raise PredictorContractError("V9 program ID mismatch")
        manifest_dropout = float(manifest.get("numeric_context_drop_probability", -1.0))
        if manifest.get("teacher_target_used") is not False:
            raise PredictorContractError("V9 model manifest is not explicitly teacher-free")
    if manifest.get("teacher_target_used") not in (None, False):
        raise PredictorContractError("accepted checkpoint manifest has teacher-target use")
    if not np.isclose(manifest_dropout, expected_dropout, rtol=0.0, atol=0.0):
        raise PredictorContractError("training numeric-context dropout identity mismatch")
    task_ids = tuple(int(value) for value in manifest.get("task_ids", TASK_IDS))
    if task_ids != TASK_IDS:
        raise PredictorContractError("model manifest task order differs from accepted V9 task order")
    if str(manifest.get("final_state_hash", "")) != pinned_state_hash:
        raise PredictorContractError("model manifest state hash differs from accepted source map")
    return pinned_state_hash


def validate_normalizer(normalizer: Mapping[str, Any], *, video_mode: str) -> np.ndarray:
    required = {
        "state_mean": (71,),
        "state_safe_std": (71,),
    }
    for field, shape in required.items():
        array = np.asarray(normalizer.get(field), dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise PredictorContractError(f"normalizer {field} shape/finite check failed")
        if field.endswith("safe_std") and np.any(array <= 0):
            raise PredictorContractError(f"normalizer {field} must be positive")
    for field in ("relative_delta_mean", "relative_delta_safe_std"):
        value = np.asarray(normalizer.get(field), dtype=np.float64)
        if value.size != 1 or not np.isfinite(value).all():
            raise PredictorContractError(f"normalizer {field} must be one finite scalar")
        if field.endswith("safe_std") and float(value.reshape(-1)[0]) <= 0:
            raise PredictorContractError(f"normalizer {field} must be positive")
    scale = np.asarray(normalizer.get("target_sigma"), dtype=np.float32)
    if scale.shape != (28,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise PredictorContractError("normalizer target_sigma must be finite, positive, and [28]")
    if video_mode == "TS11-D":
        for field in ("video_mean", "video_safe_std"):
            array = np.asarray(normalizer.get(field), dtype=np.float64)
            if array.shape != (768,) or not np.isfinite(array).all():
                raise PredictorContractError(f"normalizer {field} shape/finite check failed")
            if field.endswith("safe_std") and np.any(array <= 0):
                raise PredictorContractError(f"normalizer {field} must be positive")
    return scale


def _string_vector(value: Any, field: str, n_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != n_rows or array.dtype.kind not in "US":
        raise PredictorContractError(f"prefix {field} must be a string vector of length N")
    result = array.astype(str, copy=False)
    if any(not item for item in result.tolist()):
        raise PredictorContractError(f"prefix {field} contains an empty value")
    return result


def _int_vector(value: Any, field: str, n_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != n_rows or array.dtype.kind not in "iu":
        raise PredictorContractError(f"prefix {field} must be an integer vector of length N")
    return array.astype(np.int64, copy=False)


def validate_prefix_data(
    data: Mapping[str, Any],
    *,
    canonical_ids: Sequence[str],
    split: str,
) -> dict[str, np.ndarray]:
    """Validate and return the label-free V9 prefix fields in archive row order."""
    if split not in SPLITS:
        raise PredictorContractError(f"unsupported split identifier: {split}")
    keys = {str(key) for key in data}
    bad = sorted(key for key in keys if _FORBIDDEN_KEY.search(key))
    if bad:
        raise PredictorContractError(f"prefix data contains forbidden keys: {bad}")
    missing = PREFIX_REQUIRED_KEYS - keys
    extra = keys - PREFIX_KEYS
    if missing or extra:
        raise PredictorContractError(f"prefix schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    anchor_ids_raw = np.asarray(data["anchor_id"])
    if anchor_ids_raw.ndim != 1 or anchor_ids_raw.dtype.kind not in "US":
        raise PredictorContractError("prefix anchor_id must be a string vector")
    n_rows = len(anchor_ids_raw)
    if n_rows < 1:
        raise PredictorContractError("prefix archive is empty")
    anchor_ids = _string_vector(anchor_ids_raw, "anchor_id", n_rows)
    episode_ids = _int_vector(data["episode_id"], "episode_id", n_rows)
    task_ids = _int_vector(data["task_id"], "task_id", n_rows)
    task_ranks = _int_vector(data["task_rank"], "task_rank", n_rows)
    cache_indices = _int_vector(data["cache_index"], "cache_index", n_rows)
    anchor_frames = _int_vector(data["anchor_frame"], "anchor_frame", n_rows)
    source_clusters = _string_vector(data["source_cluster"], "source_cluster", n_rows)
    if np.any(episode_ids < 0) or np.any(cache_indices < 0) or np.any(anchor_frames < 0):
        raise PredictorContractError("prefix episode/cache/frame indices must be nonnegative")
    if any(int(task) not in TASK_IDS for task in task_ids.tolist()):
        raise PredictorContractError("prefix contains an unrecognized task ID")
    expected_ranks = np.asarray([TASK_IDS.index(int(task)) for task in task_ids], dtype=np.int64)
    if not np.array_equal(task_ranks, expected_ranks):
        raise PredictorContractError("prefix task_rank disagrees with the fixed V9 task order")

    shapes = {
        "history_native": (n_rows, 32, 71),
        "history_mask": (n_rows, 32),
        "relative_delta": (n_rows, 32),
        "anchor_native": (n_rows, 28),
    }
    numeric_fields = set(shapes)
    for field, shape in shapes.items():
        array = np.asarray(data[field])
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise PredictorContractError(f"prefix {field} has wrong shape/dtype: {array.shape}/{array.dtype}")
        if not np.isfinite(array).all():
            raise PredictorContractError(f"prefix {field} contains nonfinite values")
    history_mask = np.asarray(data["history_mask"])
    if not np.isin(history_mask, (0, 1)).all():
        raise PredictorContractError("prefix history_mask must contain only 0/1")

    ids = np.asarray(canonical_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) != n_rows:
        raise PredictorContractError("ordered canonical IDs must be a string vector of length N")
    ids = ids.astype(str, copy=False)
    if any(not item for item in ids.tolist()) or len(set(ids.tolist())) != n_rows:
        raise PredictorContractError("canonical IDs must be nonempty and unique")
    if split == "final_test" and "canonical_id" not in keys:
        raise PredictorContractError("final prefix must carry the parent-built canonical_id field")
    if "canonical_id" in keys:
        archived_ids = _string_vector(data["canonical_id"], "canonical_id", n_rows)
        if not np.array_equal(archived_ids, ids):
            raise PredictorContractError("canonical ID order differs from prefix archive order")
    if "split" in keys:
        split_values = _string_vector(data["split"], "split", n_rows)
        if not np.all(split_values == split):
            raise PredictorContractError("prefix split values differ from the requested split")

    if {"dataset_revision", "source_key", "literal_source_key"}.issubset(keys):
        revision_raw = np.asarray(data["dataset_revision"])
        if revision_raw.ndim == 0:
            revisions = np.repeat(str(revision_raw.item()), n_rows)
        else:
            revisions = _string_vector(revision_raw, "dataset_revision", n_rows)
        source_keys = _string_vector(data["source_key"], "source_key", n_rows)
        literal_keys = _string_vector(data["literal_source_key"], "literal_source_key", n_rows)
        split_values = _string_vector(data["split"], "split", n_rows) if "split" in keys else np.repeat(split, n_rows)
        expected_ids = np.asarray([
            f"{revisions[i]}|{source_keys[i]}|{literal_keys[i]}|{int(episode_ids[i])}|{int(task_ids[i])}|{anchor_ids[i]}|{split_values[i]}"
            for i in range(n_rows)
        ], dtype=str)
        if not np.array_equal(expected_ids, ids):
            raise PredictorContractError("canonical IDs do not bind revision/source/episode/task/anchor/split")

    clean = {key: np.asarray(data[key]) for key in keys}
    # The accepted adapter only reads this fixed prefix subset. Optional fields
    # remain metadata for order/identity validation and are never sent to model_input.
    for field in numeric_fields:
        clean[field] = np.asarray(data[field])
    return clean


def load_prefix_npz(
    path: Path,
    *,
    expected_sha256: str,
    canonical_ids: Sequence[str],
    split: str,
    final_access_authorization: FinalAccessAuthorization | None = None,
) -> dict[str, np.ndarray]:
    _require_final_access_authorization(split, final_access_authorization)
    path = _safe_path(path, "prefix")
    verify_file_sha256(path, expected_sha256, "prefix")
    try:
        with np.load(path, allow_pickle=False) as archive:
            archive_keys = set(archive.files)
            bad = sorted(key for key in archive_keys if _FORBIDDEN_KEY.search(key))
            if bad:
                raise PredictorContractError(f"prefix archive contains forbidden keys: {bad}")
            missing = PREFIX_REQUIRED_KEYS - archive_keys
            extra = archive_keys - PREFIX_KEYS
            if missing or extra:
                raise PredictorContractError(f"prefix archive key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
            arrays = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, PredictorContractError):
            raise
        raise PredictorContractError(f"cannot read label-free prefix archive {path}: {exc}") from exc
    return validate_prefix_data(arrays, canonical_ids=canonical_ids, split=split)


def bind_feature_reader(
    reader: Any,
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    split: str,
    source: AcceptedSources,
    final_access_opened_path: Path | None = None,
    expected_final_access_opened_sha256: str | None = None,
) -> Any:
    """Bind a row reader to a hash-pinned cache manifest.

    For train/dev, only the accepted Q0ShardedGrid and a manifest marked
    final_test_access=0 are allowed. final_test requires a parent-created reader,
    an explicit FINAL_ACCESS_OPENED.json hash, and a non-Q0 manifest marker.
    """
    if split not in SPLITS:
        raise PredictorContractError(f"unsupported cache split: {split}")
    if split != "final_test" and (final_access_opened_path is not None or expected_final_access_opened_sha256 is not None):
        raise PredictorContractError("final access receipt cannot be used for a non-final cache read")
    manifest_path = _safe_path(manifest_path, "artifact")
    verify_file_sha256(manifest_path, expected_manifest_sha256, "feature cache manifest")
    manifest = _read_json(manifest_path)
    _assert_safe_mapping_keys(manifest, "feature cache manifest")
    if manifest.get("status") != "PASS":
        raise PredictorContractError("feature cache manifest is not accepted")
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        raise PredictorContractError("feature cache shards must be a list")
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise PredictorContractError("feature cache shard record must be an object")
        for path_field in ("array", "meta"):
            if path_field in shard:
                _safe_path(str(shard[path_field]), "artifact")
    if not callable(getattr(reader, "row", None)):
        raise PredictorContractError("feature reader must expose row(cache_index, anchor_id, episode_id)")
    reader_manifest = getattr(reader, "manifest", None)
    if reader_manifest != manifest:
        raise PredictorContractError("feature reader is not bound to the hash-pinned manifest")

    is_q0 = isinstance(reader, source.q0_grid_class)
    is_current = isinstance(reader, CurrentProfileGrid)
    if split in {"train", "dev"}:
        if is_q0:
            if int(manifest.get("final_test_access", -1)) != 0:
                raise PredictorContractError("accepted Q0 reader requires final_test_access=0")
            if Path(manifest_path).resolve() != Path(ACCEPTED_V9_ROOT.parent / "CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json").resolve():
                raise PredictorContractError("train/dev Q0 cache path differs from the accepted source")
            if expected_manifest_sha256 != ACCEPTED_Q0_CACHE_SHA256:
                raise PredictorContractError("train/dev Q0 cache hash differs from the accepted source")
        elif is_current and split == "dev":
            if Path(manifest_path).resolve() != CURRENT_PROFILE_DEV_MANIFEST.resolve():
                raise PredictorContractError("dev current-profile reader must use the canonical V2 manifest")
            if manifest.get("split") != "dev" or manifest.get("profile_id") != "CURRENT_PROFILE_V2":
                raise PredictorContractError("dev reader is not bound to CURRENT_PROFILE_V2")
            if manifest.get("profile_lock_sha256") != sha256_file(CURRENT_PROFILE_LOCK_PATH):
                raise PredictorContractError("dev feature manifest lock hash mismatch")
        else:
            raise PredictorContractError("train/dev feature reads require accepted Q0 or explicit CURRENT_PROFILE_V2 dev grid")
    else:
        if is_q0 or not is_current:
            raise PredictorContractError("V2 final_test feature reads require CURRENT_PROFILE_V2 grid, never Q0")
        if Path(manifest_path).resolve() != CURRENT_PROFILE_FINAL_MANIFEST.resolve():
            raise PredictorContractError("final current-profile reader must use the canonical V2 manifest")
        if final_access_opened_path is None or expected_final_access_opened_sha256 is None:
            raise PredictorContractError("final_test feature reads require parent V2 FINAL_ACCESS_OPENED.json")
        receipt_path = Path(final_access_opened_path)
        final_auth = verify_final_access_receipt(receipt_path, expected_final_access_opened_sha256)
        if manifest.get("split") != "final_test" or manifest.get("profile_id") != "CURRENT_PROFILE_V2":
            raise PredictorContractError("final feature manifest is not the frozen current profile")
        if manifest.get("final_access_opened_sha256") != final_auth.sha256:
            raise PredictorContractError("final feature manifest does not bind the V2 access receipt")
        if manifest.get("core_b_v2_final_freeze_sha256") != sha256_file(V2_FINAL_FREEZE_PATH):
            raise PredictorContractError("final feature manifest does not bind the V2 pre-access freeze")
        if manifest.get("profile_lock_sha256") != sha256_file(CURRENT_PROFILE_LOCK_PATH):
            raise PredictorContractError("final feature manifest current-profile lock mismatch")
    return reader


def module_state_sha256(model: torch.nn.Module) -> str:
    """Stable generic state digest for synthetic tests and non-accepted modules."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(np.asarray(cpu.shape, dtype="<i8").tobytes())
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


class CoreBPredictor:
    """Callable whose runtime inputs are limited to prefix tensors and task IDs."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        program_id: str,
        video_mode: str,
        native_scale: np.ndarray,
        expected_state_sha256: str,
        state_hash_fn: Callable[[torch.nn.Module], str] = module_state_sha256,
    ) -> None:
        if program_id not in PROGRAM_SPECS or PROGRAM_SPECS[program_id][0] != video_mode:
            raise PredictorContractError("predictor program/video mode is unsupported")
        scale = np.asarray(native_scale, dtype=np.float32)
        if scale.shape != (28,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise PredictorContractError("native output scale must be finite, positive, and [28]")
        if model.training:
            raise PredictorContractError("model must be in eval mode before predictor construction")
        self._model = model
        self._program_id = program_id
        self._video_mode = video_mode
        self._native_scale = torch.as_tensor(scale, dtype=torch.float32)
        self._expected_state_sha256 = _validate_sha(expected_state_sha256, "expected model state SHA-256")
        self._state_hash_fn = state_hash_fn
        if self._state_hash_fn(self._model) != self._expected_state_sha256:
            raise PredictorContractError("loaded model state does not match the accepted model identity")

    def __call__(self, prefix_inputs: Mapping[str, torch.Tensor], task_id: torch.Tensor) -> np.ndarray:
        keys = {str(key) for key in prefix_inputs}
        bad = sorted(key for key in keys if _FORBIDDEN_KEY.search(key))
        if bad:
            raise PredictorContractError(f"model prefix contains forbidden keys: {bad}")
        expected_keys = MODEL_INPUT_KEYS[self._video_mode]
        if keys != expected_keys:
            raise PredictorContractError(f"model prefix key mismatch: expected={sorted(expected_keys)}, got={sorted(keys)}")
        state = prefix_inputs["state"]
        if not isinstance(state, torch.Tensor) or state.ndim != 3 or tuple(state.shape[1:]) != (32, 71):
            raise PredictorContractError("model state input must be [B,32,71]")
        batch_size = int(state.shape[0])
        expected_shapes = {
            "history_mask": (batch_size, 32),
            "relative_delta": (batch_size, 32),
            "anchor_command": (batch_size, 28),
        }
        for field, shape in expected_shapes.items():
            value = prefix_inputs[field]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise PredictorContractError(f"model input {field} must have shape {shape}")
        if self._video_mode == "TS11-D":
            for field in ("dynamic", "static"):
                value = prefix_inputs[field]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size, 16, 4, 768):
                    raise PredictorContractError(f"model input {field} must be [B,16,4,768]")
        if not isinstance(task_id, torch.Tensor) or tuple(task_id.shape) != (batch_size,) or task_id.dtype != torch.long:
            raise PredictorContractError("task_id must be a [B] long vector")
        if torch.any((task_id < 0) | (task_id >= len(TASK_IDS))):
            raise PredictorContractError("task_id contains an unsupported task rank")
        for value in (*prefix_inputs.values(), task_id):
            if not torch.isfinite(value).all():
                raise PredictorContractError("model input contains nonfinite values")
        if self._model.training:
            raise PredictorContractError("model left eval mode during inference")
        before = self._state_hash_fn(self._model)
        if before != self._expected_state_sha256:
            raise PredictorContractError("model state changed before inference call")
        with torch.inference_mode():
            output = self._model(dict(prefix_inputs), task_id)
            residual = getattr(output, "action_residual", None)
            if not isinstance(residual, torch.Tensor):
                raise PredictorContractError("accepted model output must expose action_residual")
            if tuple(residual.shape) != (batch_size, 30, 28):
                raise PredictorContractError(f"model residual must be [B,30,28], got {tuple(residual.shape)}")
            anchor = prefix_inputs["anchor_command"].to(dtype=torch.float32)
            scale = self._native_scale.to(device=residual.device, dtype=torch.float32)
            prediction = anchor[:, None, :] + scale[None, None, :] * residual.to(dtype=torch.float32)
        after = self._state_hash_fn(self._model)
        if after != before:
            raise PredictorContractError("model state changed during inference")
        native = prediction.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        if native.shape != (batch_size, 30, 28) or not np.isfinite(native).all():
            raise PredictorContractError("native prediction failed shape/finite validation")
        return native


def _ids_sha256(ids: np.ndarray) -> str:
    canonical = json.dumps(ids.astype(str).tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _prediction_sha256(prediction_native: np.ndarray) -> str:
    values = np.asarray(prediction_native, dtype="<f8", order="C")
    header = f"float64-le:{values.shape}".encode("ascii") + b"\0"
    return hashlib.sha256(header + values.tobytes(order="C")).hexdigest()


def build_prediction_bundle(
    canonical_ids: Sequence[str],
    prediction_native: np.ndarray,
    *,
    program_id: str,
    seed: int | None,
) -> PredictionBundle:
    ids = np.asarray(canonical_ids)
    prediction = np.array(prediction_native, dtype=np.float64, order="C", copy=True)
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) < 1:
        raise PredictorContractError("prediction IDs must be a nonempty string vector")
    ids = ids.astype(str, copy=True)
    if any(not value for value in ids.tolist()) or len(set(ids.tolist())) != len(ids):
        raise PredictorContractError("prediction IDs must be nonempty and unique")
    if prediction.shape != (len(ids), 30, 28):
        raise PredictorContractError(f"prediction_native must be [N,30,28], got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise PredictorContractError("prediction_native contains nonfinite values")
    if program_id != "PERSISTENCE" and (program_id not in PROGRAM_SPECS or seed not in SEEDS):
        raise PredictorContractError("prediction bundle has unsupported model identity")
    if program_id == "PERSISTENCE" and seed is not None:
        raise PredictorContractError("persistence is one reference, not a fitted training seed")
    ids.setflags(write=False)
    prediction.setflags(write=False)
    id_hash = _ids_sha256(ids)
    prediction_hash = _prediction_sha256(prediction)
    bundle_hash = hashlib.sha256(f"{id_hash}:{prediction_hash}".encode("ascii")).hexdigest()
    return PredictionBundle(program_id, seed, ids, prediction, id_hash, prediction_hash, bundle_hash)


def write_prediction_bundle(path: Path, bundle: PredictionBundle) -> dict[str, Any]:
    """Write only canonical_ids and prediction_native; never overwrite an artifact."""
    path = _safe_path(path, "artifact")
    if path.suffix.lower() != ".npz":
        raise PredictorContractError("prediction output must be a .npz file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PredictorContractError(f"refusing to overwrite prediction output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, canonical_ids=bundle.canonical_ids, prediction_native=bundle.prediction_native)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as check:
            if set(check.files) != {"canonical_ids", "prediction_native"}:
                raise PredictorContractError("written prediction archive has unexpected keys")
            stored_ids = check["canonical_ids"].astype(str)
            stored_prediction = check["prediction_native"]
        checked = build_prediction_bundle(stored_ids, stored_prediction, program_id=bundle.program_id, seed=bundle.seed)
        if checked.bundle_sha256 != bundle.bundle_sha256:
            raise PredictorContractError("prediction archive checksum differs after reload")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "canonical_ids_sha256": bundle.canonical_ids_sha256,
        "prediction_native_sha256": bundle.prediction_native_sha256,
        "bundle_sha256": bundle.bundle_sha256,
        "shape": list(bundle.prediction_native.shape),
        "dtype": "float64",
    }


def predict_rows(
    data: Mapping[str, Any],
    *,
    canonical_ids: Sequence[str],
    split: str,
    predictor: CoreBPredictor,
    normalizer: Mapping[str, Any],
    model_input_fn: Callable[..., Any],
    device: torch.device | str,
    feature_reader: Any = None,
    batch_size: int = 64,
    final_access_authorization: FinalAccessAuthorization | None = None,
) -> np.ndarray:
    """Replay ordered label-free rows using the accepted V9 model_input function."""
    _require_final_access_authorization(split, final_access_authorization)
    if batch_size < 1:
        raise PredictorContractError("batch_size must be positive")
    safe_data = validate_prefix_data(data, canonical_ids=canonical_ids, split=split)
    scale = validate_normalizer(normalizer, video_mode=predictor._video_mode)
    captured_scale = predictor._native_scale.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.array_equal(scale, captured_scale):
        raise PredictorContractError("predictor output scale differs from the hash-pinned normalizer")
    if predictor._video_mode == "TS11-D" and feature_reader is None:
        raise PredictorContractError("video model requires an injected feature reader")
    if predictor._video_mode == "STATE-CTRL" and feature_reader is not None:
        raise PredictorContractError("state-only model must not read the video feature cache")
    device = torch.device(device)
    outputs: list[np.ndarray] = []
    n_rows = len(np.asarray(safe_data["anchor_id"]))
    # Match the accepted V9 runner's no_grad inference context exactly.
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            indices = np.arange(start, min(start + batch_size, n_rows), dtype=np.int64)
            prefix_inputs, task_id, _anchor = model_input_fn(
                safe_data, indices, dict(normalizer), predictor._video_mode, device, feature_reader
            )
            # No drop_mask is passed: A0/AD/V0/VD all use full observed context at eval.
            outputs.append(predictor(prefix_inputs, task_id))
    prediction = np.concatenate(outputs, axis=0).astype(np.float64, copy=False)
    if prediction.shape != (n_rows, 30, 28) or not np.isfinite(prediction).all():
        raise PredictorContractError("batched prediction failed final shape/finite/order checks")
    return prediction


def make_persistence_bundle(
    canonical_ids: Sequence[str],
    anchor_native: np.ndarray,
) -> PredictionBundle:
    anchor = np.asarray(anchor_native, dtype=np.float64)
    ids = np.asarray(canonical_ids)
    if anchor.shape != (len(ids), 28) or not np.isfinite(anchor).all():
        raise PredictorContractError("persistence anchors must be finite [N,28]")
    repeated = np.repeat(anchor[:, None, :], 30, axis=1)
    return build_prediction_bundle(ids, repeated, program_id="PERSISTENCE", seed=None)


def make_persistence_aliases(bundle: PredictionBundle, seeds: Sequence[int] = SEEDS) -> tuple[PersistenceAlias, ...]:
    if bundle.program_id != "PERSISTENCE" or bundle.seed is not None:
        raise PredictorContractError("persistence aliases require the single unseeded persistence bundle")
    if tuple(int(seed) for seed in seeds) != SEEDS:
        raise PredictorContractError("persistence aliases must reference the three fixed V9 seed slots")
    return tuple(
        PersistenceAlias("PERSISTENCE", seed, bundle.canonical_ids_sha256,
                         bundle.prediction_native_sha256, tuple(bundle.prediction_native.shape),
                         bundle.bundle_sha256)
        for seed in SEEDS
    )


def predict_cell(
    cell: ModelCell,
    *,
    source_code_root: Path = ACCEPTED_V9_CODE_ROOT,
    canonical_ids: Sequence[str],
    split: str,
    feature_reader: Any = None,
    final_access_opened_path: Path | None = None,
    expected_final_access_opened_sha256: str | None = None,
    output_path: Path | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> tuple[PredictionBundle, dict[str, Any] | None]:
    """Verify identities, safely load an accepted checkpoint, predict, and optionally seal an NPZ."""
    if split not in SPLITS:
        raise PredictorContractError(f"unsupported prediction split: {split}")
    if split == "final_test":
        if final_access_opened_path is None or expected_final_access_opened_sha256 is None:
            raise PredictorContractError("final_test inference requires parent FINAL_ACCESS_OPENED.json")
        final_authorization = verify_final_access_receipt(
            final_access_opened_path, expected_final_access_opened_sha256
        )
    else:
        if final_access_opened_path is not None or expected_final_access_opened_sha256 is not None:
            raise PredictorContractError("final access receipt cannot be used for a non-final prediction")
        final_authorization = None
    if int(cell.seed) not in SEEDS or cell.program_id not in PROGRAM_SPECS:
        raise PredictorContractError("cell is outside the accepted Core B model matrix")
    expected_mode, _ = PROGRAM_SPECS[cell.program_id]
    if cell.video_mode != expected_mode:
        raise PredictorContractError("cell video mode differs from frozen model matrix")
    if (cell.program_id, int(cell.seed)) not in PINNED_MODEL_IDENTITIES:
        raise PredictorContractError("cell does not bind to an accepted checkpoint")
    manifest_sha, model_sha, expected_state_hash = PINNED_MODEL_IDENTITIES[(cell.program_id, int(cell.seed))]
    if cell.expected_model_sha256 != model_sha or cell.expected_model_manifest_sha256 != manifest_sha:
        raise PredictorContractError("cell hashes differ from pinned accepted model identity")
    if cell.expected_normalizer_sha256 != ACCEPTED_NORMALIZER_SHA256:
        raise PredictorContractError("cell normalizer hash differs from accepted V9 normalizer")
    verify_file_sha256(cell.model_path, cell.expected_model_sha256, "model")
    verify_file_sha256(cell.model_manifest_path, cell.expected_model_manifest_sha256, "model manifest")
    verify_file_sha256(cell.normalizer_path, cell.expected_normalizer_sha256, "normalizer")
    normalizer = _read_json(cell.normalizer_path)
    manifest = _read_json(cell.model_manifest_path)
    manifest_model_path = manifest.get("model_file")
    if not isinstance(manifest_model_path, str) or Path(manifest_model_path).resolve() != Path(cell.model_path).resolve():
        raise PredictorContractError("model manifest checkpoint path differs from the bound model path")
    checked_state_hash = validate_model_manifest(
        manifest,
        program_id=cell.program_id,
        seed=int(cell.seed),
        video_mode=cell.video_mode,
        model_sha256=cell.expected_model_sha256,
        normalizer_sha256=cell.expected_normalizer_sha256,
    )
    if checked_state_hash != expected_state_hash:
        raise PredictorContractError("accepted model state hash changed")
    scale = validate_normalizer(normalizer, video_mode=cell.video_mode)
    data = load_prefix_npz(
        cell.prefix_path,
        expected_sha256=cell.expected_prefix_sha256,
        canonical_ids=canonical_ids,
        split=split,
        final_access_authorization=final_authorization,
    )

    sources = load_accepted_sources(source_code_root)
    if cell.video_mode == "TS11-D":
        if cell.feature_cache_manifest_path is None or cell.expected_feature_cache_manifest_sha256 is None:
            raise PredictorContractError("video model cell requires a cache manifest path and expected hash")
        if feature_reader is None:
            raise PredictorContractError("video model cell requires a feature reader")
        if split == "train" and not isinstance(feature_reader, sources.q0_grid_class):
            raise PredictorContractError("train predictions remain bound to accepted Q0")
        if split == "dev" and isinstance(feature_reader, CurrentProfileGrid):
            if cell.feature_cache_manifest_path is None or Path(cell.feature_cache_manifest_path).resolve() != CURRENT_PROFILE_DEV_MANIFEST.resolve():
                raise PredictorContractError("dev cell does not bind the canonical CURRENT_PROFILE_V2 manifest")
        elif split in {"train", "dev"} and cell.expected_feature_cache_manifest_sha256 != ACCEPTED_Q0_CACHE_SHA256:
            raise PredictorContractError("train/dev cell cache hash differs from accepted Q0 source")
        feature_reader = bind_feature_reader(
            feature_reader,
            manifest_path=cell.feature_cache_manifest_path,
            expected_manifest_sha256=cell.expected_feature_cache_manifest_sha256,
            split=split,
            source=sources,
            final_access_opened_path=final_access_opened_path,
            expected_final_access_opened_sha256=expected_final_access_opened_sha256,
        )
    elif feature_reader is not None or cell.feature_cache_manifest_path is not None or cell.expected_feature_cache_manifest_sha256 is not None:
        raise PredictorContractError("state-only model cell must not bind/read video features")

    model = sources.model_class(seed=int(cell.seed), video_mode=cell.video_mode, waypoint_mode=None)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise PredictorContractError("safetensors is required for safe accepted model loading") from exc
    state_dict = load_file(str(cell.model_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    if sources.state_hash(model) != expected_state_hash:
        raise PredictorContractError("loaded checkpoint state does not match accepted state hash")
    model = model.to(torch.device(device)).eval()
    predictor = CoreBPredictor(
        model,
        program_id=cell.program_id,
        video_mode=cell.video_mode,
        native_scale=scale,
        expected_state_sha256=expected_state_hash,
        state_hash_fn=sources.state_hash,
    )
    prediction = predict_rows(
        data,
        canonical_ids=canonical_ids,
        split=split,
        predictor=predictor,
        normalizer=normalizer,
        model_input_fn=sources.model_input,
        device=device,
        feature_reader=feature_reader,
        batch_size=batch_size,
        final_access_authorization=final_authorization,
    )
    bundle = build_prediction_bundle(canonical_ids, prediction, program_id=cell.program_id, seed=int(cell.seed))
    artifact = write_prediction_bundle(output_path, bundle) if output_path is not None else None
    return bundle, artifact


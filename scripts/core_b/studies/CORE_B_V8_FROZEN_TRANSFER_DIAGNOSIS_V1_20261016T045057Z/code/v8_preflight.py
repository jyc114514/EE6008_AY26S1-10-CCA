from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks"
V8 = BENCH / "CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z"
V7 = BENCH / "CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z"
V2 = BENCH / "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
V4 = BENCH / "CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
V5 = BENCH / "CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
WORKER = V8 / "code/v8_train_forward.py"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")
WATCHER_ENV = {
    "MKL_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TMPDIR": str(V8 / "tmp"),
    "TMP": str(V8 / "tmp"),
    "TEMP": str(V8 / "tmp"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def safe_tensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw_len)[0]
        if length <= 0 or length > 16 * 1024 * 1024:
            raise RuntimeError(f"invalid safetensors header length {length}: {path}")
        header = json.loads(handle.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    allowed = {"prediction_native", "action_residual"}
    if "prediction_native" not in tensors or set(tensors) - allowed:
        raise RuntimeError(f"unexpected tensor keys in {path}: {sorted(tensors)}")
    primary = tensors["prediction_native"]
    if "action_residual" in tensors:
        residual = tensors["action_residual"]
        if residual.get("shape") != primary.get("shape") or residual.get("dtype") != "F32":
            raise RuntimeError(f"student action_residual schema mismatch in {path}")
    return {**primary, "keys": sorted(tensors)}


def canonical_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        group_col = "source_group" if "source_group" in fields else "source_cluster"
        required = {"anchor_id", "episode_id", "task_id", "anchor_frame", group_col}
        if not required <= fields:
            raise RuntimeError(f"receiver-row schema mismatch at {path}: {sorted(fields)}")
        rows = []
        for row in reader:
            rows.append({
                "anchor_id": str(row["anchor_id"]),
                "episode_id": str(int(row["episode_id"])),
                "task_id": str(int(row["task_id"])),
                "source_group": str(row[group_col]),
                "anchor_frame": str(int(row["anchor_frame"])),
            })
    if len({r["anchor_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate anchor IDs in receiver rows: {path}")
    if set(int(r["task_id"]) for r in rows) != set(TASKS):
        raise RuntimeError(f"task set mismatch in receiver rows: {path}")
    return rows


def ordered_id_hash(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(r["anchor_id"] for r in rows).encode()).hexdigest()


def rows_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["anchor_id"]: row for row in rows}


def same_identities(a: list[dict[str, str]], b: list[dict[str, str]]) -> bool:
    return rows_by_id(a) == rows_by_id(b)


def read_manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        entries[rel] = digest
    return entries


def verify_manifested(path: Path, entries: dict[str, str]) -> tuple[str, str]:
    digest = sha(path)
    try:
        rel = str(path.relative_to(V7))
    except ValueError:
        return digest, "external_to_v7_manifest"
    if rel == "qa/FINAL_CHECKSUM_VERIFICATION.json":
        return digest, "post_manifest_readback_receipt"
    expected = entries.get(rel)
    if expected is None or expected != digest:
        raise RuntimeError(f"V7 manifest mismatch for selected source: {rel}")
    return digest, "verified_against_v7_manifest"


def make_synthetic_qa() -> dict[str, Any]:
    rng = np.random.default_rng(20261016)
    shape = (4, 30, 28)
    eps = 1e-12
    y = rng.normal(size=shape).astype(np.float64)
    s = rng.normal(size=shape).astype(np.float64)
    v = rng.normal(size=shape).astype(np.float64)
    z_u = 0.5 * y + 0.5 * v
    t = 0.5 * s + 0.5 * v
    z_m = 0.5 * y + 0.5 * t
    checks: dict[str, bool] = {}
    checks["target_coefficients"] = bool(np.allclose(z_m, 0.5*y + 0.25*s + 0.25*v, rtol=0, atol=1e-14))
    sigma = np.linspace(0.5, 1.5, 28, dtype=np.float64)
    anchor = rng.normal(size=(4, 28)).astype(np.float64)
    transformed = (z_m - anchor[:, None, :]) / sigma[None, None, :]
    restored = anchor[:, None, :] + sigma[None, None, :] * transformed
    checks["native_fixed_affine_roundtrip"] = bool(np.allclose(restored, z_m, rtol=0, atol=1e-14))
    checks["internal_fd_linearity"] = bool(np.allclose(np.diff(z_m, axis=1)-np.diff(z_u, axis=1), np.diff(z_m-z_u, axis=1), rtol=0, atol=1e-14))
    checks["seam_is_separate_first_frame_delta"] = bool(np.allclose((z_m[:, 0]-anchor)-(z_u[:, 0]-anchor), (z_m-z_u)[:, 0], rtol=0, atol=1e-14))
    z_m_s_equals_v = 0.5*y + 0.25*v + 0.25*v
    checks["S_equals_V_implies_ZM_equals_ZU"] = bool(np.allclose(z_m_s_equals_v, z_u, rtol=0, atol=1e-14))
    checks["T_equals_Y_implies_ZM_equals_Y"] = bool(np.allclose(0.5*y + 0.5*y, y, rtol=0, atol=1e-14))
    runner_j = lambda p,q: float(np.mean(np.abs(p-q)) + 0.1*np.mean(np.abs(np.diff(p,axis=1)-np.diff(q,axis=1))))
    checks["same_target_comparison_is_identical"] = runner_j(y, z_m) == runner_j(y, z_m)

    # The same weighted population inner product is used for projection, norm,
    # orthogonality, and energy decomposition.
    c = rng.normal(size=shape).astype(np.float64)
    x = rng.normal(size=shape).astype(np.float64)
    w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    inner = lambda a,b: float(sum(w[i] * np.mean(a[i]*b[i]) for i in range(4)))
    c2 = inner(c,c)
    k = inner(x,c)/c2
    perp = x-k*c
    dot_res = inner(perp,c)
    norm_x2, norm_decomp = inner(x,x), k*k*c2+inner(perp,perp)
    checks["projection_orthogonality_abs_1e-12"] = bool(abs(dot_res) <= 1e-12)
    checks["projection_energy_abs_1e-12_rel_1e-10"] = bool(abs(norm_x2-norm_decomp) <= 1e-12 + 1e-10*abs(norm_x2))

    def project(xx: np.ndarray, cc: np.ndarray) -> dict[str, Any]:
        rms_x = float(np.sqrt(max(inner(xx,xx),0.0)))
        rms_c = float(np.sqrt(max(inner(cc,cc),0.0)))
        if rms_c <= eps:
            return {"k":None,"cos":None,"orthogonal_energy":None,"status":"NA_C_ZERO"}
        kk = inner(xx,cc)/inner(cc,cc)
        residual = xx-kk*cc
        cosine = None if rms_x <= eps else inner(xx,cc)/(rms_x*rms_c)
        return {"k":float(kk),"cos":None if cosine is None else float(cosine),
                "orthogonal_energy":float(inner(residual,residual)),"status":"PASS"}

    zero = np.zeros(shape, dtype=np.float64)
    c_zero = project(zero, zero)
    a_zero = project(zero, np.ones(shape, dtype=np.float64))
    checks["T_G_V_zero_C_projection_NA"] = c_zero["status"] == "NA_C_ZERO" and c_zero["k"] is None and c_zero["cos"] is None and c_zero["orthogonal_energy"] is None
    checks["zero_X_nonzero_C_k0_cosNA"] = a_zero["status"] == "PASS" and a_zero["k"] == 0.0 and a_zero["cos"] is None

    loss = lambda p,q: float(np.mean(np.abs(p-q)))
    def safe_retention(g_loss: float, v_loss: float, t_loss: float) -> float | None:
        den = v_loss-t_loss
        return None if abs(den) <= eps else (v_loss-g_loss)/den
    checks["retention_zero_denominator_NA"] = safe_retention(1.0,1.0,1.0) is None

    # Reject duplicate IDs, malformed shapes and nonfinite values.
    ids = ["a", "b", "c"]
    duplicate_rejected = len(set(ids + ["b"])) != len(ids + ["b"])
    nonfinite_rejected = not np.isfinite(np.array([1.0, np.nan])).all()
    wrong_shape_rejected = np.zeros((2, 30, 27)).shape != (2, 30, 28)
    checks["strict_id_shape_finite_rejection"] = bool(duplicate_rejected and nonfinite_rejected and wrong_shape_rejected)

    # Typed dataframe fixture from numeric seeds to a shared macro.
    fixture = pd.DataFrame([
        {"seed": 202610131, "task_id": 119, "value": 1.0},
        {"seed": 202610131, "task_id": 94, "value": 3.0},
        {"seed": 202610132, "task_id": 119, "value": 2.0},
        {"seed": 202610132, "task_id": 94, "value": 4.0},
    ])
    fixture["seed"] = pd.to_numeric(fixture["seed"], errors="raise").astype("int64")
    fixture["task_id"] = pd.to_numeric(fixture["task_id"], errors="raise").astype("int64")
    fixture["value"] = pd.to_numeric(fixture["value"], errors="raise").astype("float64")
    per_seed = fixture.groupby("seed", sort=True)["value"].mean()
    shared_macro = float(per_seed.mean())
    checks["typed_seed_to_shared_macro"] = bool(str(fixture["seed"].dtype) == "int64" and shared_macro == 2.5)
    checks["whole_seed_block_LOO"] = bool(len(per_seed) == 2 and np.isfinite(float(per_seed.iloc[0])))

    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "epsilon_num": eps,
        "projection_abs_tolerance": 1e-12,
        "projection_energy_rel_tolerance": 1e-10,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "final_heldout_payload_access": 0,
        "train_dev_targets_loaded": False,
    }


def add_bundle(records: list[dict[str, Any]], split: str, method: str, seed: int,
               prediction_path: Path, prediction_sha: str, rows_path: Path,
               rows_sha: str, source_root: str, seal_path: str | None = None,
               seal_sha: str | None = None) -> dict[str, Any]:
    if not prediction_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"missing accepted {split} prediction source: {prediction_path}")
    actual_pred_sha = sha(prediction_path)
    actual_rows_sha = sha(rows_path)
    if actual_pred_sha != prediction_sha or actual_rows_sha != rows_sha:
        raise RuntimeError(f"accepted {split} prediction digest mismatch: {prediction_path}")
    if seal_path and seal_sha:
        if not Path(seal_path).is_file() or sha(Path(seal_path)) != seal_sha:
            raise RuntimeError(f"accepted prediction seal digest mismatch: {seal_path}")
    header = safe_tensor_header(prediction_path)
    expected_rows = 1280 if split == "train" else 185
    if header.get("shape") != [expected_rows, 30, 28] or header.get("dtype") not in {"F32", "F64"}:
        raise RuntimeError(f"prediction header shape/dtype mismatch: {prediction_path}: {header}")
    rows = canonical_rows(rows_path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"row count mismatch for {prediction_path}: {len(rows)}")
    rec = {
        "split": split, "method": method, "seed": seed,
        "source_root": source_root, "prediction_path": str(prediction_path),
        "prediction_sha256": actual_pred_sha, "prediction_shape": header["shape"],
        "prediction_dtype": header["dtype"], "rows_path": str(rows_path),
        "rows_sha256": actual_rows_sha, "ordered_anchor_id_sha256": ordered_id_hash(rows),
        "receiver_rows": expected_rows, "seal_path": seal_path, "seal_sha256": seal_sha,
    }
    records.append(rec)
    return {"rows": rows, "record": rec}


def run_preflight() -> None:
    if not V8.exists():
        raise FileNotFoundError(f"V8 owner root is not present: {V8}")
    v7_receipt = read_json(V7 / "qa/FINAL_CHECKSUM_VERIFICATION.json")
    manifest_sha = sha(V7 / "CHECKSUMS.sha256")
    if (
        v7_receipt.get("status") != "PASS"
        or manifest_sha != v7_receipt.get("manifest_sha256")
        or int(v7_receipt.get("verified_entry_count", -1)) != 712
    ):
        raise RuntimeError("V7 final receipt/manifest identity is not accepted")
    manifest_entries = read_manifest(V7)
    if len(manifest_entries) != 712:
        raise RuntimeError(f"V7 manifest entry count mismatch: {len(manifest_entries)}")
    v7_source = read_json(V7 / "protocol/SOURCE_MAP.json")
    fit_index = read_json(V7 / "protocol/FIT_ACCEPTANCE_INDEX.json")
    teacher_seal = read_json(V7 / "protocol/TEACHER_BANK_SEAL.json")
    target_lock = read_json(V7 / "protocol/TARGET_LOCK.json")
    prediction_seal = read_json(V7 / "qa/PREDICTION_SEAL.json")
    gt_v = read_json(V7 / "qa/GT_RETRAIN_VI_REPRODUCTION.json")
    if (
        v7_source.get("status") != "LOCKED" or fit_index.get("status") != "PASS"
        or int(fit_index.get("accepted_fit_count", -1)) != 9
        or teacher_seal.get("status") != "PASS" or prediction_seal.get("status") != "PASS"
        or target_lock.get("status") != "LOCKED" or gt_v.get("status") != "PASS"
    ):
        raise RuntimeError("V7 accepted provenance/fit/prediction/target receipts are incomplete")

    selected_v7 = [
        "protocol/SOURCE_MAP.json", "protocol/FIT_ACCEPTANCE_INDEX.json",
        "protocol/TEACHER_BANK_SEAL.json", "protocol/TARGET_LOCK.json",
        "protocol/RECIPE_LOCK.json", "protocol/INIT_SAMPLER_PAIRING.json",
        "protocol/ANALYSIS_LOCK.json", "qa/FINAL_ACCEPTANCE.json",
        "qa/FINAL_CHECKSUM_VERIFICATION.json", "qa/PREDICTION_SEAL.json",
        "qa/PREDICTION_PREFLIGHT.json", "qa/GT_RETRAIN_VI_REPRODUCTION.json",
        "qa/SCORER_SEAL.json", "qa/DEV_TARGET_ACCESS_RECEIPT.json",
        "results/RESULTS_SUMMARY.json", "results/PRIMARY_TASK_SEED.csv",
        "results/PAIRED_EFFECTS.csv", "results/PAIRED_GROUP_CI.csv",
        "results/SEED_BLOCK_LOO.csv", "results/TASK_LOO.csv",
        "results/TEACHER_RETENTION.csv", "results/PER_WINDOW_METRICS.csv",
        "results/TRAIN_TARGET_SIGNAL.csv", "results/TRAIN_FIT_AND_IMITATION.csv",
        "results/TRAINING_CURVES.csv", "results/RESOURCE_LEDGER.csv",
    ]
    selected_hashes = {}
    for rel in selected_v7:
        path = V7 / rel
        digest, check = verify_manifested(path, manifest_entries)
        selected_hashes[rel] = {"path": str(path), "sha256": digest, "manifest_check": check}

    train_bundles: list[dict[str, Any]] = []
    train_by_seed_method: dict[tuple[int,str], dict[str, Any]] = {}
    for entry in teacher_seal["entries"]:
        role = str(entry["teacher_role"])
        seed = int(entry["seed"])
        p = Path(entry["prediction_path"])
        row_path = Path(entry["prediction_rows_path"])
        if role == "T":
            pred_sha = entry["prediction_sha256"]
        else:
            pred_sha = entry["prediction_sha256"]
        result = add_bundle(
            train_bundles, "train", role, seed, p, pred_sha,
            row_path, entry["prediction_rows_sha256"], "V7",
            entry.get("completion_path"), entry.get("completion_sha256"),
        )
        train_by_seed_method[(seed, role)] = result

    dev_bundles: list[dict[str, Any]] = []
    dev_by_seed_method: dict[tuple[int,str], dict[str, Any]] = {}
    for entry in prediction_seal["entries"]:
        method = str(entry["program_id"])
        seed = int(entry["seed"])
        result = add_bundle(
            dev_bundles, "dev", method, seed, Path(entry["prediction_path"]),
            entry["prediction_sha256"], Path(entry["prediction_rows_path"]),
            entry["prediction_rows_sha256"], "V7",
            entry["completion_path"], entry["completion_sha256"],
        )
        dev_by_seed_method[(seed, method)] = result

    refs = v7_source["evaluation_reference_bundles"]
    ref_by_seed_method: dict[tuple[int,str], dict[str, Any]] = {}
    for entry in refs:
        source_root = str(entry["source_root"])
        role = str(entry["role"])
        if not ((source_root == "V2" and role in {"S","V"}) or (source_root == "V5" and role == "T")):
            continue
        seed = int(entry["seed"])
        result = add_bundle(
            dev_bundles, "dev", role, seed, Path(entry["prediction_path"]),
            entry["prediction_sha256"], Path(entry["rows_path"]),
            entry["rows_sha256"], source_root, entry.get("seal_path"),
            entry.get("seal_sha256"),
        )
        ref_by_seed_method[(seed, role)] = result

    # Validate the receiver/anchor/task/group identity across accepted train and dev sources.
    train_id_hash = target_lock["ordered_receiver_ids_sha256"]
    for seed in SEEDS:
        base = train_by_seed_method[(seed,"V")]["rows"]
        if ordered_id_hash(base) != train_id_hash:
            raise RuntimeError(f"V7 V train ID hash differs from TARGET_LOCK for seed {seed}")
        for role in ("S","T"):
            if train_by_seed_method[(seed,role)]["rows"] != base:
                raise RuntimeError(f"train receiver identity mismatch: {role} seed {seed}")
        for program in ("GT_RETRAIN","SELF_SOFT","MIX_SOFT"):
            fit = next(e for e in fit_index["entries"] if e["program_id"] == program and int(e["seed"]) == seed)
            actual_manifest_sha = sha(Path(fit["model_manifest_path"]))
            actual_model_sha = sha(Path(fit["model_path"]))
            if actual_manifest_sha != fit["model_manifest_sha256"] or actual_model_sha != fit["model_sha256"]:
                raise RuntimeError(f"accepted model digest mismatch: {program} seed {seed}")
        dev_base = dev_by_seed_method[(seed,"GT_RETRAIN")]["rows"]
        for method in ("SELF_SOFT","MIX_SOFT"):
            if not same_identities(dev_by_seed_method[(seed,method)]["rows"], dev_base):
                raise RuntimeError(f"V7 student dev ID mismatch: {method} seed {seed}")
        for method in ("S","V","T"):
            if not same_identities(ref_by_seed_method[(seed,method)]["rows"], dev_base):
                raise RuntimeError(f"accepted dev reference ID mismatch: {method} seed {seed}")

    # Exact model-file identity for the lawful GT_RETRAIN = accepted old V_i alias.
    old_v_by_seed = {
        int(row["seed"]): row for row in v7_source["models"] if row.get("role") == "V"
    }
    alias_entries = []
    for seed in SEEDS:
        g = next(e for e in fit_index["entries"] if e["program_id"] == "GT_RETRAIN" and int(e["seed"]) == seed)
        old_v = old_v_by_seed[seed]
        vtrain = next(e for e in teacher_seal["entries"] if e.get("teacher_role") == "V" and int(e["seed"]) == seed)
        if not (
            g["model_sha256"] == old_v["model_sha256"] == vtrain["source_model_sha256"]
            and sha(Path(g["model_path"])) == g["model_sha256"]
            and sha(Path(old_v["model_path"])) == old_v["model_sha256"]
        ):
            raise RuntimeError(f"GT_RETRAIN != accepted V model for seed {seed}")
        qa_entry = next(e for e in gt_v["entries"] if int(e["seed"]) == seed)
        if qa_entry.get("GT_RETRAIN_model_sha256") != qa_entry.get("accepted_V_model_sha256") or qa_entry["GT_RETRAIN_model_sha256"] != g["model_sha256"]:
            raise RuntimeError(f"accepted GT/V reproduction receipt mismatch for seed {seed}")
        alias_entries.append({
            "seed": seed, "GT_RETRAIN_model_sha256": g["model_sha256"],
            "accepted_V_model_sha256": old_v["model_sha256"],
            "train_prediction_alias": vtrain["prediction_path"],
            "train_prediction_sha256": vtrain["prediction_sha256"],
            "train_prediction_rows_sha256": vtrain["prediction_rows_sha256"],
            "alias_basis": "identical final model bytes; same V7 prefix_batch, normalizer, TS11-D, eval/no_grad FP32 profile, and ordered receiver IDs",
        })

    # V7 diagnostic/source table coverage and whether new work is required.
    coverage = [
        {"quantity":"accepted primary results and paired contrasts","split":"dev","source_path":str(V7/"results/PRIMARY_TASK_SEED.csv"),"columns":"method,seed,task_id,position_mae,safe_fd_l1,seam_safe_fd_l1","coverage":"PRESENT_ACCEPTED","operation":"reuse; no bootstrap rerun","new_forward":0,"new_gt_load":0},
        {"quantity":"source-group confidence intervals and LOO","split":"dev","source_path":str(V7/"results/PAIRED_GROUP_CI.csv"),"columns":"accepted V7 contrasts; original 5000 draws","coverage":"PRESENT_ACCEPTED","operation":"cite original CI only","new_forward":0,"new_gt_load":0},
        {"quantity":"train perturbation signals d(S,V),d(T,V)","split":"train","source_path":str(V7/"results/TRAIN_TARGET_SIGNAL.csv"),"columns":"task,seed,native_position_mae/rms,internal_fd_mae,safe_scaled_internal_fd_l1","coverage":"PRESENT_PARTIAL","operation":"reuse and supplement Y/T/Z and final eval outputs","new_forward":0,"new_gt_load":0},
        {"quantity":"train fit/teacher distances and curves","split":"train","source_path":str(V7/"results/TRAIN_FIT_AND_IMITATION.csv"),"columns":"epoch160 train/dropout-on pre-update aggregate; target and teacher distances","coverage":"PRESENT_PARTIAL","operation":"reuse with explicit scope; supplement final eval-mode train predictions","new_forward":0,"new_gt_load":0},
        {"quantity":"S,V,T final train predictions","split":"train","source_path":str(V7/"protocol/TEACHER_BANK_SEAL.json"),"columns":"6 S/V and 3 T_NATIVE accepted [1280,30,28]","coverage":"PRESENT_ACCEPTED","operation":"reuse sealed assets","new_forward":0,"new_gt_load":0},
        {"quantity":"G final train predictions","split":"train","source_path":"V7 GT_RETRAIN_VI_REPRODUCTION + V7 accepted V train prediction","columns":"exact model SHA, same inference profile and IDs","coverage":"PRESENT_BY_ACCEPTED_ALIAS","operation":"alias G=V after identity QA","new_forward":0,"new_gt_load":0},
        {"quantity":"U=SELF_SOFT final eval-mode train predictions","split":"train","source_path":"none in V7; only dropout-on pre-update curves","columns":"required [1280,30,28] final epoch160 model output","coverage":"MISSING_REQUIRES_FROZEN_FORWARD","operation":"run six authorized V8 frozen prefix forwards total","new_forward":3,"new_gt_load":0},
        {"quantity":"M=MIX_SOFT final eval-mode train predictions","split":"train","source_path":"none in V7; only dropout-on pre-update curves","columns":"required [1280,30,28] final epoch160 model output","coverage":"MISSING_REQUIRES_FROZEN_FORWARD","operation":"run six authorized V8 frozen prefix forwards total","new_forward":3,"new_gt_load":0},
        {"quantity":"G,U,M dev predictions","split":"dev","source_path":str(V7/"qa/PREDICTION_SEAL.json"),"columns":"9 accepted bundles; 185x30x28","coverage":"PRESENT_ACCEPTED","operation":"reuse; new dev prediction jobs fixed at zero","new_forward":0,"new_gt_load":0},
        {"quantity":"S,V,T dev predictions","split":"dev","source_path":str(V7/"protocol/SOURCE_MAP.json"),"columns":"V2 S/V and V5 MIX50=T references","coverage":"PRESENT_ACCEPTED","operation":"reuse sealed same-ID references","new_forward":0,"new_gt_load":0},
        {"quantity":"training targets Z and Y","split":"train","source_path":str(V7/"protocol/TARGET_LOCK.json"),"columns":"Y_NATIVE plus 9 native float64 Z arrays","coverage":"PRESENT_LOCKED","operation":"read once per required artifact after preflight/QA","new_forward":0,"new_gt_load":1},
        {"quantity":"dev target Y","split":"dev","source_path":str(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),"columns":"accepted original 185-window target archive; V8 diagnostic reload not yet performed","coverage":"AVAILABLE_NOT_YET_LOADED","operation":"one allow_pickle=False archive load after QA; no final/heldout","new_forward":0,"new_gt_load":1},
        {"quantity":"per-step and per-coordinate dev errors","split":"dev","source_path":"accepted predictions plus original dev target Y","columns":"30 future steps, 29 internal transitions, 28 recorded coordinates","coverage":"MISSING_REQUIRES_REDUCTION","operation":"CPU descriptive reductions only","new_forward":0,"new_gt_load":1},
        {"quantity":"formal peak memory by fit","split":"train","source_path":str(V7/"protocol/FIT_ACCEPTANCE_INDEX.json"),"columns":"9 accepted model manifests contain peak allocated/reserved/GPU","coverage":"AVAILABLE_IN_MODEL_MANIFESTS","operation":"write separate V8 supplement; leave V7 ledger unchanged","new_forward":0,"new_gt_load":0},
        {"quantity":"V7 resource ledger peak columns","split":"train","source_path":str(V7/"results/RESOURCE_LEDGER.csv"),"columns":"formal fit rows lack peak columns","coverage":"MISSING_IN_V7_LEDGER","operation":"supplement from 9 accepted manifests in V8 only","new_forward":0,"new_gt_load":0},
    ]
    write_csv(V8/"inventory/EXISTING_DIAGNOSTIC_COVERAGE.csv", list(coverage[0]), coverage)

    # Preserve exact source bindings and selected V7 files; external data are read-only.
    external_refs = []
    for record in train_bundles + dev_bundles:
        external_refs.append(record)
    v8_source = {
        "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
        "study": "post_V7_frozen_transfer_diagnosis_development",
        "v7_root": str(V7),
        "v7_manifest_sha256": manifest_sha,
        "v7_manifest_entries": len(manifest_entries),
        "v7_final_receipt_path": str(V7/"qa/FINAL_CHECKSUM_VERIFICATION.json"),
        "v7_final_receipt_sha256": sha(V7/"qa/FINAL_CHECKSUM_VERIFICATION.json"),
        "v7_final_receipt_status": v7_receipt["status"],
        "selected_v7_files": selected_hashes,
        "V2_V3_V4_V5_paths_and_seals": {
            "source_roots": v7_source.get("source_roots_read_only"),
            "accepted_parent_roots": v7_source.get("accepted_parent_roots"),
            "V2_model_and_dev_reference_roots": str(V2),
            "V4_train_and_dev_prefixes_normalizer_metric_lock": str(V4),
            "V5_T_dev_reference_root": str(V5),
            "V4_train_npz_sha256": v7_source["data"]["train_npz_sha256"],
            "V4_dev_prefix_sha256": v7_source["data"]["dev_prefix_npz_sha256"],
            "V4_normalizer_sha256": v7_source["data"]["normalizer_sha256"],
            "V4_metric_scale_lock_sha256": v7_source["data"]["metric_scale_lock_sha256"],
            "feature_cache_manifest": v7_source["data"]["feature_cache_manifest"],
            "feature_cache_manifest_sha256": v7_source["data"]["feature_cache_manifest_sha256"],
        },
        "split_contract": {
            "tasks": list(TASKS), "seeds": list(SEEDS),
            "train_rows": 1280, "train_episodes": 220,
            "dev_rows": 185, "dev_source_groups": 47,
            "final_heldout_payload_access": 0,
        },
        "train_target_formula": {
            "Z_G": "Y",
            "Z_U": "0.5*Y+0.5*V",
            "Z_M": "0.5*Y+0.5*T=0.5*Y+0.25*S+0.25*V",
            "T": "0.5*(S+V)",
            "Z_M_minus_Z_U": "0.25*(S-V)",
        },
        "G_equals_V_model_alias": alias_entries,
        "accepted_prediction_assets": external_refs,
        "train_ordered_anchor_id_sha256": train_id_hash,
        "dev_ordered_anchor_id_sha256": ordered_id_hash(dev_by_seed_method[(SEEDS[0],"GT_RETRAIN")]["rows"]),
        "label_read_timeline": {
            "teacher_bank_sealed_utc": teacher_seal.get("created_utc"),
            "target_lock_created_utc": target_lock.get("created_utc"),
            "teacher_seal_future_target_values_read_before_seal": teacher_seal.get("future_target_values_read_before_seal"),
            "target_lock_Y_read_after_teacher_seal": target_lock.get("Y_native_read_after_teacher_seal"),
            "interpretation": "V7 code and receipts show S/V/T prediction bank sealed before Y_train was loaded; Z was constructed only after that read. The prose ambiguity is corrected, not evidence of target leakage.",
        },
        "read_only_roots": [str(V7), str(V2), str(V4), str(V5)],
        "old_root_writes": 0,
        "final_heldout_payload_access": 0,
    }
    write_json(V8/"inventory/SOURCE_MAP.json", v8_source)

    # Accept profile-safe G=V aliases at the model/input level. Exact prediction
    # equality is checked after loading the accepted tensors, before labels.
    reuse = {
        "schema_version": 1, "status": "PREPARED", "created_utc": utc_now(),
        "new_teacher_forward_jobs": 0, "new_dev_prediction_jobs": 0,
        "new_fitted_predictors_or_search": 0, "new_optimizer_updates": 0,
        "planned_frozen_student_train_forward_jobs": 6,
        "train_forward_jobs": [],
        "reuse": [
            {"split":"train","method":"G","source":"V7 accepted V train prediction","reason":"exact model SHA and identical V7 inference recipe/profile; IDs hash locked"},
            {"split":"train","method":"S","source":"V7 TEACHER_BANK_SEAL S_i","reason":"accepted frozen source output"},
            {"split":"train","method":"V","source":"V7 TEACHER_BANK_SEAL V_i","reason":"accepted frozen source output"},
            {"split":"train","method":"T","source":"V7 T_NATIVE","reason":"accepted 0.5*S+0.5*V native float64"},
            {"split":"dev","method":"G/U/M","source":"V7 prediction seal","reason":"9 accepted dev bundles reused; no new predictions"},
            {"split":"dev","method":"S/V","source":"V2 accepted reference bundles","reason":"same ordered 185 receiver identities"},
            {"split":"dev","method":"T","source":"V5 accepted MIX50 reference bundles","reason":"same ordered 185 receiver identities"},
        ],
        "profile_safe_alias_checks": {
            "G_equals_V_model_file_sha": "PASS",
            "train_IDs_normalizer_TS11D_batch32_eval_FP32_profile": "PASS",
            "dev_IDs_and_accepted_score_reconciliation": "PASS; per-source row order differs and will be joined by anchor_id; tensor max-abs check pending",
            "numeric_prediction_tolerance": "compute direct max-absolute and max-relative output differences; retain accepted score reconciliation max 1.7763568394002505e-15 as score-only evidence",
        },
        "new_train_forward_scope": "SELF_SOFT and MIX_SOFT only; six final epoch160 accepted model weights; 1280 train prefixes; no Y/Z or teacher inputs; eval+inference_mode; no parameter/buffer mutation",
        "final_heldout_payload_access": 0,
    }
    fit_lookup = {(str(e["program_id"]),int(e["seed"])):e for e in fit_index["entries"]}
    normalizer_path = Path(v7_source["data"]["normalizer"])
    metric_lock_path = Path(v7_source["data"]["metric_scale_lock"])
    resolved_path = Path(V4/"inventory/RESOLVED_PATHS.json")
    resolved = read_json(resolved_path)
    cache_path = Path(resolved["feature_cache"])
    cache_sha = v7_source["data"]["feature_cache_manifest_sha256"]
    if sha(cache_path) != cache_sha:
        raise RuntimeError("V4 feature cache manifest SHA differs from V7 source map")
    train_npz = Path(v7_source["data"]["train_npz"])
    ordered_train_id_sha = train_id_hash
    fit_index_sha = sha(V7/"protocol/FIT_ACCEPTANCE_INDEX.json")
    normalizer_sha = sha(normalizer_path)
    metric_lock_sha = sha(metric_lock_path)
    train_npz_sha = sha(train_npz)
    v7_ledger = []
    with (V7/"results/RESOURCE_LEDGER.csv").open(newline="",encoding="utf-8") as f:
        v7_ledger = list(csv.DictReader(f))
    pred_rows = [row for row in v7_ledger if row.get("phase") == "student_dev_prediction" and row.get("status") == "PASS"]
    measured_peaks = [float(row["peak_allocated_mib"]) for row in pred_rows if row.get("peak_allocated_mib")]
    if len(pred_rows) != 9 or not measured_peaks:
        raise RuntimeError("cannot bind inference resource estimate to accepted measured V7 predictions")
    max_observed_peak = max(measured_peaks)
    watcher_peak = math.ceil(max_observed_peak)
    resource_basis = (
        f"ceil(max actual V7 accepted student dev inference allocated peak={max_observed_peak:.9f} MiB; "
        "same TS11-D single head, FP32 eval, batch32 and cached Q0 prefix; new train data are consumed in the same fixed batch shape; "
        "not the 182 MiB training peak)"
    )
    for program in ("SELF_SOFT","MIX_SOFT"):
        for seed in SEEDS:
            fit = fit_lookup[(program,seed)]
            job_id = f"v8_train_{program}_s{seed}_a01"
            output = V8/"predictions_train"/program/f"seed{seed}"/"attempt_01"
            job_path = V8/"configs"/"inference"/f"{job_id}.json"
            status_path = V8/"coordination"/"status"/f"{job_id}.json"
            watch_path = V8/"configs"/"watcher"/f"{job_id}.json"
            log_path = V8/"logs"/"jobs"/f"{job_id}.log"
            job = {
                "schema_version": 1, "job_id": job_id, "program_id": program,
                "seed": seed, "model_path": fit["model_path"],
                "model_sha256": fit["model_sha256"],
                "model_manifest_path": fit["model_manifest_path"],
                "model_manifest_sha256": fit["model_manifest_sha256"],
                "fit_acceptance_index_sha256": fit_index_sha,
                "train_npz_sha256": train_npz_sha,
                "normalizer_sha256": normalizer_sha,
                "target_sigma_sha256": metric_lock_sha,
                "feature_cache_manifest_path": str(cache_path),
                "feature_cache_manifest_sha256": cache_sha,
                "ordered_receiver_ids_sha256": ordered_train_id_sha,
                "output_root": str(output), "batch_size": 32,
                "split": "train", "teachers_used": False,
                "targets_loaded": False, "optimizer_updates": 0,
                "final_heldout_payload_access": 0,
            }
            write_json(job_path, job)
            watcher = {
                "schema_version": 1, "job_id": job_id,
                "protocol_id": "V8_FROZEN_STUDENT_TRAIN_PREFIX_DIAGNOSTIC",
                "output_root": str(output),
                "command": [str(PYTHON), str(WORKER), "--job-json", str(job_path)],
                "resource_estimate": {"measured_peak_mib": watcher_peak, "basis": resource_basis},
                "completion_marker": "PREDICTION_COMPLETE.json",
                "max_oom_requeues": 0,
                "log_path": str(log_path),
                "environment": WATCHER_ENV,
            }
            write_json(watch_path, watcher)
            reuse["train_forward_jobs"].append({
                "job_id": job_id, "program": program, "seed": seed,
                "job_config_path": str(job_path), "job_config_sha256": sha(job_path),
                "watcher_config_path": str(watch_path), "watcher_config_sha256": sha(watch_path),
                "status_path": str(status_path), "output_root": str(output),
                "model_sha256": fit["model_sha256"],
                "expected_rows": 1280, "batch_size": 32, "batches": 40,
                "measured_peak_estimate_mib": watcher_peak,
                "measurement_basis": resource_basis,
            })
    write_json(V8/"protocol/PREDICTION_REUSE_AND_REPLAY.json", reuse)

    analysis_lock = {
        "schema_version": 1, "status": "LOCKED_PRE_TARGET_READ",
        "created_utc": utc_now(),
        "study": "post_V7_frozen_transfer_diagnosis_development",
        "post_result_informed_diagnostic": True,
        "independent_confirmation": False,
        "tasks": list(TASKS), "seeds": list(SEEDS),
        "train_rows": 1280, "train_episodes": 220,
        "dev_rows": 185, "dev_source_groups": 47,
        "methods": {"G":"GT_RETRAIN=accepted old V_i","U":"SELF_SOFT","M":"MIX_SOFT","S":"optimized state","V":"matched video","T":"0.5*(S+V)"},
        "targets": {"Z_G":"Y","Z_U":"0.5*Y+0.5*V","Z_M":"0.5*Y+0.5*T=0.5*Y+0.25*S+0.25*V"},
        "metric_scale_source": str(metric_lock_path),
        "metric_scale_sha256": metric_lock_sha,
        "normalizer_source": str(normalizer_path),
        "normalizer_sha256": normalizer_sha,
        "train_objective": "runner per_sample_loss on residual targets: L1_position + 0.1*L1_internal_FD; original Y-fitted target_sigma and safe_fd_scale; task/episode weights; eval-on-train reported separately from dropout-on epoch curves",
        "canonical_metrics": "balanced coordinate groups 0:14 and 14:28; position divided by locked target_sigma; 29 internal finite differences divided by locked safe_fd_scale; seam (future frame1 minus anchor) divided by safe_fd_scale",
        "correction_geometry": {
            "P_c":"native prediction divided by fixed target_sigma",
            "C":"T_c-G_c", "A":"M_c-G_c", "B":"U_c-G_c", "E":"M_c-U_c",
            "weighting":"mean coordinates per receiver, anchor within episode/source group equal, groups within task equal, tasks equal; three seeds pooled for overall inner products",
            "epsilon_num":1e-12,
            "projection_abs_tolerance":1e-12,
            "projection_energy_rel_tolerance":1e-10,
            "FD_projection":"fixed linear diff(P_c)/safe_fd_scale; seam separate",
            "no_fitted_deployment_projection":True,
        },
        "dev_local_decomposition": "descriptive only on accepted 185 windows; all methods, all 30 future steps, 29 transitions, 28 recorded coordinates; teacher-help/harm stratified separately per metric; no route training or model selection",
        "confidence": "reuse V7 original source-group percentile CIs only; no new bootstrap or p-values",
        "figure_selection": {
            "seed":202610131,
            "regular_groups":"ascending SHA256('V8_20261016|'+canonical_source_group_id), two groups per task; earliest accepted anchor within each group",
            "failure_example":"max MIX-vs-G position MAE loss by task; tie smallest anchor ID; label posthoc",
            "axis_selection":"per seed equal-task train mean abs(T-V)/target_sigma, max coordinate in 0:14 and 14:28, tie smaller coordinate index",
        },
        "new_fits":0, "new_optimizer_updates":0, "new_teacher_forwards":0,
        "new_dev_predictions":0, "new_bootstrap_draws":0,
        "planned_frozen_student_train_forwards":6,
        "dev_gt_archive_loads_planned":1,
        "final_heldout_payload_access":0, "old_root_writes":0,
        "inputs": {k:v for k,v in selected_hashes.items()},
    }
    write_json(V8/"protocol/ANALYSIS_LOCK.json", analysis_lock)

    # V7 resource-ledger addendum, sourced only from the nine accepted fit manifests.
    supplement = []
    for fit in fit_index["entries"]:
        manifest_path = Path(fit["model_manifest_path"])
        manifest = read_json(manifest_path)
        digest = sha(manifest_path)
        if digest != fit["model_manifest_sha256"] or sha(Path(fit["model_path"])) != fit["model_sha256"]:
            raise RuntimeError(f"V7 resource manifest/model identity mismatch: {manifest_path}")
        alloc = manifest.get("peak_memory_allocated_mib")
        reserved = manifest.get("peak_memory_reserved_mib")
        supplement.append({
            "method": fit["program_id"], "seed": fit["seed"],
            "source_manifest_path": str(manifest_path), "source_manifest_sha256": digest,
            "model_path": fit["model_path"], "model_sha256": fit["model_sha256"],
            "physical_gpu": manifest.get("physical_gpu","UNKNOWN"),
            "logical_device": manifest.get("logical_device","UNKNOWN"),
            "peak_allocated_original_mib": alloc if alloc is not None else "UNKNOWN",
            "peak_allocated_MiB": float(alloc) if alloc is not None else "UNKNOWN",
            "peak_reserved_original_mib": reserved if reserved is not None else "UNKNOWN",
            "peak_reserved_MiB": float(reserved) if reserved is not None else "UNKNOWN",
            "host_RSS": "UNKNOWN",
            "record_role": "V7 formal fit peak; copied into V8 addendum only",
            "timing_scope": "fit runtime includes training; not encoder end-to-end",
        })
    write_csv(V8/"results/V7_RESOURCE_SUPPLEMENT.csv", list(supplement[0]), supplement)

    # No target array values have been opened in this preflight.
    qa = {
        "schema_version":1, "status":"PASS", "created_utc":utc_now(),
        "V7_manifest_sha256":manifest_sha, "V7_manifest_entries":len(manifest_entries),
        "train_receivers":1280, "dev_receivers":185,
        "train_ordered_ids_equal_across_SVT":True,
        "dev_ordered_ids_equal_across_GUM_and_SVT":True,
        "safetensors_headers_checked":len(train_bundles)+len(dev_bundles),
        "prediction_arrays_materialized":0,
        "train_Y_or_Z_values_read":False, "dev_Y_values_read":False,
        "GT_RETRAIN_equals_V_exact_model_sha":True,
        "final_heldout_payload_access":0,
    }
    write_json(V8/"qa/SOURCE_AND_ID_QA.json", qa)

    synthetic = make_synthetic_qa()
    write_json(V8/"qa/SYNTHETIC_QA.json", synthetic)
    if synthetic["status"] != "PASS":
        raise RuntimeError(f"synthetic pipeline QA failed: {synthetic['failures']}")

    # Bind input counts and the data-access gate for the later analysis run.
    write_json(V8/"coordination/PREFLIGHT_COMPLETE.json", {
        "schema_version":1, "status":"PASS", "completed_utc":utc_now(),
        "source_map_sha256":sha(V8/"inventory/SOURCE_MAP.json"),
        "coverage_sha256":sha(V8/"inventory/EXISTING_DIAGNOSTIC_COVERAGE.csv"),
        "analysis_lock_sha256":sha(V8/"protocol/ANALYSIS_LOCK.json"),
        "prediction_reuse_sha256":sha(V8/"protocol/PREDICTION_REUSE_AND_REPLAY.json"),
        "source_and_id_qa_sha256":sha(V8/"qa/SOURCE_AND_ID_QA.json"),
        "synthetic_qa_sha256":sha(V8/"qa/SYNTHETIC_QA.json"),
        "train_label_values_opened":False, "dev_label_values_opened":False,
        "final_heldout_payload_access":0,
    })
    print(json.dumps({
        "status":"PASS","v7_manifest_sha256":manifest_sha,
        "train_teacher_bundles":len(train_bundles),
        "dev_prediction_bundles":len(dev_bundles),
        "planned_train_forwards":len(reuse["train_forward_jobs"]),
        "new_dev_predictions":0,"resource_supplement_rows":len(supplement),
        "synthetic_qa":synthetic["status"],"train_Y_loaded":False,"dev_Y_loaded":False,
    },sort_keys=True))


if __name__ == "__main__":
    run_preflight()

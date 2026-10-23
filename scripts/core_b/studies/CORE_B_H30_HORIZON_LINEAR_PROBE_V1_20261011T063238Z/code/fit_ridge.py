from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as _scipy_eigh_preload
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
TARGET_CACHE = EQ / "d3/train_targets_trainonly.npz"
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
THREAD_LIMIT = min(8, os.cpu_count() or 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite nonidentical artifact {path}")
        return
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    content = buf.getvalue()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            if f.read() != content:
                raise FileExistsError(f"refusing to overwrite nonidentical artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(content)


def load_folds() -> dict[str, int]:
    out = {}
    with (ROOT / "protocol/FOLD_ASSIGNMENT.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                out[row["anchor_id"]] = int(row["fold"])
    return out


def load_metadata_and_inputs():
    with np.load(INPUTS, allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U64", copy=True)
        episodes = z["episode_index"].astype(np.int64, copy=True)
        split = z["split_code"].astype(np.int8, copy=True)
        history = z["history_native"].astype(np.float32, copy=True)
        anchors = z["anchor_command_native"].astype(np.float32, copy=True)
    if history.shape != (20079, 32, 71) or anchors.shape != (20079, 28):
        raise AssertionError("input shape changed after protocol lock")
    if not np.isfinite(history).all() or not np.isfinite(anchors).all():
        raise AssertionError("input finite check failed")
    x = history.reshape(len(history), 32 * 71).astype(np.float64)
    return ids, episodes, split, x, anchors.astype(np.float64)


def load_train_targets(expected_train_ids: np.ndarray, expected_train_eps: np.ndarray, expected_train_anchor: np.ndarray):
    # This is the accepted train-only target cache; it contains no development labels.
    with np.load(TARGET_CACHE, allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U64", copy=True)
        episodes = z["episode_index"].astype(np.int64, copy=True)
        anchor = z["anchor_command_native"].astype(np.float32, copy=True)
        targets60 = z["targets_native_60"].astype(np.float32, copy=True)
    if not np.array_equal(ids, expected_train_ids):
        raise AssertionError("train-only target IDs/order differ from locked primary train")
    if not np.array_equal(episodes, expected_train_eps):
        raise AssertionError("train-only target episodes differ from input manifest")
    if targets60.shape != (len(ids), 60, 28) or anchor.shape != (len(ids), 28):
        raise AssertionError("train-only target cache shape mismatch")
    if not np.allclose(anchor, expected_train_anchor, rtol=0, atol=1e-6):
        raise AssertionError("target cache anchor differs from legal input anchor")
    if not np.isfinite(targets60).all():
        raise AssertionError("nonfinite primary train targets")
    return targets60[:, :30].astype(np.float64), anchor.astype(np.float64)


def mean_fold_metrics(pred, target, episodes, sigma, safe):
    position_row = np.abs((pred - target) / sigma[None, None, :]).mean(axis=(1, 2))
    fd_row = (np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) / safe[None, None, :]).mean(axis=(1, 2))
    position = {}
    fd = {}
    for ep in np.unique(episodes):
        m = episodes == ep
        position[int(ep)] = float(position_row[m].mean())
        fd[int(ep)] = float(fd_row[m].mean())
    return float(np.mean(list(position.values()))), float(np.mean(list(fd.values()))), position, fd


def parameterized_model(prepared, fit):
    return {
        "x_mean": prepared["x_mean"], "x_scale": prepared["x_scale"],
        "constant_mask": prepared["constant_mask"], "coef": fit["coef"],
        "intercept": fit["intercept"], "lambda": fit["lambda"],
    }


def write_prediction(path: Path, ids: np.ndarray, pred: np.ndarray, split: str) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction file {path}")
    if pred.shape != (len(ids), 30, 28) or not np.isfinite(pred).all():
        raise AssertionError(f"prediction shape/finite failure {split}: {pred.shape}")
    np.savez_compressed(path, anchor_id=ids.astype("U64"), predictions_native=pred.astype(np.float32))
    return {"path": str(path), "sha256": sha(path), "rows": len(ids), "shape": list(pred.shape), "finite": True, "ids_sha256": hashlib.sha256(ids.astype("U64").tobytes()).hexdigest(), "split": split}


def run_smoke() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ridge_core", ROOT / "code/ridge_core.py")
    core = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(core)
    ids, eps, split, x, anchor = load_metadata_and_inputs()
    train_idx = np.flatnonzero(split == 0)
    expected = train_idx
    target, target_anchor = load_train_targets(ids[train_idx], eps[train_idx], anchor[train_idx])
    fold_candidates = sorted(
        range(len(train_idx)),
        key=lambda i: hashlib.sha256((str(ids[train_idx[i]]) + "|smoke").encode()).hexdigest(),
    )[:512]
    sub = np.asarray(fold_candidates, dtype=np.int64)
    sub_eps = eps[train_idx[sub]]
    weights = core.episode_equal_weights(sub_eps)
    y = target[sub] - target_anchor[sub, None, :]
    cpu_t0 = time.process_time()
    t0 = time.perf_counter()
    prep = core.prepare_weighted_gram(x[train_idx[sub]], y.reshape(len(sub), -1), weights)
    fit = core.solve_from_gram(prep, 0.1)
    elapsed = time.perf_counter() - t0
    cpu_elapsed = time.process_time() - cpu_t0
    finite = bool(np.isfinite(fit["coef"]).all() and np.isfinite(fit["intercept"]).all())
    if not finite or fit["solver_residual_ratio"] > 1e-7:
        raise AssertionError("CPU solver smoke failed finite/residual gate")
    fold_sizes = []
    with (ROOT / "protocol/FOLD_ASSIGNMENT.csv").open(encoding="utf-8", newline="") as f:
        fold_sizes = [r for r in csv.DictReader(f) if r["split"] == "train"]
    sizes = [sum(int(r["fold"]) != fold for r in fold_sizes) for fold in range(3)]
    pools = threadpool_info()
    if any(int(p.get("num_threads", 0)) > THREAD_LIMIT for p in pools if p.get("user_api") == "blas"):
        raise AssertionError(f"BLAS thread cap not enforced: {pools}")
    smoke = {
        "status": "PASS_CPU_TRAIN_ONLY_NUMERICAL_SMOKE_THREAD_CAPPED",
        "threadpool_info": pools,
        "thread_cap": THREAD_LIMIT,
        "records": 1, "target_scope": "512 deterministically hash-selected primary-train anchors only",
        "feature_width": int(x.shape[1]), "outputs": 840, "lambda_fixed_for_smoke_only": 0.1,
        "lambda_selection_performed": False, "development_targets_read": False,
        "cpu_wall_seconds": elapsed, "cpu_seconds_solver": cpu_elapsed,
        "cpu_process_seconds_since_start": float(time.process_time()),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "solver_residual_ratio": fit["solver_residual_ratio"],
        "regularized_condition_number": fit["condition_number_regularized_gram"],
        "actual_cv_fit_rows_by_fold": sizes,
        "actual_cv_fit_budget": "12 fits; 3 full-train folds x 4 fixed lambdas",
        "estimate_note": "This smoke uses the full 2272 feature width and 840 outputs but 512 rows; Gram cost scales approximately with fitting-row count while eigendecomposition and coefficient transforms remain full-width. It is a runtime estimate only, not a lambda comparison or model result.",
        "input_sha256": sha(INPUTS), "train_target_cache_sha256": sha(TARGET_CACHE),
    }
    write_json(ROOT / "qa/CPU_TRAIN_ONLY_SMOKE_VERIFIED.json", smoke)
    print(json.dumps(smoke, sort_keys=True))
    return 0


def run_fit() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ridge_core", ROOT / "code/ridge_core.py")
    core = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(core)
    protocol = json.loads((ROOT / "protocol/PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_PRE_FIT_NO_DEV_TARGETS_READ":
        raise AssertionError("pre-fit protocol is not locked")
    if protocol["lambda_grid"]["extension_allowed"] is not False:
        raise AssertionError("lambda extension is not prohibited")
    model_lock_path = ROOT / "models/RIDGE_MODEL_LOCK.json"
    if model_lock_path.exists():
        raise FileExistsError("model lock already exists; preserve current state for explicit resume")
    expected_source = protocol["input"]["source_sha256"]
    if sha(INPUTS) != expected_source or sha(TARGET_CACHE) != protocol["target"]["train_target_sha256"]:
        raise AssertionError("locked data source hash changed")
    ids, episodes, split, x, anchors = load_metadata_and_inputs()
    train_idx = np.flatnonzero(split == 0)
    dev_idx = np.flatnonzero(split == 1)
    if len(train_idx) != 17757 or len(dev_idx) != 2322:
        raise AssertionError("train/dev row counts differ from locked population")
    train_ids, train_eps, train_groups = ids[train_idx], episodes[train_idx], None
    targets, target_anchor = load_train_targets(train_ids, train_eps, anchors[train_idx])
    residuals = targets - target_anchor[:, None, :]
    y = residuals.reshape(len(train_idx), -1)
    fold_by_id = load_folds()
    fold = np.asarray([fold_by_id[str(i)] for i in train_ids], dtype=np.int8)
    if set(fold.tolist()) != {0, 1, 2}:
        raise AssertionError("fold assignment did not cover all 3 folds")
    if len(fold_by_id) != len(train_ids):
        raise AssertionError("fold assignment row count mismatch")
    dev_group_overlap = False
    with (ROOT / "protocol/FOLD_ASSIGNMENT.csv").open(encoding="utf-8", newline="") as f:
        assign_rows = list(csv.DictReader(f))
    train_group = {r["anchor_id"]: r["source_group_id"] for r in assign_rows if r["split"] == "train"}
    dev_group = {r["source_group_id"] for r in assign_rows if r["split"] == "dev"}
    if set(train_group.values()) & dev_group:
        dev_group_overlap = True
    if dev_group_overlap:
        raise AssertionError("train/dev source group overlap after lock")

    cv_fold_rows = []
    cv_episode_rows = []
    solver_receipts = []
    fold_models: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    process_t0 = time.process_time()
    wall_t0 = time.perf_counter()
    fs_before = shutil.disk_usage(ROOT).used
    with threadpool_limits(limits=THREAD_LIMIT):
        for f in range(3):
            val_local = fold == f
            fit_local = ~val_local
            fit_idx = np.flatnonzero(fit_local)
            val_idx = np.flatnonzero(val_local)
            wfit = core.episode_equal_weights(train_eps[fit_idx])
            sigma_fold, fd_fold, safe_fold = core.episode_equal_target_scales(targets[fit_idx], train_eps[fit_idx])
            tgram = time.perf_counter()
            prepared = core.prepare_weighted_gram(x[train_idx[fit_idx]], y[fit_idx], wfit)
            gram_seconds = time.perf_counter() - tgram
            for lam in LAMBDAS:
                tsolve = time.perf_counter()
                fit_result = core.solve_from_gram(prepared, lam)
                model = parameterized_model(prepared, fit_result)
                pred_resid = core.predict_residual(x[train_idx[val_idx]], model)
                pred_native = anchors[train_idx[val_idx], None, :] + pred_resid.reshape(len(val_idx), 30, 28)
                solve_seconds = time.perf_counter() - tsolve
                pos, fd_score, pos_eps, fd_eps = mean_fold_metrics(
                    pred_native, targets[val_idx], train_eps[val_idx], sigma_fold, safe_fold
                )
                for ep, val in pos_eps.items():
                    cv_episode_rows.append({
                        "fold": f, "lambda": lam, "episode_index": ep,
                        "position_standardized_mae": val, "safe_fd_normalized": fd_eps[ep],
                        "target_sigma_fit_scope": "fold_fit_H30_only",
                    })
                cv_fold_rows.append({
                    "fold": f, "lambda": lam, "fit_rows": len(fit_idx),
                    "fit_groups": len(set(train_group[str(i)] for i in train_ids[fit_idx])),
                    "fit_episodes": len(set(train_eps[fit_idx].tolist())),
                    "heldout_rows": len(val_idx),
                    "heldout_groups": len(set(train_group[str(i)] for i in train_ids[val_idx])),
                    "heldout_episodes": len(pos_eps),
                    "position_standardized_mae_episode_equal": pos,
                    "safe_fd_normalized_episode_equal": fd_score,
                })
                receipt = {
                    "fit_id": f"CV-F{f}-L{lam:g}", "fit_type": "cv",
                    "fold": f, "lambda": lam, "fit_rows": len(fit_idx),
                    "heldout_rows": len(val_idx), "features": x.shape[1], "outputs": 840,
                    "weight_sum_W": float(wfit.sum()), "alpha_sum_loss_equiv": float(wfit.sum() * lam),
                    "gram_eigendecomposition_seconds_fold_shared": gram_seconds,
                    "solve_and_score_seconds": solve_seconds,
                    "solver_residual_ratio": fit_result["solver_residual_ratio"],
                    "regularized_condition_number": fit_result["condition_number_regularized_gram"],
                    "objective_weighted_mse": None,
                    "cv_position_standardized_mae_episode_equal": pos,
                    "cv_safe_fd_normalized_episode_equal_secondary": fd_score,
                    "final_test_access": 0,
                }
                solver_receipts.append(receipt)
                fold_models[(f, lam)] = {
                    "x_mean": prepared["x_mean"].copy(), "x_scale": prepared["x_scale"].copy(),
                    "constant_mask": prepared["constant_mask"].copy(), "coef": fit_result["coef"].copy(),
                    "intercept": fit_result["intercept"].copy(), "lambda": lam,
                }

    by_lambda: dict[float, dict[int, float]] = {lam: {} for lam in LAMBDAS}
    for row in cv_episode_rows:
        lam = float(row["lambda"])
        ep = int(row["episode_index"])
        if ep in by_lambda[lam]:
            raise AssertionError(f"duplicate OOF episode score for lambda={lam}, episode={ep}")
        by_lambda[lam][ep] = float(row["position_standardized_mae"])
    expected_episodes = set(train_eps.tolist())
    if any(set(by_lambda[lam]) != expected_episodes for lam in LAMBDAS):
        raise AssertionError("OOF episode coverage differs from primary train")
    means = {lam: float(np.mean(list(by_lambda[lam].values()))) for lam in LAMBDAS}
    best = min(means.values())
    selected = max(lam for lam in LAMBDAS if means[lam] <= best + 1e-10)
    lambda_rows = [{
        "lambda": lam, "episode_equal_oof_position_mae": means[lam],
        "episode_count": len(by_lambda[lam]), "selected": lam == selected,
        "tie_tolerance_abs": 1e-10,
    } for lam in LAMBDAS]
    write_csv(ROOT / "results/CV_FOLD_LAMBDA_SCORES.csv", list(cv_fold_rows[0].keys()), cv_fold_rows)
    write_csv(ROOT / "results/CV_EPISODE_LAMBDA_SCORES.csv", list(cv_episode_rows[0].keys()), cv_episode_rows)
    write_csv(ROOT / "results/LAMBDA_SELECTION.csv", list(lambda_rows[0].keys()), lambda_rows)
    write_json(ROOT / "models/LAMBDA_SELECTION.json", {
        "selected_global_lambda": selected, "grid": list(LAMBDAS),
        "selection_metric": "mean over all primary-train held-out episodes of fold-fit standardized H30 Full28 position MAE",
        "score_by_lambda": {str(k): v for k, v in means.items()},
        "fold_validation_scores_are_hyperparameter_selected_oof": True,
        "independent_confirmation": False, "development_targets_read": False,
        "tie_tolerance_abs": 1e-10, "tie_rule": "choose larger lambda within tolerance",
        "grid_extended": False, "selected_at_boundary": selected in (LAMBDAS[0], LAMBDAS[-1]),
    })
    oof_pred = np.empty((len(train_idx), 30, 28), dtype=np.float32)
    for f in range(3):
        val_local = fold == f
        model = fold_models[(f, selected)]
        pred_resid = core.predict_residual(x[train_idx[val_local]], model)
        oof_pred[val_local] = (anchors[train_idx[val_local], None, :] + pred_resid.reshape(-1, 30, 28)).astype(np.float32)
    oof_path = ROOT / "predictions/oof/RIDGE_SELECTED_LAMBDA_TRAIN_OOF.npz"
    if oof_path.exists():
        raise FileExistsError(oof_path)
    np.savez_compressed(oof_path, anchor_id=train_ids, predictions_native=oof_pred)
    del fold_models

    wfull = core.episode_equal_weights(train_eps)
    tgram = time.perf_counter()
    prepared_full = core.prepare_weighted_gram(x[train_idx], y, wfull)
    gram_full_seconds = time.perf_counter() - tgram
    tfit = time.perf_counter()
    full_result = core.solve_from_gram(prepared_full, selected)
    model = parameterized_model(prepared_full, full_result)
    full_solve_seconds = time.perf_counter() - tfit
    full_residual = core.predict_residual(x[train_idx], model)
    train_pred = (anchors[train_idx, None, :] + full_residual.reshape(len(train_idx), 30, 28)).astype(np.float32)
    dev_residual = core.predict_residual(x[dev_idx], model)
    dev_pred = (anchors[dev_idx, None, :] + dev_residual.reshape(len(dev_idx), 30, 28)).astype(np.float32)

    model_path = ROOT / "models/ridge_full_train.npz"
    core.save_model_npz(model_path, model)
    train_record = write_prediction(ROOT / "predictions/train/RIDGE_FULL_TRAIN.npz", train_ids, train_pred, "train")
    dev_record = write_prediction(ROOT / "predictions/dev/RIDGE_FULL_DEV.npz", ids[dev_idx], dev_pred, "dev")
    full_receipt = {
        "fit_id": "FULL-PRIMARY-TRAIN-REFIT", "fit_type": "full_primary_train",
        "fold": "all_train", "lambda": selected, "fit_rows": len(train_idx),
        "heldout_rows": 0, "features": x.shape[1], "outputs": 840,
        "weight_sum_W": float(wfull.sum()), "alpha_sum_loss_equiv": float(wfull.sum() * selected),
        "gram_eigendecomposition_seconds_fold_shared": gram_full_seconds,
        "solve_and_score_seconds": full_solve_seconds,
        "solver_residual_ratio": full_result["solver_residual_ratio"],
        "regularized_condition_number": full_result["condition_number_regularized_gram"],
        "objective_weighted_mse": float(np.sum(wfull[:, None] * (y - full_residual) ** 2) / wfull.sum() + selected * np.sum(full_result["coef"] ** 2)),
        "cv_position_standardized_mae_episode_equal": None,
        "cv_safe_fd_normalized_episode_equal_secondary": None,
        "final_test_access": 0,
    }
    solver_receipts.append(full_receipt)
    write_csv(ROOT / "results/SOLVER_FIT_RECEIPTS.csv", list(solver_receipts[0].keys()), solver_receipts)
    total_wall = time.perf_counter() - wall_t0
    total_cpu = time.process_time() - process_t0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fs_used = shutil.disk_usage(ROOT).used
    fit_summary = {
        "status": "PASS_RIDGE_FIT_AND_TARGET_FREE_PREDICTION",
        "created_utc": utc_now(), "hostname": socket.gethostname(),
        "python": sys.version, "numpy": np.__version__,
        "thread_limit": THREAD_LIMIT, "blas_threadpools_capped": True,
        "threadpool_info": threadpool_info(),
        "total_cpu_seconds": total_cpu, "total_wall_seconds": total_wall,
        "peak_rss_kib": rss, "output_root_bytes_after_fit": fs_used,
        "output_root_bytes_before_fit": fs_before,
        "new_formal_cv_fits": 12, "new_full_train_refits": 1,
        "preflight_smoke_fit_separate": True, "lambda_grid_extended": False,
        "selected_global_lambda": selected, "cv_primary_train_rows": len(train_idx),
        "cv_primary_train_groups": len(set(train_group.values())),
        "train_predictions": train_record, "dev_predictions": dev_record,
        "oof_prediction_path": str(oof_path), "oof_prediction_sha256": sha(oof_path),
        "model_path": str(model_path), "model_sha256": sha(model_path),
        "model_parameter_count_weights_plus_intercept": int(model["coef"].size + model["intercept"].size),
        "dev_targets_read": False, "target_arrays_loaded_during_prediction_generation": False,
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(ROOT / "coordination/CPU_FIT_COST.json", fit_summary)
    model_lock = {
        "status": "PASS_MODEL_AND_PREDICTIONS_LOCKED_BEFORE_DEV_SCORING",
        "locked_utc": utc_now(), "selected_global_lambda": selected,
        "lambda_grid": list(LAMBDAS), "lambda_grid_extended": False,
        "fit_counts": {"cv": 12, "full_primary_train_refit": 1, "preflight_smoke": 1},
        "preflight_smoke_excluded_from_cv_and_lambda_selection": True,
        "model_sha256": sha(model_path), "train_prediction_sha256": train_record["sha256"],
        "dev_prediction_sha256": dev_record["sha256"], "oof_prediction_sha256": sha(oof_path),
        "input_sha256": sha(INPUTS), "train_target_cache_sha256": sha(TARGET_CACHE),
        "dev_targets_read": False, "model_parameter_count_weights_plus_intercept": int(model["coef"].size + model["intercept"].size),
        "source_group_fold_assignment_sha256": protocol["folds"]["assignment_sha256"],
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(model_lock_path, model_lock)
    accept = {
        "status": "PASS_RIDGE_TRAIN_DEV_PREDICTIONS",
        "model_lock_sha256": sha(model_lock_path), "model_sha256": sha(model_path),
        "selected_global_lambda": selected,
        "train": train_record, "dev": dev_record,
        "target_arrays_present_in_prediction_files": False,
        "dev_targets_read_before_lock": False,
        "prediction_forward_target_access": False,
        "finite": True, "accepted_train_rows": len(train_idx), "accepted_dev_rows": len(dev_idx),
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(ROOT / "qa/RIDGE_PREDICTION_ACCEPTANCE.json", accept)
    run_status = (ROOT / "RUN_STATUS.ini").read_text(encoding="utf-8")
    run_status = run_status.replace("STATUS=locked\n", "STATUS=running\n", 1)
    run_status = run_status.replace("NEW_CPU_CV_FITS=0\n", "NEW_CPU_CV_FITS=12\n", 1)
    run_status = run_status.replace("NEW_CPU_FULL_TRAIN_REFIT=0\n", "NEW_CPU_FULL_TRAIN_REFIT=1\n", 1)
    run_status = run_status.replace("SELECTED_GLOBAL_LAMBDA=\n", f"SELECTED_GLOBAL_LAMBDA={selected}\n", 1)
    run_status = run_status.replace("Ridge_TRAIN_DEV_PREDICTIONS_ACCEPTED=false\n", "Ridge_TRAIN_DEV_PREDICTIONS_ACCEPTED=true\n", 1)
    (ROOT / "RUN_STATUS.ini").write_text(run_status, encoding="utf-8")
    print(json.dumps({"status": fit_summary["status"], "selected_lambda": selected, "cv_scores": means, "cpu_seconds": total_cpu, "wall_seconds": total_wall, "peak_rss_kib": rss, "model_parameters": model_lock["model_parameter_count_weights_plus_intercept"], "dev_targets_read": False}, sort_keys=True))
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("--smoke", "--fit"):
        raise SystemExit("usage: fit_ridge.py --smoke|--fit")
    with threadpool_limits(limits=THREAD_LIMIT):
        pools = threadpool_info()
        bad = [p for p in pools if p.get("user_api") == "blas" and int(p.get("num_threads", 0)) > THREAD_LIMIT]
        if bad:
            raise RuntimeError(f"BLAS thread cap check failed: {bad}")
        return run_smoke() if sys.argv[1] == "--smoke" else run_fit()


if __name__ == "__main__":
    raise SystemExit(main())


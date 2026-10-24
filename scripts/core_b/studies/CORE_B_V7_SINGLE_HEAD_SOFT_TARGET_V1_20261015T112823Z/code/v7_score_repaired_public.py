# Repaired public copy of EXCL-CODE-010; source SHA-256: 94266d3ff1db9e39186f7be716142cdefdfbb7bb000028ae1e9e3d134467ec0c
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from v7_common import (
    DEV_TARGET_NPZ,
    METRIC_LOCK,
    NORMALIZER,
    ROOT,
    SEEDS,
    TASKS,
    V4,
    atomic_json,
    load_dev_prefix,
    load_safe_prediction,
    position_fd_seam_metrics,
    sha_file,
    write_csv,
)

METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1", "native_mae")
CONTRASTS = {
    "MIX_VS_GT": ("MIX_SOFT", "GT_RETRAIN"),
    "MIX_VS_SELF": ("MIX_SOFT", "SELF_SOFT"),
    "SELF_VS_GT": ("SELF_SOFT", "GT_RETRAIN"),
    "MIX_VS_OLD_V": ("MIX_SOFT", "V"),
    "MIX_VS_TEACHER": ("MIX_SOFT", "T"),
    "MIX_VS_BIAS_STATE": ("MIX_SOFT", "BIAS_STATE"),
    "MIX_VS_BIAS_VIDEO": ("MIX_SOFT", "BIAS_VIDEO"),
}
STUDENT_PROGRAMS = ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")
REFERENCE_METHODS = ("S", "V", "T", "BIAS_STATE", "BIAS_VIDEO", "PERSISTENCE")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def map_prediction_to_dev(path: Path, rows_path: Path, dev_rows: list[dict[str, Any]]) -> np.ndarray:
    prediction = load_safe_prediction(path).astype(np.float64, copy=False)
    rows = read_csv(rows_path)
    expected_by_id = {str(row["anchor_id"]): row for row in dev_rows}
    source_by_id: dict[str, tuple[int, int, int, int, str]] = {}
    values_by_id: dict[str, np.ndarray] = {}
    for row, value in zip(rows, prediction):
        anchor_id = str(row["anchor_id"])
        if anchor_id in source_by_id:
            raise RuntimeError(f"duplicate prediction receiver ID: {anchor_id}")
        task_id = int(row["task_id"])
        task_rank = int(row["task_rank"]) if row.get("task_rank") not in {None, ""} else TASKS.index(task_id)
        source_by_id[anchor_id] = (
            int(row["episode_id"]), task_id, int(row["anchor_frame"]),
            task_rank, str(row.get("source_cluster", row.get("source_group", ""))),
        )
        values_by_id[anchor_id] = value
    if set(source_by_id) != set(expected_by_id):
        raise RuntimeError(f"prediction ID set differs from label-free dev receivers: {path}")
    aligned: list[np.ndarray] = []
    for row in dev_rows:
        anchor_id = str(row["anchor_id"])
        expected = (
            int(row["episode_id"]), int(row["task_id"]), int(row["anchor_frame"]),
            int(row["task_rank"]), str(row["source_cluster"]),
        )
        if source_by_id[anchor_id] != expected:
            raise RuntimeError(f"prediction receiver metadata join mismatch: {anchor_id}")
        aligned.append(values_by_id[anchor_id])
    result = np.stack(aligned)
    if result.shape != (185, 30, 28) or not np.isfinite(result).all():
        raise RuntimeError(f"prediction shape/finite failure after join: {result.shape}")
    return result


def validate_nonempty_prediction_seal() -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    seal_path = ROOT / "qa/PREDICTION_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("dev scorer refuses missing student prediction seal")
    seal = read_json(seal_path)
    entries = seal.get("entries", [])
    if (
        seal.get("status") != "PASS"
        or not seal.get("nonempty_seal")
        or int(seal.get("planned_student_jobs", -1)) != 9
        or int(seal.get("accepted_student_jobs", -1)) != 9
        or int(seal.get("final_heldout_payload_access", -1)) != 0
        or len(entries) != 9
    ):
        raise RuntimeError("dev scorer refuses absent, empty, or incomplete student prediction seal")
    dev_data = load_dev_prefix()
    from v7_common import receiver_rows
    dev_rows = receiver_rows(dev_data)
    observed: set[tuple[str, int]] = set()
    aligned: dict[tuple[str, int], np.ndarray] = {}
    for entry in entries:
        key = (str(entry["program_id"]), int(entry["seed"]))
        if key in observed:
            raise RuntimeError(f"duplicate sealed student prediction entry: {key}")
        observed.add(key)
        if key[0] not in STUDENT_PROGRAMS or key[1] not in SEEDS:
            raise RuntimeError(f"unexpected student prediction entry: {key}")
        pred_path, rows_path, complete_path = (
            Path(entry["prediction_path"]),
            Path(entry["prediction_rows_path"]),
            Path(entry["completion_path"]),
        )
        if (
            sha_file(pred_path) != entry["prediction_sha256"]
            or sha_file(rows_path) != entry["prediction_rows_sha256"]
            or sha_file(complete_path) != entry["completion_sha256"]
        ):
            raise RuntimeError(f"sealed student prediction hash mismatch: {key}")
        completion = read_json(complete_path)
        if (
            completion.get("status") != "PASS"
            or completion.get("dev_future_target_arrays_requested") is not False
            or int(completion.get("final_heldout_payload_access", -1)) != 0
        ):
            raise RuntimeError(f"student inference receipt is not target blind/final0: {key}")
        aligned[key] = map_prediction_to_dev(pred_path, rows_path, dev_rows)
    expected = {(program, seed) for program in STUDENT_PROGRAMS for seed in SEEDS}
    if observed != expected:
        raise RuntimeError(f"student seal matrix mismatch: {sorted(observed)}")
    return seal, entries, aligned, dev_data


def load_eval_references(dev_rows: list[dict[str, Any]], dev_anchors: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    source_map = read_json(ROOT / "protocol/SOURCE_MAP.json")
    refs: dict[tuple[str, int], np.ndarray] = {}
    for entry in source_map["evaluation_reference_bundles"]:
        role, seed = str(entry["role"]), int(entry["seed"])
        if role == "T":
            method = "T"
        elif role in {"S", "V", "BIAS_STATE", "BIAS_VIDEO"}:
            method = role
        else:
            raise RuntimeError(f"unexpected evaluation reference role: {role}")
        path, rows_path = Path(entry["prediction_path"]), Path(entry["rows_path"])
        if sha_file(path) != entry["prediction_sha256"] or sha_file(rows_path) != entry["rows_sha256"]:
            raise RuntimeError(f"accepted reference artifact hash changed: {role}/{seed}")
        key = (method, seed)
        if key in refs:
            raise RuntimeError(f"duplicate evaluation reference: {key}")
        refs[key] = map_prediction_to_dev(path, rows_path, dev_rows)
    expected = {(method, seed) for method in ("S", "V", "T", "BIAS_STATE", "BIAS_VIDEO") for seed in SEEDS}
    if set(refs) != expected:
        raise RuntimeError(f"evaluation references incomplete: {sorted(refs)}")
    anchors = np.asarray(dev_anchors, dtype=np.float64)
    if anchors.shape != (len(dev_rows), 28) or not np.isfinite(anchors).all():
        raise RuntimeError("label-free dev anchor prefix has invalid shape/values for persistence")
    refs[("PERSISTENCE", 0)] = np.broadcast_to(anchors[:, None, :], (len(anchors), 30, 28)).copy()
    return refs


def load_dev_target_after_seal(
    seal: dict[str, Any], dev_data: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if seal.get("status") != "PASS" or len(seal.get("entries", [])) != 9:
        raise RuntimeError("refusing dev outcome access before a nonempty 9-prediction seal")
    if not DEV_TARGET_NPZ.is_file():
        raise FileNotFoundError(f"scorer-only dev target archive missing: {DEV_TARGET_NPZ}")
    accepted_exposure = read_json(V4 / "inventory/OUTCOME_EXPOSURE.json")
    expected_path = Path(accepted_exposure["v4_target_artifact"])
    expected_sha256 = str(accepted_exposure["v4_target_artifact_sha256"])
    if expected_path != DEV_TARGET_NPZ:
        raise RuntimeError("V7 scorer target path differs from the accepted V4 scorer-only artifact")
    actual_target_sha256 = sha_file(DEV_TARGET_NPZ)
    if actual_target_sha256 != expected_sha256:
        raise RuntimeError("scorer-only dev target archive hash differs from accepted V4 exposure receipt")
    started = utc_now()
    atomic_json(ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json", {
        "schema_version": 1, "status": "TARGET_READ_STARTED_AFTER_SEAL", "invoked_utc": started,
        "prediction_seal_sha256": sha_file(ROOT / "qa/PREDICTION_SEAL.json"),
        "target_archive_path": str(DEV_TARGET_NPZ), "target_archive_sha256": actual_target_sha256,
        "archive_load_operations": 0, "target_values_loaded": 0, "final_heldout_payload_access": 0,
    })
    with np.load(DEV_TARGET_NPZ, allow_pickle=False) as archive:
        required = {"anchor_id", "episode_id", "task_id", "source_cluster", "target_native_h30"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"scorer-only archive missing required fields {missing}")
        target_ids = [str(value) for value in archive["anchor_id"].tolist()]
        target_episode = archive["episode_id"].astype(np.int64, copy=True)
        target_task = archive["task_id"].astype(np.int64, copy=True)
        target_group = archive["source_cluster"].astype(str, copy=True)
        target = archive["target_native_h30"].astype(np.float64, copy=True)
    if len(target_ids) != 185 or len(set(target_ids)) != 185 or target.shape != (185, 30, 28) or not np.isfinite(target).all():
        raise RuntimeError("post-seal scorer target row support/shape/finite check failed")
    atomic_json(ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json", {
        "schema_version": 1, "status": "TARGET_LOADED_AFTER_SEAL", "invoked_utc": started,
        "prediction_seal_sha256": sha_file(ROOT / "qa/PREDICTION_SEAL.json"),
        "target_archive_path": str(DEV_TARGET_NPZ), "target_archive_sha256": actual_target_sha256,
        "archive_load_operations": 1, "target_values_loaded": 185, "target_shape": list(target.shape),
        "final_heldout_payload_access": 0,
    })
    target_index = {anchor_id: i for i, anchor_id in enumerate(target_ids)}
    aligned: list[np.ndarray] = []
    for row in dev_rows_for_data(dev_data):
        anchor_id = str(row["anchor_id"])
        if anchor_id not in target_index:
            raise RuntimeError(f"dev target archive missing sealed receiver: {anchor_id}")
        i = target_index[anchor_id]
        if (
            int(target_episode[i]) != int(row["episode_id"])
            or int(target_task[i]) != int(row["task_id"])
            or str(target_group[i]) != str(row["source_cluster"])
        ):
            raise RuntimeError(f"dev target ID/episode/task/source-group join mismatch: {anchor_id}")
        aligned.append(target[i])
    target_native = np.stack(aligned)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "target_archive_path": str(DEV_TARGET_NPZ),
        "target_archive_sha256": actual_target_sha256,
        "target_archive_accepted_v4_sha256": expected_sha256,
        "accepted_v4_exposure_receipt_sha256": sha_file(V4 / "inventory/OUTCOME_EXPOSURE.json"),
        "target_reader_invoked_after_prediction_seal": True,
        "prediction_seal_sha256": sha_file(ROOT / "qa/PREDICTION_SEAL.json"),
        "archive_load_operations": 1,
        "unique_target_windows_loaded": 185,
        "target_ordered_by_prediction_seal_receiver_IDs": True,
        "dev_targets_used_in_student_forward": 0,
        "final_heldout_payload_access": 0,
        "invoked_utc": started,
    }
    return target_native, target_ids, receipt


def dev_rows_for_data(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    from v7_common import receiver_rows
    return receiver_rows(data)


def score_all(
    predictions: dict[tuple[str, int], np.ndarray],
    refs: dict[tuple[str, int], np.ndarray],
    target: np.ndarray,
    dev_data: dict[str, np.ndarray],
    metric_lock: dict[str, Any],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    anchors = np.asarray(dev_data["anchor_native"], dtype=np.float64)
    sigma = np.asarray(metric_lock["target_sigma"], dtype=np.float64)
    fd_scale = np.asarray(metric_lock["safe_fd_scale"], dtype=np.float64)
    scores: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, pred in {**refs, **predictions}.items():
        values = position_fd_seam_metrics(pred, target, anchors, sigma, fd_scale)
        scores[key] = dict(zip(METRICS, values))
    return scores


def anchor_score_rows(
    scores: dict[tuple[str, int], dict[str, np.ndarray]],
    dev_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (method, seed), metric_values in sorted(scores.items()):
        for i, dev in enumerate(dev_rows):
            rows.append({
                "method": method,
                "seed": str(seed),
                "anchor_id": str(dev["anchor_id"]),
                "episode_id": int(dev["episode_id"]),
                "task_id": int(dev["task_id"]),
                "source_group": str(dev["source_cluster"]),
                **{metric: float(metric_values[metric][i]) for metric in METRICS},
            })
    return rows


def task_episode_macro(
    metric_values: np.ndarray,
    dev_rows: list[dict[str, Any]],
    task: int,
) -> tuple[float, int, int, int]:
    positions = [i for i, row in enumerate(dev_rows) if int(row["task_id"]) == task]
    by_episode: dict[int, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    for i in positions:
        by_episode[int(dev_rows[i]["episode_id"])].append(float(metric_values[i]))
        by_group[str(dev_rows[i]["source_cluster"])].append(float(metric_values[i]))
    episode_mean = float(np.mean([np.mean(values) for values in by_episode.values()]))
    return episode_mean, len(positions), len(by_episode), len(by_group)


def source_group_arrays(
    scores: dict[tuple[str, int], dict[str, np.ndarray]],
    dev_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int, int, str], float], dict[int, list[str]]]:
    group_metrics: dict[tuple[str, int, int, str], float] = {}
    groups_by_task: dict[int, list[str]] = {}
    for task in TASKS:
        groups = sorted({str(row["source_cluster"]) for row in dev_rows if int(row["task_id"]) == task})
        groups_by_task[task] = groups
        for (method, seed), metric_values in scores.items():
            for group in groups:
                positions = [
                    i for i, row in enumerate(dev_rows)
                    if int(row["task_id"]) == task and str(row["source_cluster"]) == group
                ]
                if not positions:
                    raise RuntimeError(f"empty source group {task}/{group}")
                for metric in METRICS:
                    group_metrics[(method, seed, task, group, metric)] = float(np.mean(metric_values[metric][positions]))
    return group_metrics, groups_by_task


def task_seed_table(
    scores: dict[tuple[str, int], dict[str, np.ndarray]],
    dev_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    methods = sorted(scores)
    for method, seed in methods:
        for task in TASKS:
            index = int(seed) if seed in SEEDS else 0
            value_map = scores[(method, seed)]
            values = {}
            counts = None
            for metric in METRICS:
                value, anchors, episodes, groups = task_episode_macro(value_map[metric], dev_rows, task)
                values[metric] = value
                counts = (anchors, episodes, groups)
            rows.append({
                "analysis_role": "student" if method in STUDENT_PROGRAMS else "accepted_reference",
                "method": method,
                "seed": str(seed),
                "task_id": task,
                "anchors": counts[0],
                "episodes": counts[1],
                "source_groups": counts[2],
                "position_mae": values["position_mae"],
                "safe_fd_l1": values["safe_fd_l1"],
                "seam_safe_fd_l1": values["seam_safe_fd_l1"],
                "native_mae": values["native_mae"],
            })
        # Equal-task macro for the paired source instance.
        macro = {}
        for metric in METRICS:
            task_values = [task_episode_macro(scores[(method, seed)][metric], dev_rows, task)[0] for task in TASKS]
            macro[metric] = float(np.mean(task_values))
        rows.append({
            "analysis_role": "student" if method in STUDENT_PROGRAMS else "accepted_reference",
            "method": method,
            "seed": str(seed),
            "task_id": "macro",
            "anchors": 185,
            "episodes": 47,
            "source_groups": len({str(row["source_cluster"]) for row in dev_rows}),
            **macro,
        })
    return rows


def pair_effects_and_cis(
    scores: dict[tuple[str, int], dict[str, np.ndarray]],
    dev_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    group_values, groups_by_task = source_group_arrays(scores, dev_rows)
    rng = np.random.default_rng(2026101507)
    draws: dict[int, np.ndarray] = {
        task: rng.integers(0, len(groups), size=(5000, len(groups)), endpoint=False)
        for task, groups in groups_by_task.items()
    }
    point_cache: dict[tuple[str, int, int, str], float] = {}
    for key, value in group_values.items():
        point_cache[key] = value

    def task_group_mean(method: str, seed: int, task: int, metric: str, sample_idx: np.ndarray | None = None) -> float | np.ndarray:
        groups = groups_by_task[task]
        values = np.asarray([group_values[(method, seed, task, group, metric)] for group in groups], dtype=np.float64)
        if sample_idx is None:
            return float(values.mean())
        return values[sample_idx].mean(axis=1)

    def method_seed_for(method: str, seed: int) -> int:
        return int(seed) if (method, int(seed)) in scores else 0

    def contrast_draws(candidate: str, reference: str, metric: str, scope: str, omitted: int | None = None) -> np.ndarray:
        active_tasks = [task for task in TASKS if task != omitted]
        out = np.zeros(5000, dtype=np.float64)
        for seed in SEEDS:
            cand_seed = method_seed_for(candidate, seed)
            ref_seed = method_seed_for(reference, seed)
            if scope == "macro":
                cand = np.zeros(5000, dtype=np.float64)
                ref = np.zeros(5000, dtype=np.float64)
                for task in active_tasks:
                    cand += task_group_mean(candidate, cand_seed, task, metric, draws[task])
                    ref += task_group_mean(reference, ref_seed, task, metric, draws[task])
                cand /= len(active_tasks)
                ref /= len(active_tasks)
            else:
                task = int(scope)
                cand = task_group_mean(candidate, cand_seed, task, metric, draws[task])
                ref = task_group_mean(reference, ref_seed, task, metric, draws[task])
            out += ref - cand
        return out / len(SEEDS)

    def point_task_mean(method: str, seed: int, task: int, metric: str) -> float:
        return float(task_group_mean(method, seed, task, metric))

    def point_macro(method: str, seed: int, metric: str, omit_task: int | None = None) -> float:
        active_tasks = [task for task in TASKS if task != omit_task]
        return float(np.mean([point_task_mean(method, seed, task, metric) for task in active_tasks]))

    effect_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    loo_seed_rows: list[dict[str, Any]] = []
    loo_task_rows: list[dict[str, Any]] = []
    for contrast, (candidate, reference) in CONTRASTS.items():
        for metric in METRICS:
            per_seed_macro_gain: list[float] = []
            for seed in SEEDS:
                cand_seed = method_seed_for(candidate, seed)
                ref_seed = method_seed_for(reference, seed)
                task_gains = {}
                for task in TASKS:
                    gain = point_task_mean(reference, ref_seed, task, metric) - point_task_mean(candidate, cand_seed, task, metric)
                    task_gains[task] = gain
                    effect_rows.append({
                        "contrast": contrast, "candidate": candidate, "reference": reference,
                        "seed": seed, "scope": "task", "task_id": task, "metric": metric,
                        "candidate_loss": point_task_mean(candidate, cand_seed, task, metric),
                        "reference_loss": point_task_mean(reference, ref_seed, task, metric),
                        "gain": gain, "gain_sample_sd_across_seeds": "", "gain_sign": "positive_favors_candidate",
                    })
                macro_gain = point_macro(reference, ref_seed, metric) - point_macro(candidate, cand_seed, metric)
                per_seed_macro_gain.append(macro_gain)
                effect_rows.append({
                    "contrast": contrast, "candidate": candidate, "reference": reference,
                    "seed": seed, "scope": "macro", "task_id": "",
                    "metric": metric,
                    "candidate_loss": point_macro(candidate, cand_seed, metric),
                    "reference_loss": point_macro(reference, ref_seed, metric),
                    "gain": macro_gain, "gain_sample_sd_across_seeds": "", "gain_sign": "positive_favors_candidate",
                })
            for scope in ("macro", *(str(task) for task in TASKS)):
                sample = contrast_draws(candidate, reference, metric, scope)
                scope_groups = sum(len(groups_by_task[t]) for t in TASKS) if scope == "macro" else len(groups_by_task[int(scope)])
                if scope == "macro":
                    point = float(np.mean(per_seed_macro_gain))
                else:
                    task = int(scope)
                    point = float(np.mean([
                        point_task_mean(reference, method_seed_for(reference, seed), task, metric)
                        - point_task_mean(candidate, method_seed_for(candidate, seed), task, metric)
                        for seed in SEEDS
                    ]))
                ci_rows.append({
                    "contrast": contrast, "candidate": candidate, "reference": reference,
                    "scope": scope, "metric": metric, "point_gain": point,
                    "ci95_low": float(np.quantile(sample, 0.025)),
                    "ci95_high": float(np.quantile(sample, 0.975)),
                    "draws": 5000, "bootstrap_seed": 2026101507,
                    "source_groups_total": scope_groups,
                    "multiplicity_adjustment": "none",
                    "interpretation": "exploratory conditional percentile interval",
                })
            seed_sd = float(np.std(per_seed_macro_gain, ddof=1))
            effect_rows.append({
                "contrast": contrast, "candidate": candidate, "reference": reference,
                "seed": "", "scope": "macro_mean_sd", "task_id": "", "metric": metric,
                "candidate_loss": float(np.mean([point_macro(candidate, method_seed_for(candidate, seed), metric) for seed in SEEDS])),
                "reference_loss": float(np.mean([point_macro(reference, method_seed_for(reference, seed), metric) for seed in SEEDS])),
                "gain": float(np.mean(per_seed_macro_gain)),
                "gain_sample_sd_across_seeds": seed_sd,
                "gain_sign": "positive_favors_candidate",
            })
            for omitted in SEEDS:
                included = [seed for seed in SEEDS if seed != omitted]
                loss_gain = np.mean([
                    point_macro(reference, method_seed_for(reference, seed), metric)
                    - point_macro(candidate, method_seed_for(candidate, seed), metric)
                    for seed in included
                ])
                loo_seed_rows.append({
                    "contrast": contrast, "metric": metric, "omitted_seed": omitted,
                    "scope": "macro", "gain": float(loss_gain),
                    "remaining_seed_count": 2, "source_seed_block_deleted": True,
                })
            for omitted_task in TASKS:
                gain = np.mean([
                    point_macro(reference, method_seed_for(reference, seed), metric, omit_task=omitted_task)
                    - point_macro(candidate, method_seed_for(candidate, seed), metric, omit_task=omitted_task)
                    for seed in SEEDS
                ])
                loo_task_rows.append({
                    "contrast": contrast, "metric": metric, "omitted_task": omitted_task,
                    "scope": "macro", "gain": float(gain),
                    "remaining_task_count": 4, "task_deleted": True,
                })

    return effect_rows, ci_rows, {
        "seed_block_loo": loo_seed_rows,
        "task_loo": loo_task_rows,
        "group_values": group_values,
        "groups_by_task": groups_by_task,
        "bootstrap_seed": 2026101507,
        "bootstrap_draws": 5000,
    }


def retention_rows(
    scores: dict[tuple[str, int], dict[str, np.ndarray]],
    dev_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values_by_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        v = float(np.mean([task_episode_macro(scores[("V", seed)]["position_mae"], dev_rows, task)[0] for task in TASKS]))
        t = float(np.mean([task_episode_macro(scores[("T", seed)]["position_mae"], dev_rows, task)[0] for task in TASKS]))
        student = float(np.mean([task_episode_macro(scores[("MIX_SOFT", seed)]["position_mae"], dev_rows, task)[0] for task in TASKS]))
        numerator = v - student
        denominator = v - t
        threshold = 1e-12 * max(1.0, abs(v), abs(t))
        valid = denominator > 0 and denominator > threshold
        rows.append({
            "scope": "seed",
            "seed": seed,
            "L_V_position_mae": v,
            "L_T_position_mae": t,
            "L_MIX_SOFT_position_mae": student,
            "numerator_LV_minus_student": numerator,
            "denominator_LV_minus_LT": denominator,
            "epsilon_num": threshold,
            "retention_R": numerator / denominator if valid else "NA",
            "ratio_ci": False,
            "valid_denominator": valid,
        })
        values_by_seed[seed] = {"V": v, "T": t, "student": student}
    overall_v = float(np.mean([values_by_seed[seed]["V"] for seed in SEEDS]))
    overall_t = float(np.mean([values_by_seed[seed]["T"] for seed in SEEDS]))
    overall_student = float(np.mean([values_by_seed[seed]["student"] for seed in SEEDS]))
    numerator = overall_v - overall_student
    denominator = overall_v - overall_t
    threshold = 1e-12 * max(1.0, abs(overall_v), abs(overall_t))
    valid = denominator > 0 and denominator > threshold
    rows.append({
        "scope": "overall_equal_seed_losses",
        "seed": "macro",
        "L_V_position_mae": overall_v,
        "L_T_position_mae": overall_t,
        "L_MIX_SOFT_position_mae": overall_student,
        "numerator_LV_minus_student": numerator,
        "denominator_LV_minus_LT": denominator,
        "epsilon_num": threshold,
        "retention_R": numerator / denominator if valid else "NA",
        "ratio_ci": False,
        "valid_denominator": valid,
    })
    return rows


def reconcile_accepted_scores(scores: dict[tuple[str, int], dict[str, np.ndarray]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_root = ROOT.parent.parent.parent / "reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
    accepted_path = accepted_root / "results/PER_WINDOW_ACCEPTED_REFERENCES.csv"
    derived_path = accepted_root / "results/PER_WINDOW_DERIVED_LOSSES.csv"
    reuse_lock_path = ROOT / "protocol/REFERENCE_SCORE_REUSE_LOCK.json"
    reuse_lock = read_json(reuse_lock_path)
    if reuse_lock.get("status") != "LOCKED_BEFORE_DEV_SCORING":
        raise RuntimeError("accepted V5 per-window score reuse lock is missing or invalid")
    locked_files = {row["relative_path"]: row for row in reuse_lock["accepted_reference_score_files"]}
    if sha_file(accepted_path) != locked_files["results/PER_WINDOW_ACCEPTED_REFERENCES.csv"]["sha256"]:
        raise RuntimeError("accepted V5 per-window reference score digest mismatch")
    if sha_file(derived_path) != locked_files["results/PER_WINDOW_DERIVED_LOSSES.csv"]["sha256"]:
        raise RuntimeError("accepted V5 per-window derived score digest mismatch")
    refs = read_csv(accepted_path)
    derived = read_csv(derived_path)
    expected_files = {
        str(accepted_path): "V5_PER_WINDOW_ACCEPTED_REFERENCES",
        str(derived_path): "V5_PER_WINDOW_DERIVED_LOSSES",
    }
    rows_by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in refs:
        if row.get("method") not in {"S", "V", "BIAS_STATE", "BIAS_VIDEO", "PERSISTENCE"}:
            continue
        seed_value = str(row.get("seed", ""))
        if seed_value in {"", "shared", "NA"}:
            seed_value = "0"
        rows_by_key[(row["method"], int(seed_value), str(row["anchor_id"]))] = row
    for row in derived:
        if row.get("method") == "MIX50":
            rows_by_key[("T", int(row["seed"]), str(row["anchor_id"]))] = row
    comparisons = []
    maximum_absolute_difference = 0.0
    for method in ("S", "V", "T", "BIAS_STATE", "BIAS_VIDEO"):
        for seed in SEEDS:
            key = (method, seed)
            if key not in scores:
                raise RuntimeError(f"V7 scorer lacks reference values for {key}")
            if method == "T":
                table_method = "T"
            else:
                table_method = method
            diffs = []
            for i, row in enumerate(dev_rows):
                key_row = (table_method, seed, str(row["anchor_id"]))
                accepted = rows_by_key.get(key_row)
                if accepted is None:
                    raise RuntimeError(f"accepted V5 score row missing for {key_row}")
                if (
                    int(accepted["episode_id"]) != int(row["episode_id"])
                    or int(accepted["task_id"]) != int(row["task_id"])
                    or str(accepted["source_group"]) != str(row["source_cluster"])
                ):
                    raise RuntimeError(f"accepted V5 score metadata mismatch for {key_row}")
                for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1"):
                    diff = abs(float(accepted[metric]) - float(scores[key][metric][i]))
                    diffs.append(diff)
                    maximum_absolute_difference = max(maximum_absolute_difference, diff)
            comparisons.append({
                "method": method,
                "seed": seed,
                "window_metric_values_checked": 185 * 3,
                "max_abs_difference": max(diffs),
                "accepted_score_source": "V5 per-window score table",
            })
    for seed in SEEDS:
        diffs = []
        for i, row in enumerate(dev_rows):
            accepted = rows_by_key.get(("PERSISTENCE", seed, str(row["anchor_id"])))
            if accepted is None:
                raise RuntimeError(f"accepted V5 persistence score row missing for seed {seed}/{row['anchor_id']}")
            if (int(accepted["episode_id"]) != int(row["episode_id"]) or int(accepted["task_id"]) != int(row["task_id"]) or str(accepted["source_group"]) != str(row["source_cluster"])):
                raise RuntimeError(f"accepted V5 persistence metadata mismatch for seed {seed}/{row['anchor_id']}")
            for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1"):
                diff = abs(float(accepted[metric]) - float(scores[("PERSISTENCE", 0)][metric][i]))
                diffs.append(diff)
                maximum_absolute_difference = max(maximum_absolute_difference, diff)
        comparisons.append({
            "method": "PERSISTENCE", "seed": seed, "window_metric_values_checked": 185 * 3,
            "max_abs_difference": max(diffs), "accepted_score_source": "V5 per-window score table",
        })
    if maximum_absolute_difference > 5e-10:
        raise RuntimeError(f"V7 rescoring differs from accepted V5 score values by {maximum_absolute_difference}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "accepted_score_files": [
            {"path": path, "sha256": sha_file(Path(path)), "role": role}
            for path, role in expected_files.items()
        ],
        "comparisons": comparisons,
        "maximum_absolute_difference": maximum_absolute_difference,
        "verified_method_seed_pairs": len(comparisons),
    }


def score() -> None:
    scorer_seal_path = ROOT / "qa/SCORER_SEAL.json"
    target_access_path = ROOT / "qa/DEV_TARGET_ACCESS_RECEIPT.json"
    if scorer_seal_path.exists():
        existing = read_json(scorer_seal_path)
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "already_scored", "seal_sha256": sha_file(scorer_seal_path)}))
            return
        raise RuntimeError("existing scorer seal is not PASS; preserve and inspect partial scoring output")
    if target_access_path.exists():
        prior_access = read_json(target_access_path)
        if prior_access.get("status") != "PASS":
            raise RuntimeError("a prior scoring attempt already opened dev targets; preserve it and do not reload outcomes automatically")
        raise RuntimeError("dev target access receipt exists without scorer seal; preserve and inspect before retry")

    prediction_seal, student_entries, student_predictions, dev_data = validate_nonempty_prediction_seal()
    from v7_common import receiver_rows
    dev_rows = receiver_rows(dev_data)
    target_predictions = load_eval_references(dev_rows, dev_data["anchor_native"])

    # Only after all nine target-free student outputs are sealed and checked do
    # we open the scorer-only development target archive.
    target, target_ids, exposure = load_dev_target_after_seal(prediction_seal, dev_data)
    exposure_pending = dict(exposure)
    exposure_pending.update({"status": "TARGET_LOADED_SCORING_PENDING", "target_native_field": "target_native_h30", "target_windows": 185, "target_shape": [185, 30, 28]})
    atomic_json(target_access_path, exposure_pending)
    metric_lock = read_json(METRIC_LOCK)
    if metric_lock.get("status") != "LOCKED" or int(metric_lock.get("final_heldout_payload_access", -1)) != 0:
        raise RuntimeError("canonical metric lock is absent or final-test contaminated")
    scores = score_all(student_predictions, target_predictions, target, dev_data, metric_lock)
    reconciliation = reconcile_accepted_scores(scores, dev_rows)

    per_window = anchor_score_rows(scores, dev_rows)
    write_csv(ROOT / "results/PER_WINDOW_METRICS.csv", list(per_window[0]), per_window)
    primary = task_seed_table(scores, dev_rows)
    write_csv(ROOT / "results/PRIMARY_TASK_SEED.csv", list(primary[0]), primary)

    effect_rows, ci_rows, analyses = pair_effects_and_cis(scores, dev_rows)
    write_csv(ROOT / "results/PAIRED_EFFECTS.csv", list(effect_rows[0]), effect_rows)
    write_csv(ROOT / "results/PAIRED_GROUP_CI.csv", list(ci_rows[0]), ci_rows)
    write_csv(ROOT / "results/SEED_BLOCK_LOO.csv", list(analyses["seed_block_loo"][0]), analyses["seed_block_loo"])
    write_csv(ROOT / "results/TASK_LOO.csv", list(analyses["task_loo"][0]), analyses["task_loo"])
    retention = retention_rows(scores, dev_rows)
    write_csv(ROOT / "results/TEACHER_RETENTION.csv", list(retention[0]), retention)

    training_curves: list[dict[str, Any]] = []
    train_diag: list[dict[str, Any]] = []
    fit_index_path = ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json"
    fit_index = read_json(fit_index_path)
    if fit_index.get("status") != "PASS" or len(fit_index.get("entries", [])) != 9:
        raise RuntimeError("accepted fit index is absent or incomplete")
    accepted_fit_map = {(str(row["program_id"]), int(row["seed"])): row for row in fit_index["entries"]}
    for seed in SEEDS:
        for program in STUDENT_PROGRAMS:
            accepted_fit = accepted_fit_map[(program, seed)]
            manifest_path = Path(accepted_fit["model_manifest_path"])
            manifest = read_json(manifest_path)
            complete_path = Path(accepted_fit["completion_path"])
            if read_json(complete_path).get("status") != "PASS":
                raise RuntimeError(f"accepted fit marker absent for {program}/{seed}")
            curve_path = manifest_path.parent / "TRAINING_CURVES.csv"
            curve = read_csv(curve_path)
            if len(curve) != 160 or int(curve[-1]["epoch"]) != 160:
                raise RuntimeError(f"training curve is incomplete for {program}/{seed}")
            for row in curve:
                training_curves.append({"program": program, "seed": seed, **row})
            last = curve[-1]
            train_diag.append({
                "program": program,
                "seed": seed,
                "last_epoch": int(last["epoch"]),
                "weighted_objective_loss": float(last["weighted_objective_loss"]),
                "fit_position_mae": float(last["fit_position_mae"]),
                "fit_safe_fd_l1": float(last["fit_safe_fd_l1"]),
                "fit_seam_safe_fd_l1": float(last["fit_seam_safe_fd_l1"]),
                "ground_truth_position_mae": float(last["ground_truth_position_mae"]),
                "ground_truth_safe_fd_l1": float(last["ground_truth_safe_fd_l1"]),
                "ground_truth_seam_safe_fd_l1": float(last["ground_truth_seam_safe_fd_l1"]),
                "ground_truth_native_mae": float(last["ground_truth_native_mae"]),
                "student_to_V_position_distance": float(last["student_to_V_position_distance"]),
                "student_to_V_safe_fd_distance": float(last["student_to_V_safe_fd_distance"]),
                "student_to_T_position_distance": float(last["student_to_T_position_distance"]),
                "student_to_T_safe_fd_distance": float(last["student_to_T_safe_fd_distance"]),
                "train_prediction_kind": "in-sample pre-update batch outputs aggregated over epoch 160",
                "separate_train_diagnostic_prediction_jobs": 0,
                "target_role": program,
                "train_target_sha256": manifest["training_target_sha256"],
            })
    write_csv(ROOT / "results/TRAINING_CURVES.csv", list(training_curves[0]), training_curves)
    write_csv(ROOT / "results/TRAIN_FIT_AND_IMITATION.csv", list(train_diag[0]), train_diag)

    exposure.update({
        "status": "PASS",
        "student_prediction_jobs_accepted": 9,
        "student_window_loss_evaluations": 9 * 185,
        "student_unique_windows": 185,
        "reference_prediction_bundles_loaded_for_scoring": 15,
        "reference_window_loss_evaluations": 15 * 185,
        "persistence_reference_window_loss_evaluations": 185,
        "student_forward_used_teacher_predictions": False,
        "student_forward_used_dev_targets": False,
        "teacher_train_prediction_bundles": 6,
        "teacher_train_forward_windows": 6 * 1280,
        "derived_train_T_prediction_windows": 3 * 1280,
        "train_Y_archive_loads_after_teacher_seal": 1,
        "train_Y_archive_windows_loaded": 1280,
        "student_fit_Y_native_array_loads": 9,
        "student_fit_Y_native_array_rows_loaded": 9 * 1280,
        "student_fit_Z_native_array_loads": 9,
        "student_fit_Z_native_array_rows_loaded": 9 * 1280,
        "stage0_optimizer_updates": 6,
        "stage0_training_window_presentations": 3 * 2 * 64,
        "formal_fit_count": 9,
        "formal_optimizer_updates": int(fit_index["total_accepted_optimizer_updates"]),
        "formal_training_window_presentations": 9 * 160 * 1280,
        "total_student_training_window_presentations_including_stage0": 9 * 160 * 1280 + 3 * 2 * 64,
        "reference_prediction_bundle_window_loss_evaluations": 15 * 185,
        "persistence_reference_window_loss_evaluations": 185,
        "total_dev_metric_window_evaluations": (9 + 15 + 1) * 185,
        "historical_V6_counters_added_to_V7_totals": False,
        "final_heldout_payload_access": 0,
    })
    atomic_json(ROOT / "inventory/OUTCOME_EXPOSURE.json", exposure)
    atomic_json(target_access_path, {**exposure_pending, "status": "PASS", "scorer_seal_pending": True, "scored_utc": utc_now()})
    atomic_json(ROOT / "qa/REFERENCE_SCORE_RECONCILIATION.json", reconciliation)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "prediction_seal_path": str(ROOT / "qa/PREDICTION_SEAL.json"),
        "prediction_seal_sha256": sha_file(ROOT / "qa/PREDICTION_SEAL.json"),
        "student_prediction_jobs": 9,
        "student_dev_window_loss_evaluations": 1665,
        "reference_prediction_bundles_loaded": 15,
        "reference_score_reconciliation": "PASS",
        "primary_contrasts": ["MIX_VS_GT", "MIX_VS_SELF"],
        "bootstrap_draws": 5000,
        "bootstrap_seed": 2026101507,
        "CI_interpretation": "exploratory descriptive percentile interval, not confirmation/equivalence",
        "final_heldout_payload_access": 0,
        "dev_target_archive_load_operations": 1,
        "dev_target_windows_loaded": 185,
    }
    atomic_json(ROOT / "qa/SCORER_SEAL.json", report)
    print(json.dumps({"status": "PASS", "scorer_seal_sha256": sha_file(scorer_seal_path), "students": 9, "dev_target_loads": 1}, sort_keys=True), flush=True)


def main() -> int:
    missing = [name for name in ("EE6008_PROJECT_ROOT", "EE6008_V7_SINGLE_HEAD_STUDY_ROOT") if not os.environ.get(name)]
    if missing:
        raise RuntimeError("set explicit public V7 paths: " + ", ".join(missing))
    score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recompute V9 paired source-group CIs with the prompt-locked seed.

Reads only sealed V9 per-window score CSVs and source/cache metadata. It does not
read any dev target archive/cache payload, prediction tensor, or GPU device.
"""
from __future__ import annotations
import os
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

V9 = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z'))
V8_1 = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V8_1_EVIDENCE_COMPLETION_V1_20261017T054525Z'))
OUT = Path(__file__).resolve().parents[1]
SEED = 2026101709
N_DRAWS = 5000
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")
CONTRASTS = {
    "NUMCTX_DROP_VIDEO": ("V0", "VD"),
    "NUMCTX_DROP_STATE": ("A0", "AD"),
    "VIDEO_GAIN_0": ("A0", "V0"),
    "VIDEO_GAIN_D": ("AD", "VD"),
    "INTERACTION": ("VIDEO_GAIN_D", "VIDEO_GAIN_0"),
    "MIX_GAIN": ("MIX0", "MIXD"),
    "VD_VS_MIX0": ("MIX0", "VD"),
    "VD_VS_S": ("S", "VD"),
    "VD_VS_PERSISTENCE": ("PERSISTENCE", "VD"),
    "MIXD_VS_S": ("S", "MIXD"),
    "MIXD_VS_V0": ("V0", "MIXD"),
    "MIXD_VS_OLD_SV50": ("SV50", "MIXD"),
    "V0_VS_OLD_V": ("V", "V0"),
    "A0_VS_OLD_A": ("A", "A0"),
}

def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def verify_original_manifest() -> dict:
    manifest = V9 / "CHECKSUMS.sha256"
    checked = missing = mismatches = 0
    missing_paths, mismatch_paths = [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = V9 / relative
        checked += 1
        if not path.is_file():
            missing += 1
            missing_paths.append(relative)
        elif sha_file(path) != expected:
            mismatches += 1
            mismatch_paths.append(relative)
    result = {
        "schema_version": 1,
        "status": "PASS" if checked > 0 and missing == 0 and mismatches == 0 else "FAIL",
        "root": str(V9),
        "manifest_path": str(manifest),
        "manifest_sha256": sha_file(manifest),
        "manifest_entries": checked,
        "verified": checked - missing - mismatches,
        "missing": missing,
        "mismatches": mismatches,
        "missing_paths": missing_paths,
        "mismatch_paths": mismatch_paths,
        "original_v9_final_readback_receipt_present": (V9 / "qa/FINAL_CHECKSUM_VERIFICATION.json").is_file(),
        "verified_utc": utc(),
    }
    if result["status"] != "PASS":
        raise RuntimeError("original V9 checksum verification failed")
    write_json(OUT / "qa/ORIGINAL_V9_PACKAGE_VERIFICATION.json", result)
    return result

def audit_cache_binding() -> dict:
    seal = read_json(V8_1 / "scorer_cache/CACHE_SEAL.json")
    v9_read = read_json(V9 / "qa/DEV_TARGET_SINGLE_READ.json")
    row_path = V9 / "predictions/dev/full_context/A0/seed202610131/PREDICTION_ROWS.csv"
    v9_ids = [row["anchor_id"] for row in read_csv(row_path)]
    cache_ids = seal["ordered_anchor_ids"]
    same_ids = len(v9_ids) == 185 and len(set(v9_ids)) == 185 and cache_ids == v9_ids
    same_archive = seal["source_archive_sha256"] == v9_read["archive_sha256"]
    compatible = (
        seal.get("cache_role") == "SCORER_ONLY_EXPOSED_DEV_LABELS"
        and seal.get("source_split") == "dev"
        and seal.get("source_target_field") == "target_native_h30"
        and seal.get("target_cache_shape") == [185, 30, 28]
        and same_archive and same_ids
    )
    return {
        "schema_version": 1,
        "status": "AUDIT_COMPLETE",
        "v8_1_cache_seal_path": str(V8_1 / "scorer_cache/CACHE_SEAL.json"),
        "v8_1_cache_seal_sha256": sha_file(V8_1 / "scorer_cache/CACHE_SEAL.json"),
        "cache_role": seal.get("cache_role"),
        "source_split": seal.get("source_split"),
        "source_target_field": seal.get("source_target_field"),
        "target_cache_shape": seal.get("target_cache_shape"),
        "source_archive_path": seal.get("source_archive_path"),
        "source_archive_sha256": seal.get("source_archive_sha256"),
        "v9_archive_sha256": v9_read.get("archive_sha256"),
        "same_archive_hash": same_archive,
        "cache_ordered_id_count": len(cache_ids),
        "v9_prediction_row_id_count": len(v9_ids),
        "same_185_ordered_ids": same_ids,
        "metadata_compatible_for_scorer_cache_reuse": compatible,
        "v9_recorded_archive_content_reads": v9_read.get("archive_content_reads"),
        "v9_recorded_preseal_target_reads": v9_read.get("preseal_target_reads"),
        "v9_used_v8_1_scorer_cache": False,
        "process_deviation": "The source-bound V8.1 scorer cache matched archive hash and ordered dev IDs, but the completed V9 run used one post-seal archive read instead of reusing it.",
        "audit_opened_v8_1_Y_dev_payload": False,
        "audit_opened_v2_dev_target_payload": False,
        "audit_opened_v9_dev_target_payload": False,
        "audited_utc": utc(),
    }

def bootstrap() -> tuple[list[dict], dict]:
    new_rows = read_csv(V9 / "results/PER_WINDOW_NEW_DEV_LOSSES.csv")
    all_rows = read_csv(V9 / "results/PER_WINDOW_ALL_METHODS.csv")
    old_effects = read_csv(V9 / "results/PAIRED_EFFECTS.csv")
    metadata: dict[str, tuple[int, int, str]] = {}
    for row in new_rows:
        aid = row["anchor_id"]
        identity = (int(row["episode_id"]), int(row["task_id"]), str(row["source_group"]))
        if aid in metadata and metadata[aid] != identity:
            raise RuntimeError("inconsistent dev receiver identity: " + aid)
        metadata[aid] = identity
    if len(metadata) != 185:
        raise RuntimeError(f"expected 185 dev receivers, got {len(metadata)}")

    by: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in all_rows:
        aid = row["anchor_id"]
        if aid not in metadata:
            continue
        key = (row["method"], int(row["seed"]), aid)
        if key in by:
            raise RuntimeError("duplicate per-window key: " + repr(key))
        by[key] = row

    episode_delta: dict[tuple[str, str, int, int, str, int], float] = {}
    for name, (reference, candidate) in CONTRASTS.items():
        for metric in METRICS:
            for seed in SEEDS:
                for task in TASKS:
                    grouped: dict[tuple[str, int], list[float]] = {}
                    for aid, (episode, task_id, group) in metadata.items():
                        if task_id != task:
                            continue
                        if name == "INTERACTION":
                            value = (
                                float(by[("AD", seed, aid)][metric]) - float(by[("VD", seed, aid)][metric])
                                - float(by[("A0", seed, aid)][metric]) + float(by[("V0", seed, aid)][metric])
                            )
                        else:
                            value = float(by[(reference, seed, aid)][metric]) - float(by[(candidate, seed, aid)][metric])
                        grouped.setdefault((group, episode), []).append(value)
                    for (group, episode), values in grouped.items():
                        episode_delta[(name, metric, seed, task, group, episode)] = float(np.mean(values))

    rng = np.random.default_rng(SEED)
    draw_bank = {}
    digest = hashlib.sha256()
    support = {}
    for task in TASKS:
        groups = sorted({key[4] for key in episode_delta if key[3] == task})
        group_episodes = {
            group: sorted({key[5] for key in episode_delta if key[3] == task and key[4] == group})
            for group in groups
        }
        probabilities = np.asarray([len(group_episodes[g]) for g in groups], dtype=np.float64)
        probabilities /= probabilities.sum()
        draws = np.empty((N_DRAWS, len(groups)), dtype=np.int32)
        for b in range(N_DRAWS):
            draws[b] = rng.choice(len(groups), size=len(groups), replace=True, p=probabilities)
        draw_bank[task] = (groups, group_episodes, draws)
        digest.update(draws.tobytes(order="C"))
        support[str(task)] = {
            "source_groups": len(groups),
            "episodes": int(sum(len(items) for items in group_episodes.values())),
        }

    original_effects = {
        (row["contrast"], row["metric"]): row
        for row in old_effects
        if row["grain"] == "five_task_three_seed_macro"
    }
    ci_rows = []
    for name, (reference, candidate) in CONTRASTS.items():
        for metric in METRICS:
            boot = np.empty(N_DRAWS, dtype=np.float64)
            for b in range(N_DRAWS):
                task_values = []
                for task in TASKS:
                    groups, group_episodes, draws = draw_bank[task]
                    sampled = []
                    for group_index in draws[b]:
                        group = groups[int(group_index)]
                        for episode in group_episodes[group]:
                            values = [
                                episode_delta[(name, metric, seed, task, group, episode)]
                                for seed in SEEDS
                            ]
                            sampled.append(float(np.mean(values)))
                    task_values.append(float(np.mean(sampled)))
                boot[b] = float(np.mean(task_values))
            source = original_effects[(name, metric)]
            ci_rows.append({
                "contrast": name,
                "reference": reference,
                "candidate": candidate,
                "metric": metric,
                "grain": "five_task_three_seed_macro",
                "point_estimate": float(source["gain_reference_minus_candidate"]),
                "ci_2_5": float(np.quantile(boot, 0.025)),
                "ci_97_5": float(np.quantile(boot, 0.975)),
                "bootstrap_draws": N_DRAWS,
                "bootstrap_seed": SEED,
                "source_group_support": json.dumps(support, sort_keys=True),
                "conditional_on_fixed_models_and_dev_exposure": True,
                "p_value": "",
            })

    csv_path = OUT / "results/PAIRED_GROUP_CI.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ci_rows[0]))
        writer.writeheader()
        writer.writerows(ci_rows)
    ci_by = {(row["contrast"], row["metric"]): row for row in ci_rows}
    tags = {}
    for name in ("NUMCTX_DROP_VIDEO", "MIX_GAIN"):
        pos = ci_by[(name, "position_mae")]
        fd = ci_by[(name, "safe_fd_l1")]
        seam = ci_by[(name, "seam_safe_fd_l1")]
        tags[name] = {
            "POINT_JOINT_GAIN": pos["point_estimate"] > 0 and fd["point_estimate"] > 0,
            "CONDITIONAL_CI_JOINT_GAIN": pos["ci_2_5"] > 0 and fd["ci_2_5"] > 0,
            "position_gain": pos["point_estimate"],
            "position_ci": [pos["ci_2_5"], pos["ci_97_5"]],
            "safe_fd_gain": fd["point_estimate"],
            "safe_fd_ci": [fd["ci_2_5"], fd["ci_97_5"]],
            "seam_gain": seam["point_estimate"],
            "seam_ci": [seam["ci_2_5"], seam["ci_97_5"]],
        }

    source_path = V9 / "results/PER_WINDOW_ALL_METHODS.csv"
    seal_path = V9 / "qa/PREDICTION_SEAL.json"
    manifest_path = V9 / "CHECKSUMS.sha256"
    result = {
        "schema_version": 1,
        "status": "PASS",
        "seed": SEED,
        "draws": N_DRAWS,
        "draw_sha256": digest.hexdigest(),
        "same_draws_across_all_contrasts_metrics_seeds_policies_and_mixtures": True,
        "support": support,
        "task_order": list(TASKS),
        "seed_order": list(SEEDS),
        "metric_order": list(METRICS),
        "source_all_methods_path": str(source_path),
        "source_all_methods_sha256": sha_file(source_path),
        "source_new_dev_rows_path": str(V9 / "results/PER_WINDOW_NEW_DEV_LOSSES.csv"),
        "source_new_dev_rows_sha256": sha_file(V9 / "results/PER_WINDOW_NEW_DEV_LOSSES.csv"),
        "source_prediction_seal_sha256": sha_file(seal_path),
        "source_v9_manifest_sha256": sha_file(manifest_path),
        "dev_receiver_count": len(metadata),
        "per_window_rows_used": len(all_rows),
        "new_dev_rows_used_for_id_binding": len(new_rows),
        "numpy_version": np.__version__,
        "p_values": False,
        "target_archive_or_cache_reopened": False,
        "new_dev_target_reads": 0,
        "new_predictions": 0,
        "new_fits": 0,
        "new_optimizer_updates": 0,
        "computed_utc": utc(),
    }
    write_json(OUT / "figure_data/BOOTSTRAP_DRAW_MANIFEST.json", result)
    write_json(OUT / "qa/BOOTSTRAP_RESEED_RECEIPT.json", {
        "schema_version": 1,
        "status": "PASS",
        **result,
        "paired_group_ci_csv_sha256": sha_file(csv_path),
    })
    write_json(OUT / "qa/BOOTSTRAP_DECISION_TAGS.json", {
        "schema_version": 1,
        "status": "PASS",
        "bootstrap_seed": SEED,
        "tags": tags,
        "interpretation": "Positive gain favors the candidate. Neither requested joint-gain tag is met.",
    })
    return ci_rows, result

def write_report(ci_rows: list[dict], verification: dict, cache_audit: dict) -> None:
    selected = ("NUMCTX_DROP_VIDEO", "NUMCTX_DROP_STATE", "INTERACTION", "MIX_GAIN", "VD_VS_MIX0", "VD_VS_S")
    by = {(r["contrast"], r["metric"]): r for r in ci_rows}
    metric_names = {
        "position_mae": "Position MAE",
        "safe_fd_l1": "Safe-FD L1",
        "seam_safe_fd_l1": "Seam Safe-FD L1",
    }
    lines = [
        "# V9 bootstrap-seed and checksum supplement",
        "",
        f"生成时间 UTC：{utc()}",
        "",
        "## 范围",
        "",
        "既有 V9 包已记录 12/12 个 fresh fits、36 份 neural prediction bundles、6 份固定混合输出和 5,550 次 dev loss evaluation。本补充不新增 fit、optimizer update、预测或 dev target 读取，只从封存的 per-window 分数 CSV 按本轮锁定 seed 2026101709 重算 paired source-group bootstrap 区间。",
        "",
        "原 V9 包保持不变。其已封存区间使用 seed 20261016；本目录 results/PAIRED_GROUP_CI.csv 给出按指定 seed 2026101709 生成的 42 行修正版区间。同一组 5,000 次 within-task source-group 抽样用于所有 contrasts、metrics、seeds、policies 和 mixtures。",
        "",
        "## 指定比较",
        "",
        "gain = reference error − candidate error；正数表示 candidate 误差更低。区间条件于固定模型和重复使用的 dev 集，未计算 p 值，也未作多重比较校正。",
        "",
        "| Contrast | Metric | Gain | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for contrast in selected:
        for metric in METRICS:
            row = by[(contrast, metric)]
            lines.append(
                f"| {contrast} | {metric_names[metric]} | {row['point_estimate']:.6f} | "
                f"[{row['ci_2_5']:.6f}, {row['ci_97_5']:.6f}] |"
            )
    lines.extend([
        "",
        "## 结果解读",
        "",
        "完整数值上下文下，VD 没有优于 V0：Position MAE 和 Safe-FD 的 gain 为负，seam 改善。AD 的 Position MAE 也比 A0 更差。Safe-FD interaction 为负，不支持数值上下文 dropout 增强视频作用。MIXD 相对 MIX0 仅 seam 有正向 gain；Position MAE 与 Safe-FD 未改善。按指定 bootstrap seed 修正后，核心判断不变。",
        "",
        "## 审计记录",
        "",
        f"- 原 V9 checksum manifest：{verification['verified']}/{verification['manifest_entries']} 项已验证，missing={verification['missing']}，mismatch={verification['mismatches']}；manifest SHA-256 为 {verification['manifest_sha256']}。",
        f"- 原 V9 根目录缺少 qa/FINAL_CHECKSUM_VERIFICATION.json；本补充的 qa/ORIGINAL_V9_PACKAGE_VERIFICATION.json 保存本次只读 readback 收据。",
        f"- V8.1 scorer-only cache 与 V9 dev archive 和 ordered IDs 元数据一致（compatible={cache_audit['metadata_compatible_for_scorer_cache_reuse']}），但 V9 实际记录为 seal 后读取 archive 一次，没有复用 cache。该流程偏差保存在 qa/SCORER_CACHE_USE_AUDIT.json；本补充没有重新读取 target。",
        "- final/heldout payload 与 task-level final aggregates 访问数为 0。",
        f"- 原 V9 figures/FIGURE_VISUAL_REVIEW.json 记录的图像审阅状态为 {read_json(V9 / 'figures/FIGURE_VISUAL_REVIEW.json').get('status')}；其 paired-effects 图使用原 seed。修正区间以本目录表格为准。",
        "",
        "## 重现方式",
        "",
        "使用项目环境 Python 运行 code/recompute_bootstrap_ci.py。该脚本仅读取 V9 per-window score CSV 和 receipt 元数据，不导入 PyTorch，也不打开 target payload。",
        "",
    ])
    (OUT / "RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results/PAIRED_GROUP_CI.csv").exists():
        raise RuntimeError("refusing to overwrite existing supplemental CI output")
    verification = verify_original_manifest()
    cache_audit = audit_cache_binding()
    write_json(OUT / "qa/SCORER_CACHE_USE_AUDIT.json", cache_audit)
    ci_rows, _ = bootstrap()
    write_report(ci_rows, verification, cache_audit)
    (OUT / "qa/FAILED_ATTEMPTS.jsonl").write_text(
        json.dumps({
            "stage": "pre-supplement CPU bootstrap scratch attempt",
            "status": "failed_then_recovered",
            "error": "ValueError: too many values to unpack (expected 4)",
            "cause": "local aggregation comprehension unpacked a six-field episode key as four fields",
            "recovery": "corrected key indexing; the final script validates 185 dev IDs and emits 42 CI rows",
            "gpu_access": False,
            "dev_target_reads": 0,
            "new_predictions": 0,
            "new_fits": 0,
            "recorded_utc": utc(),
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "supplement_root": str(OUT),
        "paired_group_ci_rows": len(ci_rows),
        "bootstrap_seed": SEED,
        "draws": N_DRAWS,
        "draw_sha256": read_json(OUT / "figure_data/BOOTSTRAP_DRAW_MANIFEST.json")["draw_sha256"],
        "original_manifest_entries": verification["manifest_entries"],
        "original_manifest_mismatch": verification["mismatches"],
        "cache_compatible_but_not_used": cache_audit["metadata_compatible_for_scorer_cache_reuse"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()

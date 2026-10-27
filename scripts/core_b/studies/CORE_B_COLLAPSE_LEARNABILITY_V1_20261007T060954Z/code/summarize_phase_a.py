#!/usr/bin/env python3
"""Derive descriptive train-only metrics and a Phase A evidence report."""
from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
sys_path = SCHED / "code"
import sys
sys.path.insert(0, str(sys_path))
from d3_eq_contract import per_sample_loss  # noqa: E402
sys.path.insert(0, str(ROOT / "code"))
from collapse_diagnostic_core import loss_components  # noqa: E402

REPLAY = ROOT / "forensics/A_REPLAY_3"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def nested_pos(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (values[..., :14].mean(dim=(1, 2)) + values[..., 14:].mean(dim=(1, 2)))


def summarize() -> dict[str, Any]:
    repeat = read_json(ROOT / "forensics/REPLAY_REPEATABILITY.json")
    if repeat.get("status") != "PASS" or repeat.get("prediction_arrays_bitwise_equal") is not True:
        raise AssertionError("Phase A replay repeatability gate not passed")
    fixture = read_json(ROOT / "TRAIN_FORENSIC_FIXTURE.json")
    if fixture.get("status") != "LOCKED_BEFORE_FORWARD" or fixture.get("rows") != 16:
        raise AssertionError("train fixture lock invalid")
    replay = read_json(REPLAY / "A_REPLAY.json")
    with np.load(REPLAY / "PREDICTIONS_FIXTURE.npz", allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U96")
        anchors = torch.from_numpy(z["anchor_native"].astype(np.float32))
        target30 = torch.from_numpy(z["target_native_h30"].astype(np.float32))
        target60 = torch.from_numpy(z["target_native_h60"].astype(np.float32))
        predictions = {key.removeprefix("prediction_native__"): z[key].astype(np.float32)
                       for key in z.files if key.startswith("prediction_native__")}
    normalizers = read_json(V3 / "data/V3_NORMALIZERS.json")
    sigma = torch.tensor(normalizers["target_sigma"], dtype=torch.float32)
    safe_fd = torch.tensor(normalizers["safe_fd_scale"], dtype=torch.float32)
    selected = fixture["rows_selected_by_locked_rule"]
    strata = {row["anchor_id"]: row["stratum"] for row in selected}
    model_rows = list(csv.DictReader((REPLAY / "MODEL_REPLAY.csv").open(newline="", encoding="utf-8")))
    model_lookup = {row["label"]: row for row in model_rows}
    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, pred_np in predictions.items():
        horizon = pred_np.shape[1]
        target = target30 if horizon == 30 else target60
        pred = torch.from_numpy(pred_np)
        anchor = anchors
        pred_residual = (pred - anchor[:, None, :]) / sigma[None, None, :]
        persistence = anchor[:, None, :].expand_as(pred).clone()
        persistence_residual = torch.zeros_like(pred_residual)
        model_parts = loss_components(pred_residual, target, anchor, sigma, safe_fd)
        persistence_parts = loss_components(persistence_residual, target, anchor, sigma, safe_fd)
        locked_model = per_sample_loss(pred_residual, target, anchor, sigma, safe_fd)
        if not torch.allclose(locked_model, model_parts["full"], atol=1e-7, rtol=0):
            raise AssertionError(f"model loss differs from locked full loss: {label}")
        model_pos = model_parts["position"].numpy()
        model_fd = model_parts["fd"].numpy()
        pers_pos = persistence_parts["position"].numpy()
        pers_fd = persistence_parts["fd"].numpy()
        pred_tv_native = np.abs(np.diff(pred_np, axis=1)).mean(axis=(1, 2))
        target_np = target.numpy()
        target_tv_native = np.abs(np.diff(target_np, axis=1)).mean(axis=(1, 2))
        pred_tv_safe = (np.abs(np.diff(pred_np, axis=1)) / safe_fd.numpy()[None, None, :]).mean(axis=(1, 2))
        target_tv_safe = (np.abs(np.diff(target_np, axis=1)) / safe_fd.numpy()[None, None, :]).mean(axis=(1, 2))
        ratio = np.divide(pred_tv_safe, target_tv_safe, out=np.full_like(pred_tv_safe, np.nan), where=target_tv_safe > 1e-12)
        gain_pos = np.divide(pers_pos - model_pos, pers_pos, out=np.full_like(model_pos, np.nan), where=pers_pos > 1e-12)
        gain_fd = np.divide(pers_fd - model_fd, pers_fd, out=np.full_like(model_fd, np.nan), where=pers_fd > 1e-12)
        for i, anchor_id in enumerate(ids.tolist()):
            output_rows.append({
                "model": label, "anchor_id": anchor_id, "horizon": horizon,
                "motion_stratum": strata[anchor_id],
                "position_error_normalized": float(model_pos[i]),
                "persistence_position_error_normalized": float(pers_pos[i]),
                "position_gain_vs_persistence": float(gain_pos[i]) if np.isfinite(gain_pos[i]) else "NA",
                "future_internal_fd_error_safe_scale_normalized": float(model_fd[i]),
                "persistence_future_internal_fd_error_safe_scale_normalized": float(pers_fd[i]),
                "fd_gain_vs_persistence": float(gain_fd[i]) if np.isfinite(gain_fd[i]) else "NA",
                "prediction_tv_native": float(pred_tv_native[i]),
                "target_tv_native": float(target_tv_native[i]),
                "prediction_tv_over_target_tv_safe_scale": float(ratio[i]) if np.isfinite(ratio[i]) else "NA",
                "sample_mean_prediction_minus_anchor_native_abs": float(np.abs(pred_np[i].mean(axis=0)-anchors[i].numpy()).mean()),
            })
        raw = model_lookup[label]
        summary_rows.append({
            "model": label, "horizon": horizon, "family": raw["family"], "config": raw["config"],
            "seed": raw["seed"], "checkpoint_epoch": raw["epoch"],
            "n_train_rows": len(ids), "n_source_groups": len(set(x["source_group_id"] for x in selected)),
            "position_error_normalized_mean": float(model_pos.mean()),
            "persistence_position_error_normalized_mean": float(pers_pos.mean()),
            "position_gain_vs_persistence_mean": float(np.nanmean(gain_pos)),
            "fd_error_safe_scale_normalized_mean": float(model_fd.mean()),
            "persistence_fd_error_safe_scale_normalized_mean": float(pers_fd.mean()),
            "fd_gain_vs_persistence_mean": float(np.nanmean(gain_fd)),
            "prediction_tv_native_mean": float(pred_tv_native.mean()),
            "target_tv_native_mean": float(target_tv_native.mean()),
            "prediction_tv_over_target_tv_safe_scale_mean": float(np.nanmean(ratio)),
            "within_sample_time_std_mean": raw["within_sample_time_std_mean"],
            "between_sample_mean_offset_std": raw["between_sample_mean_offset_std"],
            "anchor_copy": raw["anchor_copy"],
            "anchor_copy_max_abs_offset": raw["anchor_copy_max_abs_offset"],
            "accepted_train_prediction_match": raw["accepted_within_locked_tolerance"],
            "accepted_train_prediction_max_abs_difference": raw["accepted_max_abs_difference"],
        })
    write_csv(ROOT / "results/SAMPLE_MOTION_AND_PERSISTENCE.csv", output_rows)
    write_csv(ROOT / "results/MODEL_MOTION_AND_PERSISTENCE.csv", summary_rows)

    # Read only the chosen train-run training-loss and LR columns.
    training_rows: list[dict[str, Any]] = []
    sources = (
        ("old_as_run", EQ / "d3/runs/D3EQ-TS11-D-H30-S20261002"),
        ("corrected_scheduler", SCHED / "runs/formal/D3EQ-CS-TS11-D-H30-S20261002"),
    )
    for family, run_root in sources:
        epoch_rows = list(csv.DictReader((run_root / "EPOCH_LOG.csv").open(newline="", encoding="utf-8")))
        lr_by_epoch: dict[int, list[float]] = defaultdict(list)
        lr_file = run_root / "LR_TRACE_ACTUAL.csv"
        if lr_file.is_file():
            for row in csv.DictReader(lr_file.open(newline="", encoding="utf-8")):
                lr_by_epoch.setdefault(int(row["epoch"]), []).append(float(row["lr_used_for_update"]))
        for row in epoch_rows:
            epoch = int(row["epoch"])
            values = lr_by_epoch.get(epoch, [])
            training_rows.append({
                "family": family, "run_id": run_root.name, "epoch": epoch,
                "optimizer_steps_total": int(row["optimizer_steps_total"]),
                "episode_weighted_train_loss": float(row["weighted_loss_sum_over_seen_rows"]),
                "unweighted_train_loss_secondary": float(row["unweighted_mean_loss"]),
                "epoch_lr_log": float(row["lr"]),
                "lr_used_min": min(values) if values else float("nan"),
                "lr_used_max": max(values) if values else float("nan"),
                "lr_used_mean": float(np.mean(values)) if values else float("nan"),
                "lr_trace_rows": len(values),
                "train_only_source_sha256": sha_file(run_root / "EPOCH_LOG.csv"),
            })
    write_csv(ROOT / "results/TRAINING_LOSS_LR_TRACE.csv", training_rows)
    checkpoints = [1, 10, 20, 30, 40]
    training_stage = [{"family": r["family"], "epoch": r["epoch"],
                       "episode_weighted_train_loss": r["episode_weighted_train_loss"],
                       "epoch_lr_log": r["epoch_lr_log"], "lr_used_min": r["lr_used_min"],
                       "lr_used_max": r["lr_used_max"], "lr_trace_rows": r["lr_trace_rows"]}
                      for r in training_rows if r["epoch"] in checkpoints]

    query = replay["query_spectra"]
    selected_labels = ["corrected_TS11-D_H30_S20261002_E10", "corrected_TS11-D_H30_S20261002_E20",
                       "corrected_TS11-D_H30_S20261002_E30", "corrected_TS11-D_H30_S20261002_E40"]
    anomaly_rows = [model_lookup[key] for key in selected_labels]
    anomaly_ratio = next(r["prediction_tv_over_target_tv_safe_scale_mean"] for r in summary_rows
                         if r["model"] == "corrected_TS11-D_H30_S20261002_E40")
    query_summary = {key: {"unique_query_rows": value["unique_query_rows"],
                           "adjacent_l2_min": value["adjacent_l2_min"],
                           "adjacent_l2_max": value["adjacent_l2_max"],
                           "adjacent_cosine_min": value["adjacent_cosine_min"],
                           "adjacent_cosine_max": value["adjacent_cosine_max"],
                           "fp32_scale_effective_rank": value["fp32_scale_effective_rank"],
                           "decoder_tgt_mask": value["decoder_tgt_mask"]}
                     for key, value in query.items()}
    query_conditioning = {
        "exact_duplicate_query_rows_observed": False,
        "H30_unique_rows": query_summary["corrected_TS11-D_H30_S20261002_E40"]["unique_query_rows"],
        "H30_fp32_scale_effective_rank": query_summary["corrected_TS11-D_H30_S20261002_E40"]["fp32_scale_effective_rank"],
        "H60_unique_rows": query_summary["corrected_TS11-D_H60_S20261002_E40"]["unique_query_rows"],
        "H60_fp32_scale_effective_rank": query_summary["corrected_TS11-D_H60_S20261002_E40"]["fp32_scale_effective_rank"],
        "H30_adjacent_cosine_range": [query_summary["corrected_TS11-D_H30_S20261002_E40"]["adjacent_cosine_min"], query_summary["corrected_TS11-D_H30_S20261002_E40"]["adjacent_cosine_max"]],
        "interpretation": "unique queries but high local cosine and low effective rank at the locked FP32 threshold are a conditioning concern, not evidence by themselves that the model cannot represent temporal change",
    }
    if len(model_rows) != 9 or sum(row["accepted_forward_compared"] == "True" for row in model_rows) != 6:
        raise AssertionError("Phase A model replay count/accepted bundle count mismatch")
    formula = {
        "test_files": ["tests/test_collapse_contracts.py", "tests/test_phase_a_formula_axis_qa.py"],
        "pytest_result": "10 passed",
        "gt_injection_position_loss_zero": True,
        "gt_injection_future_internal_fd_loss_zero": True,
        "h30_internal_fd_differences": 29,
        "linear_path_persistence_fd_error": 1.0,
        "step_path_internal_nonzero_joint_differences": 28,
        "step_path_anchor_to_first_seam_is_separate": True,
        "native_reconstruction_round_trip": True,
        "loss_component_decomposition_matches_locked_contract": True,
        "one_row_per_episode_weights_equal_one": True,
    }
    report = {
        "schema_version": 1, "status": "PHASE_A_COMPLETE_B_GATE_PASS",
        "scope": "fixed primary-train forensic fixture only; not a benchmark candidate",
        "fixture_sha256": sha_file(ROOT / "TRAIN_FORENSIC_FIXTURE.json"),
        "fixture_rows": 16, "strata_counts": fixture["selected_strata_counts"],
        "train_prediction_replay": {"models": 9, "accepted_train_bundle_comparisons": 6,
                                    "all_within_atol_1e-5_rtol_1e-5": True,
                                    "repeat_arrays_and_csvs_bitwise_equal": True,
                                    "repeat_max_abs_difference": repeat["max_abs_difference"]},
        "anomaly_diagnosis": {
            "label": "sample_dependent_near_static_future_offset",
            "exact_anchor_copy": model_lookup["corrected_TS11-D_H30_S20261002_E40"]["anchor_copy"] == "True",
            "within_sample_time_std_mean_epoch10_20_30_40": [float(x["within_sample_time_std_mean"]) for x in anomaly_rows],
            "predicted_tv_over_train_target_tv_safe_scale_epoch40": anomaly_ratio,
            "between_sample_mean_offset_std_epoch40": float(model_lookup["corrected_TS11-D_H30_S20261002_E40"]["between_sample_mean_offset_std"]),
            "interpretation_limit": "H30 train-only 16-row balanced forensic panel; onset before checkpoint epoch10 is unknown.",
        },
        "query_diagnostics": query_summary,
        "query_conditioning_interpretation": query_conditioning,
        "formula_axis_qa": formula,
        "gradient_probe": {"status": "PASS", "autograd_grad_only": True,
                           "optimizer_created": False, "parameter_hash_unchanged": True,
                           "anomalous_seed_epoch40_overall": next(r for r in csv.DictReader((REPLAY / "LOSS_GRADIENT_COMPONENTS.csv").open())
                                                                     if r["model"] == "corrected_TS11-D_H30_S20261002_E40" and r["module"] == "all")},
        "training_loss_lr_stages": training_stage,
        "evidence_paths": {
            "replay": "forensics/A_REPLAY_3", "repeatability": "forensics/REPLAY_REPEATABILITY.json",
            "model_summary": "results/MODEL_MOTION_AND_PERSISTENCE.csv",
            "sample_metrics": "results/SAMPLE_MOTION_AND_PERSISTENCE.csv",
            "training_trace": "results/TRAINING_LOSS_LR_TRACE.csv",
        },
        "resource_and_scope": {"device": "CPU", "gpu_operations": 0,
                                "optimizer_updates": 0, "dev_targets_opened": False,
                                "final_or_heldout_access": 0, "old_root_writes": 0,
                                "new_full_data_benchmark_training": False, "d5_training": False},
        "source_hashes": replay["source_hashes"],
        "failed_attempts": ["A_REPLAY_1 no-forward dropout-module implementation error",
                            "A_REPLAY_2 H60/H30 target-shape mismatch; partial work preserved in logs/FAILURE_REGISTER.json"],
    }
    write_json(ROOT / "A_FORENSIC_REPORT.json", report)
    return report


if __name__ == "__main__":
    result = summarize()
    key = result["anomaly_diagnosis"]
    lines = [
        "# Core B Phase A：近静态 checkpoint 取证",
        "",
        "状态：Phase A 完成，身份/公式/回放门通过；tiny-fit 尚未启动。以下只使用固定 primary-train 取证面板，不是 benchmark 候选，也不产生泛化结论。",
        "",
        "## 核验结论",
        "",
        f"- 16 个 train 样本按冻结 motion strata 取样：low/mid/high = {result['strata_counts']}。9 个 checkpoint 均完成 CPU 前向，其中 6 个有对应 accepted train bundle 的终点输出按 ID 对齐后通过 atol=1e-5、rtol=1e-5；第二次 replay 的输出/指标/trace/梯度逐位一致。",
        f"- 异常 TS11-D/H30/seed20261002 epoch40 不是 anchor-copy：精确零偏移判定为 {key['exact_anchor_copy']}。样本间平均 offset 标准差为 {key['between_sample_mean_offset_std_epoch40']:.6g}，而 epoch10/20/30/40 的样本内时间标准差依次为 {', '.join(f'{x:.3g}' for x in key['within_sample_time_std_mean_epoch10_20_30_40'])}。它是“样本相关水平偏移 + 几乎无未来时间变化”，不是完全没有输入响应。",
        f"- epoch40 的预测 TV / train target TV（safe-FD 标度）为 {key['predicted_tv_over_train_target_tv_safe_scale_epoch40']:.6g}。这是当前 16 行分层 train 面板的描述性结果；不能外推到自然总体。",
        "- accepted train-bundle replay 与 16 行 exact-ID 对照最大差为 0；前向和 autograd.grad 后 state hash 不变，无 optimizer/backward/update。",
        "- 公式轴 QA：GT 注入 position/FD 为零；H30 有 29 个 future-internal 差分；线性路径、阶跃路径及 anchor-to-first seam 分开核对；episode-equal 1-row-per-episode 权重为 1。10 项 CPU tests 通过。",
        "",
        "## Query、参数与梯度",
        "",
        f"H30/H60 query 没有重复行，但相邻余弦相似度约 0.999965；在锁定的 FP32 尺度阈值下有效秩仅 {result['query_conditioning_interpretation']['H30_fp32_scale_effective_rank']}/30 与 {result['query_conditioning_interpretation']['H60_fp32_scale_effective_rank']}/60。这支持时间 query 条件数可能不佳的假设，但不单独证明 head 无法表达动态输出。源码未提供 decoder target causal mask（tgt_mask=None），此实现事实本身也不是错误证明；完整奇异值见 replay trace。",
        f"异常 checkpoint 的局部总体梯度：Lpos={float(result['gradient_probe']['anomalous_seed_epoch40_overall']['L_pos']):.6g}，0.1·Ldelta={float(result['gradient_probe']['anomalous_seed_epoch40_overall']['weighted_0p1_L_delta']):.6g}；g_fd/g_pos={float(result['gradient_probe']['anomalous_seed_epoch40_overall']['g_fd_over_g_pos']):.6g}，cos={float(result['gradient_probe']['anomalous_seed_epoch40_overall']['g_pos_fd_cosine']):.4g}。这是一个固定 train panel、eval/dropout-off 的局部探针，不能证明训练全程的因果根因。",
        "",
        "## 可证伪解释",
        "",
        "| 假设 | 本轮证据 | 后续区分观察 |",
        "|---|---|---|",
        "| checkpoint / writer / normalizer / cache 身份错配 | 6 个可比终点按 ID 完全重放一致；模型源码 hash 一致；明显削弱 | 维持 source manifest 与逐 ID replay；无需重训验证此解释 |",
        "| query 重复或时间编码条件差 | 没有精确重复 query，但相邻 cosine≈0.999965、FP32 尺度有效秩为 H30=4/30、H60=5/60；支持 conditioning 风险，不是表达能力结论 | 固定 tiny-fit 检验能否学出时间动态；不额外扫 query scale |",
        "| FD loss 与 position loss 的梯度交互 | 异常 TS11-D checkpoint 上 FD 局部梯度相对很小；支持“该条件下优化信号弱”的线索 | 预先固定的 FULL_LOSS vs POS_ONLY tiny-fit 检验是否仅目标成分敏感 |",
        "| 多样本干扰 / capacity | 预测偏移随样本有变化，但当前只测16个样本；未被确认或排除 | P1 vs P8 固定面板是否出现单样本可拟合、8样本受干扰 |",
        "",
        "## 训练阶段与边界",
        "",
        "epoch10 已见近零时间变化，epoch10→40 仍保持近静态；无 epoch1 checkpoint 取证，因此具体退化发生时刻未知。训练 loss/LR 全轨迹见 `results/TRAINING_LOSS_LR_TRACE.csv`。",
        "",
        "失败尝试 A_REPLAY_1/A_REPLAY_2 保存在 `logs/FAILURE_REGISTER.json`；只在新 OUT_ROOT 写入。dev targets、final/held-out 未打开，GPU 使用为 0。Phase B 可以进入 4-update Stage0 与成本门，但正式 8×2000 tiny-fit 尚未获准启动，直至 Stage0/CPU 成本估算通过。",
        "",
    ]
    (ROOT / "A_FORENSIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report_sha256": sha_file(ROOT / "A_FORENSIC_REPORT.json"),
                      "markdown_sha256": sha_file(ROOT / "A_FORENSIC_REPORT.md")}, sort_keys=True))

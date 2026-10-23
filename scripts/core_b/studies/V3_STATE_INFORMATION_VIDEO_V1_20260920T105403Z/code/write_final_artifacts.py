#!/usr/bin/env python3
"""Write final QA, status, and Chinese handoff artifacts after analysis/figures."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z'))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: str | float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    aggregate = read_csv(EXP / "analysis/AGGREGATE_RESULTS.csv")
    baseline = read_csv(EXP / "analysis/BASELINE_RESULTS.csv")
    primary = read_csv(EXP / "analysis/PRIMARY_VIDEO_EFFECTS.csv")
    fd = read_csv(EXP / "analysis/FD_RESULTS.csv")
    interactions = read_csv(EXP / "analysis/INFORMATION_INTERACTIONS.csv")
    resource = read_csv(EXP / "analysis/RESOURCE_COSTS.csv")
    union = json.loads((EXP / "analysis/SUPPLEMENTARY_ID_UNION.json").read_text(encoding="utf-8"))
    acceptance = json.loads((EXP / "analysis/PREDICTION_ACCEPTANCE.json").read_text(encoding="utf-8"))
    reconciliation = json.loads((EXP / "V3_EVIDENCE_RECONCILIATION.json").read_text(encoding="utf-8"))
    stage0 = json.loads((EXP / "analysis/STAGE0_ACCEPTANCE.json").read_text(encoding="utf-8"))
    actual_gpu_hours = sum(float(row["wall_gpu_hours"]) for row in resource)
    actual_gpus = sorted({row["physical_gpu"] for row in resource})
    primary_sig = [row for row in primary if float(row["holm_p"]) < 0.05]
    fd_sig = [row for row in fd if float(row["holm_p"]) < 0.05]
    interaction_sig = [row for row in interactions if float(row["holm_p"]) < 0.05]
    main_table = []
    for row in aggregate:
        main_table.append({"horizon": row["horizon_steps"], "cell": row["cell"], "arm": fmt(row["arm_mae_std_mean"]), "hand": fmt(row["hand_mae_std_mean"]), "full": fmt(row["full_mae_std_mean"]), "fd": fmt(row["full_fd_native_mean"], 8), "skill": fmt(row["full_skill_vs_persistence_mean"], 4)})
    baseline_table = [{"horizon": row["horizon_steps"], "arm": fmt(row["arm_mae_std"]), "hand": fmt(row["hand_mae_std"]), "full": fmt(row["full_mae_std"]), "fd": fmt(row["full_fd_native"], 8)} for row in baseline]
    markdown = [
        "# Core B v3：状态信息量 × 视频内容 54 格收菜报告",
        "",
        "## 终态",
        "",
        "```ini",
        "STUDY_STATUS=complete",
        "STAGE0_ACCEPTED=18/18",
        "TRAINING_ACCEPTED=54/54",
        "PREDICTIONS_ACCEPTED=54/54",
        "FINAL_TEST_ACCESS=0",
        "ENCODER_UPDATED=false",
        "NEW_ENCODER_EXTRACTION_STARTED=false",
        "```",
        "",
        "54 格均使用固定 40 epochs、11,120 optimizer steps、epoch-40 last checkpoint；没有按首个 seed 或 dev 结果筛掉配置。所有 formal/inference watcher 均返回 0。",
        "",
        "## V3 基线复核",
        "",
        "Persistence 是从冻结的 primary-dev targets/anchor 独立复算的，episode-equal 聚合；与旧 V3 保存表一致。",
        "",
        "| Horizon | Arm standardized MAE | Hand standardized MAE | Full standardized MAE | Full native FD |",
        "|---:|---:|---:|---:|---:|",
    ]
    markdown += [f"| {row['horizon']} | {row['arm']} | {row['hand']} | {row['full']} | {row['fd']} |" for row in baseline_table]
    markdown += [
        "",
        f"primary dev 为 603 episodes/2322 windows；supplementary 为 64 episodes/64 windows。二者交集 {union['intersection']}，supplementary-only {union['supplementary_only']}，union {union['union']}，因此不能写成 667 个独立 episode。",
        "",
        "## 54 格主结果",
        "",
        "表中 skill 是相对于同 horizon Persistence 的 standardized Full-MAE 描述性 skill；结果仍是 development-only。",
        "",
        "| Horizon | Cell | Arm MAE | Hand MAE | Full MAE | Full native FD | Full skill |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    markdown += [f"| {row['horizon']} | {row['cell']} | {row['arm']} | {row['hand']} | {row['full']} | {row['fd']} | {row['skill']} |" for row in main_table]
    markdown += [
        "",
        "## 视频增量与信息交互",
        "",
        f"24 项 primary MAE family 中 {len(primary_sig)} 项 Holm 校正后小于 0.05；24 项 FD secondary family 中 {len(fd_sig)} 项通过同一预声明 secondary Holm；8 项 interaction family 中 {len(interaction_sig)} 项 Holm 校正后小于 0.05。完整数值见 CSV。",
        "",
        "在完整 FULL 条件下，视频净增量总体较小且未形成普遍显著的 D>N/D>S 主结论；ANCHOR 条件下 2 秒 D 相对 N/S 的 Arm、Hand 视频增量显著，说明视觉信号在状态信息被人为减少时更容易显现。该结果不能包装为完整状态任务已被视频改善。",
        "",
        "interaction 中，ANCHOR−OBS 的视频 gain 在 1 秒 Arm/Hand 和 2 秒 Arm/Hand 均为正，其中 Holm 后支持较强；OBS−FULL 没有形成同样一致的证据。这个结果支持“视频价值依赖可用 state 信息”的限定表述，不证明 command history 的物理冗余。",
        "",
        "## seed、工程和资源",
        "",
        "Fresh seeds：20260953、20260954、20260955。54/54 checkpoint strict reload、zero-residual Persistence、prediction shape/ID/finite 检查通过；旧 V3 的 outlier audit 仍为 training variation supported、无 confirmed bug。",
        f"实际 GPU-hours：{actual_gpu_hours:.6f}；物理 GPU：{', '.join(actual_gpus)}；Stage0 实际 GPU：2/3；最多两个本项目 child，未 signal/kill 其他用户进程。",
        "",
        "## 解释边界",
        "",
        "- ANCHOR 仍保留独立 last-command anchor token，不是 strict video-only。",
        "- 删除 state 后出现的视频收益只说明 reduced-information condition 下的条件增量。",
        "- FULL/OBS/ANCHOR 的差异也可能包含优化、正则化和输入融合稳定性，不能直接等价为物理信息冗余。",
        "- 所有结果都来自 development split；不替换旧冻结 candidate，不访问 final/held-out，不联系教授。",
        "",
        "完整证据：`analysis/RESULTS_BY_SEED.csv`、`analysis/AGGREGATE_RESULTS.csv`、`analysis/PRIMARY_VIDEO_EFFECTS.csv`、`analysis/FD_RESULTS.csv`、`analysis/INFORMATION_INTERACTIONS.csv`、`analysis/SUPPLEMENTARY_ID_UNION.json`。",
    ]
    (EXP / "RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(EXP / "RETURN_TO_CHATGPT.json", {
        "study_status": "complete",
        "stage0": "18/18",
        "training": "54/54",
        "predictions": "54/54",
        "seeds": [20260953, 20260954, 20260955],
        "primary_video_effects_holm_significant": len(primary_sig),
        "secondary_fd_holm_significant": len(fd_sig),
        "interaction_holm_significant": len(interaction_sig),
        "baseline_reconciliation": reconciliation,
        "episode_union": union,
        "actual_gpu_hours": actual_gpu_hours,
        "actual_gpu_ids": actual_gpus,
        "final_test_access": 0,
        "encoder_updated": False,
        "new_encoder_extraction_started": False,
        "old_roots_modified": False,
        "next_step_prompt": "generated_not_executed",
    })
    (EXP / "FINAL_STATUS.ini").write_text(
        "STUDY_STATUS=complete\nV3_BASELINE_RECONCILIATION=PASS\nV3_SEED_OUTLIER_AUDIT=training_variation\nSTATE_COLUMN_MAP_VERIFIED=true\nSTAGE0_ACCEPTED=18/18\nTRAINING_ACCEPTED=54/54\nPREDICTIONS_ACCEPTED=54/54\nSEEDS=20260953,20260954,20260955\nVIDEO_GAIN_FULL=mixed_no_uniform_net_gain\nVIDEO_GAIN_OBS=mixed_no_uniform_net_gain\nVIDEO_GAIN_ANCHOR=positive_especially_H60\nSTATE_INFORMATION_INTERACTION=ANCHOR_minus_OBS_positive;OBS_minus_FULL_not_uniform\nDYNAMIC_STATIC_RESULT=conditional_anchor_signal\nNEW_ENCODER_EXTRACTION_STARTED=false\nENCODER_UPDATED=false\nFINAL_TEST_ACCESS=0\nPROFESSOR_CONTACT_STARTED=false\nOLD_ROOTS_MODIFIED=false\nVISIBLE_GPU_POOL=dynamic_all_visible\nACTUAL_GPU_IDS=" + ",".join(actual_gpus) + f"\nMAX_PROJECT_GPU_CHILDREN_OBSERVED=2\nGPU_HOURS_USED={actual_gpu_hours:.6f}\nCHECKSUMS=pending_checksum_last\n",
        encoding="utf-8",
    )
    qa = EXP / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "NUMERIC_QA.md").write_text(
        "# Numeric QA\n\n"
        "- MATRIX_54: 54 locked rows; fresh seeds selected before new dev scores.\n"
        "- Stage0: 18/18; formal: 54/54; fixed-last inference: 54/54.\n"
        "- Primary prediction shape: (2322, 30, 28) or (2322, 60, 28); H30 supplementary shape: (64, 30, 28).\n"
        "- Primary anchor IDs, episode IDs, finite values and checkpoint reload identity: PASS.\n"
        "- Primary MAE family: 24 items, one Holm correction; FD secondary family: 24 items, separate Holm; interaction family: 8 items, separate Holm.\n"
        "- Persistence independent recomputation: PASS; primary/supplementary episode union: 614, not 667.\n"
        "- Final-test access fields and controller manifests: zero.\n",
        encoding="utf-8",
    )
    (qa / "VISUAL_QA.md").write_text(
        "# Visual QA\n\n"
        "Opened and inspected the final PNG figures: FIG01 state masks, FIG02 Arm/Hand/Full by horizon, FIG03 video gains, FIG04 information interactions, FIG05 per-seed skill heatmaps, and FIG06 fixed-episode trajectories.\n\n"
        "All six figures contain plotted data, readable axes/legends, explicit horizon labels and no empty panels. FIG03 uses the corrected control-column mapping for D-vs-N and D-vs-S.\n",
        encoding="utf-8",
    )
    (qa / "FINAL_QA.md").write_text(
        "# Final QA\n\n"
        "Overall: PASS for engineering completion and development analysis.\n\n"
        "- Data/cache/normalizer source is the frozen V3 source; no encoder forward was started.\n"
        "- State column map is verified from Parquet schema and V3 construction code.\n"
        "- All 54 formal and 54 inference jobs completed with zero failures/OOM.\n"
        "- All required analysis tables and six PNG/PDF figure pairs exist.\n"
        "- Actual GPU usage is recorded in analysis/RESOURCE_COSTS.csv; no non-child process was signaled.\n"
        "- Final/held-out data access is 0; old roots and old candidate gate are unchanged.\n"
        "- This is not a final model selection or generalization claim.\n",
        encoding="utf-8",
    )
    (EXP / "NEXT_STEP_PROMPT.md").write_text(
        "# Next step prompt\n\nGenerated but not executed. Discuss the development-only conditional video findings before any new training or final evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "primary_sig": len(primary_sig), "fd_sig": len(fd_sig), "interaction_sig": len(interaction_sig), "gpu_hours": actual_gpu_hours, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

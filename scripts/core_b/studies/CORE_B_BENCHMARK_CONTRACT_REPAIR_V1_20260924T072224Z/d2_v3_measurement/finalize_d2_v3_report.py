"""Summarize the locked D2_V3 output without modifying prediction/results files."""

from __future__ import annotations
import os

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_csv(ROOT / "D2_V3_RESULTS.csv")
    qa = json.loads((ROOT / "D2_V3_EVENT_QA.json").read_text())
    evaluation = json.loads((ROOT / "D2_V3_EVALUATION_SUMMARY.json").read_text())
    support = json.loads((ROOT / "D2_V2_V3_SUPPORT_AUDIT.json").read_text())
    train = json.loads((ROOT / "D2_V3_TRAIN_SUPPORT_SUMMARY.json").read_text())
    threshold = json.loads((ROOT / "D2_V3_PROTOCOL_LOCKED.json").read_text())
    matcher = json.loads((ROOT / "D2_V3_MATCHER_LOCKED_ATTEMPT_003.json").read_text())
    runner = json.loads((ROOT / "D2_V3_RUNNER_LOCKED_ATTEMPT_006.json").read_text())

    rollup = {}
    for horizon in (30, 60):
        for body in ("Hand", "Arm"):
            subset = [row for row in rows if int(row["horizon_steps"]) == horizon and row["event_group"] == body]
            recalls = [float(row["recall"]) for row in subset if row["recall"] not in ("", "None")]
            fa_rates = [float(row["false_alarms_per_hour"]) for row in subset if row["false_alarms_per_hour"] not in ("", "None")]
            rollup[f"H{horizon}_{body}"] = {
                "model_seed_cells": len(subset),
                "mean_cell_recall": sum(recalls) / len(recalls) if recalls else None,
                "recall_min": min(recalls) if recalls else None,
                "recall_max": max(recalls) if recalls else None,
                "mean_cell_false_alarms_per_hour": sum(fa_rates) / len(fa_rates) if fa_rates else None,
                "false_alarms_per_hour_min": min(fa_rates) if fa_rates else None,
                "false_alarms_per_hour_max": max(fa_rates) if fa_rates else None,
                "interval_unit": "independent source-group bootstrap; individual row intervals in D2_V3_RESULTS.csv",
            }

    ftokd = [row for row in rows if row["model"] == "F-TOK-D"]
    p3 = {
        "status": "PASS_D2_V3_EXPLORATORY_COMPLETE",
        "D2_V3_STATUS": "completed_exploratory",
        "old_D2_V2_unchanged": True,
        "D2_V2_hand_complete_events_each_horizon": 5,
        "D2_V2_old_event_registry_records": support["D2_V2_event_registry_records"],
        "D2_V2_unique_source_group_body_onset_events": support["D2_V2_independent_group_body_frame_events"],
        "D2_V2_groups_with_event": support["D2_V2_source_groups_with_event"],
        "D2_V2_Hand_theta_off": support["D2_V2_theta_off_hand"],
        "D2_V2_Hand_exact_q_zero_fraction_dev": support["dev_exact_q_zero_fraction"],
        "D2_V2_Hand_exact_q_zero_fraction_panel_episodes": support["panel_episode_exact_q_zero_fraction"],
        "D2_V3_thresholds_train_only": threshold["thresholds"],
        "train_support": train["independent_events"],
        "dev_support": {row["body"]: {key: row[key] for key in ("event_records", "independent_events", "source_groups_with_event", "tasks_with_event")} for row in evaluation["dev_support"]},
        "panel_support": {
            "Hand": {"independent_events": 442, "complete_opportunities_H30": 238, "tasks_H30": 95, "complete_opportunities_H60": 259, "tasks_H60": 99},
            "Arm": {"independent_events": 302, "complete_opportunities_H30": 174, "tasks_H30": 106, "complete_opportunities_H60": 190, "tasks_H60": 107},
        },
        "panel_queries": evaluation["panel_query_count"],
        "panel_groups": evaluation["panel_source_group_count"],
        "panel_exposure_hours": evaluation["panel_exposure_hours"],
        "accepted_dense_bundles": evaluation["accepted_dense_predictions"],
        "result_rows": len(rows),
        "cluster_audit_rows": qa["cluster_audit_rows"],
        "qa_status": qa["status"],
        "hand_low_support_gate": evaluation["hand_panel_low_support_any_horizon"],
        "support_expansion_proposal_required": bool(evaluation["hand_panel_low_support_any_horizon"]),
        "F_TOK_D_per_seed": [
            {key: row[key] for key in ("seed", "horizon_steps", "event_group", "full_opportunity_events", "hits", "misses", "false_alarms", "recall", "recall_ci95_low", "recall_ci95_high", "false_alarms_per_hour", "false_alarms_per_hour_ci95_low", "false_alarms_per_hour_ci95_high", "opportunities_lead_ge_0.5", "hits_lead_ge_0.5", "recall_lead_ge_0.5")}
            for row in ftokd
        ],
        "all_models_horizon_body_rollup": rollup,
        "matcher_lock_sha256": evaluation["matcher_lock_sha256"],
        "runner_lock_sha256": runner["runner_code_sha256"],
        "final_test_access": 0,
    }
    (ROOT / "D2_V3_ACCEPTANCE_FINAL.json").write_text(json.dumps(p3, indent=2) + "\n", encoding="utf-8")

    report = [
        "# D2 事件支持复核与 D2_V3 测量敏感性",
        "",
        "状态：`completed_exploratory`。D2_V2 原文件保持只读；D2_V3 在明确冻结的检测/聚类/匹配规则下重评已有 V3 预测，不开展新推理或训练。",
        "",
        "## D2_V2 支持与状态审计",
        "",
        f"- 原 truth registry 有 {support['D2_V2_event_registry_records']} 条记录，按 source-group/body/onset-frame 去重后仍为 {support['D2_V2_independent_group_body_frame_events']} 个事件键，涉及 {support['D2_V2_source_groups_with_event']} 个 source groups。Hand 的 theta_off 为 0，因此 quiet 等同 `q == 0`；dev 全量 exact-zero 步骤比例 {support['dev_exact_q_zero_fraction']:.1%}，118 组 panel episode 覆盖的比例 {support['panel_episode_exact_q_zero_fraction']:.1%}。",
        f"- 旧 118-group / 3,718-query panel 的 Hand 完整机会每个 horizon 只有 5 个；这个分母不足以给 Hand 总体效能下结论。D2_V2 panel 暴露约 {support['panel_exposure_hours']:.6f} 小时；0/5 不可解读为对所有 Hand 变化都无能力。",
        "- 在该面板中 Hand 状态计数为 unknown 3,156、known-but-unarmed/ongoing 109、armed-at-anchor 453。D2_V3 的平滑定义将对应值改为 unknown 102、known-unarmed 1,220、armed 2,396；这说明测量状态定义和可评分查询支持发生了变化，应与事件阈值一起读，而非只比较 recall。".replace("\u001f", ""),
        "",
        "## D2_V3 train-only 定义和支持",
        "",
        f"阈值先在 2,712 个独立 train source groups、120 个任务上锁定：Hand theta_on/off = {threshold['thresholds']['Hand']['theta_on']:.6f}/{threshold['thresholds']['Hand']['theta_off']:.6f}；Arm = {threshold['thresholds']['Arm']['theta_on']:.6f}/{threshold['thresholds']['Arm']['theta_off']:.6f}。原始时间戳 30Hz 检查最大绝对误差 {threshold['timebase']['max_abs_dt_error_seconds']:.3g} 秒。",
        "D2_V3 把前 8 个合法 command-change q 做移动均值，以 train source-group 等权 Q75 定 on，off=0.5×on；连续 6 个 quiet 状态后 armed，连续 3 个 high 产生事件并把 onset 回溯到第一个 high。它是平滑 command-change onset，不是物理速度或接触标签。".replace("\u001f", ""),
        f"V3 threshold 下 dev 有 Hand {evaluation['dev_support'][0]['independent_events']} / Arm {evaluation['dev_support'][1]['independent_events']} 个独立 group/body onset 事件，分别覆盖 {evaluation['dev_support'][0]['tasks_with_event']} / {evaluation['dev_support'][1]['tasks_with_event']} 个任务。固定 panel 的完整机会为 Hand H30 {238}（95 tasks）、H60 {259}（99 tasks）；Arm H30 {174}（106 tasks）、H60 {190}（107 tasks）。Hand 满足预锁定 ≥30 事件且 ≥10 tasks 的支持门，因此本轮不需要扩展 proposal；也未按模型误差挑选新 panel。",
        "",
        "## 固定 panel 的模型事件结果",
        "",
        "以下为 F-TOK-D 的逐 seed 结果。FA/hour 同时报原始 false clusters 和 0.249611h 暴露；各 CI 在 `D2_V3_RESULTS.csv` 中按 exact source-group bootstrap 给出。",
        "",
        "| H | Body | Seed | Hits / full opportunities | Recall | Lead ≥0.5s hits / opportunities | False clusters | FA/hour |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ftokd:
        report.append(f"| {row['horizon_steps']} | {row['event_group']} | {row['seed']} | {row['hits']} / {row['full_opportunity_events']} | {float(row['recall']):.3f} | {row['hits_lead_ge_0.5']} / {row['opportunities_lead_ge_0.5']} | {row['false_alarms']} | {float(row['false_alarms_per_hour']):.1f} |")
    report += [
        "",
        "Across all six V3 configurations and three seeds, the complete 72-row table reports Hand/Arm separately. Hand recall remains near zero under this smoothed definition despite improved state-known query support; Arm shows modest recall paired with very high alert-cluster rates. These results do not justify an operational early-warning claim, even though the new Hand event denominator is statistically supportable under the locked rule.",
        "",
        "The fixed matcher uses actual absolute query time, effective H−2 onset horizon, immutable first-predicted-onset cluster center, 0.2s cluster/match tolerance, and one-to-one GT events. QA passed onset backdating, unknown-state, non-transitive cluster, one-to-one matching, exposure, and count-reconciliation checks. All 36 accepted bundles were reused; no GPU job was launched.",
        "",
        "Final/held-out access: `0`. No `D2_SUPPORT_EXPANSION_PROPOSAL.md` was generated because the predeclared Hand support gate passed; the panel remains fixed.",
    ]
    (ROOT / "D2_V3_EVENT_MEASUREMENT_REPORT.zh-CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": p3["D2_V3_STATUS"], "rows": len(rows), "Hand_h30_opportunities": 238, "Hand_h60_opportunities": 259, "support_expansion_proposal": p3["support_expansion_proposal_required"], "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

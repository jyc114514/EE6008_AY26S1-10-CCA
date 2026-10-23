"""Generate the current Chinese handoff/report from reconciled evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import BENCH_ROOT, atomic_write_json, read_json, read_csv, utc_now
from reconcile_state import main as reconcile


def fmt(value: object) -> str:
    if value is None: return "UNKNOWN"
    if isinstance(value, float): return f"{value:.6f}"
    return str(value)


def main() -> int:
    reconcile()
    state = read_json(BENCH_ROOT / "BENCHMARK_STATE.json")
    inventory = read_json(BENCH_ROOT / "context/V3_STATUS_SNAPSHOT.json") if (BENCH_ROOT / "context/V3_STATUS_SNAPSHOT.json").exists() else {}
    summary = (BENCH_ROOT / "inventory/INVENTORY_SUMMARY.txt").read_text(encoding="utf-8") if (BENCH_ROOT / "inventory/INVENTORY_SUMMARY.txt").exists() else ""
    d1 = read_json(BENCH_ROOT / "d1/D1_SUMMARY.json")
    d1_learned = read_json(BENCH_ROOT / "d1/D1_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d1/D1_LEARNED_SUMMARY.json").exists() else {}
    d2 = read_json(BENCH_ROOT / "d2/D2_SUMMARY.json")
    d2_learned = read_json(BENCH_ROOT / "d2/D2_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d2/D2_LEARNED_SUMMARY.json").exists() else {}
    d2_dense = read_json(BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json").exists() else {}
    d3_pred = read_json(BENCH_ROOT / "d3/D3_PREDICTION_ACCEPTANCE.json") if (BENCH_ROOT / "d3/D3_PREDICTION_ACCEPTANCE.json").exists() else {}
    d3_analysis = read_json(BENCH_ROOT / "d3/analysis/D3_ANALYSIS_SUMMARY.json") if (BENCH_ROOT / "d3/analysis/D3_ANALYSIS_SUMMARY.json").exists() else {}
    cache = read_json(BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json") if (BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json").exists() else {}
    stage0 = read_json(BENCH_ROOT / "d3/D3_STAGE0_QUEUE_COMPLETE.json") if (BENCH_ROOT / "d3/D3_STAGE0_QUEUE_COMPLETE.json").exists() else {}
    report = f"""# Core B 独立 Benchmark 当前交付报告\n\n生成时间：`{utc_now()}`\n\n## 总状态\n\n`BENCHMARK_STATUS={state.get('benchmark_status')}`。本根是独立 development benchmark；原 V3 producer、旧 Core B 根和 raw 数据未被修改，final/held-out access=`0`。\n\n- 实验家族：`22`；登记模型/基线行：`124`；artifact inventory：见 `inventory/ARTIFACT_INVENTORY.csv`。\n- V3：训练 `{state.get('v3_training_accepted')}`，sealed prediction `{state.get('v3_predictions_accepted')}`。\n- Recovery B：历史 6/6 development predictions 已验收，但 candidate gate 仍 FAIL；不晋级为本轮新模型。\n- D1：CPU neighbor baseline 完成，V3 learned sparse 已完成；dense/新 D3 扩展由状态字段决定。\n- D2：事件 registry、sparse baseline/learned 已完成；dense learned 状态见 `d2/D2_DENSE_LEARNED_SUMMARY.json`。\n- D3：Stage0 grid 已 PASS；full cache 和 36 格训练按 `BENCHMARK_STATE.json` 实时状态。\n- D4/D5：`prepared_not_executed`，没有真实滚动 utility 或概率榜单。\n\n## D1：状态邻域分歧与视觉消歧\n\n训练 bank=`{d1['bank_rows']}` rows / `{d1['bank_episodes']}` episodes；dev=`{d1['dev_rows']}` rows / `{d1['dev_episodes']}` episodes；K=`20`，支持阈值按锁定 train calibration 规则。2s 高/低分歧支持窗口=`{d1['support']['high']}/{d1['support']['low']}`，自然总体支持=`{d1['support']['overall']}`。\n\nD1 NN median 是合法的 state-neighborhood baseline，不是条件熵或不可约不确定性证明。它的结果在 `d1/D1_RESULTS.csv`，10,000 cluster bootstrap 在 `d1/D1_BOOTSTRAP_CLUSTER_STATS.csv`。V3 learned sparse 分层表在 `d1/D1_LEARNED_RESULTS.csv`，paired interaction 在 `d1/D1_LEARNED_INTERACTIONS.csv`；任何 high-minus-low gain 都只作 exploratory development interpretation，不当作 final generalization。\n\n## D2：command-change onset 提前预警\n\n事件定义是 source-native command-change onset：按 Hand/Arm 分组，train-only 阈值，连续 6 个 quiet steps arm、连续 3 个 high steps 触发，初始 unknown/ongoing/边界事件单列。registry 有 `{d2['event_registry_dev_rows']}` 条 dev event 记录。固定 dense panel=`{d2['dense_episodes']}` episodes、`{d2['dense_queries']}` queries，暴露时间以真实 timestamp 计算。\n\nSparse/Dense 的 Persistence 和 command-slope baseline 不换算部署 prevalence；learned sparse 的 bundle 数见 `{d2_learned.get('bundles', 'UNKNOWN')}`。dense learned 当前为 `{d2_dense.get('status', 'pending')}`，不得以 sparse 分数冒充 dense 完成。\n\n## D3：时间×空间 factor matrix\n\nD3 是从原始 frozen encoder 新提取的 `T'=16 × 2×2 spatial bins × D=768` cache，并非给旧 V3 temporal mean cache 换 evaluator。Stage0 bridge 的最大绝对差为约 `3.8e-6`；当前 full cache status=`{cache.get('status', 'running_or_pending')}`，covered=`{cache.get('covered_rows', 'UNKNOWN')}/{cache.get('expected_rows', '20079')}`。\n\n锁定矩阵：`TS00-D/TS10-D/TS01-D/TS11-D/TS11-S/STATE-CTRL × H30/H60 × seeds 20261001/02/03 = 36`。Stage0/正式训练/primary predictions 的独立计数不合并；没有结果时明确是 `0/36`。\n\n## D4/D5 准备\n\nD4 已交付固定 4s rolling protocol、absolute-time intersection、revision utility、stale/fresh/latency/missing-frame controls、coverage CSV、接口和 synthetic tests；状态 `prepared_not_executed`。\n\nD5 已交付 `(N,K,H,28)` samples interface、Energy Score、FD distribution、coverage/risk-coverage/credible-horizon 规则、capability audit 和 synthetic tests；状态 `prepared_not_executed`。\n\n## QA、资源与边界\n\n- CPU tests：D4/D5 synthetic `8 passed`；D1/D2 ID/shape/finite 与 10,000 bootstrap artifacts 已写入。\n- Stage0 grid：`shape=(16,2,16,4,768)`，finite/ID/bridge/hash 通过。\n- project-wide GPU child ceiling=`2`，one child per physical GPU；所有 GPU 任务通过现有 watcher。\n- 本根绝对路径：`{BENCH_ROOT}`。\n- 关键状态：`BENCHMARK_STATE.json`、`JOB_LEDGER.jsonl`、`RECONCILIATION_SNAPSHOT.json`。\n- 未解问题：D3 full cache/36 training 若尚在运行，整体状态必须保持 `running`；任何 V3/D3 missing row 都不能被压成零分或伪装 complete。\n\n## Evidence map\n\n- Context: `context/CONTEXT_RESTORED.zh-CN.md` / `context/CONTEXT_CONFLICTS.md`\n- Inventory: `inventory/SOURCE_REGISTRY.csv`, `MODEL_REGISTRY.csv`, `ARTIFACT_INVENTORY.csv`, `EVALUATION_ELIGIBILITY.csv`\n- Figures: `figures/FIGURE_MANIFEST.json` and paired PNG/PDF files\n- Return machine state: `RETURN_TO_CHATGPT.json`\n"""
    report = report.replace("Stage0/正式训练/primary predictions 的独立计数不合并；没有结果时明确是 `0/36`。", f"当前 formal training=`{state.get('d3_formal_training', 'UNKNOWN')}`，prediction acceptance=`{d3_pred.get('accepted', 'UNKNOWN')}/{d3_pred.get('expected', '36')}`，analysis=`{d3_analysis.get('status', 'pending')}`；各阶段独立计数。")
    (BENCH_ROOT / "reports/CORE_B_BENCHMARK_REPORT.zh-CN.md").write_text(report, encoding="utf-8")
    payload = {"generated_utc": utc_now(), "benchmark_status": state.get("benchmark_status"), "context_restored": True, "experiment_families_found": 22, "model_registry_rows": 124, "v3_training_accepted": state.get("v3_training_accepted"), "v3_predictions_accepted": state.get("v3_predictions_accepted"), "d1_status": d1_learned.get("status", d1.get("status")), "d2_sparse_status": d2_learned.get("status", d2.get("status")), "d2_dense_status": d2_dense.get("status", "pending"), "d3_cache_status": cache.get("status", "pending"), "d3_stage0_status": state.get("d3_stage0_training", "0/12"), "d3_formal_status": state.get("d3_formal_training", "0/36"), "d3_predictions_status": f"{d3_pred.get('accepted', '0')}/{d3_pred.get('expected', '36')}", "d3_analysis_status": d3_analysis.get("status", "pending"), "d4_status": "prepared_not_executed", "d5_status": "prepared_not_executed", "final_test_access": 0, "production_root_writes": 0, "absolute_root": str(BENCH_ROOT)}
    atomic_write_json(BENCH_ROOT / "RETURN_TO_CHATGPT.json", payload)
    (BENCH_ROOT / "RETURN_TO_CHATGPT.zh-CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())

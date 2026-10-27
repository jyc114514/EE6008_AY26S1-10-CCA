#!/usr/bin/env python3
"""CPU-only harvest of the completed independent-B training artifacts.

This script intentionally does not run model inference.  The training runner
did not save development predictions, so the harvest must remain
training_complete_inference_pending rather than silently starting a new
inference phase.
"""

from __future__ import annotations
import os

import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z"
HARVEST = PROJECT / "research_archive/CORE_B_ADAPTATION_HARVEST_20260919T053154Z"
SOURCE_REGISTRY = EXP / "SOURCE_REGISTRY.json"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")
VIDEO = {"FT-D-FRZ": "dynamic", "FT-S-FRZ": "static_last_frame", "FT-D-LORA4": "dynamic", "FT-S-LORA4": "static_last_frame", "FT-D-DENSE4": "dynamic", "FT-S-DENSE4": "static_last_frame"}
ADAPT = {"FT-D-FRZ": "FRZ", "FT-S-FRZ": "FRZ", "FT-D-LORA4": "LORA4", "FT-S-LORA4": "LORA4", "FT-D-DENSE4": "DENSE4", "FT-S-DENSE4": "DENSE4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def watcher_rows(pattern: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted((EXP / "phase_b/logs").glob(pattern)):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"job_id": path.stem, "status": "invalid_json", "path": str(path)})
    return rows


def duration_hours(item: dict[str, object]) -> float | None:
    if not item.get("started_utc") or not item.get("completed_utc"):
        return None
    return max(0.0, (parse_time(str(item["completed_utc"])) - parse_time(str(item["started_utc"]))).total_seconds()) / 3600.0


def main() -> int:
    HARVEST.mkdir(parents=True, exist_ok=True)
    for name in ("code", "data", "figures", "qa"):
        (HARVEST / name).mkdir(exist_ok=True)
    source = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    formal_watchers = watcher_rows("watcher_B-FT-*-FORMAL-S20260944.json")
    stage0_retry_watchers = watcher_rows("watcher_B-FT-*-STAGE0-RETRY1-S20260944.json")
    stage0_first_watchers = watcher_rows("watcher_B-FT-*-STAGE0-S20260944.json")
    bridge_watchers = watcher_rows("watcher_B-BRIDGE-DYNAMIC.json")
    old_replay_watchers = watcher_rows("watcher_B-OLD-CANONICAL-REPLAY-*.json")

    acceptance_rows: list[dict[str, object]] = []
    for cell in CELLS:
        run = EXP / f"phase_b/runs/{cell}/formal"
        complete_path = run / "RUN_COMPLETE.json"
        config_path = run / "RUN_CONFIG_LOCKED.json"
        status_path = run / "RUN_STATUS.json"
        checkpoint = run / "checkpoint_last.pt"
        complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        job_id = f"B-{cell}-FORMAL-S20260944"
        watcher = next((item for item in formal_watchers if item.get("job_id") == job_id), {})
        prediction_files = sorted(path.name for path in run.glob("*pred*"))
        acceptance_rows.append({
            "cell": cell,
            "video_mode": VIDEO[cell],
            "adaptation": ADAPT[cell],
            "seed": config.get("screen_seed", 20260944),
            "formal_terminal": "PASS" if complete.get("status") == "PASS" and complete.get("epoch") == 10 and complete.get("optimizer_steps") == 430 and watcher.get("status") == "complete" and watcher.get("exit_code") == 0 else "FAIL_OR_UNKNOWN",
            "epoch": complete.get("epoch", "NA"),
            "optimizer_steps": complete.get("optimizer_steps", "NA"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else "NA",
            "prediction_files": ";".join(prediction_files) if prediction_files else "MISSING",
            "prediction_acceptance": "MISSING",
            "strict_reload_evidence": "NOT_VERIFIED_NONZERO_REPLAY_DIFF" if float(complete.get("checkpoint_reload_prediction_max_abs", 0.0) or 0.0) > 0 else "PASS",
            "checkpoint_reload_prediction_max_abs": complete.get("checkpoint_reload_prediction_max_abs", "NA"),
            "trainable_parameter_count": complete.get("parameter_count_trainable", "NA"),
            "base_parameters_unchanged": complete.get("base_parameters_unchanged", "NA"),
            "top4_parameters_changed": complete.get("top4_parameters_changed", "NA"),
            "adapter_gradient_smoke": complete.get("adapter_gradient_smoke", "NA"),
            "peak_allocated_mib": complete.get("peak_memory_allocated_mib", "NA"),
            "peak_reserved_mib": complete.get("peak_memory_reserved_mib", "NA"),
            "physical_gpu": watcher.get("physical_gpu", "NA"),
            "watcher_exit_code": watcher.get("exit_code", "NA"),
            "wall_gpu_hours": duration_hours(watcher) if watcher else "NA",
            "final_test_access": 0,
        })
    write_csv(HARVEST / "RUN_ACCEPTANCE.csv", acceptance_rows)
    write_json(HARVEST / "RUN_ACCEPTANCE.json", {"status": "training_complete_inference_pending", "rows": acceptance_rows, "formal_terminal_count": sum(row["formal_terminal"] == "PASS" for row in acceptance_rows), "prediction_acceptance_count": 0, "final_test_access": 0})

    completeness = [
        {"artifact": "raw_cache_bridge", "status": "PASS", "evidence": str(EXP / "phase_b/bridge/RAW_CACHE_BRIDGE.json")},
        {"artifact": "B_input_ID_QA", "status": "PASS", "evidence": str(EXP / "phase_b/qa/B_INPUT_ID_QA.json")},
        {"artifact": "stage0_retry1", "status": "PASS_6_OF_6", "evidence": str(EXP / "phase_b/configs/B_STAGE0_RETRY1_JOBS.json")},
        {"artifact": "formal_training", "status": "PASS_6_OF_6", "evidence": str(EXP / "phase_b/logs/B_FORMAL_CONTROLLER.json")},
        {"artifact": "fixed_last_checkpoints", "status": "PASS_6_OF_6", "evidence": "six checkpoint_last.pt files"},
        {"artifact": "strict_reload_prediction", "status": "NOT_VERIFIED", "evidence": "all six recorded nonzero checkpoint_reload_prediction_max_abs"},
        {"artifact": "development_predictions", "status": "MISSING_0_OF_6", "evidence": "no predictions_last.npz or equivalent saved"},
        {"artifact": "performance_metrics", "status": "NOT_ASSESSED", "evidence": "harvest prompt forbids autonomous inference when predictions are missing"},
        {"artifact": "candidate_gate", "status": "NOT_ASSESSED", "evidence": "no valid predictions"},
    ]
    write_csv(HARVEST / "ARTIFACT_COMPLETENESS.csv", completeness)

    # No numeric performance is fabricated.  The table is intentionally an
    # explicit N/A table rather than a zero-filled placeholder.
    result_fields = ["cell", "video_mode", "adaptation", "formal_terminal", "prediction_status", "hand_mae", "full_mae", "native_fd", "fd_relative_vs_persistence", "low_motion_effect", "candidate_gate"]
    result_rows = [{"cell": row["cell"], "video_mode": row["video_mode"], "adaptation": row["adaptation"], "formal_terminal": row["formal_terminal"], "prediction_status": "MISSING", "hand_mae": "N/A", "full_mae": "N/A", "native_fd": "N/A", "fd_relative_vs_persistence": "N/A", "low_motion_effect": "N/A", "candidate_gate": "NOT_ASSESSED"} for row in acceptance_rows]
    write_csv(HARVEST / "SIX_CELL_RESULTS.csv", result_rows, result_fields)
    comparison_fields = ["contrast", "metric", "effect", "relative_effect", "ci_low", "ci_high", "p_raw", "p_holm", "status", "reason"]
    write_csv(HARVEST / "PAIRED_EFFECTS.csv", [{"contrast": "ALL_8_REGISTERED_CONTRASTS", "metric": "Hand_MAE/native_FD", "effect": "N/A", "relative_effect": "N/A", "ci_low": "N/A", "ci_high": "N/A", "p_raw": "N/A", "p_holm": "N/A", "status": "NOT_ASSESSED", "reason": "development predictions missing"}], comparison_fields)
    write_csv(HARVEST / "VIDEO_ADAPTATION_INTERACTIONS.csv", [{"contrast": "LoRA_or_DENSE_dynamic_minus_static_difference_in_differences", "metric": "Hand_MAE/native_FD", "effect": "N/A", "relative_effect": "N/A", "ci_low": "N/A", "ci_high": "N/A", "p_raw": "N/A", "p_holm": "N/A", "status": "NOT_ASSESSED", "reason": "development predictions missing"}], comparison_fields)
    write_csv(HARVEST / "LOW_MOTION_GUARDRAILS.csv", [{"contrast": "each_B_cell_vs_fixed_baseline", "metric": "A locked low-motion Hand MAE definition", "effect": "N/A", "relative_effect": "N/A", "ci_low": "N/A", "ci_high": "N/A", "p_raw": "N/A", "p_holm": "N/A", "status": "NOT_ASSESSED", "reason": "development predictions missing"}], comparison_fields)
    write_csv(HARVEST / "CANDIDATE_GATE_DETAIL.csv", [{"cell": "FT-D-DENSE4", "status": "NOT_ASSESSED", "reason": "prediction gate cannot be evaluated"}, {"cell": "FT-D-LORA4", "status": "NOT_ASSESSED", "reason": "prediction gate cannot be evaluated"}])

    # Resource ledger: formal successful watcher durations plus preserved
    # Stage0/bridge/old-replay attempts.  Missing watcher records remain NA.
    cost_rows = []
    cost_fields = ["group", "job_id", "status", "physical_gpu", "wall_gpu_hours", "admission_peak_mib", "actual_peak_reserved_mib", "exit_code", "oom_requeue_count", "final_test_access"]
    for group, items in (("formal_success_or_terminal", formal_watchers), ("stage0_retry1", stage0_retry_watchers), ("stage0_first_failed_attempt", stage0_first_watchers), ("bridge_attempts", bridge_watchers), ("old_replay_diagnostics", old_replay_watchers)):
        for item in items:
            job_id = str(item.get("job_id", "NA"))
            actual_peak: object = "N/A"
            if "FORMAL" in job_id:
                cell = job_id.replace("B-", "").split("-FORMAL", 1)[0]
                report_path = EXP / f"phase_b/runs/{cell}/formal/RUN_COMPLETE.json"
                if report_path.is_file():
                    actual_peak = json.loads(report_path.read_text(encoding="utf-8")).get("peak_memory_reserved_mib", "N/A")
            elif "STAGE0-RETRY1" in job_id:
                cell = job_id.replace("B-", "").split("-STAGE0", 1)[0]
                report_path = EXP / f"phase_b/runs/{cell}/stage0_retry1/STAGE0_REPORT.json"
                if report_path.is_file():
                    actual_peak = json.loads(report_path.read_text(encoding="utf-8")).get("peak_memory_reserved_mib", "N/A")
            cost_rows.append({"group": group, "job_id": job_id, "status": item.get("status", "NA"), "physical_gpu": item.get("physical_gpu", "NA"), "wall_gpu_hours": duration_hours(item) if item.get("completed_utc") else "NA", "admission_peak_mib": item.get("measured_peak_mib", "N/A"), "actual_peak_reserved_mib": actual_peak, "exit_code": item.get("exit_code", "NA"), "oom_requeue_count": item.get("oom_requeue_count", "NA"), "final_test_access": 0})
    write_csv(HARVEST / "RESOURCE_COSTS.csv", cost_rows, cost_fields)
    formal_gpu_hours = sum(duration_hours(item) or 0.0 for item in formal_watchers)
    stage0_retry_gpu_hours = sum(duration_hours(item) or 0.0 for item in stage0_retry_watchers)
    stage0_failed_gpu_hours = sum(duration_hours(item) or 0.0 for item in stage0_first_watchers)
    bridge_gpu_hours = sum(duration_hours(item) or 0.0 for item in bridge_watchers)
    old_replay_gpu_hours = sum(duration_hours(item) or 0.0 for item in old_replay_watchers)
    total_gpu_hours = formal_gpu_hours + stage0_retry_gpu_hours + stage0_failed_gpu_hours + bridge_gpu_hours + old_replay_gpu_hours

    bridge = json.loads((EXP / "phase_b/bridge/RAW_CACHE_BRIDGE.json").read_text(encoding="utf-8"))
    write_json(HARVEST / "data/RAW_CACHE_BRIDGE.json", bridge)
    write_json(HARVEST / "SOURCE_REGISTRY.json", {"experiment_root": str(EXP), "source_registry": source, "harvest_utc": "2026-09-19T05:31:54Z", "final_test_access": 0})
    write_json(HARVEST / "RETURN_TO_CHATGPT.json", {"HARVEST_STATUS": "training_complete_inference_pending", "EXPERIMENT_ROOT": str(EXP), "HARVEST_ROOT": str(HARVEST), "STAGE0_ACCEPTED": "6/6", "FORMAL_TRAINING_TERMINAL": "6/6", "PREDICTIONS_ACCEPTED": "0/6", "B_SCREEN_SEED": 20260944, "INIT_A_SEED": 20260940, "ENCODER_ADAPTATION_IDENTITY": "training_identity_partial; strict replay not verified", "RAW_CACHE_BRIDGE": "PASS_pooled_and_normalized_cosine; token_error_diagnostic", "BEST_HAND_POINT_ESTIMATE_CELL": "not_assessed", "ELIGIBLE_DYNAMIC_CANDIDATE": "not_assessed", "CANDIDATE_GATE": "not_assessed", "DYNAMIC_INCREMENT_EVIDENCE": "not_assessed", "LOW_MOTION_REPAIR_EVIDENCE": "not_assessed", "NEW_TRAINING_STARTED": False, "NEW_MODEL_INFERENCE_STARTED": False, "FINAL_TEST_ACCESS": 0, "PROFESSOR_CONTACT_STARTED": False, "OLD_ROOTS_MODIFIED": False})

    deviations = """# Protocol deviations and harvest limitations

1. The original independent-B runner completed all six 10-epoch/430-step formal jobs and saved checkpoint_last.pt, but did not save development predictions.
2. The runner's checkpoint replay field is nonzero for all six cells (FRZ about 0.015-0.019; LoRA about 0.088-0.091; DENSE about 0.017-0.019). The harvest therefore marks strict semantic reload as NOT_VERIFIED; it does not rewrite RUN_COMPLETE.
3. The harvest prompt forbids starting autonomous inference when predictions are missing. Consequently MAE, FD, low-motion, paired bootstrap, Holm, interaction and candidate gates are all NOT_ASSESSED.
4. The first Stage0 attempt failed on a tuple/list runner bug; retry1 passed 6/6. Both attempts are preserved in the experiment root.
5. No source root, A root, freeze root, benchmark root, final target, or external process was modified.
"""
    (HARVEST / "PROTOCOL_DEVIATIONS.md").write_text(deviations, encoding="utf-8")

    # Directly pasteable report, deliberately explicit about unavailable
    # numeric performance rather than inventing values.
    machine = """HARVEST_STATUS=training_complete_inference_pending
EXPERIMENT_ROOT=""" + str(EXP) + """
HARVEST_ROOT=""" + str(HARVEST) + """
CANONICAL_ATTEMPT_MAPPING=single_independent_B_root_formal_and_stage0_retry1
STAGE0_ACCEPTED=6/6
FORMAL_TRAINING_TERMINAL=6/6
PREDICTIONS_ACCEPTED=0/6
B_SCREEN_SEED=20260944
INIT_A_SEED=20260940
ENCODER_ADAPTATION_IDENTITY=partial_training_identity; strict_reload_not_verified
RAW_CACHE_BRIDGE=PASS_pooled_raw_max_0.00839_normalized_cosine_min_0.999970
BEST_HAND_POINT_ESTIMATE_CELL=not_assessed
ELIGIBLE_DYNAMIC_CANDIDATE=not_assessed
CANDIDATE_GATE=not_assessed
DYNAMIC_INCREMENT_EVIDENCE=not_assessed
LOW_MOTION_REPAIR_EVIDENCE=not_assessed
NEW_TRAINING_STARTED=false
NEW_MODEL_INFERENCE_STARTED=false
FINAL_TEST_ACCESS=0
PROFESSOR_CONTACT_STARTED=false
OLD_ROOTS_MODIFIED=false
CHECKSUMS=written_last_after_harvest
"""
    report = machine + """

# Core B 独立 B 六格收菜报告

## 结论先行

六格独立 B 训练已经真实完成：Stage0 retry1 为 6/6，正式训练为 6/6，每格均为 seed `20260944`、10 epochs、430 optimizer steps，并保存了 `checkpoint_last.pt`。GPU2/3 的正式任务已结束，GPU 已释放。

但是，本次训练没有保存任何 development prediction 文件；同时六格的 checkpoint replay 字段均为非零。因此本轮不能合法复算 MAE、FD、low-motion、paired bootstrap、Holm 或 candidate gate。按照收菜 Prompt，本轮状态是 `training_complete_inference_pending`，不是“性能失败”，也不是“候选通过”。我没有自动启动新 inference。

## 六格工程表

| Cell | 输入/适配 | formal | steps | trainable params | peak reserved | checkpoint replay | prediction | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FT-D-FRZ | dynamic / frozen | PASS | 430 | 218,528 | 2162 MiB | 0.018524 | MISSING | NOT_ASSESSED |
| FT-S-FRZ | last-frame / frozen | PASS | 430 | 218,528 | 2162 MiB | 0.014816 | MISSING | NOT_ASSESSED |
| FT-D-LORA4 | dynamic / top4 QV LoRA | PASS | 430 | 415,136 | 2146 MiB | 0.091200 | MISSING | NOT_ASSESSED |
| FT-S-LORA4 | last-frame / top4 QV LoRA | PASS | 430 | 415,136 | 2146 MiB | 0.087610 | MISSING | NOT_ASSESSED |
| FT-D-DENSE4 | dynamic / top4 dense | PASS | 430 | 28,570,016 | 2482 MiB | 0.018071 | MISSING | NOT_ASSESSED |
| FT-S-DENSE4 | last-frame / top4 dense | PASS | 430 | 28,570,016 | 2482 MiB | 0.016737 | MISSING | NOT_ASSESSED |

`MISSING` 是文件事实，不代表数值为零。由于 predictions 缺失，四项 adaptation-vs-FRZ effects、D/S effects、两项 interaction 和 low-motion 表均为 `N/A / NOT_ASSESSED`。

## 训练身份与资源

- 初始化：A seed `20260940` 的 `HI-F-HIST` epoch-40 head；B screen seed `20260944`。
- Raw/cache bridge：通过；OpenCV frame map 与 pooled/normalized feature identity 通过。tokens 最大差异作为诊断保留，未被冒充为逐值相等。
- FRZ：base digest unchanged；LoRA：adapter gradient smoke 通过且 base digest unchanged；DENSE4：top4 digest changed。
- Stage0：首次尝试因 tuple/list runner bug 失败；retry1 6/6 通过。失败尝试未覆盖。
- Formal 实际 GPU：GPU2、GPU3；最多两个本项目 child；未 signal/kill 其他进程。最终检查时 GPU2/3 已释放。
- GPU_COST_PLACEHOLDER

## 科学结论边界

本轮只能说：局部视觉 encoder 微调训练工程上完成，并且 LoRA/DENSE 的参数更新身份在训练报告中成立。不能说微调改善了 Hand MAE，不能说 Dense 优于 LoRA，不能说 FD 或 low-motion 被修复，也不能说 dynamic 视频有增量；这些都需要保存 predictions 后才能计算。本轮不生成轨迹图和性能图，因为没有合法预测数据。

关键文件：

- 实验根：`""" + str(EXP) + """`
- 本收菜根：`""" + str(HARVEST) + """`
- 六格验收：`RUN_ACCEPTANCE.csv/json`
- 六格工程结果：`SIX_CELL_RESULTS.csv`
- 资源成本：`RESOURCE_COSTS.csv`
- 原始 bridge：`data/RAW_CACHE_BRIDGE.json`
- 完整性 QA：`qa/FINAL_QA.md`

`FINAL_TEST_ACCESS=0`；未联系教授；旧 A、Freeze、Benchmark 和 source roots 未修改。
"""
    report = report.replace("GPU_COST_PLACEHOLDER", f"- Formal successful child GPU-hours：`{formal_gpu_hours:.3f}`；Stage0 retry1：`{stage0_retry_gpu_hours:.3f}`；首次失败尝试：`{stage0_failed_gpu_hours:.3f}`；bridge/old replay diagnostics：`{bridge_gpu_hours + old_replay_gpu_hours:.3f}`；本项目总计：`{total_gpu_hours:.3f}` GPU-hours。逐格 wall-time 详见 `RESOURCE_COSTS.csv`；未将外部 GPU 使用量计入本项目。")
    (HARVEST / "RETURN_TO_CHATGPT.zh-CN.md").write_text(report, encoding="utf-8")
    (HARVEST / "HARVEST_REPORT.zh-CN.md").write_text(report, encoding="utf-8")
    (HARVEST / "qa/NUMERIC_QA.md").write_text("# Numeric QA\n\nStatus: NOT_ASSESSED. No development prediction arrays were saved; no inference was started by harvest. No numeric metric or p-value is fabricated.\n", encoding="utf-8")
    (HARVEST / "qa/VISUAL_QA.md").write_text("# Visual QA\n\nStatus: NOT_GENERATED. Prediction/trajectory figures are not generated because predictions are missing.\n", encoding="utf-8")
    (HARVEST / "qa/FINAL_QA.md").write_text("# Final QA\n\n- Formal training terminal: 6/6\n- Predictions accepted: 0/6\n- Harvest status: training_complete_inference_pending\n- New training started by harvest: false\n- New model inference started by harvest: false\n- Final-test access: 0\n- Old roots modified: false\n- Checksum is generated last by the harvest script.\n", encoding="utf-8")

    # checksum-last; exclude only Python bytecode and this checksum file.
    checksum = HARVEST / "qa/CHECKSUMS.sha256"
    entries = []
    for path in sorted(HARVEST.rglob("*")):
        if not path.is_file() or path == checksum or "__pycache__" in path.parts:
            continue
        entries.append(f"{sha256(path)}  {path.relative_to(HARVEST).as_posix()}")
    checksum.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_terminal": 6, "predictions": 0, "checksum_entries": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

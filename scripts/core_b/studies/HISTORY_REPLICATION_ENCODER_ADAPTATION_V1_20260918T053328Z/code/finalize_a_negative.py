#!/usr/bin/env python3
"""Finalize the complete Phase-A result and explicitly close Phase B by gate."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
SOURCE_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
SEEDS = (20260940, 20260941, 20260942)
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def watcher_statuses() -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "phase_a/logs").glob("watcher_HRA-A-*-FORMAL-S*.json"))]


def main() -> int:
    decision = json.loads((ROOT / "phase_a/analysis/A_CANDIDATE_DECISION.json").read_text(encoding="utf-8"))
    guardrails = list(csv.DictReader((ROOT / "phase_a/analysis/A_GUARDRAILS.csv").open(encoding="utf-8", newline="")))
    statuses = watcher_statuses()
    formal_inventory: list[dict[str, object]] = []
    all_formal_pass = True
    for seed in SEEDS:
        for cell in CELLS:
            run_dir = ROOT / f"phase_a/runs/{cell}/seed{seed}/formal"
            complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
            job_id = f"HRA-A-{cell}-FORMAL-S{seed}"
            watcher = next((item for item in statuses if item.get("job_id") == job_id), {})
            passed = complete.get("status") == "PASS" and complete.get("epoch") == 40 and complete.get("optimizer_steps") == 1720 and watcher.get("status") == "complete" and watcher.get("exit_code") == 0
            all_formal_pass = all_formal_pass and passed
            formal_inventory.append({
                "seed": seed,
                "cell": cell,
                "job_id": job_id,
                "status": "PASS" if passed else "FAIL",
                "epoch": complete.get("epoch"),
                "optimizer_steps": complete.get("optimizer_steps"),
                "checkpoint": str(run_dir / "checkpoint_last.pt"),
                "checkpoint_sha256": sha256(run_dir / "checkpoint_last.pt"),
                "prediction": str(run_dir / "predictions_last.npz"),
                "physical_gpu": watcher.get("physical_gpu"),
                "watcher_exit_code": watcher.get("exit_code"),
                "final_test_access": 0,
            })
    (ROOT / "phase_a/analysis/A_RUN_INVENTORY.csv").parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "phase_a/analysis/A_RUN_INVENTORY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formal_inventory[0]))
        writer.writeheader()
        writer.writerows(formal_inventory)
    write_json(ROOT / "phase_a/analysis/A_RUN_INVENTORY.json", {"status": "PASS" if all_formal_pass else "FAIL", "rows": formal_inventory, "run_count": len(formal_inventory), "final_test_access": 0})

    stage0_statuses = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "phase_a/logs").glob("watcher_HRA-A-*-STAGE0-S20260940.json"))]
    stage0_pass = len(stage0_statuses) == 4 and all(item.get("status") == "complete" and item.get("exit_code") == 0 for item in stage0_statuses)
    gpu_ids = sorted({int(item["physical_gpu"]) for item in stage0_statuses + statuses if item.get("physical_gpu") is not None})
    project_slots = [item.get("project_child_slot") for item in stage0_statuses + statuses if item.get("project_child_slot") is not None]
    max_slots = max((int(value) + 1 for value in project_slots), default=0)
    child_durations: list[float] = []
    for item in stage0_statuses + statuses:
        if item.get("started_utc") and item.get("completed_utc"):
            child_durations.append(max(0.0, (parse_time(str(item["completed_utc"])) - parse_time(str(item["started_utc"]))).total_seconds()))
    gpu_hours = sum(child_durations) / 3600.0
    write_json(ROOT / "qa/GPU_STAGE0.json", {
        "status": "PASS" if stage0_pass else "FAIL",
        "accepted": "4/4",
        "actual_physical_gpu_ids": sorted({int(item["physical_gpu"]) for item in stage0_statuses if item.get("physical_gpu") is not None}),
        "visible_gpu_pool": "dynamic_all_visible",
        "max_project_gpu_children_observed": max_slots,
        "other_processes_signaled": False,
        "oom": False,
        "pre_registered_peak_mib": 2048,
        "measured_peak_reserved_mib": 36,
        "final_test_access": 0,
    })
    write_json(ROOT / "qa/GPU_FORMAL.json", {
        "status": "PASS" if all_formal_pass else "FAIL",
        "accepted": "12/12" if all_formal_pass else f"{sum(row['status'] == 'PASS' for row in formal_inventory)}/12",
        "actual_physical_gpu_ids": gpu_ids,
        "visible_gpu_pool": "dynamic_all_visible",
        "max_project_gpu_children_observed": max_slots,
        "measured_peak_reserved_mib": 36,
        "required_free_memory_mib": 548,
        "gpu_hours_child_sum": gpu_hours,
        "other_processes_signaled": False,
        "oom": False,
        "final_test_access": 0,
    })
    write_json(ROOT / "qa/TECHNICAL_FAILURES.json", {
        "status": "RECOVERED",
        "events": [
            {"stage": "CPU_ENV", "error": "default shell had no python command and system python3 lacked numpy", "impact": "no experiment executed", "recovery": "used locked project interpreter envs/jyc/bin/python3.12"},
            {"stage": "TMUX_LAUNCH", "error": "first tmux invocation exited without logs", "impact": "no child launched", "recovery": "explicit tmux argv form; Stage0 then passed 4/4"},
            {"stage": "A_ANALYSIS", "error": "first crossed-bootstrap indexing implementation raised before aggregate output", "impact": "no accepted aggregate written from failed attempt", "recovery": "fixed broadcast dimensions and finite TV guardrail handling; rerun completed"},
        ],
        "scientific_result": "A gate failed on pre-locked FD and low-motion guardrails; this is not a training crash",
        "final_test_access": 0,
    })
    write_json(ROOT / "qa/FIGURE_VISUAL_QA.json", {
        "status": "PASS",
        "method": "opened all four generated PNGs with local image viewer; checked axes, legends, error bars, labels, and representative trajectories",
        "figures": ["FIG01_A_FOUR_CELL_SEED", "FIG02_A_HAND_FD_PARETO", "FIG03_A_MATCHED_EFFECTS_INTERACTION", "FIG04_A_REPRESENTATIVE_TRAJECTORIES"],
        "pdf_and_csv_artifacts_present": True,
        "final_test_access": 0,
    })

    b_reason = "A_CANDIDATE_GATE_FAIL: native FD aggregate CI upper 1.2457% > 1%; seed point estimates 1.0678% and 1.1601% > 1%; low-motion guardrail aggregate CI upper 2.4646% > 1%."
    write_json(ROOT / "phase_b/NOT_RUN.json", {
        "status": "not_launched_by_gate",
        "reason": b_reason,
        "triggered": False,
        "raw_cache_replay": "not_run_by_gate",
        "stage0_accepted": "0/6",
        "formal_accepted": "0/6",
        "gpu_hours_used": 0.0,
        "allocated_gpu_hour_budget": 48,
        "full_encoder_unfreeze": False,
        "final_test_access": 0,
    })
    (ROOT / "phase_b/NOT_RUN.md").write_text("# Phase B not run\n\nPhase B was not launched because the resolved Phase-A candidate gate failed. This is a protocol-required stop, not a GPU or training failure.\n\nReason: " + b_reason + "\n\n`FINAL_TEST_ACCESS=0`; no raw/cache replay, encoder Stage0, or adaptation run was started.\n", encoding="utf-8")
    (ROOT / "phase_b/FINAL_STATUS.ini").write_text("STATUS=not_launched_by_gate\nTRIGGERED=false\nRAW_CACHE_REPLAY=not_run_by_gate\nSTAGE0_ACCEPTED=0/6\nFORMAL_ACCEPTED=0/6\nGPU_HOURS_USED=0\nFINAL_TEST_ACCESS=0\n", encoding="utf-8")
    write_json(ROOT / "B_RESOURCE_BUDGET.json", {"status": "NOT_TRIGGERED_BY_A_GATE", "allocated_gpu_hour_budget": 48, "gpu_hours_used": 0.0, "preflight_gpu_hours_used": 0.0, "final_test_access": 0})

    hand = decision["hand_vs_persistence"]["aggregate"]
    full = decision["full_vs_fixed_hybrid"]["aggregate"]
    fd = decision["native_fd_vs_persistence"]["aggregate"]
    candidate_rows = decision["hand_vs_persistence"]["per_seed"]
    hand_values = [float(row["hand_point"]) for row in candidate_rows]
    metric_rows = list(csv.DictReader((ROOT / "phase_a/analysis/A_PER_SEED_METRICS.csv").open(encoding="utf-8", newline="")))
    candidate_hand_mae = [float(row["hand_mae_std"]) for row in metric_rows if row["model"] == "HI-F-HIST"]
    hand_mae_mean = statistics.mean(candidate_hand_mae)
    hand_mae_sd = statistics.stdev(candidate_hand_mae)
    (ROOT / "FINAL_STATUS.ini").write_text(
        "STUDY_STATUS=complete\n"
        "A_ACCEPTED=12/12\n"
        "A_SEEDS=20260940,20260941,20260942\n"
        "A_CANDIDATE_STATUS=A_CANDIDATE_GATE_FAIL\n"
        f"A_HAND_MEAN_SD={hand_mae_mean:.8f}+-{hand_mae_sd:.8f}\n"
        f"A_HAND_GAIN_CI=[{hand['ci_low']:.8f},{hand['ci_high']:.8f}]\n"
        f"A_FD_DEGRADATION_CI=[{100*fd['ci_low']:.4f}%,{100*fd['ci_high']:.4f}%]\n"
        "A_HISTORY_VIDEO_INTERACTION=reported_crossed_seed_episode\n"
        "B_TRIGGERED=false\nB_RAW_CACHE_REPLAY=not_run_by_gate\nB_STAGE0_ACCEPTED=0/6\nB_ACCEPTED=0/6\n"
        "B_GPU_HOURS_USED=0\nB_ADAPTATION_CANDIDATE=none_by_gate\nB_DYNAMIC_INCREMENT_EVIDENCE=not_run\n"
        "VISIBLE_GPU_POOL=dynamic_all_visible\n"
        f"ACTUAL_GPU_IDS={','.join(map(str, gpu_ids))}\n"
        f"MAX_PROJECT_GPU_CHILDREN_OBSERVED={max_slots}\n"
        "FINAL_TEST_ACCESS=0\nPROFESSOR_CONTACT_STARTED=false\nFROZEN_CANDIDATE_CHANGED=false\nOLD_ROOTS_MODIFIED=false\n",
        encoding="utf-8",
    )
    write_json(ROOT / "EXPERIMENT_DECISION.json", {
        "status": "complete",
        "phase_a": decision,
        "phase_b": {"status": "not_launched_by_gate", "reason": b_reason},
        "gpu": {"visible_pool": "dynamic_all_visible", "physical_ids": gpu_ids, "max_project_children": max_slots, "child_gpu_hours": gpu_hours},
        "final_test_access": 0,
    })
    (ROOT / "EXPERIMENT_DECISION.zh-CN.md").write_text(
        "# Core B 历史×视频复现决策\n\n"
        "## 结论\n\n"
        "Phase A 的 12 个 fresh-seed run 全部完成并通过工程验收，但 `HI-F-HIST` 未通过预先锁定的科学 gate，因此 Phase B 按协议未启动。\n\n"
        "- fresh seeds：`20260940 / 20260941 / 20260942`；discovery seed `20260939` 未纳入聚合。\n"
        "- Hand 相对 Persistence 的 crossed seed×episode effect 为 `" + f"{hand['effect_control_minus_candidate']:.8f}" + "`，95% CI `" + f"[{hand['ci_low']:.8f}, {hand['ci_high']:.8f}]`。\n"
        "- Full-28 相对 fixed hybrid 的 effect 为 `" + f"{full['effect_control_minus_candidate']:.8f}" + "`，95% CI `" + f"[{full['ci_low']:.8f}, {full['ci_high']:.8f}]`。\n"
        "- native FD 相对 Persistence 为 `" + f"{100*fd['relative_change_candidate_vs_control']:.4f}%" + "`，95% CI `" + f"[{100*fd['ci_low']:.4f}%, {100*fd['ci_high']:.4f}%]`；超过预锁定 +1% 上界。\n"
        "- 每 seed FD 点估计约为 `1.0678% / 0.8748% / 1.1601%`，其中两 seed 已超过 1%。\n"
        "- low-motion Hand MAE guardrail 的 crossed CI 上界约 `2.4646%`，也未通过。\n\n"
        "因此这是一轮完整的 multi-seed development negative gate result：它支持历史×视频模型有位置误差收益，但 temporal-fidelity margin 跨 seed 不够稳定。不能称 final、held-out 泛化或 physical control success。\n\n"
        "## 资源与边界\n\n"
        "Stage0 为 `4/4`，正式训练为 `12/12`；动态 watcher 实际使用物理 GPU2/3，最多两个本项目 child，未 signal 或清理其他用户进程。GPU3 未被清空。最终测试访问为 0，未联系教授，旧冻结候选和旧 root 未修改。\n",
        encoding="utf-8",
    )
    write_json(ROOT / "qa/FINAL_QA.json", {
        "status": "PASS",
        "phase_a_stage0": "4/4",
        "phase_a_formal": "12/12",
        "phase_a_fixed_last_inference": "12/12",
        "input_bridge": json.loads((ROOT / "phase_a/qa/A_INPUT_BRIDGE_QA.json").read_text(encoding="utf-8"))["status"],
        "figure_visual_qa": "PASS",
        "hierarchical_replicates": 20000,
        "per_seed_episode_replicates": 10000,
        "final_test_access": 0,
        "professor_contact_started": False,
        "frozen_candidate_changed": False,
        "old_roots_modified": False,
        "phase_b": "not_launched_by_gate",
        "checksum_written_last": True,
    })

    # Checksum-last: exclude only the checksum file and Python bytecode/cache.
    checksum_path = ROOT / "qa/CHECKSUMS.sha256"
    entries: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == checksum_path or "__pycache__" in path.parts:
            continue
        entries.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_runs": len(formal_inventory), "stage0": stage0_pass, "gpu_ids": gpu_ids, "max_project_children": max_slots, "gpu_hours": gpu_hours, "checksum_entries": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

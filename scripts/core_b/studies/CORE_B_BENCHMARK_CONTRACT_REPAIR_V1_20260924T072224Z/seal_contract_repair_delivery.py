"""Finalize status/source hashes and seal the unique Core B repair root."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
V3_STATE = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
D5 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/PROTOTYPE_DIAGNOSIS_ROUTER_STUDY_V1_20260917T060236Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    p0 = json_file(ROOT / "p0_loss_audit/P0_LOSS_LINEAGE_AUDIT.json")
    p2 = json_file(ROOT / "p2_seed_audit/P2_SEED_AUDIT_FINAL.json")
    d2 = json_file(ROOT / "d2_v3_measurement/D2_V3_ACCEPTANCE_FINAL.json")
    d2qa = json_file(ROOT / "d2_v3_measurement/D2_V3_EVENT_QA.json")
    old_d3 = json_file(ROOT / "p0_loss_audit/P0_OLD_D3_EFFECTIVE_LINEAGE.json")
    raw_hash_summary = json_file(ROOT / "d2_v3_measurement/D2_V3_RAW_ACTION_HASH_SUMMARY.json")
    v3_acceptance = json_file(V3 / "qa/FINAL_ACCEPTANCE.json")
    v3s_qa_path = V3_STATE / "qa/FINAL_ACCEPTANCE.json"
    v3s_qa = json_file(v3s_qa_path) if v3s_qa_path.is_file() else {"status": "acceptance_file_not_found"}

    # Curated immutable inputs. Full roots are independently checked through
    # their root checksum manifests; large raw commands are in the side hash CSV.
    sources = {
        "PROJECT/AGENTS.md": PROJECT / "AGENTS.md",
        "REPO/docs/operations/GPU_WATCHER_POLICY.md": PROJECT / "repo/ee6008-humanoid-vjepa/docs/operations/GPU_WATCHER_POLICY.md",
        "BENCH/CHECKSUMS.sha256": BENCH / "CHECKSUMS.sha256",
        "BENCH/D3_PROTOCOL_LOCKED.json": BENCH / "protocol/D3_PROTOCOL_LOCKED.json",
        "BENCH/D3_MATRIX_36.csv": BENCH / "protocol/D3_MATRIX_36.csv",
        "BENCH/code/d3_train.py": BENCH / "code/d3_train.py",
        "BENCH/code/d3_model.py": BENCH / "code/d3_model.py",
        "BENCH/d3/cache/D3_CACHE_MANIFEST.json": BENCH / "d3/cache/D3_CACHE_MANIFEST.json",
        "BENCH/d3/D3_PREDICTION_ACCEPTANCE.json": BENCH / "d3/D3_PREDICTION_ACCEPTANCE.json",
        "RESULT/CHECKSUMS.sha256": RESULT / "CHECKSUMS.sha256",
        "RESULT/01_SOURCE_ACCEPTANCE.md": RESULT / "01_SOURCE_ACCEPTANCE.md",
        "RESULT/02_SCIENTIFIC_RESULTS.zh-CN.md": RESULT / "02_SCIENTIFIC_RESULTS.zh-CN.md",
        "RESULT/D1_SOURCE_GROUP_QUERY_TABLE.csv": RESULT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv",
        "RESULT/D1_VISUAL_GAIN_EXTENSIONS.csv": RESULT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv",
        "RESULT/D2_EVENT_REANALYSIS_V2_SUMMARY.json": RESULT / "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json",
        "RESULT/D2_EVENT_REGISTRY_V2.csv": RESULT / "results/D2_EVENT_REGISTRY_V2.csv",
        "RESULT/D2_DENSE_RESULTS_V2.csv": RESULT / "results/D2_DENSE_RESULTS_V2.csv",
        "RESULT/D5_GO_NO_GO.md": RESULT / "d5/D5_GO_NO_GO.md",
        "RESULT/D4_BUNDLE_ACCEPTANCE.csv": RESULT / "d4/D4_BUNDLE_ACCEPTANCE.csv",
        "V3/qa/CHECKSUMS.sha256": V3 / "qa/CHECKSUMS.sha256",
        "V3/MODEL_LOSS_CONTRACT.md": V3 / "MODEL_LOSS_CONTRACT.md",
        "V3/PROTOCOL_LOCKED.json": V3 / "PROTOCOL_LOCKED.json",
        "V3/data/V3_MAIN_INPUTS.npz": V3 / "data/V3_MAIN_INPUTS.npz",
        "V3/data/V3_NORMALIZERS.json": V3 / "data/V3_NORMALIZERS.json",
        "V3/data/V3_VIDEO_NORMALIZER.json": V3 / "data/V3_VIDEO_NORMALIZER.json",
        "V3/code/train_v3.py": V3 / "code/train_v3.py",
        "V3/code/infer_v3.py": V3 / "code/infer_v3.py",
        "V3/ANCHOR_MANIFEST.csv": V3 / "ANCHOR_MANIFEST.csv",
        "V3_STATE/qa/CHECKSUMS.sha256": V3_STATE / "qa/CHECKSUMS.sha256",
        "V3_STATE/RETURN_TO_CHATGPT.zh-CN.md": V3_STATE / "RETURN_TO_CHATGPT.zh-CN.md",
        "D5/qa/CHECKSUMS.sha256": D5 / "qa/CHECKSUMS.sha256",
        "D5/MODE_IMBALANCE_AUDIT.json": D5 / "analysis/MODE_IMBALANCE_AUDIT.json",
        "D5/ROUTER_EXECUTION_DECISION.json": D5 / "analysis/ROUTER_EXECUTION_DECISION.json",
        "D5/COMPONENT_ORACLE_RESULTS.json": D5 / "analysis/COMPONENT_ORACLE_RESULTS.json",
        "D2_V3/raw_action_hashes.csv": ROOT / "d2_v3_measurement/D2_V3_RAW_ACTION_INPUT_HASHES.csv",
        "D2_V3/train_threshold_lock.json": ROOT / "d2_v3_measurement/D2_V3_PROTOCOL_LOCKED.json",
        "D2_V3/matcher_lock.json": ROOT / "d2_v3_measurement/D2_V3_MATCHER_LOCKED_ATTEMPT_003.json",
        "D2_V3/runner_lock.json": ROOT / "d2_v3_measurement/D2_V3_RUNNER_LOCKED_ATTEMPT_006.json",
    }
    source_hashes = {}
    for label, path in sources.items():
        if path.is_file():
            source_hashes[label] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            source_hashes[label] = {"path": str(path), "sha256": None, "status": "MISSING"}
    write_json(ROOT / "SOURCE_HASHES.json", {
        "generated_utc": now,
        "hash_algorithm": "SHA-256",
        "sources": source_hashes,
        "raw_action_source_list": raw_hash_summary,
        "final_test_access": 0,
    })

    # Root integrity summary. A listed-entry pass is stated separately from
    # source-manifest coverage; V3_STATE has 14 pre-existing pyc omissions.
    root_integrity = {
        "BENCH_ROOT": {"manifest_path": str(BENCH / "CHECKSUMS.sha256"), "entries": manifest_lines(BENCH / "CHECKSUMS.sha256"), "verification": "PASS"},
        "RESULT_ROOT": {"manifest_path": str(RESULT / "CHECKSUMS.sha256"), "entries": manifest_lines(RESULT / "CHECKSUMS.sha256"), "verification": "PASS"},
        "V3_ROOT": {"manifest_path": str(V3 / "qa/CHECKSUMS.sha256"), "entries": manifest_lines(V3 / "qa/CHECKSUMS.sha256"), "verification": "PASS"},
        "V3_STATE_ROOT": {"manifest_path": str(V3_STATE / "qa/CHECKSUMS.sha256"), "entries": 1258, "actual_files_excluding_manifest": 1272, "unlisted": 14, "unlisted_files": "generated __pycache__/*.pyc", "verification": "PASS_LISTED_ENTRIES_COVERAGE_INCOMPLETE"},
        "D5_DIAGNOSTIC_ROOT": {"manifest_path": str(D5 / "qa/CHECKSUMS.sha256"), "entries": manifest_lines(D5 / "qa/CHECKSUMS.sha256"), "verification": "PASS"},
    }

    source_registry_path = ROOT / "SOURCE_REGISTRY.json"
    source_registry = json_file(source_registry_path)
    source_registry["final_readonly_verification_utc"] = now
    source_registry["historical_root_integrity"] = root_integrity
    source_registry["d2_v3_raw_inputs"] = raw_hash_summary
    write_json(source_registry_path, source_registry)

    # Update handoff and machine status after all science/report results are final.
    status = {
        "task": "Core B D3 contract audit, D1 seed audit, D2_V3 event sensitivity, D5 decision preservation",
        "created_utc": "2026-09-24T07:22:24Z",
        "updated_utc": now,
        "NEXT_STATUS": "partial",
        "D3_DEVIATION": "unresolved",
        "D3_ATTACHMENT_TARGET_EQUIVALENCE": "confirmed_non_equivalent",
        "D3_HISTORICAL_LOCK_CONFORMANCE": "unresolved because episode-equal weighting is absent from prior lock",
        "D3_AFFECTED_OLD_RUNS": "36/36 effective old cells share the audited non-equivalent implementation relative to attachment target",
        "D3_OLD_EFFECTIVE_TRAIN_CHECKPOINT_PREDICTION": "36/36; one preserved KeyboardInterrupt attempt replaced by accepted RETRY1",
        "D3R_STAGE0_ACCEPTED": "0/12; paused pending contract reconciliation",
        "D3R_TRAINING_ACCEPTED": "0/36; not_started",
        "D3R_PRIMARY_PREDICTIONS_ACCEPTED": "0/36; not_started",
        "D3R_DENSE_PREDICTIONS_ACCEPTED": "not_run_p0_contract_unresolved",
        "D3R_MAIN_CONTRASTS_AND_CI": "not_run",
        "D3R_FACTORIAL_INTERACTION_AND_CI": "not_run",
        "V3_SEED_ANOMALY_CLASSIFICATION": "training_instability_plausible; artifact_error_not_found; data_concentration_not_primary; causal_status_unresolved",
        "D2_V2_HAND_COMPLETE_EVENTS": "5 per horizon; theta_off=0; exact q==0 dependency audited",
        "D2_V3_TRAIN_EVENTS": {"Hand": 9276, "Arm": 6156, "source_groups": 2712, "tasks": 120},
        "D2_V3_DEV_EVENTS": {"Hand": 2093, "Arm": 1411, "source_groups": 603},
        "D2_V3_PANEL_OPPORTUNITIES": {"Hand_H30": 238, "Hand_H60": 259, "Arm_H30": 174, "Arm_H60": 190, "panel_source_groups": 118, "queries": 3718, "exposure_hours": 0.2496111027399699},
        "D2_V3_STATUS": "completed_exploratory",
        "D2_SUPPORT_EXPANSION_PROPOSAL": "not_required_hand_support_gate_passed",
        "D2_V3_DENSE_BUNDLES_ACCEPTED": "36/36",
        "D2_V3_RESULT_ROWS": 72,
        "D2_V3_MATCH_QA": "PASS",
        "D5_STATUS": "redesign_no_training",
        "D5_GT_MODE_COUNTS": {"train": [174, 2567], "dev": [41, 573], "old_proto_dev_argmax": [29, 585]},
        "D5_MODE_COLLAPSE": "unresolved_not_proven_by_argmax_prevalence",
        "D5_CALIBRATION_AND_CANDIDATE_GATES": "fail; router_entry_gate=false; router_fits=0",
        "OLD_SEALED_ROOT_WRITES": 0,
        "FINAL_HELDOUT_ACCESS": 0,
        "PROJECT_GPU_CHILDREN_MAX": 0,
        "GPU_STATE_QUERY": "not_run",
        "NEW_GPU_HOURS": 0,
        "HISTORICAL_ROOT_INTEGRITY": root_integrity,
        "CHECKSUMS": "pending_checksum_last",
        "UNRESOLVED": [
            "The new episode-equal objective in the task attachment differs from the historical MODEL_LOSS_CONTRACT/D3_PROTOCOL_LOCKED objective. User direction is required before D3-R patch/training.",
            "V3_STATE checksum manifest passes its 1258 listed entries but omits 14 pre-existing generated pyc files; source-manifest coverage is incomplete."
        ],
    }
    write_json(ROOT / "STATUS.json", status)

    return_json = json_file(ROOT / "RETURN_TO_CHATGPT.json")
    return_json.update({key: status[key] for key in (
        "NEXT_STATUS", "D3_DEVIATION", "D3_ATTACHMENT_TARGET_EQUIVALENCE", "D3_HISTORICAL_LOCK_CONFORMANCE",
        "D3_AFFECTED_OLD_RUNS", "D3_OLD_EFFECTIVE_TRAIN_CHECKPOINT_PREDICTION", "D3R_STAGE0_ACCEPTED",
        "D3R_TRAINING_ACCEPTED", "D3R_PRIMARY_PREDICTIONS_ACCEPTED", "D3R_DENSE_PREDICTIONS_ACCEPTED",
        "V3_SEED_ANOMALY_CLASSIFICATION", "D2_V2_HAND_COMPLETE_EVENTS", "D2_V3_TRAIN_EVENTS", "D2_V3_DEV_EVENTS",
        "D2_V3_PANEL_OPPORTUNITIES", "D2_V3_STATUS", "D2_SUPPORT_EXPANSION_PROPOSAL", "D2_V3_DENSE_BUNDLES_ACCEPTED",
        "D2_V3_RESULT_ROWS", "D2_V3_MATCH_QA", "D5_STATUS", "D5_GT_MODE_COUNTS", "D5_MODE_COLLAPSE",
        "D5_CALIBRATION_AND_CANDIDATE_GATES", "OLD_SEALED_ROOT_WRITES", "FINAL_HELDOUT_ACCESS",
        "PROJECT_GPU_CHILDREN_MAX", "GPU_STATE_QUERY", "NEW_GPU_HOURS", "UNRESOLVED"
    )})
    return_json["CHECKSUMS"] = "pending_checksum_last"
    write_json(ROOT / "RETURN_TO_CHATGPT.json", return_json)

    (ROOT / "RETURN_TO_CHATGPT.zh-CN.md").write_text(
        "# Core B 下一阶段审计交付\n\n"
        "总体状态：`partial`。P2 与 D2_V3 的开发审计已完成；P0 确认旧 D3 代码不等价于新任务附件中的 episode-equal 公式，但原 `MODEL_LOSS_CONTRACT.md` / `D3_PROTOCOL_LOCKED.json` 没有写入 episode 等权。附件又明确要求目标不一致时暂停，因此没有擅自修约或启动 D3-R。\n\n"
        "## 关键结果\n\n"
        "- P0：train-only 数据为 17,757 windows / 2,712 episodes；实际每个64样本 batch 都混合 episode。真实首批 legacy loss `0.285918`，附件目标 `0.288485`，梯度差 `-0.001421`；32+32正确加权梯度逐参数一致，29样本尾批除以2会缩放梯度。旧 D3 有效训练/ckpt/预测 36/36，其中一格保留了 KeyboardInterrupt 失败和 RETRY1 成功。\n"
        "- P2：24个对比格按 anchor ID 重算，与封存总体 D1 MAE/FD 表最大差 `7.2e-10`，checkpoint hashes 和 ID/frame/timestamp 对齐均通过。seed 20260951/H30 的 F-TOK-D 高分歧 Arm/Hand gain `-0.029802/-0.027453`，而 F-TOK-S/F-POOL-D 不同样退化；对应 F-TOK-D train loss也高。seed 20260952/H60 的 S-HIST train loss偏高、预测幅度收缩，所有视频类模型相近改善。与训练不稳定/seed×condition 效应相符，少数任务不是主要支配因素；因每格单 seed且dev已反复使用，因果归因仍未确定。\n"
        "- D2_V3：先用 train-only 的 source-group 等权移动Q75冻结阈值，再锁0.2s簇与一对一匹配并评估已有36个dense bundles。Hand H30/H60 有238/259个完整机会（95/99 tasks），满足预注册支持门；Arm有174/190（106/107 tasks）。但 F-TOK-D Hand recall仍接近0，Arm recall约11–16%而误报率约200–990/hour，0.2496小时的短暴露不代表部署率。无需扩展panel。\n"
        "- D5：保持 `redesign_no_training`。GT minority train/dev为174/2741与41/614，旧argmax预测多数mode为585/614；类别不平衡和minority underprediction一致，但 mode collapse 未证实。CAL/candidate gate fail、router entry gate=false、router fits=0；fixed-q FD oracle比Persistence差2.2886%。\n\n"
        "## 需要用户裁决\n\n"
        "请确认：是否把附件的 episode-equal 逐样本损失视为对历史 D3 锁的正式修订？如确认，下一步再按全36格 D3-R 协议推进并单独申请宿主GPU审批；如不确认，请指定以哪份历史损失合同为准。当前没有GPU子进程，GPU状态未查询。\n\n"
        "旧实验根写入数为0，final/held-out访问为0；V3_STATE旧manifest的1258条列出hash通过，但有14个既有生成pyc未列入。详见 `BENCHMARK_VALIDITY_LEDGER.csv`、`p0_loss_audit/`、`p2_seed_audit/`、`d2_v3_measurement/` 和 `d5_decision/`。\n",
        encoding="utf-8",
    )

    qa = {
        "status": "PASS_WITH_CONTRACT_BLOCKER",
        "P0_toy_and_train_fixture": "PASS",
        "P0_effective_old_D3_lineage": "36/36 with preserved retry",
        "P2_V3_seed_recompute": "PASS 24/24; stored overall metrics match to <1e-9",
        "P2_figures": "4 PNG/PDF pairs; visually reviewed; metadata recorded in P2_FIGURE_QA.md",
        "D2_V3_bundles": "PASS 36/36",
        "D2_V3_matcher_boundary_cluster_one_to_one_QA": "PASS",
        "D2_V3_support_gate": "PASS; no expansion proposal required",
        "D5_reaudit": "PASS; redesign/no training preserved",
        "historical_root_checksums": root_integrity,
        "old_roots_modified": False,
        "final_test_access": 0,
        "new_GPU_jobs": 0,
        "unresolved_blocker": "Historical D3 lock does not specify the attachment's episode-equal target; user decision required before D3-R.",
    }
    (ROOT / "qa").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "qa/FINAL_ACCEPTANCE.json", qa)

    # Hash every file in this new root except the checksum manifest itself.
    checksum_path = ROOT / "CHECKSUMS.sha256"
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != checksum_path and not path.is_symlink())
    lines = [f"{sha256_file(path)}  ./{path.relative_to(ROOT).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    status["CHECKSUMS"] = f"PASS; {len(lines)} files hashed checksum-last"
    write_json(ROOT / "STATUS.json", status)
    # STATUS is sealed too: update checksum manifest one final time after the
    # status timestamp/count fields are final, without changing other files.
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != checksum_path and not path.is_symlink())
    lines = [f"{sha256_file(path)}  ./{path.relative_to(ROOT).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"NEXT_STATUS": status["NEXT_STATUS"], "checksum_entries": len(lines), "P0": p0["status"], "P2_cells": p2["accepted_model_cells_recomputed"], "D2V3": d2["D2_V3_STATUS"], "D2QA": d2qa["status"], "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

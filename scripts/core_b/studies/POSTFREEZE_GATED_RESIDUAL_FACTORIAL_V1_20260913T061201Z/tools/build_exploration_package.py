from __future__ import annotations
import os

import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from models import parameter_count, split_hidden_for_parameter_match


EXP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
JY02_ROOT = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
STAGE2_ATTEMPT = JY02_ROOT / "runs/stage2_authorized_20260912T151855Z"
FREEZE_ROOT = REPO_ROOT / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z"
V2_ROOT = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
INPUT_PATH = JY02_ROOT / "inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
THRESHOLD_PATH = V2_ROOT / "MOTION_THRESHOLD_SOURCE.json"
MEETING_ROOT = PROJECT_ROOT / "professor_meeting/CORE_B_JY02_DECISIONS_20260913T061201Z"
SEED = 20260924
UTC_NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

CELL_SPECS = {
    "E-S-SH-U": {"input": "STATE", "head_coupling": "shared", "residual_gate": "ungated", "input_dim": 568},
    "E-S-SH-G": {"input": "STATE", "head_coupling": "shared", "residual_gate": "group_time_gated", "input_dim": 568},
    "E-S-SP-U": {"input": "STATE", "head_coupling": "split", "residual_gate": "ungated", "input_dim": 568},
    "E-S-SP-G": {"input": "STATE", "head_coupling": "split", "residual_gate": "group_time_gated", "input_dim": 568},
    "E-F-SH-U": {"input": "FUSION", "head_coupling": "shared", "residual_gate": "ungated", "input_dim": 1336},
    "E-F-SH-G": {"input": "FUSION", "head_coupling": "shared", "residual_gate": "group_time_gated", "input_dim": 1336},
    "E-F-SP-U": {"input": "FUSION", "head_coupling": "split", "residual_gate": "ungated", "input_dim": 1336},
    "E-F-SP-G": {"input": "FUSION", "head_coupling": "split", "residual_gate": "group_time_gated", "input_dim": 1336},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_text(relative: str, value: str) -> None:
    path = EXP_ROOT / relative if not relative.startswith("MEETING::") else MEETING_ROOT / relative[9:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(relative: str, value: object, *, meeting: bool = False) -> None:
    path = (MEETING_ROOT if meeting else EXP_ROOT) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(relative: str, rows: list[dict], fields: list[str], *, meeting: bool = False) -> None:
    path = (MEETING_ROOT if meeting else EXP_ROOT) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze() -> dict:
    manifest = FREEZE_ROOT / "qa/CHECKSUMS.sha256"
    result = subprocess.run(["sha256sum", "-c", "qa/CHECKSUMS.sha256"], cwd=FREEZE_ROOT, capture_output=True, text=True, check=False)
    lines = (result.stdout + result.stderr).splitlines()
    ok = sum(line.endswith(": OK") for line in lines)
    failed = sum(line.endswith(": FAILED") for line in lines)
    if result.returncode != 0 or failed or ok != 85:
        raise RuntimeError(f"freeze checksum failed: exit={result.returncode} ok={ok} failed={failed}")
    return {"status": "PASS", "manifest": str(manifest), "manifest_sha256": sha256(manifest), "ok": ok, "failed": failed}


def build_matrix() -> list[dict]:
    rows = []
    for index, (cell_id, spec) in enumerate(CELL_SPECS.items()):
        hidden = 256 if spec["head_coupling"] == "shared" else split_hidden_for_parameter_match(spec["input_dim"])[0]
        count = parameter_count(spec["input_dim"], head_coupling=spec["head_coupling"], residual_gate=spec["residual_gate"], hidden_dim=hidden)
        shared_count = parameter_count(spec["input_dim"], head_coupling="shared", residual_gate="ungated", hidden_dim=256)
        rows.append({"cell_id": cell_id, **spec, "hidden_dim": hidden, "parameter_count": count, "shared_ungated_parameter_count": shared_count, "relative_parameter_delta_vs_shared_ungated": (count - shared_count) / shared_count, "seed": SEED, "cell_index": index, "status": "locked_before_results"})
    return rows


def build_teacher_material() -> None:
    short = """老师好，JY-02完成12/12：F-R-D Arm-14 MAE=0.1164±0.0003，相对Persistence改善约8.3%，属development候选；Hand/Full、一阶差分和TV仍不足。做8格单seed探索（STATE/FUSION×shared/split×ungated/gated residual），仅用cached train/dev，不碰final-test，不改冻结候选/协议。请确认：1 Arm-14作primary，Hand/Full作gate还是secondary？2 先blind sanitization再评估602条final，是否同意？3 仅episode split的claim边界可否接受？4 需不需要外部或group-holdout？"""
    detailed = """JY-02 Stage2 已完成 12/12：F-R-D 在三个 fresh seed 上都优于 Persistence，Arm-14 MAE 为 0.1164±0.0003，约改善 8.3%，但 Hand/Full-28、first-difference 与 TV 仍暴露出不足。下一步仅做 post-freeze exploratory：8 格 STATE/FUSION × shared/split Arm+Hand head × ungated/persistence-gated residual，固定 direct residual、1s/30×28、cached features 和训练合同，单一新 seed，结果不会替换冻结 F-R-D 或进入 pending final protocol。视频增量仍有 task/scene shortcut caveat；final 评估将固定 12 checkpoints+Persistence，不挑最佳 seed。当前只报告 command trajectory，command units、URDF/FK 未确认。请逐项确认：Arm-14 primary 与 Hand/Full gate/secondary；blind sanitization 后一次性评估 602 条 final primary；episode-level split 的 claim 边界；是否需要外部或 group-holdout 验证。"""
    questions = [
        "Arm-14 是否作为 primary，Hand/Full-28 是 hard gate 还是 secondary？",
        "是否批准先完成 blind sanitization，再一次性评估 602 条 final primary population？",
        "只有 episode-level split、没有 session/operator/scene metadata 时，最终 claim 边界是否可接受？",
        "是否需要外部数据集或 source-grouped holdout 验证？",
    ]
    write_text("MEETING::TEACHER_GROUP_MESSAGE_SHORT.zh-CN.md", short)
    write_text("MEETING::TEACHER_GROUP_MESSAGE_DETAILED.zh-CN.md", detailed)
    write_text("MEETING::TEACHER_DECISION_CHECKLIST.zh-CN.md", "# 老师决策清单\n\n" + "\n".join(f"- [ ] {item}" for item in questions) + "\n\n未发送外部消息：`TEACHER_MESSAGE_SENT=false`。")
    facts = [
        {"fact_id": "F1", "fact": "Stage2 accepted 12/12", "value": "4 configs x 3 fresh seeds", "source": str(JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json"), "evidence": "accepted_artifact"},
        {"fact_id": "F2", "fact": "F-R-D Arm-14 development MAE", "value": "0.116375070 +/- 0.000299322", "source": str(JY02_ROOT / "STAGE2_RESULTS_BY_SEED.json"), "evidence": "development_only"},
        {"fact_id": "F3", "fact": "F-R-D vs Persistence hierarchical CI", "value": "[0.007655774, 0.013557487]", "source": str(JY02_ROOT / "STAGE2_HIERARCHICAL_STATISTICS.json"), "evidence": "development_only"},
        {"fact_id": "F4", "fact": "Final boundary", "value": "final-test access 0; not sent", "source": str(FREEZE_ROOT / "BLIND_DATA_BOUNDARY.json"), "evidence": "locked_boundary"},
    ]
    write_csv("SOURCE_FACTS.csv", facts, ["fact_id", "fact", "value", "source", "evidence"], meeting=True)
    write_json("SOURCE_FACTS.json", {"status": "PASS", "rows": facts, "teacher_message_sent": False}, meeting=True)
    write_text("MEETING::README.zh-CN.md", "# Core B JY-02 老师决策材料\n\n仅生成、不发送。短消息可直接复制；详细材料保留证据边界。")


def build() -> None:
    freeze_check = verify_freeze()
    matrix = build_matrix()
    protocol_id = "core_b_postfreeze_gated_residual_factorial_v1_20260913"
    input_hash = sha256(INPUT_PATH)
    normalizer_hash = sha256(NORMALIZER_PATH)
    threshold_hash = sha256(THRESHOLD_PATH)
    freeze_manifest_hash = sha256(FREEZE_ROOT / "CANDIDATE_FREEZE_MANIFEST.json")
    stage2_manifest_hash = sha256(STAGE2_ATTEMPT / "qa/CHECKSUMS.sha256")
    hidden_state, split_state = split_hidden_for_parameter_match(568)
    hidden_fusion, split_fusion = split_hidden_for_parameter_match(1336)

    write_text("README.zh-CN.md", f"""# JY-02 Post-freeze Persistence-gated residual factorial exploration

状态：`LOCKED_BEFORE_TRAINING`。这是隔离的 single-seed exploratory sandbox，不是 final evaluation，也不改变 freeze root、pending final protocol 或冻结的 `JY02-F-R-D`。

- 8 格：STATE/FUSION × shared/split Arm+Hand head × ungated/group-time gated residual。
- exploratory seed：`{SEED}`，在结果前锁定；不使用 Stage2/final seeds。
- input：development `JY02_TARGETS.npz`，train/dev=`2741/614`，cached features，no raw-video/LoRA/encoder update。
- GPU：遵循实际 project watcher allowlist GPU2/3；最多 2 个 child；GPU0 不使用。
- final-test：`FINAL_TEST_EXPLICITLY_AUTHORIZED=false`，target access=`0`。

先完成全部 8 格 acceptance，再进行 factorial/paired bootstrap 和 figures。即使出现新候选，也必须另做 fresh-seed confirmation 和新 freeze，不得自动晋级。
""")
    write_text("README.en.md", f"""# JY-02 post-freeze Persistence-gated residual factorial exploration

Status: `LOCKED_BEFORE_TRAINING`. This is an isolated single-seed exploratory sandbox. It is not final evaluation and does not change the frozen `JY02-F-R-D`, freeze root or pending final protocol.

- 8 cells: STATE/FUSION x shared/split Arm+Hand head x ungated/group-time gated residual.
- Exploratory seed: `{SEED}`, locked before results; Stage2/final seeds are not used.
- Input: development `JY02_TARGETS.npz`, train/dev=`2741/614`, cached features, no raw-video/LoRA/encoder update.
- GPU: follow the actual project watcher allowlist GPU2/3; at most 2 children; GPU0 unused.
- Final-test: `FINAL_TEST_EXPLICITLY_AUTHORIZED=false`, target access=`0`.

All eight cells must pass acceptance before factorial/paired bootstrap and figures. Any new candidate requires a separate fresh-seed confirmation and freeze; no automatic promotion.
""")
    protocol = {
        "status": "LOCKED_BEFORE_TRAINING",
        "protocol_id": protocol_id,
        "study_type": "post_freeze_exploratory_full_2x2x2_factorial_single_seed",
        "factors": {"input": ["STATE", "FUSION"], "head_coupling": ["shared", "split_arm_hand"], "residual_gate": ["ungated", "group_time_gated"]},
        "matrix_cells": [row["cell_id"] for row in matrix],
        "exploratory_seed": SEED,
        "seed_locked_before_results": True,
        "input_contract": {"path": str(INPUT_PATH), "sha256": input_hash, "train": 2741, "development": 614, "target_shape": [30, 28], "arm_slice": "14:28", "hand_slice": "0:14", "raw_video": False},
        "normalizer": {"path": str(NORMALIZER_PATH), "sha256": normalizer_hash, "fit_population": "train only"},
        "motion_thresholds": {"path": str(THRESHOLD_PATH), "sha256": threshold_hash, "fit_population": "train only", "q25_q75_locked": True},
        "training_contract": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "physical_batch": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_epochs": 20, "min_epochs": 5, "patience": 5, "min_delta": 0.0001, "gradient_clip_norm": 1.0, "precision": "BF16 autocast; FP32 parameters", "selection_metric": "Arm-14 full-horizon standardized MAE, then Full-28 MAE, then earlier epoch"},
        "prohibitions": ["final-test target/label/prediction access", "final inference", "LoRA", "encoder update", "raw-video extraction", "GRU/Transformer", "additional sweep", "new sampling/horizon", "partial-result early stopping", "automatic multi-seed confirmation", "freeze/final protocol edits"],
        "freeze_references": {"freeze_manifest_sha256": freeze_manifest_hash, "stage2_checksum_manifest_sha256": stage2_manifest_hash},
        "final_test_explicitly_authorized": False,
    }
    write_json("EXPLORATION_PROTOCOL_LOCKED.json", protocol)
    write_text("EXPLORATION_PROTOCOL_LOCKED.md", f"""# Exploration protocol lock

This protocol is locked before the exploratory result is read. It is a complete `2x2x2` factorial screen with one fresh seed `{SEED}`. The experimental unit for inference is a source episode; steps and joints remain nested within episode. All eight cells use the same train/dev IDs, targets, persistence anchors, loader order and train-only normalizer.

The only learned output is a direct residual. Shared heads output `30x28`. Split heads independently output `30x14` for Hand and Arm, then concatenate in canonical Hand[0:14], Arm[14:28] order. A gated cell produces 30x2 group-time gates (Arm, Hand), each in `[0,1]`; it reads only past context and a learned 32-D time embedding.

All eight cells must pass acceptance before any ranking or statistics. This package cannot touch final-test or modify the frozen candidate/final protocol.
""")
    outcome = {"status": "LOCKED_BEFORE_RESULTS", "protocol_id": protocol_id, "written_utc": UTC_NOW, "metrics_thresholds_comparisons_locked": True, "results_read_before_lock": False, "candidate_gate_locked": True, "target_access": 0, "final_test_explicitly_authorized": False}
    write_json("OUTCOME_EXPOSURE.json", outcome)
    write_text("OUTCOME_EXPOSURE.md", "# Outcome exposure\n\nWritten before any exploratory result was read. The matrix, seed, gates, metrics, comparisons, train-derived thresholds, and candidate ordering are fixed. This is not final-test evaluation and no final target is accessible.")
    write_text("EXPERIMENT_RATIONALE.zh-CN.md", """# 实验动机

当前 F-R-D 在 Arm-14 平均误差上优于 Persistence，但 Hand/Full、first-difference 和 TV 仍有不足。本项目级探索假设是：无界 residual 在低置信时间段可能引入无效变化；有界 group-time Persistence gate 可以收缩残差并让 Arm/Hand 解耦，从而减少振荡和负迁移。

这不是 literature novelty 声明，也不是 final model 选择。所有 8 格完整训练；不根据中途结果停止，不增加 gate 类型、hidden、loss、smoothness、sampling、horizon 或 seed。
""")
    fields = ["cell_id", "input", "head_coupling", "residual_gate", "input_dim", "hidden_dim", "parameter_count", "shared_ungated_parameter_count", "relative_parameter_delta_vs_shared_ungated", "seed", "cell_index", "status"]
    write_csv("MATRIX.csv", matrix, fields)
    write_json("MATRIX.json", {"status": "LOCKED", "rows": matrix, "factorial": "2x2x2", "all_cells_required": True})
    seed_registry = {"status": "LOCKED_BEFORE_RESULTS", "selected_seed": SEED, "known_training_seeds": [20260903, 20260921, 20260922, 20260923], "known_nontraining_analysis_seeds": [20260910, 20260911, 20260912, 20260913, 20260930], "project_wide_registry_found": False, "selection_basis": "bounded project scan found no 20260924 occurrence; next unused positive integer after known JY02 training seeds", "must_not_change_after_results": True}
    write_json("SEED_REGISTRY.json", seed_registry)
    write_text("HEAD_CONTRACT.md", f"""# Head coupling contract

- Shared: one direct MLP `input -> 256 -> 840`, reshaped to `(30,28)`.
- Split: independent Hand and Arm MLPs, each `input -> hidden -> 420`, concatenated to `(30,28)` in Hand then Arm order.
- Hidden candidates are fixed `(128,160,192,224,256)` and selected by parameter count only. State split hidden=`{hidden_state}`; Fusion split hidden=`{hidden_fusion}`. No development metric is read for this choice.
- All cells use the same reconstructed absolute full-28 MSE and the same partial-batch weighting.
""")
    write_json("HEAD_CONTRACT.json", {"status": "LOCKED", "shared": {"hidden": 256, "output": [30, 28]}, "split": {"candidate_hidden": [128, 160, 192, 224, 256], "state_hidden": hidden_state, "fusion_hidden": hidden_fusion, "output_groups": {"hand": "0:14", "arm": "14:28"}, "selection_basis": "parameter count only"}})
    gate_contract = {"status": "LOCKED", "formula": "y_hat_std = p_std + gate * r_hat_std", "time_steps": 30, "groups": ["arm", "hand"], "time_embedding_dim": 32, "range": [0, 1], "gate_input": ["past_context", "learned_time_embedding"], "forbidden_input": ["future_target", "prediction_error", "motion_label"], "initialization": {"projection_weight": 0.0, "bias": "logit(0.9)"}, "zero_gate_equals_persistence": True, "unit_gate_equals_ungated": True, "regularizer": None, "saved_per_episode_time_group": True}
    write_json("GATE_CONTRACT.json", gate_contract)
    write_text("GATE_CONTRACT.md", """# Persistence gate contract

For gated cells, `g=sigmoid(gate(context,time_embedding))` produces 30x2 values in [0,1], ordered Arm then Hand. The Arm scalar multiplies residual dimensions 14:28; the Hand scalar multiplies 0:14. The gate sees no future target, prediction error or motion label. Output projection weights start at zero and bias at logit(0.9). Zero gate is exact Persistence; unit gate is exact ungated residual. No gate regularizer or smoothness loss is added.
""")
    budget_rows = []
    for row in matrix:
        budget_rows.append({"cell_id": row["cell_id"], "input": row["input"], "head_coupling": row["head_coupling"], "residual_gate": row["residual_gate"], "hidden_dim": row["hidden_dim"], "trainable_parameters": row["parameter_count"], "shared_ungated_reference": row["shared_ungated_parameter_count"], "relative_delta": row["relative_parameter_delta_vs_shared_ungated"], "within_10pct_split_rule": abs(row["relative_parameter_delta_vs_shared_ungated"]) <= 0.10 if row["head_coupling"] == "split" else True})
    write_csv("PARAMETER_BUDGET.csv", budget_rows, list(budget_rows[0].keys()))
    write_json("PARAMETER_BUDGET.json", {"status": "PASS_LOCKED", "rows": budget_rows, "split_match_tolerance": 0.10})
    training_contract = {"status": "LOCKED", "source": str(JY02_ROOT / "RESOLVED_TRAINING_CONTRACT.json"), "exploratory_seed": SEED, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "physical_batch": 8, "gradient_accumulation": 4, "nominal_effective_batch": 32, "scheduler": "CosineAnnealingLR", "max_epochs": 20, "min_epochs": 5, "patience": 5, "min_delta": 0.0001, "gradient_clip_norm": 1.0, "precision": "BF16 autocast; FP32 parameters", "selection_metric": "Arm-14 full-horizon standardized MAE then Full-28 MAE then earlier epoch", "encoder_or_lora_modified": False, "raw_video": False, "final_test_access": 0}
    write_json("RESOLVED_TRAINING_CONTRACT.json", training_contract)
    comparisons = ["E-F-SP-G_vs_E-F-SH-U", "E-S-SP-G_vs_E-S-SH-U", "E-F-SH-G_vs_E-F-SH-U", "E-F-SP-U_vs_E-F-SH-U", "E-F-SP-G_vs_E-S-SP-G", "E-F-SP-G_vs_PERSISTENCE"]
    write_json("COMPARISON_FAMILIES.json", {"status": "LOCKED", "primary_comparisons": comparisons, "metric": "Arm-14 standardized full-horizon MAE", "effect": "baseline MAE - candidate MAE", "bootstrap_replicates": 10000, "episode_unit": True, "holm_family": "fixed six-comparison family; exploratory CI/effect reporting"})
    write_text("COMPARISON_FAMILIES.md", "# Comparison family\n\nThe six predeclared comparisons are:\n\n1. FUSION split-gated vs FUSION shared-ungated.\n2. STATE split-gated vs STATE shared-ungated.\n3. FUSION shared-gated vs FUSION shared-ungated.\n4. FUSION split-ungated vs FUSION shared-ungated.\n5. FUSION split-gated vs STATE split-gated.\n6. FUSION split-gated vs Persistence.\n\nEpisode is the paired unit. Use 10,000 paired bootstrap replicates and the fixed Holm family if p-values are reported. Single-seed results are exploratory and are not seed robustness.")
    write_json("RESOURCE_PLAN.json", {"status": "LOCKED", "task_local_authorized_gpu_pool": [1, 2, 3], "actual_watcher_allowlist": [2, 3], "gpu0_authorized": False, "maximum_concurrent_project_gpu_children": 2, "max_children_per_gpu": 1, "idle_gate": {"utilization_percent_le": 5, "memory_mib_le": 512, "consecutive_samples": 3, "final_recheck": True}, "non_child_process_signaled": False, "lease_dir": str(REPO_ROOT / "reports/locks/gpu_lease"), "outputs_isolated": True})
    write_text("RESOURCE_PLAN.md", "# Resource plan\n\nThe user prompt names GPU1/2/3, but the actual project watcher policy allowlist is GPU2/3; the narrower policy governs. At most two project children run, one per GPU, with three idle samples, final recheck, atomic lease and no non-child signals. GPU0 is not used. Cached-feature training does not perform raw-video I/O.")
    write_csv("RESOURCE_TIMELINE.csv", [], ["utc_timestamp", "phase", "status", "physical_gpu", "uuid", "utilization_percent", "memory_used_mib", "memory_total_mib", "watcher_status", "lease_released", "notes"])
    for directory in ["checkpoints", "predictions", "logs", "statistics", "figure_data", "figures/png", "figures/svg", "figures/pdf", "figure_manifests"]:
        path = EXP_ROOT / directory / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Reserved for this isolated exploratory root; no final-test or frozen-final artifact is stored here.\n", encoding="utf-8")
    # Snapshot the policy watcher and accepted reference scripts without changing their sources.
    watcher_source = REPO_ROOT / "tools/gpu_lease_watcher.py"
    if watcher_source.is_file(): shutil.copy2(watcher_source, EXP_ROOT / "code/gpu_lease_watcher.py")
    for source_name in ["direct_chunk_decoder.py", "temporal_gru_decoder.py", "train_jy02_cell.py"]:
        source = JY02_ROOT / "code/models" / source_name if source_name.endswith("decoder.py") else JY02_ROOT / "code" / source_name
        if source.is_file(): shutil.copy2(source, EXP_ROOT / "code" / ("stage2_reference_" + source_name))
    for row in matrix:
        write_json(f"configs/{row['cell_id']}.json", {"protocol_id": protocol_id, **row, "final_test_access": 0, "fresh_initialization": True, "checkpoint_reuse": False})
    build_teacher_material()


if __name__ == "__main__":
    build()

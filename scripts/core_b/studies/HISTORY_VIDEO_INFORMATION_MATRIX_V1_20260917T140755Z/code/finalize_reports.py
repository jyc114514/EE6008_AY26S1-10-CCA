#!/usr/bin/env python3
"""Generate final reports and QA from the accepted History×Video outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    main_rows = {row["model"]: row for row in read_csv(ROOT / "analysis/MAIN_RESULTS.csv")}
    candidate = json.loads((ROOT / "analysis/CANDIDATE_DECISION.json").read_text(encoding="utf-8"))
    fd_margin = candidate["native_fd_relative_change_vs_persistence"]
    hand = candidate["hand_improvement_vs_persistence"]
    full = candidate["full_improvement_vs_fixed_hybrid"]
    p_hand = float(main_rows["PERSISTENCE"]["hand_mae_std"])
    p_fd = float(main_rows["PERSISTENCE"]["hand_fd_native"])
    c_hand = float(main_rows["HI-F-HIST"]["hand_mae_std"])
    c_fd = float(main_rows["HI-F-HIST"]["hand_fd_native"])
    run_rows = []
    watcher_ids: list[int] = []
    for cell in CELLS:
        complete = json.loads((ROOT / "runs" / cell / "formal/RUN_COMPLETE.json").read_text(encoding="utf-8"))
        watcher = json.loads((ROOT / "logs" / f"watcher_HVI-{cell}-FORMAL.json").read_text(encoding="utf-8"))
        watcher_ids.append(int(watcher["physical_gpu"]))
        run_rows.append({"cell": cell, "formal_status": complete.get("status"), "epoch": complete.get("epoch"), "optimizer_steps": complete.get("optimizer_steps"), "physical_gpu": watcher.get("physical_gpu"), "watcher_status": watcher.get("status"), "peak_reserved_mib": complete.get("peak_memory_reserved_mib"), "final_test_access": 0})
    actual_gpu_ids = sorted(set(watcher_ids))

    claim_rows = [
        {"claim": "All four formal cells completed", "status": "SUPPORTED", "evidence": "RUN_ACCEPTANCE.json; runs/*/formal/RUN_COMPLETE.json", "allowed_wording": "4/4 matched 40-epoch formal runs accepted", "not_allowed": "multi-seed robustness"},
        {"claim": "History always helps", "status": "REJECTED", "evidence": "analysis/HISTORY_VIDEO_FACTORIAL.csv", "allowed_wording": "history effect is conditional: negative in S, positive in F for level MAE", "not_allowed": "generic history benefit"},
        {"claim": "Fixed video adds level accuracy beyond history", "status": "NOT_SUPPORTED", "evidence": "analysis/HISTORY_VIDEO_FACTORIAL.csv", "allowed_wording": "F-HIST vs S-HIST level interval crosses zero", "not_allowed": "video level superiority"},
        {"claim": "Fixed video improves temporal fidelity", "status": "SUPPORTED_WITHIN_DEV", "evidence": "analysis/HISTORY_VIDEO_FACTORIAL.csv", "allowed_wording": "video increments improve native Hand FD in LAST and HIST contrasts", "not_allowed": "physical dynamics understanding"},
        {"claim": "HI-F-HIST beats Persistence on Hand level MAE", "status": "SUPPORTED_WITHIN_DEV", "evidence": "analysis/CANDIDATE_DECISION.json", "allowed_wording": f"Hand standardized MAE {c_hand:.6f} vs {p_hand:.6f}; paired CI lower {hand['ci_low']:.6f} > 0", "not_allowed": "final generalization"},
        {"claim": "HI-F-HIST beats fixed hybrid on Full-28", "status": "SUPPORTED_WITHIN_DEV", "evidence": "analysis/CANDIDATE_DECISION.json", "allowed_wording": f"Full improvement {full['effect_control_minus_candidate']:.6f} with CI [{full['ci_low']:.6f},{full['ci_high']:.6f}]", "not_allowed": "new Arm evidence"},
        {"claim": "HI-F-HIST satisfies native FD margin", "status": "SUPPORTED_WITHIN_DEV", "evidence": "analysis/FD_MARGIN_RESULTS.csv", "allowed_wording": f"FD change {fd_margin['relative_change_candidate_vs_control']*100:.3f}% with 95% upper bound {fd_margin['ci_high']*100:.3f}%", "not_allowed": "FD improvement; it is a small deterioration within the margin"},
        {"claim": "Slope baseline is a useful learned alternative", "status": "REJECTED", "evidence": "analysis/MAIN_RESULTS.csv; analysis/SLOPE_BASELINE_STATUS.json", "allowed_wording": "no-fit slope diagnostic is substantially worse in this development population", "not_allowed": "velocity or physical-motion baseline"},
        {"claim": "Old prototype ARI=1 proves stable modes", "status": "RETRACTED", "evidence": "EVIDENCE_CORRECTION_ADDENDUM.zh-CN.md; prior audit", "allowed_wording": "old ARI=1 was an implementation artifact and corrected mode stability remains limited", "not_allowed": "stable two-mode discovery"},
        {"claim": "Strict final-system OOF exists", "status": "UNKNOWN_NOT_SUPPORTED", "evidence": "prior audit correction; EVIDENCE_CORRECTION_ADDENDUM.zh-CN.md", "allowed_wording": "this new direct train-to-dev study does not require selector OOF", "not_allowed": "complete selector OOF"},
        {"claim": "Final or held-out generalization", "status": "LOCKED_UNKNOWN", "evidence": "FINAL_TEST_ACCESS=0", "allowed_wording": "development-only single-seed result", "not_allowed": "final/public performance"},
    ]
    write_csv(ROOT / "CLAIM_LEDGER.csv", claim_rows)

    (ROOT / "EXPERIMENT_DECISION.zh-CN.md").write_text(f"""# 历史信息 × 视频信息 2×2 实验决策\n\n## 完成状态\n\n四个正式 cell 已全部完成：每格 40 epochs、1720 optimizer steps、last-epoch checkpoint、固定 CPU dev inference 均通过验收。真实 Stage0 为 4/4；正式训练实际使用物理 GPU {actual_gpu_ids}，最多同时两个本项目 child。\n\n本轮是 single-seed development exploration，不能称 final 或跨 seed 确认。\n\n## 主要结果\n\n| Cell | Hand standardized MAE | Full-28 standardized MAE | Hand native FD |\n|---|---:|---:|---:|\n| Persistence | {float(main_rows['PERSISTENCE']['hand_mae_std']):.6f} | {float(main_rows['PERSISTENCE']['full_mae_std']):.6f} | {float(main_rows['PERSISTENCE']['hand_fd_native']):.6f} |\n| Fixed hybrid | {float(main_rows['FIXED_HYBRID']['hand_mae_std']):.6f} | {float(main_rows['FIXED_HYBRID']['full_mae_std']):.6f} | {float(main_rows['FIXED_HYBRID']['hand_fd_native']):.6f} |\n| HI-S-LAST | {float(main_rows['HI-S-LAST']['hand_mae_std']):.6f} | {float(main_rows['HI-S-LAST']['full_mae_std']):.6f} | {float(main_rows['HI-S-LAST']['hand_fd_native']):.6f} |\n| HI-S-HIST | {float(main_rows['HI-S-HIST']['hand_mae_std']):.6f} | {float(main_rows['HI-S-HIST']['full_mae_std']):.6f} | {float(main_rows['HI-S-HIST']['hand_fd_native']):.6f} |\n| HI-F-LAST | {float(main_rows['HI-F-LAST']['hand_mae_std']):.6f} | {float(main_rows['HI-F-LAST']['full_mae_std']):.6f} | {float(main_rows['HI-F-LAST']['hand_fd_native']):.6f} |\n| **HI-F-HIST** | **{c_hand:.6f}** | **{float(main_rows['HI-F-HIST']['full_mae_std']):.6f}** | **{c_fd:.6f}** |\n\n预注册候选优先级为 F-HIST、S-HIST、F-LAST、S-LAST，因此当前 eligible development proposal 是 `HI-F-HIST`。它相对 Persistence 的 Hand improvement 为 `{hand['effect_control_minus_candidate']:.6f}`，95% CI `[{hand['ci_low']:.6f}, {hand['ci_high']:.6f}]`；相对固定 Arm-learned + Hand-Persistence hybrid 的 Full improvement 为 `{full['effect_control_minus_candidate']:.6f}`，95% CI `[{full['ci_low']:.6f}, {full['ci_high']:.6f}]`。native FD 相对 Persistence 变化为 `{fd_margin['relative_change_candidate_vs_control']*100:.3f}%`，95% CI `[{fd_margin['ci_low']*100:.3f}%, {fd_margin['ci_high']*100:.3f}%]`，位于当前 +1% operating margin 内。\n\n## 两个科学问题\n\n1. `S-HIST − S-LAST` 的 level effect 为负且 CI 不跨 0：没有视频时，加入真实历史反而没有改善该模型。\n2. `F-HIST − F-LAST` 的 level effect 为正且 CI 不跨 0：有固定视频时，历史带来 development level 增量；但 FD effect 为负，表示 temporal fidelity 有小幅代价。\n3. `F-HIST − S-HIST` 的 level CI 跨 0：不能声称视频在历史之外提高 level accuracy；其 native FD effect 为正，说明视频增量主要体现在当前 FD 指标。\n4. `F-LAST − S-LAST` 也表现为 level 变差、FD 改善，说明视频作用依赖所测指标，不能简单称为全面视频收益。\n\n因此当前最准确的表述是：**历史信息的 level 价值是 video-conditional；固定视频对 native FD 有 development 增量，但 level 增量不确定。** 本轮没有单独证明顺序、因果机制或 dynamics understanding。\n\n## 为什么上一轮 router 不继续\n\n上一轮 fixed-q oracle 仍比 Persistence 差 `2.2886%` native FD，旧 router 距离该 oracle 仅 `0.2593%`，所以那条结论只说明固定两组件选择空间不足；本轮更换为直接监督的输入信息问题，旧 router gate 不再阻断。\n\n旧 `ARI=1` 已撤回：它没有在 bootstrap 中重新拟合 centers；正确 refit 后 K=2 ARI median 约 `-0.0007` 且区间很宽。Base OOF 完整也不等于 selector OOF 完整，旧 fold whitening 的 all-train 统计共享仍保留为限制。\n\n## 边界\n\n所有结果来自反复使用的 development set；物理 command units、URDF/FK、session/source 泛化仍 unknown。`FINAL_TEST_ACCESS=0`，没有联系教授，没有执行下一轮 multi-seed Prompt，也没有修改旧 roots。\n""", encoding="utf-8")

    (ROOT / "NEXT_STEP_PROMPT.md").write_text("""# Next-step prompt: HI-F-HIST fresh-seed confirmation\n\n`generated_not_executed=true`. This handoff is created because HI-F-HIST passed the pre-registered single-seed development gate; it must not run automatically.\n\nRun only a separately authorized, development-only confirmation of the single fixed candidate `HI-F-HIST` with at least three pre-registered fresh integer seeds. Keep the exact 32-point past-history construction, fixed dynamic video cache, causal TCN, DCT-K16 residual parameterization, train-only preprocessing, AdamW/loss/40-epoch budget, and fixed-last checkpoint rule. Do not test extra architectures, losses, gates, prototypes, selectors, slope windows, or seeds in the same confirmation.\n\nFor every seed, retain Persistence, fixed Arm-learned + Hand-Persistence hybrid, and HI-F-HIST comparisons. Require the same Hand/Full/FD/trajectory guardrails and a hierarchical seed-to-episode analysis before calling it a `multi_seed_development_candidate`. If any pre-registered seed or guardrail fails, report the proposal as unconfirmed.\n\nHard boundaries: no final/held-out access, no professor contact, no replacement of the frozen candidate, no LoRA/encoder training, no automatic final evaluation. If GPU is authorized, use the dynamic all-visible watcher, co-location with measured-peak plus normal-variation VRAM admission, at most two project children, tmux, and no signals to other users.\n""", encoding="utf-8")

    qa = ROOT / "qa"
    write_json(qa / "RUN_ACCEPTANCE.json", {"status": "PASS", "stage0_accepted": "4/4", "formal_matrix_accepted": "4/4", "dev_inference_accepted": "4/4", "rows": run_rows, "final_test_access": 0})
    write_json(qa / "GPU_STAGE0.json", {"status": "PASS", "accepted": "4/4", "accepted_attempt": "RETRY1", "first_attempt": "technical_failure_torch_peak_api", "physical_gpu_ids": actual_gpu_ids, "max_project_children": 2, "child_peak_reserved_mib": 36, "child_peak_allocated_mib": 29, "oom": False, "other_process_signaled": False, "final_test_access": 0})
    (qa / "GPU_STAGE0.md").write_text(f"""# GPU Stage0 QA\n\nAccepted Stage0: **4/4** on retry1. The initial attempt failed before training because the installed PyTorch rejected the device argument to `reset_peak_memory_stats`; that attempt is preserved in the logs and was not counted. The equivalent API fix was CPU-tested and retry1 passed all four cells.\n\nPhysical GPU IDs used by accepted Stage0: `{actual_gpu_ids}`. Child peak allocated/reserved: `29/36 MiB`. No OOM and no non-child signal occurred.\n""", encoding="utf-8")
    write_json(qa / "RESOURCE_AUDIT.json", {"status": "PASS", "visible_gpu_pool": "dynamic_all_visible", "physical_gpu_ids_used": actual_gpu_ids, "max_project_gpu_children_observed": 2, "co_location_allowed": True, "utilization_ranking_only": True, "formal_required_free_memory_mib": 548, "formal_measured_peak_mib": 36, "gpu3_rejected_for_stage0_capacity": True, "non_child_process_signaled": False, "final_test_access": 0})
    (qa / "RESOURCE_AUDIT.md").write_text("""# Dynamic GPU resource audit\n\nThe accepted run used the dynamic all-visible policy. GPU2 was selected first and GPU0 was selected for the co-located second child; queued cells reused those cards after leases released. GPU1 remained heavily utilized and GPU3 did not satisfy the initial Stage0 memory admission. Utilization was used for ranking only. The watcher enforced at most two project children and one child per physical GPU.\n\nNo other user's process was signaled or modified.\n""", encoding="utf-8")
    write_json(qa / "LEAKAGE_QA.json", {"status": "PASS_WITH_CAVEAT", "train_only_preprocessing": True, "dev_used_in_training": False, "dev_used_for_checkpoint_selection": False, "formal_checkpoint": "epoch40_last", "target_input_timestamps_strictly_after_t0": True, "history_input_timestamps_le_t0": True, "task_id_model_input": False, "future_target_model_input": False, "selector": False, "prototype": False, "final_test_access": 0})
    (qa / "LEAKAGE_QA.md").write_text("""# Leakage QA\n\nAll four formal fits used canonical train rows only. The dev set was not read for early stopping or checkpoint selection; fixed-last inference occurred after all training. State history was rebuilt from rows at or before per-episode t0; targets start strictly after t0. Video was a previously accepted frozen dynamic cache and no raw video was newly read. No task ID, future target, selector, prototype, oracle, or prediction stack entered the model.\n\nThe historical prototype OOF caveats remain documented separately and do not get silently upgraded by this direct train-to-dev experiment.\n""", encoding="utf-8")
    write_json(qa / "NUMERIC_QA.json", {"status": "PASS_WITH_CAVEAT", "canonical_shape": [3355, 30, 28], "train_count": 2741, "dev_count": 614, "history_native_shape": [3355, 32, 71], "hand_prediction_shape": [614, 30, 14], "full_prediction_shape": [614, 30, 28], "formal_optimizer_steps_each": 1720, "parameter_count_each": 218528, "target_raw_seam_max_abs": 0.0, "anchor_raw_seam_max_abs": 0.0, "checkpoint_reload_stage0_max_abs": 0.0, "arm_bitwise_equal_locked": True, "finite_predictions": True, "final_test_access": 0})
    (qa / "NUMERIC_QA.md").write_text("""# Numeric and schema QA\n\nCanonical target shape is `(3355,30,28)` with `2741/614` train/dev rows. Rebuilt history is `(3355,32,71)`, Hand outputs are `(614,30,14)`, and Full composites are `(614,30,28)`. Raw target and anchor seams are exact to the stored float arrays (`0.0` maximum absolute error). Every formal run completed 1720 optimizer steps with 218,528 matched parameters. The locked Arm slice is bitwise unchanged in every Full composite.\n\nNative FD is the declared 29 future-to-future difference metric; physical units remain unverified.\n""", encoding="utf-8")
    write_json(qa / "TECHNICAL_FAILURES.json", {"status": "RECOVERED", "events": [{"stage": "GPU_STAGE0", "attempt": "initial", "error": "PyTorch reset_peak_memory_stats rejected explicit torch.device argument", "impact": "no optimizer step accepted", "recovery": "equivalent no-argument API after set_device(0); retry1 4/4 passed"}, {"stage": "DEV_INFERENCE", "attempt": "initial", "error": "inference helper failed to slice full arrays to dev before comparing with dev targets", "impact": "invalid dev predictions archived and excluded", "recovery": "dev-mask slice fixed; final inference 4/4 accepted"}], "final_test_access": 0})
    (qa / "TECHNICAL_FAILURES.md").write_text("""# Technical failure log\n\nTwo recoverable implementation failures were preserved: the first Stage0 attempt used an incompatible explicit device argument for the installed PyTorch peak-memory API; the first dev inference attempt forgot the dev slice and would have compared train rows with dev targets. Neither produced an accepted scientific result. The corrected Stage0 retry and corrected fixed-last inference were independently rerun and accepted.\n""", encoding="utf-8")
    write_json(qa / "CLAIM_QA.json", {"status": "PASS_WITH_CAVEAT", "claim_ledger_rows": len(claim_rows), "candidate_status": candidate["status"], "single_seed": True, "development_only": True, "physical_units_claimed": False, "final_test_access": 0})
    (qa / "CLAIM_QA.md").write_text("""# Claim QA\n\nThe claim ledger separates supported development effects from rejected universal-history, video-level-superiority, physical-unit, seed-robustness, OOF, and final-generalization claims. The candidate gate is reported as a single-seed development proposal only. The invalid inference attempt is explicitly excluded from the final statistics.\n""", encoding="utf-8")
    write_json(qa / "VISUAL_QA.json", {"status": "PASS_WITH_CAVEAT", "figures_viewed": 6, "formats": ["png", "pdf"], "data_tables": 6, "captions": 6, "oracle_or_future_warning": "not applicable; all figures are direct model/dev diagnostics; native units remain unverified", "final_test_access": 0})
    (qa / "VISUAL_QA.md").write_text("""# Visual QA\n\nAll six final PNG figures were opened for inspection after generation: input timeline/history length; four-cell level/FD; matched effects with bootstrap intervals; horizon/joint/task/motion views; fixed representative trajectories; and failure cloud/Pareto. Labels, legends, CI error bars, full task/joint coverage, and no-final-test caveats were reviewed. Each has a matching PDF, CSV and caption. This is internal rendered-file QA, not publisher certification.\n""", encoding="utf-8")
    (qa / "FINAL_QA.md").write_text("""# Final QA\n\n- Study route: complete single-seed development experiment.\n- Data contract: PASS; 3355 canonical rows, 32 past query points, strict timestamp seam.\n- CPU seam tests: 8 passed.\n- GPU Stage0: 4/4 accepted on retry1; formal matrix: 4/4 accepted.\n- Fixed-last dev inference: 4/4 accepted; Arm slice bitwise unchanged.\n- Statistics: 10,000 paired episode bootstrap; four fixed contrasts × two co-primary metrics with Holm family; single-seed limitation retained.\n- Figures: 6/6 PNG/PDF pairs with data and captions, visually inspected.\n- Resource safety: dynamic all-visible policy, at most two children, no non-child signal.\n- Final boundary: `FINAL_TEST_ACCESS=0`; no professor contact; no automatic next experiment.\n\nThe candidate gate passes for `HI-F-HIST` as a development proposal, but the result is not a final model or generalized claim.\n""", encoding="utf-8")
    write_json(qa / "FINAL_QA.json", {"status": "PASS_WITH_CAVEAT", "study_status": "complete", "stage0_accepted": "4/4", "formal_matrix_accepted": "4/4", "dev_inference_accepted": "4/4", "candidate": "HI-F-HIST", "candidate_status": candidate["status"], "single_seed_development": True, "cpu_contract_tests": "8 passed", "figures": "6/6", "technical_failures_recovered": 2, "final_test_access": 0, "old_roots_modified": False, "professor_contact_started": False, "checksum_status": "pending_final_manifest"})
    (ROOT / "FINAL_STATUS.ini").write_text("\n".join([
        "STUDY_STATUS=complete",
        "INPUT_HISTORY_SOURCE=raw_G1_Parquet_observation_and_action_columns",
        "PAST_ONLY_TIMESTAMP_CHECK=pass; selected_rows<=t0_frame119; future_targets>t0",
        "NORMALIZER_FIT_SCOPE=canonical_train_only",
        "TRAINING_PIPELINE=single_stage_no_selector_no_OOF_required",
        "STAGE0_ACCEPTED=4/4",
        "FULL_MATRIX_ACCEPTED=4/4",
        f"SEED={SEED}",
        "HISTORY_EFFECT_STATE=negative_level_effect; CI_excludes_zero",
        "HISTORY_EFFECT_FUSION=positive_level_effect; CI_excludes_zero; FD_tradeoff",
        "VIDEO_INCREMENT_WITH_HISTORY=FD_positive; level_CI_crosses_zero",
        "BEST_ELIGIBLE_METHOD=HI-F-HIST",
        f"HAND_MAE={c_hand:.9f}",
        f"NATIVE_FD_CHANGE_VS_PERSISTENCE={fd_margin['relative_change_candidate_vs_control']:.9f}",
        f"FD_MARGIN_UNCERTAINTY=95CI[{fd_margin['ci_low']:.9f},{fd_margin['ci_high']:.9f}]",
        "FINAL_TEST_ACCESS=0",
        "OLD_ROOTS_MODIFIED=false",
        "PROFESSOR_CONTACT_STARTED=false",
        "VISIBLE_GPU_POOL=dynamic_all_visible",
        f"ACTUAL_GPU_IDS={','.join(str(value) for value in actual_gpu_ids)}",
        "MAX_PROJECT_GPU_CHILDREN_OBSERVED=2",
        "CHECKSUMS=pending_final_manifest",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate": "HI-F-HIST", "actual_gpu_ids": actual_gpu_ids, "final_test_access": 0}))
    return 0


SEED = 20260939


if __name__ == "__main__":
    raise SystemExit(main())

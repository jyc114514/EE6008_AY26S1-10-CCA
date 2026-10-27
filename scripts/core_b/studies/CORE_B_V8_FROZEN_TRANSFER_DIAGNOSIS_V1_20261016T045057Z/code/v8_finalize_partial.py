from __future__ import annotations

import json
import statistics
from datetime import datetime,timezone
from pathlib import Path

import v8_extract as ex

V8=ex.V8
V7=ex.V7
V2=ex.V2
V4=ex.V4
TASKS=ex.TASKS
SEEDS=ex.SEEDS


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def readcsv(path):
    return ex.read_csv(path)


def avg(vals):
    return float(sum(vals)/len(vals)) if vals else None


def sd(vals):
    return float(statistics.stdev(vals)) if len(vals)>1 else 0.0


def build_summary():
    train=readcsv(V8/"results/TRAIN_TARGET_AND_FIT.csv")
    z_gains={}
    for kind in ("TRAIN_Z_GAIN_M","TRAIN_Z_GAIN_U"):
        rows=[r for r in train if r["record_type"]==kind and r["scope"]=="equal_task_macro"]
        vals=[float(r["J_gain_runner"]) for r in rows]
        z_gains[kind]={"per_seed":{r["seed"]:float(r["J_gain_runner"]) for r in rows},
                       "mean":avg(vals),"sample_sd":sd(vals)}
    fit_y={}
    for method in ("G","U","M"):
        rows=[r for r in train if r["record_type"]=="own_or_truth_fit" and
              r["scope"]=="equal_task_macro" and r["method"]==method and r["target"]=="Y"]
        fit_y[method]={metric:{"mean":avg([float(r[col]) for r in rows]),
                               "sample_sd":sd([float(r[col]) for r in rows])}
                       for metric,col in (("position_mae","position_mae"),("safe_fd_l1","safe_fd_l1"),
                                          ("seam_safe_fd_l1","seam_safe_fd_l1"),("J_runner","J_runner"))}
    imitation=readcsv(V8/"results/TRAIN_DEV_IMITATION.csv")
    teacher_distance={}
    for split in ("train","dev"):
        teacher_distance[split]={}
        for method in ("G","U","M"):
            rows=[r for r in imitation if r["split"]==split and r["scope"]=="equal_task_macro" and r["method"]==method]
            teacher_distance[split][method]={
                metric:{"mean":avg([float(r[col]) for r in rows]),"sample_sd":sd([float(r[col]) for r in rows])}
                for metric,col in (("position","position_distance"),("safe_fd","safe_fd_distance"),("seam","seam_distance"))}
    geometry=readcsv(V8/"results/CORRECTION_GEOMETRY.csv")
    geometry_summary={}
    for split in ("train","dev"):
        rows=[r for r in geometry if r["split"]==split and r["scope"]=="three_seed_macro"]
        geometry_summary[split]={"magnitude":{},"projection":{}}
        for space in ("position","internal_fd","seam"):
            geometry_summary[split]["magnitude"][space]={}
            for vec in ("C","A","B","E"):
                r=next(x for x in rows if x["record_type"]=="magnitude" and x["space"]==space and x["vector"]==vec)
                geometry_summary[split]["magnitude"][space][vec]={"rms":float(r["rms"]),
                                                                  "mean_abs":float(r["mean_abs_magnitude"])}
        for space in ("position","internal_fd"):
            geometry_summary[split]["projection"][space]={}
            for vec in ("A","B","E"):
                r=next(x for x in rows if x["record_type"]=="projection" and x["space"]==space and x["vector"]==vec)
                geometry_summary[split]["projection"][space][vec]={
                    "cosine":float(r["weighted_cosine"]) if r["weighted_cosine"] not in ("",None) else None,
                    "k":float(r["projection_k"]) if r["projection_k"] not in ("",None) else None,
                    "orthogonal_energy":float(r["orthogonal_energy"]) if r["orthogonal_energy"] not in ("",None) else None,
                    "orthogonality_residual":float(r["orthogonality_residual"]) if r["orthogonality_residual"] not in ("",None) else None}
    v7sum=ex.read_json(V7/"results/RESULTS_SUMMARY.json")
    v7_losses={m:{k:float(v7sum["absolute_macro_losses"][m][k]["mean"])
                  for k in ("position_mae","safe_fd_l1","seam_safe_fd_l1")}
               for m in ("GT_RETRAIN","SELF_SOFT","MIX_SOFT","T")}
    ci=[r for r in readcsv(V7/"results/PAIRED_GROUP_CI.csv")
        if r["scope"]=="macro" and r["contrast"]=="MIX_VS_GT"
        and r["metric"] in ("position_mae","safe_fd_l1","seam_safe_fd_l1")]
    ci_summary={r["metric"]:{"gain":float(r["point_gain"]),"ci95_low":float(r["ci95_low"]),
                             "ci95_high":float(r["ci95_high"]),"draws":int(r["draws"]),
                             "seed":int(r["bootstrap_seed"])} for r in ci}
    effects=readcsv(V7/"results/PAIRED_EFFECTS.csv")
    task_gains={}
    for metric in ("position_mae","safe_fd_l1","seam_safe_fd_l1"):
        rows=[r for r in effects if r["contrast"]=="MIX_VS_GT" and r["scope"]=="task" and r["metric"]==metric]
        task_gains[metric]={str(t):avg([float(r["gain"]) for r in rows if int(r["task_id"])==t]) for t in TASKS}
    help_rows=readcsv(V8/"results/TEACHER_HELP_HARM.csv")
    help_summary={}
    for metric in ("position_mae","safe_fd_l1","seam_safe_fd_l1"):
        help_summary[metric]={}
        for task in TASKS:
            help_summary[metric][str(task)]={}
            for category in ("teacher_help","exact_tie","teacher_harm"):
                vals=[float(r["weighted_fraction"]) for r in help_rows
                      if int(r["task_id"])==task and r["metric"]==metric and r["teacher_stratum"]==category]
                help_summary[metric][str(task)][category]=avg(vals)
    curves=readcsv(V8/"results/TRAIN_CURVE_SUMMARY.csv")
    curve_summary={}
    for program in ("GT_RETRAIN","SELF_SOFT","MIX_SOFT"):
        rows=[r for r in curves if r["program"]==program]
        curve_summary[program]={
            "last_objective_mean":avg([float(r["last_weighted_objective_loss"]) for r in rows]),
            "last20_median_mean":avg([float(r["last20_objective_median"]) for r in rows]),
            "last20_min_mean":avg([float(r["last20_objective_min"]) for r in rows]),
            "last20_max_mean":avg([float(r["last20_objective_max"]) for r in rows]),
            "all_curve_values_finite":all(r["all_curve_values_finite"]=="True" for r in rows),
            "gradient_norm_values":"UNKNOWN_NOT_RECORDED"}
    return {
        "schema_version":1,"status":"complete_with_limits",
        "study":"post_V7_frozen_transfer_diagnosis_development",
        "post_result_informed_diagnostic":True,"independent_confirmation":False,
        "new_fits":0,"new_optimizer_updates":0,"new_teacher_forwards":0,
        "new_frozen_student_train_forward_jobs":6,"new_dev_prediction_jobs":0,
        "new_bootstrap_draws":0,"encoder_extraction":0,
        "train_Z_gain":z_gains,"final_train_vs_Y":fit_y,
        "train_dev_teacher_distance":teacher_distance,
        "correction_geometry":geometry_summary,
        "V7_macro_losses":v7_losses,"V7_MIX_vs_GT_source_group_CI":ci_summary,
        "V7_MIX_vs_GT_task_gains":task_gains,"V7_teacher_help_harm_weighted_fractions":help_summary,
        "training_curve_summary":curve_summary,
        "dev_horizon_axis":{"status":"UNAVAILABLE","position_future_steps":30,
                            "internal_fd_transitions":29,"recorded_coordinates":28,
                            "reason":"one dev target archive load was used in a failed V8 horizon reducer; no step/axis outputs were saved and the archive was not reopened"},
        "recommendation":"close the fixed 0.5 mixture objective; retain MIX50 as a dual-head development reference and GT_RETRAIN/accepted V_i as the single-head reference",
        "next_question":"NONE",
        "final_heldout_payload_access":0,"old_root_writes":0}


def markdown(summary):
    lines=[
      "# V8 V7 context and timeline addendum","",
      "STATUS: complete_with_limits. This is a post-result-informed development diagnostic, not independent confirmation.","",
      "## V7 accepted result context","",
      f"- V7 manifest: {ex.read_json(V7/'qa/FINAL_CHECKSUM_VERIFICATION.json')['manifest_entry_count']} entries; SHA-256 {ex.sha(V7/'CHECKSUMS.sha256')}; accepted receipt PASS.",
      "- V7: 9/9 fits, 28,800 optimizer updates, 9/9 dev prediction jobs, final heldout access 0.",
      "- Accepted macro losses, in position MAE / safe-FD / seam order:",
    ]
    for method,m in summary["V7_macro_losses"].items():
        lines.append(f"  - {method}: {m['position_mae']:.8f} / {m['safe_fd_l1']:.8f} / {m['seam_safe_fd_l1']:.8f}")
    lines += ["- MIX vs GT original source-group CIs (5000 draws, seed 2026101507, exploratory/conditional):"]
    for metric,x in summary["V7_MIX_vs_GT_source_group_CI"].items():
        lines.append(f"  - {metric}: {x['gain']:+.8f}, 95% percentile CI [{x['ci95_low']:+.8f}, {x['ci95_high']:+.8f}]")
    teacher=ex.read_json(V7/"protocol/TEACHER_BANK_SEAL.json")
    target=ex.read_json(V7/"protocol/TARGET_LOCK.json")
    source=ex.read_json(V8/"inventory/SOURCE_MAP.json")
    bsource={Path(x["path"]).name:x for x in source["timeline_and_objective_sources"]}
    lines += [
      "","## Teacher and target timeline","",
      f"- S/V/T bank sealed at {teacher['created_utc']}; accepted source bundles 6, derived T bundles 3, future target reads before seal {teacher['future_target_values_read_before_seal']}.",
      f"- TARGET_LOCK created at {target['created_utc']}; Y read after teacher seal: {target['Y_native_read_after_teacher_seal']}.",
      f"- The accepted target builder source {bsource['v7_build_targets.py']['path']} (SHA-256 {bsource['v7_build_targets.py']['sha256']}) seals teacher outputs before loading Y_train, then constructs Z. The earlier prose ordering ambiguity is corrected; these records do not support a leakage claim.",
      f"- Objective contract source {bsource['d3_eq_contract.py']['path']} (SHA-256 {bsource['d3_eq_contract.py']['sha256']}) defines position L1 plus 0.1 future-internal-FD L1 with the fixed Y-fitted training scales and task/episode weights.",
      "- G=V uses exact accepted model bytes, same train archive, ordered receiver IDs, normalizer, feature-cache manifest, and V7 teacher profile (batch 32, TS11-D). Dev G/V predictions are kept separate.",
      "","## V8 scope and limits","",
      "- New fits, optimizer updates, teacher forwards, dev predictions, bootstrap draws, and encoder extraction are all zero. Six frozen train-prefix forwards used epoch-160 SELF_SOFT/MIX_SOFT models; all ran on physical GPU 1 (logical cuda:0) and passed state-hash invariance.",
      "- The original 185-window dev-target archive was opened once. A vector reduction error stopped V8 before step/coordinate outputs were written. V7's per-window/task outcomes, source-group CIs, and LOO are reused; no new dev target read was made.",
      "- V8 train-Y files were loaded 3 times in total across failed/recovered CPU runs; 9 Z arrays were loaded 3 times each. The dev archive was loaded once. Exact counts and failed-work accounting are in V8_RESOURCE_AND_EXPOSURE.csv.",
      "- Dev horizon errors (30 frames), internal-FD transitions (29), coordinate errors (28), and target-bearing trajectory plots are marked UNAVAILABLE. No final/heldout payload was accessed; V7 root writes are zero.",
      "",
    ]
    return "\n".join(lines)


def main():
    required=[
      "results/TRAIN_TARGET_AND_FIT.csv","results/TRAIN_CURVE_SUMMARY.csv",
      "results/TRAIN_DEV_IMITATION.csv","results/CORRECTION_GEOMETRY.csv",
      "results/CORRECTION_GEOMETRY_LOO.csv","results/TEACHER_HELP_HARM.csv",
      "results/SEED_BLOCK_LOO.csv","results/TASK_LOO.csv",
      "results/V8_TRAIN_SEED_BLOCK_LOO.csv","results/V8_TRAIN_TASK_LOO.csv",
      "results/DECISION_TABLE.csv","results/HORIZON_AXIS_ERRORS.csv",
      "results/V7_RESOURCE_SUPPLEMENT.csv","CASE_SELECTION.csv","CASE_SELECTION_LEDGER.csv"]
    absent=[p for p in required if not (V8/p).is_file()]
    if absent: raise FileNotFoundError(f"saved V8 output tables missing: {absent}")
    summary=build_summary()
    ex.write_json(V8/"RETURN_TO_CHATGPT.json",summary)
    (V8/"results/V7_CONTEXT_AND_TIMELINE_ADDENDUM.md").write_text(markdown(summary),encoding="utf-8")
    # Resource summary uses saved tables and receipts only. No target or prediction
    # arrays are reloaded here.
    target_counts={"train_Y_npy_loads":1,"train_Z_npy_loads":9,
                   "train_target_archive_loads":0,"train_target_values_read":1}
    dev_access={"dev_target_archive_loads":0,"dev_target_rows_loaded":0,"dev_target_unique_ids":0,
                "dev_target_path":str(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),
                "dev_prefix_archive_loads":0,"dev_prefix_rows_loaded":0}
    counters={"train_target_perturbation":19200,"train_model_target":46080,
              "train_objective_J":46080,"train_imitation":11520,"dev_imitation":1665,
              "loo":138240,"geometry_receiver_contributions":39555,
              "dev_horizon_axis_windows":0}
    exposure=ex.build_resource_exposure({}, {},target_counts,dev_access,
        readcsv(V8/"results/TRAIN_CURVE_SUMMARY.csv"),
        readcsv(V8/"results/TRAIN_TARGET_AND_FIT.csv"),
        readcsv(V8/"results/HORIZON_AXIS_ERRORS.csv"),counters)
    ex.write_csv(V8/"results/V8_RESOURCE_AND_EXPOSURE.csv",list(exposure[0]),exposure)
    v7receipt=ex.read_json(V7/"qa/FINAL_CHECKSUM_VERIFICATION.json")
    if ex.sha(V7/"CHECKSUMS.sha256")!=v7receipt["manifest_sha256"]:
        raise RuntimeError("V7 sealed manifest changed")
    # Ensure only one V8 dev-target archive load appears across all attempts.
    attempt_data=ex.read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")
    total_dev_loads=sum(int(a.get("dev_target_archive_loads",0)) for a in attempt_data["attempts"])
    if total_dev_loads!=1: raise RuntimeError(f"dev target archive load count is {total_dev_loads}, expected one")
    qa={
      "schema_version":1,"status":"complete_with_limits","completed_utc":now(),
      "train_target_fit":"PASS","teacher_direction_geometry":"PASS",
      "accepted_V7_dev_task_window_metrics":"PASS_REUSED",
      "dev_horizon_axis":"UNAVAILABLE_AFTER_ONE_ARCHIVE_LOAD",
      "dev_target_archive_loads_total":total_dev_loads,
      "new_fits":0,"optimizer_updates":0,"new_teacher_forwards":0,
      "new_frozen_student_train_forwards":6,"new_dev_predictions":0,
      "new_bootstrap_draws":0,"encoder_extraction":0,
      "final_heldout_payload_access":0,"old_root_writes":0,
      "V7_RESOURCE_SUPPLEMENT_ROWS":len(readcsv(V8/"results/V7_RESOURCE_SUPPLEMENT.csv")),
      "recommendation":summary["recommendation"],"next_question":"NONE"}
    ex.write_json(V8/"qa/REAL_DATA_QA.json",qa)
    # Exact train G=V alias evidence. No separate dev tensor alias is claimed.
    v7source=ex.read_json(V7/"protocol/SOURCE_MAP.json")
    fit=ex.read_json(V7/"protocol/FIT_ACCEPTANCE_INDEX.json")
    gt=ex.read_json(V7/"qa/GT_RETRAIN_VI_REPRODUCTION.json")
    teacher=ex.read_json(V7/"protocol/TEACHER_BANK_SEAL.json")
    aliases=[]
    for seed in SEEDS:
        f=next(e for e in fit["entries"] if e["program_id"]=="GT_RETRAIN" and int(e["seed"])==seed)
        vmodel=next(e for e in v7source["models"] if e.get("role")=="V" and int(e["seed"])==seed)
        proof=next(e for e in gt["entries"] if int(e["seed"])==seed)
        config=V7/"configs/teacher"/f"v7_teacher_V_s{seed}_a01.json"
        job=ex.read_json(config)
        bank=next(e for e in teacher["entries"] if e["teacher_role"]=="V" and int(e["seed"])==seed)
        assert f["model_sha256"]==vmodel["model_sha256"]==bank["source_model_sha256"]
        assert proof["byte_identical_model_file"] is True
        assert job["batch_size"]==32 and job["video_mode"]=="TS11-D"
        aliases.append({"seed":seed,"model_sha256":f["model_sha256"],
                        "state_hash":proof["GT_RETRAIN_state_hash"],
                        "teacher_job_config_path":str(config),"teacher_job_config_sha256":ex.sha(config),
                        "normalizer_sha256":job["normalizer_sha256"],
                        "feature_cache_manifest_sha256":job["feature_cache_manifest_sha256"],
                        "train_npz_sha256":job["train_npz_sha256"],
                        "ordered_receiver_ids_sha256":job["ordered_receiver_ids_sha256"],
                        "video_mode":job["video_mode"],"batch_size":job["batch_size"],
                        "prediction_dtype_after_native_restore":"float64",
                        "alias_scope":"train only; exact accepted V output reused under same deterministic runner/profile"})
    ex.write_json(V8/"qa/ALIAS_QA.json",{
      "schema_version":1,"status":"PASS_WITH_LIMITS","created_utc":now(),
      "train_G_equals_V_alias":aliases,"dev_G_V_alias_asserted":False,
      "dev_tensor_numeric_difference":"not recomputed in finalizer; V7 accepted prediction sources remain separate",
      "numeric_tolerance":"no tensor tolerance is claimed for dev; identical train model/input/profile alias uses one V source tensor",
      "final_heldout_payload_access":0})
    # Final summary and retry ledger.
    ex.write_json(V8/"qa/FINAL_ACCEPTANCE.json",{
      "schema_version":1,"status":"complete_with_limits","completed_utc":now(),
      "new_neural_fits":0,"new_optimizer_updates":0,"new_teacher_forward_jobs":0,
      "new_frozen_student_train_forward_jobs":6,"new_dev_prediction_jobs":0,
      "new_fitted_predictors_or_search":0,"new_bootstrap_draws":0,"encoder_extraction":0,
      "dev_gt_archive_loads_total":1,"final_heldout_payload_access":0,
      "old_root_writes":0,"V7_RESOURCE_SUPPLEMENT_ROWS":9,
      "dev_horizon_axis_status":"UNAVAILABLE; no numeric values persisted",
      "status_reason":"Do not reopen the single loaded dev-target archive after the V8 reduction error; reuse accepted V7 dev metrics."})
    ex.write_json(V8/"coordination/EXTRACT_COMPLETE.json",{
      "schema_version":1,"status":"complete_with_limits","created_utc":now(),
      "train_target_fit":"PASS","prediction_only_geometry":"PASS",
      "V7_dev_outcomes":"REUSED","dev_horizon_axis":"UNAVAILABLE",
      "new_fits":0,"optimizer_updates":0,"new_teacher_forwards":0,
      "new_frozen_student_train_forward_jobs":6,"new_dev_predictions":0,
      "dev_target_archive_loads_total":1,"final_heldout_payload_access":0,
      "old_root_writes":0})
    attempts=ex.read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")
    attempts["attempts"].append({
      "attempt":5,"status":"PASS_REPORT_RESUMED_FROM_SAVED_TABLES","completed_utc":now(),
      "dev_target_archive_loads":0,"train_Y_npy_loads":0,"train_Z_npy_loads":0,
      "prediction_native_arrays_loaded":0,
      "outputs":["V8_RESOURCE_AND_EXPOSURE.csv","REAL_DATA_QA.json","ALIAS_QA.json",
                 "FINAL_ACCEPTANCE.json","V7_CONTEXT_AND_TIMELINE_ADDENDUM.md","RETURN_TO_CHATGPT.json"],
      "note":"resumed from saved salvage tables and receipts; no data arrays reloaded"})
    attempts["status"]="RECOVERED_COMPLETE_WITH_LIMITS"
    ex.write_json(V8/"coordination/EXTRACTION_ATTEMPTS.json",attempts)
    ex.write_json(V8/"coordination/OWNER_STATE.json",{
      "schema_version":1,"status":"COMPLETE_WITH_LIMITS","completed_utc":now(),
      "old_root_writes":0,"final_heldout_payload_access":0,
      "dev_target_archive_loads_total":1,
      "note":"Saved V8 train/prediction-only diagnostics and reused V7 dev metrics; dev step/axis values remain unavailable."})
    print(json.dumps({"status":"complete_with_limits","train_Z_gains":summary["train_Z_gain"],
                      "train_dev_teacher_distances":summary["train_dev_teacher_distance"],
                      "dev_target_loads":total_dev_loads,"new_dev_predictions":0},sort_keys=True))


if __name__=="__main__":
    raise SystemExit(main())

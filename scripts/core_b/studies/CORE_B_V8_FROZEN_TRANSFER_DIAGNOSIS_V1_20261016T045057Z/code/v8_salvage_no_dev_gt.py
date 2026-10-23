from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import numpy as np

import v8_extract as ex

V8=ex.V8
V7=ex.V7
V4=ex.V4
TASKS=ex.TASKS
SEEDS=ex.SEEDS
METHODS=ex.METHODS


def now()->str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def csv_rows(path:Path)->list[dict[str,str]]:
    return ex.read_csv(path)


def write_unavailable_horizon() -> list[dict[str,Any]]:
    reason=("UNAVAILABLE: the only authorized original dev-target archive load was completed, "
            "but the extractor failed before step/axis reductions were saved. The locked workflow "
            "prohibits reopening dev Y for a formatting/reduction retry.")
    rows=[]
    for quantity,description in (
        ("position_step","30 future frames"),
        ("internal_fd_transition","29 future-internal transitions"),
        ("position_axis","28 recorded coordinates"),
        ("internal_fd_axis","28 recorded coordinates")):
        rows.append({
            "split":"dev","scope":"ALL","seed":"ALL","task_id":"ALL","method":"ALL",
            "quantity":quantity,"future_frame":"","transition_start_frame":"",
            "from_frame":"","to_frame":"","coordinate_index":"",
            "mean_error":"","sample_sd_across_seeds":"","windows":185,
            "source_groups":47,"units":"standardized metric units",
            "definition":description,"status":"UNAVAILABLE","missing_reason":reason,
        })
    fields=["split","scope","seed","task_id","method","quantity","future_frame",
            "transition_start_frame","from_frame","to_frame","coordinate_index",
            "mean_error","sample_sd_across_seeds","windows","source_groups","units",
            "definition","status","missing_reason"]
    ex.write_csv(V8/"results/HORIZON_AXIS_ERRORS.csv",fields,rows)
    ex.write_csv(V8/"figure_data/horizon_axis_errors.csv",fields,rows)
    ex.write_json(V8/"figure_data/CASE_TRAJECTORY_DATA_STATUS.json",{
        "schema_version":1,"status":"UNAVAILABLE_DEV_Y_CLOSED","created_utc":now(),
        "dev_target_archive_loads_total":1,"dev_target_rows_read":185,
        "dev_target_values_persisted":False,
        "reason":reason,
        "action":"do not reopen the dev target archive; retain V7 accepted per-window/task results",
        "final_heldout_payload_access":0,
    })
    return rows


def build_decision_table()->list[dict[str,Any]]:
    pri=csv_rows(V7/"results/PRIMARY_TASK_SEED.csv")
    effects=csv_rows(V7/"results/PAIRED_EFFECTS.csv")
    cis=csv_rows(V7/"results/PAIRED_GROUP_CI.csv")
    ci={(r["contrast"],r["scope"],r["metric"]):r for r in cis}
    rows=[]
    for r in pri:
        for metric in ("position_mae","safe_fd_l1","seam_safe_fd_l1"):
            rows.append({"record_type":"accepted_absolute_loss","contrast":"",
                         "method":r["method"],"seed":r["seed"],"task_id":r["task_id"],
                         "scope":"task_seed","metric":metric,"loss":r[metric],
                         "candidate":"","reference":"","candidate_loss":"","reference_loss":"",
                         "gain":"","gain_sample_sd":"","ci95_low":"","ci95_high":"",
                         "gain_sign":"","source_table":str(V7/"results/PRIMARY_TASK_SEED.csv")})
    for r in effects:
        key=(r["contrast"],r["scope"],r["metric"])
        interval=ci.get(key,{})
        rows.append({"record_type":"accepted_paired_gain","contrast":r["contrast"],
                     "method":r["candidate"],"seed":r["seed"],"task_id":r["task_id"],
                     "scope":r["scope"],"metric":r["metric"],"loss":"",
                     "candidate":r["candidate"],"reference":r["reference"],
                     "candidate_loss":r["candidate_loss"],"reference_loss":r["reference_loss"],
                     "gain":r["gain"],"gain_sample_sd":r["gain_sample_sd_across_seeds"],
                     "ci95_low":interval.get("ci95_low",""),"ci95_high":interval.get("ci95_high",""),
                     "gain_sign":r["gain_sign"],
                     "source_table":str(V7/"results/PAIRED_EFFECTS.csv")})
    return rows


def build_selection_no_y(dev_rows:dict[int,list[dict[str,str]]],
                         train_rows:dict[int,list[dict[str,str]]],
                         train_pred:dict[int,dict[str,np.ndarray]],
                         sigma:np.ndarray) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    seed=202610131
    tr=train_rows[seed]
    d=train_pred[seed]["T"]-train_pred[seed]["V"]
    scores=[]
    for coordinate in range(28):
        task_values=[]
        for task in TASKS:
            idx=ex.task_indices(tr,task)
            w=ex.row_weights(tr,idx,"episode_id")
            per_anchor=np.mean(np.abs(d[idx])/sigma[None,None,:],axis=1)
            task_values.append(ex.weighted_mean(per_anchor[:,coordinate],w))
        scores.append(float(np.mean(task_values)))
    axes=[]
    for lo,hi in ((0,14),(14,28)):
        vals=np.asarray(scores[lo:hi]);axes.append(lo+int(np.flatnonzero(vals==vals.max())[0]))
    rows=dev_rows[seed]
    perwindow=csv_rows(V7/"results/PER_WINDOW_METRICS.csv")
    win={(int(r["seed"]),r["method"],r["anchor_id"]):float(r["position_mae"]) for r in perwindow}
    grouped={t:[] for t in TASKS}
    for i,r in enumerate(rows): grouped[int(r["task_id"])].append((i,r))
    candidates=[]
    for task in TASKS:
        groups=sorted({r["source_group"] for _,r in grouped[task]},
                      key=lambda g:(hashlib.sha256(f"V8_20261016|{g}".encode()).hexdigest(),g))
        for rank,group in enumerate(groups[:2],1):
            i,r=next((i,r) for i,r in grouped[task] if r["source_group"]==group)
            candidates.append({"task_id":task,"anchor_id":r["anchor_id"],
                               "selection_type":"regular_hash_group","source_group":group,
                               "group_hash":hashlib.sha256(f"V8_20261016|{group}".encode()).hexdigest(),
                               "group_rank":rank,"seed":seed,"posthoc":False,"anchor_row_order":i})
    for task in TASKS:
        gains=[(r["anchor_id"],win[(seed,"GT_RETRAIN",r["anchor_id"])]-
                win[(seed,"MIX_SOFT",r["anchor_id"])],r) for _,r in grouped[task]]
        best=min(x[1] for x in gains)
        aid,gain,r=min((aid,gain,r) for aid,gain,r in gains if gain==best)
        i=next(i for i,x in grouped[task] if x["anchor_id"]==aid)
        candidates.append({"task_id":task,"anchor_id":aid,"selection_type":"posthoc_min_M_gain",
                           "source_group":r["source_group"],"group_hash":"","group_rank":"",
                           "seed":seed,"posthoc":True,"anchor_row_order":i,
                           "MIX_vs_G_gain_position":gain,"all_anchors_improved":all(g>0 for _,g,_ in gains)})
    dedup={}
    for c in candidates:
        aid=c["anchor_id"]
        if aid not in dedup:
            dedup[aid]={**c,"selection_reasons":[c["selection_type"]]}
        else:
            dedup[aid]["selection_reasons"].append(c["selection_type"])
            if c["posthoc"]:
                dedup[aid]["posthoc"]=True
                dedup[aid]["MIX_vs_G_gain_position"]=c["MIX_vs_G_gain_position"]
    selected=[]
    for n,(aid,r) in enumerate(sorted(dedup.items(),key=lambda kv:(kv[1]["task_id"],kv[0])),1):
        selected.append({"plot_id":f"case_{n:02d}","seed":seed,"task_id":r["task_id"],
                         "anchor_id":aid,"source_group":r["source_group"],
                         "selection_reasons":";".join(r["selection_reasons"]),
                         "posthoc_failure_example":r["posthoc"],
                         "MIX_vs_G_gain_position":r.get("MIX_vs_G_gain_position",""),
                         "axis_0_14":axes[0],"axis_14_28":axes[1],
                         "trajectory_plot_status":"UNAVAILABLE_DEV_Y_CLOSED",
                         "anchor_row_order":r["anchor_row_order"]})
    ledger=[]
    for c in candidates:
        match=next(r for r in selected if r["anchor_id"]==c["anchor_id"])
        first=next(x for x in candidates if x["anchor_id"]==c["anchor_id"])
        ledger.append({**c,"plot_id":match["plot_id"],
                       "plot_once":c is first,"dedup_anchor":match["anchor_id"],
                       "trajectory_plot_status":"UNAVAILABLE_DEV_Y_CLOSED"})
    return selected,ledger


def build_train_loo(train_rows,train_pred,dev_rows,dev_pred,y_train,sigma,fdscale):
    rows_seed=[];rows_task=[];geom_loo=[]
    targets={s:y_train for s in SEEDS}
    for excluded in SEEDS:
        keep=[s for s in SEEDS if s!=excluded]
        losses={}
        for method in METHODS:
            losses[method]=ex.macro_loss("train",keep,list(TASKS),method,"Y",
                                          train_rows,train_pred,
                                          {s:targets[s] for s in keep},sigma,fdscale,None)
        geom=ex.geometry_rows("train","seed_block_loo",f"drop_{excluded}","ALL",
                              train_rows,train_pred,keep,list(TASKS),sigma,fdscale)
        row=ex.loo_pack("train","seed_block",excluded,losses,geom)
        row["remaining_seed_count"]=len(keep);row["remaining_task_count"]=len(TASKS)
        rows_seed.append(row);geom_loo.extend(geom)
    for excluded in TASKS:
        keep=[t for t in TASKS if t!=excluded]
        losses={}
        for method in METHODS:
            losses[method]=ex.macro_loss("train",list(SEEDS),keep,method,"Y",
                                          train_rows,train_pred,targets,sigma,fdscale,None)
        geom=ex.geometry_rows("train","task_loo","ALL",f"drop_{excluded}",
                              train_rows,train_pred,list(SEEDS),keep,sigma,fdscale)
        row=ex.loo_pack("train","task",excluded,losses,geom)
        row["remaining_seed_count"]=len(SEEDS);row["remaining_task_count"]=len(keep)
        rows_task.append(row);geom_loo.extend(geom)
    # Dev correction geometry is output-only, so it can be recomputed without Y.
    for excluded in SEEDS:
        keep=[s for s in SEEDS if s!=excluded]
        geom_loo.extend(ex.geometry_rows("dev","seed_block_loo",f"drop_{excluded}","ALL",
                                        dev_rows,dev_pred,keep,list(TASKS),sigma,fdscale))
    for excluded in TASKS:
        keep=[t for t in TASKS if t!=excluded]
        geom_loo.extend(ex.geometry_rows("dev","task_loo","ALL",f"drop_{excluded}",
                                        dev_rows,dev_pred,list(SEEDS),keep,sigma,fdscale))
    return rows_seed,rows_task,geom_loo


def main()->int:
    preflight,train,dev=ex.canonical_rows_and_arrays()
    target_lock=preflight["target_lock"];source=preflight["source"]
    norm=json.loads(Path(source["data"]["normalizer"]).read_text())
    metric=json.loads(Path(source["data"]["metric_scale_lock"]).read_text())
    sigma=np.asarray(metric["target_sigma"],dtype=np.float64)
    fdscale=np.asarray(metric["safe_fd_scale"],dtype=np.float64)
    train_sigma=np.asarray(norm["target_sigma"],dtype=np.float64)
    train_fd=np.asarray(norm["safe_fd_scale"],dtype=np.float64)
    y_pack,zs,target_counts=ex.load_training_targets(target_lock,train["rows"])
    y=y_pack["Y"]
    formula=[]
    for seed in SEEDS:
        z=zs[seed];s=train["pred"][seed]["S"];v=train["pred"][seed]["V"];t=train["pred"][seed]["T"]
        checks={
          "Z_G_equals_Y":np.array_equal(z["GT_RETRAIN"],y),
          "T_equals_half_S_plus_half_V":np.allclose(t,0.5*s+0.5*v,rtol=0,atol=1e-12),
          "Z_U_formula":np.allclose(z["SELF_SOFT"],0.5*y+0.5*v,rtol=0,atol=1e-12),
          "Z_M_formula":np.allclose(z["MIX_SOFT"],0.5*y+0.5*t,rtol=0,atol=1e-12),
          "Z_M_minus_Z_U_identity":np.allclose(z["MIX_SOFT"]-z["SELF_SOFT"],0.25*(s-v),rtol=0,atol=1e-12)}
        if not all(checks.values()): raise RuntimeError(f"target formula QA failed {seed}: {checks}")
        formula.append({"seed":seed,**{k:bool(v) for k,v in checks.items()}})
    # Crucially, do not call load_dev_targets_and_anchors: the single dev GT
    # archive load occurred in the failed extractor and is closed by protocol.
    train_table=ex.build_train_target_and_fit(train["rows"],train["pred"],y,zs,
                                               sigma,fdscale,train_sigma,train_fd)
    imitation=ex.build_train_dev_imitation(train["rows"],train["pred"],
                                            dev["rows"],dev["pred"],sigma,fdscale,
                                            {s:None for s in SEEDS})
    curves=ex.build_curve_summary()
    geometry=ex.build_geometry({"train":train["rows"],"dev":dev["rows"]},
                               {"train":train["pred"],"dev":dev["pred"]},sigma,fdscale)
    help_harm=ex.build_help_harm(dev["rows"])
    train_loo_seed,train_loo_task,geometry_loo=build_train_loo(
        train["rows"],train["pred"],dev["rows"],dev["pred"],y,sigma,fdscale)
    # V7 accepted dev outcome LOO is copied unchanged and explicitly retained as V7 evidence.
    shutil.copyfile(V7/"results/SEED_BLOCK_LOO.csv",V8/"results/SEED_BLOCK_LOO.csv")
    shutil.copyfile(V7/"results/TASK_LOO.csv",V8/"results/TASK_LOO.csv")
    ex.write_csv(V8/"results/V8_TRAIN_SEED_BLOCK_LOO.csv",list(train_loo_seed[0]),train_loo_seed)
    ex.write_csv(V8/"results/V8_TRAIN_TASK_LOO.csv",list(train_loo_task[0]),train_loo_task)
    ex.write_csv(V8/"results/CORRECTION_GEOMETRY_LOO.csv",list(geometry_loo[0]),geometry_loo)

    ex.write_csv(V8/"results/TRAIN_TARGET_AND_FIT.csv",list(train_table[0]),train_table)
    ex.write_csv(V8/"results/TRAIN_CURVE_SUMMARY.csv",list(curves[0]),curves)
    ex.write_csv(V8/"results/TRAIN_DEV_IMITATION.csv",list(imitation[0]),imitation)
    ex.write_csv(V8/"results/CORRECTION_GEOMETRY.csv",list(geometry[0]),geometry)
    ex.write_csv(V8/"results/TEACHER_HELP_HARM.csv",list(help_harm[0]),help_harm)
    ex.write_csv(V8/"results/V7_SEED_BLOCK_LOO.csv",list(ex.read_csv(V7/"results/SEED_BLOCK_LOO.csv")[0]),
                 ex.read_csv(V7/"results/SEED_BLOCK_LOO.csv"))
    ex.write_csv(V8/"results/V7_TASK_LOO.csv",list(ex.read_csv(V7/"results/TASK_LOO.csv")[0]),
                 ex.read_csv(V7/"results/TASK_LOO.csv"))
    # Existing main task metrics/contrasts are copied into a normalized table.
    decision=build_decision_table()
    ex.write_csv(V8/"results/DECISION_TABLE.csv",list(decision[0]),decision)
    # The dev step/axis table is explicit about unavailable values.
    horizon=write_unavailable_horizon()

    cases,ledger=build_selection_no_y(dev["rows"],train["rows"],train["pred"],sigma)
    ex.write_csv(V8/"CASE_SELECTION.csv",list(cases[0]),cases)
    ex.write_csv(V8/"CASE_SELECTION_LEDGER.csv",list(ledger[0]),ledger)
    # Figures use only saved metrics/prediction-only geometry. No Y trajectory is emitted.
    # Keep exact equal-task axis selection scores for later captions/review.
    axisdata=[]
    for i in range(28):
        vals=[]
        for t in TASKS:
            idx=ex.task_indices(train["rows"][202610131],t)
            w=ex.row_weights(train["rows"][202610131],idx,"episode_id")
            d=np.abs(train["pred"][202610131]["T"][idx]-train["pred"][202610131]["V"][idx])/sigma[None,None,:]
            vals.append(ex.weighted_mean(d.mean(axis=1)[:,i],w))
        axisdata.append({"coordinate_index":i,"equal_task_mean_abs_T_minus_V_over_sigma":float(np.mean(vals)),
                         "axis_group":"0:14" if i<14 else "14:28"})
    ex.write_csv(V8/"figure_data/train_axis_scores.csv",list(axisdata[0]),axisdata)
    ex.write_csv(V8/"figure_data/teacher_distances_train_dev.csv",list(imitation[0]),imitation)
    ex.write_csv(V8/"figure_data/correction_geometry.csv",list(geometry[0]),geometry)
    ex.write_csv(V8/"figure_data/help_harm.csv",list(help_harm[0]),help_harm)
    ex.write_csv(V8/"figure_data/dev_task_gains.csv",list(decision[0]),decision)
    ex.write_json(V8/"figure_data/CASE_TRAJECTORY_DATA_STATUS.json",{
      "schema_version":1,"status":"UNAVAILABLE_DEV_Y_CLOSED","created_utc":now(),
      "dev_target_archive_loads_total":1,"dev_target_rows_read":185,
      "raw_dev_target_values_persisted":False,
      "reason":"V8 dev horizon aggregation failed after the authorized archive read; per-step/axis values were not saved. No archive reopen is permitted.",
      "case_selection_metadata_saved":True,"trajectory_plot_built":False,
      "final_heldout_payload_access":0})

    attempts=ex.read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")
    attempts["attempts"].append({
      "attempt":4,"status":"PASS_TRAIN_AND_PREDICTION_ONLY_SALVAGE",
      "created_utc":now(),"prediction_native_arrays_loaded":33,"train_Y_npy_loads":1,
      "train_Z_npy_loads":9,"train_target_npz_loads":0,"dev_target_archive_loads":0,
      "dev_prefix_archive_loads":0,"new_dev_predictions":0,"new_fit_count":0,
      "optimizer_updates":0,"new_teacher_forward_jobs":0,
      "outputs":["TRAIN_TARGET_AND_FIT.csv","TRAIN_DEV_IMITATION.csv","CORRECTION_GEOMETRY.csv",
                 "TEACHER_HELP_HARM.csv","TRAIN_CURVE_SUMMARY.csv","V8_TRAIN_SEED_BLOCK_LOO.csv",
                 "V8_TRAIN_TASK_LOO.csv","CORRECTION_GEOMETRY_LOO.csv","DECISION_TABLE.csv"],
      "final_heldout_payload_access":0})
    attempts["status"]="PASS_WITH_DEV_HORIZON_AXIS_LIMIT"
    ex.write_json(V8/"coordination/EXTRACTION_ATTEMPTS.json",attempts)

    # Counters persist from this successful label-independent diagnostic pass.
    resource=ex.build_resource_exposure(train["pred"],dev["pred"],target_counts,
        {"dev_target_archive_loads":0,"dev_target_rows_loaded":0,"dev_target_unique_ids":0,
         "dev_target_path":str(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),
         "dev_prefix_archive_loads":0,"dev_prefix_rows_loaded":0},
        curves,train_table,horizon,dict(ex.COUNTERS))
    ex.write_csv(V8/"results/V8_RESOURCE_AND_EXPOSURE.csv",list(resource[0]),resource)

    primary={"schema_version":1,"status":"PASS_WITH_LIMITS","created_utc":now(),
      "post_result_informed_diagnostic":True,"independent_confirmation":False,
      "train_Y_npy_loads_total":3,"train_Z_npy_loads_total":27,
      "dev_target_archive_loads_total":1,"dev_prefix_archive_loads_total":1,
      "prediction_native_arrays_loaded_total":99,
      "new_train_frozen_forwards":6,"new_dev_predictions":0,"new_fits":0,
      "new_optimizer_updates":0,"new_teacher_forwards":0,"new_bootstrap_draws":0,
      "encoder_extraction":0,"final_heldout_payload_access":0,"old_root_writes":0,
      "dev_horizon_axis_status":"UNAVAILABLE_AFTER_SINGLE_DEV_TARGET_LOAD",
      "train_target_formula_checks":formula,
      "v7_accepted_dev_primary_and_CI_reused":True,
      "final_status_reason":"The only dev target archive load ended in a vector aggregation TypeError before any diagnostic tables were persisted. Per V8 lock, the archive was not reopened. Train-target, imitation, correction geometry, task-level accepted outcomes and help/harm evidence are delivered."}
    ex.write_json(V8/"qa/REAL_DATA_QA.json",primary)
    ex.write_json(V8/"qa/ALIAS_QA.json",{
      "schema_version":1,"status":"PASS","created_utc":now(),
      "train_G_equals_V_model_file_sha":True,"train_G_prediction_reused_from_V":True,
      "dev_G_and_V_tensors_kept_separate":True,
      "dev_G_vs_V":preflight["dev_alias_checks"],
      "dev_numeric_alias_tolerance":"not asserted; original V7 score reconciliation reused",
      "normalizer_sha256":source["data"]["normalizer_sha256"],
      "metric_scale_lock_sha256":source["data"]["metric_scale_lock_sha256"],
      "final_heldout_payload_access":0})
    ex.write_json(V8/"coordination/EXTRACT_COMPLETE.json",{
      "schema_version":1,"status":"PASS_WITH_LIMITS","created_utc":now(),
      "train_analysis_and_prediction_only_geometry":"PASS",
      "dev_target_reopened_after_failure":False,
      "dev_horizon_axis":"UNAVAILABLE",
      "new_fits":0,"optimizer_updates":0,"new_dev_predictions":0,
      "final_heldout_payload_access":0,"old_root_writes":0})
    print(json.dumps({"status":"PASS_WITH_LIMITS","train_rows":len(train_table),
      "imitation_rows":len(imitation),"geometry_rows":len(geometry),
      "help_harm_rows":len(help_harm),"train_loo_seed_rows":len(train_loo_seed),
      "train_loo_task_rows":len(train_loo_task),"dev_target_reopened":False,
      "dev_horizon_axis":"UNAVAILABLE","new_fit":0,"updates":0,"new_dev_predictions":0},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

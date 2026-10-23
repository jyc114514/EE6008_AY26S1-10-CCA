"""Compare and score sealed development replays; final inputs are never opened."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file
from analysis_metrics import core_a_classification_metrics, core_b_per_anchor_metrics, core_b_task_macro, sample_sd

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
V9_ROOT=PROJECT/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
V9_SOURCE_MAP=V9_ROOT/"inventory/SOURCE_MAP.json"
V9_TARGETS=PROJECT/"reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz"
V9_TARGETS_SHA256="233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27"
V9_TARGETS_BYTES=435083
METRIC_SCALE_LOCK=PROJECT/"reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/protocol/METRIC_SCALE_LOCK.json"
METRIC_SCALE_LOCK_SHA256="dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32"
SEEDS_A=(20260825,20260826,20260827)
SEEDS_B=(202610131,202610132,202610133)
PROGRAMS_B=("A0","AD","V0","VD","S_OPT","MIX0","MIXD","PERSISTENCE")
METRICS_B=("position_mae","safe_fd_l1","seam_safe_fd_l1")
ATOL=1e-5
RTOL=1e-5


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def utc()->str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def atomic_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temp,path)

def atomic_csv(path:Path,rows:list[dict[str,Any]],fields:list[str]|None=None)->None:
    if not rows: raise ValueError(f"refusing empty CSV: {path}")
    if fields is None:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.part")
    with temp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="raise")
        writer.writeheader();writer.writerows(rows);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def read_csv(path:Path)->list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))

def read_npz(path:Path,keys:set[str])->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as archive:
        if not keys.issubset(set(archive.files)): raise ValueError(f"missing fields in {path}: {sorted(keys-set(archive.files))}")
        return {key:archive[key].copy() for key in keys}

def _attempt_root(attempt:str)->Path:
    return RUN_ROOT/"replay"/f"dev_full_{attempt}"/"artifacts"

def verify_seal(attempt:str)->tuple[dict[str,Any],dict[tuple[str,str,str],dict[str,Any]],Path]:
    root=_attempt_root(attempt);path=root/"DEV_PREDICTION_SEAL.json"
    seal=json.loads(path.read_text(encoding="utf-8"))
    if seal.get("status")!="SEALED_BEFORE_DEV_TARGET_ACCESS" or seal.get("dev_target_archive_opened") is not False:
        raise ValueError("dev prediction seal is missing or reports preseal target access")
    by_key={}
    for record in seal.get("prediction_records",[]):
        key=(str(record["core"]),str(record["program"]),"REFERENCE" if record.get("seed") is None else str(record["seed"]))
        if key in by_key: raise ValueError(f"duplicate sealed prediction key: {key}")
        bundle_path=Path(record["path"])
        if sha256_file(bundle_path)!=record["sha256"]: raise ValueError(f"sealed bundle hash mismatch: {bundle_path}")
        by_key[key]=record
    if len(by_key)!=28: raise ValueError("dev seal must contain 28 bundles")
    return seal,by_key,root

def accepted_core_a_path(program:str,seed:int)->Path:
    matrix=read_csv(OUT_ROOT/"protocol/FINAL_MODEL_MATRIX.csv")
    found=[r for r in matrix if r["core"]=="A" and r["program"]==program and int(r["seed"])==seed]
    if len(found)!=1: raise ValueError(f"missing accepted Core A identity {program}/{seed}")
    return Path(found[0]["checkpoint_path"]).with_name("predictions_best.npz")

def accepted_core_b_paths(program:str,seed:int)->tuple[Path,Path]:
    if program=="S_OPT":
        src=json.loads(V9_SOURCE_MAP.read_text(encoding="utf-8"))["source_assets"]["accepted_S_OPT_dev_predictions"][str(seed)]
        return Path(src["prediction"]["path"]),Path(src["rows"]["path"])
    if program in {"MIX0","MIXD"}: base=V9_ROOT/"predictions/dev_derived"/program/f"seed{seed}"
    else: base=V9_ROOT/"predictions/dev/full_context"/program/f"seed{seed}"
    return base/"PREDICTIONS.safetensors",base/"PREDICTION_ROWS.csv"

def compare_core_a_raw(by_key:dict[tuple[str,str,str],dict[str,Any]])->tuple[list[dict[str,Any]],np.ndarray,np.ndarray,list[dict[str,str]]]:
    identity=read_csv(RUN_ROOT/"replay/core_a/core_a_dev_identity.csv")
    episode_ids=np.asarray([int(r["episode_id"]) for r in identity],dtype=np.int64)
    canonical=[r["canonical_id"] for r in identity]
    results=[]
    for program in ("A_DYNAMIC","A_STATIC"):
        for seed in SEEDS_A:
            record=by_key[("A",program,str(seed))]
            new=read_npz(Path(record["path"]),{"episode_index","logits"})
            ref_path=accepted_core_a_path(program,seed)
            with np.load(ref_path,allow_pickle=False) as ref:
                ref_ids=ref["episode_index"].astype(np.int64)
                ref_logits=ref["logits"].astype(np.float64)
                ref_pred=ref["predictions"].astype(np.int64)
            if not np.array_equal(new["episode_index"].astype(np.int64),episode_ids) or not np.array_equal(ref_ids,episode_ids):
                raise ValueError(f"Core A dev order mismatch: {program}/{seed}")
            allclose=bool(np.allclose(new["logits"],ref_logits,atol=ATOL,rtol=RTOL))
            top_match=bool(np.array_equal(np.argmax(new["logits"],axis=1),ref_pred))
            row={"core":"A","program":program,"seed":seed,"reference_path":str(ref_path),"reference_sha256":sha256_file(ref_path),"raw_output_allclose":allclose,"top1_exact_match":top_match,"max_abs_difference":float(np.max(np.abs(new["logits"]-ref_logits))),"rows":617}
            results.append(row)
            if not allclose or not top_match: raise ValueError(f"Core A raw dev replay mismatch: {program}/{seed}")
    return results,episode_ids,np.asarray(canonical,dtype=str),identity

def compare_core_b_raw(by_key:dict[tuple[str,str,str],dict[str,Any]])->tuple[list[dict[str,Any]],np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[dict[str,str]]]:
    prefix_path=RUN_ROOT/"replay/core_b/DEV_PREFIXES_NO_LABELS.npz"
    prefix=read_npz(prefix_path,{"canonical_id","anchor_id","episode_id","task_id","source_cluster","anchor_native"})
    canonical=prefix["canonical_id"].astype(str);anchors=prefix["anchor_id"].astype(str)
    episodes=prefix["episode_id"].astype(np.int64);tasks=prefix["task_id"].astype(np.int64);groups=prefix["source_cluster"].astype(str)
    anchor_native=np.asarray(prefix["anchor_native"],dtype=np.float64)
    identity=read_csv(RUN_ROOT/"replay/core_b/core_b_dev_identity.csv")
    if len(canonical)!=185 or len(set(canonical.tolist()))!=185 or [r["canonical_id"] for r in identity]!=canonical.tolist(): raise ValueError("Core B canonical dev identity/order mismatch")
    results=[]
    for program in ("A0","AD","V0","VD","S_OPT","MIX0","MIXD"):
        for seed in SEEDS_B:
            record=by_key[("B",program,str(seed))]
            new=read_npz(Path(record["path"]),{"canonical_ids","prediction_native"})
            if not np.array_equal(new["canonical_ids"].astype(str),canonical): raise ValueError(f"Core B new prediction order mismatch: {program}/{seed}")
            ref_path,rows_path=accepted_core_b_paths(program,seed)
            ref_tensor=load_file(str(ref_path),device="cpu")
            old=np.asarray(ref_tensor["prediction_native"].numpy(),dtype=np.float64)
            old_rows=read_csv(rows_path)
            old_ids=[r["anchor_id"] for r in old_rows]
            if len(old_ids)!=185 or len(set(old_ids))!=185: raise ValueError(f"accepted Core B row IDs invalid: {program}/{seed}")
            lookup={anchor_id:i for i,anchor_id in enumerate(old_ids)}
            if set(lookup)!=set(anchors.tolist()): raise ValueError(f"accepted Core B support differs: {program}/{seed}")
            order=np.asarray([lookup[a] for a in anchors.tolist()],dtype=np.int64)
            old=old[order]
            close=bool(np.allclose(new["prediction_native"],old,atol=ATOL,rtol=RTOL))
            row={"core":"B","program":program,"seed":seed,"reference_path":str(ref_path),"reference_sha256":sha256_file(ref_path),"raw_output_allclose":close,"max_abs_difference":float(np.max(np.abs(new["prediction_native"]-old))),"rows":185}
            results.append(row)
            if not close: raise ValueError(f"Core B raw dev replay mismatch: {program}/{seed}")
    persistence=read_npz(Path(by_key[("B","PERSISTENCE","REFERENCE")]["path"]),{"canonical_ids","prediction_native"})
    if not np.array_equal(persistence["canonical_ids"].astype(str),canonical): raise ValueError("Persistence canonical dev IDs differ")
    expected=np.repeat(anchor_native[:,None,:],30,axis=1)
    if not np.array_equal(persistence["prediction_native"],expected): raise ValueError("Persistence output is not exact anchor repetition")
    results.append({"core":"B","program":"PERSISTENCE","seed":"REFERENCE","reference_path":"same observed anchor repeated 30 times","reference_sha256":"not_applicable","raw_output_allclose":True,"max_abs_difference":0.0,"rows":185})
    return results,canonical,anchors,episodes,tasks,groups,identity

def score_core_a(by_key:dict[tuple[str,str,str],dict[str,Any]],episode_ids:np.ndarray)->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    # One accepted dev label vector is read after the new prediction seal.
    label_path=accepted_core_a_path("A_DYNAMIC",SEEDS_A[0])
    with np.load(label_path,allow_pickle=False) as ref:
        labels=ref["labels"].astype(np.int64)
        label_ids=ref["episode_index"].astype(np.int64)
    if labels.shape!=(617,) or not np.array_equal(label_ids,episode_ids):
        raise ValueError("accepted Core A dev label vector/order differs from the locked population")
    per_seed=[];scores={}
    for program in ("A_DYNAMIC","A_STATIC"):
        for seed in SEEDS_A:
            new=read_npz(Path(by_key[("A",program,str(seed))]["path"]),{"episode_index","logits"})
            if not np.array_equal(new["episode_index"].astype(np.int64),episode_ids):
                raise ValueError(f"new Core A output order differs from the sealed population: {program}/{seed}")
            metric=core_a_classification_metrics(new["logits"],labels,class_count=120)
            scores[(program,seed)]=metric
            per_seed.append({"program":program,"seed":seed,"support":617,**metric})
    summaries=[]
    for program in ("A_DYNAMIC","A_STATIC"):
        rows=[r for r in per_seed if r["program"]==program]
        summaries.append({"program":program,"seed_count":3,"support_per_seed":617,
            "top1_accuracy_mean":float(np.mean([r["top1_accuracy"] for r in rows])),"top1_accuracy_sample_sd":sample_sd([r["top1_accuracy"] for r in rows]),
            "top5_accuracy_mean":float(np.mean([r["top5_accuracy"] for r in rows])),"top5_accuracy_sample_sd":sample_sd([r["top5_accuracy"] for r in rows]),
            "macro_f1_mean":float(np.mean([r["macro_f1"] for r in rows])),"macro_f1_sample_sd":sample_sd([r["macro_f1"] for r in rows])})
    contrasts=[]
    for seed in SEEDS_A:
        a=scores[("A_DYNAMIC",seed)];b=scores[("A_STATIC",seed)]
        contrasts.append({"contrast":"A_DYNAMIC_minus_A_STATIC","seed":seed,"top1_accuracy_difference":a["top1_accuracy"]-b["top1_accuracy"],"macro_f1_difference":a["macro_f1"]-b["macro_f1"],"support":617})
    return per_seed,summaries,contrasts

def score_core_b(by_key:dict[tuple[str,str,str],dict[str,Any]],canonical:np.ndarray,anchors:np.ndarray,episodes:np.ndarray,tasks:np.ndarray,groups:np.ndarray,identity:list[dict[str,str]],target:np.ndarray,anchor_native:np.ndarray,sigma:np.ndarray,fdscale:np.ndarray):
    per_anchor_rows=[];task_seed_rows=[];seed_rows=[];seed_metrics={}
    identity_by_anchor={r["anchor_id"]:r for r in identity}
    for program in PROGRAMS_B:
        seeds=(None,) if program=="PERSISTENCE" else SEEDS_B
        for seed in seeds:
            key=("B",program,"REFERENCE" if seed is None else str(seed))
            bundle=read_npz(Path(by_key[key]["path"]),{"canonical_ids","prediction_native"})
            pred=np.asarray(bundle["prediction_native"],dtype=np.float64)
            per=core_b_per_anchor_metrics(pred,target,anchor_native,sigma,fdscale)
            macro,task_rows,episode_rows=core_b_task_macro(per,episodes,tasks)
            if macro["status"]!="estimable": raise ValueError(f"Core B fixed dev task macro not estimable: {program}/{seed}")
            seed_metrics[(program,seed)]={m:float(macro[m]) for m in METRICS_B}
            for task_row in task_rows: task_seed_rows.append({"program":program,"seed":seed,**task_row})
            seed_rows.append({"program":program,"seed":seed,"episode_support":macro["episode_support"],"anchor_support":macro["anchor_support"],**{m:float(macro[m]) for m in METRICS_B}})
            for i in range(185):
                meta=identity_by_anchor[str(anchors[i])]
                per_anchor_rows.append({"program":program,"seed":seed,"canonical_id":str(canonical[i]),"source_key":meta["source_key"],"episode_id":int(episodes[i]),"task_id":int(tasks[i]),"anchor_id":str(anchors[i]),"source_cluster":str(groups[i]),**{m:float(per[m][i]) for m in METRICS_B}})
    program_summary=[]
    for program in PROGRAMS_B:
        rows=[r for r in seed_rows if r["program"]==program]
        summary={"program":program,"training_seed_count":0 if program=="PERSISTENCE" else len(rows),"episode_support":47,"anchor_support":185}
        for metric in METRICS_B:
            values=[float(r[metric]) for r in rows]
            summary[f"{metric}_mean"]=float(np.mean(values))
            summary[f"{metric}_sample_sd"]=sample_sd(values) if len(values)>=2 else None
        program_summary.append(summary)
    comparisons=(("A0","V0"),("PERSISTENCE","V0"),("V0","VD"),("A0","AD"),("AD","VD"),("MIX0","MIXD"),("V0","MIX0"))
    contrasts=[]
    for reference,candidate in comparisons:
        contrast_seeds=(SEEDS_B if reference!="PERSISTENCE" else SEEDS_B)
        for metric in METRICS_B:
            gains=[seed_metrics[(reference,None if reference=="PERSISTENCE" else seed)][metric]-seed_metrics[(candidate,seed)][metric] for seed in contrast_seeds]
            contrasts.append({"contrast":f"{reference}_minus_{candidate}","reference":reference,"candidate":candidate,"metric":metric,"grain":"three_seed_task_macro","mean_gain":float(np.mean(gains)),"sample_sd_gain_across_seeds":sample_sd(gains),"seed_count":3,"support_tasks":5,"support_episodes":47,"support_anchors":185,"direction":"positive favors candidate lower error"})
    return per_anchor_rows,task_seed_rows,seed_rows,program_summary,contrasts

def _score_after_raw_checks(args:Any,score_root:Path,attempt_root:Path,seal:dict[str,Any],by_key:dict[tuple[str,str,str],dict[str,Any]],raw_checks:list[dict[str,Any]],episode_ids:np.ndarray,canonical_a:np.ndarray,identity_a:list[dict[str,str]],canonical_b:np.ndarray,anchors:np.ndarray,episodes:np.ndarray,tasks:np.ndarray,groups:np.ndarray,identity_b:list[dict[str,str]])->int:
    # This latch is written before opening any dev label/target payload.
    exposure=score_root/"DEV_SCORE_EXPOSURE.json"
    score_retry = int(getattr(args, "score_retry", 0))
    prior_reads = score_retry
    prior_core_a_labels = 617 * score_retry
    prior_target_rows = 185 * max(0, score_retry - 1)
    prior_target_values = prior_target_rows * 30 * 28
    prior_latch_name = args.attempt if score_retry == 1 else f"{args.attempt}_score_retry_{score_retry-1:02d}"
    prior_latch_path = OUT_ROOT/"replay/dev_replay"/prior_latch_name/"DEV_SCORE_EXPOSURE.json"
    if score_retry:
        prior = json.loads(prior_latch_path.read_text(encoding="utf-8"))
        previous_read_total = int(prior.get("core_b_target_archive_content_reads_total", prior.get("core_b_target_archive_content_reads", -1)))
        previous_label_total = int(prior.get("core_a_label_rows_materialized_total", prior.get("core_a_label_rows_materialized", -1)))
        previous_target_rows = int(prior.get("core_b_target_rows_materialized_total", prior.get("core_b_target_rows_materialized", -1)))
        previous_target_values = int(prior.get("core_b_target_values_materialized_total", prior.get("core_b_target_values_materialized", -1)))
        expected_status = "FAILED" if score_retry == 1 else "PASS"
        if (prior.get("status") != expected_status or previous_read_total != prior_reads
            or previous_label_total != prior_core_a_labels or previous_target_rows != prior_target_rows
            or previous_target_values != prior_target_values):
            raise ValueError("scorer retry prior exposure ledger does not match the registered read count")
    latch={"schema_version":1,"status":"READ_STARTED","split":"dev","started_utc":utc(),"prediction_seal_path":str(attempt_root/"DEV_PREDICTION_SEAL.json"),"prediction_seal_sha256":sha256_file(attempt_root/"DEV_PREDICTION_SEAL.json"),"raw_prediction_comparisons":len(raw_checks),"raw_comparisons_allclose":all(bool(r["raw_output_allclose"]) for r in raw_checks),"core_a_dev_label_source":"accepted predictions_best.npz labels array; loaded after new prediction seal","prior_core_a_label_rows_materialized":prior_core_a_labels,"current_core_a_label_rows_materialized":0,"core_a_label_rows_materialized_total":prior_core_a_labels,"core_b_target_archive_path":str(V9_TARGETS),"core_b_target_archive_sha256_expected":V9_TARGETS_SHA256,"prior_core_b_target_archive_content_reads":prior_reads,"current_core_b_target_archive_content_reads":0,"core_b_target_archive_content_reads_total":prior_reads,"prior_core_b_target_rows_materialized":prior_target_rows,"current_core_b_target_rows_materialized":0,"core_b_target_rows_materialized_total":prior_target_rows,"core_b_target_values_materialized_current":0,"core_b_target_values_materialized_total":prior_target_values,"final_payload_reads":0}
    atomic_json(exposure,latch)
    per_seed_a,summary_a,contrasts_a=score_core_a(by_key,episode_ids)
    latch["current_core_a_label_rows_materialized"]=617
    latch["core_a_label_rows_materialized_total"]=prior_core_a_labels+617
    latch["first_dev_score_exposure_utc"]=utc();atomic_json(exposure,latch)
    if V9_TARGETS.stat().st_size!=V9_TARGETS_BYTES: raise ValueError("Core B dev target archive size differs from accepted exposure")
    target_start=utc();blob=V9_TARGETS.read_bytes();actual_sha=hashlib.sha256(blob).hexdigest()
    if len(blob)!=V9_TARGETS_BYTES or actual_sha!=V9_TARGETS_SHA256:
        latch.update({"status":"FAILED","failure_reason":"dev archive expected SHA/size mismatch","current_core_b_target_archive_content_reads":1,"core_b_target_archive_content_reads_total":prior_reads+1,"actual_dev_archive_sha256":actual_sha,"actual_dev_archive_bytes":len(blob),"decoded_target_arrays":0,"failure_utc":utc()})
        atomic_json(exposure,latch)
        raise ValueError("Core B dev target archive hash/size mismatch")
    latch["core_b_target_archive_sha256"]=actual_sha
    latch["current_core_b_target_archive_content_reads"]=1
    latch["core_b_target_archive_content_reads_total"]=prior_reads+1
    latch["core_b_target_read_started_utc"]=target_start
    atomic_json(exposure,latch)
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
        required={"anchor_id","episode_id","task_id","source_cluster","target_native_h30"}
        if not required.issubset(archive.files): raise ValueError(f"dev target schema missing: {sorted(required-set(archive.files))}")
        target_ids=archive["anchor_id"].astype(str);target_episode=archive["episode_id"].astype(np.int64);target_task=archive["task_id"].astype(np.int64);target_group=archive["source_cluster"].astype(str);target_raw=np.asarray(archive["target_native_h30"],dtype=np.float64).copy()
    if len(target_ids)!=185 or len(set(target_ids.tolist()))!=185 or set(target_ids.tolist())!=set(anchors.tolist()): raise ValueError("Core B dev target support differs from sealed predictions")
    lookup={value:i for i,value in enumerate(target_ids.tolist())};order=np.asarray([lookup[value] for value in anchors.tolist()],dtype=np.int64);target=target_raw[order]
    for i,j in enumerate(order.tolist()):
        if (int(target_episode[j]),int(target_task[j]),str(target_group[j]))!=(int(episodes[i]),int(tasks[i]),str(groups[i])): raise ValueError(f"Core B dev target metadata mismatch at {anchors[i]}")
    if target.shape!=(185,30,28) or not np.isfinite(target).all(): raise ValueError("Core B dev target shape/finite check failed")
    latch["current_core_b_target_rows_materialized"]=185
    latch["core_b_target_rows_materialized_total"]=prior_target_rows+185
    latch["core_b_target_values_materialized_current"]=int(target.size)
    latch["core_b_target_values_materialized_total"]=prior_target_values+int(target.size)
    latch["decoded_target_arrays_current"]=1
    latch["decoded_target_arrays_total"]=max(0,score_retry)+1
    atomic_json(exposure,latch)
    observed_metric_lock_sha=sha256_file(METRIC_SCALE_LOCK)
    if observed_metric_lock_sha!=METRIC_SCALE_LOCK_SHA256:
        raise ValueError("accepted Core B metric scale lock SHA changed")
    metric_scale=json.loads(METRIC_SCALE_LOCK.read_text(encoding="utf-8"))
    if metric_scale.get("status")!="LOCKED":
        raise ValueError("Core B metric scale lock is not LOCKED")
    sigma=np.asarray(metric_scale["target_sigma"],dtype=np.float64)
    fdscale=np.asarray(metric_scale["safe_fd_scale"],dtype=np.float64)
    with np.load(RUN_ROOT/"replay/core_b/DEV_PREFIXES_NO_LABELS.npz",allow_pickle=False) as prefix: anchor_native=np.asarray(prefix["anchor_native"],dtype=np.float64).copy()
    per_anchor_b,task_seed_b,seed_b,summary_b,contrasts_b=score_core_b(by_key,canonical_b,anchors,episodes,tasks,groups,identity_b,target,anchor_native,sigma,fdscale)
    atomic_csv(score_root/"CORE_A_DEV_REPLAY_PER_SEED.csv",per_seed_a);atomic_csv(score_root/"CORE_A_DEV_REPLAY_SUMMARY.csv",summary_a);atomic_csv(score_root/"CORE_A_DEV_REPLAY_CONTRASTS.csv",contrasts_a)
    atomic_csv(score_root/"CORE_B_DEV_REPLAY_PER_ANCHOR.csv",per_anchor_b);atomic_csv(score_root/"CORE_B_DEV_REPLAY_TASK_SEED.csv",task_seed_b);atomic_csv(score_root/"CORE_B_DEV_REPLAY_SEED.csv",seed_b);atomic_csv(score_root/"CORE_B_DEV_REPLAY_SUMMARY.csv",summary_b);atomic_csv(score_root/"CORE_B_DEV_REPLAY_CONTRASTS.csv",contrasts_b)
    metric_reconciliation={"status":"not_compared"}
    metric_path=V9_ROOT/"results/PER_WINDOW_NEW_DEV_LOSSES.csv"
    if metric_path.is_file():
        old_rows=read_csv(metric_path);old={}
        for row in old_rows:
            if row.get("input_policy") not in {"full_context","derived_full_context"}: continue
            program=str(row.get("program_id",row.get("method","")))
            if program.endswith("_MASKED"): continue
            try: seed=int(row["seed"])
            except (KeyError,ValueError): continue
            old[(program,seed,str(row["anchor_id"]))]=row
        matched=0;max_difference={metric:0.0 for metric in METRICS_B}
        for row in per_anchor_b:
            if row["program"]=="PERSISTENCE": continue
            key=(row["program"],int(row["seed"]),row["anchor_id"]);ref=old.get(key)
            if ref is None: continue
            matched+=1
            for metric in METRICS_B: max_difference[metric]=max(max_difference[metric],abs(float(ref[metric])-float(row[metric])))
        metric_reconciliation={"status":"COMPARED" if matched else "NO_MATCHING_ROWS","source_path":str(metric_path),"source_sha256":sha256_file(metric_path),"matched_rows":matched,"max_abs_metric_difference":max_difference}
    latch.update({"status":"PASS","core_b_target_read_completed_utc":utc(),"completed_utc":utc(),"new_dev_metric_evaluations":185*22,"final_payload_reads":0});atomic_json(exposure,latch)
    score_receipt={"schema_version":1,"status":"PASS","attempt":args.attempt,"prediction_seal_sha256":sha256_file(attempt_root/"DEV_PREDICTION_SEAL.json"),"raw_comparison_rows":len(raw_checks),"raw_comparisons_allclose":all(bool(row["raw_output_allclose"]) for row in raw_checks),"core_a_dev_cells":6,"core_b_dev_neural_cells":15,"core_b_dev_derived_mixtures":6,"core_b_dev_persistence_references":1,"new_training_runs":0,"core_a_label_rows_materialized_total":prior_core_a_labels+617,"core_b_target_archive_content_reads_total":prior_reads+1,"core_b_target_rows_materialized_total":prior_target_rows+185,"core_b_target_values_materialized_current":int(target.size),"core_b_target_values_materialized_total":prior_target_values+int(target.size),"decoded_target_arrays_total":score_retry+1,"core_b_new_dev_metric_evaluations":185*22,"metric_reconciliation":metric_reconciliation,"metric_scale_lock_sha256":observed_metric_lock_sha,"final_payload_reads":0,"score_root":str(score_root)}
    atomic_json(score_root/"DEV_SCORE_RECEIPT.json",score_receipt)
    print(json.dumps({"status":"PASS","score_root":str(score_root),"raw_comparison_rows":len(raw_checks),"target_archive_reads":1,"metric_reconciliation":metric_reconciliation},sort_keys=True),flush=True)
    return 0

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--attempt",choices=("attempt_01","attempt_02","attempt_03","attempt_04","attempt_05","attempt_06"),required=True)
    parser.add_argument("--score-retry",type=int,choices=(0,1,2),default=0)
    args=parser.parse_args()
    attempt_root=RUN_ROOT/"replay"/f"dev_full_{args.attempt}"/"artifacts"
    if not (attempt_root/"DEV_PREDICTION_SEAL.json").is_file():
        raise FileNotFoundError("dev prediction seal must exist before scoring")
    seal,by_key,_attempt_root=verify_seal(args.attempt)
    # Compare raw predictions with accepted dev outputs before opening any labels/targets.
    core_a_raw,episode_ids,canonical_a,identity_a=compare_core_a_raw(by_key)
    core_b_raw,canonical_b,anchors,episodes,tasks,groups,identity_b=compare_core_b_raw(by_key)
    raw_checks=core_a_raw+core_b_raw
    score_name = args.attempt if args.score_retry == 0 else f"{args.attempt}_score_retry_{args.score_retry:02d}"
    score_root=OUT_ROOT/"replay/dev_replay"/score_name
    score_root.mkdir(parents=True,exist_ok=False)
    atomic_csv(score_root/"RAW_REPLAY_COMPARISON.csv",raw_checks)
    return _score_after_raw_checks(args,score_root,attempt_root,seal,by_key,raw_checks,episode_ids,canonical_a,identity_a,canonical_b,anchors,episodes,tasks,groups,identity_b)

if __name__=="__main__":
    raise SystemExit(main())

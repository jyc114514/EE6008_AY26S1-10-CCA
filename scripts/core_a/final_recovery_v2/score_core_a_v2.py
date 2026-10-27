#!/usr/bin/env python3
"""V2 CPU scorer for the six sealed Core A logits bundles."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

def utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path: Path,obj,replace=False):
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_name("."+path.name+"."+str(os.getpid())+".part")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def atomic_csv(path: Path,rows,fields):
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_name("."+path.name+"."+str(os.getpid())+".part")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def stable_rank(logits: np.ndarray) -> np.ndarray:
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError("logits must be finite rank-2")
    cols=np.arange(x.shape[1],dtype=np.int64)
    # Descending score; exact ties use the lower original output-column index.
    return np.vstack([np.lexsort((cols,-row)) for row in x]).astype(np.int64)

def confusion_matrix_union(y_true: np.ndarray,y_pred: np.ndarray,labels: list[int]) -> np.ndarray:
    pos={int(x):i for i,x in enumerate(labels)}
    out=np.zeros((len(labels),len(labels)),dtype=np.int64)
    for t,p in zip(np.asarray(y_true,dtype=np.int64),np.asarray(y_pred,dtype=np.int64)):
        if int(t) not in pos or int(p) not in pos: raise ValueError("value outside frozen C_union")
        out[pos[int(t)],pos[int(p)]]+=1
    return out

def metrics_from_rank(y_true,rank,column_to_task,f1_labels,union_labels,verify_sklearn=True):
    y=np.asarray(y_true,dtype=np.int64); cols=np.asarray(column_to_task,dtype=np.int64)
    pred=cols[rank[:,0]]; top5=cols[rank[:,:min(5,rank.shape[1])]]
    cm=confusion_matrix_union(y,pred,union_labels)
    upos={int(x):i for i,x in enumerate(union_labels)}
    per=[]; f1s=[]
    for label in f1_labels:
        i=upos[int(label)];tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp);den=2*tp+fp+fn
        f1=(2*tp/den) if den else 0.0;f1s.append(f1);support=int(cm[i,:].sum());pcount=int(cm[:,i].sum())
        m=y==int(label)
        top5rec=float(np.mean(np.any(top5[m]==int(label),axis=1))) if m.any() else 0.0
        per.append({"task_id":int(label),"support":support,"predicted_count":pcount,"tp":tp,"fp":fp,"fn":fn,
                    "precision":tp/pcount if pcount else 0.0,"recall":tp/support if support else 0.0,
                    "f1":f1,"top5_recall":top5rec})
    macro=float(np.mean(f1s)) if f1s else 0.0
    if verify_sklearn:
        sk=float(f1_score(y,pred,labels=list(map(int,f1_labels)),average="macro",zero_division=0))
        if not np.isclose(macro,sk,atol=1e-12,rtol=0.0):
            raise AssertionError(f"NumPy confusion oracle and sklearn F1 differ: {macro} != {sk}")
    return {"top1_accuracy":float(np.mean(pred==y)),
            "top5_accuracy":float(np.mean(np.any(top5==y[:,None],axis=1))),
            "macro_f1_true_classes":macro,"confusion_matrix":cm,"per_class":per,
            "ranked_task_ids_top5":top5}

def estimand_masks(y_true: np.ndarray,model_classes:set[int],final_classes:set[int]):
    if model_classes==final_classes:
        return {"A-FULL602":np.arange(len(y_true),dtype=np.int64)}
    common=model_classes&final_classes
    return {"A-FULL602":np.arange(len(y_true),dtype=np.int64),
            "A-COMMON590":np.flatnonzero(np.isin(y_true,sorted(common)))}

def record_bootstrap_draw_manifest(collection,estimand,path,draw_sha256,shape,missing_by_task):
    """Add one serializable estimand entry without aliasing the collection itself."""
    entry={"path":str(path),"sha256":str(draw_sha256),"shape":list(shape),
        "rng":"numpy.random.default_rng","seed":2026101901,"draws":5000,
        "resampling_unit":"V1 split_assignment_key cluster; one episode per group",
        "same_cluster_draw_for_all_six_cells":True,
        "macro_f1_zero_support_count_by_task":{int(k):int(v) for k,v in missing_by_task.items()}}
    collection[str(estimand)]=entry
    return entry

def load_preflight(v1:Path,out:Path):
    seal_path=v1/"predictions/CORE_A_PREDICTION_SEAL.json"
    manifest_path=v1/"predictions/CORE_A_PREDICTION_MANIFEST.csv"
    identity_path=v1/"protocol/CORE_A_FINAL_IDENTITY.csv"
    seal=json.loads(seal_path.read_text(encoding="utf-8"))
    manifest=list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    identity=list(csv.DictReader(identity_path.open(encoding="utf-8-sig")))
    identity.sort(key=lambda r:int(r["episode_id"]))
    canonical=[r["canonical_id"] for r in identity]
    episodes=np.asarray([int(r["episode_id"]) for r in identity],dtype="<i8")
    canonical_sha=hashlib.sha256("\n".join(canonical).encode()).hexdigest()
    episode_sha=hashlib.sha256(np.ascontiguousarray(episodes,dtype="<i8").tobytes()).hexdigest()
    if len(identity)!=602 or len(set(episodes.tolist()))!=602: raise ValueError("locked Core A IDs invalid")
    if seal.get("prediction_count")!=6 or len(manifest)!=6: raise ValueError("expected the six sealed Core A cells")
    map_obj=json.loads((out/"core_a/CORRECTED_CLASS_MAP_V2.json").read_text(encoding="utf-8"))
    class_rows=sorted(map_obj["classes"],key=lambda r:int(r["column_index"]))
    col_to_task=[int(r["canonical_task_id"]) for r in class_rows]
    if [int(r["column_index"]) for r in class_rows]!=list(range(120)): raise ValueError("corrected output map is not dense 0..119")
    preds={}
    for row in manifest:
        key=(row["program"],int(row["seed"]))
        path=Path(row["prediction_path"]);sidepath=Path(row["sidecar_path"])
        if sha(path)!=row["prediction_sha256"] or sha(sidepath)!=row["sidecar_sha256"]:
            raise ValueError(f"sealed prediction or sidecar hash mismatch: {key}")
        if row["canonical_ids_sha256"]!=canonical_sha: raise ValueError(f"canonical ID order hash mismatch: {key}")
        if row["shape"]!="[602,120]" or int(row["rows"])!=602: raise ValueError(f"sealed prediction shape mismatch: {key}")
        side=json.loads(sidepath.read_text(encoding="utf-8"))
        if side.get("checkpoint_sha256")!=row["checkpoint_sha256"]: raise ValueError(f"checkpoint sidecar mismatch: {key}")
        if side.get("episode_index_sha256")!=episode_sha: raise ValueError(f"episode order hash mismatch: {key}")
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={"episode_index","logits"}: raise ValueError(f"unexpected sealed NPZ members: {key}")
            ids=np.asarray(z["episode_index"],dtype="<i8")
            logits=np.asarray(z["logits"],dtype=np.float64)
        if not np.array_equal(ids,episodes): raise ValueError(f"sealed row order mismatch: {key}")
        if logits.shape!=(602,120) or not np.isfinite(logits).all(): raise ValueError(f"invalid sealed logits: {key}")
        preds[key]={"logits":logits,"manifest_row":row,"sidecar":side}
    expected={(p,s) for p in ("A_DYNAMIC","A_STATIC") for s in (20260825,20260826,20260827)}
    if set(preds)!=expected: raise ValueError("V1 sealed matrix is not Core A dynamic/static x 3 seeds")
    return {"seal_path":seal_path,"seal":seal,"manifest_path":manifest_path,"manifest_sha256":sha(manifest_path),
        "identity_path":identity_path,"identity_sha256":sha(identity_path),"identity":identity,
        "canonical_ids":canonical,"canonical_ids_sha256":canonical_sha,"episode_ids":episodes,
        "episode_index_sha256":episode_sha,"predictions":preds,"column_map":map_obj,
        "column_map_sha256":sha(out/"core_a/CORRECTED_CLASS_MAP_V2.json")}

def preflight_only(v1,out):
    pre=load_preflight(v1,out)
    print(json.dumps({"status":"PREDICTION_PREFLIGHT_PASS","cells":6,"rows_per_cell":602,
        "canonical_ids_sha256":pre["canonical_ids_sha256"],"prediction_manifest_sha256":pre["manifest_sha256"],
        "class_map_sha256":pre["column_map_sha256"]},ensure_ascii=False))
    return 0

def run_score(v1:Path,out:Path,run:Path):
    pre=load_preflight(v1,out)
    freeze_path=out/"core_a/FREEZE_RECEIPT.json"
    if not freeze_path.is_file(): raise FileNotFoundError("score requires frozen Core A V2 receipt")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status")!="LOCKED_BEFORE_V2_SCORE": raise ValueError("Core A V2 contract is not locked")
    if freeze["code_sha256"].get("score_core_a_v2.py")!=sha(Path(__file__)): raise ValueError("score code differs from frozen hash")
    if freeze.get("prediction_manifest_sha256")!=pre["manifest_sha256"]: raise ValueError("sealed prediction manifest hash differs from freeze")
    if freeze.get("corrected_class_map_sha256")!=pre["column_map_sha256"]: raise ValueError("corrected class map differs from freeze")
    exposure_path=out/"core_a/SCORE_EXPOSURE.json"
    if not exposure_path.is_file(): raise FileNotFoundError("attempt 01 exposure event is required for this documented retry")
    prior_exposure=json.loads(exposure_path.read_text(encoding="utf-8"))
    if prior_exposure.get("metrics_computed") is not False or prior_exposure.get("first_v2_metric_computed_utc") is not None:
        raise RuntimeError("prior Core A score attempt exposed a metric; do not run this retry")
    prior_attempt=json.loads((out/"core_a/CORE_A_SCORE_ATTEMPT_01.json").read_text(encoding="utf-8"))
    if prior_attempt.get("status")!="FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC":
        raise RuntimeError("prior attempt status does not authorize this same-contract retry")
    attempt_path=out/"core_a/CORE_A_SCORE_ATTEMPT_02.json"
    if attempt_path.exists(): raise FileExistsError("attempt 02 already exists; refusing duplicate scoring")
    label_path=Path(freeze.get("final_label_manifest_path",freeze.get("label_manifest_path","")))
    label_sha=freeze.get("final_label_manifest_sha256",freeze.get("label_manifest_sha256"))
    if not label_path.is_file(): raise FileNotFoundError(label_path)
    attempt={"schema_version":1,"protocol_id":out.name,"attempt_id":"CORE_A_SCORE_ATTEMPT_02",
        "freeze_revision":freeze.get("revision"),"status":"STARTED",
        "started_utc":utc(),"final_label_rows_expected":602,"final_label_rows_materialized":0,
        "metric_values_computed":False,"first_metric_computed_utc":None,"bootstrap_draws_run":0,
        "score_tables_created":False,"prior_failed_attempt_path":str(out/"core_a/CORE_A_SCORE_ATTEMPT_01.json"),
        "repeat_read_same_ids":True}
    atomic_json(attempt_path,attempt)
    exposure={
        "schema_version":1,"protocol_id":out.name,"status":"FINAL_LABEL_READ_STARTED",
        "v1_prior_final_label_values_decoded":1204,
        "v2_pre_score_label_metadata_reads":{
            "final_test_label_manifest_rows_materialized":614,
            "full_split_manifest_reads":6,"rows_per_split_manifest_read":4060,
            "final_task_label_metadata_rows_per_split_read":614,
            "unique_final_rows_seen_before_score":614
        },
        "v2_scoring_label_source":str(label_path),"v2_scoring_label_manifest_sha256":sha(label_path),
        "v2_score_rows_expected":602,"v2_score_rows_materialized":0,
        "second_v2_label_read_started_utc":attempt["started_utc"],"first_v2_metric_computed_utc":None,
        "score_table_commit_utc":None,"metrics_computed":False,"final_score_exposed":False,
        "attempt_id":"CORE_A_SCORE_ATTEMPT_02","attempt01_prior_labels_materialized":602
    }
    if sha(label_path)!=label_sha: raise ValueError("final label manifest hash differs from freeze")
    # Materialize exactly the frozen 602 IDs; the separate 12 duplicate-conflict rows are excluded by the pre-label v3 contract.
    table=pq.read_table(label_path,columns=["episode_index","split","original_task_index","contiguous_label_index","task_name_raw","category_raw"],
                        filters=[("episode_index","in",pre["episode_ids"].tolist())])
    label_rows=table.to_pylist()
    if len(label_rows)!=602: raise ValueError(f"expected 602 V1 primary label rows, got {len(label_rows)}")
    by_id={int(r["episode_index"]):int(r["original_task_index"]) for r in label_rows}
    if len(by_id)!=602 or set(by_id)!=set(pre["episode_ids"].tolist()): raise ValueError("final labels do not match exact frozen IDs")
    if any(r["split"]!="final_test" for r in label_rows): raise ValueError("label source returned a non-final row")
    y=np.asarray([by_id[int(i)] for i in pre["episode_ids"]],dtype=np.int64)
    attempt["status"]="FINAL_LABELS_MATERIALIZED"
    attempt["final_label_rows_materialized"]=602
    attempt["label_read_completed_utc"]=utc()
    atomic_json(attempt_path,attempt,replace=True)
    class_map=pre["column_map"]
    classes=sorted(class_map["classes"],key=lambda r:int(r["column_index"]))
    col_to_task=[int(r["canonical_task_id"]) for r in classes]
    frozen_core=freeze.get("core_a",freeze)
    model_set=set(col_to_task); final_set=set(map(int,frozen_core["C_final"]))
    if set(y.tolist())!=final_set: raise ValueError("observed final class IDs differ from the frozen C_final")
    if model_set!=set(map(int,frozen_core["C_model"])): raise ValueError("observed output map differs from freeze")
    if len(y)!=602 or len(set(pre["episode_ids"].tolist()))!=602: raise ValueError("population cardinality changed")
    union=sorted(model_set|final_set)
    f1_full=sorted(final_set)
    common=model_set&final_set
    if model_set==final_set:
        estimands={"A-FULL602":np.arange(602,dtype=np.int64)}
    else:
        estimands={"A-FULL602":np.arange(602,dtype=np.int64),
                   "A-COMMON590":np.flatnonzero(np.isin(y,sorted(common)))}
    attempt["status"]="METRIC_COMPUTATION_STARTED"
    attempt["final_label_rows_materialized"]=602
    atomic_json(attempt_path,attempt,replace=True)

    per_rows=[];detail_rows=[];summary_rows=[];loo_rows=[];contrast_rows=[];estimand_inputs={}
    draw_manifest_by_estimand={}
    first_metric_recorded=False
    for estimand,rows_idx in estimands.items():
        if not len(rows_idx): continue
        true=y[rows_idx]
        f1_labels=sorted(final_set if estimand=="A-FULL602" else common)
        estimand_inputs[estimand]={"rows":len(rows_idx),"labels":f1_labels}
        cell_metrics={}
        rank_cache={}
        for program in ("A_DYNAMIC","A_STATIC"):
            for seed in (20260825,20260826,20260827):
                logits=pre["predictions"][(program,seed)]["logits"][rows_idx]
                rank=stable_rank(logits)
                rank_cache[(program,seed)]=rank
                m=metrics_from_rank(true,rank,col_to_task,f1_labels,union,verify_sklearn=True)
                if not first_metric_recorded:
                    attempt["first_metric_computed_utc"]=utc()
                    attempt["status"]="FIRST_FINAL_METRIC_COMPUTED"
                    attempt["metric_values_computed"]=True
                    atomic_json(attempt_path,attempt,replace=True)
                    first_metric_recorded=True
                cell_metrics[(program,seed)]=m
                per_rows.append({
                    "estimand":estimand,"program":program,"seed":seed,"n_rows":len(rows_idx),
                    "n_classes_macro_f1":len(f1_labels),"sample_coverage":len(rows_idx)/602,
                    "class_coverage":len(common)/len(final_set),"top1_accuracy":m["top1_accuracy"],
                    "top5_accuracy":m["top5_accuracy"],"macro_f1_true_classes":m["macro_f1_true_classes"],
                    "macro_f1_labels":json.dumps(f1_labels,separators=(",",":")),
                    "zero_division":0,"ranking_columns":120,
                    "tie_rule":"descending logit; equal logits ordered by lower original output-column index"})
                cm=m["confusion_matrix"]
                for i,t in enumerate(union):
                    for j,p in enumerate(union):
                        detail_rows.append({"record_type":"confusion","estimand":estimand,"program":program,"seed":seed,
                            "true_task_id":t,"predicted_task_id":p,"count":int(cm[i,j])})
                name_by_task={int(c["canonical_task_id"]):c["canonical_task_name"] for c in classes}
                for pc in m["per_class"]:
                    detail_rows.append({"record_type":"per_class","estimand":estimand,"program":program,"seed":seed,
                        "true_task_id":pc["task_id"],"predicted_task_id":"","count":"",
                        "support":pc["support"],"predicted_count":pc["predicted_count"],
                        "tp":pc["tp"],"fp":pc["fp"],"fn":pc["fn"],"precision":pc["precision"],
                        "recall":pc["recall"],"f1":pc["f1"],"top5_recall":pc["top5_recall"],
                        "true_task_name":name_by_task.get(pc["task_id"],"")})
        if estimand!="A-FULL602" and not (model_set!=final_set): continue
        # Use the V1 split_assignment_key groups in identity order; groups are one episode each in this frozen population.
        cluster_file=v1/"protocol/CORE_A_FINAL_BOOTSTRAP_CLUSTER_MAP.csv"
        cluster_rows=list(csv.DictReader(cluster_file.open(encoding="utf-8-sig")))
        by_cid={r["canonical_id"]:r["cluster_id_sha256"] for r in cluster_rows}
        if set(by_cid)!=set(pre["canonical_ids"]): raise ValueError("cluster map IDs differ from V1 identity")
        cluster_ids=[by_cid[c] for c in pre["canonical_ids"]]
        row_idx=np.asarray(rows_idx,dtype=np.int64)
        local_cluster_ids=[cluster_ids[i] for i in row_idx]
        groups={}
        for i,cid in enumerate(local_cluster_ids):groups.setdefault(cid,[]).append(i)
        group_keys=sorted(groups)
        if len(group_keys)!=len(row_idx) or any(len(v)!=1 for v in groups.values()):
            raise ValueError("V1 Core A cluster unit is not one unique episode per frozen row")
        rng=np.random.default_rng(2026101901)
        draw_indices=rng.integers(0,len(group_keys),size=(5000,len(group_keys)),dtype=np.int16)
        out_draw=run/f"core_a/{estimand}_BOOTSTRAP_DRAWS.npy"
        if out_draw.exists(): raise FileExistsError(out_draw)
        out_draw.parent.mkdir(parents=True,exist_ok=True)
        np.save(out_draw,draw_indices,allow_pickle=False)
        draws={key:{metric:np.empty(5000,dtype=np.float64) for metric in ("top1_accuracy","top5_accuracy","macro_f1_true_classes")}
               for key in cell_metrics}
        missing={int(label):0 for label in f1_labels}
        for b,group_draw in enumerate(draw_indices):
            idx=np.asarray([groups[group_keys[int(g)]][0] for g in group_draw],dtype=np.int64)
            yb=true[idx]
            for label in f1_labels:
                if not np.any(yb==int(label)):missing[int(label)]+=1
            for key in cell_metrics:
                program,seed=key
                ranks=rank_cache[key][idx]
                m=metrics_from_rank(yb,ranks,col_to_task,f1_labels,union,verify_sklearn=False)
                for metric in draws[key]:draws[key][metric][b]=m[metric]
        record_bootstrap_draw_manifest(draw_manifest_by_estimand,estimand,out_draw,sha(out_draw),
                                       draw_indices.shape,missing)
        # Per-condition seed mean/sample SD and grouped interval conditional on fitted weights.
        for program in ("A_DYNAMIC","A_STATIC"):
            for metric in ("top1_accuracy","top5_accuracy","macro_f1_true_classes"):
                values=[cell_metrics[(program,s)][metric] for s in (20260825,20260826,20260827)]
                boot=np.mean(np.stack([draws[(program,s)][metric] for s in (20260825,20260826,20260827)],axis=0),axis=0)
                summary_rows.append({"estimand":estimand,"program":program,"metric":metric,"n_rows":len(rows_idx),
                    "n_seeds":3,"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),
                    "conditional_grouped_bootstrap_ci_low":float(np.quantile(boot,0.025)),
                    "conditional_grouped_bootstrap_ci_high":float(np.quantile(boot,0.975)),
                    "bootstrap_draws":5000,"bootstrap_seed":2026101901,"cluster_count":len(group_keys)})
                for omitted in (20260825,20260826,20260827):
                    kept=[s for s in (20260825,20260826,20260827) if s!=omitted]
                    loo_rows.append({"estimand":estimand,"program":program,"metric":metric,
                        "omitted_seed":omitted,"loo_mean":float(np.mean([cell_metrics[(program,s)][metric] for s in kept]))})
        for metric in ("top1_accuracy","top5_accuracy","macro_f1_true_classes"):
            deltas={s:cell_metrics[("A_DYNAMIC",s)][metric]-cell_metrics[("A_STATIC",s)][metric]
                    for s in (20260825,20260826,20260827)}
            boot=np.mean(np.stack([draws[("A_DYNAMIC",s)][metric]-draws[("A_STATIC",s)][metric]
                                   for s in (20260825,20260826,20260827)],axis=0),axis=0)
            row={"estimand":estimand,"contrast":"A_DYNAMIC_minus_A_STATIC","metric":metric,
                 "direction":"positive favors dynamic","mean_difference":float(np.mean(list(deltas.values()))),
                 "sample_sd_across_three_paired_seed_differences":float(np.std(list(deltas.values()),ddof=1)),
                 "seed_20260825_difference":deltas[20260825],"seed_20260826_difference":deltas[20260826],
                 "seed_20260827_difference":deltas[20260827],
                 "conditional_grouped_bootstrap_ci_low":float(np.quantile(boot,0.025)),
                 "conditional_grouped_bootstrap_ci_high":float(np.quantile(boot,0.975)),
                 "bootstrap_seed":2026101901,"bootstrap_draws":5000,"cluster_count":len(group_keys)}
            for omitted in (20260825,20260826,20260827):
                row[f"loo_without_{omitted}"]=float(np.mean([deltas[s] for s in deltas if s!=omitted]))
            contrast_rows.append(row)
        # Draw manifest is committed after score tables below.

    if not first_metric_recorded: raise RuntimeError("no final metric was computed")
    # Sidecar-correction branch: class sets are identical; common-class subset is the same 602/120 estimand.
    if model_set==final_set:
        common_rows=[{"status":"NOT_APPLICABLE_CLASS_SETS_IDENTICAL_TO_FULL","estimand":"A-COMMON590",
                      "n_rows":602,"common_class_count":120,"full_population_rows":602,
                      "reason":"The train-target-derived model class set equals the v3 primary final set after correcting the V1 sidecar; a separate common-class subset would be identical to A-FULL602."}]
    else:
        common_rows=[dict(r) for r in per_rows if r["estimand"]=="A-COMMON590"]
    atomic_csv(out/"core_a/FULL602_PER_SEED.csv",per_rows,list(per_rows[0]))
    atomic_csv(out/"core_a/FULL602_SUMMARY.csv",summary_rows,list(summary_rows[0]))
    atomic_csv(out/"core_a/PAIRED_CONTRASTS_AND_CI.csv",contrast_rows,list(contrast_rows[0]))
    atomic_csv(out/"core_a/SEED_LOO.csv",loo_rows,list(loo_rows[0]))
    detail_fields=["record_type","estimand","program","seed","true_task_id","predicted_task_id","count",
                   "support","predicted_count","tp","fp","fn","precision","recall","f1","top5_recall","true_task_name"]
    atomic_csv(out/"core_a/CONFUSION_AND_PER_CLASS.csv",detail_rows,detail_fields)
    atomic_csv(out/"core_a/COMMON_CLASS_PER_SEED.csv",common_rows,list(common_rows[0]))
    boot_manifest={"schema_version":1,"status":"PASS","protocol_id":out.name,
        "estimands":draw_manifest_by_estimand,"interval_semantics":"95 percent percentile cluster interval conditional on the six fitted checkpoints; excludes training-seed uncertainty"}
    atomic_json(out/"core_a/BOOTSTRAP_DRAW_MANIFEST.json",boot_manifest)
    commit_utc=utc()
    attempt.update({"status":"COMPLETED","score_table_commit_utc":commit_utc,
        "metric_values_computed":True,"score_tables_created":True,"bootstrap_draws_run":5000,
        "output_paths":[str(out/"core_a/FULL602_PER_SEED.csv"),str(out/"core_a/FULL602_SUMMARY.csv"),
                         str(out/"core_a/PAIRED_CONTRASTS_AND_CI.csv"),str(out/"core_a/CONFUSION_AND_PER_CLASS.csv")]})
    atomic_json(attempt_path,attempt,replace=True)
    exposure_summary={"schema_version":1,"protocol_id":out.name,"status":"METRICS_COMPUTED_AND_TABLES_COMMITTED",
        "v1_prior_label_reads":2,"v1_values_decoded":1204,
        "v2_pre_score_label_manifest_rows_materialized":614,"v2_split_manifest_reads":6,
        "v2_split_manifest_final_task_metadata_rows":3684,
        "v2_score_attempts":[str(out/"core_a/CORE_A_SCORE_ATTEMPT_01.json"),str(attempt_path)],
        "v2_score_label_rows_materialized_total":1204,"v2_unique_primary_rows_scored":602,
        "cumulative_explicit_final_label_task_id_values_decoded_including_v1":2408,
        "v2_pre_score_label_bearing_metadata_row_materializations":{
            "final_test_label_manifest_rows":614,
            "split_manifest_reads":6,
            "final_task_label_metadata_rows_per_split_read":614,
            "split_manifest_final_task_label_metadata_rows_total":3684,
            "all_pre_score_label_bearing_metadata_rows_total":4298},
        "unique_final_source_rows_touched_total":614,"first_v2_label_read_utc":prior_exposure["first_v2_label_read_started_utc"],
        "attempt02_label_read_started_utc":attempt["started_utc"],
        "first_v2_metric_computed_utc":attempt["first_metric_computed_utc"],"score_table_commit_utc":commit_utc,
        "metrics_computed":True,"final_score_exposed":True,"estimands_computed":list(estimands),
        "bootstrap_draws":5000,"first_attempt_metrics_computed":False}
    atomic_json(out/"core_a/SCORE_EXPOSURE_SUMMARY.json",exposure_summary)
    receipt={"schema_version":1,"protocol_id":out.name,"status":"COMPLETED_AS_AMENDED_ANALYSIS",
        "estimand":"A-FULL602_corrected_closed_set_task_recognition",
        "correction":"V1 class-map sidecar used source contiguous labels 0..119; accepted training targets used the train/dev-derived local_label_map. The corrected map matches v3 primary final classes; predictions/checkpoints unchanged.",
        "population_rows":602,"model_classes":120,"final_primary_classes":120,"C_model_equals_C_final":model_set==final_set,
        "unmapped_final_rows":int(sum(int(x) not in model_set for x in y)),"sample_coverage":1.0,"class_coverage":1.0,
        "macro_f1_definition":{"labels_by_estimand":{k:v["labels"] for k,v in estimand_inputs.items()},
            "average":"macro","zero_division":0},
        "ranking_tie_rule":"descending logit; ties ordered by lower original output column index",
        "results":{"per_seed":str(out/"core_a/FULL602_PER_SEED.csv"),"summary":str(out/"core_a/FULL602_SUMMARY.csv"),
                   "paired_contrasts":str(out/"core_a/PAIRED_CONTRASTS_AND_CI.csv"),"seed_loo":str(out/"core_a/SEED_LOO.csv"),
                   "confusion_and_per_class":str(out/"core_a/CONFUSION_AND_PER_CLASS.csv"),
                   "common_class_status":str(out/"core_a/COMMON_CLASS_PER_SEED.csv"),
                   "bootstrap_manifest":str(out/"core_a/BOOTSTRAP_DRAW_MANIFEST.json")},
        "score_exposure_path":str(out/"core_a/SCORE_EXPOSURE_SUMMARY.json"),
        "first_v2_label_read_utc":prior_exposure["first_v2_label_read_started_utc"],
        "first_v2_metric_computed_utc":attempt["first_metric_computed_utc"],
        "score_tables_committed_utc":commit_utc,
        "final_label_values_materialized_this_successful_score":602,"successful_score_runs":1,
        "total_v2_score_attempts":2,"v2_score_attempt01_label_rows_materialized":602,
        "bootstrap_draws_per_estimand":5000,
        "evidence_mode":"same-source v3 primary partition; prior label-bearing metadata and V1 label reads disclosed; not untouched blind or session-independent confirmation",
        "new_training_runs":0}
    atomic_json(out/"core_a/CORE_A_SCORE_RECEIPT.json",receipt)
    print(json.dumps({"status":receipt["status"],"per_seed_rows":len(per_rows),"summary_rows":len(summary_rows),
                      "contrast_rows":len(contrast_rows),"common_status":common_rows[0]["status"] if common_rows else "COMPUTED",
                      "first_metric_utc":receipt["first_v2_metric_computed_utc"]},ensure_ascii=False))
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v1-out",type=Path,required=True)
    ap.add_argument("--v2-out",type=Path,required=True)
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--preflight-only",action="store_true")
    args=ap.parse_args()
    v1=args.v1_out.resolve();out=args.v2_out.resolve();run=args.run_root.resolve()
    if args.preflight_only:
        pre=load_preflight(v1,out)
        print(json.dumps({"status":"PREDICTION_PREFLIGHT_PASS","cells":6,"rows_per_cell":602,
            "prediction_manifest_sha256":pre["manifest_sha256"],"canonical_ids_sha256":pre["canonical_ids_sha256"],
            "corrected_class_map_sha256":pre["column_map_sha256"]},ensure_ascii=False))
        return 0
    return run_score(v1,out,run)

if __name__=="__main__":
    raise SystemExit(main())


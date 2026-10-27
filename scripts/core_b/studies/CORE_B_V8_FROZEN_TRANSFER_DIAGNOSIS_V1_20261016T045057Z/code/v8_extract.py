from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT/"reports/benchmarks"
V8 = Path(__file__).resolve().parents[1]
V7 = BENCH/"CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z"
V2 = BENCH/"CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
V4 = BENCH/"CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
V5 = BENCH/"CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)
METHODS=("G","U","M","S","V","T")
METHOD_MAP={"G":"GT_RETRAIN","U":"SELF_SOFT","M":"MIX_SOFT","S":"S","V":"V","T":"T"}
GROUP_KEY={"train":"episode_id","dev":"source_group"}
EPS=1e-12
COUNTERS=defaultdict(int)
sys.path.insert(0,str(V7/"code"))
from v7_common import position_fd_seam_metrics  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def sha(path: Path|str) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def read_json(path: Path|str) -> dict[str,Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path|str, value: dict[str,Any]) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(f".{p.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tmp.replace(p)


def write_csv(path: Path|str, fields: list[str], rows: list[dict[str,Any]]) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(f".{p.name}.{os.getpid()}.part")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(rows)
    tmp.replace(p)


def read_csv(path: Path|str) -> list[dict[str,str]]:
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canon_rows(path: Path|str) -> list[dict[str,str]]:
    rows=read_csv(path)
    if not rows: raise RuntimeError(f"empty receiver rows: {path}")
    result=[]
    for r in rows:
        group=r.get("source_group",r.get("source_cluster"))
        if group is None: raise RuntimeError(f"missing source group field: {path}")
        result.append({"anchor_id":str(r["anchor_id"]),"episode_id":str(int(r["episode_id"])),
                       "task_id":str(int(r["task_id"])),"source_group":str(group),
                       "anchor_frame":str(int(r.get("anchor_frame",0)))})
    ids=[r["anchor_id"] for r in result]
    if len(ids)!=len(set(ids)): raise RuntimeError(f"duplicate anchor IDs: {path}")
    if set(int(r["task_id"]) for r in result)!=set(TASKS): raise RuntimeError(f"task IDs mismatch: {path}")
    return result


def same_identity_set(a: list[dict[str,str]], b: list[dict[str,str]]) -> bool:
    return {r["anchor_id"]:r for r in a}=={r["anchor_id"]:r for r in b}


def load_prediction(path: Path|str) -> np.ndarray:
    with safe_open(str(path),framework="numpy",device="cpu") as f:
        keys=set(f.keys())
        allowed={"prediction_native","action_residual"}
        if "prediction_native" not in keys or keys-allowed:
            raise RuntimeError(f"prediction tensor key whitelist failure: {path}: {sorted(keys)}")
        arr=np.asarray(f.get_tensor("prediction_native"))
    if arr.ndim!=3 or arr.shape[1:]!=(30,28) or not np.isfinite(arr).all():
        raise RuntimeError(f"prediction shape/finite failure: {path}: {arr.shape}")
    if arr.dtype not in (np.float32,np.float64):
        raise RuntimeError(f"prediction dtype failure: {path}: {arr.dtype}")
    return arr.astype(np.float64,copy=False)


def id_ordered_prediction(path: Path|str, row_path: Path|str,
                          canonical: list[dict[str,str]]) -> np.ndarray:
    rows=canon_rows(row_path)
    if not same_identity_set(rows,canonical):
        raise RuntimeError(f"prediction ID/metadata support differs: {path}")
    pred=load_prediction(path)
    if pred.shape[0]!=len(rows): raise RuntimeError(f"prediction row count mismatch: {path}")
    src={r["anchor_id"]:i for i,r in enumerate(rows)}
    order=np.asarray([src[r["anchor_id"]] for r in canonical],dtype=np.int64)
    return pred[order]


def task_indices(rows: list[dict[str,str]], task: int) -> np.ndarray:
    return np.asarray([i for i,r in enumerate(rows) if int(r["task_id"])==int(task)],dtype=np.int64)


def row_weights(rows: list[dict[str,str]], indices: np.ndarray, group_key: str) -> np.ndarray:
    groups=Counter(str(rows[int(i)][group_key]) for i in indices)
    if not len(indices) or not groups: raise RuntimeError("cannot weight empty task support")
    out=np.empty(len(indices),dtype=np.float64)
    for j,i in enumerate(indices):
        group=str(rows[int(i)][group_key])
        out[j]=1.0/(len(groups)*groups[group])
    if not np.isclose(out.sum(),1.0,rtol=0.0,atol=1e-12):
        raise RuntimeError(f"within-task weights sum to {out.sum()}")
    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float|np.ndarray:
    values=np.asarray(values,dtype=np.float64)
    if values.shape[0]!=len(weights) or not np.isfinite(values).all():
        raise RuntimeError("weighted mean input shape/finite failure")
    result=np.tensordot(weights,values,axes=(0,0))
    if np.ndim(result)==0: return float(result)
    return np.asarray(result,dtype=np.float64)


def metric_and_rms(pred: np.ndarray, target: np.ndarray, rows: list[dict[str,str]],
                   task: int, split: str, sigma: np.ndarray,
                   fd_scale: np.ndarray, anchor: np.ndarray|None=None,
                   count_key: str="canonical_metric") -> dict[str,float]:
    idx=task_indices(rows,task)
    COUNTERS[count_key]+=len(idx)
    w=row_weights(rows,idx,GROUP_KEY[split])
    p=pred[idx]; y=target[idx]
    a=np.zeros((len(idx),28),dtype=np.float64) if anchor is None else anchor[idx]
    pos,fd,seam,_=position_fd_seam_metrics(p,y,a,sigma,fd_scale)
    pos_delta=(p-y)/sigma[None,None,:]
    fd_delta=np.diff(p-y,axis=1)/fd_scale[None,None,:]
    seam_delta=(p[:,0]-y[:,0])/fd_scale[None,:]
    return {
        "position_mae":weighted_mean(pos,w),
        "safe_fd_l1":weighted_mean(fd,w),
        "seam_safe_fd_l1":weighted_mean(seam,w),
        "position_rms":math.sqrt(max(weighted_mean(np.mean(pos_delta**2,axis=(1,2)),w),0.0)),
        "safe_fd_rms":math.sqrt(max(weighted_mean(np.mean(fd_delta**2,axis=(1,2)),w),0.0)),
        "seam_rms":math.sqrt(max(weighted_mean(np.mean(seam_delta**2,axis=1),w),0.0)),
    }


def runner_j_windows(pred: np.ndarray, target: np.ndarray,
                     train_sigma: np.ndarray, train_fd: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    # Exact V7 runner reduction, applied to final eval-mode native float64 output.
    pos_abs=np.abs((pred-target)/train_sigma[None,None,:])
    pos=0.5*(pos_abs[...,:14].mean(axis=(1,2))+pos_abs[...,14:].mean(axis=(1,2)))
    fd_abs=np.abs(np.diff(pred-target,axis=1)/train_fd[None,None,:])
    fd=0.5*(fd_abs[...,:14].mean(axis=(1,2))+fd_abs[...,14:].mean(axis=(1,2)))
    return pos,fd,pos+0.1*fd


def runner_j_task(pred: np.ndarray,target: np.ndarray,rows: list[dict[str,str]],
                  task: int,train_sigma: np.ndarray,train_fd: np.ndarray,
                  count_key: str="runner_J") -> dict[str,float]:
    idx=task_indices(rows,task)
    COUNTERS[count_key]+=len(idx)
    w=row_weights(rows,idx,GROUP_KEY["train"])
    p,fd,j=runner_j_windows(pred[idx],target[idx],train_sigma,train_fd)
    return {"J_position_term":weighted_mean(p,w),"J_internal_fd_term":weighted_mean(fd,w),
            "J_runner":weighted_mean(j,w)}


def canonical_rows_and_arrays() -> tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    source=read_json(V7/"protocol/SOURCE_MAP.json")
    fit_index=read_json(V7/"protocol/FIT_ACCEPTANCE_INDEX.json")
    teacher_seal=read_json(V7/"protocol/TEACHER_BANK_SEAL.json")
    target_lock=read_json(V7/"protocol/TARGET_LOCK.json")
    pred_seal=read_json(V7/"qa/PREDICTION_SEAL.json")
    if read_json(V8/"qa/SYNTHETIC_QA.json").get("status")!="PASS":
        raise RuntimeError("pre-label synthetic QA is not PASS")
    if read_json(V8/"coordination/PREFLIGHT_COMPLETE.json").get("status")!="PASS":
        raise RuntimeError("metadata-only source preflight is not PASS")
    if read_json(V8/"coordination/FROZEN_FORWARD_EXECUTION.json").get("status")!="PASS":
        raise RuntimeError("six frozen train prefix forwards are not all accepted")
    replay=read_json(V8/"protocol/PREDICTION_REUSE_AND_REPLAY.json")
    if replay.get("status")!="PASS" or replay.get("new_dev_prediction_jobs")!=0:
        raise RuntimeError("prediction reuse/replay receipt violates the zero-dev-prediction lock")

    # No labels are loaded in this section.
    train_entries={(int(e["seed"]),str(e["teacher_role"])):e for e in teacher_seal["entries"]}
    train_rows={}
    train_pred={s:{} for s in SEEDS}
    for seed in SEEDS:
        base_e=train_entries[(seed,"V")]
        base_rows=canon_rows(base_e["prediction_rows_path"])
        train_rows[seed]=base_rows
        for role in ("S","V","T"):
            e=train_entries[(seed,role)]
            if not same_identity_set(canon_rows(e["prediction_rows_path"]),base_rows):
                raise RuntimeError(f"train {role} IDs differ for seed {seed}")
            train_pred[seed][role]=id_ordered_prediction(e["prediction_path"],e["prediction_rows_path"],base_rows)
    fit_by={(str(e["program_id"]),int(e["seed"])):e for e in fit_index["entries"]}
    alias_records=[]
    for seed in SEEDS:
        gt=fit_by[("GT_RETRAIN",seed)]
        old_v=next(e for e in source["models"] if e.get("role")=="V" and int(e["seed"])==seed)
        entry=train_entries[(seed,"V")]
        if not (gt["model_sha256"]==old_v["model_sha256"]==entry["source_model_sha256"]):
            raise RuntimeError(f"G=V train model SHA mismatch for seed {seed}")
        # G is the accepted V_i model output on the same V7 train prefix recipe.
        train_pred[seed]["G"]=train_pred[seed]["V"]
        alias_records.append({
            "seed":seed,"train_G_alias":"V_i train prediction",
            "GT_RETRAIN_model_sha256":gt["model_sha256"],
            "accepted_V_model_sha256":old_v["model_sha256"],
            "V_train_prediction_path":entry["prediction_path"],
            "V_train_prediction_sha256":entry["prediction_sha256"],
            "ordered_receiver_ids_sha256":entry["ordered_receiver_ids_sha256"],
            "normalizer_sha256":source["data"]["normalizer_sha256"],
            "inference_profile":"V7 same v7_teacher worker, TS11-D, FP32 eval/inference_mode, batch32, native float64 restore, accepted Q0 cache",
            "alias_basis":"exact accepted model-file bytes and same input/preprocessing/profile/IDs",
        })
        for program,method in (("SELF_SOFT","U"),("MIX_SOFT","M")):
            base=V8/"predictions_train"/program/f"seed{seed}"/"attempt_01"
            complete=read_json(base/"PREDICTION_COMPLETE.json")
            if complete.get("status")!="PASS" or int(complete.get("seed",-1))!=seed:
                raise RuntimeError(f"new frozen train forward missing for {program}/{seed}")
            if complete.get("model_sha256")!=fit_by[(program,seed)]["model_sha256"]:
                raise RuntimeError(f"frozen train output model identity mismatch {program}/{seed}")
            train_rows_path=Path(complete["prediction_rows_path"])
            if not same_identity_set(canon_rows(train_rows_path),train_rows[seed]):
                raise RuntimeError(f"frozen train IDs mismatch {program}/{seed}")
            train_pred[seed][method]=id_ordered_prediction(complete["prediction_path"],train_rows_path,train_rows[seed])

    # Dev predictions are kept as separate methods. The V2/V5 receiver order is
    # different, so each tensor is joined to V7 GT_RETRAIN by explicit anchor ID.
    refs=source["evaluation_reference_bundles"]
    dev_ref={(int(e["seed"]),str(e["role"])):e for e in refs
             if (e["source_root"]=="V2" and e["role"] in {"S","V"}) or
                (e["source_root"]=="V5" and e["role"]=="T")}
    pred_entries={(int(e["seed"]),str(e["program_id"])):e for e in pred_seal["entries"]}
    dev_rows={}
    dev_pred={s:{} for s in SEEDS}
    dev_alias_checks=[]
    for seed in SEEDS:
        ge=pred_entries[(seed,"GT_RETRAIN")]
        base_rows=canon_rows(ge["prediction_rows_path"])
        dev_rows[seed]=base_rows
        for program,method in (("GT_RETRAIN","G"),("SELF_SOFT","U"),("MIX_SOFT","M")):
            e=pred_entries[(seed,program)]
            dev_pred[seed][method]=id_ordered_prediction(e["prediction_path"],e["prediction_rows_path"],base_rows)
        for method in ("S","V","T"):
            e=dev_ref[(seed,method)]
            dev_pred[seed][method]=id_ordered_prediction(e["prediction_path"],e["rows_path"],base_rows)
        diff=np.abs(dev_pred[seed]["G"]-dev_pred[seed]["V"])
        dev_alias_checks.append({
            "seed":seed,"dev_G_vs_V_prediction_tensor_alias":"NOT_ASSERTED; retain separate accepted tensors",
            "same_receiver_ID_set":True,
            "max_abs_output_difference":float(diff.max()),
            "mean_abs_output_difference":float(diff.mean()),
            "max_relative_output_difference":float((diff/np.maximum(np.abs(dev_pred[seed]["V"]),EPS)).max()),
            "accepted_score_reconciliation_max_abs":1.7763568394002505e-15,
            "interpretation":"the score reconciliation tolerance is not a tensor tolerance; direct tensor difference is reported without collapsing G and V",
        })

    # Before loading labels, verify accepted train grouping matches the V7
    # episode/task weighting contract.
    train_group_qa=[]
    for seed in SEEDS:
        rows=train_rows[seed]
        for task in TASKS:
            subset=[r for r in rows if int(r["task_id"])==task]
            ep_to_group=defaultdict(set);group_to_ep=defaultdict(set)
            for r in subset:
                ep_to_group[r["episode_id"]].add(r["source_group"])
                group_to_ep[r["source_group"]].add(r["episode_id"])
            if any(len(x)!=1 for x in ep_to_group.values()) or any(len(x)!=1 for x in group_to_ep.values()):
                raise RuntimeError(f"train episode/source-group mapping is not one-to-one for task {task}")
            train_group_qa.append({"seed":seed,"task_id":task,"anchors":len(subset),
                                   "episodes":len(ep_to_group),"source_groups":len(group_to_ep),
                                   "episode_group_one_to_one":True})
    return ({"source":source,"fit_index":fit_index,"teacher_seal":teacher_seal,
             "target_lock":target_lock,"pred_seal":pred_seal,"alias_records":alias_records,
             "dev_alias_checks":dev_alias_checks,"train_group_qa":train_group_qa},
            {"rows":train_rows,"pred":train_pred},
            {"rows":dev_rows,"pred":dev_pred})


def load_training_targets(target_lock: dict[str,Any], train_rows: dict[int,list[dict[str,str]]]
                          ) -> tuple[dict[str,np.ndarray],dict[int,dict[str,np.ndarray]],dict[str,int]]:
    if target_lock.get("status")!="LOCKED" or not target_lock.get("Y_native_read_after_teacher_seal"):
        raise RuntimeError("accepted V7 train target lock not ready")
    y_path=Path(target_lock["Y_native_path"])
    if sha(y_path)!=target_lock["Y_native_sha256"]: raise RuntimeError("V7 Y_NATIVE digest mismatch")
    y=np.load(y_path,allow_pickle=False)
    if y.shape!=(1280,30,28) or y.dtype!=np.float64 or not np.isfinite(y).all():
        raise RuntimeError(f"V7 Y_NATIVE shape/dtype/finite mismatch: {y.shape}/{y.dtype}")
    targets={}; target_by_seed={s:{} for s in SEEDS}
    entries=target_lock["target_entries"]
    if len(entries)!=9: raise RuntimeError(f"expected 9 locked Z arrays, found {len(entries)}")
    for e in entries:
        seed=int(e["seed"]); program=str(e["program"])
        p=Path(e["target_path"])
        if sha(p)!=e["target_sha256"]: raise RuntimeError(f"V7 target digest mismatch: {p}")
        z=np.load(p,allow_pickle=False)
        if z.shape!=(1280,30,28) or z.dtype!=np.float64 or not np.isfinite(z).all():
            raise RuntimeError(f"V7 target shape/dtype/finite mismatch: {p}")
        target_by_seed[seed][program]=z
        targets[f"{seed}:{program}"]=z
    return {"Y":y},target_by_seed,{"train_Y_npy_loads":1,"train_Z_npy_loads":9,
                                    "train_target_archive_loads":0,"train_target_values_read":1}


def load_dev_targets_and_anchors(source: dict[str,Any], dev_rows: dict[int,list[dict[str,str]]]
                                      ) -> tuple[dict[int,np.ndarray],dict[int,np.ndarray],dict[str,Any]]:
    receipt=read_json(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json")
    target_path=Path(receipt["target_archive_path"])
    if sha(target_path)!=receipt["target_archive_sha256"]:
        raise RuntimeError("accepted original dev target archive hash mismatch")
    if int(receipt.get("target_windows",-1))!=185 or int(receipt.get("final_heldout_payload_access",-1))!=0:
        raise RuntimeError("original dev target receipt is not 185-window/final0")
    required={"anchor_id","episode_id","task_id","source_cluster","target_native_h30"}
    with np.load(target_path,allow_pickle=False) as archive:
        if not required.issubset(set(archive.files)):
            raise RuntimeError(f"dev target archive schema missing fields: {sorted(required-set(archive.files))}")
        target_ids=np.asarray(archive["anchor_id"]).astype(str)
        target_episode=np.asarray(archive["episode_id"]).astype(np.int64)
        target_task=np.asarray(archive["task_id"]).astype(np.int64)
        target_group=np.asarray(archive["source_cluster"]).astype(str)
        raw_target=np.asarray(archive["target_native_h30"],dtype=np.float64)
    if raw_target.shape!=(185,30,28) or not np.isfinite(raw_target).all() or len(set(target_ids))!=185:
        raise RuntimeError("dev target shape/finite/ID support mismatch")
    target_map={target_ids[i]:i for i in range(len(target_ids))}

    prefix_path=Path(source["data"]["dev_prefix_npz"])
    if sha(prefix_path)!=source["data"]["dev_prefix_npz_sha256"]:
        raise RuntimeError("dev prefix archive digest mismatch")
    prefix_required={"anchor_id","episode_id","task_id","source_cluster","anchor_native"}
    with np.load(prefix_path,allow_pickle=False) as archive:
        if not prefix_required.issubset(set(archive.files)):
            raise RuntimeError(f"dev prefix schema missing fields: {sorted(prefix_required-set(archive.files))}")
        prefix_ids=np.asarray(archive["anchor_id"]).astype(str)
        prefix_episode=np.asarray(archive["episode_id"]).astype(np.int64)
        prefix_task=np.asarray(archive["task_id"]).astype(np.int64)
        prefix_group=np.asarray(archive["source_cluster"]).astype(str)
        raw_anchor=np.asarray(archive["anchor_native"],dtype=np.float64)
    if raw_anchor.shape!=(185,28) or len(set(prefix_ids))!=185 or set(prefix_ids)!=set(target_ids):
        raise RuntimeError("dev anchor metadata identity mismatch")
    prefix_map={prefix_ids[i]:i for i in range(len(prefix_ids))}
    dev_target={}; dev_anchor={}
    for seed in SEEDS:
        rows=dev_rows[seed]
        y=np.empty((185,30,28),dtype=np.float64); a=np.empty((185,28),dtype=np.float64)
        for i,row in enumerate(rows):
            aid=row["anchor_id"]
            ti=target_map[aid]; pi=prefix_map[aid]
            if (int(target_episode[ti])!=int(row["episode_id"]) or int(target_task[ti])!=int(row["task_id"])
                or str(target_group[ti])!=str(row["source_group"])):
                raise RuntimeError(f"dev target metadata mismatch for {aid}")
            if (int(prefix_episode[pi])!=int(row["episode_id"]) or int(prefix_task[pi])!=int(row["task_id"])
                or str(prefix_group[pi])!=str(row["source_group"])):
                raise RuntimeError(f"dev prefix metadata mismatch for {aid}")
            y[i]=raw_target[ti];a[i]=raw_anchor[pi]
        dev_target[seed]=y;dev_anchor[seed]=a
    if not np.isfinite(raw_anchor).all():
        raise RuntimeError("dev anchors contain nonfinite values")
    access={
        "dev_target_archive_loads":1,"dev_target_rows_loaded":185,"dev_target_unique_ids":len(set(target_ids)),
        "dev_prefix_archive_loads":1,"dev_prefix_rows_loaded":185,
        "dev_target_path":str(target_path),"dev_target_sha256":sha(target_path),
        "dev_target_receipt_path":str(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),
        "dev_target_receipt_sha256":sha(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),
        "dev_target_keys_loaded":sorted(required),"dev_prefix_keys_loaded":sorted(prefix_required),
        "final_heldout_payload_access":0,
    }
    return dev_target,dev_anchor,access


def build_train_target_and_fit(rows:dict[int,list[dict[str,str]]],pred:dict[int,dict[str,np.ndarray]],
                               y_train:np.ndarray,zs:dict[int,dict[str,np.ndarray]],
                               metric_sigma:np.ndarray,metric_fd:np.ndarray,
                               train_sigma:np.ndarray,train_fd:np.ndarray) -> list[dict[str,Any]]:
    raw=[]
    perturb=[("Y","T"),("V","T"),("Z_M","Y"),("Z_M","Z_U"),("Z_U","Y")]
    for seed in SEEDS:
        r=rows[seed]; z_seed=zs[seed]
        targets={"Y":y_train,"T":pred[seed]["T"],"Z_G":z_seed["GT_RETRAIN"],
                 "Z_U":z_seed["SELF_SOFT"],"Z_M":z_seed["MIX_SOFT"]}
        for task in TASKS:
            for left,right in perturb:
                l=y_train if left=="Y" else pred[seed]["T"] if left=="T" else pred[seed]["V"] if left=="V" else targets[left]
                rr=y_train if right=="Y" else pred[seed]["T"] if right=="T" else pred[seed]["V"] if right=="V" else targets[right]
                m=metric_and_rms(l,rr,r,task,"train",metric_sigma,metric_fd,
                                 count_key="train_target_perturbation")
                raw.append({"split":"train","scope":"task_seed","seed":seed,"task_id":task,
                            "record_type":"target_perturbation","method":left,"target":right,"reference":left,
                            **m,"J_position_term":"","J_internal_fd_term":"","J_runner":"",
                            "J_gain_position":"","J_gain_internal_fd":"","J_gain_runner":"",
                            "notes":"native float64 target distance; canonical position/internal-FD/seam scales"})
            method_pred={"G":pred[seed]["G"],"U":pred[seed]["U"],"M":pred[seed]["M"]}
            method_own={"G":"Z_G","U":"Z_U","M":"Z_M"}
            selected_targets={
                "G":["Y","T","Z_G","Z_M","Z_U"],
                "U":["Y","T","Z_U","Z_M","Z_U"],
                "M":["Y","T","Z_M","Z_M","Z_U"],
            }
            # Include own-target and both shared-target comparisons; duplicate
            # own/shared rows are retained with distinct record_type labels.
            metric_cache={}
            j_cache={}
            for method in ("G","U","M"):
                for j,target_name in enumerate(selected_targets[method]):
                    record_type="own_or_truth_fit" if j<3 else ("common_ZM_comparison" if j==3 else "common_ZU_comparison")
                    target=targets[target_name]
                    cache_target="Y" if target_name=="Z_G" else target_name
                    cache_key=(method,cache_target)
                    if cache_key not in metric_cache:
                        metric_cache[cache_key]=metric_and_rms(
                            method_pred[method],target,r,task,"train",metric_sigma,metric_fd,
                            count_key="train_model_target")
                        j_cache[cache_key]=runner_j_task(
                            method_pred[method],target,r,task,train_sigma,train_fd,
                            count_key="train_objective_J")
                    m=metric_cache[cache_key]
                    jv=j_cache[cache_key]
                    raw.append({"split":"train","scope":"task_seed","seed":seed,"task_id":task,
                                "record_type":record_type,"method":method,"target":target_name,
                                "reference":"Y" if target_name=="Y" else ("T" if target_name=="T" else ""),
                                **m,**jv,"J_gain_position":"","J_gain_internal_fd":"",
                                "J_gain_runner":"",
                                "notes":"J_eval_on_train: eval-mode final output, native float64, fixed Y-fitted training scales; differs from dropout-on pre-update curve"})
            for name,cand,ref,target_name in (
                ("TRAIN_Z_GAIN_M","M","G","Z_M"),("TRAIN_Z_GAIN_U","U","G","Z_U")):
                target=targets[target_name]
                cache_target="Y" if target_name=="Z_G" else target_name
                jc=j_cache[(cand,cache_target)]
                jr=j_cache[(ref,cache_target)]
                raw.append({"split":"train","scope":"task_seed","seed":seed,"task_id":task,
                            "record_type":name,"method":cand,"target":target_name,"reference":ref,
                            "position_mae":"","safe_fd_l1":"","seam_safe_fd_l1":"",
                            "position_rms":"","safe_fd_rms":"","seam_rms":"",
                            "J_position_term":jc["J_position_term"],
                            "J_internal_fd_term":jc["J_internal_fd_term"],"J_runner":jc["J_runner"],
                            "J_gain_position":jr["J_position_term"]-jc["J_position_term"],
                            "J_gain_internal_fd":jr["J_internal_fd_term"]-jc["J_internal_fd_term"],
                            "J_gain_runner":jr["J_runner"]-jc["J_runner"],
                            "notes":"positive gain means candidate has lower fixed-target runner objective than hard-supervised comparator"})
    keys=("record_type","method","target","reference")
    numeric=("position_mae","safe_fd_l1","seam_safe_fd_l1","position_rms","safe_fd_rms","seam_rms",
             "J_position_term","J_internal_fd_term","J_runner","J_gain_position","J_gain_internal_fd","J_gain_runner")
    macros=[]
    for seed in SEEDS:
        subset=[r for r in raw if r["seed"]==seed]
        groups=defaultdict(list)
        for row in subset: groups[tuple(row[k] for k in keys)].append(row)
        for key,items in groups.items():
            out={"split":"train","scope":"equal_task_macro","seed":seed,"task_id":"ALL",
                 **dict(zip(keys,key)),"notes":"equal mean of the five locked task-specific values"}
            for col in numeric:
                vals=[float(x[col]) for x in items if x.get(col) not in ("",None)]
                out[col]=float(np.mean(vals)) if vals else ""
            macros.append(out)
    fields=["split","scope","seed","task_id","record_type","method","target","reference",
            "position_mae","safe_fd_l1","seam_safe_fd_l1","position_rms","safe_fd_rms","seam_rms",
            "J_position_term","J_internal_fd_term","J_runner","J_gain_position","J_gain_internal_fd",
            "J_gain_runner","notes"]
    return raw+macros


def build_train_dev_imitation(train_rows:dict[int,list[dict[str,str]]],
                              train_pred:dict[int,dict[str,np.ndarray]],
                              dev_rows:dict[int,list[dict[str,str]]],
                              dev_pred:dict[int,dict[str,np.ndarray]],
                              metric_sigma:np.ndarray,metric_fd:np.ndarray,
                              dev_anchor:dict[int,np.ndarray]) -> list[dict[str,Any]]:
    rows=[]
    for split,all_rows,all_pred,anchors in (
        ("train",train_rows,train_pred,{s:None for s in SEEDS}),
        ("dev",dev_rows,dev_pred,dev_anchor)):
        raw=[]
        for seed in SEEDS:
            for task in TASKS:
                for method in ("G","U","M"):
                    m=metric_and_rms(all_pred[seed][method],all_pred[seed]["T"],all_rows[seed],
                                     task,split,metric_sigma,metric_fd,anchors[seed],
                                     count_key=f"{split}_imitation")
                    raw.append({"split":split,"scope":"task_seed","seed":seed,"task_id":task,
                                "method":method,"target":"T","position_distance":m["position_mae"],
                                "safe_fd_distance":m["safe_fd_l1"],"seam_distance":m["seam_safe_fd_l1"],
                                "position_rms":m["position_rms"],"safe_fd_rms":m["safe_fd_rms"],
                                "seam_rms":m["seam_rms"],"position_imitation_gain_vs_G":"",
                                "safe_fd_imitation_gain_vs_G":"","seam_imitation_gain_vs_G":"",
                                "note":"gain=d(G,T)-d(method,T); positive is closer to T"})
        for seed in SEEDS:
            for task in TASKS:
                g=next(r for r in raw if r["seed"]==seed and r["task_id"]==task and r["method"]=="G")
                for row in [r for r in raw if r["seed"]==seed and r["task_id"]==task]:
                    row["position_imitation_gain_vs_G"]=float(g["position_distance"])-float(row["position_distance"])
                    row["safe_fd_imitation_gain_vs_G"]=float(g["safe_fd_distance"])-float(row["safe_fd_distance"])
                    row["seam_imitation_gain_vs_G"]=float(g["seam_distance"])-float(row["seam_distance"])
        for seed in SEEDS:
            for method in ("G","U","M"):
                items=[r for r in raw if r["seed"]==seed and r["method"]==method]
                out={"split":split,"scope":"equal_task_macro","seed":seed,"task_id":"ALL",
                     "method":method,"target":"T","note":"equal mean across five tasks"}
                for col in ("position_distance","safe_fd_distance","seam_distance","position_rms","safe_fd_rms","seam_rms",
                            "position_imitation_gain_vs_G","safe_fd_imitation_gain_vs_G","seam_imitation_gain_vs_G"):
                    vals=[float(r[col]) for r in items]
                    out[col]=float(np.mean(vals))
                rows.append(out)
        rows.extend(raw)
    return rows


def build_curve_summary() -> list[dict[str,Any]]:
    curves=read_csv(V7/"results/TRAINING_CURVES.csv")
    cols=set(curves[0])
    expected={"program","seed","epoch","optimizer_steps","weighted_objective_loss",
              "fit_position_mae","fit_safe_fd_l1","fit_seam_safe_fd_l1",
              "ground_truth_position_mae","ground_truth_safe_fd_l1","ground_truth_seam_safe_fd_l1",
              "student_to_T_position_distance","student_to_T_safe_fd_distance","student_to_T_seam_distance"}
    if not expected.issubset(cols): raise RuntimeError(f"V7 curve schema missing {sorted(expected-cols)}")
    rows=[]
    for program in ("GT_RETRAIN","SELF_SOFT","MIX_SOFT"):
        for seed in SEEDS:
            subset=[r for r in curves if r["program"]==program and int(r["seed"])==seed]
            subset.sort(key=lambda r:int(r["epoch"]))
            if len(subset)!=160 or int(subset[-1]["epoch"])!=160 or int(subset[-1]["optimizer_steps"])!=3200:
                raise RuntimeError(f"V7 training curve support mismatch {program}/{seed}")
            last=subset[-1];tail=subset[-20:]
            objective=np.asarray([float(r["weighted_objective_loss"]) for r in tail],dtype=np.float64)
            finite_all=all(math.isfinite(float(r[c])) for r in subset for c in cols if c not in {"program","seed"})
            rows.append({
                "program":program,"seed":seed,"epochs":len(subset),"last_epoch":160,
                "last_optimizer_steps":3200,"curve_semantics":"dropout-on pre-update epoch aggregation",
                "last_weighted_objective_loss":float(last["weighted_objective_loss"]),
                "last20_objective_min":float(objective.min()),
                "last20_objective_median":float(np.median(objective)),
                "last20_objective_max":float(objective.max()),
                "last_fit_position_mae":float(last["fit_position_mae"]),
                "last_fit_safe_fd_l1":float(last["fit_safe_fd_l1"]),
                "last_fit_seam_safe_fd_l1":float(last["fit_seam_safe_fd_l1"]),
                "last_ground_truth_position_mae":float(last["ground_truth_position_mae"]),
                "last_ground_truth_safe_fd_l1":float(last["ground_truth_safe_fd_l1"]),
                "last_ground_truth_seam_safe_fd_l1":float(last["ground_truth_seam_safe_fd_l1"]),
                "last_student_to_T_position_distance":float(last["student_to_T_position_distance"]),
                "last_student_to_T_safe_fd_distance":float(last["student_to_T_safe_fd_distance"]),
                "last_student_to_T_seam_distance":float(last["student_to_T_seam_distance"]),
                "all_curve_values_finite":finite_all,
                "gradient_norm_values":"UNKNOWN_NOT_RECORDED",
                "gradient_finite_guard":"present in accepted runner; all 160 epochs completed with no failure",
                "notes":"no intermediate output replay; epoch values retained as their original dropout-on, pre-update scope",
            })
    return rows


def build_horizon_axis(dev_rows:dict[int,list[dict[str,str]]],dev_pred:dict[int,dict[str,np.ndarray]],
                       dev_y:dict[int,np.ndarray],sigma:np.ndarray,fd_scale:np.ndarray) -> list[dict[str,Any]]:
    detail=[]
    raw_by_seed_task=defaultdict(list)
    for seed in SEEDS:
        rows=dev_rows[seed]; y=dev_y[seed]
        for method in METHODS:
            p=dev_pred[seed][method]
            pos=np.abs(p-y)/sigma[None,None,:]
            pos_step=0.5*(pos[...,:14].mean(axis=2)+pos[...,14:].mean(axis=2))
            pos_axis=pos.mean(axis=1)
            fd=np.abs(np.diff(p-y,axis=1)/fd_scale[None,None,:])
            fd_step=0.5*(fd[...,:14].mean(axis=2)+fd[...,14:].mean(axis=2))
            fd_axis=fd.mean(axis=1)
            for task in TASKS:
                idx=task_indices(rows,task);w=row_weights(rows,idx,"source_group")
                COUNTERS["dev_horizon_axis_windows"]+=len(idx)*len(METHODS)
                types=[
                    ("position_step",pos_step[idx],list(range(1,31)),"future_frame","canonical target_sigma scaled abs error"),
                    ("internal_fd_transition",fd_step[idx],list(range(1,30)),"transition_start_frame","29 future-internal transitions: frame t to t+1"),
                    ("position_axis",pos_axis[idx],list(range(28)),"coordinate_index","30-frame mean normalized abs error; raw index only"),
                    ("internal_fd_axis",fd_axis[idx],list(range(28)),"coordinate_index","29-transition mean safe-FD abs error; raw index only"),
                ]
                for typ,mat,indices,index_name,definition in types:
                    means=weighted_mean(mat,w)
                    for j,ix in enumerate(indices):
                        val=float(means[j])
                        row={"split":"dev","scope":"task_seed","seed":seed,"task_id":task,
                             "method":method,"quantity":typ,index_name:ix,"mean_error":val,
                             "sample_sd_across_seeds":"","windows":len(idx),
                             "source_groups":len({rows[int(i)]["source_group"] for i in idx}),
                             "units":"standardized metric units","definition":definition}
                        if typ=="internal_fd_transition":
                            row["from_frame"]=ix;row["to_frame"]=ix+1
                        detail.append(row);raw_by_seed_task[(seed,method,typ,index_name,ix)].append((task,val))
    # Equal-task macro per seed and three-seed mean/sample SD.
    macro=[]
    macro_values=defaultdict(list)
    for seed in SEEDS:
        for method in METHODS:
            subset=[r for r in detail if r["seed"]==seed and r["method"]==method]
            groups=defaultdict(list)
            for r in subset:
                groups[(r["quantity"],r.get("future_frame"),r.get("transition_start_frame"),r.get("coordinate_index"))].append(r)
            for key,items in groups.items():
                out=dict(items[0]);out.update({"scope":"equal_task_macro","task_id":"ALL","windows":None,"source_groups":None})
                out["mean_error"]=float(np.mean([float(x["mean_error"]) for x in items]))
                macro.append(out);macro_values[(method,*key)].append(out["mean_error"])
    combined=detail+macro
    for method in METHODS:
        all_keys=set(k[1:] for k in macro_values if k[0]==method)
        for key in all_keys:
            vals=macro_values[(method,*key)]
            if len(vals)!=3: raise RuntimeError("horizon axis seed macro support mismatch")
            typ,frame,trans,axis=key
            out={"split":"dev","scope":"three_seed_mean_sample_sd","seed":"ALL","task_id":"ALL",
                 "method":method,"quantity":typ,"future_frame":frame,"transition_start_frame":trans,
                 "coordinate_index":axis,"mean_error":float(np.mean(vals)),
                 "sample_sd_across_seeds":float(np.std(vals,ddof=1)),
                 "windows":None,"source_groups":None,"units":"standardized metric units",
                 "definition":"mean/sample SD of the three seed-specific equal-task macros"}
            if typ=="internal_fd_transition" and trans is not None:
                out["from_frame"]=trans;out["to_frame"]=trans+1
            combined.append(out)
    return combined


def build_help_harm(dev_rows:dict[int,list[dict[str,str]]]) -> list[dict[str,Any]]:
    accepted=read_csv(V7/"results/PER_WINDOW_METRICS.csv")
    table={}
    for r in accepted:
        if r["method"] in {"GT_RETRAIN","MIX_SOFT","T"}:
            table[(int(r["seed"]),r["method"],r["anchor_id"])]=r
    outputs=[]
    for seed in SEEDS:
        rows=dev_rows[seed]
        for task in TASKS:
            idx=task_indices(rows,task)
            ids=[rows[int(i)]["anchor_id"] for i in idx]
            group_rows=[rows[int(i)] for i in idx]
            w=row_weights(rows,idx,"source_group")
            for metric,v7col in (("position_mae","position_mae"),("safe_fd_l1","safe_fd_l1"),("seam_safe_fd_l1","seam_safe_fd_l1")):
                g=np.asarray([float(table[(seed,"GT_RETRAIN",aid)][v7col]) for aid in ids])
                t=np.asarray([float(table[(seed,"T",aid)][v7col]) for aid in ids])
                m=np.asarray([float(table[(seed,"MIX_SOFT",aid)][v7col]) for aid in ids])
                if not np.isfinite(g).all() or not np.isfinite(t).all() or not np.isfinite(m).all():
                    raise RuntimeError("accepted per-window help/harm table has nonfinite values")
                teacher_gap=g-t; m_gain=g-m
                categories=np.where(teacher_gap>0,"teacher_help",np.where(teacher_gap<0,"teacher_harm","exact_tie"))
                for category in ("teacher_help","exact_tie","teacher_harm"):
                    mask=categories==category
                    support=float(w[mask].sum())
                    outputs.append({
                        "split":"dev","scope":"task_seed","seed":seed,"task_id":task,
                        "metric":metric,"teacher_stratum":category,
                        "window_count":int(mask.sum()),"source_group_count":len({group_rows[i]["source_group"] for i in np.flatnonzero(mask)}),
                        "weighted_fraction":support,
                        "teacher_gap_weighted_contribution_G_minus_T":float(np.dot(w[mask],teacher_gap[mask])),
                        "teacher_gap_weighted_mean_G_minus_T":float(np.dot(w[mask],teacher_gap[mask])/support) if support>0 else "",
                        "M_vs_G_gain_weighted_contribution":float(np.dot(w[mask],m_gain[mask])),
                        "M_vs_G_gain_weighted_mean":float(np.dot(w[mask],m_gain[mask])/support) if support>0 else "",
                        "definition":"G loss - T loss >0 help; <0 harm; exact zero tie; weights equal source groups then task; counts are descriptive windows",
                    })
    return outputs


def geom_vectors(rows_by_seed:dict[int,list[dict[str,str]]],pred_by_seed:dict[int,dict[str,np.ndarray]],
                 seeds:list[int],tasks:list[int],sigma:np.ndarray,fd_scale:np.ndarray,
                 split:str) -> tuple[dict[str,np.ndarray],np.ndarray]:
    vectors={k:[] for k in ("C","A","B","E")}
    weights=[]
    for seed in seeds:
        rows=rows_by_seed[seed];pred=pred_by_seed[seed]
        for task in tasks:
            idx=task_indices(rows,task)
            w=row_weights(rows,idx,GROUP_KEY[split])/(len(seeds)*len(tasks))
            COUNTERS["geometry_receiver_contributions"]+=len(idx)
            g=pred["G"][idx];u=pred["U"][idx];m=pred["M"][idx];t=pred["T"][idx]
            vectors["C"].append(t-g);vectors["A"].append(m-g)
            vectors["B"].append(u-g);vectors["E"].append(m-u)
            weights.extend(w.tolist())
    arrays={k:np.concatenate(v,axis=0).astype(np.float64,copy=False) for k,v in vectors.items()}
    w=np.asarray(weights,dtype=np.float64)
    if not np.isclose(w.sum(),1.0,rtol=0.0,atol=1e-12):
        raise RuntimeError(f"geometry weights sum to {w.sum()}")
    spaces={
        "position":{k:v/sigma[None,None,:] for k,v in arrays.items()},
        "internal_fd":{k:np.diff(v,axis=1)/fd_scale[None,None,:] for k,v in arrays.items()},
        "seam":{k:v[:,0]/fd_scale[None,:] for k,v in arrays.items()},
    }
    return spaces,w


def inner_weighted(a:np.ndarray,b:np.ndarray,w:np.ndarray) -> float:
    if a.shape!=b.shape or a.shape[0]!=len(w): raise RuntimeError("geometry inner-product shape mismatch")
    row=np.mean(a*b,axis=tuple(range(1,a.ndim)))
    return weighted_mean(row,w)


def geometry_rows(split:str,scope:str,seed_field:Any,task_field:Any,
                  rows_by_seed:dict[int,list[dict[str,str]]],
                  pred_by_seed:dict[int,dict[str,np.ndarray]],
                  seeds:list[int],tasks:list[int],sigma:np.ndarray,fd_scale:np.ndarray) -> list[dict[str,Any]]:
    spaces,w=geom_vectors(rows_by_seed,pred_by_seed,seeds,tasks,sigma,fd_scale,split)
    out=[]
    for space,vecs in spaces.items():
        for name,arr in vecs.items():
            rms=math.sqrt(max(inner_weighted(arr,arr,w),0.0))
            absmean=weighted_mean(np.mean(np.abs(arr),axis=tuple(range(1,arr.ndim))),w)
            out.append({"split":split,"scope":scope,"seed":seed_field,"task_id":task_field,
                        "record_type":"magnitude","space":space,"vector":name,
                        "rms":rms,"mean_abs_magnitude":absmean,
                        "inner_X_C":"","C_energy":"","X_energy":"","projection_k":"",
                        "weighted_cosine":"","orthogonal_energy":"","orthogonal_rms":"",
                        "orthogonality_residual":"","energy_identity_abs_error":"",
                        "energy_identity_rel_error":"","na_reason":""})
        if space=="seam": continue
        c=vecs["C"]; c_energy=inner_weighted(c,c,w); c_rms=math.sqrt(max(c_energy,0.0))
        for xname in ("A","B","E"):
            x=vecs[xname];x_energy=inner_weighted(x,x,w);x_rms=math.sqrt(max(x_energy,0.0))
            dot=inner_weighted(x,c,w)
            if c_rms<=EPS:
                row={"split":split,"scope":scope,"seed":seed_field,"task_id":task_field,
                     "record_type":"projection","space":space,"vector":xname,
                     "rms":x_rms,"mean_abs_magnitude":weighted_mean(np.mean(np.abs(x),axis=tuple(range(1,x.ndim))),w),
                     "inner_X_C":dot,"C_energy":c_energy,"X_energy":x_energy,
                     "projection_k":"","weighted_cosine":"","orthogonal_energy":"",
                     "orthogonal_rms":"","orthogonality_residual":"",
                     "energy_identity_abs_error":"","energy_identity_rel_error":"",
                     "na_reason":"RMS(C)<=epsilon_num; k/cos/orthogonal residual are NA"}
            else:
                k=dot/c_energy
                residual=x-k*c
                orth=inner_weighted(residual,residual,w)
                orth_dot=inner_weighted(residual,c,w)
                decomposition=k*k*c_energy+orth
                abs_err=abs(x_energy-decomposition)
                rel_err=abs_err/max(abs(x_energy),EPS)
                if abs(orth_dot)>1e-12 or abs_err>1e-12+1e-10*abs(x_energy):
                    raise RuntimeError(f"projection algebra QA failed {split}/{scope}/{space}/{xname}: dot={orth_dot}, abs={abs_err}, rel={rel_err}")
                cosine="" if x_rms<=EPS else dot/(x_rms*c_rms)
                row={"split":split,"scope":scope,"seed":seed_field,"task_id":task_field,
                     "record_type":"projection","space":space,"vector":xname,
                     "rms":x_rms,"mean_abs_magnitude":weighted_mean(np.mean(np.abs(x),axis=tuple(range(1,x.ndim))),w),
                     "inner_X_C":dot,"C_energy":c_energy,"X_energy":x_energy,
                     "projection_k":k,"weighted_cosine":cosine,
                     "orthogonal_energy":orth,"orthogonal_rms":math.sqrt(max(orth,0.0)),
                     "orthogonality_residual":orth_dot,
                     "energy_identity_abs_error":abs_err,"energy_identity_rel_error":rel_err,
                     "na_reason":"cos NA because RMS(X)<=epsilon_num" if x_rms<=EPS else ""}
            out.append(row)
    return out


def build_geometry(rows_by_split:dict[str,dict[int,list[dict[str,str]]]],
                   preds_by_split:dict[str,dict[int,dict[str,np.ndarray]]],
                   sigma:np.ndarray,fd_scale:np.ndarray) -> list[dict[str,Any]]:
    out=[]
    for split in ("train","dev"):
        for seed in SEEDS:
            for task in TASKS:
                out.extend(geometry_rows(split,"task_seed",seed,task,rows_by_split[split],
                                          preds_by_split[split],[seed],[task],sigma,fd_scale))
            out.extend(geometry_rows(split,"equal_task_macro",seed,"ALL",rows_by_split[split],
                                     preds_by_split[split],[seed],list(TASKS),sigma,fd_scale))
        out.extend(geometry_rows(split,"three_seed_macro","ALL","ALL",rows_by_split[split],
                                 preds_by_split[split],list(SEEDS),list(TASKS),sigma,fd_scale))
    return out


def macro_loss(split:str,seed_ids:list[int],task_ids:list[int],method:str,target_name:str,
               rows_by_seed:dict[int,list[dict[str,str]]],pred_by_seed:dict[int,dict[str,np.ndarray]],
               targets_by_seed:dict[int,np.ndarray],sigma:np.ndarray,fd_scale:np.ndarray,
               anchors_by_seed:dict[int,np.ndarray]|None=None) -> dict[str,float]:
    seed_values=[]
    for seed in seed_ids:
        task_vals=[]
        for task in task_ids:
            anchor=None if anchors_by_seed is None else anchors_by_seed[seed]
            task_vals.append(metric_and_rms(pred_by_seed[seed][method],targets_by_seed[seed],
                                            rows_by_seed[seed],task,split,sigma,fd_scale,anchor,
                                            count_key="loo"))
        seed_values.append({k:float(np.mean([x[k] for x in task_vals])) for k in task_vals[0]})
    return {k:float(np.mean([x[k] for x in seed_values])) for k in seed_values[0]}


def build_loo(rows_by_split:dict[str,dict[int,list[dict[str,str]]]],
              preds_by_split:dict[str,dict[int,dict[str,np.ndarray]]],
              targets:dict[str,dict[int,np.ndarray]],sigma:np.ndarray,fd_scale:np.ndarray,
              anchors_by_seed:dict[int,np.ndarray]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    seed_rows=[];task_rows=[]
    for split in ("train","dev"):
        ydict=targets[split]
        for excluded in SEEDS:
            keep=[s for s in SEEDS if s!=excluded]
            losses={}
            for method in METHODS:
                losses[method]=macro_loss(split,keep,list(TASKS),method,"Y",rows_by_split[split],
                                           preds_by_split[split],{s:ydict[s] for s in keep},sigma,fd_scale,
                                           {s:anchors_by_seed[s] for s in keep} if split=="dev" else None)
            geom=geometry_rows(split,"seed_block_loo",f"drop_{excluded}","ALL",rows_by_split[split],
                               preds_by_split[split],keep,list(TASKS),sigma,fd_scale)
            seed_rows.append(loo_pack(split,"seed_block",excluded,losses,geom))
        for excluded in TASKS:
            keep=[t for t in TASKS if t!=excluded]
            losses={}
            for method in METHODS:
                losses[method]=macro_loss(split,list(SEEDS),keep,method,"Y",rows_by_split[split],
                                           preds_by_split[split],ydict,sigma,fd_scale,
                                           anchors_by_seed if split=="dev" else None)
            geom=geometry_rows(split,"task_loo","ALL",f"drop_{excluded}",rows_by_split[split],
                               preds_by_split[split],list(SEEDS),keep,sigma,fd_scale)
            task_rows.append(loo_pack(split,"task",excluded,losses,geom))
    return seed_rows,task_rows


def loo_pack(split:str,kind:str,excluded:Any,losses:dict[str,dict[str,float]],
             geom_rows_value:list[dict[str,Any]]) -> dict[str,Any]:
    out={"split":split,"loo_type":kind,"excluded_value":excluded,
         "remaining_seed_count":2 if kind=="seed_block" else 3,
         "remaining_task_count":5 if kind=="seed_block" else 4}
    for method in METHODS:
        for metric,base in (("position_mae","position_mae"),("safe_fd_l1","safe_fd_l1"),("seam_safe_fd_l1","seam_safe_fd_l1")):
            out[f"{metric}_loss_{method}"]=losses[method][base]
    for metric,base in (("position_mae","position_mae"),("safe_fd_l1","safe_fd_l1"),("seam_safe_fd_l1","seam_safe_fd_l1")):
        for cand,ref in (("M","G"),("M","U"),("M","T")):
            out[f"{metric}_gain_{cand}_vs_{ref}"]=losses[ref][base]-losses[cand][base]
    for x in ("A","B","E"):
        for space in ("position","internal_fd"):
            g=next((r for r in geom_rows_value if r["record_type"]=="projection" and r["vector"]==x and r["space"]==space),None)
            c_mag=next((r for r in geom_rows_value if r["record_type"]=="magnitude" and r["vector"]=="C" and r["space"]==space),None)
            x_mag=next((r for r in geom_rows_value if r["record_type"]=="magnitude" and r["vector"]==x and r["space"]==space),None)
            out[f"{x}_vs_C_{space}_rms_C"]=None if c_mag is None else c_mag["rms"]
            out[f"{x}_vs_C_{space}_rms_X"]=None if x_mag is None else x_mag["rms"]
            out[f"{x}_vs_C_{space}_mean_abs_C"]=None if c_mag is None else c_mag["mean_abs_magnitude"]
            out[f"{x}_vs_C_{space}_mean_abs_X"]=None if x_mag is None else x_mag["mean_abs_magnitude"]
            out[f"{x}_vs_C_{space}_cosine"]=None if g is None else g["weighted_cosine"]
            out[f"{x}_vs_C_{space}_k"]=None if g is None else g["projection_k"]
            out[f"{x}_vs_C_{space}_orthogonal_energy"]=None if g is None else g["orthogonal_energy"]
            out[f"{x}_vs_C_{space}_NA_reason"]="" if g is None else g["na_reason"]
    return out


def build_case_data(train_rows:dict[int,list[dict[str,str]]],train_pred:dict[int,dict[str,np.ndarray]],
                    dev_rows:dict[int,list[dict[str,str]]],dev_pred:dict[int,dict[str,np.ndarray]],
                    dev_y:dict[int,np.ndarray],dev_anchor:dict[int,np.ndarray],
                    sigma:np.ndarray) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    seed=202610131
    tr=train_rows[seed];tp=train_pred[seed]
    axis_scores=np.zeros(28,dtype=np.float64)
    per_task=[]
    delta=np.abs(tp["T"]-tp["V"])/sigma[None,None,:]
    for task in TASKS:
        idx=task_indices(tr,task);w=row_weights(tr,idx,"episode_id")
        per_axis=delta[idx].mean(axis=1)
        per_task.append(weighted_mean(per_axis,w))
    axis_scores=np.mean(np.stack(per_task),axis=0)
    axes=[]
    for lo,hi in ((0,14),(14,28)):
        vals=axis_scores[lo:hi]
        mx=vals.max()
        axis=lo+int(np.flatnonzero(vals==mx)[0])
        axes.append(axis)
    rows=dev_rows[seed]
    by_task={t:[] for t in TASKS}
    for i,r in enumerate(rows): by_task[int(r["task_id"])].append((i,r))
    candidates=[]
    for task in TASKS:
        groups=sorted({r["source_group"] for _,r in by_task[task]},
                      key=lambda g:(hashlib.sha256(f"V8_20261016|{g}".encode()).hexdigest(),g))
        for rank,group in enumerate(groups[:2],start=1):
            first=next((i,r) for i,r in by_task[task] if r["source_group"]==group)
            candidates.append({"task_id":task,"anchor_id":first[1]["anchor_id"],
                               "example_type":"regular_hash_group","source_group":group,
                               "group_hash":hashlib.sha256(f"V8_20261016|{group}".encode()).hexdigest(),
                               "group_rank":rank,"selected_seed":seed,"posthoc":False})
    # Accepted V7 per-window losses drive the posthoc failure examples.
    accepted=read_csv(V7/"results/PER_WINDOW_METRICS.csv")
    idx_loss={(int(r["seed"]),r["method"],r["anchor_id"]):float(r["position_mae"]) for r in accepted}
    for task in TASKS:
        supports=[(r["anchor_id"],idx_loss[(seed,"GT_RETRAIN",r["anchor_id"])]-
                   idx_loss[(seed,"MIX_SOFT",r["anchor_id"])]) for _,r in by_task[task]]
        min_gain=min(x[1] for x in supports)
        anchor=min(a for a,g in supports if g==min_gain)
        row=next(r for _,r in by_task[task] if r["anchor_id"]==anchor)
        candidates.append({"task_id":task,"anchor_id":anchor,"example_type":"posthoc_min_MIX_gain",
                           "source_group":row["source_group"],"group_hash":"",
                           "group_rank":"","selected_seed":seed,"posthoc":True,
                           "MIX_vs_G_gain_position":min_gain,
                           "all_anchors_improved":all(g>0 for _,g in supports)})
    # De-duplicate overlapping routine/failure anchors while keeping every selection reason.
    dedup={}
    for c in candidates:
        key=c["anchor_id"]
        if key not in dedup: dedup[key]={**c,"selection_reasons":[c["example_type"]]}
        else:
            dedup[key]["selection_reasons"].append(c["example_type"])
            if c["posthoc"]: dedup[key]["posthoc"]=True
            dedup[key]["MIX_vs_G_gain_position"]=c.get("MIX_vs_G_gain_position",dedup[key].get("MIX_vs_G_gain_position",""))
    case_rows=[]
    for plot_index,(aid,c) in enumerate(sorted(dedup.items(),key=lambda kv:(kv[1]["task_id"],kv[0])),start=1):
        i=next(i for i,r in enumerate(rows) if r["anchor_id"]==aid)
        case_rows.append({"plot_id":f"case_{plot_index:02d}","seed":seed,"task_id":c["task_id"],
                          "anchor_id":aid,"source_group":c["source_group"],
                          "selection_reasons":";".join(c["selection_reasons"]),
                          "posthoc_failure_example":c["posthoc"],
                          "MIX_vs_G_gain_position":c.get("MIX_vs_G_gain_position",""),
                          "axis_0_14":axes[0],"axis_14_28":axes[1],
                          "coordinate_selection_basis":"seed train equal-task mean abs(T-V)/canonical target_sigma",
                          "anchor_row_order":i})
    selection_ledger=[]
    for c in candidates:
        selected=next(r for r in case_rows if r["anchor_id"]==c["anchor_id"])
        selection_ledger.append({**c,"plot_id":selected["plot_id"],
                                 "plot_once":c==next(x for x in candidates if x["anchor_id"]==c["anchor_id"]),
                                 "dedup_anchor":selected["anchor_id"]})
    figure_rows=[]
    for case in case_rows:
        i=int(case["anchor_row_order"])
        for axis_group,axis in (("coordinate_0_13",int(case["axis_0_14"])),
                                ("coordinate_14_27",int(case["axis_14_28"]))):
            for method in ("Y","G","U","M","T"):
                if method=="Y": trajectory=dev_y[seed][i,:,axis]
                else: trajectory=dev_pred[seed][method][i,:,axis]
                values=[float(dev_anchor[seed][i,axis])]+[float(x) for x in trajectory]
                for frame,value in enumerate(values):
                    figure_rows.append({"plot_id":case["plot_id"],"seed":seed,"task_id":case["task_id"],
                                        "anchor_id":case["anchor_id"],"selection_reasons":case["selection_reasons"],
                                        "posthoc_failure_example":case["posthoc_failure_example"],
                                        "axis_group":axis_group,"coordinate_index":axis,
                                        "method":method,"frame_index":frame,"frame_label":"anchor" if frame==0 else f"future_{frame}",
                                        "native_value":value,"value_units":"native recorded value; no physical unit asserted"})
    return case_rows,selection_ledger,figure_rows


def build_resource_exposure(train_pred:dict[int,dict[str,np.ndarray]],
                            dev_pred:dict[int,dict[str,np.ndarray]],
                            target_counts:dict[str,int],dev_access:dict[str,Any],
                            curve_rows:list[dict[str,Any]],train_fit_rows:list[dict[str,Any]],
                            horizon_rows:list[dict[str,Any]],counters:dict[str,int]) -> list[dict[str,Any]]:
    summary=read_json(V7/"results/RESULTS_SUMMARY.json")
    exec_receipt=read_json(V8/"coordination/FROZEN_FORWARD_EXECUTION.json")
    actual_forwards=exec_receipt["jobs"]
    rows=[]
    def add(phase,metric,count,unit,source,scope,notes):
        rows.append({"phase":phase,"metric":metric,"count":count,"unit":unit,
                     "source":source,"scope":scope,"notes":notes})
    add("V7_historical","accepted_new_head_fits",summary["new_head_fits_accepted"],"fits",str(V7/"qa/FINAL_ACCEPTANCE.json"),"historical only","not added to V8 cost")
    add("V7_historical","optimizer_updates",summary["formal_optimizer_updates"],"updates",str(V7/"results/RESULTS_SUMMARY.json"),"historical only","28,800 V7 updates; not V8 activity")
    add("V7_historical","student_dev_prediction_jobs",summary["student_dev_prediction_jobs_accepted"],"jobs",str(V7/"qa/FINAL_ACCEPTANCE.json"),"historical only","V8 reuses; no new dev predictions")
    add("V7_historical","student_dev_window_loss_evaluations",summary["student_dev_window_loss_evaluations"],"window-method evaluations",str(V7/"qa/SCORER_SEAL.json"),"historical only","1,665; not combined with rows or jobs")
    add("V7_historical","dev_target_archive_loads",summary["dev_target_archive_loads"],"archive loads",str(V7/"qa/DEV_TARGET_ACCESS_RECEIPT.json"),"historical only","one earlier V7 scorer load")
    add("V8_actual","new_fits",0,"fits",str(V8),"this run","frozen inference/CPU diagnosis only")
    add("V8_actual","optimizer_updates",0,"updates",str(V8),"this run","no optimizer instantiated")
    add("V8_actual","new_teacher_forward_jobs",0,"jobs",str(V8),"this run","V7 S/V/T assets reused")
    add("V8_actual","new_frozen_student_train_forward_jobs",len(actual_forwards),"jobs",str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json"),"this run","six U/M epoch160 frozen forwards")
    add("V8_actual","frozen_train_forward_windows",len(actual_forwards)*1280,"job-window rows",str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json"),"this run","six jobs times 1,280 accepted train prefixes")
    add("V8_actual","new_dev_prediction_jobs",0,"jobs",str(V8/"protocol/PREDICTION_REUSE_AND_REPLAY.json"),"this run","nine V7 outputs and accepted S/V/T references reused")
    add("V8_actual","new_bootstrap_draws",0,"draws",str(V8),"this run","V7 CI quoted only")
    add("V8_actual","encoder_extraction",0,"jobs",str(V8),"this run","precomputed Q0 cache read as prefix input only")
    attempts=read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")
    failed=[a for a in attempts.get("attempts",[]) if a.get("status") in {"FAILED_BEFORE_DEV_TARGET_LOAD","FAILED_AFTER_ONE_DEV_TARGET_LOAD"}]
    prior_y=sum(int(a.get("train_Y_npy_loads",0)) for a in failed)
    prior_z=sum(int(a.get("train_Z_npy_loads",0)) for a in failed)
    prior_pred=sum(int(a.get("prediction_native_arrays_loaded",0)) for a in failed)
    probe_pred=int(attempts.get("prediction_native_schema_probe",{}).get("arrays_loaded",0))
    prior_pert=sum(int(a.get("train_target_perturbation_window_metric_evaluations",0)) for a in failed)
    prior_model_metric=sum(int(a.get("train_model_target_metric_window_evaluations",0)) for a in failed)
    prior_j=sum(int(a.get("train_objective_J_window_evaluations",0)) for a in failed)
    prior_train_imitation=sum(int(a.get("train_imitation_window_metric_evaluations",0)) for a in failed)
    prior_dev_imitation=sum(int(a.get("dev_imitation_window_metric_evaluations",0)) for a in failed)
    prior_dev_loads=sum(int(a.get("dev_target_archive_loads",0)) for a in failed)
    prior_dev_windows=sum(int(a.get("dev_target_windows_loaded",0)) for a in failed)
    prior_dev_ids=max([int(a.get("dev_target_windows_loaded",0)) for a in failed]+[0])
    prior_prefix_loads=sum(int(a.get("dev_prefix_archive_loads",0)) for a in failed)
    prior_prefix_rows=sum(int(a.get("dev_prefix_rows_loaded",0)) for a in failed)
    add("V8_successful_extract","train_Y_npy_loads",target_counts["train_Y_npy_loads"],"array loads",str(V7/"training_targets/Y_NATIVE.npy"),"successful extract","one accepted native Y array")
    add("V8_successful_extract","train_Z_npy_loads",target_counts["train_Z_npy_loads"],"array loads",str(V7/"protocol/TARGET_LOCK.json"),"successful extract","nine locked Z arrays")
    add("V8_failed_attempts","train_Y_npy_loads",prior_y,"array loads",str(V7/"training_targets/Y_NATIVE.npy"),"failed pre-dev extract","arrays loaded before report-construction KeyError; counted, no dev target opened")
    add("V8_failed_attempts","train_Z_npy_loads",prior_z,"array loads",str(V7/"protocol/TARGET_LOCK.json"),"failed pre-dev extract","arrays loaded before report-construction KeyError; counted")
    add("V8_total","train_Y_npy_loads",target_counts["train_Y_npy_loads"]+prior_y,"array loads",str(V7/"training_targets/Y_NATIVE.npy"),"all V8 attempts","successful plus failed pre-dev attempt")
    add("V8_total","train_Z_npy_loads",target_counts["train_Z_npy_loads"]+prior_z,"array loads",str(V7/"protocol/TARGET_LOCK.json"),"all V8 attempts","successful plus failed pre-dev attempt")
    add("V8_actual","train_target_npz_loads",target_counts["train_target_archive_loads"],"archive loads",str(V7/"training_targets"),"this run","the train outcome NPZ target field was not opened")
    add("V8_successful_extract","dev_target_archive_loads",dev_access["dev_target_archive_loads"],"archive loads",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"successful salvage","zero target reloads after the one failed analysis attempt")
    add("V8_failed_attempts","dev_target_archive_loads",prior_dev_loads,"archive loads",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed dev decomposition attempt","one original archive load; no results persisted; archive not reopened")
    add("V8_total","dev_target_archive_loads",dev_access["dev_target_archive_loads"]+prior_dev_loads,"archive loads",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"all V8 attempts","actual count, including failed attempt")
    add("V8_failed_attempts","dev_target_windows_loaded",prior_dev_windows,"windows",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed dev decomposition attempt","185 original dev windows read once; no per-step table saved")
    add("V8_total","dev_target_windows_loaded",dev_access["dev_target_rows_loaded"]+prior_dev_windows,"windows",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"all V8 attempts","counted actual loads, not independent sample size")
    add("V8_total","dev_target_unique_ids",max(dev_access.get("dev_target_unique_ids",0),prior_dev_ids),"IDs",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"all V8 attempts","185 unique IDs; no final/heldout payload")
    add("V8_successful_extract","dev_prefix_archive_loads",dev_access.get("dev_prefix_archive_loads",0),"archive loads",str(V4/"data/DEV_PREFIXES_NO_LABELS.npz"),"successful salvage","no additional prefix reload")
    add("V8_failed_attempts","dev_prefix_archive_loads",prior_prefix_loads,"archive loads",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed dev decomposition attempt","one prefix-only anchor read")
    add("V8_total","dev_prefix_archive_loads",dev_access.get("dev_prefix_archive_loads",0)+prior_prefix_loads,"archive loads",str(V4/"data/DEV_PREFIXES_NO_LABELS.npz"),"all V8 attempts","total actual prefix archive opens")
    add("V8_total","dev_prefix_rows_loaded",dev_access.get("dev_prefix_rows_loaded",0)+prior_prefix_rows,"rows",str(V4/"data/DEV_PREFIXES_NO_LABELS.npz"),"all V8 attempts","anchor_native only")
    add("V8_actual","V7_training_curve_rows_reused",len(read_csv(V7/"results/TRAINING_CURVES.csv")),"rows",str(V7/"results/TRAINING_CURVES.csv"),"this run","original curve scope preserved")
    add("V8_successful_extract","train_model_target_metric_window_evaluations",counters["train_model_target"],"window-target evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"successful extract","G/U/M against Y,T,own Z and common ZM/ZU; duplicate method-target pairs cached")
    add("V8_successful_extract","train_target_perturbation_window_evaluations",counters["train_target_perturbation"],"window-pair evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"successful extract","five locked target-difference pairs")
    add("V8_successful_extract","train_objective_J_window_evaluations",counters["train_objective_J"],"window-target evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"successful extract","exact fixed-scale runner objective on final eval outputs")
    add("V8_failed_attempts","train_model_target_metric_window_evaluations",prior_model_metric,"window-target evaluations",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed pre-dev extract","only first seed/task G-on-Y calculation completed")
    add("V8_failed_attempts","train_target_perturbation_window_evaluations",prior_pert,"window-pair evaluations",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed pre-dev extract","five pairs on first seed/task support")
    add("V8_failed_attempts","train_objective_J_window_evaluations",prior_j,"window-target evaluations",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed pre-dev extract","only first seed/task G-on-Y objective completed")
    add("V8_total","train_model_target_metric_window_evaluations",counters["train_model_target"]+prior_model_metric,"window-target evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"all V8 attempts","successful plus failed in-memory work")
    add("V8_total","train_target_perturbation_window_evaluations",counters["train_target_perturbation"]+prior_pert,"window-pair evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"all V8 attempts","successful plus failed in-memory work")
    add("V8_total","train_objective_J_window_evaluations",counters["train_objective_J"]+prior_j,"window-target evaluations",str(V8/"results/TRAIN_TARGET_AND_FIT.csv"),"all V8 attempts","successful plus failed in-memory work")
    add("V8_failed_attempts","prediction_native_arrays_loaded",prior_pred,"tensor arrays",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed pre-dev extract","read-only accepted prediction assets")
    add("V8_preflight","prediction_native_arrays_loaded",probe_pred,"tensor arrays","V7 GT_RETRAIN dev prediction","one-time schema API probe","prediction_native only; no target")
    add("V8_total","prediction_native_arrays_loaded",33+prior_pred+probe_pred,"tensor arrays","V7/V8 accepted prediction safetensors","all V8 attempts","15 train assets and 18 dev assets per extraction attempt plus one schema probe")
    add("V8_successful_extract","train_dev_imitation_window_evaluations",counters["train_imitation"]+counters["dev_imitation"],"window-method comparisons",str(V8/"results/TRAIN_DEV_IMITATION.csv"),"successful salvage","G/U/M distance to T, train and dev; prediction-only")
    add("V8_failed_attempts","train_imitation_window_evaluations",prior_train_imitation,"window-method comparisons",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed post-target attempt","computed in memory before horizon aggregation failed")
    add("V8_failed_attempts","dev_imitation_window_evaluations",prior_dev_imitation,"window-method comparisons",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed post-target attempt","prediction-to-teacher distances; no GT used")
    add("V8_total","train_dev_imitation_window_evaluations",counters["train_imitation"]+counters["dev_imitation"]+prior_train_imitation+prior_dev_imitation,"window-method comparisons",str(V8/"results/TRAIN_DEV_IMITATION.csv"),"all V8 attempts","successful salvage plus failed in-memory attempt")
    add("V8_actual","LOO_metric_window_evaluations",counters["loo"],"window-method-target evaluations",str(V8/"results/SEED_BLOCK_LOO.csv"),"this run","whole-seed and whole-task LOO; no bootstrap")
    add("V8_actual","dev_step_axis_diagnostic_window_methods",counters.get("dev_horizon_axis_windows",0),"window-method bundles",str(V8/"results/HORIZON_AXIS_ERRORS.csv"),"this run","six methods x three seeds x 185 windows; reduction includes steps and coordinates")
    add("V8_actual","geometry_receiver_contributions",counters["geometry_receiver_contributions"],"weighted receiver contributions",str(V8/"results/CORRECTION_GEOMETRY.csv"),"this run","pooled position/FD/seam geometry scopes; not independent N")
    add("V8_actual","prediction_native_arrays_loaded",33,"tensor arrays","V7/V8 accepted prediction safetensors","this run","15 train assets and 18 dev assets; action_residual key not loaded")
    add("V8_actual","V7_existing_dev_loss_rows_reused",summary["student_dev_window_loss_evaluations"],"window-method evaluations",str(V7/"results/PER_WINDOW_METRICS.csv"),"this run","no bootstrap or V7 rescoring")
    alloc=[float(j["accepted_outputs"]["peak_memory_allocated_mib"]) for j in actual_forwards]
    reserv=[float(j["accepted_outputs"]["peak_memory_reserved_mib"]) for j in actual_forwards]
    add("V8_actual","GPU_jobs_on_physical_GPU1",sum(str(j["accepted_outputs"]["physical_gpu"])=="1" for j in actual_forwards),"jobs",str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json"),"this run","logical device cuda:0; six sequential watcher leases")
    add("V8_actual","max_peak_allocated_mib",max(alloc),"MiB",str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json"),"this run","measured from V8 forward receipts")
    add("V8_actual","max_peak_reserved_mib",max(reserv),"MiB",str(V8/"coordination/FROZEN_FORWARD_EXECUTION.json"),"this run","measured from V8 forward receipts")
    horizon_values=sum(1 for r in horizon_rows if r.get("mean_error") not in ("",None))
    horizon_status_rows=len(horizon_rows)-horizon_values
    failed_horizon_arrays=sum(int(a.get("dev_error_array_method_windows",0)) for a in failed)
    add("V8_actual","horizon_axis_numeric_values_written",horizon_values,"values",str(V8/"results/HORIZON_AXIS_ERRORS.csv"),"this run","zero numeric values; explicit UNAVAILABLE rows explain the closed-target boundary")
    add("V8_actual","horizon_axis_status_rows_written",horizon_status_rows,"status rows",str(V8/"results/HORIZON_AXIS_ERRORS.csv"),"this run","four quantity rows marked UNAVAILABLE")
    add("V8_failed_attempts","dev_error_array_method_windows",failed_horizon_arrays,"window-method arrays",str(V8/"coordination/EXTRACTION_ATTEMPTS.json"),"failed dev decomposition attempt","one G seed-131 error tensor was computed before aggregation failed; no values were persisted")
    return rows


def synthetic_pipeline_qa() -> dict[str,Any]:
    rng=np.random.default_rng(4808)
    rows=[]
    for task in (119,94):
        for ep in range(2):
            for a in range(2):
                rows.append({"anchor_id":f"{task}-{ep}-{a}","episode_id":str(ep),
                             "task_id":str(task),"source_group":f"{task}-{ep}","anchor_frame":str(a)})
    data=rng.normal(size=(len(rows),30,28)).astype(np.float64)
    target=rng.normal(size=data.shape).astype(np.float64)
    sigma=np.ones(28,dtype=np.float64);fd=np.ones(28,dtype=np.float64)
    m=metric_and_rms(data,target,rows,119,"train",sigma,fd)
    p1,f1,j1=runner_j_windows(data[:2],target[:2],sigma,fd)
    vector_mean=weighted_mean(np.asarray([[1.0,2.0],[3.0,4.0]]),np.asarray([0.25,0.75]))
    vector_mean_ok=bool(np.allclose(vector_mean,np.asarray([2.5,3.5]),rtol=0,atol=1e-14))
    # Main schema fixture checks the required output axes and J decomposition.
    fixture_rows=[]
    for task in TASKS:
        for episode in range(2):
            for anchor_i in range(2):
                fixture_rows.append({"anchor_id":f"{task}-{episode}-{anchor_i}",
                                     "episode_id":str(episode),"task_id":str(task),
                                     "source_group":f"{task}-{episode}","anchor_frame":str(anchor_i)})
    n=len(fixture_rows)
    fy=rng.normal(size=(n,30,28)).astype(np.float64)
    pred_fixture={};z_fixture={}
    for seed in SEEDS:
        fs=rng.normal(size=(n,30,28)).astype(np.float64)
        fv=rng.normal(size=(n,30,28)).astype(np.float64)
        ft=0.5*fs+0.5*fv
        pred_fixture[seed]={"G":fv,"U":rng.normal(size=(n,30,28)),
                            "M":rng.normal(size=(n,30,28)),"S":fs,"V":fv,"T":ft}
        z_fixture[seed]={"GT_RETRAIN":fy,"SELF_SOFT":0.5*fy+0.5*fv,
                         "MIX_SOFT":0.5*fy+0.5*ft}
    fit_fixture=build_train_target_and_fit(
        {seed:fixture_rows for seed in SEEDS},pred_fixture,fy,z_fixture,
        np.ones(28),np.ones(28),np.ones(28),np.ones(28))
    has_T_fit=all(any(r["method"]==method and r["target"]=="T" for r in fit_fixture)
                  for method in ("G","U","M"))
    has_gains=all(any(r["record_type"]==name for r in fit_fixture)
                  for name in ("TRAIN_Z_GAIN_M","TRAIN_Z_GAIN_U"))
    pass_checks=(
        np.isfinite(list(m.values())).all()
        and np.allclose(j1,p1+0.1*f1,rtol=0,atol=1e-14)
        and data.shape==(len(rows),30,28)
        and len(task_indices(rows,119))==4
        and abs(row_weights(rows,task_indices(rows,119),"episode_id").sum()-1.0)<1e-12
        and has_T_fit and has_gains and vector_mean_ok
    )
    return {"status":"PASS" if pass_checks else "FAIL","checks":{
        "canonical_metric_and_rms_finite":bool(np.isfinite(list(m.values())).all()),
        "runner_J_equals_position_plus_0p1_FD":bool(np.allclose(j1,p1+0.1*f1,rtol=0,atol=1e-14)),
        "typed_30x28_and_group_equal_weights":bool(data.shape==(len(rows),30,28) and abs(row_weights(rows,task_indices(rows,119),"episode_id").sum()-1)<1e-12),
        "train_fit_table_accepts_T_and_emits_both_Z_gains":bool(has_T_fit and has_gains),
        "vector_weighted_mean_support":vector_mean_ok,
        "synthetic_windows":len(rows),"targets_loaded":False,"gpu_access":False},
        "created_utc":now(),"final_heldout_payload_access":0}


def run_extract() -> None:
    attempts=read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")
    failed=[a for a in attempts.get("attempts",[]) if a.get("status") in {"FAILED_BEFORE_DEV_TARGET_LOAD","FAILED_AFTER_ONE_DEV_TARGET_LOAD"}]
    prior_y=sum(int(a.get("train_Y_npy_loads",0)) for a in failed)
    prior_z=sum(int(a.get("train_Z_npy_loads",0)) for a in failed)
    prior_pred=sum(int(a.get("prediction_native_arrays_loaded",0)) for a in failed)
    probe_pred=int(attempts.get("prediction_native_schema_probe",{}).get("arrays_loaded",0))
    preflight_data,train_data,dev_data=canonical_rows_and_arrays()
    print(json.dumps({"stage":"prediction_ids_profiles_prechecked","prediction_native_arrays_loaded":33,
                      "train_Y_loaded":False,"dev_Y_loaded":False,"new_dev_predictions":0},sort_keys=True),flush=True)
    source=preflight_data["source"]
    target_lock=preflight_data["target_lock"]
    norm=json.loads(Path(source["data"]["normalizer"]).read_text(encoding="utf-8"))
    metric_lock=json.loads(Path(source["data"]["metric_scale_lock"]).read_text(encoding="utf-8"))
    if metric_lock.get("status")!="LOCKED" or int(metric_lock.get("final_heldout_payload_access",-1))!=0:
        raise RuntimeError("accepted metric lock is not valid/final0")
    metric_sigma=np.asarray(metric_lock["target_sigma"],dtype=np.float64)
    metric_fd=np.asarray(metric_lock["safe_fd_scale"],dtype=np.float64)
    train_sigma=np.asarray(norm["target_sigma"],dtype=np.float64)
    train_fd=np.asarray(norm["safe_fd_scale"],dtype=np.float64)
    if any(a.shape!=(28,) or not np.isfinite(a).all() or np.any(a<=0) for a in (metric_sigma,metric_fd,train_sigma,train_fd)):
        raise RuntimeError("accepted fixed scale shape/finite/positive check failed")

    # The order is deliberate: all source/header/ID/profile checks and synthetic
    # QA have passed; now load accepted V7 train targets, then one original dev
    # target archive. No final/heldout data are referenced.
    train_y_pack,z_by_seed,target_counts=load_training_targets(target_lock,train_data["rows"])
    y_train=train_y_pack["Y"]
    print(json.dumps({"stage":"accepted_train_Y_Z_loaded","train_Y_npy_loads":target_counts["train_Y_npy_loads"],
                      "train_Z_npy_loads":target_counts["train_Z_npy_loads"],
                      "dev_Y_loaded":False},sort_keys=True),flush=True)
    s_native={s:train_data["pred"][s]["S"] for s in SEEDS}
    v_native={s:train_data["pred"][s]["V"] for s in SEEDS}
    t_native={s:train_data["pred"][s]["T"] for s in SEEDS}
    target_formula_checks=[]
    for seed in SEEDS:
        z=z_by_seed[seed]
        if not np.array_equal(z["GT_RETRAIN"],y_train):
            raise RuntimeError(f"Z_G differs from Y for seed {seed}")
        if not np.allclose(t_native[seed],0.5*s_native[seed]+0.5*v_native[seed],rtol=0,atol=1e-12):
            raise RuntimeError(f"T is not 0.5(S+V) for train seed {seed}")
        if not np.allclose(z["SELF_SOFT"],0.5*y_train+0.5*v_native[seed],rtol=0,atol=1e-12):
            raise RuntimeError(f"Z_U identity mismatch seed {seed}")
        if not np.allclose(z["MIX_SOFT"],0.5*y_train+0.5*t_native[seed],rtol=0,atol=1e-12):
            raise RuntimeError(f"Z_M identity mismatch seed {seed}")
        if not np.allclose(z["MIX_SOFT"]-z["SELF_SOFT"],0.25*(s_native[seed]-v_native[seed]),rtol=0,atol=1e-12):
            raise RuntimeError(f"Z_M-Z_U identity mismatch seed {seed}")
        target_formula_checks.append({"seed":seed,"Z_G_equals_Y":True,"T_equals_half_S_plus_half_V":True,
                                      "Z_U_formula":True,"Z_M_formula":True,"Z_M_minus_Z_U_identity":True})
    dev_y,dev_anchor,dev_access=load_dev_targets_and_anchors(source,dev_data["rows"])

    # Exact source-based V7 train curves and V7 accepted dev per-window metrics
    # are retained; new local dev arrays use the already accepted predictions.
    train_fit_rows=build_train_target_and_fit(train_data["rows"],train_data["pred"],y_train,z_by_seed,
                                               metric_sigma,metric_fd,train_sigma,train_fd)
    imitation_rows=build_train_dev_imitation(train_data["rows"],train_data["pred"],
                                               dev_data["rows"],dev_data["pred"],
                                               metric_sigma,metric_fd,dev_anchor)
    horizon_rows=build_horizon_axis(dev_data["rows"],dev_data["pred"],dev_y,metric_sigma,metric_fd)
    help_rows=build_help_harm(dev_data["rows"])
    rows_by_split={"train":train_data["rows"],"dev":dev_data["rows"]}
    preds_by_split={"train":train_data["pred"],"dev":dev_data["pred"]}
    geometry=build_geometry(rows_by_split,preds_by_split,metric_sigma,metric_fd)
    loo_seed,loo_task=build_loo(rows_by_split,preds_by_split,
                                {"train":{s:y_train for s in SEEDS},"dev":dev_y},
                                metric_sigma,metric_fd,dev_anchor)
    case_rows,selection_rows,case_figure_rows=build_case_data(
        train_data["rows"],train_data["pred"],dev_data["rows"],dev_data["pred"],
        dev_y,dev_anchor,metric_sigma)

    curves=build_curve_summary()
    # Persist all target-derived tables/figure data before the dev archive reader
    # goes out of scope; no raw target values are written.
    write_csv(V8/"results/TRAIN_TARGET_AND_FIT.csv",list(train_fit_rows[0]),train_fit_rows)
    write_csv(V8/"results/TRAIN_DEV_IMITATION.csv",list(imitation_rows[0]),imitation_rows)
    write_csv(V8/"results/CORRECTION_GEOMETRY.csv",list(geometry[0]),geometry)
    write_csv(V8/"results/HORIZON_AXIS_ERRORS.csv",list(horizon_rows[0]),horizon_rows)
    write_csv(V8/"results/TEACHER_HELP_HARM.csv",list(help_rows[0]),help_rows)
    write_csv(V8/"results/SEED_BLOCK_LOO.csv",list(loo_seed[0]),loo_seed)
    write_csv(V8/"results/TASK_LOO.csv",list(loo_task[0]),loo_task)
    write_csv(V8/"results/TRAIN_CURVE_SUMMARY.csv",list(curves[0]),curves)
    write_csv(V8/"CASE_SELECTION.csv",list(case_rows[0]),case_rows)
    write_csv(V8/"figure_data/case_trajectories.csv",list(case_figure_rows[0]),case_figure_rows)
    write_csv(V8/"figure_data/train_axis_scores.csv",["coordinate_index","equal_task_mean_abs_T_minus_V_over_sigma","axis_group"],
              [{"coordinate_index":i,"equal_task_mean_abs_T_minus_V_over_sigma":float(v),
                "axis_group":"0:14" if i<14 else "14:28"} for i,v in enumerate(
                  np.mean([weighted_mean(np.mean(np.abs(train_data["pred"][202610131]["T"]-
                     train_data["pred"][202610131]["V"])/metric_sigma[None,None,:],axis=1),
                     row_weights(train_data["rows"][202610131],
                       task_indices(train_data["rows"][202610131],t),"episode_id"))
                    for t in TASKS],axis=0))])
    write_csv(V8/"figure_data/horizon_axis_errors.csv",horizon_fields,horizon_rows)
    write_csv(V8/"figure_data/teacher_distances_train_dev.csv",list(imitation_rows[0]),imitation_rows)
    write_csv(V8/"CASE_SELECTION_LEDGER.csv",list(selection_rows[0]),selection_rows)

    # Alias proof and actual label-read receipt are separate from the pre-label QA.
    alias_qa={
      "schema_version":1,"status":"PASS","created_utc":now(),
      "train_G_equals_V_model_file_sha":True,
      "train_G_prediction_reused_from_V":True,
      "train_inference_inputs_and_profile":"same V7 train-prefix loader/normalizer/TS11-D/batch32/eval-inference_mode; final model SHA exact; receiver identity hash locked",
      "dev_G_and_V_tensors_kept_separate":True,
      "dev_G_vs_V":preflight_data["dev_alias_checks"],
      "dev_numeric_alias_tolerance":"not asserted; original V7 accepted score reconciliation retained; direct output deltas reported",
      "normalizer_sha256":source["data"]["normalizer_sha256"],
      "metric_scale_lock_sha256":source["data"]["metric_scale_lock_sha256"],
      "final_heldout_payload_access":0,
    }
    write_json(V8/"qa/ALIAS_QA.json",alias_qa)
    real_qa={
      "schema_version":1,"status":"PASS","created_utc":now(),
      "preflight_status":"PASS","synthetic_qa_status":"PASS",
      "prediction_identity":{"train_teacher_rows":9,"new_train_student_forwards":6,
                             "dev_student_reused":9,"dev_reference_reused":9,
                             "ordered_train_ids_sha256":target_lock["ordered_receiver_ids_sha256"],
                             "dev_rows":185,"dev_unique_ids":dev_access["dev_target_unique_ids"]},
      "target_formula_checks":target_formula_checks,
      "train_group_weighting":preflight_data["train_group_qa"],
      "scale_sources":{"normalizer_path":source["data"]["normalizer"],
                       "normalizer_sha256":source["data"]["normalizer_sha256"],
                       "metric_lock_path":source["data"]["metric_scale_lock"],
                       "metric_lock_sha256":source["data"]["metric_scale_lock_sha256"],
                       "training_target_sigma":"normalizer.target_sigma",
                       "training_internal_fd_scale":"normalizer.safe_fd_scale",
                       "canonical_position_scale":"METRIC_SCALE_LOCK.target_sigma",
                       "canonical_safe_fd_scale":"METRIC_SCALE_LOCK.safe_fd_scale"},
      "access_counts":{**target_counts,**dev_access,
                       "prediction_native_arrays_loaded":33,
                       "prediction_arrays_source":"V7 train S/V/T, six V8 U/M, V7 dev G/U/M and V2/V5 dev S/V/T",
                       "action_residual_tensor_loaded":False,
                       "train_Y_npy_loads_total":target_counts["train_Y_npy_loads"]+prior_y,
                       "train_Z_npy_loads_total":target_counts["train_Z_npy_loads"]+prior_z,
                       "prediction_native_arrays_loaded_successful":33,
                       "prediction_native_arrays_loaded_prior_failed":prior_pred,
                       "prediction_native_arrays_loaded_schema_probe":probe_pred,
                       "prediction_native_arrays_loaded_total":33+prior_pred+probe_pred,
                       "diagnostic_metric_window_counters_successful":dict(COUNTERS)},
      "new_dev_prediction_jobs":0,"new_fits":0,"new_optimizer_updates":0,
      "new_teacher_forward_jobs":0,"final_heldout_payload_access":0,
    }
    write_json(V8/"qa/REAL_DATA_QA.json",real_qa)
    exposure=build_resource_exposure(train_data["pred"],dev_data["pred"],target_counts,dev_access,
                                     curves,train_fit_rows,horizon_rows,dict(COUNTERS))
    write_csv(V8/"results/V8_RESOURCE_AND_EXPOSURE.csv",list(exposure[0]),exposure)
    extract_receipt={
      "schema_version":1,"status":"PASS","created_utc":now(),
      "train_Y_and_Z_loaded_after_preflight_and_synthetic_QA":True,
      "dev_target_archive_loads":dev_access["dev_target_archive_loads"],
      "new_fits":0,"optimizer_updates":0,"new_teacher_forward_jobs":0,"new_dev_prediction_jobs":0,
      "new_frozen_student_train_forward_jobs":6,"new_bootstrap_draws":0,"encoder_extraction":0,
      "final_heldout_payload_access":0,"old_root_writes":0,
      "outputs":[
        "results/TRAIN_TARGET_AND_FIT.csv","results/TRAIN_CURVE_SUMMARY.csv",
        "results/TRAIN_DEV_IMITATION.csv","results/CORRECTION_GEOMETRY.csv",
        "results/HORIZON_AXIS_ERRORS.csv","results/TEACHER_HELP_HARM.csv",
        "results/SEED_BLOCK_LOO.csv","results/TASK_LOO.csv",
        "results/V8_RESOURCE_AND_EXPOSURE.csv","CASE_SELECTION.csv",
        "CASE_SELECTION_LEDGER.csv","figure_data/case_trajectories.csv",
        "figure_data/horizon_axis_errors.csv","figure_data/teacher_distances_train_dev.csv",
        "figure_data/train_axis_scores.csv","qa/REAL_DATA_QA.json","qa/ALIAS_QA.json"
      ]
    }
    write_json(V8/"coordination/EXTRACT_COMPLETE.json",extract_receipt)
    print(json.dumps({
      "status":"PASS","train_target_rows":len(train_fit_rows),"imitation_rows":len(imitation_rows),
      "geometry_rows":len(geometry),"horizon_axis_rows":len(horizon_rows),"help_harm_rows":len(help_rows),
      "seed_loo_rows":len(loo_seed),"task_loo_rows":len(loo_task),
      "cases":len(case_rows),"dev_target_archive_loads":dev_access["dev_target_archive_loads"],
      "new_dev_predictions":0,"new_fits":0,"updates":0,
    },sort_keys=True),flush=True)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--synthetic-only",action="store_true")
    args=parser.parse_args()
    if args.synthetic_only:
        result=synthetic_pipeline_qa()
        write_json(V8/"qa/EXTRACT_SYNTHETIC_QA.json",result)
        print(json.dumps(result,sort_keys=True))
        return 0 if result["status"]=="PASS" else 1
    run_extract()
    return 0


if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import os
import csv, hashlib, io, json, os, resource, socket, time, uuid, fcntl, shutil
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors import safe_open

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V5=PROJECT/"reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
ROOT=PROJECT/"reports/benchmarks/CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z"
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
PROGRAMS=["S","A","V"]
METRICS=["position_mae","safe_fd_l1","seam_safe_fd_l1"]
DRAWS=5000
BOOT_SEED=2026101506
ATOL=1e-12
RTOL=1e-10

def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def log(s): print(f"[{utc()}] {s}",flush=True)
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+".tmp")
    t.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(t,p)
def read_csv(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields:fields.append(k)
    if not fields:fields=["status"]
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        w.writeheader();w.writerows(rows)
def as_text(x):
    if isinstance(x,bytes):return x.decode("utf-8")
    if isinstance(x,np.bytes_):return bytes(x).decode("utf-8")
    return str(x)
def id_hash(ids):return sha_bytes(("".join(f"{x}\n" for x in ids)).encode())
def agg(rows,metric,task):
    ep=defaultdict(list)
    for r in rows:
        if int(r["task_id"])==int(task):ep[str(r["episode_id"])].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in ep.values()]))
def macro(tv):return float(np.mean([tv[t] for t in TASKS]))
def sd(vals):
    a=np.asarray(list(vals),dtype=np.float64)
    return float(np.std(a,ddof=1)) if a.size>1 else float("nan")
def load_tensor(path,field="prediction_native"):
    with safe_open(str(path),framework="numpy",device="cpu") as f:
        if field not in f.keys():raise RuntimeError(f"missing tensor field {field} in {path}")
        a=np.asarray(f.get_tensor(field))
    if a.dtype.hasobject or not np.isfinite(a).all():raise RuntimeError(f"invalid tensor {path}")
    return np.asarray(a,dtype=np.float64,order="C")
def generate_pair(x,y):return np.asarray(0.5*np.asarray(x,dtype=np.float64)+0.5*np.asarray(y,dtype=np.float64),dtype=np.float64,order="C")

def main():
    lock= (ROOT/"coordination/OWNER.lock").open("a+")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("V6 resume refused: owner lock is active")
    owner_path=ROOT/"coordination/OWNER_STATE.json"
    exposure_path=ROOT/"inventory/OUTCOME_EXPOSURE.json"
    # The original post-score failure is retained; timestamped backups precede authorized ledger/state updates.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(owner_path,owner_path.with_name(owner_path.name+f".pre-resume-{stamp}.bak"))
    shutil.copy2(exposure_path,exposure_path.with_name(exposure_path.name+f".pre-resume-{stamp}.bak"))
    lock_backup=ROOT/"coordination"/f"OWNER.lock.pre-resume-{stamp}.bak"
    lock.seek(0);old_lock=lock.read()
    lock_backup.write_text(old_lock,encoding="utf-8")
    token=uuid.uuid4().hex
    lock_info={"host":socket.gethostname(),"owner_uid":os.getuid(),"owner_pid":os.getpid(),
        "owner_token_sha256":sha_bytes(token.encode()),"study":"post_V5_equal_half_ensemble_controls_development",
        "out_root":str(ROOT),"owner_lock_acquired":True,"resume_attempt":1}
    lock.seek(0);lock.truncate();lock.write(json.dumps(lock_info,sort_keys=True)+"\n");lock.flush();os.fsync(lock.fileno())
    owner=read_json(owner_path)
    owner.update({"status":"running","phase":"resume_from_complete_score_table","host":socket.gethostname(),
        "owner_uid":os.getuid(),"owner_pid":os.getpid(),"owner_token_sha256":lock_info["owner_token_sha256"],
        "updated_utc":utc(),"previous_failure_preserved":True,"resume_attempt":1})
    write_json(owner_path,owner)
    failure=read_json(ROOT/"qa/FAILURE.json")
    failure_record={"attempt":1,"phase":"post_score_pre_aggregation",
        "created_utc":utc(),"error_type":failure.get("error_type"),"error_message":failure.get("error_message"),
        "failed_at":"source-member lookup while building L1 cancellation ledger",
        "complete_score_table_path":"results/PER_WINDOW_PAIR_METRICS.csv",
        "complete_score_table_sha256":sha_file(ROOT/"results/PER_WINDOW_PAIR_METRICS.csv"),
        "complete_score_rows":len(read_csv(ROOT/"results/PER_WINDOW_PAIR_METRICS.csv")),
        "loss_recomputation_on_resume":0,"recovery_action":"reuse sealed arrays and complete pair-window score table; reload dev archive only for elementwise cancellation diagnostics"}
    with (ROOT/"qa/FAILED_ATTEMPTS.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(failure_record,ensure_ascii=False)+"\n")

    t0=time.perf_counter()
    try:
        # Immutable target-free locks and complete score table must match the original seal.
        seal=read_json(ROOT/"qa/PREDICTION_SEAL.json")
        formula=read_json(ROOT/"protocol/FORMULA_LOCK.json")
        pair_plan=read_csv(ROOT/"protocol/PAIR_PLAN.csv")
        index=read_csv(ROOT/"predictions/PREDICTION_INDEX.csv")
        source_map=read_json(ROOT/"inventory/SOURCE_MODEL_MAP.json")
        if seal.get("status")!="PASS" or seal.get("nonempty_seal") is not True or seal.get("sealed_bundles")!=36:
            raise RuntimeError("resume refuses invalid/incomplete prediction seal")
        formula_sha=sha_file(ROOT/"protocol/FORMULA_LOCK.json")
        if seal.get("formula_lock_sha256")!=formula_sha or seal.get("pair_plan_sha256")!=sha_file(ROOT/"protocol/PAIR_PLAN.csv"):
            raise RuntimeError("resume lock hash differs from the sealed prediction batch")
        if seal.get("source_model_map_sha256")!=sha_file(ROOT/"inventory/SOURCE_MODEL_MAP.json"):
            raise RuntimeError("resume source map changed after seal")
        if len(pair_plan)!=36 or len(index)!=36:
            raise RuntimeError("resume pair plan/index is incomplete")
        score_path=ROOT/"results/PER_WINDOW_PAIR_METRICS.csv"
        score_rows=read_csv(score_path)
        if len(score_rows)!=6660:
            raise RuntimeError("preserved pair-window score table is incomplete")
        counts=Counter(r["pair_id"] for r in score_rows)
        if len(counts)!=36 or any(n!=185 for n in counts.values()):
            raise RuntimeError("preserved score table does not contain 36 complete bundles")
        score_source_count=Counter(r["score_source"] for r in score_rows)
        if score_source_count["V6_target_score"]!=6105 or score_source_count["V5_MIX50_reused_after_reconciliation"]!=555:
            raise RuntimeError(f"preserved score table source counts invalid: {score_source_count}")
        qa_pre=read_json(ROOT/"qa/SCHEMA_PREFLIGHT.json")
        if qa_pre.get("status")!="PASS" or qa_pre.get("prediction_target_reads_before_seal")!=0:
            raise RuntimeError("V6 target-free preflight is not accepted")
        v5map=read_json(V5/"inventory/SOURCE_MAP.json")
        order=read_csv(V5/"protocol/ORDERED_RECEIVER_IDS.csv")
        order.sort(key=lambda r:int(r["canonical_row_index"]))
        ids=[r["anchor_id"] for r in order]
        if id_hash(ids)!=formula["ordered_receiver_ids_sha256"] or len(ids)!=185:
            raise RuntimeError("locked canonical IDs failed resume validation")
        canonical={a:i for i,a in enumerate(ids)}
        joins=read_csv(ROOT/"inventory/SOURCE_JOINS.csv")
        meta={}
        nodes={}
        for r in joins:
            aid=r["anchor_id"]
            mm=(str(r["episode_id"]),str(r["task_id"]),str(r["source_group"]))
            if aid in meta and meta[aid]!=mm:raise RuntimeError(f"source join drift for {aid}")
            meta[aid]=mm
            k=(r["program"],int(r["seed"]))
            nodes.setdefault(k,{"node_id":r["node_id"],"path":r["source_path"],"sha":r["source_sha256"],"role":r["source_role"]})
        if set(meta)!=set(ids) or len(nodes)!=9:
            raise RuntimeError("locked V6 source joins or node count incomplete")
        groups_by_task=defaultdict(set);group_task={}
        for ep,task_s,group in meta.values():
            task=int(task_s)
            if group in group_task and group_task[group]!=task:raise RuntimeError("source group crosses tasks")
            group_task[group]=task;groups_by_task[task].add(group)
        if len(group_task)!=47:
            raise RuntimeError("expected 47 source groups on resume")

        # Verify every sealed output and reload only already existing arrays; no new prediction bundle is written.
        pair_arrays={};pair_map={r["pair_id"]:r for r in pair_plan};index_map={r["pair_id"]:r for r in index}
        node_arrays={}
        for key,rec in nodes.items():
            p=Path(rec["path"])
            if sha_file(p)!=rec["sha"]:raise RuntimeError(f"source prediction changed: {key}")
            smrec=next(x for x in v5map["source_prediction_bundles"] if x["role"]==rec["role"] and int(x["seed"])==key[1])
            rows=read_csv(smrec["prediction_rows_path"])
            source_ids=[r["anchor_id"] for r in rows]
            if len(source_ids)!=185 or set(source_ids)!=set(ids):raise RuntimeError(f"source row sidecar mismatch: {key}")
            node_arrays[key]=load_tensor(p,smrec["native_field"])[[source_ids.index(a) for a in ids]]
        for p in pair_plan:
            pid=p["pair_id"];ent=index_map[pid];pp=Path(ent["prediction_path"])
            if sha_file(pp)!=ent["prediction_sha256"]:raise RuntimeError(f"sealed output changed: {pid}")
            if pp.suffix==".npy":
                arr=np.load(pp,allow_pickle=False)
                if arr.dtype.hasobject or tuple(arr.shape)!=(185,30,28) or not np.isfinite(arr).all():
                    raise RuntimeError(f"invalid saved pair array {pid}")
                arr=np.asarray(arr,dtype=np.float64)
            else:
                arr=load_tensor(pp)
            if arr.shape!=(185,30,28):raise RuntimeError(f"pair array shape mismatch {pid}")
            pair_arrays[pid]=arr
        score_by_pair=defaultdict(list)
        for r in score_rows:
            if r["anchor_id"] not in meta:raise RuntimeError("score table has unknown receiver ID")
            expected=(meta[r["anchor_id"]][0],meta[r["anchor_id"]][1],meta[r["anchor_id"]][2])
            if (r["episode_id"],str(r["task_id"]),r["source_group"])!=expected:
                raise RuntimeError(f"score table join changed: {r['pair_id']}/{r['anchor_id']}")
            for m in METRICS:
                r[m]=float(r[m])
                if not np.isfinite(r[m]):raise RuntimeError("nonfinite preserved pair loss")
            score_by_pair[r["pair_id"]].append(r)

        # The previous scorer load is recorded; this one reload is solely for elementwise cancellation diagnostics.
        exposure=read_json(exposure_path)
        if exposure.get("target_archive_load_operations")!=1 or exposure.get("target_archive_physical_read_operations")!=1:
            raise RuntimeError("unexpected pre-recovery V6 target exposure count")
        v5_exposure=read_json(V5/"inventory/OUTCOME_EXPOSURE.json")
        target_path=Path(v5_exposure["target_archive_path"])
        expected_target_sha=v5_exposure["target_archive_sha256"]
        prefix_rec=next(x for x in v5map["source_artifacts"] if Path(x["path"]).name=="DEV_PREFIXES_NO_LABELS.npz")
        prefix_path=Path(prefix_rec["path"])
        if sha_file(prefix_path)!=prefix_rec["sha256"]:raise RuntimeError("label-free prefix source changed")
        exposure["status"]="RECOVERY_DIAGNOSTIC_ARCHIVE_READ_STARTED"
        exposure["recovery_diagnostic_target_archive_read_started_utc"]=utc()
        exposure["history"].append({"phase":"resume_elementwise_cancellation","status":"started",
            "additional_target_archive_loads":0,"pair_loss_rescoring":0})
        write_json(exposure_path,exposure)
        archive_start=time.perf_counter()
        blob=target_path.read_bytes()
        got_sha=sha_bytes(blob)
        if got_sha!=expected_target_sha:raise RuntimeError("locked development archive checksum changed during recovery")
        with np.load(io.BytesIO(blob),allow_pickle=False) as z:
            fields={"anchor_id","episode_id","task_id","source_cluster","target_native_h30"}
            if not fields.issubset(set(z.files)):raise RuntimeError("development archive missing whitelisted scorer fields")
            tids=[as_text(x) for x in z["anchor_id"].tolist()]
            target_raw=np.asarray(z["target_native_h30"],dtype=np.float64).copy()
            target_meta={tids[i]:(as_text(z["episode_id"][i]),as_text(z["task_id"][i]),as_text(z["source_cluster"][i]))
                         for i in range(len(tids))}
        with np.load(prefix_path,allow_pickle=False) as z:
            pids=[as_text(x) for x in z["anchor_id"].tolist()]
            anchors_raw=np.asarray(z["anchor_native"],dtype=np.float64).copy()
            prefix_meta={pids[i]:(as_text(z["episode_id"][i]),as_text(z["task_id"][i]),as_text(z["source_cluster"][i]))
                         for i in range(len(pids))}
        if set(tids)!=set(ids) or set(pids)!=set(ids) or target_raw.shape!=(185,30,28) or anchors_raw.shape!=(185,28):
            raise RuntimeError("recovery target/prefix schema or IDs invalid")
        ti={a:i for i,a in enumerate(tids)};pi={a:i for i,a in enumerate(pids)}
        target=np.stack([target_raw[ti[a]] for a in ids])
        anchors=np.stack([anchors_raw[pi[a]] for a in ids])
        for aid in ids:
            if target_meta[aid]!=prefix_meta[aid] or target_meta[aid]!=meta[aid]:
                raise RuntimeError(f"target/prefix/source join mismatch {aid}")
        if not np.isfinite(target).all() or not np.isfinite(anchors).all():
            raise RuntimeError("nonfinite recovery target/prefix values")
        archive_read_seconds=time.perf_counter()-archive_start

        # Load the locked metric scale; outcome independent and hash-bound to the source map.
        mref=v5map["metric_scale_lock"];metric_path=Path(mref["path"])
        if sha_file(metric_path)!=mref["sha256"]:raise RuntimeError("metric scale lock changed")
        metric_lock=read_json(metric_path)
        sigma=np.asarray(metric_lock["target_sigma"],dtype=np.float64)
        fdscale=np.asarray(metric_lock["safe_fd_scale"],dtype=np.float64)
        if sigma.shape!=(28,) or fdscale.shape!=(28,) or np.any(sigma<=0) or np.any(fdscale<=0):
            raise RuntimeError("metric scales invalid during recovery")
        # Source member losses come from accepted V5 rows; no member score recomputation.
        refs=read_csv(V5/"results/PER_WINDOW_ACCEPTED_REFERENCES.csv")
        source_loss=defaultdict(dict)
        baseline_loss=defaultdict(dict)
        for r in refs:
            method=r["method"];seed=int(r["seed"]);aid=r["anchor_id"]
            if method not in {"S","A","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"}:continue
            row={**r,"seed":str(seed)}
            for m in METRICS:row[m]=float(r[m])
            if method in {"S","A","V"}:source_loss[(method,seed)][aid]=row
            else:baseline_loss[(method,seed)][aid]=row
        if any(len(source_loss[(m,s)])!=185 for m in ["S","A","V"] for s in SEEDS):
            raise RuntimeError("accepted V5 source component loss rows incomplete")

        # Pure CPU composition timing on the already accepted arrays; excludes file loading and model inference.
        comp0=time.perf_counter()
        composed_count=0
        for r in pair_plan:
            if r["reuse_v5_mix50"]=="true":continue
            generate_pair(node_arrays[(r["left_program"],int(r["left_seed"]))],
                          node_arrays[(r["right_program"],int(r["right_seed"]))])
            composed_count+=1
        composition_arithmetic_seconds=time.perf_counter()-comp0
        if composed_count!=33:raise RuntimeError("CPU composition timing did not visit all 33 new pairs")

        # Build raw member normalized errors and the locked elementwise L1 identity; scalar losses remain the saved table values.
        cancel_anchor=defaultdict(list);max_identity=defaultdict(float);zero_counts=defaultdict(int)
        finite_counts=defaultdict(int);min_abs=defaultdict(lambda:float("inf"))
        l1_start=time.perf_counter()
        for p in pair_plan:
            pid=p["pair_id"]
            x=node_arrays[(p["left_program"],int(p["left_seed"]))]
            y=node_arrays[(p["right_program"],int(p["right_seed"]))]
            err_sets=[
                ("position_mae",(x-target)/sigma[None,None,:],(y-target)/sigma[None,None,:]),
                ("safe_fd_l1",(np.diff(x,axis=1)-np.diff(target,axis=1))/fdscale[None,None,:],
                              (np.diff(y,axis=1)-np.diff(target,axis=1))/fdscale[None,None,:]),
                ("seam_safe_fd_l1",((x[:,0,:]-anchors)-(target[:,0,:]-anchors))/fdscale[None,:],
                                    ((y[:,0,:]-anchors)-(target[:,0,:]-anchors))/fdscale[None,:])
            ]
            for metric,u,v in err_sets:
                opposite=((u<0)&(v>0))|((u>0)&(v<0))
                lhs=0.5*(np.abs(u)+np.abs(v))-np.abs(0.5*(u+v))
                rhs=np.where(opposite,np.minimum(np.abs(u),np.abs(v)),0.0)
                resid=np.abs(lhs-rhs)
                max_identity[(pid,metric)]=float(np.max(resid))
                zero_counts[(pid,metric)]=int(np.count_nonzero((u==0)|(v==0)))
                finite_counts[(pid,metric)]=int(np.isfinite(u).sum())
                nonzero=np.concatenate([np.abs(u[np.nonzero(u)]),np.abs(v[np.nonzero(v)])])
                if nonzero.size:min_abs[(pid,metric)]=float(np.min(nonzero))
                for i,aid in enumerate(ids):
                    ep,task,group=meta[aid]
                    cancel_anchor[(pid,metric)].append({"anchor_id":aid,"episode_id":ep,"task_id":task,
                        "source_group":group,"cancel_amount":float(rhs[i].mean()),
                        "opposite_sign_fraction":float(opposite[i].sum()/u[i].size),
                        "zero_member_error_elements":int(np.count_nonzero((u[i]==0)|(v[i]==0))),
                        "element_count":int(u[i].size),"max_element_identity_residual":float(np.max(resid[i]))})

        ledger=[]
        family_names=["SS","AA","VV","SA","SV","AV"]
        for p in pair_plan:
            pid=p["pair_id"];left=source_loss[(p["left_program"],int(p["left_seed"]))]
            right=source_loss[(p["right_program"],int(p["right_seed"]))]
            joint={r["anchor_id"]:r for r in score_by_pair[pid]}
            ll={r["anchor_id"]:r for r in left.values()};rl={r["anchor_id"]:r for r in right.values()}
            ca={r["anchor_id"]:r for r in cancel_anchor[(pid,METRICS[0])]}
            for metric in METRICS:
                ca={r["anchor_id"]:r for r in cancel_anchor[(pid,metric)]}
                for scope,task in [("task",t) for t in TASKS]+[("macro",None)]:
                    sub=[a for a in ids if task is None or int(meta[a][1])==task]
                    def ep_equal(vals):
                        eps=defaultdict(list)
                        for aid,val in vals:eps[meta[aid][0]].append(float(val))
                        return float(np.mean([np.mean(v) for v in eps.values()]))
                    lx=ep_equal([(a,ll[a][metric]) for a in sub])
                    ly=ep_equal([(a,rl[a][metric]) for a in sub])
                    le=ep_equal([(a,joint[a][metric]) for a in sub])
                    jgap=.5*(lx+ly)-le
                    cancel=ep_equal([(a,ca[a]["cancel_amount"]) for a in sub])
                    opposite=ep_equal([(a,ca[a]["opposite_sign_fraction"]) for a in sub])
                    residual=jgap-cancel
                    maxelem=max(ca[a]["max_element_identity_residual"] for a in sub)
                    zc=sum(ca[a]["zero_member_error_elements"] for a in sub)
                    ec=sum(ca[a]["element_count"] for a in sub)
                    ledger.append({"pair_id":pid,"family":p["family"],"left_program":p["left_program"],
                        "left_seed":p["left_seed"],"right_program":p["right_program"],"right_seed":p["right_seed"],
                        "same_seed":p["same_seed"],"scope":scope,"task_id":task if task is not None else "",
                        "metric":metric,"left_loss":lx,"right_loss":ly,"ensemble_loss":le,
                        "j_gap_member_mean_minus_ensemble":jgap,"gain_vs_left_member":lx-le,
                        "gain_vs_right_member":ly-le,"gain_vs_better_member":min(lx,ly)-le,
                        "elementwise_cancellation_amount":cancel,"opposite_sign_error_fraction":opposite,
                        "j_gap_reconstruction_residual":residual,"max_element_identity_residual":maxelem,
                        "identity_atol":ATOL,"identity_rtol":RTOL,
                        "identity_pass":bool(abs(residual)<=ATOL+RTOL*max(1.0,abs(jgap),abs(cancel)) and maxelem<=ATOL+RTOL),
                        "member_error_exact_zero_elements":zc,"element_count":ec,
                        "min_nonzero_abs_normalized_member_error":min_abs[(pid,metric)],
                        "nonfinite_elements":0,"source":"saved V5/V6 scalar loss rows plus recovery elementwise diagnostic"})
        if not all(r["identity_pass"] for r in ledger):
            raise RuntimeError("recovery J_gap identity failed: "+repr([r for r in ledger if not r["identity_pass"]][:3]))
        write_csv(ROOT/"results/L1_CANCELLATION_LEDGER.csv",ledger)
        l1_seconds=time.perf_counter()-l1_start
        recovery_elapsed=time.perf_counter()-t0

        # Exposure ledger: two archive loads total, no loss-table recomputation on recovery.
        exposure.update({"status":"COMPLETE","target_archive_load_operations":2,
            "target_archive_physical_read_operations":2,"target_array_window_rows_loaded":370,
            "target_archive_unique_windows":185,"target_archive_sha256_actual":got_sha,
            "additional_target_archive_loads_for_recovery":1,"recovery_archive_purpose":"elementwise L1 cancellation diagnostics only",
            "pair_loss_rescoring_on_recovery":0,"new_pair_window_loss_evaluations":6105,
            "mix50_reconciliation_window_evaluations":555,"reused_v5_mix50_loss_rows":555,
            "cancellation_diagnostic_pair_windows":6660,"cancellation_diagnostic_metrics":METRICS,
            "final_heldout_payload_access":0,"old_root_writes":0,"source_roots_read_only":True,
            "recovery_completed_utc":utc(),
            "history":exposure["history"]+[{"phase":"recovery_archive_read","status":"complete",
                 "additional_target_archive_loads":1,"purpose":"elementwise L1 cancellation diagnostics only",
                 "pair_loss_rescoring":0,"target_windows_loaded":185},
                {"phase":"recovery_aggregation","status":"complete","pair_score_rows_reused":6660,
                 "new_pair_window_loss_evaluations":6105,"reused_v5_mix50_loss_rows":555}]})
        write_json(exposure_path,exposure)

        # Save recovery receipt for the preserved scorer pass and the exact failure/resume chain.
        write_json(ROOT/"qa/SCORER_RECOVERY.json",{"schema_version":1,"status":"PASS","created_utc":utc(),
            "original_failure_path":"qa/FAILURE.json","failed_attempt_record_path":"qa/FAILED_ATTEMPTS.jsonl",
            "preserved_score_table":"results/PER_WINDOW_PAIR_METRICS.csv","preserved_score_table_sha256":sha_file(score_path),
            "preserved_score_rows":len(score_rows),"new_pair_score_window_evaluations":6105,
            "reused_v5_mix50_loss_rows":555,"mix50_reconciliation_window_evaluations":555,
            "loss_table_recomputed_on_resume":False,"target_archive_loads_total":2,
            "additional_recovery_archive_loads":1,"recovery_read_used_for":"elementwise L1 cancellation diagnostics",
            "final_heldout_payload_access":0})

        # Figure data and visual outputs are derived from accepted tables plus the second diagnostic array read.
        pair_task=read_csv(ROOT/"results/PAIR_TASK_METRICS.csv")
        fam_rows=read_csv(ROOT/"results/FAMILY_EFFECTS.csv")
        principal=read_csv(ROOT/"results/PRINCIPAL_MIX50_EFFECTS.csv")
        ci=read_csv(ROOT/"results/PAIRED_GROUP_CI.csv")
        seed_loo=read_csv(ROOT/"results/SEED_BLOCK_LOO.csv")
        task_loo=read_csv(ROOT/"results/TASK_LOO.csv")
        write_csv(ROOT/"figure_data/PRINCIPAL_GAINS.csv",[
            {"reference":r["reference"],"metric":r["metric"],"gain_mean":r["gain"],
             "gain_sample_sd_across_three_fixed_pairs":r.get("gain_sample_sd_across_three_fixed_pairs",""),
             "ci_low":next((c["ci_low"] for c in ci if c["contrast"]==f"MIX50_DIAG_MEAN3_vs_{r['reference']}" and c["metric"]==r["metric"]),""),
             "ci_high":next((c["ci_high"] for c in ci if c["contrast"]==f"MIX50_DIAG_MEAN3_vs_{r['reference']}" and c["metric"]==r["metric"]),""),
             "ci_scope":"three fixed diagonal pairs; four locked references" if r["reference"]!="PERSISTENCE" else "not computed by locked CI plan"}
            for r in principal if r["scope"]=="seed_mean_macro"])
        write_csv(ROOT/"figure_data/FAMILY_MEANS_AND_PAIR_SPREAD.csv",
            [r for r in fam_rows if r["record_type"]=="family_mean" and r["scope"]=="macro"])
        write_csv(ROOT/"figure_data/SEED_BLOCK_LOO.csv",seed_loo)
        write_csv(ROOT/"figure_data/JGAP_AND_GAINS.csv",[
            r for r in ledger if r["scope"]=="macro"])
        selected=read_json(V5/"protocol/FORMULA_LOCK.json")["metadata_selected_cases"]["selected_anchor_ids"]
        mix_seed=SEEDS[0]
        principal_pid=next(p["pair_id"] for p in pair_plan if p["family"]=="SV" and p["same_seed"]=="true" and int(p["paired_seed"])==mix_seed)
        mix_arr=pair_arrays[principal_pid]
        traj=[]
        for task_s,aid in selected.items():
            idx=canonical[aid];task=int(task_s)
            for axis in [0,14]:
                for h in range(30):
                    for series,arr in [("target_GT",target),("S_state_member",node_arrays[("S",mix_seed)]),
                                       ("V_fusion_member",node_arrays[("V",mix_seed)]),("MIX50_average",mix_arr)]:
                        val=float(arr[idx,h,axis]) if series!="target_GT" else float(arr[idx,h,axis])
                        traj.append({"task_id":task,"anchor_id":aid,"seed":mix_seed,"axis":axis,
                                     "horizon_step":h+1,"series":series,"native_value":val})
        write_csv(ROOT/"figure_data/TRAJECTORY_DATA.csv",traj)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fam_order=["SS","AA","VV","SA","SV","AV"]
        colors={"SS":"#3465a4","AA":"#73d216","VV":"#cc0000","SA":"#75507b","SV":"#f57900","AV":"#c17d11"}
        refs_order=["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]
        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        for ax,metric in zip(axs,METRICS):
            rows=[r for r in read_csv(ROOT/"figure_data/PRINCIPAL_GAINS.csv") if r["metric"]==metric]
            yy=[float(next(r["gain_mean"] for r in rows if r["reference"]==ref)) for ref in refs_order]
            lo=[];hi=[]
            for ref,yv in zip(refs_order,yy):
                rr=next(r for r in rows if r["reference"]==ref)
                lo.append(yv-float(rr["ci_low"]) if rr["ci_low"]!="" else 0.0)
                hi.append(float(rr["ci_high"])-yv if rr["ci_high"]!="" else 0.0)
            ax.axhline(0,color="black",linewidth=.8)
            ax.errorbar(np.arange(5),yy,yerr=[lo,hi],fmt="o",capsize=4,color="#204a87")
            ax.set_xticks(np.arange(5),refs_order,rotation=30,ha="right")
            ax.set_title(metric);ax.set_ylabel("reference loss - MIX50 loss")
        fig.suptitle("Fixed V5 same-seed MIX50: direct paired gains")
        fig.savefig(ROOT/"figures/MIX50_DIRECT_REFERENCE_GAINS.png",dpi=160);plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        for ax,metric in zip(axs,METRICS):
            for i,fam in enumerate(fam_order):
                pids=[r["pair_id"] for r in pair_task if r["record_type"]=="pair_system" and r["scope"]=="macro" and r["family"]==fam and r["metric"]==metric]
                vals=[float(next(r["loss"] for r in pair_task if r["pair_id"]==pid and r["metric"]==metric and r["scope"]=="macro")) for pid in pids]
                jitter=np.linspace(-.16,.16,len(vals)) if len(vals)>1 else np.array([0.0])
                ax.scatter(np.full(len(vals),i)+jitter,vals,s=26,color=colors[fam],alpha=.8)
                fr=next(r for r in fam_rows if r["record_type"]=="family_mean" and r["scope"]=="macro" and r["family"]==fam and r["metric"]==metric)
                sdev=float(fr["pair_sample_sd"]) if fr.get("pair_sample_sd") not in ("","nan","NaN") else np.nan
                ax.errorbar([i],[float(fr["mean_pair_loss"])],yerr=[[sdev],[sdev]] if np.isfinite(sdev) else None,
                            fmt="D",capsize=4,color="black")
            ax.set_xticks(np.arange(6),fam_order);ax.set_title(metric)
            ax.set_ylabel("family mean loss; finite pair SD")
        fig.suptitle("Six fixed families: all pair points and family means")
        fig.savefig(ROOT/"figures/FAMILY_MEANS_AND_PAIR_SPREAD.png",dpi=160);plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        loo_names=["SV_vs_SS","SV_vs_VV","SV_vs_SA","AV_vs_AA","AV_vs_VV","SV_pairing_effect"]
        for ax,metric in zip(axs,METRICS):
            for name in loo_names:
                vals=[]
                for seed in SEEDS:
                    row=next(r for r in seed_loo if int(r["leave_out_seed"])==seed and r["metric"]==metric and r.get("contrast_name")==name)
                    vals.append(float(row["gain"]))
                ax.plot(range(3),vals,marker="o",label=name)
            ax.axhline(0,color="black",linewidth=.8);ax.set_xticks(range(3),[str(s) for s in SEEDS],rotation=20)
            ax.set_title(metric);ax.set_ylabel("gain after full seed-block removal")
        axs[-1].legend(fontsize=7,loc="best")
        fig.suptitle("True seed-block leave-one-out contrasts")
        fig.savefig(ROOT/"figures/SEED_BLOCK_LOO_CONTRASTS.png",dpi=160);plt.close(fig)

        fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        jrows=[r for r in ledger if r["scope"]=="macro"]
        for ax,metric in zip(axs,METRICS):
            for fam in fam_order:
                rr=[r for r in jrows if r["family"]==fam and r["metric"]==metric]
                ax.scatter([float(r["j_gap_member_mean_minus_ensemble"]) for r in rr],
                           [float(r["gain_vs_better_member"]) for r in rr],
                           s=24,alpha=.8,label=fam,color=colors[fam])
            ax.axhline(0,color="black",linewidth=.8);ax.axvline(0,color="black",linewidth=.8)
            ax.set_title(metric);ax.set_xlabel("J_gap (mean member loss - ensemble loss)")
            ax.set_ylabel("gain over better member")
        axs[-1].legend(fontsize=8)
        fig.suptitle("L1 cancellation and realized improvement over the better member")
        fig.savefig(ROOT/"figures/JGAP_VS_BETTER_MEMBER_GAIN.png",dpi=160);plt.close(fig)

        trace_files=[]
        for task_s,aid in selected.items():
            task=int(task_s);idx=canonical[aid]
            fig,axs=plt.subplots(1,2,figsize=(11,4),sharex=True,constrained_layout=True)
            for ax,axis in zip(axs,[0,14]):
                ax.plot(range(1,31),target[idx,:,axis],label="target GT",linewidth=2,color="black")
                ax.plot(range(1,31),node_arrays[("S",mix_seed)][idx,:,axis],label=f"S state {mix_seed}",alpha=.85)
                ax.plot(range(1,31),node_arrays[("V",mix_seed)][idx,:,axis],label=f"V fusion {mix_seed}",alpha=.85)
                ax.plot(range(1,31),mix_arr[idx,:,axis],label="MIX50 average",linewidth=2)
                ax.set_title(f"axis {axis}; native command values");ax.set_xlabel("forecast step");ax.grid(alpha=.2)
            axs[0].set_ylabel("native command value");axs[1].legend(fontsize=8)
            fig.suptitle(f"V5 locked case; task {task}; {aid}; seed {mix_seed}")
            name=f"TRAJECTORY_TASK_{task}.png";fig.savefig(ROOT/"figures"/name,dpi=160);plt.close(fig);trace_files.append(name)
        figures=["MIX50_DIRECT_REFERENCE_GAINS.png","FAMILY_MEANS_AND_PAIR_SPREAD.png",
                 "SEED_BLOCK_LOO_CONTRASTS.png","JGAP_VS_BETTER_MEMBER_GAIN.png"]+trace_files
        captions=[
            "MIX50_DIRECT_REFERENCE_GAINS.png: 三个固定V5同seed MIX50对S、V、BIAS_STATE、BIAS_VIDEO和Persistence的直接收益；前四项为锁定的source-group 95%区间，Persistence显示三对固定成员的点估计。",
            "FAMILY_MEANS_AND_PAIR_SPREAD.png: 六个固定family的全体pair宏平均loss、family均值和有限pair样本标准差；误差线表示pair离散度。",
            "SEED_BLOCK_LOO_CONTRASTS.png: 删除一个seed对应的S、A、V节点及所有相关pair后重新计算固定family contrast。",
            "JGAP_VS_BETTER_MEMBER_GAIN.png: 横轴为J_gap误差抵消量，纵轴为ensemble相对较好成员的实际gain。",
            "轨迹图复用V5预先锁定且不按结果选择的case ID，展示seed 202610131的native command index 0/14。"
        ]
        write_json(ROOT/"captions/FIGURE_CAPTIONS.zh-CN.json",{"captions":captions,"figure_files":figures})
        from PIL import Image,ImageDraw
        thumbs=[]
        for name in figures:
            im=Image.open(ROOT/"figures"/name).convert("RGB");im.thumbnail((520,260))
            canvas=Image.new("RGB",(540,300),"white");canvas.paste(im,((540-im.width)//2,28))
            ImageDraw.Draw(canvas).text((8,6),name,fill="black");thumbs.append(canvas)
        cols=2;nr=(len(thumbs)+1)//2
        contact=Image.new("RGB",(1080,nr*300),(238,238,238))
        for i,im in enumerate(thumbs):contact.paste(im,((i%2)*540,(i//2)*300))
        contact.save(ROOT/"qa/V6_FIGURE_CONTACT_SHEET.jpg",quality=92)
        if any((ROOT/"figures"/name).stat().st_size<1000 for name in figures):
            raise RuntimeError("one or more V6 figures were unexpectedly small")
        write_json(ROOT/"qa/FIGURE_BUILD_QA.json",{"schema_version":1,"status":"PASS","created_utc":utc(),
            "figure_files":figures,"figure_count":len(figures),"figure_data_from_saved_loss_tables_and_recovery_memory":True,
            "target_archive_reopens_for_figures":0,"representative_case_ids_source":"V5 protocol/FORMULA_LOCK.json",
            "representative_case_selection_uses_outcomes":False,"human_visual_review_status":"pending"})

        # Resource and exposure ledger; keep failed first pass and diagnostic re-read separate.
        resource_obj={"schema_version":1,"created_utc":utc(),"device":"CPU","physical_gpu_access":False,
            "new_neural_fits":0,"optimizer_updates":0,"new_calibration_fits":0,"new_encoder_extractions":0,"weight_search":0,
            "new_pair_bundles":33,"reused_v5_mix50_bundles":3,"pair_outputs_per_system":2,"equal_FLOPs_assumed":False,
            "cpu_composition_arithmetic_only_seconds":composition_arithmetic_seconds,"composition_timing_passes":1,
            "composition_cost_includes_source_file_loading":False,"composition_cost_includes_model_forward":False,
            "original_score_pass_new_pair_window_loss_evaluations":6105,
            "original_score_pass_v5_mix50_reconciliation_windows":555,
            "score_table_rows_preserved_and_reused":6660,"loss_recomputations_on_resume":0,
            "target_archive_load_operations_total":2,"target_archive_physical_read_operations_total":2,
            "target_array_rows_loaded_total":370,"unique_target_windows":185,
            "recovery_target_read_purpose":"elementwise L1 cancellation only; no scalar pair-score recomputation",
            "cancellation_diagnostic_pair_windows":6660,"cancellation_diagnostic_wall_seconds":l1_seconds,
            "prefix_archive_loads_total":2,"source_node_arrays_loads_initial":9,"mix50_arrays_loaded_initial":3,
            "source_node_arrays_loads_recovery":9,"pair_arrays_loaded_recovery":36,
            "v5_loss_rows_parsed_initial":7215,"v5_reference_rows_parsed_recovery":3885,
            "source_groups":47,"bootstrap_draws":DRAWS,"bootstrap_seed":BOOT_SEED,
            "prior_v5_archive_loads":4,"prior_v5_target_rows_loaded":740,"prior_v5_primary_loss_evaluations":3330,
            "prior_v5_l2_diagnostic_evaluations":2775,
            "prior_v5_legacy_counter":35523,"prior_v5_legacy_counter_name":"cumulative_dev_future_action_row_reads",
            "prior_v5_legacy_counter_unit_status":"unverified_as_same_grain; kept separate",
            "v2_accepted_per_job_child_seconds_by_seed":[
                {"seed":202610131,"S":0.767859,"V":1.190094,"S_plus_V_child_sum":1.957953},
                {"seed":202610132,"S":0.766805,"V":0.558181,"S_plus_V_child_sum":1.324986},
                {"seed":202610133,"S":0.542778,"V":0.526567,"S_plus_V_child_sum":1.069345}],
            "v2_accepted_per_job_child_seconds_mean":{"S":0.6924806666666666,"V":0.7582806666666667,"S_plus_V_child_sum":1.4507613333333333},
            "v2_latency_scope":"separate historical 185-window child job seconds; not integrated end-to-end latency",
            "A_program_inference_latency":"not present in accepted V5 inference-cost summary; unknown",
            "source_program_dependencies":{"S":"optimized state-only G-S","A":"V4 common RGB recipe with RGB mask at inference","V":"matched state+RGB fusion G-V"},
            "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "recovery_elapsed_wall_seconds":recovery_elapsed,"archive_read_wall_seconds":archive_read_seconds,
            "final_heldout_payload_access":0,"old_root_writes":0}
        write_json(ROOT/"results/RESOURCE_LEDGER.json",resource_obj)
        write_csv(ROOT/"results/RESOURCE_LEDGER.csv",[{"key":k,"value":json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v} for k,v in resource_obj.items()])
        write_json(ROOT/"results/RESULTS_SUMMARY.json",{"schema_version":1,"status":"analysis_complete_pending_visual_review_and_checksum",
            "study":"post_V5_equal_half_ensemble_controls_development","v6_root":str(ROOT),
            "formula_lock_sha256":formula_sha,"target_archive_loads_total":2,"additional_recovery_reads":1,
            "new_pair_window_loss_evaluations":6105,"mix50_reconciliation_window_evaluations":555,
            "reused_v5_mix50_loss_rows":555,"final_heldout_payload_access":0,"old_root_writes":0})
        owner.update({"status":"running","phase":"recovery_analysis_and_figures_complete_pending_visual_review",
            "updated_utc":utc(),"source_nodes_accepted":9,"pair_systems_accepted":36,"new_pair_bundles_accepted":33,
            "reused_v5_mix50_bundles":3,"target_archive_loads":2,"target_array_rows_loaded":370,
            "new_pair_window_loss_evaluations":6105,"pair_loss_recomputations_on_resume":0,
            "final_heldout_payload_access":0,"old_root_writes":0})
        write_json(owner_path,owner)
        log("recovery complete: preserved score pass reused; extra archive read restricted to L1 cancellation diagnostics")
    except Exception as exc:
        owner.update({"status":"technical_failure","phase":"resume_failed","updated_utc":utc(),
            "error_type":type(exc).__name__,"error_message":str(exc),"final_heldout_payload_access":0,"old_root_writes":0})
        write_json(owner_path,owner)
        with (ROOT/"qa/FAILED_ATTEMPTS.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"attempt":2,"created_utc":utc(),"error_type":type(exc).__name__,
                "error_message":str(exc),"phase":"resume_post_score_diagnostics"},ensure_ascii=False)+"\n")
        log(f"RESUME FAIL {type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()

if __name__=="__main__":main()


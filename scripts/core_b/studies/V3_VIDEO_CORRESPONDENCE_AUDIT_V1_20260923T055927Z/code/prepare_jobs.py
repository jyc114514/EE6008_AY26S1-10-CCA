#!/usr/bin/env python3
"""Prepare immutable watcher jobs for authorized fixed-checkpoint inference."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
EXP=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV=REPO/"reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
DELAY=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
PYTHON=PROJECT/"envs/jyc/bin/python"
WATCHER=REPO/"tools/gpu_lease_watcher.py"
RUNNER=EXP/"code/run_fixed_context.py"
CONDITIONS=8

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def readcsv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=content:raise FileExistsError(f"refusing to overwrite differing locked artifact: {path}")
        return
    tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(content,encoding="utf-8");os.replace(tmp,path)

def source_identity():
    registry=json.loads((EXP/"SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    if registry.get("status")!="LOCKED" or registry.get("final_test_access")!=0:raise RuntimeError("source registry is not locked or final boundary is not zero")
    manifest=readcsv(EXP/"MODEL_MANIFEST.csv")
    if len(manifest)!=6:raise RuntimeError(f"expected exactly six locked models, got {len(manifest)}")
    inputs=V3/"data/V3_MAIN_INPUTS.npz";cache=V3/"cache/V3_TEMPORAL_FEATURES.npy"
    input_hash=digest(inputs);cache_hash=digest(cache)
    normalizer_hash=digest(V3/"data/V3_NORMALIZERS.json");video_normalizer_hash=digest(V3/"data/V3_VIDEO_NORMALIZER.json")
    if cache_hash!=registry["feature_cache_sha256"]:raise RuntimeError("frozen temporal cache hash changed")
    rows=[]
    for r in manifest:
        family=r["family"];seed=int(r["seed"])
        if r["status"]!="PASS" or r["epoch"]!="40" or r["optimizer_steps"]!="11120" or r["input_hash_matches_current"]!="True" or r["cache_hash_matches_locked"]!="True" or r["final_test_access"]!="0":
            raise RuntimeError(f"locked model manifest failed: {family} seed {seed}")
        if r["input_sha256"]!=input_hash or r["cache_sha256"]!=cache_hash:raise RuntimeError(f"input/cache identity mismatch: {family} seed {seed}")
        checkpoint=Path(r["checkpoint_path"]).resolve()
        source_root=(SIV if family=="SIV" else DELAY).resolve()
        if not checkpoint.is_relative_to(source_root) or not checkpoint.is_file():raise RuntimeError(f"checkpoint outside approved root or missing: {checkpoint}")
        cp_hash=digest(checkpoint)
        if cp_hash!=r["checkpoint_sha256"]:raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        model_code=(SIV/"code/state_video_model.py") if family=="SIV" else (DELAY/"code/delay_model.py")
        model_hash=digest(model_code)
        if model_hash!=r["model_code_sha256"]:raise RuntimeError(f"model code hash mismatch: {model_code}")
        pred_root=SIV/"predictions" if family=="SIV" else DELAY/"predictions"
        predictions=[pred_root/f"SIV-ANCHOR-D-H60-S{seed}/primary.npz"] if family=="SIV" else [pred_root/f"ODR-ROBUST-D-H60-S{seed}/delay_{i}.npz" for i in range(4)]
        missing=[str(p) for p in predictions if not p.is_file()]
        if missing:raise RuntimeError(f"source replay prediction(s) missing: {missing}")
        rows.append({"family":family,"cell":r["cell"],"seed":seed,"checkpoint":str(checkpoint),"checkpoint_sha256":cp_hash,"model_code_sha256":model_hash,"source_prediction":str(predictions[0])})
    identity={"status":"PASS","checked_utc":datetime.now(timezone.utc).isoformat(),"input_path":str(inputs),"input_sha256":input_hash,"cache_path":str(cache),"cache_sha256":cache_hash,"normalizer_sha256":normalizer_hash,"video_normalizer_sha256":video_normalizer_hash,"locked_models":rows,"final_test_access":0,"training_started":False,"encoder_extraction_started":False}
    target=EXP/"SOURCE_IDENTITY_CHECK.json"
    if target.exists():
        previous=json.loads(target.read_text(encoding="utf-8"))
        stable_keys=("input_sha256","cache_sha256","normalizer_sha256","video_normalizer_sha256","locked_models","final_test_access","training_started","encoder_extraction_started")
        if previous.get("status")!="PASS" or any(previous.get(k)!=identity.get(k) for k in stable_keys):raise RuntimeError("source identity changed since its first lock")
        identity=previous
    else:write_json(target,identity)
    return manifest,identity

def context_rows():
    rows=readcsv(EXP/"EVAL_MATRIX_120.csv")
    contexts={}
    for r in rows:contexts.setdefault(r["context_id"],r)
    return rows,list(contexts.values())

def next_attempt(base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    i=1
    while (base/f"attempt_{i:02d}").exists():i+=1
    return base/f"attempt_{i:02d}"

def historical_peak(family):
    path=(SIV/"analysis/RESOURCE_COSTS.csv") if family=="SIV" else (DELAY/"analysis/RESOURCE_COSTS.csv")
    rows=readcsv(path)
    tag="SIV-ANCHOR-D-H60-" if family=="SIV" else "ODR-ROBUST-D-H60-"
    candidates=[int(r["peak_contract_mib"]) for r in rows if r["kind"]=="inference" and r["job_id"].startswith(tag) and r["status"]=="complete"]
    if not candidates:raise RuntimeError(f"no historical inference peak contract for {family}")
    return max(candidates)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("stage0","formal"),required=True);args=ap.parse_args()
    manifest,identity=source_identity();matrix,contexts=context_rows()
    if len(matrix)!=120 or len(contexts)!=15:raise RuntimeError(f"locked matrix mismatch: {len(matrix)} rows / {len(contexts)} contexts")
    stage0_contexts=[r for r in contexts if r["source_family"]=="SIV" or (r["source_family"]=="DELAY" and abs(float(r["delay_seconds"])-0.5)<1e-9)]
    selected=stage0_contexts if args.stage=="stage0" else contexts
    if args.stage=="stage0" and len(selected)!=6:raise RuntimeError(f"expected six Stage0 contexts, got {len(selected)}")
    stage0_by_key={(r["source_family"],int(r["model_seed"])):r for r in stage0_contexts}
    jobs=[]
    for ctx in selected:
        family=ctx["source_family"];seed=int(ctx["model_seed"]);context=ctx["context_id"]
        if args.stage=="stage0":
            out=next_attempt(EXP/f"runs/stage0/{context}")
            peak=historical_peak(family)
            suffix=out.name
        else:
            smoke=stage0_by_key[(family,seed)]
            stage0_path=EXP/f"runs/stage0/{smoke['context_id']}"
            accepted=sorted(stage0_path.glob("attempt_*/STAGE0_REPLAY.json"))
            if not accepted:raise RuntimeError(f"missing Stage0 result for {family} seed {seed}")
            records=[json.loads(p.read_text(encoding="utf-8")) for p in accepted]
            valid=[x for x in records if x.get("status")=="PASS" and x.get("allclose_at_1e-6") is True]
            if not valid:raise RuntimeError(f"no accepted Stage0 replay for {family} seed {seed}")
            peak=max(int(x["peak_memory_reserved_mib"]) for x in valid)
            if peak<=0:raise RuntimeError(f"invalid Stage0 peak memory {family} seed {seed}")
            out=next_attempt(EXP/f"runs/formal/{context}")
            suffix=out.name
        job_id=f"VCA-{args.stage.upper()}-{family}-S{seed}-{context[-8:]}-{suffix.upper()}"
        cmd=[str(PYTHON),str(RUNNER),"--context-id",context,"--output-dir",str(out)]
        if args.stage=="stage0":cmd.append("--smoke")
        log=EXP/f"logs/child_{job_id}.log"
        job={"job_id":job_id,"protocol_id":"V3_VIDEO_CORRESPONDENCE_AUDIT_V1_"+args.stage.upper(),"output_root":str(EXP/f"predictions/{context}" if args.stage=="formal" else out),"command":cmd,"resource_estimate":{"estimate_type":"historical_peak_contract" if args.stage=="stage0" else "measured_stage0_peak_reserved","peak_memory_mib":peak},"log_path":str(log),"max_oom_requeues":1,"environment":{"MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1"},"final_test_access":0,"optimizer_steps":0,"encoder_forward":0,"context_id":context,"family":family,"model_seed":seed}
        job_path=EXP/f"configs/jobs/{job_id}.json"
        write_json(job_path,job);jobs.append({"job_id":job_id,"job_json":str(job_path),"context_id":context,"family":family,"model_seed":seed,"output_dir":str(out),"resource_peak_mib":peak})
    queue={"status":"LOCKED","stage":args.stage,"protocol_id":"V3_VIDEO_CORRESPONDENCE_AUDIT_V1_"+args.stage.upper(),"jobs":jobs,"job_count":len(jobs),"max_project_gpu_children":2,"dynamic_all_visible":True,"utilization_is_ranking_only":True,"gpu_hours_upper_bound":6,"additional_disk_gib_upper_bound":12,"training_started":False,"encoder_extraction_started":False,"final_test_access":0,"source_identity_check":str(EXP/"SOURCE_IDENTITY_CHECK.json")}
    manifest_path=EXP/f"configs/{args.stage.upper()}_JOBS.json";write_json(manifest_path,queue)
    print(json.dumps({"status":"PASS","stage":args.stage,"job_count":len(jobs),"manifest":str(manifest_path),"source_identity":"PASS","context_ids":[j["context_id"] for j in jobs]},sort_keys=True))
if __name__=="__main__":raise SystemExit(main())

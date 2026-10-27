#!/usr/bin/env python3
"""Run registered watcher jobs with a bounded two-child project queue."""
from __future__ import annotations
import os
import argparse,json,os,subprocess,sys,time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
EXP=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
PYTHON=PROJECT/"envs/jyc/bin/python"
WATCHER=REPO/"tools/gpu_lease_watcher.py"
LEASE_DIR=REPO/"reports/locks/gpu_lease"

def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--poll-seconds",type=float,default=1.0);ap.add_argument("--max-wait-seconds",type=int,default=21600);args=ap.parse_args()
    manifest=read_json(args.manifest)
    if manifest.get("status")!="LOCKED" or manifest.get("final_test_access")!=0 or manifest.get("max_project_gpu_children")!=2 or not manifest.get("dynamic_all_visible"):raise RuntimeError("queue manifest failed authorization/resource contract")
    stage=manifest["stage"];jobs=manifest["jobs"]
    if len(jobs)!=manifest["job_count"] or len({j["job_id"] for j in jobs})!=len(jobs):raise RuntimeError("queue manifest count/identity mismatch")
    pending=deque(jobs);active={};records=[];max_active=0;started=utc()
    status_path=EXP/f"logs/{stage.upper()}_QUEUE_STATUS.json"
    queue={"status":"running","stage":stage,"manifest":str(args.manifest),"started_utc":started,"job_count":len(jobs),"max_project_gpu_children":2,"max_active_watchers":0,"jobs":[],"final_test_access":0}
    atomic_json(status_path,queue)
    while pending or active:
        while pending and len(active)<2:
            item=pending.popleft();job_path=Path(item["job_json"])
            job=read_json(job_path)
            if job["job_id"]!=item["job_id"]:raise RuntimeError(f"job identity mismatch: {job_path}")
            watcher_status=EXP/f"status/{job['job_id']}.json"
            controller_log=EXP/f"logs/controller_{job['job_id']}.log"
            controller_log.parent.mkdir(parents=True,exist_ok=True)
            command=[str(PYTHON),str(WATCHER),"--job-json",str(job_path),"--lease-dir",str(LEASE_DIR),"--status-path",str(watcher_status),"--poll-seconds","20","--max-wait-seconds",str(args.max_wait_seconds),"--max-oom-requeues","1"]
            log_handle=controller_log.open("w",encoding="utf-8")
            proc=subprocess.Popen(command,cwd=str(REPO),stdout=log_handle,stderr=subprocess.STDOUT,text=True)
            active[proc.pid]={"proc":proc,"item":item,"status_path":str(watcher_status),"log_path":str(controller_log),"log_handle":log_handle,"controller_pid":proc.pid,"controller_started_utc":utc(),"monotonic_start":time.monotonic()}
            max_active=max(max_active,len(active));queue["max_active_watchers"]=max_active
            queue["jobs"].append({"job_id":job["job_id"],"state":"watcher_started","controller_pid":proc.pid,"status_path":str(watcher_status),"log_path":str(controller_log),"started_utc":active[proc.pid]["controller_started_utc"]})
            atomic_json(status_path,queue)
        finished=[]
        for pid,entry in list(active.items()):
            code=entry["proc"].poll()
            if code is None:continue
            entry["log_handle"].close()
            child_status={}
            status_file=Path(entry["status_path"])
            if status_file.is_file():
                try:child_status=read_json(status_file)
                except Exception as exc:child_status={"status":"status_read_error","error":f"{type(exc).__name__}: {exc}"}
            record={"job_id":entry["item"]["job_id"],"context_id":entry["item"]["context_id"],"family":entry["item"]["family"],"model_seed":entry["item"]["model_seed"],"watcher_exit_code":code,"watcher_status":child_status.get("status","missing"),"physical_gpu":child_status.get("physical_gpu",child_status.get("selected_gpu")),"watcher_status_path":entry["status_path"],"controller_log":entry["log_path"],"controller_pid":pid,"controller_started_utc":entry["controller_started_utc"],"completed_utc":utc(),"non_child_process_signaled":child_status.get("non_child_process_signaled",False)}
            if child_status.get("launch_utc") and child_status.get("completed_utc"):
                try:
                    start=datetime.fromisoformat(str(child_status["launch_utc"]).replace("Z","+00:00"))
                    end=datetime.fromisoformat(str(child_status["completed_utc"]).replace("Z","+00:00"))
                    record["gpu_seconds"]=max(0.0,(end-start).total_seconds())
                except Exception:record["gpu_seconds"]=None
            records.append(record);queue["jobs"]=[r for r in queue["jobs"] if r.get("job_id")!=record["job_id"]]+[{**record,"state":"finished"}]
            atomic_json(status_path,queue);finished.append(pid)
        for pid in finished:active.pop(pid)
        if pending or active:time.sleep(max(0.1,args.poll_seconds))
    failures=[r for r in records if r["watcher_exit_code"]!=0 or r["watcher_status"]!="complete" or r["non_child_process_signaled"]]
    queue.update({"status":"PASS" if not failures else "FAIL","completed_utc":utc(),"completed_count":len(records),"failed_job_ids":[r["job_id"] for r in failures],"max_active_watchers":max_active,"actual_gpu_ids":sorted({int(r["physical_gpu"]) for r in records if r["physical_gpu"] is not None}),"gpu_hours_used":sum((r.get("gpu_seconds") or 0.0) for r in records)/3600.0,"job_records":records,"final_test_access":0,"optimizer_steps_total":0,"encoder_forward_total":0})
    atomic_json(status_path,queue)
    print(json.dumps({"status":queue["status"],"stage":stage,"completed":len(records),"failed_job_ids":queue["failed_job_ids"],"max_active_watchers":max_active,"actual_gpu_ids":queue["actual_gpu_ids"],"gpu_hours_used":queue["gpu_hours_used"],"status_path":str(status_path)},sort_keys=True))
    return 0 if not failures else 1

if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""V4 train pilot, full fixed matrix, inference, and nonempty pre-target sealing."""
from __future__ import annotations
import os
import csv,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python'
SESSION='ee6008-v4-20261014t083805z'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def owner(**updates):
    p=ROOT/'coordination/OWNER_STATE.json';d=json.loads(p.read_text());d.update(updates);d['updated_utc']=utc();d['owner_pid']=os.getpid();d['tmux_session']=SESSION;write(p,d)
def run_phase(queue:str)->None:
    q=ROOT/'coordination/queues'/queue
    print(f'[{utc()}] start {q.name}',flush=True)
    res=subprocess.run([str(PYTHON),str(ROOT/'code/run_v4_queue.py'),'--queue',str(q)],cwd=str(PROJECT),check=False)
    if res.returncode!=0:raise RuntimeError(f'queue failed: {q.name} rc={res.returncode}')
    print(f'[{utc()}] complete {q.name}',flush=True)
def verify(job_id:str)->None:
    p=ROOT/'configs/fits'/f'{job_id}.json'
    res=subprocess.run([str(PYTHON),str(ROOT/'code/verify_v4_model.py'),'--job-json',str(p)],cwd=str(PROJECT),check=False)
    if res.returncode!=0:raise RuntimeError(f'fresh CPU reload failed for {job_id}')
def resource_ledger():
    rows=[]
    qpaths=[ROOT/'coordination/queues/TRAIN_PILOT.status.json',ROOT/'coordination/queues/TRAIN_REMAINDER.status.json',ROOT/'coordination/queues/INFERENCE.status.json']
    for qp in qpaths:
        if not qp.exists():continue
        q=json.loads(qp.read_text())
        for r in q['results']:
            marker=Path(r['marker']) if r.get('marker') else None
            data=json.loads(marker.read_text()) if marker and marker.exists() else {}
            rows.append({'job_id':r['job_id'],'phase':q['phase'],'status':r['status'],'physical_gpu':r.get('selected_gpu') or data.get('physical_gpu',''),'child_pid':r.get('child_pid',''),'peak_allocated_mib':data.get('peak_memory_allocated_mib',''),'peak_reserved_mib':data.get('peak_memory_reserved_mib',''),'elapsed_seconds':data.get('elapsed_seconds',''),'lease_released':r.get('lease_released',''),'watcher_log':r.get('console_log','')})
    p=ROOT/'results/RESOURCE_LEDGER.csv';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        fields=list(rows[0]) if rows else ['job_id','phase','status','physical_gpu','child_pid','peak_allocated_mib','peak_reserved_mib','elapsed_seconds','lease_released','watcher_log']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return rows
def main():
    started=utc();owner(status='running',phase='GPU_pilot',active_jobs=0,started_utc=started)
    try:
        run_phase('TRAIN_PILOT.json')
        pilot='v4_fit_C_STATE_NEIGHBOR_REASSIGNED_s202610131'
        verify(pilot)
        write(ROOT/'qa/PILOT_ACCEPTANCE.json',{'schema_version':1,'status':'PASS','pilot_fit':pilot,'fresh_process_reload':'PASS','remainder_auto_expanded':True,'fits_planned_total':6,'old_B_refits':0,'final_heldout_payload_access':0,'completed_utc':utc()})
        print(f'[{utc()}] pilot fit and fresh CPU reload PASS; automatically expanding the five remaining fits',flush=True)
        owner(phase='GPU_train_remainder',active_jobs=0)
        run_phase('TRAIN_REMAINDER.json')
        all_fit_jobs=[f'v4_fit_{program}_s{seed}' for program in ['A_STATE_COMMON','C_STATE_NEIGHBOR_REASSIGNED'] for seed in [202610131,202610132,202610133]]
        for jid in all_fit_jobs:
            verify(jid)
        owner(phase='GPU_prediction_matrix',active_jobs=0)
        run_phase('INFERENCE.json')
        owner(phase='prediction_seal',active_jobs=0)
        res=subprocess.run([str(PYTHON),str(ROOT/'code/seal_v4_predictions.py')],cwd=str(PROJECT),check=False,stdout=sys.stdout,stderr=sys.stderr)
        if res.returncode!=0:raise RuntimeError('V4 pre-target prediction seal failed')
        rows=resource_ledger()
        owner(phase='sealed_waiting_for_CPU_scoring',active_jobs=0,prediction_seal_status='PASS',prediction_jobs_accepted=24,new_model_fits_accepted=6,completed_gpu_utc=utc())
        write(ROOT/'coordination/GPU_EXECUTION.json',{'schema_version':1,'status':'GPU_PHASE_COMPLETE','started_utc':started,'completed_utc':utc(),'tmux_session':SESSION,'main_pid':os.getpid(),'pilot_fit':pilot,'new_model_fits':6,'prediction_jobs':24,'reused_B_MATCHED_jobs':3,'resource_ledger_rows':len(rows),'shared_watcher_sha256':json.loads((ROOT/'inventory/WATCHER_POLICY.json').read_text())['source_sha256'],'final_heldout_payload_access':0})
        print(f'[{utc()}] V4 GPU matrix and nonempty prediction seal complete; dev targets remain unopened',flush=True)
        owner(phase='sealed_waiting_for_CPU_scoring',active_jobs=0)
        return 0
    except Exception as e:
        owner(status='partial',phase='GPU_pipeline_failure',active_jobs=0,last_error=f'{type(e).__name__}: {e}',failed_utc=utc())
        write(ROOT/'coordination/GPU_PIPELINE_FAILURE.json',{'schema_version':1,'status':'partial','error_type':type(e).__name__,'error':str(e),'failed_utc':utc(),'final_heldout_payload_access':0})
        print(f'[{utc()}] GPU pipeline failed: {type(e).__name__}: {e}',flush=True)
        return 2
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Reconcile future watcher peaks against the first accepted V7 training slot."""
from __future__ import annotations
import json,math,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(p:Path,obj)->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def main()->int:
 seed=202610131;slot=ROOT/'model_only/ST_SV'/f'seed{seed}'/'attempt_01';manifest=json.loads((slot/'MODEL_MANIFEST.json').read_text())
 if manifest.get('status')!='PASS':raise RuntimeError('pilot model is not accepted; refusing to register memory')
 actual=math.ceil(float(manifest['peak_memory_reserved_mib']));plan_path=ROOT/'coordination/JOB_PLAN.json';plan=json.loads(plan_path.read_text())
 previous=int(plan['registered_peak_mib']);updated=max(previous,actual);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backups=[];changed=0
 if updated>previous:
  for name in ('FITS_REMAINDER.json','TRAIN_INFERENCE.json','DEV_INFERENCE.json','LATENCY.json'):
   p=ROOT/'coordination/queues'/name
   if not p.is_file():continue
   q=json.loads(p.read_text());backup=p.with_name(f'{p.name}.pre-v7-pilot-memory-{stamp}.bak');shutil.copy2(p,backup);backups.append(str(backup))
   for job in q['jobs']:
    job.setdefault('resource_estimate',{})['measured_peak_mib']=updated
    job['resource_estimate']['basis']=f'V7 pilot measured peak reserved {actual} MiB; admission safety margin computed by shared watcher'
    changed+=1
   write(p,q)
 plan['registered_peak_mib']=updated;plan['registered_free_memory_admission_mib']=updated+max(512,math.ceil(.1*updated));write(plan_path,plan)
 result={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
         'pilot_job_id':manifest['job_id'],'pilot_peak_allocated_mib':float(manifest['peak_memory_allocated_mib']),
         'pilot_peak_reserved_mib':float(manifest['peak_memory_reserved_mib']),'pilot_reserved_ceil_mib':actual,
         'previous_registered_peak_mib':previous,'new_registered_peak_mib':updated,'future_jobs_updated':changed,
         'queue_backups':backups,'admission_margin_policy':'max(512 MiB, ceil(10 percent of registered peak))',
         'final_heldout_payload_access':0}
 write(ROOT/'qa/PILOT_RESOURCE_UPDATE.json',result)
 print(json.dumps({'status':'PASS','pilot_peak_reserved_mib':manifest['peak_memory_reserved_mib'],'registered_peak_mib':updated,'future_jobs_updated':changed},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

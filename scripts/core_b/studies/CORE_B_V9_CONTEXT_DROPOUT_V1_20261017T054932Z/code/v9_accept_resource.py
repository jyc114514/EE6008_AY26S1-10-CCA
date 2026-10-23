#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 stage=ROOT/'tmp/stage0/retry02/attempt_01/STAGE0_COMPLETE.json'
 qa=ROOT/'qa/MASK_SENTINEL_AND_RNG.json'
 if not stage.is_file() or not qa.is_file():raise RuntimeError('successful Stage0 receipt or sentinel missing')
 s=rj(stage);q=rj(qa)
 if s.get('status')!='PASS' or int(s.get('updates',-1))!=2 or q.get('status')!='PASS':raise RuntimeError('Stage0 acceptance failed')
 status=ROOT/'coordination/watcher_status'/f"{s['job_id']}.json"
 w=rj(status)
 if w.get('status')!='complete' or int(w.get('exit_code',-1))!=0:raise RuntimeError('Stage0 watcher did not complete cleanly')
 peak=math.ceil(float(s['peak_memory_reserved_mib']))
 if peak<=0:raise RuntimeError('invalid measured Stage0 peak')
 lock={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'measured_peak_mib':peak,'peak_memory_reserved_mib':s['peak_memory_reserved_mib'],
  'peak_memory_allocated_mib':s['peak_memory_allocated_mib'],'stage0_receipt_path':str(stage),'stage0_receipt_sha256':sha(stage),
  'stage0_watcher_status_path':str(status),'stage0_watcher_status_sha256':sha(status),
  'stage0_completed_utc_from_watcher':w.get('completed_utc'),'stage0_job_id':s['job_id'],
  'rule':'register measured Stage0 same-runner peak; watcher margin is max(512 MiB, ceil(0.10*peak))',
  'required_free_memory_mib':peak+max(512,math.ceil(0.10*peak)),
  'physical_gpu':s['physical_gpu'],'cuda_visible_devices':s['cuda_visible_devices'],'logical_device':s['logical_device'],
  'training_code_lock_sha256':sha(ROOT/'protocol/CODE_LOCK.json'),
  'final_heldout_payload_access':0}
 out=ROOT/'protocol/RESOURCE_LOCK.json'
 if out.exists():raise RuntimeError('RESOURCE_LOCK already exists')
 tmp=out.with_name('.'+out.name+'.part');tmp.write_text(json.dumps(lock,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,out)
 print(json.dumps(lock,sort_keys=True))
if __name__=='__main__':main()


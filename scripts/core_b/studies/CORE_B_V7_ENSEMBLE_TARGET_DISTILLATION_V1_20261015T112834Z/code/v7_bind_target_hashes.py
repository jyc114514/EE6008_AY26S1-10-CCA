#!/usr/bin/env python3
"""Bind each formal fit config to its accepted target array after construction."""
from __future__ import annotations
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,obj)->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def main()->int:
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');mix=ROOT/'protocol/TARGET_MIX_LOCK.json';mix_sha=sha(mix)
 bound=[]
 for queue_name in ('PILOT.json','FITS_REMAINDER.json'):
  queue_path=ROOT/'coordination/queues'/queue_name;queue=json.loads(queue_path.read_text(encoding='utf-8'))
  for watcher_job in queue['jobs']:
   cfg_path=Path(watcher_job['command'][-1]);cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
   target_manifest=Path(cfg['teacher_target_manifest_path']);manifest=json.loads(target_manifest.read_text(encoding='utf-8'))
   target_path=Path(manifest['target_path']);target_sha=sha(target_path)
   if manifest.get('status')!='PASS' or manifest.get('program_id')!=cfg['program_id'] or int(manifest.get('seed',-1))!=int(cfg['seed']) or target_sha!=manifest.get('target_sha256'):
    raise RuntimeError(f'target manifest/array mismatch: {cfg_path}')
   if cfg.get('teacher_target_sha256') not in {'filled_after_target_build',target_sha}:
    raise RuntimeError(f'existing bound target hash differs; preserving config: {cfg_path}')
   if sha(mix)!=mix_sha:raise RuntimeError('target mix lock changed during config binding')
   if cfg.get('teacher_target_sha256')!=target_sha or cfg.get('target_mix_lock_sha256')!=mix_sha:
    backup=cfg_path.with_name(f'{cfg_path.name}.pre-target-sha-{stamp}.bak')
    if not backup.exists():shutil.copy2(cfg_path,backup)
    cfg['teacher_target_sha256']=target_sha;cfg['target_mix_lock_sha256']=mix_sha
    write(cfg_path,cfg)
   bound.append({'job_id':cfg['job_id'],'config_path':str(cfg_path),'config_sha256':sha(cfg_path),'target_path':str(target_path),'target_sha256':target_sha,'target_mix_lock_sha256':mix_sha})
 plan_path=ROOT/'coordination/JOB_PLAN.json';plan=json.loads(plan_path.read_text(encoding='utf-8'));plan['bound_fit_configs']=len(bound);plan['target_mix_lock_sha256']=mix_sha;plan['config_hashes']={r['job_id']:r['config_sha256'] for r in bound};write(plan_path,plan)
 write(ROOT/'qa/CONFIG_BINDING.json',{'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'fit_configs_bound':len(bound),'target_mix_lock_sha256':mix_sha,'configs':bound,'dev_target_opened':False,'final_heldout_payload_access':0})
 print(json.dumps({'status':'PASS','fit_configs_bound':len(bound),'target_mix_lock_sha256':mix_sha},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

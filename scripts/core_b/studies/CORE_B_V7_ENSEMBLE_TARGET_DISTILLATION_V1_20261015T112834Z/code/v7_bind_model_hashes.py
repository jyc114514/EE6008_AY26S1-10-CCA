#!/usr/bin/env python3
"""Bind standalone inference configs to their accepted model and manifest hashes."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV');SEEDS=(202610131,202610132,202610133)
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,obj)->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--program');ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');args=ap.parse_args()
 if not args.all and not (args.program and args.seed):raise SystemExit('choose --all or --program NAME --seed SEED')
 pairs=[(p,s) for p in PROGRAMS for s in SEEDS] if args.all else [(args.program,args.seed)]
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');rows=[]
 for program,seed in pairs:
  if program not in PROGRAMS or seed not in SEEDS:raise RuntimeError(f'invalid V7 model key: {program}/{seed}')
  slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';model=slot/'model.safetensors';manifest_path=slot/'MODEL_MANIFEST.json'
  if not model.is_file() or not manifest_path.is_file():raise FileNotFoundError(f'student model not complete: {program}/{seed}')
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'));model_sha=sha(model);manifest_sha=sha(manifest_path)
  if manifest.get('status')!='PASS' or manifest.get('model_file_sha256')!=model_sha or manifest.get('student_inference_requires_teacher') is not False:raise RuntimeError(f'student model manifest/hash is not accepted: {program}/{seed}')
  for dataset in ('train','dev'):
   cfg_path=ROOT/'configs/fits'/f'v7_pred_{dataset}_{program}_s{seed}.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
   if Path(cfg['model_path']).resolve()!=model.resolve() or Path(cfg['model_manifest_path']).resolve()!=manifest_path.resolve():raise RuntimeError(f'inference config model path mismatch: {cfg_path}')
   forbidden={'teacher_target_path','teacher_target_rows_path','teacher_target_manifest_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path'}
   if forbidden.intersection(cfg):raise RuntimeError(f'inference config contains a forbidden path: {cfg_path}')
   if cfg.get('model_sha256')!=model_sha or cfg.get('model_manifest_sha256')!=manifest_sha:
    backup=cfg_path.with_name(f'{cfg_path.name}.pre-model-hash-{stamp}.bak')
    if not backup.exists():shutil.copy2(cfg_path,backup)
    cfg['model_sha256']=model_sha;cfg['model_manifest_sha256']=manifest_sha
    write(cfg_path,cfg)
   rows.append({'job_id':cfg['job_id'],'program_id':program,'seed':seed,'dataset_role':dataset,'config_path':str(cfg_path),'config_sha256':sha(cfg_path),
                'model_path':str(model),'model_sha256':model_sha,'model_manifest_path':str(manifest_path),'model_manifest_sha256':manifest_sha,
                'teacher_or_GT_paths_in_config':False})
 plan_path=ROOT/'coordination/JOB_PLAN.json';plan=json.loads(plan_path.read_text());bindings=plan.get('model_hash_bindings',{})
 for r in rows:bindings[r['job_id']]=r['config_sha256']
 plan['model_hash_bindings']=bindings;write(plan_path,plan)
 receipt={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'configs_bound':len(rows),'bindings':rows,'dev_target_opened':False}
 write(ROOT/'qa/MODEL_HASH_BINDING.json',receipt)
 print(json.dumps({'status':'PASS','configs_bound':len(rows),'program_seed_pairs':len(pairs),'dev_target_opened':False},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""CPU-only fresh INIT safe-model save/reload and locked-target preflight."""
from __future__ import annotations
import os
import csv,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
sys.path.insert(0,str(ROOT/'code'))
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
SEEDS=(202610131,202610132,202610133)

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):return json.loads(p.read_text())
def writej(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 role_map=readj(V7/'protocol/TEACHER_ROLE_MAP.json');out=ROOT/'qa/FRESH_INIT_ROUNDTRIP';out.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in SEEDS:
  source=role_map['sources'][str(seed)]['V_MATCHED']; init_path=Path(source['init_path'])
  model=V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None)
  missing,unexpected=model.load_state_dict(load_file(str(init_path),device='cpu'),strict=True)
  if missing or unexpected:raise RuntimeError(f'fresh INIT strict load mismatch {seed}: {missing}/{unexpected}')
  before=state_hash(model)
  if before!=source['init_state_hash']:raise RuntimeError(f'fresh INIT hash mismatch {seed}')
  qa_path=out/f'INIT_seed{seed}.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(qa_path))
  reloaded=V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None)
  missing,unexpected=reloaded.load_state_dict(load_file(str(qa_path),device='cpu'),strict=True)
  if missing or unexpected:raise RuntimeError(f'saved fresh INIT strict reload mismatch {seed}')
  after=state_hash(reloaded)
  exact=all(torch.equal(model.state_dict()[k],reloaded.state_dict()[k]) for k in model.state_dict())
  if not exact or before!=after:raise RuntimeError(f'fresh INIT safe-model roundtrip differs {seed}')
  rows.append({'seed':seed,'source_init_path':str(init_path),'source_init_sha256':sha(init_path),'source_init_state_hash':before,'qa_safe_model_path':str(qa_path),'qa_safe_model_sha256':sha(qa_path),'reloaded_state_hash':after,'strict_load':True,'tensor_values_exact':True,'parameter_count':parameter_count(reloaded),'optimizer_created':False,'cuda_accessed':False})
 # All fixed target files and row manifests are sealed before training.
 target_checks=[]
 role=readj(ROOT/'protocol/TARGET_ROLE_MAP.json')
 if role.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or len(role.get('targets',[]))!=9:raise RuntimeError('target role map does not match locked B matrix')
 for entry in role['targets']:
  path=Path(entry['target_path']); manifest=readj(Path(entry['manifest_path']))
  if sha(path)!=entry['target_sha256'] or manifest.get('target_sha256')!=entry['target_sha256'] or manifest.get('status')!='PASS':raise RuntimeError(f'target bank hash/schema not sealed: {entry}')
  target_checks.append({'program_id':entry['program_id'],'seed':entry['seed'],'sha256':entry['target_sha256'],'rows':manifest['row_count'],'target_mode':entry['target_mode']})
 result={'schema_version':1,'status':'PASS','created_utc':utc(),'cpu_only':True,'fresh_init_seed_count':len(rows),'fresh_init_rows':rows,
  'target_bank_count':len(target_checks),'target_banks':target_checks,'training_candidates_started':0,'dev_target_archive_opened':False,'final_heldout_payload_access':0}
 writej(ROOT/'qa/FRESH_INIT_SAVE_RELOAD_QA.json',result)
 with (ROOT/'qa/FRESH_INIT_ROUNDTRIP.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'status':'PASS','fresh_init_seeds':len(rows),'target_banks':len(target_checks),'cuda_accessed':False},sort_keys=True))
if __name__=='__main__':main()

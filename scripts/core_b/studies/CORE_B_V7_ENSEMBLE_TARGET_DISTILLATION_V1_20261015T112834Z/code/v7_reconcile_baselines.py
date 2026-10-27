#!/usr/bin/env python3
"""Check accepted A/V hard-only support against the V7 common student recipe."""
from __future__ import annotations
import os
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z';V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
SEEDS=(202610131,202610132,202610133)
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read_manifest(root:Path,role:str,seed:int)->tuple[Path,dict]:
 if role=='A':p=root/'model_only/formal/A_STATE_COMMON'/f'seed{seed}'/'attempt_01'/'MODEL_MANIFEST.json'
 else:p=V2/'model_only/formal'/f'v2_formal_G_V_lr0p0005_s{seed}'/'MODEL_MANIFEST.json'
 d=json.loads(p.read_text(encoding='utf-8'));return p,d
def main()->int:
 v4lock=json.loads((V4/'protocol/MODEL_RECIPE_LOCK.json').read_text(encoding='utf-8'))['recipe'];resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'))
 v2eq=V2/'code/locked_source/d3_eq_contract.py';v4eq=V4/'code/locked_source/d3_eq_contract.py';v7eq=ROOT/'code/locked_source/d3_eq_contract.py'
 v2model=V2/'code/v2_model.py';v4model=V4/'code/v2_model.py';v7model=ROOT/'code/v2_model.py'
 v2scheduler=V2/'code/v2_scheduler.py';v4scheduler=V4/'code/v2_scheduler.py';v7scheduler=ROOT/'code/v2_scheduler.py'
 code_hashes={name:{'V2':sha(a),'V4':sha(b),'V7_copy':sha(c),'identical':sha(a)==sha(b)==sha(c)} for name,a,b,c in [('sampler_loss_contract',v2eq,v4eq,v7eq),('model',v2model,v4model,v7model),('scheduler',v2scheduler,v4scheduler,v7scheduler)]}
 if not all(v['identical'] for v in code_hashes.values()):raise RuntimeError(f'V2/V4/V7 accepted source components differ: {code_hashes}')
 rows=[]
 for seed in SEEDS:
  init_path=V2/'model_only/INIT'/f'seed_{seed}'/'shared_5task_base.safetensors'
  for role in ('A','V'):
   root=V4 if role=='A' else V2;path,d=read_manifest(root,role,seed)
   expected_mode='STATE-CTRL' if role=='A' else 'TS11-D'
   checks={'status':d.get('status')=='PASS','mode':d.get('video_mode')==expected_mode,'lr':float(d.get('lr',-1))==0.0005,
           'epochs':int(d.get('epochs',-1))==160,'effective_batch':int(d.get('effective_batch',-1))==64,
           'microbatch':int(d.get('microbatch_size',-1))==64,'accumulation':int(d.get('accumulation_steps',-1))==1,
           'optimizer_steps':int(d.get('optimizer_steps',-1))==3200,'steps_per_epoch':int(d.get('steps_per_epoch',-1))==20,
           'scheduler_tmax':int(d.get('scheduler',{}).get('T_max',-1))==3200,'scheduler_eta':float(d.get('scheduler',{}).get('eta_min',-1))==1e-5,
           'adamw':d.get('optimizer',{}).get('name')=='AdamW' and d.get('optimizer',{}).get('betas')==[0.9,0.999] and float(d.get('optimizer',{}).get('eps',-1))==1e-8 and float(d.get('optimizer',{}).get('weight_decay',-1))==1e-4,
           'fp32_no_amp_compile':d.get('dtype')=='float32' and d.get('amp') is False and d.get('compile') is False,
           'train_archive':d.get('train_selected_npz_sha256')==resolved['train_npz_sha256'],'normalizer':d.get('normalizer_sha256')==resolved['source_formal_normalizer_sha256'],
           'fresh_init':Path(d['init_path']).resolve()==init_path.resolve() and d.get('init_state_hash')==json.loads((V4/'qa/COMMON_INIT_AND_RECIPE.json').read_text())['fit_receipts'][SEEDS.index(seed)]['init_state_hash']}
   if not all(checks.values()):raise RuntimeError(f'{role} hard-only seed {seed} mismatches locked V7 recipe: {checks}')
   rows.append({'role':'A_hard' if role=='A' else 'V_hard','seed':seed,'manifest_path':str(path),'manifest_sha256':sha(path),'model_path':d['model_file'],'model_sha256':d['model_file_sha256'],'checks':checks})
 result={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
         'hard_only_models_reused':6,'additional_hard_only_baseline_fits_required':0,'A_state_common_reused':True,'V_matched_reused':True,
         'recipe':v4lock,'source_code_hashes':code_hashes,'models':rows,'reason':'accepted V4 A and V2 V share exact train data, INIT per seed, normalizer, model/sampler/loss/scheduler source hashes, batch, LR, epoch, optimizer and update schedule with locked student hard-only references'}
 (ROOT/'qa/HARD_ONLY_BASELINE_RECONCILIATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','hard_only_fits_required':0,'models_verified':len(rows),'source_components_identical':True},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[1]
SEEDS=(20260950,20260951,20260952)
N_BOOT=10000;BOOT_SEED=20261007

def rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main()->int:
 path=EXP/'analysis/P2_GAIN_BY_ANCHOR.csv'
 if not path.is_file():raise RuntimeError('H60 gate requires accepted H30 P2 interactions')
 values=rows(path);analysis=json.loads((EXP/'protocol/ANALYSIS_LOCKED.json').read_text());delta=float(analysis['display_delta']['value'])
 groups=sorted({r['source_group_id'] for r in values});group_episode={}
 for r in values:
  g=r['source_group_id'];e=int(r['episode_index'])
  if g in group_episode and group_episode[g]!=e:raise RuntimeError('nested source group requires group-episode bootstrap implementation')
  group_episode[g]=e
 rng=np.random.default_rng(BOOT_SEED);checks=[]
 for model in ('F-POOL-D','ALT-POOL-R3D18'):
  for diag,criterion in [('supported_high_A',lambda r:r['A_stratum']=='high' and bool(r['A_supported'])),('large_M',lambda r:r['M_stratum']=='large')]:
   by_seed={};task_by_group={}
   for seed in SEEDS:
    selected=[r for r in values if r['model']==model and int(r['seed'])==seed and criterion(r)]
    group_values={};
    for g in groups:
     local=[float(r['gain_vs_history']) for r in selected if r['source_group_id']==g]
     if local:group_values[g]=float(np.mean(local))
    by_seed[seed]=group_values
    for r in selected:task_by_group[r['source_group_id']]=int(r['task'])
   common=sorted(set.intersection(*(set(by_seed[s]) for s in SEEDS))) if all(by_seed[s] for s in SEEDS) else []
   n_groups=len(common);n_tasks=len({task_by_group[g] for g in common})
   if common:
    seed_means={str(s):float(np.mean([by_seed[s][g] for g in common])) for s in SEEDS}
    per_group=np.asarray([np.mean([by_seed[s][g] for s in SEEDS]) for g in common],dtype=np.float64)
    draws=rng.integers(0,len(per_group),size=(N_BOOT,len(per_group)),dtype=np.int32)
    boot=per_group[draws].mean(axis=1);pooled_mean=float(per_group.mean());lo=float(np.quantile(boot,.025));hi=float(np.quantile(boot,.975))
   else:seed_means={str(s):None for s in SEEDS};pooled_mean=lo=hi=float('nan')
   support=n_groups>=50 and n_tasks>=5
   passing=(support and pooled_mean>delta and all(seed_means[str(s)] is not None and seed_means[str(s)]>0 for s in SEEDS) and lo>0)
   checks.append({'model':model,'diagnostic':diag,'seed_mean_gains':seed_means,'pooled_gain':pooled_mean,'pooled_ci95_low':lo,'pooled_ci95_high':hi,'source_groups':n_groups,'tasks':n_tasks,'support_gate':support,'display_delta':delta,'data_gate_pass':bool(passing)})
 d4path=EXP/'analysis/P2_UPDATE_GATE.json'
 d4=json.loads(d4path.read_text()) if d4path.exists() else {'status':'not_run'}
 d4_pass=d4.get('status')=='PASS_EXPLORATORY'
 data_pass=any(c['data_gate_pass'] for c in checks) or d4_pass
 resource=json.loads((EXP/'coordination/RESOURCE_ESTIMATE.json').read_text())
 measured=resource.get('h60_incremental_gpu_hours')
 if not data_pass:overall='NOT_PASS'
 elif measured is None:overall='DATA_GATE_PASS_RESOURCE_PENDING'
 elif float(measured)>8 or float(resource.get('h60_peak_vram_gib') or 0)>10:overall='BUDGET_INSUFFICIENT'
 else:overall='PASS'
 record={'status':overall,'h60_data_gate_pass':data_pass,'h60_resource_gate_status':'pending' if measured is None else 'measured','permitted_h60_addition':'ALT-POOL-R3D18 H60 x3 only; reuse V3 H60 S-HIST/F-POOL-D; no extra dense matrix','conditional_gain_gates':checks,'d4_gate':d4,'h60_incremental_gpu_hours':measured,'h60_peak_vram_gib':resource.get('h60_peak_vram_gib'),'max_incremental_gpu_hours':8,'max_peak_vram_gib':10,'final_test_access':0}
 out=EXP/'analysis/H60_GATE_STATUS.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':overall,'data_gate':data_pass,'candidate_paths':[c for c in checks if c['data_gate_pass']],'d4_gate':d4_pass,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

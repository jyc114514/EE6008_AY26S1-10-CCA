from __future__ import annotations
import os
import csv,hashlib,json
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEEDS=(20260950,20260951,20260952)
MAX_PER_CLASS=4

def read_csv(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def save_csv(path,rows,fields):
 if path.exists():raise FileExistsError(path)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def case_hash(episode_id:str,anchor_id:str)->str:
 return hashlib.sha256(f'p1p2-case-v1|{episode_id}|{anchor_id}'.encode()).hexdigest()

def main()->int:
 data_path=V3/'data/V3_MAIN_INPUTS.npz'
 with np.load(data_path,allow_pickle=False) as z:
  keys=('anchor_id','episode_index','original_task_index','split_code','anchor_frame','anchor_timestamp','targets_native','anchor_command_native')
  d={k:z[k].copy() for k in keys}
 if set(np.unique(d['split_code']).tolist())-{0,1}:raise RuntimeError('unexpected split code in canonical V3 inputs')
 dev=np.flatnonzero(d['split_code']==1);dev_id=[str(x) for x in d['anchor_id'][dev]];id_to_row={x:int(r) for x,r in zip(dev_id,dev)}
 target=d['targets_native'][:, :30];sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 metrics={}
 for model in ('S-HIST','F-POOL-D'):
  for seed in SEEDS:
   path=V3/'predictions'/f'V3-{model}-H30-S{seed}'/'primary.npz'
   with np.load(path,allow_pickle=False) as z:
    pred_ids=[str(x) for x in z['anchor_id']];pred=z['predictions_native'].copy()
   if len(set(pred_ids))!=len(pred_ids) or set(pred_ids)!=set(dev_id):raise RuntimeError(f'prediction IDs mismatch for {model}/{seed}')
   lookup={aid:i for i,aid in enumerate(pred_ids)}
   ordered=pred[np.asarray([lookup[aid] for aid in dev_id],dtype=np.int64)]
   if ordered.shape!=(len(dev),30,28) or not np.isfinite(ordered).all():raise RuntimeError('prediction shape/finite failure')
   metrics[(model,seed)]=np.mean(np.abs(ordered-target[dev])/sigma[None,None,:],axis=(1,2))
 gain=np.mean(np.stack([metrics[('S-HIST',s)]-metrics[('F-POOL-D',s)] for s in SEEDS]),axis=0)
 vj_seed={s:metrics[('F-POOL-D',s)]-metrics[('S-HIST',s)] for s in SEEDS}
 motion={r['anchor_id']:r for r in read_csv(EXP/'analysis/P2_MOTION.csv')}
 amb={r['anchor_id']:r for r in read_csv(EXP/'analysis/D1_SUPPORT_DEV.csv')}
 lock=json.loads((EXP/'analysis/D1_SUPPORT_ADAPTED_LOCK.json').read_text())
 thresholds=lock['thresholds']['30'];q25=thresholds['q25'];q75=thresholds['q75'];d95=thresholds['support_k20_distance_q95']
 motion_lock=json.loads((EXP/'analysis/P2_MOTION_LOCK.json').read_text())
 m_q25=motion_lock['train_episode_equal_q25'];m_q75=motion_lock['train_episode_equal_q75']
 with (V3/'analysis/BASELINE_RESULTS.csv').open(newline='',encoding='utf-8') as f:
  persistence=next(float(x['full_mae_std']) for x in csv.DictReader(f) if int(x['horizon_steps'])==30 and x['model']=='PERSISTENCE')
 delta=0.01*persistence
 categories={
  'vj_gain_above_delta':lambda row:row['gain']>delta,
  'vj_gain_below_negative_delta':lambda row:row['gain'] < -delta,
  'vj_gain_within_delta':lambda row:abs(row['gain'])<=delta,
  'd1_supported_high_A':lambda row:row['A']>=q75 and row['neighbor_k20_distance']<=d95,
  'train_defined_large_M':lambda row:row['M']>=m_q75,
  'train_defined_stable_M':lambda row:row['M']<=m_q25,
 }
 base=[]
 for i,row in enumerate(dev):
  aid=str(d['anchor_id'][row]);ep=int(d['episode_index'][row]);episode_id=aid.split('_a')[0]
  ar=amb[aid];mr=motion[aid]
  base.append({'anchor_id':aid,'episode_id':episode_id,'episode_index':ep,'task':int(d['original_task_index'][row]),'anchor_frame':int(d['anchor_frame'][row]),'anchor_timestamp':float(d['anchor_timestamp'][row]),'source_group_id':ar['source_group_id'],'gain':float(gain[i]),'vj_gain_by_seed':{str(s):float(vj_seed[s][i]) for s in SEEDS},'history_mae_by_seed':{str(s):float(metrics[('S-HIST',s)][i]) for s in SEEDS},'vjepa_mae_by_seed':{str(s):float(metrics[('F-POOL-D',s)][i]) for s in SEEDS},'A':float(ar['A_h30']),'neighbor_k20_distance':float(ar['neighbor_k20_distance']),'M':float(mr['M_h30']),'hash':case_hash(episode_id,aid),'final_test_access':0})
 candidates=[];selected=[]
 for category,predicate in categories.items():
  eligible=[r for r in base if predicate(r)]
  eligible.sort(key=lambda r:r['hash'])
  candidates.extend({'category':category,**r} for r in eligible)
  episodes=set();taken=0
  for row in eligible:
   if row['episode_index'] in episodes:continue
   episodes.add(row['episode_index']);selected.append({'category':category,**row});taken+=1
   if taken>=MAX_PER_CLASS:break
 candidate_fields=['category','anchor_id','episode_id','episode_index','task','anchor_frame','anchor_timestamp','source_group_id','gain','vj_gain_by_seed','history_mae_by_seed','vjepa_mae_by_seed','A','neighbor_k20_distance','M','hash','final_test_access']
 save_csv(EXP/'qualitative/CASE_CANDIDATES.csv',[{**r,'vj_gain_by_seed':json.dumps(r['vj_gain_by_seed'],sort_keys=True),'history_mae_by_seed':json.dumps(r['history_mae_by_seed'],sort_keys=True),'vjepa_mae_by_seed':json.dumps(r['vjepa_mae_by_seed'],sort_keys=True)} for r in candidates],candidate_fields)
 selected_by_id={}
 for r in selected:selected_by_id.setdefault(r['anchor_id'],[]).append(r)
 manifest=[]
 for aid,tagged in sorted(selected_by_id.items(),key=lambda x:min(r['hash'] for r in x[1])):
  r=tagged[0]
  manifest.append({'render_slot':len(manifest)+1,'anchor_id':aid,'episode_id':r['episode_id'],'episode_index':r['episode_index'],'task':r['task'],'anchor_frame':r['anchor_frame'],'anchor_timestamp':r['anchor_timestamp'],'source_group_id':r['source_group_id'],'tags':';'.join(sorted({x['category'] for x in tagged})),'selection_hash':r['hash'],'selection_reason':'V3-reference gain category and/or train-defined D1/M class; outcome-exposed development illustration','history_mae_by_seed':json.dumps(r['history_mae_by_seed'],sort_keys=True),'vjepa_mae_by_seed':json.dumps(r['vjepa_mae_by_seed'],sort_keys=True),'alternative_status':'not_scored_pending_three_accepted_H30_predictions','final_test_access':0})
 fields=list(manifest[0]) if manifest else ['render_slot','anchor_id','episode_id','episode_index','task','anchor_frame','anchor_timestamp','source_group_id','tags','selection_hash','selection_reason','history_mae_by_seed','vjepa_mae_by_seed','alternative_status','final_test_access']
 save_csv(EXP/'qualitative/CASE_MANIFEST.csv',manifest,fields)
 counts={c:sum(1 for r in candidates if r['category']==c) for c in categories}
 lock={'status':'LOCKED_CANDIDATES_BEFORE_ALT_SCORE','source':'accepted V3 H30 dev predictions; all three fixed seeds','seeds':list(SEEDS),'display_delta':delta,'classes':{c:{'eligible_anchors':counts[c],'selected_slots':sum(c in r['tags'].split(';') for r in manifest)} for c in categories},'unique_render_slots':len(manifest),'max_slots':24,'duplicate_anchor_tags_render_once':True,'alternative_used_for_selection':False,'selection_hash':'sha256(p1p2-case-v1|episode_id|anchor_id)','final_test_access':0}
 q=EXP/'qualitative/CASE_SELECTION_LOCK.json'
 if q.exists():raise FileExistsError(q)
 q.write_text(json.dumps(lock,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','classes':counts,'unique_render_slots':len(manifest),'delta_display':delta,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

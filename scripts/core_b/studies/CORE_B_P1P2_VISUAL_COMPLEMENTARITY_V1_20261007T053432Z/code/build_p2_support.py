from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import csv,json,time,hashlib
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
DATA=V3/'data/V3_MAIN_INPUTS.npz'
K=20
BATCH_QUERIES=32
CANDIDATE_ROWS=1024

def load_groups():
 out={}
 with (EXP/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):out[int(r['episode_index'])]=r['source_group_id']
 return out

def stable_rows(indices,episodes,frames,ids):
 return indices[np.lexsort((ids[indices],frames[indices],episodes[indices]))]

def choose_neighbors(distances,bank_groups,query_group,k=K):
 d=distances.copy();d[bank_groups==query_group]=np.inf
 take=min(len(d),max(CANDIDATE_ROWS,k*64))
 cand=np.argpartition(d,take-1)[:take]
 order=cand[np.lexsort((cand,d[cand]))]
 chosen=[];seen=set()
 for j in order:
  g=bank_groups[j]
  if g in seen:continue
  seen.add(g);chosen.append(int(j))
  if len(chosen)==k:break
 if len(chosen)<k:
  order=np.lexsort((np.arange(len(d)),d))
  for j in order:
   g=bank_groups[j]
   if g in seen:continue
   seen.add(g);chosen.append(int(j))
   if len(chosen)==k:break
 if len(chosen)!=k:raise RuntimeError(f'found {len(chosen)} unique groups, K={k} required')
 return np.asarray(chosen,dtype=np.int64)

def moments_for_queries(query_rows,kind,data,bank_rows,bank_state,bank_groups,bank_norm,sigma,group_lookup):
 qstate=data['history_standardized'].reshape(len(data['episode_index']),-1).astype(np.float32)
 qmat=qstate[query_rows]
 qnorm=np.sum(qmat.astype(np.float64)**2,axis=1)
 distances_out=np.empty((len(query_rows),K),dtype=np.float32)
 chosen_rows=np.empty((len(query_rows),K),dtype=np.int64)
 a30=np.empty(len(query_rows),dtype=np.float32);a60=np.empty(len(query_rows),dtype=np.float32)
 anchor=data['anchor_command_native'];targets=data['targets_native'];episodes=data['episode_index'];frames=data['anchor_frame'];ids=data['anchor_id'];D=bank_state.shape[1]
 for start in range(0,len(query_rows),BATCH_QUERIES):
  stop=min(start+BATCH_QUERIES,len(query_rows));qb=qmat[start:stop]
  dot=qb@bank_state.T
  all_d=(qnorm[start:stop,None]+bank_norm[None,:]-2.0*dot.astype(np.float64))/D
  all_d=np.maximum(all_d,0.0)
  for local in range(stop-start):
   qidx=int(query_rows[start+local]);qgroup=group_lookup[int(episodes[qidx])]
   loc=choose_neighbors(all_d[local],bank_groups,qgroup)
   src=bank_rows[loc]
   # Recompute selected distances in float64 and preserve deterministic tie ordering.
   exact=np.mean((bank_state[loc].astype(np.float64)-qmat[start+local].astype(np.float64))**2,axis=1)
   order=np.lexsort((loc,exact));loc=loc[order];src=bank_rows[loc];exact=exact[order]
   chosen_rows[start+local]=src;distances_out[start+local]=exact.astype(np.float32)
   for horizon,target_out in ((30,a30),(60,a60)):
    residual=(targets[src,:horizon]-anchor[src,None,:])/sigma[None,None,:]
    center=np.median(residual,axis=0)
    target_out[start+local]=np.mean(np.abs(residual-center))
 return {'query_rows':query_rows,'query_kind':np.full(len(query_rows),kind,dtype='U16'),'query_anchor_id':ids[query_rows].astype('U64'),'query_episode':episodes[query_rows],'query_task':data['original_task_index'][query_rows],'query_frame':frames[query_rows],'query_group':np.asarray([group_lookup[int(e)] for e in episodes[query_rows]],dtype='U64'),'A_h30':a30,'A_h60':a60,'neighbor_k20_distance':distances_out[:,-1],'neighbor_rows':chosen_rows,'neighbor_distances':distances_out}

def main()->int:
 t0=time.perf_counter()
 with np.load(DATA,allow_pickle=False) as z:
  keys=('history_standardized','targets_native','anchor_command_native','episode_index','split_code','anchor_id','anchor_frame','original_task_index')
  data={k:z[k].copy() for k in keys}
 if set(np.unique(data['split_code']).tolist())-{0,1}:raise RuntimeError('V3 inputs contain non train/dev split code')
 groups=load_groups();episodes=data['episode_index'];split=data['split_code'];frames=data['anchor_frame'];ids=data['anchor_id']
 train=np.flatnonzero(split==0);dev=np.flatnonzero(split==1)
 bank_rows=stable_rows(train,episodes,frames,ids)
 calibration=[];seen=set()
 for row in bank_rows:
  g=groups[int(episodes[row])]
  if g in seen:continue
  seen.add(g);calibration.append(int(row))
 calibration=np.asarray(calibration,dtype=np.int64)
 if len(calibration)!=len(seen):raise RuntimeError('train calibration group mapping is not one row per source group')
 bank_state=data['history_standardized'][bank_rows].reshape(len(bank_rows),-1).astype(np.float32)
 bank_groups=np.asarray([groups[int(e)] for e in episodes[bank_rows]],dtype='U64')
 bank_norm=np.sum(bank_state.astype(np.float64)**2,axis=1)
 sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 cal=moments_for_queries(calibration,'train_calibration',data,bank_rows,bank_state,bank_groups,bank_norm,sigma,groups)
 validation=moments_for_queries(dev,'dev',data,bank_rows,bank_state,bank_groups,bank_norm,sigma,groups)
 thresholds={}
 for h,key in ((30,'A_h30'),(60,'A_h60')):
  values=cal[key].astype(np.float64);dist=cal['neighbor_k20_distance'].astype(np.float64)
  q25=float(np.quantile(values,.25,method='linear'));q75=float(np.quantile(values,.75,method='linear'));d95=float(np.quantile(dist,.95,method='linear'))
  thresholds[str(h)]={'train_calibration_rows':len(values),'train_calibration_groups':len(set(cal['query_group'].tolist())),'q25':q25,'q75':q75,'support_k20_distance_q95':d95,'threshold_degenerate':bool(q75<=q25),'final_test_access':0}
 rows=[]
 for i in range(len(dev)):
  item={'anchor_id':str(validation['query_anchor_id'][i]),'episode_index':int(validation['query_episode'][i]),'task':int(validation['query_task'][i]),'anchor_frame':int(validation['query_frame'][i]),'source_group_id':str(validation['query_group'][i]),'A_h30':float(validation['A_h30'][i]),'A_h60':float(validation['A_h60'][i]),'neighbor_k20_distance':float(validation['neighbor_k20_distance'][i]),'final_test_access':0}
  for h,key in ((30,'A_h30'),(60,'A_h60')):
   th=thresholds[str(h)];a=item[key]
   item[f'stratum_h{h}']='NA' if th['threshold_degenerate'] else ('low' if a<=th['q25'] else ('high' if a>=th['q75'] else 'middle'))
   item[f'supported_h{h}']=item['neighbor_k20_distance']<=th['support_k20_distance_q95']
  rows.append(item)
 out=EXP/'analysis/D1_SUPPORT_ADAPTED_LOCK.json'
 if out.exists():raise FileExistsError(out)
 summary={'status':'PASS','adaptation':'full canonical primary train history bank (17757 rows), duplicate-safe source_group_id, K20 one neighbor per group; train calibration one earliest anchor per group; leave-group-out for train calibration queries. This is a minimal D1 adaptation: historical D1 capped bank at 8 anchors per episode and used episode_index because duplicate group field was absent.','distance':'mean squared distance over flattened standardized history32x71','dispersion':'neighbor residual r=(future-own_anchor)/target_sigma; coordinatewise median center; A=mean(abs(r-center)) over K,time,Full-28','bank_rows':len(bank_rows),'bank_episodes':len(set(bank_groups.tolist())),'train_calibration_queries':len(calibration),'dev_queries':len(dev),'thresholds':thresholds,'source_group_map':str(EXP/'inventory/SOURCE_GROUP_MAP.csv'),'runtime_seconds':time.perf_counter()-t0,'final_test_access':0}
 out.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 for filename,items in [('D1_SUPPORT_TRAIN_CALIBRATION.csv',cal),('D1_SUPPORT_DEV.csv',validation)]:
  path=EXP/'analysis'/filename
  if path.exists():raise FileExistsError(path)
  fields=['query_kind','anchor_id','episode_index','task','anchor_frame','source_group_id','A_h30','A_h60','neighbor_k20_distance','neighbor_anchor_ids','neighbor_source_group_ids','final_test_access']
  with path.open('w',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for i,row in enumerate(items['query_rows']):
    nrows=items['neighbor_rows'][i]
    writer.writerow({'query_kind':str(items['query_kind'][i]),'anchor_id':str(items['query_anchor_id'][i]),'episode_index':int(items['query_episode'][i]),'task':int(data['original_task_index'][row]),'anchor_frame':int(data['anchor_frame'][row]),'source_group_id':str(items['query_group'][i]),'A_h30':float(items['A_h30'][i]),'A_h60':float(items['A_h60'][i]),'neighbor_k20_distance':float(items['neighbor_k20_distance'][i]),'neighbor_anchor_ids':';'.join(str(ids[x]) for x in nrows),'neighbor_source_group_ids':';'.join(groups[int(episodes[x])] for x in nrows),'final_test_access':0})
 print(json.dumps({'status':'PASS','bank_rows':len(bank_rows),'calibration':len(calibration),'dev':len(dev),'thresholds':thresholds,'elapsed_seconds':round(time.perf_counter()-t0,3),'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

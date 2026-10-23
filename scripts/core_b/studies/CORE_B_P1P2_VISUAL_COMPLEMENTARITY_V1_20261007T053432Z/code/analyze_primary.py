from __future__ import annotations
import os
import csv,json,hashlib
from pathlib import Path
import numpy as np
from analysis_stats import shared_group_bootstrap_counts, episode_equal_cluster_means

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEEDS=(20260950,20260951,20260952)
MODELS=('S-HIST','F-POOL-D','ALT-POOL-R3D18')
N_BOOT=10000
BOOT_SEED=20261007

def read_csv(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 if path.exists():raise FileExistsError(path)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_bundle(path,expected_ids):
 with np.load(path,allow_pickle=False) as z:
  ids=np.asarray(z['anchor_id']).astype('U64');pred=z['predictions_native'].copy()
 if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(expected_ids):raise RuntimeError(f'prediction ID set mismatch: {path}')
 lookup={aid:i for i,aid in enumerate(ids)};ordered=pred[np.asarray([lookup[x] for x in expected_ids],dtype=np.int64)]
 if ordered.shape!=(len(expected_ids),30,28) or not np.isfinite(ordered).all():raise RuntimeError(f'prediction shape/finiteness mismatch: {path}')
 return ordered

def load_alt_paths():
 path=EXP/'predictions/PRIMARY_MANIFEST.csv'
 if not path.is_file():raise RuntimeError('ALT primary manifest missing; score gate remains closed')
 rows=read_csv(path)
 expected={str(s) for s in SEEDS}
 selected=[r for r in rows if r.get('condition')=='ALT-POOL-R3D18' and r.get('horizon_steps')=='30' and r.get('status')=='accepted']
 if {r['seed'] for r in selected}!=expected or len(selected)!=3:raise RuntimeError('all three accepted H30 ALT bundles are required before scoring')
 return {int(r['seed']):EXP/r['bundle_path'] for r in selected}

def main()->int:
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
  data={k:z[k].copy() for k in ('anchor_id','episode_index','original_task_index','split_code','targets_native','anchor_command_native')}
 if set(np.unique(data['split_code']).tolist())-{0,1}:raise RuntimeError('input contains a split outside canonical train/dev')
 dev_rows=np.flatnonzero(data['split_code']==1);ids=np.asarray(data['anchor_id'][dev_rows]).astype('U64');id_list=ids.tolist()
 target=data['targets_native'][dev_rows,:30].astype(np.float64);anchor=data['anchor_command_native'][dev_rows].astype(np.float64)
 sigma=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 if target.shape!=(2322,30,28) or not np.isfinite(target).all():raise RuntimeError('primary dev target contract mismatch')
 groups={}
 for r in read_csv(EXP/'inventory/SOURCE_GROUP_MAP.csv'):groups[int(r['episode_index'])]=r['source_group_id']
 dev_episode=data['episode_index'][dev_rows];dev_task=data['original_task_index'][dev_rows]
 group_ids=np.asarray([groups[int(e)] for e in dev_episode],dtype='U64')
 models={}
 models['PERSISTENCE']={s:np.broadcast_to(anchor[:,None,:],target.shape).copy() for s in SEEDS}
 for model in ('S-HIST','F-POOL-D'):
  for seed in SEEDS:
   path=V3/'predictions'/f'V3-{model}-H30-S{seed}'/'primary.npz'
   models.setdefault(model,{})[seed]=load_bundle(path,id_list)
 alt_paths=load_alt_paths()
 for seed,path in alt_paths.items():models.setdefault('ALT-POOL-R3D18',{})[seed]=load_bundle(path,id_list)
 errors={};anchor_rows=[];episode_rows=[]
 for model,pred_by_seed in models.items():
  for seed,pred in pred_by_seed.items():
   window=np.mean(np.abs(pred.astype(np.float64)-target)/sigma[None,None,:],axis=(1,2))
   errors[(model,seed)]=window
   for i,row in enumerate(dev_rows):
    anchor_rows.append({'model':model,'seed':seed,'anchor_id':str(ids[i]),'episode_index':int(dev_episode[i]),'source_group_id':str(group_ids[i]),'task':int(dev_task[i]),'full28_mae_std':float(window[i]),'final_test_access':0})
   for episode in np.unique(dev_episode):
    selected=np.flatnonzero(dev_episode==episode)
    episode_rows.append({'model':model,'seed':seed,'episode_index':int(episode),'source_group_id':str(group_ids[selected[0]]),'task':int(dev_task[selected[0]]),'N_anchor':len(selected),'full28_mae_std':float(window[selected].mean()),'final_test_access':0})
 episode_index=sorted(set(int(e) for e in dev_episode));episode_lookup={e:i for i,e in enumerate(episode_index)}
 group_for_episode={e:groups[e] for e in episode_index};groups_order=sorted(set(group_for_episode.values()))
 episode_by_group={g:[] for g in groups_order}
 for e,g in group_for_episode.items():episode_by_group[g].append(e)
 # Current canonical allowlist has one source group per episode; retain a general nested cluster resample.
 score=np.empty((4,len(SEEDS),len(episode_index)),dtype=np.float64)
 model_order=('PERSISTENCE','S-HIST','F-POOL-D','ALT-POOL-R3D18')
 for mi,model in enumerate(model_order):
  for si,seed in enumerate(SEEDS):
   for e in episode_index:
    local=np.flatnonzero(dev_episode==e)
    score[mi,si,episode_lookup[e]]=float(errors[(model,seed)][local].mean())
 global_rows=[]
 for mi,model in enumerate(model_order):
  seed_values=[float(score[mi,si].mean()) for si in range(len(SEEDS))]
  if model=='PERSISTENCE':
   global_rows.append({'model':model,'seed':'NOT_APPLICABLE','full28_mae_std':seed_values[0],'N_episode':len(episode_index),'N_anchor':len(dev_rows),'aggregation':'episode_equal','final_test_access':0})
  else:
   for si,seed in enumerate(SEEDS):global_rows.append({'model':model,'seed':seed,'full28_mae_std':seed_values[si],'N_episode':len(episode_index),'N_anchor':len(dev_rows),'aggregation':'episode_equal','final_test_access':0})
 aggregates=[]
 for mi,model in enumerate(model_order):
  vals=np.asarray([score[mi,si].mean() for si in range(len(SEEDS))],dtype=np.float64)
  aggregates.append({'model':model,'mean_full28_mae_std':float(vals.mean()),'seed_sample_sd':None if model=='PERSISTENCE' else float(vals.std(ddof=1)),'seed_count':1 if model=='PERSISTENCE' else len(SEEDS),'N_episode':len(episode_index),'N_anchor':len(dev_rows),'recipe_class':'legacy_actual_recipe' if model not in ('PERSISTENCE',) else 'baseline','final_test_access':0})
 with (V3/'analysis/BASELINE_RESULTS.csv').open(newline='',encoding='utf-8') as f:
  persistence=next(float(r['full_mae_std']) for r in csv.DictReader(f) if int(r['horizon_steps'])==30 and r['model']=='PERSISTENCE')
 delta=0.01*persistence
 contrasts=[('F-POOL-D','S-HIST'),('ALT-POOL-R3D18','S-HIST'),('ALT-POOL-R3D18','F-POOL-D')]
 effect_rows=[];top_rows=[]
 for model,control in contrasts:
  for seed in SEEDS:
   m=score[model_order.index(model),SEEDS.index(seed)];c=score[model_order.index(control),SEEDS.index(seed)];d=m-c
   effect_rows.append({'contrast':f'{model}-{control}','model':model,'control':control,'seed':seed,'estimand':'episode_mean_model_minus_control','mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'q10':float(np.quantile(d,.1)),'q25':float(np.quantile(d,.25)),'q75':float(np.quantile(d,.75)),'q90':float(np.quantile(d,.9)),'improved_beyond_delta':int(np.sum(d < -delta)),'within_display_delta':int(np.sum(np.abs(d)<=delta)),'degraded_beyond_delta':int(np.sum(d>delta)),'N_episode':len(d),'display_delta':delta,'final_test_access':0})
   order=np.argsort(d)
   for rank,i in enumerate(list(order[:10])+list(order[-10:][::-1]),1):
    e=episode_index[i];top_rows.append({'contrast':f'{model}-{control}','seed':seed,'rank_within_improve_then_degrade':rank,'episode_index':e,'task':int(next(t for t,x in zip(dev_task,dev_episode) if int(x)==e)),'delta_mae':float(d[i]),'direction':'improved' if rank<=10 else 'degraded','final_test_access':0})
 # Seed-averaged encoder gain quadrants and a group bootstrap shared across contrasts.
 vj_gain=score[0].mean(axis=0)-score[2].mean(axis=0)
 alt_gain=score[0].mean(axis=0)-score[3].mean(axis=0)
 quadrants=[]
 for i,e in enumerate(episode_index):
  q='both_improve' if vj_gain[i]>0 and alt_gain[i]>0 else ('vjepa_only_improves' if vj_gain[i]>0 else ('r3d_only_improves' if alt_gain[i]>0 else 'neither_improves'))
  quadrants.append({'episode_index':e,'source_group_id':group_for_episode[e],'vjepa_gain':float(vj_gain[i]),'r3d_gain':float(alt_gain[i]),'quadrant':q,'vj_within_display_delta':abs(float(vj_gain[i]))<=delta,'r3d_within_display_delta':abs(float(alt_gain[i]))<=delta,'final_test_access':0})
 gcount=len(groups_order);batches=10000;counts=shared_group_bootstrap_counts(gcount,batches,BOOT_SEED)
 n_episode=np.asarray([len(episode_by_group[g]) for g in groups_order],dtype=np.float64)
 group_sums=np.zeros((4,len(SEEDS),gcount),dtype=np.float64)
 for gi,g in enumerate(groups_order):
  epis=[episode_lookup[e] for e in episode_by_group[g]]
  group_sums[:,:,gi]=score[:,:,epis].sum(axis=2)
 denom=counts@n_episode
 means=np.empty((4,len(SEEDS),batches),dtype=np.float64)
 for mi in range(4):
  for si in range(len(SEEDS)):
   means[mi,si]=episode_equal_cluster_means(group_sums[mi,si],n_episode,counts)
 ci_rows=[]
 for model,control in contrasts:
  mi=model_order.index(model);ci=model_order.index(control)
  draw_delta=np.mean(means[mi]-means[ci],axis=0)
  point=float(np.mean(score[mi]-score[ci]))
  ci_rows.append({'contrast':f'{model}-{control}','estimand':'mean_across_fixed_seeds_of_episode_equal_difference','estimate':point,'ci95_low':float(np.quantile(draw_delta,.025)),'ci95_high':float(np.quantile(draw_delta,.975)),'bootstrap_replicates':batches,'bootstrap_seed':BOOT_SEED,'bootstrap_unit':'duplicate-safe source_group_id; all group episodes/anchors/methods/seeds retained','fixed_models':True,'final_test_access':0})
 outdir=EXP/'analysis'
 write_csv(outdir/'ANCHOR_PAIRED.csv',anchor_rows);write_csv(outdir/'EPISODE_PAIRED.csv',episode_rows);write_csv(outdir/'GLOBAL_BY_SEED.csv',global_rows);write_csv(outdir/'GLOBAL_AGGREGATE.csv',aggregates);write_csv(outdir/'EFFECT_DISTRIBUTIONS.csv',effect_rows);write_csv(outdir/'ENCODER_QUADRANTS.csv',quadrants);write_csv(outdir/'BOOTSTRAP_CI.csv',ci_rows);write_csv(outdir/'TOP_EPISODES.csv',top_rows)
 check={'status':'PASS','canonical_dev_episodes':len(episode_index),'canonical_dev_anchors':len(dev_rows),'group_count':gcount,'final_test_access':0,'score_release_gate':'all three ALT H30 bundles accepted before this script read any ALT scores','delta_display':delta,'reference_data_sha256':digest(V3/'analysis/PRIMARY_RESULTS_BY_SEED.csv'),'scored_predictions':{'ALT-POOL-R3D18':[str(alt_paths[s].relative_to(EXP)) for s in SEEDS]}}
 q=EXP/'qa/PRIMARY_ANALYSIS_ACCEPTANCE.json'
 if q.exists():raise FileExistsError(q)
 q.write_text(json.dumps(check,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','global':aggregates,'contrasts':ci_rows,'delta_display':delta,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

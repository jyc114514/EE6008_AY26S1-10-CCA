from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[1]
SEEDS=(20260950,20260951,20260952)
MODELS=('F-POOL-D','ALT-POOL-R3D18')
N_BOOT=10000;BOOT_SEED=20261007

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 if p.exists():raise FileExistsError(p)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main()->int:
 anchors=read_csv(EXP/'analysis/ANCHOR_PAIRED.csv')
 if not anchors:raise RuntimeError('accepted primary analysis anchor table required')
 motion={r['anchor_id']:r for r in read_csv(EXP/'analysis/P2_MOTION.csv')}
 ambiguity={r['anchor_id']:r for r in read_csv(EXP/'analysis/D1_SUPPORT_DEV.csv')}
 d1lock=json.loads((EXP/'analysis/D1_SUPPORT_ADAPTED_LOCK.json').read_text());m_lock=json.loads((EXP/'analysis/P2_MOTION_LOCK.json').read_text())
 q25=d1lock['thresholds']['30']['q25'];q75=d1lock['thresholds']['30']['q75'];d95=d1lock['thresholds']['30']['support_k20_distance_q95']
 mq25=m_lock['train_episode_equal_q25'];mq75=m_lock['train_episode_equal_q75']
 groups={};episode_task={};anchor_by_seed={}
 for r in anchors:
  if r['model'] not in ('S-HIST',*MODELS):continue
  key=(r['model'],int(r['seed']),r['anchor_id']);anchor_by_seed[key]=float(r['full28_mae_std'])
  e=int(r['episode_index']);groups[e]=r['source_group_id'];episode_task[e]=int(r['task'])
 episode_ids=sorted(groups);group_ids=sorted(set(groups.values()))
 if len(group_ids)!=len(episode_ids):raise RuntimeError('nested source groups require the general group-episode aggregator')
 index={aid:r for r in anchors for aid in [r['anchor_id']]}
 gain_rows=[]
 for aid,mr in motion.items():
  ar=ambiguity[aid]
  for model in MODELS:
   for seed in SEEDS:
    hist=anchor_by_seed[('S-HIST',seed,aid)];value=anchor_by_seed[(model,seed,aid)]
    gain=hist-value
    gain_rows.append({'model':model,'seed':seed,'anchor_id':aid,'episode_index':int(index[aid]['episode_index']),'task':int(index[aid]['task']),'source_group_id':index[aid]['source_group_id'],'gain_vs_history':gain,'M_h30':float(mr['M_h30']),'M_stratum':mr['stratum'],'A_h30':float(ar['A_h30']),'A_k20_distance':float(ar['neighbor_k20_distance']),'A_supported':bool(float(ar['neighbor_k20_distance'])<=d95),'A_stratum':'low' if float(ar['A_h30'])<=q25 else ('high' if float(ar['A_h30'])>=q75 else 'middle'),'final_test_access':0})
 # Episode means within each locked stratum, preserving all eligible episodes.
 conditions=[('M_H30_large_minus_stable','large','stable',False),('D1_A_H30_supported_high_minus_low','high','low',True)]
 interaction_rows=[];bootstrap=rng=np.random.default_rng(BOOT_SEED);G=len(group_ids)
 counts=rng.integers(0,G,size=(N_BOOT,G),dtype=np.int32)
 # Convert sampled group IDs into multiplicities; groups and episodes are verified one-to-one.
 count_matrix=np.zeros((N_BOOT,G),dtype=np.uint16);draw_rows=np.broadcast_to(np.arange(N_BOOT,dtype=np.int32)[:,None],counts.shape);np.add.at(count_matrix,(draw_rows,counts),1)
 group_to_index={g:i for i,g in enumerate(group_ids)}
 episode_for_group={g:next(e for e in episode_ids if groups[e]==g) for g in group_ids}
 for model in MODELS:
  for seed in SEEDS:
   local=[r for r in gain_rows if r['model']==model and r['seed']==seed]
   for name,high_name,low_name,require_support in conditions:
    high_ep={};low_ep={};support_rows=[]
    for e in episode_ids:
     ep_rows=[r for r in local if r['episode_index']==e]
     high=[r for r in ep_rows if (r['M_stratum']==high_name if name.startswith('M_') else (r['A_stratum']==high_name and (not require_support or bool(r['A_supported']))))]
     low=[r for r in ep_rows if (r['M_stratum']==low_name if name.startswith('M_') else (r['A_stratum']==low_name and (not require_support or bool(r['A_supported']))))]
     if high:high_ep[e]=float(np.mean([r['gain_vs_history'] for r in high]))
     if low:low_ep[e]=float(np.mean([r['gain_vs_history'] for r in low]))
    high_groups={groups[e] for e in high_ep};low_groups={groups[e] for e in low_ep}
    high_tasks={episode_task[e] for e in high_ep};low_tasks={episode_task[e] for e in low_ep}
    point_high=float(np.mean(list(high_ep.values()))) if high_ep else float('nan')
    point_low=float(np.mean(list(low_ep.values()))) if low_ep else float('nan')
    point_inter=point_high-point_low if high_ep and low_ep else float('nan')
    draw_high=np.full(N_BOOT,np.nan);draw_low=np.full(N_BOOT,np.nan);draw_inter=np.full(N_BOOT,np.nan)
    for b in range(N_BOOT):
     sampled_groups=np.repeat(np.arange(G,dtype=np.int32),count_matrix[b].astype(np.int32))
     sampled_episodes=[episode_for_group[group_ids[int(gi)]] for gi in sampled_groups]
     hes=[high_ep[e] for e in sampled_episodes if e in high_ep]
     les=[low_ep[e] for e in sampled_episodes if e in low_ep]
     if hes:draw_high[b]=float(np.mean(hes))
     if les:draw_low[b]=float(np.mean(les))
     if hes and les:draw_inter[b]=draw_high[b]-draw_low[b]
    def ci(x):
     v=x[np.isfinite(x)]
     return (float(np.quantile(v,.025)),float(np.quantile(v,.975)),len(v)) if len(v) else (float('nan'),float('nan'),0)
    hl,hh,hn=ci(draw_high);ll,lh,ln=ci(draw_low);il,ih,inn=ci(draw_inter)
    support_ok=len(high_groups)>=50 and len(low_groups)>=50 and len(high_tasks)>=5 and len(low_tasks)>=5
    interaction_rows.append({'diagnostic':name,'model':model,'seed':seed,'high_stratum':high_name,'low_stratum':low_name,'high_gain':point_high,'high_gain_ci95_low':hl,'high_gain_ci95_high':hh,'low_gain':point_low,'low_gain_ci95_low':ll,'low_gain_ci95_high':lh,'high_minus_low_gain':point_inter,'interaction_ci95_low':il,'interaction_ci95_high':ih,'N_high_episodes':len(high_ep),'N_high_groups':len(high_groups),'N_high_tasks':len(high_tasks),'N_low_episodes':len(low_ep),'N_low_groups':len(low_groups),'N_low_tasks':len(low_tasks),'support_gate_50groups_5tasks':support_ok,'bootstrap_replicates':N_BOOT,'bootstrap_seed':BOOT_SEED,'final_test_access':0})
 write_csv(EXP/'analysis/P2_GAIN_BY_ANCHOR.csv',gain_rows)
 write_csv(EXP/'analysis/P2_INTERACTIONS.csv',interaction_rows)
 print(json.dumps({'status':'PASS','rows':len(gain_rows),'interactions':len(interaction_rows),'support':[{k:r[k] for k in ('diagnostic','model','seed','N_high_groups','N_high_tasks','N_low_groups','N_low_tasks','support_gate_50groups_5tasks')} for r in interaction_rows[:4]],'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

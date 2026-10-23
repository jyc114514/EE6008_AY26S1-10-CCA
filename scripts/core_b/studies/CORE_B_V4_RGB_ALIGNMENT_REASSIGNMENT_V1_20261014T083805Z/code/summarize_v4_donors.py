#!/usr/bin/env python3
"""Summarize fixed train-rotation and dev-map donor quality without labels or RGB values."""
from __future__ import annotations
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def stable_hash(text):return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16],16)
def summ(x):
 x=np.asarray(x,dtype=np.float64)
 return {'n':int(len(x)),'min':float(x.min()) if len(x) else 'NA','median':float(np.median(x)) if len(x) else 'NA','mean':float(x.mean()) if len(x) else 'NA','max':float(x.max()) if len(x) else 'NA'}
def main():
 with np.load(ROOT/'data/TRAIN_SELECTED_H30.npz',allow_pickle=False) as z:
  ids=z['anchor_id'].astype(str);tasks=z['task_id'].astype(np.int64);episodes=z['episode_id'].astype(np.int64);groups=z['source_cluster'].astype(str)
 with np.load(ROOT/'protocol/TRAIN_DONOR_CANDIDATES.npz',allow_pickle=False) as z:
  cand=z['candidate_row'];dist=z['candidate_distance'];ks=z['candidate_k']
 train_rows=read(ROOT/'protocol/TRAIN_DONOR_CANDIDATES.csv');eval_cand=read(ROOT/'protocol/EVAL_DONOR_CANDIDATES.csv');eval_maps=read(ROOT/'protocol/EVAL_DONOR_MAPS.csv');v3=read(ROOT/'inventory/V3_DONOR_STATE_DISTANCE.csv')
 out=[]
 for task in TASKS:
  ix=np.flatnonzero(tasks==task);nearest=dist[ix,0];sel_dist=[];reuse_fr=[];donor_unique=[];fallback=int(np.sum(ks[ix]==0));all_selections=0
  for seed in SEEDS:
   for epoch0 in range(160):
    chosen=[];local_dist=[]
    for i in ix:
     k=int(ks[i])
     if k<=0:continue
     off=stable_hash(f'{seed}|{ids[i]}');rank=(off+epoch0)%k;di=int(cand[i,rank]);chosen.append(di);local_dist.append(float(dist[i,rank]))
    all_selections+=len(chosen);sel_dist.extend(local_dist)
    c=Counter(chosen);reuse_fr.append((len(chosen)-len(c))/len(chosen) if chosen else 0.0);donor_unique.append(len(c))
  out.append({'split':'train_rotation_over_160_epochs_x_3_seeds','task_id':task,'receivers':int(len(ix)),'episodes':int(len(np.unique(episodes[ix]))),'candidate_k_min':int(ks[ix].min()),'candidate_k_max':int(ks[ix].max()),'fallback_receivers':fallback,'nearest_candidate_distance_json':json.dumps(summ(nearest),sort_keys=True),'selected_rotation_distance_json':json.dumps(summ(sel_dist),sort_keys=True),'v3_same_task_donor_distance_json':'NA','selected_donor_assignments':all_selections,'mean_unique_donor_anchors_per_seed_epoch':float(np.mean(donor_unique)),'mean_same_epoch_donor_reuse_fraction':float(np.mean(reuse_fr)),'maps_independent_replicates':False})
 for task in TASKS:
  q=[r for r in eval_cand if int(r['task_id'])==task];near=[]
  first={}
  for r in q:
   if r['candidate_rank']=='1':first[r['receiver_anchor_id']]=float(r['state_distance'])
  near=list(first.values());v3dist=[float(r['v3_state_distance']) for r in v3 if int(r['task_id'])==task]
  taskmaps=[]
  for map_id in ('D1','D2','D3'):
   r=[x for x in eval_maps if x['map_id']==map_id and int(x['task_id'])==task]
   d=[float(x['state_distance']) for x in r if x['state_distance']!='NA'];don=[x['donor_anchor_id'] for x in r if x['donor_anchor_id']]
   cnt=Counter(don);taskmaps.append({'map_id':map_id,'selected_distance':summ(d),'unique_donor_anchors':len(cnt),'donor_assignments':len(don),'reuse_fraction':float((len(don)-len(cnt))/len(don)) if don else 'NA','fallback':sum(x['donor_status']!='PASS' for x in r)})
  ks_eval=[int(r['candidate_k']) for r in eval_maps if int(r['task_id'])==task and r['map_id']=='D1']
  out.append({'split':'dev_eval_maps','task_id':task,'receivers':sum(1 for x in eval_maps if int(x['task_id'])==task and x['map_id']=='D1'),'episodes':len(set(int(x['receiver_episode_id']) for x in eval_maps if int(x['task_id'])==task and x['map_id']=='D1')),'candidate_k_min':min(ks_eval),'candidate_k_max':max(ks_eval),'fallback_receivers':sum(x['donor_status']!='PASS' for x in eval_maps if int(x['task_id'])==task and x['map_id']=='D1'),'nearest_candidate_distance_json':json.dumps(summ(near),sort_keys=True),'selected_rotation_distance_json':json.dumps({x['map_id']:x['selected_distance'] for x in taskmaps},sort_keys=True),'v3_same_task_donor_distance_json':json.dumps(summ(v3dist),sort_keys=True),'selected_donor_assignments':sum(x['donor_assignments'] for x in taskmaps),'mean_unique_donor_anchors_per_seed_epoch':json.dumps({x['map_id']:x['unique_donor_anchors'] for x in taskmaps},sort_keys=True),'mean_same_epoch_donor_reuse_fraction':json.dumps({x['map_id']:x['reuse_fraction'] for x in taskmaps},sort_keys=True),'maps_independent_replicates':False})
 p=ROOT/'results/DONOR_MATCH_QUALITY.csv'
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 receipt=json.loads((ROOT/'protocol/DONOR_BANK_RECEIPT.json').read_text());receipt.update({'donor_match_quality_csv_sha256':sha(p),'train_rotation_seeds':list(SEEDS),'train_rotation_epochs':160,'quality_summary_status':'PASS','targets_requested':False,'final_heldout_payload_access':0});(ROOT/'protocol/DONOR_BANK_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','rows':len(out),'sha256':sha(p),'targets_requested':False,'final_access':0},indent=2))
if __name__=='__main__':main()

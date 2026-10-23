#!/usr/bin/env python3
"""Lock train-derived recording-command-change strata; does not read dev targets."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    train_path=ROOT/'data/TRAIN_SELECTED_H30.npz';normalizer_path=ROOT/'normalizers/NORMALIZER_FORMAL.json'
    norm=json.loads(normalizer_path.read_text());scale=np.asarray(norm['safe_fd_scale'],dtype=np.float64)
    with np.load(train_path,allow_pickle=False) as z:
        task=z['task_id'].copy();ep=z['episode_id'].copy();y=z['target_native_h30'].astype(np.float64,copy=True)
    rows=[];locks={}
    for t in TASKS:
        task_eps=sorted(int(e) for e in np.unique(ep[task==t]));scores=[]
        for e in task_eps:
            idx=np.flatnonzero((task==t)&(ep==e))
            fd=np.abs(np.diff(y[idx],axis=1))/scale[None,None,:]
            score=float(fd.mean(axis=(1,2)).mean())
            scores.append(score)
        q25=float(np.quantile(scores,0.25,method='linear'));q75=float(np.quantile(scores,0.75,method='linear'))
        group={'LOW':[],'MID':[],'HIGH':[]}
        for e,v in zip(task_eps,scores):
            name='LOW' if v<=q25 else ('MID' if v<=q75 else 'HIGH');group[name].append(e)
            rows.append({'task_id':t,'episode_id':e,'M_fd_train':v,'q25_train':q25,'q75_train':q75,'stratum':name})
        locks[str(t)]={'train_episodes':len(task_eps),'q25_linear':q25,'q75_linear':q75,'support_episodes':{k:len(v) for k,v in group.items()},'empty_groups':[k for k,v in group.items() if not v],'q25_q75_tied':q25==q75,'ties_at_q25':sum(v==q25 for v in scores),'ties_at_q75':sum(v==q75 for v in scores)}
    p=ROOT/'protocol/TRAIN_MOTION_SCORES.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    obj={'schema_version':1,'status':'LOCKED_FROM_TRAIN_BEFORE_V4_DEV_SCORING','definition':'M_fd(episode)=mean over receiver anchors and future t=2..30, axes j of abs(y[t,j]-y[t-1,j])/accepted safe_fd_scale[j]','future_internal_only':True,'scale_path':str(normalizer_path),'scale_sha256':sha(normalizer_path),'train_npz_path':str(train_path),'train_npz_sha256':sha(train_path),'quantile_method':'linear','q25_q75':[0.25,0.75],'strata_rule':'LOW<=q25; MID>q25 and <=q75; HIGH>q75','by_task':locks,'dev_targets_read':False,'outcome_defined_descriptive_strata':True,'not_physical_motion_or_speed':True,'not_model_input_routing_or_sample_weight':True,'final_heldout_payload_access':0,'train_scores_csv_sha256':sha(p)}
    (ROOT/'protocol/MOTION_STRATA.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(obj,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

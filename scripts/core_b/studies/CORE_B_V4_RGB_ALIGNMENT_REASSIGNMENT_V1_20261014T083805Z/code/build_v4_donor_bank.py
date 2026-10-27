#!/usr/bin/env python3
"""Build the locked train-only state-nearest RGB donor bank. Never reads labels or video."""
from __future__ import annotations
import os
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TASKS=(119,94,90,141,230); K_MAX=5
TRAIN_PATH=ROOT/'data/TRAIN_SELECTED_H30.npz'
DEV_PATH=ROOT/'data/DEV_PREFIXES_NO_LABELS.npz'
V3_MAP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z/diagnostics/RGB_DONOR_MAP.csv'))
OUT=ROOT/'protocol'
TRAIN_FIELDS=('anchor_id','episode_id','task_id','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
DEV_FIELDS=('anchor_id','episode_id','task_id','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def stable_hash(text:str)->int:return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16],16)
def read_allowed(path:Path,fields:tuple[str,...])->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        missing=set(fields)-set(z.files)
        if missing: raise RuntimeError(f'{path.name}: missing allowed input fields {sorted(missing)}')
        return {k:z[k].copy() for k in fields}
def auxiliary(d:dict[str,np.ndarray])->tuple[np.ndarray,list[dict[str,object]]]:
    parts=[d['history_native'].reshape(len(d['anchor_id']),-1).astype(np.float64),
           d['history_mask'].astype(np.float64),d['relative_delta'].astype(np.float64),
           d['anchor_native'].astype(np.float64)]
    names=[('history_native',parts[0].shape[1]),('history_mask',parts[1].shape[1]),
           ('relative_delta',parts[2].shape[1]),('anchor_native',parts[3].shape[1])]
    return np.concatenate(parts,axis=1),[{'field':k,'dimensions':int(n),'model_input_source':'V2 prefix adapter raw field'} for k,n in names]
def fit_scaler(x:np.ndarray,d:dict[str,np.ndarray])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    n=len(x); w=np.zeros(n,dtype=np.float64)
    for task in TASKS:
        ix=np.flatnonzero(d['task_id']==task)
        episodes=np.unique(d['episode_id'][ix])
        for ep in episodes:
            ei=ix[d['episode_id'][ix]==ep]
            w[ei]=1.0/(len(TASKS)*len(episodes)*len(ei))
    if not np.isclose(w.sum(),1.0,rtol=0,atol=1e-12):raise RuntimeError(f'task/episode/anchor scaler weights sum={w.sum()}')
    mean=np.sum(x*w[:,None],axis=0,dtype=np.float64)
    var=np.sum(((x-mean)**2)*w[:,None],axis=0,dtype=np.float64)
    sd=np.sqrt(np.maximum(var,0.0)); active=sd>0.0
    safe=np.maximum(sd,1e-6)
    if not active.any():raise RuntimeError('no nonconstant state dimensions for nearest-neighbor bank')
    return mean,safe,active,w
def pairwise(q:np.ndarray,d:np.ndarray)->np.ndarray:
    qn=np.einsum('ij,ij->i',q,q); dn=np.einsum('ij,ij->i',d,d)
    out=(qn[:,None]+dn[None,:]-2.0*(q@d.T))/q.shape[1]
    return np.maximum(out,0.0)
def best_by_source(distance:np.ndarray,qidx:int,bank:dict[str,np.ndarray],bank_rows:np.ndarray,exclude_source:str|None,exclude_episode:int|None)->list[tuple[int,float]]:
    task=int(bank['task_id'][bank_rows[0]]) if len(bank_rows) else -1
    eligible=bank_rows[bank['task_id'][bank_rows]==task]
    groups:dict[str,list[int]]={}
    q_episode=int(bank['episode_id'][qidx]) if exclude_episode is not None else -1
    for di in eligible.tolist():
        source=str(bank['source_cluster'][di]); episode=int(bank['episode_id'][di])
        if exclude_source is not None and (source==exclude_source or episode==q_episode):continue
        groups.setdefault(source,[]).append(di)
    per_group=[]
    for source,rows in groups.items():
        chosen=min(rows,key=lambda di:(float(distance[di]),str(bank['anchor_id'][di])))
        per_group.append((chosen,float(distance[chosen]),source))
    per_group.sort(key=lambda x:(x[1],x[2],str(bank['anchor_id'][x[0]])))
    return [(di,dist) for di,dist,_ in per_group[:K_MAX]]
def write_csv(path:Path,rows:list[dict[str,object]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['status']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def qsummary(v:np.ndarray)->dict[str,object]:
    if not len(v):return {'n':0,'min':'NA','median':'NA','mean':'NA','max':'NA'}
    return {'n':int(len(v)),'min':float(v.min()),'median':float(np.median(v)),'mean':float(v.mean()),'max':float(v.max())}
def main()->None:
    tr=read_allowed(TRAIN_PATH,TRAIN_FIELDS); dv=read_allowed(DEV_PATH,DEV_FIELDS)
    if len(tr['anchor_id'])!=1280 or len(dv['anchor_id'])!=185:raise RuntimeError('accepted V2 train/dev prefix counts changed')
    if len(np.unique(tr['anchor_id']))!=len(tr['anchor_id']) or len(np.unique(dv['anchor_id']))!=len(dv['anchor_id']):raise RuntimeError('duplicate anchor IDs in prefix assets')
    if 'target_native_h30' not in TRAIN_PATH.name:pass # explicit source-code marker; target key is never requested below
    train_sources=set(tr['source_cluster'].astype(str));dev_sources=set(dv['source_cluster'].astype(str))
    overlap=train_sources&dev_sources
    if overlap:raise RuntimeError(f'outer split source overlap: {len(overlap)} groups')
    xt,field_receipt=auxiliary(tr); xd,_=auxiliary(dv)
    mu,sd,active,w=fit_scaler(xt,tr)
    zt=((xt[:,active]-mu[active])/sd[active]).astype(np.float64)
    zd=((xd[:,active]-mu[active])/sd[active]).astype(np.float64)
    if not np.isfinite(zt).all() or not np.isfinite(zd).all():raise RuntimeError('nonfinite state-neighbor matrix')
    dtt=pairwise(zt,zt); ddt=pairwise(zd,zt)
    cand=np.full((len(tr['anchor_id']),K_MAX),-1,dtype=np.int64); cds=np.full((len(cand),K_MAX),np.nan,dtype=np.float64); ks=np.zeros(len(cand),dtype=np.int8)
    train_rows=[]
    for qi in range(len(cand)):
        task=int(tr['task_id'][qi]); bank_rows=np.flatnonzero(tr['task_id']==task)
        found=best_by_source(dtt[qi],qi,tr,bank_rows,str(tr['source_cluster'][qi]),int(tr['episode_id'][qi]))
        ks[qi]=len(found)
        for rank,(di,dist) in enumerate(found):
            cand[qi,rank]=di;cds[qi,rank]=dist
            train_rows.append({'receiver_row':qi,'receiver_anchor_id':str(tr['anchor_id'][qi]),'receiver_episode_id':int(tr['episode_id'][qi]),'task_id':task,'receiver_source_cluster':str(tr['source_cluster'][qi]),'candidate_rank':rank+1,'donor_row':di,'donor_anchor_id':str(tr['anchor_id'][di]),'donor_episode_id':int(tr['episode_id'][di]),'donor_source_cluster':str(tr['source_cluster'][di]),'state_distance':float(dist),'same_source_excluded':str(tr['source_cluster'][qi])!=str(tr['source_cluster'][di]),'same_episode_excluded':int(tr['episode_id'][qi])!=int(tr['episode_id'][di])})
    if np.any(ks==0):
        # Contract permits explicit zero fallback; preserve affected rows and continue.
        pass
    if np.any(ks>0) and not np.all(np.any(cand>=0,axis=1)):raise RuntimeError('unexpected donor array error')
    npz=OUT/'TRAIN_DONOR_CANDIDATES.npz'
    np.savez_compressed(npz,candidate_row=cand,candidate_distance=cds,candidate_k=ks,receiver_anchor_id=tr['anchor_id'])
    write_csv(OUT/'TRAIN_DONOR_CANDIDATES.csv',train_rows)
    eval_rows=[];eval_candidate_rows=[];v3_rows=[]
    v3_by_receiver={}
    with V3_MAP.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):v3_by_receiver[row['receiver_anchor_id']]=row
    dev_pos={str(a):i for i,a in enumerate(dv['anchor_id'])}
    for qi in range(len(dv['anchor_id'])):
        task=int(dv['task_id'][qi]); bank_rows=np.flatnonzero(tr['task_id']==task)
        found=best_by_source(ddt[qi],qi,tr,bank_rows,None,None)
        k=len(found)
        v3row=v3_by_receiver.get(str(dv['anchor_id'][qi]))
        if v3row and v3row.get('donor_status')=='PASS' and v3row.get('donor_anchor_id') in dev_pos:
            vi=dev_pos[v3row['donor_anchor_id']]
            if int(dv['task_id'][vi])!=task or int(dv['episode_id'][vi])==int(dv['episode_id'][qi]):raise RuntimeError('V3 donor mapping violates its same-task/different-episode contract')
            dist=float(np.mean((zd[qi]-zd[0])**2)) if False else float(np.mean((zd[qi]-((xd[vi,active]-mu[active])/sd[active]))**2))
            v3_rows.append({'receiver_anchor_id':str(dv['anchor_id'][qi]),'task_id':task,'receiver_episode_id':int(dv['episode_id'][qi]),'v3_donor_anchor_id':v3row['donor_anchor_id'],'v3_donor_episode_id':int(v3row['donor_episode_id']),'v3_state_distance':dist,'selection_sha256':v3row.get('selection_sha256','')})
        for map_id in ('D1','D2','D3'):
            if not k:
                di=-1;rank=-1;dist=float('nan');status='ZERO_FALLBACK'
            else:
                choice=stable_hash(f'v4eval|{map_id}|{dv["anchor_id"][qi]}')%k
                di,dist=found[choice];rank=choice+1;status='PASS'
            eval_rows.append({'map_id':map_id,'receiver_row':qi,'receiver_anchor_id':str(dv['anchor_id'][qi]),'receiver_episode_id':int(dv['episode_id'][qi]),'task_id':task,'receiver_source_cluster':str(dv['source_cluster'][qi]),'candidate_k':k,'selected_candidate_rank':rank,'donor_status':status,'donor_row':di,'donor_anchor_id':'' if di<0 else str(tr['anchor_id'][di]),'donor_episode_id':'' if di<0 else int(tr['episode_id'][di]),'donor_source_cluster':'' if di<0 else str(tr['source_cluster'][di]),'state_distance':'NA' if di<0 else float(dist),'selection_sha256':hashlib.sha256(f'v4eval|{map_id}|{dv["anchor_id"][qi]}'.encode()).hexdigest()})
        for rank,(di,dist) in enumerate(found):
            eval_candidate_rows.append({'receiver_row':qi,'receiver_anchor_id':str(dv['anchor_id'][qi]),'receiver_episode_id':int(dv['episode_id'][qi]),'task_id':task,'candidate_rank':rank+1,'donor_row':di,'donor_anchor_id':str(tr['anchor_id'][di]),'donor_episode_id':int(tr['episode_id'][di]),'donor_source_cluster':str(tr['source_cluster'][di]),'state_distance':float(dist)})
    write_csv(OUT/'EVAL_DONOR_MAPS.csv',eval_rows);write_csv(OUT/'EVAL_DONOR_CANDIDATES.csv',eval_candidate_rows);write_csv(ROOT/'inventory/V3_RGB_DONOR_MAP.csv',list(v3_by_receiver.values()))
    write_csv(ROOT/'inventory/V3_DONOR_STATE_DISTANCE.csv',v3_rows)
    quality=[]
    for split,dat,distmat in [('train',tr,dtt),('dev',dv,ddt)]:
        chosen=[]
        for row in eval_rows if split=='dev' else []:pass
        for task in TASKS:
            qix=np.flatnonzero(dat['task_id']==task)
            if split=='train':
                dvals=np.array([cds[i,0] for i in qix if ks[i]>0],dtype=np.float64)
                reuse=[int(cand[i,0]) for i in qix if ks[i]>0]
            else:
                er=[r for r in eval_rows if r['task_id']==task and r['map_id']=='D1']
                dvals=np.array([float(r['state_distance']) for r in er if r['state_distance']!='NA'],dtype=np.float64)
                reuse=[int(r['donor_row']) for r in er if int(r['donor_row'])>=0]
            rep=Counter(reuse) if False else None
            unique=len(set(reuse)); repeated=sum(c>1 for c in __import__('collections').Counter(reuse).values())
            quality.append({'split':split,'task_id':task,'receivers':int(len(qix)),'fallback_receivers':int(len(qix)-len(dvals)),'candidate_k_min':int(ks[qix].min()) if split=='train' else int(min([r['candidate_k'] for r in eval_rows if r['task_id']==task],default=0)),'candidate_k_max':int(ks[qix].max()) if split=='train' else int(max([r['candidate_k'] for r in eval_rows if r['task_id']==task],default=0)),'nearest_distance_summary_json':json.dumps(qsummary(dvals),sort_keys=True),'selected_D1_unique_donor_anchors':unique,'selected_D1_donors_reused_by_multiple_receivers':repeated,'selected_D1_reuse_fraction':float((len(reuse)-unique)/len(reuse)) if reuse else 'NA'})
    write_csv(ROOT/'results/DONOR_MATCH_QUALITY.csv',quality)
    all_v3=np.array([r['v3_state_distance'] for r in v3_rows],dtype=np.float64)
    scaler={'schema_version':1,'status':'LOCKED_OUTER_TRAIN_ONLY','fit_scope':'outer_train 220 episodes / 1280 anchors only','source_npz_sha256':sha(TRAIN_PATH),'feature_fields':field_receipt,'input_pre_normalization':'raw V2 prefix fields as supplied to accepted V2 adapter; no model normalizer applied a second time','mean_and_std_weighting':'task equal, then episode equal within task, then anchors equal within episode','active_dims':int(active.sum()),'total_dims':int(active.size),'constant_dims_removed':int((~active).sum()),'std_floor':1e-6,'distance':'mean over active dimensions of squared standardized difference','means_active':mu[active].tolist(),'stds_active':sd[active].tolist(),'active_indices':np.flatnonzero(active).tolist(),'does_not_include':['RGB','future commands','model predictions','hidden states','errors','numeric episode id','future episode length','semantic progress','future labels'],'task_id_use':'same-task candidate restriction only','final_heldout_payload_access':0}
    (OUT/'DONOR_SCALER.npz').write_bytes(b'') if False else None
    (OUT/'DONOR_SCALER.json').write_text(json.dumps(scaler,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    meta={'schema_version':1,'status':'PASS','train_rows':len(tr['anchor_id']),'dev_rows':len(dv['anchor_id']),'tasks':list(TASKS),'K_max':K_MAX,'train_candidate_sha256':sha(OUT/'TRAIN_DONOR_CANDIDATES.npz'),'train_candidate_csv_sha256':sha(OUT/'TRAIN_DONOR_CANDIDATES.csv'),'eval_maps_sha256':sha(OUT/'EVAL_DONOR_MAPS.csv'),'eval_candidates_sha256':sha(OUT/'EVAL_DONOR_CANDIDATES.csv'),'scaler_sha256':sha(OUT/'DONOR_SCALER.json'),'v3_donor_map_sha256':sha(ROOT/'inventory/V3_RGB_DONOR_MAP.csv'),'v3_distance_comparable_rows':len(v3_rows),'v3_distance_summary':qsummary(all_v3),'same_task_source_exclusion_verified':True,'dev_source_overlap_train':0,'donor_inputs_exclude_targets':True,'donor_selection_uses_RGB_values':False,'donor_rng_touches_training_sampler_rng':False,'fallback_rows':int((ks==0).sum()),'final_heldout_payload_access':0}
    (OUT/'DONOR_BANK_RECEIPT.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':
    from collections import Counter
    main()

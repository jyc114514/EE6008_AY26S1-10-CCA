#!/usr/bin/env python3
"""Join frozen train predictions by identity and build fixed beta=.5 native targets."""
from __future__ import annotations
import os
import csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.torch import load_file, save_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
TRAIN=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/TRAIN_SELECTED_H30.npz'
NORM=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/normalizers/NORMALIZER_FORMAL.json'
TASKS=(119,94,90,141,230); SEEDS=(202610131,202610132,202610133); BETA=0.5
FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','anchor_native','target_native_h30')
def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def arr_sha(x:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(x).tobytes(order='C')).hexdigest()
def atomic_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def load_train()->dict[str,np.ndarray]:
    with np.load(TRAIN,allow_pickle=False) as z:
        if set(FIELDS)-set(z.files):raise RuntimeError(f'train archive missing fields: {sorted(set(FIELDS)-set(z.files))}')
        return {k:z[k].copy() for k in FIELDS}
def row_map(path:Path,data:dict[str,np.ndarray])->tuple[np.ndarray,list[dict[str,str]]]:
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    n=len(data['anchor_id']); ids=data['anchor_id'].astype(str)
    if len(rows)!=n or len({str(r['anchor_id']) for r in rows})!=n:raise RuntimeError(f'prediction rows incomplete/duplicated: {path}')
    by_id={str(r['anchor_id']):r for r in rows}
    if set(by_id)!=set(ids):raise RuntimeError(f'prediction anchor IDs differ from accepted train archive: {path}')
    order=np.empty(n,dtype=np.int64)
    for i,aid in enumerate(ids.tolist()):
        r=by_id[aid]
        if int(r['episode_id'])!=int(data['episode_id'][i]) or int(r['task_id'])!=int(data['task_id'][i]) or str(r['source_cluster'])!=str(data['source_cluster'][i]):
            raise RuntimeError(f'prediction identity/task join mismatch at anchor_id={aid}')
        order[i]=int(r['row'])
    if not np.array_equal(np.sort(order),np.arange(n)):raise RuntimeError('prediction row indices are not a one-to-one mapping')
    return order,rows
def load_prediction(role:str,seed:int,data:dict[str,np.ndarray])->tuple[np.ndarray,dict[str,Any]]:
    out=ROOT/'teacher_predictions/source'/role/f'seed{seed}'
    receipt=json.loads((out/'PREDICTION_COMPLETE.json').read_text(encoding='utf-8'))
    if receipt.get('status')!='PASS' or receipt.get('teacher_role')!=role or int(receipt.get('seed',-1))!=seed:raise RuntimeError(f'teacher replay not accepted: {role}/{seed}')
    pred_path=Path(receipt['prediction_path']); rows_path=Path(receipt['prediction_rows_path'])
    if sha(pred_path)!=receipt['prediction_sha256'] or sha(rows_path)!=receipt['prediction_rows_sha256']:raise RuntimeError(f'teacher prediction checksum mismatch: {role}/{seed}')
    tensors=load_file(str(pred_path),device='cpu')
    if 'prediction_native' not in tensors:raise RuntimeError(f'teacher native prediction field absent: {role}/{seed}')
    values=tensors['prediction_native'].numpy()
    if values.shape!=(1280,30,28) or values.dtype!=np.float64 or not np.isfinite(values).all():raise RuntimeError(f'teacher array shape/dtype/finite failure: {role}/{seed}')
    order,_=row_map(rows_path,data)
    return values[order].copy(),receipt
def metric_rows(data:dict[str,np.ndarray],pred:np.ndarray,y:np.ndarray,norm:dict[str,Any],role:str,seed:int)->list[dict[str,Any]]:
    sigma=np.asarray(norm['target_sigma'],dtype=np.float64); fdscale=np.asarray(norm['safe_fd_scale'],dtype=np.float64)
    anchor=np.asarray(data['anchor_native'],dtype=np.float64)
    p=np.abs(pred-y)/sigma[None,None,:]
    position=0.5*(p[...,:14].mean(axis=(1,2))+p[...,14:].mean(axis=(1,2)))
    df=np.diff(pred,axis=1)/fdscale[None,None,:]-np.diff(y,axis=1)/fdscale[None,None,:]
    fd=0.5*(np.abs(df[...,:14]).mean(axis=(1,2))+np.abs(df[...,14:]).mean(axis=(1,2)))
    seam=(pred[:,0]-anchor)/fdscale[None,:]-(y[:,0]-anchor)/fdscale[None,:]
    seam=0.5*(np.abs(seam[:,:14]).mean(axis=1)+np.abs(seam[:,14:]).mean(axis=1))
    by_task={}
    for task in TASKS:
        ix=np.flatnonzero(np.asarray(data['task_id'])==task)
        episodes=np.asarray(data['episode_id'])[ix]
        ep=np.unique(episodes)
        pos_ep=np.asarray([position[ix[episodes==e]].mean() for e in ep])
        fd_ep=np.asarray([fd[ix[episodes==e]].mean() for e in ep])
        seam_ep=np.asarray([seam[ix[episodes==e]].mean() for e in ep])
        by_task[task]={'position_mae':float(pos_ep.mean()),'safe_fd_l1':float(fd_ep.mean()),'seam_safe_fd_l1':float(seam_ep.mean()),'episodes':len(ep),'anchors':len(ix)}
    macro={key:float(np.mean([v[key] for v in by_task.values()])) for key in ('position_mae','safe_fd_l1','seam_safe_fd_l1')}
    rows=[{'role':role,'seed':seed,'task_id':t,**by_task[t]} for t in TASKS]
    rows.append({'role':role,'seed':seed,'task_id':'macro',**macro,'episodes':sum(v['episodes'] for v in by_task.values()),'anchors':len(y)})
    return rows
def save_csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main()->int:
    data=load_train(); n=len(data['anchor_id']);
    if n!=1280 or len(set(data['anchor_id'].astype(str)))!=n:raise RuntimeError('accepted outer-train data is not 1280 unique anchors')
    norm=json.loads(NORM.read_text(encoding='utf-8')); train_sha=sha(TRAIN); norm_sha=sha(NORM)
    sources={}; source_receipts={}; gt_rows=[]
    for seed in SEEDS:
        for role in ('S_OPT','A_COMMON','V_MATCHED'):
            pred,receipt=load_prediction(role,seed,data);sources[(role,seed)]=pred;source_receipts[(role,seed)]=receipt
            gt_rows.extend(metric_rows(data,pred,data['target_native_h30'].astype(np.float64),norm,role,seed))
        ss=0.5*sources[('S_OPT',seed)]+0.5*sources[('A_COMMON',seed)]
        sv=0.5*sources[('S_OPT',seed)]+0.5*sources[('V_MATCHED',seed)]
        if not (np.isfinite(ss).all() and np.isfinite(sv).all()):raise RuntimeError(f'nonfinite teacher composition for seed={seed}')
        for role,values in (('SS50',ss),('SV50',sv)):
            out=ROOT/'teacher_predictions/composed'/role/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
            pred_path=out/'PREDICTIONS.safetensors';tmp=pred_path.with_name(f'.{pred_path.name}.{os.getpid()}.part')
            from safetensors.numpy import save_file as save_numpy
            save_numpy({'prediction_native':np.ascontiguousarray(values,dtype=np.float64)},str(tmp));os.replace(tmp,pred_path)
            rows_path=out/'PREDICTION_ROWS.csv'
            with rows_path.open('w',newline='',encoding='utf-8') as f:
                fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for i in range(n):w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),'task_rank':int(data['task_rank'][i]),'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i])})
            gt_rows.extend(metric_rows(data,values,data['target_native_h30'].astype(np.float64),norm,role,seed))
            receipt={'schema_version':1,'status':'PASS','role':role,'seed':seed,'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),'prediction_rows_path':str(rows_path),'prediction_rows_sha256':sha(rows_path),'prediction_shape':list(values.shape),'dtype':'float64','native_command_space':True,'formula':'0.5*S+0.5*A' if role=='SS50' else '0.5*S+0.5*V','source_prediction_sha256':{r:source_receipts[(r,seed)]['prediction_sha256'] for r in (('S_OPT','A_COMMON') if role=='SS50' else ('S_OPT','V_MATCHED'))},'anchor_ids_joined_explicitly':True,'outer_train_prediction_is_oof':False,'final_heldout_payload_access':0}
            atomic_json(out/'PREDICTION_COMPLETE.json',receipt)
    all_rows=[]
    programs={
        'ST_SELF':('A_COMMON',lambda s:sources[('A_COMMON',s)]),
        'ST_SS':('SS50',lambda s:0.5*sources[('S_OPT',s)]+0.5*sources[('A_COMMON',s)]),
        'ST_SV':('SV50',lambda s:0.5*sources[('S_OPT',s)]+0.5*sources[('V_MATCHED',s)]),
        'VD_SELF':('V_MATCHED',lambda s:sources[('V_MATCHED',s)]),
        'VD_SS':('SS50',lambda s:0.5*sources[('S_OPT',s)]+0.5*sources[('A_COMMON',s)]),
        'VD_SV':('SV50',lambda s:0.5*sources[('S_OPT',s)]+0.5*sources[('V_MATCHED',s)]),
    }
    y=np.asarray(data['target_native_h30'],dtype=np.float64)
    for program,(teacher_role,teacher_fn) in programs.items():
        for seed in SEEDS:
            teacher=teacher_fn(seed); z=y+BETA*(teacher-y)
            if z.shape!=(1280,30,28) or not np.isfinite(z).all():raise RuntimeError(f'mixed target shape/finite failure: {program}/{seed}')
            out=ROOT/'teacher_targets'/program/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
            target_path=out/'Z_TRAIN_NATIVE.npy';np.save(target_path,z,allow_pickle=False)
            rows_path=out/'TARGET_ROWS.csv'
            with rows_path.open('w',newline='',encoding='utf-8') as f:
                fields=['target_row','anchor_id','episode_id','task_id','source_cluster'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for i in range(n):w.writerow({'target_row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':int(data['task_id'][i]),'source_cluster':str(data['source_cluster'][i])})
            if teacher_role in {'A_COMMON','V_MATCHED'}: teacher_hash=source_receipts[(teacher_role,seed)]['prediction_sha256']
            else: teacher_hash=sha(ROOT/'teacher_predictions/composed'/teacher_role/f'seed{seed}'/'PREDICTIONS.safetensors')
            manifest={'schema_version':1,'status':'PASS','program_id':program,'seed':seed,'teacher_role':teacher_role,'beta':BETA,
                      'target_formula':'Z=(1-beta)*Y+beta*T in native numeric command values','target_path':str(target_path),'target_sha256':sha(target_path),
                      'target_array_sha256':arr_sha(z),'target_shape':list(z.shape),'target_dtype':'float64','rows_path':str(rows_path),'rows_sha256':sha(rows_path),
                      'train_npz_path':str(TRAIN),'train_npz_sha256':train_sha,'train_target_array_sha256':arr_sha(y),'teacher_prediction_sha256':teacher_hash,
                      'normalizer_path':str(NORM),'normalizer_sha256':norm_sha,'normalizer_refit_on_Z':False,'teacher_train_predictions_are_oof':False,
                      'episode_task_ids_sha256':arr_sha(np.asarray(data['task_id'],dtype=np.int64)),'final_heldout_payload_access':0,'created_utc':utc()}
            atomic_json(out/'TARGET_MANIFEST.json',manifest)
            all_rows.append({'program_id':program,'seed':seed,'teacher_role':teacher_role,'target_path':str(target_path),'target_sha256':manifest['target_sha256'],'rows_path':str(rows_path),'rows_sha256':manifest['rows_sha256'],'teacher_prediction_sha256':teacher_hash,'train_npz_sha256':train_sha,'beta':BETA,'target_dtype':'float64'})
    save_csv(ROOT/'results/TEACHER_TRAIN_GT_METRICS.csv',gt_rows)
    save_csv(ROOT/'results/TARGET_MIX_MANIFEST.csv',all_rows)
    print(json.dumps({'status':'PASS','source_teacher_bundles':9,'composed_teacher_bundles':6,'student_target_bundles':18,'train_anchors':n,'target_dtype':'float64','dev_labels_opened':False},sort_keys=True))
    return 0
if __name__=='__main__':raise SystemExit(main())

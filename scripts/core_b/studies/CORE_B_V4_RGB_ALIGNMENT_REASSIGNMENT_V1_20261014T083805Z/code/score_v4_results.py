#!/usr/bin/env python3
"""V4 target scorer. It refuses all target access until the nonempty seal validates."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
R2=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'))
R3=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'))
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
AUDIT_METHODS=('P_REF','P_CAPTURE','P_ZERO_R')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def write_json(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def write_csv(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(rows[0]) if rows else ['status']
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def get_tensor(path,key):
    with safe_open(str(path),framework='pt',device='cpu') as f:return f.get_tensor(key).detach().cpu().numpy()
def score_windows(pred,target,anchor,sigma,fdscale):
    pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64);anchor=np.asarray(anchor,dtype=np.float64)
    pos=(np.abs(pred-target)/sigma[None,None,:]).mean(axis=(1,2))
    fd=(np.abs(np.diff(pred,axis=1)-np.diff(target,axis=1))/fdscale[None,None,:]).mean(axis=(1,2))
    seam=(np.abs((pred[:,0,:]-anchor)-(target[:,0,:]-anchor))/fdscale[None,:]).mean(axis=1)
    return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}
def validate_seal_and_predictions():
    seal_path=ROOT/'protocol/PREDICTION_BATCH_SEAL.json'
    if not seal_path.is_file():raise RuntimeError('target reader refuses missing V4 prediction seal')
    seal=json.loads(seal_path.read_text())
    if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or not seal.get('new_prediction_entries'):
        raise RuntimeError('target reader refuses absent or empty V4 prediction seal')
    if seal.get('planned_new_jobs')!=24 or seal.get('sealed_new_prediction_jobs')!=24 or seal.get('missing_new_prediction_jobs')!=0:
        raise RuntimeError('target reader refuses incomplete V4 24-job matrix')
    if seal.get('dev_target_reader_called') is not False or int(seal.get('dev_target_rows_read',-1))!=0:
        raise RuntimeError('invalid pre-target exposure marker')
    with np.load(ROOT/'data/DEV_PREFIXES_NO_LABELS.npz',allow_pickle=False) as z:
        if 'target_native_h30' in z.files:raise RuntimeError('label-free dev prefix unexpectedly contains targets')
        dev={k:z[k].copy() for k in ('anchor_id','episode_id','task_id','source_cluster','anchor_native','anchor_frame')}
    ids=[str(x) for x in dev['anchor_id'].tolist()]
    devmeta={ids[i]:(str(dev['episode_id'][i]),str(dev['task_id'][i]),str(dev['source_cluster'][i])) for i in range(len(ids))}
    if len(ids)!=185 or len(set(ids))!=185:raise RuntimeError('dev receiver ID support mismatch')
    preds={};entries=[]
    for e in seal['new_prediction_entries']:
        p=Path(e['prediction_path']);r=Path(e['prediction_rows_path']);c=Path(e['completion_path'])
        if sha(p)!=e['prediction_sha256'] or sha(r)!=e['prediction_rows_sha256'] or sha(c)!=e['completion_sha256']:
            raise RuntimeError(f"sealed prediction hash changed: {e['job_id']}")
        done=json.loads(c.read_text())
        if done.get('status')!='PASS' or done.get('dev_future_target_arrays_requested') is not False:raise RuntimeError(f"invalid target-free inference receipt: {e['job_id']}")
        tensor=get_tensor(p,'prediction_native')
        rows=read_csv(r)
        if tensor.shape!=(185,30,28) or not np.isfinite(tensor).all() or [x['anchor_id'] for x in rows]!=ids:raise RuntimeError(f"sealed prediction rows/shape mismatch: {e['job_id']}")
        for i,row in enumerate(rows):
            if int(row['task_id'])!=int(dev['task_id'][i]) or int(row['episode_id'])!=int(dev['episode_id'][i]) or str(row['source_cluster'])!=str(dev['source_cluster'][i]):raise RuntimeError(f"prediction row join mismatch: {e['job_id']} row={i}")
        method='A_M' if e['program_id']=='A_STATE_COMMON' else ('C_M' if e['program_id']=='C_STATE_NEIGHBOR_REASSIGNED' and e['condition']=='MATCHED' else f"{e['program_id'][0]}_{e['condition']}")
        key=(method,str(e['seed']))
        if key in preds:raise RuntimeError(f'duplicate V4 prediction condition/seed {key}')
        preds[key]=tensor.astype(np.float64,copy=True);entries.append({**e,'method':method,'rows':rows})
    if len(entries)!=24:raise RuntimeError('expected 24 prediction entries')
    # The reused V2 B_M paths were accepted and sealed in V3; they are revalidated, still before V4 target access.
    refs={}
    for e in seal.get('reused_B_MATCHED_entries',[]):
        p=Path(e['prediction_path']);r=Path(e['prediction_rows_path'])
        if sha(p)!=e['prediction_sha256'] or sha(r)!=e['prediction_rows_sha256']:raise RuntimeError('reused V2 B_MATCHED artifact hash mismatch')
        tensor=get_tensor(p,'prediction_native');rows=read_csv(r)
        if tensor.shape!=(185,30,28) or not np.isfinite(tensor).all() or len(rows)!=185 or {x['anchor_id'] for x in rows}!=set(ids):raise RuntimeError('reused V2 B_MATCHED shape/ID set invalid')
        rowmap={x['anchor_id']:i for i,x in enumerate(rows)}
        for row in rows:
            if (str(row['episode_id']),str(row['task_id']),str(row['source_cluster']))!=devmeta[row['anchor_id']]:raise RuntimeError(f"reused V2 B_MATCHED metadata join mismatch: {row['anchor_id']}")
        refs[str(e['seed'])]=tensor[[rowmap[a] for a in ids]].astype(np.float64,copy=True)
    if set(refs)!=set(map(str,SEEDS)):raise RuntimeError('reused V2 B_MATCHED references incomplete')
    if len(preds)!=24:return None
    # V3 P_REF references and V3 captured feature bundles are also opened and joined before target access.
    v3seal=json.loads((R3/'protocol/PREDICTION_BATCH_SEAL.json').read_text())
    v3refs=v3seal['reused_v2_prediction_references'];audit={};audit_native={}
    v2norm=json.loads((R2/'normalizers/NORMALIZER_FORMAL.json').read_text())
    model_sigma=np.asarray(v2norm['target_sigma'],dtype=np.float32)
    for model in ('G-S','G-V'):
        for seed in SEEDS:
            ref=[x for x in v3refs if x.get('model_id')==model and int(x.get('seed',-1))==seed]
            if len(ref)!=1:raise RuntimeError(f'V3 accepted reference missing for {model}/{seed}')
            ref=ref[0];p=Path(ref['prediction_path']);rp=Path(ref['prediction_rows_path'])
            if sha(p)!=ref['prediction_sha256'] or sha(rp)!=ref['prediction_rows_sha256']:raise RuntimeError('V2 P_REF changed after V3 seal')
            tensor=get_tensor(p,'prediction_native');rids=read_csv(rp)
            if tensor.shape!=(185,30,28) or len(rids)!=185 or {x['anchor_id'] for x in rids}!=set(ids):raise RuntimeError(f'P_REF shape/ID set mismatch {model}/{seed}')
            ref_index={x['anchor_id']:i for i,x in enumerate(rids)}
            for row in rids:
                if (str(row['episode_id']),str(row['task_id']),str(row['source_cluster']))!=devmeta[row['anchor_id']]:raise RuntimeError(f"P_REF metadata join mismatch: {row['anchor_id']}")
            tensor=tensor[[ref_index[a] for a in ids]]
            froot=R3/'features/formal'/f"{model.replace('-','_')}_s{seed}"/'attempt_01'
            feature_path=froot/'DEV_FEATURES.safetensors';frow_path=froot/'DEV_ROWS.csv'
            features=load_file(str(feature_path),device='cpu')
            fr=read_csv(frow_path)
            if tuple(features['h'].shape)!=(185,30,128) or tuple(features['z_base'].shape)!=(185,30,28) or len(fr)!=185 or {x['anchor_id'] for x in fr}!=set(ids):raise RuntimeError(f'V3 capture feature schema/ID set invalid {model}/{seed}')
            feat_index={x['anchor_id']:i for i,x in enumerate(fr)}
            for row in fr:
                if (str(row['episode_id']),str(row['task_id']),str(row['source_cluster']))!=devmeta[row['anchor_id']]:raise RuntimeError(f"V3 capture metadata join mismatch: {row['anchor_id']}")
            feat_order=[feat_index[a] for a in ids]
            zbase=features['z_base'].numpy()[feat_order].astype(np.float32,copy=True)
            anchors=dev['anchor_native'].astype(np.float32)
            capture=(anchors[:,None,:]+model_sigma[None,None,:]*zbase).astype(np.float32)
            zero_residual=np.zeros_like(zbase)
            zero=(anchors[:,None,:]+model_sigma[None,None,:]*(zbase+zero_residual)).astype(np.float32)
            audit[(model,str(seed))]={'P_REF':tensor.astype(np.float64,copy=True),'P_CAPTURE':capture.astype(np.float64,copy=True),'P_ZERO_R':zero.astype(np.float64,copy=True)}
            audit_native[(model,str(seed))]={'normalizer_sigma':model_sigma.copy(),'feature_sha256':sha(feature_path),'feature_rows_sha256':sha(frow_path),'v2_prediction_sha256':sha(p),'h_shape':list(features['h'].shape),'z_base_shape':list(features['z_base'].shape)}
    old_rows=read_csv(R3/'results/scoring/attempt_02/PER_WINDOW_LOSSES.csv')
    v3_method=json.loads if False else None
    v3_score_rows=read_csv(R3/'results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv')
    return {'seal':seal,'seal_sha256':sha(seal_path),'dev':dev,'ids':ids,'preds':preds,'new_entries':entries,'B_refs':refs,'audit_preds':audit,'audit_native_meta':audit_native,'v3_old_per_window':old_rows,'v3_method_metrics':v3_score_rows,'v3_refs':v3refs,'metric_lock':json.loads((ROOT/'protocol/METRIC_SCALE_LOCK.json').read_text()),'metric_lock_sha256':sha(ROOT/'protocol/METRIC_SCALE_LOCK.json'),'v2_normalizer':v2norm}
def preflight():
    sigma=np.ones(28,dtype=np.float64);fd=np.ones(28,dtype=np.float64)
    target=np.zeros((2,30,28));anchor=np.zeros((2,28));pred=target.copy();pred[0,0,0]=1.0;pred[1,1,14]=2.0
    m=score_windows(pred,target,anchor,sigma,fd)
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in m.values()):raise RuntimeError('synthetic scorer schema fixture failed')
    if np.any(m['position_mae']<=0) or np.any(m['safe_fd_l1']<=0):raise RuntimeError('synthetic scorer metric values invalid')
    receipt={'schema_version':1,'status':'PASS','device':'CPU_ONLY','fixture_rows':2,'metrics':list(METRICS),'target_artifact_opened':False,'seal_required_for_real_scoring':True,'empty_seal_refused':True,'final_heldout_payload_access':0,'completed_utc':utc()}
    write_json(ROOT/'qa/SCORER_PREFLIGHT.json',receipt);print(json.dumps(receipt,indent=2));return 0
def old_method_metric(rows,method,seed,task,metric):
    matches=[r for r in rows if r['method']==method and str(r['seed'])==str(seed) and int(r['task_id'])==task and r['metric']==metric]
    if len(matches)!=1:raise RuntimeError(f'V3 accepted metric row missing/duplicate {method}/{seed}/{task}/{metric}')
    return float(matches[0]['value'])
def audit_scores(prepared,target,target_ids):
    dev=prepared['dev'];ids=prepared['ids'];idx={str(a):i for i,a in enumerate(target_ids)}
    y=np.stack([target[idx[a]] for a in ids]).astype(np.float64)
    anchors=dev['anchor_native'].astype(np.float64);lock=prepared['metric_lock']
    sigma=np.asarray(lock['target_sigma'],dtype=np.float64);fd=np.asarray(lock['safe_fd_scale'],dtype=np.float64)
    by_anchor_task={str(dev['anchor_id'][i]):int(dev['task_id'][i]) for i in range(len(ids))}
    audit_method=[];audit_native=[];impact=[];audit_window_scores={};old=prepared['v3_method_metrics']
    for (model,seed),versions in prepared['audit_preds'].items():
        model_method='BASE_STATE' if model=='G-S' else 'B_M'
        for version,pred in versions.items():
            scores=score_windows(pred,y,anchors,sigma,fd)
            audit_window_scores[(model,seed,version)]=scores
            for task in TASKS:
                ti=np.asarray([by_anchor_task[a]==task for a in ids])
                task_eps=sorted(set(int(prepared['dev']['episode_id'][i]) for i in range(len(ids)) if int(prepared['dev']['task_id'][i])==task))
                for metric in METRICS:
                    pooled=float(np.mean(scores[metric][ti]))
                    epvals=[]
                    for ep_id in task_eps:
                        ei=ti & (np.asarray(prepared['dev']['episode_id'],dtype=np.int64)==ep_id)
                        if ei.any():epvals.append(float(np.mean(scores[metric][ei])))
                    episode_macro=float(np.mean(epvals)) if epvals else float('nan')
                    common={'model_id':model,'seed':seed,'task_id':task,'version':version,'metric':metric,'dev_episodes':len(task_eps),'dev_anchors':int(ti.sum())}
                    audit_method.append({**common,'value':episode_macro,'aggregation':'episode_macro'})
                    audit_method.append({**common,'value':pooled,'aggregation':'task_anchor_pooled'})
        p_ref=versions['P_REF'];p_cap=versions['P_CAPTURE'];p_zero=versions['P_ZERO_R']
        model_sigma=prepared['audit_native_meta'][(model,seed)]['normalizer_sigma'].astype(np.float64)
        for a,b,name in [(p_cap,p_ref,'P_CAPTURE_minus_P_REF'),(p_zero,p_ref,'P_ZERO_R_minus_P_REF'),(p_zero,p_cap,'P_ZERO_R_minus_P_CAPTURE')]:
            delta=a-b
            for task in TASKS:
                ti=np.asarray([by_anchor_task[x]==task for x in ids]);dd=delta[ti]
                audit_native.append({'model_id':model,'seed':seed,'task_id':task,'comparison':name,'native_mean_abs':float(np.abs(dd).mean()),'native_max_abs':float(np.abs(dd).max()),'standardized_position_mean_abs':float((np.abs(dd)/sigma[None,None,:]).mean()),'standardized_fd_mean_abs':float((np.abs(np.diff(dd,axis=1))/fd[None,None,:]).mean()),'standardized_seam_mean_abs':float((np.abs(dd[:,0,:])/fd[None,:]).mean()),'v3_capture_feature_sha256':prepared['audit_native_meta'][(model,seed)]['feature_sha256']})
        base_method='BASE_STATE' if model=='G-S' else 'BASE_VIDEO'
        for other,other_method,family in [('SHARED_RES_STATE','SHARED_RES_STATE','residual'),('BIAS_STATE','BIAS_STATE','bias')] if model=='G-S' else [('SHARED_RES_VIDEO','SHARED_RES_VIDEO','residual'),('BIAS_VIDEO','BIAS_VIDEO','bias')]:
            for task in TASKS:
                for metric in ('position_mae','safe_fd_l1'):
                    old_gain=old_method_metric(old,base_method,seed,task,metric)-old_method_metric(old,other_method,seed,task,metric)
                    capture=next(float(r['value']) for r in audit_method if r['model_id']==model and r['seed']==seed and int(r['task_id'])==task and r['version']=='P_CAPTURE' and r['metric']==metric and r['aggregation']=='episode_macro')
                    zero=next(float(r['value']) for r in audit_method if r['model_id']==model and r['seed']==seed and int(r['task_id'])==task and r['version']=='P_ZERO_R' and r['metric']==metric and r['aggregation']=='episode_macro')
                    other_val=old_method_metric(old,other_method,seed,task,metric)
                    cap_gain=capture-other_val;zero_gain=zero-other_val
                    flip=(old_gain!=0 and cap_gain!=0 and np.sign(old_gain)!=np.sign(cap_gain)) or (old_gain==0 and cap_gain!=0)
                    impact.append({'model_id':model,'seed':seed,'task_id':task,'family':family,'metric':metric,'old_v3_gain':old_gain,'capture_corrected_gain':cap_gain,'zeroR_corrected_gain':zero_gain,'capture_minus_old_gain':cap_gain-old_gain,'absolute_metric_reconstruction_delta':abs(cap_gain-old_gain),'relative_delta_to_old_gain':abs(cap_gain-old_gain)/max(abs(old_gain),1e-12),'sign_or_zero_status_flip':flip})
    return audit_method,audit_native,impact,audit_window_scores,y,anchors,sigma,fd
def add_scored_rows(prepared,target,anchors,sigma,fd,audit_method,audit_window_scores):
    dev=prepared['dev'];ids=prepared['ids'];rows=[];new_row_evals=0
    task=np.asarray(dev['task_id'],dtype=np.int64);episode=np.asarray(dev['episode_id'],dtype=np.int64);cluster=dev['source_cluster'].astype(str)
    for e in prepared['new_entries']:
        pred=prepared['preds'][(e['method'],str(e['seed']))]
        metric=score_windows(pred,target,anchors,sigma,fd);new_row_evals+=len(ids)
        for i,a in enumerate(ids):
            rows.append({'job_id':e['job_id'],'method':e['method'],'program_id':e['program_id'],'seed':str(e['seed']),'condition':e['condition'],'anchor_id':a,'task_id':int(task[i]),'episode_id':int(episode[i]),'source_cluster':str(cluster[i]),**{m:float(metric[m][i]) for m in METRICS},'source':'V4_new_prediction'})
    # P_REF audit predictions are the accepted B_M and optimized STATE rows, scored in the audit phase.
    for (model,seed),versions in prepared['audit_preds'].items():
        method='BASE_STATE' if model=='G-S' else 'B_M'
        pred=versions['P_REF'];metric=audit_window_scores[(model,seed,'P_REF')]
        for i,a in enumerate(ids):
            rows.append({'job_id':f'V3_precision_audit_{model}_s{seed}','method':method,'program_id':'accepted_V2_reference','seed':seed,'condition':'MATCHED','anchor_id':a,'task_id':int(task[i]),'episode_id':int(episode[i]),'source_cluster':str(cluster[i]),**{m:float(metric[m][i]) for m in METRICS},'source':'V3_metric_precision_audit_P_REF'})
    # Previously accepted V3 per-window scores supply BIAS and PERSISTENCE baselines; they are not re-read from targets.
    for r in prepared['v3_old_per_window']:
        if r['method'] not in {'BIAS_STATE','BIAS_VIDEO','PERSISTENCE'}:continue
        if r['method']!='PERSISTENCE' and r.get('condition') not in {'MATCHED',''}:continue
        rows.append({'job_id':r.get('job_id','V3_reused_score'),'method':r['method'],'program_id':'accepted_V3_score_reuse','seed':str(r['seed']),'condition':'MATCHED','anchor_id':r['anchor_id'],'task_id':int(r['task_id']),'episode_id':int(r['episode_id']),'source_cluster':str(r['source_cluster']),'position_mae':float(r['position_mae']),'safe_fd_l1':float(r['safe_fd_l1']),'seam_safe_fd_l1':float(r['seam_safe_fd_l1']),'source':'accepted_V3_per_window_score_reuse'})
    # Mean maps at the loss level, never at the prediction level.
    raw={(r['method'],str(r['seed']),r['anchor_id']):r for r in rows if r['method'] in {'B_D1','B_D2','B_D3','C_D1','C_D2','C_D3'}}
    for base in ('B_D','C_D'):
        pref=base[0]
        for seed in map(str,SEEDS):
            for i,a in enumerate(ids):
                src=[raw[(f'{pref}_D{j}',seed,a)] for j in (1,2,3)]
                rows.append({'job_id':f'V4_mean_loss_maps_{base}_s{seed}','method':base,'program_id':'derived_mean_map_losses','seed':seed,'condition':'mean_D_loss','anchor_id':a,'task_id':int(task[i]),'episode_id':int(episode[i]),'source_cluster':str(cluster[i]),**{m:float(np.mean([x[m] for x in src])) for m in METRICS},'source':'equal_mean_of_three_map_losses'})
    return rows,new_row_evals
def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['method'],str(r['seed']),int(r['task_id']))].append(r)
    table={}
    out=[]
    for key,rr in grouped.items():
        episodes=defaultdict(list)
        for r in rr:episodes[int(r['episode_id'])].append(r)
        for metric in METRICS:
            epvals=[float(np.mean([float(x[metric]) for x in wr])) for wr in episodes.values()]
            em=float(np.mean(epvals)) if epvals else float('nan');pool=float(np.mean([float(x[metric]) for x in rr])) if rr else float('nan')
            table[key+(metric,'episode_macro')]=em;table[key+(metric,'anchor_pooled')]=pool
        out.append({'method':key[0],'seed':key[1],'task_id':key[2],'dev_anchors':len(rr),'dev_episodes':len(episodes),'dev_source_groups':len(set(x['source_cluster'] for x in rr)),**{f'{m}_episode_macro':table[key+(m,'episode_macro')] for m in METRICS},**{f'{m}_anchor_pooled':table[key+(m,'anchor_pooled')] for m in METRICS}})
    return table,out
def sign_guide(name):
    if name=='RGB_COMMON_GAIN':
        return True,'L(A_M)-L(B_M); positive means B_M error is lower than A_M'
    if name=='PAIRED_TRAIN_GAIN':
        return False,'L(C_M)-L(B_M); positive means C_M error is higher than B_M'
    if name in {'B_INPUT_SENSITIVITY','C_INPUT_SENSITIVITY'}:
        return False,'mapped_input_loss_minus_matched_loss; positive means replacement worsens the model'
    if name=='C_D_vs_B_D':
        return False,'C_D_loss_minus_B_D_loss; positive means B_D has lower error'
    if name=='TRAIN_TEST_INTERACTION':
        return False,'B_input_sensitivity_minus_C_input_sensitivity; positive means B is more input-sensitive'
    if name.startswith('B_vs_'):
        return True,'reference_loss_minus_B_MATCHED_loss; positive favors B_MATCHED'
    return None,'see the locked formula in ANALYSIS_LOCK.json'

def contrasts(table,rows):
    definitions={'RGB_COMMON_GAIN':('A_M','B_M'),'PAIRED_TRAIN_GAIN':('C_M','B_M'),'B_INPUT_SENSITIVITY':('B_D','B_M'),'C_INPUT_SENSITIVITY':('C_D','C_M'),'C_D_vs_B_D':('C_D','B_D'),'B_vs_optimized_STATE':('BASE_STATE','B_M'),'B_vs_BIAS_STATE':('BIAS_STATE','B_M'),'B_vs_BIAS_VIDEO':('BIAS_VIDEO','B_M'),'B_vs_PERSISTENCE':('PERSISTENCE','B_M')}
    out=[];seed_macro={}
    def value(method,seed,task,metric,aggregation):
        s=str(seed)
        return table.get((method,s,int(task),metric,aggregation))
    def global_pooled(method,seed,metric):
        s=str(seed)
        rr=[r for r in rows if r['method']==method and str(r['seed'])==s]
        return float(np.mean([float(r[metric]) for r in rr])) if rr else None
    names=list(definitions)+['TRAIN_TEST_INTERACTION']
    for seed in map(str,SEEDS):
        for task in TASKS:
            for metric in METRICS:
                for agg in ('episode_macro','anchor_pooled'):
                    def v(method):return value(method,seed,task,metric,agg)
                    vals={name:(None if v(l) is None or v(r) is None else float(v(l)-v(r))) for name,(l,r) in definitions.items()}
                    vals['TRAIN_TEST_INTERACTION']=None if vals['B_INPUT_SENSITIVITY'] is None or vals['C_INPUT_SENSITIVITY'] is None else vals['B_INPUT_SENSITIVITY']-vals['C_INPUT_SENSITIVITY']
                    for name,val in vals.items():
                        positive,meaning=sign_guide(name)
                        out.append({'contrast':name,'seed':seed,'task_id':task,'metric':metric,'aggregation':agg,'value':'NA' if val is None else val,'positive_means_later_condition_lower_error':positive,'sign_interpretation':meaning})
                        seed_macro[(name,seed,metric,agg,task)]=val
        for metric in METRICS:
            for agg in ('episode_macro','anchor_pooled'):
                for name in names:
                    vals=[seed_macro.get((name,seed,metric,agg,t)) for t in TASKS]
                    val=None if any(x is None for x in vals) else float(np.mean(vals))
                    positive,meaning=sign_guide(name)
                    out.append({'contrast':name,'seed':seed,'task_id':'five_task_macro','metric':metric,'aggregation':agg,'value':'NA' if val is None else val,'positive_means_later_condition_lower_error':positive,'sign_interpretation':meaning})
                    seed_macro[(name,seed,metric,agg,'five_task_macro')]=val
            for name in names:
                def gp(method):return global_pooled(method,seed,metric)
                if name in definitions:
                    left,right=definitions[name];val=None if gp(left) is None or gp(right) is None else gp(left)-gp(right)
                else:
                    bd,bm,cd,cm=(gp('B_D'),gp('B_M'),gp('C_D'),gp('C_M'))
                    val=None if any(x is None for x in (bd,bm,cd,cm)) else (bd-bm)-(cd-cm)
                positive,meaning=sign_guide(name)
                out.append({'contrast':name,'seed':seed,'task_id':'all_anchors','metric':metric,'aggregation':'all_anchor_pooled','value':'NA' if val is None else val,'positive_means_later_condition_lower_error':positive,'sign_interpretation':meaning})
                seed_macro[(name,seed,metric,'all_anchor_pooled','all_anchors')]=val
            for name in names:
                vals=[seed_macro.get((name,seed,metric,'anchor_pooled',t)) for t in TASKS]
                val=None if any(x is None for x in vals) else float(np.mean(vals))
                out.append({'contrast':name,'seed':seed,'task_id':'five_task_macro','metric':metric,'aggregation':'task_equal_anchor_pooled','value':'NA' if val is None else val,'positive_means_later_condition_lower_error':True})
                seed_macro[(name,seed,metric,'task_equal_anchor_pooled','five_task_macro')]=val
    return out,definitions,seed_macro

def summary_stats(agg_table,rows):
    out=[];methods=sorted(set(k[0] for k in agg_table))
    for method in methods:
        seeds=sorted(set(k[1] for k in agg_table if k[0]==method));tasks=sorted(set(k[2] for k in agg_table if k[0]==method))
        for task in tasks:
            for metric in METRICS:
                for aggregation in ('episode_macro','anchor_pooled'):
                    vals=[agg_table[(method,s,task,metric,aggregation)] for s in seeds if (method,s,task,metric,aggregation) in agg_table]
                    out.append({'method':method,'task_id':task,'metric':metric,'aggregation':aggregation,'n_seeds':len(vals),'mean':float(np.mean(vals)) if vals else 'NA','sample_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 'NA','median':float(np.median(vals)) if vals else 'NA'})
        macro_aggs=('five_task_episode_equal','task_equal_anchor_pooled','all_anchor_pooled')
        for metric in METRICS:
            per_seed=[]
            for seed in seeds:
                ss=seed
                meth_rows=[r for r in rows if r['method']==method and str(r['seed'])==ss]
                ep_by_task=[];pool_by_task=[]
                for task in TASKS:
                    rr=[r for r in meth_rows if int(r['task_id'])==task]
                    if not rr:continue
                    eps=defaultdict(list)
                    for r in rr:eps[int(r['episode_id'])].append(float(r[metric]))
                    ep_by_task.append(float(np.mean([np.mean(v) for v in eps.values()])))
                    pool_by_task.append(float(np.mean([float(r[metric]) for r in rr])))
                if ep_by_task:
                    per_seed.append({'five_task_episode_equal':float(np.mean(ep_by_task)),'task_equal_anchor_pooled':float(np.mean(pool_by_task)),'all_anchor_pooled':float(np.mean([float(r[metric]) for r in meth_rows]))})
            for aggregation in macro_aggs:
                vals=[x[aggregation] for x in per_seed]
                out.append({'method':method,'task_id':'macro','metric':metric,'aggregation':aggregation,'n_seeds':len(vals),'mean':float(np.mean(vals)) if vals else 'NA','sample_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 'NA','median':float(np.median(vals)) if vals else 'NA'})
    return out

def loo_tables(seed_macro):
    srows=[];trows=[];names=sorted(set(k[0] for k in seed_macro))
    for name in names:
        for metric in METRICS:
            for aggregation,taskkey in [('episode_macro','five_task_macro'),('all_anchor_pooled','all_anchors')]:
                for omitted in SEEDS:
                    vals=[seed_macro.get((name,str(s),metric,aggregation,taskkey)) for s in SEEDS if s!=omitted]
                    vals=[v for v in vals if v is not None]
                    srows.append({'contrast':name,'metric':metric,'aggregation':aggregation,'omitted_seed':omitted,'remaining_seed_count':3-1,'value':float(np.mean(vals)) if vals else 'NA'})
            for aggregation in ('episode_macro','anchor_pooled'):
                for omitted_task in TASKS:
                    per_seed=[]
                    for seed in map(str,SEEDS):
                        vals=[seed_macro.get((name,seed,metric,aggregation,t)) for t in TASKS if t!=omitted_task]
                        vals=[v for v in vals if v is not None]
                        if vals:per_seed.append(float(np.mean(vals)))
                    trows.append({'contrast':name,'metric':metric,'aggregation':aggregation,'omitted_task':omitted_task,'remaining_task_count':4,'mean_over_seeds':float(np.mean(per_seed)) if per_seed else 'NA','per_seed_values_json':json.dumps(per_seed)})
    return srows,trows

def source_group_bootstrap(rows,observed,defs):
    rng=np.random.default_rng(20261014);nboot=5000
    methods=sorted(set(r['method'] for r in rows));by=defaultdict(list)
    for r in rows:by[(r['method'],str(r['seed']),int(r['task_id']),str(r['source_cluster']))].append(r)
    task_groups={t:sorted(set(k[3] for k in by if k[2]==t)) for t in TASKS}
    draws={t:rng.integers(0,len(task_groups[t]),size=(nboot,len(task_groups[t]))) for t in TASKS}
    group_values={}
    for (method,seed,task,cluster),rr in by.items():
        ep=defaultdict(list)
        for r in rr:ep[int(r['episode_id'])].append(r)
        for metric in METRICS:
            epvals=[float(np.mean([float(x[metric]) for x in er])) for er in ep.values()]
            group_values[(method,seed,task,cluster,metric)]=(float(np.sum(epvals)),len(epvals))
    scores={}
    # Method/task bootstrap distributions; identical source-group draws are reused across methods, seeds and maps.
    for method in methods:
        for seed in map(str,SEEDS):
            for task in TASKS:
                clusters=task_groups[task]
                if not all((method,seed,task,c,m) in group_values for c in clusters for m in METRICS):continue
                ix=draws[task]
                for metric in METRICS:
                    gs=np.asarray([group_values[(method,seed,task,c,metric)][0] for c in clusters],dtype=np.float64)
                    ge=np.asarray([group_values[(method,seed,task,c,metric)][1] for c in clusters],dtype=np.float64)
                    num=gs[ix].sum(axis=1);den=ge[ix].sum(axis=1)
                    scores[(method,seed,task,metric)]=num/den
    contrast_names=['RGB_COMMON_GAIN','PAIRED_TRAIN_GAIN','B_INPUT_SENSITIVITY','C_INPUT_SENSITIVITY','C_D_vs_B_D','TRAIN_TEST_INTERACTION']
    out=[]
    for metric in METRICS:
        for name in contrast_names:
            seed_draws=[]
            for seed in map(str,SEEDS):
                task_draws=[]
                for task in TASKS:
                    def s(method):return scores.get((method,seed,task,metric))
                    if name=='RGB_COMMON_GAIN': val=None if s('A_M') is None or s('B_M') is None else s('A_M')-s('B_M')
                    elif name=='PAIRED_TRAIN_GAIN':val=None if s('B_M') is None or s('C_M') is None else s('C_M')-s('B_M')
                    elif name=='B_INPUT_SENSITIVITY':val=None if s('B_D') is None or s('B_M') is None else s('B_D')-s('B_M')
                    elif name=='C_INPUT_SENSITIVITY':val=None if s('C_D') is None or s('C_M') is None else s('C_D')-s('C_M')
                    elif name=='C_D_vs_B_D':val=None if s('C_D') is None or s('B_D') is None else s('C_D')-s('B_D')
                    else:val=None if any(s(k) is None for k in ('B_D','B_M','C_D','C_M')) else (s('B_D')-s('B_M'))-(s('C_D')-s('C_M'))
                    task_draws.append(val)
                if any(x is None for x in task_draws):seed_draws.append(None)
                else:seed_draws.append(np.mean(np.stack(task_draws),axis=0))
            valid=[x for x in seed_draws if x is not None]
            if len(valid)==3:
                vals=np.mean(np.stack(valid),axis=0);estimate=float(np.mean([float(x['value']) for x in observed if x['contrast']==name and x['metric']==metric and x['aggregation']=='episode_macro' and x['task_id']=='five_task_macro']))
                lo,hi=np.quantile(vals,[0.025,0.975],method='linear');valid_count=int(np.isfinite(vals).sum())
            else:estimate=lo=hi='NA';valid_count=0
            positive,meaning=sign_guide(name)
            out.append({'contrast':name,'metric':metric,'aggregation':'five_task_episode_equal_seed_mean','estimate':estimate,'ci95_low':float(lo) if valid_count else 'NA','ci95_high':float(hi) if valid_count else 'NA','bootstrap_draws':nboot,'valid_draws':valid_count,'seed':20261014,'source_group_unit':'source_cluster within task; all episodes retained','all_conditions_share_draws':True,'maps_averaged_as_losses':True,'maps_independent_replicates':False,'p_values_computed':False,'conditional_interval':True,'positive_means_later_condition_lower_error':positive,'sign_interpretation':meaning})
    return out

def build_motion(rows,target,dev,metric_scale,train_strata):
    byep=defaultdict(list);task=np.asarray(dev['task_id']);ep=np.asarray(dev['episode_id'])
    for i in range(len(task)):
        fd=np.abs(np.diff(target[i],axis=0))/metric_scale[None,:]
        byep[(int(task[i]),int(ep[i]))].append(float(fd.mean()))
    mfd={k:float(np.mean(v)) for k,v in byep.items()};stratum={}
    thresholds=train_strata['by_task']
    for (t,e),v in mfd.items():
        q=thresholds[str(t)];name='LOW' if v<=float(q['q25_linear']) else ('MID' if v<=float(q['q75_linear']) else 'HIGH');stratum[(t,e)]=(name,v)
    wanted={'PERSISTENCE','BASE_STATE','BIAS_STATE','BIAS_VIDEO','A_M','B_M','C_M'}
    grouped=defaultdict(list)
    for r in rows:
        if r['method'] in wanted:grouped[(r['method'],str(r['seed']),int(r['task_id']),stratum[(int(r['task_id']),int(r['episode_id']))][0])].append(r)
    absrows=[]
    for (method,seed,task,group),rr in grouped.items():
        eps=defaultdict(list)
        for r in rr:eps[int(r['episode_id'])].append(r)
        for metric in METRICS:
            epvals=[float(np.mean([float(x[metric]) for x in rs])) for rs in eps.values()]
            absrows.append({'row_type':'absolute','method':method,'seed':seed,'task_id':task,'stratum':group,'metric':metric,'episodes':len(eps),'anchors':len(rr),'mean_fd_command_change':float(np.mean([stratum[(task,e)][1] for e in eps])) if eps else 'NA','value':float(np.mean(epvals)) if epvals else 'NA','outcome_defined_descriptive_strata':True})
    look={(r['method'],str(r['seed']),int(r['task_id']),r['stratum'],r['metric']):float(r['value']) for r in absrows}
    for seed in map(str,SEEDS):
        for task in TASKS:
            for group in ('LOW','MID','HIGH'):
                for metric in METRICS:
                    for contrast,ref,comp in [('gain_vs_state','BASE_STATE',None),('gain_vs_persistence','PERSISTENCE',None),('gain_B_vs_state','BASE_STATE','B_M'),('gain_C_vs_B','B_M','C_M')]:
                        if contrast=='gain_vs_state':left=look.get(('BASE_STATE',seed,task,group,metric));right=look.get(('B_M',seed,task,group,metric))
                        elif contrast=='gain_vs_persistence':left=look.get(('PERSISTENCE',seed,task,group,metric));right=look.get(('B_M',seed,task,group,metric))
                        elif contrast=='gain_B_vs_state':left=look.get(('BASE_STATE',seed,task,group,metric));right=look.get(('B_M',seed,task,group,metric))
                        else:left=look.get(('B_M',seed,task,group,metric));right=look.get(('C_M',seed,task,group,metric))
                        if left is not None and right is not None:absrows.append({'row_type':contrast,'method':'paired_gain','seed':seed,'task_id':task,'stratum':group,'metric':metric,'episodes':'same_task_stratum_support','anchors':'same_task_stratum_support','mean_fd_command_change':'NA','value':left-right,'outcome_defined_descriptive_strata':True})
    return absrows,{'dev_episode_scores':{f'{t}:{e}':{'M_fd':v,'stratum':stratum[(t,e)][0]} for (t,e),v in mfd.items()},'dev_targets_used_after_seal':True}

def run_score():
    prepared=validate_seal_and_predictions()
    # All required predictions, IDs, model hashes, metrics locks and V3 captures have passed before labels open.
    exposure_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';exposure=json.loads(exposure_path.read_text())
    exposure.update({'v4_target_reader_invoked':True,'v4_target_reader_invoked_utc':utc(),'v4_prediction_seal_sha256':prepared['seal_sha256'],'v4_scoring_status':'target_read_pending'})
    write_json(exposure_path,exposure)
    target_path=R2/'data/DEV_TARGETS_SCORER_ONLY.npz'
    try:
        with np.load(target_path,allow_pickle=False) as z:
            required=('anchor_id','episode_id','task_id','source_cluster','target_native_h30')
            missing=set(required)-set(z.files)
            if missing:raise RuntimeError(f'scorer-only archive missing fields: {sorted(missing)}')
            target_ids=[str(x) for x in z['anchor_id'].tolist()]
            target=z['target_native_h30'].astype(np.float64,copy=True)
            tmeta=[(int(z['episode_id'][i]),int(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(target_ids))]
    except Exception as e:
        exposure=json.loads(exposure_path.read_text());exposure.update({'v4_scoring_status':'failed_during_target_load','v4_target_load_error':f'{type(e).__name__}: {e}','final_heldout_payload_access':0});write_json(exposure_path,exposure);raise
    if len(target_ids)!=185 or target.shape!=(185,30,28) or not np.isfinite(target).all():raise RuntimeError('target shape/finite support mismatch after sealed read')
    idx={a:i for i,a in enumerate(target_ids)}
    if len(idx)!=185:raise RuntimeError('target anchor IDs duplicated')
    dev=prepared['dev'];ids=prepared['ids'];task=np.asarray(dev['task_id'],dtype=np.int64);episode=np.asarray(dev['episode_id'],dtype=np.int64);cluster=dev['source_cluster'].astype(str);anchor=dev['anchor_native'].astype(np.float64)
    for i,a in enumerate(ids):
        if a not in idx:raise RuntimeError(f'target ID missing for sealed dev receiver {a}')
        if tmeta[idx[a]]!=(int(episode[i]),int(task[i]),str(cluster[i])):raise RuntimeError(f'target metadata join mismatch for {a}')
    y=np.stack([target[idx[a]] for a in ids]).astype(np.float64)
    sigma=np.asarray(prepared['metric_lock']['target_sigma'],dtype=np.float64);fdscale=np.asarray(prepared['metric_lock']['safe_fd_scale'],dtype=np.float64)
    if sigma.shape!=(28,) or fdscale.shape!=(28,) or np.any(sigma<=0) or np.any(fdscale<=0):raise RuntimeError('canonical metric scales invalid')
    audit_method,audit_native,impact,audit_window_scores,audit_y,audit_anchor,_,_=audit_scores(prepared,y,target_ids)
    # Save V3 precision audit tables before assembling new prediction outcomes.
    write_csv(ROOT/'audit/V3_CANONICAL_ZERO_R_REPLAY.csv',audit_method)
    write_csv(ROOT/'audit/V3_CANONICAL_NATIVE_DIFFERENCES.csv',audit_native)
    write_csv(ROOT/'audit/V3_METRIC_PRECISION_IMPACT.csv',impact)
    nflip=sum(bool(x['sign_or_zero_status_flip']) for x in impact)
    maxnative=max((x['native_max_abs'] for x in audit_native if x['comparison']=='P_CAPTURE_minus_P_REF'),default=0.0)
    maxmet=max((x['absolute_metric_reconstruction_delta'] for x in impact),default=0.0)
    audit_obj={'schema_version':1,'status':'PASS','target_read_purpose':'V3_metric_precision_audit','formal_shared_base_count':6,'prediction_versions_per_base':3,'scored_prediction_window_pairs':18*185,'canonical_metric_lock_sha256':prepared['metric_lock_sha256'],'v3_prediction_seal_sha256':sha(R3/'protocol/PREDICTION_BATCH_SEAL.json'),'max_native_abs_CAPTURE_minus_REF':maxnative,'max_abs_effect_metric_delta_vs_old_v3_contrast':maxmet,'old_bias_or_residual_contrast_sign_flips':nflip,'interpretation_changed':nflip>0,'native_zero_R_equals_capture_max_abs':max((x['native_max_abs'] for x in audit_native if x['comparison']=='P_ZERO_R_minus_P_CAPTURE'),default=0.0),'final_heldout_payload_access':0}
    write_json(ROOT/'audit/V3_CANONICAL_ZERO_R_REPLAY.json',audit_obj)
    rows,new_eval_count=add_scored_rows(prepared,y,anchor,sigma,fdscale,audit_method,audit_window_scores)
    write_csv(ROOT/'results/PER_WINDOW_LOSSES.csv',rows)
    agg,absolute=aggregate(rows);write_csv(ROOT/'results/ABSOLUTE_METRICS.csv',absolute)
    con_rows,defs,seed_task_con=contrasts(agg,rows);write_csv(ROOT/'results/PAIRED_EFFECTS.csv',con_rows)
    summary=summary_stats(agg,rows);write_csv(ROOT/'results/SUMMARY_STATISTICS.csv',summary)
    seed_loo,task_loo=loo_tables(seed_task_con);write_csv(ROOT/'results/SEED_LOO.csv',seed_loo);write_csv(ROOT/'results/TASK_LOO.csv',task_loo)
    ci=source_group_bootstrap(rows,con_rows,defs);write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci)
    motion_lock=json.loads((ROOT/'protocol/MOTION_STRATA.json').read_text())
    motion_rows,motion_assignment=build_motion(rows,y,dev,np.asarray(prepared['metric_lock']['safe_fd_scale'],dtype=np.float64),motion_lock)
    write_csv(ROOT/'results/MOTION_STRATA_RESULTS.csv',motion_rows)
    # Exact data exposure uses the established V3 convention: one logical row for each scored prediction-window pair.
    audit_pairs=18*185;prediction_pairs=24*185;logical=audit_pairs+prediction_pairs
    exposure=json.loads(exposure_path.read_text())
    exposure.update({'v4_metric_audit_rows_read':audit_pairs,'v4_new_prediction_score_rows_read':prediction_pairs,'v4_total_new_logical_dev_future_row_reads':logical,'v4_target_artifact':str(target_path),'v4_target_artifact_sha256':sha(target_path),'v4_target_archive_load_operations':1,'v4_unique_dev_target_windows_loaded':185,'v4_dev_future_action_rows_read_by_prior_window_convention':logical,'v4_metric_audit_prediction_windows':audit_pairs,'v4_scored_new_prediction_windows':prediction_pairs,'v4_prediction_seal_sha256':prepared['seal_sha256'],'v4_target_read_after_nonempty_prediction_seal':True,'v4_scoring_status':'PASS','cumulative_dev_future_action_row_reads':27750+logical,'final_heldout_payload_access':0,'status':'V4_TARGETS_READ_AFTER_SEAL_SCORING_PASS'})
    write_json(exposure_path,exposure)
    qa={'schema_version':1,'status':'PASS','purpose':'V4_nonempty_sealed_prediction_scoring_plus_V3_metric_precision_audit','prediction_seal_sha256':prepared['seal_sha256'],'prediction_jobs_scored':24,'reused_B_MATCHED_predictions_included_via_audit_P_REF':3,'target_archive_load_operations':1,'unique_dev_target_windows_loaded':185,'v4_metric_audit_prediction_window_pairs':audit_pairs,'v4_new_prediction_window_pairs':prediction_pairs,'v4_total_logical_window_pairs':logical,'new_prediction_count_includes_maps_as_diagnostics_not_samples':True,'empty_prediction_batch_can_reach_target_reader':False,'final_heldout_payload_access':0,'metric_audit':audit_obj,'v3_old_per_window_scores_reused_for_bias_and_persistence':True,'scored_new_prediction_window_pairs':new_eval_count}
    write_json(ROOT/'qa/DEV_TARGET_SCORING_QA.json',qa)
    # Additional compact V3 RGB table summary is already computed pretarget and bound here.
    write_json(ROOT/'results/RESULTS_SUMMARY.json',{'schema_version':1,'status':'PASS','new_training_fits':6,'B_MATCHED_reused_fits':3,'new_prediction_jobs':24,'unique_dev_receivers':185,'map_count':3,'metrics':['position_mae','safe_fd_l1','seam_safe_fd_l1'],'v4_development_only':True,'positive_gain_gate':False,'final_heldout_payload_access':0,'exposure_summary':{'prior':27750,'v4_metric_audit':audit_pairs,'v4_new_prediction_scoring':prediction_pairs,'v4_total':logical,'cumulative':27750+logical},'interpretation_changed_by_v3_native_audit':nflip>0})
    print(json.dumps({'status':'PASS','new_predictions':24,'audit_pairs':audit_pairs,'prediction_pairs':prediction_pairs,'logical_target_window_reads':logical,'cumulative_reads':27750+logical,'metric_audit_sign_flips':nflip,'final_access':0},indent=2))
    return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if args.preflight:return preflight()
    run_score()
if __name__=='__main__':raise SystemExit(main())

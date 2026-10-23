#!/usr/bin/env python3
from __future__ import annotations
import os
import csv, hashlib, io, json, os, platform, resource, socket, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V3=PROJECT/'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
METRICS=['position_mae','safe_fd_l1','seam_safe_fd_l1']
METHODS=['S','V','A','C','H_SV','H_VS','H_AV','H_SC','H_PV','MIX50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE']
CONTRASTS={
 'LEVEL_TRANSFER_GAIN':('V','H_SV'),
 'HYBRID_VS_OPT_STATE':('S','H_SV'),
 'HYBRID_VS_BIAS_STATE':('BIAS_STATE','H_SV'),
 'HYBRID_VS_BIAS_VIDEO':('BIAS_VIDEO','H_SV'),
 'MATCHED_SHAPE_GAIN':('H_SC','H_SV'),
 'OPT_VS_COMMON_LEVEL_GAIN':('H_AV','H_SV'),
 'HYBRID_VS_ANCHOR_LEVEL':('H_PV','H_SV'),
 'HYBRID_VS_MIX50':('MIX50','H_SV'),
}
V4_REF_MAP={'S':'BASE_STATE','V':'B_M','A':'A_M','C':'C_M','BIAS_STATE':'BIAS_STATE','BIAS_VIDEO':'BIAS_VIDEO','PERSISTENCE':'PERSISTENCE'}
CORE_ROLES={'S':'S_OPT','V':'V_MATCHED','A':'A_COMMON','C':'C_REASSIGNED'}

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def write_csv(p,rows,fields=None):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        if not fields: fields=['status']
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def target_score(pred,y,anchor,sigma,fdscale):
    pred=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64);anchor=np.asarray(anchor,dtype=np.float64)
    if pred.shape!=y.shape or pred.ndim!=3 or pred.shape[1:]!=(30,28):raise ValueError('score tensor must be aligned [N,30,28]')
    pos=(np.abs(pred-y)/sigma[None,None,:]).mean(axis=(1,2))
    fd=(np.abs(np.diff(pred,axis=1)-np.diff(y,axis=1))/fdscale[None,None,:]).mean(axis=(1,2))
    seam=(np.abs((pred[:,0,:]-anchor)-(y[:,0,:]-anchor))/fdscale[None,:]).mean(axis=1)
    return {'position_mae':pos,'safe_fd_l1':fd,'seam_safe_fd_l1':seam}
def aggregate_rows(rows,metric):
    by_task={}
    for task in TASKS:
        rr=[r for r in rows if int(r['task_id'])==task]
        eps={}
        for r in rr:eps.setdefault(str(r['episode_id']),[]).append(float(r[metric]))
        by_task[task]={'episode_macro':float(np.mean([np.mean(v) for v in eps.values()])) if eps else np.nan,
                       'anchor_pooled':float(np.mean([float(r[metric]) for r in rr])) if rr else np.nan,
                       'anchors':len(rr),'episodes':len(eps),'source_groups':len(set(r['source_group'] for r in rr))}
    return by_task
def bootstrap_group_stats(win_rows, metric, draws_by_task):
    """Return shared-draw source-group bootstrap values for one task/seed outcome."""
    output={}
    for task in TASKS:
        rr=[r for r in win_rows if int(r['task_id'])==task]
        if not rr:
            output[task]=None;continue
        by_episode={}
        for r in rr:by_episode.setdefault(str(r['episode_id']),[]).append(float(r[metric]))
        group_episode={}
        for r in rr:group_episode.setdefault(str(r['source_group']),set()).add(str(r['episode_id']))
        groups=sorted(group_episode)
        group_sums=np.asarray([sum(float(np.mean(by_episode[e])) for e in sorted(group_episode[g])) for g in groups],dtype=np.float64)
        group_counts=np.asarray([len(group_episode[g]) for g in groups],dtype=np.float64)
        idx=draws_by_task[task]
        num=group_sums[idx].sum(axis=1)
        den=group_counts[idx].sum(axis=1)
        vals=num/den
        point=float(np.mean([np.mean(v) for v in by_episode.values()]))
        output[task]={'draws':vals,'point':point,'n_groups':len(groups),'n_episodes':len(by_episode)}
    return output

def main():
    t0=time.perf_counter()
    seal_path=ROOT/'qa/PREDICTION_SEAL.json'
    if not seal_path.is_file():raise RuntimeError('target reader refuses missing prediction seal')
    seal=json.loads(seal_path.read_text())
    if seal.get('status')!='PASS' or seal.get('nonempty_seal') is not True or int(seal.get('sealed_bundles',-1))!=18 or int(seal.get('missing_bundles',-1))!=0:
        raise RuntimeError('target reader refuses absent, empty, or incomplete prediction seal')
    if int(seal.get('attempt',0))!=2:raise RuntimeError('unexpected prediction attempt in seal')
    formula_path=ROOT/'protocol/FORMULA_LOCK.json';formula=json.loads(formula_path.read_text());formula_sha=sha_file(formula_path)
    if seal.get('formula_lock_sha256')!=formula_sha:raise RuntimeError('formula lock changed after prediction seal')
    contrast=json.loads((ROOT/'protocol/CONTRAST_LOCK.json').read_text())
    if contrast.get('formula_lock_sha256')!=formula_sha:raise RuntimeError('contrast lock not bound to formula lock')
    ordered_rows=read_csv(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv')
    ids=[r['anchor_id'] for r in sorted(ordered_rows,key=lambda x:int(x['canonical_row_index']))]
    idhash=sha_bytes(('\n'.join(ids)+'\n').encode())
    if len(ids)!=185 or len(set(ids))!=185 or idhash!=formula.get('ordered_receiver_ids_sha256'):raise RuntimeError('ordered receiver ID lock invalid')
    # Revalidate the complete prediction matrix, its ID sidecars, and finite values before opening any target.
    derived={};bundle_meta={}
    for e in seal['entries']:
        if int(e['attempt'])!=2:raise RuntimeError('unaccepted bundle attempt')
        pp=Path(e['prediction_path']);rp=Path(e['prediction_rows_path']);cp=Path(e['completion_path'])
        if sha_file(pp)!=e['prediction_sha256'] or sha_file(rp)!=e['prediction_rows_sha256'] or sha_file(cp)!=e['completion_sha256']:
            raise RuntimeError(f'sealed file hash changed for {e["bundle_id"]}')
        done=json.loads(cp.read_text())
        if done.get('status')!='PASS' or done.get('target_input_to_transform') is not False or done.get('future_labels_used_in_forward') is not False:
            raise RuntimeError(f'bad target-free completion {e["bundle_id"]}')
        arr=np.asarray(load_file(str(pp))['prediction_native'])
        rows=read_csv(rp);rowids=[r['anchor_id'] for r in rows]
        if arr.shape!=(185,30,28) or arr.dtype!=np.float64 or not np.isfinite(arr).all() or rowids!=ids:
            raise RuntimeError(f'prediction schema/ID validation failed {e["bundle_id"]}')
        ah=sha_bytes(np.asarray(arr,dtype='<f8',order='C').tobytes(order='C'))
        if ah!=done.get('canonical_array_sha256'):raise RuntimeError(f'canonical array hash mismatch {e["bundle_id"]}')
        key=(e['method'],int(e['seed']))
        if key in derived:raise RuntimeError(f'duplicate derived bundle {key}')
        derived[key]=arr;bundle_meta[key]=e
    if len(derived)!=18 or {(m,s) for m in ['H_SV','H_VS','H_AV','H_SC','H_PV','MIX50'] for s in SEEDS}!=set(derived):
        raise RuntimeError('derived prediction grid incomplete')
    # Record the exact pre-open state before the sole target archive read.
    exposure_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';exposure=json.loads(exposure_path.read_text())
    prior_target_loads=int(exposure.get('target_archive_load_operations_total',0))
    if exposure.get('status')=='PREDICTIONS_SEALED_PRE_SCORE':
        if prior_target_loads!=0 or int(exposure.get('v5_preseal_target_archive_opens',-1))!=0: raise RuntimeError('preseal exposure ledger invalid')
    elif exposure.get('status')=='RETRY_REQUIRED_AFTER_RECORDED_POSTSEAL_READ':
        if prior_target_loads not in (1,2,3) or int(exposure.get('failed_attempt_target_array_window_rows_loaded',-1))!=185*prior_target_loads: raise RuntimeError('recorded failed scorer reads are incomplete')
    else: raise RuntimeError(f'unexpected exposure state before target read: {exposure.get("status")}')
    exposure.update({'status':'TARGET_READ_PENDING','target_access_retry_started_utc':utc() if prior_target_loads else None,
        'target_archive_loads_this_v5':prior_target_loads,'target_archive_load_operations_total':prior_target_loads,'new_dev_future_window_logical_reads':0})
    write_json(exposure_path,exposure)

    # Resolve and validate label-free anchors before the sole target read for this attempt.
    source_map=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())
    prefix_rec=next(x for x in source_map['source_artifacts'] if Path(x['path']).name=='DEV_PREFIXES_NO_LABELS.npz')
    prefix_path=Path(prefix_rec['path'])
    if sha_file(prefix_path)!=prefix_rec['sha256']: raise RuntimeError('locked label-free prefix hash changed')
    with np.load(prefix_path,allow_pickle=False) as z:
        prefix_ids=[str(x) for x in z['anchor_id'].tolist()]
        anchor_by_id={str(z['anchor_id'][i]):np.asarray(z['anchor_native'][i],dtype=np.float64) for i in range(len(prefix_ids))}
        prefix_meta={str(z['anchor_id'][i]):(str(z['episode_id'][i]),str(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(prefix_ids))}
    if len(prefix_ids)!=185 or set(prefix_ids)!=set(ids): raise RuntimeError('locked label-free prefix IDs invalid')

    # One physical archive read per recorded attempt after the complete prediction seal; parse from memory.
    target_path=V2/'data/DEV_TARGETS_SCORER_ONLY.npz'
    target_blob=target_path.read_bytes();target_sha=sha_bytes(target_blob);target_blob_size=len(target_blob)
    with np.load(io.BytesIO(target_blob),allow_pickle=False) as z:
        required=('anchor_id','episode_id','task_id','source_cluster','target_native_h30')
        missing=set(required)-set(z.files)
        if missing:raise RuntimeError(f'scorer-only archive missing fields: {sorted(missing)}')
        target_ids=[str(x) for x in z['anchor_id'].tolist()]
        target_native=np.asarray(z['target_native_h30'],dtype=np.float64).copy()
        target_meta={target_ids[i]:(str(z['episode_id'][i]),str(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(target_ids))}
    if len(target_ids)!=185 or len(set(target_ids))!=185 or set(target_ids)!=set(ids):raise RuntimeError('scorer-only target receiver IDs mismatch')
    if any(target_meta[a]!=prefix_meta[a] for a in ids):raise RuntimeError('target metadata does not match locked receiver IDs')
    tidx={a:i for i,a in enumerate(target_ids)}
    target=np.stack([target_native[tidx[a]] for a in ids])
    anchors=np.stack([anchor_by_id[a] for a in ids])
    if target.shape!=(185,30,28) or not np.isfinite(target).all() or not np.isfinite(anchors).all():raise RuntimeError('target/current-anchor shape or finite validation failed')
    metric_path=Path(json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())['metric_scale_lock']['path'])
    metric=json.loads(metric_path.read_text())
    sigma=np.asarray(metric['target_sigma'],dtype=np.float64);fdscale=np.asarray(metric['safe_fd_scale'],dtype=np.float64)
    if sigma.shape!=(28,) or fdscale.shape!=(28,) or np.any(sigma<=0) or np.any(fdscale<=0):raise RuntimeError('train-only metric scales invalid')

    metadata={a:prefix_meta[a] for a in ids}
    derived_rows={};window_loss={}
    loss_table=ROOT/'results/PER_WINDOW_DERIVED_LOSSES.csv'
    prior_failure_path=ROOT/'qa/SCORER_ATTEMPT_02_FAILURE.json'
    derived_score_evaluations_this_attempt=0
    if loss_table.is_file() and prior_failure_path.is_file():
        prior_failure=json.loads(prior_failure_path.read_text())
        loss_table_sha=sha_file(loss_table)
        if prior_failure.get('status')!='FAILED_AFTER_COMPLETE_DERIVED_WINDOW_SCORING_BEFORE_AGGREGATION' or prior_failure.get('derived_window_loss_table_sha256')!=loss_table_sha or prior_failure.get('prediction_seal_sha256')!=sha_file(seal_path) or prior_failure.get('formula_lock_sha256')!=formula_sha:
            raise RuntimeError('preserved target-scored rows are not bound to the current seal/formula')
        saved=read_csv(loss_table)
        if len(saved)!=3330:raise RuntimeError('preserved derived score table is incomplete')
        for r in saved:
            key_int=(r['method'],int(r['seed']))
            if key_int not in derived or r['anchor_id'] not in set(ids):raise RuntimeError('preserved derived score row has unknown method/seed/ID')
            key=(r['method'],str(int(r['seed'])))
            if (r['episode_id'],r['task_id'],r['source_group'])!=metadata[r['anchor_id']]:raise RuntimeError('preserved derived score row metadata mismatch')
            for metric_name in METRICS:r[metric_name]=float(r[metric_name])
            if any(not np.isfinite(r[m]) for m in METRICS):raise RuntimeError('preserved derived score contains nonfinite value')
            derived_rows.setdefault(key,[]).append(r);window_loss[(key[0],key[1],r['anchor_id'])]=r
        expected={(m,str(seed)) for m in ['H_SV','H_VS','H_AV','H_SC','H_PV','MIX50'] for seed in SEEDS}
        if set(derived_rows)!=expected or any(len(v)!=185 for v in derived_rows.values()):raise RuntimeError('preserved derived score table does not contain the full locked matrix')
        derived_score_evaluations_preserved=3330
    else:
        for (method,seed),pred in derived.items():
            scores=target_score(pred,target,anchors,sigma,fdscale)
            rec=[]
            for i,a in enumerate(ids):
                ep,task,group=metadata[a]
                row={'method':method,'seed':str(seed),'anchor_id':a,'episode_id':ep,'task_id':task,'source_group':group,
                     'position_mae':float(scores['position_mae'][i]),'safe_fd_l1':float(scores['safe_fd_l1'][i]),'seam_safe_fd_l1':float(scores['seam_safe_fd_l1'][i]),
                     'source':'V5_derived_prediction_target_score'}
                rec.append(row);window_loss[(method,str(seed),a)]=row
            derived_rows[(method,str(seed))]=rec
        if sum(len(v) for v in derived_rows.values())!=3330:raise RuntimeError('new derived window loss evaluation count is not 3330')
        write_csv(loss_table,[r for key in sorted(derived_rows) for r in derived_rows[key]])
        derived_score_evaluations_this_attempt=3330
        derived_score_evaluations_preserved=0
    if sum(len(v) for v in derived_rows.values())!=3330:raise RuntimeError('derived window score table is not complete')

    # Reuse the target-free preflighted V4 reference copy; no V4 raw score table is reopened here.
    preflight=json.loads((ROOT/'qa/SCORER_RESUME_PREFLIGHT.json').read_text())
    recon_proof=json.loads((ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json').read_text())
    v4_pw_path=Path(preflight['v4_per_window_path']);v4_pw_hash=preflight['v4_per_window_sha256']
    ref_path=Path(preflight['accepted_reference_copy_path'])
    if preflight.get('status')!='PASS' or preflight.get('formula_lock_sha256')!=formula_sha or preflight.get('prediction_seal_sha256')!=sha_file(seal_path):
        raise RuntimeError('target-free source-score resume preflight is not bound to current locks')
    if recon_proof.get('v4_per_window_sha256')!=v4_pw_hash or sha_file(ref_path)!=preflight['accepted_reference_copy_sha256']:
        raise RuntimeError('preflighted V4 source-score copy/proof changed')
    copied=read_csv(ref_path);selected=len(copied);parsed=0;excluded=0
    if selected!=7*3*185 or recon_proof.get('status')!='PASS':raise RuntimeError('preflighted accepted V4 reference copy incomplete')
    ref_rows={m:[] for m in V4_REF_MAP}
    for r in copied:
        method=r['method']
        if method not in ref_rows:raise RuntimeError(f'preflight source copy has unaccepted method {method}')
        row={'method':method,'seed':str(r['seed']),'anchor_id':r['anchor_id'],'episode_id':str(r['episode_id']),
            'task_id':str(r['task_id']),'source_group':str(r['source_group']),'position_mae':float(r['position_mae']),
            'safe_fd_l1':float(r['safe_fd_l1']),'seam_safe_fd_l1':float(r['seam_safe_fd_l1']),'source':'accepted_V4_per_window_loss_reuse'}
        if row['anchor_id'] not in metadata or (row['episode_id'],row['task_id'],row['source_group'])!=metadata[row['anchor_id']]:
            raise RuntimeError(f'preflight source row metadata mismatch {method}/{row["seed"]}/{row["anchor_id"]}')
        ref_rows[method].append(row)
    for method in V4_REF_MAP:
        for seed in SEEDS:
            rr=[r for r in ref_rows[method] if r['seed']==str(seed)]
            if len(rr)!=185 or {r['anchor_id'] for r in rr}!=set(ids):raise RuntimeError(f'accepted V4 reference incomplete {method}/{seed}')
    all_rows={m:{} for m in METHODS}
    for method in V4_REF_MAP:
        for seed in SEEDS:all_rows[method][str(seed)]=[r for r in ref_rows[method] if r['seed']==str(seed)]
    for (method,seed),rr in derived_rows.items():all_rows[method][str(seed)]=rr
    reconcile=read_csv(ROOT/'results/SOURCE_SCORE_RECONCILIATION.csv')
    for r in reconcile:r['abs_delta']=float(r['abs_delta']);r['pass']=str(r['pass']).lower()=='true'
    if len(reconcile)!=210 or not all(r['pass'] for r in reconcile):raise RuntimeError('preflight source-score reconciliation invalid')
    v4_primary_rows_count=15

    # Absolute task-by-seed estimates, anchor-pooled supplements, and macro tables.
    task_agg={}; macro={}
    for method in METHODS:
        for seed in SEEDS:
            task_agg[(method,str(seed))]={}
            for metric_name in METRICS:
                vals=aggregate_rows(all_rows[method][str(seed)],metric_name)
                for task in TASKS:task_agg[(method,str(seed))].setdefault(task,{})[metric_name]=vals[task]
            for task in TASKS:
                task_agg[(method,str(seed))][task]['support']=aggregate_rows(all_rows[method][str(seed)],'position_mae')[task]
            macro[(method,str(seed))]={}
            for metric_name in METRICS:
                macro[(method,str(seed))][metric_name]=float(np.mean([task_agg[(method,str(seed))][t][metric_name]['episode_macro'] for t in TASKS]))
                macro[(method,str(seed))][metric_name+'_anchor_pooled_task_equal']=float(np.mean([task_agg[(method,str(seed))][t][metric_name]['anchor_pooled'] for t in TASKS]))
    primary_rows=[]
    for task in TASKS:
        for seed in SEEDS:
            support=next(x for x in json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())['fold_membership']['task_support'] if x['task_id']==task)
            row={'task_id':task,'seed':seed,'aggregation':'episode_equal_within_task','anchors':support['anchors'],'episodes':support['episodes'],'source_groups':support['source_groups'],
                 'persistence_seed_role':'shared_reference'}
            for method in METHODS:
                for metric_name in METRICS:
                    v=task_agg[(method,str(seed))][task][metric_name]['episode_macro'];row[f'{method}_{metric_name}']=v
                    row[f'{method}_{metric_name}_anchor_pooled']=task_agg[(method,str(seed))][task][metric_name]['anchor_pooled']
            primary_rows.append(row)
    write_csv(ROOT/'results/PRIMARY_TASK_SEED.csv',primary_rows)
    macro_rows=[]
    for method in METHODS:
        seeds=['shared'] if method=='PERSISTENCE' else list(map(str,SEEDS))
        for seed in seeds:
            srcseed=str(SEEDS[0]) if seed=='shared' else seed
            row={'method':method,'seed':seed,'seed_role':'shared_reference' if method=='PERSISTENCE' else 'formal_model_seed',
                 'aggregation':'five_task_equal_episode_macro'}
            for metric_name in METRICS:row[metric_name]=macro[(method,srcseed)][metric_name]
            macro_rows.append(row)
        for metric_name in METRICS:
            vals=[macro[(method,str(seed))][metric_name] for seed in SEEDS]
            macro_rows.append({'method':method,'seed':'mean_across_formal_seeds' if method!='PERSISTENCE' else 'shared','seed_role':'three_formal_seeds' if method!='PERSISTENCE' else 'shared_reference',
                'aggregation':'mean_and_sample_sd_across_seeds' if method!='PERSISTENCE' else 'not_applicable_shared','metric':metric_name,
                'mean':float(np.mean(vals)) if method!='PERSISTENCE' else vals[0],'sample_sd':float(np.std(vals,ddof=1)) if method!='PERSISTENCE' else 'NA','n_seeds':3 if method!='PERSISTENCE' else 1})
    write_csv(ROOT/'results/FIVE_TASK_MACRO.csv',macro_rows)

    # Paired gains use the locked sign L(reference)-L(candidate), paired on same task/seed/anchor.
    pair_rows=[];gain_window={}
    for name,(reference,candidate) in CONTRASTS.items():
        for seed in SEEDS:
            rmap={r['anchor_id']:r for r in all_rows[reference][str(seed)]};cmap={r['anchor_id']:r for r in all_rows[candidate][str(seed)]}
            if set(rmap)!=set(cmap):raise RuntimeError(f'contrast ID mismatch {name}/{seed}')
            rec=[]
            for a in ids:
                rr=rmap[a];cc=cmap[a]
                if (rr['episode_id'],rr['task_id'],rr['source_group'])!=(cc['episode_id'],cc['task_id'],cc['source_group']):raise RuntimeError(f'contrast metadata mismatch {name}/{seed}/{a}')
                ep,task,group=metadata[a]
                row={'contrast':name,'reference_method':reference,'candidate_method':candidate,'seed':seed,'anchor_id':a,'episode_id':ep,'task_id':int(task),'source_group':group}
                for metric_name in METRICS:row[metric_name]=float(rr[metric_name])-float(cc[metric_name])
                rec.append(row);gain_window[(name,str(seed),a)]=row
            for task in TASKS:
                rr=[r for r in rec if int(r['task_id'])==task]
                eps={}
                for r in rr:eps.setdefault(r['episode_id'],[]).append(r)
                out={'contrast':name,'reference_method':reference,'candidate_method':candidate,'seed':seed,'task_id':task,
                     'anchors':len(rr),'episodes':len(eps),'source_groups':len(set(r['source_group'] for r in rr)),'aggregation':'episode_equal_within_task'}
                for m in METRICS:out[m]=float(np.mean([np.mean([x[m] for x in vv]) for vv in eps.values()]))
                pair_rows.append(out)
    write_csv(ROOT/'results/PAIRED_EFFECTS.csv',pair_rows)

    # True seed and task leave-one-out summaries; shared persistence has no fake seed LOO.
    seed_loo=[]
    for method in METHODS:
        if method=='PERSISTENCE':continue
        for omit in SEEDS:
            remain=[s for s in SEEDS if s!=omit]
            for m in METRICS:
                vals=[macro[(method,str(s))][m] for s in remain]
                seed_loo.append({'analysis':'absolute','method_or_contrast':method,'metric':m,'omitted_seed':omit,'remaining_seeds':','.join(map(str,remain)),
                    'n_remaining_seeds':2,'loo_five_task_macro':float(np.mean(vals))})
    pair_macro={}
    for name,(reference,candidate) in CONTRASTS.items():
        for seed in SEEDS:
            pmap={(int(r['task_id']),int(r['seed'])):r for r in pair_rows if r['contrast']==name}
            pair_macro[(name,str(seed))]={m:float(np.mean([pmap[(t,seed)][m] for t in TASKS])) for m in METRICS}
        for omit in SEEDS:
            remain=[s for s in SEEDS if s!=omit]
            for m in METRICS:
                vals=[pair_macro[(name,str(s))][m] for s in remain]
                seed_loo.append({'analysis':'paired_gain','method_or_contrast':name,'metric':m,'omitted_seed':omit,'remaining_seeds':','.join(map(str,remain)),
                    'n_remaining_seeds':2,'loo_five_task_macro':float(np.mean(vals))})
    write_csv(ROOT/'results/SEED_LOO.csv',seed_loo)
    task_loo=[]
    for method in METHODS:
        seeds=['shared'] if method=='PERSISTENCE' else list(map(str,SEEDS))
        for seed in seeds:
            srcseed=str(SEEDS[0]) if seed=='shared' else seed
            for omitted in TASKS:
                remaining=[t for t in TASKS if t!=omitted]
                for m in METRICS:
                    vals=[task_agg[(method,srcseed)][t][m]['episode_macro'] for t in remaining]
                    task_loo.append({'analysis':'absolute','method_or_contrast':method,'metric':m,'seed':seed,'omitted_task':omitted,
                        'remaining_tasks':','.join(map(str,remaining)),'n_remaining_tasks':4,'loo_macro':float(np.mean(vals))})
    for name in CONTRASTS:
        for seed in SEEDS:
            pmap={(int(r['task_id']),int(r['seed'])):r for r in pair_rows if r['contrast']==name}
            for omitted in TASKS:
                remaining=[t for t in TASKS if t!=omitted]
                for m in METRICS:
                    vals=[float(pmap[(t,seed)][m]) for t in remaining]
                    task_loo.append({'analysis':'paired_gain','method_or_contrast':name,'metric':m,'seed':seed,'omitted_task':omitted,
                        'remaining_tasks':','.join(map(str,remaining)),'n_remaining_tasks':4,'loo_macro':float(np.mean(vals))})
    write_csv(ROOT/'results/TASK_LOO.csv',task_loo)

    # Paired source-group bootstrap with one shared set of within-task draws for all methods/seeds.
    draws_rng=np.random.default_rng(20261015);draws_by_task={};draw_hashes={}
    source_map=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())
    for task in TASKS:
        groups=sorted(set(metadata[a][2] for a in ids if int(metadata[a][1])==task))
        if not groups:draws_by_task[task]=None;draw_hashes[str(task)]={'groups':0,'sha256':None};continue
        d=draws_rng.integers(0,len(groups),size=(5000,len(groups)),dtype=np.int32)
        draws_by_task[task]=d
        draw_hashes[str(task)]={'groups':len(groups),'group_ids':groups,'sha256':sha_bytes(np.asarray(d,dtype='<i4',order='C').tobytes())}
    contrast_boot={}
    ci_rows=[]
    for name in CONTRASTS:
        for seed in SEEDS:
            rec=[gain_window[(name,str(seed),a)] for a in ids]
            for m in METRICS:
                boot=bootstrap_group_stats(rec,m,draws_by_task)
                for task in TASKS:
                    b=boot[task]
                    if b is None or b['n_groups']<2:
                        ci_rows.append({'contrast':name,'metric':m,'seed':seed,'task_id':task,'aggregation':'within_task','n_source_groups':0 if b is None else b['n_groups'],
                            'draws':5000,'ci_level':0.95,'ci_low':'NA','ci_high':'NA','point_estimate':'NA','interval':'NA_single_or_empty_group'})
                    else:
                        vals=b['draws'];ci_rows.append({'contrast':name,'metric':m,'seed':seed,'task_id':task,'aggregation':'within_task','n_source_groups':b['n_groups'],
                            'draws':5000,'ci_level':0.95,'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),
                            'point_estimate':b['point'],'interval':'percentile_source_group_bootstrap'})
                if any(boot[t] is None for t in TASKS): macro_draw=np.full(5000,np.nan)
                else: macro_draw=np.mean(np.stack([boot[t]['draws'] for t in TASKS]),axis=0)
                point_macro=float(np.mean([boot[t]['point'] for t in TASKS]))
                contrast_boot[(name,str(seed),m)]=macro_draw
                ci_rows.append({'contrast':name,'metric':m,'seed':seed,'task_id':'five_task_macro','aggregation':'five_task_equal_within_task_source_group',
                    'n_source_groups':sum(boot[t]['n_groups'] for t in TASKS if boot[t] is not None),'draws':5000,'ci_level':0.95,
                    'ci_low':float(np.nanquantile(macro_draw,.025)),'ci_high':float(np.nanquantile(macro_draw,.975)),'point_estimate':point_macro,'interval':'percentile_source_group_bootstrap'})
        for m in METRICS:
            seed_draw=np.mean(np.stack([contrast_boot[(name,str(s),m)] for s in SEEDS]),axis=0)
            seed_point=float(np.mean([pair_macro[(name,str(s))][m] for s in SEEDS]))
            ci_rows.append({'contrast':name,'metric':m,'seed':'mean_across_formal_seeds','task_id':'five_task_macro','aggregation':'five_task_macro_mean_over_three_fixed_seeds',
                'n_source_groups':sum(draw_hashes[str(t)]['groups'] for t in TASKS),'draws':5000,'ci_level':0.95,
                'ci_low':float(np.nanquantile(seed_draw,.025)),'ci_high':float(np.nanquantile(seed_draw,.975)),'point_estimate':seed_point,
                'interval':'percentile_source_group_bootstrap_conditional_on_three_fixed_seeds'})
    write_csv(ROOT/'results/PAIRED_GROUP_CI.csv',ci_rows)
    write_json(ROOT/'qa/BOOTSTRAP_DRAW_MANIFEST.json',{'schema_version':1,'status':'PASS','rng':'numpy.default_rng','seed':20261015,'draws':5000,
        'same_draw_matrices_shared_across_all_contrasts_and_seeds':True,'within_task_source_groups':draw_hashes,
        'interval':'percentile 95%','p_values':False,'created_utc':utc()})

    # L2 diagnostic decomposition; it is separate from L1 MAE and uses train-only target_sigma.
    source_entries={(x['role'],int(x['seed'])):x for x in source_map['source_prediction_bundles']}
    l2_methods=['S','V','A','C','H_SV'];l2_window=[];l2_task_rows=[];identity_max={'level_H_SV_minus_S':0.0,'shape_H_SV_minus_V':0.0,'position_minus_level_shape':0.0}
    core_arrays={}
    for seed in SEEDS:
        for method,role in CORE_ROLES.items():
            source=source_entries[(role,seed)]
            arr=np.asarray(load_file(source['prediction_path'])['prediction_native'],dtype=np.float64)
            source_rows=read_csv(source['prediction_rows_path']);idx={r['anchor_id']:i for i,r in enumerate(source_rows)}
            core_arrays[(method,str(seed))]=np.stack([arr[idx[a]] for a in ids])
        # Score source L2 and derived H_SV in standardized train target units.
        for method in l2_methods:
            pred=derived[(method,seed)] if method=='H_SV' else core_arrays[(method,str(seed))]
            x=pred/sigma[None,None,:];y=target/sigma[None,None,:]
            xm=x.mean(axis=1,keepdims=True);ym=y.mean(axis=1,keepdims=True)
            level=((xm[:,0,:]-ym[:,0,:])**2).mean(axis=1)
            shape=(((x-xm)-(y-ym))**2).mean(axis=(1,2))
            pos=((x-y)**2).mean(axis=(1,2))
            for i,a in enumerate(ids):
                ep,task,group=metadata[a]
                l2_window.append({'method':method,'seed':seed,'anchor_id':a,'episode_id':ep,'task_id':task,'source_group':group,
                    'level_mse':float(level[i]),'shape_mse':float(shape[i]),'position_mse':float(pos[i]),'decomposition_abs_error':abs(float(pos[i]-level[i]-shape[i]))})
            if method=='H_SV':
                S=core_arrays[('S',str(seed))];V=core_arrays[('V',str(seed))]
                xs=S/sigma[None,None,:];xv=V/sigma[None,None,:]
                sm=((xs.mean(axis=1)-y.mean(axis=1))**2).mean(axis=1)
                vs=(((xv-xv.mean(axis=1,keepdims=True))-(y-y.mean(axis=1,keepdims=True)))**2).mean(axis=(1,2))
                identity_max['level_H_SV_minus_S']=max(identity_max['level_H_SV_minus_S'],float(np.max(np.abs(level-sm))))
                identity_max['shape_H_SV_minus_V']=max(identity_max['shape_H_SV_minus_V'],float(np.max(np.abs(shape-vs))))
            identity_max['position_minus_level_shape']=max(identity_max['position_minus_level_shape'],float(np.max(np.abs(pos-level-shape))))
            for task in TASKS:
                rr=[r for r in l2_window if r['method']==method and int(r['seed'])==seed and int(r['task_id'])==task]
                eps={}
                for r in rr:eps.setdefault(r['episode_id'],[]).append(r)
                l2_task_rows.append({'method':method,'seed':seed,'task_id':task,'aggregation':'episode_equal_within_task',
                    'anchors':len(rr),'episodes':len(eps),'source_groups':len(set(r['source_group'] for r in rr)),
                    'level_mse':float(np.mean([np.mean([x['level_mse'] for x in vv]) for vv in eps.values()])),
                    'shape_mse':float(np.mean([np.mean([x['shape_mse'] for x in vv]) for vv in eps.values()])),
                    'position_mse':float(np.mean([np.mean([x['position_mse'] for x in vv]) for vv in eps.values()]))})
            rr=[r for r in l2_task_rows if r['method']==method and int(r['seed'])==seed]
            l2_task_rows.append({'method':method,'seed':seed,'task_id':'five_task_macro','aggregation':'five_task_equal_episode_macro',
                'anchors':sum(r['anchors'] for r in rr),'episodes':sum(r['episodes'] for r in rr),'source_groups':sum(r['source_groups'] for r in rr),
                'level_mse':float(np.mean([r['level_mse'] for r in rr])),'shape_mse':float(np.mean([r['shape_mse'] for r in rr])),
                'position_mse':float(np.mean([r['position_mse'] for r in rr]))})
    write_csv(ROOT/'figure_data/LEVEL_SHAPE_L2_PER_WINDOW.csv',l2_window)
    # Add three-seed summaries with actual sample SD; the shared persistence reference is outside this L2 set.
    for taskval in TASKS+['five_task_macro']:
        for method in l2_methods:
            seeds_rows=[r for r in l2_task_rows if r['method']==method and r['task_id']==taskval]
            if len(seeds_rows)!=3:raise RuntimeError(f'L2 task/seed support incomplete {method}/{taskval}')
            l2_task_rows.append({'method':method,'seed':'mean_across_formal_seeds','task_id':taskval,'aggregation':'mean_and_sample_sd_across_seeds',
                'level_mse':float(np.mean([r['level_mse'] for r in seeds_rows])),'level_mse_sample_sd':float(np.std([r['level_mse'] for r in seeds_rows],ddof=1)),
                'shape_mse':float(np.mean([r['shape_mse'] for r in seeds_rows])),'shape_mse_sample_sd':float(np.std([r['shape_mse'] for r in seeds_rows],ddof=1)),
                'position_mse':float(np.mean([r['position_mse'] for r in seeds_rows])),'position_mse_sample_sd':float(np.std([r['position_mse'] for r in seeds_rows],ddof=1)),
                'anchors':'NA','episodes':'NA','source_groups':'NA'})
    write_csv(ROOT/'results/LEVEL_SHAPE_L2.csv',l2_task_rows)

    # Metadata-selected cases are fixed before outcome access; only values are exported here after the seal.
    cases={int(k):v for k,v in formula['metadata_selected_cases']['selected_anchor_ids'].items()}
    case_rows=[];traj=[]
    for task in TASKS:
        aid=cases[task];ep,task_str,group=metadata[aid]
        case_rows.append({'task_id':task,'episode_id':ep,'anchor_id':aid,'source_group':group,'selection':'V4 case' if task==119 else 'metadata lowest episode/frame/id','outcome_selected':False})
        i=ids.index(aid)
        for seed in SEEDS:
            matrices={'GT':target,'S':core_arrays[('S',str(seed))],'V':core_arrays[('V',str(seed))],
                      'H_SV':derived[('H_SV',seed)],'MIX50':derived[('MIX50',seed)]}
            for axis in (0,14):
                for method,matrix in matrices.items():
                    for step in range(30):
                        traj.append({'task_id':task,'episode_id':ep,'anchor_id':aid,'source_group':group,'seed':seed,'action_index':axis,
                            'future_step':step+1,'method':method,'native_command_value':float(matrix[i,step,axis]),
                            'unit_note':'numeric native command; physical units unresolved'})
    write_csv(ROOT/'figure_data/TRAJECTORY_CASES.csv',case_rows)
    write_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv',traj)

    # Discard all loaded future values before regenerating from source predictions and anchors alone.
    loaded_target_sha=target_sha
    del target_native,target,target_blob,target_ids,target_meta,traj
    # Verify generation is independent of outcomes after discarding the loaded target array.
    sys.path.insert(0,str(ROOT/'code'))
    from compose_v5 import compose as pure_compose
    regenerate={}
    for seed in SEEDS:
        src={k:core_arrays[(k,str(seed))] for k in ('S','V','A','C')}
        new=pure_compose(src,ids,anchors)
        for method in ['H_SV','H_VS','H_AV','H_SC','H_PV','MIX50']:
            tmp=ROOT/'tmp'/f'POST_SCORE_REGEN_{method}_{seed}.safetensors'
            save_file({'prediction_native':np.asarray(new[method],dtype=np.float64,order='C')},str(tmp))
            accepted=Path(bundle_meta[(method,seed)]['prediction_path'])
            same=sha_file(tmp)==sha_file(accepted)
            if not same:raise RuntimeError(f'postscore target-free regeneration changed bundle {method}/{seed}')
            regenerate[f'{method}:{seed}']=sha_file(accepted);tmp.unlink()
    # A prediction transformer signature excludes any future observations/labels; future values were discarded before regeneration.

    # Build clean seed-level and macro summaries used for the handoff.
    gain_mean_sd={}
    for name in CONTRASTS:
        gain_mean_sd[name]={}
        for m in METRICS:
            vals=[pair_macro[(name,str(s))][m] for s in SEEDS]
            gain_mean_sd[name][m]={'per_seed':vals,'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1))}
    absolute_mean_sd={}
    for method in METHODS:
        absolute_mean_sd[method]={}
        for m in METRICS:
            vals=[macro[(method,str(s))][m] for s in SEEDS]
            absolute_mean_sd[method][m]={'per_seed':vals,'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1)) if method!='PERSISTENCE' else None,
                'seed_role':'shared_reference' if method=='PERSISTENCE' else 'three_formal_seeds'}
    ci_summary={}
    for r in ci_rows:
        if r.get('task_id')=='five_task_macro' and r.get('seed')=='mean_across_formal_seeds':ci_summary.setdefault(r['contrast'],{})[r['metric']]={'low':r['ci_low'],'high':r['ci_high'],'point':r['point_estimate']}

    elapsed=time.perf_counter()-t0
    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Runtime ledger distinguishes historical frozen model inference from this CPU-only V5 computation.
    resource_rows=[]
    srcs=source_map['source_prediction_bundles']
    for role in ['S_OPT','V_MATCHED','A_COMMON','C_REASSIGNED']:
        for seed in SEEDS:
            s=next(x for x in srcs if x['role']==role and int(x['seed'])==seed)
            resource_rows.append({'phase':'historical_source_inference','role':role,'seed':seed,'wall_seconds':s.get('elapsed_seconds','NA'),
                'compute_device':'historical_gpu','physical_gpu_historical':s.get('physical_gpu_historical','NA'),'new_v5_gpu_job':False,
                'prediction_hash':s['prediction_sha256'],'scope':'accepted source inference; not executed in V5'})
    transform=json.loads((ROOT/'qa/TRANSFORM_QA.json').read_text())
    resource_rows.extend([
      {'phase':'v5_formula_composition','role':'six_fixed_composition_methods','seed':'all','wall_seconds':transform['composition_wall_seconds'],'compute_device':'CPU','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':'NA','scope':'output serialization and regeneration QA; 18 bundles'},
      {'phase':'v5_scoring_statistics_l2','role':'resume scoring, paired CI, L2 and trajectory-data export','seed':'all','wall_seconds':elapsed,'compute_device':'CPU','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':'NA','scope':'one archive read this attempt; '+str(prior_target_loads+1)+' total V5 archive reads including recorded failures; peak RSS KiB='+str(rss_kib)},
      {'phase':'v5_target_exposure','role':'development target reads','seed':'all','wall_seconds':'NA','compute_device':'CPU','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':'NA','scope':'archive_loads_total='+str(prior_target_loads+1)+'; loaded_window_rows_total='+str(int(exposure.get('failed_attempt_target_array_window_rows_loaded',0))+185)+'; unique_windows=185; all after prediction seal'},
      {'phase':'v5_source_score_reference_reuse','role':'preflighted accepted V4 source per-window scores','seed':'all','wall_seconds':'NA','compute_device':'CPU','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':sha_file(ref_path),'scope':'unique rows reused='+str(selected)+'; raw V4 file reads before resume='+str(exposure.get('v4_per_window_file_read_operations_before_retry',0))+'; donor-map rows used=0'},
      {'phase':'v5_gpu_activity','role':'new fits, feature extraction, GPU replay','seed':'none','wall_seconds':0,'compute_device':'none','physical_gpu_historical':'NA','new_v5_gpu_job':False,'prediction_hash':'NA','scope':'all three counts are zero; watcher not invoked'}])
    write_csv(ROOT/'results/RESOURCE_LEDGER.csv',resource_rows)

    # Publish exposure only after all sealed predictions passed and this archive read completed.
    failed_target_rows=int(exposure.get('failed_attempt_target_array_window_rows_loaded',0))
    exposure.update({
        'status':'V5_TARGET_SCORING_COMPLETE','target_archive_path':str(target_path),'target_archive_sha256':loaded_target_sha,
        'target_archive_load_operations_total':prior_target_loads+1,'target_archive_physical_read_operations_total':prior_target_loads+1,'target_archive_bytes':target_blob_size,
        'failed_scorer_attempts':int(exposure.get('failed_scorer_attempts',0)),
        'failed_attempt_target_array_window_rows_loaded':failed_target_rows,'target_array_window_rows_loaded_total':failed_target_rows+185,
        'target_array_windows_loaded_this_attempt':185,'unique_dev_receiver_windows':185,
        'new_derived_prediction_bundles_scored':18,'new_derived_dev_task_views_scored':90,'new_derived_dev_future_window_logical_reads':3330,
        'derived_window_loss_evaluations_this_attempt':derived_score_evaluations_this_attempt,
        'derived_window_loss_evaluations_preserved_from_attempt_02':derived_score_evaluations_preserved,
        'derived_window_loss_evaluations_total':3330,'derived_window_loss_recomputations_after_attempt_02':0,
        'l2_diagnostic_window_evaluations':2775,'source_reference_rows_reused_unique':selected,
        'source_reference_methods_reused':list(V4_REF_MAP.values()),'v5_preflighted_reference_copy_rows_read_this_attempt':selected,
        'v5_preflighted_reference_copy_sha256':sha_file(ref_path),'v4_per_window_rows_parsed_this_attempt':0,
        'v4_per_window_rows_parsed_total':int(exposure.get('failed_attempt_02_v4_per_window_rows_parsed',0))+int(exposure.get('v4_preflight_per_window_rows_parsed',0))+int(exposure.get('v4_resume_preflight_per_window_rows_parsed',0)),
        'v4_reference_rows_selected_total':int(exposure.get('failed_attempt_02_v4_reference_rows_selected',0))+int(exposure.get('v4_preflight_reference_rows_selected',0))+int(exposure.get('v4_resume_preflight_reference_rows_selected',0)),
        'v4_per_window_file_read_operations_total':int(exposure.get('v4_per_window_file_read_operations_before_retry',0)),
        'v4_primary_task_seed_rows_parsed_total':int(exposure.get('failed_attempt_02_primary_rows_parsed',0))+int(exposure.get('v4_preflight_primary_rows_parsed',0))+int(exposure.get('v4_resume_preflight_primary_rows_parsed',0)),
        'v4_primary_table_read_operations_total':int(exposure.get('v4_primary_table_read_operations_before_retry',0)),
        'v4_unselected_rows_excluded_total':int(exposure.get('failed_attempt_02_v4_unselected_rows_excluded',0))+int(exposure.get('v4_preflight_unselected_rows_excluded',0))+int(exposure.get('v4_resume_preflight_unselected_rows_excluded',0)),
        'trajectory_unique_target_windows_exported':5,'trajectory_target_scalar_values':5*2*30,
        'additional_target_archive_reads_for_figures':0,'preseal_target_reads':0,'final_heldout_payload_access':0,
        'cumulative_dev_future_window_scoring_reads':35523+3330,
        'cumulative_including_l2_diagnostic_evaluations':35523+3330+2775,
        'cumulative_including_l2_and_failed_target_load_rows':35523+3330+2775+failed_target_rows,
        'gt_independent_regeneration_hashes':regenerate,'prediction_seal_sha256':sha_file(seal_path),'score_completed_utc':utc()})
    exposure['history'].append({'phase':'V5_scoring_plus_l2_diagnostics','target_archive_loads_this_attempt':1,'target_archive_load_operations_total':prior_target_loads+1,'new_derived_window_loss_evaluations':3330,
        'l2_diagnostic_window_evaluations':2775,'target_reader_invoked_after_nonempty_seal':True,'final_heldout_payload_access':0,'status':'PASS'})
    exposure['history'].append({'phase':'V5_trajectory_figure_data_export','additional_target_archive_loads':0,'unique_case_windows':5,'status':'values exported from score-pass memory'})
    write_json(exposure_path,exposure)

    # A concise machine-readable metrics summary for direct handoff.
    handoff={'schema_version':1,'status':'SCORED','study':'post_V4_frozen_level_shape_composition_development','principal_candidate':'H_SV',
        'tasks':TASKS,'formal_seeds':SEEDS,'formula_lock_sha256':formula_sha,'prediction_seal_sha256':sha_file(seal_path),
        'absolute_five_task_episode_equal_mean_and_sample_sd':absolute_mean_sd,'paired_gain_five_task_mean_and_sample_sd':gain_mean_sd,
        'paired_source_group_ci_mean_across_seeds':ci_summary,'paired_gain_sign':'L(reference)-L(candidate)',
        'source_score_reconciliation':{'status':'PASS','rows':len(reconcile),'max_abs_delta':max(r['abs_delta'] for r in reconcile)},
        'new_prediction_bundles_scored':18,'new_window_loss_evaluations':3330,'derived_window_loss_evaluations_this_attempt':derived_score_evaluations_this_attempt,
        'derived_window_loss_evaluations_preserved_from_attempt_02':derived_score_evaluations_preserved if 'derived_score_evaluations_preserved' in locals() else 0,'l2_diagnostic_window_evaluations':2775,
        'target_archive_load_operations_total':prior_target_loads+1,'target_archive_sha256':loaded_target_sha,'preseal_target_access':0,'final_heldout_payload_access':0,
        'l2_identity_max_abs':identity_max,'gt_independent_regeneration_verified':len(regenerate)==18,'created_utc':utc()}
    write_json(ROOT/'results/V5_METRICS_SUMMARY.json',handoff)
    owner_path=ROOT/'coordination/OWNER_STATE.json';owner=json.loads(owner_path.read_text())
    owner.update({'phase':'scores_l2_and_trajectory_data_complete','status':'running','target_archive_loads_total':prior_target_loads+1,'target_archive_rows_loaded_total':int(exposure.get('failed_attempt_target_array_window_rows_loaded',0))+185,
        'new_prediction_bundles_scored':18,'new_window_loss_evaluations':3330,'active_jobs':0,'active_gpu_jobs':0,'updated_utc':utc()})
    write_json(owner_path,owner)
    print(json.dumps({'status':'SCORED','target_archive_sha256':loaded_target_sha,'target_archive_loads_total':prior_target_loads+1,'new_derived_window_loss_evaluations':3330,
        'source_reference_rows_reused_unique':selected,'v4_per_window_rows_parsed_total':int(exposure.get('failed_attempt_02_v4_per_window_rows_parsed',0))+int(exposure.get('v4_preflight_per_window_rows_parsed',0))+int(exposure.get('v4_resume_preflight_per_window_rows_parsed',0)),
        'target_archive_load_operations_total':prior_target_loads+1,'v4_source_reconciliation_max_abs_delta':max(r['abs_delta'] for r in reconcile),
        'l2_identity_max_abs':identity_max,'bootstrap_ci_rows':len(ci_rows),'seed_loo_rows':len(seed_loo),'task_loo_rows':len(task_loo),
        'total_wall_seconds':elapsed,'peak_rss_kib':rss_kib,'final_heldout_payload_access':0},indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
# Frozen CPU scorer. Real labels are opened once, only after sample and forecast locks exist.
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

HORIZONS=(1,3,6,10,15,30)
VIEWS=('OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND')
BLOCKS=(('RAW_OBS_0_14',0,14),('FULL43',0,43),('HAND_14_28',14,28),('LEG_28_43',28,43))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def by_cluster(values, cluster_ids):
    ids=np.asarray(cluster_ids).astype(str)
    unique=sorted(set(ids.tolist()))
    return unique,np.stack([np.asarray(values)[ids==cid].mean(axis=0) for cid in unique])


def score_arrays(prefix,forecast,labels,model,synthetic=False,draws=20000,seed=20261012):
    ids=np.asarray(prefix['anchor_id']).astype(str)
    if not np.array_equal(ids,np.asarray(forecast['anchor_id']).astype(str)) or not np.array_equal(ids,np.asarray(labels['anchor_id']).astype(str)):
        raise ValueError('anchor IDs differ across prefix/forecast/label arrays')
    clusters=np.asarray(prefix['canonical_cluster_id']).astype(str)
    episode=np.asarray(prefix['episode_id']).astype(str)
    if len(set(zip(clusters,episode)))!=len(episode):
        # Multiple anchors per episode are expected: only reject if a cluster contains >1 selected episode.
        for c in set(clusters):
            if len(set(episode[clusters==c]))!=1: raise ValueError('more than one selected episode per cluster')
    if 'source_group_id' in labels and not np.array_equal(np.asarray(prefix['source_group_id']).astype(str),np.asarray(labels['source_group_id']).astype(str)):
        raise ValueError('source_group IDs differ between sealed prefix and labels')
    if 'episode_id' in labels and not np.array_equal(np.asarray(prefix['episode_id']).astype(str),np.asarray(labels['episode_id']).astype(str)):
        raise ValueError('episode IDs differ between sealed prefix and labels')
    y=np.asarray(labels['obs_future'],dtype=np.float64)
    if y.shape!=(len(ids),30,43) or not np.isfinite(y).all():
        return {'status':'numeric_failure','reason':'future labels missing/nonfinite or shape mismatch',
                'new_scores_reported':False,'confirmation_verdict':'inconclusive'}
    sigma=np.asarray(model['sigma_obs'],dtype=np.float64)[:14]
    safe=np.asarray(model['safe_fd_obs'],dtype=np.float64)[:14]
    all_sigma=np.asarray(model['sigma_obs'],dtype=np.float64)
    all_safe=np.asarray(model['safe_fd_obs'],dtype=np.float64)
    per_anchor={}; rows=[]
    for view in VIEWS:
        p=np.asarray(forecast[view],dtype=np.float64)
        if p.shape!=(len(ids),30,43) or not np.isfinite(p).all():
            return {'status':'numeric_failure','reason':'forecast missing/nonfinite or shape mismatch',
                    'new_scores_reported':False,'confirmation_verdict':'inconclusive'}
        for h in HORIZONS:
            for name,lo,hi in BLOCKS:
                sl=slice(lo,hi)
                scale=all_sigma[sl]
                fdscale=all_safe[sl]
                e=p[:,:h,sl]-y[:,:h,sl]
                pos_std=np.mean(np.abs(e)/scale[None,None,:],axis=(1,2))
                pos_native=np.mean(np.abs(e),axis=(1,2))
                if h==1:
                    fd_std=np.full(len(ids),np.nan); fd_native=np.full(len(ids),np.nan)
                else:
                    fd_err=np.diff(p[:,:h,sl],axis=1)-np.diff(y[:,:h,sl],axis=1)
                    fd_std=np.mean(np.abs(fd_err)/fdscale[None,None,:],axis=(1,2))
                    fd_native=np.mean(np.abs(fd_err),axis=(1,2))
                for metric,vals in (('position_standardized_mae',pos_std),('position_native_mae',pos_native),
                                    ('future_internal_fd_safe_normalized',fd_std),('future_internal_fd_native_mae',fd_native)):
                    unique,groupvals=by_cluster(vals,clusters)
                    value=float(np.mean(groupvals[np.isfinite(groupvals)])) if np.isfinite(groupvals).any() else None
                    rows.append({'view':view,'horizon':h,'block':name,'metric':metric,
                                 'value':value,'clusters':len(unique),'anchors':len(ids)})
                    per_anchor[(view,h,name,metric)]=vals
    primary={}
    for metric in ('position_standardized_mae','future_internal_fd_safe_normalized'):
        endpoint={}
        for view in VIEWS:
            vals=per_anchor[(view,30,'RAW_OBS_0_14',metric)]
            _,cv=by_cluster(vals,clusters)
            endpoint[view]=cv.reshape(len(cv),-1).mean(axis=1)
        d=endpoint['OBS_PERSISTENCE']-endpoint['ARX_HOLD_COMMAND']
        rng=np.random.default_rng(seed)
        idx=rng.integers(0,len(d),size=(draws,len(d)),dtype=np.int32)
        boot=d[idx].mean(axis=1)
        loss_p=float(endpoint['OBS_PERSISTENCE'].mean())
        loss_h=float(endpoint['ARX_HOLD_COMMAND'].mean())
        loss_cv=float(endpoint['OBS_CONSTANT_VELOCITY'].mean())
        primary[metric]={'persistence_loss':loss_p,'hold_loss':loss_h,'constant_velocity_loss':loss_cv,
            'gain_persistence_minus_hold':float(d.mean()),
            'hold_skill_vs_persistence':(1.0-loss_h/loss_p) if loss_p>0 else None,
            'cv_skill_vs_persistence':(1.0-loss_cv/loss_p) if loss_p>0 else None,
            'ordinary_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'co_primary_97_5_ci':[float(np.quantile(boot,.0125)),float(np.quantile(boot,.9875))],
            'clusters':len(d),'bootstrap_draws':draws,'seed':seed}
    if synthetic:
        verdict='synthetic_not_confirmatory'
    elif all(x['co_primary_97_5_ci'][0]>0 for x in primary.values()):
        verdict='replicated_recorded_domain_gain'
    elif any(x['co_primary_97_5_ci'][1]<0 for x in primary.values()):
        verdict='dual_gain_not_supported'
    else:
        verdict='inconclusive'
    return {'status':'PASS_SYNTHETIC_QA' if synthetic else 'SCORED_ONCE',
        'primary_metrics':primary,'secondary_rows':rows,'cluster_count':len(set(clusters)),
        'episode_count':len(set(episode)),'anchor_count':len(ids),
        'estimand':'one episode per connected independent cluster; mean anchors within episode; equal cluster mean',
        'p_values':False,'synthetic_fixture':bool(synthetic),
        'confirmation_verdict':verdict if not synthetic else 'synthetic_not_confirmatory',
        'old_dev_is_discovery_only':True,'original_full43_and_command28_unchanged':True,
        'oracle_or_teacher_used':False,'new_model_fits':0,'gpu_actions':0}


def score_once(root, label_path, forecast_path, prefix_path, sample_lock_path):
    root=Path(root)
    receipt=root/'logs/OUTCOME_READ_RECEIPT.json'
    score_path=root/'results/PROSPECTIVE_SCORE.json'
    if receipt.exists() or score_path.exists(): raise RuntimeError('outcome read already started or score exists; do not reopen or rescore')
    sample=json.loads(Path(sample_lock_path).read_text(encoding='utf-8'))
    if sample.get('status')!='SAMPLE_SEALED' or not sample.get('sample_sealed'):
        raise RuntimeError('sample must be sealed before scoring')
    admission=json.loads((root/'data/DATA_ADMISSION_RESULT.json').read_text(encoding='utf-8'))
    plan=json.loads((root/'data/N_PLAN.json').read_text(encoding='utf-8'))
    if admission.get('strict_pool_clusters',0)<plan['n_required_independent_clusters']:
        raise RuntimeError('strict admitted cluster count is below the frozen N')
    if sample.get('selected_clusters')!=plan['n_required_independent_clusters']:
        raise RuntimeError('sample cluster count differs from locked N')
    forecast_hash=sha(forecast_path); prefix_hash=sha(prefix_path)
    lock_path=root/'protocol/FORECAST_LOCK.json'
    if not lock_path.is_file(): raise RuntimeError('FORECAST_LOCK must exist before label access')
    forecast_lock=json.loads(lock_path.read_text(encoding='utf-8'))
    if forecast_lock.get('status')!='SEALED_BEFORE_LABEL_ACCESS' or forecast_lock.get('forecast_sha256')!=forecast_hash or forecast_lock.get('prefix_sha256')!=prefix_hash:
        raise RuntimeError('forecast/prefix hash lock mismatch before label access')
    if forecast_lock.get('sample_lock_sha256')!=sha(sample_lock_path): raise RuntimeError('sample lock hash mismatch before label access')
    admission_path=root/'data/DATA_ADMISSION_RESULT.json'
    if forecast_lock.get('admission_result_sha256')!=sha(admission_path): raise RuntimeError('admission hash mismatch before label access')
    model_path=root/'frozen_model/ARX_OBS43_FULL_TRAIN.npz'
    freeze_path=root/'protocol/MODEL_FREEZE.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
    model_hash=sha(model_path)
    if model_hash!=freeze.get('packaged_model_sha256') or forecast_lock.get('model_sha256')!=model_hash:
        raise RuntimeError('frozen model hash mismatch')
    if sample.get('model_sha256')!=model_hash or sample.get('model_freeze_sha256')!=sha(freeze_path):
        raise RuntimeError('sealed sample is not bound to the current frozen model lock')
    if forecast_lock.get('oracle_used') is not False or forecast_lock.get('teacher_forcing_used') is not False:
        raise RuntimeError('forecast lock reports forbidden oracle/teacher use')
    if forecast_lock.get('views')!=list(VIEWS): raise RuntimeError('forecast lock views differ from the three-view contract')
    # Prefix and prediction arrays are read and validated before the single outcome read.
    with np.load(prefix_path,allow_pickle=False) as z: prefix={k:z[k].copy() for k in z.files}
    with np.load(forecast_path,allow_pickle=False) as z: forecast={k:z[k].copy() for k in z.files}
    expected_views=set(VIEWS)
    if set(forecast)-{'anchor_id','source_group_id','episode_id','canonical_cluster_id'} != expected_views:
        raise RuntimeError('forecast file must contain exactly Persistence/CV/HOLD; oracle/teacher are forbidden')
    for identity in ('anchor_id','source_group_id','episode_id'):
        if identity not in forecast or not np.array_equal(np.asarray(prefix[identity]).astype(str),np.asarray(forecast[identity]).astype(str)):
            raise RuntimeError(f'prefix/forecast {identity} mismatch before outcome access')
    start=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
    receipt.write_text(json.dumps({'status':'OPEN_START','first_new_label_read_utc':start,
        'label_path':str(label_path),'forecast_sha256':forecast_hash,'prefix_sha256':prefix_hash,
        'sample_lock_sha256':sha(sample_lock_path),'read_count':1,'purpose':'single independent scoring pass'},
        indent=2,sort_keys=True)+'\n')
    # This is the only open of the new outcome file in this runner. Validate archive keys before loading arrays.
    required_label_keys={'anchor_id','source_group_id','episode_id','obs_future'}
    with np.load(label_path,allow_pickle=False) as z:
        if set(z.files)!=required_label_keys: raise ValueError('scorer labels archive must contain exactly identity fields and obs_future')
        labels={k:z[k].copy() for k in required_label_keys}
    with np.load(model_path,allow_pickle=False) as z: model={k:z[k].copy() for k in ('sigma_obs','safe_fd_obs')}
    result=score_arrays(prefix,forecast,labels,model,synthetic=False,draws=20000,seed=20261012)
    result['label_first_read_utc']=start; result['forecast_sha256']=forecast_hash
    result['prefix_sha256']=prefix_hash; result['label_sha256']=sha(label_path)
    out=root/'results/PROSPECTIVE_SCORE.json'
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    receipt.write_text(json.dumps({'status':'LOAD_COMPLETE_SCORE_WRITTEN','first_new_label_read_utc':start,
        'label_path':str(label_path),'forecast_sha256':forecast_hash,'prefix_sha256':prefix_hash,
        'label_sha256':result['label_sha256'],'read_count':1,'score_status':result['status'],
        'confirmation_verdict':result['confirmation_verdict']},indent=2,sort_keys=True)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description='Single-pass scorer; requires sealed sample and forecast hash lock.')
    ap.add_argument('--out-root',required=True); ap.add_argument('--labels',required=True)
    ap.add_argument('--forecasts',required=True); ap.add_argument('--prefix',required=True)
    ap.add_argument('--sample-lock',required=True)
    a=ap.parse_args()
    result=score_once(a.out_root,a.labels,a.forecasts,a.prefix,a.sample_lock)
    print(json.dumps({'status':result['status'],'confirmation_verdict':result['confirmation_verdict'],
                      'clusters':result['cluster_count'],'labels_opened_once':True}))


if __name__=='__main__':main()

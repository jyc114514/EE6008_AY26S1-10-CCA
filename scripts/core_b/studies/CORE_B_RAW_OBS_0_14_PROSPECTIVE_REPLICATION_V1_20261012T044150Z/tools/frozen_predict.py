#!/usr/bin/env python3
# Pure CPU inference for the frozen 43D ARX. Reads only prefix inputs and emits 3 views.
import argparse
import json
from pathlib import Path
import numpy as np

DELTA=1.0/30.0
VIEWS=('OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND')


def load_model(path):
    with np.load(path,allow_pickle=False) as z:
        required=('x_mean','x_std','coef','intercept','sigma_obs','safe_fd_obs')
        missing=[k for k in required if k not in z.files]
        if missing: raise ValueError('frozen model member missing: '+repr(missing))
        m={k:z[k].copy() for k in z.files}
    if m['coef'].shape!=(114,43) or m['x_mean'].shape!=(114,) or m['x_std'].shape!=(114,):
        raise ValueError('frozen model dimensions mismatch')
    if m['intercept'].shape!=(43,) or m['sigma_obs'].shape!=(43,) or m['safe_fd_obs'].shape!=(43,):
        raise ValueError('frozen scale/output dimensions mismatch')
    if any(a.dtype.hasobject or not np.isfinite(a).all() for a in m.values() if a.dtype.kind in 'fiu'):
        raise ValueError('frozen model has nonfinite or unsafe numeric arrays')
    return m


def source_step(model,s,v,u):
    x=np.concatenate((s,v,u),axis=1).astype(np.float64,copy=False)
    xz=(x-model['x_mean'])/model['x_std']
    vn=(model['intercept']+xz@model['coef'])*model['sigma_obs']
    return s+DELTA*vn,vn


def predict_views(model_path,prefix_path):
    model=load_model(model_path)
    with np.load(prefix_path,allow_pickle=False) as z:
        required=('anchor_id','source_group_id','episode_id','s_prev','s0','u0')
        allowed=set(required)|{'canonical_cluster_id','anchor_frame','anchor_timestamp','timestamp_window','episode_index'}
        if any(k not in z.files for k in required): raise ValueError('prefix-only file missing required key')
        unexpected=set(z.files)-allowed
        if unexpected: raise ValueError('unexpected non-prefix fields in predictor input: '+repr(sorted(unexpected)))
        inp={k:z[k].copy() for k in required}
    ids=inp['anchor_id']
    sp=inp['s_prev'].astype(np.float64); s0=inp['s0'].astype(np.float64); u0=inp['u0'].astype(np.float64)
    n=len(ids)
    if sp.shape!=(n,43) or s0.shape!=(n,43) or u0.shape!=(n,28): raise ValueError('prefix tensor shape mismatch')
    if not (np.isfinite(sp).all() and np.isfinite(s0).all() and np.isfinite(u0).all()):
        raise ValueError('nonfinite prefix input')
    s0f=inp['s0'].astype(np.float32); spf=inp['s_prev'].astype(np.float32)
    v0f=(s0f-spf)/DELTA
    persistence=np.repeat(s0f[:,None,:],30,axis=1)
    cv=(s0f[:,None,:]+np.arange(1,31,dtype=np.float32)[None,:,None]*DELTA*v0f[:,None,:]).astype(np.float32)
    s=s0.copy(); v=(s-sp)/DELTA
    hold=np.empty((n,30,43),dtype=np.float32)
    for j in range(30):
        s,v=source_step(model,s,v,u0)
        hold[:,j]=s.astype(np.float32)
        if not np.isfinite(s).all(): raise FloatingPointError('nonfinite frozen HOLD rollout')
    return {'anchor_id':ids,'source_group_id':inp['source_group_id'],'episode_id':inp['episode_id'],
            'OBS_PERSISTENCE':persistence,'OBS_CONSTANT_VELOCITY':cv,'ARX_HOLD_COMMAND':hold}


def main():
    ap=argparse.ArgumentParser(description='Predict Persistence, Constant Velocity and frozen HOLD from prefix only.')
    ap.add_argument('--model',required=True); ap.add_argument('--prefix',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    out=predict_views(a.model,a.prefix)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**out)
    print(json.dumps({'status':'PREDICTION_WRITTEN','anchors':len(out['anchor_id']),
                      'views':list(VIEWS),'new_fits':0,'gpu_actions':0}))


if __name__=='__main__':
    main()

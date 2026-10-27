#!/usr/bin/env python3
"""Pure arithmetic for N4; no file reads, fitting, or model execution."""
from __future__ import annotations
import argparse, json
import numpy as np

ATOL=1e-12
RTOL=1e-10
def _paths(yhat,y,sigma):
    yhat=np.asarray(yhat,dtype=np.float64);y=np.asarray(y,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)
    if yhat.shape!=y.shape or yhat.ndim!=3 or y.shape[1:]!=(30,28):
        raise ValueError('paths must share shape [N,30,28]')
    if sigma.shape!=(28,) or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('sigma must be finite, positive and length 28')
    if not np.isfinite(yhat).all() or not np.isfinite(y).all():raise ValueError('paths contain nonfinite values')
    return yhat,y,sigma
def _scale(v,name):
    v=np.asarray(v,dtype=np.float64)
    if v.shape!=(28,) or not np.isfinite(v).all() or np.any(v<=0):
        raise ValueError(name+' must be finite, positive and length 28')
    return v
def position_budget(yhat,y,sigma):
    yhat,y,sigma=_paths(yhat,y,sigma)
    e=(yhat-y)/sigma[None,None,:]
    level_offset=e.mean(axis=1)
    centered=e-level_offset[:,None,:]
    level_mse=np.mean(level_offset**2,axis=1)
    centered_mse=np.mean(centered**2,axis=(1,2))
    pos_mse=np.mean(e**2,axis=(1,2))
    if not np.allclose(pos_mse,level_mse+centered_mse,atol=ATOL,rtol=RTOL):
        raise ArithmeticError('position L2 level/time-centered identity failed')
    delta=np.median(e,axis=1)
    oracle_error=e-delta[:,None,:]
    level_mse_axis=level_offset**2
    centered_mse_axis=np.mean(centered**2,axis=1)
    oracle_original_mae_axis=np.mean(np.abs(e),axis=1)
    oracle_remaining_mae_axis=np.mean(np.abs(oracle_error),axis=1)
    oracle_removable_mae_axis=oracle_original_mae_axis-oracle_remaining_mae_axis
    if np.any(oracle_removable_mae_axis < -ATOL):raise ArithmeticError('oracle MAE increased on an axis')
    original_native_axis=np.mean(np.abs(yhat-y),axis=1)
    remaining_native_axis=np.mean(np.abs(oracle_error*sigma[None,None,:]),axis=1)
    removable_native_axis=original_native_axis-remaining_native_axis
    if np.any(removable_native_axis < -ATOL):raise ArithmeticError('native oracle MAE increased on an axis')
    original_norm_mae=np.mean(np.abs(e),axis=(1,2))
    remaining_norm_mae=np.mean(np.abs(oracle_error),axis=(1,2))
    removable_norm_mae=original_norm_mae-remaining_norm_mae
    original_native_mae=np.mean(np.abs(yhat-y),axis=(1,2))
    remaining_native_mae=np.mean(np.abs(oracle_error*sigma[None,None,:]),axis=(1,2))
    removable_native_mae=original_native_mae-remaining_native_mae
    if np.any(removable_norm_mae < -ATOL) or np.any(removable_native_mae < -ATOL):
        raise ArithmeticError('oracle MAE increased')
    if not np.allclose(original_norm_mae,removable_norm_mae+remaining_norm_mae,atol=ATOL,rtol=RTOL):
        raise ArithmeticError('oracle standardized MAE decomposition failed')
    if not np.allclose(original_native_mae,removable_native_mae+remaining_native_mae,atol=ATOL,rtol=RTOL):
        raise ArithmeticError('oracle native MAE decomposition failed')
    if not np.allclose(level_mse_axis.mean(axis=1),level_mse,atol=ATOL,rtol=RTOL):
        raise ArithmeticError('level axis aggregation failed')
    if not np.allclose(centered_mse_axis.mean(axis=1),centered_mse,atol=ATOL,rtol=RTOL):
        raise ArithmeticError('centered axis aggregation failed')
    output_constant=yhat.mean(axis=1,keepdims=True)
    output_constant=np.broadcast_to(output_constant,yhat.shape)
    return {'error_standardized':e,'level_offset':level_offset,'centered_error':centered,
      'level_mse_axis':level_mse_axis,'time_centered_error_mse_axis':centered_mse_axis,
      'level_mse':level_mse,'time_centered_error_mse':centered_mse,'pos_mse':pos_mse,
      'oracle_level_delta':delta,'oracle_remaining_error':oracle_error,
      'oracle_original_standardized_mae_axis':oracle_original_mae_axis,
      'oracle_remaining_standardized_mae_axis':oracle_remaining_mae_axis,
      'oracle_removable_standardized_mae_axis':oracle_removable_mae_axis,
      'oracle_original_native_mae_axis':original_native_axis,
      'oracle_remaining_native_mae_axis':remaining_native_axis,
      'oracle_removable_native_mae_axis':removable_native_axis,
      'oracle_original_standardized_mae':original_norm_mae,
      'oracle_remaining_standardized_mae':remaining_norm_mae,
      'oracle_removable_standardized_mae':removable_norm_mae,
      'oracle_original_native_mae':original_native_mae,
      'oracle_remaining_native_mae':remaining_native_mae,
      'oracle_removable_native_mae':removable_native_mae,
      'output_constant_native':output_constant}
def fd_energy(yhat,y,safe_scale):
    yhat=np.asarray(yhat,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if yhat.shape!=y.shape or yhat.ndim!=3 or y.shape[1:]!=(30,28):
        raise ValueError('paths must share shape [N,30,28]')
    safe=_scale(safe_scale,'safe_fd_scale')
    if not np.isfinite(yhat).all() or not np.isfinite(y).all():raise ValueError('paths contain nonfinite values')
    d=np.diff(y,axis=1)/safe[None,None,:]
    dh=np.diff(yhat,axis=1)/safe[None,None,:]
    T=np.mean(d*d,axis=(1,2));P=np.mean(dh*dh,axis=(1,2));C=np.mean(d*dh,axis=(1,2))
    mse=T+P-2*C
    if not np.allclose(mse,np.mean((dh-d)**2,axis=(1,2)),atol=ATOL,rtol=RTOL):
        raise ArithmeticError('FD MSE identity failed')
    if np.any(C*C>T*P+ATOL+RTOL*T*P):
        raise ArithmeticError('C^2 <= T*P failed')
    active=T>0;pred_active=P>0
    r=np.full(T.shape,np.nan);c=np.full(T.shape,np.nan);skill=np.full(T.shape,np.nan)
    norm=np.full(T.shape,np.nan);align=np.full(T.shape,np.nan);mse_ratio=np.full(T.shape,np.nan)
    r[active]=np.sqrt(P[active]/T[active])
    skill[active]=(2*C[active]-P[active])/T[active]
    mse_ratio[active]=mse[active]/T[active]
    norm[active]=(1-r[active])**2
    good=active & pred_active
    c[good]=C[good]/np.sqrt(T[good]*P[good])
    c_for_alignment=np.zeros(T.shape,dtype=np.float64)
    c_for_alignment[good]=c[good]
    align[active]=2*r[active]*(1-c_for_alignment[active])
    if not np.allclose(mse_ratio[active],norm[active]+align[active],atol=ATOL,rtol=RTOL):
        raise ArithmeticError('FD energy geometry identity failed')
    # T>0,P=0: the prescribed values are r=0, c=NA, skill=0, norm=1, alignment=0.
    r[active & ~pred_active]=0.0;skill[active & ~pred_active]=0.0
    norm[active & ~pred_active]=1.0;align[active & ~pred_active]=0.0
    # T=0: FD_MSE=P, and all ratios/skill/geometry terms remain NA.
    if not np.allclose(mse[~active],P[~active],atol=ATOL,rtol=RTOL):
        raise ArithmeticError('T=0 FD_MSE=P rule failed')
    return {'target_energy_T':T,'prediction_energy_P':P,'cross_alignment_C':C,
      'fd_mse':mse,'fd_mse_over_T':mse_ratio,'r_norm_ratio':r,'c_zero_lag_cosine':c,
      'fd_mse_skill':skill,'norm_error_term':norm,'alignment_error_term':align,
      'target_derivative_normalized':d,'prediction_derivative_normalized':dh}
def episode_equal(values,episodes):
    v=np.asarray(values,dtype=np.float64);ep=np.asarray(episodes,dtype=np.int64)
    if v.ndim!=1 or len(v)!=len(ep) or not len(v):raise ValueError('episode values/IDs must be paired vectors')
    if not np.isfinite(v).all():raise ValueError('nonfinite values cannot be silently dropped')
    ids=np.unique(ep)
    return ids,np.asarray([v[ep==x].mean() for x in ids],dtype=np.float64)
def gvi(cells):
    # cells[(system,query)]; positive G favors Q1, positive V favors TS11-D.
    q0ts=float(cells[('TS11-D','Q0')]);q1ts=float(cells[('TS11-D','Q1')])
    q0st=float(cells[('STATE-CTRL','Q0')]);q1st=float(cells[('STATE-CTRL','Q1')])
    gts=q0ts-q1ts;gst=q0st-q1st;vq0=q0st-q0ts;vq1=q1st-q1ts
    inter=gts-gst
    if not np.isclose(inter,vq1-vq0,atol=ATOL,rtol=RTOL):raise ArithmeticError('G/V/I interaction identity failed')
    return {'G_TS11-D':gts,'G_STATE-CTRL':gst,'V_Q0':vq0,'V_Q1':vq1,'I':inter}
def run_fixtures():
    sigma=np.ones(28);safe=np.ones(28);x=np.linspace(0,1,30,dtype=np.float64)
    y=np.broadcast_to(x[None,:,None],(1,30,28)).copy()
    # GT injection.
    p=position_budget(y,y,sigma);f=fd_energy(y,y,safe)
    assert np.allclose(p['pos_mse'],0,atol=ATOL) and np.allclose(f['fd_mse'],0,atol=ATOL)
    # Constant future offset moves level/seam while leaving future-internal derivatives intact.
    off=np.linspace(.01,.28,28)
    shifted=y+off[None,None,:]
    pb=position_budget(shifted,y,sigma);fb=fd_energy(shifted,y,safe)
    assert float(pb['level_mse'][0])>0 and np.allclose(pb['time_centered_error_mse'],0,atol=ATOL)
    assert np.allclose(fb['target_energy_T'],fb['prediction_energy_P'],atol=ATOL)
    assert np.allclose(fb['cross_alignment_C'],fb['target_energy_T'],atol=ATOL)
    seam0=np.abs(y[:,0]-0).mean();seam1=np.abs(shifted[:,0]-0).mean()
    assert seam1>seam0 and np.diff(shifted,axis=1).shape[1]==29
    # Same derivative norm, opposite direction.
    opposite=-y
    fo=fd_energy(opposite,y,safe)
    assert np.allclose(fo['r_norm_ratio'],1,atol=ATOL)
    assert np.allclose(fo['c_zero_lag_cosine'],-1,atol=ATOL)
    assert np.allclose(fo['fd_mse'],4*fo['target_energy_T'],atol=ATOL)
    assert np.allclose(fo['fd_mse_skill'],-3,atol=ATOL)
    # Zero prediction derivative and positive target energy.
    const=np.zeros_like(y);fz=fd_energy(const,y,safe)
    assert np.all(fz['target_energy_T']>0) and np.all(fz['prediction_energy_P']==0)
    assert np.all(fz['r_norm_ratio']==0) and np.isnan(fz['c_zero_lag_cosine']).all()
    assert np.all(fz['fd_mse_skill']==0) and np.all(fz['norm_error_term']==1) and np.all(fz['alignment_error_term']==0)
    # Zero target derivative with nonzero prediction derivative.
    flat=np.zeros_like(y);moving=y
    ft=fd_energy(moving,flat,safe)
    assert np.all(ft['target_energy_T']==0) and np.all(ft['fd_mse']==ft['prediction_energy_P'])
    assert np.isnan(ft['r_norm_ratio']).all() and np.isnan(ft['fd_mse_skill']).all()
    # Oracle level correction is a nonnegative removable MAE component.
    noisy=y+np.sin(np.arange(30)[None,:,None]/4)*.2+off[None,None,:]
    po=position_budget(noisy,y,sigma)
    assert np.all(po['oracle_removable_standardized_mae']>=-ATOL)
    assert np.allclose(po['oracle_original_standardized_mae'],
      po['oracle_removable_standardized_mae']+po['oracle_remaining_standardized_mae'],atol=ATOL,rtol=RTOL)
    # Per-window Cauchy-Schwarz and geometry identities are covered on nontrivial data above.
    assert np.all(fo['cross_alignment_C']**2<=fo['target_energy_T']*fo['prediction_energy_P']+ATOL)
    # Unequal anchors per episode are still episode-equal.
    ids,means=episode_equal([0.,2.,10.],[1,1,2])
    assert np.array_equal(ids,np.array([1,2])) and np.isclose(means.mean(),5.5) and not np.isclose(np.mean([0.,2.,10.]),5.5)
    # G/V/I and additive component gain identities.
    cells={('TS11-D','Q0'):5.,('TS11-D','Q1'):4.,('STATE-CTRL','Q0'):7.,('STATE-CTRL','Q1'):6.5}
    c=gvi(cells);assert np.isclose(c['I'],c['V_Q1']-c['V_Q0'])
    level={k:v*.3 for k,v in cells.items()};center={k:v*.7 for k,v in cells.items()}
    cg=gvi(cells);lg=gvi(level);tg=gvi(center)
    for key in cg:assert np.isclose(cg[key],lg[key]+tg[key],atol=ATOL,rtol=RTOL)
    # Input/schema rejection.
    rejected=0
    for bad in (np.full_like(y,np.nan),np.zeros((1,29,28))):
        try:position_budget(bad,y,sigma)
        except ValueError:rejected+=1
    try:position_budget(y,y,np.zeros(28))
    except ValueError:rejected+=1
    try:fd_energy(y,y,np.full(28,np.inf))
    except ValueError:rejected+=1
    assert rejected==4
    return {'status':'PASS_N4_ARITHMETIC_FIXTURES','fixtures':[
      'GT injection','constant offset level and future-FD invariance','same norm opposite direction r=1 c=-1 MSE=4T skill=-3',
      'zero predicted derivative','zero target derivative','oracle MAE nonincrease and decomposition',
      'Cauchy-Schwarz and FD geometry','unequal anchor counts with episode-equal weighting',
      'G/V/I identity and additive component gains','29 differences and separate seam',
      'nonfinite path rejection','wrong horizon rejection','invalid sigma rejection','invalid safe scale rejection'],
      'tolerance':{'atol':ATOL,'rtol':RTOL},'files_read':False,'targets_read':False,'models_loaded':False,
      'supervised_fits':0}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fixtures',action='store_true');args=ap.parse_args()
    if not args.fixtures:raise SystemExit('Use --fixtures; this helper never reads files.')
    print(json.dumps(run_fixtures(),indent=2,ensure_ascii=False))

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from analyze_full_query import nested_group_values,metric_arrays,group_summary

def test_nested_window_episode_group_mean():
    values=np.asarray([1.,3.,10.,20.,30.,50.])
    episodes=np.asarray([1,1,2,3,4,4])
    groups=np.asarray(['a','a','b','b','b','b'])
    names,means=nested_group_values(values,episodes,groups)
    assert names==['a','b']
    np.testing.assert_allclose(means,[2.,(10.+20.+40.)/3.])
    assert np.isclose(np.mean(means),(2.+(10.+20.+40.)/3.)/2.)

def test_safe_fd_uses_future_internal_29_differences_and_direction_na_when_no_support():
    anchor=np.zeros((2,28),dtype=np.float32)
    target=np.zeros((2,30,28),dtype=np.float32)
    pred=np.zeros_like(target)
    target[1,1:,0]=np.arange(1,30,dtype=np.float32)*0.01
    pred[1]=target[1]
    metrics=metric_arrays(pred,target,anchor,np.ones(28,dtype=np.float32),np.ones(28,dtype=np.float32),np.full(28,1e-6,dtype=np.float32))
    assert target.shape[1]-1==29
    assert metrics['active_step_count'][0]==0 and np.isnan(metrics['direction_accuracy_active'][0])
    assert metrics['active_step_count'][1]==29
    assert metrics['safe_fd_normalized'][1]==0
    assert metrics['zero_gt_tv_window'].tolist()==[1,0]

def test_tv_ratio_uses_aggregate_numerator_denominator_and_support_counts():
    episodes=np.asarray([1,2]);groups=np.asarray(['a','b'])
    metrics={'tv_pred':np.asarray([1.,3.]),'tv_gt':np.asarray([1.,9.]),'tv_ratio_window':np.asarray([1.,1/3]),
             'direction_accuracy_active':np.asarray([np.nan,1.]),'active_step_count':np.asarray([0,4]),
             'direction_correct_count':np.asarray([0,4]),'near_zero_gt_step_count':np.asarray([1,0]),
             'small_nonzero_gt_step_count':np.asarray([0,0]),'zero_gt_tv_window':np.asarray([1,0])}
    result=group_summary(metrics,episodes,groups)
    assert np.isclose(result['tv_ratio_aggregate'],0.4)
    assert np.isclose(result['tv_ratio_window'],(1+1/3)/2)
    assert result['zero_gt_tv_window_total']==1

def test_system_interaction_identity():
    qts_old,qts_had=0.11,0.09
    qstate_old,qstate_had=0.10,0.095
    gts=qts_old-qts_had;gstate=qstate_old-qstate_had
    vold=qstate_old-qts_old;vhad=qstate_had-qts_had
    assert np.isclose(gts-gstate,vhad-vold)

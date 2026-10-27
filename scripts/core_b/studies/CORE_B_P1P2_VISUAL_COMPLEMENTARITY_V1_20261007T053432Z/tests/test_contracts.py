from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from experiment_contracts import (common_pixels_r3d,episode_equal_moments,legal_frame_indices,make_matched_alt_predictor,validate_physical_gpu)
from train_alt import legacy_weighted_batch_objective,batch_loss
from infer_alt import InferenceDataset
from build_p2_support import choose_neighbors
from analysis_stats import shared_group_bootstrap_counts,episode_equal_cluster_means
from evaluate_d4_alt import pair_metrics
import d2_v3_evaluator_reference as d2ref
from model import parameter_count
from torchvision.models.video import r3d_18
from torch import nn
from bin.nvidia_smi_allow23 import build_command


def test_frame_schedule_is_32_past_frames_ending_at_anchor():
    idx=legal_frame_indices(119)
    assert idx.shape==(32,)
    assert idx[0]==0 and idx[-1]==119
    assert np.all(idx<=119)
    with pytest.raises(ValueError):legal_frame_indices(118)


def test_common_pixels_shape_and_finiteness():
    frames=np.zeros((32,8,16,3),dtype=np.uint8)
    frames[:,:,:,0]=127
    out=common_pixels_r3d(frames,resolution=8)
    assert tuple(out.shape)==(3,32,8,8)
    assert torch.isfinite(out).all()


def test_train_only_episode_equal_normalizer():
    x=np.array([[0.,2.],[2.,4.],[10.,12.],[100.,100.]],dtype=np.float32)
    episodes=np.array([0,0,1,2])
    train=np.array([True,True,True,False])
    mean,std=episode_equal_moments(x,episodes,train)
    # Episode 0 mean=[1,3], episode 1 mean=[10,12], episodes receive equal weight.
    np.testing.assert_allclose(mean,[5.5,7.5])
    assert np.all(std>0)


def test_legacy_weighted_batch_denominator_is_explicit():
    losses=torch.tensor([1.,2.])
    weights=torch.tensor([1.,3.])
    actual=legacy_weighted_batch_objective(losses,weights)
    assert actual.item()==pytest.approx(1.75)
    assert actual.item()!=pytest.approx((losses*weights).sum().item()/len(losses))
    with pytest.raises(ValueError):legacy_weighted_batch_objective(losses,torch.zeros_like(weights))


def test_alt_initialization_copies_shared_modules_and_has_expected_dims():
    torch.random.default_generator.manual_seed(17)
    model,report=make_matched_alt_predictor(30,20260950)
    assert model.video_projection.in_features==512
    assert report['shared_initialization_verified'] is True
    assert report['canonical_video_projection_parameters']==98432
    assert report['alternative_video_projection_parameters']==65664
    assert report['canonical_predictor_parameters']==1155356
    assert report['alternative_predictor_parameters']==1122588
    assert parameter_count(model)==1122588


def test_zero_output_is_persistence_on_cpu_fixture():
    model,_=make_matched_alt_predictor(30,20260950)
    b=2
    state=torch.zeros((b,32,71));mask=torch.ones((b,32));rel=torch.zeros((b,32))
    video=torch.zeros((b,16,512));anchor=torch.randn((b,28));target=anchor[:,None,:].expand(-1,30,-1).clone()
    episode=torch.tensor([0,1])
    stats={'target_sigma':torch.ones(28),'safe_fd':torch.ones(28)}
    total,_,_,pred,_=batch_loss(model,(state,mask,rel,video,anchor,target,episode),stats,30,torch.device('cpu'))
    torch.testing.assert_close(pred,anchor[:,None,:].expand_as(pred),rtol=0,atol=0)
    torch.testing.assert_close(total,torch.zeros_like(total),rtol=0,atol=0)


def test_gpu_allowlist_guard_and_watcher_adapter_are_fail_closed():
    assert validate_physical_gpu('2')==2
    assert validate_physical_gpu('3')==3
    for value in (None,'0','4','gpu2'):
        with pytest.raises(RuntimeError):validate_physical_gpu(value)
    assert build_command(['--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])==['/usr/bin/nvidia-smi','--id=2,3','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']
    with pytest.raises(ValueError):build_command(['--query-compute-apps=pid,process_name'])


def test_inference_subset_uses_local_feature_row_alignment():
    rows=np.array([2,5],dtype=np.int64)
    features=np.stack([np.full(512,11,dtype=np.float32),np.full(512,22,dtype=np.float32)])
    data={'history_standardized':np.zeros((6,32,71),dtype=np.float32),'history_mask':np.ones((6,32),dtype=np.float32),'history_relative_delta_standardized':np.zeros((6,32),dtype=np.float32),'anchor_command_native':np.zeros((6,28),dtype=np.float32)}
    ds=InferenceDataset(data,features,rows,np.zeros(512,dtype=np.float32),np.ones(512,dtype=np.float32))
    assert float(ds[0][3][0,0])==11.0
    assert float(ds[1][3][0,0])==22.0


def test_d1_neighbors_exclude_query_group_and_deduplicate_groups():
    distances=np.array([0.01,0.02,0.03,0.04,0.05,0.06],dtype=np.float64)
    groups=np.array(['same','same','query','group_a','group_b','group_c'])
    selected=choose_neighbors(distances,groups,'query',k=3)
    selected_groups=groups[selected].tolist()
    assert 'query' not in selected_groups
    assert len(selected_groups)==len(set(selected_groups))==3


def test_cluster_bootstrap_reweights_episodes_not_group_means():
    # Group A contains two episodes with scores 1 and 1; group B has one episode scored 7.
    group_sums=np.array([2.0,7.0])
    episode_counts=np.array([2.0,1.0])
    sampled_groups=np.array([[0,2]],dtype=np.uint16)
    value=episode_equal_cluster_means(group_sums,episode_counts,sampled_groups)
    assert value[0]==pytest.approx(7.0)
    both_groups=np.array([[1,1]],dtype=np.uint16)
    value=episode_equal_cluster_means(group_sums,episode_counts,both_groups)
    assert value[0]==pytest.approx(3.0)  # (1+1+7)/3, not mean of group means (=4)

def test_bootstrap_counts_are_shared_group_resamples():
    counts=shared_group_bootstrap_counts(group_count=5,replicates=32,seed=20261007)
    assert counts.shape==(32,5)
    assert np.all(counts.sum(axis=1)==5)
    np.testing.assert_array_equal(counts,shared_group_bootstrap_counts(5,32,20261007))


def test_r3d18_prelogit_parameter_accounting_without_forward():
    model=r3d_18(weights=None)
    assert model.fc.in_features==512
    assert sum(p.numel() for p in model.parameters())==33371472
    model.fc=nn.Identity()
    assert sum(p.numel() for p in model.parameters())==33166272


def test_d4_same_target_pair_utility_is_old_minus_new():
    truth=np.zeros((4,28),dtype=np.float64)
    old=np.ones((4,28),dtype=np.float64)
    new=np.full((4,28),0.5,dtype=np.float64)
    result=pair_metrics(old,new,truth,np.ones(28,dtype=np.float64))
    assert result['Full28|L_old']==pytest.approx(1.0)
    assert result['Full28|L_new']==pytest.approx(0.5)
    assert result['Full28|U']==pytest.approx(0.5)
    assert result['Full28|R']==pytest.approx(0.5)
    assert result['Full28|FD_old']==pytest.approx(0.0)
    assert result['Full28|FD_new']==pytest.approx(0.0)


def test_d2_v3_locked_onset_state_machine_contract():
    signal=np.array([0.,0.,0.,0.,0.,0.,0.2,0.2,0.2,0.0],dtype=np.float64)
    trace=d2ref.state_trace(signal,1,11,on=0.1,off=0.05)
    assert trace['known'][6] and trace['armed'][6]
    assert len(trace['events'])==1
    assert trace['events'][0]['onset_frame']==7
    assert trace['events'][0]['confirmation_frame']==9

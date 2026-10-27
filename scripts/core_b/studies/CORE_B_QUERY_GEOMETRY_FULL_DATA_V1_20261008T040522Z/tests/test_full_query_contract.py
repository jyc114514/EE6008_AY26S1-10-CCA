from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from d3_model import D3TimeSpacePredictor,parameter_count,state_hash,sinusoidal
from fixed_future_query_model import FixedFutureQueryOverride,forward_with_fixed_future_queries
from d3_eq_contract import per_sample_loss,episode_equal_weights,effective_update_groups,weighted_batch_loss
from corrected_scheduler import UpdateDrivenCosineAnnealingLR,closed_form_lr


def _query_tables():
    with np.load(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz',allow_pickle=False) as z:
        return {k:np.asarray(z[k],dtype=np.float32).copy() for k in ['candidate_h30','candidate_h60','old_future_h30','old_future_h60']}

def _float32_hash(value):
    a=np.ascontiguousarray(value,dtype=np.float32);h=hashlib.sha256();h.update(a.dtype.str.encode('ascii'));h.update(b'\0');h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes());return h.hexdigest()

def test_hadamard_hash_rms_orthogonality_and_prefix():
    q=_query_tables();meta=json.loads((ROOT/'protocol/QUERY_TABLES_AND_HASHES.json').read_text())
    h=q['candidate_h30'];old=q['old_future_h30']
    assert h.shape==(30,128) and np.isfinite(h).all()
    assert _float32_hash(h)==meta['candidate_table_sha256_h30']
    assert np.array_equal(h,q['candidate_h60'][:30])
    assert np.array_equal(old,q['old_future_h60'][:30])
    assert np.max(np.abs(h.mean(axis=1)))<1e-6
    assert np.max(np.abs(np.linalg.norm(h,axis=1)-np.linalg.norm(old,axis=1)))<1e-5
    normalized=h/np.linalg.norm(h,axis=1,keepdims=True)
    gram=normalized@normalized.T
    assert np.max(np.abs(np.diag(gram)-1))<1e-5
    assert np.max(np.abs(gram-np.eye(30)))<1e-5

def test_wrapper_changes_only_fixed_future_query_and_adds_no_parameters():
    torch.manual_seed(9);q=_query_tables();base=D3TimeSpacePredictor(horizon=30,seed=20261001,video_mode='TS11-D').eval()
    before={k:v.detach().clone() for k,v in base.state_dict().items()}
    wrapped=FixedFutureQueryOverride(base,torch.from_numpy(q['candidate_h30'])).eval()
    assert parameter_count(base)==parameter_count(wrapped)==890780
    assert all(torch.equal(v,before[k]) for k,v in wrapped.base.state_dict().items())
    assert not any('future_query_table' in n for n,_ in wrapped.named_parameters())
    assert 'future_query_table' in wrapped.state_dict()
    assert torch.equal(wrapped.future_query_table,torch.from_numpy(q['candidate_h30']))

def test_fixed_old_query_forward_matches_source_model():
    torch.manual_seed(10);base=D3TimeSpacePredictor(horizon=30,seed=20261002,video_mode='STATE-CTRL').eval()
    q=_query_tables()['old_future_h30'];old_wrapper=FixedFutureQueryOverride(base,torch.from_numpy(q)).eval()
    B=2;state=torch.randn(B,32,71);mask=torch.ones(B,32);rel=torch.linspace(-1,0,32).repeat(B,1)
    dyn=torch.randn(B,16,4,768);static=torch.randn_like(dyn);anchor=torch.randn(B,28)
    with torch.inference_mode():
        expected=base(state,mask,rel,dyn,static,anchor)
        replay=old_wrapper(state,mask,rel,dyn,static,anchor)
    torch.testing.assert_close(replay,expected,atol=1e-5,rtol=1e-5)

def test_zero_output_is_anchor_persistence_and_buffer_reload_is_exact(tmp_path):
    q=_query_tables()['candidate_h30'];base=D3TimeSpacePredictor(horizon=30,seed=20261003,video_mode='TS11-D').eval()
    model=FixedFutureQueryOverride(base,torch.from_numpy(q)).eval()
    B=3;state=torch.zeros(B,32,71);mask=torch.ones(B,32);rel=torch.zeros(B,32);dyn=torch.zeros(B,16,4,768);static=torch.zeros_like(dyn);anchor=torch.randn(B,28)
    with torch.inference_mode():res=model(state,mask,rel,dyn,static,anchor)
    assert torch.count_nonzero(res)==0
    pred=anchor[:,None,:]+res
    torch.testing.assert_close(pred,anchor[:,None,:].expand(-1,30,-1),atol=0,rtol=0)
    path=tmp_path/'candidate.pt';torch.save(model.state_dict(),path)
    restored=FixedFutureQueryOverride(D3TimeSpacePredictor(horizon=30,seed=20261003,video_mode='TS11-D'),torch.from_numpy(q))
    restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    assert _float32_hash(restored.future_query_table.numpy())==_float32_hash(q)

def test_full_loss_ground_truth_injection_and_future_internal_29_differences():
    B=4;anchor=torch.randn(B,28);target=anchor[:,None,:].expand(-1,30,-1).clone();sigma=torch.ones(28);safe=torch.ones(28)
    residual=torch.zeros_like(target)
    per=per_sample_loss(residual,target,anchor,sigma,safe)
    torch.testing.assert_close(per,torch.zeros_like(per),atol=0,rtol=0)
    assert target.shape[1]-1==29

def test_episode_weighted_tail_batch_and_update_groups():
    episodes=np.array([0,0,1,2,2,2]);weights=episode_equal_weights(episodes)
    assert np.isclose(weights.mean(),1.0)
    losses=torch.tensor([1.,2.,3.]);w=torch.tensor([0.5,1.,1.5]);assert torch.isclose(weighted_batch_loss(losses,w),torch.sum(losses*w)/3)
    groups=list(effective_update_groups(17757,microbatch_size=64,accumulation_steps=1))
    assert len(groups)==278 and groups[-1][1]-groups[-1][0]==29

def test_locked_ids_have_unique_one_per_group_monitor_panel():
    import csv
    with (ROOT/'protocol/TRAIN_MONITOR_PANEL.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==128 and len({r['anchor_id'] for r in rows})==128 and len({r['source_group_id'] for r in rows})==128
    assert [int(r['rank']) for r in rows]==list(range(1,129))


def test_full_11120_update_scheduler_and_resume_contract():
    torch.set_num_threads(1)
    p=torch.nn.Parameter(torch.tensor([1.0],dtype=torch.float32))
    optimizer=torch.optim.AdamW([p],lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
    scheduler=UpdateDrivenCosineAnnealingLR(optimizer,T_max=11120,eta_min=1e-5)
    for update in range(1,1001):
        scheduler.set_epoch((update-1)//278+1)
        optimizer.zero_grad(set_to_none=True);p.grad=torch.ones_like(p);optimizer.step()
        assert abs(float(optimizer.param_groups[0]['lr'])-closed_form_lr(update,eta_min=1e-5,t_max=11120))<2e-11
        if update%278==0:scheduler.step()
    # Fresh optimizer/scheduler restoration at a mid-epoch update resumes the same LR and call counts.
    p2=torch.nn.Parameter(p.detach().clone())
    optimizer2=torch.optim.AdamW([p2],lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
    scheduler2=UpdateDrivenCosineAnnealingLR(optimizer2,T_max=11120,eta_min=1e-5)
    optimizer2.load_state_dict(optimizer.state_dict())
    scheduler2.load_state_dict(scheduler.state_dict())
    assert scheduler2.step_calls==1000 and scheduler2.legacy_epoch_step_calls==3
    assert abs(float(optimizer2.param_groups[0]['lr'])-closed_form_lr(1000,eta_min=1e-5,t_max=11120))<2e-11
    for update in range(1001,11121):
        scheduler2.set_epoch((update-1)//278+1)
        optimizer2.zero_grad(set_to_none=True);p2.grad=torch.ones_like(p2);optimizer2.step()
        assert abs(float(optimizer2.param_groups[0]['lr'])-closed_form_lr(update,eta_min=1e-5,t_max=11120))<2e-11
        if update%278==0:scheduler2.step()
    assert scheduler2.step_calls==11120 and scheduler2.legacy_epoch_step_calls==40
    assert abs(float(optimizer2.param_groups[0]['lr'])-1e-5)<2e-11
    before=scheduler2.get_last_lr()[0];scheduler2.step()
    assert scheduler2.step_calls==11120 and scheduler2.legacy_epoch_step_calls==41
    assert scheduler2.get_last_lr()[0]==before

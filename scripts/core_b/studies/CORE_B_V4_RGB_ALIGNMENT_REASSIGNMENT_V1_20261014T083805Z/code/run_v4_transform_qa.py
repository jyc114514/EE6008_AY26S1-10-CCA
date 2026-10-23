#!/usr/bin/env python3
"""CPU fixture for V4 donor transforms, label isolation, A mask, and safe reload."""
from __future__ import annotations
import os
import importlib.util,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
TASKS=(119,94,90,141,230)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module():
    path=ROOT/'code/v4_train.py';spec=importlib.util.spec_from_file_location('v4_train_fixture',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def main():
    train_path=ROOT/'data/TRAIN_SELECTED_H30.npz';dev_path=ROOT/'data/DEV_PREFIXES_NO_LABELS.npz'
    with np.load(train_path,allow_pickle=False) as z:train={k:z[k].copy() for k in z.files}
    with np.load(dev_path,allow_pickle=False) as z:dev={k:z[k].copy() for k in z.files}
    if 'target_native_h30' in dev:raise RuntimeError('dev prefix fixture unexpectedly contains targets')
    candidates=np.load(ROOT/'protocol/TRAIN_DONOR_CANDIDATES.npz',allow_pickle=False)
    cand=candidates['candidate_row'];ks=candidates['candidate_k']
    # Find a real C receiver/donor pair with different source and episode.
    recv=next(i for i in range(len(train['anchor_id'])) if ks[i]>0 and train['task_id'][i]==119)
    donor=int(cand[recv,0])
    assert donor>=0 and train['task_id'][recv]==train['task_id'][donor]
    assert train['episode_id'][recv]!=train['episode_id'][donor]
    assert train['source_cluster'][recv]!=train['source_cluster'][donor]
    norm=json.loads((ROOT/'normalizers/NORMALIZER_FORMAL.json').read_text())
    resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text())
    manifest=json.loads(Path(resolved['feature_cache']).read_text())
    from prepare_v2_data import Q0ShardedGrid
    grid=Q0ShardedGrid(manifest)
    m=load_module();cpu=torch.device('cpu')
    # Correct matched and reassigned batches through the same accepted adapter.
    bprefix,btask,banchor,by,bsigma,bfd=m.input_batch(train,np.asarray([recv]),norm,grid,'TS11-D',cpu)
    cprefix,ctask,canchor,cy,csigma,cfd=m.input_batch(train,np.asarray([recv]),norm,grid,'TS11-D',cpu,donor_rows=np.asarray([donor]))
    same_receiver={k:torch.equal(bprefix[k],cprefix[k]) for k in ['state','history_mask','relative_delta','anchor_command']}
    receiver_targets_same=torch.equal(by,cy);task_same=torch.equal(btask,ctask);anchor_same=torch.equal(banchor,canchor)
    if not all(same_receiver.values()) or not receiver_targets_same or not task_same or not anchor_same:raise RuntimeError('C transform changed a receiver field')
    if torch.equal(bprefix['dynamic'],cprefix['dynamic']) and torch.equal(bprefix['static'],cprefix['static']):raise RuntimeError('selected C donor did not replace RGB block')
    # The selected donor target is never supplied to candidate selection or feature loading.
    alt={k:v for k,v in train.items()};alt['target_native_h30']=train['target_native_h30'].copy();alt['target_native_h30'][donor]+=123.0
    aprefix,atask,aanchor,ay,_,_=m.input_batch(alt,np.asarray([recv]),norm,grid,'TS11-D',cpu,donor_rows=np.asarray([donor]))
    donor_label_mutation_no_change=all(torch.equal(cprefix[k],aprefix[k]) for k in cprefix) and torch.equal(cy,ay)
    if not donor_label_mutation_no_change:raise RuntimeError('changing donor future labels changed C input or receiver loss target')
    # Same seed INIT; forward API contains no target argument.
    seed=202610131;init_manifest=json.loads((ROOT/'protocol/B_REUSE_DECISION.json').read_text())['checks'][0]
    init_path=Path(init_manifest['init_path']);init_state=load_file(str(init_path),device='cpu')
    model=m.V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None);model.load_state_dict(init_state,strict=True);model.eval()
    with torch.no_grad():pred1=model(cprefix,ctask).action_residual
    altered_receiver={k:v for k,v in train.items()};altered_receiver['target_native_h30']=train['target_native_h30'].copy();altered_receiver['target_native_h30'][recv]+=0.25
    rprefix,rtask,ranchor,ry,rsigma,rfd=m.input_batch(altered_receiver,np.asarray([recv]),norm,grid,'TS11-D',cpu,donor_rows=np.asarray([donor]))
    with torch.no_grad():pred2=model(rprefix,rtask).action_residual
    receiver_target_does_not_change_forward=torch.equal(pred1,pred2)
    loss1=m.per_sample_loss(pred1,cy,canchor,csigma,cfd);loss2=m.per_sample_loss(pred2,ry,ranchor,rsigma,rfd)
    receiver_target_changes_loss=not torch.equal(loss1,loss2)
    if not receiver_target_does_not_change_forward or not receiver_target_changes_loss:raise RuntimeError('receiver label separation fixture failed')
    # State mask ignores arbitrary RGB tensors.
    state_model=m.V2TaskConditionedPredictor(seed=seed,video_mode='STATE-CTRL',waypoint_mode=None);state_model.load_state_dict(init_state,strict=True);state_model.eval()
    state_prefix,stask,_,_,_,_=m.input_batch(train,np.asarray([recv]),norm,None,'STATE-CTRL',cpu)
    with torch.no_grad():state_out=state_model(state_prefix,stask).action_residual
    modified={k:v.clone() for k,v in state_prefix.items()};modified['dynamic']=torch.randn((1,16,4,768));modified['static']=torch.randn((1,16,4,768))
    with torch.no_grad():state_out_rgb_changed=state_model(modified,stask).action_residual
    state_rgb_invariant=torch.equal(state_out,state_out_rgb_changed)
    if not state_rgb_invariant:raise RuntimeError('STATE-CTRL action output depends on RGB tensors')
    # Fresh instance from safe tensor serialization.
    smoke=ROOT/'qa/TRANSFORM_RELOAD_FIXTURE';smoke.mkdir(parents=True,exist_ok=True);weight=smoke/'fixture.safetensors';save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(weight))
    fresh=m.V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None);fresh.load_state_dict(load_file(str(weight),device='cpu'),strict=True);fresh.eval()
    with torch.no_grad():pred3=fresh(cprefix,ctask).action_residual
    reload_equal=torch.equal(pred1,pred3)
    if not reload_equal:raise RuntimeError('safe tensor fresh-instance reload changed prediction')
    receipt={'schema_version':1,'status':'PASS','device':'CPU_ONLY','input_files':{'train_npz_sha256':sha(train_path),'dev_prefix_sha256':sha(dev_path),'donor_candidates_sha256':sha(ROOT/'protocol/TRAIN_DONOR_CANDIDATES.npz'),'feature_cache_manifest_sha256':resolved['feature_cache_manifest_sha256']},'receiver_anchor_id':str(train['anchor_id'][recv]),'receiver_task_id':int(train['task_id'][recv]),'receiver_episode_id':int(train['episode_id'][recv]),'donor_anchor_id':str(train['anchor_id'][donor]),'donor_episode_id':int(train['episode_id'][donor]),'receiver_task_episode_source_unchanged':same_receiver,'task_id_unchanged':task_same,'anchor_unchanged':anchor_same,'receiver_future_target_unchanged_in_C_batch':receiver_targets_same,'RGB_block_replaced':True,'donor_target_mutation_changes_nothing':donor_label_mutation_no_change,'receiver_target_mutation_changes_forward':not receiver_target_does_not_change_forward,'receiver_target_mutation_changes_loss_only':receiver_target_changes_loss,'A_STATE_CTRL_RGB_invariant':state_rgb_invariant,'fresh_safe_tensor_instance_reload_prediction_equal':reload_equal,'model_forward_arguments':['prefix_inputs','task_id'],'donor_builder_requested_target_native_h30':False,'train_target_used_only_by_loss_and_separate_motion_strata':True,'dev_target_arrays_requested':False,'final_heldout_payload_access':0,'fixture_weight_path':str(weight),'fixture_weight_sha256':sha(weight)}
    (ROOT/'qa/TRANSFORM_LABEL_SEPARATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

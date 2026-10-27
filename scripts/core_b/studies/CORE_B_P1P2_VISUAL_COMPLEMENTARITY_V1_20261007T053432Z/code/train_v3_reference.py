#!/usr/bin/env python3
"""V3 single-run trainer; formal entry has no inference or final-test path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import V3Full28Predictor, parameter_count, state_hash


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
CELLS = {"S-HIST": (True, "none"), "F-POOL-D": (True, "pool"), "F-TOK-D": (True, "dynamic"), "F-TOK-S": (True, "static"), "V-TOK-D-A": (False, "dynamic"), "V-TOK-S-A": (False, "static")}
EPOCHS = 40
ACTION_DIM = 28
HAND = 14


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part"); part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x))) + "\n", encoding="utf-8"); os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle: torch.save(value, handle)
    os.replace(part, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class V3Dataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], features: np.ndarray, rows: np.ndarray, video_mean: Tensor, video_std: Tensor, use_state: bool, video_mode: str, horizon: int):
        self.data=data; self.features=features; self.rows=rows.astype(np.int64); self.video_mean=video_mean; self.video_std=video_std; self.use_state=use_state; self.video_mode=video_mode; self.horizon=horizon
    def __len__(self): return len(self.rows)
    def __getitem__(self, index: int):
        row=int(self.rows[index]); video=np.asarray(self.features[row,0 if self.video_mode in ("dynamic","pool") else 1],dtype=np.float32)
        video=(video-self.video_mean.numpy())/self.video_std.numpy()
        if self.video_mode == "none": video=np.zeros_like(video)
        return (torch.from_numpy(self.data["history_standardized"][row]), torch.from_numpy(self.data["history_mask"][row]), torch.from_numpy(self.data["history_relative_delta_standardized"][row]), torch.from_numpy(video), torch.from_numpy(self.data["anchor_command_native"][row]), torch.from_numpy(self.data["targets_native"][row,:self.horizon]), torch.tensor(int(self.data["episode_index"][row]),dtype=torch.long))


def load_data() -> tuple[dict[str,np.ndarray], np.ndarray, dict[str,Tensor]]:
    with np.load(EXP/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
    features=np.load(EXP/"cache/V3_TEMPORAL_FEATURES.npy",mmap_mode="r")
    norm=json.loads((EXP/"data/V3_NORMALIZERS.json").read_text()); video=json.loads((EXP/"data/V3_VIDEO_NORMALIZER.json").read_text())
    stats={"target_sigma":torch.tensor(norm["target_sigma"],dtype=torch.float32),"safe_fd":torch.tensor(norm["safe_fd_scale"],dtype=torch.float32),"video_mean":torch.tensor(video["mean"],dtype=torch.float32),"video_std":torch.tensor(video["safe_std"],dtype=torch.float32)}
    return data,features,stats


def batch_loss(model: V3Full28Predictor, batch, stats: dict[str,Tensor], cell: str, horizon: int, device: torch.device):
    state, mask, rel, video, anchor, target, episode = [value.to(device) for value in batch]
    use_state, video_mode = CELLS[cell]
    if video_mode == "pool": video=video.mean(dim=1,keepdim=True).expand(-1,16,-1)
    residual=model(state,mask,rel,video,anchor,use_state=use_state,use_video=video_mode!="none")
    prediction=anchor.unsqueeze(1)+residual*stats["target_sigma"].to(device).view(1,1,ACTION_DIM)
    sigma=stats["target_sigma"].to(device).view(1,1,ACTION_DIM); fd_scale=stats["safe_fd"].to(device).view(1,1,ACTION_DIM)
    position=0.5*torch.mean(torch.abs(prediction[:,:,:HAND]-target[:,:,:HAND])/sigma[:,:,:HAND],dim=(1,2))+0.5*torch.mean(torch.abs(prediction[:,:,HAND:]-target[:,:,HAND:])/sigma[:,:,HAND:],dim=(1,2))
    fd_error=torch.abs(torch.diff(prediction,dim=1)-torch.diff(target,dim=1))/fd_scale
    fd=0.5*torch.mean(fd_error[:,:,:HAND],dim=(1,2))+0.5*torch.mean(fd_error[:,:,HAND:],dim=(1,2))
    return position+0.1*fd,position,fd,prediction,episode


def save_checkpoint(path: Path, model, optimizer, scheduler, config, epoch, steps):
    atomic_torch(path,{"model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"config":config,"epoch":epoch,"optimizer_steps":steps,"seed":config["seed"],"rng_state_torch":torch.get_rng_state(),"rng_state_numpy":np.random.get_state(),"rng_state_python":random.getstate()})


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--cell",choices=sorted(CELLS),required=True); parser.add_argument("--horizon",type=int,choices=(30,60),required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--stage0",action="store_true"); parser.add_argument("--max-optimizer-steps",type=int); parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--accumulation",type=int,default=1); parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args(); run_root=args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()): raise FileExistsError(run_root)
    if args.stage0 and args.max_optimizer_steps!=4: raise ValueError("V3 Stage0 requires exactly 4 optimizer steps")
    if not args.stage0 and args.max_optimizer_steps is not None: raise ValueError("max optimizer steps only for Stage0")
    device=torch.device("cuda:0" if args.device=="cuda" else "cpu")
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.device=="cuda": torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed); torch.set_num_threads(1); run_root.mkdir(parents=True)
    data,features,stats=load_data(); train_rows=np.flatnonzero(data["split_code"]==0)
    if args.stage0: train_rows=train_rows[:min(len(train_rows),args.batch_size*args.accumulation*args.max_optimizer_steps)]
    episode_counts={int(e):int(np.sum(data["episode_index"][train_rows]==e)) for e in np.unique(data["episode_index"][train_rows])}; episode_count=len(episode_counts); total_windows=len(train_rows)
    weights=np.asarray([total_windows/(episode_count*episode_counts[int(e)]) for e in data["episode_index"][train_rows]],dtype=np.float32)
    dataset=V3Dataset(data,features,train_rows,stats["video_mean"],stats["video_std"],*CELLS[args.cell],args.horizon)
    model=V3Full28Predictor(horizon=args.horizon,seed=args.seed).to(device); model_hash_initial=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4); steps_per_epoch=math.ceil(total_windows/args.batch_size/args.accumulation); expected=args.max_optimizer_steps if args.stage0 else EPOCHS*steps_per_epoch; scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(expected,1),eta_min=1e-5)
    config={"status":"LOCKED","protocol_id":"CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1","cell":args.cell,"horizon_steps":args.horizon,"seed":args.seed,"epochs":0 if args.stage0 else EPOCHS,"batch_size":args.batch_size,"accumulation":args.accumulation,"effective_batch":args.batch_size*args.accumulation,"steps_per_epoch":steps_per_epoch,"expected_optimizer_steps":expected,"train_windows":total_windows,"train_episodes":episode_count,"parameter_count":parameter_count(model),"initial_model_hash":model_hash_initial,"input_sha256":digest(EXP/"data/V3_MAIN_INPUTS.npz"),"cache_sha256":digest(EXP/"cache/V3_TEMPORAL_FEATURES.npy"),"model_code_sha256":digest(EXP/"code/model.py"),"final_test_access":0,"encoder_updated":False}
    atomic_json(run_root/"RUN_CONFIG_LOCKED.json",config); atomic_json(run_root/"RUN_STATUS.json",{"status":"running","optimizer_steps":0,"final_test_access":0})
    # Zero residual must be exact Persistence on a fixed CPU/GPU fixture.
    fixture=next(iter(DataLoader(dataset,batch_size=min(args.batch_size,8),shuffle=False,num_workers=0))); model.eval()
    with torch.inference_mode():
        _,_,_,fixture_pred,_=batch_loss(model,fixture,stats,args.cell,args.horizon,device)
    persistence_error=float(torch.max(torch.abs(fixture_pred-fixture[4].to(device).unsqueeze(1))))
    if persistence_error>1e-6: raise RuntimeError(f"zero residual is not Persistence: {persistence_error}")
    steps=0; curves=[]
    try:
        for epoch in range(1,(EPOCHS+1 if not args.stage0 else 100000)):
            if steps>=expected: break
            model.train(); generator=torch.Generator().manual_seed(args.seed+epoch); loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,generator=generator,num_workers=0); optimizer.zero_grad(set_to_none=True); accum=0; micro=0; sum_loss=sum_pos=sum_fd=sum_weight=0.0
            for batch in loader:
                if steps>=expected: break
                total,pos,fd,_pred,episode=batch_loss(model,batch,stats,args.cell,args.horizon,device); values=total.detach().cpu().numpy(); episode_np=episode.detach().cpu().numpy(); batch_weights=np.asarray([total_windows/(episode_count*episode_counts[int(e)]) for e in episode_np],dtype=np.float32); batch_weight=torch.from_numpy(batch_weights).to(device); loss=(total*batch_weight).sum()/batch_weight.sum(); loss.backward(); accum+=len(episode); micro+=1; sum_loss+=float(np.sum(values*batch_weights)); sum_pos+=float(np.sum(pos.detach().cpu().numpy()*batch_weights)); sum_fd+=float(np.sum(fd.detach().cpu().numpy()*batch_weights)); sum_weight+=float(batch_weights.sum())
                if micro==args.accumulation or steps+1>=expected:
                    params=list(model.parameters()); torch.nn.utils.clip_grad_norm_(params,1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); steps+=1; accum=0; micro=0
            curves.append({"epoch":epoch,"optimizer_steps":steps,"loss_weighted":sum_loss/max(sum_weight,1e-12),"position_weighted":sum_pos/max(sum_weight,1e-12),"fd_weighted":sum_fd/max(sum_weight,1e-12),"learning_rate":scheduler.get_last_lr()[0]}); atomic_json(run_root/"RUN_STATUS.json",{"status":"running","last_completed_epoch":epoch,"optimizer_steps":steps,"final_test_access":0})
            if not args.stage0 and epoch in (10,20,30): save_checkpoint(run_root/f"checkpoint_epoch{epoch}.pt",model,optimizer,scheduler,config,epoch,steps)
        if steps!=expected: raise RuntimeError(f"optimizer steps {steps} != {expected}")
        name="checkpoint_stage0.pt" if args.stage0 else "checkpoint_last.pt"; model.eval(); save_checkpoint(run_root/name,model,optimizer,scheduler,config,epoch,steps)
        with torch.inference_mode():
            _,_,_,pred,_=batch_loss(model,fixture,stats,args.cell,args.horizon,device)
        reload=V3Full28Predictor(horizon=args.horizon,seed=args.seed).to(device); payload=torch.load(run_root/name,map_location="cpu",weights_only=False); reload.load_state_dict(payload["model_state_dict"],strict=True); reload.eval()
        with torch.inference_mode():
            _,_,_,reloaded,_=batch_loss(reload,fixture,stats,args.cell,args.horizon,device)
        reload_diff=float(torch.max(torch.abs(pred-reloaded)).detach().cpu())
        if reload_diff>1e-6: raise RuntimeError(f"reload mismatch {reload_diff}")
        peak_allocated=int(torch.cuda.max_memory_allocated()//(1024*1024)) if device.type=="cuda" else 0; peak_reserved=int(torch.cuda.max_memory_reserved()//(1024*1024)) if device.type=="cuda" else 0
        atomic_json(run_root/"EPOCH_CURVES.json",{"status":"PASS","rows":curves,"final_test_access":0}); atomic_json(run_root/("STAGE0_REPORT.json" if args.stage0 else "RUN_COMPLETE.json"),{"status":"PASS","cell":args.cell,"horizon":args.horizon,"seed":args.seed,"epoch":epoch,"optimizer_steps":steps,"checkpoint":name,"parameter_count":parameter_count(reload),"initial_persistence_max_abs":persistence_error,"checkpoint_reload_prediction_max_abs":reload_diff,"peak_memory_allocated_mib":peak_allocated,"peak_memory_reserved_mib":peak_reserved,"train_windows":total_windows,"final_test_access":0}); atomic_json(run_root/"RUN_STATUS.json",{"status":"complete","optimizer_steps":steps,"checkpoint":name,"final_test_access":0}); print(json.dumps({"status":"PASS","cell":args.cell,"horizon":args.horizon,"seed":args.seed,"optimizer_steps":steps,"peak_memory_reserved_mib":peak_reserved},sort_keys=True)); return 0
    except Exception as exc:
        atomic_json(run_root/"RUN_STATUS.json",{"status":"failed","optimizer_steps":steps,"error":f"{type(exc).__name__}: {exc}","final_test_access":0}); raise


if __name__ == "__main__": raise SystemExit(main())

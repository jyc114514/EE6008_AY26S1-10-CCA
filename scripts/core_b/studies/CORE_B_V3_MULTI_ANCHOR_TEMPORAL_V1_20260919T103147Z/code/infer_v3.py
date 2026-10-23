#!/usr/bin/env python3
"""Fixed-last V3 development inference; targets are not read by the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path: sys.path.insert(0, str(CODE))
from model import V3Full28Predictor, state_hash

EXP = CODE.parent
CELLS = {"S-HIST": (True, "none"), "F-POOL-D": (True, "pool"), "F-TOK-D": (True, "dynamic"), "F-TOK-S": (True, "static"), "V-TOK-D-A": (False, "dynamic"), "V-TOK-S-A": (False, "static")}
ACTION_DIM = 28


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def load_bundle(name: str):
    with np.load(EXP/"data"/name,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def checkpoint_for(run_id: str, seed: int) -> Path:
    return EXP/f"runs/{run_id}/{'formal_pilot_retry1' if seed==20260950 else 'formal'}/checkpoint_last.pt"


def atomic_npz(path: Path, **values):
    path.parent.mkdir(parents=True,exist_ok=True); part=path.with_name(f".{path.name}.part")
    with part.open("wb") as handle: np.savez_compressed(handle,**values)
    part.replace(path)


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--cell",choices=sorted(CELLS),required=True); parser.add_argument("--horizon",type=int,choices=(30,60),required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(0); device=torch.device("cuda:0")
    main_data=load_bundle("V3_MAIN_INPUTS.npz"); supp_data=load_bundle("V3_SUPPLEMENTARY_1S_INPUTS.npz"); features=np.load(EXP/"cache/V3_TEMPORAL_FEATURES.npy",mmap_mode="r"); norm=json.loads((EXP/"data/V3_NORMALIZERS.json").read_text()); vnorm=json.loads((EXP/"data/V3_VIDEO_NORMALIZER.json").read_text())
    target_sigma=torch.tensor(norm["target_sigma"],dtype=torch.float32,device=device); video_mean=torch.tensor(vnorm["mean"],dtype=torch.float32,device=device); video_std=torch.tensor(vnorm["safe_std"],dtype=torch.float32,device=device)
    payload=torch.load(checkpoint_for(args.run_id,args.seed),map_location="cpu",weights_only=False); model=V3Full28Predictor(horizon=args.horizon,seed=args.seed).to(device); model.load_state_dict(payload["model_state_dict"],strict=True); model.eval(); before=state_hash(model)
    use_state,video_mode=CELLS[args.cell]
    def run(data, indices, cache_offset, bundle_name):
        outputs=[]
        for start in range(0,len(indices),64):
            rows=indices[start:start+64]; state=torch.from_numpy(data["history_standardized"][rows]).to(device); mask=torch.from_numpy(data["history_mask"][rows]).to(device); rel=torch.from_numpy(data["history_relative_delta_standardized"][rows]).to(device); anchor=torch.from_numpy(data["anchor_command_native"][rows]).to(device); token_rows=np.asarray(features[cache_offset+rows,0 if video_mode in ("dynamic","pool") else 1],dtype=np.float32); video=(torch.from_numpy(token_rows).to(device)-video_mean)/video_std
            if video_mode=="none": video=torch.zeros_like(video)
            if video_mode=="pool": video=video.mean(dim=1,keepdim=True).expand(-1,16,-1)
            with torch.inference_mode(): residual=model(state,mask,rel,video,anchor,use_state=use_state,use_video=video_mode!="none"); prediction=anchor.unsqueeze(1)+residual*target_sigma.view(1,1,ACTION_DIM)
            outputs.append(prediction.cpu().numpy().astype(np.float32))
        prediction=np.concatenate(outputs,axis=0)
        atomic_npz(EXP/f"predictions/{args.run_id}/{bundle_name}.npz",anchor_id=data["anchor_id"][indices],episode_index=data["episode_index"][indices],original_task_index=data["original_task_index"][indices],anchor_frame=data["anchor_frame"][indices],anchor_timestamp=data["anchor_timestamp"][indices],horizon_steps=np.asarray(args.horizon,dtype=np.int64),predictions_native=prediction)
        return prediction.shape
    primary_dev=np.flatnonzero(main_data["split_code"]==1); run(main_data,primary_dev,0,"primary")
    if args.horizon==30: run(supp_data,np.arange(len(supp_data["episode_index"])),len(main_data["episode_index"]),"supplementary_1s")
    after=state_hash(model); meta={"status":"PASS","run_id":args.run_id,"cell":args.cell,"horizon_steps":args.horizon,"seed":args.seed,"checkpoint":str(checkpoint_for(args.run_id,args.seed)),"checkpoint_sha256":digest(checkpoint_for(args.run_id,args.seed)),"primary_rows":int(len(primary_dev)),"supplementary_rows":int(len(supp_data["episode_index"])) if args.horizon==30 else 0,"state_hash_before":before,"state_hash_after":after,"state_unchanged":before==after,"final_test_access":0,"encoder_updated":False,"physical_gpu":os.environ.get("GPU_WATCHER_PHYSICAL_GPU","unknown")}
    (EXP/f"predictions/{args.run_id}/PREDICTION_EXPORT_COMPLETE.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(meta,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

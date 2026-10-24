# Repaired public copy of EXCL-CODE-008; source SHA-256: 934a0afb5598913013e3b3ae75977c074b77695f5f1f0d7855c363c1cff446ec
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18

from experiment_contracts import common_pixels_r3d, legal_frame_indices, validate_physical_gpu

PROJECT: Path | None = None
EXP: Path | None = None
PRIMARY_INPUTS: Path | None = None
DENSE_INPUTS: Path | None = None
RAW_ROOT: Path | None = None
WEIGHTS: Path | None = None
WEIGHTS_SHA256 = "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"
FEATURE_DIM = 512
RESOLUTION = 384
BATCH_SIZE = 1



def video_artifact_id(path: str | Path) -> str:
    rel=Path(path)
    if rel.is_absolute(): raise ValueError("video reference must be relative to --raw-root")
    return "VIDEO-"+hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:24]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def load_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int,int]]:
    with np.load(PRIMARY_INPUTS,allow_pickle=False) as z:
        required=("anchor_id","episode_index","split_code","anchor_frame","anchor_timestamp","video_path")
        p={k:z[k].copy() for k in required}
    if set(np.unique(p["split_code"]).tolist()) - {0,1}:
        raise RuntimeError("V3 primary input has a split code outside train/dev")
    primary=[]
    for i in range(len(p["anchor_id"])):
        split="train" if int(p["split_code"][i])==0 else "dev"
        primary.append({"kind":"primary","source_id":str(p["anchor_id"][i]),"primary_row":i,"episode_index":int(p["episode_index"][i]),"split":split,"anchor_frame":int(p["anchor_frame"][i]),"anchor_timestamp":float(p["anchor_timestamp"][i]),"video_path":str(p["video_path"][i])})
    dev_episodes={r["episode_index"] for r in primary if r["split"]=="dev"}
    dense=[]
    with np.load(DENSE_INPUTS,allow_pickle=False) as z:
        required=("dense_id","episode_index","anchor_frame","anchor_timestamp","video_path")
        d={k:z[k].copy() for k in required}
    for i in range(len(d["dense_id"])):
        episode=int(d["episode_index"][i])
        if episode not in dev_episodes: raise RuntimeError("D4 dense input contains an episode outside canonical primary development")
        dense.append({"kind":"dense","source_id":str(d["dense_id"][i]),"dense_row":i,"episode_index":episode,"split":"dev","anchor_frame":int(d["anchor_frame"][i]),"anchor_timestamp":float(d["anchor_timestamp"][i]),"video_path":str(d["video_path"][i])})
    group_map={}
    with (EXP/"inventory/SOURCE_GROUP_MAP.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f): group_map[int(r["episode_index"])]=r["source_group_id"]
    for row in primary+dense: row["source_group_id"]=group_map[int(row["episode_index"])]
    return primary,dense,{r["episode_index"]:int(r["primary_row"]) for r in primary}


def union_rows(primary: list[dict[str, object]], dense: list[dict[str, object]]) -> tuple[list[dict[str, object]],np.ndarray,np.ndarray]:
    rows=[]; index={}; pmap=np.empty(len(primary),dtype=np.int64); dmap=np.empty(len(dense),dtype=np.int64)
    for kind,items,mapping in (("primary",primary,pmap),("dense",dense,dmap)):
        for i,row in enumerate(items):
            key=(int(row["episode_index"]),int(row["anchor_frame"]))
            if key not in index:
                index[key]=len(rows)
                item=dict(row);item["kind_tags"]=[kind];item["alias_source_ids"]=[str(row["source_id"])];rows.append(item)
            else:
                stored=rows[index[key]]
                if stored["video_path"]!=row["video_path"]:
                    raise RuntimeError(f"same episode/frame maps to different video paths: {key}")
                if kind not in stored["kind_tags"]: stored["kind_tags"].append(kind)
                if str(row["source_id"]) not in stored["alias_source_ids"]:stored["alias_source_ids"].append(str(row["source_id"]))
            mapping[i]=index[key]
    return rows,pmap,dmap


def decode_wanted_frames(path: Path,wanted: np.ndarray) -> dict[int,np.ndarray]:
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open allowlisted video: {path}")
    wanted_set={int(v) for v in wanted.tolist()}; out={}; maximum=max(wanted_set)
    try:
        frame=0
        while frame<=maximum:
            ok,image=cap.read()
            if not ok: raise RuntimeError(f"decode failed at source frame {frame}: {path}")
            if frame in wanted_set: out[frame]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            frame+=1
    finally: cap.release()
    if set(out)!=wanted_set: raise RuntimeError("decoded frame set differs from requested frame set")
    return out


def build_encoder(device: torch.device) -> tuple[nn.Module,str,str,int]:
    actual=sha256(WEIGHTS)
    if actual!=WEIGHTS_SHA256: raise RuntimeError(f"R3D weight hash mismatch: {actual}")
    state=torch.load(WEIGHTS,map_location="cpu",weights_only=True)
    model=r3d_18(weights=None)
    model.load_state_dict(state,strict=True)
    model.fc=nn.Identity()
    effective_parameters=sum(int(x.numel()) for x in model.parameters())
    final_conv_channels=next((module.out_channels for module in model.layer4[-1].conv2.modules() if isinstance(module,nn.Conv3d)),None)
    if model.fc.__class__.__name__!='Identity' or final_conv_channels!=512 or effective_parameters!=33166272:
        raise RuntimeError(f'unexpected R3D pre-logit architecture/parameter count: {effective_parameters}')
    model.eval().requires_grad_(False)
    before=buffer_hash(model)
    params_before=parameter_hash(model)
    model.to(device).eval()
    return model,before,params_before,effective_parameters


def parameter_hash(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def buffer_hash(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,value in sorted(model.named_buffers()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    global PROJECT, EXP, PRIMARY_INPUTS, DENSE_INPUTS, RAW_ROOT, WEIGHTS
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True,type=Path)
    ap.add_argument("--study-root",required=True,type=Path)
    ap.add_argument("--weights",required=True,type=Path)
    ap.add_argument("--primary-inputs",required=True,type=Path)
    ap.add_argument("--dense-inputs",required=True,type=Path)
    ap.add_argument("--raw-root",required=True,type=Path)
    ap.add_argument("--plan-only",action="store_true")
    ap.add_argument("--stage0",action="store_true")
    ap.add_argument("--stage0-rows",type=int,default=8)
    ap.add_argument("--stage0-output-root",type=Path)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=ap.parse_args()
    PROJECT=args.project_root.expanduser().resolve()
    EXP=args.study_root.expanduser().resolve()
    WEIGHTS=args.weights.expanduser().resolve()
    if not PROJECT.is_dir(): raise FileNotFoundError("project root does not exist")
    if not EXP.is_dir(): raise FileNotFoundError("study root does not exist")
    if not WEIGHTS.is_file(): raise FileNotFoundError("R3D weights must be supplied locally; downloading is not supported")
    if not PRIMARY_INPUTS.is_file(): raise FileNotFoundError("--primary-inputs file not found")
    if not DENSE_INPUTS.is_file(): raise FileNotFoundError("--dense-inputs file not found")
    if not RAW_ROOT.is_dir(): raise NotADirectoryError("--raw-root directory not found")
    PRIMARY_INPUTS=args.primary_inputs.expanduser().resolve()
    DENSE_INPUTS=args.dense_inputs.expanduser().resolve()
    RAW_ROOT=args.raw_root.expanduser().resolve()
    if args.stage0_output_root is not None and not args.stage0:raise ValueError("stage0-output-root is only valid with --stage0")
    primary,dense,_=load_rows()
    rows,pmap,dmap=union_rows(primary,dense)
    if args.plan_only:
        print(json.dumps({'status':'PLAN_PASS','primary_rows':len(primary),'dense_rows':len(dense),'union_clip_keys':len(rows),'d4_only_extra_keys':len(set(dmap.tolist())-set(pmap.tolist())),'final_test_access':0},sort_keys=True));return 0
    if args.device!='cuda': raise RuntimeError('formal R3D feature extraction is host CUDA only')
    physical=validate_physical_gpu(os.environ.get('GPU_WATCHER_PHYSICAL_GPU'))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable after approved host launch')
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    encoder,buffers_before,params_before,encoder_parameters=build_encoder(device)
    torch.cuda.reset_peak_memory_stats()
    train_primary=[r for r in primary if r["split"]=="train"]
    rows_to_encode=train_primary[:args.stage0_rows] if args.stage0 else rows
    if args.stage0:
        stage_root=(args.stage0_output_root.resolve() if args.stage0_output_root else EXP/'cache/stage0').resolve()
        if not stage_root.is_relative_to(EXP.resolve()):raise RuntimeError('Stage0 outputs must stay inside EXP_ROOT')
        stage_root.mkdir(parents=True,exist_ok=True)
        output=stage_root/'R3D_STAGE0_FEATURES.npz'
        if output.exists(): raise FileExistsError(output)
        ids=[];values=[];stage_started=time.perf_counter()
        for start in range(0,len(rows_to_encode),BATCH_SIZE):
            batch=rows_to_encode[start:start+BATCH_SIZE]
            clips=[]
            for row in batch:
                wanted=legal_frame_indices(int(row['anchor_frame']))
                rel=Path(str(row['video_path']))
                video=(RAW_ROOT/rel).resolve()
                if not video.is_relative_to(RAW_ROOT.resolve()) or not video.is_file(): raise FileNotFoundError(video)
                frames=decode_wanted_frames(video,wanted)
                clips.append(common_pixels_r3d(np.stack([frames[int(i)] for i in wanted])))
            x=torch.stack(clips).to(device)
            with torch.inference_mode(): y=encoder(x)
            if y.shape!=(len(batch),FEATURE_DIM) or not torch.isfinite(y).all(): raise RuntimeError(f'R3D pooled output schema failure {tuple(y.shape)}')
            ids.extend(range(start,start+len(batch)));values.extend(y.float().cpu().numpy())
        np.savez_compressed(output,cache_index=np.asarray(ids,dtype=np.int64),features=np.asarray(values,dtype=np.float32))
        elapsed=time.perf_counter()-stage_started;params_after=parameter_hash(encoder);buffers_after=buffer_hash(encoder)
        if params_after!=params_before or buffers_after!=buffers_before:raise RuntimeError('frozen R3D state changed during Stage0')
        manifest={'status':'PASS','stage0_rows':len(rows_to_encode),'physical_gpu':physical,'logical_device':'cuda:0','weights_sha256':WEIGHTS_SHA256,'effective_encoder_parameters':encoder_parameters,'parameter_hash_before':params_before,'parameter_hash_after':params_after,'buffers_before':buffers_before,'buffers_after':buffers_after,'peak_allocated_mib':int(torch.cuda.max_memory_allocated()//(1024*1024)),'peak_reserved_mib':int(torch.cuda.max_memory_reserved()//(1024*1024)),'elapsed_seconds':elapsed,'seconds_per_clip':elapsed/max(len(rows_to_encode),1),'resolution':RESOLUTION,'clip_frames':32,'final_test_access':0}
        (stage_root/'R3D_STAGE0_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        print(json.dumps(manifest,sort_keys=True));return 0
    final=EXP/'cache/R3D_FEATURES_UNION.npy';partial=EXP/'cache/R3D_FEATURES_UNION.partial.npy';manifest_path=EXP/'cache/R3D_FEATURE_MANIFEST.csv'
    EXP.joinpath('cache').mkdir(parents=True,exist_ok=True)
    if final.exists(): raise FileExistsError(final)
    if partial.exists() and not args.resume: raise FileExistsError(f'partial cache exists; pass --resume: {partial}')
    if partial.exists():
        mm=np.lib.format.open_memmap(partial,mode='r+')
        if mm.shape!=(len(rows),FEATURE_DIM): raise RuntimeError('partial cache shape mismatch')
    else:
        mm=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(rows),FEATURE_DIM));mm[:]=np.nan;mm.flush()
    complete=set(np.flatnonzero(np.isfinite(mm).all(axis=1)).tolist())
    done_count=len(complete)
    groups={}
    for i,row in enumerate(rows):
        if i in complete: continue
        rel=Path(str(row['video_path']));video=(RAW_ROOT/rel).resolve()
        if not video.is_relative_to(RAW_ROOT.resolve()) or not video.is_file(): raise FileNotFoundError(video)
        groups.setdefault(str(video),[]).append((i,row))
    write_header=not manifest_path.exists()
    with manifest_path.open('a',newline='',encoding='utf-8') as f:
        fields=['cache_index','kind_tags','alias_source_ids','episode_index','source_group_id','anchor_frame','anchor_timestamp','source_id','video_artifact_id','source_frame_ids','source_frame_ids_sha256','preprocess_id','weights_sha256','pooling','feature_dim','final_test_access']
        writer=csv.DictWriter(f,fieldnames=fields)
        if write_header: writer.writeheader()
        for group_no,(video_name,items) in enumerate(groups.items(),1):
            group_started=time.perf_counter()
            wanted=sorted({int(v) for _,row in items for v in legal_frame_indices(int(row['anchor_frame']))})
            frame_map=decode_wanted_frames(Path(video_name),np.asarray(wanted,dtype=np.int64))
            for cache_i,row in items:
                frame_ids=legal_frame_indices(int(row['anchor_frame']))
                rgb=np.stack([frame_map[int(t)] for t in frame_ids])
                x=common_pixels_r3d(rgb).unsqueeze(0).to(device)
                with torch.inference_mode(): y=encoder(x)
                if y.shape!=(1,FEATURE_DIM) or not torch.isfinite(y).all(): raise RuntimeError(f'feature output invalid at cache row {cache_i}')
                mm[cache_i]=y[0].float().cpu().numpy();mm.flush()
                payload=','.join(str(int(t)) for t in frame_ids)
                writer.writerow({'cache_index':cache_i,'kind_tags':';'.join(row['kind_tags']),'alias_source_ids':';'.join(row['alias_source_ids']),'episode_index':row['episode_index'],'source_group_id':row['source_group_id'],'anchor_frame':row['anchor_frame'],'anchor_timestamp':row['anchor_timestamp'],'source_id':row['source_id'],'video_artifact_id':video_artifact_id(row['video_path']),'source_frame_ids':payload,'source_frame_ids_sha256':hashlib.sha256(payload.encode()).hexdigest(),'preprocess_id':'V3_BILINEAR_SHORTSIDE_CENTER384_R3D_K400_NORM_V1','weights_sha256':WEIGHTS_SHA256,'pooling':'R3D18 avgpool pre-fc 512D','feature_dim':FEATURE_DIM,'final_test_access':0})
            f.flush()
            done_count+=len(items)
            print(json.dumps({'group':group_no,'groups_total':len(groups),'anchors':len(items),'cache_rows_done':done_count,'elapsed_seconds':time.perf_counter()-group_started,'physical_gpu':physical},sort_keys=True),flush=True)
    if not np.isfinite(mm).all(): raise RuntimeError('cache incomplete; partial memmap and receipts retained')
    after=buffer_hash(encoder);params_after=parameter_hash(encoder)
    if after!=buffers_before or params_after!=params_before: raise RuntimeError('frozen R3D parameters/buffers changed during feature extraction')
    mm.flush();del mm;os.replace(partial,final)
    accepted_manifest=EXP/'cache/R3D_FEATURE_MANIFEST_ACCEPTED.csv'
    if accepted_manifest.exists():raise FileExistsError(accepted_manifest)
    full=np.load(final,mmap_mode='r')
    accepted_fields=['cache_index','kind_tags','alias_source_ids','episode_index','source_group_id','anchor_frame','anchor_timestamp','source_id','video_artifact_id','source_frame_ids','source_frame_ids_sha256','feature_sha256','preprocess_id','weights_sha256','pooling','feature_dim','final_test_access']
    with accepted_manifest.open('w',newline='',encoding='utf-8') as af:
        aw=csv.DictWriter(af,fieldnames=accepted_fields);aw.writeheader()
        for cache_i,row in enumerate(rows):
            frame_ids=legal_frame_indices(int(row['anchor_frame']));payload=','.join(str(int(t)) for t in frame_ids)
            feature_sha=hashlib.sha256(np.asarray(full[cache_i],dtype=np.float32).tobytes()).hexdigest()
            aw.writerow({'cache_index':cache_i,'kind_tags':';'.join(row['kind_tags']),'alias_source_ids':';'.join(row['alias_source_ids']),'episode_index':row['episode_index'],'source_group_id':row['source_group_id'],'anchor_frame':row['anchor_frame'],'anchor_timestamp':row['anchor_timestamp'],'source_id':row['source_id'],'video_artifact_id':video_artifact_id(row['video_path']),'source_frame_ids':payload,'source_frame_ids_sha256':hashlib.sha256(payload.encode()).hexdigest(),'feature_sha256':feature_sha,'preprocess_id':'V3_BILINEAR_SHORTSIDE_CENTER384_R3D_K400_NORM_V1','weights_sha256':WEIGHTS_SHA256,'pooling':'R3D18 avgpool pre-fc 512D','feature_dim':FEATURE_DIM,'final_test_access':0})
    (EXP/'cache/R3D_FEATURE_PLAN.json').write_text(json.dumps({'status':'PASS','rows':len(rows),'primary_rows':len(primary),'dense_rows':len(dense),'unique_union_keys':len(rows),'d4_only_extra_keys':len(set(dmap.tolist())-set(pmap.tolist())),'primary_index_artifact_id':'R3D_PRIMARY_INDEX','dense_index_artifact_id':'R3D_DENSE_INDEX','accepted_manifest_artifact_id':'R3D_ACCEPTED_MANIFEST','weights_sha256':WEIGHTS_SHA256,'effective_encoder_parameters':encoder_parameters,'effective_encoder_parameters':encoder_parameters,'parameter_hash_before':params_before,'parameter_hash_after':params_after,'buffer_hash_before':buffers_before,'buffer_hash_after':after,'physical_gpu':physical,'final_test_access':0},indent=2,sort_keys=True)+'\n')
    np.save(EXP/'cache/R3D_PRIMARY_INDEX.npy',pmap);np.save(EXP/'cache/R3D_DENSE_INDEX.npy',dmap)
    print(json.dumps({'status':'PASS','rows':len(rows),'physical_gpu':physical,'output_artifact_id':'R3D_FEATURE_OUTPUT','final_test_access':0},sort_keys=True));return 0

if __name__=='__main__': raise SystemExit(main())

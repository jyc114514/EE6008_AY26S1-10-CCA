#!/usr/bin/env python3
"""Measure the source-faithful BF16/batch-one D3 baseline on two fixed train rows."""
from __future__ import annotations
import os
import csv, hashlib, importlib.util, json, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
D3_ROOT=PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
D3_SOURCE=D3_ROOT/"code/d3_extract_grid.py"
CACHE_MANIFEST=D3_ROOT/"d3/cache/D3_CACHE_MANIFEST.json"
BRIDGE=V1_RUN/"replay/core_b/core_b_bridge_manifest.csv"
CPU_AUDIT=OUT/"core_b/CORE_B_TRAIN_BRIDGE_CPU_AUDIT.json"
ATTEMPT=RUN/"core_b/gpu_probes/P1_STAGE0_ATTEMPT.json"
RESULT=OUT/"core_b/CORE_B_GPU_PROBE_STAGE0_P1.json"
FEATURES=RUN/"core_b/gpu_probes/P1_D3_BF16_BATCH1_STAGE0.npz"
FIXED_IDS=("e001726_a01_f000149","e001769_a00_f000119")

def utc(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def hash_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+"."+str(os.getpid())+".part")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)
def import_source(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {path}")
 module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(path.parent));spec.loader.exec_module(module);return module
def diff(a,b):
 x=np.asarray(a,dtype=np.float32);y=np.asarray(b,dtype=np.float32);d=np.abs(x.astype(np.float64)-y.astype(np.float64))
 return {"shape":list(x.shape),"max_abs_difference":float(d.max()),"mean_abs_difference":float(d.mean()),"p95_abs_difference":float(np.quantile(d,.95)),"p99_abs_difference":float(np.quantile(d,.99)),"relative_l2_difference":float(np.linalg.norm(d.ravel())/max(float(np.linalg.norm(y.astype(np.float64).ravel())),1e-30)),"allclose_atol_1e-5_rtol_1e-5":bool(np.allclose(x,y,atol=1e-5,rtol=1e-5))}

def main():
 physical=os.environ.get("GPU_WATCHER_PHYSICAL_GPU","");visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
 # The watcher assignment guard is evaluated before any torch.cuda API.
 if physical!="3" or visible!="3":raise RuntimeError(f"GPU3-only guard: physical={physical!r}, CUDA_VISIBLE_DEVICES={visible!r}")
 if ATTEMPT.exists() or RESULT.exists() or FEATURES.exists():raise FileExistsError("P1 Stage0 output already exists")
 ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
 attempt={"schema_version":1,"attempt_id":"P1_D3_BF16_BATCH1_STAGE0","status":"STARTED","started_utc":utc(),"physical_gpu":3,"logical_device_expected":"cuda:0","cuda_visible_devices":visible,"rows":list(FIXED_IDS),"new_training_runs":0,"final_payload_read":False,"future_targets_materialized":False}
 atomic_json(ATTEMPT,attempt)
 try:
  if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable after watcher assigned physical GPU3")
  logical=int(torch.cuda.current_device())
  if logical!=0:raise RuntimeError(f"expected logical cuda:0, got cuda:{logical}")
  gpu_name=torch.cuda.get_device_name(logical)
  cpu=json.loads(CPU_AUDIT.read_text(encoding="utf-8"));audit={r["anchor_id"]:r for r in cpu["rows"]}
  if cpu.get("status")!="PASS_CPU_METADATA_FRAME_STORAGE_PREPROCESS_AUDIT" or set(audit)!=set(FIXED_IDS):raise RuntimeError("two-row CPU identity/preprocess audit is not accepted")
  bridge=list(csv.DictReader(BRIDGE.open(encoding="utf-8-sig",newline="")));by_id={r["anchor_id"]:r for r in bridge if r["dataset_role"]=="train" and r["anchor_id"] in FIXED_IDS}
  if set(by_id)!=set(FIXED_IDS):raise RuntimeError("fixed train bridge rows missing")
  d3=import_source(D3_SOURCE,"v2_stage0_historical_d3");preprocess,load_base=d3.import_stack()
  clips_module=sys.modules[preprocess.__module__]
  if sha(D3_SOURCE)!=cpu["source_hashes"]["d3_extractor_script_sha256"] or sha(Path(clips_module.__file__))!=cpu["source_hashes"]["preprocess_module_sha256"]:raise RuntimeError("source hash drift after CPU audit")
  manifest=json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"));q0mod=import_source(PROJECT/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/code/prepare_v2_data.py","v2_stage0_q0_reader");q0=q0mod.Q0ShardedGrid(manifest)
  clips=[];q0_rows=[];rows=[]
  for anchor in FIXED_IDS:
   row=by_id[anchor];ar=audit[anchor];video=(d3.RAW_ROOT/row["video_path"]).resolve()
   if not video.is_relative_to(d3.RAW_ROOT.resolve()):raise RuntimeError("train video escaped fixed raw root")
   wanted=np.floor(np.linspace(int(row["anchor_frame"])-119,int(row["anchor_frame"]),32)).astype(np.int64)
   frame_map=d3.read_frames(video,wanted);frames=np.stack([frame_map[int(i)] for i in wanted],axis=0);static=np.repeat(frames[-1:,...],32,axis=0)
   prehash={}
   for mode,raw in (("dynamic",frames),("static",static)):
    tensor=preprocess(raw,resolution=384).detach().cpu().contiguous()
    ph=hash_array(tensor.numpy());want=next(x["sha256"] for x in ar["preprocessed"] if x["mode"]==mode)
    if ph!=want:raise RuntimeError(f"preprocess hash mismatch {anchor}/{mode}")
    clips.append({"anchor_id":anchor,"mode":mode,"tensor":tensor});prehash[mode]=ph
   qrow=q0.row(int(row["cache_index"]),anchor,str(row["episode_id"]));q0_rows.append(qrow)
   rows.append({"anchor_id":anchor,"episode_id":int(row["episode_id"]),"task_id":int(row["task_id"]),"anchor_frame":int(row["anchor_frame"]),"cache_index":int(row["cache_index"]),"v3_main_input_row":ar["v3_main_input_row"],"video_sha256":sha(video),"frame_indices":wanted.tolist(),"frame_indices_sha256":hash_array(wanted.astype("<i8")),"rgb_frames_sha256":hash_array(frames),"preprocessed_sha256":prehash,"q0_row_sha256":hash_array(qrow)})
  encoder,encoder_info=load_base(source_root=d3.VJEPA_SOURCE,checkpoint=d3.CHECKPOINT,device=torch.device("cuda:0"),num_frames=32,resolution=384);encoder.eval()
  outputs=[];raw_records=[]
  for item in clips:
   clip=item["tensor"].to("cuda:0")
   with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):tokens=encoder(clip.unsqueeze(0))[0]
   pooled,_=d3.grid_pool(tokens);raw32=tokens.detach().float().cpu().contiguous().numpy()
   if not np.isfinite(raw32).all() or not np.isfinite(pooled).all():raise FloatingPointError("nonfinite output")
   outputs.append(pooled);raw_records.append({"anchor_id":item["anchor_id"],"mode":item["mode"],"raw_token_dtype":str(tokens.dtype),"raw_token_shape":list(tokens.shape),"raw_token_float32_sha256":hash_array(raw32),"pooled_sha256":hash_array(pooled)})
  grids=np.stack(outputs).reshape(2,2,16,4,768).astype(np.float32)
  qstack=np.stack(q0_rows).astype(np.float32)
  comparisons={}
  for row_index,anchor in enumerate(FIXED_IDS):
   comparisons[anchor]={mode:diff(grids[row_index,mode_index],qstack[row_index,mode_index]) for mode_index,mode in enumerate(("dynamic","static"))}
  ftemp=FEATURES.with_name("."+FEATURES.name+"."+str(os.getpid())+".part")
  with ftemp.open("wb") as f:
   np.savez_compressed(f,anchor_ids=np.asarray(FIXED_IDS,dtype="U64"),features=grids);f.flush();os.fsync(f.fileno())
  os.replace(ftemp,FEATURES)
  report={"schema_version":1,"status":"PASS_STAGE0_P1_D3_BF16_BATCH1_TWO_TRAIN_ROWS","created_utc":utc(),"physical_gpu":3,"logical_device":"cuda:0","gpu_name":gpu_name,"torch":str(torch.__version__),"torch_cuda_build":str(torch.version.cuda),"encoder_info":encoder_info,"encoder_eval":True,"autocast":"bfloat16","encoder_batch_size":1,"grid_dtype":"float32","grid_shape":[2,2,16,4,768],"historical_d3_source_sha256":sha(D3_SOURCE),"preprocess_module_sha256":sha(Path(clips_module.__file__)),"checkpoint_sha256":sha(d3.CHECKPOINT),"q0_manifest_sha256":sha(CACHE_MANIFEST),"rows":rows,"raw_token_outputs":raw_records,"comparisons_to_q0":comparisons,"feature_archive_path":str(FEATURES),"feature_archive_sha256":sha(FEATURES),"scope":"train only; no dev outcomes, final payload, future targets, or training","new_training_runs":0,"final_payload_read":False,"future_targets_materialized":False}
  atomic_json(RESULT,report)
  attempt.update({"status":"COMPLETED","completed_utc":utc(),"logical_device":logical,"result_path":str(RESULT),"result_sha256":sha(RESULT),"feature_archive_path":str(FEATURES),"feature_archive_sha256":sha(FEATURES)})
  atomic_json(ATTEMPT,attempt)
  print(json.dumps({"status":report["status"],"physical_gpu":3,"gpu":gpu_name,"anchor_ids":list(FIXED_IDS),"result":str(RESULT)},ensure_ascii=False),flush=True)
  return 0
 except Exception as exc:
  attempt.update({"status":"FAILED","failed_utc":utc(),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()});atomic_json(ATTEMPT,attempt);raise
if __name__=="__main__":raise SystemExit(main())

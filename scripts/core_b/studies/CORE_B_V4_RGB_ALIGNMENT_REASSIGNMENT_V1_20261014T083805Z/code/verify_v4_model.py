#!/usr/bin/env python3
"""Fresh CPU process reload and provenance verification for one V4 fit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/locked_source'))
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
from datetime import datetime,timezone
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);job=json.loads(ap.parse_args().job_json.read_text())
    out=Path(job['attempt_dir']);manifest=json.loads((out/'MODEL_MANIFEST.json').read_text());done=json.loads((out/'RUN_COMPLETE.json').read_text())
    model_path=Path(manifest['model_file']);normalizer=Path(manifest['normalizer_path']);init=Path(manifest['init_path'])
    if done.get('status')!='PASS' or manifest.get('status')!='PASS':raise RuntimeError('fit markers are not PASS')
    if sha(model_path)!=manifest['model_file_sha256'] or sha(model_path)!=done['weights_sha256']:raise RuntimeError('model file hash does not reconcile')
    if sha(normalizer)!=manifest['normalizer_sha256']:raise RuntimeError('normalizer hash changed')
    if sha(Path(job['init_path']))!=sha(init):raise RuntimeError('fit INIT path differs from locked source')
    model=V2TaskConditionedPredictor(seed=int(job['seed']),video_mode=str(job['video_mode']),waypoint_mode=None)
    model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
    loaded=state_hash(model)
    if loaded!=manifest['final_state_hash']:raise RuntimeError('fresh process state hash mismatch')
    if parameter_count(model)!=manifest['parameter_count']:raise RuntimeError('fresh process parameter count mismatch')
    finite=all(torch.isfinite(x).all().item() for x in model.state_dict().values())
    if not finite:raise RuntimeError('nonfinite model tensor')
    receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'program_id':job['program_id'],'seed':int(job['seed']),'fresh_process':True,'device':'CPU','model_path':str(model_path),'model_sha256':sha(model_path),'model_state_hash':loaded,'parameter_count':parameter_count(model),'finite_model_tensors':finite,'init_path':str(init),'init_sha256':sha(init),'init_state_hash':manifest['init_state_hash'],'epochs_completed':done['epochs_completed'],'optimizer_steps':done['optimizer_steps'],'normalizer_sha256':sha(normalizer),'train_npz_sha256':manifest['train_selected_npz_sha256'],'final_heldout_payload_access':0,'checked_utc':utc()}
    p=ROOT/'qa/FIT_RELOAD'/f"{job['job_id']}.json";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=='__main__':main()

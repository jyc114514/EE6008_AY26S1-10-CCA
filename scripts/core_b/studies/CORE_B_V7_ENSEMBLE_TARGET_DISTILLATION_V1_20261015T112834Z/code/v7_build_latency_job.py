#!/usr/bin/env python3
"""Register one optional cached-feature latency job through the shared watcher."""
from __future__ import annotations
import os
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python';WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,obj)->None:p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.part');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def main()->int:
 role_map=json.loads((ROOT/'protocol/TEACHER_ROLE_MAP.json').read_text());seed=202610131
 teacher=role_map['sources'][str(seed)]
 student=ROOT/'model_only/VD_SV'/f'seed{seed}'/'attempt_01'
 out=ROOT/'results/latency';cfg={'job_id':'v7_cached_feature_head_latency_s202610131','seed':seed,'train_prefix_path':str(ROOT/'data/TRAIN_PREFIXES_NO_LABELS.npz'),
      'normalizer_path':str(ROOT/'normalizers/NORMALIZER_FORMAL.json'),'warmup':10,'repeats':100,'output_root':str(out),
      'models':[{'name':'VD_SV','role':'student','program_id':'VD_SV','video_mode':'TS11-D','model_path':str(student/'model.safetensors'),'model_manifest_path':str(student/'MODEL_MANIFEST.json')},
                {'name':'S_OPT','role':'teacher_head','video_mode':'STATE-CTRL','model_path':teacher['S_OPT']['model_path'],'model_manifest_path':teacher['S_OPT']['model_manifest_path']},
                {'name':'V_MATCHED','role':'teacher_head','video_mode':'TS11-D','model_path':teacher['V_MATCHED']['model_path'],'model_manifest_path':teacher['V_MATCHED']['model_manifest_path']}]}
 cp=ROOT/'configs/fits/v7_cached_feature_head_latency.json';write(cp,cfg)
 job={'job_id':cfg['job_id'],'protocol_id':'coreb_v7_cached_feature_head_latency','output_root':str(out),'completion_marker':'LATENCY.json',
      'command':[str(PYTHON),str(ROOT/'code/v7_latency.py'),'--job-json',str(cp)],
      'resource_estimate':{'measured_peak_mib':384,'basis':'accepted V4 same-Q0 batch64 train estimate; V7 models/head inputs are architecture-matched'},'log_path':str(out/'watcher_child.log')}
 write(ROOT/'coordination/queues/LATENCY.json',{'schema_version':1,'phase':'cached_feature_head_latency','jobs':[job]})
 print(json.dumps({'status':'PASS','job_id':cfg['job_id'],'models':3,'seed':seed,'warmup':10,'repeats':100,'registered_peak_mib':384},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

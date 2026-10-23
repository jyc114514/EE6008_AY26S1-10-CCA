#!/usr/bin/env python3
"""Build immutable V7 watcher and standalone-inference job specifications."""
from __future__ import annotations
import os
import hashlib,json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python'
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
TASKS=[119,94,90,141,230];SEEDS=[202610131,202610132,202610133]
PROGRAMS={'ST_SELF':('STATE-CTRL','A'),'ST_SS':('STATE-CTRL','SS50'),'ST_SV':('STATE-CTRL','SV50'),
          'VD_SELF':('TS11-D','V'),'VD_SS':('TS11-D','SS50'),'VD_SV':('TS11-D','SV50')}
PEAK_MIB=384

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(path)
def watcher_job(job_id:str,protocol_id:str,output:Path,script:Path,config:Path)->dict[str,Any]:
    marker='PREDICTION_COMPLETE.json' if script.name in {'v7_teacher_predict.py','v7_infer.py'} else 'RUN_COMPLETE.json'
    return {'job_id':job_id,'protocol_id':protocol_id,'output_root':str(output),'command':[str(PYTHON),str(script),'--job-json',str(config)],
            'completion_marker':marker,
            'resource_estimate':{'measured_peak_mib':PEAK_MIB,'basis':'pre-registered from accepted same-architecture V4 batch64 train peak estimate 384 MiB; update from V7 pilot actual reserved peak'},
            'log_path':str(output/'watcher_child.log')}
def main()->int:
    role_map=json.loads((ROOT/'protocol/TEACHER_ROLE_MAP.json').read_text(encoding='utf-8'))
    resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'))
    norm=Path(resolved['formal_normalizer_path']);norm_sha=resolved['formal_normalizer_sha256']
    train_prefix=Path(resolved['train_prefix_npz']);dev_prefix=Path(resolved['dev_prefix_npz'])
    train_sha=resolved['train_npz_sha256'];mix_lock=ROOT/'protocol/TARGET_MIX_LOCK.json';mix_sha=sha(mix_lock)
    teacher_jobs=[];fit_jobs=[];pilot_jobs=[];train_infer_jobs=[];dev_infer_jobs=[]
    for seed in SEEDS:
        for role in ('S_OPT','A_COMMON','V_MATCHED'):
            source=role_map['sources'][str(seed)][role]
            job_id=f'v7_teacher_{role}_s{seed}';out=ROOT/'teacher_predictions/source'/role/f'seed{seed}'
            config={'job_id':job_id,'teacher_role':role,'seed':seed,'video_mode':source['video_mode'],'model_path':source['model_path'],
                    'model_sha256':source['model_sha256'],'model_manifest_path':source['model_manifest_path'],
                    'model_manifest_sha256':source['model_manifest_sha256'],'normalizer_path':str(norm),'normalizer_sha256':norm_sha,
                    'train_prefix_path':str(train_prefix),'train_prefix_sha256':resolved['train_prefix_npz_sha256'],'batch_size':64,
                    'output_root':str(out),'expected_rows':1280,'forward_receives_GT':False}
            cp=ROOT/'configs/fits'/f'{job_id}.json';write(cp,config)
            teacher_jobs.append(watcher_job(job_id,f'coreb_v7_teacher_{role.lower()}_s{seed}',out,ROOT/'code/v7_teacher_predict.py',cp))
    for seed in SEEDS:
        for program,(mode,teacher_role) in PROGRAMS.items():
            job_id=f'v7_fit_{program}_s{seed}';slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';slot.mkdir(parents=True,exist_ok=True)
            teacher_dir=ROOT/'teacher_targets'/program/f'seed{seed}'
            source=role_map['sources'][str(seed)]['A_COMMON']
            config={'job_id':job_id,'phase':'formal','program_id':program,'model_id':program,'teacher_role':teacher_role,
                    'seed':seed,'scope':'shared','task_id':None,'task_ids':TASKS,'video_mode':mode,'waypoint_mode':None,
                    'init_path':source['init_path'],'expected_init_state_hash':source['init_state_hash'],
                    'normalizer_path':str(norm),'normalizer_sha256':norm_sha,'train_npz_path':resolved['train_npz'],
                    'train_role':'outer_train','teacher_target_path':str(teacher_dir/'Z_TRAIN_NATIVE.npy'),
                    'teacher_target_rows_path':str(teacher_dir/'TARGET_ROWS.csv'),'teacher_target_manifest_path':str(teacher_dir/'TARGET_MANIFEST.json'),
                    'teacher_target_sha256':'filled_after_target_build','target_mix_lock_sha256':mix_sha,'beta':0.5,
                    'lr':0.0005,'epochs':160,'effective_batch':64,'microbatch_size':64,'accumulation_steps':1,
                    'scheduler_tmax':3200,'checkpoint_epochs':[160],'gradient_clip_norm':1.0,'weight_decay':0.0001,
                    'dtype':'float32','amp':False,'compile':False,'oom_fallback':{'microbatch_size':32,'accumulation_steps':2},
                    'job_root':str(slot),'attempt_dir':str(slot),'output_root':str(slot),'final_heldout_payload_access':0}
            cp=ROOT/'configs/fits'/f'{job_id}.json';write(cp,config)
            fit_jobs.append(watcher_job(job_id,f'coreb_v7_fit_{program.lower()}_s{seed}',slot,ROOT/'code/v7_train.py',cp))
            if program=='ST_SV' and seed==SEEDS[0]:pilot_jobs.append(watcher_job(job_id,f'coreb_v7_fit_{program.lower()}_s{seed}',slot,ROOT/'code/v7_train.py',cp))
            for dataset_role,prefix,expected in (('train',train_prefix,1280),('dev',dev_prefix,185)):
                infer_id=f'v7_pred_{dataset_role}_{program}_s{seed}';out=ROOT/'predictions'/dataset_role/program/f'seed{seed}'/'attempt_01'
                model_path=slot/'model.safetensors';model_manifest=slot/'MODEL_MANIFEST.json'
                infer={'job_id':infer_id,'program_id':program,'seed':seed,'video_mode':mode,'dataset_role':dataset_role,
                       'model_path':str(model_path),'model_manifest_path':str(model_manifest),'normalizer_path':str(norm),
                       'normalizer_sha256':norm_sha,'prefix_path':str(prefix),'prefix_sha256':sha(prefix),
                       'expected_rows':expected,'batch_size':64,'output_root':str(out)}
                icp=ROOT/'configs/fits'/f'{infer_id}.json';write(icp,infer)
                job=watcher_job(infer_id,f'coreb_v7_pred_{dataset_role}_{program.lower()}_s{seed}',out,ROOT/'code/v7_infer.py',icp)
                (train_infer_jobs if dataset_role=='train' else dev_infer_jobs).append(job)
    # Training configs receive exact target hashes after the CPU target builder seals the arrays.
    queues=ROOT/'coordination/queues'
    write(queues/'TEACHER.json',{'schema_version':1,'phase':'teacher_train_prefix_replay','jobs':teacher_jobs})
    write(queues/'PILOT.json',{'schema_version':1,'phase':'one_formal_student_pilot','jobs':pilot_jobs})
    pilot_ids={j['job_id'].replace('v7_fit_','v7_pred_train_').replace('_s202610131','_s202610131') for j in pilot_jobs}
    pilot_infer=[j for j in train_infer_jobs+dev_infer_jobs if j['job_id'] in {'v7_pred_train_ST_SV_s202610131','v7_pred_dev_ST_SV_s202610131'}]
    write(queues/'PILOT_INFERENCE.json',{'schema_version':1,'phase':'pilot_standalone_train_dev_inference','jobs':pilot_infer})
    write(queues/'FITS_REMAINDER.json',{'schema_version':1,'phase':'remaining_17_formal_student_fits','jobs':[j for j in fit_jobs if j not in pilot_jobs]})
    write(queues/'TRAIN_INFERENCE.json',{'schema_version':1,'phase':'student_train_prefix_predictions','jobs':train_infer_jobs})
    write(queues/'DEV_INFERENCE.json',{'schema_version':1,'phase':'student_dev_prefix_predictions','jobs':dev_infer_jobs})
    summary={'schema_version':1,'status':'PASS','teacher_prediction_jobs':len(teacher_jobs),'formal_fit_jobs':len(fit_jobs),'pilot_fit_jobs':len(pilot_jobs),
             'remaining_fit_jobs':len(fit_jobs)-len(pilot_jobs),'student_train_inference_jobs':len(train_infer_jobs),'student_dev_inference_jobs':len(dev_infer_jobs),
             'student_programs':list(PROGRAMS),'seeds':SEEDS,'registered_peak_mib':PEAK_MIB,'registered_free_memory_admission_mib':PEAK_MIB+512,
             'watcher_path':str(WATCHER),'watcher_sha256':sha(WATCHER),'concurrency_cap_source':'shared watcher code; parsed by v7_run_queue.py',
             'standalone_inference_config_forbidden_keys':['teacher_target_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path']}
    write(ROOT/'coordination/JOB_PLAN.json',summary)
    print(json.dumps(summary,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

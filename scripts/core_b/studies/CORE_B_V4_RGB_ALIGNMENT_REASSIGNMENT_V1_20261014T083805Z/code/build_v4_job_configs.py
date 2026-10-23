import os
#!/usr/bin/env python3
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
PYTHON=str(PROJECT/'envs/jyc/bin/python')
WATCHER=str(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py')
SEEDS=[202610131,202610132,202610133]
TASKS=[119,94,90,141,230]
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    fit_watchers={}; fit_paths={}
    for seed in SEEDS:
        gv=V2/f'model_only/formal/v2_formal_G_V_lr0p0005_s{seed}'
        mf=json.loads((gv/'MODEL_MANIFEST.json').read_text())
        for program,mode in [('A_STATE_COMMON','STATE-CTRL'),('C_STATE_NEIGHBOR_REASSIGNED','TS11-D')]:
            job_id=f'v4_fit_{program}_s{seed}'
            out=ROOT/'model_only/formal'/program/f'seed{seed}'/'attempt_01'
            cfg={'schema_version':1,'job_id':job_id,'phase':'formal','program_id':program,'model_id':program,'scope':'shared','task_id':None,'task_ids':TASKS,'video_mode':mode,'waypoint_mode':None,'seed':seed,'lr':0.0005,'epochs':160,'checkpoint_epochs':[],'microbatch_size':64,'accumulation_steps':1,'effective_batch':64,'oom_fallback':{'microbatch_size':32,'accumulation_steps':2,'effective_batch':64},'normalizer_path':str(ROOT/'normalizers/NORMALIZER_FORMAL.json'),'train_role':'outer_train','init_path':mf['init_path'],'expected_init_state_hash':mf['init_state_hash'],'job_root':str(out),'attempt_dir':str(out),'final_heldout_payload_access':0}
            if program.startswith('C_'):cfg['donor_candidate_path']=str(ROOT/'protocol/TRAIN_DONOR_CANDIDATES.npz')
            cp=ROOT/'configs/fits'/f'{job_id}.json';put(cp,cfg);fit_paths[job_id]=cp
            watch={'schema_version':1,'job_id':job_id,'protocol_id':'V4_FIXED_RGB_ALIGNMENT_REASSIGNMENT_V1','output_root':str(out),'command':[PYTHON,str(ROOT/'code/v4_train.py'),'--job-json',str(cp)],'resource_estimate':{'measured_peak_mib':384,'basis':'V2 accepted G-V formal training allocated 137.5/reserved 176 MiB; V3 observed model-side maximum allocated 224/reserved 276 MiB; preregistered 384 MiB before V4 pilot'},'completion_marker':'RUN_COMPLETE.json','max_oom_requeues':1}
            wp=ROOT/'configs/watcher'/f'{job_id}.json';put(wp,watch);fit_watchers[job_id]=wp
    pilot='v4_fit_C_STATE_NEIGHBOR_REASSIGNED_s202610131'
    remainder=[k for k in fit_watchers if k!=pilot]
    put(ROOT/'coordination/queues/TRAIN_PILOT.json',{'schema_version':1,'phase':'TRAIN_PILOT','jobs':[str(fit_watchers[pilot])]})
    put(ROOT/'coordination/queues/TRAIN_REMAINDER.json',{'schema_version':1,'phase':'TRAIN_REMAINDER','jobs':[str(fit_watchers[k]) for k in remainder]})
    # Frozen inference plan: 6 new matched predictions and 18 fixed cross-evaluation maps.
    infjobs=[]
    for seed in SEEDS:
        for program,cond in [('A_STATE_COMMON','MATCHED'),('C_STATE_NEIGHBOR_REASSIGNED','MATCHED')]:
            jid=f'v4_pred_{program}_s{seed}_M'
            modeldir=ROOT/'model_only/formal'/program/f'seed{seed}'/'attempt_01'
            out=ROOT/'predictions'/program/f'seed{seed}'/cond/'attempt_01'
            cfg={'schema_version':1,'job_id':jid,'phase':'prediction','program_id':program,'model_id':program,'condition':cond,'seed':seed,'task_ids':TASKS,'video_mode':'STATE-CTRL' if program=='A_STATE_COMMON' else 'TS11-D','model_path':str(modeldir/'model.safetensors'),'model_manifest_path':str(modeldir/'MODEL_MANIFEST.json'),'model_sha256_expected':None,'data_path':str(ROOT/'data/DEV_PREFIXES_NO_LABELS.npz'),'train_data_path':str(ROOT/'data/TRAIN_SELECTED_H30.npz'),'normalizer_path':str(ROOT/'normalizers/NORMALIZER_FORMAL.json'),'eval_donor_maps_path':str(ROOT/'protocol/EVAL_DONOR_MAPS.csv'),'resolved_paths_path':str(ROOT/'inventory/RESOLVED_PATHS.json'),'output_root':str(out),'batch_size':32,'expected_rows':185,'final_heldout_payload_access':0}
            cp=ROOT/'configs/inference'/f'{jid}.json';put(cp,cfg)
            wp=ROOT/'configs/watcher'/f'{jid}.json';put(wp,{'schema_version':1,'job_id':jid,'protocol_id':'V4_FIXED_RGB_ALIGNMENT_REASSIGNMENT_V1','output_root':str(out),'command':[PYTHON,str(ROOT/'code/v4_infer.py'),'--job-json',str(cp)],'resource_estimate':{'measured_peak_mib':160,'basis':'V2 accepted inference and V3 feature replay peaks under 100 MiB; preregistered 160 MiB'},'completion_marker':'PREDICTION_COMPLETE.json','max_oom_requeues':1})
            infjobs.append(str(wp))
        for program in ['B_MATCHED','C_STATE_NEIGHBOR_REASSIGNED']:
            for m in ['D1','D2','D3']:
                jid=f'v4_pred_{program}_s{seed}_{m}'
                if program=='B_MATCHED':
                    old=V2/f'model_only/formal/v2_formal_G_V_lr0p0005_s{seed}'
                    mf=json.loads((old/'MODEL_MANIFEST.json').read_text())
                    model_path=Path(mf['model_file']);model_manifest=old/'MODEL_MANIFEST.json';sha=mf['model_file_sha256'];mode='TS11-D'
                else:
                    modeldir=ROOT/'model_only/formal/C_STATE_NEIGHBOR_REASSIGNED'/f'seed{seed}'/'attempt_01'
                    model_path=modeldir/'model.safetensors';model_manifest=modeldir/'MODEL_MANIFEST.json';sha=None;mode='TS11-D'
                out=ROOT/'predictions'/program/f'seed{seed}'/m/'attempt_01'
                cfg={'schema_version':1,'job_id':jid,'phase':'prediction','program_id':program,'model_id':program,'condition':m,'seed':seed,'task_ids':TASKS,'video_mode':mode,'model_path':str(model_path),'model_manifest_path':str(model_manifest),'model_sha256_expected':sha,'data_path':str(ROOT/'data/DEV_PREFIXES_NO_LABELS.npz'),'train_data_path':str(ROOT/'data/TRAIN_SELECTED_H30.npz'),'normalizer_path':str(ROOT/'normalizers/NORMALIZER_FORMAL.json'),'eval_donor_maps_path':str(ROOT/'protocol/EVAL_DONOR_MAPS.csv'),'resolved_paths_path':str(ROOT/'inventory/RESOLVED_PATHS.json'),'output_root':str(out),'batch_size':32,'expected_rows':185,'final_heldout_payload_access':0}
                cp=ROOT/'configs/inference'/f'{jid}.json';put(cp,cfg)
                wp=ROOT/'configs/watcher'/f'{jid}.json';put(wp,{'schema_version':1,'job_id':jid,'protocol_id':'V4_FIXED_RGB_ALIGNMENT_REASSIGNMENT_V1','output_root':str(out),'command':[PYTHON,str(ROOT/'code/v4_infer.py'),'--job-json',str(cp)],'resource_estimate':{'measured_peak_mib':160,'basis':'V2 accepted inference and V3 feature replay peaks under 100 MiB; preregistered 160 MiB'},'completion_marker':'PREDICTION_COMPLETE.json','max_oom_requeues':1})
                infjobs.append(str(wp))
    put(ROOT/'coordination/queues/INFERENCE.json',{'schema_version':1,'phase':'INFERENCE','jobs':infjobs})
    fit_plan=[]
    for program in ['A_STATE_COMMON','C_STATE_NEIGHBOR_REASSIGNED']:
      for seed in SEEDS:fit_plan.append({'program_id':program,'seed':seed,'fit_job_id':f'v4_fit_{program}_s{seed}','accepted_init_from_v2_seed':seed,'new_fit':True})
    import csv
    with (ROOT/'protocol/FIT_PLAN.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(fit_plan[0]));w.writeheader();w.writerows(fit_plan)
    print(json.dumps({'fit_jobs':len(fit_watchers),'pilot':pilot,'remaining':len(remainder),'inference_jobs':len(infjobs),'new_fit_limit':6},indent=2))
if __name__=='__main__':main()

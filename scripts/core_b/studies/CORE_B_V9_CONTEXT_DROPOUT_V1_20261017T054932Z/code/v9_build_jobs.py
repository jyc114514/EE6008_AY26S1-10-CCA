#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse,hashlib,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
V8=PROJECT/'reports/benchmarks/CORE_B_V8_CONTROLLED_REPLAY_V1_20261016T045154Z'
PYTHON=PROJECT/'envs/jyc/bin/python'
WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
PROGRAMS={'A0':('STATE-CTRL',0.0),'AD':('STATE-CTRL',0.5),'V0':('TS11-D',0.0),'VD':('TS11-D',0.5)}
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
    tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def run_dry(job_path,job,status):
    status.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(PYTHON),str(WATCHER),'--job-json',str(job_path),'--lease-dir',str(ROOT/'coordination/leases'),
        '--status-path',str(status),'--dry-run']
    result=subprocess.run(cmd,cwd=str(ROOT),capture_output=True,text=True)
    if result.returncode!=0: raise RuntimeError('watcher dry-run failed for '+job['job_id']+': '+result.stdout+result.stderr)
    return json.loads(result.stdout)

def locked_inputs(seed,program,stage0=False):
    recipe=rj(ROOT/'protocol/RECIPE_LOCK.json')
    source=rj(ROOT/'inventory/SOURCE_MAP.json')
    init=recipe['init_by_seed'][str(seed)]
    bank=source['source_assets']
    mask_meta=rj(ROOT/'protocol/MASK_SCHEDULE.json')['bank_hashes'][str(seed)]
    sampler_path=ROOT/'protocol/SAMPLER_ORDERS.npz'
    with np.load(sampler_path,allow_pickle=False) as z: sampler=z[f'seed_{seed}'].copy()
    sampler_seq=hashlib.sha256(np.ascontiguousarray(sampler).tobytes(order='C')).hexdigest()
    cache=recipe['feature_cache_manifest']
    metric=recipe['metric_scale_lock']
    metric_lock=rj(metric['path'])
    return {'schema_version':1,'phase':'formal','job_id':('stage0_'+program+'_s'+str(seed)) if stage0 else ('v9_'+program+'_s'+str(seed)),
        'program_id':program,'seed':seed,'video_mode':PROGRAMS[program][0],'numeric_context_drop_probability':PROGRAMS[program][1],
        'target_mode':'true_y','target_expression':'ground_truth_future_Y','teacher_target_used':False,
        'train_npz_path':bank['train_npz']['path'],'train_npz_sha256':bank['train_npz']['sha256'],
        'normalizer_path':recipe['normalizer']['path'],'normalizer_sha256':recipe['normalizer']['sha256'],
        'feature_cache_manifest_path':cache['path'],'feature_cache_manifest_sha256':cache['sha256'],
        'sampler_orders_path':str(sampler_path),'sampler_orders_sha256':recipe['training']['sampler_archive_sha256'],
        'sampler_sequence_sha256':sampler_seq,
        'mask_bank_path':mask_meta['path'],'mask_bank_file_sha256':mask_meta['file_sha256'],'mask_bank_bits_sha256':mask_meta['bits_sha256'],
        'init_path':init['path'],'init_file_sha256':init['file_sha256'],'expected_init_state_hash':init['state_hash'],
        'metric_sigma':metric_lock['target_sigma'],'metric_safe_fd_scale':metric_lock['safe_fd_scale'],
        'lr':0.0005,'epochs':160,'effective_batch':64,'microbatch_size':64,'accumulation_steps':1,
        'oom_fallback':{'microbatch_size':32,'accumulation_steps':2,'effective_batch':64},
        'final_heldout_payload_access':0,'dev_or_final_target_rows':0}

def watcher_job(job,config_path,output_root,peak,protocol_id,log_path):
    return {'job_id':job['job_id'],'protocol_id':protocol_id,'output_root':str(output_root),
        'command':[str(PYTHON),str(ROOT/'code'/('v9_train.py' if protocol_id.startswith('v9_fit') or protocol_id=='v9_stage0' else 'v9_infer.py')),'--job-json',str(config_path)],
        'cwd':str(ROOT),'log_path':str(log_path),'environment':{'PYTHONUNBUFFERED':'1'},
        'resource_estimate':{'measured_peak_mib':int(peak)},
        'max_oom_requeues':0 if protocol_id=='v9_stage0' else 1}

def build_stage0(retry_label=None):
    label=str(retry_label or 'initial')
    if label=='initial':
        job_root=ROOT/'tmp/stage0';attempt=job_root/'attempt_01';config=ROOT/'configs/stage0_job.json'
        watcher_path=ROOT/'coordination/jobs/stage0.json';log_path=ROOT/'logs/stage0_watcher_child.log'
        dry_path=ROOT/'coordination/watcher_status/stage0_dryrun.json';queue_path=ROOT/'coordination/queue_stage0.json'
    else:
        job_root=ROOT/'tmp/stage0'/label;attempt=job_root/'attempt_01';config=ROOT/'configs'/f'stage0_{label}_job.json'
        watcher_path=ROOT/'coordination/jobs'/f'stage0_{label}.json';log_path=ROOT/'logs'/f'stage0_{label}_watcher_child.log'
        dry_path=ROOT/'coordination/watcher_status'/f'stage0_{label}_dryrun.json';queue_path=ROOT/'coordination'/f'queue_stage0_{label}.json'
    if attempt.exists() and any(attempt.iterdir()): raise RuntimeError('Stage0 attempt directory already contains evidence: '+str(attempt))
    job=locked_inputs(202610131,'VD',True)
    if label!='initial': job['job_id']='stage0_'+label+'_VD_s202610131'
    job['code_lock_sha256']=sha(ROOT/'protocol/CODE_LOCK.json')
    job['job_root']=str(job_root);job['attempt_dir']=str(attempt)
    job['stage0']=True;job['updates_limit']=2;job['training_runner_sha256']=sha(ROOT/'code/v9_train.py');job['v9_common_sha256']=sha(ROOT/'code/v9_common.py')
    wj(config,job)
    watch=watcher_job(job,config,attempt,256,'v9_stage0',log_path)
    wj(watcher_path,watch)
    dry=run_dry(watcher_path,watch,dry_path)
    wj(ROOT/'qa'/('WATCHER_DRYRUN_STAGE0.json' if label=='initial' else f'WATCHER_DRYRUN_STAGE0_{label.upper()}.json'),dry)
    wj(queue_path,{'schema_version':1,'jobs':[str(watcher_path)],'max_concurrent_watchers':1,'lease_dir':str(ROOT/'coordination/leases')})
    print(json.dumps({'status':'STAGE0_READY','label':label,'job_config':str(config),'watcher_job':str(watcher_path),'queue':str(queue_path),'dry_run':dry},sort_keys=True))

def accept_resource():
    candidates=sorted((ROOT/'tmp/stage0').rglob('STAGE0_COMPLETE.json'),key=lambda p:p.stat().st_mtime_ns)
    if not candidates: raise RuntimeError('no successful Stage0 receipt found')
    stage=candidates[-1];qa=ROOT/'qa/MASK_SENTINEL_AND_RNG.json'
    if not qa.is_file(): raise RuntimeError('Stage0 sentinel missing')
    s=rj(stage);q=rj(qa)
    if s.get('status')!='PASS' or s.get('updates')!=2 or q.get('status')!='PASS': raise RuntimeError('Stage0 or sentinel failed')
    peak=math.ceil(float(s['peak_memory_reserved_mib']))
    if peak<=0: raise RuntimeError('Stage0 memory peak is invalid')
    watcher_path=ROOT/'coordination/watcher_status'/f"{s['job_id']}.json"
    watcher=rj(watcher_path)
    if watcher.get('status')!='complete' or watcher.get('exit_code')!=0: raise RuntimeError('Stage0 watcher did not complete cleanly')
    lock={'schema_version':1,'status':'PASS','measured_peak_mib':peak,'peak_memory_reserved_mib':s['peak_memory_reserved_mib'],
        'peak_memory_allocated_mib':s['peak_memory_allocated_mib'],'stage0_receipt_path':str(stage),'stage0_receipt_sha256':sha(stage),
        'stage0_watcher_status_path':str(watcher_path),'stage0_watcher_status_sha256':sha(watcher_path),
        'rule':'register measured Stage0 same-runner peak; watcher margin is max(512 MiB, ceil(0.10*peak))',
        'required_free_memory_mib':peak+max(512,math.ceil(0.1*peak)),'physical_gpu':s['physical_gpu'],
        'cuda_visible_devices':s['cuda_visible_devices'],'logical_device':s['logical_device'],'created_utc':s['completed_utc']}
    wj(ROOT/'protocol/RESOURCE_LOCK.json',lock)
    print(json.dumps(lock,sort_keys=True))

def build_formal():
    lock=rj(ROOT/'protocol/RESOURCE_LOCK.json')
    if lock.get('status')!='PASS': raise RuntimeError('measured Stage0 resource lock missing')
    if rj(ROOT/'qa/MASK_SENTINEL_AND_RNG.json').get('status')!='PASS': raise RuntimeError('mask sentinel QA missing')
    # Rotate arm order by seed, preserving a complete matched fresh-start matrix.
    order={202610131:['A0','AD','V0','VD'],202610132:['AD','V0','VD','A0'],202610133:['V0','VD','A0','AD']}
    queue=[]
    for seed in SEEDS:
        for program in order[seed]:
            job=locked_inputs(seed,program)
            job['job_root']=str(ROOT/'model_only'/program/f'seed{seed}')
            job['attempt_dir']=str(Path(job['job_root'])/'attempt_01')
            job['training_runner_sha256']=sha(ROOT/'code/v9_train.py')
            job['code_lock_sha256']=sha(ROOT/'protocol/CODE_LOCK.json')
            job['v9_common_sha256']=sha(ROOT/'code/v9_common.py')
            job['recipe_lock_sha256']=sha(ROOT/'protocol/RECIPE_LOCK.json')
            job['study_lock_sha256']=sha(ROOT/'protocol/STUDY_LOCK.json')
            job['stage0']=False
            config=ROOT/'configs'/'fits'/f'{program}_seed{seed}.json';wj(config,job)
            out=Path(job['job_root'])
            out.mkdir(parents=True,exist_ok=True)
            watch=watcher_job(job,config,out,int(lock['measured_peak_mib']),'v9_fit_'+program,ROOT/'logs'/f'fit_{program}_seed{seed}.watcher_child.log')
            watch['job_id']=job['job_id']
            watch_path=ROOT/'coordination/jobs'/f'{program}_seed{seed}.json';wj(watch_path,watch)
            dry=run_dry(watch_path,watch,ROOT/'coordination/watcher_status'/f'{job["job_id"]}_dryrun.json')
            queue.append({'program_id':program,'seed':seed,'job_json':str(watch_path),'config_path':str(config),'watcher_dryrun':dry})
    if len(queue)!=12: raise RuntimeError('formal fit queue must have 12 jobs')
    wj(ROOT/'qa/WATCHER_DRYRUN_FORMAL.json',{'schema_version':1,'status':'PASS','count':len(queue),'jobs':[x['watcher_dryrun'] for x in queue],
        'dry_run_does_not_query_GPU':True,'measured_peak_mib':lock['measured_peak_mib']})
    wj(ROOT/'coordination/queue_train.json',{'schema_version':1,'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),
        'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),'resource_lock_sha256':sha(ROOT/'protocol/RESOURCE_LOCK.json'),
        'max_concurrent_watchers':2,'lease_dir':str(ROOT/'coordination/leases'),'max_wait_seconds':0,
        'rotated_order_by_seed':order,'jobs':queue})
    print(json.dumps({'status':'FORMAL_QUEUE_READY','jobs':len(queue),'measured_peak_mib':lock['measured_peak_mib'],
        'queue_path':str(ROOT/'coordination/queue_train.json')},sort_keys=True))

def build_inference():
    lock=rj(ROOT/'protocol/RESOURCE_LOCK.json')
    if lock.get('status')!='PASS': raise RuntimeError('resource lock missing')
    source=rj(ROOT/'inventory/SOURCE_MAP.json');recipe=rj(ROOT/'protocol/RECIPE_LOCK.json')
    prefix_train=source['source_assets']['train_prefix_npz'];prefix_dev=source['source_assets']['dev_prefix_npz']
    if 'target_native_h30' in np.load(prefix_train['path'],allow_pickle=False).files: raise RuntimeError('train inference prefix includes Y')
    if 'target_native_h30' in np.load(prefix_dev['path'],allow_pickle=False).files: raise RuntimeError('dev inference prefix includes Y')
    queue=[]
    for seed in SEEDS:
        for program in ('A0','AD','V0','VD'):
            model_dir=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'
            complete=model_dir/'RUN_COMPLETE.json';manifest=model_dir/'MODEL_MANIFEST.json'
            if not complete.is_file() or not manifest.is_file(): raise RuntimeError(f'formal fit missing before inference: {program}/{seed}')
            fit=rj(complete);model=rj(manifest)
            if fit.get('status')!='PASS' or int(fit.get('epochs_completed',-1))!=160 or int(fit.get('optimizer_steps',-1))!=3200 or fit.get('target_role')!='TRUE_Y':
                raise RuntimeError(f'formal fit not accepted: {program}/{seed}')
            mode=PROGRAMS[program][0]
            for dataset_role,policy,prefix,expected in (
                ('train','full_context',prefix_train,1280),
                ('dev','full_context',prefix_dev,185),
                ('dev','force_numeric_context_drop',prefix_dev,185)):
                job_id=f'v9pred_{dataset_role}_{policy}_{program}_s{seed}'
                job={'schema_version':1,'job_id':job_id,'program_id':program,'seed':seed,'video_mode':mode,
                    'dataset_role':dataset_role,'input_policy':policy,'expected_rows':expected,'batch_size':64,
                    'model_path':model['model_file'],'model_sha256':model['model_file_sha256'],
                    'model_manifest_path':str(manifest),'model_manifest_sha256':sha(manifest),
                    'normalizer_path':recipe['normalizer']['path'],'normalizer_sha256':recipe['normalizer']['sha256'],
                    'prefix_path':prefix['path'],'prefix_sha256':prefix['sha256'],
                    'feature_cache_manifest_path':recipe['feature_cache_manifest']['path'],
                    'feature_cache_manifest_sha256':recipe['feature_cache_manifest']['sha256'],
                    'output_root':str(ROOT/'predictions'/dataset_role/policy/program/f'seed{seed}'),
                    'final_heldout_payload_access':0,'code_lock_sha256':sha(ROOT/'protocol/CODE_LOCK.json')}
                config=ROOT/'configs'/'inference'/f'{job_id}.json';wj(config,job)
                out=Path(job['output_root'])
                if out.exists() and any(out.iterdir()): raise RuntimeError('inference output slot already contains data: '+str(out))
                out.mkdir(parents=True,exist_ok=True)
                job['inference_runner_sha256']=sha(ROOT/'code/v9_infer.py')
                job['v9_common_sha256']=sha(ROOT/'code/v9_common.py')
                wj(config,job)
                watch={'job_id':job_id,'protocol_id':'v9_inference_'+dataset_role+'_'+policy,'output_root':str(out),
                    'command':[str(PYTHON),str(ROOT/'code/v9_infer.py'),'--job-json',str(config)],'cwd':str(ROOT),
                    'log_path':str(ROOT/'logs'/f'{job_id}.watcher_child.log'),'environment':{'PYTHONUNBUFFERED':'1'},
                    'resource_estimate':{'measured_peak_mib':int(lock['measured_peak_mib'])},'max_oom_requeues':1}
                watch_path=ROOT/'coordination/jobs'/(job_id+'.json');wj(watch_path,watch)
                dry=run_dry(watch_path,watch,ROOT/'coordination/watcher_status'/(job_id+'_dryrun.json'))
                queue.append({'job_id':job_id,'program_id':program,'seed':seed,'dataset_role':dataset_role,'input_policy':policy,'job_json':str(watch_path),'watcher_dryrun':dry})
    if len(queue)!=36: raise RuntimeError('prediction queue must contain 36 neural bundles')
    wj(ROOT/'qa/WATCHER_DRYRUN_INFERENCE.json',{'schema_version':1,'status':'PASS','count':36,'jobs':[x['watcher_dryrun'] for x in queue],
        'dev_target_archive_opened':False,'measured_peak_mib':lock['measured_peak_mib']})
    wj(ROOT/'coordination/queue_inference.json',{'schema_version':1,'max_concurrent_watchers':2,'lease_dir':str(ROOT/'coordination/leases'),
        'max_wait_seconds':0,'new_train_bundles':12,'new_dev_full_bundles':12,'new_dev_forced_mask_bundles':12,'jobs':queue})
    print(json.dumps({'status':'INFERENCE_QUEUE_READY','jobs':len(queue),'queue_path':str(ROOT/'coordination/queue_inference.json')},sort_keys=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['stage0','accept-resource','formal','inference']);ap.add_argument('--retry-label',default=None);args=ap.parse_args()
    if args.mode=='stage0':build_stage0(args.retry_label)
    elif args.mode=='accept-resource':accept_resource()
    elif args.mode=='formal':build_formal()
    elif args.mode=='inference':build_inference()
if __name__=='__main__':main()


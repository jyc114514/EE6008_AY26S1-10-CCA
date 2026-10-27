#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code';LOCKED=CODE/'locked_source'
sys.path[:0]=[str(CODE),str(LOCKED)]
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
from v9_common import atomic_json,load_npz,model_input,prefix_row_csv,read_json,sha_file,utc
TASKS=(119,94,90,141,230)

def run(job_path:Path)->int:
    job=read_json(job_path);out=Path(job['output_root']);out.mkdir(parents=True,exist_ok=True)
    status=out/'RUN_STATUS.json';complete=out/'PREDICTION_COMPLETE.json';timer=time.perf_counter()
    forbidden={'teacher_target_path','teacher_target_rows_path','teacher_target_manifest_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path'}
    if forbidden.intersection(job): raise RuntimeError('inference job contains forbidden target/train paths: '+str(sorted(forbidden.intersection(job))))
    try:
        if not torch.cuda.is_available(): raise RuntimeError('V9 student inference requires watcher-authorized CUDA')
        if job.get('code_lock_sha256')!=sha_file(ROOT/'protocol/CODE_LOCK.json'): raise RuntimeError('V9 code lock SHA differs from inference job')
        if sha_file(Path(__file__))!=job['inference_runner_sha256']: raise RuntimeError('inference runner code hash differs from job config')
        if sha_file(ROOT/'code/v9_common.py')!=job['v9_common_sha256']: raise RuntimeError('V9 shared adapter hash differs from job config')
        seed=int(job['seed']);program=str(job['program_id']);mode=str(job['video_mode']);dataset=str(job['dataset_role']);policy=str(job['input_policy'])
        if dataset not in {'train','dev'} or mode not in {'STATE-CTRL','TS11-D'}: raise RuntimeError('invalid dataset role or video mode')
        if policy not in {'full_context','force_numeric_context_drop'}: raise RuntimeError('invalid evaluation input policy')
        if dataset=='train' and policy!='full_context': raise RuntimeError('train prediction must be full-context only')
        model_path=Path(job['model_path']);manifest_path=Path(job['model_manifest_path'])
        norm_path=Path(job['normalizer_path']);prefix_path=Path(job['prefix_path'])
        model_manifest=read_json(manifest_path)
        if model_manifest.get('status')!='PASS' or model_manifest.get('teacher_target_used') is not False: raise RuntimeError('student model manifest not accepted teacher-free')
        if model_manifest.get('program_id')!=program or int(model_manifest.get('seed',-1))!=seed or model_manifest.get('video_mode')!=mode: raise RuntimeError('model/program/seed/mode mismatch')
        if sha_file(model_path)!=job['model_sha256'] or sha_file(manifest_path)!=job['model_manifest_sha256']: raise RuntimeError('model file or manifest SHA mismatch')
        if sha_file(norm_path)!=job['normalizer_sha256'] or model_manifest.get('normalizer_sha256')!=job['normalizer_sha256']: raise RuntimeError('normalizer hash mismatch')
        if sha_file(prefix_path)!=job['prefix_sha256']: raise RuntimeError('label-free prefix hash mismatch')
        with np.load(prefix_path,allow_pickle=False) as z:
            if 'target_native_h30' in z.files: raise RuntimeError('prediction prefix archive contains future Y')
        seed_everything(seed)
        device=torch.device('cuda:0')
        model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
        missing,unexpected=model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
        if missing or unexpected: raise RuntimeError('fresh-process model strict load mismatch')
        loaded_hash=state_hash(model)
        if loaded_hash!=model_manifest.get('final_state_hash'): raise RuntimeError('fresh-process state hash mismatch')
        model=model.to(device).eval();torch.cuda.reset_peak_memory_stats(device)
        data=load_npz(prefix_path)
        n=len(data['anchor_id']);expected=int(job['expected_rows'])
        if n!=expected or n!=(1280 if dataset=='train' else 185): raise RuntimeError('prefix support count differs from lock')
        ids=data['anchor_id'].astype(str)
        if len(set(ids.tolist()))!=n: raise RuntimeError('prefix receiver IDs are not unique')
        if set(int(x) for x in np.unique(data['task_id']))!=set(TASKS): raise RuntimeError('prefix task set differs from V9')
        normalizer=read_json(norm_path)
        grid=None
        if mode!='STATE-CTRL':
            cache_path=Path(job['feature_cache_manifest_path'])
            if sha_file(cache_path)!=job['feature_cache_manifest_sha256']: raise RuntimeError('feature cache manifest hash mismatch')
            grid=Q0ShardedGrid(read_json(cache_path))
        sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
        rows=[];before=state_hash(model);forward_rows=0;masked_rows=0
        with torch.no_grad():
            for start in range(0,n,int(job.get('batch_size',64))):
                idx=np.arange(start,min(start+int(job.get('batch_size',64)),n),dtype=np.int64)
                drop=np.ones(len(idx),dtype=np.bool_) if policy=='force_numeric_context_drop' else None
                prefix,task,anchor=model_input(data,idx,normalizer,mode,device,grid,drop_mask=drop)
                output=model(prefix,task)
                native=anchor[:,None]+sigma[None,None]*output.action_residual.float()
                rows.append(native.detach().to(dtype=torch.float64).cpu())
                forward_rows+=len(idx)
                masked_rows+=int(len(idx) if drop is not None else 0)
        native=torch.cat(rows,dim=0).contiguous()
        if tuple(native.shape)!=(n,30,28) or not torch.isfinite(native).all(): raise RuntimeError('prediction shape/finite check failed')
        if state_hash(model)!=before: raise RuntimeError('model state changed during inference')
        pred_path=out/'PREDICTIONS.safetensors';tmp=pred_path.with_name('.'+pred_path.name+'.'+str(os.getpid())+'.part')
        save_file({'prediction_native':native},str(tmp));os.replace(tmp,pred_path)
        prefix_row_csv(out/'PREDICTION_ROWS.csv',data,dataset)
        loaded=load_file(str(pred_path),device='cpu')['prediction_native']
        if loaded.dtype!=torch.float64 or tuple(loaded.shape)!=(n,30,28) or not torch.isfinite(loaded).all(): raise RuntimeError('prediction safe-tensor reload failed')
        payload_sha=hashlib.sha256(loaded.numpy().tobytes(order='C')).hexdigest()
        if payload_sha!=hashlib.sha256(native.numpy().tobytes(order='C')).hexdigest(): raise RuntimeError('prediction payload changed after reload')
        receipt={'schema_version':1,'status':'PASS','completed_utc':utc(),'job_id':job['job_id'],'program_id':program,'seed':seed,
            'video_mode':mode,'training_numeric_context_drop_probability':float(model_manifest['numeric_context_drop_probability']),
            'dataset_role':dataset,'input_policy':policy,'prediction_rows':n,'forward_rows':forward_rows,
            'forced_mask_rows':masked_rows,'forced_mask_fraction':1.0 if policy=='force_numeric_context_drop' else 0.0,
            'mask_field':'prefix_inputs.state [B,32,71]','mask_location':'after accepted normalization; before state_projection',
            'independent_command_anchor_preserved':True,'task_preserved':True,'video_feature_used':mode!='STATE-CTRL',
            'state_only_RGB_masked_by_locked_model':mode=='STATE-CTRL','video_preserved_when_present':mode!='STATE-CTRL',
            'target_Y_or_teacher_paths_supplied':False,'future_Y_requested_by_inference_loader':False,
            'model_path':str(model_path),'model_sha256':sha_file(model_path),'model_state_hash':loaded_hash,
            'model_manifest_path':str(manifest_path),'model_manifest_sha256':sha_file(manifest_path),
            'normalizer_path':str(norm_path),'normalizer_sha256':sha_file(norm_path),
            'prefix_path':str(prefix_path),'prefix_sha256':sha_file(prefix_path),
            'prediction_path':str(pred_path),'prediction_sha256':sha_file(pred_path),
            'prediction_tensor_sha256':payload_sha,'prediction_rows_path':str(out/'PREDICTION_ROWS.csv'),
            'prediction_rows_sha256':sha_file(out/'PREDICTION_ROWS.csv'),'prediction_shape':list(native.shape),'prediction_dtype':'float64',
            'fresh_model_state_reload':True,'prediction_safetensors_reload':True,'parameter_count':parameter_count(model),
            'job_config_path':str(job_path),'job_config_sha256':sha_file(job_path),'inference_runner_path':str(Path(__file__).resolve()),
            'inference_runner_sha256':sha_file(Path(__file__).resolve()),
            'runtime':{'hostname':os.uname().nodename,'python_version':sys.version.split()[0],'torch_version':torch.__version__,
                'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False},
            'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device),
            'peak_memory_allocated_mib':float(torch.cuda.max_memory_allocated(device)/(1024*1024)),
            'peak_memory_reserved_mib':float(torch.cuda.max_memory_reserved(device)/(1024*1024)),
            'elapsed_seconds':time.perf_counter()-timer,'final_heldout_payload_access':0}
        atomic_json(complete,receipt);atomic_json(status,{'status':'complete','completed_utc':utc(),'job_id':job['job_id']})
        print(json.dumps({'status':'PASS','job_id':job['job_id'],'rows':n,'policy':policy,'prediction_sha256':receipt['prediction_sha256'],
            'peak_reserved_mib':receipt['peak_memory_reserved_mib']},sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        atomic_json(status,{'status':'failed','failed_utc':utc(),'job_id':job.get('job_id'),'error_type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc(),'partial_forward_rows':int(locals().get('forward_rows',0)),
            'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'final_heldout_payload_access':0})
        print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),flush=True)
        return 1

def seed_everything(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);args=ap.parse_args()
    raise SystemExit(run(args.job_json))
if __name__=='__main__':main()


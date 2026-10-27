#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
PROGRAMS=('A0','AD','V0','VD')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.part')
 tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def read_rows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def prefix_metadata(path,expected):
 with np.load(path,allow_pickle=False) as z:
  forbidden={'target_native_h30','target','future_y'} & set(z.files)
  if forbidden:raise RuntimeError('label-free prefix contains target keys: '+str(sorted(forbidden)))
  required={'anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame'}
  if not required.issubset(z.files):raise RuntimeError('prefix metadata fields missing')
  data={k:z[k].copy() for k in required}
 ids=data['anchor_id'].astype(str).tolist()
 if len(ids)!=expected or len(set(ids))!=expected:raise RuntimeError('prefix ID support invalid')
 return ids,data
def fit_acceptance():
 train=0;rows=[]
 for seed in SEEDS:
  for program in PROGRAMS:
   out=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'
   complete=rj(out/'RUN_COMPLETE.json');manifest=rj(out/'MODEL_MANIFEST.json')
   if complete.get('status')!='PASS' or manifest.get('status')!='PASS':raise RuntimeError('formal fit completion missing')
   if int(complete.get('epochs_completed',-1))!=160 or int(complete.get('optimizer_steps',-1))!=3200:raise RuntimeError('formal fit not 160x3200')
   if complete.get('target_role')!='TRUE_Y' or complete.get('teacher_target_used') is not False:raise RuntimeError('fit target role is not true Y only')
   if int(complete.get('dev_or_final_target_rows',-1))!=0:raise RuntimeError('fit read a dev/final target')
   if complete.get('model_state_hash')!=manifest.get('final_state_hash') or complete.get('model_sha256')!=manifest.get('model_file_sha256'):raise RuntimeError('fit complete/manifest mismatch')
   model_path=Path(manifest['model_file'])
   if sha(model_path)!=manifest['model_file_sha256']:raise RuntimeError('fit model SHA mismatch')
   cfg_path=Path(manifest['job_config_path']);cfg=rj(cfg_path)
   if sha(cfg_path)!=manifest.get('job_config_sha256') or cfg.get('code_lock_sha256')!=sha(ROOT/'protocol/CODE_LOCK.json'):raise RuntimeError('fit job/code lock mismatch')
   if cfg.get('training_runner_sha256')!=sha(ROOT/'code/v9_train.py'):raise RuntimeError('fit runner code hash mismatch')
   if manifest.get('training_runner_sha256')!=sha(ROOT/'code/v9_train.py'):raise RuntimeError('fit runner SHA mismatch')
   if manifest.get('target_mode')!='true_y' or manifest.get('target_role')!='TRUE_Y':raise RuntimeError('fit manifest target is not true Y')
   if int(manifest.get('target_presentations',-1))!=204800:raise RuntimeError('formal true Y presentation count differs from 160x1280')
   if len(read_rows(out/'EPOCH_LOG.csv'))!=160:raise RuntimeError('formal training curve is incomplete')
   if len(read_rows(out/'MASK_EPOCH_LOG.csv'))!=800:raise RuntimeError('formal per-task mask log is incomplete')
   rows.append({'program_id':program,'seed':seed,'model_sha256':manifest['model_file_sha256'],'final_state_hash':manifest['final_state_hash'],
     'sampler_sequence_sha256':manifest['sampler_sequence_sha256'],'mask_bank_bits_sha256':manifest['mask_bank_bits_sha256'],
     'target_presentations':manifest['target_presentations'],'peak_reserved_mib':manifest['peak_memory_reserved_mib']})
   train+=1
 execution=rj(ROOT/'coordination/EXECUTION_TRAIN.json')
 if execution.get('status')!='complete' or len(execution.get('results',[]))!=12 or execution.get('failed_jobs'):raise RuntimeError('formal queue not complete 12/12')
 for row in execution['results']:
  if int(row.get('exit_code',1))!=0:raise RuntimeError('formal watcher exit nonzero')
 return rows
def main():
 seal_path=ROOT/'qa/PREDICTION_SEAL.json'
 if seal_path.exists():
  if (ROOT/'qa/DEV_TARGET_READ_STARTED.json').exists() or (ROOT/'qa/DEV_TARGET_SINGLE_READ.json').exists():raise SystemExit('dev target read started; prediction seal is immutable')
  old_seal=rj(seal_path)
  if old_seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or old_seal.get('dev_target_archive_opened') is not False:raise RuntimeError('existing seal is not an unexposed pre-target seal')
  backup=seal_path.with_name('PREDICTION_SEAL.json.pre-metadata-reseal-'+utc().replace(':','').replace('-','')+'.bak')
  shutil.copy2(seal_path,backup)
 study=rj(ROOT/'protocol/STUDY_LOCK.json');recipe=rj(ROOT/'protocol/RECIPE_LOCK.json')
 scorer_lock=rj(ROOT/'protocol/SCORING_CODE_LOCK.json')
 if scorer_lock.get('status')!='LOCKED_BEFORE_DEV_TARGET_ACCESS' or scorer_lock.get('files',{}).get('code/v9_seal_predictions.py',{}).get('sha256')!=sha(ROOT/'code/v9_seal_predictions.py'):
  raise RuntimeError('prediction sealer code hash differs from pre-target lock')
 resource=rj(ROOT/'protocol/RESOURCE_LOCK.json');preflight=rj(ROOT/'qa/FORMAL_JOB_PREFLIGHT.json')
 if study.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV' or recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV':raise RuntimeError('study/recipe lock absent')
 if resource.get('status')!='PASS' or preflight.get('status')!='PASS':raise RuntimeError('resource or formal job preflight absent')
 if int(resource.get('final_heldout_payload_access',-1))!=0:raise RuntimeError('final access lock broken')
 fits=fit_acceptance()
 source=rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']
 train_ids,train_meta=prefix_metadata(Path(source['train_prefix_npz']['path']),1280)
 dev_ids,dev_meta=prefix_metadata(Path(source['dev_prefix_npz']['path']),185)
 queue=rj(ROOT/'coordination/queue_inference.json')
 if len(queue.get('jobs',[]))!=36:raise RuntimeError('inference queue does not contain 36 jobs')
 records=[];seen=set();expected={}
 for item in queue['jobs']:
  watcher_path=Path(item['job_json']);watcher_job=rj(watcher_path)
  job_path=ROOT/'configs/inference'/f"{item['job_id']}.json";job=rj(job_path)
  if watcher_job.get('job_id')!=item['job_id'] or Path(watcher_job['command'][-1])!=job_path:raise RuntimeError('watcher job does not bind to expected inference config')
  key=(job['dataset_role'],job['input_policy'],job['program_id'],int(job['seed']))
  if key in seen:raise RuntimeError('duplicate prediction job key')
  seen.add(key)
  if job.get('teacher_target_path') or job.get('dev_target_path') or 'train_npz_path' in job:raise RuntimeError('inference job contains target/train path')
  role=job['dataset_role'];policy=job['input_policy'];program=job['program_id'];seed=int(job['seed'])
  if program not in PROGRAMS or seed not in SEEDS:raise RuntimeError('prediction program/seed outside lock')
  if role=='train':
   if policy!='full_context':raise RuntimeError('only full-context train prediction is allowed')
   ids=train_ids;meta=train_meta;expected_n=1280
  elif role=='dev':
   if policy not in {'full_context','force_numeric_context_drop'}:raise RuntimeError('dev policy invalid')
   ids=dev_ids;meta=dev_meta;expected_n=185
  else:raise RuntimeError('dataset role invalid')
  out=Path(job['output_root']);complete=rj(out/'PREDICTION_COMPLETE.json')
  if complete.get('status')!='PASS' or complete.get('dataset_role')!=role or complete.get('input_policy')!=policy:raise RuntimeError('prediction receipt role/policy mismatch')
  if int(complete.get('seed',-1))!=seed or complete.get('program_id')!=program:raise RuntimeError('prediction receipt program/seed mismatch')
  if int(complete.get('prediction_rows',-1))!=expected_n or int(complete.get('forward_rows',-1))!=expected_n:raise RuntimeError('prediction receipt row count mismatch')
  if int(complete.get('final_heldout_payload_access',-1))!=0:raise RuntimeError('prediction receipt final access not zero')
  if complete.get('target_Y_or_teacher_paths_supplied') is not False or complete.get('future_Y_requested_by_inference_loader') is not False:raise RuntimeError('inference target isolation failed')
  expected_mask=1.0 if policy=='force_numeric_context_drop' else 0.0
  if float(complete.get('forced_mask_fraction',-1))!=expected_mask:raise RuntimeError('prediction input policy mask fraction mismatch')
  if complete.get('independent_command_anchor_preserved') is not True or complete.get('task_preserved') is not True:raise RuntimeError('inference lost command anchor/task')
  cfg_path=Path(complete['job_config_path']);cfg=rj(cfg_path)
  if sha(cfg_path)!=complete.get('job_config_sha256') or cfg.get('code_lock_sha256')!=sha(ROOT/'protocol/CODE_LOCK.json'):raise RuntimeError('prediction job/code lock mismatch')
  if cfg.get('inference_runner_sha256')!=sha(ROOT/'code/v9_infer.py') or cfg.get('v9_common_sha256')!=sha(ROOT/'code/v9_common.py'):raise RuntimeError('inference code hash mismatch')
  pred_path=Path(complete['prediction_path']);rows_path=Path(complete['prediction_rows_path'])
  if sha(pred_path)!=complete['prediction_sha256'] or sha(rows_path)!=complete['prediction_rows_sha256']:raise RuntimeError('prediction bundle SHA mismatch')
  tensor=load_file(str(pred_path),device='cpu')
  if set(tensor)!={'prediction_native'}:raise RuntimeError('prediction tensor keys invalid')
  pred=tensor['prediction_native']
  if tuple(pred.shape)!=(expected_n,30,28) or pred.dtype.__str__()!='torch.float64' or not __import__('torch').isfinite(pred).all():raise RuntimeError('prediction tensor shape/dtype/finite invalid')
  rows=read_rows(rows_path)
  if len(rows)!=expected_n or [r['anchor_id'] for r in rows]!=ids:raise RuntimeError('prediction ordered IDs differ from locked prefix')
  for i,row in enumerate(rows):
   m=meta
   if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']),int(row['anchor_frame'])) != (
      int(m['episode_id'][i]),int(m['task_id'][i]),str(m['source_cluster'][i]),int(m['anchor_frame'][i])):
      raise RuntimeError('prediction row identity mismatch for '+str(row['anchor_id']))
  expected[key]=complete
  records.append({'job_id':complete['job_id'],'program_id':program,'seed':seed,'dataset_role':role,'input_policy':policy,
      'prediction_path':str(pred_path),'prediction_sha256':complete['prediction_sha256'],'prediction_rows_path':str(rows_path),'prediction_rows_sha256':complete['prediction_rows_sha256'],
      'model_sha256':complete['model_sha256'],'model_state_hash':complete['model_state_hash'],'prediction_rows':expected_n,
      'input_policy_masked_fraction':complete['forced_mask_fraction'],'physical_gpu':complete['physical_gpu'],'peak_reserved_mib':complete['peak_memory_reserved_mib']})
 if len(seen)!=36 or len(records)!=36:raise RuntimeError('expected exactly 36 neural bundles')
 if sum(1 for k in seen if k[0]=='train')!=12 or sum(1 for k in seen if k[0]=='dev' and k[1]=='full_context')!=12 or sum(1 for k in seen if k[0]=='dev' and k[1]=='force_numeric_context_drop')!=12:raise RuntimeError('train/full/masked bundle counts invalid')
 for seed in SEEDS:
  for program in PROGRAMS:
   a=expected[('dev','full_context',program,seed)];b=expected[('dev','force_numeric_context_drop',program,seed)]
   if a['model_sha256']!=b['model_sha256'] or a['model_state_hash']!=b['model_state_hash']:raise RuntimeError('full/masked prediction model differs')
 # Freeze the six fixed-alpha dev mixtures before reading any new dev target.
 source_map=rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']['accepted_S_OPT_dev_predictions']
 dev_index={aid:i for i,aid in enumerate(dev_ids)};mixture_records=[]
 for seed in SEEDS:
  srec=source_map[str(seed)];sp=Path(srec['prediction']['path']);sr=Path(srec['rows']['path'])
  if sha(sp)!=srec['prediction']['sha256'] or sha(sr)!=srec['rows']['sha256']:raise RuntimeError('accepted S_OPT source checksum mismatch')
  st=load_file(str(sp),device='cpu')
  if 'prediction_native' not in st:raise RuntimeError('S_OPT prediction tensor key missing')
  sv=np.asarray(st['prediction_native'].numpy(),dtype=np.float64)
  if sv.shape!=(185,30,28) or not np.isfinite(sv).all():raise RuntimeError('S_OPT prediction array invalid')
  srows=read_rows(sr)
  if len(srows)!=185:raise RuntimeError('S_OPT row map support invalid')
  sid=[r['anchor_id'] for r in srows]
  if len(set(sid))!=185 or set(sid)!=set(dev_ids):raise RuntimeError('S_OPT dev IDs differ from locked V4 prefix')
  sorder=np.asarray([sid.index(aid) for aid in dev_ids],dtype=np.int64);sv=sv[sorder]
  for row in srows:
   aid=row['anchor_id'];i=dev_index[aid];m=dev_meta
   if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']),int(row['anchor_frame']))!=(int(m['episode_id'][i]),int(m['task_id'][i]),str(m['source_cluster'][i]),int(m['anchor_frame'][i])):
    raise RuntimeError('S_OPT row identity differs from locked dev prefix')
  v0=load_file(expected[('dev','full_context','V0',seed)]['prediction_path'],device='cpu')['prediction_native'].numpy()
  vd=load_file(expected[('dev','full_context','VD',seed)]['prediction_path'],device='cpu')['prediction_native'].numpy()
  for method,values,video_method in (('MIX0',0.5*sv+0.5*v0,'V0'),('MIXD',0.5*sv+0.5*vd,'VD')):
   if values.shape!=(185,30,28) or values.dtype!=np.float64 or not np.isfinite(values).all():raise RuntimeError('derived mixture array invalid')
   out=ROOT/'predictions/dev_derived'/method/f'seed{seed}'
   pred_path=out/'PREDICTIONS.safetensors';rows_path=out/'PREDICTION_ROWS.csv';done=out/'DERIVED_COMPLETE.json'
   if out.exists() and done.is_file():
    rec=rj(done)
    if rec.get('status')!='PASS' or rec.get('formula')!='0.5*S_OPT + 0.5*'+video_method or sha(pred_path)!=rec.get('prediction_sha256') or sha(rows_path)!=rec.get('prediction_rows_sha256'):
     raise RuntimeError('existing derived prediction receipt/hash mismatch')
    if not np.array_equal(load_file(str(pred_path),device='cpu')['prediction_native'].numpy(),values):raise RuntimeError('existing derived mix differs from locked source arrays')
   else:
    if out.exists() and any(out.iterdir()):raise RuntimeError('partial derived output preserved; refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    tmp=pred_path.with_name('.'+pred_path.name+'.part')
    save_file({'prediction_native':__import__('torch').from_numpy(np.ascontiguousarray(values))},str(tmp));os.replace(tmp,pred_path)
    rows_src=Path(expected[('dev','full_context',video_method,seed)]['prediction_rows_path']);shutil.copy2(rows_src,rows_path)
    rec={'schema_version':1,'status':'PASS','created_utc':utc(),'program_id':method,'seed':seed,'dataset_role':'dev','input_policy':'derived_full_context',
     'formula':'0.5*S_OPT + 0.5*'+video_method,'alpha_S_OPT':0.5,'alpha_'+video_method:0.5,'source_S_OPT_prediction_sha256':srec['prediction']['sha256'],
     'source_video_prediction_sha256':expected[('dev','full_context',video_method,seed)]['prediction_sha256'],
     'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),'prediction_rows_path':str(rows_path),'prediction_rows_sha256':sha(rows_path),
     'prediction_shape':[185,30,28],'prediction_dtype':'float64','final_heldout_payload_access':0}
    wj(done,rec)
   mixture_records.append({'program_id':method,'seed':seed,'formula':rec['formula'],'prediction_path':str(pred_path),'prediction_sha256':rec['prediction_sha256'],
    'prediction_rows_path':str(rows_path),'prediction_rows_sha256':rec['prediction_rows_sha256'],'source_S_OPT_prediction_sha256':rec['source_S_OPT_prediction_sha256'],
    'source_video_prediction_sha256':rec['source_video_prediction_sha256'],'prediction_shape':[185,30,28]})
 if len(mixture_records)!=6:raise RuntimeError('expected six pre-seal derived mixture bundles')
 seal={'schema_version':1,'status':'SEALED_BEFORE_TARGET_ACCESS','created_utc':utc(),'study':'post_V8_training_only_numeric_context_dropout_development',
  'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
  'resource_lock_sha256':sha(ROOT/'protocol/RESOURCE_LOCK.json'),'code_lock_sha256':sha(ROOT/'protocol/CODE_LOCK.json'),
  'formal_fit_count':len(fits),'accepted_train_bundles':12,'accepted_dev_full_context_bundles':12,'accepted_dev_forced_mask_bundles':12,
  'accepted_dev_bundles':30,'accepted_dev_neural_bundles':24,'accepted_dev_derived_bundles':6,'accepted_dev_bundles_full_context_primary':18,
  'accepted_neural_prediction_bundles':36,'accepted_derived_prediction_bundles':6,'missing_bundles':0,'dev_target_archive_opened':False,
  'preseal_dev_target_reads':0,'dev_target_content_reads_before_seal':0,'final_heldout_payload_access':0,
  'prediction_records':records,'derived_prediction_records':mixture_records,'fit_records':fits,'train_prefix_sha256':source['train_prefix_npz']['sha256'],'dev_prefix_sha256':source['dev_prefix_npz']['sha256']}
 wj(ROOT/'qa/PREDICTION_SEAL.json',seal)
 exposure=rj(ROOT/'inventory/OUTCOME_EXPOSURE.json')
 exposure.update({'status':'predictions_sealed_before_dev_target','v9_formal_fits_expected':12,'v9_formal_fits_completed':12,
  'v9_true_Y_presentations':sum(int(x['target_presentations']) for x in fits),'v9_train_prediction_bundles':12,
  'v9_dev_full_prediction_bundles':12,'v9_dev_forced_mask_prediction_bundles':12,'v9_dev_derived_mixture_bundles':6,'v9_neural_prediction_bundles':36,
  'v9_dev_target_archive_content_reads':0,'v9_dev_target_archive_opened_before_seal':False,'v9_expected_new_dev_logic_bundles':30,
  'v9_expected_primary_new_dev_loss_evaluations':3330,'v9_expected_forced_mask_loss_evaluations':2220,
  'v9_expected_total_new_dev_loss_evaluations':5550,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0})
 wj(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 print(json.dumps({'status':'SEALED_BEFORE_TARGET_ACCESS','fits':len(fits),'neural_predictions':len(records),'derived_dev_predictions':len(mixture_records),
  'train_bundles':12,'dev_full':12,'dev_masked':12,'dev_mixtures':6,'dev_target_reads':0,'final_access':0},sort_keys=True))
if __name__=='__main__':main()


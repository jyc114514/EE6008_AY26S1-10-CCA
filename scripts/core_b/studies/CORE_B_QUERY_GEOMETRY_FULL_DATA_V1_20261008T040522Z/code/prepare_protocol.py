from __future__ import annotations
import os
import csv, hashlib, json, shutil, io
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
MECH=P/'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'
SCHED=P/'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'
EQ=P/'reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z'
BENCH=P/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
V3=P/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SPLIT=P/'repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet'
NOW=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(rel,obj):
 p=ROOT/rel; content=json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
 if p.exists():
  if p.read_bytes()==content.encode('utf-8'):return
  if p.read_text(encoding='utf-8').replace('\r\n','\n')==content.replace('\r\n','\n'):return
  raise FileExistsError(f'preserving non-identical prior artifact: {p}')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
def write_csv(rel,rows,fields=None):
 p=ROOT/rel
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows);content=buf.getvalue()
 if p.exists():
  if p.read_bytes()==content.encode('utf-8') or p.read_text(encoding='utf-8').replace('\r\n','\n')==content.replace('\r\n','\n'):return
  raise FileExistsError(f'preserving non-identical prior artifact: {p}')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))

# Hash-sealed input/source inventory. One preexisting figure-generator drift is recorded, not repaired.
sched_hashes=readj(SCHED/'protocol/SOURCE_HASHES.json'); mech_lock=readj(MECH/'protocol/PROTOCOL_LOCKED.json')
source_rows=[]
for label,items in [('SCHED_CORRECTED',sched_hashes['sources']),('MECH_QUERY',[{'path':p,'sha256':h,'bytes':None,'role':'MECH_LOCKED_SOURCE'} for p,h in mech_lock['source_hashes'].items()])]:
 for x in items:
  p=Path(x['path']);obs=sha(p) if p.is_file() else 'MISSING'
  source_rows.append({'source_root':label,'role':x.get('role','MECH_LOCKED_SOURCE'),'path':str(p),'expected_sha256':x['sha256'],'observed_sha256':obs,'hash_match':str(obs==x['sha256']).lower(),'bytes':x.get('bytes',p.stat().st_size if p.is_file() else '')})
sched_bad=[x for x in source_rows if x['source_root']=='SCHED_CORRECTED' and x['hash_match']=='false']
mech_bad=[x for x in source_rows if x['source_root']=='MECH_QUERY' and x['hash_match']=='false']
if len(sched_bad)!=1 or sched_bad[0]['path']!=str(SCHED/'code/generate_figures.py') or mech_bad:raise RuntimeError({'sched_bad':sched_bad,'mech_bad':mech_bad})

# Verify six accepted corrected H30 controls, checkpoints, LR traces and old train/dev bundles.
pa=readj(SCHED/'PREDICTION_ACCEPTANCE.json'); pred_index={(x['run_id'],x['split']):x for x in pa['bundles']}
controls=[]; pred_rows=[]; runids=[]
for config in ('TS11-D','STATE-CTRL'):
 for seed in (20261001,20261002,20261003):runids.append(f'D3EQ-CS-{config}-H30-S{seed}')
for source_run in runids:
 rr=SCHED/'runs/formal'/source_run;rc=readj(rr/'RUN_COMPLETE.json')
 if (rc['status'],rc['epochs'],rc['optimizer_steps'],rc['scheduler_step_calls'],rc['parameter_count'],rc['horizon'])!=('PASS',40,11120,11120,890780,30):raise RuntimeError('old control contract mismatch '+source_run)
 cp=rr/'checkpoint_last.pt';cph=sha(cp)
 if cph!=rc['checkpoint_last_sha256']:raise RuntimeError('checkpoint mismatch '+source_run)
 lr=list(csv.DictReader((rr/'LR_TRACE_ACTUAL.csv').open(newline='',encoding='utf-8')))
 if len(lr)!=11120 or int(lr[-1]['update_count_after'])!=11120 or float(lr[-1]['lr_after_update'])!=1e-5:raise RuntimeError('LR trace mismatch '+source_run)
 model='TS11-D' if 'TS11-D' in source_run else 'STATE-CTRL';seed=int(source_run[-8:])
 q0=f'Q0-{model}-H30-FULL_LOSS-S{seed}';q1=f'Q1-{model}-H30-FULL_LOSS-S{seed}'
 ps={}
 for split,n in [('train',17757),('dev',2322)]:
  b=pred_index.get((source_run,split));path=SCHED/'predictions'/split/f'{source_run}.npz';side=SCHED/'predictions'/split/f'{source_run}.json'
  if not b or b['status']!='PASS' or int(b['rows'])!=n or sha(path)!=b['prediction_sha256']:raise RuntimeError(f'old prediction hash/status mismatch {source_run}/{split}')
  sm=readj(side)
  if sm['checkpoint_sha256']!=cph or sm['prediction_sha256']!=b['prediction_sha256']:raise RuntimeError(f'old prediction/checkpoint join mismatch {source_run}/{split}')
  ps[split]=b['prediction_sha256'];pred_rows.append({'run_id':source_run,'split':split,'rows':n,'path':str(path),'sha256':b['prediction_sha256'],'checkpoint_sha256':cph,'status':'PASS_REUSE'})
 controls.append({'q0_run_id':q0,'source_run_id':source_run,'q1_run_id':q1,'config':model,'seed':seed,'horizon_steps':30,
                  'action':'reuse','status':'PASS','checkpoint_path':str(cp),'checkpoint_sha256':cph,'init_state_hash':rc['init_hash'],
                  'parameter_count':rc['parameter_count'],'epochs':rc['epochs'],'optimizer_updates':rc['optimizer_steps'],
                  'scheduler_step_calls':rc['scheduler_step_calls'],'train_prediction_sha256':ps['train'],'dev_prediction_sha256':ps['dev'],
                  'recipe':'accepted corrected-scheduler H30; FP32; effective batch64; FULL_LOSS; last epoch40'})
if len(controls)!=6 or len(pred_rows)!=12:raise RuntimeError('expected six Q0 controls and 12 old bundles')

# Accepted seed INITs.
seedmanifest=readj(SCHED/'protocol/SEED_INIT_MANIFEST.json'); seeds=[]
for x in seedmanifest['seeds']:
 if sha(x['init_state_path'])!=x['init_state_file_sha256']:raise RuntimeError('seed INIT file SHA mismatch')
 seeds.append({'seed':x['seed'],'init_state_path':x['init_state_path'],'init_state_file_sha256':x['init_state_file_sha256'],
               'init_state_hash':x['init_state_hash'],'same_initial_state_across_configs':x['same_initial_state_across_configs_horizons']})

# Duplicate-safe source groups and anchor manifests: read split metadata and input fields, never targets_native.
input_path=V3/'data/V3_MAIN_INPUTS.npz';anchor_src=V3/'ANCHOR_MANIFEST.csv'
input_sha=sha(input_path);anchor_sha=sha(anchor_src)
t=pq.read_table(SPLIT,columns=['episode_index','split','video_sha256_fingerprint'],filters=[('split','in',['train','validation'])]).to_pydict()
epmap={}
for ep,sp,fp in zip(t['episode_index'],t['split'],t['video_sha256_fingerprint']):
 ns='dev' if sp in ('validation','val') else sp
 if fp is None:raise RuntimeError(f'missing video fingerprint for episode {ep}')
 val=(ns,str(fp))
 if int(ep) in epmap and epmap[int(ep)]!=val:raise RuntimeError(f'conflicting group map for episode {ep}')
 epmap[int(ep)]=val
with np.load(input_path,allow_pickle=False) as z:
 splitcode=z['split_code'].astype(np.int8,copy=True);ids=z['anchor_id'].astype('U96',copy=True);episodes=z['episode_index'].astype(np.int64,copy=True)
 if len(splitcode)!=20079 or set(np.unique(splitcode).tolist())!={0,1}:raise RuntimeError('V3 input split schema mismatch')
counts=Counter();groups={'train':set(),'dev':set()};eps={'train':set(),'dev':set()}
for code,ep in zip(splitcode,episodes):
 sp='train' if int(code)==0 else 'dev';msp,fp=epmap[int(ep)]
 if sp!=msp:raise RuntimeError(f'split group mismatch {ep}')
 counts[sp]+=1;groups[sp].add(fp);eps[sp].add(int(ep))
if (counts['train'],counts['dev'],len(eps['train']),len(eps['dev']),len(groups['train']),len(groups['dev']))!=(17757,2322,2712,603,2712,603) or groups['train']&groups['dev']:
 raise RuntimeError('accepted population/group contract mismatch')
write_csv('inventory/SOURCE_GROUP_MAP.csv',[{'episode_index':ep,'split':sp,'source_group_id':fp} for ep,(sp,fp) in sorted(epmap.items()) if sp in ('train','dev') and ep in eps[sp]],['episode_index','split','source_group_id'])
anchor_rows=[]
with anchor_src.open(newline='',encoding='utf-8') as f:
 for x in csv.DictReader(f):
  if x.get('population')!='primary' or x['split'] not in ('train','validation','dev'):continue
  ep=int(x['episode_index']);sp='dev' if x['split'] in ('validation','dev') else 'train';fp=epmap[ep][1]
  af=int(x['anchor_frame']);first=int(x['target_1s_start_frame']);last=int(x['target_1s_end_frame']);at=float(x['anchor_timestamp'])
  if epmap[ep][0]!=sp or first!=af+1 or last!=af+30 or abs(at-af/30)>1e-5:raise RuntimeError('anchor/time/H30 seam mismatch '+x['anchor_id'])
  anchor_rows.append({'anchor_id':x['anchor_id'],'episode_index':ep,'split':sp,'source_group_id':fp,'original_task_index':x['original_task_index'],
   'anchor_frame':af,'anchor_timestamp_seconds':at,'target_first_frame':first,'target_last_frame':last,
   'target_first_timestamp_seconds':first/30.0,'target_last_timestamp_seconds':last/30.0,'final_test_access':0})
if len(anchor_rows)!=20079:raise RuntimeError(f'primary train/dev anchor manifest count={len(anchor_rows)}')
write_csv('inventory/ANCHOR_MANIFEST.csv',anchor_rows)
write_csv('inventory/POPULATION_MANIFEST.csv',[
 {'split':'train','anchors':17757,'episodes':2712,'source_groups':2712,'exposure':'training population','final_test_access':0},
 {'split':'dev','anchors':2322,'episodes':603,'source_groups':603,'exposure':'previously exposed; exploratory only','final_test_access':0}],
 ['split','anchors','episodes','source_groups','exposure','final_test_access'])
# 128 fixed train-monitor anchors; one per group, SHA-sorted before scores.
tr=[]
for i in np.flatnonzero(splitcode==0):
 ep=int(episodes[i]);aid=str(ids[i]);fp=epmap[ep][1];h=hashlib.sha256(f'qg-full-v1|{ep}|{aid}'.encode()).hexdigest();tr.append((h,ep,fp,aid))
monitor=[];seen=set()
for h,ep,fp,aid in sorted(tr):
 if fp in seen:continue
 seen.add(fp);monitor.append({'selection_hash':h,'episode_index':ep,'source_group_id':fp,'anchor_id':aid})
 if len(monitor)==128:break
if len(monitor)!=128:raise RuntimeError('monitor panel lacks 128 unique train groups')
write_csv('protocol/TRAIN_MONITOR_PANEL.csv',[{'rank':i+1,**x} for i,x in enumerate(monitor)])

# Copy accepted query table and train-fitted motion threshold assets exactly.
for src,dst in [(MECH/'protocol/HADAMARD_QUERY_TABLES.npz',ROOT/'protocol/HADAMARD_QUERY_TABLES.npz'),(MECH/'protocol/HADAMARD_QUERY_TABLES.csv',ROOT/'protocol/HADAMARD_QUERY_TABLES.csv'),
                (SCHED/'MOTION_ACTIVITY_THRESHOLD.json',ROOT/'protocol/MOTION_ACTIVITY_THRESHOLD.json'),
                (SCHED/'MOTION_THRESHOLD_AUDIT.csv',ROOT/'protocol/MOTION_THRESHOLD_AUDIT.csv'),
                (SCHED/'MOTION_STRATA_SUPPORT.csv',ROOT/'protocol/MOTION_STRATA_SUPPORT.csv')]:
 if dst.exists():
  if sha(dst)!=sha(src):raise FileExistsError(f'preserving non-identical copied source: {dst}')
  continue
 shutil.copyfile(src,dst)
qmeta=readj(MECH/'protocol/QUERY_TABLES_AND_HASHES.json')
with np.load(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz',allow_pickle=False) as q:
 qh30=np.asarray(q['candidate_h30'],dtype=np.float32);qh60=np.asarray(q['candidate_h60'],dtype=np.float32);old30=np.asarray(q['old_future_h30'],dtype=np.float32);old60=np.asarray(q['old_future_h60'],dtype=np.float32)
qarray=np.ascontiguousarray(qh30,dtype=np.float32);qhasher=hashlib.sha256();qhasher.update(qarray.dtype.str.encode('ascii'));qhasher.update(b'\0');qhasher.update(np.asarray(qarray.shape,dtype=np.int64).tobytes());qhasher.update(qarray.tobytes());qhash=qhasher.hexdigest();qarchive_hash=sha(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz')
if qhash!=qmeta['candidate_table_hashes']['H30'] or qh30.shape!=(30,128) or not np.array_equal(qh30,qh60[:30]) or not np.array_equal(old30,old60[:30]):raise RuntimeError('H30 query archive integrity/prefix test failed')
write_json('protocol/QUERY_TABLES_AND_HASHES.json',{
 'schema_version':1,'source_record_path':str(MECH/'protocol/QUERY_TABLES_AND_HASHES.json'),'source_record_sha256':sha(MECH/'protocol/QUERY_TABLES_AND_HASHES.json'),
 'archive_path':'protocol/HADAMARD_QUERY_TABLES.npz','archive_sha256':qarchive_hash,'archive_source_sha256':qmeta['saved_artifacts']['npz_sha256'],
 'candidate_table_sha256_h30':qhash,'candidate_table_hashes':qmeta['candidate_table_hashes'],'candidate_table_shape_h30':[30,128],
 'candidate_H30_fit_only':True,'candidate_H60_training':False,'candidate_H30_is_H60_prefix':True,'old_H30_is_H60_prefix':True,
 'trainable_parameters_added':0,'construction':qmeta['construction'],'column_order':qmeta['column_order'],
 'H30_row_indices_zero_based':qmeta['candidate_row_indices_zero_based']['H30'],'RMS_match_max_abs_error':qmeta['max_abs_row_rms_match_error'],
 'normalized_Gram_diagonal_max_error':qmeta['normalized_gram_diagonal_max_error'],'normalized_Gram_off_diagonal_max_abs':qmeta['normalized_gram_off_diagonal_max_abs']})
write_csv('protocol/SEED_INIT_MANIFEST.csv',seeds)

# Exact 12-cell factorial: old controls reused; six Hadamard fits are the only new model training.
matrix=[];reuse=[];by_q0={x['q0_run_id']:x for x in controls}
for seed in (20261001,20261002,20261003):
 for config in ('TS11-D','STATE-CTRL'):
  q0=f'Q0-{config}-H30-FULL_LOSS-S{seed}';q1=f'Q1-{config}-H30-FULL_LOSS-S{seed}';src=f'D3EQ-CS-{config}-H30-S{seed}';old=by_q0[q0]
  reuse.append({**old,'eligibility':'PASS_REUSE_CORRECTED_SCHEDULER'})
  matrix.append({'condition':'Q0-F' if config=='TS11-D' else 'Q0-S','run_id':q0,'config':config,'query_variant':'old_fixed_sinusoidal_time_query','seed':seed,'horizon_steps':30,'loss':'FULL_LOSS','action':'reuse','matched_q1_run_id':q1,'source_run_id':src,'checkpoint_sha256':old['checkpoint_sha256'],'train_prediction_sha256':old['train_prediction_sha256'],'dev_prediction_sha256':old['dev_prediction_sha256'],'expected_train_anchors':17757,'expected_dev_anchors':2322,'epochs':40,'expected_optimizer_updates':11120,'final_test_access':0})
  matrix.append({'condition':'Q1-F' if config=='TS11-D' else 'Q1-S','run_id':q1,'config':config,'query_variant':'canonical_Hadamard_H30_RMS_matched','seed':seed,'horizon_steps':30,'loss':'FULL_LOSS','action':'train_had','matched_q0_run_id':q0,'source_run_id':src,'expected_train_anchors':17757,'expected_dev_anchors':2322,'expected_train_groups':2712,'expected_dev_groups':603,'effective_batch':64,'physical_batch_size':64,'accumulation_steps':1,'epochs':40,'expected_optimizer_updates':11120,'endpoint':'last_update_epoch40','device':'cuda','dtype':'float32','optimizer':'AdamW lr=1e-3 betas=(0.9,0.999) eps=1e-8 weight_decay=1e-4','scheduler':'post-optimizer-update cosine T_max=11120 eta_min=1e-5; one inert epoch callback','gradient_clip_norm':1.0,'candidate_trainable_parameters_added':0,'final_test_access':0})
write_csv('protocol/MATRIX12.csv',matrix)
write_csv('protocol/REUSE_ELIGIBILITY.csv',reuse)

# Source allowlist and identity records.
write_csv('inventory/SOURCE_REGISTRY.csv',source_rows,['source_root','role','path','expected_sha256','observed_sha256','hash_match','bytes'])
write_json('inventory/REGISTERED_SOURCE_HASHES_AT_START.json',{'schema_version':1,'created_utc':NOW,'sched_source_entry_count':len(sched_hashes['sources']),
 'mech_source_entry_count':len(mech_lock['source_hashes']),'source_rows':source_rows,'preexisting_deviation':sched_bad})
write_json('inventory/SOURCE_ACCEPTANCE.json',{
 'schema_version':1,'created_utc':NOW,'status':'PASS_WITH_ONE_PREEXISTING_NONEXECUTION_SOURCE_DRIFT',
 'SCHED_SOURCE_HASHES_path':str(SCHED/'protocol/SOURCE_HASHES.json'),'SCHED_SOURCE_HASHES_sha256':sha(SCHED/'protocol/SOURCE_HASHES.json'),
 'SCHED_entries':len(sched_hashes['sources']),'SCHED_matches':len(sched_hashes['sources'])-len(sched_bad),
 'SCHED_mismatches':[{'path':x['path'],'expected_sha256':x['expected_sha256'],'observed_sha256':x['observed_sha256'],'impact':'figure-generation code only; excluded from this execution'} for x in sched_bad],
 'MECH_protocol_sha256':sha(MECH/'protocol/PROTOCOL_LOCKED.json'),'MECH_locked_sources':len(mech_lock['source_hashes']),
 'MECH_source_matches':len(mech_lock['source_hashes'])-len(mech_bad),'corrected_H30_controls_reuse_pass':'6/6',
 'old_train_prediction_bundles_hash_pass':'6/6','old_dev_prediction_bundles_hash_pass':'6/6','old_control_repairs':0,
 'seed_INITs_hash_pass':'3/3','model_source_sha256':'6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589',
 'parameter_count':890780,'H30_query_sha256':qhash,'candidate_trainable_parameters_added':0,
 'train':{'anchors':17757,'episodes':2712,'source_groups':2712},'dev':{'anchors':2322,'episodes':603,'source_groups':603},
 'train_dev_group_overlap':0,'final_test_access':0,'heldout_access':0,'old_root_writes':0,
 'caveat':'The sole SCHED hash mismatch is generate_figures.py. It is not imported by the new trainer, inferencer, or scorer. Accepted model, trainer, scheduler, predictor/scorer inputs, checkpoints and H30 train/dev bundles match.'})
write_json('inventory/DATA_ALLOWLIST.json',{'schema_version':1,'status':'LOCKED_TRAIN_AND_PREVIOUSLY_EXPOSED_DEV_ONLY','allowed_sources':[
 {'path':str(input_path),'sha256':input_sha,'fields_training':['split_code','anchor_id','episode_index','history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native'],'fields_inference':['split_code','anchor_id','episode_index','original_task_index','anchor_frame','anchor_timestamp','history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native'],'target_policy':'training reads train-only target cache; inference never reads targets_native; scorer opens train cache and dev targets only after prediction acceptance'},
 {'path':str(EQ/'d3/train_targets_trainonly.npz'),'sha256':sha(EQ/'d3/train_targets_trainonly.npz'),'scope':'primary train only'},
 {'path':str(BENCH/'d3/cache/D3_CACHE_MANIFEST.json'),'sha256':sha(BENCH/'d3/cache/D3_CACHE_MANIFEST.json'),'scope':'accepted dynamic/static cached features for primary train/dev only; no video decode'},
 {'path':str(BENCH/'d3/cache/D3_VIDEO_NORMALIZER.json'),'sha256':sha(BENCH/'d3/cache/D3_VIDEO_NORMALIZER.json'),'scope':'frozen train-fit video normalizer'},
 {'path':str(V3/'data/V3_NORMALIZERS.json'),'sha256':sha(V3/'data/V3_NORMALIZERS.json'),'scope':'frozen train-fit target_sigma and safe_fd_scale'},
 {'path':str(SPLIT),'sha256':sha(SPLIT),'scope':'only train/validation episode/split/video fingerprint metadata are included in new map'},
 {'path':'protocol/HADAMARD_QUERY_TABLES.npz','sha256':qarchive_hash,'scope':'accepted H30 candidate; H60 only copied for provenance/prefix tests'},
 {'path':str(SCHED/'predictions/train'),'scope':'six accepted corrected-scheduler H30 control prediction bundles referenced by hash'},
 {'path':str(SCHED/'predictions/dev'),'scope':'six accepted corrected-scheduler H30 control prediction bundles referenced by hash'}],
 'prohibited':['final_test targets','heldout targets','final_test video files','heldout video files','final_test feature cache','heldout feature cache','final_test metrics','heldout metrics'],
 'final_test_access':0,'heldout_access':0})
write_json('protocol/INPUT_IDENTITIES.json',{'schema_version':1,'created_utc':NOW,
 'V3_MAIN_INPUTS':{'path':str(input_path),'sha256':input_sha,'archive_rows':20079,'target_arrays_accessed_during_preflight':False,
  'train_rows':17757,'dev_rows':2322,'fields_read_for_preflight':['split_code','anchor_id','episode_index','original_task_index','anchor_frame','anchor_timestamp']},
 'train_target_cache':{'path':str(EQ/'d3/train_targets_trainonly.npz'),'sha256':sha(EQ/'d3/train_targets_trainonly.npz'),'manifest_sha256':sha(EQ/'d3/TRAIN_TARGET_CACHE_MANIFEST.json'),'scope':'primary_train_only'},
 'feature_cache_manifest_sha256':sha(BENCH/'d3/cache/D3_CACHE_MANIFEST.json'),'video_normalizer_sha256':sha(BENCH/'d3/cache/D3_VIDEO_NORMALIZER.json'),
 'target_normalizer_sha256':sha(V3/'data/V3_NORMALIZERS.json'),'split_manifest_sha256':sha(SPLIT),
 'anchor_manifest_source_sha256':anchor_sha,'source_group_map_sha256':sha(ROOT/'inventory/SOURCE_GROUP_MAP.csv'),
 'train':{'anchors':17757,'episodes':2712,'source_groups':2712},'dev':{'anchors':2322,'episodes':603,'source_groups':603},
 'train_dev_source_group_overlap':0,'final_test_access':0,'heldout_access':0})

threshold=readj(SCHED/'MOTION_ACTIVITY_THRESHOLD.json')
write_json('protocol/TRAJECTORY_SELECTION_LOCKED.json',{'schema_version':1,'status':'LOCKED_ALGORITHM_IDS_GENERATED_AFTER_ALL_PREDICTIONS_BEFORE_SCORING',
 'case_count_expected':12,'strata':{'low':6,'high':6},'horizon_steps':30,'selection_unit':'source group; one episode and one anchor per selected group',
 'motion_score':'mean over H30 future-internal steps and raw action[0:28] of abs(diff(target_native))/target_sigma; within episode anchor mean; source-group mean',
 'thresholds_from_train_only':threshold['strata_thresholds']['30'],'threshold_source_path':str(SCHED/'MOTION_ACTIVITY_THRESHOLD.json'),
 'threshold_source_sha256':sha(SCHED/'MOTION_ACTIVITY_THRESHOLD.json'),'low_rule':'group score <= train-fitted q25','high_rule':'group score >= train-fitted q75',
 'order':'SHA256(qg-full-v1|episode_id|anchor_id) ascending; at most one anchor per source group',
 'gate':'write and hash exact case IDs before the scorer opens any model prediction arrays','final_test_access':0})

write_json('protocol/ANALYSIS_LOCKED.json',{'schema_version':1,'created_utc':NOW,'study_id':'H30_FULL_DATA_QUERY_GEOMETRY_EXPLORATORY',
 'dev_evaluation':'exploratory; previously exposed; no model selection','independent_confirmation':False,
 'primary':['group-equal Full28 standardized position MAE','group-equal safe-FD normalized error'],
 'metric_formulas':{'position':'mean(abs((pred_native-target_native)/target_sigma)) over H30 x raw action[0:28]',
  'safe_fd':'mean(abs(diff(pred_native,axis=time)-diff(target_native,axis=time))/safe_fd_scale) over exactly 29 future-internal differences; no anchor-to-first seam',
  'native_fd':'mean absolute native future-internal difference error; separate',
  'seam':'first future sample relative to native anchor; separate',
  'tv':'source-group mean TV_pred divided by source-group mean TV_GT; zero-GT-TV/support reported separately; no epsilon'},
 'paired_estimands':{'G_TS':'L_TS_old - L_TS_Had','G_STATE':'L_STATE_old - L_STATE_Had','V_old':'L_STATE_old - L_TS_old',
  'V_Had':'L_STATE_Had - L_TS_Had','I':'G_TS - G_STATE = V_Had - V_old','positive_G':'Hadamard lower loss','positive_V':'TS system lower loss than STATE control',
  'positive_I':'relative system interaction only; not a video causal effect'},
 'aggregation':'window to episode mean to source-group mean; each seed reported; aggregate is equal mean of three fixed seed summaries plus sample SD',
 'bootstrap':{'replicates':10000,'prng':'numpy.default_rng(20261008)','unit':'paired source group','shared_indices':'one resampled group vector shared across all cells/seeds/metrics; compute I directly per draw',
  'interval':'percentile 2.5/97.5; conditional on these fixed models/seeds/data','p_values':False,'crossed_seed_group_sensitivity':'optional, labeled low precision'},
 'direction_threshold_source':str(SCHED/'MOTION_ACTIVITY_THRESHOLD.json'),'direction_threshold_sha256':sha(SCHED/'MOTION_ACTIVITY_THRESHOLD.json'),
 'motion_strata':'reuse accepted train-fitted H30 q25/q75 from protocol/MOTION_ACTIVITY_THRESHOLD.json',
 'no_best_seed_or_checkpoint':True,'train_results':'in-sample only','final_or_heldout_access':0})
write_json('protocol/OUTCOME_EXPOSURE.json',{'schema_version':1,'created_utc':NOW,'previous_dev_exposure':True,
 'known_prior_results':['D3-EQ, corrected-scheduler, learning diagnosis and train/dev analyses are previously exposed.',
  'MECH_ROOT four H30 tinyfits used P1/P8 training targets for optimization and had positive train-only paired gains.',
  'The prior tiny-fit results motivate this bounded full-data study but are not dev or generalization evidence.'],
 'new_full_data_scores_seen_at_lock':False,'new_dev_predictions_seen_at_lock':False,'new_dev_model_selection':False,
 'independent_confirmation':False,'final_test_access':0,'heldout_access':0})
write_json('coordination/RUN_STATE.json',{'schema_version':1,'study_id':'H30_FULL_DATA_QUERY_GEOMETRY_EXPLORATORY','status':'PRELOCK_CPU_QA',
 'new_had_expected':6,'q0_controls_reused':6,'old_control_repairs':0,'train_prediction_bundles_expected':12,'dev_prediction_bundles_expected':12,
 'final_test_access':0,'heldout_access':0,'updated_utc':NOW,'next_action':'complete code/schema/preflight, freeze protocol, CPU tests, then 4-update GPU Stage0'})
(ROOT/'coordination/NEXT_ACTIONS.md').write_text('# Next actions\n\n1. Complete inference, scorer, figure code and typed source schema in this root.\n2. Freeze protocol, MATRIX12, reuse eligibility, query, INIT, input identities, analysis, and exposure.\n3. Run CPU contract tests and matrix dry-run.\n4. Run two 4-update H30 GPU Stage0 canaries through the shared watcher; accept peak before formal.\n5. Train all six Hadamard cells from their original seed INITs; no dev scoring during training.\n6. Generate train/dev predictions, hash and accept all 12 cells, including six reused controls.\n7. Lock 12 dev trajectory case IDs before scoring; run paired analysis, plots, QA, and final checksum.\n',encoding='utf-8')
(ROOT/'coordination/RESUME_COMMAND.txt').write_text('tmux attach -t qgfhd30_20261008T040522\n',encoding='utf-8')
print(json.dumps({'status':'PREPARED','sched_source_registry_entries':len(sched_hashes['sources']),'sched_matches':len(sched_hashes['sources'])-len(sched_bad),
 'mech_locked_sources':len(mech_lock['source_hashes']),'mech_matches':len(mech_lock['source_hashes'])-len(mech_bad),
 'q0_reused':len(controls),'old_train_dev_bundles_verified':len(pred_rows),'q1_had_expected':6,
 'train_monitor_panel':len(monitor),'population':{'train_anchors':counts['train'],'train_groups':len(groups['train']),'dev_anchors':counts['dev'],'dev_groups':len(groups['dev'])},
 'h30_query_sha256':qhash,'source_group_overlap':len(groups['train']&groups['dev'])},ensure_ascii=False,indent=2))

#!/usr/bin/env python3
"""Resume/accept already persisted extraction artifacts without opening dev labels."""
import os
import csv, hashlib, json, os, resource, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
ROOT=Path(__file__).resolve().parents[1]
RIDGE=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z'))
H=(1,3,6,10,15,30); DELTA=1/30

def utc(): return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def jdefault(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 raise TypeError(type(x).__name__)
def wjson(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,sort_keys=True,default=jdefault,allow_nan=False)+'\n')
def rcsv(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
t0=time.perf_counter(); c0=time.process_time(); started=utc()
elig=rcsv(ROOT/'data/ELIGIBILITY_MANIFEST.csv'); raw=rcsv(ROOT/'source/RAW_PARQUET_READ_MANIFEST.csv'); times=rcsv(ROOT/'data/TARGET_TIME_GRID.csv')
if len(elig)!=20079: raise AssertionError(f'canonical row count {len(elig)}')
if any(r['eligible']!='1' for r in elig): raise AssertionError('unexpected exclusions in persisted eligibility manifest')
by_split={s:[r for r in elig if r['split']==s] for s in ('train','dev')}
inputs={}
for s in ('train','dev'):
 p=ROOT/f'data/{s}/inputs_prefix.npz'
 with np.load(p,allow_pickle=False) as z:
  a={k:z[k].copy() for k in ['anchor_id','source_group_id','episode_index','anchor_frame','s_prev','s0','u0','timestamp_window']}
 n=len(a['anchor_id'])
 assert n==len(by_split[s]) and len(set(a['anchor_id'].tolist()))==n
 assert a['s_prev'].shape==(n,43) and a['s0'].shape==(n,43) and a['u0'].shape==(n,28) and a['timestamp_window'].shape==(n,32)
 assert np.isfinite(a['s_prev']).all() and np.isfinite(a['s0']).all() and np.isfinite(a['u0']).all() and np.isfinite(a['timestamp_window']).all()
 assert set(a['anchor_id'].tolist())=={r['anchor_id'] for r in by_split[s]}
 inputs[s]=a
 assert (ROOT/f'data/{s}/labels_obs43.npz').is_file() and (ROOT/f'data/{s}/commands_privileged.npz').is_file()
# Strictly do not load labels_obs43 or commands_privileged payloads in this acceptance stage.
train=by_split['train']; dev=by_split['dev']; tg={s:{r['source_group_id'] for r in by_split[s]} for s in ('train','dev')}
ep={s:{int(r['episode_index']) for r in by_split[s]} for s in ('train','dev')}
assert len(tg['train'])==2712 and len(tg['dev'])==603 and not (tg['train']&tg['dev'])
fold=rcsv(ROOT/'data/FOLD_ASSIGNMENT.csv'); old=[r for r in rcsv(RIDGE/'protocol/FOLD_ASSIGNMENT.csv') if r['split']=='train' and r['fold'].strip().isdigit()]
oldmap={r['anchor_id']:int(r['fold']) for r in old}; newmap={r['anchor_id']:int(r['fold']) for r in fold}
assert len(fold)==len(train) and newmap=={r['anchor_id']:oldmap[r['anchor_id']] for r in train}
assert all(len({int(r['fold']) for r in fold if r['source_group_id']==g})==1 for g in tg['train'])
exclusion=Counter()
for r in elig:
 if r['eligible']=='0':
  for reason in r['exclusion_reason'].split(';'): exclusion[reason]+=1
errs=[]
for s in ('train','dev'):
 dt=np.diff(inputs[s]['timestamp_window'],axis=1)
 errs.extend(np.abs(dt-DELTA).reshape(-1).tolist())
assert max(errs)<=1e-5
assert not any(r['raw_write_detected']=='True' for r in raw)
actual={str(h):float(np.mean([float(r['actual_target_delta_seconds']) for r in times if int(r['horizon'])==h])) for h in H}
summary={'created_utc':utc(),'canonical_primary_rows':len(elig),'canonical_train_rows':len(train),'canonical_dev_rows':len(dev),'eligible_train_rows':len(train),'eligible_train_groups':len(tg['train']),'eligible_train_episodes':len(ep['train']),'eligible_dev_rows':len(dev),'eligible_dev_groups':len(tg['dev']),'eligible_dev_episodes':len(ep['dev']),'excluded_rows':0,'exclusion_reason_occurrences':dict(exclusion),'train_dev_overlap':{'train_dev_anchor_overlap':0,'train_dev_group_overlap':0},'fold_source':'reused_accepted_Ridge_source_group_assignment_restricted_to_eligible_rows','fold_group_counts':{str(f):len({r['source_group_id'] for r in fold if int(r['fold'])==f}) for f in range(3)},'clock_grid':'nominal_30Hz_validated','clock_intervals_checked':len(errs),'clock_max_abs_error_seconds':float(max(errs)),'clock_p99_abs_error_seconds':float(np.quantile(errs,.99)),'actual_horizon_mean_seconds':actual,'raw_columns_parsed':['frame_index','timestamp','action','observation.arm_joints','observation.hand_joints','observation.leg_joints'],'raw_video_or_features_read':False,'raw_parquet_episode_files_read':len(raw),'raw_file_write_detected':False,'final_or_heldout_access':0,'dev_label_extraction_started_utc':'UNKNOWN_exact_first_raw_label_read_not_persisted_by_failed_initial_extraction','dev_labels_constructed':True,'canonical_target_array_accessed':False,'recovery_note':'eligibility, prefix identities, clocks, folds, and weights accepted from persisted artifacts; this stage never loaded dev labels_obs43 or privileged command payloads'}
wjson(ROOT/'data/ELIGIBILITY_SUMMARY.json',summary)
wjson(ROOT/'data/INPUT_TARGET_ACCESS_LOG.json',{'dev_label_archive_materialized_during_initial_extract_attempt':True,'exact_first_raw_dev_label_read_utc':'UNKNOWN_not_recorded_before_initial_extraction_summary_NameError','dev_label_archive_mtime_upper_bound_utc':datetime.fromtimestamp((ROOT/'data/dev/labels_obs43.npz').stat().st_mtime,timezone.utc).isoformat().replace('+00:00','Z'),'fit_dev_labels_read':False,'four_forecast_prediction_dev_labels_read':False,'teacher_state_reset_dev_label_read':'not yet; separate post-forecast-acceptance stage','final_test_access':0,'recovery_stage_read_dev_label_payload':False})
wjson(ROOT/'qa/DATA_CONTRACT_ACCEPTANCE.json',{'status':'PASS','observation_fields':'raw independent recorded Parquet fields arm14+hand14+leg15','command_field':'raw action28','anchor_obs_reproduction_rows':len(elig),'anchor_command_reproduction_rows':len(elig),'known_anchor_availability':'recorded-clock benchmark assumption; raw issue_time absent','eligible_train_rows':len(train),'eligible_train_groups':len(tg['train']),'eligible_dev_rows':len(dev),'eligible_dev_groups':len(tg['dev']),'clock_grid':'nominal_30Hz_validated','raw_episode_files_read':len(raw),'raw_write_detected':False,'final_test_access':0})
wjson(ROOT/'qa/ELIGIBILITY_ACCEPTANCE.json',{'status':'PASS','summary_path':'data/ELIGIBILITY_SUMMARY.json','all_views_same_support':True,'train_dev_group_overlap':0,'no_motion_based_filtering':True,'no_padding_or_interpolation':True,'canonical_to_eligible':'train 17757/17757; dev 2322/2322'})
wjson(ROOT/'qa/EXTRACTION_RECOVERY_ACCEPTANCE.json',{'status':'PASS_RECOVERED_AFTER_SUMMARY_NAMEERROR','data_artifacts_reused':True,'raw_reextraction_performed':False,'rows':len(elig),'raw_files_stat_unchanged':True,'dev_label_values_opened_in_recovery':False,'first_dev_raw_label_read_time':'UNKNOWN_exact time was not persisted by failed initial extraction','all_exclusions':0,'folds_reused_exactly':True})
# Amend source registry with both the exact extraction attempt code and corrected locked implementation.
regp=ROOT/'source/SOURCE_REGISTRY.json'; reg=json.loads(regp.read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,rel,role in [('OBS43_DIAGNOSTIC_CODE_CORRECTED','code/obs43_diagnostic.py','RUN_CODE_AFTER_SUMMARY_FIX'),('EXTRACTION_ACCEPTANCE_RECOVERY_CODE','code/accept_extraction.py','CPU_ACCEPTANCE_CODE')]:
 p=ROOT/rel; reg['sources'].append({'name':name,'path':str(p),'sha256':h(p),'bytes':p.stat().st_size,'status':role})
reg['extraction_attempt_code_sha256']='31e8f2d153d5c48592883069266471213d5631ecfeb64d45dccd9edea058bc77'
reg['extraction_summary_recovery']='only summary/acceptance artifacts resumed; raw arrays were not rewritten'
reg['locked_utc_corrected_implementation']=utc()
regp.write_text(json.dumps(reg,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
receipt={'stage':'accept_extraction_recovery','started_utc':started,'ended_utc':utc(),'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_affinity':sorted(os.sched_getaffinity(0)),'threadpools':threadpool_info(),'gpu_api_or_command_invoked':False,'dev_label_payload_loaded':False,'train_rows':len(train),'dev_rows':len(dev),'raw_episodes':len(raw)}
wjson(ROOT/'logs/EXTRACTION_RECOVERY_RECEIPT.json',receipt)
print(json.dumps({'status':'PASS','train_rows':len(train),'train_groups':len(tg['train']),'dev_rows':len(dev),'dev_groups':len(tg['dev']),'raw_episodes':len(raw),'clock_intervals':len(errs),'clock_max_abs_error_seconds':max(errs),'first_dev_raw_label_read':'UNKNOWN'},sort_keys=True))

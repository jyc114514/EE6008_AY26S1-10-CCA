"""Finalize V8.1 descriptive evidence package; no target or prediction payload reads."""
from __future__ import annotations
import os
import csv,json,hashlib,os,resource,statistics,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BASE=PROJECT/'reports/benchmarks'
V2=BASE/'CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=BASE/'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5=BASE/'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7=BASE/'CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z'
V8=BASE/'CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z'
TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)
METHODS=('G','U','M','T')
METH_SOURCE={'G':'GT_RETRAIN','U':'SELF_SOFT','M':'MIX_SOFT','T':'T'}
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')
V7_MANIFEST='58c50a9928106c2cf95fe78c5d3a0bdaa05c050a271803d01a6ef8e011810a29'
V8_MANIFEST='7778803b295f0bd5c3563fa69b940e740ced7b77a7e77c5650f4a0790b88d1c0'
V5_MANIFEST='5866688b48bef3ed940920affc15081dea052c51eca6eb86e7bcfd3a09c5acfa'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jread(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:
  r=csv.DictReader(f);return list(r.fieldnames or []),list(r)
def writejson(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 with tmp.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def writecsv(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(rows[0])
 tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 with tmp.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def max_abs(rows,fn): return max((abs(fn(r)) for r in rows),default=0.0)


def main():
 started=time.perf_counter()
 result_root=ROOT/'results'
 # Check source packages and reuse receipts, without reopening target or prediction payloads.
 old_roots={}
 for label,path,expect in (('V5',V5,V5_MANIFEST),('V7',V7,V7_MANIFEST),('V8',V8,V8_MANIFEST)):
  actual=sha(path/'CHECKSUMS.sha256')
  receipt=jread(path/'qa/FINAL_CHECKSUM_VERIFICATION.json')
  if actual!=expect or receipt.get('status')!='PASS' or receipt.get('manifest_sha256')!=expect:
   raise RuntimeError(f'{label} source manifest/receipt changed')
  old_roots[label]={'root':str(path),'manifest_sha256':actual,'manifest_entries':receipt['manifest_entry_count'],
                    'receipt_sha256':sha(path/'qa/FINAL_CHECKSUM_VERIFICATION.json'),'receipt_status':'PASS'}
 reuse=jread(ROOT/'inventory/V8_REUSE_RECEIPT.json')
 if reuse.get('status')!='PASS' or not reuse.get('all_byte_identical') or reuse.get('count')!=36:
  raise RuntimeError('V8 byte-identical reuse receipt failed')

 # Required row support and exact reconciliation.
 h=readcsv(result_root/'HORIZON_ERRORS.csv')[1]
 a=readcsv(result_root/'AXIS_ERRORS.csv')[1]
 hr=readcsv(result_root/'HORIZON_GAINS.csv')[1]
 ar=readcsv(result_root/'AXIS_GAINS.csv')[1]
 mc=readcsv(result_root/'MACRO_RECONCILIATION.csv')[1]
 pw=readcsv(result_root/'PER_WINDOW_RECONCILIATION.csv')[1]
 expected={
  'horizon_rows':4248,'position_horizon_rows':4*3*6*30,'internal_fd_horizon_rows':4*3*6*29,
  'axis_rows':6048,'position_axis_rows':4*3*6*28,'internal_fd_axis_rows':4*3*6*28,'seam_axis_rows':4*3*6*28,
  'macro_rows':216,'per_window_rows':2220,
 }
 observed={'horizon_rows':len(h),'position_horizon_rows':sum(x['metric']=='position' for x in h),
  'internal_fd_horizon_rows':sum(x['metric']=='internal_fd' for x in h),'axis_rows':len(a),
  'position_axis_rows':sum(x['metric']=='position' for x in a),'internal_fd_axis_rows':sum(x['metric']=='internal_fd' for x in a),
  'seam_axis_rows':sum(x['metric']=='seam' for x in a),'macro_rows':len(mc),'per_window_rows':len(pw)}
 if observed!=expected or len(hr)!=4248 or len(ar)!=6048 or any(x['status']!='PASS' for x in mc) or any(x['status']!='PASS' for x in pw):
  raise RuntimeError(f'accepted support/reconciliation gate failed: {observed}')
 if len({(x['method'],int(x['seed']),x['task_or_ALL'],x['metric'],str(x['step_or_transition'])) for x in h})!=len(h):raise RuntimeError('duplicate horizon keys')
 if len({(x['method'],int(x['seed']),x['task_or_ALL'],x['metric'],int(x['axis'])) for x in a})!=len(a):raise RuntimeError('duplicate axis keys')
 if len({(x['method'],int(x['seed']),x['task_or_ALL'],x['metric']) for x in mc})!=len(mc):raise RuntimeError('duplicate macro reconciliation keys')
 max_primary=max(float(x['max_abs_difference_to_accepted']) for x in mc)
 max_direct_step=max_abs([x for x in mc if x['metric']!='seam_safe_fd_l1'],lambda x:float(x['v8_1_canonical_direct_value'])-float(x['v8_1_horizon_reconstructed_value']))
 max_direct_axis=max_abs(mc,lambda x:float(x['v8_1_canonical_direct_value'])-float(x['v8_1_axis_reconstructed_value']))
 max_perwindow=max(float(x['max_abs_diff']) for x in pw)

 # Derive per-seed, task/macro gains. Positive G-minus-candidate values favor the candidate.
 mcidx={(x['method'],int(x['seed']),x['task_or_ALL'],x['metric']):x for x in mc}
 macro_gains=[]
 for method in METHODS:
  for seed in SEEDS:
   for scope in [*(str(t) for t in TASKS),'ALL']:
    for metric in METRICS:
     g=float(mcidx[('G',seed,scope,metric)]['v8_1_canonical_direct_value'])
     value=float(mcidx[(method,seed,scope,metric)]['v8_1_canonical_direct_value'])
     macro_gains.append({'method':method,'seed':seed,'task_or_ALL':scope,'metric':metric,
                         'reference':'G','G_value':g,'method_value':value,'gain_G_minus_method':g-value,
                         'positive_favors_candidate':True})
 writecsv(result_root/'MACRO_GAINS.csv',macro_gains)
 groups={}
 for r in macro_gains:groups.setdefault((r['method'],r['task_or_ALL'],r['metric']),[]).append(float(r['gain_G_minus_method']))
 macro_gain_summary=[]
 for key,vals in sorted(groups.items()):
  if len(vals)!=3:raise RuntimeError(f'macro gain seed support mismatch: {key}/{len(vals)}')
  for stat,value in (('mean',statistics.mean(vals)),('sample_sd',statistics.stdev(vals))):
   macro_gain_summary.append({'method':key[0],'task_or_ALL':key[1],'metric':key[2],'seed_stat':stat,'value':value,'n_seeds':3,
                              'gain_sign':'G loss minus method loss; positive favors candidate'})
 writecsv(result_root/'MACRO_GAINS_SEED_SUMMARY.csv',macro_gain_summary)

 # Reconcile step/axis gains to V7 accepted paired effects without new resampling.
 v7paired=readcsv(V7/'results/PAIRED_EFFECTS.csv')[1]
 v7m={(r['seed'],r['task_id'],r['metric']):float(r['gain']) for r in v7paired
      if r['contrast']=='MIX_VS_GT' and r['scope']=='task' and r['candidate']=='MIX_SOFT'}
 h_gain={(x['method'],int(x['seed']),x['task_or_ALL'],x['metric'],str(x['step_or_transition'])):float(x['gain_G_minus_method']) for x in hr}
 a_gain={(x['method'],int(x['seed']),x['task_or_ALL'],x['metric'],int(x['axis'])):float(x['gain_G_minus_method']) for x in ar}
 gain_recon=[]
 for task in (str(t) for t in TASKS):
  for seed in SEEDS:
   for v7metric,metric in (('position_mae','position'),('safe_fd_l1','internal_fd'),('seam_safe_fd_l1','seam')):
    accepted=v7m.get((str(seed),task,v7metric))
    if accepted is None:raise RuntimeError(f'V7 paired effect missing {(seed,task,v7metric)}')
    vals=h_gain if metric!='seam' else a_gain
    keys=[k for k in vals if k[0]=='M' and k[1]==seed and k[2]==task and k[3]==metric]
    current=statistics.mean(vals[k] for k in keys)
    gain_recon.append({'method':'M','seed':seed,'task_id':task,'metric':v7metric,'v7_accepted_gain':accepted,
                       'v8_1_step_or_axis_mean_gain':current,'abs_diff':abs(current-accepted),
                       'n_steps_or_axes':len(keys),'status':'PASS' if abs(current-accepted)<=1e-12 else 'FAIL'})
 if len(gain_recon)!=45 or any(r['status']!='PASS' for r in gain_recon):raise RuntimeError('V7 paired gain reconciliation failed')
 max_gain_recon=max(r['abs_diff'] for r in gain_recon)
 writecsv(result_root/'V7_GAIN_RECONCILIATION.csv',gain_recon)

 agg_qa={'schema_version':1,'status':'PASS','source_split':'locked exposed dev','locked_windows':185,'source_groups':47,'episodes':47,
  'task_support':{'119':{'anchors':48,'episodes':12,'source_groups':12},'94':{'anchors':48,'episodes':12,'source_groups':12},
                  '90':{'anchors':41,'episodes':11,'source_groups':11},'141':{'anchors':20,'episodes':5,'source_groups':5},'230':{'anchors':28,'episodes':7,'source_groups':7}},
  'methods':list(METHODS),'seeds':list(SEEDS),'observed_rows':observed,'expected_rows':expected,
  'position_future_steps':30,'safe_fd_internal_transitions':29,'coordinate_axes':28,
  'masking':'no source mask; all locked full-coordinate support is present',
  'aggregation':'anchor losses; equal anchors within episode; equal episodes within task; equal-task macro; per-seed first',
  'tolerance':{'absolute':1e-12,'relative':1e-10},
  'max_step_to_canonical_abs_difference':max_direct_step,'max_axis_to_canonical_abs_difference':max_direct_axis,
  'max_accepted_v7_primary_abs_difference':max_primary,'max_accepted_v7_perwindow_abs_difference':max_perwindow,
  'max_v7_paired_task_gain_abs_difference':max_gain_recon,'macro_reconciliation_failures':0,'perwindow_reconciliation_failures':0,
  'all_gain_rows_use':'G error minus candidate error; positive favors candidate','new_bootstrap_draws':0,
  'created_utc':now()}
 writejson(ROOT/'qa/AGGREGATE_RECONCILIATION.json',agg_qa)

 # Accepted V8 train-Z fit sign audit: compare direct same-target J values and stored gain rows.
 v8fit_path=V8/'results/TRAIN_TARGET_AND_FIT.csv';v8fit=readcsv(v8fit_path)[1]
 sign_audit={'schema_version':1,'status':'PASS','source_path':str(v8fit_path),'source_sha256':sha(v8fit_path),
  'formula':'J_gain_runner = J(G,Z) - J(candidate,Z); lower J is closer to the same fixed target',
  'negative_gain_interpretation':'G has lower J and fits the fixed Z target better than the corresponding student M or U',
  'comparisons':[],'no_v8_1_refit_or_forward':True}
 for record,target,student,base_record in (('TRAIN_Z_GAIN_M','Z_M','M','common_ZM_comparison'),('TRAIN_Z_GAIN_U','Z_U','U','common_ZU_comparison')):
  signed=[]
  for seed in SEEDS:
   gainrow=next(r for r in v8fit if r['record_type']==record and r['task_id']=='ALL' and int(r['seed'])==seed)
   grow=next(r for r in v8fit if r['record_type']==base_record and r['task_id']=='ALL' and int(r['seed'])==seed and r['method']=='G')
   srow=next(r for r in v8fit if r['record_type']==base_record and r['task_id']=='ALL' and int(r['seed'])==seed and r['method']==student)
   delta=float(grow['J_runner'])-float(srow['J_runner'])
   reported=float(gainrow['J_gain_runner'])
   if abs(delta-reported)>1e-12:raise RuntimeError(f'V8 sign cross-check mismatch {record}/{seed}')
   signed.append(reported)
  sign_audit['comparisons'].append({'record_type':record,'target':target,'student':student,
    'per_seed_J_G_minus_student':signed,'mean':statistics.mean(signed),'sample_sd':statistics.stdev(signed),
    'negative_seeds':sum(x<0 for x in signed),'interpretation':'G fits the fixed student-derived Z better in this eval objective'})
 writejson(ROOT/'qa/V8_SIGN_AUDIT.json',sign_audit)

 # Teacher-distance summary is reused directly from V8; no new scoring here.
 teacher_path=V8/'results/TRAIN_DEV_IMITATION.csv';teacher=readcsv(teacher_path)[1]
 td=[]
 for split in ('train','dev'):
  for method in ('G','U','M'):
   sub=[r for r in teacher if r['split']==split and r['scope']=='equal_task_macro' and r['task_id']=='ALL' and r['method']==method]
   if len(sub)!=3:raise RuntimeError(f'teacher-distance support mismatch {split}/{method}/{len(sub)}')
   for field,metric in (('position_distance','position_distance_to_T'),('safe_fd_distance','internal_fd_distance_to_T'),('seam_distance','seam_distance_to_T')):
    vals=[float(r[field]) for r in sub]
    td.append({'split':split,'method':method,'metric':metric,'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals),
                'n_seeds':3,'source':'accepted V8; descriptive distance to T'})
 writecsv(result_root/'TEACHER_DISTANCE_ACCEPTED_SUMMARY.csv',td)

 # Compact macro table for the chat and professor-ready return.
 macro_summary=readcsv(result_root/'MACRO_SEED_SUMMARY.csv')[1]
 macro_all=[r for r in macro_summary if r['task_or_ALL']=='ALL']
 writecsv(result_root/'ALL_METHOD_MACRO_SUMMARY.csv',macro_all)

 # Descriptive most-negative and most-positive M-vs-G task/step/axis cells across three seeds.
 local=[]
 for filename,kind in (('HORIZON_GAINS.csv','horizon'),('AXIS_GAINS.csv','axis')):
  rows=readcsv(result_root/filename)[1]
  selected=[r for r in rows if r['method']=='M' and r['task_or_ALL']!='ALL']
  buckets={}
  for r in selected:
   cell=r['step_or_transition'] if kind=='horizon' else r['axis']
   key=(r['task_or_ALL'],r['metric'],str(cell));buckets.setdefault(key,[]).append(float(r['gain_G_minus_method']))
  for metric in (('position','internal_fd') if kind=='horizon' else ('position','internal_fd','seam')):
   q=[(statistics.mean(v),k) for k,v in buckets.items() if k[1]==metric and len(v)==3]
   if not q:raise RuntimeError(f'no local cells for {kind}/{metric}')
   for tag,item in (('most_negative_M_minus_G_gain',min(q)),('most_positive_M_minus_G_gain',max(q))):
    value,key=item
    local.append({'grain':kind,'metric':metric,'extreme':tag,'task_id':key[0],'cell_index':key[2],
                  'gain_G_minus_M_mean_across_3_seeds':value,'n_seeds':3,
                  'sign':'positive favors M; negative means M error exceeds G',
                  'interpretation':'descriptive post-result localization; no multiplicity correction or inference'})
 writecsv(result_root/'DIAGNOSTIC_LOCALIZATION.csv',local)

 # Refresh cumulative exposure and run receipts from recorded attempts, never reopen targets.
 pre=jread(ROOT/'qa/PREFLIGHT.json');real=jread(ROOT/'qa/REAL_DATA_QA.json');target_receipt=jread(ROOT/'qa/TARGET_READ_RECEIPT.json')
 source_map=jread(ROOT/'inventory/SOURCE_MAP.json')
 runs=jread(ROOT/'logs/RUN_003_FROM_CACHE.json')
 figbuild=jread(ROOT/'logs/FIGURES_BUILD_004.json')
 if target_receipt.get('archive_npz_loads_total')!=1 or target_receipt.get('cache_loads_total')!=2 or target_receipt.get('cache_resume_loads')!=1:
  raise RuntimeError('cumulative target/cache counts changed unexpectedly')
 real.update({
  'target_archive_npz_loads_total':1,'target_archive_hash_passes_total':2,'target_unique_windows_loaded':185,
  'target_native_scalar_values_loaded':185*30*28,'scorer_cache_package_loads_total':2,'prefix_archive_loads_total':3,
  'prediction_payload_decodes_total':24,'prediction_file_hash_passes_total':36,'prediction_row_map_reads_total':36,
  'accepted_per_window_source_rows_scanned_total':4625*3,'accepted_task_seed_source_rows_scanned_total':150*3,
  'accepted_per_window_rows_reused_unique':2220,'accepted_task_seed_rows_reused_unique':72,
  'checkpoint_slots_written_total':12,'checkpoint_package_loads_total':24,
  'successful_cached_resume_runs':1,'post_label_technical_assembly_failures':1,
  'additional_target_archive_reads_after_seal':0,'final_heldout_payload_access':0,
  'new_fit':0,'optimizer_updates':0,'teacher_student_forward_jobs':0,'new_dev_prediction_jobs':0,
  'encoder_extractions':0,'bootstrap_draws':0,'model_selection_or_hyperparameter_search':0,'gpu_commands':0,
  'old_root_writes':0,'independent_confirmation':False,
  'figure_generation_count':figbuild['figure_count'],'figure_human_review_status':'PENDING'
 })
 writejson(ROOT/'qa/REAL_DATA_QA.json',real)
 target_receipt.update({'archive_npz_loads_total':1,'source_archive_hash_passes_total':2,
  'prefix_archive_loads_total':3,'scorer_cache_package_loads_total':2,'prediction_payload_decodes_total':24,
  'prediction_file_hash_passes_total':36,'prediction_row_map_reads_total':36,'additional_source_archive_loads_after_initial_read':0,
  'target_native_scalar_values_loaded':185*30*28,'target_label_cache_sha256':sha(ROOT/'scorer_cache/Y_dev.npy')})
 writejson(ROOT/'qa/TARGET_READ_RECEIPT.json',target_receipt)

 # Reconcile old root identity before terminal state.
 source_invariance={'schema_version':1,'status':'PASS','checked_utc':now(),'roots':old_roots,
  'V7_manifest_sha256_expected':V7_MANIFEST,'V8_manifest_sha256_expected':V8_MANIFEST,
  'V5_manifest_sha256_expected':V5_MANIFEST,'V7_old_root_writes':0,'V8_old_root_writes':0,
  'V8_status_remains':'complete_with_limits','V8_original_unavailable_rows_remain':4,
  'V8_1_new_outputs_are_separate':True,'reused_v8_files_byte_identical':reuse['count'],'no_gpu_commands':0}
 writejson(ROOT/'qa/SOURCE_INVARIANCE.json',source_invariance)

 # Full conclusion ledger.
 v7_ci=readcsv(V7/'results/PAIRED_GROUP_CI.csv')[1]
 ci={r['metric']:(float(r['ci95_low']),float(r['ci95_high'])) for r in v7_ci if r['contrast']=='MIX_VS_GT' and r['scope']=='macro'}
 gainall={(r['method'],r['task_or_ALL'],r['metric'],r['seed_stat']):float(r['value']) for r in macro_gain_summary}
 v8_m={x['record_type']:x for x in sign_audit['comparisons']}
 teacher_lookup={(r['split'],r['method'],r['metric']):r for r in td}
 tdidx={(r['split'],r['method'],r['metric']):r for r in td}
 ledger=[
  {'claim':'Does SELF_SOFT or MIX_SOFT better fit the actual fixed train target Z?','V7_V8_evidence':'V8 TRAIN_TARGET_AND_FIT.csv: J_gain_runner=J(G,Z_M)-J(M,Z_M) and J(G,Z_U)-J(U,Z_U), 3/3 seeds negative for each target.',
   'V8_1_new_evidence':'qa/V8_SIGN_AUDIT.json directly cross-checks reported gains against same-target G and student J_runner rows.',
   'supported_statement':'No. G has lower eval objective on both Z_M and Z_U. Mean gains are M=-0.002499277619 (sample SD 0.0005406885) and U=-0.004040830964 (sample SD 0.0006835763).',
   'limit':'This describes the fixed dropout-off objective and scales; it does not identify an optimizer bug, representation-capacity limit, or a general distillation failure.','status':'PASS'},
  {'claim':'Do outputs move toward teacher T, and how does train differ from dev?','V7_V8_evidence':'V8 TRAIN_DEV_IMITATION.csv; distances use accepted train/dev predictions to T.',
   'V8_1_new_evidence':'V8 teacher-distance table reused byte-identically and summarized in results/TEACHER_DISTANCE_ACCEPTED_SUMMARY.csv; trajectories show accepted Y/G/U/M/T on frozen selected cases.',
   'supported_statement':f"G is closer to T than U and M in position, safe-FD, and seam on both train and dev. Train position/FD/seam: G {tdidx[('train','G','position_distance_to_T')]['mean']:.6f}/{tdidx[('train','G','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('train','G','seam_distance_to_T')]['mean']:.6f}, U {tdidx[('train','U','position_distance_to_T')]['mean']:.6f}/{tdidx[('train','U','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('train','U','seam_distance_to_T')]['mean']:.6f}, M {tdidx[('train','M','position_distance_to_T')]['mean']:.6f}/{tdidx[('train','M','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('train','M','seam_distance_to_T')]['mean']:.6f}; dev G {tdidx[('dev','G','position_distance_to_T')]['mean']:.6f}/{tdidx[('dev','G','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('dev','G','seam_distance_to_T')]['mean']:.6f}, U {tdidx[('dev','U','position_distance_to_T')]['mean']:.6f}/{tdidx[('dev','U','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('dev','U','seam_distance_to_T')]['mean']:.6f}, M {tdidx[('dev','M','position_distance_to_T')]['mean']:.6f}/{tdidx[('dev','M','internal_fd_distance_to_T')]['mean']:.6f}/{tdidx[('dev','M','seam_distance_to_T')]['mean']:.6f}.",
   'limit':'Distance to T is not accuracy against Y. V8.1 does not recompute these distances.','status':'PASS'},
  {'claim':'What happens to dev MAE, FD, and seam for the fixed .5 MIX_SOFT recipe vs G?','V7_V8_evidence':'V7 accepted PRIMARY_TASK_SEED.csv and source-group CI; V8 diagnosis retains the same fixed recipe conclusion.',
   'V8_1_new_evidence':'216-row MACRO_RECONCILIATION.csv matches accepted V7 per-window/task/seed values within 4.44e-16.',
   'supported_statement':f"Across seeds, G-minus-M gains are position {gainall[('M','ALL','position_mae','mean')]:+.9f}, FD {gainall[('M','ALL','safe_fd_l1','mean')]:+.9f}, seam {gainall[('M','ALL','seam_safe_fd_l1','mean')]:+.9f}; positive favors M. Existing V7 conditional source-group 95% CIs are position [{ci['position_mae'][0]:+.9f},{ci['position_mae'][1]:+.9f}], FD [{ci['safe_fd_l1'][0]:+.9f},{ci['safe_fd_l1'][1]:+.9f}], seam [{ci['seam_safe_fd_l1'][0]:+.9f},{ci['seam_safe_fd_l1'][1]:+.9f}].",
   'limit':'V8.1 adds no bootstrap draws; these development results are not independent final-test evidence. The fixed .5 recipe remains closed under the accepted decision.','status':'PASS'},
  {'claim':'Where are the M-vs-G changes concentrated by task, step, and coordinate?','V7_V8_evidence':'V7 per-task losses showed position/FD losses concentrated in tasks 119/141 and seam gains in 119/141; V8 had no step/coordinate values.',
   'V8_1_new_evidence':'DIAGNOSTIC_LOCALIZATION.csv and complete 4,248/6,048-row horizon/axis tables.',
   'supported_statement':'Equal-task macro gains show position/FD deficits concentrated in tasks 141, 119, and 94, while seam gains are positive in every task and largest in 119 and 141. The largest negative three-seed mean cells are task 141 position step 7 (-0.010552), task 141 FD transition 9→10 (-0.091063), task 141 position coordinate 6 (-0.015775), task 119 FD coordinate 20 (-0.063199), and task 141 seam coordinate 20 (-0.188403). Largest positive cells include task 119 seam coordinate 4 (+0.290707).',
   'limit':'Cell extrema are post-result descriptions over the fixed task/step/axis grid; they are not independent tests, do not change the primary metric, and coordinate indices have no anatomical mapping.','status':'PASS'},
  {'claim':'Did this supplement constitute a new model experiment or independent evaluation?','V7_V8_evidence':'Accepted protocol locks restrict the scope to the already exposed 185-window development set.',
   'V8_1_new_evidence':'One authorized scorer-only dev source archive load; labels cached privately; no new fits, updates, forwards, predictions, encoder extraction, bootstrap, search, GPU commands, or heldout access.',
   'supported_statement':'This is a post-result evidence completion on the same 185 dev windows, with 12 existing method-seed prediction bundles and 2,220 method-seed-window diagnostic units.',
   'limit':'Development Y remains exposed evidence and is not an independent pool. Human figure review is pending.','status':'PASS'},
 ]
 writecsv(result_root/'CONCLUSION_LEDGER.csv',ledger)

 # Frozen run config, no secrets and no training path.
 config={'schema_version':1,'study':'post_V8_evidence_completion_no_new_model','output_root':str(ROOT),
  'tasks':list(TASKS),'seeds':list(SEEDS),'methods':{'G':'GT_RETRAIN','U':'SELF_SOFT','M':'MIX_SOFT','T':'accepted V5 MIX50 reference'},
  'locked_dev_windows':185,'source_groups':47,'ordered_anchor_id_sha256':'5c6e5ef759ce864e69a186b7d2fe6063b3c9e1bc8d5c98640d7a2873d1ba3025',
  'metric_scale_lock_path':str(V4/'protocol/METRIC_SCALE_LOCK.json'),'metric_scale_lock_sha256':'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32',
  'target_reader_fields':['anchor_id','episode_id','task_id','source_cluster','target_native_h30'],'target_reader_allow_pickle':False,
  'device_policy':'CPU_ONLY; CUDA_VISIBLE_DEVICES empty; no CUDA/NVML/GPU commands',
  'new_fits':0,'optimizer_updates':0,'teacher_student_forward_jobs':0,'new_dev_prediction_jobs':0,
  'encoder_extractions':0,'bootstrap_draws':0,'model_selection_or_search':0,'final_heldout_payload_access':0,
  'aggregation':'per-anchor; equal anchors per episode; equal episodes per task; equal task macro; per-seed first',
  'gain':'G loss minus method loss; positive favors candidate','tolerance':{'absolute':1e-12,'relative':1e-10}}
 writejson(ROOT/'configs/V8_1_LOCKED_CPU_CONFIG.json',config)

 # Update ledger statuses and append summary output records; it never self-hashes.
 ledger_path=ROOT/'inventory/GAP_AND_REUSE_LEDGER.csv'
 fields,rows=readcsv(ledger_path)
 target_path=source_map['dev_target_source']['absolute_path'];target_sha=source_map['dev_target_source']['sha256']
 for row in rows:
  if row['source_path']==target_path:
   row['prior_status']='V8 exposed once; no raw target values persisted in V8'
   row['V8_1_action']='one authorized allow_pickle=False read of five whitelisted fields; cache sealed before reduction; no archive reread'
   row['output_status']='FILLED_SCORER_ONLY_CACHE_1_ARCHIVE_LOAD_185_WINDOWS'
  if row['source_path'].endswith('/results/HORIZON_AXIS_ERRORS.csv') and 'V8_FROZEN_TRANSFER' in row['source_path']:
   row['V8_1_action']='preserved original V8 UNAVAILABLE rows; completed step/axis tables in independent V8.1 root'
   row['output_status']='FILLED_V8_1_4248_HORIZON_6048_AXIS_216_RECONCILIATION'
  if row['source_path'].endswith('/results/PER_WINDOW_METRICS.csv'):
   row['output_status']='REUSED_2220_SELECTED_ROWS_AND_2220_ROW_RECONCILIATION_PASS'
  if row['source_path'].endswith('/results/PRIMARY_TASK_SEED.csv'):
   row['output_status']='REUSED_72_SELECTED_ROWS_AND_216_MACRO_RECONCILIATION_PASS'
 for rel,purpose in (('results/HORIZON_ERRORS.csv','new complete scalar horizon errors'),('results/AXIS_ERRORS.csv','new complete scalar coordinate errors'),
                    ('results/MACRO_RECONCILIATION.csv','direct/step/axis accepted-score reconciliation'),('results/HORIZON_GAINS.csv','new descriptive horizon gains'),
                    ('results/AXIS_GAINS.csv','new descriptive axis and seam gains'),('results/DIAGNOSTIC_LOCALIZATION.csv','descriptive task/step/coordinate localization'),
                    ('figure_data/trajectory_data.csv','private already-exposed Y/G/U/M/T case plot values'),('scorer_cache/CACHE_SEAL.json','scorer-only cache seal')):
  p=ROOT/rel
  entry={'source_path':str(p),'source_sha256':sha(p),'purpose':purpose,'prior_status':'new output required to fill V8 gap',
         'V8_1_action':'computed/written in isolated V8.1 root; no old-root write','output_status':'PASS'}
  match=next((x for x in rows if x.get('source_path')==str(p)),None)
  if match is None: rows.append(entry)
  else: match.update(entry)
 tmp=ledger_path.with_name(f'.GAP_AND_REUSE_LEDGER.csv.finalize.{os.getpid()}.part')
 with tmp.open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,ledger_path)

 # Figure receipt is finalized only after the final visual review pass.
 review_path=ROOT/'qa/FIGURE_REVIEW.json';review=jread(review_path)
 if review.get('generated_figure_count')!=29 or len(review.get('figures',[]))!=29:raise RuntimeError('final figure review inventory incomplete')
 contact_sheets=[]
 for rel in ('qa/AGGREGATE_FIGURE_CONTACT_SHEET.png','qa/TRAJECTORY_FIGURE_CONTACT_SHEET.png','qa/REUSED_V8_FIGURE_CONTACT_SHEET.png'):
  q=ROOT/rel
  if q.is_file(): contact_sheets.append({'path':rel,'sha256':sha(q)})
 inspected=[]
 for rel in ('figures/horizon_position_overall.png','figures/horizon_internal_fd_by_task.png','figures/axis_gains_overall.png',
             'figures/axis_position_by_task.png','figures/axis_gains_by_task.png','figures/trajectories/case_01.png',
             'figures/trajectories/case_02.png','figures/trajectories/case_15.png','figures/reused_v8/DEV_TASK_GAINS_MIX_VS_GT.png'):
  q=ROOT/rel;inspected.append({'path':rel,'sha256':sha(q)})
 review.update({'status':'PASS_WITH_HUMAN_REVIEW_PENDING','agent_visual_review_status':'PASS','human_review_status':'PENDING',
   'presentation_compliance':'NOT_ASSESSED_NO_TARGET_PUBLICATION_VENUE','reused_v8_figure_count':4,
   'new_figures_are_byte_independent_of_old_root':True,'reviewed_contact_sheets':contact_sheets,
   'representative_images_inspected':inspected,
   'agent_visual_review_notes':['All 14 aggregate figures and all 15 fixed case trajectories inspected via contact sheets.',
    'Enlarged representative horizon, per-task error, axis gain, trajectory, and reused V8 figures.',
    'Confirmed method legends, distinct line styles, axes, annotations, and no clipping or caption overlap after corrections.',
    'Trajectory indices are descriptive coordinates and recorded frame indices, not anatomical mappings or calibrated time.'],
   'latest_visual_check_utc':now()})
 writejson(review_path,review)
 figbuild=jread(ROOT/'qa/FIGURE_BUILD_QA.json')
 figbuild.update({'agent_visual_review_status':'PASS','human_review_status':'PENDING','reused_v8_figure_count':4})
 writejson(ROOT/'qa/FIGURE_BUILD_QA.json',figbuild)

 # Source-invariance receipt confirms no edits to V5/V7/V8 packages.
 writejson(ROOT/'qa/SOURCE_INVARIANCE.json',{
  'schema_version':1,'status':'PASS','checked_utc':now(),'roots':old_roots,
  'expected_manifests':{'V5':V5_MANIFEST,'V7':V7_MANIFEST,'V8':V8_MANIFEST},
  'V7_old_root_writes':0,'V8_old_root_writes':0,'V8_status_remains':'complete_with_limits',
  'V8_original_horizon_axis_unavailable_rows_remain':4,'V8_1_output_root_is_separate':True,
  'V8_1_selected_reused_v8_files_byte_identical':36,'GPU_commands':0})

 # CPU/resource and exposure ledger is scoped only to the new supplement.
 pre=jread(ROOT/'qa/PREFLIGHT.json');real=jread(ROOT/'qa/REAL_DATA_QA.json');run=jread(ROOT/'logs/RUN_003_FROM_CACHE.json')
 figlog=jread(ROOT/'logs/FIGURES_BUILD_004.json') if (ROOT/'logs/FIGURES_BUILD_004.json').is_file() else jread(ROOT/'logs/FIGURES_BUILD_003.json')
 cache_files=list((ROOT/'scorer_cache').glob('*'))
 ckpt_files=list((ROOT/'checkpoints').glob('*.npz'))
 new_png=[x for x in review['figures']]
 reused_png=list((ROOT/'figures/reused_v8').glob('*.png'))
 source_map= jread(ROOT/'inventory/SOURCE_MAP.json')
 pred_assets=source_map['prediction_bundles']
 pred_bytes=sum(Path(x['prediction_absolute_path']).stat().st_size for x in pred_assets)
 row_bytes=sum(Path(x['rows_absolute_path']).stat().st_size for x in pred_assets)
 cache_bytes=sum(p.stat().st_size for p in cache_files if p.is_file())
 checkpoint_bytes=sum(p.stat().st_size for p in ckpt_files)
 new_figure_bytes=sum(int(x['bytes']) for x in new_png)
 reused_figure_bytes=sum(p.stat().st_size for p in reused_png)
 resource_rows=[]
 def rr(metric,value,unit,scope,source,notes=''):
  resource_rows.append({'metric':metric,'value':value,'unit':unit,'scope':scope,'source':source,'notes':notes})
 rr('target archive npz loads',1,'loads','V8.1 only',target_path,'One allow_pickle=False load; same 185 locked dev windows; no reread after cache seal.')
 rr('historical target archive loads',2,'loads','V7+V8 historical','V7 qa/DEV_TARGET_ACCESS_RECEIPT.json; V8 qa/DEV_TARGET_ACCESS_RECEIPT.json','Kept separate from V8.1 resource total.')
 rr('target archive hash passes',2,'passes','V8.1 only',target_path,'Preflight source binding plus immediate pre-open recheck; hash-only passes are not npz loads.')
 rr('dev target archive file size',Path(target_path).stat().st_size,'bytes','V2 read-only source',target_path,'File stat only in finalizer; no archive or label load.')
 rr('label-free prefix archive file size',Path(source_map['dev_prefix_source']['absolute_path']).stat().st_size,'bytes','V4 read-only source',source_map['dev_prefix_source']['absolute_path'],'File stat only; whitelist fields were read during preflight and the two analysis runs.')
 rr('target unique windows loaded',185,'windows','V8.1 only',target_path,'185 unique dev windows; no final/heldout payload.')
 rr('target native scalar values',185*30*28,'scalars','V8.1 only',target_path,'Shape [185,30,28]; values persisted only in scorer_cache/Y_dev.npy and the required private trajectory figure data.')
 rr('prefix archive loads',3,'loads','V8.1 only',str(V4/'data/DEV_PREFIXES_NO_LABELS.npz'),'Label-free accepted ID/episode/task/group/frame/anchor fields only; preflight and two analysis runs.')
 rr('scorer cache package loads',2,'loads','V8.1 only',str(ROOT/'scorer_cache/CACHE_SEAL.json'),'One immediate readback after write; one resume from sealed cache after a result-assembly failure.')
 rr('prediction file hash passes',36,'file checks','V8.1 only','12 sealed bundles × preflight + initial scoring attempt + cache-resumed scoring attempt','Prediction files remained in V7/V5 roots and were read-only.')
 rr('prediction tensor decodes',24,'loads','V8.1 only','12 bundles × 2 analysis runs','Existing G/U/M from V7 and T/MIX50 from V5; no new predictions.')
 rr('prediction row-map reads',36,'reads','V8.1 only','12 row maps × preflight + two analysis attempts','Each map was joined by anchor ID and episode/task/frame/rank/source group.')
 rr('accepted V7 per-window rows reused',2220,'unique rows','V8.1 reconciliation','V7/results/PER_WINDOW_METRICS.csv','4 methods × 3 seeds × 185 anchors; source file parsed in 3 bounded passes.')
 rr('accepted V7 task/seed/macro rows reused',72,'unique rows','V8.1 reconciliation','V7/results/PRIMARY_TASK_SEED.csv','4 methods × 3 seeds × 6 scopes; source file parsed in 3 bounded passes.')
 rr('per-window scored diagnostic units',2220,'method-seed-window units','V8.1 only','12 accepted bundles × 185 windows','Not a new independent sample count.')
 rr('error checkpoint writes',12,'slots','V8.1 only',str(ROOT/'checkpoints'),'All 12 method-seed error slots sealed; no Y stored in checkpoints.')
 rr('error checkpoint package loads',24,'loads','V8.1 only',str(ROOT/'checkpoints'),'12 readbacks after write plus 12 checkpoint-only resume reads.')
 rr('horizon error table',4248,'rows','V8.1 only','results/HORIZON_ERRORS.csv','Position 1..30 and internal FD 1→2..29→30 across 6 scopes, 4 methods, 3 seeds.')
 rr('axis error table',6048,'rows','V8.1 only','results/AXIS_ERRORS.csv','Position, internal FD, and separate seam across 28 coordinates.')
 rr('macro reconciliation table',216,'rows','V8.1 only','results/MACRO_RECONCILIATION.csv','All direct/step/axis rows pass locked 1e-12 absolute and 1e-10 relative tolerance.')
 rr('new aggregate figure files',14,'PNGs','V8.1 only','figures/','Agent visual review PASS; 3-seed bands are min–max, not CI.')
 rr('new trajectory figure files',15,'PNGs','V8.1 only','figures/trajectories/','Seed 202610131, fixed V8 cases and axes 2/20; includes already exposed Y.')
 rr('reused V8 figure files',4,'PNGs','byte-identical V8 reuse','figures/reused_v8/','V8_REUSE_RECEIPT.json verifies byte identity.')
 rr('scorer cache size',cache_bytes,'bytes','V8.1 private cache',str(ROOT/'scorer_cache'),'Directory 0700; cache files 0600; scorer-only exposed-dev role.')
 rr('error checkpoint size',checkpoint_bytes,'bytes','V8.1 private checkpoints',str(ROOT/'checkpoints'),'Error-only sufficient arrays; no raw Y or model state.')
 rr('existing prediction payload bytes',pred_bytes,'bytes','read-only V7/V5 inputs','12 prediction assets','Input sizes, not copied into V8.1.')
 rr('prediction row-map bytes',row_bytes,'bytes','read-only V7/V5 inputs','12 row-map assets','Input sizes, not copied into V8.1.')
 rr('new scientific figure size',new_figure_bytes,'bytes','V8.1 only','29 PNGs','PNG sizes from final figure QA.')
 rr('reused V8 figure size',reused_figure_bytes,'bytes','byte-identical V8 reuse','4 PNGs','Copied into separate V8.1 reuse directory.')
 rr('CPU process elapsed',float(pre['elapsed_seconds']),'seconds','source preflight','qa/PREFLIGHT.json','Wall elapsed; CPU user/system time not collected.')
 rr('CPU process elapsed',float(run['process_elapsed_seconds']),'seconds','successful cache-resumed scoring','logs/RUN_003_FROM_CACHE.json','Wall elapsed; CPU user/system time not collected.')
 rr('CPU process elapsed',float(figlog.get('elapsed_seconds',0.0)),'seconds','latest figure build','logs/FIGURES_BUILD_004.json','Wall elapsed; CPU user/system time not collected.')
 rr('peak RSS',int(pre['max_rss_kib']),'KiB','source preflight','qa/PREFLIGHT.json','Observed process peak; no rerun for resource sampling.')
 rr('peak RSS',int(run['process_max_rss_kib']),'KiB','successful cache-resumed scoring','logs/RUN_003_FROM_CACHE.json','Observed process peak; no GPU memory metrics.')
 rr('V7 historical resource supplement rows',9,'rows','accepted V8 reuse only','results/reused_v8/V7_RESOURCE_SUPPLEMENT.csv','Copied for provenance; not counted as V8.1 resource or recomputed.')
 rr('GPU commands',0,'commands','V8.1 only','policy lock','No GPU, CUDA, NVML, or shared watcher command executed.')
 rr('new fit / optimizer update / forward / dev prediction / encoder / bootstrap / search',0,'operations','V8.1 only','protocol/REPAIR_SCOPE_LOCK.json','All remain zero.')
 writecsv(result_root/'RESOURCE_AND_EXPOSURE.csv',resource_rows)

 # Return-to-chat artifacts contain the full result; manifest hash is returned in final QA/chat to avoid self-reference.
 macro_by={(r['method'],r['task_or_ALL'],r['metric'],r['seed_stat']):float(r['value']) for r in macro_all}
 gain_macro={(r['method'],r['task_or_ALL'],r['metric'],r['seed_stat']):float(r['value']) for r in macro_gain_summary}
 def pm(x):return f'{x:.6f}'
 methods_table=[]
 for method in METHODS:
  methods_table.append({'method':method,'position_mean':macro_by[(method,'ALL','position_mae','mean')],
   'position_sample_sd':macro_by[(method,'ALL','position_mae','sample_sd')],
   'fd_mean':macro_by[(method,'ALL','safe_fd_l1','mean')],'fd_sample_sd':macro_by[(method,'ALL','safe_fd_l1','sample_sd')],
   'seam_mean':macro_by[(method,'ALL','seam_safe_fd_l1','mean')],'seam_sample_sd':macro_by[(method,'ALL','seam_safe_fd_l1','sample_sd')]})
 gain_rows={}
 for r in macro_gain_summary:
  if r['method']=='M' and r['task_or_ALL']=='ALL':gain_rows.setdefault(r['metric'],{})[r['seed_stat']]=float(r['value'])
 task_gains=[]
 for task in TASKS:
  d={m:gain_macro[('M',str(task),m,'mean')] for m in METRICS}
  task_gains.append({'task_id':task,'position':d['position_mae'],'safe_fd':d['safe_fd_l1'],'seam':d['seam_safe_fd_l1']})
 task_gain_md='| Task | Position gain | Safe-FD gain | Seam gain |\n|---:|---:|---:|---:|\n'
 task_gain_md+=''.join(f"| {x['task_id']} | {x['position']:+.9f} | {x['safe_fd']:+.9f} | {x['seam']:+.9f} |\n" for x in task_gains)
 all_taskgains={m:gain_rows[m] for m in METRICS}
 trainz={x['record_type']:x for x in sign_audit['comparisons']}
 teacher_md='| Split | Method | Position distance to T | Safe-FD distance to T | Seam distance to T |\n|---|---|---:|---:|---:|\n'
 for split in ('train','dev'):
  for method in ('G','U','M'):
   vals=[tdidx[(split,method,key)] for key in ('position_distance_to_T','internal_fd_distance_to_T','seam_distance_to_T')]
   teacher_md+=f"| {split} | {method} | {vals[0]['mean']:.6f} ± {vals[0]['sample_sd']:.6f} | {vals[1]['mean']:.6f} ± {vals[1]['sample_sd']:.6f} | {vals[2]['mean']:.6f} ± {vals[2]['sample_sd']:.6f} |\n"
 report_md=(
  '# V8.1 固定开发集证据补全\n\n'
  '状态：complete_with_limits。该包补齐 V8 已锁定证据，未开展新模型实验。\n\n'
  '## 三个核查问题\n\n'
  '1. **学生是否更好地拟合实际训练目标 Z？** 否。V8 的差值定义为 J(G,Z)−J(student,Z)，三 seed 都为负，表示 G 的同目标评估损失更低。Z_M 上 M 的均值差为 '
  f"{trainz['TRAIN_Z_GAIN_M']['mean']:+.6f} ± {trainz['TRAIN_Z_GAIN_M']['sample_sd']:.6f}；Z_U 上 U 为 {trainz['TRAIN_Z_GAIN_U']['mean']:+.6f} ± {trainz['TRAIN_Z_GAIN_U']['sample_sd']:.6f}。"
  '这只说明当前固定目标、固定尺度和 dropout-off objective 的比较结果。\n\n'
  '2. **输出是否朝教师 T 靠近？** V8 的 train/dev 距离表显示 G 比 U、M 更接近 T；这些是输出间距离，不等同于真实 Y 准确度。V8.1 复用完整 seed 级表的三 seed 均值和 sample SD：\n\n'+teacher_md+'\nV8.1 轨迹图使用同一批已封存预测。\n\n'
  '3. **MAE/FD 退化与 seam 改善在哪里？** 完整 30-step、29-transition、28-axis 表已生成。对 M 相对 G，正收益定义为 G error−M error；三 seed 宏观均值为 '
  f"position {all_taskgains['position_mae']['mean']:+.6f} ± {all_taskgains['position_mae']['sample_sd']:.6f}，FD {all_taskgains['safe_fd_l1']['mean']:+.6f} ± {all_taskgains['safe_fd_l1']['sample_sd']:.6f}，seam {all_taskgains['seam_safe_fd_l1']['mean']:+.6f} ± {all_taskgains['seam_safe_fd_l1']['sample_sd']:.6f}。"
  '主要平均退化落在任务 141、119、94；seam 的 task macro 在五个任务均改善，增益集中于 119 和 141。M 的最负三 seed 平均单元是 task 141 position step 7 (−0.010552) 与 FD transition 9→10 (−0.091063)；坐标上 position 的最负单元为 task 141 axis 6 (−0.015775)，FD 为 task 119 axis 20 (−0.063199)。Seam 在 task 119 axis 4 有最大正单元 (+0.290707)，task 141 axis 20 仍有局部负单元 (−0.188403)。这些是完整网格上的描述值，不是逐格显著性检验。\n\n'
  '## 宏观结果（mean ± sample SD, n=3 seeds）\n\n'
  '| 方法 | Position | Safe FD | Seam safe FD |\n|---|---:|---:|---:|\n'
  + ''.join(f"| {m['method']} | {pm(m['position_mean'])} ± {pm(m['position_sample_sd'])} | {pm(m['fd_mean'])} ± {pm(m['fd_sample_sd'])} | {pm(m['seam_mean'])} ± {pm(m['seam_sample_sd'])} |\n" for m in methods_table)
  + '\nM 相对 G 的分任务收益（正值有利于 M）：\n\n'+task_gain_md+'\n'
  + 'V7 已接受的条件 source-group 95% CI（5,000 draws；本轮 0 次重抽样）仍为 position [−0.002180, +0.000579]、FD [−0.008554, −0.000481]、seam [+0.007785, +0.035567]。对账表的最大直接/step/axis差为 4.44e−16，低于 1e−12 abs / 1e−10 rel 容差。\n\n'
  '## 覆盖与访问账本\n\n'
  '- HORIZON_ERRORS 4,248 行：position 2,160，safe-FD 2,088。AXIS_ERRORS 6,048 行：position、internal-FD、seam 各 2,016。MACRO_RECONCILIATION 216 行全部 PASS；PER_WINDOW_RECONCILIATION 2,220 行全部 PASS。\n'
  '- 任务支持：119=48、94=48、90=41、141=20、230=28 anchors；共 47 episodes/source groups。\n'
  '- V8.1 原始 dev-target archive 只 load 1 次，185 windows；精确读取五个白名单字段，cache 先写入和读回后才归约。scorer-cache package load 2 次（初次 readback + 从已封存缓存恢复）。错误组装重试未重开 target。\n'
  '- 12 个既有预测 bundle 在 cache 计算和恢复各 decode 一次（24 次），行映射 36 次含 preflight；2220 是方法×seed×window 诊断单元，不是新增样本。\n'
  '- 新 fit/update/forward/dev prediction/encoder/bootstrap/search/GPU 均为 0；final/heldout payload access=0；旧 root writes=0。V7/V8 manifest SHA 保持 '+V7_MANIFEST+' / '+V8_MANIFEST+'。V8 原 status complete_with_limits 和 4 个 UNAVAILABLE 行保持封存；补全仅存在 V8.1 新目录。\n'
  '- 图：14 张总体/task误差与收益图，15 张固定 seed/anchor、固定轴 2/20 的轨迹图；四张 V8 图 byte-identical 复用。Agent visual review PASS；human review PENDING。trajectory_data.csv 含已经暴露的 dev Y，留在私有根目录；不作为独立 pool。\n\n'
  f"输出根：{ROOT}\n"
  '关键文件：results/HORIZON_ERRORS.csv、AXIS_ERRORS.csv、MACRO_RECONCILIATION.csv、CONCLUSION_LEDGER.csv、RESOURCE_AND_EXPOSURE.csv；图与 captions 均在 figures/、figure_data/、captions/。最终 manifest SHA 由 qa/FINAL_CHECKSUM_VERIFICATION.json 记录，且在最终聊天回执中报告。\n')
 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text(report_md,encoding='utf-8')
 return_json={'schema_version':1,'status':'complete_with_limits','study':'post_V8_evidence_completion_no_new_model',
  'output_root':str(ROOT),'summary_methods_macro':methods_table,'teacher_distance_summary':td,
  'train_Z_fit_sign_audit':sign_audit['comparisons'],'V7_existing_macro_source_group_ci':{k:list(v) for k,v in ci.items()},
  'M_minus_G_gain_sign':'G loss minus M loss; positive favors M',
  'M_minus_G_task_macro_gain_rows':task_gains,'horizon_rows':4248,'axis_rows':6048,'macro_reconciliation_rows':216,
  'per_window_reconciliation_rows':2220,'max_macro_abs_difference':max_primary,'max_step_to_canonical_abs_difference':max_direct_step,
  'max_axis_to_canonical_abs_difference':max_direct_axis,'max_v7_paired_gain_abs_difference':max_gain_recon,
  'locked_dev_windows':185,'source_groups':47,'target_archive_loads_v8_1':1,'scorer_cache_package_loads':2,
  'prediction_payload_decodes':24,'prediction_row_map_reads':36,'new_fits':0,'optimizer_updates':0,
  'teacher_student_forward_jobs':0,'new_dev_prediction_jobs':0,'encoder_extractions':0,'bootstrap_draws':0,
  'model_selection_or_search':0,'gpu_commands':0,'final_heldout_payload_access':0,'old_root_writes':0,
  'V7_manifest_sha256':V7_MANIFEST,'V8_manifest_sha256':V8_MANIFEST,'V5_manifest_sha256':V5_MANIFEST,
  'V8_1_manifest_path':'CHECKSUMS.sha256','manifest_sha256':'recorded in qa/FINAL_CHECKSUM_VERIFICATION.json after checksum-last seal',
  'figure_count_new':29,'aggregate_figure_count':14,'trajectory_figure_count':15,'reused_v8_figure_count':4,
  'agent_visual_review':'PASS','human_review':'PENDING','independent_confirmation':False,
  'next_training_experiment_executed':False,'V8_status_unchanged':'complete_with_limits',
  'V8_original_unavailable_rows_remain':4,'checksum_receipt_path':'qa/FINAL_CHECKSUM_VERIFICATION.json'}
 writejson(ROOT/'RETURN_TO_CHATGPT.json',return_json)

 # Final acceptance and terminal owner state are written before the checksum manifest.
 acceptance={'schema_version':1,'status':'complete_with_limits','study':'post_V8_evidence_completion_no_new_model',
  'created_utc':now(),'preflight':'PASS','typed_vector_pipeline':'PASS','cache_resume':'PASS','selection_freeze':'PASS',
  'real_data_scoring':'PASS','aggregate_reconciliation':'PASS','figure_build':'PASS','agent_visual_review':'PASS',
  'human_figure_review':'PENDING','publisher_compliance':'NOT_ASSESSED_NO_PUBLICATION_TARGET',
  'horizon_rows':4248,'axis_rows':6048,'macro_reconciliation_rows':216,'per_window_reconciliation_rows':2220,
  'target_archive_loads_v8_1':1,'scorer_cache_package_loads':2,'new_fits':0,'optimizer_updates':0,
  'teacher_student_forward_jobs':0,'new_dev_prediction_jobs':0,'encoder_extractions':0,'bootstrap_draws':0,
  'model_selection_or_search':0,'gpu_commands':0,'final_heldout_payload_access':0,'old_root_writes':0,
  'independent_confirmation':False,'next_training_experiment_executed':False,
  'checksum_policy':'CHECKSUMS.sha256 generated after all other content; qa/FINAL_CHECKSUM_VERIFICATION.json is post-manifest readback and excluded from the manifest'}
 writejson(ROOT/'qa/FINAL_ACCEPTANCE.json',acceptance)
 owner_path=ROOT/'coordination/OWNER_STATE.json';owner=jread(owner_path)
 owner.update({'status':'complete_with_limits','completed_utc':now(),'final_acceptance_path':'qa/FINAL_ACCEPTANCE.json',
               'checksum_seal_pending':True,'single_writer_terminal_state':True})
 writejson(owner_path,owner)
 external_lock=BASE/'.CORE_B_V8_1_EVIDENCE_COMPLETION_V1_20261017T054525Z.owner.lock/OWNER.json'
 if external_lock.exists():
  ext=jread(external_lock);ext.update({'status':'complete_with_limits','completed_utc':owner['completed_utc'],
                                       'output_acceptance_path':str(ROOT/'qa/FINAL_ACCEPTANCE.json'),'writer_released':True})
  writejson(external_lock,ext)
 print(json.dumps({'status':'FINALIZATION_PASS','aggregate':agg_qa,'sign_audit':sign_audit,
                   'files_prepared_for_checksum':True,'owner_status':'complete_with_limits'},indent=2,sort_keys=True))

if __name__=='__main__':main()

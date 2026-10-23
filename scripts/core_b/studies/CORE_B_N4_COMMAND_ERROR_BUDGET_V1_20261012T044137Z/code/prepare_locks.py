#!/usr/bin/env python3
"""Lock N4 from accepted metadata before reading N4 targets or prediction values."""
import os
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N3=P/'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
N1=P/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
V3=P/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
B=P/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
SEEDS=[20261001,20261002,20261003]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def C(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def W(rel,obj):
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name('.'+p.name+'.part')
 t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 t.replace(p)
def main():
 if (R/'protocol/PROTOCOL_LOCK.json').exists():raise SystemExit('Existing N4 lock: inspect owner before resume.')
 stamp=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 W('logs/ATTEMPT_001_PRELOCK.json',{
  'attempt':1,'timestamp_utc':'2026-10-12T04:42:12Z','phase':'pre-outcome-lock-generation',
  'status':'FAILED_PRELOCK_VALIDATION',
  'error':'Validator incorrectly required the Q1 Hadamard table hash on Q0 sinusoidal-query rows; stopped before source/analysis locks.',
  'target_arrays_read':False,'prediction_values_read':False,'model_loaded':False,'gpu_access':False,
  'old_root_writes':0,'partial_preoutcome_artifacts':['protocol/CASE_SELECTION_LOCK.csv','protocol/MATRIX24.csv','inventory/PREDICTION_REUSE_PLAN.json']})
 expect='21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f'
 if sha(N2/'CHECKSUMS.sha256')!=expect:raise SystemExit('N2 checksum manifest changed.')
 cm={}
 for line in (N2/'CHECKSUMS.sha256').read_text().splitlines():
  x=line.split('  ',1)
  if len(x)==2:cm[x[1].removeprefix('./')]=x[0]
 selected=['RETURN_TO_CHATGPT.zh-CN.md','qa/FINAL_ACCEPTANCE.json','qa/SAFE_EXPORT_ACCEPTANCE.json',
 'qa/SAFE_ONLY_RELOAD.json','qa/N2_PREDICTION_ACCEPTANCE.json','qa/N2_PROFILE_ACCEPTANCE.json',
 'qa/N1_STORED_SCORE_RECONCILIATION.json','qa/OUTCOME_EXPOSURE.json','protocol/N2_MATRIX24.csv',
 'protocol/N2_MODEL_REGISTRY.csv','protocol/N2_PROFILE_LOCK.json','protocol/ANALYSIS_LOCKED.json',
 'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz','results/N2_VIEW_ACCEPTANCE.csv','MODEL_EXPORT_REQUEST.csv',
 'n1_refs/accepted_n1_refs/N1_ARTIFACT_MANIFEST.csv','n1_refs/accepted_n1_refs/FINAL_ACCEPTANCE.json',
 'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv','n3_refs/N3_DIRECT_REUSE.json',
 'n3_refs/ACCEPTED_N3_ARTIFACT_MANIFEST.csv','code/run_n2_inference.py','code/score_n2.py',
 'source/d3_model.py','source/fixed_future_query_model.py','source/analyze_n1_metrics.py']
 for d in (N2/'safe_models').iterdir():
  if d.is_dir():
   selected += [str((d/x).relative_to(N2)) for x in
    ('model.safetensors','EXPORT_MANIFEST.json','BUFFER_INVENTORY.json','QUERY_SPEC.json','TENSOR_CATALOGUE.json')]
 views=C(N2/'results/N2_VIEW_ACCEPTANCE.csv')
 cuda=[r for r in views if r.get('profile')=='CUDA_COMMON' and r.get('status')=='PASS']
 if len(cuda)!=12:raise SystemExit('N2 CUDA_COMMON accepted views !=12.')
 for r in cuda:
  selected += [str(Path(r[k]).relative_to(N2)) for k in ('prediction_path','sidecar_path')]
 checked=[]
 for rel in sorted(set(selected)):
  if rel not in cm or sha(N2/rel)!=cm[rel]:raise SystemExit('Selected N2 sealed-source hash mismatch: '+rel)
  checked.append({'path':str(N2/rel),'sha256':cm[rel]})
 safe=J(N2/'qa/SAFE_EXPORT_ACCEPTANCE.json');reload=J(N2/'qa/SAFE_ONLY_RELOAD.json')
 pred=J(N2/'qa/N2_PREDICTION_ACCEPTANCE.json');prof=J(N2/'qa/N2_PROFILE_ACCEPTANCE.json')
 plock=J(N2/'protocol/N2_PROFILE_LOCK.json')
 if reload.get('promoted_count')!=12 or reload.get('torch_load_called') is not False:raise SystemExit('N2 safe-only reload gate failed.')
 if pred.get('accepted_views')!=24 or pred.get('cuda_common_views')!=12 or pred.get('target_arrays_loaded_by_inference') is not False:raise SystemExit('N2 prediction gate failed.')
 if prof.get('CUDA_COMMON',{}).get('status')!='PASS':raise SystemExit('N2 common CUDA profile gate failed.')
 n1man=C(N2/'n1_refs/accepted_n1_refs/N1_ARTIFACT_MANIFEST.csv')
 n1train=[r for r in n1man if r.get('artifact_type')=='train_prediction']
 profiles=C(N2/'protocol/AS_RUN_PROFILE_REGISTRY.csv')
 profmap={(r['model'],r['query'],r['seed']):r for r in profiles}
 if len(n1train)!=12 or len(profmap)!=12:raise SystemExit('N1 accepted train manifest/profile incomplete.')
 # Correctly bind the previous N2 draws to the exact dev source-group/seed order.
 sgm=C(N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv')
 es={int(r['episode_index']):r['split'] for r in sgm}
 eg={int(r['episode_index']):r['source_group_id'] for r in sgm}
 tr_groups=sorted({eg[e] for e,s in es.items() if s=='train'})
 dv_groups=sorted({eg[e] for e,s in es.items() if s=='dev'})
 if len(tr_groups)!=2712 or len(dv_groups)!=603:raise SystemExit('N1 split group counts mismatch.')
 drawfile=N2/'protocol/N2_GROUP_BOOTSTRAP_DRAWS.npz'
 with np.load(drawfile,allow_pickle=False) as z:
  gids=np.asarray(z['source_group_ids']).astype('U128')
  gi=np.asarray(z['group_draw_indices']);si=np.asarray(z['seed_draw_indices'])
  dseed=int(np.asarray(z['prng_seed']).item());reps=int(np.asarray(z['replicates']).item())
 if not (reps==10000 and dseed==20261011 and np.array_equal(gids,np.asarray(dv_groups,dtype='U128'))
  and gi.shape==(10000,603) and si.shape==(10000,3) and np.all((gi>=0)&(gi<603)) and np.all((si>=0)&(si<3))):
  raise SystemExit('N2 group/seed draw manifest differs from N4; keep target gate closed.')
 if '20261001' not in str(SEEDS):raise SystemExit('fixed seed order mismatch.')
 # Check canonical train/dev metadata only. Do not access targets_native.
 with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as a:
  for k in ('split_code','anchor_id','episode_index'):
   if k not in a.files:raise SystemExit('Missing target-blind metadata '+k)
  split=np.asarray(a['split_code'],dtype=np.int8)
  aids=np.asarray(a['anchor_id']).astype('U96')
  eps=np.asarray(a['episode_index'],dtype=np.int64)
 if len(split)!=20079 or int((split==0).sum())!=17757 or int((split==1).sum())!=2322:raise SystemExit('Canonical anchor population mismatch.')
 amap={str(aids[i]):(int(eps[i]),'train' if split[i]==0 else 'dev') for i in range(len(aids))}
 cases=C(R/'protocol/CASE_SELECTION_LOCK.csv')
 if len(cases)!=14 or len({x['source_group_id'] for x in cases})!=14:raise SystemExit('Case selection must be 14 distinct groups.')
 for x in cases:
  ep=int(x['episode_index'])
  if amap.get(x['anchor_id'])!=(ep,'dev') or es.get(ep)!='dev' or eg.get(ep)!=x['source_group_id']:
   raise SystemExit('Case selection ID/group metadata mismatch.')
 # All source hashes are explicit. No old full checkpoint is deserialized or read.
 paths={
 'n2_return':N2/'RETURN_TO_CHATGPT.zh-CN.md','n2_final_acceptance':N2/'qa/FINAL_ACCEPTANCE.json',
 'n2_safe_export_acceptance':N2/'qa/SAFE_EXPORT_ACCEPTANCE.json','n2_safe_only_reload':N2/'qa/SAFE_ONLY_RELOAD.json',
 'n2_prediction_acceptance':N2/'qa/N2_PREDICTION_ACCEPTANCE.json','n2_profile_acceptance':N2/'qa/N2_PROFILE_ACCEPTANCE.json',
 'n2_profile_lock':N2/'protocol/N2_PROFILE_LOCK.json','n2_analysis_lock':N2/'protocol/ANALYSIS_LOCKED.json',
 'n2_matrix':N2/'protocol/N2_MATRIX24.csv','n2_view_acceptance':N2/'results/N2_VIEW_ACCEPTANCE.csv',
 'n1_artifact_manifest':N2/'n1_refs/accepted_n1_refs/N1_ARTIFACT_MANIFEST.csv',
 'n1_acceptance':N2/'n1_refs/accepted_n1_refs/FINAL_ACCEPTANCE.json',
 'source_group_map':N2/'n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv',
 'case_manifest':N1/'qualitative/CASE_MANIFEST.csv','v3_inputs':V3/'data/V3_MAIN_INPUTS.npz',
 'target_normalizer':V3/'data/V3_NORMALIZERS.json','feature_cache_manifest':B/'d3/cache/D3_CACHE_MANIFEST.json',
 'video_normalizer':B/'d3/cache/D3_VIDEO_NORMALIZER.json','query_archive':N1/'protocol/HADAMARD_QUERY_TABLES.npz',
 'query_table_registry':N1/'protocol/QUERY_TABLES_AND_HASHES.json','bootstrap_draws':drawfile,
 'n3_direct_reference':N2/'n3_refs/N3_DIRECT_REUSE.json','n3_manifest':N2/'n3_refs/ACCEPTED_N3_ARTIFACT_MANIFEST.csv'}
 sh={k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()}
 reqs=C(N2/'MODEL_EXPORT_REQUEST.csv')
 qhash=J(N1/'protocol/QUERY_TABLES_AND_HASHES.json')['candidate_table_sha256_h30']
 for r in reqs:
  if r['query']=='Q1' and r['accepted_query_table_sha256']!=qhash:raise SystemExit('Q1 table hash mismatch.')
  if r['query']=='Q0' and r['accepted_query_table_sha256'] not in ('',None):raise SystemExit('Q0 should have no Hadamard table hash.')
  for key,ref in [('input_sha256','v3_inputs'),('target_normalizer_sha256','target_normalizer'),
   ('feature_cache_manifest_sha256','feature_cache_manifest'),('video_normalizer_sha256','video_normalizer')]:
   if r[key]!=sh[ref]['sha256']:raise SystemExit('Request hash mismatch: '+key)
  if r['query']=='Q1' and r['query_archive_sha256']!=sh['query_archive']['sha256']:
   raise SystemExit('Request hash mismatch: query_archive_sha256')
  if r['query']=='Q0' and r['query_archive_sha256'] not in ('',None):
   raise SystemExit('Q0 should not bind the Q1 query archive.')
 matrix=C(R/'protocol/MATRIX24.csv');reuse=J(R/'inventory/PREDICTION_REUSE_PLAN.json')
 if len(matrix)!=24 or reuse.get('dev_reused')!=12 or reuse.get('train_new_forward')!=12:raise SystemExit('Partial matrix/reuse plan invalid.')
 if sum(r['source_status']=='REUSE_ACCEPTED_N2_CUDA_COMMON' for r in matrix)!=12:raise SystemExit('N2 dev matrix rows invalid.')
 n3direct=J(N2/'n3_refs/N3_DIRECT_REUSE.json')
 n1accept=J(N2/'n1_refs/accepted_n1_refs/FINAL_ACCEPTANCE.json')
 bg=C(N1/'inventory/BACKGROUND_BASELINE_STATUS.csv') if (N1/'inventory/BACKGROUND_BASELINE_STATUS.csv').is_file() else []
 stamp2='2026-10-12T04:42:12Z'
 W('inventory/SOURCE_ACCEPTANCE.json',{'schema_version':1,'status':'PASS_INCREMENTAL_N2_N3_SOURCE_ACCEPTANCE',
  'created_utc':stamp,'verification_scope':'Named acceptance files and only files used by N4; sealed N2 manifest identity and selected N2 sources verified; no full checkpoint deserialization.',
  'n2_root':str(N2),'n2_checksum_manifest_sha256':sha(N2/'CHECKSUMS.sha256'),
  'n2_checksum_manifest_expected_sha256':expect,'n2_selected_files_verified_against_manifest':len(checked),
  'n2_selected_file_inventory':checked,'n2_safe_model_count':12,'n2_safe_reload_status':reload.get('status'),
  'n2_cuda_common_dev_views':12,'n2_target_blind':True,'n2_cuda_profile':prof['CUDA_COMMON'],
  'n1_accepted_train_candidates':12,'n1_matching_common_train_profiles':0,'n1_train_reuse_rejected_due_profile':12,
  'n1_acceptance_status':n1accept.get('status'),'n3_direct_reference':n3direct,
  'source_hashes':sh,'old_root_writes':0,'full_checkpoint_recovery_rerun':False})
 W('protocol/PROTOCOL_LOCK.json',{'schema_version':1,'study_id':'CORE_B_N4_COMMAND_ERROR_BUDGET_V1',
  'lock_status':'LOCKED_BEFORE_N4_OUTCOME_READ','locked_utc':stamp,
  'hypothesis_timing':'retrospective/exploratory after prior N1 train and N2/N3 dev outcome exposure',
  'questions':['Seed 20261002 gains: level error vs time-centered error and oracle level correction.',
   'Dynamic derivative norm vs zero-lag alignment, retaining original MAE/safe-FD.',
   'G/V/I query/system contrasts and fixed-seed/LOO stability; absolute co-primary advantage.'],
  'models':{'heads':['TS11-D','STATE-CTRL'],'queries':['Q0','Q1'],'seeds':SEEDS,'horizon':30,'epochs':40,
   'optimizer_updates':11120,'models_fixed':12,'analysis_bundles':24},
  'population_expected':{'primary_train':{'anchors':17757,'episodes':2712,'source_groups':2712},
   'exposed_dev':{'anchors':2322,'episodes':603,'source_groups':603}},
  'scope':{'supervised_fits':0,'new_neural_training':0,'optimizer_steps':0,'encoder_extraction':0,
   'new_train_forward_max':12,'new_dev_formal_forward':0,'n2_rerun':False,'n3_rerun_or_rescore':False,
   'ridge_obs43_other_windows':False,'final_heldout_access':0},
  'evaluation':'All results exploratory; train in-sample; dev previously exposed; no p-values or independent confirmation.',
  'target_gate':'Read no targets until all 12 train forwards accepted and all 12 reused N2 CUDA_COMMON dev views revalidated.'})
 W('protocol/PROFILE_LOCK.json',{'schema_version':1,'profile_id':'N2_CUDA_COMMON_REUSED_FOR_N4',
  'source_acceptance_path':str(N2/'qa/N2_PROFILE_ACCEPTANCE.json'),
  'source_profile_acceptance_sha256':sha(N2/'qa/N2_PROFILE_ACCEPTANCE.json'),
  'source_profile_lock_sha256':sha(N2/'protocol/N2_PROFILE_LOCK.json'),
  'runtime':prof['CUDA_COMMON'],'locked_profile':plock,
  'physical_gpu':'Shared watcher dynamic_all_visible; prefer source physical GPU 2 and record actual GPU/UUID.',
  'canary_rule':'If assigned GPU UUID differs from N2 CUDA_COMMON GPU 2 UUID, run 32 fixed dev IDs through all 12 models, compare against accepted N2 CUDA_COMMON predictions using existing locked tolerance; canary is not a formal view.',
  'safe_model_only':True,'inference_mode':True,'targets_loaded':False})
 W('protocol/METRIC_BUDGET_LOCK.json',{'schema_version':1,'status':'LOCKED_BEFORE_N4_OUTCOME_READ',
  'target_normalizer_sha256':sh['target_normalizer']['sha256'],
  'native_contract':'native = anchor_command_native + target_sigma * residual; score native target/prediction once.',
  'co_primary':['N1 position MAE','N1 safe-FD L1 with existing seam contract unchanged'],
  'support':['native FD','native TV','frozen active-mask direction','29 future-internal differences','anchor-to-first-future seam separately'],
  'axes':{'raw28':'[0:28]','block0':'[0:14]','block1':'[14:28]','physical_units_verified':False},
  'E1':{'e':'((yhat-a)-(y-a))/sigma','level':'b=mean_t(e); mean_j(b^2)',
   'centered':'mean_tj((e-b)^2)','identity':'POS_MSE=level+time_centered',
   'output_constant':'repeat mean_t(yhat); output-only diagnostic',
   'oracle':'delta=median_t(e); score e-delta; GT used, diagnostics only, not deployable'},
  'E2':{'d':'D29(y)/safe_fd_scale','dhat':'D29(yhat)/safe_fd_scale',
   'moments':'T=E[d^2], P=E[dhat^2], C=E[d*dhat]','fd_mse':'T+P-2C',
   'r':'sqrt(P/T)','c':'C/sqrt(T*P)','skill':'(2C-P)/T',
   'geometry':'FD_MSE/T=(1-r)^2+2r(1-c)','aggregate':'aggregate moments first, then ratios; also per-seed ratios',
   'zero_rules':{'T>0,P=0':'r=0,skill=0,c=NA,norm=1,alignment=0','T=0':'FD_MSE=P, ratios/skill NA; retain rows'},
   'c_interpretation':'zero-lag cosine inner product, not Pearson or lag detection'},
  'diagnostics':'No fit of scale, time lag, basis or frequency; L2 does not replace L1 co-primary.',
  'fp64_tolerance':{'atol':1e-12,'rtol':1e-10},
  'contrasts':{'G':'L_Q0-L_Q1','V':'L_STATE-CTRL-L_TS11-D','I':'G_TS11-D-G_STATE-CTRL'},
  'interaction':'I=V_Q1-V_Q0; calculate separately per endpoint/component.'})
 W('protocol/ANALYSIS_LOCK.json',{'schema_version':1,'status':'LOCKED_BEFORE_N4_OUTCOME_READ',
  'episode_weighting':'Within episode average anchors; then equally weighted episodes; retain source-group cluster membership.',
  'dev_bootstrap':{'method':'paired source-group percentile; same draw for all models and endpoints',
   'replicates':10000,'source_groups':603,'source_group_order_sha256':hashlib.sha256('\n'.join(dv_groups).encode()).hexdigest(),
   'seed_order':SEEDS,'draw_source':'REUSE_N2_COMPLETE_DRAW_MANIFEST','draw_manifest_sha256':sha(drawfile),
   'prng_seed':dseed,'recompute':'Each draw first recomputes each seed ratio; then apply recorded seed resampling weights; linear components use same weights; aggregate T/P/C before ratios.',
   'scope':'conditional on fixed models and previously exposed dev only'},
  'train_ci':'none, in-sample','p_values':'not computed','three_seed_summary':'mean, sample SD, median, three LOO; never average predictions'})
 W('protocol/OUTCOME_EXPOSURE.json',{'schema_version':1,'status':'LOCKED_BEFORE_N4_OUTCOME_READ',
  'n4_outcomes_opened':False,'previous_exposure':{'n1_train_target_rows':17757,'n1_train_bundles':12,
  'n2_dev_target_rows':2322,'n2_access_attempts':2,'n2_failed_attempts_before_commit':1,
  'n3_predictions':18,'n3_reconciliation_rows':162,'n3_dev_exposed':True},
  'n4_access_plan':{'train':'Only after predictions accepted; in-sample','dev':'Only after predictions accepted; previously exposed',
  'final_test_access':0,'heldout_access':0}})
 W('protocol/CASE_SELECTION_LOCK.json',{'schema_version':1,'status':'LOCKED_BEFORE_N4_OUTCOME_READ',
  'source_manifest':str(N1/'qualitative/CASE_MANIFEST.csv'),'source_manifest_sha256':sha(N1/'qualitative/CASE_MANIFEST.csv'),
  'fields_used':['case_id','source_group_id','episode_index','anchor_id','selection_hash'],
  'fields_ignored':['dev_group_motion_score','train_q25','train_q75','target_first_frame','target_last_frame'],
  'selection_rule':'Prior accepted case ID intersection by canonical IDs; add hash-order metadata-only groups to 14; one anchor per group.',
  'case_groups':cases,'future_gt_used_for_selection':False})
 W('inventory/OTHER_WINDOW_REFERENCE_STATUS.json',{'schema_version':1,
  'status':'CHECKED_REFERENCE_ONLY_NOT_RELAUNCHED','reference':'N1 inventory/BACKGROUND_BASELINE_STATUS.csv and accepted N1 return.',
  'background_status_rows':bg,'ridge_obs43_horizon_relaunched':False,'other_window_modifications':False})
 W('n3_refs/N3_DIRECT_REUSE.json',{'schema_version':1,'status':'DIRECT_REFERENCE_ONLY','source_root':str(N3),
  'source_acceptance_path':str(N3/'qa/FINAL_ACCEPTANCE.json'),'n2_reference_path':str(N2/'n3_refs/N3_DIRECT_REUSE.json'),
  'accepted_predictions':18,'score_reconciliation_rows':162,'rerun':False,'rescore':False,
  'interpretation':'FULL near zero, OBS negative, ANCHOR positive; non-monotonic, fixed seeds, exposed dev.'})
 W('coordination/OWNER_STATE.json',{'schema_version':1,'study_id':'CORE_B_N4_COMMAND_ERROR_BUDGET_V1',
  'owner_active':True,'status':'RUNNING','owner':'Codex Luna Max','root':str(R),'created_utc':stamp,
  'phase':'preoutcome_protocol_locked','last_verified_utc':stamp,'final_test_access':0})
 W('coordination/JOB_LEDGER.json',{'schema_version':1,'jobs':[],'new_neural_training':0,'optimizer_steps':0,
  'encoder_extraction':0,'new_train_forward_started':0,'new_dev_formal_forward_started':0,
  'max_project_gpu_children':2,'watcher':'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py',
  'status':'preflight_pending'})
 W('coordination/RESOURCE_LOG.json',{'schema_version':1,'preflight_pending':True,'gpu_jobs':[],
  'disk_free_at_preflight':'pending','training_cost':0,'inference_gpu_hours':'pending watcher receipt'})
 (R/'coordination/RESUME_COMMANDS.md').write_text(
  '# N4 resume commands\n\nCheck coordination/OWNER_STATE.json, coordination/JOB_LEDGER.json and qa/PREDICTION_ACCEPTANCE.json. '
  'Resume only using the command in the ledger through the shared project GPU lease watcher.\n',encoding='utf-8')
 print(json.dumps({'status':'PASS_PREOUTCOME_LOCKS','root':str(R),'utc':stamp,
  'n2_selected_files_verified':len(checked),'n1_train_candidates':len(n1train),
  'matching_common_train_profiles':0,'train_forward_required':12,'n2_cuda_dev_reused':len(cuda),
  'matrix_rows':len(matrix),'train_anchors':17757,'dev_anchors':2322,'train_groups':2712,'dev_groups':603,
  'bootstrap_reused':True,'case_count':len(cases),'target_arrays_read':False,'n3_rerun_or_rescore':False},
  ensure_ascii=False,indent=2))
if __name__=='__main__':main()

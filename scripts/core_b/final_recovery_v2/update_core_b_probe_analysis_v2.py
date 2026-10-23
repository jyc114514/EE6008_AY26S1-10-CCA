#!/usr/bin/env python3
"""Update Core B route/hypothesis evidence from the completed P1–P6 probes."""
from __future__ import annotations
import os
import csv,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

OUT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z'))
RUN=Path(os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'outputs', 'ee6008')), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z'))
PROBE=OUT/"core_b/CORE_B_GPU_DIFFERENTIAL_PROBES_ATTEMPT_02.json"
ARCHIVE=RUN/"core_b/gpu_probes/CURRENT_PROFILE_BRIDGE_FEATURES_ATTEMPT_02.npz"
HYP_CSV=OUT/"core_b/CORE_B_CAUSAL_HYPOTHESES.csv"
HYP_MD=OUT/"core_b/CORE_B_DIFFERENTIAL_HARNESS_AND_HYPOTHESES.md"
TABLE=OUT/"core_b/PROFILE_AND_SOURCE_COMPARISON.csv"
ROUTE=OUT/"core_b/ROUTE_DECISION.json"
PROBE_KEYS=[
 ("P1_D3_BF16_BATCH1_DEFAULT","BF16 autocast; batch 1; runtime backend defaults","BF16","default","default","Reference; exact current replay of historical D3 code."),
 ("P2_FP32_BATCH1_DEFAULT","FP32; autocast off; batch 1; runtime backend defaults","FP32","default","default","Autocast/precision contrast from P1."),
 ("P3_FP32_BATCH1_MATH_SDPA","FP32; batch 1; force math SDPA","FP32","math","default","SDPA backend contrast from P2."),
 ("P4_FP32_BATCH1_TF32_OFF","FP32; batch 1; matmul/cuDNN TF32 off","FP32","default","off","TF32 policy contrast from P2."),
 ("P5_CURRENT_PROFILE_V2_REPEAT_0","FP32; batch 1; math SDPA; matmul/cuDNN TF32 off","FP32","math","off","Pre-score candidate profile repeat 1."),
 ("P6_CURRENT_PROFILE_V2_REPEAT_1","FP32; batch 1; math SDPA; matmul/cuDNN TF32 off","FP32","math","off","Pre-score candidate profile repeat 2."),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def backup(p):
 b=p.with_name(p.name+'.pre-p2-p6-route-update-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.bak')
 shutil.copy2(p,b);return b
def atomic_json(p,obj):
 t=p.with_name(f'.{p.name}.{os.getpid()}.part');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 d=json.loads(PROBE.read_text(encoding='utf-8'))
 if d.get('status')!='COMPLETED_TRAIN_ONLY_CONTROLLED_GPU3_PROBES' or d.get('physical_gpu')!=3 or d.get('final_payload_read') or d.get('future_targets_materialized'):
  raise ValueError("train-only six-probe result does not pass its scope gate")
 with np.load(ARCHIVE,allow_pickle=False) as z:
  arrays={k:z[k].copy() for k in z.files}
 c=d['comparisons'];rows=[]
 q0_max={k:max(x['max_abs_difference'] for x in v['pooled_vs_q0_by_input']) for k,v in c.items()}
 q0_mean={k:float(np.mean([x['mean_abs_difference'] for x in v['pooled_vs_q0_by_input']])) for k,v in c.items()}
 q0_close={k:all(x['allclose_atol_1e-5_rtol_1e-5'] for x in v['pooled_vs_q0_by_input']) for k,v in c.items()}
 direct={}
 for a,b in [('P3_FP32_BATCH1_MATH_SDPA','P2_FP32_BATCH1_DEFAULT'),('P4_FP32_BATCH1_TF32_OFF','P2_FP32_BATCH1_DEFAULT'),('P5_CURRENT_PROFILE_V2_REPEAT_0','P4_FP32_BATCH1_TF32_OFF'),('P6_CURRENT_PROFILE_V2_REPEAT_1','P5_CURRENT_PROFILE_V2_REPEAT_0')]:
  delta=np.abs(arrays[a].astype(np.float64)-arrays[b].astype(np.float64))
  direct[(a,b)]={'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'bitwise_equal':bool(np.array_equal(arrays[a],arrays[b]))}
 for key,config,precision,sdpa,tf32,note in PROBE_KEYS:
  drow={
   'probe_id':key,'support':'2 fixed train rows × dynamic/static',
   'configuration':config,'precision':precision,'sdpa':sdpa,'tf32':tf32,'batch_size':1,
   'pooled_max_abs_vs_q0':q0_max[key],'pooled_mean_abs_vs_q0_mean_over_4':q0_mean[key],
   'q0_allclose_atol_1e-5_rtol_1e-5_all_4':q0_close[key],
   'pooled_max_abs_vs_historical_d3':max(x['max_abs_difference'] for x in c[key]['pooled_vs_legacy_d3_by_input']),
   'pooled_mean_abs_vs_historical_d3_mean_over_4':float(np.mean([x['mean_abs_difference'] for x in c[key]['pooled_vs_legacy_d3_by_input']])),
   'raw_token_max_abs_vs_historical_d3':max(x['max_abs_difference'] for x in c[key]['raw_token_vs_legacy_by_input']),
   'candidate_repeatability_bitwise':None,'direct_reference':None,'direct_max_abs_difference':None,'direct_mean_abs_difference':None,
   'interpretation':note}
  if key=='P3_FP32_BATCH1_MATH_SDPA':
   x=direct[(key,'P2_FP32_BATCH1_DEFAULT')];drow.update(candidate_repeatability_bitwise=x['bitwise_equal'],direct_reference='P2_FP32_BATCH1_DEFAULT',direct_max_abs_difference=x['max_abs'],direct_mean_abs_difference=x['mean_abs'])
  elif key=='P4_FP32_BATCH1_TF32_OFF':
   x=direct[(key,'P2_FP32_BATCH1_DEFAULT')];drow.update(direct_reference='P2_FP32_BATCH1_DEFAULT',direct_max_abs_difference=x['max_abs'],direct_mean_abs_difference=x['mean_abs'])
  elif key=='P5_CURRENT_PROFILE_V2_REPEAT_0':
   x=direct[(key,'P4_FP32_BATCH1_TF32_OFF')];drow.update(direct_reference='P4_FP32_BATCH1_TF32_OFF',direct_max_abs_difference=x['max_abs'],direct_mean_abs_difference=x['mean_abs'])
  elif key=='P6_CURRENT_PROFILE_V2_REPEAT_1':
   x=direct[(key,'P5_CURRENT_PROFILE_V2_REPEAT_0')];drow.update(candidate_repeatability_bitwise=x['bitwise_equal'],direct_reference='P5_CURRENT_PROFILE_V2_REPEAT_0',direct_max_abs_difference=x['max_abs'],direct_mean_abs_difference=x['mean_abs'])
  rows.append(drow)
 if TABLE.exists():backup(TABLE)
 with TABLE.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 hypotheses=[
  {'hypothesis_id':'H1','hypothesis':'row_or_frame_alignment','prediction_if_true':'Q0 shard row metadata or source frame schedule differs from matching bridge row','discriminating_check':'Join fixed train IDs across accepted bridge, V3 identity, Q0 shard sidecars, source video and selected frame indices','observed_evidence':'Both selected indices/episodes/tasks/anchors and the 32 frame schedules agree; current video/RGB/preprocess hashes are source-bound for these rows.','status':'NOT_OBSERVED_FOR_TWO_TRAIN_ROWS; HISTORICAL_Q0_GENERATION_IDENTITY_STILL_UNBOUND'},
  {'hypothesis_id':'H2','hypothesis':'storage_cast_axis_or_padding','prediction_if_true':'Q0 row read differs from stored slice, dtype/layout differs, or a source-documented conversion explains cache values','discriminating_check':'Check direct shard slice, dtype/shape and only documented conversions; do not fit a correction','observed_evidence':'Q0 rows are float32 [2,16,4,768]; reader equals direct shard slice. No historical generation script/hash or payload hash proves a pre-save conversion.','status':'READ_TIME_CAST_OR_LAYOUT_RULED_OUT; GENERATION_TIME_CONVERSION_UNKNOWN'},
  {'hypothesis_id':'H3','hypothesis':'decoder_or_preprocessing','prediction_if_true':'Identical frame IDs produce different RGB pixels or preprocess tensors','discriminating_check':'Compare video/frame/preprocess source hashes and exact RGB/preprocessed tensor hashes on fixed train IDs','observed_evidence':'Current OpenCV frame lists, RGB hashes, preprocess module hash and dynamic/static tensor hashes are recorded and matched the V2 CPU audit; Q0 has no pixel/tensor reference.','status':'CURRENT_INPUT_PIPELINE_PINNED; HISTORICAL_INPUT_BYTES_UNAVAILABLE'},
  {'hypothesis_id':'H4','hypothesis':'encoder_numeric_profile','prediction_if_true':'A controlled precision/backend change yields repeatable drift on identical rows','discriminating_check':'Six registered train-only probes vary autocast, SDPA and TF32; P5/P6 repeat the candidate; no batch sweep','observed_evidence':'P1 reproduces historical D3 exactly but fails Q0; P2 changes raw/grid output; P3 equals P2; P4 differs modestly from P2; P5/P6 are bitwise identical yet fail Q0.','status':'NUMERIC_PROFILE_SENSITIVITY_OBSERVED; Q0_CAUSE_UNRESOLVED; NO_GPU_NUMBER_ATTRIBUTION'},
  {'hypothesis_id':'H5','hypothesis':'token_grid_pooling','prediction_if_true':'Token layout, reshape or pooling differs from accepted D3 contract','discriminating_check':'Verify raw token shape and source pooling code against stored grid descriptor; inspect source-backed transforms only','observed_evidence':'Current/historical D3 code yields [16,24,24,768] and four disjoint 12x12 means; Q0 records float32 grid but no raw-token evidence or extractor hash.','status':'CURRENT_POOLING_CONTRACT_PINNED; HISTORICAL_EXECUTION_NOT_FULLY_BOUND'},
 ]
 backup(HYP_CSV)
 with HYP_CSV.open('w',encoding='utf-8',newline='') as f:
  fields=list(hypotheses[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(hypotheses)
 md=HYP_MD.read_text(encoding='utf-8')
 backup(HYP_MD)
 md=md.replace('**Status:** CPU identity/storage/preprocessing audit passed for two preselected train rows; controlled GPU3 probe set is pending. No Core B development score or final target has been read in this V2 work.','**Status:** CPU identity/storage/preprocessing audit passed for two fixed train rows; all six registered GPU3 probes completed. Full current-profile dev features/scores and Core B final RGB/targets remain unaccessed at this decision point.')
 md=md.replace('P2–P6 remain pending; the candidate profile will be selected from source traceability and a fixed numerical definition, before full development scores.','P2–P6 have now completed; the candidate profile is selected from source traceability, an explicit numerical definition and repeatability, before full development scores.')
 md += '''
\n\n## GPU3 P2–P6 probe results and route decision\n\nThe six registered settings ran through the shared watcher on physical GPU3 (RTX 3090, process-local cuda:0), on the same two fixed train rows with dynamic/static inputs. All outputs were finite. The candidate repeats P5/P6 matched bitwise for both pooled grids and raw tokens. No dev targets, final RGB, final predictions or training steps were accessed.\n\n| Probe | Numeric profile | Pooled max abs vs Q0 | Mean abs vs Q0 (four grids) | Q0 1e-5 allclose | Finding |\n|---|---|---:|---:|---|---|\n'''
 for key,config,precision,sdpa,tf32,note in PROBE_KEYS:
  drow=next(r for r in rows if r['probe_id']==key)
  md+=f"| {key.split('_')[0]} | {config} | {drow['pooled_max_abs_vs_q0']:.8f} | {drow['pooled_mean_abs_vs_q0_mean_over_4']:.8f} | {'pass' if drow['q0_allclose_atol_1e-5_rtol_1e-5_all_4'] else 'fail'} | {note} |\n"
 md+='''\nP3's pooled and raw-token hashes equal P2; forcing math SDPA did not change this device's result. P4 changed the TF32 policy but remained outside the Q0 tolerance. P5 adds the explicit math-SDPA policy and P6 repeats it; the two candidate outputs are bitwise equal. The candidate profile is therefore **FP32 without autocast, batch one, matmul/cuDNN TF32 disabled, math SDPA forced**, with the observed cuDNN benchmark/determinism and deterministic-algorithm flags recorded. It was chosen before any full dev score and was not selected for Q0 similarity or model performance.\n\n**Route: R2 — CURRENT_PROFILE_V2.** The old Q0 bridge remains FAIL_UNRESOLVED. Historical cache-build precision/backend/source bindings are incomplete, so this audit does not identify why Q0 differs and does not attribute the difference to physical GPU1 or GPU3. The new profile defines a reproducible current inference path using the pinned encoder, source/preprocessing, anchor-limited frame schedule, pooling, environment and predictor checkpoints. The complete 185-window dev evaluation must use the fixed matrix and metric contract already locked in the candidate profile.\n\n'''
 HYP_MD.write_text(md,encoding='utf-8')
 route={
  'schema_version':1,'protocol_id':'EE6008_FINAL_RECOVERY_V2_20261020T053953Z',
  'status':'R2_SELECTED_BEFORE_CURRENT_PROFILE_DEV_FEATURES_OR_SCORES',
  'route':'R2','selected_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'old_cached_q0_development_replay':'PASS_ACCEPTED_V9_DEVELOPMENT_ONLY',
  'original_q0_fresh_feature_bridge':'FAIL_UNRESOLVED','r1_status':'NOT_ESTABLISHED',
  'current_profile_id':'CURRENT_PROFILE_V2','current_profile_lock_path':str(OUT/'core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json'),
  'root_cause':'UNKNOWN; Q0 historical generation profile/source is not fully bound; no GPU-number causal attribution',
  'basis':['source/video/anchor/frame identity checks passed on the two fixed train rows','P1 current GPU3 replay exactly matches historical D3 source but both differ from Q0','P2–P4 isolate precision/backend changes without restoring Q0','P5/P6 candidate output is bitwise repeatable on fixed train bridge inputs','current 185-row dev inputs are no-label and preserve original population/order'],
  'candidate_selection_basis':'source traceability, explicit FP32/backend/batch definition and train-only repeatability; no development/final metric was used',
  'probe_result_sha256':sha(PROBE),'probe_feature_archive_sha256':sha(ARCHIVE),
  'new_training_runs':0,'current_profile_full_dev_features_extracted':False,'current_profile_dev_scores_revealed':False,
  'final_rgb_extracted':False,'final_targets_materialized':False,
  'allowed_next_step':'freeze CURRENT_PROFILE_CANDIDATE_LOCK.json before full 185-window dev feature extraction and scoring'
 }
 if ROUTE.exists():backup(ROUTE)
 atomic_json(ROUTE,route)
 print(json.dumps({'route':route,'table_sha256':sha(TABLE),'hypothesis_csv_sha256':sha(HYP_CSV),'hypothesis_md_sha256':sha(HYP_MD),'route_sha256':sha(ROUTE)},indent=2,sort_keys=True))
 return 0
if __name__=='__main__':raise SystemExit(main())

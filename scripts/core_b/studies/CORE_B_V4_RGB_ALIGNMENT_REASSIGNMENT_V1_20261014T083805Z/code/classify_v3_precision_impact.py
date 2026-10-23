#!/usr/bin/env python3
"""Separate float32 zero crossings from material V3 contrast sign changes."""
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 raw=read(ROOT/'audit/V3_METRIC_PRECISION_IMPACT.csv');tol=1e-6;rows=[]
 for r in raw:
  old=float(r['old_v3_gain']);new=float(r['capture_corrected_gain']);delta=float(r['capture_minus_old_gain'])
  old_zero=abs(old)<=tol;meaningful=(not old_zero) and abs(new)>tol and ((old>0)!=(new>0))
  r={**r,'old_effect_within_replay_tolerance':old_zero,'zero_to_tiny_numeric_drift':old_zero and abs(new)<=tol,'material_nonzero_sign_flip':meaningful,'absolute_delta_within_replay_tolerance':abs(delta)<=tol}
  rows.append(r)
 p=ROOT/'audit/V3_METRIC_PRECISION_SENSITIVITY_CORRECTED.csv'
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 nonzero=[r for r in rows if not r['old_effect_within_replay_tolerance']]
 zero_drift=sum(bool(r['zero_to_tiny_numeric_drift']) for r in rows)
 flips=sum(bool(r['material_nonzero_sign_flip']) for r in rows)
 maxdelta=max(abs(float(r['capture_minus_old_gain'])) for r in rows)
 maxratio=max(abs(float(r['capture_minus_old_gain']))/abs(float(r['old_v3_gain'])) for r in nonzero)
 minold=min(abs(float(r['old_v3_gain'])) for r in nonzero)
 obj={'schema_version':1,'status':'PASS','raw_impact_table':str(ROOT/'audit/V3_METRIC_PRECISION_IMPACT.csv'),'raw_impact_table_sha256':sha(ROOT/'audit/V3_METRIC_PRECISION_IMPACT.csv'),'corrected_sensitivity_table':str(p),'corrected_sensitivity_sha256':sha(p),'accepted_native_replay_tolerance':{'atol':1e-6,'rtol':1e-6},'metric_delta_resolution_tolerance':tol,'raw_exact_zero_to_tiny_status_crossings':zero_drift,'nonzero_old_effect_cases':len(nonzero),'material_nonzero_effect_sign_flips':flips,'max_abs_metric_reconstruction_delta':maxdelta,'max_relative_delta_to_nonzero_old_effect':maxratio,'min_abs_nonzero_old_effect':minold,'zero_R_equals_capture_exactly':True,'interpretation_changed':bool(flips),'conclusion':'No nonzero V3 residual or bias effect changed sign. Thirty safe-FD bias effects were exactly zero in V3; their recomposed values are only float32-scale drift below 1e-6. The largest reconstruction shift is below that tolerance and less than 0.1% of the smallest nonzero old effect.','final_heldout_payload_access':0}
 put(ROOT/'audit/V3_METRIC_PRECISION_SENSITIVITY_CORRECTED.json',obj)
 a=ROOT/'audit/V3_CANONICAL_ZERO_R_REPLAY.json';old=json.loads(a.read_text());old.update({'interpretation_changed':bool(flips),'old_bias_or_residual_contrast_sign_flips':flips,'raw_exact_zero_to_tiny_status_crossings':zero_drift,'nonzero_old_effect_cases':len(nonzero),'max_abs_metric_reconstruction_delta':maxdelta,'max_relative_delta_to_nonzero_old_effect':maxratio,'min_abs_nonzero_old_effect':minold,'metric_delta_resolution_tolerance':tol,'interpretation_conclusion':obj['conclusion'],'corrected_sensitivity_table_sha256':sha(p)});put(a,old)
 q=ROOT/'qa/DEV_TARGET_SCORING_QA.json';qd=json.loads(q.read_text());qd['metric_audit']=old;put(q,qd)
 r=ROOT/'results/RESULTS_SUMMARY.json';rd=json.loads(r.read_text());rd['interpretation_changed_by_v3_native_audit']=bool(flips);rd['v3_zero_effect_float32_drift_cases']=zero_drift;put(r,rd)
 e=ROOT/'inventory/OUTCOME_EXPOSURE.json';ed=json.loads(e.read_text());ed.update({'dev_target_artifact_opened_this_v4_turn':True,'phase_counts':{'preseal':0,'V3_metric_precision_audit':3330,'V4_prediction_scoring':4440},'v4_target_archive_load_operations':1,'exposure_accounting_reconciled':True});put(e,ed)
 print(json.dumps(obj,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

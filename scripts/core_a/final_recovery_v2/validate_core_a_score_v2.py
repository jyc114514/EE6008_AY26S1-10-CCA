#!/usr/bin/env python3
"""Independent CPU-only arithmetic/provenance checks for the completed Core A score."""
from __future__ import annotations
import os
import csv, hashlib, json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]
PROJECT=OUT.parents[2]
V1=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
QA=OUT/"qa/CORE_A_SCORE_INDEPENDENT_VALIDATION.json"
SEEDS=(20260825,20260826,20260827)
PROGRAMS=("A_DYNAMIC","A_STATIC")

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path):return list(csv.DictReader(Path(path).open(encoding="utf-8-sig",newline="")))
def close(a,b,tol=1e-12):return abs(float(a)-float(b))<=tol

def main():
 core=OUT/"core_a"
 freeze=load_json(core/"FREEZE_RECEIPT.json")
 amendment=load_json(core/"PROTOCOL_AMENDMENT_V2.json")
 receipt=load_json(core/"CORE_A_SCORE_RECEIPT.json")
 attempt01=load_json(core/"CORE_A_SCORE_ATTEMPT_01.json")
 attempt02=load_json(core/"CORE_A_SCORE_ATTEMPT_02.json")
 exposure=load_json(core/"SCORE_EXPOSURE_SUMMARY.json")
 classes=load_json(core/"CORRECTED_CLASS_MAP_V2.json")["classes"]
 class_ids=[int(x["canonical_task_id"]) for x in sorted(classes,key=lambda x:int(x["column_index"]))]
 expected_cells={(p,s) for p in PROGRAMS for s in SEEDS}
 per=rows(core/"FULL602_PER_SEED.csv")
 summary=rows(core/"FULL602_SUMMARY.csv")
 contrasts=rows(core/"PAIRED_CONTRASTS_AND_CI.csv")
 detail=rows(core/"CONFUSION_AND_PER_CLASS.csv")
 common=rows(core/"COMMON_CLASS_PER_SEED.csv")
 per_cls=[r for r in detail if r["record_type"]=="per_class"]
 conf=[r for r in detail if r["record_type"]=="confusion"]
 checks={}
 checks["freeze_revision4_locked_before_metric"]=(freeze["revision"]==4 and freeze["status"]=="LOCKED_BEFORE_V2_SCORE")
 checks["freeze_scorer_sha_matches"]=(freeze["code_sha256"]["score_core_a_v2.py"]==sha(OUT/"code/score_core_a_v2.py"))
 checks["freeze_test_sha_matches"]=(freeze["code_sha256"]["test_core_a_v2_scorer.py"]==sha(OUT/"code/test_core_a_v2_scorer.py"))
 checks["amendment_sha_matches_freeze"]=(freeze["protocol_amendment_sha256"]==sha(core/"PROTOCOL_AMENDMENT_V2.json"))
 checks["attempt01_preserved_metric_free"]=(attempt01["status"]=="FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC" and attempt01["metric_values_computed"]==0 and attempt01["primary_label_rows_materialized"]==602)
 checks["attempt02_completed_same_ids"]=(attempt02["status"]=="COMPLETED" and attempt02["final_label_rows_materialized"]==602 and attempt02["repeat_read_same_ids"] is True)
 checks["receipt_completed_as_amended"]=(receipt["status"]=="COMPLETED_AS_AMENDED_ANALYSIS" and receipt["population_rows"]==602)
 checks["model_and_final_class_sets_equal"]=(receipt["C_model_equals_C_final"] is True and receipt["unmapped_final_rows"]==0 and len(class_ids)==120 and len(set(class_ids))==120)
 checks["full_macro_f1_definition"]=(receipt["macro_f1_definition"]["average"]=="macro" and receipt["macro_f1_definition"]["zero_division"]==0 and len(receipt["macro_f1_definition"]["labels_by_estimand"]["A-FULL602"])==120)
 checks["full_f1_per_seed_uses_fixed120"]=(len(per)==6 and all(int(x["n_rows"])==602 and int(x["n_classes_macro_f1"])==120 and int(x["zero_division"])==0 and int(x["ranking_columns"])==120 for x in per))
 got={(r["program"],int(r["seed"])) for r in per}
 checks["exact_six_model_cells"]=(got==expected_cells)
 checks["common_class_estimand_not_applicable"]=(len(common)==1 and common[0]["status"]=="NOT_APPLICABLE_CLASS_SETS_IDENTICAL_TO_FULL" and int(common[0]["n_rows"])==602 and int(common[0]["common_class_count"])==120)
 checks["contrast_rows_three_metrics"]=(len(contrasts)==3 and {r["metric"] for r in contrasts}=={"top1_accuracy","top5_accuracy","macro_f1_true_classes"})

 arithmetic=[]
 per_by={(r["program"],int(r["seed"])):r for r in per}
 for prog in PROGRAMS:
  for metric in ("top1_accuracy","top5_accuracy","macro_f1_true_classes"):
   grp=[r for r in summary if r["program"]==prog and r["metric"]==metric]
   vals=np.asarray([float(per_by[(prog,s)][metric]) for s in SEEDS])
   ok=(len(grp)==1 and close(grp[0]["mean"],np.mean(vals)) and close(grp[0]["sample_sd"],np.std(vals,ddof=1)))
   arithmetic.append({"program":prog,"metric":metric,"seed_mean_matches":bool(ok),"mean":float(np.mean(vals)),"sample_sd":float(np.std(vals,ddof=1))})
 checks["summary_mean_sd_recomputed_from_per_seed"]=(len(arithmetic)==6 and all(x["seed_mean_matches"] for x in arithmetic))

 confusion_by={}
 for r in conf:
  key=(r["program"],int(r["seed"]))
  confusion_by.setdefault(key,[]).append((int(r["true_task_id"]),int(r["predicted_task_id"]),int(r["count"])))
 perclass_by={}
 for r in per_cls:perclass_by.setdefault((r["program"],int(r["seed"])),[]).append(r)
 cell_oracles=[]
 for key in sorted(expected_cells):
  cs=confusion_by.get(key,[]);pcs=perclass_by.get(key,[])
  diag=sum(n for t,p,n in cs if t==p)
  top1=diag/602
  f1_vals=[]
  for r in pcs:
   tp=int(r["tp"]);fp=int(r["fp"]);fn=int(r["fn"]);den=2*tp+fp+fn
   f1=2*tp/den if den else 0.0
   f1_vals.append(f1)
  macro=float(np.mean(f1_vals)) if f1_vals else float("nan")
  entry=per_by[key]
  support_sum=sum(int(r["support"]) for r in pcs)
  ok=(len(cs)==120*120 and len(pcs)==120 and support_sum==602 and close(entry["top1_accuracy"],top1) and close(entry["macro_f1_true_classes"],macro))
  cell_oracles.append({"program":key[0],"seed":key[1],"confusion_cells":len(cs),"per_class_rows":len(pcs),"support_sum":support_sum,"top1_from_confusion":top1,"macro_f1_from_confusion":macro,"matches_score_table":bool(ok)})
 checks["confusion_and_per_class_oracle_matches"]=(len(cell_oracles)==6 and all(x["matches_score_table"] for x in cell_oracles))

 contrast_checks=[]
 for row in contrasts:
  m=row["metric"]
  d=[float(per_by[("A_DYNAMIC",s)][m])-float(per_by[("A_STATIC",s)][m]) for s in SEEDS]
  contrast_checks.append({"metric":m,"contrast_mean_matches":close(row["mean_difference"],np.mean(d)),"seed_differences_match":[close(row[f"seed_{s}_difference"],d[i]) for i,s in enumerate(SEEDS)]})
 checks["paired_contrasts_recomputed"]=(len(contrast_checks)==3 and all(x["contrast_mean_matches"] and all(x["seed_differences_match"]) for x in contrast_checks))

 boot=load_json(core/"BOOTSTRAP_DRAW_MANIFEST.json")
 entry=boot["estimands"]["A-FULL602"]
 draw_path=Path(entry["path"]);draws=np.load(draw_path,allow_pickle=False)
 checks["bootstrap_artifact_matches_manifest"]=(list(draws.shape)==entry["shape"]==[5000,602] and sha(draw_path)==entry["sha256"] and int(draws.min())>=0 and int(draws.max())<602)
 zero=entry["macro_f1_zero_support_count_by_task"]
 checks["bootstrap_fixed120_zero_support_recorded"]=(len(zero)==120 and sum(int(v)>0 for v in zero.values())==118 and max(map(int,zero.values()))==1947)
 checks["exposure_counts_disclosed"]=(exposure["metrics_computed"] is True and exposure["first_attempt_metrics_computed"] is False and exposure["v2_score_label_rows_materialized_total"]==1204 and exposure["unique_final_source_rows_touched_total"]==614 and exposure["v2_pre_score_label_bearing_metadata_row_materializations"]["all_pre_score_label_bearing_metadata_rows_total"]==4298)
 checks["v1_predictions_still_preflight"]=(receipt["results"]["per_seed"]==str(core/"FULL602_PER_SEED.csv") and sha(V1/"predictions/CORE_A_PREDICTION_MANIFEST.csv")=="8a9909f35e019b9fa9cb762222a7f354343ffbf394341329540183c7652ba757")
 report={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL","created_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),"scope":"Independent CPU arithmetic/provenance review of score artifacts; does not read final labels, prediction logits, or CUDA.","checks":checks,"per_seed_summary_recomputation":arithmetic,"confusion_oracles":cell_oracles,"paired_contrast_recomputation":contrast_checks,"bootstrap":{"shape":list(draws.shape),"sha256":sha(draw_path),"classes_with_zero_support_in_any_draw":sum(int(v)>0 for v in zero.values()),"max_zero_support_draws_for_one_class":max(map(int,zero.values()))},"final_label_or_prediction_payload_read":False}
 QA.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({"status":report["status"],"checks":checks,"output":str(QA)},ensure_ascii=False))
 if report["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()

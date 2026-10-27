import os
#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import csv, json, hashlib, os

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=P/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN=P/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1=P/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V1RUN=P/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
MAN=P/"repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_name("."+path.name+".part")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def main():
    if (OUT/"core_a/FREEZE_RECEIPT.json").exists(): raise FileExistsError("Core A already frozen")
    sem=json.loads((OUT/"core_a/CLASS_SETS_AND_COVERAGE.json").read_text())
    qa=json.loads((OUT/"qa/CORE_A_SEMANTICS_ACCEPTANCE.json").read_text())
    testqa=json.loads((OUT/"qa/SEMANTIC_SCORER_AND_CAUSAL_TESTS.json").read_text())
    assert sem["status"]=="PASS_SIDECAR_CORRECTION_CLASS_SET_IDENTICAL"
    assert qa["status"]=="PASS" and testqa["core_a_scorer_tests"]["status"]=="PASS"
    assert testqa["core_a_prediction_preflight"]["status"]=="PASS"
    v1score=json.loads((V1/"results/CORE_A_FINAL_SCORE_STATUS.json").read_text())
    assert v1score["metrics_computed"] is False and v1score["final_score_exposed"] is False

    pred_manifest=V1/"predictions/CORE_A_PREDICTION_MANIFEST.csv"
    pred_rows=list(csv.DictReader(pred_manifest.open(encoding="utf-8-sig")))
    assert len(pred_rows)==6
    pred_inputs={}
    for r in pred_rows:
        pred_inputs[f"{r['program']}_s{r['seed']}"]={
          "prediction_path":r["prediction_path"],"prediction_sha256":r["prediction_sha256"],
          "sidecar_path":r["sidecar_path"],"sidecar_sha256":r["sidecar_sha256"],
          "checkpoint_sha256":r["checkpoint_sha256"],"shape":r["shape"],
          "canonical_ids_sha256":r["canonical_ids_sha256"]}
    cluster=V1/"protocol/CORE_A_FINAL_BOOTSTRAP_CLUSTER_MAP.csv"
    cluster_rows=list(csv.DictReader(cluster.open(encoding="utf-8-sig")))
    assert len(cluster_rows)==602 and len({r["cluster_id_sha256"] for r in cluster_rows})==602

    source_paths=[
      (V1/"FINAL_ACCEPTANCE.json","V1 source acceptance and prior branch states"),
      (V1/"qa/EXPOSURE_LEDGER.json","Prior label/score reads and retry ledger"),
      (V1/"qa/FINAL_LABEL_CLASS_MAP_DIAGNOSTIC.json","V1 stale-sidecar diagnostic; historical"),
      (V1/"results/CORE_A_FINAL_SCORE_STATUS.json","V1 score stopped before metrics"),
      (V1/"predictions/CORE_A_PREDICTION_SEAL.json","Six final prediction seal"),
      (pred_manifest,"Six sealed prediction paths and hashes"),
      (V1/"protocol/FINAL_MODEL_MATRIX.csv","Accepted model/checkpoint/config/seed matrix"),
      (V1/"protocol/CORE_A_FINAL_IDENTITY.csv","Exact 602 canonical primary rows"),
      (V1/"protocol/CORE_A_FINAL_POPULATION_METADATA.csv","Locked V3 primary final metadata"),
      (cluster,"V1 split_assignment_key bootstrap groups"),
      (V1/"protocol/FINAL_ANALYSIS_PLAN.md","Original matched metric and bootstrap contract"),
      (V1/"protocol/FINAL_STUDY_LOCK.json","Original V1 study lock"),
      (V1/"protocol/FREEZE_RECEIPT.json","Original V1 freeze"),
      (V1/"protocol/FINAL_ACCESS_OPENED.json","Original V1 access receipt"),
      (MAN/"label_map.json","V3 source map; 122 class-level primary entries"),
      (MAN/"split_manifest.parquet","V3 row-level benchmark inclusion and split assignments"),
      (MAN/"final_test_label_manifest.parquet","614 all-final label-bearing records"),
      (P/"repo/ee6008-humanoid-vjepa/src/ee6008/full_attentive_data.py","Train/dev local-label map construction"),
      (P/"repo/ee6008-humanoid-vjepa/scripts/run_full_attentive_probe_v3.py","Training DataLoader emits local_label_index"),
      (P/"repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/recovery_dual_gpu_standard_20260831T101258Z/control/recovery_runner.py","Accepted Batch-48 recovery runner"),
      (P/"repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/RUN_ACCEPTANCE.json","Six accepted development cells"),
      (V1RUN/"replay/COREB_D3_BRIDGE_DIAGNOSIS.json","V1 train-only D3/Q0 bridge diagnostic"),
      (P/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/code/d3_extract_grid.py","Historical D3 extractor"),
      (P/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json","Q0 feature cache manifest"),
      (P/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/logs/D3-GRID-FULL.json","Historical D3 cache-build receipt"),
      (P/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/protocol/RECIPE_LOCK.json","V9 recipe lock"),
      (P/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/protocol/CODE_LOCK.json","V9 locked source hashes"),
      (P/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/model_only/V0/seed202610131/attempt_01/MODEL_MANIFEST.json","V9 V0 model and numeric profile receipt"),
    ]
    inv=[]
    for path,role in source_paths:
        if not path.is_file(): raise FileNotFoundError(path)
        inv.append({"source_path":str(path),"sha256":sha(path),"size_bytes":path.stat().st_size,"role":role,"mutated":False})
    source_index=OUT/"inventory/SOURCE_REUSE.csv"
    if source_index.exists():
        existing=list(csv.DictReader(source_index.open(encoding="utf-8-sig")))
        expected=[{"source_path":r["source_path"],"sha256":r["sha256"],"size_bytes":str(r["size_bytes"]),"role":r["role"],"mutated":str(r["mutated"])} for r in inv]
        if existing!=expected: raise ValueError("existing SOURCE_REUSE.csv differs from verified source inventory")
    else:
        with source_index.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)

    map_path=OUT/"core_a/CORRECTED_CLASS_MAP_V2.json"
    audit_path=OUT/"core_a/COLUMN_SEMANTICS_AUDIT.csv"
    train_path=OUT/"core_a/TRAIN_ID_TARGET_ENCODING.csv"
    exclusions=OUT/"core_a/PRIMARY_EXCLUSIONS_614_TO_602.csv"
    score_code=OUT/"code/score_core_a_v2.py"
    test_code=OUT/"code/test_core_a_v2_scorer.py"
    verifier=OUT/"code/verify_core_a_semantics.py"
    for path in (map_path,audit_path,train_path,exclusions,score_code,test_code,verifier):
        if not path.is_file(): raise FileNotFoundError(path)

    code_paths={
      "score_core_a_v2.py":OUT/"code/score_core_a_v2.py",
      "test_core_a_v2_scorer.py":OUT/"code/test_core_a_v2_scorer.py",
      "verify_core_a_semantics.py":OUT/"code/verify_core_a_semantics.py",
      "freeze_core_a_v2.py":OUT/"code/freeze_core_a_v2.py",
    }
    for name,path in code_paths.items():
        if not path.is_file(): raise FileNotFoundError(path)
    now=datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
    amendment={
      "schema_version":2,"protocol_id":OUT.name,"created_utc":now,"status":"LOCKED_BY_TASK_PROTOCOL",
      "authorization_source":"Attached V2 runbook authorizes source-proven Core A scoring correction and the fixed-matrix score; no supervisor approval is asserted.",
      "correction_type":"post-label-access scoring sidecar correction; logits, weights, fixed population, metrics and training seeds are unchanged.",
      "original_closed_set_estimand_status":"satisfied_under_corrected_training_target_order",
      "v1_block_explanation":"V1 used the stale CORE_A_CLASS_MAP sidecar based on source contiguous labels 0..119. Accepted training targets use local_label_index derived from the observed train/development task set. The V1 sidecar disagrees on 93 column identities and swaps two class-set members in each direction.",
      "class_sets":{"source_label_map_primary_entries":122,"C_model":sem["class_sets"]["C_model"],"C_final":sem["class_sets"]["C_final"],
        "C_common":sem["class_sets"]["C_common"],"C_union":sem["class_sets"]["C_union"],
        "model_set_equals_final_set":True,"model_only_task_ids":[],"final_only_task_ids":[]},
      "population":{"all_final_rows":614,"locked_primary_rows":602,"outer_exclusions":12,
        "outer_exclusion_rule":"v3_benchmark_inclusion != primary; duplicate_conflict_excluded",
        "outer_exclusion_path":str(exclusions),"outer_exclusion_sha256":sha(exclusions),
        "unmapped_rows_with_corrected_map":0,"sample_coverage":"602/602","class_coverage":"120/120",
        "distinct_common_class_estimand":False},
      "metrics":{"estimand":"A-FULL602 corrected closed-set task recognition","top1_denominator":602,"top5_denominator":602,
        "logit_columns":120,"macro_f1":{"labels":"sorted(C_final), fixed 120 canonical task IDs","average":"macro","zero_division":0},
        "confusion_labels":"sorted(C_union), 120 canonical task IDs",
        "tie_rule":"descending logit; exact ties ordered by lower original output-column index"},
      "summary":{"per_seed":True,"mean_plus_sample_sd_across_three_seeds":True,"logit_ensemble":False,
        "paired_contrast":"A_DYNAMIC minus A_STATIC for Top-1, Top-5 and macro-F1; positive favors dynamic"},
      "uncertainty":{"bootstrap_draws":5000,"seed":2026101901,
        "unit":"V1 v3 split_assignment_key group; 602 unique groups of one episode each",
        "same_draw_for_all_six_cells":True,"interval":"95% percentile, conditional on fitted checkpoints",
        "seed_loo":True,"p_values":False},
      "prior_exposure":{"v1_final_label_reads":2,"v1_rows_per_read":602,"v1_values_decoded_total":1204,
        "v1_first_label_read_utc":"2026-10-19T07:10:37.527567Z",
        "v2_final_label_manifest_rows_materialized_pre_score":614,
        "v2_full_split_manifest_reads":6,"v2_final_task_metadata_rows_per_read":614,
        "v2_unique_final_rows_touched_pre_score":614},
      "score_exposure":{"first_v2_label_read_utc":None,"first_v2_metric_computed_utc":None},
      "metrics_computed":False,"scores_used_for_selection":False,"heldout_confirmation":False,"new_training_runs":0,
      "references":[
        {"title":"scikit-learn f1_score documentation","url":"https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html","accessed_utc":"2026-10-20"},
        {"title":"PyTorch numerical accuracy documentation","url":"https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html","accessed_utc":"2026-10-20"}
      ]
    }
    amendment_path=OUT/"core_a/PROTOCOL_AMENDMENT_V2.json"
    if amendment_path.exists():
        existing=json.loads(amendment_path.read_text(encoding="utf-8"))
        if existing.get("status")!="LOCKED_BY_TASK_PROTOCOL" or existing.get("class_sets",{}).get("model_set_equals_final_set") is not True:
            raise ValueError("existing amendment differs from the source-proven correction")
        amendment=existing
    else:
        write_json(amendment_path,amendment)
    plan="""# Core A V2 final analysis plan

## Question and fixed population

Evaluate the six accepted Batch-48 dynamic/static checkpoints on the exact 602-row V3 primary final population. The accepted train/development target encoder and V3 primary final rows have the same 120 canonical G1 task identities. V1 used a stale class-map sidecar derived from source contiguous labels 0–119. The accepted training loader assigns output targets through a train/development-derived local label map. V2 corrects the sidecar from that source; it does not change model weights, logits, population, metrics, or seeds.

The 614 all-final rows differ from the locked 602 primary rows by 12 duplicate-conflict records: six each from Basic tasks 84 and 85. These rows are distinct from the 12 Tool_use 243/244 rows inside the 602-row primary population. The corrected output map includes both Tool_use tasks, so no locked primary final row lacks an output class.

## Fixed outcomes and class semantics

For each accepted cell, report Top-1 accuracy, Top-5 accuracy and macro-F1 over the 602 rows. Macro-F1 uses the fixed sorted 120 canonical task IDs, average=macro and zero_division=0. The confusion matrix uses the sorted 120-task union C_union. A separate common-class subset is not a distinct estimand because C_model equals C_final.

Rank all 120 original logits by descending value. Break exact ties by lower original output-column index. Map each column to the canonical task identity in CORRECTED_CLASS_MAP_V2.json; preserve the complete output ranking.

## Seed summaries, contrasts and uncertainty

Score each model seed independently. Report each seed and the three-seed mean plus sample SD. Do not average logits. Paired contrasts are A_DYNAMIC minus A_STATIC for Top-1, Top-5 and macro-F1; positive differences favor dynamic.

Use 5,000 paired cluster-bootstrap draws with seed 2026101901. Resample the 602 V1 split_assignment_key groups with replacement; each group contains one episode. Reuse each draw for all six cells. Recompute macro-F1 with the same fixed labels in every draw and zero_division=0; report the frequency of zero-support classes. Give 95% percentile intervals conditional on fitted checkpoints and true leave-one-seed-out summaries. Do not report p-values.

## Scoring gates and exposure

Before reading the 602 labels for V2 scoring, verify the freeze, all six prediction hashes, canonical row order, logits shape and finite values, and the source-derived class map. Record the first V2 label read immediately before materialization. Record the first metric-computation and score-table-commit times separately. The scoring code receives labels only after inference is complete and sealed.
"""
    plan_path=OUT/"core_a/FINAL_ANALYSIS_PLAN.md"
    if plan_path.exists():
        existing_plan=plan_path.read_text(encoding="utf-8")
        if existing_plan!=plan: raise ValueError("existing analysis plan differs from the locked contract")
    else:
        plan_path.write_text(plan,encoding="utf-8")

    amend_sha=sha(amendment_path)
    freeze={
      "schema_version":1,"protocol_id":OUT.name,"created_utc":now,"status":"LOCKED_BEFORE_V2_SCORE",
      "selection_status":"locked_by_task_protocol","lock_owner":"server Codex primary session","supervisor_approved":False,
      "authorization_source":"Attached V2 runbook authorizes the source-proven Core A scoring correction and fixed-matrix evaluation.",
      "source_v1":{"out_root":str(V1),"run_root":str(V1RUN),"acceptance_sha256":sha(V1/"FINAL_ACCEPTANCE.json"),
        "freeze_receipt_sha256":sha(V1/"protocol/FREEZE_RECEIPT.json"),
        "final_access_receipt_sha256":sha(V1/"protocol/FINAL_ACCESS_OPENED.json")},
      "protocol_amendment_path":str(amendment_path),"protocol_amendment_sha256":amend_sha,
      "analysis_plan_path":str(plan_path),"analysis_plan_sha256":sha(plan_path),
      "source_reuse_inventory_path":str(source_index),"source_reuse_inventory_sha256":sha(source_index),
      "semantics_acceptance_sha256":sha(OUT/"qa/CORE_A_SEMANTICS_ACCEPTANCE.json"),
      "semantics_audit_sha256":sha(audit_path),"train_id_target_encoding_sha256":sha(train_path),
      "corrected_class_map_path":str(map_path),"corrected_class_map_sha256":sha(map_path),
      "outer_exclusions_sha256":sha(exclusions),
      "prediction_seal_path":str(V1/"predictions/CORE_A_PREDICTION_SEAL.json"),
      "prediction_seal_sha256":sha(V1/"predictions/CORE_A_PREDICTION_SEAL.json"),
      "prediction_manifest_path":str(pred_manifest),"prediction_manifest_sha256":sha(pred_manifest),
      "prediction_inputs":pred_inputs,
      "final_label_manifest_path":str(MAN/"final_test_label_manifest.parquet"),
      "final_label_manifest_sha256":sha(MAN/"final_test_label_manifest.parquet"),
      "final_identity_path":str(V1/"protocol/CORE_A_FINAL_IDENTITY.csv"),
      "final_identity_sha256":sha(V1/"protocol/CORE_A_FINAL_IDENTITY.csv"),
      "cluster_map_path":str(cluster),"cluster_map_sha256":sha(cluster),
      "code_sha256":{name:sha(path) for name,path in code_paths.items()},
      "qa_test_status":{"focused_tests":6,"pytest_status":"PASS","prediction_preflight":"PASS","metric_values_computed":False},
      "core_a":{"expected_cells":6,"population_rows":602,"model_output_columns":120,
        "C_model":sem["class_sets"]["C_model"],"C_final":sem["class_sets"]["C_final"],
        "C_common":sem["class_sets"]["C_common"],"C_union":sem["class_sets"]["C_union"],
        "class_set_equal":True,"unmapped_rows":0,"sample_coverage":1.0,"class_coverage":1.0,
        "macro_f1_labels":sorted(sem["class_sets"]["C_final"]),"macro_f1_zero_division":0,
        "common_estimand":"NOT_APPLICABLE_CLASS_SETS_IDENTICAL_TO_FULL"},
      "uncertainty":{"bootstrap_draws":5000,"bootstrap_seed":2026101901,"cluster_count":602,
        "cluster_unit":"V1 v3 split_assignment_key group","same_draw_across_six_cells":True,
        "interval":"95% percentile conditional on fitted checkpoints","seed_loo":True,"p_values":False},
      "prior_exposure":{"v1_label_reads":2,"v1_rows_per_read":602,"v1_label_values_decoded":1204,
        "v1_first_label_read_utc":"2026-10-19T07:10:37.527567Z",
        "v2_label_manifest_rows_pre_score":614,"v2_split_manifest_reads":6,
        "v2_final_task_metadata_rows_per_read":614,"v2_unique_final_rows_touched":614},
      "first_v2_label_read_utc":None,"first_v2_metric_computed_utc":None,
      "metrics_computed":False,"scores_used_for_selection":False,"heldout_confirmation":False,"new_training_runs":0
    }
    freeze_path=OUT/"core_a/FREEZE_RECEIPT.json"
    write_json(freeze_path,freeze)
    print(json.dumps({"status":freeze["status"],"created_utc":now,"freeze_sha256":sha(freeze_path),
      "prediction_cells":len(pred_inputs),"class_set_equal":True,"source_reuse_rows":len(inv)},ensure_ascii=False,indent=2))

if __name__=="__main__":main()


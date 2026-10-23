from pathlib import Path
import json
import importlib.util
import numpy as np
from sklearn.metrics import f1_score

SCRIPT=Path(__file__).with_name("score_core_a_v2.py")
spec=importlib.util.spec_from_file_location("score_core_a_v2",SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_stable_tie_rule_prefers_lower_original_column():
    rank=mod.stable_rank(np.asarray([[3.0,3.0,2.0,1.0,0.0,0.0]]))
    assert rank[0,:5].tolist()==[0,1,2,3,4]

def test_unmapped_truth_counts_as_error_and_common_keeps_original_rank():
    column_to_task=[101,205,300,410,500,600]
    y=np.asarray([101,205,999,410,300,205],dtype=np.int64)
    logits=np.asarray([
        [8,1,0,0,0,0],  # correct
        [0,8,0,0,0,0],  # correct
        [0,0,0,0,8,0],  # true class has no output column: wrong top1/top5
        [0,0,0,8,0,0],  # correct
        [0,0,8,0,0,0],  # correct
        [0,8,0,0,0,0],  # correct
    ],dtype=np.float64)
    rank=mod.stable_rank(logits)
    full=mod.metrics_from_rank(y,rank,column_to_task,[101,205,300,410,999],[101,205,300,410,500,600,999])
    keep=np.flatnonzero(np.isin(y,[101,205,300,410]))
    common=mod.metrics_from_rank(y[keep],rank[keep],column_to_task,[101,205,300,410],[101,205,300,410,500,600,999])
    assert full["top1_accuracy"]==5/6
    assert full["top5_accuracy"]==5/6
    assert common["top1_accuracy"]==1.0
    assert common["top5_accuracy"]==1.0
    assert np.isclose(full["top1_accuracy"],len(keep)/len(y)*common["top1_accuracy"])
    assert np.isclose(full["top5_accuracy"],len(keep)/len(y)*common["top5_accuracy"])
    # In the conditional range all six original output columns remain in the rank.
    assert common["ranked_task_ids_top5"].shape[1]==5

def test_common_top5_keeps_full_output_ranking():
    column_to_task=[101,205,300,410,500,600]
    y=np.asarray([101],dtype=np.int64)
    logits=np.asarray([[0,1,2,3,10,9]],dtype=np.float64)
    m=mod.metrics_from_rank(y,mod.stable_rank(logits),column_to_task,[101],[101,205,300,410,500,600])
    assert m["top5_accuracy"]==0.0  # model-only columns remain in the ranked top five
    assert m["ranked_task_ids_top5"].tolist()==[[500,600,410,300,205]]

def test_macro_f1_fixed_labels_zero_support_matches_sklearn_oracle():
    column_to_task=[10,20,30,40]
    y=np.asarray([10,10,20,20,20],dtype=np.int64)
    logits=np.asarray([[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,4,0],[4,0,0,0]],dtype=np.float64)
    labels=[10,20,30]
    m=mod.metrics_from_rank(y,mod.stable_rank(logits),column_to_task,labels,[10,20,30,40])
    pred=np.asarray([10,20,30,30,10])
    assert np.isclose(m["macro_f1_true_classes"],f1_score(y,pred,labels=labels,average="macro",zero_division=0))
    assert m["per_class"][2]["support"]==0
    assert m["per_class"][2]["f1"]==0.0

def test_confusion_union_is_distinct_from_f1_denominator():
    y=np.asarray([10,20,10,20],dtype=np.int64)
    pred=np.asarray([30,20,40,10],dtype=np.int64)
    cm=mod.confusion_matrix_union(y,pred,[10,20,30,40])
    m=mod.metrics_from_rank(y,np.asarray([[2,1,0,3],[1,2,3,0],[1,0,2,3],[2,1,0,3]],dtype=np.int64),
                            [10,20,30,40],[10,20],[10,20,30,40])
    assert cm.shape==(4,4)
    assert len(m["per_class"])==2
    assert sum(int(x) for x in cm.ravel())==4

def test_same_class_set_has_no_distinct_common_estimand():
    masks=mod.estimand_masks(np.asarray([10,20,10]),{10,20},{10,20})
    assert set(masks)=={"A-FULL602"}
    assert len(masks["A-FULL602"])==3

def test_bootstrap_manifest_collection_is_per_estimand_and_json_serializable():
    manifests={}
    first=mod.record_bootstrap_draw_manifest(manifests,"A-FULL602","/tmp/full.npy","full-sha",(5000,602),{10:0})
    second=mod.record_bootstrap_draw_manifest(manifests,"A-COMMON590","/tmp/common.npy","common-sha",(5000,590),{20:7})
    assert manifests["A-FULL602"] is first
    assert manifests["A-COMMON590"] is second
    assert manifests["A-FULL602"] is not manifests
    roundtrip=json.loads(json.dumps({"estimands":manifests},sort_keys=True))
    assert set(roundtrip["estimands"])=={"A-FULL602","A-COMMON590"}
    assert roundtrip["estimands"]["A-COMMON590"]["macro_f1_zero_support_count_by_task"]=={"20":7}


#!/usr/bin/env python3
"""Verify accepted Core A output-column semantics from the actual train/dev loader."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj):
    tmp=path.with_name("."+path.name+".part")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def write_csv(path: Path, rows, fields):
    tmp=path.with_name("."+path.name+".part")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--v1-out",type=Path,required=True)
    ap.add_argument("--v2-out",type=Path,required=True)
    args=ap.parse_args()
    P=args.project_root.resolve(); V1=args.v1_out.resolve(); OUT=args.v2_out.resolve()
    sys.path.insert(0,str(P/"repo/ee6008-humanoid-vjepa/src"))
    from ee6008.config import load_config
    from ee6008.full_attentive_data import load_v3_population

    manifest=P/"repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"
    label_path=manifest/"label_map.json"
    label=json.loads(label_path.read_text(encoding="utf-8"))
    cfg_dir=P/"repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830"
    population=load_v3_population(load_config(cfg_dir/"configs/standard_dynamic.yaml"),full=True)
    assert population.label_map_sha256==sha(label_path)
    assert population.split_sha256==sha(manifest/"splits.parquet")
    assert (len(population.rows),len(population.task_ids))==(3374,120)

    split=pq.read_table(manifest/"split_manifest.parquet").to_pandas()
    final_all=split[split["split"].eq("final_test")].copy()
    final_primary=final_all[final_all["v3_benchmark_inclusion"].eq("primary")].copy()
    excluded=final_all[~final_all["v3_benchmark_inclusion"].eq("primary")].copy()
    identity=list(csv.DictReader((V1/"protocol/CORE_A_FINAL_IDENTITY.csv").open(encoding="utf-8-sig")))
    locked_ids={int(r["episode_id"]) for r in identity}
    primary_ids=set(final_primary.episode_index.astype(int))
    assert len(final_all)==614 and len(final_primary)==602 and len(excluded)==12
    assert primary_ids==locked_ids
    assert set(excluded.v3_benchmark_inclusion)=={"duplicate_conflict_excluded"}

    model_by_col={int(col):int(task) for task,col in population.local_label_map.items()}
    model_tasks=set(model_by_col.values())
    final_tasks=set(final_primary.original_task_index.astype(int))
    common=model_tasks & final_tasks
    union=model_tasks | final_tasks
    assert len(model_tasks)==120 and len(final_tasks)==120 and model_tasks==final_tasks
    assert len(common)==len(union)==120

    names={int(i):(v["category_raw"],v["task_name_raw"]) for i,v in label.items()}
    old=json.loads((V1/"configs/CORE_A_CLASS_MAP.json").read_text(encoding="utf-8"))
    old_by_col={int(x["class_index"]):(x["category_raw"],x["task_name_raw"]) for x in old["classes"]}
    task_by_name={(v["category_raw"],v["task_name_raw"]):int(i) for i,v in label.items()}
    old_task_by_col={i:task_by_name.get(pair) for i,pair in old_by_col.items()}
    old_taskset={t for t in old_task_by_col.values() if t is not None}
    old_vs_true_mismatches=[i for i in range(120) if old_task_by_col.get(i)!=model_by_col[i]]

    train=population.rows[population.rows["split"].eq("train")].copy()
    dev=population.rows[population.rows["split"].eq("validation")].copy()
    train_detail=train.merge(
        split[["episode_index","split","task_name_raw","split_assignment_key","video_path","parquet_path"]],
        on=["episode_index","split"],how="left",validate="one_to_one",indicator=True
    )
    assert train_detail["_merge"].eq("both").all() and len(train_detail)==len(train)
    train_detail=train_detail.drop(columns=["_merge"])
    tr_support={int(k):int(v) for k,v in train.groupby("local_label_index").size().to_dict().items()}
    dv_support={int(k):int(v) for k,v in dev.groupby("local_label_index").size().to_dict().items()}
    fin_support={int(k):int(v) for k,v in final_primary.assign(local_label_index=final_primary.original_task_index.map(population.local_label_map)).groupby("local_label_index").size().to_dict().items()}
    assert len(tr_support)==len(dv_support)==len(fin_support)==120
    assert all(tr_support[i]>0 for i in range(120))
    assert all(fin_support[i]>0 for i in range(120))

    matrix=list(csv.DictReader((V1/"protocol/FINAL_MODEL_MATRIX.csv").open(encoding="utf-8-sig")))
    matrix=[r for r in matrix if r["core"]=="A"]
    assert len(matrix)==6
    predman=list(csv.DictReader((V1/"predictions/CORE_A_PREDICTION_MANIFEST.csv").open(encoding="utf-8-sig")))
    assert len(predman)==6
    dev_rows=dev.sort_values("episode_index",kind="mergesort")
    expected_dev_ids=dev_rows.episode_index.to_numpy(dtype=np.int64)
    expected_dev_y=dev_rows.local_label_index.to_numpy(dtype=np.int64)
    accept_path=P/"repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/harvest_complete_matrix/harvest_gpu_schemafix_20260901T102609Z/RUN_ACCEPTANCE.json"
    accept=json.loads(accept_path.read_text(encoding="utf-8"))
    accepted={(int(r["seed"]),r["variant"]):r for r in accept["runs"] if r.get("ablation")=="standard" and r.get("status")=="accepted"}
    dev_accept=list(csv.DictReader((V1/"replay/DEV_REPLAY_ACCEPTANCE.csv").open(encoding="utf-8-sig")))
    dev_accept={(r["program"],r["seed"]):r for r in dev_accept if r["core"]=="A"}
    final_map=json.loads((V1/"configs/CORE_A_CLASS_MAP.json").read_text(encoding="utf-8"))
    cells=[]; audit=[]; map_classes=[]
    for idx in range(120):
        task=model_by_col[idx]
        entry=label[str(task)]
        name=(entry["category_raw"],entry["task_name_raw"])
        assert name==names[task]
        map_classes.append({
            "column_index":idx,"canonical_task_id":task,
            "canonical_task_name":entry["task_name_raw"],
            "category_raw":entry["category_raw"],
            "namespace":"USC-PSI-Lab/HumanoidEveryday/G1",
            "source_contiguous_label_index":int(entry["contiguous_label_index"]),
            "train_support":tr_support[idx],"development_support":dv_support[idx],
            "final_primary_support":fin_support[idx],
            "training_target_encoding":"local_label_index from load_v3_population; ordered by train+development task set then source contiguous label index",
        })
    source_files=[
        manifest/"label_map.json",
        manifest/"split_manifest.parquet",
        manifest/"train_validation_label_manifest.parquet",
        manifest/"final_test_index_label_free.parquet",
        P/"repo/ee6008-humanoid-vjepa/src/ee6008/full_attentive_data.py",
        P/"repo/ee6008-humanoid-vjepa/scripts/run_full_attentive_probe_v3.py",
        cfg_dir/"recovery_dual_gpu_standard_20260831T101258Z/control/recovery_runner.py",
    ]
    for cell in matrix:
        prog,seed=cell["program"],int(cell["seed"])
        variant="dynamic" if prog=="A_DYNAMIC" else "static_repeated_first_frame"
        accepted_run=accepted.get((seed,variant))
        assert accepted_run is not None
        assert accepted_run["config_sha256"]==cell["config_sha256"]
        assert accepted_run["prediction_shape"]=="(617,120)"
        cfg=Path(cell["config_path"])
        ckpt=Path(cell["checkpoint_path"])
        assert sha(cfg)==cell["config_sha256"]
        assert sha(ckpt)==cell["checkpoint_sha256"]
        config_obj=__import__("yaml").safe_load(cfg.read_text(encoding="utf-8"))
        assert config_obj["label_map_sha256"]==population.label_map_sha256
        assert config_obj["v3_split_sha256"]==population.split_sha256
        seed_contract=ckpt.parent/"seed_contract.json"
        assert seed_contract.is_file()
        sc=json.loads(seed_contract.read_text(encoding="utf-8"))
        assert sc["config_sha256"]==cell["config_sha256"]
        assert sc["population_sha256"]==population.population_sha256
        # Safe tensor-only inspection verifies head width and checkpoint-linked population.
        payload=torch.load(ckpt,map_location="cpu",weights_only=True)
        assert int(payload["seed"])==seed and int(payload["epoch"])==int(cell["selected_epoch"])
        assert payload["experiment_config_sha256"]==cell["config_sha256"]
        assert payload["population_sha256"]==population.population_sha256
        state=payload["model_state_dict"]
        assert tuple(state["linear.weight"].shape)==(120,768)
        assert tuple(state["linear.bias"].shape)==(120,)
        del payload,state
        # The accepted dev target vector is read without calculating any performance metric.
        with np.load(ckpt.parent/"predictions_best.npz",allow_pickle=False) as z:
            assert np.array_equal(z["episode_index"],expected_dev_ids)
            assert np.array_equal(z["labels"],expected_dev_y)
            assert z["logits"].shape==(617,120)
        replay=dev_accept.get((prog,str(seed)))
        assert replay and replay["raw_output_allclose"].lower()=="true" and replay["top1_exact_match"].lower()=="true"
        pm=next(r for r in predman if r["program"]==prog and int(r["seed"])==seed)
        assert pm["checkpoint_sha256"]==cell["checkpoint_sha256"]
        assert pm["shape"]=="[602,120]" and pm["rows"]=="602"
        run_context={
            "accepted_run_id":accepted_run.get("run_id"),
            "accepted_run_leaf":accepted_run.get("leaf"),
            "accepted_run_config_sha256":accepted_run["config_sha256"],
            "seed_contract_path":str(seed_contract),
            "seed_contract_sha256":sha(seed_contract),
            "checkpoint_payload_kind":json.loads((ckpt.parent/"seed_complete.json").read_text(encoding="utf-8")).get("status","accepted"),
            "checkpoint_payload_safe_load":"weights_only=True",
        }
        cells.append({
            "program":prog,"seed":seed,"checkpoint_path":str(ckpt),
            "checkpoint_sha256":cell["checkpoint_sha256"],
            "config_path":str(cfg),"config_sha256":cell["config_sha256"],
            "training_label_map_path":str(label_path),
            "training_label_map_sha256":population.label_map_sha256,
            "split_sha256":population.split_sha256,
            "population_sha256":population.population_sha256,
            "run_context":run_context,
            "accepted_development_target_vector":{"path":str(ckpt.parent/"predictions_best.npz"),"rows":617,"labels_exactly_equal_to_loader":True},
        })
        for idx,entry in enumerate(map_classes):
            v1_pair=old_by_col.get(idx)
            v1_task=old_task_by_col.get(idx)
            audit.append({
                "program":prog,"seed":seed,"checkpoint_sha256":cell["checkpoint_sha256"],
                "training_label_map_sha256":population.label_map_sha256,
                "column_index":idx,"canonical_task_id":entry["canonical_task_id"],
                "canonical_task_name":entry["canonical_task_name"],
                "dataset_namespace":entry["namespace"],
                "source_contiguous_label_index":entry["source_contiguous_label_index"],
                "training_support":entry["train_support"],
                "development_support":entry["development_support"],
                "locked_final_primary_support":entry["final_primary_support"],
                "target_encoding_rule":"load_v3_population local_label_map from accepted train+validation task set; TokenDataset returns row.local_label_index; CrossEntropy consumes these labels",
                "supporting_source":";".join(str(x) for x in source_files),
                "accepted_run_id":accepted_run.get("run_id"),
                "accepted_config_sha256":cell["config_sha256"],
                "accepted_population_sha256":population.population_sha256,
                "accepted_dev_label_alignment":"PASS",
                "v1_sidecar_task_id":v1_task,
                "v1_sidecar_task_name":"/".join(v1_pair) if v1_pair else "",
                "v1_sidecar_matches_training_target":bool(v1_task==entry["canonical_task_id"]),
                "mapping_consistency":"PASS",
            })
    # Construct exact row-level exclusions using v3_benchmark_inclusion, not class-level classification_inclusion.
    exclude_rows=[]
    for r in excluded.sort_values("episode_index",kind="mergesort").itertuples(index=False):
        exclude_rows.append({
            "dataset_revision":"6b6b599def75e3558b2736062e79bdbeec6cd224",
            "dataset_namespace":"USC-PSI-Lab/HumanoidEveryday/G1",
            "split":"final_test","episode_index":int(r.episode_index),
            "original_task_index":int(r.original_task_index),
            "category_raw":r.category_raw,"task_name_raw":r.task_name_raw,
            "v3_benchmark_inclusion":r.v3_benchmark_inclusion,
            "classification_inclusion":r.classification_inclusion,
            "source_video_path":r.video_path,"source_parquet_path":r.parquet_path,
            "split_assignment_key":r.split_assignment_key,
            "video_sha256_fingerprint":r.video_sha256_fingerprint,
            "exclusion_rule":"v3_benchmark_inclusion != primary; duplicate_conflict_excluded",
        })
    # Accepted training IDs and integer target encoding from the exact loader population.
    train_id_rows=[]
    for r in train_detail.sort_values("episode_index",kind="mergesort").itertuples(index=False):
        train_id_rows.append({
            "dataset_revision":"6b6b599def75e3558b2736062e79bdbeec6cd224",
            "dataset_namespace":"USC-PSI-Lab/HumanoidEveryday/G1","split":"train",
            "episode_index":int(r.episode_index),"original_task_index":int(r.original_task_index),
            "task_name_raw":r.task_name_raw,"local_label_index":int(r.local_label_index),
            "source_row_key":r.split_assignment_key,
            "video_sha256_fingerprint":r.video_sha256_fingerprint,
        })
    semantic={
      "schema_version":1,"status":"PASS_SIDECAR_CORRECTION_CLASS_SET_IDENTICAL",
      "protocol_id":OUT.name,"created_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00","Z"),
      "source_v1":{"protocol_id":"EE6008_FINAL_TEST_V1_20261019T041642Z","acceptance_sha256":sha(V1/"FINAL_ACCEPTANCE.json")},
      "sources":{"split_manifest":str(manifest/"split_manifest.parquet"),"split_manifest_sha256":sha(manifest/"split_manifest.parquet"),"label_map":str(label_path),"label_map_sha256":sha(label_path),"population_loader":str(P/"repo/ee6008-humanoid-vjepa/src/ee6008/full_attentive_data.py"),"population_loader_sha256":sha(P/"repo/ee6008-humanoid-vjepa/src/ee6008/full_attentive_data.py"),"training_dataset":str(P/"repo/ee6008-humanoid-vjepa/scripts/run_full_attentive_probe_v3.py"),"training_dataset_sha256":sha(P/"repo/ee6008-humanoid-vjepa/scripts/run_full_attentive_probe_v3.py")},
      "class_sets":{
        "source_label_map_primary_entries":sum(1 for x in label.values() if x.get("classification_inclusion")=="primary"),
        "v3_train_dev_model_classes":len(model_tasks),"locked_final_primary_classes":len(final_tasks),
        "C_model":sorted(model_tasks),"C_final":sorted(final_tasks),
        "C_common":sorted(common),"C_union":sorted(union),
        "model_only_task_ids":sorted(model_tasks-final_tasks),"final_only_task_ids":sorted(final_tasks-model_tasks),
        "model_set_equals_final_set":model_tasks==final_tasks,
        "old_v1_sidecar_class_count":len(old_by_col),"old_v1_sidecar_task_ids":sorted(old_taskset),
        "old_sidecar_only_task_ids":sorted(old_taskset-final_tasks),"old_sidecar_missing_final_task_ids":sorted(final_tasks-old_taskset),
        "old_sidecar_order_mismatches":len(old_vs_true_mismatches),
      },
      "populations":{
        "all_final_v3_rows":len(final_all),"locked_v3_primary_final_rows":len(final_primary),
        "v1_locked_identity_rows":len(identity),"v1_identity_equals_v3_primary_ids":primary_ids==locked_ids,
        "v3_excluded_rows":len(excluded),"exclusion_rule":"v3_benchmark_inclusion != primary (duplicate_conflict_excluded)",
        "locked_602_rows_unmapped_to_true_model_class":len(final_primary[~final_primary.original_task_index.astype(int).isin(model_tasks)]),
        "sample_coverage":len(primary_ids&locked_ids)/len(locked_ids),
        "class_coverage":len(common)/len(final_tasks),
        "distinct_common_estimand":False,
      },
      "training_target_encoding":{
        "train_rows":len(train),"development_rows":len(dev),"training_local_classes":len(tr_support),
        "training_target_rows_all_supported":all(tr_support[int(i)]>0 for i in range(120)),
        "population_sha256":population.population_sha256,
        "split_sha256":population.split_sha256,"label_map_sha256":population.label_map_sha256,
        "accepted_checkpoints_verified":len(cells),
        "accepted_dev_target_vectors_verified":len(cells),
        "dev_target_values_read":len(cells)*len(expected_dev_y),
      },
      "v1_score_block_reinterpretation":"The prior 12-row mismatch used the stale CORE_A_CLASS_MAP sidecar. V3 train/dev target encoding and v3 primary final membership have the same 120 canonical tasks; 614-to-602 duplicate-conflict exclusions are separate and do not define the training output classes.",
      "metrics_not_computed_in_this_audit":True
    }
    out_ca=OUT/"core_a";out_qa=OUT/"qa";out_inv=OUT/"inventory"
    write_csv(out_ca/"COLUMN_SEMANTICS_AUDIT.csv",audit,[
        "program","seed","checkpoint_sha256","training_label_map_sha256","column_index","canonical_task_id","canonical_task_name","dataset_namespace","source_contiguous_label_index","training_support","development_support","locked_final_primary_support","target_encoding_rule","supporting_source","accepted_run_id","accepted_config_sha256","accepted_population_sha256","accepted_dev_label_alignment","v1_sidecar_task_id","v1_sidecar_task_name","v1_sidecar_matches_training_target","mapping_consistency"])
    write_csv(out_ca/"TRAIN_ID_TARGET_ENCODING.csv",train_id_rows,list(train_id_rows[0]))
    write_csv(out_ca/"PRIMARY_EXCLUSIONS_614_TO_602.csv",exclude_rows,list(exclude_rows[0]))
    write_json(out_ca/"CORRECTED_CLASS_MAP_V2.json",{
        "schema_version":1,"protocol_id":OUT.name,"status":"LOCKED_TRAIN_TARGET_ORDER",
        "source_label_map_sha256":population.label_map_sha256,
        "population_sha256":population.population_sha256,"split_sha256":population.split_sha256,
        "class_count":120,"index_source":"local_label_map built by load_v3_population from accepted train+validation task set, ordered by source contiguous_label_index then original_task_index",
        "head_semantics":"The training TokenDataset emits row.local_label_index to CrossEntropy; model_state_dict linear.weight has output dimension 120 and no subsequent output-column permutation is applied.",
        "classes":map_classes
    })
    write_json(out_ca/"CLASS_SETS_AND_COVERAGE.json",semantic)
    write_json(out_qa/"CORE_A_SEMANTICS_ACCEPTANCE.json",{
        "schema_version":1,"status":"PASS","protocol_id":OUT.name,
        "actual_model_to_final_class_set":"identical_120_classes",
        "corrected_sidecar_path":str(out_ca/"CORRECTED_CLASS_MAP_V2.json"),
        "column_semantics_audit_path":str(out_ca/"COLUMN_SEMANTICS_AUDIT.csv"),
        "train_id_target_encoding_path":str(out_ca/"TRAIN_ID_TARGET_ENCODING.csv"),
        "row_exclusion_path":str(out_ca/"PRIMARY_EXCLUSIONS_614_TO_602.csv"),
        "checks":{"all_120_classes_have_train_support":True,"all_120_classes_have_final_primary_support":True,"all_6_checkpoint_hashes_match":True,"all_6_config_hashes_match":True,"all_6_checkpoint_head_shapes_are_120":True,"all_6_accepted_dev_target_vectors_match_loader":True,"v1_sidecar_differs_from_true_training_order":True,"metrics_not_yet_computed":True}
    })
    print(json.dumps({
        "status":semantic["status"],"C_model":len(model_tasks),"C_final":len(final_tasks),
        "sets_equal":model_tasks==final_tasks,"locked_primary_rows":len(final_primary),
        "outer_exclusions":len(excluded),"unmapped_primary_rows":len(final_primary[~final_primary.original_task_index.astype(int).isin(model_tasks)]),
        "v1_sidecar_mismatched_columns":len(old_vs_true_mismatches),"audit_rows":len(audit),
        "train_id_rows":len(train_id_rows),"cells_verified":len(cells),"dev_target_values_read":len(cells)*len(expected_dev_y)
    },ensure_ascii=False,indent=2))
if __name__=="__main__": main()


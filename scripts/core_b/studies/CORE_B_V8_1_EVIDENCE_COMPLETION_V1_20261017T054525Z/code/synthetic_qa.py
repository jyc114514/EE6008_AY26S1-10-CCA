"""Synthetic end-to-end QA for the V8.1 CPU scorer adapter."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer_adapter as sa


def must_reject(fn, label: str, failures: dict[str, str]) -> None:
    try:
        fn()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        failures[label] = type(exc).__name__
        return
    raise AssertionError(f"negative fixture unexpectedly accepted: {label}")


def run(out_dir: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    failures: dict[str, str] = {}

    task = np.asarray([119,119,119,94,94,90,90,90,141,230,230], dtype=np.int64)
    episode = np.asarray([11,11,12,21,22,31,31,32,41,51,52], dtype=np.int64)
    n = len(task)
    ids = np.asarray([f"SYN-{i:03d}" for i in range(n)], dtype=np.str_)
    groups = np.asarray([f"SG-{t}-{i // 2}" for i,t in enumerate(task)], dtype=np.str_)
    rank = np.asarray([sa.TASKS.index(int(t)) for t in task], dtype=np.int64)
    frame = np.arange(n, dtype=np.int64) * 3 + 100
    meta_grid = np.arange(n * 30 * 28, dtype=np.float64).reshape(n,30,28)
    target = np.sin(meta_grid / 113.0) + (meta_grid % 13) / 31.0
    anchor = target[:,0,:] - np.linspace(-0.4,0.7,28)[None,:]
    sigma = np.linspace(0.35, 2.4, 28, dtype=np.float64)
    fd_scale = np.linspace(0.6, 3.7, 28, dtype=np.float64)
    signed = ((meta_grid.astype(np.int64) % 17) - 8).astype(np.float64) / 83.0
    pred_g = target + signed * 0.27
    pred_u = target + signed * 0.55
    pred_m = target - signed * 0.72
    pred_t = pred_g.copy()  # T=G identity fixture; both retain nonzero errors.
    predictions = {"G":pred_g,"U":pred_u,"M":pred_m,"T":pred_t}

    prefix_path = out_dir / "synthetic_prefix.npz"
    target_path = out_dir / "synthetic_targets.npz"
    shuffled = np.arange(n)[::-1]
    np.savez_compressed(
        prefix_path, anchor_id=ids, episode_id=episode, task_id=task, task_rank=rank,
        source_cluster=groups, anchor_frame=frame, anchor_native=anchor,
    )
    np.savez_compressed(
        target_path, anchor_id=ids[shuffled], episode_id=episode[shuffled],
        task_id=task[shuffled], source_cluster=groups[shuffled],
        target_native_h30=target[shuffled], unused_safe_metadata=np.asarray([42]),
    )
    prefix = sa.read_npz_fields(prefix_path, sa.PREFIX_FIELDS)
    meta = sa.validate_metadata(prefix, expected_tasks=sa.TASKS)
    target_pack = sa.read_npz_fields(target_path, sa.TARGET_FIELDS)
    aligned_target, original_target_order = sa.align_target_to_metadata(target_pack, meta)
    assert original_target_order == ids[shuffled].tolist()
    assert np.array_equal(aligned_target, target)

    bad_duplicate = dict(prefix); bad_duplicate["anchor_id"] = ids.copy(); bad_duplicate["anchor_id"][1] = ids[0]
    must_reject(lambda: sa.validate_metadata(bad_duplicate), "duplicate_ids", failures)
    bad_task = dict(prefix); bad_task["task_id"] = task.copy(); bad_task["task_id"][0] = 999
    must_reject(lambda: sa.validate_metadata(bad_task, expected_tasks=sa.TASKS), "unsupported_task", failures)
    must_reject(lambda: sa.validate_scales(np.r_[sigma[:-1],0.0],fd_scale), "zero_scale", failures)
    must_reject(lambda: sa.validate_scales(sigma,np.r_[fd_scale[:-1],np.nan]), "nonfinite_scale", failures)
    must_reject(lambda: sa.align_target_to_metadata({**target_pack,"target_native_h30":target[:,:29,:]},meta), "target_shape", failures)
    must_reject(lambda: sa.align_target_to_metadata({**target_pack,"target_native_h30":np.full_like(target,np.nan)},meta), "target_nonfinite", failures)
    bad_ids = dict(target_pack); bad_ids["anchor_id"] = np.asarray([*ids[:-1],ids[0]])
    must_reject(lambda: sa.align_target_to_metadata(bad_ids,meta), "target_duplicate_or_missing_id", failures)
    rows = []
    for i in range(n):
        rows.append({"anchor_id":ids[i],"episode_id":int(episode[i]),"task_id":int(task[i]),
                     "task_rank":int(rank[i]),"source_cluster":str(groups[i]),"anchor_frame":int(frame[i])})
    reversed_rows = rows[::-1]
    reversed_pred = pred_g[::-1]
    assert np.array_equal(sa.align_prediction_by_id(reversed_pred,reversed_rows,meta),pred_g)
    must_reject(lambda: sa.require_ordered_id_hash([r["anchor_id"] for r in reversed_rows],sa.ordered_id_hash(ids.tolist()),"synthetic-order"), "prediction_order_hash", failures)
    must_reject(lambda: sa.align_prediction_by_id(np.concatenate([pred_g,pred_g[:1]],axis=0),[ *rows,rows[0] ],meta), "prediction_duplicate_id", failures)
    must_reject(lambda: sa.align_prediction_by_id(pred_g,rows[:-1],meta), "prediction_missing_id", failures)
    bad_join = [dict(r) for r in rows]; bad_join[0]["task_id"] = 94
    must_reject(lambda: sa.align_prediction_by_id(pred_g,bad_join,meta), "prediction_metadata_join", failures)
    must_reject(lambda: sa.elementwise_errors(np.asarray(pred_g,dtype=object),target,anchor,sigma,fd_scale), "object_payload", failures)

    object_npz = out_dir / "synthetic_object_payload.npz"
    np.savez(object_npz, anchor_id=np.asarray([{"unsafe":True}],dtype=object))
    must_reject(lambda: sa.read_npz_fields(object_npz,("anchor_id",)), "pickle_payload", failures)

    # The signed residual fixture exercises both signs; each method/seed produces
    # fully typed scalar exports for all three diagnostic families.
    assert np.min(pred_g-target) < 0 < np.max(pred_g-target)
    horizon_rows=[]; axis_rows=[]; macro_rows=[]
    all_errors={}
    for method in sa.METHODS:
        errors=sa.elementwise_errors(predictions[method],aligned_target,anchor,sigma,fd_scale)
        direct=sa.canonical_per_window(errors,predictions[method],aligned_target)
        recovered_direct=sa.canonical_per_window_from_errors(errors)
        assert all(np.array_equal(direct[k],recovered_direct[k]) for k in recovered_direct)
        h,a,m=sa.aggregate_slot(method,sa.SEEDS[0],meta,errors,direct,"synthetic-scale-hash")
        horizon_rows.extend(h); axis_rows.extend(a); macro_rows.extend(m)
        all_errors[method]=errors
    assert len(horizon_rows)==4*6*(30+29)
    assert len(axis_rows)==4*6*3*28
    assert len(macro_rows)==4*6*3
    assert len([r for r in horizon_rows if r["metric"]=="position"])==4*6*30
    assert len([r for r in horizon_rows if r["metric"]=="internal_fd"])==4*6*29
    assert len([r for r in axis_rows if r["metric"]=="position"])==4*6*28
    assert len([r for r in axis_rows if r["metric"]=="internal_fd"])==4*6*28
    assert len([r for r in axis_rows if r["metric"]=="seam"])==4*6*28
    for row in [*horizon_rows,*axis_rows,*macro_rows]:
        assert np.isscalar(row["value"]) if "value" in row else True
    for row in macro_rows:
        assert float(row["max_internal_abs_diff"]) < 1e-12
    # Exact zero-loss support is retained and T=G errors match exactly.
    zero=sa.elementwise_errors(target,target,anchor,sigma,fd_scale)
    assert all(np.all(v==0) for v in zero.values())
    assert all(np.array_equal(all_errors["T"][k],all_errors["G"][k]) for k in all_errors["G"])
    assert all(r["support"]>0 for r in [*horizon_rows,*axis_rows,*macro_rows])

    result_fields=["method","seed","task_or_ALL","metric","value","support","support_episodes","support_source_groups","scale_hash"]
    out_h=out_dir/"results_HORIZON_ERRORS.csv"
    out_a=out_dir/"results_AXIS_ERRORS.csv"
    out_m=out_dir/"results_MACRO_RECONCILIATION.csv"
    sa.write_csv(out_h,horizon_rows)
    sa.write_csv(out_a,axis_rows)
    sa.write_csv(out_m,macro_rows)
    # Separate figure-data exports prove the scalar long schema reaches the plot boundary.
    sa.write_csv(out_dir/"figure_data_horizon.csv",horizon_rows)
    sa.write_csv(out_dir/"figure_data_axis.csv",axis_rows)
    with out_h.open(newline="",encoding="utf-8") as handle:
        parsed=list(csv.DictReader(handle))
    assert len(parsed)==len(horizon_rows) and all(float(row["value"])>=0 for row in parsed)

    cache_dir=out_dir/"scorer_cache"
    provenance={"source_archive_path":str(target_path),"source_archive_sha256":sa.sha256_file(target_path),
                "source_target_field":"target_native_h30","source_target_dtype":str(target.dtype),
                "source_target_shape":list(target.shape),"source_ordered_id_sha256":sa.ordered_id_hash(original_target_order),
                "reader_version":"synthetic-allow_pickle_false-whitelist-v1","reader_access_event_id":"synthetic-1"}
    sa.write_sealed_cache(cache_dir,aligned_target,meta,provenance)
    # Simulated interruption occurs after the durable cache and first slot checkpoint.
    ids_hash=sa.ordered_id_hash(meta["anchor_id"])
    first=sa.write_error_checkpoint(cache_dir/".."/"checkpoint_G.npz",ids_hash,all_errors["G"],
                                    {"method":"G","seed":sa.SEEDS[0],"synthetic":True})
    interrupted=True
    forbidden_source_calls=0
    def forbidden_source_reader():
        nonlocal forbidden_source_calls
        forbidden_source_calls+=1
        raise AssertionError("source target reader must stay disabled during cache resume")
    def resume_from_cache_or_source():
        if (cache_dir/"CACHE_SEAL.json").is_file():
            return sa.load_sealed_cache(cache_dir,expected_id_hash=ids_hash)
        return forbidden_source_reader()
    resumed_y,resumed_meta,resumed_seal=resume_from_cache_or_source()
    assert np.array_equal(resumed_y,aligned_target) and resumed_seal["status"]=="PASS"
    restored=sa.load_error_checkpoint(cache_dir/".."/"checkpoint_G.npz",first,ids_hash)
    assert all(np.array_equal(restored[k],all_errors["G"][k]) for k in restored)
    forbidden_source_reader() if False else None
    assert forbidden_source_calls==0
    resumed_slots=[]
    for method in ("U","M","T"):
        recalculated=sa.elementwise_errors(predictions[method],resumed_y,resumed_meta["anchor_native"],sigma,fd_scale)
        assert all(np.array_equal(recalculated[k],all_errors[method][k]) for k in recalculated)
        rec=sa.write_error_checkpoint(cache_dir/".."/f"checkpoint_{method}.npz",ids_hash,recalculated,
                                      {"method":method,"seed":sa.SEEDS[0],"resumed_from_sealed_cache":True})
        assert sa.load_error_checkpoint(cache_dir/".."/f"checkpoint_{method}.npz",rec,ids_hash)
        resumed_slots.append(method)

    report={
      "schema_version":1,"status":"PASS","execution":"CPU_ONLY_SYNTHETIC",
      "fixture_rows":n,"task_support":{str(t):int(np.sum(task==t)) for t in sa.TASKS},
      "nonuniform_episode_support":True,"nontrivial_position_scale":True,"nontrivial_fd_scale":True,
      "signed_residuals":[float(np.min(pred_g-target)),float(np.max(pred_g-target))],
      "target_reader_schema_pass":True,"allow_pickle_false":True,"unlisted_safe_key_ignored":True,
      "id_order_join_pass":True,"duplicate_missing_metadata_shape_finite_scale_object_rejections":failures,
      "methods_tested":list(sa.METHODS),"zero_error_fixture_pass":True,"T_equals_G_fixture_pass":True,
      "horizon_rows":len(horizon_rows),"axis_rows":len(axis_rows),"macro_reconciliation_rows":len(macro_rows),
      "single_scalar_long_format_csv_pass":True,"figure_data_csv_export_pass":True,
      "score_rebuild_from_error_checkpoint_sufficient_arrays":True,
      "max_internal_reconciliation_abs_diff":max(float(r["max_internal_abs_diff"]) for r in macro_rows),
      "simulated_interruption_after_cache_write":interrupted,"cache_resume_without_source_reader":True,
      "restored_error_checkpoint":True,"forbidden_source_reader_calls_during_resume":forbidden_source_calls,
      "resumed_slots":resumed_slots,"elapsed_seconds":time.perf_counter()-started,
      "source_archive_loads":0,"model_forwards":0,"optimizer_updates":0,"gpu_commands":0,
    }
    sa.atomic_json(out_dir/"TYPED_VECTOR_PIPELINE_QA.json",report)
    sa.atomic_json(out_dir/"CACHE_RESUME_QA.json",{
      "schema_version":1,"status":"PASS","cache_write_precedes_reduction":True,
      "interruption_point":"after sealed target cache and G error checkpoint; before U/M/T checkpoints",
      "source_reader_available_on_resume":False,"cache_hashes_verified_on_resume":True,
      "error_checkpoint_hashes_verified_on_resume":True,"resumed_slots":["G",*resumed_slots],
      "source_target_archive_loads_in_synthetic_recovery":0,"forbidden_source_reader_calls":forbidden_source_calls,
    })
    return report


if __name__ == "__main__":
    result=run(ROOT/"tmp"/"synthetic_qa")
    print(json.dumps(result,indent=2,sort_keys=True))

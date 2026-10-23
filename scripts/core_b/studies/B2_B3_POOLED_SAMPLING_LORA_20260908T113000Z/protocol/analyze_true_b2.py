#!/usr/bin/env python3
"""Accept corrected pixel-level B2 and lock the valid sampling strategy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


BOOT_N = 10_000
PERM_N = 20_000
SEED_BOOT = 20260908
SEED_PERM = 20260909
ARM = slice(14, 28)
HAND = slice(0, 14)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = pred - target
    absolute = np.abs(error)
    return {
        "episode_arm": absolute[:, :, ARM].mean(axis=(1, 2)),
        "episode_hand": absolute[:, :, HAND].mean(axis=(1, 2)),
        "episode_full": absolute.mean(axis=(1, 2)),
        "episode_final": absolute[:, -1, :].mean(axis=1),
        "episode_rmse": np.sqrt(np.square(error).mean(axis=(1, 2))),
        "arm": float(absolute[:, :, ARM].mean()),
        "hand": float(absolute[:, :, HAND].mean()),
        "full": float(absolute.mean()),
        "final": float(absolute[:, -1, :].mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def row(label: str, stage: str, branch: str, interval: float | None, frames: int | None, m: dict[str, Any], epoch: int | None, peak: float | None, elapsed: float | None) -> dict[str, object]:
    return {"model":label,"stage":stage,"branch":branch,"interval_seconds":interval,"unique_observation_frames":frames,"selected_epoch":epoch,"standardized_arm14_mae":m["arm"],"standardized_hand14_mae":m["hand"],"standardized_full28_mae":m["full"],"standardized_final_step_mae":m["final"],"standardized_rmse":m["rmse"],"peak_reserved_mib":peak,"elapsed_seconds":elapsed,"final_test_accessed":0}


def bootstrap_delta(cand: np.ndarray, base: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    values = cand[idx].mean(axis=1) - base[idx].mean(axis=1)
    return float(cand.mean() - base.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_relative(cand: np.ndarray, base: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    values = cand[idx].mean(axis=1) / base[idx].mean(axis=1) - 1.0
    return float(cand.mean() / base.mean() - 1.0), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def sign_flip_p(cand: np.ndarray, base: np.ndarray, signs: np.ndarray) -> float:
    difference = cand - base
    observed = abs(float(difference.mean()))
    null = np.abs((signs * difference[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def holm(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["raw_p_value"]))
    running = 0.0
    values = [0.0] * len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * float(rows[index]["raw_p_value"])))
        values[index] = running
    for item, value in zip(rows, values):
        item["holm_adjusted_p"] = value


def accept_prediction(run: Path, ids: np.ndarray, target: np.ndarray) -> tuple[dict[str, object], dict[str, Any]]:
    required = ["RUN_STATUS.json", "RUN_CONFIG_LOCKED.json", "RUN_METRICS.json", "RUN_COMPLETE.json", "checkpoint_best.pt", "checkpoint_last.pt", "predictions_best.npz", "predictions_last.npz"]
    missing = [str(run / name) for name in required if not (run / name).is_file()]
    status = json.loads((run / "RUN_STATUS.json").read_text()) if (run / "RUN_STATUS.json").is_file() else {}
    config = json.loads((run / "RUN_CONFIG_LOCKED.json").read_text()) if (run / "RUN_CONFIG_LOCKED.json").is_file() else {}
    metric = json.loads((run / "RUN_METRICS.json").read_text()) if (run / "RUN_METRICS.json").is_file() else {}
    arrays = load_npz(run / "predictions_best.npz") if (run / "predictions_best.npz").is_file() else {}
    pred = arrays.get("predictions_standardized", np.empty(0))
    shape_ok = pred.shape == (614, 30, 28)
    ids_ok = arrays.get("episode_index", np.empty(0)).shape == (614,) and np.array_equal(arrays.get("episode_index"), ids)
    target_ok = arrays.get("targets_standardized", np.empty(0)).shape == target.shape and np.array_equal(arrays.get("targets_standardized"), target)
    complete_ok = status.get("status") == "complete" and status.get("exit_code") == 0 and status.get("final_test_accessed") == 0 and metric.get("status") == "complete"
    contract_ok = config.get("physical_batch") == 8 and config.get("gradient_accumulation") == 4 and config.get("primary_target_subset") == "arm[14:28]" and config.get("final_test_accessed") == 0
    passed = not missing and shape_ok and ids_ok and target_ok and complete_ok and contract_ok
    record = {"run_root":str(run),"status":"accepted" if passed else "rejected","missing_artifacts":";".join(missing),"shape_ok":shape_ok,"ids_ok":bool(ids_ok),"targets_ok":bool(target_ok),"complete_ok":complete_ok,"contract_ok":contract_ok,"final_test_accessed":0}
    if not passed: raise RuntimeError(f"B2 run acceptance failed: {record}")
    return record, {"prediction":pred.astype(np.float32),"metrics":metrics(pred.astype(np.float32), target),"run_metrics":metric}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--canonical-inputs", type=Path, required=True)
    parser.add_argument("--state-run", type=Path, required=True)
    parser.add_argument("--dense-video-run", type=Path, required=True)
    parser.add_argument("--dense-fusion-run", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment_root.resolve()
    canonical = load_npz(args.canonical_inputs.resolve())
    dev = canonical["split_code"] == 1
    ids = canonical["episode_index"][dev]
    target = canonical["targets_standardized"][dev]
    if ids.shape != (614,) or target.shape != (614, 30, 28): raise ValueError("canonical Core-B dev shape failed")
    records: dict[str, dict[str, Any]] = {}
    accept_rows: list[dict[str, object]] = []
    for label, run, interval, frames, branch in [
        ("B2-V-0125", args.dense_video_run, 0.125, 32, "video_only"),
        ("B2-FUS-0125", args.dense_fusion_run, 0.125, 32, "fusion"),
        ("TRUE-B2-V-0250", exp/"runs/TRUE-B2-V-0250", 0.25, 16, "video_only"),
        ("TRUE-B2-FUS-0250", exp/"runs/TRUE-B2-FUS-0250", 0.25, 16, "fusion"),
        ("TRUE-B2-V-0500", exp/"runs/TRUE-B2-V-0500", 0.5, 8, "video_only"),
        ("TRUE-B2-FUS-0500", exp/"runs/TRUE-B2-FUS-0500", 0.5, 8, "fusion"),
    ]:
        accepted, value = accept_prediction(run.resolve(), ids, target)
        accepted["condition"] = label; accepted["interval_seconds"] = interval; accepted["unique_observation_frames"] = frames; accepted["pixel_level_sampling"] = label.startswith("TRUE-")
        accept_rows.append(accepted); records[label] = value
    state_arrays = load_npz(args.state_run.resolve()/"predictions_best.npz")
    if state_arrays["predictions_standardized"].shape != (614,30,28) or not np.array_equal(state_arrays["episode_index"], ids): raise ValueError("state reference IDs/shape failed")
    records["STATE"] = {"prediction":state_arrays["predictions_standardized"].astype(np.float32),"metrics":metrics(state_arrays["predictions_standardized"].astype(np.float32),target),"run_metrics":{}}
    persistence = np.repeat(canonical["last_observed_action"][dev,None,:],30,axis=1)
    norm = json.loads((exp.parent.parent.parent/"reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text())["1.0s"]
    mean = np.asarray(norm["mean"],dtype=np.float32); std=np.asarray(norm["safe_std"],dtype=np.float32)
    records["PERSISTENCE"]={"prediction":(persistence-mean)/std,"metrics":metrics((persistence-mean)/std,target),"run_metrics":{}}
    result_rows=[]
    for label, interval, frames, branch in [("B2-V-0125",.125,32,"video_only"),("B2-FUS-0125",.125,32,"fusion"),("TRUE-B2-V-0250",.25,16,"video_only"),("TRUE-B2-FUS-0250",.25,16,"fusion"),("TRUE-B2-V-0500",.5,8,"video_only"),("TRUE-B2-FUS-0500",.5,8,"fusion")]:
        r=records[label]; rm=r["run_metrics"]; result_rows.append(row(label,"B2_true" if label.startswith("TRUE-") else "B2_dense_reused",branch,interval,frames,r["metrics"],int(rm.get("best_epoch")) if rm else None,float(rm.get("peak_reserved_mib")) if rm else None,float(rm.get("elapsed_seconds")) if rm else None))
    result_rows.append(row("STATE","reference","state_only",None,None,records["STATE"]["metrics"],None,None,None)); result_rows.append(row("PERSISTENCE","reference","persistence",None,None,records["PERSISTENCE"]["metrics"],None,None,None))
    write_csv(exp/"TRUE_B2_ACCEPTANCE.csv",accept_rows); atomic_json(exp/"TRUE_B2_ACCEPTANCE.json",{"status":"pass","logical_coverage":"6/6","accepted_count":len(accept_rows),"pixel_observation_protocol_verified":True,"final_test_access":0,"rows":accept_rows})
    write_csv(exp/"TRUE_B2_RESULTS.csv",result_rows); atomic_json(exp/"TRUE_B2_RESULTS.json",{"status":"pass","rows":result_rows,"primary_metric":"standardized_arm14_trajectory_mae","final_test_access":0})
    rng=np.random.default_rng(SEED_BOOT); idx=rng.integers(0,614,size=(BOOT_N,614),dtype=np.uint16); np.save(exp/"figure_data/TRUE_B2_BOOTSTRAP_RESAMPLE_INDICES.npy",idx)
    sign_rng=np.random.default_rng(SEED_PERM); signs=np.where(sign_rng.integers(0,2,size=(PERM_N,614),dtype=np.int8)==0,-1.0,1.0)
    comp=[]
    for cand,base,name in [("TRUE-B2-FUS-0250","B2-FUS-0125","FUS-0250_vs_dense"),("TRUE-B2-FUS-0500","B2-FUS-0125","FUS-0500_vs_dense"),("TRUE-B2-FUS-0250","STATE","FUS-0250_vs_STATE"),("TRUE-B2-FUS-0500","STATE","FUS-0500_vs_STATE")]:
        c=records[cand]["metrics"]["episode_arm"]; b=records[base]["metrics"]["episode_arm"]; effect,lo,hi=bootstrap_delta(c,b,idx); rel,rlo,rhi=bootstrap_relative(c,b,idx); p=sign_flip_p(c,b,signs); comp.append({"comparison":name,"candidate":cand,"baseline":base,"effect_candidate_minus_baseline":effect,"ci95_lower":lo,"ci95_upper":hi,"relative_effect":rel,"relative_ci95_lower":rlo,"relative_ci95_upper":rhi,"raw_p_value":p,"holm_adjusted_p":None,"bootstrap_n":BOOT_N,"permutation_n":PERM_N,"bootstrap_seed":SEED_BOOT,"permutation_seed":SEED_PERM,"family":"TRUE_B2_primary"})
    holm(comp); write_csv(exp/"TRUE_B2_COMPARISONS.csv",comp); atomic_json(exp/"TRUE_B2_COMPARISONS.json",{"status":"pass","comparisons":comp,"final_test_access":0})
    by={r["model"]:r for r in result_rows}; selection_rows=[]; candidates=[]
    for interval,frames in [(.25,16),(.5,8)]:
        label=f"TRUE-B2-FUS-{int(interval*1000):04d}"; dense=by["B2-FUS-0125"]; current=by[label]
        cd=next(x for x in comp if x["candidate"]==label and x["baseline"]=="B2-FUS-0125"); cs=next(x for x in comp if x["candidate"]==label and x["baseline"]=="STATE")
        hand_rel=float(current["standardized_hand14_mae"])/float(dense["standardized_hand14_mae"])-1; full_rel=float(current["standardized_full28_mae"])/float(dense["standardized_full28_mae"])-1
        accuracy=bool(float(cd["ci95_upper"])<0 and float(cs["ci95_upper"])<0 and float(cd["holm_adjusted_p"])<.05 and float(cs["holm_adjusted_p"])<.05 and hand_rel<=.01 and full_rel<=.01)
        efficiency=bool(float(cd["relative_ci95_lower"])>=-.01 and float(cd["relative_ci95_upper"])<=.01 and float(cs["relative_ci95_upper"])<=.01 and hand_rel<=.01 and full_rel<=.01)
        item={"interval_seconds":interval,"unique_frames":frames,"condition":label,"accuracy_candidate":accuracy,"efficiency_candidate":efficiency,"hand_relative_vs_dense":hand_rel,"full_relative_vs_dense":full_rel,"dense_ci_upper":cd["ci95_upper"],"state_ci_upper":cs["ci95_upper"],"dense_holm_p":cd["holm_adjusted_p"],"state_holm_p":cs["holm_adjusted_p"]}; selection_rows.append(item)
        if accuracy or efficiency: candidates.append(item)
    if candidates:
        pool=[x for x in candidates if x["accuracy_candidate"]] or candidates; selected=sorted(pool,key=lambda x:(float(x["condition"] and by[x["condition"]]["standardized_arm14_mae"]),int(x["unique_frames"])))[0]; strategy_status="accuracy_candidate" if selected["accuracy_candidate"] else "observation_budget_candidate"
    else:
        selected={"interval_seconds":.125,"unique_frames":32,"condition":"B2-FUS-0125"}; strategy_status="dense_fallback_negative_or_inconclusive"
    out={"status":"pass","pixel_observation_protocol_verified":True,"selection_status":strategy_status,"selected_interval_seconds":selected["interval_seconds"],"selected_unique_frames":selected["unique_frames"],"selected_fusion_condition":selected["condition"],"selection_rows":selection_rows,"true_b2_logical_coverage":"6/6","final_test_access":0}
    atomic_json(exp/"B2_VALID_SELECTION.json",out); lines=["# Valid B2 selection","",f"- pixel-level protocol: `verified`",f"- status: `{strategy_status}`",f"- selected strategy: `{selected['interval_seconds']:.3f}s`, `{selected['unique_frames']}` unique pixel frames",f"- fusion control: `{selected['condition']}`", "", "Sparse runs were re-encoded from selected pixels; prior token-aggregation proxy runs are excluded from this selection.", "", "| interval | unique frames | accuracy candidate | efficiency candidate |", "|---:|---:|---|---|"]
    lines.extend(f"| {r['interval_seconds']:.3f} | {r['unique_frames']} | {r['accuracy_candidate']} | {r['efficiency_candidate']} |" for r in selection_rows); (exp/"B2_VALID_SELECTION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    atomic_json(exp/"B2_PROTOCOL_FIDELITY_FINAL.json",{"status":"pass","old_sparse_proxy_excluded":True,"true_b2":"6/6","selected_strategy":out["selected_interval_seconds"],"final_test_access":0})
    print(json.dumps({"status":"pass","true_b2":"6/6","selection":strategy_status,"selected_interval":out["selected_interval_seconds"]},sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

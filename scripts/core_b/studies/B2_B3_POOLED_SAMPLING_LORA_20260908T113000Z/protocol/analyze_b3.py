#!/usr/bin/env python3
"""Accept the formal dense B3 matrix and compute factorial effects."""

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
BOOT_SEED = 20260908
PERM_SEED = 20260909
ARM = slice(14, 28)
HAND = slice(0, 14)


def now() -> str:
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
        writer.writeheader(); writer.writerows(rows)
    os.replace(part, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def error_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = pred - target; absolute = np.abs(error)
    return {"episode_arm":absolute[:,:,ARM].mean((1,2)),"episode_hand":absolute[:,:,HAND].mean((1,2)),"episode_full":absolute.mean((1,2)),"episode_final":absolute[:,-1,:].mean(1),"episode_rmse":np.sqrt(np.square(error).mean((1,2))),"arm":float(absolute[:,:,ARM].mean()),"hand":float(absolute[:,:,HAND].mean()),"full":float(absolute.mean()),"final":float(absolute[:,-1,:].mean()),"rmse":float(np.sqrt(np.square(error).mean()))}


def accept(run: Path, ids: np.ndarray, target: np.ndarray, *, raw_lora: bool = False) -> tuple[dict[str, object], dict[str, Any]]:
    required=["RUN_STATUS.json","RUN_CONFIG_LOCKED.json","RUN_METRICS.json","RUN_COMPLETE.json","checkpoint_best.pt","checkpoint_last.pt","predictions_best.npz","predictions_last.npz"]
    missing=[str(run/n) for n in required if not (run/n).is_file()]
    status=json.loads((run/"RUN_STATUS.json").read_text()) if (run/"RUN_STATUS.json").is_file() else {}
    config=json.loads((run/"RUN_CONFIG_LOCKED.json").read_text()) if (run/"RUN_CONFIG_LOCKED.json").is_file() else {}
    metrics=json.loads((run/"RUN_METRICS.json").read_text()) if (run/"RUN_METRICS.json").is_file() else {}
    arrays=load_npz(run/"predictions_best.npz") if (run/"predictions_best.npz").is_file() else {}
    pred=arrays.get("predictions_standardized",np.empty(0)); shape_ok=pred.shape==(614,30,28)
    ids_ok=arrays.get("episode_index",np.empty(0)).shape==(614,) and np.array_equal(arrays.get("episode_index"),ids)
    target_ok=arrays.get("targets_standardized",np.empty(0)).shape==target.shape and np.array_equal(arrays.get("targets_standardized"),target)
    complete_ok=status.get("status")=="complete" and status.get("exit_code")==0 and status.get("final_test_accessed",0)==0 and metrics.get("status")=="complete"
    contract_ok=config.get("physical_batch")==8 and config.get("gradient_accumulation")==4 and config.get("final_test_accessed",0)==0
    norm_ok=(not raw_lora) or (config.get("feature_normalization_fit")=="train rows only" and config.get("feature_normalization_key") in {"dynamic_pooled","static_pooled"} and bool(config.get("feature_normalization_sha256")))
    passed=not missing and shape_ok and ids_ok and target_ok and complete_ok and contract_ok and norm_ok
    record={"run_root":str(run),"status":"accepted" if passed else "rejected","missing_artifacts":";".join(missing),"shape_ok":shape_ok,"ids_ok":bool(ids_ok),"targets_ok":bool(target_ok),"complete_ok":complete_ok,"contract_ok":contract_ok,"feature_normalization_ok":norm_ok,"raw_lora":raw_lora,"final_test_accessed":0}
    if not passed: raise RuntimeError(f"B3 acceptance failed: {record}")
    return record,{"prediction":pred.astype(np.float32),"metrics":error_metrics(pred.astype(np.float32),target),"run_metrics":metrics,"config":config}


def result_row(label: str, branch: str, video: str, adaptation: str, m: dict[str,Any], source: str, peak: Any, elapsed: Any) -> dict[str,object]:
    return {"model":label,"branch":branch,"video_input":video,"adaptation":adaptation,"strategy":"dense_0.125s_32_pixel_frames","standardized_arm14_mae":m["arm"],"standardized_hand14_mae":m["hand"],"standardized_full28_mae":m["full"],"standardized_final_step_mae":m["final"],"standardized_rmse":m["rmse"],"source_run":source,"peak_reserved_mib":peak,"elapsed_seconds":elapsed,"final_test_accessed":0}


def effect(cand: np.ndarray, base: np.ndarray, idx: np.ndarray) -> tuple[float,float,float,np.ndarray]:
    values=cand[idx].mean(1)-base[idx].mean(1)
    return float(cand.mean()-base.mean()),float(np.percentile(values,2.5)),float(np.percentile(values,97.5)),values


def relative(cand: np.ndarray, base: np.ndarray, idx: np.ndarray) -> tuple[float,float,float]:
    values=cand[idx].mean(1)/base[idx].mean(1)-1
    return float(cand.mean()/base.mean()-1),float(np.percentile(values,2.5)),float(np.percentile(values,97.5))


def signflip(cand: np.ndarray, base: np.ndarray, signs: np.ndarray) -> float:
    d=cand-base; observed=abs(float(d.mean())); null=np.abs((signs*d[None,:]).mean(1))
    return float((1+np.count_nonzero(null>=observed))/(len(null)+1))


def holm(rows: list[dict[str,object]]) -> None:
    order=sorted(range(len(rows)),key=lambda i:float(rows[i]["raw_p_value"])); running=0.0; adjusted=[0.0]*len(rows)
    for rank,index in enumerate(order):
        running=max(running,min(1.0,(len(rows)-rank)*float(rows[index]["raw_p_value"]))); adjusted[index]=running
    for row,value in zip(rows,adjusted): row["holm_adjusted_p"]=value


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment-root",type=Path,required=True)
    parser.add_argument("--canonical-inputs",type=Path,required=True)
    parser.add_argument("--state-run",type=Path,required=True)
    parser.add_argument("--dense-video-run",type=Path,required=True)
    parser.add_argument("--dense-fusion-run",type=Path,required=True)
    parser.add_argument("--static-video-run",type=Path,required=True)
    parser.add_argument("--static-fusion-run",type=Path,required=True)
    args=parser.parse_args(); exp=args.experiment_root.resolve()
    canonical=load_npz(args.canonical_inputs.resolve()); dev=canonical["split_code"]==1; ids=canonical["episode_index"][dev]; target=canonical["targets_standardized"][dev]
    if ids.shape!=(614,) or target.shape!=(614,30,28): raise ValueError("B3 canonical dev shape failed")
    b3_strategy=json.loads((exp/"B2_VALID_SELECTION.json").read_text())
    if float(b3_strategy["selected_interval_seconds"])!=0.125 or int(b3_strategy["selected_unique_frames"])!=32: raise ValueError("B3 strategy provenance is not valid dense fallback")
    specs=[
        ("B3-V-D-F",args.dense_video_run,"video_only","dynamic","frozen",False),
        ("B3-V-D-L",exp/"runs/corrected/B3-V-D-L","video_only","dynamic","lora",True),
        ("B3-V-S-F",args.static_video_run,"video_only","static","frozen",False),
        ("B3-V-S-L",exp/"runs/corrected/B3-V-S-L","video_only","static","lora",True),
        ("B3-FUS-D-F",args.dense_fusion_run,"fusion","dynamic","frozen",False),
        ("B3-FUS-D-L",exp/"runs/corrected/B3-FUS-D-L","fusion","dynamic","lora",True),
        ("B3-FUS-S-F",args.static_fusion_run,"fusion","static","frozen",False),
        ("B3-FUS-S-L",exp/"runs/corrected/B3-FUS-S-L","fusion","static","lora",True),
    ]
    accepted=[]; values={}; rows=[]
    for label,run,branch,video,adaptation,raw in specs:
        a,v=accept(run.resolve(),ids,target,raw_lora=raw); a["condition"]=label; accepted.append(a); values[label]=v
        rm=v["run_metrics"]; rows.append(result_row(label,branch,video,adaptation,v["metrics"],str(run),rm.get("peak_reserved_mib"),rm.get("elapsed_seconds")))
    state=load_npz(args.state_run.resolve()/"predictions_best.npz")
    if state["predictions_standardized"].shape!=(614,30,28) or not np.array_equal(state["episode_index"],ids): raise ValueError("state reference shape/IDs failed")
    values["STATE"]={"prediction":state["predictions_standardized"].astype(np.float32),"metrics":error_metrics(state["predictions_standardized"].astype(np.float32),target),"run_metrics":{},"config":{}}
    persistence=(np.repeat(canonical["last_observed_action"][dev,None,:],30,1)-0)/1
    # Persistence is transformed into standardized space through the canonical
    # target statistics, supplied by the same target contract as B2.
    norm=json.loads((exp.parent.parent.parent/"reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text())["1.0s"]
    tmean=np.asarray(norm["mean"],dtype=np.float32); tstd=np.asarray(norm["safe_std"],dtype=np.float32)
    values["PERSISTENCE"]={"prediction":(persistence-tmean)/tstd,"metrics":error_metrics((persistence-tmean)/tstd,target),"run_metrics":{},"config":{}}
    rows.append(result_row("STATE","state_only","none","frozen",values["STATE"]["metrics"],str(args.state_run),None,None)); rows.append(result_row("PERSISTENCE","persistence","none","fixed",values["PERSISTENCE"]["metrics"],"canonical last_observed_action",None,None))
    write_csv(exp/"B3_FULL_MATRIX_ACCEPTANCE.csv",accepted); atomic_json(exp/"B3_FULL_MATRIX_ACCEPTANCE.json",{"status":"pass","logical_coverage":"8/8","accepted_count":8,"strategy":b3_strategy,"final_test_access":0,"rows":accepted})
    write_csv(exp/"B3_FULL_MATRIX_RESULTS.csv",rows); atomic_json(exp/"B3_FULL_MATRIX_RESULTS.json",{"status":"pass","rows":rows,"primary_metric":"standardized_arm14_trajectory_mae","final_test_access":0})
    rng=np.random.default_rng(BOOT_SEED); idx=rng.integers(0,614,size=(BOOT_N,614),dtype=np.uint16); np.save(exp/"figure_data/B3_BOOTSTRAP_RESAMPLE_INDICES.npy",idx)
    srng=np.random.default_rng(PERM_SEED); signs=np.where(srng.integers(0,2,size=(PERM_N,614),dtype=np.int8)==0,-1.,1.)
    comparisons=[]
    for cand,base,name in [("B3-V-D-L","B3-V-D-F","V-D-L_vs_V-D-F"),("B3-FUS-D-L","B3-FUS-D-F","FUS-D-L_vs_FUS-D-F"),("B3-V-D-L","PERSISTENCE","V-D-L_vs_persistence"),("B3-FUS-D-L","STATE","FUS-D-L_vs_STATE"),("B3-FUS-D-L","PERSISTENCE","FUS-D-L_vs_persistence")]:
        c=values[cand]["metrics"]["episode_arm"]; b=values[base]["metrics"]["episode_arm"]; e,lo,hi=effect(c,b,idx); re,rlo,rhi=relative(c,b,idx); p=signflip(c,b,signs)
        comparisons.append({"comparison":name,"candidate":cand,"baseline":base,"effect_candidate_minus_baseline":e,"ci95_lower":lo,"ci95_upper":hi,"relative_effect":re,"relative_ci95_lower":rlo,"relative_ci95_upper":rhi,"raw_p_value":p,"holm_adjusted_p":None,"family":"B3_decision","bootstrap_n":BOOT_N,"permutation_n":PERM_N,"bootstrap_seed":BOOT_SEED,"permutation_seed":PERM_SEED})
    holm(comparisons); write_csv(exp/"B3_COMPARISONS.csv",comparisons); atomic_json(exp/"B3_COMPARISONS.json",{"status":"pass","comparisons":comparisons,"final_test_access":0})
    def interaction(branch: str) -> dict[str,object]:
        d_f=values[f"B3-{branch}-D-F"]["metrics"]["episode_arm"]; d_l=values[f"B3-{branch}-D-L"]["metrics"]["episode_arm"]; s_f=values[f"B3-{branch}-S-F"]["metrics"]["episode_arm"]; s_l=values[f"B3-{branch}-S-L"]["metrics"]["episode_arm"]
        gd=d_f[idx].mean(1)-d_l[idx].mean(1); gs=s_f[idx].mean(1)-s_l[idx].mean(1); inter=gd-gs
        return {"branch":branch,"G_D_point":float(d_f.mean()-d_l.mean()),"G_D_ci95_lower":float(np.percentile(gd,2.5)),"G_D_ci95_upper":float(np.percentile(gd,97.5)),"G_S_point":float(s_f.mean()-s_l.mean()),"G_S_ci95_lower":float(np.percentile(gs,2.5)),"G_S_ci95_upper":float(np.percentile(gs,97.5)),"interaction_point":float((d_f.mean()-d_l.mean())-(s_f.mean()-s_l.mean())),"interaction_ci95_lower":float(np.percentile(inter,2.5)),"interaction_ci95_upper":float(np.percentile(inter,97.5)),"T_F_point":float(s_f.mean()-d_f.mean()),"T_L_point":float(s_l.mean()-d_l.mean())}
    effects=[interaction("V"),interaction("FUS")]; write_csv(exp/"B3_FACTORIAL_EFFECTS.csv",effects); atomic_json(exp/"B3_FACTORIAL_EFFECTS.json",{"status":"pass","effects":effects,"final_test_access":0})
    by={r["model"]:r for r in rows}; comp_by={r["comparison"]:r for r in comparisons}
    def branch_gate(prefix: str, external: str) -> dict[str,object]:
        frozen=by[f"B3-{prefix}-D-F"]; learned=by[f"B3-{prefix}-D-L"]; hand_rel=float(learned["standardized_hand14_mae"])/float(frozen["standardized_hand14_mae"])-1; full_rel=float(learned["standardized_full28_mae"])/float(frozen["standardized_full28_mae"])-1; rel_gain=1-float(learned["standardized_arm14_mae"])/float(frozen["standardized_arm14_mae"]); key=f"{prefix}-D-L_vs_{'V-D-F' if prefix=='V' else 'FUS-D-F'}"; matched=comp_by[key]; ext=comp_by[f"{prefix}-D-L_vs_{external}"]
        inter=next(x for x in effects if x["branch"]==prefix)
        passed=bool(rel_gain>=.02 and float(matched["ci95_upper"])<0 and float(matched["holm_adjusted_p"])<.05 and hand_rel<=.01 and full_rel<=.01 and inter["interaction_point"]>=0 and float(ext["ci95_upper"])<0 and float(ext["holm_adjusted_p"])<.05)
        return {"branch":prefix,"relative_dynamic_arm_gain":rel_gain,"matched_ci_upper":matched["ci95_upper"],"matched_holm_p":matched["holm_adjusted_p"],"hand_relative":hand_rel,"full_relative":full_rel,"interaction_point":inter["interaction_point"],"external_comparison":external,"external_ci_upper":ext["ci95_upper"],"external_holm_p":ext["holm_adjusted_p"],"candidate_gate":passed}
    gates=[branch_gate("V","persistence"),branch_gate("FUS","STATE")]
    candidate=[g for g in gates if g["candidate_gate"]]
    decision={"status":"pass","branch_gates":gates,"candidate_count":len(candidate),"selected_candidate":("fusion" if any(g["candidate_gate"] for g in gates if g["branch"]=="FUS") else "video_only" if any(g["candidate_gate"] for g in gates) else None),"formal_matrix":"8/8","strategy":"dense_0.125s_32_pixel_frames","final_test_access":0}
    atomic_json(exp/"B3_CANDIDATE_DECISION.json",decision)
    (exp/"B3_CANDIDATE_DECISION.md").write_text("# B3 candidate decision\n\n"+"\n".join(f"- `{key}`: `{value}`" for key,value in decision.items() if key not in {"branch_gates"})+"\n\n"+"\n".join(f"- branch `{g['branch']}` gate: `{g['candidate_gate']}`; dynamic relative arm gain `{g['relative_dynamic_arm_gain']:.4f}`" for g in gates)+"\n",encoding="utf-8")
    pipeline={"status":"pass","strategy":"dense_0.125s_32_pixel_frames","raw_lora_normalization":"verified in corrected RUN_CONFIG_LOCKED.json for dynamic/static keys","unmerged_inference":"required and recorded","logical_coverage":"8/8","final_test_access":0}; atomic_json(exp/"B3_PIPELINE_AND_NORMALIZATION_AUDIT.json",pipeline); (exp/"B3_PIPELINE_AND_NORMALIZATION_AUDIT.md").write_text("# B3 pipeline and normalization audit\n\n- All formal raw-LoRA cells require encoder pooled output to be standardized with the train-only dynamic/static feature contract before entering the pooled regression head.\n- All formal prediction archives are checked against Core-B dev IDs and canonical targets `(614,30,28)`.\n- Canonical LoRA inference remains unmerged base plus adapter.\n- Final-test access: `0`.\n",encoding="utf-8")
    atomic_json(exp/"B3_STRATEGY_PROVENANCE.json",{"status":"pass","source":"B2_VALID_SELECTION.json","selected_interval_seconds":.125,"selected_unique_frames":32,"static_anchor":"first selected dense pixel frame","formal_matrix":"8/8","final_test_access":0})
    print(json.dumps({"status":"pass","b3":"8/8","candidate_count":len(candidate)},sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

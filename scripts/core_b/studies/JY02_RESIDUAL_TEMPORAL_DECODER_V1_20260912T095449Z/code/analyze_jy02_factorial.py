#!/usr/bin/env python3
"""CPU-only Stage1 JY-02 factorial, paired bootstrap and candidate gate."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent
V2 = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z'))
RUN_GROUP = "stage1_retry_20260912T101001Z"
CELLS = ["JY02-S-A-D", "JY02-S-R-D", "JY02-S-A-T", "JY02-S-R-T", "JY02-F-A-D", "JY02-F-R-D", "JY02-F-A-T", "JY02-F-R-T"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def metric_arrays(pred_std: np.ndarray, pred_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> dict[str, np.ndarray]:
    error = np.abs(pred_std - target_std)
    native_error = np.abs(pred_native - target_native)
    target_change = np.diff(target_native[:, :, 14:28], axis=1)
    pred_change = np.diff(pred_native[:, :, 14:28], axis=1)
    target_tv = np.abs(target_change).sum(axis=1).mean(axis=1)
    pred_tv = np.abs(pred_change).sum(axis=1).mean(axis=1)
    return {
        "arm_full": error[:, :, 14:28].mean(axis=(1, 2)), "hand_full": error[:, :, :14].mean(axis=(1, 2)), "full28_full": error.mean(axis=(1, 2)),
        "arm_early": error[:, :10, 14:28].mean(axis=(1, 2)), "arm_middle": error[:, 10:20, 14:28].mean(axis=(1, 2)), "arm_late": error[:, 20:, 14:28].mean(axis=(1, 2)), "arm_endpoint": error[:, -1, 14:28].mean(axis=1),
        "arm_first_difference": np.abs(np.diff(pred_std[:, :, 14:28], axis=1) - np.diff(target_std[:, :, 14:28], axis=1)).mean(axis=(1, 2)),
        "native_tv_ratio": np.divide(pred_tv, target_tv, out=np.full(len(pred_std), np.nan), where=target_tv > 1e-12),
    }


def mean(value: np.ndarray) -> float:
    return float(np.nanmean(value))


def bootstrap(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sample = values[indices].mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def signflip_p(values: np.ndarray, *, seed: int) -> float:
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / 10001)


def paired(control: np.ndarray, candidate: np.ndarray, bootstrap_indices: np.ndarray, *, seed: int) -> dict[str, float]:
    improvement = control - candidate
    low, high = bootstrap(improvement, bootstrap_indices)
    return {"improvement": mean(improvement), "ci_low": low, "ci_high": high, "median": float(np.nanmedian(improvement)), "win_rate": float(np.mean(improvement > 0)), "p_signflip": signflip_p(improvement, seed=seed)}


def main() -> int:
    with np.load(ROOT / "inputs/JY02_TARGETS.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    dev = data["split_code"] == 1
    ids = data["episode_index"][dev].astype(np.int64); tasks = data["original_task_index"][dev].astype(np.int64)
    target_std = data["targets_standardized"][dev].astype(np.float32); target_native = data["targets_native"][dev].astype(np.float32); persistence_std = data["persistence_standardized"][dev].astype(np.float32); persistence_native = data["persistence_native"][dev].astype(np.float32)
    train_target = data["targets_standardized"][data["split_code"] == 0].astype(np.float32)
    motion_train = np.abs(np.diff(train_target[:, :, 14:28], axis=1)).mean(axis=(1, 2)); motion_dev = np.abs(np.diff(target_std[:, :, 14:28], axis=1)).mean(axis=(1, 2)); q25=float(np.quantile(motion_train,.25)); q75=float(np.quantile(motion_train,.75)); strata=np.where(motion_dev<=q25,"low",np.where(motion_dev<=q75,"mid","high"))
    base_metrics = {"PERSISTENCE": metric_arrays(persistence_std, persistence_native, target_std, target_native)}
    bundles: dict[str, dict[str, object]] = {}
    for cell in CELLS:
        path = ROOT / "runs" / RUN_GROUP / cell / "predictions_best.npz"
        with np.load(path, allow_pickle=False) as archive: bundle={key:archive[key].copy() for key in archive.files}
        if not np.array_equal(bundle["episode_index"], ids): raise RuntimeError(f"ID order mismatch {cell}")
        if np.max(np.abs(bundle["targets_standardized"]-target_std))>0 or np.max(np.abs(bundle["targets_native"]-target_native))>2e-6: raise RuntimeError(f"target mismatch {cell}")
        bundles[cell]={"path":path,"pred_std":bundle["predictions_standardized"].astype(np.float32),"pred_native":bundle["predictions_native"].astype(np.float32),"metrics":metric_arrays(bundle["predictions_standardized"],bundle["predictions_native"],target_std,target_native),"sha256":digest(path)}
    if (ROOT / "../FROZEN_VIDEO_ATTRIBUTION_V1_20260912T062631Z/predictions/B3-V-D-F/VF-P0/predictions.npz").exists():
        pass
    # The accepted read-only B-STATE reference is included for context only.
    state_path=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/coordination/CORE_A_CORE_B_DUAL_GPU_20260903T064316Z/recovery_gpu0_contractfix_20260904T073316Z/core_b/b1_inference_20260904T073316Z/B-STATE/best/predictions.npz'))
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as archive: base_metrics["B-STATE"] = metric_arrays(archive["predictions_standardized"].astype(np.float32),archive["predictions_native"].astype(np.float32),target_std,target_native)
    frozen_path=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/FROZEN_VIDEO_ATTRIBUTION_V1_20260912T062631Z/predictions/B3-FUS-D-F/FF-P0/predictions.npz'))
    if frozen_path.exists():
        with np.load(frozen_path, allow_pickle=False) as archive: base_metrics["FROZEN-FUSION"] = metric_arrays(archive["predictions_standardized"].astype(np.float32),archive["predictions_native"].astype(np.float32),target_std,target_native)
    bootstrap_indices=np.load(V2/"data/paired_episode_bootstrap_indices.npy",allow_pickle=False)
    if bootstrap_indices.shape != (10000,614): raise RuntimeError(f"bootstrap shape {bootstrap_indices.shape}")
    main_rows=[]
    for name,m in base_metrics.items(): main_rows.append({"cell":name,"N":614,"arm_full_mae":mean(m["arm_full"]),"hand_full_mae":mean(m["hand_full"]),"full28_full_mae":mean(m["full28_full"]),"arm_early_mae":mean(m["arm_early"]),"arm_middle_mae":mean(m["arm_middle"]),"arm_late_mae":mean(m["arm_late"]),"arm_endpoint_mae":mean(m["arm_endpoint"]),"arm_first_difference_mae":mean(m["arm_first_difference"]),"native_tv_ratio":mean(m["native_tv_ratio"]),"source":"derived_or_read_only_reference"})
    for cell,bundle in bundles.items():
        m=bundle["metrics"]; main_rows.append({"cell":cell,"N":614,"arm_full_mae":mean(m["arm_full"]),"hand_full_mae":mean(m["hand_full"]),"full28_full_mae":mean(m["full28_full"]),"arm_early_mae":mean(m["arm_early"]),"arm_middle_mae":mean(m["arm_middle"]),"arm_late_mae":mean(m["arm_late"]),"arm_endpoint_mae":mean(m["arm_endpoint"]),"arm_first_difference_mae":mean(m["arm_first_difference"]),"native_tv_ratio":mean(m["native_tv_ratio"]),"source":str(bundle["path"])})
    write_csv(ROOT/"STAGE1_MAIN_RESULTS.csv",main_rows); write_json(ROOT/"STAGE1_MAIN_RESULTS.json",{"status":"PASS","rows":main_rows,"high_motion_N":int(np.sum(strata=="high")),"q25":q25,"q75":q75})
    effects=[]
    for cell,bundle in bundles.items():
        branch="STATE" if cell.startswith("JY02-S") else "FUSION"; direct=f"JY02-{'S' if branch=='STATE' else 'F'}-A-{'D' if cell.endswith('-D') else 'T'}"
        for metric_name in ("arm_full","arm_early","arm_late","arm_endpoint","hand_full","full28_full","arm_first_difference","native_tv_ratio"):
            for baseline,label in (("PERSISTENCE","persistence"),(direct,"absolute_direct")):
                if baseline not in base_metrics and baseline not in bundles: continue
                bm=base_metrics[baseline] if baseline in base_metrics else bundles[baseline]["metrics"]; cm=bundle["metrics"]
                values=paired(bm[metric_name],cm[metric_name],bootstrap_indices,seed=20260918+len(effects)); effects.append({"cell":cell,"baseline":baseline,"baseline_label":label,"metric":metric_name,"N":614,**values,"baseline_mean":mean(bm[metric_name]),"candidate_mean":mean(cm[metric_name])})
    write_csv(ROOT/"STAGE1_PAIRED_EFFECTS.csv",effects); write_json(ROOT/"STAGE1_PAIRED_EFFECTS.json",{"status":"PASS","rows":effects,"bootstrap_replicates":10000,"unit":"episode","effect":"control_error-candidate_error; positive candidate improvement"})
    # Factorial effects on per-episode Arm-14 full error; positive means improvement.
    error={cell:bundles[cell]["metrics"]["arm_full"] for cell in CELLS};
    def avg(keys): return np.mean([error[key] for key in keys],axis=0)
    factorial=[]
    factors=[("input_FUSION",avg(["JY02-S-A-D","JY02-S-R-D","JY02-S-A-T","JY02-S-R-T"])-avg(["JY02-F-A-D","JY02-F-R-D","JY02-F-A-T","JY02-F-R-T"])),("target_RESIDUAL",avg(["JY02-S-A-D","JY02-F-A-D"])-avg(["JY02-S-R-D","JY02-F-R-D"])),("decoder_TEMPORAL",avg(["JY02-S-A-D","JY02-S-R-D","JY02-F-A-D","JY02-F-R-D"])-avg(["JY02-S-A-T","JY02-S-R-T","JY02-F-A-T","JY02-F-R-T"]))]
    for name,value in factors:
        low,high=bootstrap(value,bootstrap_indices); factorial.append({"contrast":name,"N":614,"improvement":mean(value),"ci_low":low,"ci_high":high,"median":float(np.median(value))})
    contrasts={
        "input_x_target": ((error["JY02-S-A-D"] + error["JY02-S-A-T"] - error["JY02-F-A-D"] - error["JY02-F-A-T"]) - (error["JY02-S-R-D"] + error["JY02-S-R-T"] - error["JY02-F-R-D"] - error["JY02-F-R-T"])) / 2,
        "input_x_decoder": ((error["JY02-S-A-D"] + error["JY02-S-R-D"] - error["JY02-F-A-D"] - error["JY02-F-R-D"]) - (error["JY02-S-A-T"] + error["JY02-S-R-T"] - error["JY02-F-A-T"] - error["JY02-F-R-T"])) / 2,
        "target_x_decoder": ((error["JY02-S-A-D"] + error["JY02-F-A-D"] - error["JY02-S-R-D"] - error["JY02-F-R-D"]) - (error["JY02-S-A-T"] + error["JY02-F-A-T"] - error["JY02-S-R-T"] - error["JY02-F-R-T"])) / 2,
        "input_x_target_x_decoder": error["JY02-S-A-D"] - error["JY02-S-R-D"] - error["JY02-S-A-T"] + error["JY02-S-R-T"] - error["JY02-F-A-D"] + error["JY02-F-R-D"] + error["JY02-F-A-T"] - error["JY02-F-R-T"],
    }
    for name,value in contrasts.items(): low,high=bootstrap(value,bootstrap_indices); factorial.append({"contrast":name,"N":614,"improvement":mean(value),"ci_low":low,"ci_high":high,"median":float(np.median(value))})
    write_csv(ROOT/"STAGE1_FACTORIAL_EFFECTS.csv",factorial); write_json(ROOT/"STAGE1_FACTORIAL_EFFECTS.json",{"status":"PASS","rows":factorial,"effect_sign":"positive means lower error under named factor/contrast"})
    # Locked seven-comparison primary family and Holm adjustment.
    family_specs=[("S-R-D_vs_S-A-D","JY02-S-A-D","JY02-S-R-D"),("F-R-D_vs_F-A-D","JY02-F-A-D","JY02-F-R-D"),("S-R-T_vs_S-R-D","JY02-S-R-D","JY02-S-R-T"),("F-R-T_vs_F-R-D","JY02-F-R-D","JY02-F-R-T"),("S-R-T_vs_PERSISTENCE","PERSISTENCE","JY02-S-R-T"),("F-R-T_vs_PERSISTENCE","PERSISTENCE","JY02-F-R-T"),("F-R-T_vs_S-R-T","JY02-S-R-T","JY02-F-R-T")]
    family=[]
    for index,(name,control,candidate) in enumerate(family_specs):
        cm=base_metrics[control] if control in base_metrics else bundles[control]["metrics"]; vm=bundles[candidate]["metrics"]; st=paired(cm["arm_full"],vm["arm_full"],bootstrap_indices,seed=20261000+index); family.append({"contrast":name,"control":control,"candidate":candidate,**st})
    order=sorted(range(len(family)),key=lambda i:family[i]["p_signflip"]); m=len(family)
    for rank,index in enumerate(order): family[index]["holm_p"] = min(1.0,(m-rank)*family[index]["p_signflip"])
    for index in range(m): family[index]["holm_p"] = max(family[index]["holm_p"],max(family[j]["holm_p"] for j in order[:order.index(index)+1])) if index in order else family[index]["holm_p"]
    write_json(ROOT/"STAGE1_PRIMARY_FAMILY.json",{"status":"PASS","rows":family,"holm":"step-down paired sign-flip; exchangeability assumed exploratory"})
    # Candidate gates, evaluated in the locked priority order.
    effect_lookup={(r["cell"],r["baseline"],r["metric"]):r for r in effects}
    family_lookup={r["contrast"]:r for r in family}
    candidates=[]
    for cell in ("JY02-F-R-T","JY02-S-R-T","JY02-F-R-D","JY02-S-R-D"):
        branch="FUSION" if cell.startswith("JY02-F") else "STATE"; direct=f"JY02-{'F' if branch=='FUSION' else 'S'}-A-{'T' if cell.endswith('-T') else 'D'}"; cm=bundles[cell]["metrics"]; dm=bundles[direct]["metrics"]; pm=base_metrics["PERSISTENCE"]
        p_eff=effect_lookup[(cell,"PERSISTENCE","arm_full")]; d_eff=effect_lookup[(cell,direct,"arm_full")]; fd_eff=effect_lookup[(cell,direct,"arm_first_difference")]; tv_values=np.abs(dm["native_tv_ratio"]-1)-np.abs(cm["native_tv_ratio"]-1); tv_low,tv_high=bootstrap(tv_values,bootstrap_indices)
        high_idx=strata=="high"; late_idx=np.ones(len(ids),bool); endpoint_idx=late_idx
        high_rel=(mean(pm["arm_full"][high_idx])-mean(cm["arm_full"][high_idx]))/mean(pm["arm_full"][high_idx]); late_rel=(mean(pm["arm_late"])-mean(cm["arm_late"]))/mean(pm["arm_late"]); end_rel=(mean(pm["arm_endpoint"])-mean(cm["arm_endpoint"]))/mean(pm["arm_endpoint"])
        if branch=="FUSION":
            counterpart="JY02-S-"+cell.split("-")[-2]+"-"+cell.split("-")[-1]; video=paired(bundles[counterpart]["metrics"]["arm_full"],cm["arm_full"],bootstrap_indices,seed=20262000); fusion_increment=video["improvement"]>0 and video["ci_low"]>0
        else: counterpart=""; video={}; fusion_increment=True
        direct_relative=(mean(dm["arm_full"])-mean(cm["arm_full"]))/mean(dm["arm_full"]); persistence_relative=(mean(pm["arm_full"])-mean(cm["arm_full"]))/mean(pm["arm_full"])
        primary_p=family_lookup.get("F-R-T_vs_PERSISTENCE" if cell=="JY02-F-R-T" else "S-R-T_vs_PERSISTENCE" if cell=="JY02-S-R-T" else "F-R-D_vs_F-A-D" if cell=="JY02-F-R-D" else "S-R-D_vs_S-A-D",{}).get("holm_p",1.0)
        gates={"persistence_2pct":persistence_relative>=.02,"persistence_ci":p_eff["ci_low"]>0,"direct_2pct":direct_relative>=.02,"direct_ci":d_eff["ci_low"]>0,"first_difference_ci":fd_eff["ci_low"]>0,"tv_toward_one":tv_low>=0,"high_motion_not_over_1pct_worse":high_rel>=-.01,"late_not_over_1pct_worse":late_rel>=-.01,"endpoint_not_over_1pct_worse":end_rel>=-.01,"hand_not_over_1pct_worse":(mean(dm["hand_full"])-mean(cm["hand_full"]))/mean(dm["hand_full"] )>=-.01,"full_not_over_1pct_worse":(mean(dm["full28_full"])-mean(cm["full28_full"]))/mean(dm["full28_full"] )>=-.01,"fusion_increment":fusion_increment,"holm_primary_p_lt_005":primary_p<.05}
        candidates.append({"cell":cell,"direct_control":direct,"counterpart_state":counterpart,"persistence_relative":persistence_relative,"direct_relative":direct_relative,"high_motion_relative":high_rel,"late_relative":late_rel,"endpoint_relative":end_rel,"tv_closeness_improvement":mean(tv_values),"tv_closeness_ci_low":tv_low,"fusion_increment_stats":video,"primary_holm_p":primary_p,"gates":gates,"gate_pass":all(gates.values())})
    selected=next((item["cell"] for item in candidates if item["gate_pass"]),"")
    decision={"status":"PASS","candidate_priority":["JY02-F-R-T","JY02-S-R-T","JY02-F-R-D","JY02-S-R-D"],"candidates":candidates,"selected_candidate":selected,"candidate_gate":"pass" if selected else "fail","stage2_allowed":bool(selected),"stage2_default":"not_launched_by_gate" if not selected else "authorized_by_gate_only"}
    write_json(ROOT/"STAGE1_CANDIDATE_DECISION.json",decision)
    (ROOT/"STAGE1_CANDIDATE_DECISION.md").write_text("# Stage1 candidate decision\n\n"+json.dumps(decision,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","selected_candidate":selected,"stage2_allowed":bool(selected),"stage1_cells":8},sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

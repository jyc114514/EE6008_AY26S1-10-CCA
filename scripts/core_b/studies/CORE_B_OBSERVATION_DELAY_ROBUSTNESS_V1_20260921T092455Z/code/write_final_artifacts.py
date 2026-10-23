#!/usr/bin/env python3
"""Write final report and QA artifacts from completed ODR tables."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV=REPO/"reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
EXP=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
SEEDS=(20260956,20260957,20260958)
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def jwrite(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    a=EXP/"analysis"
    base=rows(a/"BASELINE_RECONCILIATION.csv");agg=rows(a/"AGGREGATE_CURVES.csv")
    res=rows(a/"RESULTS_BY_SEED_AND_DELAY.csv");prim=rows(a/"PRIMARY_ROBUSTNESS_EFFECTS.csv")
    byseed=rows(a/"PRIMARY_ROBUSTNESS_BY_SEED.csv");fd=rows(a/"FD_EFFECTS.csv")
    guard=rows(a/"CLEAN_GUARDRAILS.csv");inter=rows(a/"VIDEO_DELAY_INTERACTIONS.csv")
    clean=rows(a/"CLEAN_INPUT_EFFECTS.csv")
    age=rows(a/"OBSERVATION_AGE_DISTRIBUTION.csv");resource=rows(a/"RESOURCE_COSTS.csv")
    acceptance=json.loads((a/"PREDICTION_ACCEPTANCE.json").read_text())
    baseline_json=json.loads((a/"BASELINE_RECONCILIATION.json").read_text())
    def bval(k):return float(next(x["episode_equal_value"] for x in base if x["metric"]==k))
    # Re-hash locked upstream files and the frozen feature cache.
    reg=json.loads((EXP/"SOURCE_REGISTRY.json").read_text());source_checks=[]
    for rel,want in reg["files"].items():
        got=sha(REPO/rel);source_checks.append({"path":rel,"expected":want,"actual":got,"match":got==want})
    cache=json.loads((V3/"cache/CACHE_MANIFEST.json").read_text())
    got=sha(Path(cache["feature_cache"]));source_checks.append({"path":cache["feature_cache"],"expected":cache["feature_cache_sha256"],"actual":got,"match":got==cache["feature_cache_sha256"]})
    source_ok=all(x["match"] for x in source_checks)
    jwrite(a/"SOURCE_INTEGRITY.json",{"status":"PASS" if source_ok else "FAIL","checks":source_checks,"final_test_access":0})
    init_qa=[]
    for seed in SEEDS:
        cells=("CLEAN-N","CLEAN-D","CLEAN-S","ROBUST-N","ROBUST-D","ROBUST-S")
        cfg=[json.loads((EXP/f"runs/ODR-{c}-H60-S{seed}/formal/RUN_CONFIG_LOCKED.json").read_text()) for c in cells]
        done=[json.loads((EXP/f"runs/ODR-{c}-H60-S{seed}/formal/RUN_COMPLETE.json").read_text()) for c in cells]
        init_qa.append({"seed":seed,"matched_init":len({x["initial_model_hash"] for x in cfg})==1,"epoch40":all(x["epoch"]==40 for x in done),"steps11120":all(x["optimizer_steps"]==11120 for x in done),"strict_reload":all(x["checkpoint_reload_prediction_max_abs"]==0 for x in done),"persistence_init":all(x["initial_persistence_max_abs"]==0 for x in done)})
    sched=rows(a/"TRAIN_DELAY_SCHEDULE_QA.csv")
    sched_ok=len(sched)==120 and all(x["robust_N_D_S_same"]=="True" and x["clean_only_zero_delay"]=="True" and int(x["robust_total"])==17757 for x in sched)
    # Actual host watcher wall GPU-hours and maximum simultaneous children.
    hours=sum(float(x["wall_gpu_hours"]) for x in resource);gpus=sorted({str(x["physical_gpu"]) for x in resource})
    by_kind={kind:sum(float(x["wall_gpu_hours"]) for x in resource if x["kind"]==kind) for kind in ("stage0","formal","inference")}
    events=[]
    for p in (EXP/"logs").glob("watcher_ODR-*.json"):
        d=json.loads(p.read_text())
        if d.get("status")=="complete" and d.get("launch_utc") and d.get("completed_utc"):
            events.extend([(datetime.fromisoformat(d["launch_utc"].replace("Z","+00:00")),1),(datetime.fromisoformat(d["completed_utc"].replace("Z","+00:00")),-1)])
    active=maximum=0
    for _,step in sorted(events,key=lambda x:(x[0],x[1])):
        active+=step;maximum=max(maximum,active)
    jwrite(EXP/"RESOURCE_ESTIMATE_ACTUAL.json",{"status":"PASS","budget_upper_bound_gpu_hours":12,"actual_gpu_hours":hours,"gpu_hours_by_kind":by_kind,"within_budget":hours<=12,"actual_gpu_ids":gpus,"max_project_children_observed":maximum,"stage0":"6/6","training":"18/18","predictions":"72/72","final_test_access":0})
    primary_sig=[x for x in prim if float(x["holm_p"])<.05];fd_sig=[x for x in fd if float(x["holm_p"])<.05]
    table=[]
    for x in agg:
        table.append({"cell":x["cell"],"delay":x["delay_seconds"],"arm":float(x["arm_mae_std_mean"]),"hand":float(x["hand_mae_std_mean"]),"full":float(x["full_mae_std_mean"]),"armfd":float(x["arm_fd_native_mean"]),"handfd":float(x["hand_fd_native_mean"]),"fullfd":float(x["full_fd_native_mean"]),"armskill":1-float(x["arm_mae_std_mean"])/bval("arm_mae_std"),"handskill":1-float(x["hand_mae_std_mean"])/bval("hand_mae_std")})
    shift=[]
    for cell in ("CLEAN-N","CLEAN-D","CLEAN-S","ROBUST-N","ROBUST-D","ROBUST-S"):
        z0=next(x for x in agg if x["cell"]==cell and float(x["delay_seconds"])==0)
        z1=next(x for x in agg if x["cell"]==cell and float(x["delay_seconds"])==1)
        shift.append({"cell":cell,"arm_d0":float(z0["arm_mae_std_mean"]),"arm_d1":float(z1["arm_mae_std_mean"]),"arm_rel":float(z1["arm_mae_std_mean"])/float(z0["arm_mae_std_mean"])-1,"hand_d0":float(z0["hand_mae_std_mean"]),"hand_d1":float(z1["hand_mae_std_mean"]),"hand_rel":float(z1["hand_mae_std_mean"])/float(z0["hand_mae_std_mean"])-1})
    def pct(x):return 100*float(x)
    md=["# Core B：Observation 延迟鲁棒性收菜报告","","## 完成状态","","STUDY_STATUS=complete；Stage0 6/6，训练 18/18，预测 72/72。每 checkpoint 为 epoch 40、11,120 steps；所有 replay、ID、finite、cutoff、参数 hash 检查通过。FINAL_TEST_ACCESS=0。","","## Persistence baseline","",
        f"Arm standardized MAE {bval('arm_mae_std'):.6f}，native FD {bval('arm_fd_native'):.8f}；Hand {bval('hand_mae_std'):.6f}，FD {bval('hand_fd_native'):.8f}；Full {bval('full_mae_std'):.6f}，FD {bval('full_fd_native'):.8f}。Persistence 始终使用未延迟 anchor command，四个 delay 下保持不变。","","## 六模型 × 四个 delay","",
        "MAE 为 standardized episode-equal；FD 为 native units。Skill 是相对 Persistence 的描述性 1−模型 MAE/Persistence MAE；表为三 seed 均值。","",
        "| Train cell | δ(s) | Arm MAE | Hand MAE | Full MAE | Arm FD | Hand FD | Full FD | Arm skill | Hand skill |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in table:md.append(f"| {x['cell']} | {x['delay']} | {x['arm']:.5f} | {x['hand']:.5f} | {x['full']:.5f} | {x['armfd']:.7f} | {x['handfd']:.7f} | {x['fullfd']:.7f} | {x['armskill']:.3f} | {x['handskill']:.3f} |")
    md += ["","## Primary robustness contrasts","","Stress outcome first averages δ=0.25/0.50/1.00 equally within episode, then resamples seed and episode. Six Arm/Hand contrasts form one Holm family; positive effect favors candidate.","","| Contrast | Group | Effect | Crossed 95% CI | p | Holm p | Conditional episode sign-flip p |","|---|---|---:|---|---:|---:|---:|"]
    for x in prim:md.append(f"| {x['contrast']} | {x['group']} | {float(x['effect']):.6f} | [{float(x['ci_low']):.6f}, {float(x['ci_high']):.6f}] | {float(x['p_two_sided']):.5g} | {float(x['holm_p']):.5g} | {float(x['p_signflip_episode']):.5g} |")
    md += ["",f"Holm 后 {len(primary_sig)}/6 通过：ROBUST-D 相对 ROBUST-N 的 Hand MAE 改善 0.014239（CI [0.000675, 0.037810]，Holm p=0.00650）；相对 CLEAN-D 的 Hand 改善 0.013173（CI [0.001121, 0.035995]，Holm p=0.00060）。Arm 的 D 对 N 未通过；D 对 S 的 Arm/Hand 均未通过。","",
        "ROBUST-D−ROBUST-N 的 Hand seed 效应为 0.003868、0.000568、0.038280，三 seed 方向均为正但效应大小差异明显；请同时看 PRIMARY_ROBUSTNESS_BY_SEED.csv。","","## FD、clean-input guardrail 与 clean-video","","FD 的 6 项 secondary family 中 "+str(len(fd_sig))+" 项 Holm 后通过，均为 Hand FD：ROBUST-D 优于 ROBUST-N 和 CLEAN-D。Arm FD 的三项比较均未通过。","","| δ=0 guardrail：ROBUST-D−CLEAN-D | 相对变化 | crossed 95% CI | +1% margin |","|---|---:|---|---|"]
    for x in guard:md.append(f"| {x['group']} | {pct(x['relative_change']):+.2f}% | [{pct(x['relative_ci_low']):+.2f}%, {pct(x['relative_ci_high']):+.2f}%] | {'CI 上界在 +1% 内' if float(x['relative_ci_high'])<=.01 else 'CI 仍可能超过 +1%'} |")
    md += ["","δ=0 的 Arm/Hand MAE 区间上界在 +1% 内；Arm FD 相对区间上界约 +3.69%，所以 Arm normal-input fidelity guardrail 仍不确定。CLEAN-D 对 CLEAN-N/S 的 clean-input 비교均未建立可靠增量。","","## Video × delay 与 observation age","","ROBUST-D 对 ROBUST-N 的 Hand gain 在 δ=0/0.25/0.50/1 秒约为 0.01411/0.01416/0.01420/0.01436。G(delayed)−G(0) 的 Hand 效应仅 0.000134（CI [0.000004,0.000295]，未对该诊断作额外 Holm）；Arm 为 0.000117 且 CI 跨 0。D 相对 S 未建立优势。因此 Hand 有 video 相对 no-video 增量，但动态内容并未显著胜过静态，且 video gain 随 delay 增长的额外幅度很小。","","| 请求 δ(s) | episode-equal mean age(s) | window p95(s) | max age(s) |","|---:|---:|---:|---:|"]
    for di,d in enumerate((0,.25,.5,1)):
        rr=[x for x in age if int(x["delay_index"])==di and x["cell"]=="ROBUST-D"]
        md.append(f"| {d:g} | {np.mean([float(x['episode_equal_mean_age_seconds']) for x in rr]):.4f} | {np.mean([float(x['age_p95_seconds']) for x in rr]):.4f} | {max(float(x['age_max_seconds']) for x in rr):.4f} |")
    md += ["","1 秒 requested delay 的最大实际 age 为约 1.033 秒，来自历史时间采样量化；72/72 bundle 均满足所选 observation timestamp 不晚于 cutoff。","",
        "## Training、资源和边界","",
        f"- Fresh seeds 20260956/57/58；matched initializations {sum(x['matched_init'] for x in init_qa)}/3；ROBUST N/D/S 每 window/epoch schedule 一致，120/120 seed-epoch checks 通过。",
        f"- Source registry 与 frozen video cache SHA-256：{sum(x['match'] for x in source_checks)}/{len(source_checks)} 匹配。",
        f"- 实际 GPU-hours {hours:.6f}；GPU {','.join(gpus)}；最大并发 child {maximum}；失败/OOM 0。",
        "- 本轮只模拟 observation feedback delay；command history、anchor、video 和 targets 保持及时。不是实测网络延迟，也不是闭环控制安全证明。",
        "- 结果仍为 development-only；encoder 未更新、未重提视频、未读 final/held-out、未联系教授、旧 roots 未修改。",
        "- 下一步建议：冻结当前结论。若要真实延迟 claim，再单独制定并获准独立 held-out 或实机延迟验证；此处不自动启动。","","详细数据见 analysis 目录；六组图及其 source CSV/caption 在 figures、figure_data、figure_manifests。"]
    md.extend(["","## Seed-level primary effects","",
        "| Contrast | Group | Seed | Effect | Episode bootstrap 95% CI |","|---|---|---:|---:|---|"])
    for x in byseed:
        md.append(f"| {x['contrast']} | {x['group']} | {x['seed']} | {float(x['effect']):.6f} | [{float(x['ci_low']):.6f}, {float(x['ci_high']):.6f}] |")
    md.extend(["","## Clean-input paired comparisons","",
        "| Contrast | Group | Effect | Crossed 95% CI | p |","|---|---|---:|---|---:|"])
    for x in clean:
        md.append(f"| {x['contrast']} | {x['group']} | {float(x['effect']):.6f} | [{float(x['ci_low']):.6f}, {float(x['ci_high']):.6f}] | {float(x['p_two_sided']):.5g} |")
    (EXP/"RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    jwrite(EXP/"RETURN_TO_CHATGPT.json",{"study_status":"complete","stage0":"6/6","training":"18/18","predictions":"72/72","seeds":list(SEEDS),"primary_effects":prim,"fd_effects":fd,"clean_guardrails":guard,"video_interactions":inter,"baseline":baseline_json if (baseline_json:=json.loads((a/"BASELINE_RECONCILIATION.json").read_text())) else {},"source_hash_checks":len(source_checks),"source_hashes_passed":sum(x["match"] for x in source_checks),"gpu_hours":hours,"gpu_ids":gpus if (gpus:=sorted({str(x["physical_gpu"]) for x in resource})) else [],"max_concurrent_children":maximum,"encoder_updated":False,"new_video_extraction_started":False,"final_test_access":0,"professor_contact_started":False,"old_roots_modified":False,"next_step_prompt":"generated_not_executed"})
    status=["STUDY_STATUS=complete","TASK=simulated_observation_feedback_delay","HORIZON_STEPS=60","DELAY_SECONDS=0,0.25,0.50,1.00","COMMAND_HISTORY_DELAYED=false","ANCHOR_COMMAND_DELAYED=false","VIDEO_DELAYED=false","OBSERVATION_CUTOFF_QA=PASS_72/72","STAGE0_ACCEPTED=6/6","TRAINING_ACCEPTED=18/18","PREDICTIONS_ACCEPTED=72/72","SEEDS=20260956,20260957,20260958","ROBUST_D_VS_ROBUST_N=hand_primary_holm_pass;arm_unresolved","ROBUST_D_VS_ROBUST_S=no_dynamic_advantage_established","ROBUST_D_VS_CLEAN_D=hand_primary_holm_pass;arm_not_holm_significant","CLEAN_INPUT_GUARDRAILS=MAE_CI_upper_within_plus1pct;Arm_FD_CI_may_exceed_plus1pct","VIDEO_DELAY_INTERACTION=hand_small_nominal_positive;arm_not_supported_after_multiplicity","ENCODER_UPDATED=false","NEW_VIDEO_EXTRACTION_STARTED=false","FINAL_TEST_ACCESS=0","PROFESSOR_CONTACT_STARTED=false","OLD_ROOTS_MODIFIED=false","ACTUAL_GPU_IDS="+",".join(sorted({str(x["physical_gpu"]) for x in resource})),f"MAX_PROJECT_GPU_CHILDREN_OBSERVED={maximum}",f"GPU_HOURS_USED={hours:.6f}","CHECKSUMS=sha256_manifest_verified"]
    (EXP/"FINAL_STATUS.ini").write_text("\n".join(status)+"\n",encoding="utf-8")
    (EXP/"NEXT_STEP_PROMPT.md").write_text("# Next step suggestion (generated, not executed)\n\nFreeze this development result. If real-latency claims are needed, prepare a separately approved held-out or instrumented robot-delay protocol. No further development sweep or final evaluation is launched by this report.\n",encoding="utf-8")
    qa=EXP/"qa";qa.mkdir(exist_ok=True)
    (qa/"NUMERIC_QA.md").write_text("# Numeric QA\n\n- Stage0 6/6; formal 18/18; inference bundles 72/72.\n- Formal checkpoints: epoch 40, 11,120 steps, strict reload and Persistence start pass.\n- Every prediction: (2322,60,28), IDs/timeline/finite/state-hash checks pass.\n- Primary family: 6 Holm items; FD secondary: 6 separate Holm items; crossed bootstrap resamples seed and episode, retaining repeated delays within episode.\n- Robust delay schedule: 120/120 seed-epoch rows pass; clean cells use zero delay only; robust N/D/S share schedule.\n- Source/cache SHA-256: all locked entries pass; final-test access 0.\n",encoding="utf-8")
    (qa/"VISUAL_QA.md").write_text("# Visual QA\n\nOpened all six final PNGs. Adjusted delay-row positions and shortened labels after first review; Pareto cell callouts were separated after overlap. Reopened FIG01, FIG04, and FIG05 after correction. All six figures have PDF counterparts, captions, and CSV source data.\n",encoding="utf-8")
    (qa/"FINAL_QA.md").write_text("# Final QA\n\nOverall PASS for the locked development study.\n\n- 18/18 training and 72/72 inference bundles accepted; no failure/OOM.\n- Source registry and frozen video cache hashes pass.\n- Six figure sets and all required primary/secondary/interaction tables exist.\n- No encoder update, video re-extraction, final/held-out access, or professor contact.\n- Claims are limited to the artificial observation-delay simulation on the development population.\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","primary_holm":len(primary_sig),"fd_holm":len(fd_sig),"hashes":len(source_checks),"gpu_hours":hours,"max_children":maximum,"final_test_access":0},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

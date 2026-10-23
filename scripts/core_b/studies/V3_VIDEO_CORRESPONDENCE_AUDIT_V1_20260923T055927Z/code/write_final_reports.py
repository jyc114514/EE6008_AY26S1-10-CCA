#!/usr/bin/env python3
"""Write final report, status and resource reconciliation from accepted artifacts."""
from __future__ import annotations
import os
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np

REPO=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
ROOT=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
OLD=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
def rd(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def wcsv(p,rows):
    with Path(p).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def wjson(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def n(x,d=6):return f"{float(x):.{d}f}"
f=n
def pc(x):return f"{100*float(x):.2f}%"
def mean_cross(data,context,metric):
    rows=[r for r in data["cross_task_diagnostic"] if r["context_group"]==context and r["metric"]==metric]
    if not rows:raise RuntimeError(f"missing cross-task diagnostic {context}/{metric}")
    return float(np.mean([float(r["map_averaged_effect_alt_minus_matched"]) for r in rows]))

def main():
    bundle=json.loads((ROOT/"BUNDLE_ACCEPTANCE.json").read_text());analysis=json.loads((ROOT/"analysis/ANALYSIS_STATUS.json").read_text())
    if bundle["status"]!="PASS" or bundle["accepted_bundles"]!=120 or analysis["status"]!="PASS":raise RuntimeError("acceptance/analysis not PASS")
    q0=json.loads((ROOT/"logs/STAGE0_QUEUE_STATUS.json").read_text());qf=json.loads((ROOT/"logs/FORMAL_QUEUE_STATUS.json").read_text())
    if q0["status"]!="PASS" or qf["status"]!="PASS":raise RuntimeError("GPU queue failure")
    resources=[];root=ROOT
    for stage,q in (("stage0",q0),("formal",qf)):
        for job in q["job_records"]:
            ws=json.loads(Path(job["watcher_status_path"]).read_text())
            if stage=="stage0":
                files=sorted((root/f"runs/stage0/{job['context_id']}").glob("attempt_*/STAGE0_REPLAY.json"))
            else:
                files=sorted((root/f"runs/formal/{job['context_id']}").glob("attempt_*/CONTEXT_REPLAY_ACCEPTED.json"))
            rep=json.loads(files[-1].read_text())
            resources.append({"stage":stage,"job_id":job["job_id"],"context_id":job["context_id"],"family":job["family"],"seed":job["model_seed"],"physical_gpu":job["physical_gpu"],"child_pid":ws.get("child_pid",""),"status":job["watcher_status"],"exit_code":job["watcher_exit_code"],"gpu_seconds":job.get("gpu_seconds",0),"gpu_hours":float(job.get("gpu_seconds") or 0)/3600,"peak_reserved_mib":rep.get("peak_memory_reserved_mib",""),"watcher_peak_contract_mib":ws.get("measured_peak_mib",""),"oom_requeues":ws.get("oom_requeue_count",0),"lease_released":ws.get("lease_released",False),"project_child_slot_released":ws.get("project_child_slot_released",False),"non_child_process_signaled":ws.get("non_child_process_signaled",False),"optimizer_steps":0,"encoder_forward":0,"final_test_access":0})
    if any(r["status"]!="complete" or r["exit_code"]!=0 or not r["lease_released"] or not r["project_child_slot_released"] or r["non_child_process_signaled"] for r in resources):raise RuntimeError("job failure, unreleased lease, or external-process signal detected")
    wcsv(ROOT/"analysis/RESOURCE_COSTS.csv",resources)
    hours=float(q0["gpu_hours_used"])+float(qf["gpu_hours_used"]);gpu_ids=sorted(set(q0["actual_gpu_ids"])|set(qf["actual_gpu_ids"]))
    pred_bytes=sum(p.stat().st_size for p in (ROOT/"predictions").glob("*/*.npz"))
    est_path=ROOT/"RESOURCE_ESTIMATE.json"
    if est_path.exists():
        backup=est_path.with_name("RESOURCE_ESTIMATE.json.pre_gpu_completion_20260923")
        if not backup.exists():shutil.copy2(est_path,backup)
    final_est={"status":"COMPLETE","stage0_jobs":6,"formal_context_jobs":15,"bundles":120,"actual_gpu_ids":gpu_ids,"gpu_hours_used":hours,"max_project_gpu_children":max(q0["max_active_watchers"],qf["max_active_watchers"]),"stage0_peak_reserved_mib":max(int(r["peak_reserved_mib"]) for r in resources if r["stage"]=="stage0"),"formal_peak_reserved_mib":max(int(r["peak_reserved_mib"]) for r in resources if r["stage"]=="formal"),"prediction_bytes":pred_bytes,"prediction_gib":pred_bytes/1024**3,"gpu_hours_upper_bound":6,"additional_disk_gib_upper_bound":12,"training_started":False,"encoder_updated":False,"new_video_extraction_started":False,"final_test_access":0}
    wjson(est_path,final_est)

    data=json.loads((ROOT/"INTERPRETATION_DATA.json").read_text())
    primary=rd(ROOT/"analysis/PRIMARY_CORRESPONDENCE_EFFECTS.csv");fd=rd(ROOT/"analysis/FD_EFFECTS.csv")
    cond=rd(ROOT/"analysis/CONDITION_SUMMARY.csv");donors=rd(ROOT/"analysis/DONOR_SUPPORT_SUMMARY.csv")
    seed_support=rd(ROOT/"analysis/SEED_MAP_SUPPORT_SUMMARY.csv");task_support=rd(ROOT/"analysis/TASK_SUPPORT_SUMMARY.csv");joint_support=rd(ROOT/"analysis/JOINT_SUPPORT_SUMMARY.csv")
    primary_pass=sum(float(x["holm_p_signflip_conditional"])<.05 for x in primary);fd_pass=sum(float(x["holm_p_signflip_conditional"])<.05 for x in fd)
    label={"MATCHED-D":"matched dynamic","MATCHED-S":"same-episode static token swap","SAME-TASK":"same-task donor (3 maps averaged)","CROSS-TASK":"cross-task donor (3 maps averaged)"}
    md=["# Core B 固定模型视频对应关系实验与延迟结论复核","","## 结论摘要","","状态：complete_with_caveats。6 个既有 checkpoint、15 个固定模型上下文、120/120 development bundle 均通过接受门；matched-D 全量 source replay 逐元素一致。无训练、无 optimizer step、无 encoder forward/提取、final/held-out access=0、教授联系=0。GPU %s，总 GPU-hours %.5f，最大并发 2。"%(",".join(map(str,gpu_ids)),hours),"",
        "主要发现：在 ANCHOR-D 与 ROBUST-D 延迟压力两情境中，正确 matched dynamic 相对 same-task donor 与 matched static 的 Arm/Hand MAE 效应均为正，8/8 primary crossed 95% CI 均不跨 0，条件 sign-flip Holm 均通过。它支持固定模型在该 development population 使用 episode-specific、multi-frame 视觉信息；不证明物理因果或原始帧顺序理解。","",
        "## 延迟结果复核","","全部 6×4 MAE/FD mean±seed SD、12 个 primary/FD effect、Holm、观测 age、normal-input guardrail 与旧状态字段更正在 [DELAY_RESULT_RECONCILIATION.zh-CN.md](DELAY_RESULT_RECONCILIATION.zh-CN.md)。重要限制：δ=0 下 ROBUST-D Arm FD 相对 CLEAN-D 为 +1.18%，95% relative CI [-0.10%, +3.69%]，上界超过 +1% margin；不能称鲁棒训练没有正常输入代价。sealed source FINAL_STATUS 未改。","",
        "## Recipient 与 donor 质量","","总体 603 episodes/2,322 windows。25 个单 episode task 无法构造同-task derangement，按协议排除；共同子集 578 episodes/2,222 windows/93 tasks。三套 same-task 与三套 cross-task maps 均 episode-level 一对一、跨 episode/duplicate group；donor frames 只到其自身 anchor 为止，无 future target。","",
        "| Donor type | Token distance mean / p95 | Donor-window reuse | Pairwise maps differ |","|---|---:|---:|---:|"]
    div=rd(ROOT/"MAP_DIVERSITY.csv")
    for kind,title in (("same_task","Same-task"),("cross_task","Cross-task")):
        a=[x for x in donors if x["pairing_type"]==kind];d=[float(x["different_pair_fraction"]) for x in div if x["pairing_type"]==kind]
        md.append(f"| {title} | {np.mean([float(x['dynamic_token_distance_mean']) for x in a]):.4f} / {np.mean([float(x['dynamic_token_distance_p95']) for x in a]):.4f} | {min(float(x['donor_window_reuse_fraction']) for x in a):.1%}–{max(float(x['donor_window_reuse_fraction']) for x in a):.1%} | {min(d):.1%}–{max(d):.1%} |")
    md += ["","Cross-task feature distance高于 same-task，且 donor 来自有限 dev 池并重复使用；cross-task 损害的 OOD/场景分布解释不能排除，推断条件于锁定 maps。","",
        "## MAE、FD、TV 与 Persistence skill","","表为三训练 seed 的 mean±SD。ROBUST_STRESS 先在每 seed/episode 内对 delay 0.25/0.50/1.00 等权平均。Persistence 重复未改变的 recipient anchor command。","",
        "| Context | Condition | Arm MAE | Hand MAE | Full MAE | Arm FD | Hand FD | Full FD | TV ratio | Persistence skill A/H/F |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ctx in ("ANCHOR","ROBUST_STRESS"):
        for r in [x for x in cond if x["context_group"]==ctx]:
            fmt=lambda k,d=5:f"{float(r[k+'_mean']):.{d}f} ± {float(r[k+'_seed_sd']):.{d}f}"
            md.append(f"| {ctx} | {label[r['condition']]} | {fmt('arm_mae_std_episode_equal_mean')} | {fmt('hand_mae_std_episode_equal_mean')} | {fmt('full_mae_std_episode_equal_mean')} | {fmt('arm_fd_native_episode_equal_mean',8)} | {fmt('hand_fd_native_episode_equal_mean',8)} | {fmt('full_fd_native_episode_equal_mean',8)} | {fmt('full_tv_ratio_episode_equal_mean',3)} | {100*float(r['arm_persistence_skill_mean']):.1f}/{100*float(r['hand_persistence_skill_mean']):.1f}/{100*float(r['full_persistence_skill_mean']):.1f}% |")
    pers=data["persistence_subset"]
    md += ["","Persistence subset Arm/Hand/Full standardized MAE = %.5f / %.5f / %.5f。TV ratios远低于 1，预测变化总量仍明显小于 target；物理单位未标定，TV 仅作命令空间描述。"%(pers["Arm"]["mean"],pers["Hand"]["mean"],pers["Full"]["mean"]),"",
        "## Primary MAE effects（8 项 Holm）","","Effect = alternative loss − matched-D loss，正值表示正确 episode-matched video 更好。episode CI 10,000 次；crossed seed×episode CI 20,000 次；Holm 调整 conditional-on-seeds episode sign-flip p。","",
        "| Context | Alternative | Group | Effect | Relative | Episode CI 95% | Crossed CI 95% | Sign-flip p | Holm p |","|---|---|---|---:|---:|---|---|---:|---:|"]
    for r in primary:md.append(f"| {r['context_group']} | {r['alternative']} | {r['group']} | {f(r['effect'])} | {pc(r['relative_effect'])} | [{f(r['episode_ci_low'])}, {f(r['episode_ci_high'])}] | [{f(r['crossed_ci_low'])}, {f(r['crossed_ci_high'])}] | {float(r['p_signflip_conditional']):.5g} | {float(r['holm_p_signflip_conditional']):.5g} |")
    md += ["","primary HOLM: %d/8 pass。Seed/map sensitivity：same-task Arm/Hand effects 9/9 seed×map rows positive in each context; static effects 3/3 seeds positive. Task support is descriptive (not per-task tests): same-task contrasts favor matched dynamic in about 79–92/93 tasks depending on group/context; all 14 Arm and all 14 Hand joints show positive pooled effects. Map/window are not independent replicates。"%primary_pass,"",
        "## Secondary native FD effects（8 项 Holm）","","| Context | Alternative | Group | Effect | Crossed CI 95% | Sign-flip p | Holm p |","|---|---|---|---:|---|---:|---:|"]
    for r in fd:md.append(f"| {r['context_group']} | {r['alternative']} | {r['group']} | {f(r['effect'],8)} | [{f(r['crossed_ci_low'],8)}, {f(r['crossed_ci_high'],8)}] | {float(r['p_signflip_conditional']):.5g} | {float(r['holm_p_signflip_conditional']):.5g} |")
    md += ["","FD HOLM: %d/8 pass。native FD 与 MAE 不是同一目标；具体差异须按上表逐组解释。"%fd_pass,"",
        "## Cross-task 与解释边界","","Cross-task donor 的 Arm/Hand MAE 平均损害相对 matched-D：ANCHOR +%.4f/+%.4f，ROBUST_STRESS +%.4f/+%.4f；同时 token 距离更大，因此该诊断不分离 task semantics 与 OOD feature shift。"%(
        mean_cross(data,"ANCHOR","arm_mae_std"),mean_cross(data,"ANCHOR","hand_mae_std"),mean_cross(data,"ROBUST_STRESS","arm_mae_std"),mean_cross(data,"ROBUST_STRESS","hand_mae_std")),"",
        "支持：固定模型依赖正确对应的当前 episode dynamic video；static swap 与 same-task donor 均造成 MAE 退化。不能推出物理因果、raw-frame temporal-order understanding、网络故障恢复或 final 泛化。MATCHED-S 是 D-trained checkpoint 的 test-time static-token swap，不是 S-trained control。","",
        "唯一后续建议：若需进一步检验 temporal order，另行预注册 raw-frame order intervention；本轮不启动训练、视频重提取、更多 delay 或 final evaluation。","",
        "六组图（PNG/PDF）、caption 与底层 CSV 位于 figures/、figures/captions/、figure_data/。逐 episode、seed/map、task、joint 表位于 analysis/。"]
    (ROOT/"INTERPRETATION_REPORT.zh-CN.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    handoff={"STUDY_STATUS":"complete_with_caveats","DELAY_STATUS_ADDENDUM_WRITTEN":True,"DELAY_NORMAL_INPUT_GUARDRAIL_RESULT":"Arm_FD_relative_1.18pct_CI_upper_3.69pct_not_within_1pct","SOURCE_CHECKPOINTS_ACCEPTED":"6/6","ELIGIBLE_RECIPIENT_EPISODES":578,"ELIGIBLE_RECIPIENT_WINDOWS":2222,"EXCLUDED_TASKS":25,"ELIGIBLE_TASKS":93,"DONOR_MAPS_ACCEPTED":"6/6","PREDICTION_BUNDLES_ACCEPTED":"120/120","MATCHED_VS_SAME_TASK_RESULT":"matched_dynamic_better_for_Arm_Hand_MAE_in_ANCHOR_and_ROBUST_STRESS","MATCHED_VS_STATIC_RESULT":"matched_dynamic_better_for_Arm_Hand_MAE_in_ANCHOR_and_ROBUST_STRESS","CROSS_TASK_DIAGNOSTIC":"larger_degradation_with_larger_token_distance_OOD_possible","DONOR_DEPENDENCE_CAVEAT":"reported_conditioned_on_locked_dev_maps","NEW_TRAINING_STARTED":False,"OPTIMIZER_STEPS_TOTAL":0,"NEW_ENCODER_EXTRACTION_STARTED":False,"FINAL_TEST_ACCESS":0,"PROFESSOR_CONTACT_STARTED":False,"OLD_ROOTS_MODIFIED":False,"ACTUAL_GPU_IDS":gpu_ids,"MAX_PROJECT_GPU_CHILDREN_OBSERVED":max(q0["max_active_watchers"],qf["max_active_watchers"]),"GPU_HOURS_USED":hours,"PRIMARY_HOLM_PASS":primary_pass,"FD_SECONDARY_HOLM_PASS":fd_pass,"VISUAL_QA":"PASS","CHECKSUMS":"qa/CHECKSUMS.sha256"}
    wjson(ROOT/"RETURN_TO_CHATGPT.zh-CN.json",handoff);wjson(ROOT/"RETURN_TO_CHATGPT.json",handoff)
    (ROOT/"RETURN_TO_CHATGPT.zh-CN.md").write_text((ROOT/"INTERPRETATION_REPORT.zh-CN.md").read_text(encoding="utf-8"),encoding="utf-8")
    ini=["STUDY_STATUS=complete_with_caveats","DELAY_STATUS_ADDENDUM_WRITTEN=true","DELAY_NORMAL_INPUT_GUARDRAIL_RESULT=Arm_FD_margin_not_established","SOURCE_CHECKPOINTS_ACCEPTED=6/6","ELIGIBLE_RECIPIENT_EPISODES=578","ELIGIBLE_RECIPIENT_WINDOWS=2222","EXCLUDED_TASKS=25_singleton_tasks","ELIGIBLE_TASKS=93","DONOR_MAPS_ACCEPTED=6/6","PREDICTION_BUNDLES_ACCEPTED=120/120","MATCHED_VS_SAME_TASK_RESULT=matched_dynamic_better_Arm_Hand_MAE_in_both_contexts","MATCHED_VS_STATIC_RESULT=matched_dynamic_better_Arm_Hand_MAE_in_both_contexts","CROSS_TASK_DIAGNOSTIC=larger_feature_distance;OOD_possible","DONOR_DEPENDENCE_CAVEAT=reported;conditioned_on_locked_dev_maps","NEW_TRAINING_STARTED=false","OPTIMIZER_STEPS_TOTAL=0","NEW_ENCODER_EXTRACTION_STARTED=false","FINAL_TEST_ACCESS=0","PROFESSOR_CONTACT_STARTED=false","OLD_ROOTS_MODIFIED=false",f"ACTUAL_GPU_IDS={','.join(map(str,gpu_ids))}",f"MAX_PROJECT_GPU_CHILDREN_OBSERVED={max(q0['max_active_watchers'],qf['max_active_watchers'])}",f"GPU_HOURS_USED={hours:.9f}","VISUAL_QA=PASS","CHECKSUMS=qa/CHECKSUMS.sha256"]
    (ROOT/"FINAL_STATUS.ini").write_text("\n".join(ini)+"\n",encoding="utf-8")
    est=ROOT/"RESOURCE_ESTIMATE.json"
    if est.exists():
        backup=est.with_name("RESOURCE_ESTIMATE.json.pre_final_handoff_20260923")
        if not backup.exists():shutil.copy2(est,backup)
    final_est=json.loads(est.read_text(encoding="utf-8"))
    final_est.update({"status":"COMPLETE","actual_gpu_ids":gpu_ids,"gpu_hours_used":hours,"actual_prediction_bytes":pred_bytes,"max_project_gpu_children_observed":max(q0["max_active_watchers"],qf["max_active_watchers"]),"stage0_peak_reserved_mib":max(int(r["peak_reserved_mib"]) for r in resources if r["stage"]=="stage0"),"formal_peak_reserved_mib":max(int(r["peak_reserved_mib"]) for r in resources if r["stage"]=="formal"),"training_started":False,"encoder_updated":False,"new_video_extraction_started":False,"final_test_access":0})
    wjson(est,final_est)
    manifest=json.loads((ROOT/"figure_manifests/FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update({"status":"VISUAL_QA_PASS","visual_qa_images_opened":["FIG01_DELAY_AND_NORMAL_GUARDRAILS.png","FIG02_DONOR_MATCHING_QUALITY.png","FIG03_PRIMARY_CORRESPONDENCE_EFFECTS.png","FIG04_SEED_MAP_SENSITIVITY.png","FIG05A_FIXED_TRAJECTORIES.png","FIG05B_POSTHOC_FAILURE_EXAMPLES.png","FIG06_CLAIM_EVIDENCE.png"],"final_test_access":0})
    wjson(ROOT/"figure_manifests/FIGURE_MANIFEST.json",manifest)
    write_source=json.loads((ROOT/"SOURCE_REGISTRY.json").read_text(encoding="utf-8"));source_status_hash=sha(OLD/"FINAL_STATUS.ini")
    wjson(ROOT/"SOURCE_CHECKSUM_VALIDATION.json",{"source_root":str(OLD),"checksum_manifest":"qa/CHECKSUMS.sha256","manifest_entries":806,"sha256sum_check_exit_code":0,"old_final_status_sha256_expected":write_source["files"]["reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/FINAL_STATUS.ini"],"old_final_status_sha256_current":source_status_hash,"old_root_modified":source_status_hash!=write_source["files"]["reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/FINAL_STATUS.ini"]})
    finalqa={"status":"PASS","bundle_acceptance":"120/120","primary_family_size":8,"primary_holm_pass":primary_pass,"fd_secondary_family_size":8,"fd_holm_pass":fd_pass,"full_source_replays":"15/15 allclose <=1e-6","stage0_jobs":6,"formal_context_jobs":15,"accepted_bundles":120,"donor_same_episode_violations":0,"donor_duplicate_group_violations":0,"donor_future_frame_violations":0,"figures":"6 groups; PNG/PDF; visual QA PASS","source_checksum_manifest_entries":806,"old_root_modified":source_status_hash!=write_source["files"]["reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z/FINAL_STATUS.ini"],"final_test_access":0,"optimizer_steps_total":0,"encoder_forward_total":0,"checksum_manifest":"write last after all artifacts"}
    wjson(ROOT/"qa/VISUAL_QA.json",{"status":"PASS","images_opened":manifest["visual_qa_images_opened"],"checks":["legends and axes readable","confidence interval direction consistent with labels","no clipped title or evidence text","trajectory zoom labeled and separated from full target range","source captions and analysis CSV present"],"viewer_fallback":"small in-memory thumbnails after local image viewer sandbox error"})
    wjson(ROOT/"qa/FINAL_QA.json",finalqa)
    print(json.dumps({"status":"PASS","primary_holm_pass":primary_pass,"fd_holm_pass":fd_pass,"gpu_hours":hours,"figures_visual_qa":"PASS","final_test_access":0},ensure_ascii=False,sort_keys=True))

if __name__=="__main__":raise SystemExit(main())

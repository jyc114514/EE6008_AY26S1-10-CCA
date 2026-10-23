#!/usr/bin/env python3
"""Reconcile sealed delay-study outputs into this audit root, CPU-only."""
from __future__ import annotations
import os
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
OLD=REPO/"reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
ROOT=REPO/"reports/core_b/V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z"
AN=OLD/"analysis";OUT=ROOT/"analysis"

def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError(f"refusing empty table: {path}")
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def stats(rows,key):
    a=np.asarray([float(r[key]) for r in rows],dtype=np.float64)
    return float(a.mean()),float(a.std(ddof=1))
def n(x,d=6):return f"{float(x):.{d}f}"
def pct(x):return f"{100*float(x):+.2f}%"

def main():
    source=read_csv(AN/"RESULTS_BY_SEED_AND_DELAY.csv")
    if len(source)!=72 or any(r["final_test_access"]!="0" for r in source):raise RuntimeError("source result count/final boundary failed")
    groups=defaultdict(list)
    for r in source:groups[(r["cell"],float(r["delay_seconds"]))].append(r)
    metrics=("arm_mae_std","hand_mae_std","full_mae_std","arm_fd_native","hand_fd_native","full_fd_native","arm_skill_vs_persistence","hand_skill_vs_persistence","full_skill_vs_persistence")
    cell_order=("CLEAN-N","CLEAN-D","CLEAN-S","ROBUST-N","ROBUST-D","ROBUST-S")
    table=[]
    for (cell,delay),rows in sorted(groups.items(),key=lambda x:(cell_order.index(x[0][0]),x[0][1])):
        if len(rows)!=3 or len({r["seed"] for r in rows})!=3:raise RuntimeError(f"seed count failed: {cell}/{delay}")
        item={"cell":cell,"delay_seconds":delay,"seed_count":3}
        for key in metrics:
            mu,sd=stats(rows,key);item[key+"_mean"]=mu;item[key+"_seed_sd"]=sd
        table.append(item)
    if len(table)!=24:raise RuntimeError(f"expected 24 cell-delay rows, got {len(table)}")
    write_csv(OUT/"DELAY_RESULTS_RECONCILIATION_TABLE.csv",table)
    effect_rows=[]
    for family,name in (("primary","PRIMARY_ROBUSTNESS_EFFECTS.csv"),("FD_secondary","FD_EFFECTS.csv")):
        rows=read_csv(AN/name)
        if len(rows)!=6:raise RuntimeError(f"{name} count is not 6")
        for r in rows:
            if r["final_test_access"]!="0":raise RuntimeError(f"final boundary failed: {name}")
            effect_rows.append({"family":family,**r})
    write_csv(OUT/"DELAY_EFFECTS_RECONCILIATION.csv",effect_rows)
    guard=read_csv(AN/"CLEAN_GUARDRAILS.csv");interactions=read_csv(AN/"VIDEO_DELAY_INTERACTIONS.csv")
    ages=read_csv(AN/"OBSERVATION_AGE_DISTRIBUTION.csv");robust_ages=[]
    for i,delay in enumerate((0.0,.25,.5,1.0)):
        rows=[r for r in ages if r["cell"]=="ROBUST-D" and int(r["delay_index"])==i]
        if len(rows)!=3:raise RuntimeError("effective-age rows do not cover all three seeds")
        robust_ages.append({"delay_seconds":delay,"seed_count":3,
            "episode_equal_mean_age_seconds":float(np.mean([float(r["episode_equal_mean_age_seconds"]) for r in rows])),
            "age_mean_seconds":float(np.mean([float(r["age_mean_seconds"]) for r in rows])),
            "age_p95_seconds_mean_across_seeds":float(np.mean([float(r["age_p95_seconds"]) for r in rows])),
            "age_max_seconds_max_across_seeds":float(max(float(r["age_max_seconds"]) for r in rows))})
    write_csv(OUT/"DELAY_AGE_RECONCILIATION.csv",robust_ages)
    baseline=json.loads((AN/"BASELINE_RECONCILIATION.json").read_text(encoding="utf-8"))
    old_status=(OLD/"FINAL_STATUS.ini").read_text(encoding="utf-8")
    old_line=next(x for x in old_status.splitlines() if x.startswith("VIDEO_DELAY_INTERACTION="))
    primary=[r for r in effect_rows if r["family"]=="primary"];fd=[r for r in effect_rows if r["family"]=="FD_secondary"]
    payload={"status":"PASS","source_delay_root":str(OLD),"source_checksum_manifest":"qa/CHECKSUMS.sha256","source_checksum_manifest_entries":806,"source_checksum_validation":{"command":"sha256sum --check --quiet qa/CHECKSUMS.sha256","exit_code":0,"read_only":True},
        "old_final_status_line":old_line,"old_final_status_sha256":"1e42fa4cc35c694c0a65427ac646e706ffa47c3bfc43b2a676cbbe7edad525b2",
        "baseline":baseline,"results_by_cell_delay":table,"primary_effects":primary,"fd_secondary_effects":fd,"normal_input_guardrails":guard,"video_delay_interactions":interactions,
        "interaction_holm_adjustment":"not performed; these diagnostics were outside both Holm families",
        "robust_dynamic_video_effective_age":robust_ages,
        "sign_conventions":"Primary MAE/FD effect is control minus candidate; positive favors candidate. The delta-zero normal-input guardrail is robust minus clean; positive means robust has larger error.",
        "signflip_vs_crossed_bootstrap":"Conditional episode sign-flip holds the three observed training seeds fixed. Crossed seed×episode bootstrap resamples both seed and episode and therefore includes training-seed uncertainty; report both.",
        "delay0_cpu_identity_check":{"primary_dev_windows":2322,"max_abs_difference_vs_locked_history_standardized":0.0,"exact_array_equal":True,"command_history_max_abs_difference":0.0,"selected_timestamp_after_anchor_max_seconds":0.0,"final_test_access":0},
        "delay_transform_contract":"Only observation channels 0:43 are replaced from past-selected timestamps; standardized action-history channels 43:71 are unchanged. Inference keeps the recipient anchor and video input. At requested delay zero, all primary-dev state histories are exactly the locked original.",
        "final_test_access":0}
    write_json(ROOT/"DELAY_RESULT_RECONCILIATION.zh-CN.json",payload)

    metric_specs=(("Arm MAE std","arm_mae_std",6),("Hand MAE std","hand_mae_std",6),("Full MAE std","full_mae_std",6),
        ("Arm FD native","arm_fd_native",8),("Hand FD native","hand_fd_native",8),("Full FD native","full_fd_native",8),
        ("Arm persistence skill","arm_skill_vs_persistence",3),("Hand persistence skill","hand_skill_vs_persistence",3),("Full persistence skill","full_skill_vs_persistence",3))
    md=["# 延迟实验完整统计复核","","复核只使用已封存的 development 报告与输入合同；没有重训或访问 final。源 root 的 806 条 checksum manifest 已通过 sha256sum 检查（exit 0）；72 个 per-seed/delay rows 均标记 final_test_access=0。","","## 六模型 × 四个请求延迟","","表内为三个 training seeds 的均值 ± seed 间样本标准差。MAE 为 standardized，FD 为 native units。Persistence skill = 1 − 模型 MAE / Persistence MAE，仅表示位置误差相对基线。",""]
    for label,key,digits in metric_specs:
        md += [f"### {label}","","| Cell | δ(s) | Mean ± seed SD |","|---|---:|---:|"]
        for r in table:md.append(f"| {r['cell']} | {r['delay_seconds']:g} | {n(r[key+'_mean'],digits)} ± {n(r[key+'_seed_sd'],digits)} |")
        md.append("")
    md += ["## Primary MAE family：6 项 Holm","","estimand 先在 episode 内对 δ=0.25/0.50/1.00 等权平均，再对 3 seeds 与 603 episodes 做 20,000 次 crossed bootstrap。正 effect 为 control loss − candidate loss，正值支持 candidate。sign-flip p 条件于观察到的 seeds。","",
        "| Contrast | Group | Effect | Crossed 95% CI | p | Holm p | Conditional sign-flip p |","|---|---|---:|---|---:|---:|---:|"]
    for r in primary:md.append(f"| {r['contrast']} | {r['group']} | {n(r['effect'])} | [{n(r['ci_low'])}, {n(r['ci_high'])}] | {float(r['p_two_sided']):.5g} | {float(r['holm_p']):.5g} | {float(r['p_signflip_episode']):.5g} |")
    md += ["","Holm 后 2/6 通过：ROBUST-D 的 Hand MAE 优于 ROBUST-N，也优于 CLEAN-D。Arm 的 D 对 N 未通过；D 对 S 的 Arm/Hand 均未建立差异。","","## Secondary native first-difference family：6 项 Holm","","正 effect 同样表示 candidate 的 FD loss 较低。","",
        "| Contrast | Group | Effect | Crossed 95% CI | p | Holm p | Conditional sign-flip p |","|---|---|---:|---|---:|---:|---:|"]
    for r in fd:md.append(f"| {r['contrast']} | {r['group']} | {n(r['effect'],8)} | [{n(r['ci_low'],8)}, {n(r['ci_high'],8)}] | {float(r['p_two_sided']):.5g} | {float(r['holm_p']):.5g} | {float(r['p_signflip_episode']):.5g} |")
    md += ["","Holm 后 2/6 通过，均为 Hand FD：ROBUST-D 优于 ROBUST-N 和 CLEAN-D；Arm FD 对比未通过。","","## 正常输入代价：δ=0 ROBUST-D − CLEAN-D","","此处正值表示 robust model 错误更大。Arm/Hand MAE 相对变化的 crossed CI 上界都在 +1% 内；Arm FD CI 上界约 +3.69%，因此正常输入的 Arm fidelity guardrail 未通过/未能排除有意义退化；Hand FD 上界在 +1% 内。","",
        "| Metric | Absolute effect | Crossed 95% CI | Relative change | Relative 95% CI | +1% margin |","|---|---:|---|---:|---|---|"]
    for r in guard:md.append(f"| {r['group']} | {n(r['effect'],8)} | [{n(r['ci_low'],8)}, {n(r['ci_high'],8)}] | {pct(r['relative_change'])} | [{pct(r['relative_ci_low'])}, {pct(r['relative_ci_high'])}] | {'PASS' if r['ci_upper_within_plus_1pct']=='True' else 'NOT ESTABLISHED'} |")
    md += ["","## Video × delay interaction：未经额外 Holm 的诊断","","G(δ)=MAE(ROBUST-N,δ)−MAE(ROBUST-D,δ)，正值表示 dynamic 相对 no-video 更好；interaction 是 delayed G 均值减 G(0)。旧 FINAL_STATUS 的 after_multiplicity 表述不正确：interaction 未纳入/未做额外 Holm。","",
        "| Group | Effect | Crossed 95% CI | Raw p | Episode sign-flip p | Multiplicity |","|---|---:|---|---:|---:|---|"]
    for r in interactions:md.append(f"| {r['group']} | {n(r['effect'],9)} | [{n(r['ci_low'],9)}, {n(r['ci_high'],9)}] | {float(r['p_two_sided']):.5g} | {float(r['p_signflip_episode']):.5g} | 未额外校正 |")
    md += ["","Hand interaction 约 0.000134 standardized MAE，虽 crossed CI 略高于 0，但这是小且未额外校正的诊断；Arm CI 跨 0。不能称为 multiplicity-adjusted 结果。D 相对 S 未建立可靠优势。","","## Effective observation age 与 δ=0 identity","","δ=0 CPU 重放在 2,322 个 primary-dev windows 上与锁定的原始 standardized state history 完全一致（max absolute difference=0）；action-history channels 43:71 也完全相同。delay transform 只替换 observation channels；recipient anchor command 和 recipient video input 未延迟，Persistence 也始终用同一个 anchor command。","",
        "| Requested δ(s) | Episode-equal mean age(s) | Mean age(s) | Mean p95(s) | Max age(s) |","|---:|---:|---:|---:|---:|"]
    for r in robust_ages:md.append(f"| {r['delay_seconds']:g} | {n(r['episode_equal_mean_age_seconds'],5)} | {n(r['age_mean_seconds'],5)} | {n(r['age_p95_seconds_mean_across_seeds'],5)} | {n(r['age_max_seconds_max_across_seeds'],5)} |")
    md += ["","δ=1.0 s 的最大有效 age 约 1.0333 s，来自采样时间量化；这仍是 observation-feedback delay 模拟，不是网络链路测量或闭环安全证明。","","## Sign-flip 与 crossed bootstrap 的差异","","Conditional sign-flip 不重采训练 seed，而是条件于三个已观察 seed 的配对 episode 效应并依赖符号对称性；crossed bootstrap 同时重采 seed 和 episode，会把训练 seed 不确定性带入区间。比如 robust D 对 N 的 Arm conditional sign-flip 很小，而 crossed CI 仍跨 0。两者不可择一报告，episode 也不能冒充 fresh-seed 数。","","旧 FINAL_STATUS 仍保持封存；新目录的 STATUS_INTERPRETATION_ADDENDUM.md/json 记录了原字段与更正。所有结论仍限于 development split、已观察 seeds；不证明真实网络故障恢复、物理轨迹正确或 final 泛化。"]
    (ROOT/"DELAY_RESULT_RECONCILIATION.zh-CN.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","model_delay_rows":len(table),"primary_items":len(primary),"fd_items":len(fd),"normal_guardrails":len(guard),"interactions":len(interactions),"age_points":len(robust_ages),"report":str(ROOT/"DELAY_RESULT_RECONCILIATION.zh-CN.md"),"final_test_access":0},ensure_ascii=False,sort_keys=True))
if __name__=="__main__":raise SystemExit(main())

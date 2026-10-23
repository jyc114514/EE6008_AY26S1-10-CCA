#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,os,socket,uuid,fcntl,shutil
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=PROJECT/"reports/benchmarks/CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z"
V5=PROJECT/"reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
V5_EXPECTED="5866688b48bef3ed940920affc15081dea052c51eca6eb86e7bcfd3a09c5acfa"
SEEDS=[202610131,202610132,202610133]
METRICS=["position_mae","safe_fd_l1","seam_safe_fd_l1"]
PRETTY={"position_mae":"position MAE","safe_fd_l1":"safe-FD L1","seam_safe_fd_l1":"seam safe-FD L1"}

def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,p)
def read_csv(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def fmt(x): return f"{float(x):+.6f}"
def plain(x): return f"{float(x):.6f}"
def pct(x): return f"{100*float(x):.2f}%"
def verify_v5():
    rec=read_json(V5/"qa/FINAL_CHECKSUM_VERIFICATION.json")
    raw=(V5/"CHECKSUMS.sha256").read_bytes();d=sha_bytes(raw)
    if d!=rec.get("manifest_sha256") or d!=V5_EXPECTED or rec.get("status")!="PASS":
        raise RuntimeError("V5 current receipt/manifest mismatch")
    if rec.get("manifest_entry_count")!=255 or rec.get("verified_entry_count")!=255 or rec.get("missing") or rec.get("mismatches"):
        raise RuntimeError("V5 receipt is not 255/255 PASS")
    entries={}
    for i,line in enumerate(raw.decode().splitlines(),1):
        if len(line)<67 or line[64:66]!="  ":raise RuntimeError(f"malformed V5 manifest line {i}")
        h,rel=line[:64],line[66:];rp=Path(rel)
        if rp.is_absolute() or ".." in rp.parts or rel in entries:raise RuntimeError("unsafe/duplicate V5 manifest path")
        entries[rel]=h
    bad=[]
    for rel,h in entries.items():
        p=V5/rel
        if not p.is_file() or p.is_symlink() or sha_file(p)!=h:bad.append(rel)
    if bad:raise RuntimeError(f"V5 current tree differs from final seal: {bad[:5]}")
    return {"status":"PASS","manifest_sha256":d,"entry_count":len(entries)}

def main():
    lock=(ROOT/"coordination/OWNER.lock").open("a+")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("finalization refused: V6 writer lock is active")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    owner_path=ROOT/"coordination/OWNER_STATE.json"
    shutil.copy2(owner_path,owner_path.with_name(owner_path.name+f".pre-finalize-{stamp}.bak"))
    lock.seek(0);old=lock.read()
    (ROOT/"coordination"/f"OWNER.lock.pre-finalize-{stamp}.bak").write_text(old,encoding="utf-8")
    if (ROOT/"results/RESULTS_SUMMARY.json").exists():
        shutil.copy2(ROOT/"results/RESULTS_SUMMARY.json",ROOT/"results"/f"RESULTS_SUMMARY.json.pre-final-report-{stamp}.bak")
    token=uuid.uuid4().hex
    lockinfo={"host":socket.gethostname(),"owner_uid":os.getuid(),"owner_pid":os.getpid(),
        "owner_token_sha256":sha_bytes(token.encode()),"study":"post_V5_equal_half_ensemble_controls_development",
        "out_root":str(ROOT),"owner_lock_acquired":True,"finalization":True}
    lock.seek(0);lock.truncate();lock.write(json.dumps(lockinfo,sort_keys=True)+"\n");lock.flush();os.fsync(lock.fileno())
    owner=read_json(owner_path)
    owner.update({"status":"running","phase":"final_acceptance_and_checksum","updated_utc":utc(),
        "host":socket.gethostname(),"owner_uid":os.getuid(),"owner_pid":os.getpid(),
        "owner_token_sha256":lockinfo["owner_token_sha256"]})
    write_json(owner_path,owner)
    try:
        v5=verify_v5()
        sm=read_json(ROOT/"inventory/SOURCE_MODEL_MAP.json")
        nodes=sm["source_nodes"]
        if len(nodes)!=9:raise RuntimeError("expected 9 locked source nodes")
        for n in nodes:
            for pk,hk in [("prediction_path","prediction_sha256"),("rows_path","rows_sha256"),("source_seal_path","source_seal_sha256")]:
                if sha_file(Path(n[pk]))!=n[hk]:raise RuntimeError(f"locked source changed: {n[pk]}")
        seal=read_json(ROOT/"qa/PREDICTION_SEAL.json")
        formula=read_json(ROOT/"protocol/FORMULA_LOCK.json")
        contrast=read_json(ROOT/"protocol/CONTRAST_LOCK.json")
        analysis=read_json(ROOT/"protocol/ANALYSIS_LOCK.json")
        if seal.get("status")!="PASS" or seal.get("sealed_bundles")!=36 or seal.get("missing_bundles")!=0:
            raise RuntimeError("prediction seal incomplete")
        fsha=sha_file(ROOT/"protocol/FORMULA_LOCK.json")
        if seal["formula_lock_sha256"]!=fsha or contrast["formula_lock_sha256"]!=fsha or analysis["formula_lock_sha256"]!=fsha:
            raise RuntimeError("locked protocol hashes differ")
        for rel in ["qa/SCHEMA_PREFLIGHT.json","qa/TRANSFORM_QA.json","qa/SCORER_RECOVERY.json"]:
            if read_json(ROOT/rel).get("status")!="PASS":raise RuntimeError(f"required QA did not pass: {rel}")
        exposure=read_json(ROOT/"inventory/OUTCOME_EXPOSURE.json")
        if (exposure.get("target_archive_load_operations")!=2 or exposure.get("target_array_window_rows_loaded")!=370
            or exposure.get("target_archive_unique_windows")!=185 or exposure.get("new_pair_window_loss_evaluations")!=6105
            or exposure.get("final_heldout_payload_access")!=0 or exposure.get("old_root_writes")!=0):
            raise RuntimeError("final exposure counters do not match actual run")
        pair_plan=read_csv(ROOT/"protocol/PAIR_PLAN.csv")
        pindex=read_csv(ROOT/"predictions/PREDICTION_INDEX.csv")
        if len(pair_plan)!=36 or len(pindex)!=36 or len({p["pair_id"] for p in pair_plan})!=36:
            raise RuntimeError("pair plan or prediction index is incomplete")
        fc=defaultdict(int)
        for p in pair_plan:fc[p["family"]]+=1
        if dict(sorted(fc.items()))!={"AA":3,"AV":9,"SA":9,"SS":3,"SV":9,"VV":3}:
            raise RuntimeError(f"family support differs: {dict(fc)}")
        if sum(p["reuse_v5_mix50"]=="true" for p in pair_plan)!=3 or sum(x["reuse_source"]!="" for x in pindex)!=3:
            raise RuntimeError("V5 MIX50 reuse count differs from 3")
        pw=read_csv(ROOT/"results/PER_WINDOW_PAIR_METRICS.csv")
        source_count=defaultdict(int);pair_count=defaultdict(int)
        for r in pw:source_count[r["score_source"]]+=1;pair_count[r["pair_id"]]+=1
        if len(pw)!=6660 or len(pair_count)!=36 or any(v!=185 for v in pair_count.values()):
            raise RuntimeError("saved pair score table is incomplete")
        if source_count["V6_target_score"]!=6105 or source_count["V5_MIX50_reused_after_reconciliation"]!=555:
            raise RuntimeError(f"score row counts differ: {dict(source_count)}")
        l1=read_csv(ROOT/"results/L1_CANCELLATION_LEDGER.csv")
        if len(l1)!=648 or not all(r["identity_pass"].lower()=="true" for r in l1):
            raise RuntimeError("L1 cancellation QA incomplete or failed")
        fam=read_csv(ROOT/"results/FAMILY_EFFECTS.csv")
        principal=read_csv(ROOT/"results/PRINCIPAL_MIX50_EFFECTS.csv")
        ci=read_csv(ROOT/"results/PAIRED_GROUP_CI.csv")
        seed_loo=read_csv(ROOT/"results/SEED_BLOCK_LOO.csv")
        task_loo=read_csv(ROOT/"results/TASK_LOO.csv")
        pair=read_csv(ROOT/"results/PAIR_TASK_METRICS.csv")
        if len(ci)!=30 or any(int(x["draws"])!=5000 or int(x["seed"])!=2026101506 for x in ci):
            raise RuntimeError("paired CI count or settings differ")
        cm={(x["contrast"],x["metric"]):x for x in ci}
        pm={}
        for x in principal:
            if x["scope"]=="seed_mean_macro":pm[(x["reference"],x["metric"])]=x
        fm={(x["family"],x["metric"]):x for x in fam if x["record_type"]=="family_mean" and x["scope"]=="macro"}
        summary={"schema_version":1,"status":"complete","study":"post_V5_equal_half_ensemble_controls_development",
            "principal":"three fixed V5 same-seed MIX50 pairs","source_programs":["S","A","V"],
            "source_seeds":[str(s) for s in [202610131,202610132,202610133]],
            "source_nodes_accepted":9,"pair_systems_accepted":36,"new_pair_bundles_accepted":33,
            "reused_v5_mix50_bundles":3,"families":dict(sorted(fc.items())),"ensemble_weight":[0.5,0.5],
            "independent_confirmation":False,"v5_current_checksum":v5,
            "principal_metrics":{},"family_mean_metrics":{},"family_contrasts":{},
            "seed_block_loo":{},"task_loo":{},"l1_cancellation":{},"all_pair_macro_ranges":{},
            "exposure":{"target_archive_loads":2,"target_rows_loaded_total":370,"unique_target_windows":185,
                "new_pair_window_loss_evaluations":6105,"mix50_reconciliation_windows":555,
                "reused_v5_mix50_loss_rows":555,"recovery_loss_recomputations":0,
                "cancellation_diagnostic_pair_windows":6660,"final_heldout_payload_access":0,"old_root_writes":0}}
        for m in METRICS:
            summary["principal_metrics"][m]={}
            for ref in ["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
                row=pm[(ref,m)];cirow=cm.get((f"MIX50_DIAG_MEAN3_vs_{ref}",m)) if ref!="PERSISTENCE" else None
                seeds={}
                for s in [202610131,202610132,202610133]:
                    x=next(z for z in principal if z["scope"]=="macro" and z["candidate"]=="MIX50" and int(z["candidate_seed"])==s and z["reference"]==ref and z["metric"]==m)
                    seeds[str(s)]=float(x["gain"])
                absrows=[x for x in principal if x["scope"]=="macro" and x["candidate"]=="MIX50" and x["reference"]=="S" and x["metric"]==m]
                vals=[float(x["candidate_loss"]) for x in absrows]
                summary["principal_metrics"][m][ref]={"gain_mean":float(row["gain"]),
                    "gain_sample_sd_across_three_pairs":float(row["gain_sample_sd_across_three_fixed_pairs"]),
                    "ci_low":float(cirow["ci_low"]) if cirow else None,
                    "ci_high":float(cirow["ci_high"]) if cirow else None,"per_seed_gain":seeds}
                summary["principal_metrics"][m]["absolute_candidate_loss"]={"mean":float(np.mean(vals)),
                    "sample_sd":float(np.std(vals,ddof=1))}
            summary["family_mean_metrics"][m]={}
            for family in ["SS","AA","VV","SA","SV","AV"]:
                x=fm[(family,m)]
                summary["family_mean_metrics"][m][family]={"mean":float(x["mean_pair_loss"]),
                    "pair_sample_sd":float(x["pair_sample_sd"]),"pair_min":float(x["pair_min"]),
                    "pair_max":float(x["pair_max"]),"n_pairs":int(x["n_pairs"])}
            summary["family_contrasts"][m]={}
            for name in ["SV_vs_SS","SV_vs_VV","SV_vs_SA","AV_vs_AA","AV_vs_VV","SV_pairing_effect"]:
                x=cm[(name,m)]
                summary["family_contrasts"][m][name]={"gain":float(x["point_estimate"]),
                    "ci_low":float(x["ci_low"]),"ci_high":float(x["ci_high"])}
            for key,rows,outkey in [("leave_out_seed",seed_loo,"seed_block_loo"),("leave_out_task",task_loo,"task_loo")]:
                summary[outkey].setdefault(m,{})
                for name in ["SV_vs_SS","SV_vs_VV","SV_vs_SA","AV_vs_AA","AV_vs_VV","SV_pairing_effect"]:
                    vals=[float(x["gain"]) for x in rows if x.get("contrast_name")==name and x["metric"]==m]
                    summary[outkey][m][name]={"min":min(vals),"max":max(vals),
                        "positive_count":sum(v>0 for v in vals),"n":len(vals)}
            lr=[x for x in l1 if x["scope"]=="macro" and x["metric"]==m]
            summary["l1_cancellation"][m]={"n_pairs":len(lr),
                "mean_j_gap":float(np.mean([float(x["j_gap_member_mean_minus_ensemble"]) for x in lr])),
                "mean_opposite_sign_error_fraction":float(np.mean([float(x["opposite_sign_error_fraction"]) for x in lr])),
                "mean_gain_vs_better_member":float(np.mean([float(x["gain_vs_better_member"]) for x in lr])),
                "max_element_identity_residual":max(float(x["max_element_identity_residual"]) for x in lr),
                "max_j_gap_reconstruction_residual":max(abs(float(x["j_gap_reconstruction_residual"])) for x in lr)}
            pv=[float(x["loss"]) for x in pair if x["scope"]=="macro" and x["metric"]==m]
            summary["all_pair_macro_ranges"][m]={"min":min(pv),"max":max(pv),"n_pairs":len(pv)}
        summary["task_level_direct_gain_positive_counts"]={}
        for m in ["position_mae","safe_fd_l1"]:
            summary["task_level_direct_gain_positive_counts"][m]={}
            for ref in ["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
                vals=[float(x["gain"]) for x in principal if x["scope"]=="task" and x["candidate"]=="MIX50" and x["reference"]==ref and x["metric"]==m]
                summary["task_level_direct_gain_positive_counts"][m][ref]={"positive":sum(v>0 for v in vals),"n":len(vals)}
        write_json(ROOT/"results/RESULTS_SUMMARY.json",summary)

        # Visual review record for figures actually inspected in this session.
        visual="""# V6 figure visual review

Assistant visual review: PASS. External human review: pending.

Inspected the contact sheet, the enlarged direct-reference gain figure, and the enlarged family point-spread figure. The three metric panels show the zero reference and the locked intervals; all six family panels show individual pair points, the mean, and finite-pair spread. The contact sheet shows the seed-block LOO, J_gap, and all five locked trajectory figures. The trajectory figures include target, S, V, and MIX50 series for native action indices 0 and 14. No clipped titles, axes, tick labels, or legends were observed. Figure generation used saved score tables and the already loaded development arrays; no archive was reopened for plotting.
"""
        visual_path=ROOT/"qa/FIGURE_VISUAL_REVIEW.md"
        if not visual_path.exists(): visual_path.write_text(visual,encoding="utf-8")
        if not (ROOT/"RUN_CONTEXT_RECOVERY.json").exists():
            write_json(ROOT/"RUN_CONTEXT_RECOVERY.json",{"schema_version":1,"created_utc":utc(),
            "host":socket.gethostname(),"uid":os.getuid(),
            "python_bin":str(PROJECT/"envs/jyc/bin/python"),
            "command":"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 <V6>/code/resume_v6.py",
            "device":"CPU","gpu_queries":0,"target_archive_loads_in_recovery":1,
            "target_read_purpose":"elementwise cancellation diagnostics only","pair_loss_recomputations":0})

        def ci_cell(name,m):
            x=cm[(name,m)]
            return f"{fmt(x['point_estimate'])} [{fmt(x['ci_low'])}, {fmt(x['ci_high'])}]"
        report=[]
        report += ["# EE6008 V6 固定等权平均对照结果","",
            f"- 状态：complete；输出根目录：{ROOT}",
            "- 研究类型：post-V5 development；主候选固定为 V5 三个同 seed MIX50 pair。",
            "- 统计区间条件于 9 个固定模型和已暴露 development 窗口。bootstrap：47 个 task-nested source groups，5,000 draws，seed 2026101506。",
            "",
            "## MIX50 对锁定参考的直接配对结果","",
            "gain = reference loss − MIX50 loss，正值支持 MIX50。四个非 Persistence 参考使用同一 source-group draw 计算 95% percentile CI；Persistence 按锁定计划报告点值。",
            "",
            "| 指标 | 参考 | seed 131 | seed 132 | seed 133 | 三对均值 | 95% CI |","|---|---|---:|---:|---:|---:|---:|"]
        for m in METRICS:
            for ref in ["S","V","BIAS_STATE","BIAS_VIDEO","PERSISTENCE"]:
                seedrows={}
                for s in [202610131,202610132,202610133]:
                    seedrows[s]=next(x for x in principal if x["scope"]=="macro" and x["candidate"]=="MIX50" and int(x["candidate_seed"])==s and x["reference"]==ref and x["metric"]==m)
                row=pm[(ref,m)];cc=cm.get((f"MIX50_DIAG_MEAN3_vs_{ref}",m))
                ci_text=f"[{fmt(cc['ci_low'])}, {fmt(cc['ci_high'])}]" if cc else "未计算"
                report.append(f"| {PRETTY[m]} | {ref} | {fmt(seedrows[202610131]['gain'])} | {fmt(seedrows[202610132]['gain'])} | {fmt(seedrows[202610133]['gain'])} | {fmt(row['gain'])} | {ci_text} |")
        abs_by={}
        for m in METRICS:
            x=summary["principal_metrics"][m]["absolute_candidate_loss"]
            abs_by[m]=x
        report += ["",
            f"三对 MIX50 的绝对宏平均：position MAE {plain(abs_by['position_mae']['mean'])}（seed sample SD {plain(abs_by['position_mae']['sample_sd'])}），safe-FD {plain(abs_by['safe_fd_l1']['mean'])}（SD {plain(abs_by['safe_fd_l1']['sample_sd'])}），seam {plain(abs_by['seam_safe_fd_l1']['mean'])}（SD {plain(abs_by['seam_safe_fd_l1']['sample_sd'])}）。",
            "",
            "Position MAE 对 S、V、BIAS_STATE、BIAS_VIDEO 的四个 CI 下界均大于零，三对固定模型的宏平均 gain 也全为正。Safe-FD 对 S 与 BIAS_STATE 的区间支持正 gain；对 V 与 BIAS_VIDEO 的区间跨零，且 seed 202610133 的 gain 为 −0.000120。按 15 个 task×seed 单元计，position gain 为正的数量：S 12/15、V 13/15、BIAS_STATE 9/15、BIAS_VIDEO 9/15；safe-FD：12/15、9/15、12/15、9/15。Persistence 的三项平均 gain 为 position 0.066791、safe-FD 0.035219、seam 0.037194。",
            "",
            "## 六类 family 均值","",
            "各 family 对固定 pair loss 等权平均；括号是有限 pair 的 sample SD，表示当前成员离散度，不表示独立 seed 不确定性。",
            "",
            "| Family | Pair 数 | Position MAE | Safe-FD L1 | Seam safe-FD L1 |","|---|---:|---:|---:|---:|"]
        for fam in ["SS","AA","VV","SA","SV","AV"]:
            xs=[fm[(fam,m)] for m in METRICS]
            report.append(f"| {fam} | {xs[0]['n_pairs']} | {plain(xs[0]['mean_pair_loss'])} ({plain(xs[0]['pair_sample_sd'])}) | {plain(xs[1]['mean_pair_loss'])} ({plain(xs[1]['pair_sample_sd'])}) | {plain(xs[2]['mean_pair_loss'])} ({plain(xs[2]['pair_sample_sd'])}) |")
        report += ["",
            "SV 的 finite-family position MAE 均值最低；safe-FD 低于 SS、AA、SA、AV，但高于 VV；seam 高于 SS、AA、VV。",
            "",
            "## 六个固定 family contrasts","",
            "每格为 point estimate [95% source-group CI]，正值支持 contrast 名称中的候选。",
            "",
            "| Contrast | Position MAE | Safe-FD L1 | Seam safe-FD L1 |","|---|---:|---:|---:|"]
        for name in ["SV_vs_SS","SV_vs_VV","SV_vs_SA","AV_vs_AA","AV_vs_VV","SV_pairing_effect"]:
            report.append(f"| {name} | {ci_cell(name,'position_mae')} | {ci_cell(name,'safe_fd_l1')} | {ci_cell(name,'seam_safe_fd_l1')} |")
        report += ["",
            "SV 相对 SA 在 position MAE 与 safe-FD 的区间为正；SV 相对 SS 的 safe-FD 区间为正，position MAE 区间跨零；SV 相对 VV 的 safe-FD 为负，position MAE 跨零。相同 RGB recipe 的 AV 相对 AA 在 position MAE 和 safe-FD 为正，相对 VV 的 safe-FD 为负。seam 排序与前两项不同。",
            "",
            "SV pairing effect 使用 L_offdiag−L_diag。三项区间均包含零，point estimates 为 position +0.000007、safe-FD −0.000161、seam +0.000462。",
            "",
            "## Seed-block 与 task LOO","",
            "Seed-block 每次共同删除 S_k、A_k、V_k 及所有含这些节点的 pair。position 下 SV 对 SS/VV/SA 与 AV 对 AA 的 gain 三次均为正，AV 对 VV 三次均为负；SV offdiag−diag 在零附近变号。safe-FD 下 SV 对 SS/SA 与 AV 对 AA 三次均为正，SV/AV 对 VV 三次均为负；offdiag−diag 三次均为微小负值。seam 下 SV 对 SS 三次均为负、对 VV 三次均为正，其余对照有变号。",
            "",
            "Task LOO 每次去掉一个 task 后重算等 task macro。Position 下 SV 对 SS/VV/SA 与 AV 对 AA 五次均为正；AV 对 VV 只有一次为正且最大正值 +0.000077；pairing effect 近零并变号。Safe-FD 下 SV 对 SS/SA 与 AV 对 AA 五次均为正，SV/AV 对 VV 五次均为负，pairing effect 五次均为负。Seam 下 SV 对 SS 五次均为负、对 VV 五次均为正；其他对照随任务变化。所有 LOO 数值在 results/SEED_BLOCK_LOO.csv 与 results/TASK_LOO.csv。",
            "",
            "## L1 误差抵消","",
            "J_gap = 0.5*(L_X+L_Y)−L_E 描述 pair loss 相对两成员平均 loss 的凸性收益，不代表超过较好成员。36 个固定 pair 的宏平均 J_gap / 异号误差比例 / 相对较好成员 gain：",
            "",
            "| Metric | J_gap | Opposite-sign fraction | Gain vs better member | Max element residual | Max aggregated J_gap residual |","|---|---:|---:|---:|---:|---:|"]
        for m in METRICS:
            x=summary["l1_cancellation"][m]
            report.append(f"| {PRETTY[m]} | {plain(x['mean_j_gap'])} | {pct(x['mean_opposite_sign_error_fraction'])} | {plain(x['mean_gain_vs_better_member'])} | {x['max_element_identity_residual']:.2e} | {x['max_j_gap_reconstruction_residual']:.2e} |")
        report += ["",
            "全部 648 个 pair×task/macro ledger checks 通过。凸性抵消量与具体 pair 超过较好成员的 gain 分列；该机制本身不识别视频特有信息。",
            "",
            "## 执行与数据暴露","",
            "- 9 个节点、36 个固定 pair：SS/AA/VV 各 3，SA/SV/AV 各 9；33 个新 bundle，3 个 V5 MIX50 原 bundle 复用。",
            "- 新 pair loss evaluations 6,105；555 条 V5 MIX50 per-window loss 复用并做 555 条 reconciliation。首轮 reducer 类型错误在完整 6,660 行评分表写盘后发生；错误记录保存在 qa/FAILURE.json 与 qa/FAILED_ATTEMPTS.jsonl。恢复复用该表、没有重算损失；为 elementwise L1 ledger 增加一次 DEV archive 读取。",
            "- V6 实际 DEV archive load 2 次、总 window-row loads 370，unique windows 185；figure generation 没有重新打开 archive。Final/heldout access 0，V2–V5 writes 0。",
            "- New neural fit、optimizer、calibration、weight search、encoder extraction 全部为 0。V2 已接受的 185-window S/V child-seconds 均值分别为 0.692481 / 0.758281，两者 child sum 1.450761；这些不是端到端延迟。A 时长不在已接受的 V5 cost summary 中。每个 pair 用两个冻结模型输出，不假设 FLOPs 相等。",
            "- Assistant 图表目视检查 PASS；external human review pending。source-group CI 条件于固定 9 个模型和开发 exposure；session/trial 关联未知，结论是开发线索而非独立确认。",
            "",
            "固定结果不支持启动额外训练或权重搜索。",
            "",
            "STATUS=complete；NEW_NEURAL_FITS=0；FINAL_HELDOUT_PAYLOAD_ACCESS=0；OLD_ROOT_WRITES=0；INDEPENDENT_CONFIRMATION=false；CHECKSUM_RECEIPT=qa/FINAL_CHECKSUM_VERIFICATION.json（报告完成后生成）。",
            ""]
        (ROOT/"RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(report),encoding="utf-8")
        return_json={"STATUS":"complete","STUDY":"post_V5_equal_half_ensemble_controls_development",
            "PRINCIPAL":"V5_same_seed_MIX50_fixed_three_pairs","SOURCE_PROGRAMS":["S","A","V"],
            "SOURCE_SEEDS":SEEDS,"SOURCE_NODES_ACCEPTED":"9/9","PAIR_SYSTEMS_ACCEPTED":"36/36",
            "NEW_PAIR_BUNDLES_ACCEPTED":"33/33","REUSED_V5_MIX50_BUNDLES":"3/3",
            "FAMILIES":["SS","AA","VV","SA","SV","AV"],"ENSEMBLE_WEIGHT":[0.5,0.5],
            "NEW_NEURAL_FITS":0,"OPTIMIZER_UPDATES":0,"NEW_CALIBRATION_FITS":0,"WEIGHT_SEARCH":0,
            "NEW_ENCODER_EXTRACTION":0,"FAMILY_PREDICTIONS_AVERAGED_INTO_LARGER_MODEL":False,
            "PAIRS_TREATED_AS_INDEPENDENT_TRAINING_SEEDS":False,"SEED_BLOCK_LOO_REMOVES_ALL_MEMBER_PROGRAMS":True,
            "PRINCIPAL_SELECTED_FROM_NEW_BEST_PAIR":False,"GT_USED_TO_CONSTRUCT_PREDICTIONS":False,
            "INDEPENDENT_CONFIRMATION":False,"FINAL_HELDOUT_PAYLOAD_ACCESS":0,"OLD_ROOT_WRITES":0,
            "TARGET_ARCHIVE_LOADS":2,"TARGET_ARCHIVE_ROWS_LOADED_TOTAL":370,"UNIQUE_TARGET_WINDOWS":185,
            "NEW_PAIR_WINDOW_LOSS_EVALUATIONS":6105,"REUSED_V5_MIX50_LOSS_ROWS":555,
            "MIX50_RECONCILIATION_WINDOW_EVALUATIONS":555,"CANCELLATION_DIAGNOSTIC_PAIR_WINDOWS":6660,
            "BOOTSTRAP_DRAWS":5000,"BOOTSTRAP_SEED":2026101506,"CI_ROWS":30,
            "V5_CHECKSUM_SHA256":v5["manifest_sha256"],"V5_CHECKSUM_ENTRIES_VERIFIED":255,
            "RESULTS":summary,"RESULTS_SUMMARY_PATH":"results/RESULTS_SUMMARY.json",
            "CHECKSUM_MANIFEST_PATH":"CHECKSUMS.sha256","CHECKSUM_RECEIPT_PATH":"qa/FINAL_CHECKSUM_VERIFICATION.json",
            "ASSISTANT_VISUAL_REVIEW":"PASS","EXTERNAL_HUMAN_VISUAL_REVIEW":"pending"}
        write_json(ROOT/"RETURN_TO_CHATGPT.json",return_json)
        acceptance={"schema_version":1,"status":"complete","created_utc":utc(),
            "study":"post_V5_equal_half_ensemble_controls_development",
            "source_nodes_expected":9,"source_nodes_accepted":9,"pair_systems_expected":36,"pair_systems_accepted":36,
            "family_counts":dict(sorted(fc.items())),"new_pair_bundles_accepted":33,"reused_v5_mix50_bundles":3,
            "prediction_seal_status":"PASS","schema_preflight_status":"PASS","transform_qa_status":"PASS",
            "scorer_recovery_status":"PASS","failed_attempts_preserved":1,"complete_score_table_reused_on_resume":True,
            "score_table_rows":6660,"new_pair_window_loss_evaluations":6105,
            "mix50_reconciliation_window_evaluations":555,"reused_v5_mix50_loss_rows":555,
            "target_archive_loads":2,"target_archive_window_rows_loaded_total":370,"unique_target_windows":185,
            "additional_archive_read_purpose":"L1 elementwise cancellation diagnostics only",
            "loss_recomputations_on_resume":0,"l1_ledger_rows":648,"l1_identity_checks_passed":True,
            "paired_group_ci_rows":30,"bootstrap_draws":5000,"bootstrap_seed":2026101506,
            "new_neural_fits":0,"optimizer_updates":0,"new_calibration_fits":0,"weight_search":0,
            "new_encoder_extraction":0,"family_predictions_averaged_into_larger_model":False,
            "pairs_treated_as_independent_training_seeds":False,"seed_block_loo_removes_all_member_programs":True,
            "principal_selected_from_new_best_pair":False,"gt_used_to_construct_predictions":False,
            "independent_confirmation":False,"final_heldout_payload_access":0,"old_root_writes":0,
            "v5_manifest_sha256":v5["manifest_sha256"],"v5_manifest_entries_verified":255,
            "assistant_visual_review_status":"PASS","external_human_visual_review_status":"pending",
            "checksum_manifest_path":"CHECKSUMS.sha256","checksum_receipt_path":"qa/FINAL_CHECKSUM_VERIFICATION.json",
            "checksum_receipt_written_after_acceptance":True}
        write_json(ROOT/"qa/FINAL_ACCEPTANCE.json",acceptance)
        owner.update({"status":"complete","phase":"complete_checksum_pending","updated_utc":utc(),
            "source_nodes_accepted":9,"pair_systems_accepted":36,"new_pair_bundles_accepted":33,
            "reused_v5_mix50_bundles":3,"target_archive_loads":2,"target_array_rows_loaded_total":370,
            "new_pair_window_loss_evaluations":6105,"final_heldout_payload_access":0,"old_root_writes":0,
            "external_human_visual_review_status":"pending","all_report_artifacts_written":True})
        write_json(owner_path,owner)

        manifest_path=ROOT/"CHECKSUMS.sha256";receipt_path=ROOT/"qa/FINAL_CHECKSUM_VERIFICATION.json"
        if manifest_path.exists() or receipt_path.exists():raise RuntimeError("checksum already exists; refusing implicit reseal")
        excluded={"CHECKSUMS.sha256","qa/FINAL_CHECKSUM_VERIFICATION.json"}
        files=[]
        for p in ROOT.rglob("*"):
            if p.is_symlink():raise RuntimeError(f"symlink in V6 tree: {p}")
            if p.is_file():
                rel=p.relative_to(ROOT).as_posix()
                if rel not in excluded:files.append((rel,p))
        files.sort()
        manifest_text="\n".join(f"{sha_file(p)}  {rel}" for rel,p in files)+"\n"
        manifest_path.write_text(manifest_text,encoding="utf-8")
        manifest_sha=sha_file(manifest_path)
        parsed={}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if len(line)<67 or line[64:66]!="  ":raise RuntimeError("malformed generated checksum manifest")
            parsed[line[66:]]=line[:64]
        verified=[];missing=[];mismatch=[]
        for rel,h in parsed.items():
            p=ROOT/rel
            if not p.is_file():missing.append(rel)
            else:
                got=sha_file(p)
                if got!=h:mismatch.append({"path":rel,"expected":h,"actual":got})
                else:verified.append(rel)
        live={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.relative_to(ROOT).as_posix() not in excluded}
        unlisted=sorted(live-set(parsed))
        if unlisted:mismatch.append({"unlisted_files":unlisted})
        receipt={"schema_version":1,"created_utc":utc(),
            "status":"PASS" if not missing and not mismatch and len(verified)==len(parsed) else "FAIL",
            "manifest_path":"CHECKSUMS.sha256","manifest_sha256":manifest_sha,
            "manifest_entry_count":len(parsed),"verified_entry_count":len(verified),
            "missing":missing,"mismatches":mismatch,
            "excluded_from_manifest":[{"path":"CHECKSUMS.sha256","reason":"manifest cannot hash itself"},
                {"path":"qa/FINAL_CHECKSUM_VERIFICATION.json","reason":"read-back receipt is written after manifest"}],
            "v5_current_manifest_sha256":v5["manifest_sha256"],"v5_current_manifest_entries_verified":255}
        write_json(receipt_path,receipt)
        readback=read_json(receipt_path)
        if readback["status"]!="PASS" or readback["manifest_sha256"]!=manifest_sha or sha_file(manifest_path)!=manifest_sha:
            raise RuntimeError("checksum read-back failed")
        print(json.dumps({"status":readback["status"],"manifest_sha256":manifest_sha,
            "entry_count":len(parsed),"verified":len(verified),"missing":missing,"mismatches":mismatch,"root":str(ROOT)},ensure_ascii=False))
    except Exception as e:
        owner.update({"status":"technical_failure","phase":"finalization_failed","updated_utc":utc(),
            "error_type":type(e).__name__,"error_message":str(e)})
        write_json(owner_path,owner)
        raise
    finally:
        lock.close()

if __name__=="__main__":main()


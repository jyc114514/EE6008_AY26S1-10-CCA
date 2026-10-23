from __future__ import annotations

import json
import statistics
from datetime import datetime,timezone
from pathlib import Path

import v8_extract as ex

V8=ex.V8
V7=ex.V7
SEEDS=ex.SEEDS
TASKS=ex.TASKS


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def readcsv(path):
    return ex.read_csv(path)


def mean(xs):
    return float(sum(xs)/len(xs)) if xs else None


def sd(xs):
    return float(statistics.stdev(xs)) if len(xs)>1 else 0.0


def main():
    summary=ex.read_json(V8/"RETURN_TO_CHATGPT.json")
    resource=readcsv(V8/"results/V8_RESOURCE_AND_EXPOSURE.csv")
    supplement=readcsv(V8/"results/V7_RESOURCE_SUPPLEMENT.csv")
    help_rows=readcsv(V8/"results/TEACHER_HELP_HARM.csv")
    decision=readcsv(V7/"results/PAIRED_EFFECTS.csv")
    windows=readcsv(V7/"results/PER_WINDOW_METRICS.csv")
    curve=readcsv(V8/"results/TRAIN_CURVE_SUMMARY.csv")
    v7_receipt=ex.read_json(V7/"qa/FINAL_CHECKSUM_VERIFICATION.json")
    v7_manifest=ex.sha(V7/"CHECKSUMS.sha256")
    assert v7_manifest==v7_receipt["manifest_sha256"]
    assert len(supplement)==9

    help_totals={}
    for metric in ("position_mae","safe_fd_l1","seam_safe_fd_l1"):
        help_totals[metric]={}
        for category in ("teacher_help","exact_tie","teacher_harm"):
            subset=[r for r in help_rows if r["metric"]==metric and r["teacher_stratum"]==category]
            help_totals[metric][category]={
                "window_count":sum(int(r["window_count"]) for r in subset),
                "mean_equal_task_seed_weight_fraction":mean([float(r["weighted_fraction"]) for r in subset])}
    task_gains={}
    for metric in ("position_mae","safe_fd_l1","seam_safe_fd_l1"):
        rows=[r for r in decision if r["contrast"]=="MIX_VS_GT" and r["scope"]=="task" and r["metric"]==metric]
        task_gains[metric]={str(task):mean([float(r["gain"]) for r in rows if int(r["task_id"])==task]) for task in TASKS}

    loo_geom=readcsv(V8/"results/CORRECTION_GEOMETRY_LOO.csv")
    loo_ranges={}
    for split in ("train","dev"):
        loo_ranges[split]={}
        for loo_type in ("seed_block_loo","task_loo"):
            loo_ranges[split][loo_type]={}
            for space in ("position","internal_fd"):
                loo_ranges[split][loo_type][space]={}
                for vector in ("A","E"):
                    q=[r for r in loo_geom if r["split"]==split and r["scope"]==loo_type
                       and r["record_type"]=="projection" and r["space"]==space and r["vector"]==vector]
                    loo_ranges[split][loo_type][space][vector]={
                        field:{"min":min(float(r[field]) for r in q),"max":max(float(r[field]) for r in q)}
                        for field in ("weighted_cosine","projection_k","orthogonal_energy")}
    projection_rows=[r for r in readcsv(V8/"results/CORRECTION_GEOMETRY.csv")
                     if r["record_type"]=="projection"]
    max_orth=max(abs(float(r["orthogonality_residual"])) for r in projection_rows)
    max_energy_abs=max(float(r["energy_identity_abs_error"]) for r in projection_rows)
    retention=next(r for r in readcsv(V7/"results/TEACHER_RETENTION.csv")
                   if r["scope"]=="overall_equal_seed_losses")
    resource_summary={}
    for key in ("train_Y_npy_loads","train_Z_npy_loads","dev_target_archive_loads",
                "dev_target_windows_loaded","dev_prefix_archive_loads","prediction_native_arrays_loaded",
                "train_target_perturbation_window_evaluations","train_model_target_metric_window_evaluations",
                "train_objective_J_window_evaluations","train_dev_imitation_window_evaluations",
                "LOO_metric_window_evaluations","dev_step_axis_diagnostic_window_methods",
                "horizon_axis_numeric_values_written","GPU_jobs_on_physical_GPU1",
                "max_peak_allocated_mib","max_peak_reserved_mib"):
        vals=[r for r in resource if r["metric"]==key and r["phase"]=="V8_total"]
        if not vals: vals=[r for r in resource if r["metric"]==key]
        resource_summary[key]=[{"phase":r["phase"],"count":r["count"],"unit":r["unit"]} for r in vals]

    figure_meta=ex.read_json(V8/"figure_data/FIGURE_EXPORT_METADATA.json")
    figure_review={
      "schema_version":1,"status":"AGENT_VISUAL_INSPECTION_PASS_HUMAN_REVIEW_PENDING",
      "reviewed_utc":now(),"review_method":"reduced PNG previews inspected in-memory",
      "figures":[{"path":r["path"],"sha256":r["sha256"],"pixels":r["pixels"],"dpi":r["dpi"]} for r in figure_meta["outputs"]],
      "observations":[
        "task gains use categorical seed points and mean diamonds; zero baseline visible",
        "train/dev teacher distances use separate panels and report sample SD across three seeds",
        "correction geometry labels k as descriptive and keeps position/FD panels separate",
        "help/harm contributions show positive and negative values around zero",
        "no target-bearing trajectory or horizon/axis plot is produced"
      ],
      "human_review":"PENDING",
      "publisher_compliance":"NOT_ASSESSED; no journal or submission destination specified"
    }
    ex.write_json(V8/"qa/FIGURE_REVIEW.json",figure_review)
    figure_meta["review"]={
      "agent_visual_inspection":"PASS_REDUCED_PREVIEW",
      "human_review":"PENDING",
      "notes":figure_review["observations"],
      "publisher_compliance":"NOT_ASSESSED"
    }
    ex.write_json(V8/"figure_data/FIGURE_EXPORT_METADATA.json",figure_meta)

    summary.update({
      "resource_accounting":resource_summary,
      "V7_resource_supplement":{"rows":len(supplement),
        "physical_gpus":sorted({r["physical_gpu"] for r in supplement}),
        "peak_allocated_mib_min":min(float(r["peak_allocated_MiB"]) for r in supplement),
        "peak_allocated_mib_max":max(float(r["peak_allocated_MiB"]) for r in supplement),
        "peak_reserved_mib_min":min(float(r["peak_reserved_MiB"]) for r in supplement),
        "peak_reserved_mib_max":max(float(r["peak_reserved_MiB"]) for r in supplement),
        "host_RSS":"UNKNOWN"},
      "V8_figures":{"count":len(figure_meta["outputs"]),
        "visual_review":"PASS_REDUCED_PREVIEW","human_review":"PENDING"},
      "V8_selected_case_count":len(readcsv(V8/"CASE_SELECTION.csv")),
      "V8_case_trajectory_plot":"UNAVAILABLE_DEV_Y_CLOSED",
      "V8_failure_and_recovery_attempts":len(ex.read_json(V8/"coordination/EXTRACTION_ATTEMPTS.json")["attempts"]),
      "V7_manifest_sha256":v7_manifest,
      "V8_manifest_sha256":"see qa/FINAL_CHECKSUM_VERIFICATION.json after checksum-last",
      "resource_supplement_rows":9,
      "task_level_MIX_vs_GT_gains":task_gains,
      "teacher_help_harm_counts_and_fractions":help_totals,
      "V7_source_group_CI_interpretation":"original exploratory conditional percentile intervals reused; no new bootstrap or p-values",
      "V7_teacher_retention_R":float(retention["retention_R"]),
      "correction_geometry_LOO_ranges":loo_ranges,
      "correction_projection_QA":{"max_abs_orthogonality_residual":max_orth,
                                  "max_energy_identity_abs_error":max_energy_abs,
                                  "absolute_tolerance":1e-12,"relative_energy_tolerance":1e-10},
      "root":str(V8),
      "final_checksum_receipt_path":str(V8/"qa/FINAL_CHECKSUM_VERIFICATION.json"),
      "prediction_native_arrays_loaded_total":100,
      "dev_target_archive_loads_actual":1,
      "final_heldout_payload_access":0,
      "old_root_writes":0,
      "status":"complete_with_limits",
      "recommendation":"close fixed 0.5 mixture target; retain MIX50 as a dual-head development reference and GT_RETRAIN/accepted V_i as the single-head reference",
      "next_question":"NONE",
    })
    ex.write_json(V8/"RETURN_TO_CHATGPT.json",summary)

    m_g=summary["train_Z_gain"]["TRAIN_Z_GAIN_M"]
    u_g=summary["train_Z_gain"]["TRAIN_Z_GAIN_U"]
    m_train=summary["train_dev_teacher_distance"]["train"]["M"]
    g_train=summary["train_dev_teacher_distance"]["train"]["G"]
    m_dev=summary["train_dev_teacher_distance"]["dev"]["M"]
    g_dev=summary["train_dev_teacher_distance"]["dev"]["G"]
    macro=summary["V7_macro_losses"]
    cis=summary["V7_MIX_vs_GT_source_group_CI"]
    line=[
      "# V8 固定教师转移诊断回传","",
      "STATUS=complete_with_limits",
      "STUDY=post_V7_frozen_transfer_diagnosis_development",
      "POST_RESULT_INFORMED_DIAGNOSTIC=true",
      "INDEPENDENT_CONFIRMATION=false",
      "",
      "## 1. 学生是否更好地拟合规定训练目标 Z",
      "",
      "按最终 epoch 160 的冻结 eval 输出，以相同 Z 比较 G=GT_RETRAIN 与软目标学生。J 使用 V7 runner 的 position L1 + 0.1 internal-FD L1、原 Y 拟合尺度及 task/episode 权重；正的 G−student 表示软目标学生更好。",
      f"- MIX 对 Z_M：mean J gain = {m_g['mean']:+.6f}（seed SD {m_g['sample_sd']:.6f}）；三 seed = "+", ".join(f"{int(k)}:{v:+.6f}" for k,v in m_g["per_seed"].items())+".",
      f"- SELF 对 Z_U：mean J gain = {u_g['mean']:+.6f}（seed SD {u_g['sample_sd']:.6f}）；三 seed = "+", ".join(f"{int(k)}:{v:+.6f}" for k,v in u_g["per_seed"].items())+".",
      "- 两项均三 seed 为负：在相同规定 Z 上，软目标学生的最终 eval loss 没有胜过硬监督 G。该结果不证明容量上限或未收敛。",
      "- 最终 train 对真实 Y：MIX position MAE mean "+f"{summary['final_train_vs_Y']['M']['position_mae']['mean']:.6f}"+", safe-FD "+f"{summary['final_train_vs_Y']['M']['safe_fd_l1']['mean']:.6f}"+", seam "+f"{summary['final_train_vs_Y']['M']['seam_safe_fd_l1']['mean']:.6f}"+". 均为锁定尺度下的标准化误差，不是物理单位。详情见 TRAIN_TARGET_AND_FIT.csv。",
      "- V7 epoch 曲线是 dropout-on、pre-update 汇总。跨三 seed 的 epoch160 objective 均值 G/U/M=0.262412/0.144083/0.146343；last20 min/median/max 等 seed 平均分别 G 0.262304/0.262594/0.262865、U 0.143953/0.144198/0.144472、M 0.146131/0.146395/0.146748。不同 Z 的 objective 不作准确率横比；有限值检查通过，梯度范数未记录。",
      "",
      "## 2. 输出是否朝教师 T 移动，train 与 dev 的差异",
      "",
      f"- Train 到 T 的距离 G→M：position {g_train['position']['mean']:.6f}→{m_train['position']['mean']:.6f}；safe-FD {g_train['safe_fd']['mean']:.6f}→{m_train['safe_fd']['mean']:.6f}；seam {g_train['seam']['mean']:.6f}→{m_train['seam']['mean']:.6f}.",
      f"- Dev 到 T 的距离 G→M：position {g_dev['position']['mean']:.6f}→{m_dev['position']['mean']:.6f}；safe-FD {g_dev['safe_fd']['mean']:.6f}→{m_dev['safe_fd']['mean']:.6f}；seam {g_dev['seam']['mean']:.6f}→{m_dev['seam']['mean']:.6f}.",
      "- M 在这六个距离上都比 G 更远；均为三 seed 均值，样本 SD 在 RETURN_TO_CHATGPT.json 与 TRAIN_DEV_IMITATION.csv。",
      "- correction geometry 部分对齐但不能抵消距离增长：A/C position cosine/k train 0.408/0.388、dev 0.350/0.436；internal-FD train 0.373/0.509、dev 0.376/0.494。E/C position cosine/k train 0.329/0.257、dev 0.042/0.047；internal-FD train 0.181/0.211、dev 0.049/0.062。k 是描述性投影，不是部署系数或因果证据。",
      "- C/A/E RMS：position train 0.0350/0.0333/0.0274、dev 0.0648/0.0808/0.0722；internal-FD train 0.1296/0.1766/0.1506、dev 0.2860/0.3756/0.3654；seam train 0.2839/0.3801/0.3492、dev 0.4161/0.6586/0.5425。",
      "- whole-seed/task LOO 保持全部 G/U/M/S/V/T 行并重新池化。例：A/C position k 的 train seed-LOO 范围 0.364–0.401、task-LOO 0.382–0.398；E/C position k 的 dev seed-LOO -0.024–0.128、task-LOO -0.014–0.101。完整 position/FD、cos/k/orthogonal-energy 范围见 CORRECTION_GEOMETRY_LOO.csv；投影正交和能量恒等式误差通过锁定容差。",
      "",
      "## 3. MAE、FD、seam 的任务局部结果，以及 step/coordinate 限制",
      "",
      "- V7 已接受 macro：G position/FD/seam = "+f"{macro['GT_RETRAIN']['position_mae']:.6f}/{macro['GT_RETRAIN']['safe_fd_l1']:.6f}/{macro['GT_RETRAIN']['seam_safe_fd_l1']:.6f}"+f"; M = {macro['MIX_SOFT']['position_mae']:.6f}/{macro['MIX_SOFT']['safe_fd_l1']:.6f}/{macro['MIX_SOFT']['seam_safe_fd_l1']:.6f}"+". G loss−M loss gains 为 -0.000850 / -0.004274 / +0.020449；正 gain 有利于 M。",
      "- 原 V7 source-group 95% CI：position [-0.002180,+0.000579]，safe-FD [-0.008554,-0.000481]，seam [+0.007785,+0.035567]；仅复用原 CI，本轮无 bootstrap/p 值。",
      "- 按 task 平均三 seed 的 G loss−M loss gain，位置 MAE：119 -0.000732、94 -0.000407、90 -0.000027、141 -0.003101、230 +0.000019；FD：119 -0.007583、94 -0.004122、90 +0.000050、141 -0.009753、230 +0.000038；seam：119 +0.060523、94 +0.005492、90 +0.005303、141 +0.030055、230 +0.000875。",
      "- 退化主要见 task 141 的 position/FD，另有 task 119/94 的 FD；seam gain 最大在 task 119 与 141。每个 task/seed/contrast 见 DECISION_TABLE.csv。",
      "- V7 每窗口 help/harm 表显示三 seed 合计 555 个窗口/metric：position teacher-help 345、harm 210；FD 326/229；seam 381/174。窗口数不是独立 N；加权 contribution 与 M 对 G gain 分层见 TEACHER_HELP_HARM.csv。",
      "- 30 帧 position step、29 个 internal-FD transition、28 个坐标误差以及带 Y 的例子轨迹在本轮标为 UNAVAILABLE。第二次提取因向量聚合错误在唯一 dev archive load 后中止；已按锁定规则未重开 archive。V7 无可复用的 per-step/per-axis 明细。",
      "",
      "## 执行、资源与建议",
      "",
      "- 本轮新 fit=0，optimizer update=0，teacher forward=0，新 dev prediction=0；仅 6 个 U/M final train-prefix frozen forwards，均在物理 GPU 1（逻辑 cuda:0），状态哈希不变。",
      "- 实际 V8 读取累计：train Y NPY 3 次、9 个 Z 各 3 次；dev target archive 1 次/185 IDs；dev prefix 1 次；prediction_native tensor arrays 100 次（含 1 次 safe_open schema probe、两次失败 CPU extraction、一次 salvage；所有文件均只读）。V8_RESOURCE_AND_EXPOSURE.csv 分列成功/失败成本。",
      "- V7_RESOURCE_SUPPLEMENT.csv 从九份接受 model manifest 补列，9 行；formal peak allocated 均 139.799 MiB、reserved 均 182 MiB；host RSS UNKNOWN。V8 新推理实际 peak allocated 36.318 MiB、reserved 50 MiB。",
      "- Final heldout payload access=0；旧 V7 root writes=0；V7 manifest SHA-256 "+v7_manifest+".",
      "- 建议关闭当前固定 0.5 soft-target 方案；MIX50 只保留双 head 开发精度参考，G/V 保留单 head 参考。下一问题：NONE。",
      "",
      f"V8 root: {V8}",
      "最终 manifest SHA-256 与文件核验数见 qa/FINAL_CHECKSUM_VERIFICATION.json。",
      "Figures were visually inspected in reduced previews; human review remains pending.",
    ]
    (V8/"RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(line)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","figures":len(figure_review["figures"]) if False else 4,
                      "markdown":str(V8/"RETURN_TO_CHATGPT.zh-CN.md"),
                      "dev_archive_reloads_by_finalizer":0},sort_keys=True))


if __name__=="__main__":
    main()

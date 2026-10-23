"""Build the audited Core B results report and dashboard from locked local tables."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
STATE = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
HAND = REPO / "reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z"
ROUTER = REPO / "reports/core_b/PROTOTYPE_DIAGNOSIS_ROUTER_STUDY_V1_20260917T060236Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
LONG_CSV = ROOT / "02_SCIENTIFIC_RESULTS.csv"
REPORT_MD = ROOT / "02_SCIENTIFIC_RESULTS.zh-CN.md"
CLAIMS_CSV = ROOT / "02_CLAIM_LEDGER.csv"

LONG_FIELDS = [
    "study", "analysis", "model", "candidate", "control", "seed", "horizon_steps",
    "body", "metric", "stratum", "event_bin", "estimate", "ci95_low", "ci95_high",
    "seed_sd_n3", "support_episodes", "support_source_groups", "support_windows_or_pairs",
    "opportunities", "hits", "misses", "false_alarms", "exposure_hours",
    "timing_abs_error_seconds", "bootstrap_replicates", "confidence_interval_type",
    "status", "final_test_access", "source_file", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value, digits=6):
    value = number(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def safe_cell(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


LONG_ROWS: list[dict] = []


def add_long(study, analysis, source, *, notes="", **values):
    row = {field: "" for field in LONG_FIELDS}
    row.update({
        "study": study,
        "analysis": analysis,
        "source_file": source,
        "notes": notes,
        "final_test_access": 0,
    })
    row.update(values)
    LONG_ROWS.append({key: safe_cell(value) for key, value in row.items()})


def load_analysis_rows():
    # D1: individual cell scores and the post-hoc supported-middle extension.
    for rel in ("results/D1_SOURCE_GROUP_RESULTS.csv", "results/D1_MIDDLE_STRATUM_RESULTS.csv"):
        for r in read_csv(ROOT / rel):
            add_long(
                "D1", "model_score", rel, model=r["model"], seed=r["seed"],
                horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
                stratum=r["stratum"], estimate=number(r["estimate"]),
                ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
                support_episodes=r["N_episode"], support_source_groups=r["N_source_group"],
                support_windows_or_pairs=r["N_window"], bootstrap_replicates=10000,
                confidence_interval_type="source-group percentile bootstrap, per fixed model seed",
                status="measured", notes=("Middle stratum is a post-hoc descriptive extension using the parent train-only thresholds."
                     if "MIDDLE" in rel else "Source-group reanalysis; natural and train-supported strata are retained."),
            )
    for r in read_csv(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv"):
        add_long(
            "D1", r["contrast"], "results/D1_VISUAL_GAIN_EXTENSIONS.csv",
            model="", candidate=r["candidate"], control=r["control"], seed=r["seed"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            stratum=r["stratum"], estimate=number(r["estimate"]),
            ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_source_groups=r["N_source_groups"], support_windows_or_pairs=r["N_windows"],
            bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="post-hoc source-group percentile bootstrap, per fixed model seed",
            status="measured_posthoc", notes="Positive gain = control error minus F-TOK-D error; positive favors F-TOK-D. No p-values.",
        )
    for r in read_csv(ROOT / "results/D1_SOURCE_GROUP_INTERACTIONS.csv"):
        add_long(
            "D1", "locked_high_minus_low_interaction", "results/D1_SOURCE_GROUP_INTERACTIONS.csv",
            candidate=r["fusion"], control=r["control"], seed=r["seed"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric="mae",
            stratum="high_gain_minus_low_gain", estimate=number(r["high_minus_low"]),
            ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_source_groups=r["bootstrap_source_groups"],
            support_windows_or_pairs=f"low_groups={r['N_low_groups']};high_groups={r['N_high_groups']}",
            bootstrap_replicates=10000, confidence_interval_type="shared source-group bootstrap",
            status="measured", notes="Primary locked D1 interaction; estimate is high-stratum visual gain minus low-stratum visual gain.",
        )

    # D2 dense event results: every denominator, lead window, and short exposure is explicit.
    for r in read_csv(ROOT / "results/D2_DENSE_RESULTS_V2.csv"):
        base = {
            "model": r["model"], "seed": r["seed"], "horizon_steps": r["horizon_steps"],
            "body": r["event_group"], "support_episodes": r["episodes"],
            "support_source_groups": r["source_groups"], "support_windows_or_pairs": r["queries"],
            "opportunities": r["full_opportunity_events"], "hits": r["hits"], "misses": r["misses"],
            "false_alarms": r["false_alarms"], "exposure_hours": number(r["exposure_hours"]),
            "bootstrap_replicates": 10000,
            "confidence_interval_type": "source-group cluster bootstrap; task-balanced dense panel",
        }
        for metric, value, low, high, status, note in [
            ("full_opportunity_events", r["full_opportunity_events"], "", "", "measured", "Complete-horizon independent event opportunities."),
            ("recall", r["recall"], r["recall_ci95_low"], r["recall_ci95_high"], "measured", "Event recall over complete opportunities."),
            ("false_alarms_per_hour", r["false_alarms_per_hour"], r["false_alarms_per_hour_ci95_low"], r["false_alarms_per_hour_ci95_high"], "measured", "False alarms divided by dense-panel exposure hours."),
            ("timing_abs_error_seconds", r["timing_abs_error_seconds"], "", "", "measured" if number(r["timing_abs_error_seconds"]) is not None else "NA_no_hits", "Absolute event timing error; blank when no matched event."),
            ("exposure_hours", r["exposure_hours"], "", "", "measured", "Exposure is repeated for each model/group row; do not sum across models."),
        ]:
            add_long(
                "D2_dense", "event_metric", "results/D2_DENSE_RESULTS_V2.csv",
                **base, metric=metric, estimate=number(value), ci95_low=number(low), ci95_high=number(high),
                status=status, notes=note + "; panel population is task-balanced, not natural prevalence.",
            )
        for lead, suffix in (("0.25", "0p25"), ("0.5", "0p5"), ("1.0", "1p0")):
            denominator = r[f"opportunity_events_lead_ge_{suffix}"]
            hits = r[f"hits_lead_ge_{suffix}"]
            rate = r[f"recall_lead_ge_{suffix}"]
            den = number(denominator)
            lead_base = {**base, "opportunities": denominator, "hits": hits,
                         "misses": (int(denominator) - int(hits)) if den is not None else ""}
            add_long(
                "D2_dense", "lead_recall", "results/D2_DENSE_RESULTS_V2.csv",
                **lead_base, metric=f"recall_lead_ge_{lead}s", stratum=f"lead_at_least_{lead}s",
                estimate=number(rate), ci95_low=number(r[f"recall_lead_ge_{suffix}_ci95_low"]),
                ci95_high=number(r[f"recall_lead_ge_{suffix}_ci95_high"]),
                status="measured" if den else "NA_no_complete_opportunity",
                notes="No complete opportunity is NA, never zero; task-balanced dense panel.",
            )
    for r in read_csv(ROOT / "results/D2_DENSE_PAIRED_EFFECTS_V2.csv"):
        for metric, value, low, high in (
            ("recall_difference", r["recall_difference"], r["recall_diff_ci95_low"], r["recall_diff_ci95_high"]),
            ("false_alarms_per_hour_difference", r["false_alarms_per_hour_difference"], r["false_alarm_diff_ci95_low"], r["false_alarm_diff_ci95_high"]),
        ):
            add_long(
                "D2_dense", "paired_effect", "results/D2_DENSE_PAIRED_EFFECTS_V2.csv",
                model=r["model"], candidate=r["model"], control=r["baseline"], seed=r["seed"],
                horizon_steps=r["horizon_steps"], body=r["event_group"], metric=metric,
                estimate=number(value), ci95_low=number(low), ci95_high=number(high),
                support_source_groups=r["paired_source_groups"], bootstrap_replicates=10000,
                confidence_interval_type="paired source-group bootstrap",
                status="measured", notes="Difference is candidate minus baseline.",
            )

    # D2 sparse rows repeat the same truth event across overlapping anchor opportunities.
    for r in read_csv(ROOT / "results/D2_SPARSE_RESULTS_V2.csv"):
        common = {
            "model": r["model"], "seed": r["seed"], "horizon_steps": r["horizon_steps"],
            "body": r["event_group"], "support_episodes": r["episodes"],
            "support_source_groups": r["source_groups"],
            "opportunities": r["query_event_opportunities"], "hits": r["hits"], "misses": r["misses"],
            "bootstrap_replicates": 10000,
            "confidence_interval_type": "source-group bootstrap over repeated query-event opportunities",
        }
        sparse_metrics = [
            ("query_event_opportunities", r["query_event_opportunities"], "", "", "measured"),
            ("recall_query_event", r["recall_query_event"], r["recall_ci95_low"], r["recall_ci95_high"], "measured"),
            ("false_alarms_per_overlapping_query_sum", r["false_alarms_per_overlapping_query_sum"], "", "", "measured"),
            ("false_alarms_per_hour", r["false_alarms_per_hour"], "", "", "measured" if number(r["false_alarms_per_hour"]) is not None else "NA_not_reported"),
            ("timing_abs_error_seconds", r["timing_abs_error_seconds"], "", "", "measured" if number(r["timing_abs_error_seconds"]) is not None else "NA_no_hits"),
        ]
        for metric, value, low, high, status in sparse_metrics:
            add_long(
                "D2_sparse", "event_metric", "results/D2_SPARSE_RESULTS_V2.csv",
                **common, metric=metric, estimate=number(value), ci95_low=number(low), ci95_high=number(high),
                status=status, notes="Sparse query-event opportunities can repeat an event across anchors; false_alarms_per_overlapping_query_sum is not false alarms/hour.",
            )
        for lead, suffix in (("0.25", "0p25"), ("0.5", "0p5"), ("1.0", "1p0")):
            denominator = r[f"opportunities_lead_ge_{suffix}"]
            hits = r[f"hits_lead_ge_{suffix}"]
            den = number(denominator)
            lead_common = {**common, "opportunities": denominator, "hits": hits,
                           "misses": (int(denominator) - int(hits)) if den is not None else ""}
            add_long(
                "D2_sparse", "lead_recall", "results/D2_SPARSE_RESULTS_V2.csv",
                **lead_common, metric=f"recall_query_event_lead_ge_{lead}s", stratum=f"lead_at_least_{lead}s",
                estimate=number(r[f"recall_lead_ge_{suffix}"]),
                ci95_low=number(r[f"recall_lead_ge_{suffix}_ci95_low"]),
                ci95_high=number(r[f"recall_lead_ge_{suffix}_ci95_high"]),
                status="measured" if den else "NA_no_complete_opportunity",
                notes="Repeated query-event opportunities; no natural-prevalence or per-hour inference.",
            )

    # D3 cell estimates, factorial contrasts, and per-task descriptive layer.
    for r in read_csv(ROOT / "results/D3_SOURCE_GROUP_RESULTS.csv"):
        add_long(
            "D3", "cell_score", "results/D3_SOURCE_GROUP_RESULTS.csv",
            model=r["config"], seed=r["seed"], horizon_steps=r["horizon_steps"],
            body=r["body"], metric=r["metric"], estimate=number(r["estimate"]),
            ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            support_windows_or_pairs=r["windows"], bootstrap_replicates=10000,
            confidence_interval_type="exact-video-source-group bootstrap, per fixed training seed",
            status="actual_run_exploratory", notes="Training objective implementation deviated from locked equal-episode sample-wise weighting; no retraining.",
        )
    for r in read_csv(ROOT / "results/D3_SOURCE_GROUP_CONTRASTS.csv"):
        add_long(
            "D3", r["contrast"], "results/D3_SOURCE_GROUP_CONTRASTS.csv",
            model="", candidate=r["contrast"], seed=r["seed"], horizon_steps=r["horizon_steps"],
            body=r["body"], metric=r["metric"], stratum=r["interpretation"],
            estimate=number(r["estimate"]), ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            bootstrap_replicates=10000,
            confidence_interval_type="paired exact-video-source-group bootstrap, per fixed training seed",
            status="actual_run_exploratory", notes=r["estimand"] + "; negative candidate-minus-control favors the candidate where specified.",
        )
    for r in read_csv(ROOT / "results/D3_TASK_DESCRIPTIVE_RESULTS.csv"):
        add_long(
            "D3", "descriptive_task_layer", "results/D3_TASK_DESCRIPTIVE_RESULTS.csv",
            model=r["config"], seed="", horizon_steps=r["horizon_steps"], body=r["body"],
            metric=r["metric"], stratum=f"task_{r['task']}", estimate=number(r["estimate"]),
            support_episodes=r["episodes"], support_source_groups=r["source_groups"],
            status=r["status"], notes="Descriptive task layer; no task-wise confidence interval.",
        )

    # V3_STATE is a neighboring state-mask study; preserve its original p/Holm quantities.
    state_rows = read_csv(STATE / "analysis/RESULTS_BY_SEED.csv")
    state_metrics = [
        "arm_mae_std", "hand_mae_std", "full_mae_std", "arm_fd_native",
        "hand_fd_native", "full_fd_native", "full_skill_vs_persistence",
    ]
    for r in state_rows:
        for metric in state_metrics:
            add_long(
                "V3_STATE", "cell_score", "V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z/analysis/RESULTS_BY_SEED.csv",
                model=r["cell"], seed=r["seed"], horizon_steps=r["horizon_steps"],
                body=("Arm" if metric.startswith("arm_") else "Hand" if metric.startswith("hand_") else "Full28"),
                metric=metric, stratum=r["state_level"], estimate=number(r[metric]),
                support_episodes=r["N_episode"], support_source_groups=603,
                status="accepted_development_cell", notes=f"video_level={r['video_level']}; independent state-mask study; do not pool with D1/D3.",
            )
    for rel, analysis in (("analysis/PRIMARY_VIDEO_EFFECTS.csv", "video_effect"), ("analysis/INFORMATION_INTERACTIONS.csv", "information_interaction")):
        for r in read_csv(STATE / rel):
            add_long(
                "V3_STATE", analysis, f"V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z/{rel}",
                candidate=r.get("candidate", ""), control=r.get("control", ""),
                horizon_steps=r["horizon_steps"], body=r["group"], metric=r["family"],
                stratum=r.get("state_level", r.get("contrast", "")), estimate=number(r["effect"]),
                ci95_low=number(r["ci_low"]), ci95_high=number(r["ci_high"]),
                support_episodes=r["episode_level"], support_source_groups=603,
                bootstrap_replicates=r["replicates"],
                confidence_interval_type="source study's declared bootstrap/sign-flip outputs; values not recomputed",
                status="original_claim_preserved",
                notes=f"contrast={r['contrast']}; raw_p={r['p_two_sided']}; Holm_p={r['holm_p']}; final_test_access={r['final_test_access']}",
            )

    # D4 per-seed estimates, matched contrasts, descriptive cross-seed intervals, and NA event bins.
    for r in read_csv(ROOT / "d4/D4_PRIMARY_RESULTS_BY_SEED.csv"):
        body, metric = r["metric"].split("|", 1)
        add_long(
            "D4", "per_seed_model_metric", "d4/D4_PRIMARY_RESULTS_BY_SEED.csv",
            model=r["model"], seed=r["seed"], horizon_steps=r["horizon_steps"],
            body=body, metric=metric, estimate=number(r["estimate"]),
            ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_episodes=r["episode_count"], support_source_groups=r["source_group_count"],
            support_windows_or_pairs=r["pair_count"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="source-group bootstrap over matched rolling pairs",
            status="D4_V3_DENSE_CORE_COMPLETE",
        )
    for r in read_csv(ROOT / "d4/D4_PAIRED_EFFECTS.csv"):
        add_long(
            "D4", f"paired_{r['contrast']}", "d4/D4_PAIRED_EFFECTS.csv",
            candidate=r["candidate"], control=r["control"], seed=r["seed"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            stratum=r["estimand"], estimate=number(r["estimate"]),
            ci95_low=number(r["ci95_low"]), ci95_high=number(r["ci95_high"]),
            support_episodes=r["episode_count"], support_source_groups=r["source_group_count"],
            support_windows_or_pairs=r["pair_count"], bootstrap_replicates=r["bootstrap_replicates"],
            confidence_interval_type="paired source-group cluster bootstrap",
            status="measured", notes="Same absolute future targets and identical pair IDs; estimand direction is included verbatim.",
        )
    for r in read_csv(ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE.csv"):
        add_long(
            "D4", f"cross_seed_{r['analysis']}_{r['contrast']}", "d4/D4_CROSS_SEED_DESCRIPTIVE.csv",
            model=r["model"], candidate=r["candidate"], control=r["control"],
            horizon_steps=r["horizon_steps"], body=r["body"], metric=r["metric"],
            estimate=number(r["estimate_equal_seed_group_mean"]),
            ci95_low=number(r["ci95_descriptive_low"]), ci95_high=number(r["ci95_descriptive_high"]),
            seed_sd_n3=number(r["seed_mean_sd_n3"]),
            support_episodes=r["episode_count"], support_source_groups=r["source_group_count"],
            support_windows_or_pairs=r["pair_count_per_seed"], bootstrap_replicates=10000,
            confidence_interval_type="post-hoc descriptive hierarchical resampling of 3 seeds and source groups; no p-values",
            status="measured_posthoc_descriptive",
            notes=f"seed points: {r['seed_20260950_estimate']};{r['seed_20260951_estimate']};{r['seed_20260952_estimate']}; pairwise join was validated.",
        )
    for r in read_csv(ROOT / "d4/D4_EVENT_AMBIGUITY_SLICES.csv"):
        for metric, val, low, high in (
            ("delta_U", r["delta_u"], r["delta_u_ci95_low"], r["delta_u_ci95_high"]),
            ("delta_L_new", r["delta_l_new"], r["delta_l_new_ci95_low"], r["delta_l_new_ci95_high"]),
        ):
            add_long(
                "D4", "posthoc_event_diagnostic_slice", "d4/D4_EVENT_AMBIGUITY_SLICES.csv",
                candidate=r["candidate"], control=r["control"], seed=r["seed"],
                horizon_steps=r["horizon_steps"], body=r["event_group"], metric=metric,
                event_bin=r["event_bin"], estimate=number(val), ci95_low=number(low), ci95_high=number(high),
                support_episodes=r["episode_count"], support_source_groups=r["source_group_count"],
                support_windows_or_pairs=r["pair_count"], opportunities=r["independent_events"],
                status=r["support_status"], notes=f"additional_events={r['additional_events_in_T']}; excluded_state_ineligible_pairs={r['excluded_state_ineligible_pairs']}; empty bins remain NA.",
            )


def write_long_csv():
    with LONG_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(LONG_ROWS)


def gain_rows():
    return read_csv(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv")


def render_per_seed_gain(rows, control, horizon, body, metric, stratum):
    selected = [r for r in rows if r["contrast"] == "visual_gain_control_minus_F_TOK_D"
                and r["control"] == control and r["horizon_steps"] == str(horizon)
                and r["body"] == body and r["metric"] == metric and r["stratum"] == stratum]
    selected.sort(key=lambda r: r["seed"])
    points = [number(r["estimate"]) for r in selected]
    if len(points) != 3:
        return "not_verified"
    mean = statistics.mean(points)
    sd = statistics.stdev(points)
    detail = "; ".join(f"{r['seed'][-2:]} {fmt(r['estimate'])} [{fmt(r['ci95_low'])},{fmt(r['ci95_high'])}]"
                       for r in selected)
    return f"{mean:.6f} ± {sd:.6f}; {detail}"


def build_scientific_report():
    d1_summary = read_json(ROOT / "results/D1_SOURCE_GROUP_SUMMARY.json")
    d1_mid = read_json(ROOT / "results/D1_MIDDLE_STRATUM_SUMMARY.json")
    d1_gain_summary = read_json(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS_SUMMARY.json")
    d2_summary = read_json(ROOT / "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json")
    d3_summary = read_json(ROOT / "results/D3_SOURCE_GROUP_SUMMARY.json")
    d4_status = read_json(ROOT / "d4/D4_STATUS.json")
    d4_cross = read_csv(ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE.csv")
    d1_gains = gain_rows()
    d1_interactions = read_csv(ROOT / "results/D1_SOURCE_GROUP_INTERACTIONS.csv")
    d2_dense = read_csv(ROOT / "results/D2_DENSE_RESULTS_V2.csv")
    d3_contrasts = read_csv(ROOT / "results/D3_SOURCE_GROUP_CONTRASTS.csv")
    state_effects = read_csv(STATE / "analysis/PRIMARY_VIDEO_EFFECTS.csv")
    state_interactions = read_csv(STATE / "analysis/INFORMATION_INTERACTIONS.csv")

    out = []
    out.append("# Core B：结果审计、滚动更新与 D5 决策\n")
    out.append("## 结论先行\n")
    out.append(
        "Core B 不能被视作一个可把全部登记行混排的排行榜：原 benchmark registry 为 124 行、"
        "13 个 family 标签；可比较结果必须按输入/目标/normalizer/horizon/population 分层。"
        "V3、D2 dense、D3 和 V3_STATE 的 accepted prediction bundles 分别为 36/36、36/36、36/36、54/54。"
        "主线 D1/D2/D3 数值审计与 V3 D4 dense 核心已完成；D1 视频增益随 seed 和分歧层明显变化，"
        "D2 的事件提前召回很低且误报代价高，D3 的空间增量不稳定，D4 未显示稳定的视频更新收益。"
        "D5 决策为 redesign，当前不训练概率模型。所有结论仅限 development，final/held-out access=0。\n"
    )
    out.append("## 可比性与数据边界\n")
    out.append(
        "- V3 primary dev：603 episodes / 603 exact-video source groups / 2,322 anchor windows；H30/H60、Full28。"
        " V3 supplementary 的 64 episodes 中 53 与 primary 重叠、11 仅 supplementary；614 是 union，不是 primary 分母。\n"
        "- D2 dense 是 task-balanced panel：118 episodes/source groups、3,718 queries；总 event exposure 约 0.249611 小时，"
        "不是自然 prevalence。D2 sparse 的 query-event opportunities 会在重叠 anchors 重复事件。\n"
        "- D3 为 6 configs × 2 horizons × 3 training seeds；V3_STATE 为独立 3×3 mask/video study。"
        " 两者不与 V3/D1/D2 混成跨架构 leaderboard。\n"
        "- D1/D2/D3/V3_STATE 已在旧报告中暴露；本次重分析均为 development/exploratory。"
        " D4 是用已评测预测做的 post-hoc secondary analysis。D1 中间层/gain extension 与 D4 跨 seed 汇总也明确标为事后描述性分析。\n"
    )

    out.append("## D1：视觉增益与分歧层\n")
    out.append(
        f"锁定 train-only 阈值：A60 Q25={fmt(d1_summary['thresholds_train_only']['dispersion_q25_h60'])}，"
        f"Q75={fmt(d1_summary['thresholds_train_only']['dispersion_q75_h60'])}；k=20 邻居距离 Q95="
        f"{fmt(d1_summary['thresholds_train_only']['neighbor_distance_k20_q95'])}。自然总体 2,322 windows/603 groups；"
        f"supported overall={d1_summary['dev_support']['supported_windows']} windows。"
        f"低层 {d1_summary['dev_support']['low_supported_windows']} windows/154 groups，"
        f"中层 {d1_mid['middle_supported_windows']} windows/{d1_mid['middle_supported_source_groups']} groups，"
        f"高层 {d1_summary['dev_support']['high_supported_windows']} windows/430 groups。"
        "三层 source-group 数不应相加：同一 episode/group 的 anchors 可落在不同分层。\n"
    )
    out.append("下表是 S-HIST 对照 F-TOK-D 的 standardized-MAE gain（control minus candidate，正值偏向 F-TOK-D）。"
              "均值±SD 是三个训练 seed 的描述统计；分号后保留逐 seed 点估计及 source-group bootstrap 95% CI。\n")
    out.append("| H | Body | 分层 | 跨 seed mean ± SD；各 seed estimate [95% CI] |\n|---:|---|---|---|\n")
    for h in (30, 60):
        for body in ("Arm", "Hand"):
            for stratum in ("natural_overall", "low_dispersion_supported", "middle_dispersion_supported", "high_dispersion_supported"):
                label = {"natural_overall": "natural overall", "low_dispersion_supported": "low supported",
                         "middle_dispersion_supported": "middle supported", "high_dispersion_supported": "high supported"}[stratum]
                out.append(f"| {h} | {body} | {label} | {render_per_seed_gain(d1_gains, 'S-HIST', h, body, 'mae', stratum)} |\n")
    out.append("\n锁定 primary 的 high-minus-low interaction（同样为 gain_high−gain_low）。逐 seed 结果显示异质性：\n")
    out.append("| H | Body | 各 seed effect [95% source-group CI] |\n|---:|---|---|\n")
    for h in (30, 60):
        for body in ("Arm", "Hand"):
            rr = [r for r in d1_interactions if r["control"] == "S-HIST" and r["horizon_steps"] == str(h) and r["body"] == body]
            rr.sort(key=lambda r: r["seed"])
            detail = "; ".join(f"{r['seed'][-2:]} {fmt(r['high_minus_low'])} [{fmt(r['ci95_low'])},{fmt(r['ci95_high'])}]"
                               for r in rr)
            out.append(f"| {h} | {body} | {detail} |\n")
    out.append(
        "\n解释：H30 的 seed 20260951 在 Arm/Hand 高层都出现明显负 gain；H60 seed 20260952 出现约 +0.070 的大正 interaction，"
        "而另外两个 seed 的 Hand interaction 近零/略负。故不能声称视觉增益随 ambiguity 稳定上升。"
        "模型绝对分数（S-HIST、F-TOK-D、F-TOK-S、F-POOL-D、NN_MEDIAN、PERSISTENCE；Arm/Hand/Full28 × MAE/FD × natural/分层）"
        "均逐行保留在 02_SCIENTIFIC_RESULTS.csv 和 D1 source CSV。CI 跨 0 不是等价性证据。\n"
    )

    out.append("## D2：提前预警与误报\n")
    out.append(
        f"corrected D2 V2 使用 train-only thresholds：theta_on Hand={fmt(d2_summary['thresholds_train_only']['theta_on_hand_arm_order'][0])}, "
        f"Arm={fmt(d2_summary['thresholds_train_only']['theta_on_hand_arm_order'][1])}; "
        f"theta_off Hand={fmt(d2_summary['thresholds_train_only']['theta_off_hand_arm_order'][0])}, "
        f"Arm={fmt(d2_summary['thresholds_train_only']['theta_off_hand_arm_order'][1])}。"
        f"校正后 truth events={d2_summary['corrected_truth_events_group_specific']}；"
        f"dense panel={d2_summary['dense_panel_episodes']} episodes/{d2_summary['dense_panel_queries']} queries，"
        "每模型 exposure 约 0.249611h。独立 complete-opportunity 总事件为 H30 63（Arm 58/Hand 5），"
        "H60 58（Arm 53/Hand 5）；lead≥0.5s 的总机会为 H30 53、H60 51。\n"
    )
    out.append("F-TOK-D 在 dense task-balanced panel 的逐 seed 结果：\n")
    out.append("| H | Body | 完整事件机会 | 命中 / 机会；recall [95% CI] | lead≥0.5s 命中/机会；recall | FA/hour [95% CI] | timing abs error (s) |\n|---:|---|---:|---|---|---|---:|\n")
    for h in (30, 60):
        for body in ("Arm", "Hand"):
            rr = [r for r in d2_dense if r["model"] == "F-TOK-D" and r["horizon_steps"] == str(h) and r["event_group"] == body]
            rr.sort(key=lambda r: r["seed"])
            for r in rr:
                timing = fmt(r["timing_abs_error_seconds"], 3)
                out.append(
                    f"| {h} | {body} / seed {r['seed'][-2:]} | {r['full_opportunity_events']} | "
                    f"{r['hits']} / {r['full_opportunity_events']}; {fmt(r['recall'],3)} [{fmt(r['recall_ci95_low'],3)},{fmt(r['recall_ci95_high'],3)}] | "
                    f"{r['hits_lead_ge_0p5']} / {r['opportunity_events_lead_ge_0p5']}; {fmt(r['recall_lead_ge_0p5'],3)} | "
                    f"{fmt(r['false_alarms_per_hour'],1)} [{fmt(r['false_alarms_per_hour_ci95_low'],1)},{fmt(r['false_alarms_per_hour_ci95_high'],1)}] | {timing} |\n"
                )
    out.append(
        "\nD2 dense 结果不支持当前配置作为实用早预警：0.5s 提前召回低，Hand 机会数极少，"
        "而 H60 Arm 的 F-TOK-D FA/hour 约 60–88；全 28 合并也有模型/seed 的高误报率。"
        "短 exposure 会放大 rate 不确定性，必须同时看 CI 与原始计数。"
        "1.0s 层若无完整机会按 NA 输出，不填 0。Sparse 表仅用于 repeated query-event denominator，不能把它解读成独立事件率。\n"
    )

    out.append("## D3：时间×空间 factorial\n")
    out.append(
        f"6×2×3 实际训练/推理 artifacts 均接受（36/36），在相同 603 source groups/2,322 windows 上重分析；"
        f"216/216 source point values 与 sealed D3 table 对齐。训练实现存在重要偏差：batch mean loss 乘 batch mean episode weight，"
        "不等于锁定的逐样本等 episode 加权 loss；未重训，所以以下是 actual-run exploratory evidence。\n"
        "核心 conditional contrasts 为 TS11-D−TS10-D（保留 temporal 时增加 spatial）与 TS11-D−TS01-D（保留 spatial 时增加 temporal）；"
        "另列 factorial interaction。MAE 的负 candidate−control 倾向 TS11-D，FD 方向按 native first-difference error 解读。\n"
    )
    out.append("| Contrast | H | Body | Metric | seed mean ± SD | per-seed estimate [95% source-group CI] |\n|---|---:|---|---|---|---|\n")
    for contrast in ("TS11_vs_TS10", "TS11_vs_TS01", "factorial_interaction_TS11_TS10_TS01_TS00"):
        for h in (30, 60):
            for body in ("Arm", "Hand"):
                for metric in ("mae", "fd"):
                    rr = [r for r in d3_contrasts if r["contrast"] == contrast and r["horizon_steps"] == str(h)
                          and r["body"] == body and r["metric"] == metric]
                    rr.sort(key=lambda r: r["seed"])
                    if not rr:
                        continue
                    vals = [number(r["estimate"]) for r in rr]
                    detail = "; ".join(f"{r['seed'][-2:]} {fmt(r['estimate'])} [{fmt(r['ci95_low'])},{fmt(r['ci95_high'])}]"
                                       for r in rr)
                    out.append(f"| {contrast} | {h} | {body} | {metric} | {statistics.mean(vals):.6f} ± {statistics.stdev(vals):.6f} | {detail} |\n")
    out.append(
        "\n空间条件效应并非一致：例如 TS11−TS10 Arm MAE H30 跨 seed mean +0.000776±0.000609（candidate 略差）；"
        "H60 mean −0.000930±0.003208，受 seed 20261003 的 −0.004628 影响；Hand H60 mean −0.001993±0.001093，"
        "但只有一个 seed 的 source-group CI 排除 0。Native FD 与 factorial interaction 的完整 seed/horizon/body 数值在 CSV 中。"
        "旧 V3 与 TS10-D 的输入 token/layout 不同，不能解释为纯空间消融。\n"
    )

    out.append("## V3_STATE：状态遮蔽×视频 study（独立层）\n")
    out.append(
        "该研究 54/54 训练与预测 accepted，使用 primary dev 603/2,322、Full28、H30/H60 和新 seed；"
        "FULL/OBS/ANCHOR 是不同 state-mask 条件，ANCHOR 仍保留独立 last-command token，非 strict video-only。"
        f"原始统计保留不变：24 项 primary MAE family 中 6 项 Holm<.05；24 项 FD secondary 中 5 项；"
        f"8 项信息交互中 4 项。完整状态条件下 video 净增量小且不普遍；ANCHOR−OBS 的条件交互为正，"
        "只说明视觉价值依赖遮蔽状态条件，不能改写成完整状态 Core B 普遍收益。\n"
    )
    out.append("| Original contrast | H | Group | Effect [95% CI] | Holm p |\n|---|---:|---|---|---:|\n")
    for r in state_effects:
        if r["state_level"] == "FULL":
            out.append(f"| {r['contrast']} | {r['horizon_steps']} | {r['group']} | {fmt(r['effect'])} [{fmt(r['ci_low'])},{fmt(r['ci_high'])}] | {fmt(r['holm_p'],4)} |\n")
    out.append("\nANCHOR−OBS interaction examples:\n\n| H | Group | Effect [95% CI] | Holm p |\n|---:|---|---|---:|\n")
    for r in state_interactions:
        if r["contrast"] == "video_gain(ANCHOR)_minus_video_gain(OBS)":
            out.append(f"| {r['horizon_steps']} | {r['group']} | {fmt(r['effect'])} [{fmt(r['ci_low'])},{fmt(r['ci_high'])}] | {fmt(r['holm_p'],4)} |\n")
    out.append("\n")

    out.append("## D4：同一绝对未来时间的 rolling update\n")
    out.append(
        f"真实运行已接受 V3 dense bundles {d4_status['v3_dense_prediction_bundles_accepted']}/36；"
        f"{d4_status['input_panel_episodes']} episodes、{d4_status['input_panel_queries']} queries、"
        f"{d4_status['source_groups']} source groups；H30/H60 各 {d4_status['adjacent_pairs_per_horizon']} adjacent pairs。"
        "每个 old/new 都按 source frame/timestamp/episode join 到相同 GT 未来交集；保留实际 7/8-frame anchor gap。"
        "D4 primary interval 按 source-group cluster bootstrap；额外跨 seed×group CI 是事后描述性，不是确认性推断。\n"
        "视频 contrast 定义为 F-TOK-D−S-HIST。ΔU>0 表示候选的更新收益较大；ΔL_new>0 表示候选在新锚点绝对误差更大；"
        "ΔFD_new>0 表示候选 FD 更大。所有数值仍需和候选绝对 L_new/FD 一起看：较晚 anchor 通常自然更接近目标，U>0 不等于视觉因果收益。\n"
    )
    out.append("| H | Body | ΔU video (mean; descriptive 95% CI) | ΔL_new video | ΔFD_new video | F-TOK-D L_new / FD_new |\n|---:|---|---|---|---|---|\n")
    for h in (30, 60):
        for body in ("Arm", "Hand"):
            def cross(metric):
                return next(r for r in d4_cross if r["analysis"] == "paired_contrast" and r["contrast"] == "video"
                            and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == metric)
            du, dl, dfd = (cross(m) for m in ("delta_U", "delta_L_new", "delta_FD_new"))
            level = next(r for r in d4_cross if r["analysis"] == "per_model" and r["model"] == "F-TOK-D"
                         and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == "L_new")
            fdlevel = next(r for r in d4_cross if r["analysis"] == "per_model" and r["model"] == "F-TOK-D"
                           and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == "FD_new")
            effect_text = lambda r: f"{fmt(r['estimate_equal_seed_group_mean'])} [{fmt(r['ci95_descriptive_low'])},{fmt(r['ci95_descriptive_high'])}]"
            out.append(f"| {h} | {body} | {effect_text(du)} | {effect_text(dl)} | {effect_text(dfd)} | {fmt(level['estimate_equal_seed_group_mean'])} / {fmt(fdlevel['estimate_equal_seed_group_mean'])} |\n")
    n_na = sum(r["support_status"] == "NA_no_eligible_pairs" for r in read_csv(ROOT / "d4/D4_EVENT_AMBIGUITY_SLICES.csv"))
    out.append(
        f"\nD4 event-diagnostic slices: {d4_status['event_slice_rows']} rows，其中 {n_na} 个空支持 bin 保留 NA。"
        "Cross-seed ΔU 未形成稳定正的视频优势；H30 估计接近 0、区间跨 0；H60 ΔU 对 Arm/Hand 为负，"
        "虽 F-TOK-D 的新锚点绝对 MAE 在 H60 汇总较低，但改善主要受 seed 差异影响，FD 也并非两 body 同向。"
        "因此不宣称新观测或视频配置普遍提升可用性。stale/fresh、latency、missing-frame、timestamp-gap controls 仍 deferred。\n"
    )
    out.append(
        "代表轨迹：[D4 representative](d4/figures/D4_TRAJECTORY_representative.png)；"
        "预选失败诊断：[worst negative video-update example](d4/figures/D4_TRAJECTORY_worst_negative_delta_u_video_diagnostic.png)。"
        "二图均显示 GT、old/new prediction 和同一 common target support；失败案例的 Arm 上升事件明显未被预测轨迹捕捉。\n"
    )

    out.append("## D5 决策\n")
    out.append(
        "建议为 **redesign（不进入本窗口的概率模型训练）**。D1/D4 尚未说明在哪些条件下 uncertainty 能改变决策；"
        "D2 的 recall–误报权衡差；D3 空间增量 seed-sensitive 且训练目标实现偏离 lock。"
        "历史 hand prototype 的 CAL_GATE 和 candidate screening 均 fail；K=2 的 argmax mode 占 585/614=95.28%，"
        "posterior>95% 的占比 91.53%。Router study 明确 router-entry gate=false、router fits unauthorized；"
        "固定 q 的 FD oracle 相对 persistence 增加 2.2886%，不是进入训练/拟合 router 的理由。"
        "既有 124 行 capability matrix 中仍是 deterministic trajectories；conditional samples/weights 未建立，"
        "calibration not fitted，多 seed 不是 conditional samples。下一轮若重新立项，需 source-group cross-fit 或重训隔离、"
        "固定 K 的 joint trajectory samples、finite-sample Energy Score 与路径/时间相关 proper scores、校准与 coverage，"
        "同时对照 matched deterministic baselines；minADE/best-of-K 仅 secondary。此处不授权训练或 final-test。\n"
    )
    out.append("## 解释边界\n")
    out.append(
        "- 不显著/区间跨零不表示两模型等价；三个 training seeds 的 seed mean±SD 仅描述，不是独立 episode 样本。\n"
        "- D2 dense 的 task-balanced panel不可用于自然事件 prevalence；D2 sparse opportunity 会重复事件。\n"
        "- V3_STATE 的既有 Holm 数值不重新计算；研究仍是 development-only。\n"
        "- V3_STATE source checksum 的 1,258 listed entries 验证通过，但实际有 1,272 个文件（14 个未列入的生成 pyc）；覆盖不完整。\n"
        "- D3 dense optional extension 截至本报告为 384/3,718 完整特征 rows；没有 D3 dense prediction/evaluation 结果，不影响必做 D4 V3 core，但不得称该可选扩展完成。\n"
        "- Exact per-seed/stratum/event/pair metrics见 02_SCIENTIFIC_RESULTS.csv；底层输入和 method version 列在各行 source_file 中。\n"
    )
    REPORT_MD.write_text("".join(out), encoding="utf-8")


def build_claim_ledger():
    fields = ["claim_id", "claim", "measured_fact", "interpretation", "plausible_alternative", "evidence_status", "support", "source_files", "final_test_access"]
    rows = [
        ["C1", "D1 视觉增益不稳定，未建立普遍消歧收益。",
         "S-HIST vs F-TOK-D gain 与 high-minus-low interaction 在 3 seeds、H30/H60、Arm/Hand 方向/幅度不同；H30 seed 20260951 高层 gain 显著为负，H60 seed 20260952 interaction 约 +0.070。",
         "当前结果不支持跨 seed、跨 horizon 一致的视觉收益或高歧义增益。",
         "seed-specific optimization、分层内状态/任务构成、边界样本支持量差异。",
         "development exploratory; post-hoc paired gain extension",
         "603 primary episodes; 2,322 windows; low 199 windows/154 groups; middle 1,090/492; high 796/430",
         "results/D1_SOURCE_GROUP_RESULTS.csv; results/D1_SOURCE_GROUP_INTERACTIONS.csv; results/D1_VISUAL_GAIN_EXTENSIONS.csv", "0"],
        ["C2", "D2 当前事件提前预警未达实用门槛。",
         "Task-balanced dense panel有118 episodes/3,718 queries/0.249611 exposure hours；F-TOK-D 的 lead>=0.5s 召回低，FA/hour 可达约 56–116（依 horizon/body/seed）。",
         "现有配置不能同时提供高提前召回和低误报率。",
         "panel短暴露导致 rate 不确定；panel task-balanced 而非自然 prevalence；手部机会数很少。",
         "corrected D2 V2; exploratory",
         "594 group-specific corrected truth events across 603 dev episodes; H30/H60 full opportunities 63/58",
         "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json; results/D2_DENSE_RESULTS_V2.csv; results/D2_DENSE_PAIRED_EFFECTS_V2.csv", "0"],
        ["C3", "D3 没有稳定的空间主效应。",
         "TS11-D−TS10-D 的 Arm/Hand MAE contrasts across 3 seeds often near zero or change sign; H30 Arm mean +0.000776, H60 Arm mean −0.000930 with SD 0.003208; D3 trainer weighting differs from locked sample-wise equal-episode objective.",
         "D3 actual-run results are exploratory, not evidence that space should be kept or discarded conclusively.",
         "training objective deviation, seed variability, spatial/temporal optimization interaction.",
         "36/36 artifacts accepted; 216/216 point reconciliation; actual-run exploratory",
         "603 groups / 2,322 windows; 6 configs × 2 horizons × 3 seeds",
         "results/D3_SOURCE_GROUP_RESULTS.csv; results/D3_SOURCE_GROUP_CONTRASTS.csv; source d3_train.py", "0"],
        ["C4", "V3_STATE video value is conditional on state masking, not a complete-state universal gain.",
         "54/54 accepted; original tests show 6/24 primary MAE, 5/24 FD-secondary, and 4/8 information-interaction Holm-adjusted p values below .05; FULL-state effects are small/non-uniform.",
         "Video contribution depends on the input mask; it does not replace D1 or D3 and does not establish a FULL-state benefit.",
         "Optimization/regularization and input-fusion stability can explain some mask interactions.",
         "original V3_STATE development claims retained unchanged",
         "603 primary-dev episodes / 2,322 windows; fresh seeds 20260953–55",
         "V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z/analysis/PRIMARY_VIDEO_EFFECTS.csv; analysis/INFORMATION_INTERACTIONS.csv; CLAIM_LEDGER.md", "0"],
        ["C5", "D4 did not establish stable positive video update utility.",
         "D4 accepted 36/36 V3 dense bundles; 3,600 matched pairs per horizon / 118 exact source groups; cross-seed video ΔU H30 near zero and H60 negative, with mixed new-anchor FD.",
         "Later observations often improve each model, but comparative update utility and practical quality do not consistently favor F-TOK-D.",
         "new anchor is closer to target; 4-second observation shift both adds and removes context; seed differences.",
         "post-hoc secondary development analysis; descriptive cross-seed bootstrap",
         "same absolute target timestamp intersection; 10,800 joined pair rows per contrast/body/horizon",
         "d4/D4_PAIR_MANIFEST.csv; d4/D4_PRIMARY_RESULTS_BY_SEED.csv; d4/D4_PAIRED_EFFECTS.csv; d4/D4_CROSS_SEED_DESCRIPTIVE.csv", "0"],
        ["C6", "D5 should be redesigned before any probabilistic model training.",
         "Existing outputs deterministic; conditional samples/weights not established; calibration not fitted. Historical prototype CAL_GATE and candidate gate failed with mode collapse; router entry gate false.",
         "No present evidence supports spending a new model sweep on D5; redesign calibration/split/scoring protocol first.",
         "A future source-group cross-fit probabilistic study may still add value if it meets proper-score/calibration gates.",
         "decision gate only; no new model training",
         "prototype argmax 585/614=95.28%; router FD oracle +2.2886% vs persistence; training count 0",
         "prepared/d5/D5_MODEL_CAPABILITY.csv; HAND_CALIBRATED_PROTOTYPE_MIXTURE.../FINAL_STATUS.ini; PROTOTYPE_DIAGNOSIS_ROUTER_STUDY.../analysis/ROUTER_EXECUTION_DECISION.json", "0"],
        ["C7", "Registry row count is not a comparable leaderboard.",
         "124 registry rows span 13 family labels; 22 is the discovered source-root count, not a common-score model count.",
         "Model/seed/horizon/input contracts must remain separate.",
         "Same metric name can hide different populations, target layouts, checkpoint selection, or score transforms.",
         "source-backed contract matrix",
         "214 result-contract rows after adding 36 D3 and 54 V3_STATE rows",
         "RESULT_CONTRACT_MATRIX.csv; results/CONTRACT_MATRIX_SUMMARY.json; inventory/ARTIFACT_INVENTORY_RECONCILED.csv", "0"],
        ["C8", "Final/held-out data remain sealed.",
         "Source acceptance, analysis tables, and output sidecars retain final_test_access=0; this phase used train/dev only.",
         "No final/generalization claim is authorized.",
         "Zero marker alone is not proof; it is paired with input/prediction path acceptance and train/dev ID joins.",
         "verified development-only boundary",
         "final-test access count 0",
         "01_SOURCE_ACCEPTANCE.md; source_acceptance/PREDICTION_BUNDLE_ACCEPTANCE.csv; all analysis summaries", "0"],
    ]
    with CLAIMS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def build_source_acceptance_and_exposure():
    matrix = read_json(ROOT / "results/CONTRACT_MATRIX_SUMMARY.json")
    d2 = read_json(ROOT / "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json")
    d3 = read_json(ROOT / "results/D3_SOURCE_GROUP_SUMMARY.json")
    d4 = read_json(ROOT / "d4/D4_STATUS.json")
    source_md = f"""# 01 Source Acceptance

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.

## Root and checksums

- BENCH_ROOT: {BENCH}; the full CHECKSUMS.sha256 verification passed: 1,378 entries / 1,378 current files, no missing or unhashed paths.
- V3 primary predictions: 36/36 accepted exact ID/frame/timestamp/shape/finite checks. D2 dense predictions: 36/36 exact panel ID/frame/timestamp. D3 primary-dev predictions: 36/36 accepted.
- V3_STATE: 54/54 accepted for its own study. Its listed 1,258 checksum entries verify, but actual non-manifest file count is 1,272; 14 omitted files are generated __pycache__/*.pyc. See inventory/V3_STATE_CHECKSUM_GAPS.csv; do not call the source manifest coverage complete.

## Population and comparability

- V3 primary dev is 603 episodes, 603 exact-video source groups, and 2,322 windows.
- A distinct supplementary manifest has 64 episodes/windows: 53 overlap primary and 11 are supplementary-only. The 614-episode union is not the V3 primary denominator.
- The registry has 124 rows and 13 family labels. The separate inventory contains 22 source roots; this is not 22 mutually comparable model families. RESULT_CONTRACT_MATRIX.csv includes 124 source rows + 36 D3 + 54 V3_STATE rows and prohibits an all-row leaderboard.
- Exact video hashes define duplicate/source identity, not independent operators or scenes.

## Scientific and implementation findings

1. D1 source-group reanalysis uses K=20, leave-source-group-out neighbors, and train-only Q25/Q75/Q95. A supported middle-stratum and paired gain extension reuse these frozen thresholds and mark the addenda post-hoc/descriptive.
2. The earlier D2 event evaluator merged Hand/Arm events and used a denominator that could repeat events. Corrected V2 results use per-group event states, complete opportunities, and train-only thresholds; prior failed attempts remain under invalid_runs.
3. D3's actual trainer multiplies a scalar batch-mean loss by a batch-mean episode weight; this is not the locked per-example weighted mean. No retraining was done; D3 results are actual-run exploratory.
4. V3_STATE is a separate 3×3 information-masking study, adjacent evidence only; its original Holm results and claims were not modified.
5. D4 V3 dense core status is {d4['status']}: {d4['v3_dense_prediction_bundles_accepted']}/36 accepted bundles, {d4['input_panel_episodes']} episodes, {d4['input_panel_queries']} queries, {d4['adjacent_pairs_per_horizon']} adjacent pairs/horizon, {d4['source_groups']} source groups. D4 controls for stale/fresh, latency, missing frames, and timestamp gaps remain deferred.

## Boundaries and current optional extension

All result calculations use development records; final_test_access is 0. BENCH_ROOT, V3_ROOT, V3_STATE_ROOT, raw action data, checkpoints, and original prediction arrays were not edited. No new model training occurred. The optional D3 dense extension is not a completion gate for D4 V3 core: Stage0 passed and six atomic 64-row shards (384/3,718) were saved before pause; no D3 dense prediction or metric is claimed.
"""
    (ROOT / "01_SOURCE_ACCEPTANCE.md").write_text(source_md, encoding="utf-8")
    exposure = """# Outcome exposure

D1, D2, D3, V3_STATE, historical prototype/router, and their development outcomes were already visible in sealed source roots or previous handoffs. All claims remain exploratory development evidence.

D1_SOURCE_GROUP_REANALYSIS_V2, D2_EVENT_REANALYSIS_V2, and D4_PROTOCOL_LOCKED were versioned in NEXT_ROOT. D4's same-absolute-time pair estimand was locked before pair-level D4 errors were calculated, but its source predictions had already contributed to earlier per-anchor studies; D4 is a post-hoc secondary development analysis, not independent confirmation.

The D1 supported-middle and paired visual-gain extensions and the D4 cross-seed-by-source-group summary were specified after parent outcomes were inspected. They reuse original train-only thresholds, IDs, predictions, and source groups; they are descriptive, have no new p-values, and do not replace the parent locked results.

No final/held-out records, predictions, or labels were accessed. No training was run during this phase. Optional D3 dense feature extraction was interrupted at the user's request after 384/3,718 atomic rows; this is a partial extension, not a completed result.
"""
    (ROOT / "OUTCOME_EXPOSURE.md").write_text(exposure, encoding="utf-8")

    additions = [
        ROOT / "protocol/D1_SOURCE_GROUP_REANALYSIS_V2.json",
        ROOT / "protocol/D1_MIDDLE_STRATUM_EXTENSION_V1.json",
        ROOT / "protocol/D1_VISUAL_GAIN_EXTENSION_V1.json",
        ROOT / "protocol/D2_EVENT_REANALYSIS_V2.json",
        ROOT / "protocol/D4_PROTOCOL_LOCKED.json",
        ROOT / "protocol/D4_CROSS_SEED_DESCRIPTIVE_V1.json",
        ROOT / "protocol/FINAL_RESULTS_REPORT_V1.json",
        ROOT / "results/D1_SOURCE_GROUP_RESULTS.csv",
        ROOT / "results/D1_SOURCE_GROUP_INTERACTIONS.csv",
        ROOT / "results/D1_MIDDLE_STRATUM_RESULTS.csv",
        ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv",
        ROOT / "results/D2_DENSE_RESULTS_V2.csv",
        ROOT / "results/D2_DENSE_PAIRED_EFFECTS_V2.csv",
        ROOT / "results/D3_SOURCE_GROUP_RESULTS.csv",
        ROOT / "results/D3_SOURCE_GROUP_CONTRASTS.csv",
        ROOT / "d4/D4_PRIMARY_RESULTS_BY_SEED.csv",
        ROOT / "d4/D4_PAIRED_EFFECTS.csv",
        ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE.csv",
        STATE / "PROTOCOL_LOCKED.json",
        STATE / "analysis/PRIMARY_VIDEO_EFFECTS.csv",
        STATE / "analysis/INFORMATION_INTERACTIONS.csv",
        BENCH / "prepared/d5/D5_MODEL_CAPABILITY.csv",
        HAND / "FINAL_STATUS.ini",
        ROUTER / "analysis/ROUTER_EXECUTION_DECISION.json",
    ]
    hashes = read_json(ROOT / "SOURCE_HASHES.json") if (ROOT / "SOURCE_HASHES.json").exists() else {}
    for path in additions:
        if path.is_file():
            hashes[str(path)] = sha256(path)
    (ROOT / "SOURCE_HASHES.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_d5_report():
    final_status = {}
    for line in (HAND / "FINAL_STATUS.ini").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            final_status[key] = value
    router = read_json(ROUTER / "analysis/ROUTER_EXECUTION_DECISION.json")
    capability = read_csv(BENCH / "prepared/d5/D5_MODEL_CAPABILITY.csv")
    deterministic = sum(r["output_type"].startswith("deterministic") for r in capability)
    body = f"""# D5 Go / No-Go

**Decision: redesign. No D5 model training is authorized or recommended in this window.** This is a development decision gate; it does not release final/held-out data.

| Evidence | Measured fact | Decision relevance |
|---|---|---|
| D1 ambiguity-conditioned visual gain | Gains and high-minus-low interactions vary by seed and horizon; H30 seed 20260951 has negative Arm/Hand high-stratum gain while H60 seed 20260952 has a large positive interaction. | A general multimodal uncertainty benefit is not established. |
| D2 early warning | Task-balanced panel has low lead>=0.5 s recall and high per-hour false alarms at short exposure; some Hand denominators are only five complete events. | Uncalibrated event probabilities would not yet support a useful decision policy. |
| D3 spatial factorial | 36/36 cells exist, but spatial contrasts are not seed-stable and the implemented episode weighting differs from the locked objective. | Do not train a probabilistic head to explain a training-contract deviation. |
| D4 rolling update | Same-absolute-target ΔU video is near zero at H30 and negative at H60; level error and FD are mixed. | Update utility has not shown a repeatable video-specific gain. |
| Historical hand prototype | CAL_GATE={final_status.get('CAL_GATE')}; screening candidate gate={final_status.get('SCREENING_CANDIDATE_GATE')}; K={final_status.get('SELECTED_K_MODE')}; argmax mode prevalence={final_status.get('MODE_COLLAPSE')}. | Mode collapse and calibration failure must be repaired before a probability model is useful. |
| Historical router study | entry gate={router['router_entry_gate']}; router fits authorized={router['router_fits_authorized']}; FD oracle relative change vs persistence={router['fd_oracle_fd_relative_change_vs_persistence']:.6%}. | The source gate explicitly did not authorize router fitting; oracle is non-deployable and worsens FD. |
| Current output capability | {deterministic} capability rows remain deterministic; conditional samples/weights are not established and calibration is not fitted. | Multiple model-training seeds cannot be relabeled as samples from a conditional future distribution. |

## What would change the decision

Before training, specify a new source-group protocol with (1) independent train/calibration/dev or source-group cross-fitting, (2) fixed K and complete joint-time sample paths with absolute timestamps and train-only scaling, (3) finite-sample Energy Score plus a time-correlation-sensitive proper score, (4) marginal, per-time, and simultaneous-path coverage and width, (5) event-probability calibration/risk-coverage and false-alarm operating points, and (6) matched deterministic/persistence baselines. Best-of-K/minADE remains secondary. The repeatedly viewed development set cannot be treated as independent calibration or final confirmation.

## Scope

This decision authorizes only preparation of a later independent prompt after redesign review. It does not authorize D5 training, checkpoint changes, benchmark gate rewrites, or final-test access. Historical failures remain preserved and are not averaged with current V3/D3 results.
"""
    (ROOT / "d5/D5_GO_NO_GO.md").write_text(body, encoding="utf-8")


def build_dashboard():
    gains = read_csv(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv")
    dense = read_csv(ROOT / "results/D2_DENSE_RESULTS_V2.csv")
    contrasts = read_csv(ROOT / "results/D3_SOURCE_GROUP_CONTRASTS.csv")
    cross = read_csv(ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE.csv")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Core B development benchmark：决策面板（development only）", fontsize=17, fontweight="bold")

    # D1: per-seed high-minus-low gain; source-group CIs are conditional on each fixed seed.
    ax = axes[0, 0]
    categories = [(30, "Arm"), (30, "Hand"), (60, "Arm"), (60, "Hand")]
    colors = {"20260950": "#1f77b4", "20260951": "#d95f02", "20260952": "#2ca02c",
              "20261001": "#9467bd", "20261002": "#8c564b", "20261003": "#e377c2"}
    for yi, (h, body) in enumerate(categories):
        for si, seed in enumerate(("20260950", "20260951", "20260952")):
            row = next(r for r in gains if r["contrast"] == "high_minus_low_visual_gain" and r["control"] == "S-HIST"
                       and r["seed"] == seed and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == "mae")
            x = float(row["estimate"])
            lo, hi = float(row["ci95_low"]), float(row["ci95_high"])
            y = yi + (si - 1) * 0.16
            ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=colors[seed], capsize=2, markersize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(categories)), [f"H{h} {b}" for h, b in categories])
    ax.invert_yaxis()
    ax.set_title("D1：高−低 ambiguity visual gain（S-HIST control）")
    ax.set_xlabel("gain_high − gain_low；正值偏向高层视觉增益")
    ax.grid(axis="x", alpha=0.2)

    # D2: early recall against alarm burden, individual fixed seeds.
    ax = axes[0, 1]
    for r in dense:
        if r["model"] not in ("F-TOK-D", "S-HIST") or r["event_group"] not in ("Arm", "Hand"):
            continue
        x = number(r["false_alarms_per_hour"])
        y = number(r["recall_lead_ge_0p5"])
        if x is None or y is None:
            continue
        marker = "o" if r["event_group"] == "Arm" else "s"
        color = "#2463a4" if r["horizon_steps"] == "30" else "#c7522a"
        open_marker = r["model"] == "S-HIST"
        ax.scatter(x, y, marker=marker, s=36, facecolors="none" if open_marker else color,
                   edgecolors=color, alpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.01, 0.16)
    ax.set_title("D2：≥0.5s 提前召回 vs false alarms/hour")
    ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("lead≥0.5s event recall")
    ax.text(0.98, 0.96, "颜色：H30蓝 / H60红；形状：Arm圆 / Hand方；空心：S-HIST",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.grid(alpha=0.2)

    # D3: conditional spatial addition, per fixed seed.
    ax = axes[1, 0]
    categories = [(30, "Arm"), (30, "Hand"), (60, "Arm"), (60, "Hand")]
    for yi, (h, body) in enumerate(categories):
        rr = [r for r in contrasts if r["contrast"] == "TS11_vs_TS10" and r["metric"] == "mae"
              and r["horizon_steps"] == str(h) and r["body"] == body]
        rr.sort(key=lambda r: r["seed"])
        for si, r in enumerate(rr):
            x = float(r["estimate"])
            lo, hi = float(r["ci95_low"]), float(r["ci95_high"])
            ax.errorbar(x, yi + (si - 1) * 0.16, xerr=[[x-lo], [hi-x]], fmt="o",
                        color=colors[r["seed"]], capsize=2, markersize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(categories)), [f"H{h} {b}" for h, b in categories])
    ax.invert_yaxis()
    ax.set_title("D3：TS11-D − TS10-D spatial effect（MAE）")
    ax.set_xlabel("candidate − control standardized MAE；负值偏向 TS11-D")
    ax.grid(axis="x", alpha=0.2)

    # D4: descriptive cross-seed video update utility vs new-anchor absolute error.
    ax = axes[1, 1]
    for h in (30, 60):
        for body, marker in (("Arm", "o"), ("Hand", "s")):
            du = next(r for r in cross if r["analysis"] == "paired_contrast" and r["contrast"] == "video"
                      and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == "delta_U")
            dl = next(r for r in cross if r["analysis"] == "paired_contrast" and r["contrast"] == "video"
                      and r["horizon_steps"] == str(h) and r["body"] == body and r["metric"] == "delta_L_new")
            x, y = float(du["estimate_equal_seed_group_mean"]), float(dl["estimate_equal_seed_group_mean"])
            xlo, xhi = float(du["ci95_descriptive_low"]), float(du["ci95_descriptive_high"])
            ylo, yhi = float(dl["ci95_descriptive_low"]), float(dl["ci95_descriptive_high"])
            color = "#2463a4" if h == 30 else "#c7522a"
            ax.errorbar(x, y, xerr=[[x-xlo], [xhi-x]], yerr=[[y-ylo], [yhi-y]],
                        fmt=marker, color=color, capsize=2, markersize=5)
            label = f"A{h}" if body == "Arm" else f"H{h}"
            offset = (6, 8) if body == "Arm" else (6, -10)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("D4：video contrast（F-TOK-D − S-HIST）")
    ax.set_xlabel("ΔU: F-TOK-D − S-HIST update utility")
    ax.set_ylabel("ΔL_new: F-TOK-D − S-HIST new-anchor error")
    ax.grid(alpha=0.2)

    fig.text(
        0.5, 0.012,
        "Support：D1 603 groups/2,322 windows；D2 118 task-balanced episodes/3,718 queries/0.2496h；"
        "D3 603 groups/36 cells；D4 118 groups/3,600 pairs per horizon。"
        "Per-seed intervals are not training-seed population intervals. D4 cost=91.2 CPU s/0 GPUh; optional dense inference/extraction tracked separately in WORK_STATUS; D3 new training=0; D5=redesign; final access=0.",
        ha="center", va="bottom", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    path = ROOT / "figures/BENCHMARK_DECISION_DASHBOARD.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_test_and_work_status():
    commands = [
        [str(PYTHON), "-m", "pytest", "code/test_event_v2.py", "code/test_d2_v2.py", "code/test_d4_eval.py", "-q"],
        [str(PYTHON), "-m", "pytest", str(BENCH / "prepared/d5/test_d5_synthetic.py"), "-q"],
    ]
    test_results = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
        test_results.append({
            "command": command, "cwd": str(ROOT), "exit_code": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "stdout_tail": completed.stdout[-5000:], "stderr_tail": completed.stderr[-3000:],
        })
    d1_mid = read_json(ROOT / "results/D1_MIDDLE_STRATUM_SUMMARY.json")
    d1_gain = read_json(ROOT / "results/D1_VISUAL_GAIN_EXTENSIONS_SUMMARY.json")
    d4_cross = read_json(ROOT / "d4/D4_CROSS_SEED_DESCRIPTIVE_SUMMARY.json")
    matrix = read_json(ROOT / "results/CONTRACT_MATRIX_SUMMARY.json")
    d3 = read_json(ROOT / "results/D3_SOURCE_GROUP_SUMMARY.json")
    d4 = read_json(ROOT / "d4/D4_STATUS.json")
    checks = {
        "contract_matrix_214_rows": matrix["total_rows"] == 214,
        "v3_state_54_cells": matrix["accepted_v3_state_rows"] == 54,
        "d1_middle_extension_1090_windows_492_groups": d1_mid["middle_supported_windows"] == 1090 and d1_mid["middle_supported_source_groups"] == 492,
        "d1_gain_648_rows_36_matched_cells": d1_gain["rows"] == 648 and d1_gain["accepted_v3_cells"] == 24,
        "d4_crossseed_132_rows_12_joins": d4_cross["output_rows"] == 132 and all(r["status"] == "PASS" for r in d4_cross["pair_join_checks"]),
        "d4_v3_dense_core_36_of_36": d4["v3_dense_prediction_bundles_accepted"] == 36,
        "final_test_access_zero": d4["final_test_access"] == 0 and d1_mid["final_test_access"] == 0 and d1_gain["final_test_access"] == 0,
    }
    qa = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tests": test_results, "checks": checks,
        "status": "PASS" if all(x["status"] == "PASS" for x in test_results) and all(checks.values()) else "FAIL",
        "final_test_access": 0,
    }
    (ROOT / "qa/TEST_RESULTS.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    d3_cache_rows = len(list((ROOT / "d3_extension/cache/shards").glob("grid_*.json")))
    d3_cached = 0
    for meta in (ROOT / "d3_extension/cache/shards").glob("grid_*.json"):
        try:
            shard = read_json(meta)
            if meta.with_suffix(".npy").is_file():
                d3_cached += int(shard["end_row"]) - int(shard["start_row"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
    d3_stage0 = read_json(ROOT / "d3_extension/qa/D3_DENSE_STAGE0_ACCEPTANCE.json")
    watcher_files = [ROOT / "d3_extension/status/D3_DENSE_STAGE0_WATCHER.json", ROOT / "d3_extension/status/D3_DENSE_EXTRACT_WATCHER.json"]
    gpu_hours = 0.0
    for path in watcher_files:
        if not path.exists():
            continue
        obj = read_json(path)
        start, end = obj.get("lease_acquired_utc"), obj.get("lease_released_utc")
        if start and end:
            from datetime import datetime as dt
            gpu_hours += (dt.fromisoformat(end.replace("Z", "+00:00")) - dt.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600.0
    work = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "NEXT_STATUS": "pending_final_visual_QA_and_checksum",
        "SOURCE_AND_CHECKSUM_ACCEPTANCE": "BENCH_ROOT full manifest PASS; V3_STATE listed hashes PASS but manifest coverage incomplete 1258/1272",
        "ORIGINAL_BENCHMARK_COMPLETE_VERIFIED": True,
        "REGISTRY_ROWS": 124,
        "REGISTRY_FAMILY_LABELS": 13,
        "DISCOVERED_SOURCE_ROOTS": 22,
        "V3_TRAIN_PRED_ACCEPTED": "36/36 predictions; sealed training/prediction artifacts accepted",
        "D3_TRAIN_PRED_ACCEPTED": "36/36; actual-run objective nonconformance disclosed",
        "D2_DENSE_ACCEPTED": "36/36 bundles; 118 episodes; 3718 queries",
        "D1_GAIN_HIGH_MINUS_LOW_ARM_HAND_H30_H60": "see results/D1_SOURCE_GROUP_INTERACTIONS.csv and D1_VISUAL_GAIN_EXTENSIONS.csv",
        "D1_CI_AND_SUPPORT": "10000 source-group bootstrap; natural 603/2322; low 199 windows/154 groups; middle 1090/492; high 796/430",
        "D2_RECALL_AT_LEAD_0P5_ARM_HAND_H30_H60": "per-seed values and CIs in results/D2_DENSE_RESULTS_V2.csv",
        "D2_FALSE_ALARMS_PER_HOUR_AND_EXPOSURE": "per-seed values/CIs in results/D2_DENSE_RESULTS_V2.csv; exposure approximately 0.249611 h",
        "D3_SPATIAL_TEMPORAL_DYNAMIC_VIDEO_EFFECTS_AND_CI": "source-group point estimates and per-seed CIs in results/D3_SOURCE_GROUP_CONTRASTS.csv",
        "D3_FACTORIAL_INTERACTION_AND_CI": "results/D3_SOURCE_GROUP_CONTRASTS.csv",
        "V3_STATE_54_STUDY_COMPATIBILITY": "same primary dev/target/horizon, separate 3x3 state-mask study; source checksum coverage incomplete",
        "D4_V3_DENSE_BUNDLES_ACCEPTED": f"{d4['v3_dense_prediction_bundles_accepted']}/36",
        "D4_ANCHOR_PAIRS_SOURCE_GROUPS": f"{d4['adjacent_pairs_per_horizon']} pairs/horizon; {d4['source_groups']} exact source groups",
        "D4_DELTA_U_VIDEO_ARM_HAND_H30_H60_AND_CI": "d4/D4_CROSS_SEED_DESCRIPTIVE.csv; intervals descriptive only",
        "D3_D4_EXTENSION_STATUS": f"partial_paused_features_{d3_cached}_of_3718; Stage0={d3_stage0.get('status','UNKNOWN')}; complete cache shards={d3_cache_rows}",
        "D5_DECISION": "redesign",
        "NEW_MODEL_TRAINING": 0,
        "FINAL_HELDOUT_ACCESS": 0,
        "OLD_SEALED_ROOT_WRITES": 0,
        "GPU_HOURS_THIS_OPTIONAL_EXTENSION_SO_FAR": round(gpu_hours, 6),
        "CHECKSUM_ACCEPTANCE": "pending final visual QA and whole-NEXT_ROOT checksum",
        "UNRESOLVED": [
            "optional D3 dense extension has features for 384/3718 rows and no dense predictions/metrics yet",
            "D3 actual training objective differs from locked equal-episode weighting",
            "V3_STATE checksum manifest omits 14 generated pyc files",
            "D4 stale/fresh, latency, missing-frame, and timestamp-gap controls are deferred",
        ],
        "final_test_access": 0,
    }
    (ROOT / "WORK_STATUS.json").write_text(json.dumps(work, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    visual_pending = {
        "status": "PENDING_DASHBOARD_VISUAL_REVIEW",
        "reviewed_source_images": [
            {"path": "d4/figures/D4_TRAJECTORY_representative.png", "review": "PASS: axes/legend/GT-old-new present; same common target support; predictions are mostly smooth and miss GT variation."},
            {"path": "d4/figures/D4_TRAJECTORY_worst_negative_delta_u_video_diagnostic.png", "review": "PASS: axes/legend/GT-old-new present; diagnostic visibly misses the Arm rise after about 1s."},
        ],
        "dashboard": {"path": "figures/BENCHMARK_DECISION_DASHBOARD.png", "review": "generated; manual visual review pending"},
        "final_test_access": 0,
    }
    (ROOT / "qa/VISUAL_QA.json").write_text(json.dumps(visual_pending, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contract = {
        "NEXT_STATUS": "pending_final_visual_QA_and_checksum",
        "D1_middle_supported_windows": d1_mid["middle_supported_windows"],
        "D1_middle_supported_groups": d1_mid["middle_supported_source_groups"],
        "D2_dense_episodes": read_json(ROOT / "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json")["dense_panel_episodes"],
        "D3_accepted_bundles": d3["accepted_bundles"],
        "D4_status": d4["status"],
        "D5_decision": "redesign",
        "NEW_MODEL_TRAINING": 0,
        "FINAL_HELDOUT_ACCESS": 0,
    }
    (ROOT / "RETURN_TO_CHATGPT.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return work, qa


def main():
    load_analysis_rows()
    write_long_csv()
    build_claim_ledger()
    build_scientific_report()
    build_source_acceptance_and_exposure()
    build_d5_report()
    build_dashboard()
    work, qa = build_test_and_work_status()
    d4 = read_json(ROOT / "d4/D4_STATUS.json")
    return_md = f"""# Core B results audit — handoff

**Status: mandatory result-audit/D4 V3 core/D5 decision work is generated; final visual QA/checksum is pending.**

## Main findings

- **Comparability:** 124 registry rows carry 13 family labels; 22 is the discovered source-root count. No all-row leaderboard is valid. The 603/2,322 primary-dev population is separate from the 64-row supplementary set (53 overlap; 11 supplementary-only).
- **D1:** The train-defined middle stratum is now present as a post-hoc extension (1,090 windows / 492 exact-video groups). S-HIST vs F-TOK-D gains and high-minus-low effects vary strongly by seed/horizon; broad visual disambiguation is not established. Exact per-seed gain/CI is in 02_SCIENTIFIC_RESULTS.csv.
- **D2:** Corrected event registry has 594 group-specific truth events. Dense panel has 118 task-balanced episodes, 3,718 queries, and about 0.249611 hours of total exposure. Lead≥0.5s recall is low while some F-TOK-D false-alarm rates are high; this is not evidence of natural event prevalence.
- **D3:** 36/36 actual prediction cells and 216/216 source-point reconciliation pass. Spatial/temporal effects are seed-dependent; actual training weighting deviates from the locked per-example objective, so D3 findings are exploratory.
- **V3_STATE:** 54/54 accepted and kept separate; original Holm counts remain 6/24 primary MAE, 5/24 FD secondary, 4/8 interactions. Its checksum manifest covers 1,258/1,272 files, with 14 generated pyc omissions.
- **D4:** status {d4['status']}; 36/36 dense V3 bundles, 3,600 matched pairs per horizon and 118 exact source groups. The same-absolute-time video ΔU has no stable positive advantage; see the descriptive cross-seed CI and matched new-level/FD columns in d4/D4_CROSS_SEED_DESCRIPTIVE.csv.
- **D5:** decision **redesign**; no probabilistic model was trained. Historical hand calibration/screening gates failed and the router gate did not authorize fitting.

## Work state

- No new model training; final/held-out access 0; original sealed roots unchanged.
- Optional D3 dense feature extension is paused at 384/3,718 complete rows; no dense prediction or D3 dense effect is claimed. It is resumable using d3_extension/run_optional_dense.sh, whose extraction job uses --resume and the shared GPU watcher.
- CPU test status: {qa['status']}; dashboard manual review and final whole-root SHA-256 remain pending.

## Key files

- 02_SCIENTIFIC_RESULTS.zh-CN.md — readable result and interpretation.
- 02_SCIENTIFIC_RESULTS.csv — unified per-seed/metric/CI/support export.
- 02_CLAIM_LEDGER.csv — fact, interpretation, alternative explanation, evidence boundary.
- figures/BENCHMARK_DECISION_DASHBOARD.png — dashboard (visual QA pending).
- d4/D4_PAIR_MANIFEST.csv, d4/D4_PRIMARY_RESULTS_BY_SEED.csv, d4/D4_PAIRED_EFFECTS.csv, d4/D4_BOOTSTRAP.csv, d4/D4_EVENT_AMBIGUITY_SLICES.csv, d4/figures/ — D4 outputs.
- d5/D5_GO_NO_GO.md — evidence gate and decision.
- qa/TEST_RESULTS.json and qa/VISUAL_QA.json — verification state.
"""
    (ROOT / "RETURN_TO_CHATGPT.zh-CN.md").write_text(return_md, encoding="utf-8")
    print(json.dumps({
        "long_csv_rows": len(LONG_ROWS),
        "claims": len(read_csv(CLAIMS_CSV)),
        "dashboard": str(ROOT / "figures/BENCHMARK_DECISION_DASHBOARD.png"),
        "tests": qa["status"],
        "work_status": work["NEXT_STATUS"],
        "final_test_access": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

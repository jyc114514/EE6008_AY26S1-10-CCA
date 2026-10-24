# Repaired public copy of EXCL-CODE-006; source SHA-256: b5bfb770aaa5492a9c05278b966ad2fc7952cd65f184be2027a77dc5a2966db7
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path

PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
ROOT = Path(os.environ.get("EE6008_FUSION_QUERY_STUDY_ROOT", PROJECT / "reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z")).expanduser().resolve()
TASK = Path(os.environ.get("EE6008_FUSION_QUERY_TASK_ATTACHMENT", ".")).expanduser().resolve()
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
DIAG = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
IDS = [
    "TINYFIT-TS11-D-H30-FULL_LOSS-P1-S20261002",
    "TINYFIT-TS11-D-H30-FULL_LOSS-P8-S20261002",
    "TINYFIT-STATE-CTRL-H30-FULL_LOSS-P1-S20261002",
    "TINYFIT-STATE-CTRL-H30-FULL_LOSS-P8-S20261002",
]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")

learn_sa = read_json(LEARN / "SOURCE_ACCEPTANCE.json")
learn_fa = read_json(LEARN / "FINAL_ACCEPTANCE.json")
learn_qa = read_json(LEARN / "FINAL_QA.json")
sched_fa = read_json(SCHED / "FINAL_ACCEPTANCE.json")
sched_sa = read_json(SCHED / "SOURCE_ACCEPTANCE.json")
eq_status = read_json(EQ / "STATUS.json")
diag_fa = read_json(DIAG / "FINAL_ACCEPTANCE.json")
diag_qa = read_json(DIAG / "FINAL_QA.json")

sources = []
def add(path: Path, role: str, expected: str | None = None, basis: str | None = None):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"hash mismatch for {path}: {actual} != {expected}")
    item = {"role": role, "path": str(path), "bytes": path.stat().st_size,
            "observed_sha256": actual, "read_only": True}
    if expected is not None:
        item.update({"manifest_expected_sha256": expected, "expected_basis": basis,
                     "expected_hash_match": True})
    sources.append(item)

add(TASK, "current user task attachment")
add(PROJECT / "AGENTS.md", "project operating policy")
old_prompt = Path(learn_sa["original_task_prompt_path"])
if old_prompt.is_file():
    add(old_prompt, "historical prompt referenced by prior accepted source lock",
        learn_sa["original_task_prompt_sha256"], "LEARN_ROOT/SOURCE_ACCEPTANCE.json")

for name, item in learn_sa["source_file_hashes"].items():
    add(Path(item["path"]), "LEARN_ROOT locked source: " + name,
        item["sha256"], "LEARN_ROOT/SOURCE_ACCEPTANCE.json")

extra = [
    (LEARN/"A_FORENSIC_REPORT.md", "Phase A report"),
    (LEARN/"B_FINAL_REPORT.zh-CN.md", "prior train-only tiny-fit report"),
    (LEARN/"FINAL_ACCEPTANCE.json", "tiny-fit acceptance"),
    (LEARN/"FINAL_QA.json", "tiny-fit QA"),
    (LEARN/"CHECKSUMS.sha256", "tiny-fit checksum manifest"),
    (LEARN/"TRAIN_FORENSIC_FIXTURE.json", "locked train forensic fixture"),
    (LEARN/"protocol/B_PANEL_SELECTION_LOCK.json", "locked P1/P8 panels"),
    (LEARN/"protocol/B_PROTOCOL_LOCKED.json", "prior tiny-fit protocol"),
    (LEARN/"protocol/B_MATRIX_8.csv", "prior eight-cell matrix"),
    (LEARN/"results/TINYFIT_FINAL_RUN_SUMMARY.csv", "prior run summary"),
    (LEARN/"results/TINYFIT_FINAL_SAMPLE_METRICS.csv", "prior train sample metrics"),
    (LEARN/"results/TINYFIT_LEARNING_CURVES.csv", "prior train learning curves"),
    (LEARN/"code/run_tiny_fit.py", "accepted CPU tiny-fit implementation"),
    (LEARN/"code/collapse_diagnostic_core.py", "locked diagnostic metrics"),
    (LEARN/"forensics/A_REPLAY_4/A_REPLAY.json", "accepted Phase A replay"),
    (LEARN/"forensics/REPLAY_REPEATABILITY.json", "Phase A replay repeatability"),
    (SCHED/"02_SCIENTIFIC_RESULTS.zh-CN.md", "corrected-scheduler report"),
    (SCHED/"FORMAL_TRAINING_ACCEPTANCE.md", "corrected-scheduler training acceptance"),
    (SCHED/"FINAL_ACCEPTANCE.json", "corrected-scheduler acceptance"),
    (SCHED/"FINAL_REPORT_QA.json", "corrected-scheduler QA"),
    (SCHED/"SOURCE_ACCEPTANCE.json", "corrected-scheduler source acceptance"),
    (SCHED/"PROTOCOL_LOCKED.json", "corrected-scheduler locked protocol"),
    (SCHED/"OUTCOME_EXPOSURE.md", "scheduler outcome exposure"),
    (SCHED/"CHECKSUMS.sha256", "scheduler checksum manifest"),
    (SCHED/"code/d3_model.py", "locked D3 model"),
    (SCHED/"code/d3_eq_contract.py", "locked loss contract"),
    (SCHED/"code/d3_train_corrected.py", "train loader and corrected trainer"),
    (SCHED/"MOTION_ACTIVITY_THRESHOLD.json", "frozen train motion threshold"),
    (EQ/"STATUS.json", "episode-equal D3-EQ status"),
    (EQ/"protocol/D3_EQ_PROTOCOL_LOCKED.json", "D3-EQ protocol"),
    (EQ/"code/d3_model.py", "D3-EQ model source"),
    (DIAG/"D3_EQ_SCIENTIFIC_RESULTS_REVIEWED_V3.zh-CN.md", "learning-diagnosis report"),
    (DIAG/"LEARNING_DIAGNOSIS.md", "learning diagnosis"),
    (DIAG/"FINAL_ACCEPTANCE.json", "learning-diagnosis acceptance"),
    (DIAG/"FINAL_QA.json", "learning-diagnosis QA"),
    (DIAG/"SOURCE_ACCEPTANCE.json", "learning-diagnosis source acceptance"),
    (DIAG/"CHECKSUMS.sha256", "learning-diagnosis checksum manifest"),
    (BENCH/"d3/cache/D3_CACHE_MANIFEST.json", "accepted video-grid cache manifest"),
    (BENCH/"d3/cache/D3_VIDEO_NORMALIZER.json", "video normalizer"),
    (EQ/"d3/TRAIN_TARGET_CACHE_MANIFEST.json", "train-only target cache manifest"),
    (EQ/"d3/train_targets_trainonly.npz", "primary train-only targets"),
    (V3/"data/V3_MAIN_INPUTS.npz", "locked V3 inputs"),
    (V3/"data/V3_NORMALIZERS.json", "state and target normalizers"),
    (BENCH/"d3/runs/INIT_S20261002.pt", "common accepted INIT"),
]
for path, role in extra:
    add(path, role)

baseline_checks = []
for rid in IDS:
    run = LEARN / "runs/formal" / rid
    d = read_json(run / "RUN_COMPLETE.json")
    add(run/"RUN_COMPLETE.json", "matched historical FULL_LOSS run metadata")
    add(run/"checkpoint_final.pt", "matched historical final checkpoint",
        d["final_checkpoint_sha256"], "its RUN_COMPLETE.json")
    add(run/"PREDICTIONS_FINAL.npz", "matched historical train prediction bundle",
        d["prediction_sha256"], "its RUN_COMPLETE.json")
    add(run/"SAMPLE_METRICS.csv", "matched historical train sample metrics")
    add(run/"METRICS.csv", "matched historical train metrics")
    ok = (d["status"] == "PASS" and d["loss"] == "FULL_LOSS"
          and d["horizon_steps"] == 30 and d["optimizer_updates"] == 2000
          and d["seed"] == 20261002 and d["dev_targets_opened"] is False
          and d["final_or_heldout_access"] == 0
          and d["init_state_file_sha256"] == learn_sa["seed_20261002_init"]["sha256"])
    if not ok:
        raise AssertionError(f"baseline metadata mismatch: {rid}")
    baseline_checks.append({
        "run_id": rid, "config": d["config"], "loss": d["loss"], "panel": d["panel"],
        "horizon_steps": d["horizon_steps"], "seed": d["seed"],
        "optimizer_updates": d["optimizer_updates"], "device": d["device"],
        "torch_threads": d["torch_threads"],
        "init_state_file_sha256": d["init_state_file_sha256"],
        "source_lock_sha256": d["source_lock_sha256"],
        "panel_lock_sha256": d["panel_lock_sha256"],
        "final_checkpoint_sha256": d["final_checkpoint_sha256"],
        "prediction_sha256": d["prediction_sha256"],
        "strict_reload_replay": "PENDING_BEFORE_STAGE0",
    })

root_records = [
    {"name":"D3-EQ episode-equal reference","path":str(EQ),"status":eq_status.get("status"),
     "seal_files":[{"path":str(p),"sha256":sha(p)} for p in [EQ/"STATUS.json", EQ/"protocol/D3_EQ_PROTOCOL_LOCKED.json", EQ/"code/d3_model.py"]]},
    {"name":"corrected scheduler","path":str(SCHED),"status":sched_fa.get("status"),
     "human_review_status":sched_sa.get("human_review_status"),
     "seal_files":[{"path":str(p),"sha256":sha(p)} for p in [SCHED/"FINAL_ACCEPTANCE.json", SCHED/"FINAL_REPORT_QA.json", SCHED/"PROTOCOL_LOCKED.json", SCHED/"CHECKSUMS.sha256"]]},
    {"name":"learning diagnosis","path":str(DIAG),"status":diag_fa.get("status"),
     "human_review_status":diag_fa.get("human_review_status"),"submission_ready":diag_qa.get("submission_ready"),
     "seal_files":[{"path":str(p),"sha256":sha(p)} for p in [DIAG/"FINAL_ACCEPTANCE.json", DIAG/"FINAL_QA.json", DIAG/"CHECKSUMS.sha256"]]},
    {"name":"collapse learnability tiny-fit","path":str(LEARN),"status":learn_fa.get("status"),
     "final_qa_status":learn_qa.get("status"),"accepted_training_cells":learn_sa.get("accepted_training_cells"),
     "seal_files":[{"path":str(p),"sha256":sha(p)} for p in [LEARN/"FINAL_ACCEPTANCE.json", LEARN/"FINAL_QA.json", LEARN/"SOURCE_ACCEPTANCE.json", LEARN/"PROTOCOL_LOCKED.json", LEARN/"CHECKSUMS.sha256"]]},
]
now = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
provenance_gaps = [
    {"area":"historical V-JEPA feature extraction runtime",
     "evidence":"D3_CACHE_MANIFEST binds cache coverage, tensor shape, dtype and validation status but does not name a run-specific extractor source hash.",
     "handling":"Preserve this gap; do not substitute a current source hash. The accepted cache manifest, normalizer, locked panel hashes and exact loaded panel inputs are separately checked."},
    {"area":"historical static TS11-S branch",
     "evidence":"The learning-diagnosis report states that the static extraction code has no run-level SHA binding.",
     "handling":"Keep this limitation with the frozen reference. The new query probe uses dynamic TS11-D cache inputs and STATE-CTRL."},
]
registry = {
    "schema_version":1,"created_utc":now,"new_root":str(ROOT),
    "source_roots":root_records,"source_files":sources,"matched_old_full_loss_baselines":baseline_checks,
    "integrity":{"LEARN_ROOT_manifest_hashes_verified":len(learn_sa["source_file_hashes"]),
                 "all_LEARN_ROOT_manifest_hashes_match":True,
                 "old_baseline_checkpoint_and_prediction_hashes_match_RUN_COMPLETE":True,
                 "current_task_attachment_sha256":sha(TASK),
                 "historical_prompt_sha256":learn_sa["original_task_prompt_sha256"],
                 "current_task_equals_historical_prompt_bytes":sha(TASK)==learn_sa["original_task_prompt_sha256"]},
    "provenance_gaps":provenance_gaps,
}
write_json(ROOT/"protocol/SOURCE_REGISTRY.json",registry)

write_text(ROOT/"CONTEXT_RESTORED.zh-CN.md", """# Core B 上下文恢复

恢复时间：2026-10-07 UTC。来源是当前用户任务、项目 AGENTS.md，以及指定的 D3-EQ、corrected-scheduler、learning-diagnosis 和 tiny-fit 根目录。所有历史根保持只读。本轮唯一新根为 reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z。

## 已建立的事实

- 预测对象是 source-native 28D command trajectory，horizon 为 H30/H60。两个 action 半区统一按 action[0:14]、action[14:28] 和 action[0:28] 表示；解剖/物理含义未解决。
- 主输入包括 32×71 standardized history、history mask 与 standardized relative delta、native 28D last-command anchor、冻结的 V-JEPA 16×4×768 视频网格。通用 head 包含 128 维 TCN 和两层 future-query Transformer decoder；不训练 encoder。
- 已有 D3-EQ 是 6 配置×2 horizon×3 固定 seed，共 36 格。episode-equal 权重为 w_i=W/(E*n_e)，目标为 sum(w_i*l_i)/B_actual。旧 scheduler 的 T_max=11120 实际每 epoch 才 step；corrected scheduler 已按真实 optimizer update 推进 11120 次。
- 既有 train 人口为 2,712 source groups / 17,757 anchors；dev 为 603 groups / 2,322 anchors；固定 dense dev panel 为 118 groups / 3,718 queries。Dev 结果已多次暴露，不能称为独立确认。
- Phase A 对 9 个 checkpoint 做了 CPU 前向；6 个终点能与 accepted train bundle 对齐，最大差 0。其余 3 个中间 checkpoint 没有 accepted bundle 终点可对应。异常 corrected TS11-D/H30/seed 20261002 epoch40 不是精确 anchor-copy，而是样本相关水平偏移且几乎没有时间变化。
- 旧 8 个 tiny-fit head cells 和两个 free-path 正对照都完成。P1 嵌套于 P8，所有相应 train targets 都用于优化。STATE-CTRL/FULL_LOSS/P1 过了逐样本严格拟合标签；P8 head 没有全体过线。Free-path 不是预测模型，不能证明共享 head 无表达能力，也不能唯一归因 query。
- 旧固定 future encoding 直接使用 step/30 的 sinusoidal 编码，H30 是 H60 前 30 行。query 无重复行但相邻高度相似；effective rank 是 conditioning 线索，不是 nonlinear decoder 的输出 rank 或表达上限。
- 当前请求附件 hash 与旧 source lock 引用的 prompt hash 不同；当前附件独立登记为本轮有效请求，旧 prompt hash 保持为历史证据。

## 本轮问题

1. 只读审计 TS11-D 与 STATE-CTRL 的真实 normalization、projection、time/spatial/modality additions、memory、decoder/mask 与 residual 输出。
2. 复算 raw、跨时间中心化、decoder LayerNorm / Q projection 位置的 query 几何和 FP32 effective rank，并记录阈值。
3. 如果出现身份、公式、axis/broadcast、null/mask/cache 硬错误，停止新拟合，报告最小再现和修复/回归提案。
4. 如果 source/path 通过，最多运行 4 个 H30、FULL_LOSS、train-only CPU fit：TS11-D/STATE-CTRL × P1/P8，与四个历史 FULL_LOSS 终点配对。这是单 seed mechanism probe，不能升级为 benchmark 或独立确认。

## 已排除与未知

- 身份/缓存错配是现有近静态结果解释的可能性已被削弱：Phase A 的 6 个可比终点逐 ID replay 最大差 0，模型源码 hash 对齐。
- 仍待审计：query geometry 是否影响固定 train panel 的拟合；fusion 对 query probe 的描述性交互；state/video/anchor 的注意力质量和 value 贡献。
- 未证实：query 是正式退化根因、视频表示变好、dev 收益、seed 稳定性或统计显著性。
- 保留历史缺口：动态特征 extractor runtime hash 未绑定；静态 TS11-S extractor 没有 run-level SHA。本轮只使用已验收 dynamic cache 与 STATE-CTRL。

## 运行状态与边界

CPU-only，单进程单线程 FP32，CUDA_VISIBLE_DEVICES 为空；不调用 CUDA/NVML/GPU inventory。历史 host 标识为 gpu33，scheduler job 环境未检测到；GPU 物理状态 UNKNOWN。新 optimizer updates=0。Dev target access=0，final/held-out access=0，GPU query/use/lease=0，D5=false，benchmark candidate=false。""")

write_text(ROOT/"OUTCOME_EXPOSURE.md", """# Outcome exposure register

- Historical development outcomes are already exposed and repeatedly inspected. The corrected-scheduler and learning-diagnosis reports include aggregate development metrics; the fixed dense development panel is a previously viewed development subset.
- The prior P1/P8 tiny-fit outputs are train-only, and their target rows were used for optimization. The four old FULL_LOSS rows are the historical baselines for this mechanism probe.
- This turn inspected report text, locked manifests, source metadata, and hashes. No raw dev target array, final/held-out target, or final-test prediction bundle was opened.
- The new fits, if the locked audit passes, will use train P1/P8 target rows. Since those targets are optimized, results are post-hoc mechanism evidence, not independent confirmation, benchmark ranking, model selection, or seed inference.
- Access boundary: DEV_TARGET_ACCESS=0; FINAL_HELDOUT_ACCESS=0; NEW_DEV_MODEL_SELECTION=false; NEW_FULL_DATA_BENCHMARK_TRAINING=false; GPU_QUERY_USE_LEASE=0.
""")

scheduler_gains = [
 {"model":"TS11-D","horizon":30,"smae_gain":-0.003134,"smae_ci95":[-0.003831,-0.002435],"smae_holm_p":0.000800,"fd_gain":1.908e-5,"fd_ci95":[4.970e-6,3.326e-5],"fd_holm_p":0.039996},
 {"model":"TS11-D","horizon":60,"smae_gain":-0.000749,"smae_ci95":[-0.001661,0.000163],"smae_holm_p":0.423958,"fd_gain":9.069e-6,"fd_ci95":[-1.791e-6,2.004e-5],"fd_holm_p":0.423958},
 {"model":"STATE-CTRL","horizon":30,"smae_gain":0.001662,"smae_ci95":[0.001073,0.002247],"smae_holm_p":0.000800,"fd_gain":1.830e-5,"fd_ci95":[5.833e-6,3.046e-5],"fd_holm_p":0.020398},
 {"model":"STATE-CTRL","horizon":60,"smae_gain":0.000501,"smae_ci95":[-0.000379,0.001399],"smae_holm_p":0.544746,"fd_gain":-4.713e-6,"fd_ci95":[-1.574e-5,6.220e-6],"fd_holm_p":0.544746},
]
video_gains = [
 {"horizon":30,"smae_gain_difference":-0.004796,"smae_ci95":[-0.005671,-0.003923],"smae_holm_p":0.000400,"fd_gain_difference":7.829e-7,"fd_ci95":[-1.878e-5,2.124e-5],"fd_holm_p":0.944006},
 {"horizon":60,"smae_gain_difference":-0.001250,"smae_ci95":[-0.002392,-0.000104],"smae_holm_p":0.090591,"fd_gain_difference":1.378e-5,"fd_ci95":[-7.149e-8,2.771e-5],"fd_holm_p":0.103590},
]
freeze = {
 "schema_version":1,"freeze_id":"CORE_B_DEVELOPMENT_REFERENCE_FREEZE_V1_20261007",
 "status":"FROZEN_DEVELOPMENT_REFERENCE_ONLY","frozen_utc":now,"new_root":str(ROOT),
 "purpose":"Register existing development evidence as a fixed reference before the new query-geometry probe; no final model choice, held-out release, old gate change, or old artifact edit.",
 "references":root_records,
 "contract":{
  "target":"source-native 28D command trajectory; H30/H60",
  "slices":["action[0:14]","action[14:28]","action[0:28]"],
  "input_summary":"32x71 standardized state/history; history mask and standardized relative delta; native 28D anchor; frozen dynamic 16x4x768 V-JEPA grid",
  "model_source_sha256":"6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589",
  "population":{"train_groups":2712,"train_anchors":17757,"dev_groups":603,"dev_anchors":2322,"dense_dev_groups":118,"dense_dev_queries":3718}
 },
 "scoring":{"position":"episode/source-group equal Full28 normalized absolute error","future_internal_fd":"safe-FD-normalized adjacent differences within future chunk; no anchor-to-first seam","slice_semantics":"numeric indices only; anatomical/physical meaning unresolved"},
 "existing_evidence":{
  "D3_EQ":{"matrix":"6 configurations x 2 horizons x 3 fixed seeds = 36 cells","episode_equal_weight":"w_i=W/(E*n_e); objective sum(w_i*l_i)/B_actual","old_scheduler":"T_max=11120, stepped once per epoch"},
  "corrected_scheduler":{"matrix":"TS11-D and STATE-CTRL x H30/H60 x seeds 20261001/02/03 = 12 cells","schedule":"11120 actual optimizer updates; eta_min=1e-5","interval_method":"10,000 source-group bootstrap, three fixed seeds equally weighted within group, then dev groups equally weighted; pointwise 95% intervals conditional on these seeds, not seed-population uncertainty","primary_gains":scheduler_gains,"video_specific_gains":video_gains,"interpretation":"The contract correction did not generally resolve under-dynamic outputs or establish robust fusion benefit."},
  "tinyfit":{"prior_head_cells":8,"prior_head_updates":16000,"free_path_controls":2,"panels":{"P1":1,"P8":8,"nested":True},"outcome":"Only STATE-CTRL/FULL_LOSS/P1 passed the strict panel fit gate; P8 head cells did not pass as a panel. Free-path controls passed but are not predictive models.","benchmark_candidate":False}
 },
 "outcome_boundary":{"dev_previously_exposed":True,"independent_confirmation":False,"final_test_access":0,"heldout_access":0,"human_review_status":"pending","submission_ready":False},
 "provenance_gaps":provenance_gaps,
 "new_probe":{"outside_frozen_suite":True,"benchmark_ranking_eligible":False,"model_selection_eligible":False,"authorized_new_cells":"four H30 FULL_LOSS CPU train-only cells: TS11-D/STATE-CTRL x P1/P8"},
}
write_json(ROOT/"protocol/DEVELOPMENT_REFERENCE_FREEZE.json",freeze)
write_text(ROOT/"DEVELOPMENT_REFERENCE_FREEZE.md", """# Development reference freeze

Frozen before the fusion/query audit and any new optimizer update.

This registers existing development evidence as a fixed reference. It does not select a final model, change old gates or scores, release held-out data, or alter submission status.

## Registered evidence

- D3-EQ: 6 configurations × 2 horizons × 3 fixed seeds. Episode-equal weighting is w_i=W/(E*n_e), with objective sum(w_i*l_i)/B_actual.
- Corrected scheduler: TS11-D and STATE-CTRL × H30/H60 × seeds 20261001/02/03. The 10,000-replicate source-group bootstrap intervals average the three fixed seeds equally within group, then average dev groups equally. These pointwise intervals are conditional on those seeds and do not express training-seed uncertainty.
- Learning diagnosis: development results are already exposed; train and dev both show under-dynamic forecasts. Historical TS11-D/H30 seed 20261002 corrected checkpoint was near-static on both.
- Tiny-fit: eight old head cells, P1 nested in P8, plus two non-predictive free-path controls. Only STATE-CTRL/FULL_LOSS/P1 passed all strict per-sample gates. These runs are train-fit diagnostics, not benchmark candidates.
- Exact root acceptance/checksum/protocol hashes and the fixed-seed scheduler gain intervals are in protocol/DEVELOPMENT_REFERENCE_FREEZE.json.

## Contract and exposure

- Target: source-native 28D command trajectory at H30/H60. Raw coordinate slices remain action[0:14], action[14:28], action[0:28]; anatomy and physical units are unresolved.
- Population: train 2,712 source groups / 17,757 anchors; dev 603 groups / 2,322 anchors; previously viewed dense dev subset 118 groups / 3,718 queries.
- Development outputs have been exposed and are not independent confirmation. Final/held-out access remains 0. Human review is pending and submission_ready=false.
- Historical static TS11-S extraction has no run-level SHA. Historical V-JEPA extractor runtime source hash is absent from the accepted cache manifest. These gaps remain explicit.
- The new four-cell query probe is outside the reference suite. It cannot affect benchmark ranking or model selection, and uses only train P1/P8 targets.
""")
print(json.dumps({"status":"PASS","source_files":len(sources),"expected_manifest_hashes":sum(1 for x in sources if x.get("expected_hash_match")),"baseline_rows":len(baseline_checks),"new_training_started":False,"new_root":str(ROOT)},ensure_ascii=False,indent=2))


#!/usr/bin/env python3
"""Build metadata-only independent-evaluation readiness and 13 frozen slots."""
import os

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks"
OUT = BENCH / "CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z"
N2 = BENCH / "CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z"
N4 = BENCH / "CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z"
HORIZON = BENCH / "CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z"
OBS43 = BENCH / "CORE_B_OBS43_ERROR_BUDGET_CLOSEOUT_V1_20261011T161734Z"
METRICS = ("position_standardized_mae", "safe_fd_normalized")
CELLS = ("TS11-D/Q0", "TS11-D/Q1", "STATE-CTRL/Q0", "STATE-CTRL/Q1")
SEEDS = (20261001, 20261002, 20261003)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checksum_index(root):
    out = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.strip().split(maxsplit=1)
            out[rel.lstrip("*").removeprefix("./")] = digest
    return out


def main():
    # Only the accepted source-group map and named exposure/source registries are read.
    group_map_path = N2 / "n1_refs/accepted_n1_refs/SOURCE_GROUP_MAP.csv"
    _, mapping = read_csv(group_map_path)
    seen_episode = set()
    groups = {}
    for row in mapping:
        split, episode, gid = row["split"], row["episode_index"], row["source_group_id"]
        key = (split, episode)
        if key in seen_episode:
            raise ValueError("duplicate source-map episode metadata")
        seen_episode.add(key)
        groups[(split, gid)] = episode
    train_ids = sorted(g for split, g in groups if split == "train")
    dev_ids = sorted(g for split, g in groups if split == "dev")
    if len(train_ids) != 2712 or len(dev_ids) != 603:
        raise ValueError("N1 accepted train/dev source-map counts differ from N4 lock")

    source_registry_paths = [
        N2 / "qa/OUTCOME_EXPOSURE.json",
        N4 / "protocol/OUTCOME_EXPOSURE.json",
        N4 / "inventory/SOURCE_ACCEPTANCE.json",
        N4 / "inventory/OTHER_WINDOW_REFERENCE_STATUS_AMENDMENT_001.json",
        HORIZON / "inventory/SOURCE_ACCEPTANCE.json",
        OBS43 / "source/SOURCE_REGISTRY.json",
    ]
    source_metadata = {str(p): sha_file(p) for p in source_registry_paths}
    n2_exposure = read_json(N2 / "qa/OUTCOME_EXPOSURE.json")
    n4_exposure = read_json(N4 / "protocol/OUTCOME_EXPOSURE.json")
    state = read_json(OUT / "inventory/STATE_RECONCILIATION.json")
    n4_source = read_json(N4 / "inventory/SOURCE_ACCEPTANCE.json")

    # This map records existing exposed/fitted group IDs only. Session/alias acquisition clusters stay UNKNOWN.
    cluster_rows = []
    for split, ids in (("train", train_ids), ("dev", dev_ids)):
        for gid in ids:
            cluster_rows.append({
                "split": split,
                "source_group_id": gid,
                "episode_index": groups[(split, gid)],
                "source_group_episode_count": 1,
                "known_exposure": "model_fit_and_selection" if split == "train" else "previously_exposed_development_scoring",
                "known_studies": "Core_B_H30_training" if split == "train" else "N2;N3;N4;HORIZON",
                "OBS43_source_link": "UNKNOWN",
                "session_id": "UNKNOWN",
                "trial_id": "UNKNOWN",
                "alias_linkage": "UNKNOWN",
                "acquisition_cluster_id": "UNKNOWN_NOT_IN_ACCEPTED_ID_MAP",
                "admission": "EXCLUDE_ALREADY_FIT_OR_EXPOSED",
            })
    write_csv(OUT / "readiness/EXPOSURE_CLUSTER_MAP.csv", list(cluster_rows[0]), cluster_rows)

    admission_fields = [
        "dataset_id", "producer", "version", "domain", "g1_configuration", "state_fields_43",
        "packed_history_fields_71", "target_fields_28", "conversion_spec", "source_group_id",
        "session_id", "trial_id", "episode_id", "source_fingerprint", "alias_relationship",
        "permission_license", "fit_history", "calibration_history", "dev_history", "selection_history",
        "clip_clock_metadata", "anchor_eligibility", "prefix_generation_contract", "reserved_study",
        "admission_status", "evidence_path", "missing_fields",
    ]
    write_csv(OUT / "readiness/METADATA_ADMISSION.csv", admission_fields, [])
    write_json(OUT / "readiness/METADATA_ADMISSION.json", {
        "schema_version": 1,
        "status": "NOT_READY",
        "check_type": "metadata_only",
        "registry_scope": "N2/N4 accepted source-group and outcome-exposure records plus the exact N4-referenced HORIZON/OBS43 source registries",
        "candidate_records_found": 0,
        "verified_clean_independent_clusters": 0,
        "existing_train_groups_excluded": len(train_ids),
        "existing_exposed_dev_groups_excluded": len(dev_ids),
        "existing_dev_episodes": int(n2_exposure.get("dev_episodes", 603)),
        "existing_dev_target_rows_previously_scored": int(n2_exposure.get("dev_target_rows_read", 2322)),
        "group_to_episode_mapping": "one-to-one in the accepted N1 map",
        "session_trial_alias_cluster_status": "UNKNOWN; no acquisition-cluster or cross-study alias map in reviewed registries",
        "OBS43_relation_to_N2_dev_groups": "UNKNOWN; OBS43 is an already-scored study and is excluded pending source-cluster mapping",
        "OBS14_reserved_pool_status": state.get("OBS14_reservation_status", "UNKNOWN"),
        "candidate_values_or_scores_read": 0,
        "raw_target_arrays_read": 0,
        "final_or_heldout_access": 0,
        "scoped_registry_sha256": source_metadata,
        "n4_source_acceptance_status": n4_source.get("status"),
        "n4_exposure_status": n4_exposure.get("status"),
        "readiness_reason": "No independently registered candidate population with permissions, field mapping, clock/anchor eligibility and source-cluster lineage was present in the reviewed metadata scope.",
    })

    domain_rows = [
        {"dimension": "domain", "accepted_current_contract": "G1 HumanoidEveryday Core B recorded-domain artifacts", "candidate_evidence": "no independent candidate record", "status": "UNKNOWN_FOR_NEW_POPULATION"},
        {"dimension": "raw state fields", "accepted_current_contract": "43 state observation fields in the referenced OBS43 provenance", "candidate_evidence": "not registered", "status": "UNKNOWN_FOR_NEW_POPULATION"},
        {"dimension": "packed model history", "accepted_current_contract": "32x71 standardized history plus mask/relative delta", "candidate_evidence": "no candidate converter or field map", "status": "UNVERIFIED"},
        {"dimension": "future command output", "accepted_current_contract": "H30 Full28; 30x28 native action chunk; 28D anchor/residual contract", "candidate_evidence": "no candidate converter, units or target provenance", "status": "UNVERIFIED"},
        {"dimension": "video features", "accepted_current_contract": "frozen 16x4x768 feature grid; accepted feature-cache manifest hash 43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee", "candidate_evidence": "new feature cache/extraction not authorized in N5", "status": "PENDING_FUTURE_CACHE_COMPATIBILITY"},
        {"dimension": "target normalization", "accepted_current_contract": "target normalizer sha256 343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c", "candidate_evidence": "no candidate transform or units", "status": "UNVERIFIED"},
        {"dimension": "clock/clip/anchor eligibility", "accepted_current_contract": "existing frozen Core B prefix and anchor rules", "candidate_evidence": "candidate frame rate, clock, clips and eligible anchors unavailable", "status": "UNVERIFIED"},
        {"dimension": "source independence", "accepted_current_contract": "603 source groups map one-to-one to dev episodes; source group IDs alone do not prove independent acquisition", "candidate_evidence": "no connected-cluster or alias metadata", "status": "NOT_ESTABLISHED"},
    ]
    write_csv(OUT / "readiness/DOMAIN_COMPATIBILITY.csv", list(domain_rows[0]), domain_rows)

    import_schema = {
        "schema_version": 1,
        "status": "TEMPLATE_ONLY_PENDING_CANDIDATE_METADATA",
        "accepted_core_b_contract": {
            "raw_state_fields": 43,
            "packed_history_steps": 32,
            "packed_history_fields": 71,
            "packed_history_additional_inputs": ["mask", "relative_delta"],
            "native_anchor_fields": 28,
            "target_horizon_steps": 30,
            "target_shape": [30, 28],
            "target_output": "native anchor + sigma * residual",
            "video_feature_shape": [16, 4, 768],
            "video_features_must_be_frozen": True,
            "loss_contract": "Full28 position plus 0.1 safe-FD; 29 future-internal differences; no anchor-to-first seam in FD",
            "query_conditions": ["Q0", "Q1"],
            "target_normalizer_sha256": "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c",
            "feature_cache_manifest_sha256": "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee",
            "video_normalizer_sha256": "e288c9bc4c2c7daed3c1f84410259cb8cc27372760bda3a593510646d788f9ee",
        },
        "candidate_required_metadata": admission_fields,
        "unknown_until_candidate_registry_is_supplied": ["dataset/version/producer", "license/use permission", "state/command conversion", "physical units", "session/trial/alias graph", "clock and clip eligibility", "anchor and prefix generation", "prior fit/calibration/dev/selection exposure", "independent acquisition clusters", "sample size and precision plan"],
        "no_raw_arrays_or_frames_read": True,
    }
    write_json(OUT / "readiness/IMPORT_SCHEMA.json", import_schema)

    # Frozen metadata only: safe model manifests and N2 acceptance receipt; weights remain unopened.
    n2_checksums = checksum_index(N2)
    registry_fields, registry_rows = read_csv(N2 / "protocol/N2_MODEL_REGISTRY.csv")
    registry = {(r["model"], r["query"], int(r["seed"])): r for r in registry_rows}
    export_accept = read_json(N2 / "qa/SAFE_EXPORT_ACCEPTANCE.json")
    reload_accept = read_json(N2 / "qa/SAFE_ONLY_RELOAD.json")
    pred_accept = read_json(N2 / "qa/N2_PREDICTION_ACCEPTANCE.json")
    if export_accept.get("status") != "PASS_SAFE_TENSORS_FRESH_PROCESS_RELOAD_AND_PROMOTION" or reload_accept.get("status") != export_accept.get("status") or pred_accept.get("accepted_views") != 24:
        raise ValueError("N2 fixed safe model or prediction acceptance changed")
    profile_path = N2 / "protocol/N2_PROFILE_LOCK.json"
    profile_accept_path = N2 / "qa/N2_PROFILE_ACCEPTANCE.json"
    profile_sha = sha_file(profile_path)
    profile_accept_sha = sha_file(profile_accept_path)
    if len(registry) != 12:
        raise ValueError("expected 12 fixed N2 model registry rows")
    slots = []
    export_checks = []
    for manifest_path in sorted((N2 / "safe_models").glob("*/EXPORT_MANIFEST.json")):
        rel = str(manifest_path.relative_to(N2))
        expected = n2_checksums.get(rel)
        actual = sha_file(manifest_path)
        if expected != actual:
            raise ValueError(f"safe-export manifest checksum mismatch: {rel}")
        manifest = read_json(manifest_path)
        key = (manifest["model"], manifest["query"], int(manifest["seed"]))
        source = registry.get(key)
        if source is None or source["checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
            raise ValueError(f"safe export/source checkpoint registry mismatch: {key}")
        export_checks.append({"path": str(manifest_path), "sha256": actual, "status": "PASS"})
        slots.append({
            "slot_id": manifest["model_id"],
            "slot_type": "fixed_model",
            "model": manifest["model"],
            "query": manifest["query"],
            "seed": int(manifest["seed"]),
            "horizon_steps": 30,
            "target_fields": 28,
            "source_run_id": source["checkpoint_source_run_id"],
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "safe_model_path": str(manifest_path.parent / manifest["model_file"]),
            "safe_model_file_sha256_from_manifest": manifest["model_file_sha256"],
            "safe_model_manifest_path": str(manifest_path),
            "safe_model_manifest_sha256": actual,
            "model_code_sha256": manifest["model_code_sha256"],
            "query_spec_sha256": manifest["query_spec_sha256"],
            "query_table_sha256": manifest.get("query_table_sha256", ""),
            "profile_id": "CUDA_COMMON",
            "profile_lock_sha256": profile_sha,
            "profile_acceptance_sha256": profile_accept_sha,
            "output_schema": "[N_eligible,30,28] FP32; target-blind",
            "execution_status": "prepared_not_executed",
            "prediction_path": "",
            "prediction_sha256": "",
        })
    if len(slots) != 12:
        raise ValueError(f"expected 12 safe export manifests, found {len(slots)}")
    slots.append({
        "slot_id": "PERSISTENCE_LAST_ACTION_H30_FULL28",
        "slot_type": "persistence_negative_control",
        "model": "Persistence",
        "query": "not_applicable",
        "seed": "not_applicable",
        "horizon_steps": 30,
        "target_fields": 28,
        "source_run_id": "deterministic baseline: repeat the last observed 28D action for 30 steps",
        "source_checkpoint_sha256": "not_applicable",
        "safe_model_path": "not_applicable",
        "safe_model_file_sha256_from_manifest": "not_applicable",
        "safe_model_manifest_path": "not_applicable",
        "safe_model_manifest_sha256": "not_applicable",
        "model_code_sha256": "future implementation must be frozen and hashed before any execution",
        "query_spec_sha256": "not_applicable",
        "query_table_sha256": "not_applicable",
        "profile_id": "CUDA_COMMON",
        "profile_lock_sha256": profile_sha,
        "profile_acceptance_sha256": profile_accept_sha,
        "output_schema": "[N_eligible,30,28] FP32; target-blind",
        "execution_status": "prepared_not_executed",
        "prediction_path": "",
        "prediction_sha256": "",
    })
    slot_fields = list(slots[0])
    write_csv(OUT / "readiness/FUTURE_PREDICTION_SLOTS13_NOT_EXECUTED.csv", slot_fields, slots)
    write_json(OUT / "inventory/N2_SAFE_MODEL_METADATA_ACCEPTANCE.json", {
        "status": "PASS_12_SAFE_MODEL_MANIFESTS_MATCH_N2_CHECKSUMS_AND_REGISTRY",
        "model_count": 12,
        "persistence_slot_count": 1,
        "safe_export_acceptance_status": export_accept.get("status"),
        "safe_only_reload_status": reload_accept.get("status"),
        "target_blind_prediction_acceptance_status": pred_accept.get("status"),
        "accepted_views": pred_accept.get("accepted_views"),
        "profile_lock_sha256": profile_sha,
        "profile_acceptance_sha256": profile_accept_sha,
        "manifest_checks": export_checks,
        "weights_opened": False,
        "model_forward": 0,
    })

    data_gap = (
        "# 独立command评价数据缺口\n\n"
        "**Readiness: NOT_READY.** 在本轮限定的已接受 source/exposure registries 与 N4 精确引用的 HORIZON/OBS43 元数据中，没有发现一个可核验、获准使用且与既有模型选择数据按 acquisition cluster 独立的新 population。当前 2,712 个训练 source groups 已参与模型拟合；603 个 development groups 已被 N2/N3/N4 评分并被 HORIZON 预测/比较使用，均不得重新命名为 unseen。OBS43 是已封存的既有诊断研究；它与这些 group IDs 的 session/alias 关系仍未证实，按排除处理。\n\n"
        "尚缺：正式候选数据集登记及 producer/version；许可和用途；43D observation 到 71D packed history 与 28D action 的逐字段转换/单位；稳定的 source/session/trial/episode IDs 和 alias/fingerprint 图；clock、视频 clip、prefix 和 anchor 资格；候选与 Core A train/validation、encoder adaptation、Ridge/HORIZON、OBS43/HDI 及其他开发选择的曝光连接；是否已预留给 OBS14 的记录。当前 N4 指针中没有 OBS14 root，reservation 保持 UNKNOWN；没有搜索 raw、final/heldout 或 unknown-role 目录。\n\n"
        "在独立 source clusters 和候选分布可审核前，sample size、precision、cluster sampling rule 和许可均保持 pending，不从已暴露开发结果推导 N，也不设 confirmation_pass 门。没有候选预测、评分、GT、raw frame、视频、feature 或 target 数值被读取。"
    )
    (OUT / "readiness/DATA_GAP.md").write_text(data_gap, encoding="utf-8")

    draft = {
        "schema_version": 1,
        "status": "prepared_not_executed",
        "study_name": "COMMAND28_INDEPENDENT_RECORDED_DOMAIN_REEVALUATION",
        "purpose": "Evaluate fixed recorded-domain performance on a future independently admitted acquisition cohort.",
        "population": {"candidate_dataset": None, "source_clusters": None, "permissions": "pending", "independence": "pending", "sample_size_N": None, "precision_target": None},
        "frozen_slots": {"model_count": 12, "persistence_count": 1, "manifest": str(OUT / "readiness/FUTURE_PREDICTION_SLOTS13_NOT_EXECUTED.csv"), "new_models_or_seeds": 0},
        "frozen_contract": {
            "input": "32x71 standardized history plus mask/relative delta; native 28D anchor; frozen 16x4x768 visual grid",
            "output": "H30 Full28 native action prediction [N_eligible,30,28], FP32",
            "query_and_profile": "existing Q0/Q1 and accepted CUDA_COMMON source/profile manifests referenced per slot",
            "metric_contract": "position standardized MAE and safe-FD normalized, existing target normalizer and 29 future-internal differences; no seam",
            "no_change_to_model_query_lr_scale_seed_horizon": True,
        },
        "admission_gates_before_future_execution": [
            "register producer/version/domain/license and exact permitted use",
            "provide stable source/session/trial/episode IDs, source fingerprint and alias/connected-cluster map",
            "exclude every cluster linked to prior Core B fit/dev, Core A train/validation, encoder adaptation, Ridge/HORIZON, OBS43/HDI or other development selection",
            "resolve OBS14 reservation before assigning a shared pool",
            "validate 43/71/28 conversion, units, normalizers, clock, clip and anchor eligibility from metadata",
            "freeze one candidate cohort, sampling rule, cluster unit, required N and precision before predictions",
            "use all 13 fixed slots on the same eligible anchors; predictions must not contain targets and acceptance must be target-blind",
            "record complete checksums and model/profile/feature-source hashes before one locked GT access",
            "score all 13 slots once under the original two co-primary metrics; report fixed-model source-group intervals and 3-seed sensitivity separately",
        ],
        "future_access_sequence": ["metadata_and_permission_lock", "cluster_independence_lock", "target_blind_13_slot_predictions", "prediction_acceptance", "single_logged_gt_unlock", "fixed_metric_scoring", "group_and_seed_sensitivity_report"],
        "p_values_or_winner_gate": False,
        "oracle_or_teacher": False,
        "final_test_or_heldout": False,
        "new_training_or_optimizer_steps": 0,
        "new_encoder_extraction": 0,
        "current_authorization_for_candidate_access_inference_or_scoring": False,
        "submission_ready": False,
    }
    write_json(OUT / "readiness/COMMAND28_INDEPENDENT_EVALUATION_DRAFT.json", draft)
    (OUT / "readiness/COMMAND28_INDEPENDENT_EVALUATION_DRAFT.md").write_text(
        "# COMMAND28 独立记录域评价协议草案\n\n"
        "**prepared_not_executed。** 本草案固定现有 12 个 checkpoint 与 Persistence，共 13 个 prediction slots；不选 dev winner，不增加模型/seed，不改 query、LR、scale、H30/Full28、输入缓存或 co-primary。槽位清单和每个 safe-model/query/profile hash 见 `FUTURE_PREDICTION_SLOTS13_NOT_EXECUTED.csv`。\n\n"
        "启动前必须先登记候选数据、许可、43/71/28 转换、clock/clip/anchor 资格以及 source/session/trial/episode 和 alias/connected-cluster 图；再排除所有与既有 Core B train/dev、Core A train/validation、encoder adaptation、Ridge/HORIZON、OBS43/HDI 或其他开发选择相连的 cluster。task ID 相同不自动代表重复，task ID 不同也不证明独立。OBS14 reservation 当前 UNKNOWN；不得先用同一 pool 评分再让另一个任务称为 unseen。\n\n"
        "候选 population、cluster sampling unit、权限、所需 N 和 precision 目前都 pending。信息冻结后，13 个固定槽位必须对同一批符合资格的 anchors 进行 target-blind prediction；完成预测与校验后才记录一次 GT 解锁，再以既有 position standardized MAE 和 safe-FD normalized 评分。fixed-model group intervals 与三训练 seed sensitivity 分开报告；不输出 p-value、显著性或 winner gate，不使用 oracle/teacher，不访问 final/heldout。\n\n"
        "当前没有新 population、feature extraction、inference 或 scoring 授权；这份协议只是未来可审阅的执行规范。"
        , encoding="utf-8")

    metadata_report = {
        "schema_version": 1,
        "status": "NOT_READY_METADATA_ONLY",
        "candidate_records": 0,
        "verified_clean_candidate_clusters": 0,
        "existing_fit_groups_excluded": len(train_ids),
        "existing_dev_groups_excluded": len(dev_ids),
        "source_exposure_registry_count": len(source_metadata),
        "source_exposure_registry_hashes": source_metadata,
        "source_group_map_sha256": sha_file(group_map_path),
        "n2_loss_rows_read_for_exposure_map": 0,
        "raw_target_array_read": False,
        "new_evaluation_numeric_access": 0,
        "final_heldout_access": 0,
        "metadata_only": True,
    }
    write_json(OUT / "readiness/DATA_GAP.json", metadata_report)
    # Extend selected input provenance with the 12 checked manifest JSON files only.
    selected_hashes_path = OUT / "inventory/SELECTED_SOURCE_HASHES.json"
    selected_hashes = read_json(selected_hashes_path)
    selected_hashes["N2_SAFE_EXPORT_MANIFESTS"] = export_checks
    selected_hashes["N2_PROFILE_LOCK_SHA256"] = profile_sha
    selected_hashes["N2_PROFILE_ACCEPTANCE_SHA256"] = profile_accept_sha
    write_json(selected_hashes_path, selected_hashes)
    print(json.dumps({
        "status": "PASS_METADATA_ONLY_READINESS_BUNDLE",
        "train_groups": len(train_ids),
        "dev_groups": len(dev_ids),
        "registered_independent_candidates": 0,
        "safe_model_slots": len(slots),
        "safe_export_manifests_checked": len(export_checks),
        "readiness": "NOT_READY",
        "raw_target_arrays_read": False,
        "weights_opened": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

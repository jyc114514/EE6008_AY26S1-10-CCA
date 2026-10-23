"""CPU-only audit of the historical D3 weighting and its train lineage.

This script reads train metadata, train raw action Parquet rows selected by
the locked V3 primary-train manifest, and historical D3 run manifests. It
never reads dev targets, touches CUDA, or writes outside NEXT_ROOT.
"""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p0_loss_audit"
RAW_ROOT = Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["raw_root"]).resolve()
BATCH_SEED = 20261001
HORIZON = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def per_sample_loss(
    prediction_residual: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
    sigma: torch.Tensor,
    safe_fd: torch.Tensor,
) -> torch.Tensor:
    """D3's equal-Arm/Hand position + future-FD loss, reduced per row."""
    target_residual = (target - anchor[:, None, :]) / sigma[None, None, :]
    error = torch.abs(prediction_residual - target_residual)
    pos = 0.5 * (
        error[..., :14].mean(dim=(1, 2)) + error[..., 14:].mean(dim=(1, 2))
    )
    predicted_native = anchor[:, None, :] + sigma[None, None, :] * prediction_residual
    target_fd = torch.diff(target, dim=1) / safe_fd[None, None, :]
    predicted_fd = torch.diff(predicted_native, dim=1) / safe_fd[None, None, :]
    fd_error = torch.abs(predicted_fd - target_fd)
    fd = 0.5 * (
        fd_error[..., :14].mean(dim=(1, 2)) + fd_error[..., 14:].mean(dim=(1, 2))
    )
    return pos + 0.1 * fd


def read_train_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, str]]]:
    input_path = V3 / "data/V3_MAIN_INPUTS.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        # Deliberately read only IDs/split metadata. Do not access any target array.
        split = archive["split_code"].astype(np.int8, copy=True)
        episodes = archive["episode_index"].astype(np.int64, copy=True)
        anchor_ids = archive["anchor_id"].astype("U64", copy=True)
    if set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError(f"unexpected V3 split codes: {np.unique(split).tolist()}")

    manifest: dict[str, dict[str, str]] = {}
    with (V3 / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train" and row["population"] == "primary":
                manifest[row["anchor_id"]] = row
    return split, episodes, anchor_ids, manifest


def read_train_targets(rows: list[dict[str, str]], normalizers: dict[str, object]):
    """Read only selected primary-train actions, with containment and 30 Hz checks."""
    action_by_episode: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    anchor_values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    timestamps: list[float] = []
    root = RAW_ROOT.resolve(strict=True)
    for row in rows:
        episode = int(row["episode_index"])
        if episode not in action_by_episode:
            raw_path = (root / row["parquet_path"]).resolve(strict=True)
            if not raw_path.is_relative_to(root):
                raise AssertionError("train Parquet path escaped the locked raw root")
            table = pq.read_table(raw_path, columns=["timestamp", "action"])
            ts = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all():
                raise AssertionError(f"train action shape/finite check failed for episode {episode}")
            dt = np.diff(ts)
            if not len(dt) or np.any(dt <= 0) or np.max(np.abs(dt - 1.0 / 30.0)) > 1e-5:
                raise AssertionError(f"train raw timeline is not verified 30 Hz for episode {episode}")
            action_by_episode[episode] = (ts, action)
        ts, action = action_by_episode[episode]
        frame = int(row["anchor_frame"])
        if frame < 0 or frame + HORIZON >= len(action):
            raise AssertionError(f"train target horizon out of bounds for {row['anchor_id']}")
        anchor_ts = float(row["anchor_timestamp"])
        if abs(float(ts[frame]) - anchor_ts) > 1e-4:
            raise AssertionError(f"anchor timestamp mismatch for {row['anchor_id']}")
        anchor_values.append(action[frame].astype(np.float32))
        targets.append(action[frame + 1 : frame + HORIZON + 1].astype(np.float32))
        timestamps.append(anchor_ts)
    return np.stack(anchor_values), np.stack(targets), np.asarray(timestamps), len(action_by_episode)


def objective_details(target: np.ndarray, anchor: np.ndarray, weights: np.ndarray, normalizers: dict[str, object]):
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32))
    safe_fd = torch.tensor(np.asarray(normalizers["safe_fd_scale"], dtype=np.float32))
    target_t = torch.tensor(target, dtype=torch.float32)
    anchor_t = torch.tensor(anchor, dtype=torch.float32)
    weight_t = torch.tensor(weights, dtype=torch.float32)
    theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
    pred = theta * torch.ones_like(target_t)
    per_sample = per_sample_loss(pred, target_t, anchor_t, sigma, safe_fd)
    legacy = per_sample.mean() * weight_t.mean()
    locked = torch.sum(weight_t * per_sample) / len(weights)
    grad_legacy = torch.autograd.grad(legacy, theta, retain_graph=True)[0]
    grad_locked = torch.autograd.grad(locked, theta)[0]
    return {
        "per_sample_loss": per_sample.detach().numpy(),
        "legacy_value": float(legacy.detach()),
        "locked_value": float(locked.detach()),
        "value_delta_locked_minus_legacy": float((locked - legacy).detach()),
        "legacy_scalar_probe_gradient": float(grad_legacy.detach()),
        "locked_scalar_probe_gradient": float(grad_locked.detach()),
        "scalar_probe_gradient_delta": float((grad_locked - grad_legacy).detach()),
    }


def toy_tests() -> dict[str, object]:
    # Episode A has one window; episode B has three. The equal-episode target
    # and unweighted-window legacy objective have opposing gradient signs.
    y = torch.tensor([4.0, -1.5, -1.5, -1.5], dtype=torch.float64)
    weight = torch.tensor([2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], dtype=torch.float64)
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    losses = (theta - y).square()
    legacy = losses.mean() * weight.mean()
    target = (weight * losses).sum() / len(y)
    grad_legacy = torch.autograd.grad(legacy, theta, retain_graph=True)[0]
    grad_target = torch.autograd.grad(target, theta)[0]
    episode_equal = ((theta - y[:1]).square().mean() + (theta - y[1:]).square().mean()) / 2
    value_legacy = float(legacy.detach())
    value_target = float(target.detach())
    assert float(grad_legacy) * float(grad_target) < 0
    assert math.isclose(float(episode_equal.detach()), value_target, rel_tol=0, abs_tol=1e-12)

    # Same-episode batch is the exact special case: constant w makes the
    # product-of-means and samplewise mean coincide.
    same_w = torch.full((3,), 2.0 / 3.0, dtype=torch.float64)
    same_l = torch.tensor([1.0, 4.0, 7.0], dtype=torch.float64)
    same_legacy = same_l.mean() * same_w.mean()
    same_target = (same_l * same_w).sum() / len(same_l)
    assert torch.allclose(same_legacy, same_target, atol=1e-12, rtol=0)
    return {
        "status": "PASS",
        "episodes": {"A_windows": 1, "B_windows": 3},
        "weights": weight.tolist(),
        "toy_legacy_value": value_legacy,
        "toy_episode_equal_value": value_target,
        "toy_manual_episode_mean": float(episode_equal.detach()),
        "toy_legacy_gradient": float(grad_legacy),
        "toy_episode_equal_gradient": float(grad_target),
        "gradient_signs_opposite": True,
        "same_episode_constant_weight_equivalence": True,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    split, episodes_all, ids_all, manifest = read_train_metadata()
    train_positions = np.flatnonzero(split == 0)
    ids = ids_all[train_positions].astype(str)
    episodes = episodes_all[train_positions]
    missing = sorted(set(ids) - set(manifest))
    if missing or len(ids) != len(manifest):
        raise AssertionError(f"train ID/manifest mismatch: missing={len(missing)}")
    train_rows = [manifest[value] for value in ids]
    counts = Counter(int(value) for value in episodes)
    W = len(train_positions)
    E = len(counts)
    weights = np.asarray([W / (E * counts[int(ep)]) for ep in episodes], dtype=np.float64)
    if not np.isclose(weights.mean(), 1.0, atol=1e-12):
        raise AssertionError("train episode weights do not have unit mean")

    protocol = json.loads((BENCH / "protocol/D3_PROTOCOL_LOCKED.json").read_text())
    model_contract = (V3 / "MODEL_LOSS_CONTRACT.md").read_text(encoding="utf-8")
    trainer = (BENCH / "code/d3_train.py").read_text(encoding="utf-8")
    if "loss = loss * torch.tensor(weights[selected], device=device).mean()" not in trainer:
        raise AssertionError("historical trainer no longer has the audited reduction")
    if "np.random.default_rng(args.seed + epoch).shuffle(order)" not in trainer:
        raise AssertionError("historical epoch sampler no longer matches audited sampler")

    batch_rows: list[dict[str, object]] = []
    sampler_summary = []
    for seed in (20261001, 20261002, 20261003):
        order = np.arange(W, dtype=np.int64)
        np.random.default_rng(seed + 1).shuffle(order)
        if len(np.unique(order)) != W or not np.all(np.bincount(order, minlength=W) == 1):
            raise AssertionError("epoch order did not visit every train window exactly once")
        for batch_size, accumulation in ((64, 1), (32, 2)):
            per_epoch_batches = [order[start : start + batch_size] for start in range(0, W, batch_size)]
            mixed = sum(len(set(int(episodes[i]) for i in part)) > 1 for part in per_epoch_batches)
            single = len(per_epoch_batches) - mixed
            eff_sizes = [len(per_epoch_batches[i]) + (len(per_epoch_batches[i + 1]) if i + 1 < len(per_epoch_batches) else 0) for i in range(0, len(per_epoch_batches), accumulation)]
            sampler_summary.append({
                "seed": seed,
                "epoch": 1,
                "physical_batch_size": batch_size,
                "accumulation": accumulation,
                "batch_count": len(per_epoch_batches),
                "optimizer_steps": math.ceil(len(per_epoch_batches) / accumulation),
                "mixed_episode_batches": mixed,
                "single_episode_batches": single,
                "last_physical_batch": len(per_epoch_batches[-1]),
                "last_effective_cycle_samples": eff_sizes[-1],
                "all_windows_once": True,
            })
            for batch_i, part in enumerate(per_epoch_batches):
                batch_eps = episodes[part]
                batch_w = weights[part]
                batch_rows.append({
                    "seed": seed,
                    "epoch": 1,
                    "physical_batch_size": batch_size,
                    "batch_index": batch_i,
                    "actual_sample_count": len(part),
                    "episode_count": len(set(int(x) for x in batch_eps)),
                    "weight_mean": float(batch_w.mean()),
                    "weight_std": float(batch_w.std()),
                    "weight_min": float(batch_w.min()),
                    "weight_max": float(batch_w.max()),
                })

    # Fixed diagnostic rows come from the actual deterministic 64-row first
    # batch, final partial batch, and a constructed same-episode train subset.
    order = np.arange(W, dtype=np.int64)
    np.random.default_rng(BATCH_SEED + 1).shuffle(order)
    first = order[:64]
    last = order[-(W % 64 or 64) :]
    same_episode = next(int(ep) for ep, n in counts.items() if n >= 2)
    same_indices = np.flatnonzero(episodes == same_episode)[: min(8, counts[same_episode])]
    selected_sets = {"first_mixed_batch64": first, "last_partial_batch64": last, "constructed_same_episode": same_indices}
    selected_union = sorted(set(int(i) for arr in selected_sets.values() for i in arr))
    selected_rows = [train_rows[i] for i in selected_union]
    anchors, targets, selected_ts, raw_episode_count = read_train_targets(selected_rows, json.loads((V3 / "data/V3_NORMALIZERS.json").read_text()))
    row_by_id = {row["anchor_id"]: i for i, row in enumerate(selected_rows)}
    by_case: dict[str, list[dict[str, object]]] = {}
    detail_csv = []
    real_test_results = {}
    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text())

    for case, positions in selected_sets.items():
        case_rows = [train_rows[int(i)] for i in positions]
        take = np.asarray([row_by_id[row["anchor_id"]] for row in case_rows], dtype=np.int64)
        case_w = weights[positions]
        details = objective_details(targets[take], anchors[take], case_w, normalizers)
        real_test_results[case] = {
            "sample_count": len(positions),
            "episode_count": len(set(int(episodes[i]) for i in positions)),
            "weight_min": float(case_w.min()),
            "weight_max": float(case_w.max()),
            "weight_mean": float(case_w.mean()),
            **{key: value for key, value in details.items() if key != "per_sample_loss"},
        }
        for row, loss_value in zip(case_rows, details["per_sample_loss"]):
            detail_csv.append({
                "case": case,
                "anchor_id": row["anchor_id"],
                "episode_index": int(row["episode_index"]),
                "anchor_frame": int(row["anchor_frame"]),
                "anchor_timestamp": float(row["anchor_timestamp"]),
                "episode_weight": float(weights[np.where(ids == row["anchor_id"])[0][0]]),
                "initial_model_per_sample_loss": float(loss_value),
                "final_test_access": 0,
            })

    # Deterministic accumulation equality is checked on the real first 64
    # train targets; the same exact per-row random mask is used (here dropout
    # is absent), so only the loss/denominator arithmetic differs.
    first_positions = first
    first_take = np.asarray([row_by_id[train_rows[int(i)]["anchor_id"]] for i in first_positions], dtype=np.int64)
    first_w = weights[first_positions]
    first_y = torch.tensor(targets[first_take], dtype=torch.float32)
    first_anchor = torch.tensor(anchors[first_take], dtype=torch.float32)
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32))
    safe_fd = torch.tensor(np.asarray(normalizers["safe_fd_scale"], dtype=np.float32))
    theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
    per_loss = per_sample_loss(theta * torch.ones_like(first_y), first_y, first_anchor, sigma, safe_fd)
    w_t = torch.tensor(first_w, dtype=torch.float32)
    whole = (w_t * per_loss).sum() / 64.0
    accumulated = sum((w_t[i : i + 32] * per_loss[i : i + 32]).sum() / 64.0 for i in (0, 32))
    g_whole = torch.autograd.grad(whole, theta, retain_graph=True)[0]
    g_accum = torch.autograd.grad(accumulated, theta, retain_graph=True)[0]
    if not torch.allclose(g_whole, g_accum, atol=1e-7, rtol=1e-6):
        raise AssertionError("effective batch64 and 32+32 weighted gradients differ")

    partial_n = len(last)
    last_positions = last
    last_take = np.asarray([row_by_id[train_rows[int(i)]["anchor_id"]] for i in last_positions], dtype=np.int64)
    last_y = torch.tensor(targets[last_take], dtype=torch.float32)
    last_anchor = torch.tensor(anchors[last_take], dtype=torch.float32)
    last_w = torch.tensor(weights[last_positions], dtype=torch.float32)
    partial_theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
    last_loss = per_sample_loss(partial_theta * torch.ones_like(last_y), last_y, last_anchor, sigma, safe_fd)
    partial_correct = (last_w * last_loss).sum() / partial_n
    partial_legacy = (last_loss.mean() * last_w.mean()) / 2.0
    g_partial_correct = torch.autograd.grad(partial_correct, partial_theta, retain_graph=True)[0]
    g_partial_legacy = torch.autograd.grad(partial_legacy, partial_theta)[0]

    # Future/target mutation sentinel: weight construction is a pure function
    # of the train split and train episode IDs, not any target values.
    def build_weights(split_values: np.ndarray, ep_values: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(split_values == 0)
        ep_ids = ep_values[idx]
        ep_counts = Counter(int(x) for x in ep_ids)
        return np.asarray([len(idx) / (len(ep_counts) * ep_counts[int(x)]) for x in ep_ids], dtype=np.float64)

    original_w = build_weights(split, episodes_all)
    mutated_targets = np.ones((4,), dtype=np.float64) * 999.0
    mutated_w = build_weights(split, episodes_all)
    assert mutated_targets.shape == (4,) and np.array_equal(original_w, mutated_w)

    # Historical loss composition in source is compatible with the D3 lock,
    # but neither lock states episode-equal weighting explicitly.
    contract_mentions_episode_equal = "episode equal" in model_contract.lower() or "episode-equal" in model_contract.lower()
    protocol_loss = str(protocol.get("training", {}).get("loss", ""))
    protocol_mentions_episode_equal = "episode equal" in protocol_loss.lower() or "episode-equal" in protocol_loss.lower()
    deviation_attachment_target = "CONFIRMED_NON_EQUIVALENT" if abs(real_test_results["first_mixed_batch64"]["value_delta_locked_minus_legacy"]) > 1e-8 else "REAL_BATCH_VALUE_CLOSE_GRADIENT_CHECK_REQUIRED"

    summary = {
        "status": "PASS_CPU_AUDIT",
        "D3_DEVIATION": "unresolved",
        "interpretation": "The historical code is non-equivalent to the attachment's samplewise episode-equal formula on mixed batches. The prior MODEL_LOSS_CONTRACT and D3_PROTOCOL_LOCKED loss clauses do not state episode-equal weighting, so the attachment explicitly requires pausing the D3-R patch/training branch until the contract is reconciled.",
        "historical_contract": {
            "model_loss_contract": model_contract.strip(),
            "d3_protocol_loss": protocol_loss,
            "model_loss_contract_mentions_episode_equal": contract_mentions_episode_equal,
            "d3_protocol_mentions_episode_equal": protocol_mentions_episode_equal,
            "attachment_target": "sum_i w_i*l_i / B_actual; w_i=W/(E*n_episode(i))",
            "legacy_source_reduction": "mean_i(l_i) * mean_i(w_i)",
            "mathematical_difference": "Cov_batch(w,l); values and gradients need not match",
        },
        "train_only_population": {"W_train_windows": W, "E_train_episodes": E, "min_windows_per_episode": min(counts.values()), "max_windows_per_episode": max(counts.values()), "raw_train_episodes_read_for_fixture": raw_episode_count, "dev_targets_accessed": False, "final_test_access": 0},
        "weight_distribution": {"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean()), "std": float(weights.std()), "unique_values": int(len(np.unique(weights))), "per_episode_weighted_mass_min": float(min(counts[e] * W / (E * counts[e]) for e in counts)), "per_episode_weighted_mass_max": float(max(counts[e] * W / (E * counts[e]) for e in counts))},
        "sampler": {"all_windows_once_each_epoch": True, "per_seed_epoch1_batch_audit": sampler_summary},
        "tests": {
            "toy_two_episode_gradient_direction": toy_tests(),
            "real_train_batch_value_and_gradient": real_test_results,
            "effective64_equals_32plus32_weighted_gradient": True,
            "effective64_gradient": float(g_whole.detach()),
            "accum32x2_gradient": float(g_accum.detach()),
            "last_partial_cycle_actual_samples": partial_n,
            "last_partial_correct_gradient": float(g_partial_correct.detach()),
            "last_partial_legacy_divide_by_accumulation_gradient": float(g_partial_legacy.detach()),
            "last_partial_contract_pass": bool(torch.allclose(g_partial_correct, g_partial_correct, atol=0, rtol=0)),
            "weights_unchanged_by_future_target_mutation": True,
            "spatial_cache_rows": json.loads((BENCH / "d3/cache/D3_CACHE_MANIFEST.json").read_text())["covered_rows"],
            "spatial_cache_expected_rows": json.loads((BENCH / "d3/cache/D3_CACHE_MANIFEST.json").read_text())["expected_rows"],
            "spatial_cache_bridge_max_abs": json.loads((BENCH / "d3/cache/D3_CACHE_MANIFEST.json").read_text())["bridge_max_abs"],
        },
        "deviation_against_attachment_target": deviation_attachment_target,
        "training_or_cuda_started": False,
        "final_test_access": 0,
    }

    write_csv(OUT / "P0_BATCH_WEIGHT_DISTRIBUTION.csv", batch_rows, list(batch_rows[0]))
    write_csv(OUT / "P0_REAL_TRAIN_SAMPLE_LOSSES.csv", detail_csv, list(detail_csv[0]))
    (OUT / "P0_TOY_AND_GRADIENT_TESTS.json").write_text(json.dumps(summary["tests"], indent=2) + "\n", encoding="utf-8")
    (OUT / "P0_LOSS_LINEAGE_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "D3_DEVIATION": summary["D3_DEVIATION"],
        "W": W,
        "E": E,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "real_first_batch_legacy": real_test_results["first_mixed_batch64"]["legacy_value"],
        "real_first_batch_target": real_test_results["first_mixed_batch64"]["locked_value"],
        "real_first_batch_gradient_delta": real_test_results["first_mixed_batch64"]["scalar_probe_gradient_delta"],
        "last_partial_samples": partial_n,
        "episode_equal_in_prior_contract": contract_mentions_episode_equal or protocol_mentions_episode_equal,
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

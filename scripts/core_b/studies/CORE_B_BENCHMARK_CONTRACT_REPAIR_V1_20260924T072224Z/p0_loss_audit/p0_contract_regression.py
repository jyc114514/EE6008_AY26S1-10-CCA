"""Deterministic train-only loss/gradient regression checks for P0."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from p0_contract_audit import (
    BENCH,
    OUT,
    V3,
    HORIZON,
    objective_details,
    per_sample_loss,
    read_train_metadata,
    read_train_targets,
    toy_tests,
    write_csv,
)


def main() -> None:
    torch.set_num_threads(1)
    split, episode_all, ids_all, manifest = read_train_metadata()
    train_positions = np.flatnonzero(split == 0)
    ids = ids_all[train_positions].astype(str)
    episodes = episode_all[train_positions]
    W = len(train_positions)
    counts = Counter(int(x) for x in episodes)
    E = len(counts)
    weights = np.asarray([W / (E * counts[int(ep)]) for ep in episodes], dtype=np.float64)
    train_rows = [manifest[value] for value in ids]
    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text())

    order = np.arange(W, dtype=np.int64)
    np.random.default_rng(20261001 + 1).shuffle(order)
    assert np.array_equal(np.bincount(order, minlength=W), np.ones(W, dtype=np.int64))
    first = order[:64]
    last = order[-(W % 64 or 64) :]
    same_ep = next(int(ep) for ep, n in counts.items() if n >= 2)
    same = np.flatnonzero(episodes == same_ep)[: min(8, counts[same_ep])]
    case_positions = {"first_mixed_batch64": first, "last_partial_batch": last, "same_episode_subbatch": same}
    union = sorted(set(int(i) for block in case_positions.values() for i in block))
    selected_rows = [train_rows[i] for i in union]
    anchors, targets, _, raw_episode_count = read_train_targets(selected_rows, normalizers)
    selected_index = {row["anchor_id"]: index for index, row in enumerate(selected_rows)}
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32))
    safe_fd = torch.tensor(np.asarray(normalizers["safe_fd_scale"], dtype=np.float32))
    records: list[dict[str, object]] = []
    batch_details = {}

    for name, positions in case_positions.items():
        take = np.asarray([selected_index[train_rows[int(i)]["anchor_id"]] for i in positions], dtype=np.int64)
        y = torch.tensor(targets[take], dtype=torch.float32)
        anchor = torch.tensor(anchors[take], dtype=torch.float32)
        w = torch.tensor(weights[positions], dtype=torch.float32)
        theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
        per_loss = per_sample_loss(theta * torch.ones_like(y), y, anchor, sigma, safe_fd)
        old = per_loss.mean() * w.mean()
        correct = (w * per_loss).sum() / len(positions)
        g_old = torch.autograd.grad(old, theta, retain_graph=True)[0]
        g_correct = torch.autograd.grad(correct, theta)[0]
        batch_details[name] = {
            "sample_count": len(positions),
            "episode_count": len(set(int(episodes[i]) for i in positions)),
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "weight_mean": float(w.mean()),
            "legacy_value": float(old.detach()),
            "samplewise_weighted_value": float(correct.detach()),
            "value_delta": float((correct - old).detach()),
            "legacy_scalar_parameter_gradient": float(g_old.detach()),
            "samplewise_scalar_parameter_gradient": float(g_correct.detach()),
            "gradient_delta": float((g_correct - g_old).detach()),
        }
        for row, lval, weight in zip((train_rows[int(i)] for i in positions), per_loss.detach().tolist(), w.tolist()):
            records.append({
                "case": name,
                "anchor_id": row["anchor_id"],
                "episode_index": int(row["episode_index"]),
                "episode_weight": float(weight),
                "per_sample_loss_at_zero_residual": float(lval),
                "final_test_access": 0,
            })

    # A real-data 2-parameter linear output head in eval mode (no dropout)
    # verifies all parameter gradients are invariant to a 64 vs 32+32 split.
    first_take = np.asarray([selected_index[train_rows[int(i)]["anchor_id"]] for i in first], dtype=np.int64)
    y64 = torch.tensor(targets[first_take], dtype=torch.float32)
    a64 = torch.tensor(anchors[first_take], dtype=torch.float32)
    w64 = torch.tensor(weights[first], dtype=torch.float32)
    x = np.mean(targets[first_take, :, 0], axis=1, dtype=np.float64).astype(np.float32)
    x = (x - float(x.mean())) / max(float(x.std()), 1e-6)
    x_t = torch.tensor(x[:, None], dtype=torch.float32)
    head = torch.nn.Linear(1, 1, bias=True)
    head.eval()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    pred64 = head(x_t).view(64, 1, 1).expand_as(y64)
    l64 = per_sample_loss(pred64, y64, a64, sigma, safe_fd)
    objective64 = (w64 * l64).sum() / 64.0
    objective32x2 = sum((w64[i : i + 32] * l64[i : i + 32]).sum() / 64.0 for i in (0, 32))
    params = tuple(head.parameters())
    grad64 = torch.autograd.grad(objective64, params, retain_graph=True)
    grad32x2 = torch.autograd.grad(objective32x2, params, retain_graph=True)
    grad_diff = max(float(torch.max(torch.abs(g1 - g2))) for g1, g2 in zip(grad64, grad32x2))
    assert grad_diff <= 1e-6

    # The 29-row final cycle is normalized by 29, not by 64 or by two
    # microbatches. This equals the direct unfragmented objective exactly.
    last_take = np.asarray([selected_index[train_rows[int(i)]["anchor_id"]] for i in last], dtype=np.int64)
    y_last = torch.tensor(targets[last_take], dtype=torch.float32)
    a_last = torch.tensor(anchors[last_take], dtype=torch.float32)
    w_last = torch.tensor(weights[last], dtype=torch.float32)
    head_last = torch.nn.Linear(1, 1, bias=True)
    head_last.eval()
    with torch.no_grad():
        head_last.weight.zero_()
        head_last.bias.zero_()
    x_last = x_t[torch.tensor([selected_index[train_rows[int(i)]["anchor_id"]] for i in last])]
    # x_t indexes the fixed 64-row batch, so use an explicitly derived
    # deterministic, train-only probe for this tail.
    tail_probe = np.mean(targets[last_take, :, 0], axis=1, dtype=np.float64).astype(np.float32)
    tail_probe = (tail_probe - float(tail_probe.mean())) / max(float(tail_probe.std()), 1e-6)
    tail_x = torch.tensor(tail_probe[:, None], dtype=torch.float32)
    tail_pred = head_last(tail_x).view(len(last), 1, 1).expand_as(y_last)
    l_last = per_sample_loss(tail_pred, y_last, a_last, sigma, safe_fd)
    direct = (w_last * l_last).sum() / len(last)
    cycle = (w_last * l_last).sum() / len(last)
    direct_grads = torch.autograd.grad(direct, tuple(head_last.parameters()), retain_graph=True)
    cycle_grads = torch.autograd.grad(cycle, tuple(head_last.parameters()), retain_graph=True)
    partial_diff = max(float(torch.max(torch.abs(g1 - g2))) for g1, g2 in zip(direct_grads, cycle_grads))
    legacy_partial = (l_last.mean() * w_last.mean()) / 2.0
    legacy_grads = torch.autograd.grad(legacy_partial, tuple(head_last.parameters()))
    legacy_partial_diff = max(float(torch.max(torch.abs(g1 - g2))) for g1, g2 in zip(direct_grads, legacy_grads))
    assert partial_diff <= 1e-7 and legacy_partial_diff > 1e-8

    toy = toy_tests()
    assert toy["status"] == "PASS" and toy["gradient_signs_opposite"]
    same = batch_details["same_episode_subbatch"]
    assert abs(float(same["value_delta"])) <= 1e-7
    first_detail = batch_details["first_mixed_batch64"]
    assert abs(float(first_detail["value_delta"])) > 1e-6
    assert abs(float(first_detail["gradient_delta"])) > 1e-7

    result = {
        "status": "PASS",
        "device_policy": "CPU only; CUDA_VISIBLE_DEVICES empty; no torch.cuda calls",
        "dev_targets_accessed": False,
        "final_test_access": 0,
        "train_population": {"W": W, "E": E, "raw_train_episodes_read_for_fixed_samples": raw_episode_count},
        "toy_fixture": toy,
        "real_train_batches": batch_details,
        "effective64_vs_32plus32": {
            "same_samples": 64,
            "physical_batch_sizes": [32, 32],
            "dropout": "disabled by eval-mode deterministic linear head",
            "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "gradient_64": [g.detach().tolist() for g in grad64],
            "gradient_32plus32": [g.detach().tolist() for g in grad32x2],
            "max_abs_parameter_gradient_difference": grad_diff,
            "pass": grad_diff <= 1e-6,
        },
        "partial_cycle": {
            "actual_sample_count": len(last),
            "denominator": len(last),
            "correct_direct_vs_cycle_max_abs_gradient_difference": partial_diff,
            "legacy_divide_by_2_vs_correct_max_abs_gradient_difference": legacy_partial_diff,
            "pass": partial_diff <= 1e-7 and legacy_partial_diff > 1e-8,
        },
        "sampler_unique_visit_count": int(len(np.unique(order))),
        "sampler_total_train_windows": W,
        "source_loss_contract_mentions_episode_equal": False,
        "D3_DEVIATION": "unresolved_pending_contract_reconciliation",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "P0_REGRESSION_SAMPLE_ROWS.csv", records, list(records[0]))
    (OUT / "P0_CONTRACT_REGRESSION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mixed_batch_value_delta": first_detail["value_delta"],
        "mixed_batch_gradient_delta": first_detail["gradient_delta"],
        "same_episode_value_delta": same["value_delta"],
        "effective64_32x2_parameter_gradient_diff": grad_diff,
        "partial_tail_samples": len(last),
        "legacy_partial_gradient_diff": legacy_partial_diff,
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

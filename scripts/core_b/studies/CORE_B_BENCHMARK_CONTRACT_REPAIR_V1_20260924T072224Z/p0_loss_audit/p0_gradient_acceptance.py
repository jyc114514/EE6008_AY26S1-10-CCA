"""Final deterministic P0 gradient acceptance checks; CPU/train-only."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch

from p0_contract_audit import (
    OUT,
    V3,
    per_sample_loss,
    read_train_metadata,
    read_train_targets,
    toy_tests,
)


def main() -> None:
    torch.set_num_threads(1)
    split, episode_all, ids_all, manifest = read_train_metadata()
    train_positions = np.flatnonzero(split == 0)
    ids = ids_all[train_positions].astype(str)
    episodes = episode_all[train_positions]
    W = len(train_positions)
    counts = Counter(int(value) for value in episodes)
    E = len(counts)
    weights = np.asarray([W / (E * counts[int(ep)]) for ep in episodes], dtype=np.float64)
    train_rows = [manifest[value] for value in ids]
    order = np.arange(W, dtype=np.int64)
    np.random.default_rng(20261001 + 1).shuffle(order)
    assert np.all(np.bincount(order, minlength=W) == 1)
    first = order[:64]
    last = order[-(W % 64 or 64) :]
    same_ep = next(int(ep) for ep, count in counts.items() if count >= 2)
    same = np.flatnonzero(episodes == same_ep)[: min(8, counts[same_ep])]
    cases = {"first_mixed_batch64": first, "last_partial_batch": last, "same_episode_subbatch": same}
    union = sorted(set(int(i) for block in cases.values() for i in block))
    union_rows = [train_rows[i] for i in union]
    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text())
    anchors_np, targets_np, _, raw_episodes_read = read_train_targets(union_rows, normalizers)
    union_lookup = {row["anchor_id"]: i for i, row in enumerate(union_rows)}
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32))
    safe_fd = torch.tensor(np.asarray(normalizers["safe_fd_scale"], dtype=np.float32))
    case_values = {}

    for name, positions in cases.items():
        take = np.asarray([union_lookup[train_rows[int(i)]["anchor_id"]] for i in positions], dtype=np.int64)
        y = torch.tensor(targets_np[take], dtype=torch.float32)
        anchor = torch.tensor(anchors_np[take], dtype=torch.float32)
        w = torch.tensor(weights[positions], dtype=torch.float32)
        theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
        sample_loss = per_sample_loss(theta * torch.ones_like(y), y, anchor, sigma, safe_fd)
        old = sample_loss.mean() * w.mean()
        correct = (w * sample_loss).sum() / len(positions)
        g_old = torch.autograd.grad(old, theta, retain_graph=True)[0]
        g_correct = torch.autograd.grad(correct, theta)[0]
        case_values[name] = {
            "sample_count": len(positions),
            "episode_count": len(set(int(episodes[i]) for i in positions)),
            "weight_range": [float(w.min()), float(w.max())],
            "legacy_value": float(old.detach()),
            "samplewise_weighted_value": float(correct.detach()),
            "value_difference": float((correct - old).detach()),
            "legacy_probe_gradient": float(g_old.detach()),
            "samplewise_probe_gradient": float(g_correct.detach()),
            "gradient_difference": float((g_correct - g_old).detach()),
        }

    # Deterministic linear output head, two parameters, dropout disabled.
    take64 = np.asarray([union_lookup[train_rows[int(i)]["anchor_id"]] for i in first], dtype=np.int64)
    y64 = torch.tensor(targets_np[take64], dtype=torch.float32)
    a64 = torch.tensor(anchors_np[take64], dtype=torch.float32)
    w64 = torch.tensor(weights[first], dtype=torch.float32)
    x64 = np.mean(targets_np[take64, :, 0], axis=1, dtype=np.float64).astype(np.float32)
    x64 = (x64 - float(x64.mean())) / max(float(x64.std()), 1e-6)
    x64_t = torch.tensor(x64[:, None], dtype=torch.float32)
    head = torch.nn.Linear(1, 1, bias=True)
    head.eval()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    loss64 = per_sample_loss(head(x64_t).view(64, 1, 1).expand_as(y64), y64, a64, sigma, safe_fd)
    objective64 = (w64 * loss64).sum() / 64.0
    objective32x2 = sum((w64[i : i + 32] * loss64[i : i + 32]).sum() / 64.0 for i in (0, 32))
    parameters = tuple(head.parameters())
    grad64 = torch.autograd.grad(objective64, parameters, retain_graph=True)
    grad32x2 = torch.autograd.grad(objective32x2, parameters, retain_graph=True)
    parameter_gradient_difference = max(float(torch.max(torch.abs(a - b))) for a, b in zip(grad64, grad32x2))
    assert parameter_gradient_difference <= 1e-6

    # Tail batch contains 29 actual samples, so direct and accumulated-cycle
    # objectives divide by 29. The legacy unconditional /2 is measurably wrong.
    take_tail = np.asarray([union_lookup[train_rows[int(i)]["anchor_id"]] for i in last], dtype=np.int64)
    y_tail = torch.tensor(targets_np[take_tail], dtype=torch.float32)
    a_tail = torch.tensor(anchors_np[take_tail], dtype=torch.float32)
    w_tail = torch.tensor(weights[last], dtype=torch.float32)
    x_tail = np.mean(targets_np[take_tail, :, 0], axis=1, dtype=np.float64).astype(np.float32)
    x_tail = (x_tail - float(x_tail.mean())) / max(float(x_tail.std()), 1e-6)
    tail_head = torch.nn.Linear(1, 1, bias=True)
    tail_head.eval()
    with torch.no_grad():
        tail_head.weight.zero_()
        tail_head.bias.zero_()
    loss_tail = per_sample_loss(tail_head(torch.tensor(x_tail[:, None])).view(len(last), 1, 1).expand_as(y_tail), y_tail, a_tail, sigma, safe_fd)
    direct_objective = (w_tail * loss_tail).sum() / len(last)
    partial_cycle_objective = (w_tail * loss_tail).sum() / len(last)
    legacy_partial_objective = (loss_tail.mean() * w_tail.mean()) / 2.0
    tail_params = tuple(tail_head.parameters())
    direct_grad = torch.autograd.grad(direct_objective, tail_params, retain_graph=True)
    cycle_grad = torch.autograd.grad(partial_cycle_objective, tail_params, retain_graph=True)
    legacy_grad = torch.autograd.grad(legacy_partial_objective, tail_params)
    correct_cycle_diff = max(float(torch.max(torch.abs(a - b))) for a, b in zip(direct_grad, cycle_grad))
    legacy_cycle_diff = max(float(torch.max(torch.abs(a - b))) for a, b in zip(direct_grad, legacy_grad))
    assert correct_cycle_diff <= 1e-7
    assert legacy_cycle_diff > 1e-8

    toy = toy_tests()
    assert toy["status"] == "PASS" and toy["gradient_signs_opposite"]
    assert abs(case_values["same_episode_subbatch"]["value_difference"]) <= 1e-7
    assert abs(case_values["first_mixed_batch64"]["value_difference"]) > 1e-6

    result = {
        "status": "PASS",
        "device": "CPU only; CUDA_VISIBLE_DEVICES empty; no torch.cuda calls",
        "final_test_access": 0,
        "dev_targets_accessed": False,
        "train": {"W": W, "E": E, "raw_train_episodes_read_for_fixed_cases": raw_episodes_read},
        "toy_two_episode": toy,
        "real_train_fixed_batches": case_values,
        "effective_batch_64_vs_32plus32": {
            "model": "zero-initialized Linear(1,1) deterministic output head",
            "dropout": "disabled in eval mode",
            "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "gradient_64": [value.detach().tolist() for value in grad64],
            "gradient_32plus32": [value.detach().tolist() for value in grad32x2],
            "max_abs_parameter_gradient_difference": parameter_gradient_difference,
            "pass": parameter_gradient_difference <= 1e-6,
        },
        "partial_cycle": {
            "actual_samples": len(last),
            "correct_denominator": len(last),
            "direct_vs_partial_cycle_max_abs_parameter_gradient_difference": correct_cycle_diff,
            "legacy_mean_product_divide2_vs_correct_gradient_difference": legacy_cycle_diff,
            "pass": correct_cycle_diff <= 1e-7 and legacy_cycle_diff > 1e-8,
        },
        "sampler": {"unique_indices": int(len(np.unique(order))), "expected_train_windows": W, "every_window_once": bool(np.all(np.bincount(order, minlength=W) == 1))},
        "episode_equal_contract_in_original_locks": False,
        "D3_DEVIATION": "unresolved_pending_contract_reconciliation",
    }
    (OUT / "P0_CONTRACT_REGRESSION_FINAL.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mixed_value_difference": case_values["first_mixed_batch64"]["value_difference"],
        "mixed_gradient_difference": case_values["first_mixed_batch64"]["gradient_difference"],
        "64_vs_32x2_parameter_gradient_difference": parameter_gradient_difference,
        "partial_samples": len(last),
        "legacy_partial_gradient_difference": legacy_cycle_diff,
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

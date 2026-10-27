#!/usr/bin/env python3
"""CPU-only synthetic contracts required before the first V2 GPU fit."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LOCKED = CODE / "locked_source"
sys.path.insert(0, str(LOCKED))
sys.path.insert(0, str(CODE))

from d3_eq_contract import episode_equal_weights, weighted_batch_loss  # noqa: E402
from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash  # noqa: E402
from v2_scheduler import UpdateDrivenCosineAnnealingLR  # noqa: E402


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(881122)
    np.random.seed(881122)
    device = torch.device("cpu")
    checks: dict[str, Any] = {}

    # Unequal episode window counts reject mean(loss) * mean(weight).
    episode_ids = np.asarray([11, 11, 11, 29], dtype=np.int64)
    weights_np = episode_equal_weights(episode_ids)
    losses = torch.tensor([1.0, 1.0, 1.0, 5.0], dtype=torch.float32)
    weights = torch.as_tensor(weights_np, dtype=torch.float32)
    exact = weighted_batch_loss(losses, weights)
    incorrect = losses.mean() * weights.mean()
    if not torch.isclose(exact, torch.tensor(3.0)) or torch.isclose(exact, incorrect):
        raise AssertionError("weighted batch objective failed to distinguish the incorrect reduction")
    checks["episode_weight_objective"] = {"status": "PASS", "exact": float(exact), "incorrect_mean_times_mean": float(incorrect)}

    batch = 2
    prefix = {
        "state": torch.randn(batch, 32, 71),
        "history_mask": torch.ones(batch, 32),
        "relative_delta": torch.randn(batch, 32),
        "anchor_command": torch.randn(batch, 28),
        "dynamic": torch.randn(batch, 16, 4, 768),
        "static": torch.randn(batch, 16, 4, 768),
    }
    task_id = torch.tensor([0, 0], dtype=torch.long)

    state_model = V2TaskConditionedPredictor(seed=202610130, video_mode="STATE-CTRL").to(device).eval()
    fusion_model = V2TaskConditionedPredictor(seed=202610130, video_mode="TS11-D").to(device).eval()
    state_sd = state_model.state_dict(); fusion_sd = fusion_model.state_dict()
    if set(state_sd) != set(fusion_sd) or any(not torch.equal(state_sd[k], fusion_sd[k]) for k in state_sd):
        raise AssertionError("state/fusion fresh initialization is not an exact common-parameter clone")
    with torch.no_grad():
        state_model.output.weight.normal_(0.0, 0.02)
        state_model.output.bias.zero_()
        fusion_model.output.weight.copy_(state_model.output.weight)
        fusion_model.output.bias.copy_(state_model.output.bias)
        before = state_model(prefix, task_id).action_residual
        changed = dict(prefix)
        changed["dynamic"] = torch.randn_like(prefix["dynamic"]) * 100.0
        changed["static"] = torch.randn_like(prefix["static"]) * 100.0
        after = state_model(changed, task_id).action_residual
        if not torch.equal(before, after):
            raise AssertionError("state-only output depends on RGB cache values")
        fusion_before = fusion_model(prefix, task_id).action_residual
        fusion_after = fusion_model(changed, task_id).action_residual
        if torch.equal(fusion_before, fusion_after):
            raise AssertionError("fusion output did not respond to a changed RGB prefix fixture")
    checks["state_video_mask_and_common_init"] = {"status": "PASS", "parameter_count": parameter_count(state_model)}

    aux = V2TaskConditionedPredictor(seed=202610131, video_mode="TS11-D", waypoint_mode="AUX_ONLY").to(device)
    goal = V2TaskConditionedPredictor(seed=202610131, video_mode="TS11-D", waypoint_mode="GOAL_COND").to(device)
    aux_sd = aux.state_dict(); goal_sd = goal.state_dict()
    if set(aux_sd) != set(goal_sd) or any(not torch.equal(aux_sd[k], goal_sd[k]) for k in aux_sd):
        raise AssertionError("AUX_ONLY/GOAL_COND fresh initialization differs")
    aux.eval(); goal.eval()
    with torch.no_grad():
        goal.output.weight.normal_(0.0, 0.02)
        goal.output.bias.zero_()
        aux.output.weight.copy_(goal.output.weight)
        aux.output.bias.copy_(goal.output.bias)
        goal_out_1 = goal(prefix, task_id)
        goal_out_2 = goal(prefix, task_id)
        if not torch.equal(goal_out_1.action_residual, goal_out_2.action_residual):
            raise AssertionError("fixed-prefix forward is not deterministic in eval mode")
        aux_out = aux(prefix, task_id)
        if goal_out_1.goal_prediction is None or aux_out.goal_prediction is None:
            raise AssertionError("waypoint heads are missing")
        zero_token = torch.zeros(batch, 1, 128)
        predicted_token = goal.goal_projection(goal_out_1.goal_prediction).unsqueeze(1)
        zero_action = goal.qa_decode_with_goal_token(prefix, task_id, zero_token)
        predicted_action = goal.qa_decode_with_goal_token(prefix, task_id, predicted_token)
        if torch.equal(zero_action, predicted_action):
            raise AssertionError("predicted goal memory token does not affect GOAL_COND action path")
        if goal_out_1.action_residual.shape != (batch, 30, 28):
            raise AssertionError("waypoint action output shape mismatch")
    checks["waypoint_forward_and_injection"] = {"status": "PASS", "aux_only_goal_projection_active": False, "goal_cond_projection_active": True}

    # Future labels affect only external loss values; forward accepts no target argument.
    future_a = torch.zeros(batch, 30, 28)
    future_b = torch.ones(batch, 30, 28)
    sigma = torch.ones(28)
    anchor = prefix["anchor_command"]
    pred_before = goal(prefix, task_id).action_residual.detach().clone()
    g_target_a = (future_a[:, -1] - anchor) / sigma
    g_target_b = (future_b[:, -1] - anchor) / sigma
    g_loss_a = torch.abs(goal(prefix, task_id).goal_prediction - g_target_a).mean()
    g_loss_b = torch.abs(goal(prefix, task_id).goal_prediction - g_target_b).mean()
    pred_after = goal(prefix, task_id).action_residual.detach().clone()
    if not torch.equal(pred_before, pred_after) or torch.equal(g_loss_a, g_loss_b):
        raise AssertionError("future-label change affected forward or failed to affect external goal loss")

    # Both waypoint gradients exist on GOAL_COND; AUX_ONLY goal projection is intentionally unused.
    goal.train(); goal.zero_grad(set_to_none=True)
    out = goal(prefix, task_id)
    action_loss = out.action_residual.square().mean()
    aux_loss = torch.abs(out.goal_prediction - g_target_a).mean()
    (action_loss + 0.1 * aux_loss).backward()
    if goal.goal_head[0].weight.grad is None or goal.goal_projection.weight.grad is None:
        raise AssertionError("GOAL_COND lacks expected prediction/condition gradient paths")
    aux.train(); aux.zero_grad(set_to_none=True)
    aux_out = aux(prefix, task_id)
    (aux_out.action_residual.square().mean() + 0.1 * torch.abs(aux_out.goal_prediction - g_target_a).mean()).backward()
    if aux.goal_head[0].weight.grad is None or aux.goal_projection.weight.grad is not None:
        raise AssertionError("AUX_ONLY gradient-path accounting mismatch")
    checks["future_label_separation_and_gradients"] = {"status": "PASS", "future_labels_in_forward": False}

    # Fresh-process-safe tensor serialization contract, exercised in a CPU temporary artifact.
    state_cpu = {name: value.detach().cpu().contiguous() for name, value in goal.state_dict().items()}
    temp_path = ROOT / "tmp/CPU_MODEL_ROUNDTRIP.safetensors"
    save_file(state_cpu, str(temp_path))
    reloaded = V2TaskConditionedPredictor(seed=202610131, video_mode="TS11-D", waypoint_mode="GOAL_COND").to(device)
    reloaded.load_state_dict(load_file(str(temp_path), device="cpu"), strict=True)
    reloaded.eval(); goal.eval()
    with torch.no_grad():
        original = goal(prefix, task_id).action_residual
        restored = reloaded(prefix, task_id).action_residual
    if not torch.allclose(original, restored, rtol=0.0, atol=1e-7):
        raise AssertionError("safetensors strict reload changed fixed-prefix output")
    checks["safetensors_strict_reload"] = {"status": "PASS", "model_hash": state_hash(reloaded), "tensor_file": str(temp_path)}

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4, weight_decay=1e-4)
    scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    for epoch in range(1, 6):
        scheduler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        (parameter.square().sum()).backward()
        optimizer.step()
    if scheduler.step_calls != 5 or scheduler.legacy_epoch_step_calls != 0:
        raise AssertionError("update-driven scheduler call count mismatch")
    if abs(float(optimizer.param_groups[0]["lr"]) - 1e-5) > 2e-11:
        raise AssertionError("update-driven scheduler failed to reach locked eta_min")
    scheduler.close()
    checks["scheduler_update_count"] = {"status": "PASS", "T_max": 5, "updates": 5, "eta_min": 1e-5}

    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "device": "CPU",
        "tests": checks,
        "future_label_reads_in_forward": 0,
        "final_heldout_payload_access": 0,
    }
    path = ROOT / "qa/FIRST_REAL_PILOT_PRECHECK.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

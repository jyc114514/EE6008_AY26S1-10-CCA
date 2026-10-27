#!/usr/bin/env python3
"""CPU-only contract tests for the state/video information masks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from state_video_model import StateVideoPredictor, parameter_count  # noqa: E402


def main() -> int:
    torch.manual_seed(7)
    state = torch.randn(4, 32, 71)
    video = torch.randn(4, 16, 768)
    mask = torch.ones(4, 32)
    relative = torch.randn(4, 32)
    anchor = torch.randn(4, 28)
    model = StateVideoPredictor(horizon=30, seed=20260953)
    model.eval()
    assert parameter_count(model) == 1_155_356
    full_state, full_video = model.masked_inputs(state, video, state_level="FULL", video_level="D")
    obs_state, obs_video = model.masked_inputs(state, video, state_level="OBS", video_level="D")
    anchor_state, anchor_video = model.masked_inputs(state, video, state_level="ANCHOR", video_level="N")
    assert torch.equal(full_state, state) and torch.equal(full_video, video)
    assert torch.equal(obs_state[..., :43], state[..., :43]) and torch.count_nonzero(obs_state[..., 43:]) == 0
    assert torch.equal(obs_video, video)
    assert torch.count_nonzero(anchor_state) == 0 and torch.count_nonzero(anchor_video) == 0
    changed = state.clone()
    changed[..., 43:] += 1000.0
    changed_all = state + 1000.0
    assert torch.equal(model.masked_inputs(changed, video, state_level="OBS", video_level="D")[0], obs_state)
    assert torch.equal(model.masked_inputs(changed_all, video, state_level="ANCHOR", video_level="N")[0], anchor_state)
    assert not torch.equal(model.masked_inputs(changed, video, state_level="FULL", video_level="D")[0], full_state)
    output_obs = model.forward_locked(state, mask, relative, video, anchor, state_level="OBS", video_level="D")
    output_obs_changed = model.forward_locked(changed, mask, relative, video, anchor, state_level="OBS", video_level="D")
    output_anchor = model.forward_locked(state, mask, relative, video, anchor, state_level="ANCHOR", video_level="N")
    output_anchor_changed = model.forward_locked(changed_all, mask, relative, video, anchor, state_level="ANCHOR", video_level="N")
    assert torch.equal(output_obs, output_obs_changed)
    assert torch.equal(output_anchor, output_anchor_changed)
    assert float(output_obs.detach().abs().max()) == 0.0 and float(output_anchor.detach().abs().max()) == 0.0
    print({"status": "PASS", "parameter_count": parameter_count(model), "masked_column_sentinel": "PASS", "zero_residual_persistence": "PASS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Conditional train-only free-path positive controls for the locked P8 panel."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_tiny_fit as base  # noqa: E402

TRIGGER_PATH = ROOT / "protocol/B_FREE_PATH_TRIGGER_LOCK.json"
SOURCE_LOCK_PATH = ROOT / "protocol/B_FREE_PATH_SOURCE_LOCK.json"
OUT_ROOT = ROOT / "runs/free_path"


class IndependentFuturePath(nn.Module):
    """One independent standardized future residual per sample/time/raw-coordinate."""
    def __init__(self, n_samples: int, horizon: int = 30, action_dim: int = 28) -> None:
        super().__init__()
        self.residual = nn.Parameter(torch.zeros((n_samples, horizon, action_dim), dtype=torch.float32))

    def forward(self, state: torch.Tensor, history_mask: torch.Tensor, relative_delta: torch.Tensor,
                dynamic: torch.Tensor, static: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return self.residual


def verify_free_path_source_lock() -> str:
    lock = base.read_json(SOURCE_LOCK_PATH)
    for raw, expected in lock["file_hashes"].items():
        actual = base.sha256_file(Path(raw))
        if actual != expected:
            raise AssertionError(f"free-path source changed after lock: {raw}")
    return base.sha256_file(SOURCE_LOCK_PATH)


def run(loss_name: str, resume: bool = False) -> dict[str, Any]:
    if loss_name not in {"FULL_LOSS", "POS_ONLY"}:
        raise ValueError(loss_name)
    source_lock_hash = verify_free_path_source_lock()
    trigger = base.read_json(TRIGGER_PATH)
    if trigger.get("status") != "LOCKED_BEFORE_FREE_PATH_CONTROLS" or trigger.get("selected_panel") != "P8":
        raise AssertionError("P8 trigger lock is absent")
    if not all(not x["strong_training_fit"] for x in trigger["P8"]["cells"]):
        raise AssertionError("P8 free-path trigger does not match the four head-cell outcomes")
    p8_ids = trigger["selected_anchor_ids"]
    protocol = base.read_json(base.PROTO_PATH)
    protocol_hash = base.sha256_file(base.PROTO_PATH)
    trigger_hash = base.sha256_file(TRIGGER_PATH)
    panel_hash = base.sha256_file(base.PANEL_PATH)
    execution = base.read_json(base.EXECUTION_LOCK_PATH)
    gate = base.read_json(base.RESOURCE_GATE_PATH)
    if execution.get("device") != "cpu" or gate.get("status") != "PASS":
        raise AssertionError("locked CPU/resource execution gate is not PASS")
    base.set_cpu_runtime(int(execution["torch_threads"]))
    panels, metadata, _ = base.load_panels()
    if metadata["P8"]["anchor_id"].tolist() != p8_ids:
        raise AssertionError("free-path P8 IDs differ from frozen selection")
    tensors = panels["P8"]
    run_id = f"FREEPATH-H30-P8-{loss_name}-S20261002"
    run_root = OUT_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    complete_path = run_root / "RUN_COMPLETE.json"
    if complete_path.exists():
        complete = base.read_json(complete_path)
        if complete.get("status") == "PASS" and complete.get("trigger_lock_sha256") == trigger_hash:
            return {"status": "PASS_REUSED", "run_id": run_id}
        raise AssertionError("existing free-path completion conflicts with current trigger lock")
    latest = run_root / "checkpoint_latest.pt"
    if not resume and any(run_root.iterdir()):
        raise FileExistsError(f"partial free-path run exists; resume explicitly: {run_root}")
    model = IndependentFuturePath(len(p8_ids))
    if not torch.equal(model.residual.detach(), torch.zeros_like(model.residual)):
        raise AssertionError("free-path parameters must initialize exactly to zero")
    zero_init_hash = base.sha256_file(TRIGGER_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
    history: dict[str, Any] = {"summary": [], "samples": []}
    start_step=0
    if resume:
        if not latest.is_file(): raise FileNotFoundError("free-path resume checkpoint missing")
        state=torch.load(latest,map_location="cpu",weights_only=False)
        for key,expected in (("run_id",run_id),("loss_name",loss_name),("trigger_lock_sha256",trigger_hash),
                             ("protocol_sha256",protocol_hash),("panel_lock_sha256",panel_hash),
                             ("source_lock_sha256",source_lock_hash)):
            if state.get(key)!=expected: raise AssertionError(f"free-path resume mismatch: {key}")
        model.load_state_dict(state["model"],strict=True);optimizer.load_state_dict(state["optimizer"])
        history=state["history"];start_step=int(state["optimizer_updates"])
    step_times=[];start_wall=time.perf_counter()
    for step in range(start_step+1,2001):
        t0=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        residual=model(tensors["state"],tensors["mask"],tensors["relative"],tensors["dynamic"],tensors["static"],tensors["anchor"])
        parts=base.loss_components(residual,tensors["target"],tensors["anchor"],tensors["sigma"],tensors["safe_fd"])
        locked=base.per_sample_loss(residual,tensors["target"],tensors["anchor"],tensors["sigma"],tensors["safe_fd"])
        if not torch.allclose(parts["full"],locked,atol=1e-7,rtol=0): raise AssertionError("free-path loss contract mismatch")
        objective=parts["full"].mean() if loss_name=="FULL_LOSS" else parts["position"].mean()
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();model.zero_grad(set_to_none=True)
        step_times.append(time.perf_counter()-t0)
        if step%100==0:
            summary,samples,pred=base.evaluate(model,tensors,metadata["P8"],step,run_id,loss_name)
            history["summary"]=[x for x in history["summary"] if int(x["step"])!=step]+[summary]
            history["samples"]=[x for x in history["samples"] if int(x["step"])!=step]+samples
            base.write_csv(run_root/"METRICS.csv",history["summary"]);base.write_csv(run_root/"SAMPLE_METRICS.csv",history["samples"])
            with (run_root/f"PREDICTIONS_STEP_{step:04d}.npz").open("wb") as f:
                np.savez_compressed(f,anchor_id=metadata["P8"]["anchor_id"].astype("U96"),
                                    target_native=tensors["target"].numpy(),anchor_native=tensors["anchor"].numpy(),
                                    prediction_native=pred.astype(np.float32))
            payload={"run_id":run_id,"loss_name":loss_name,"optimizer_updates":step,"model":model.state_dict(),
                     "optimizer":optimizer.state_dict(),"history":history,"trigger_lock_sha256":trigger_hash,
                     "protocol_sha256":protocol_hash,"panel_lock_sha256":panel_hash,"source_lock_sha256":source_lock_hash,
                     "independent_residual_shape":list(model.residual.shape),"device":"cpu"}
            base.atomic_checkpoint(latest,payload)
    summary,samples,pred=base.evaluate(model,tensors,metadata["P8"],2000,run_id,loss_name)
    history["summary"]=[x for x in history["summary"] if int(x["step"])!=2000]+[summary]
    history["samples"]=[x for x in history["samples"] if int(x["step"])!=2000]+samples
    base.write_csv(run_root/"METRICS.csv",history["summary"]);base.write_csv(run_root/"SAMPLE_METRICS.csv",history["samples"])
    with (run_root/"PREDICTIONS_FINAL.npz").open("wb") as f:
        np.savez_compressed(f,anchor_id=metadata["P8"]["anchor_id"].astype("U96"),
                            target_native=tensors["target"].numpy(),anchor_native=tensors["anchor"].numpy(),
                            prediction_native=pred.astype(np.float32))
    final_payload={"model":model.state_dict(),"run_id":run_id,"loss_name":loss_name,"optimizer_updates":2000,
                   "trigger_lock_sha256":trigger_hash,"protocol_sha256":protocol_hash,"panel_lock_sha256":panel_hash,
                   "source_lock_sha256":source_lock_hash,"independent_residual_shape":list(model.residual.shape),
                   "parameter_count":int(model.residual.numel()),"initialized_at_zero":True,"device":"cpu"}
    base.atomic_checkpoint(run_root/"checkpoint_final.pt",final_payload)
    if latest.exists():latest.unlink()
    complete={"status":"PASS","run_id":run_id,"loss_name":loss_name,"panel":"P8","seed":20261002,
              "optimizer_updates":2000,"independent_trainable_values":int(model.residual.numel()),
              "independent_residual_shape":list(model.residual.shape),"init_value":"exact zeros",
              "protocol_sha256":protocol_hash,"panel_lock_sha256":panel_hash,"trigger_lock_sha256":trigger_hash,
              "source_lock_sha256":source_lock_hash,"device":"cpu","torch_threads":torch.get_num_threads(),
              "elapsed_seconds":time.perf_counter()-start_wall,
              "mean_update_seconds":float(np.mean(step_times)),"p95_update_seconds":float(np.quantile(step_times,.95)),
              "strong_training_fit":bool(summary["all_samples_strong_fit"]),"strong_fit_samples":int(summary["per_sample_strong_fit_count"]),
              "panel_rows":8,"dev_targets_opened":False,"final_or_heldout_access":0,"new_benchmark_candidate":False,
              "final_checkpoint_sha256":base.sha256_file(run_root/"checkpoint_final.pt"),
              "prediction_sha256":base.sha256_file(run_root/"PREDICTIONS_FINAL.npz"),
              "training_target_data_used":True,"not_a_forecasting_model":True}
    base.write_json(complete_path,complete)
    return complete


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument('--loss',choices=['FULL_LOSS','POS_ONLY'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    result=run(args.loss,args.resume);print(json.dumps(result,sort_keys=True,default=str));return 0

if __name__=='__main__':raise SystemExit(main())

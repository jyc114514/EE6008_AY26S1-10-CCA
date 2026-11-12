"""Four CPU updates comparing reconstructed team/personal state-control paths."""

import argparse
import importlib
import json
from pathlib import Path
import random
import sys
import os

from evaluate_videomae_core_b import digest, require


from path_config import scratch_root
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    scratch = scratch_root(project)
    require(output.is_relative_to(scratch) and output != scratch and not output.exists(), "Fresh personal output required")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = args.device
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
    if device == "cuda":
        require(physical in {"0", "1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == physical,
                "GPU check requires shared watcher assignment")
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.set_device(0)
    code = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
    expected = {"d3_model.py": "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589",
                "d3_eq_contract.py": "7359c956bf405092dade2e49e72f236759099330357652469c610d422c770b1c",
                "corrected_scheduler.py": "493c54acfedb7a1471b5b63c6e584846f01df5bf425964efe9b4129c4ef5c0f4",
                "d3_train_corrected.py": "342cf4f3423f3e5f0a4701661f53bf9e1498d2f3f7daa1351f83902eb2029e6a"}
    for name, sha in expected.items():
        require(digest(code / name) == sha, f"Source changed: {name}")
    sys.path.insert(0, str(code))
    models, contract, sched = [importlib.import_module(name) for name in
                              ("d3_model", "d3_eq_contract", "corrected_scheduler")]
    v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data"
    input_path = v3 / "V3_MAIN_INPUTS.npz"
    require(digest(input_path) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input mismatch")
    keys = ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")
    with np.load(input_path, allow_pickle=False) as source:
        rows = np.flatnonzero(source["split_code"] == 0)
        ids = source["anchor_id"][rows].copy()
        episodes = source["episode_index"][rows].copy()
        team_inputs = {key: source[key][rows].copy() for key in keys}
    require(len(rows) == 17757 and len(np.unique(episodes)) == 2712, "Train population mismatch")
    own_inputs = {key: value.astype(np.float32) for key, value in team_inputs.items()}
    eq = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    target_path = eq / "d3/train_targets_trainonly.npz"
    require(digest(target_path) == "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1", "Train cache mismatch")
    with np.load(target_path, allow_pickle=False) as source:
        require(np.array_equal(source["anchor_id"], ids) and np.array_equal(source["episode_index"], episodes), "Train join mismatch")
        team_anchors = source["anchor_command_native"].astype(np.float32)
        targets = source["targets_native_60"].astype(np.float32)
    norm_path = v3 / "V3_NORMALIZERS.json"
    require(digest(norm_path) == "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c", "Scale mismatch")
    norm = json.loads(norm_path.read_text())
    sigma, fd = [torch.tensor(norm[key], dtype=torch.float32, device=device) for key in ("target_sigma", "safe_fd_scale")]
    weights = contract.episode_equal_weights(episodes)
    seed = 20261001
    registry = json.loads((eq / "protocol/D3_EQ_SEED_REGISTRY.json").read_text())["seeds"][str(seed)]
    initial_path = project / f"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/runs/INIT_S{seed}.pt"
    require(digest(initial_path) == registry["init_state_file_sha256"], "Init mismatch")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def differences(left, right):
        require(set(left) == set(right), "Tensor schema mismatch")
        maximum = 0.0
        first = None
        for key in left:
            if left[key] is None or right[key] is None:
                require(left[key] is right[key], f"Missing gradient mismatch: {key}")
                continue
            error = float((left[key] - right[key]).abs().max())
            maximum = max(maximum, error)
            if error and first is None:
                first = key
        return {"max_abs": maximum, "first_differing_tensor": first}

    for horizon in (30, 60):
        reference = []
        branches = ("team", "team_repeat", "personal") if device == "cuda" else ("team", "personal")
        for branch in branches:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = models.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode="STATE-CTRL")
            model.load_state_dict(initial, strict=True)
            require(models.state_hash(model) == registry["init_state_hash"], "Initial model mismatch")
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
            scheduler = sched.UpdateDrivenCosineAnnealingLR(optimizer, T_max=11120, eta_min=1e-5)
            scheduler.set_epoch(1)
            order = contract.shuffled_epoch_order(len(rows), seed, 1)
            model.train()
            for step in range(5 if device == "cuda" else 4):
                positions = order[step * 64:(step + 1) * 64] if step < 4 else order[-29:]
                optimizer.zero_grad(set_to_none=True)
                inputs = team_inputs if branch.startswith("team") else own_inputs
                state, mask, relative = [torch.as_tensor(inputs[key][positions], device=device) for key in keys[:3]]
                anchor = torch.as_tensor((team_anchors if branch.startswith("team") else own_inputs[keys[3]])[positions], device=device)
                zeros = torch.zeros(len(positions), 16, 4, 768, device=device)
                rng_before = (torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()).clone()
                residual = model(state, mask, relative, zeros, zeros, anchor)
                sample = contract.per_sample_loss(residual, torch.as_tensor(targets[positions, :horizon], device=device), anchor, sigma, fd)
                loss = torch.sum(torch.as_tensor(weights[positions], dtype=torch.float32, device=device) * sample) / len(positions)
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                gradients = {key: None if p.grad is None else p.grad.detach().cpu().clone() for key, p in model.named_parameters()}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=(branch == "personal"))
                optimizer.step()
                state_after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if branch == "team":
                    reference.append((loss.item(), rng_before, gradients, state_after, residual.detach().cpu().clone(), scheduler.get_last_lr()[0]))
                else:
                    old_loss, old_rng, old_gradients, old_state, old_residual, old_lr = reference[step]
                    result = {"horizon": horizon, "step": step + 1, "branch": branch,
                              "batch_rows": len(positions), "rng_equal": bool(torch.equal(old_rng, rng_before)),
                              "loss_abs_difference": abs(old_loss - loss.item()),
                              "forward_max_abs_difference": float((old_residual - residual.detach().cpu()).abs().max()),
                              "gradients": differences(old_gradients, gradients),
                              "parameters": differences(old_state, state_after), "lr_equal": old_lr == scheduler.get_last_lr()[0]}
                    results.append(result)
                    print(json.dumps(result), flush=True)
            require(scheduler.step_calls == (5 if device == "cuda" else 4), "Scheduler count mismatch")
    passed = all(r["rng_equal"] and r["lr_equal"] and r["loss_abs_difference"] == 0
                 and r["forward_max_abs_difference"] == r["gradients"]["max_abs"] == r["parameters"]["max_abs"] == 0 for r in results)
    status = ("PASS_GPU_SHORT_PARITY" if passed else "GPU_SHORT_DIVERGENCE") if device == "cuda" else ("PASS_CPU_FOUR_STEP_PARITY" if passed else "CPU_FOUR_STEP_DIVERGENCE")
    report = {"status": status, "results": results,
              "input_dtypes": {key: str(value.dtype) for key, value in team_inputs.items()},
              "train_anchor_max_abs_difference": float(np.abs(team_anchors - own_inputs[keys[3]]).max()),
              "source_hashes": expected, "script_sha256": digest(Path(__file__)), "seed": seed,
              "torch": torch.__version__, "numpy": np.__version__, "gpu_used": device == "cuda", "physical_gpu": physical,
              "backend_settings": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
              "train_targets_loaded": True, "dev_targets_loaded": False, "final_test_access": 0,
              "scope": "Reconstructed state-only update paths, not execution of original full trainers; seed20261001, H30/H60. CUDA adds same-path repeat and 29-row batch after four full batches.",
              "limitations": "Tail probe is not reached through a full epoch. Passing does not establish full training reproduction; backend flags are observed, not changed."}
    (output / "UPDATE_PARITY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "UPDATE_PARITY.json"),
                      "gpu_used": device == "cuda", "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()

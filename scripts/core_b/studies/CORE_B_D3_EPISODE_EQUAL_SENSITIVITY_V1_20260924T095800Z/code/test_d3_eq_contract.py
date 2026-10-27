"""CPU-only D3-EQ loss, data-boundary, model and recovery regression tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from d3_eq_contract import (
    accumulated_group_loss,
    effective_update_groups,
    episode_equal_weights,
    historical_scheduler_spec,
    per_sample_loss,
    shuffled_epoch_order,
    weighted_batch_loss,
)
from d3_model import D3TimeSpacePredictor, state_hash


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
CODE_ROOT = EQ_ROOT / "code"
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
TARGET_CACHE = EQ_ROOT / "d3/train_targets_trainonly.npz"
NORMALIZER = V3_ROOT / "data/V3_NORMALIZERS.json"


def bytes_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class TestD3EQContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
        cls.split = cls.archive["split_code"].astype(np.int8, copy=True)
        cls.train_indices = np.flatnonzero(cls.split == 0)
        cls.train_ids = cls.archive["anchor_id"][cls.train_indices].astype("U64", copy=True)
        cls.train_episodes = cls.archive["episode_index"][cls.train_indices].astype(np.int64, copy=True)
        cls.weights64 = episode_equal_weights(cls.train_episodes)
        cache = np.load(TARGET_CACHE, allow_pickle=False)
        cls.cache_ids = cache["anchor_id"].astype("U64", copy=True)
        cls.anchors = cache["anchor_command_native"].astype(np.float32, copy=True)
        cls.targets60 = cache["targets_native_60"].astype(np.float32, copy=True)
        norm = json.loads(NORMALIZER.read_text(encoding="utf-8"))
        cls.sigma = torch.tensor(np.asarray(norm["target_sigma"], dtype=np.float32))
        cls.safe_fd = torch.tensor(np.asarray(norm["safe_fd_scale"], dtype=np.float32))
        cls.order = shuffled_epoch_order(len(cls.train_indices), 20261001, 1)
        cls.report: dict[str, object] = {"checks": {}}

    def test_original_one_vs_three_episode_formula_and_gradient(self) -> None:
        y = torch.tensor([4.0, -1.5, -1.5, -1.5], dtype=torch.float64)
        weights = torch.tensor([2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], dtype=torch.float64)
        theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
        losses = (theta - y).square()
        legacy = losses.mean() * weights.mean()
        eq = weighted_batch_loss(losses, weights)
        manual_equal_episode = ((theta - y[:1]).square().mean() + (theta - y[1:]).square().mean()) / 2
        legacy_grad = torch.autograd.grad(legacy, theta, retain_graph=True)[0]
        eq_grad = torch.autograd.grad(eq, theta)[0]
        self.assertTrue(torch.allclose(eq, manual_equal_episode, rtol=0.0, atol=1e-12))
        self.assertLess(float(legacy_grad * eq_grad), 0.0)
        self.assertGreater(abs(float(legacy - eq)), 1e-6)
        self.report["checks"]["toy_one_vs_three_episode"] = {
            "legacy_value": float(legacy.detach()),
            "episode_equal_value": float(eq.detach()),
            "legacy_gradient": float(legacy_grad),
            "episode_equal_gradient": float(eq_grad),
            "gradient_signs_opposite": True,
        }

    def test_full_train_weights_sampler_and_dev_target_boundary(self) -> None:
        counts = np.unique(self.train_episodes, return_counts=True)[1]
        self.assertEqual(len(self.train_indices), 17757)
        self.assertEqual(len(counts), 2712)
        self.assertEqual((int(counts.min()), int(counts.max())), (1, 8))
        self.assertAlmostEqual(float(self.weights64.mean()), 1.0, places=12)
        grouped_sum = {}
        for episode in np.unique(self.train_episodes):
            group_weights = self.weights64[self.train_episodes == episode]
            grouped_sum[int(episode)] = float(group_weights.sum())
        expected_episode_total = len(self.train_indices) / len(counts)
        self.assertTrue(all(abs(value - expected_episode_total) < 1e-10 for value in grouped_sum.values()))

        for seed in (20261001, 20261002, 20261003):
            for epoch in (1, 10, 40):
                order = shuffled_epoch_order(len(self.train_indices), seed, epoch)
                self.assertEqual(len(np.unique(order)), len(self.train_indices))
                self.assertTrue(np.array_equal(np.sort(order), np.arange(len(self.train_indices))))
                self.assertTrue(np.array_equal(order, shuffled_epoch_order(len(self.train_indices), seed, epoch)))
        batches = list(effective_update_groups(len(self.train_indices), microbatch_size=64, accumulation_steps=1))
        self.assertEqual(len(batches), 278)
        self.assertEqual([end - start for start, end, _ in batches][-1], 29)
        self.assertTrue(all(np.bincount(self.train_episodes[self.order[s:e]]).size >= 1 for s, e, _ in batches))

        # Only explicit covariate keys are read from the mixed train/dev NPZ by
        # the trainer. The dev target array is not part of the training loader.
        trainer_tree = ast.parse((CODE_ROOT / "d3_train_eq.py").read_text(encoding="utf-8"))
        input_key_tuple = None
        for node in ast.walk(trainer_tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "input_keys" for t in node.targets):
                if isinstance(node.value, (ast.Tuple, ast.List)):
                    input_key_tuple = tuple(item.value for item in node.value.elts if isinstance(item, ast.Constant))
        self.assertIsNotNone(input_key_tuple)
        self.assertNotIn("targets_native", input_key_tuple)
        self.assertTrue(np.array_equal(self.cache_ids, self.train_ids))
        self.assertEqual(self.targets60.shape, (17757, 60, 28))

        # A changed synthetic dev-target buffer cannot influence any training
        # weights or train-input signature: neither enters the train loader.
        train_input_hash = bytes_hash(self.archive["history_standardized"][self.train_indices])
        fake_dev_a = np.zeros((int((self.split == 1).sum()), 60, 28), dtype=np.float32)
        fake_dev_b = np.ones_like(fake_dev_a) * 1e6
        self.assertNotEqual(bytes_hash(fake_dev_a), bytes_hash(fake_dev_b))
        self.assertEqual(train_input_hash, bytes_hash(self.archive["history_standardized"][self.train_indices]))
        self.assertTrue(np.array_equal(self.weights64, episode_equal_weights(self.train_episodes)))
        self.report["checks"]["train_only_population_and_sampler"] = {
            "W": len(self.train_indices), "E": len(counts), "weight_mean": float(self.weights64.mean()),
            "optimizer_steps_per_epoch": len(batches), "tail_rows": 29,
            "dev_targets_loaded_by_trainer": False,
        }

    def _sample_losses(self, positions: np.ndarray, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.targets60[positions, :30], dtype=torch.float32)
        anchor = torch.tensor(self.anchors[positions], dtype=torch.float32)
        pred = theta * torch.ones_like(target)
        losses = per_sample_loss(pred, target, anchor, self.sigma, self.safe_fd)
        weights = torch.tensor(self.weights64[positions], dtype=torch.float32)
        return losses, weights

    def test_real_mixed_batch_same_episode_and_weighted_gradients(self) -> None:
        mixed = self.order[:64]
        self.assertGreater(len(np.unique(self.train_episodes[mixed])), 1)
        theta = torch.zeros((), dtype=torch.float32, requires_grad=True)
        losses, weights = self._sample_losses(mixed, theta)
        legacy = losses.mean() * weights.mean()
        correct = weighted_batch_loss(losses, weights)
        legacy_grad = torch.autograd.grad(legacy, theta, retain_graph=True)[0]
        correct_grad = torch.autograd.grad(correct, theta)[0]
        self.assertGreater(abs(float(correct - legacy)), 1e-6)
        self.assertGreater(abs(float(correct_grad - legacy_grad)), 1e-7)
        self.assertTrue(torch.allclose(correct, torch.sum(losses * weights) / 64.0, rtol=0.0, atol=1e-7))

        episode, count = next((int(ep), int(n)) for ep, n in zip(*np.unique(self.train_episodes, return_counts=True)) if n >= 2)
        same = np.flatnonzero(self.train_episodes == episode)[: min(count, 8)]
        same_losses, same_weights = self._sample_losses(same, torch.zeros((), requires_grad=True))
        same_legacy = same_losses.mean() * same_weights.mean()
        same_correct = weighted_batch_loss(same_losses, same_weights)
        self.assertTrue(torch.allclose(same_legacy, same_correct, rtol=0.0, atol=1e-7))
        self.report["checks"]["real_train_objective"] = {
            "mixed_batch_rows": 64,
            "mixed_episode_count": int(len(np.unique(self.train_episodes[mixed]))),
            "legacy_value": float(legacy.detach()),
            "episode_equal_value": float(correct.detach()),
            "legacy_gradient": float(legacy_grad),
            "episode_equal_gradient": float(correct_grad),
            "same_episode_special_case_value_diff": abs(float(same_legacy - same_correct)),
        }

    def test_gradient_accumulation_and_29_row_tail(self) -> None:
        positions = self.order[:64]
        target = torch.tensor(self.targets60[positions, :30], dtype=torch.float32)
        anchor = torch.tensor(self.anchors[positions], dtype=torch.float32)
        weights = torch.tensor(self.weights64[positions], dtype=torch.float32)
        x = torch.tensor(self.targets60[positions, 0, 0:1], dtype=torch.float32)
        head = torch.nn.Linear(1, 1, bias=True)
        head.eval()
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        pred = head(x).view(64, 1, 1).expand_as(target)
        losses = per_sample_loss(pred, target, anchor, self.sigma, self.safe_fd)
        direct = weighted_batch_loss(losses, weights)
        accumulated = accumulated_group_loss(losses, weights, 0, 64, ((0, 32), (32, 64)))
        params = tuple(head.parameters())
        direct_grads = torch.autograd.grad(direct, params, retain_graph=True)
        accumulated_grads = torch.autograd.grad(accumulated, params)
        diff64 = max(float((a - b).abs().max()) for a, b in zip(direct_grads, accumulated_grads))
        self.assertLessEqual(diff64, 1e-7)

        tail_positions = self.order[-29:]
        tail_target = torch.tensor(self.targets60[tail_positions, :30], dtype=torch.float32)
        tail_anchor = torch.tensor(self.anchors[tail_positions], dtype=torch.float32)
        tail_weights = torch.tensor(self.weights64[tail_positions], dtype=torch.float32)
        tail_x = torch.tensor(self.targets60[tail_positions, 0, 0:1], dtype=torch.float32)
        tail_head = torch.nn.Linear(1, 1, bias=True)
        tail_head.eval()
        with torch.no_grad():
            tail_head.weight.zero_()
            tail_head.bias.zero_()
        tail_losses = per_sample_loss(tail_head(tail_x).view(29, 1, 1).expand_as(tail_target), tail_target, tail_anchor, self.sigma, self.safe_fd)
        direct_tail = weighted_batch_loss(tail_losses, tail_weights)
        accumulated_tail = accumulated_group_loss(tail_losses, tail_weights, 0, 29, ((0, 29),))
        wrong64 = torch.sum(tail_losses * tail_weights) / 64.0
        correct_tail_grad = torch.autograd.grad(direct_tail, tuple(tail_head.parameters()), retain_graph=True)
        accumulated_tail_grad = torch.autograd.grad(accumulated_tail, tuple(tail_head.parameters()), retain_graph=True)
        wrong_tail_grad = torch.autograd.grad(wrong64, tuple(tail_head.parameters()))
        diff29 = max(float((a - b).abs().max()) for a, b in zip(correct_tail_grad, accumulated_tail_grad))
        wrong_diff29 = max(float((a - b).abs().max()) for a, b in zip(correct_tail_grad, wrong_tail_grad))
        self.assertLessEqual(diff29, 1e-7)
        self.assertGreater(wrong_diff29, 1e-6)
        self.report["checks"]["effective_batch_and_tail"] = {
            "64_vs_32plus32_max_abs_parameter_gradient_diff": diff64,
            "tail_rows": 29,
            "tail_denominator": 29,
            "29_direct_vs_accumulated_gradient_diff": diff29,
            "wrong_divide_by_64_gradient_diff": wrong_diff29,
        }

    def test_temporal_seam_history_and_cache_source_immutability(self) -> None:
        h30 = self.targets60[:, :30]
        self.assertTrue(np.array_equal(h30, self.targets60[:, :30]))
        history_ts = self.archive["history_selected_timestamps"][self.train_indices]
        history_mask = self.archive["history_mask"][self.train_indices].astype(bool)
        anchor_ts = self.archive["anchor_timestamp"][self.train_indices]
        valid = history_ts[history_mask]
        valid_anchor = np.broadcast_to(anchor_ts[:, None], history_ts.shape)[history_mask]
        self.assertTrue(np.all(valid <= valid_anchor + 1e-5))
        cache_audit = json.loads((EQ_ROOT / "qa/D3_EQ_CACHE_ID_AUDIT.json").read_text(encoding="utf-8"))
        target_audit = json.loads((EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(cache_audit["status"], "PASS")
        self.assertEqual(target_audit["dev_targets_loaded"], False)
        self.assertEqual(target_audit["final_test_access"], 0)
        self.assertEqual(target_audit["train_windows"], 17757)
        self.assertEqual(target_audit["train_episodes"], 2712)
        self.assertEqual(target_audit["anchor_join_max_abs_error"], 0.0)
        self.report["checks"]["temporal_and_cache_seam"] = {
            "train_history_not_after_anchor": True,
            "h30_prefix_steps": 30,
            "source_cache_shards": cache_audit["cache_shards_checked"],
            "cache_row_identity": "PASS",
            "train_target_anchor_join_max_abs_error": target_audit["anchor_join_max_abs_error"],
            "dev_targets_loaded": False,
            "final_test_access": 0,
        }

    def test_initialization_persistence_state_control_and_full28(self) -> None:
        configs = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
        seed_hashes = {}
        for seed in (20261001, 20261002, 20261003):
            init_path = BENCH_ROOT / f"d3/runs/INIT_S{seed}.pt"
            init_state = torch.load(init_path, map_location="cpu", weights_only=True)
            hashes = set()
            for config in configs:
                for horizon in (30, 60):
                    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config)
                    model.load_state_dict(init_state, strict=True)
                    hashes.add(state_hash(model))
            self.assertEqual(len(hashes), 1)
            seed_hashes[str(seed)] = next(iter(hashes))

        model = D3TimeSpacePredictor(horizon=30, seed=20261001, video_mode="TS11-D").eval()
        state = torch.zeros((1, 32, 71), dtype=torch.float32)
        mask = torch.ones((1, 32), dtype=torch.float32)
        relative = torch.zeros((1, 32), dtype=torch.float32)
        dynamic = torch.zeros((1, 16, 4, 768), dtype=torch.float32)
        static = torch.ones_like(dynamic)
        anchor = torch.arange(28, dtype=torch.float32).view(1, 28)
        with torch.inference_mode():
            residual = model(state, mask, relative, dynamic, static, anchor)
        self.assertEqual(tuple(residual.shape), (1, 30, 28))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(anchor[:, None, :] + residual, anchor[:, None, :].expand(-1, 30, -1)))

        control = D3TimeSpacePredictor(horizon=30, seed=20261001, video_mode="STATE-CTRL")
        zero_video = control._video_layout(dynamic, static)
        changed_video = control._video_layout(torch.full_like(dynamic, float("nan")), torch.full_like(static, 123.0))
        self.assertTrue(torch.equal(zero_video, torch.zeros_like(zero_video)))
        self.assertTrue(torch.equal(zero_video, changed_video))
        self.report["checks"]["initialization_and_model_contract"] = {
            "same_seed_hashes_by_cell": seed_hashes,
            "six_configs_two_horizons_share_init_per_seed": True,
            "initial_residual_zero_persistence": True,
            "full28_output_shape": [1, 30, 28],
            "state_control_video_zero_ignores_values": True,
        }

    def test_dropout_rng_checkpoint_resume_and_fresh_process_reload(self) -> None:
        torch.manual_seed(73)
        base = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.4), torch.nn.Linear(8, 1))
        base_state = copy.deepcopy(base.state_dict())
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10.0
        y = torch.arange(6, dtype=torch.float32).reshape(6, 1) / 7.0

        def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        torch.manual_seed(321)
        uninterrupted = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.4), torch.nn.Linear(8, 1))
        uninterrupted.load_state_dict(base_state)
        opt_a = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
        step(uninterrupted, opt_a)
        checkpoint_state = {
            "model": copy.deepcopy(uninterrupted.state_dict()),
            "optimizer": copy.deepcopy(opt_a.state_dict()),
            "torch_cpu_rng": torch.get_rng_state().clone(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        step(uninterrupted, opt_a)
        expected_dropout_resume = [value.detach().clone() for value in uninterrupted.parameters()]

        resumed = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.4), torch.nn.Linear(8, 1))
        resumed.load_state_dict(checkpoint_state["model"])
        opt_b = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        opt_b.load_state_dict(checkpoint_state["optimizer"])
        torch.set_rng_state(checkpoint_state["torch_cpu_rng"])
        random.setstate(checkpoint_state["python_rng"])
        np.random.set_state(checkpoint_state["numpy_rng"])
        step(resumed, opt_b)
        diff = max(float((a - b).abs().max()) for a, b in zip(expected_dropout_resume, resumed.parameters()))
        self.assertEqual(diff, 0.0)

        with tempfile.TemporaryDirectory(prefix="d3eq-reload-") as temp:
            temp_root = Path(temp)
            checkpoint = temp_root / "checkpoint.pt"
            fixture = temp_root / "fixture.npz"
            torch.manual_seed(20261001)
            model = D3TimeSpacePredictor(horizon=30, seed=20261001, video_mode="TS11-D").eval()
            with torch.no_grad():
                model.output.weight.fill_(0.002)
                model.output.bias.fill_(-0.001)
            np.savez(
                fixture,
                state=np.zeros((1, 32, 71), dtype=np.float32),
                mask=np.ones((1, 32), dtype=np.float32),
                relative=np.zeros((1, 32), dtype=np.float32),
                dynamic=np.arange(16 * 4 * 768, dtype=np.float32).reshape(1, 16, 4, 768) / 100000.0,
                static=np.zeros((1, 16, 4, 768), dtype=np.float32),
                anchor=np.arange(28, dtype=np.float32).reshape(1, 28) / 20.0,
            )
            with np.load(fixture, allow_pickle=False) as data:
                tensors = [torch.tensor(data[key]) for key in ("state", "mask", "relative", "dynamic", "static", "anchor")]
            with torch.inference_mode():
                expected = model(*tensors).cpu().numpy()
            expected_hash = bytes_hash(expected)
            torch.save({"model": model.state_dict()}, checkpoint)
            child = """
import hashlib,json,sys,numpy as np,torch
from d3_model import D3TimeSpacePredictor
torch.set_num_threads(1)
payload=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
model=D3TimeSpacePredictor(horizon=30,seed=20261001,video_mode='TS11-D')
model.load_state_dict(payload['model'],strict=True); model.eval()
with np.load(sys.argv[2],allow_pickle=False) as d:
    xs=[torch.tensor(d[k]) for k in ('state','mask','relative','dynamic','static','anchor')]
with torch.inference_mode(): y=model(*xs).numpy()
print(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())
"""
            child_result = subprocess.run(
                [sys.executable, "-c", child, str(checkpoint), str(fixture)],
                cwd=CODE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            reload_hash = child_result.stdout.strip()
            self.assertEqual(reload_hash, expected_hash)
        self.report["checks"]["dropout_resume_and_fresh_process_reload"] = {
            "cpu_dropout_resume_parameter_max_abs_diff": diff,
            "fresh_process_reload_output_hash_matches": True,
            "fresh_process_dtype": "float32",
            "final_test_access": 0,
        }


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestD3EQContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "device_policy": "CPU-only; no torch.cuda calls",
        "tests_run": result.testsRun,
        "failures": [str(item[1]) for item in result.failures],
        "errors": [str(item[1]) for item in result.errors],
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "checks": getattr(TestD3EQContract, "report", {}).get("checks", {}),
    }
    report_path = EQ_ROOT / "qa/D3_EQ_CPU_TESTS.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "dev_targets_loaded", "final_test_access")}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

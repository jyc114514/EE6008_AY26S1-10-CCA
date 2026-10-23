"""CPU-only contracts for the corrected-scheduler D3-EQ sensitivity."""
from __future__ import annotations
import os

import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq
import torch

TASK_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = pathlib.Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
sys.path.insert(0, str(TASK_ROOT / "code"))

from corrected_scheduler import UpdateDrivenCosineAnnealingLR, closed_form_lr
from d3_eq_contract import effective_update_groups, weighted_batch_loss
from d3_model import D3TimeSpacePredictor, state_hash


class TestCorrectedScheduler(unittest.TestCase):
    def test_full_11120_step_trace_and_endpoints(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
        scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=11120, eta_min=1e-5)
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 1e-3)
        for update in range(11120):
            scheduler.set_epoch(update // 278 + 1)
            optimizer.zero_grad(set_to_none=True)
            param.square().sum().backward()
            optimizer.step()
            if (update + 1) % 278 == 0:
                scheduler.step()  # legacy epoch callback must not advance LR
        self.assertEqual(scheduler.step_calls, 11120)
        self.assertEqual(scheduler.legacy_epoch_step_calls, 40)
        self.assertEqual(scheduler.inner.last_epoch, 11120)
        self.assertEqual(len(scheduler.update_records), 11120)
        self.assertAlmostEqual(scheduler.update_records[0]["lr_used_for_update"], 1e-3, places=15)
        self.assertAlmostEqual(scheduler.update_records[-1]["lr_after_update"], 1e-5, delta=2e-11)
        lr = np.asarray([row["lr_after_update"] for row in scheduler.update_records])
        self.assertTrue(np.isfinite(lr).all())
        self.assertTrue(np.all(np.diff(lr) <= 1e-14))
        max_error = max(
            abs(row["lr_after_update"] - closed_form_lr(row["update_count_after"]))
            for row in scheduler.update_records
        )
        self.assertLess(max_error, 2e-11)

    def test_effective_batch_accumulation_and_tail_29(self) -> None:
        torch.manual_seed(9)
        x = torch.randn(64, 5)
        y = torch.randn(64, 2)
        weights = torch.linspace(0.3, 1.7, 64)
        seed_model = torch.nn.Linear(5, 2)
        initial = {key: value.detach().clone() for key, value in seed_model.state_dict().items()}
        full = torch.nn.Linear(5, 2)
        full.load_state_dict(initial)
        accumulated = torch.nn.Linear(5, 2)
        accumulated.load_state_dict(initial)
        full_loss = weighted_batch_loss((full(x) - y).abs().mean(dim=1), weights)
        full_loss.backward()
        for start in (0, 32):
            end = start + 32
            loss = (torch.sum((accumulated(x[start:end]) - y[start:end]).abs().mean(dim=1)
                              * weights[start:end]) / 64.0)
            loss.backward()
        for full_param, accum_param in zip(full.parameters(), accumulated.parameters()):
            self.assertTrue(torch.allclose(full_param.grad, accum_param.grad, rtol=0.0, atol=1e-7))
        tail = weighted_batch_loss(torch.ones(29), torch.full((29,), 0.5))
        self.assertEqual(float(tail), 0.5)
        groups = list(effective_update_groups(17757, microbatch_size=64, accumulation_steps=1))
        self.assertEqual(len(groups), 278)
        self.assertEqual(groups[-1][1] - groups[-1][0], 29)
        accum_groups = list(effective_update_groups(64, microbatch_size=32, accumulation_steps=2))
        self.assertEqual(len(accum_groups), 1)
        self.assertEqual(accum_groups[0][2], ((0, 32), (32, 64)))

    def test_checkpoint_resume_restores_dropout_rng_and_scheduler(self) -> None:
        def train(interrupt: bool):
            torch.manual_seed(1234)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 12), torch.nn.Dropout(0.2),
                torch.nn.Tanh(), torch.nn.Linear(12, 2),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=11120, eta_min=1e-5)
            x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 17
            y = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 13

            def step(update: int) -> None:
                scheduler.set_epoch(update // 10 + 1)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                (model(x) - y).square().mean().backward()
                optimizer.step()
                if (update + 1) % 10 == 0:
                    scheduler.step()

            if interrupt:
                with tempfile.TemporaryDirectory() as directory:
                    for update in range(20):
                        step(update)
                    checkpoint = pathlib.Path(directory) / "resume.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": torch.get_rng_state(),
                    }, checkpoint)
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    resumed_model = torch.nn.Sequential(
                        torch.nn.Linear(4, 12), torch.nn.Dropout(0.2),
                        torch.nn.Tanh(), torch.nn.Linear(12, 2),
                    )
                    resumed_model.load_state_dict(payload["model"])
                    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
                    resumed_scheduler = UpdateDrivenCosineAnnealingLR(
                        resumed_optimizer, T_max=11120, eta_min=1e-5
                    )
                    # Match the production order: scheduler exists before state restoration.
                    resumed_optimizer.load_state_dict(payload["optimizer"])
                    resumed_scheduler.load_state_dict(payload["scheduler"])
                    torch.set_rng_state(payload["rng"])
                    model, optimizer, scheduler = resumed_model, resumed_optimizer, resumed_scheduler
                    for update in range(20, 40):
                        step(update)
            else:
                for update in range(40):
                    step(update)
            return model.state_dict(), optimizer.state_dict(), scheduler.state_dict()

        uninterrupted = train(False)
        resumed = train(True)
        for key in uninterrupted[0]:
            self.assertTrue(torch.equal(uninterrupted[0][key], resumed[0][key]))
        self.assertEqual(uninterrupted[2]["step_calls"], resumed[2]["step_calls"])
        self.assertEqual(uninterrupted[2]["update_records"], resumed[2]["update_records"])

    def test_seed_init_matches_all_config_horizon_pairs_and_zero_residual(self) -> None:
        registry = __import__("json").loads((EQ / "protocol/D3_EQ_SEED_REGISTRY.json").read_text())
        for seed in (20261001, 20261002, 20261003):
            init_path = OLD / f"d3/runs/INIT_S{seed}.pt"
            init = torch.load(init_path, map_location="cpu", weights_only=True)
            hashes = []
            for config in ("TS11-D", "STATE-CTRL"):
                for horizon in (30, 60):
                    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config)
                    model.load_state_dict(init, strict=True)
                    hashes.append(state_hash(model))
            self.assertEqual(set(hashes), {registry["seeds"][str(seed)]["init_state_hash"]})
        seed = 20261001
        init = torch.load(OLD / f"d3/runs/INIT_S{seed}.pt", map_location="cpu", weights_only=True)
        model = D3TimeSpacePredictor(horizon=30, seed=seed, video_mode="TS11-D")
        model.load_state_dict(init, strict=True)
        model.eval()
        state = torch.zeros((1, 32, 71))
        history_mask = torch.zeros((1, 32))
        relative = torch.zeros((1, 32))
        video = torch.zeros((1, 16, 4, 768))
        anchor = torch.linspace(-1.0, 1.0, 28).reshape(1, 28)
        with torch.inference_mode():
            residual = model(state, history_mask, relative, video, video, anchor)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        expanded_anchor = anchor[:, None, :].expand_as(residual)

    def test_target_blind_input_selector_never_requests_future_targets(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("corrected_inference_for_test", TASK_ROOT / "code/d3_corrected_infer.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

        allowed = set(module.TARGET_BLIND_INPUT_KEYS)
        payload = {key: np.arange(6).reshape(3, 2) for key in allowed}
        payload["targets_native"] = np.full((3, 60, 28), 999999.0)
        accessed: list[str] = []

        class SentinelArchive:
            def __getitem__(self, key: str):
                accessed.append(key)
                if key not in allowed:
                    raise AssertionError(f"inference read forbidden key: {key}")
                return payload[key]

        selected = module.select_target_blind_inputs(SentinelArchive(), np.asarray([0, 2]))
        self.assertEqual(set(selected), allowed)
        self.assertNotIn("targets_native", accessed)
        payload["targets_native"][:] = -123456.0
        selected_again = module.select_target_blind_inputs(SentinelArchive(), np.asarray([0, 2]))
        for key in allowed:
            self.assertTrue(np.array_equal(selected[key], selected_again[key]))

    def test_train_dev_source_groups_are_disjoint(self) -> None:
        table = pq.read_table(SPLIT_MANIFEST, columns=[
            "episode_index", "split", "video_sha256_fingerprint"
        ])
        episodes = table["episode_index"].to_numpy(zero_copy_only=False)
        splits = table["split"].to_pylist()
        fingerprints = table["video_sha256_fingerprint"].to_pylist()
        mapping = {}
        for episode, split, fingerprint in zip(episodes, splits, fingerprints):
            split = "dev" if str(split) in {"validation", "val"} else str(split)
            if fingerprint is None or not str(fingerprint).strip():
                raise AssertionError(f"missing fingerprint for episode {episode}")
            mapping[int(episode)] = (split, str(fingerprint))
        archive = np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
        split_code = archive["split_code"].astype(np.int8, copy=False)
        episode = archive["episode_index"].astype(np.int64, copy=False)
        train_eps = set(map(int, episode[split_code == 0]))
        dev_eps = set(map(int, episode[split_code == 1]))
        train_groups = {mapping[ep][1] for ep in train_eps}
        dev_groups = {mapping[ep][1] for ep in dev_eps}
        self.assertEqual(len(train_eps), 2712)
        self.assertEqual(len(dev_eps), 603)
        self.assertEqual(train_groups & dev_groups, set())
        self.assertEqual(len(train_groups), 2712)
        self.assertEqual(len(dev_groups), 603)


if __name__ == "__main__":
    unittest.main()

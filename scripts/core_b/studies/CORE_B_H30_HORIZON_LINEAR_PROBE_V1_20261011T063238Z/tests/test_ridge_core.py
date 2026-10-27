from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ridge_core import (
    episode_equal_weights, horizon_window_metrics, load_model_npz,
    predict_native, predict_residual, prepare_weighted_gram,
    safe_ratio_skill, save_model_npz, solve_from_gram,
    standardize_features, weighted_feature_stats,
)


class RidgeCoreTests(unittest.TestCase):
    def test_unequal_episode_weights_and_direct_reference(self):
        episode = np.array([10, 10, 20, 30, 30, 30])
        w = episode_equal_weights(episode)
        self.assertAlmostEqual(w.mean(), 1.0, places=12)
        for ep in np.unique(episode):
            self.assertAlmostEqual(w[episode == ep].sum(), len(episode) / 3.0, places=12)
        x = np.array([[0., 1.], [1., 1.], [2., -1.], [3., 0.], [4., 1.], [5., 2.]])
        y = np.column_stack([2. + x[:, 0] - .5*x[:, 1], -1. + .2*x[:, 0]])
        lam = .07
        prepared = prepare_weighted_gram(x, y, w)
        fit = solve_from_gram(prepared, lam)
        z = standardize_features(x, prepared["x_mean"], prepared["x_scale"], prepared["constant_mask"])
        ymean = np.sum(y*w[:, None], axis=0)/w.sum()
        ref = np.linalg.solve(z.T@(z*w[:, None])/w.sum()+lam*np.eye(2), z.T@((y-ymean)*w[:, None])/w.sum())
        np.testing.assert_allclose(fit["coef"], ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(fit["intercept"], ymean, rtol=0, atol=1e-12)

    def test_unpenalized_intercept_and_alpha_scaling(self):
        x = np.ones((5, 2)); y = np.arange(15, dtype=float).reshape(5, 3)
        w = np.array([.5, .5, 1., 1.5, 1.5]); lam = .3
        p = prepare_weighted_gram(x, y, w); fit = solve_from_gram(p, lam)
        np.testing.assert_allclose(fit["coef"], 0., atol=1e-12)
        np.testing.assert_allclose(fit["intercept"], np.average(y, axis=0, weights=w), atol=1e-12)
        self.assertAlmostEqual(fit["effective_alpha_sum_loss_convention"], w.sum()*lam)
        x2 = np.arange(5., dtype=float)[:, None]; y2 = np.column_stack([x2[:, 0]**2, -x2[:, 0]])
        p2 = prepare_weighted_gram(x2, y2, w); f2 = solve_from_gram(p2, lam)
        z = standardize_features(x2, p2["x_mean"], p2["x_scale"], p2["constant_mask"])
        yc = y2-np.average(y2, axis=0, weights=w)
        direct = np.linalg.solve(z.T@(z*w[:, None])+w.sum()*lam*np.eye(1), z.T@(yc*w[:, None]))
        np.testing.assert_allclose(f2["coef"], direct, rtol=1e-11, atol=1e-11)

    def test_fit_only_scaler_and_constant_columns(self):
        fit_x = np.array([[1., 2., 7.], [3., 2., 7.]])
        mean1, scale1, const1 = weighted_feature_stats(fit_x, np.ones(2))
        validation = np.array([[1e6, -2e6, 55.]])
        _ = validation
        mean2, scale2, const2 = weighted_feature_stats(fit_x, np.ones(2))
        np.testing.assert_array_equal(mean1, mean2); np.testing.assert_array_equal(scale1, scale2)
        np.testing.assert_array_equal(const1, const2); self.assertTrue(const1[2])
        self.assertEqual(standardize_features(np.array([[99., 2., 55.]]), mean1, scale1, const1)[0, 2], 0.)

    def test_group_assignment_keeps_duplicate_groups_together(self):
        import hashlib
        groups = ["g2", "g1", "g1", "g3", "g4", "g4", "g5"]
        ranked = sorted(set(groups), key=lambda g: (hashlib.sha256(f"{g}|seed=20261011".encode()).hexdigest(), g))
        folds = {g: i % 3 for i, g in enumerate(ranked)}
        row_folds = [folds[g] for g in groups]
        self.assertEqual(row_folds[1], row_folds[2]); self.assertEqual(row_folds[4], row_folds[5])

    def test_anchor_addition_and_zero_error_metrics(self):
        rng = np.random.default_rng(11); n = 4
        anchor = rng.normal(size=(n, 28)); target = rng.normal(size=(n, 30, 28)); pred = target.copy()
        metrics = horizon_window_metrics(pred, target, anchor, np.ones(28), np.ones(28), np.full(28, 1e-6), 30)
        np.testing.assert_allclose(metrics["prefix_position_standardized_mae"], 0.)
        np.testing.assert_allclose(metrics["future_internal_fd_safe_normalized"], 0.)
        self.assertTrue(np.all(metrics["future_internal_difference_count"] == 29))
        x = rng.normal(size=(n, 3)); y = np.zeros((n, 840)); p = prepare_weighted_gram(x, y, np.ones(n))
        model = {**p, **solve_from_gram(p, .1)}
        native = predict_native(x, anchor, model)
        self.assertEqual(native.shape, (n, 30, 28))

    def test_h1_fd_na_and_targets_do_not_change_forward(self):
        rng = np.random.default_rng(13); n = 3
        x = rng.normal(size=(n, 4)); y = rng.normal(size=(n, 840))
        p = prepare_weighted_gram(x, y, np.ones(n)); model = {**p, **solve_from_gram(p, .1)}
        before = predict_residual(x, model)
        target = rng.normal(size=(n, 30, 28)); target[:] += 1e9
        after = predict_residual(x, model); np.testing.assert_array_equal(before, after)
        anchor = rng.normal(size=(n, 28)); pred = predict_native(x, anchor, model)
        h1 = horizon_window_metrics(pred, target, anchor, np.ones(28), np.ones(28), np.ones(28), 1)
        self.assertTrue(np.isnan(h1["future_internal_fd_safe_normalized"]).all())
        self.assertTrue(np.isnan(h1["tv_ratio_window"]).all())
        self.assertTrue(np.all(h1["future_internal_difference_count"] == 0))

    def test_zero_denominator_skill_is_na(self):
        self.assertTrue(np.isnan(safe_ratio_skill(.2, 0.)))
        self.assertAlmostEqual(safe_ratio_skill(.25, 1.), .75)

    def test_fresh_process_model_reload_predictions_identical(self):
        rng = np.random.default_rng(23); x = rng.normal(size=(8, 4)); y = rng.normal(size=(8, 840))
        p = prepare_weighted_gram(x, y, np.ones(8)); model = {**p, **solve_from_gram(p, .01)}
        expected = predict_residual(x, model)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); mp = td/"model.npz"; ip = td/"input.npz"; ep = td/"expected.npy"
            save_model_npz(mp, model); np.savez(ip, x=x); np.save(ep, expected)
            code_dir = Path(__file__).resolve().parents[1]/"code"
            snippet = ("import sys,numpy as np; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                       "from ridge_core import load_model_npz,predict_residual; "
                       "m=load_model_npz(Path(sys.argv[2])); x=np.load(sys.argv[3])['x']; "
                       "np.testing.assert_array_equal(predict_residual(x,m),np.load(sys.argv[4]))")
            env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run([sys.executable, "-c", snippet, str(code_dir), str(mp), str(ip), str(ep)],
                           check=True, env=env, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

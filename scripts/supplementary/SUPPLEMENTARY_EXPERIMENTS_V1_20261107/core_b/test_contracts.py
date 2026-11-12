"""Focused leakage, rejection and metric-denominator checks; no GPU or real targets."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_cascade import load_classification, TASKS, PAIRS, REC, exact_profile_lock, sha
from score_cascade import (aggregate, components, PROGRAMS, BOOTSTRAP_SEED,
                           validate_bundle_masks, shared_episode_resamples,
                           bootstrap_from_shared_indices)


class Contracts(unittest.TestCase):
    def test_profile_lock_uses_only_an_exact_matching_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / REC / 'core_b'
            directory.mkdir(parents=True)
            current = directory / 'CURRENT_PROFILE_CANDIDATE_LOCK.json'
            backup = directory / 'CURRENT_PROFILE_CANDIDATE_LOCK.json.pre-final.bak'
            current.write_text('{"revision":"final"}', encoding='utf-8')
            backup.write_text('{"revision":"dev"}', encoding='utf-8')
            self.assertEqual(exact_profile_lock(tmp, sha(current)), current)
            self.assertEqual(exact_profile_lock(tmp, sha(backup)), backup)
            with self.assertRaisesRegex(ValueError, 'No current or historical candidate lock'):
                exact_profile_lock(tmp, '0' * 64)

    def test_unrestricted_task_prediction_and_frame_cutoff(self):
        data = {'anchor_id': np.asarray(['a', 'b', 'c', 'd']), 'episode_id': np.arange(1, 5),
                'task_id': np.asarray([119, 119, 94, 90]), 'anchor_frame': np.asarray([119, 119, 119, 119])}
        rows = [{'episode_id': 1, 'predicted_task_id': 94, 'observation_end_frame': 116},
                {'episode_id': 2, 'predicted_task_id': 777, 'observation_end_frame': 116},
                {'episode_id': 3, 'predicted_task_id': 94, 'observation_end_frame': 120}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pred.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            predicted, valid, audit = load_classification(path, data, 20260825)
        self.assertEqual(predicted.tolist(), [94, 777, 94, -1])
        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertEqual(audit[0]['true_task_id'], 119)
        self.assertEqual(audit[1]['status'], 'unsupported_core_a_task')
        self.assertEqual(audit[2]['status'], 'classification_observes_after_core_b_anchor')
        np.testing.assert_array_equal(data['task_id'], [119, 119, 94, 90])

    def test_position_temporal_seam_have_distinct_meanings(self):
        target = np.zeros((2, 30, 28))
        pred = np.zeros_like(target)
        pred[0] += 2
        pred[1, :, :] = np.arange(1, 31)[:, None]
        got = components(pred, target, np.zeros((2, 28)), np.ones(28), np.ones(28))
        np.testing.assert_allclose(got[0], [2, 0, 2])
        np.testing.assert_allclose(got[1], [15.5, 1, 1])

    def test_episode_then_task_equal_weighting(self):
        # Unequal anchors and unequal episodes must not become anchor-weighted.
        data = {'task_id': np.asarray([119, 119, 119, 94]), 'episode_id': np.asarray([1, 1, 2, 3])}
        values = np.asarray([[0, 0, 0], [0, 0, 0], [6, 6, 6], [1, 1, 1]], float)
        avg, tasks, eps = aggregate(values, np.ones(4, bool), data)
        self.assertEqual(avg['position_mae'], 2.0)  # task119=(ep1=0,ep2=6)/2=3; task94=1; macro=2.
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(eps), 3)
        avg_subset, tasks_subset, _ = aggregate(values, np.asarray([True, True, False, True]), data)
        self.assertEqual(avg_subset['position_mae'], 0.5)

    def test_every_program_mask_matches_audit_and_oracle_is_complete(self):
        bundles, audits = {}, {}
        for a_seed, b_seed in PAIRS:
            audits[a_seed] = [{'status': 'accepted'}, {'status': 'unsupported_core_a_task'}]
            for program in PROGRAMS:
                bundles[('oracle_task', program, b_seed)] = (None, np.ones(2, bool))
                bundles[('core_a_task', program, b_seed)] = (None, np.asarray([True, False]))
        bundles[('persistence', 'PERSISTENCE', None)] = (None, np.ones(2, bool))
        validate_bundle_masks(bundles, audits, 2)
        key = ('core_a_task', 'MIXD', PAIRS[1][1])
        bundles[key] = (None, np.ones(2, bool))
        with self.assertRaisesRegex(ValueError, 'differs from classification audit'):
            validate_bundle_masks(bundles, audits, 2)
        bundles[key] = (None, np.asarray([True, False]))
        bundles[('oracle_task', 'S_OPT', PAIRS[2][1])] = (None, np.asarray([True, False]))
        with self.assertRaisesRegex(ValueError, 'Oracle valid mask is not all True'):
            validate_bundle_masks(bundles, audits, 2)

    def test_common_bootstrap_uses_original_seed_and_shared_draws(self):
        data = {'task_id': np.repeat(TASKS, 3), 'episode_id': np.arange(15)}
        episodes, indices = shared_episode_resamples(data, np.ones(15, bool))
        self.assertEqual(BOOTSTRAP_SEED, 2026101902)
        expected_rng = np.random.default_rng(2026101902)
        for index in indices:
            np.testing.assert_array_equal(index, expected_rng.integers(0, 3, size=(5000, 3)))
            self.assertFalse(index.flags.writeable)
        model_a = [np.arange(9).reshape(3, 3) + i for i in range(5)]
        model_b = [2 * values for values in model_a]
        draws_a = bootstrap_from_shared_indices(model_a, indices)
        draws_b = bootstrap_from_shared_indices(model_b, indices)
        np.testing.assert_array_equal(draws_b, 2 * draws_a)
        self.assertEqual(draws_a.shape, (5000, 3))


if __name__ == '__main__':
    unittest.main()

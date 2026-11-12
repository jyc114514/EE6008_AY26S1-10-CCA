"""CPU-only protocol tests; no original data, network, or GPU operations."""
import os
import tempfile
from pathlib import Path
import unittest

import numpy as np

from train_experts import (training_indices, training_weights, filter_epoch_order,
                           schedule_counts, freeze_cache_content, read_json)


class TrainingContracts(unittest.TestCase):
    def test_filter_preserves_global_relative_order_without_cross_class_rows(self):
        tasks = np.asarray([119, 94, 119, 90, 94, 119])
        selected = training_indices(tasks, 'expert', 119)
        global_order = np.asarray([4, 2, 1, 5, 0, 3], dtype=np.int64)
        np.testing.assert_array_equal(filter_epoch_order(global_order, selected, 6), [2, 5, 0])
        shared = training_indices(tasks, 'shared', None)
        np.testing.assert_array_equal(filter_epoch_order(global_order, shared, 6), global_order)
        with self.assertRaises(ValueError):
            filter_epoch_order(np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64), selected, 6)

    def test_class_and_episode_weights_remain_mean_one(self):
        tasks = np.asarray([119, 119, 119, 94, 94])
        episodes = np.asarray([1, 1, 2, 3, 3])
        selected = np.arange(5, dtype=np.int64)
        shared = training_weights(tasks, episodes, selected)
        np.testing.assert_allclose(shared, [0.625, 0.625, 1.25, 1.25, 1.25])
        np.testing.assert_allclose(shared.mean(), 1)
        expert_ix = training_indices(tasks, 'expert', 119)
        expert = training_weights(tasks, episodes, expert_ix)
        np.testing.assert_allclose(expert, [0.75, 0.75, 1.5])
        # Episode 1 has twice the windows but equal total loss weight to episode 2.
        np.testing.assert_allclose(expert[:2].sum(), expert[2])
        self.assertEqual(float(expert.sum()), 3.0)

    def test_schedules_keep_160_epochs_and_actual_partial_batch(self):
        self.assertEqual(schedule_counts(1280), (20, 3200))
        self.assertEqual(schedule_counts(129), (3, 480))
        self.assertEqual(schedule_counts(64), (1, 160))
        self.assertEqual(schedule_counts(65), (2, 320))

    def test_cache_freeze_is_first_write_then_exact_content_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / 'hardlink_probe'
            probe.write_text('synthetic fixture', encoding='utf-8')
            try:
                os.link(probe, Path(tmp) / 'hardlink_probe_copy')
            except OSError:
                self.skipTest('This filesystem does not permit the hardlinks required by the atomic cache-lock test')
            path = Path(tmp) / 'cache_lock.json'
            record = {'source': 'pinned', 'hash': 'a'}
            freeze_cache_content(path, record)
            freeze_cache_content(path, record)
            self.assertEqual(read_json(path), record)
            with self.assertRaisesRegex(ValueError, 'differs'):
                freeze_cache_content(path, {'source': 'pinned', 'hash': 'b'})


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from apply_fusion_feature_interventions import cross_task_donors, same_task_donors


def test_same_task_derangement_has_no_fixed_points():
    ids = np.arange(8)
    tasks = np.repeat([1, 2], 4)
    mapping = same_task_donors(ids, tasks, seed=20260911)
    assert all(mapping[k] != k for k in mapping)
    assert all(tasks[mapping[k]] == tasks[k] for k in mapping)


def test_cross_task_derangement_changes_task():
    ids = np.arange(6)
    tasks = np.array([1, 1, 2, 2, 3, 3])
    mapping = cross_task_donors(ids, tasks, seed=20260912)
    assert all(mapping[k] != k for k in mapping)
    assert all(tasks[mapping[k]] != tasks[k] for k in mapping)

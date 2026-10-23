"""Pure NumPy routines for the predeclared N5 paired resampling scopes."""

from itertools import product
import math

import numpy as np


EXPECTED_METRICS = ("position_standardized_mae", "safe_fd_normalized")
EXPECTED_SEEDS = (20261001, 20261002, 20261003)
EXPECTED_CELLS = ("TS11-D/Q0", "TS11-D/Q1", "STATE-CTRL/Q0", "STATE-CTRL/Q1")
CONTRASTS = ("G_TS", "G_STATE", "V_Q0", "V_Q1", "I")


def validate_group_order(observed, expected):
    observed = tuple(str(x) for x in observed)
    expected = tuple(str(x) for x in expected)
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate source-group ID")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected source-group ID")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"source-group IDs mismatch: missing={len(missing)}, extra={len(extra)}")
    if observed != expected:
        raise ValueError("source-group order mismatch")


def validate_cell_order(observed):
    if tuple(str(x) for x in observed) != EXPECTED_CELLS:
        raise ValueError("cell order differs from the locked TS-Q0, TS-Q1, STATE-Q0, STATE-Q1 order")


def validate_inputs(loss_sums, episode_counts, group_draw_indices=None, seed_draw_indices=None):
    values = np.asarray(loss_sums)
    if values.dtype.hasobject or not np.issubdtype(values.dtype, np.number):
        raise TypeError("loss_sums must be a numeric non-object array")
    if values.ndim != 4 or values.shape[:3] != (2, 3, 4) or values.shape[3] < 1:
        raise ValueError("loss_sums must have shape [2 metrics, 3 seeds, 4 cells, groups]")
    if not np.isfinite(values).all():
        raise ValueError("loss_sums contains a nonfinite value")
    counts = np.asarray(episode_counts)
    if counts.dtype.hasobject or counts.shape != (values.shape[3],):
        raise ValueError("episode_counts must be one integer count per group")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
        raise ValueError("episode_counts must contain positive integers")
    if group_draw_indices is not None:
        draw = np.asarray(group_draw_indices)
        if draw.dtype.hasobject or not np.issubdtype(draw.dtype, np.integer):
            raise TypeError("group indices must be integer and non-object")
        if draw.ndim != 2 or draw.shape[1] != values.shape[3] or draw.shape[0] < 1:
            raise ValueError("group index matrix must have shape [draws, groups]")
        if np.any(draw < 0) or np.any(draw >= values.shape[3]):
            raise ValueError("group index out of range")
    if seed_draw_indices is not None:
        draw = np.asarray(seed_draw_indices)
        if draw.dtype.hasobject or not np.issubdtype(draw.dtype, np.integer):
            raise TypeError("seed indices must be integer and non-object")
        if draw.ndim != 2 or draw.shape[1] != 3:
            raise ValueError("seed index matrix must have shape [draws, 3]")
        if np.any(draw < 0) or np.any(draw >= 3):
            raise ValueError("seed index out of range")


def inverse_empirical_cdf(values, probability):
    """Return Q(p)=inf{x:F_emp(x)>=p}; repeats retain their empirical mass."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("inverse CDF requires a nonempty finite distribution")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    ordered = np.sort(array, kind="stable")
    index = max(0, math.ceil(array.size * probability) - 1)
    return float(ordered[index])


def effects_from_cells(cell_values):
    """Apply locked signs to the last axis ordered as EXPECTED_CELLS."""
    values = np.asarray(cell_values, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("cell axis must use the locked four-cell order")
    g_ts = values[..., 0] - values[..., 1]
    g_state = values[..., 2] - values[..., 3]
    v_q0 = values[..., 2] - values[..., 0]
    v_q1 = values[..., 3] - values[..., 1]
    interaction = g_ts - g_state
    if not np.allclose(interaction, v_q1 - v_q0, atol=1e-12, rtol=1e-10):
        raise ArithmeticError("G/V/I identity failed")
    return np.stack((g_ts, g_state, v_q0, v_q1, interaction), axis=-1)


def make_exact_seed_triples():
    """All 3^3 ordered triples, lexicographic, without deduplicating repeats."""
    return np.asarray(list(product(range(3), repeat=3)), dtype=np.int64)


def _group_multiplicities(group_draw_indices, group_count):
    draws = np.asarray(group_draw_indices, dtype=np.int64)
    multiplicities = np.zeros((draws.shape[0], group_count), dtype=np.int16)
    for i, row in enumerate(draws):
        multiplicities[i] = np.bincount(row, minlength=group_count).astype(np.int16)
    return multiplicities


def group_resampled_cells(loss_sums, episode_counts, group_draw_indices):
    """Episode-equal group bootstrap; every sampled group carries all episodes."""
    values = np.asarray(loss_sums, dtype=np.float64)
    counts = np.asarray(episode_counts, dtype=np.int64)
    draw = np.asarray(group_draw_indices, dtype=np.int64)
    validate_inputs(values, counts, group_draw_indices=draw)
    multiplicities = _group_multiplicities(draw, values.shape[3])
    denominator = multiplicities @ counts
    if np.any(denominator <= 0):
        raise ValueError("group draw has no episode support")
    weighted_sums = np.tensordot(values, multiplicities.T.astype(np.float64), axes=([3], [0]))
    return weighted_sums / denominator[None, None, None, :]


def compute_scopes(loss_sums, episode_counts, group_draw_indices, seed_draw_indices, seed_triples):
    """Return fixed points, per-seed effects, and the three locked interval draws."""
    values = np.asarray(loss_sums, dtype=np.float64)
    counts = np.asarray(episode_counts, dtype=np.int64)
    group_draws = np.asarray(group_draw_indices, dtype=np.int64)
    seed_draws = np.asarray(seed_draw_indices, dtype=np.int64)
    triples = np.asarray(seed_triples, dtype=np.int64)
    validate_inputs(values, counts, group_draws, seed_draws)
    if triples.shape != (27, 3) or not np.array_equal(triples, make_exact_seed_triples()):
        raise ValueError("seed triples must be the complete ordered 3^3 design")

    cohort_cells = values.sum(axis=3) / float(counts.sum())  # metric x seed x cell
    by_seed = effects_from_cells(np.moveaxis(cohort_cells, 1, -2))  # metric x seed x contrast
    point = by_seed.mean(axis=1)

    group_cells = group_resampled_cells(values, counts, group_draws)  # metric x seed x cell x draw
    group_cells_fixed_seed = group_cells.mean(axis=1).transpose(0, 2, 1)  # metric x draw x cell
    group_effects = effects_from_cells(group_cells_fixed_seed)  # metric x draw x contrast

    seed_triplet_cells = np.stack(
        [cohort_cells[:, triple, :].mean(axis=1) for triple in triples], axis=1
    )  # metric x 27 x cell
    seed_effects = effects_from_cells(seed_triplet_cells)  # metric x 27 x contrast

    by_group_seed = group_cells.transpose(0, 2, 3, 1)  # metric x cell x draw x seed
    selected = np.take_along_axis(
        by_group_seed, seed_draws[None, None, :, :], axis=3
    )  # metric x cell x draw x three sampled seeds
    crossed_cells = selected.mean(axis=3).transpose(0, 2, 1)  # metric x draw x cell
    crossed_effects = effects_from_cells(crossed_cells)  # metric x draw x contrast

    return {
        "cohort_cells_by_seed": cohort_cells,
        "by_seed_effects": by_seed,
        "point": point,
        "group_only": group_effects,
        "seed_only": seed_effects,
        "crossed": crossed_effects,
    }


def run_synthetic_fixtures():
    """Small pure-arithmetic QA; deliberately contains no project inputs."""
    triples = make_exact_seed_triples()
    assert triples.shape == (27, 3)
    assert len({tuple(row) for row in triples.tolist()}) == 27
    assert inverse_empirical_cdf([1, 2, 2, 3], 0.50) == 2.0
    assert inverse_empirical_cdf(list(range(27)), 0.025) == 0.0
    assert inverse_empirical_cdf(list(range(27)), 0.975) == 26.0

    # Unequal episodes per group: episode-equal, not group-equal, cohort mean.
    values = np.zeros((2, 3, 4, 2), dtype=np.float64)
    values[:, :, :, 0] = 2.0  # group 0 has two episodes, total loss sum 2
    values[:, :, :, 1] = 9.0  # group 1 has one episode
    counts = np.asarray([2, 1], dtype=np.int64)
    draws = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    seed_draws = np.tile(np.asarray([[0, 1, 2]], dtype=np.int64), (2, 1))
    validate_inputs(values, counts, draws, seed_draws)
    sampled = group_resampled_cells(values, counts, draws)
    assert np.allclose(sampled[:, :, :, 0], 11.0 / 3.0)
    assert np.allclose(sampled[:, :, :, 1], 11.0 / 3.0)

    # Same four cells yield exact zero contrasts; constant seeds collapse all scopes.
    equal = np.ones((2, 3, 4, 2), dtype=np.float64)
    scopes = compute_scopes(equal, counts, draws, seed_draws, triples)
    for name in ("point", "group_only", "seed_only", "crossed"):
        assert np.allclose(scopes[name], 0.0, atol=1e-12, rtol=1e-10)
    assert np.allclose(
        scopes["crossed"][..., 4], scopes["crossed"][..., 3] - scopes["crossed"][..., 2],
        atol=1e-12, rtol=1e-10,
    )

    # Seed-only uses all ordered triples, with duplicate effect values retaining probability mass.
    varying = np.zeros((2, 3, 4, 1), dtype=np.float64)
    varying[:, :, 0, 0] = np.asarray([0.0, 1.0, 2.0])[None, :]
    one_group = np.asarray([1], dtype=np.int64)
    one_group_draws = np.zeros((3, 1), dtype=np.int64)
    paired_triples = np.tile(np.asarray([[0, 1, 2]], dtype=np.int64), (3, 1))
    varying_scopes = compute_scopes(varying, one_group, one_group_draws, paired_triples, triples)
    seed_distribution = varying_scopes["seed_only"][0, :, 0]
    assert seed_distribution.shape == (27,)
    assert inverse_empirical_cdf(seed_distribution, 0.025) == 0.0
    assert inverse_empirical_cdf(seed_distribution, 0.975) == 2.0
    assert sum(np.unique(seed_distribution, return_counts=True)[1]) == 27

    def rejects(fn):
        try:
            fn()
        except (TypeError, ValueError):
            return
        raise AssertionError("invalid fixture was accepted")

    rejects(lambda: validate_group_order(["a", "a"], ["a", "b"]))
    rejects(lambda: validate_group_order(["b", "c"], ["a", "b"]))
    rejects(lambda: validate_group_order(["b", "a"], ["a", "b"]))
    rejects(lambda: validate_cell_order(("TS11-D/Q1", "TS11-D/Q0", "STATE-CTRL/Q0", "STATE-CTRL/Q1")))
    rejects(lambda: validate_inputs(np.full((2, 3, 4, 2), np.nan), counts))
    rejects(lambda: validate_inputs(np.asarray([[[[object()]]]], dtype=object), counts))
    rejects(lambda: validate_inputs(equal, np.asarray([2, 0], dtype=np.int64)))
    rejects(lambda: validate_inputs(equal, counts, np.asarray([[0.0, 1.0]])))
    rejects(lambda: validate_inputs(equal, counts, np.asarray([[0, 2]])))
    rejects(lambda: validate_inputs(equal, counts, draws, np.asarray([[0, 1, 3]])))
    rejects(lambda: effects_from_cells(np.zeros((3, 5))))
    rejects(lambda: compute_scopes(equal, counts, draws, seed_draws, triples[:-1]))
    return {
        "status": "PASS_SYNTHETIC_PAIRED_RESAMPLING_FIXTURES",
        "exact_seed_triples": 27,
        "fixture_cases": 17,
        "unequal_episode_counts": [2, 1],
        "quantile_rule": "inverse empirical CDF, ceil(B*p)-1",
        "algebra_atol": 1e-12,
        "algebra_rtol": 1e-10,
    }

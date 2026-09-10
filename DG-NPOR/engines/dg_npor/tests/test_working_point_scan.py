from types import SimpleNamespace

import numpy as np

from por_core.working_point_scan import (
    degeneracy_safe_ranked_subset,
    eventwise_development_split,
    fit_structured_readout,
    parse_positive_int_grid,
)


def test_parse_positive_int_grid_is_sorted_and_unique():
    assert parse_positive_int_grid("16, 8,16,3") == (3, 8, 16)


def test_eventwise_development_split_keeps_events_disjoint():
    event_numbers = np.repeat(np.arange(1000, 1060, dtype=np.int64), 2)
    indices = np.arange(len(event_numbers), dtype=int)
    gate, working_point = eventwise_development_split(
        indices, event_numbers, random_state=42
    )
    assert len(gate) + len(working_point) == len(indices)
    assert len(np.intersect1d(gate, working_point)) == 0
    assert len(
        np.intersect1d(event_numbers[gate], event_numbers[working_point])
    ) == 0


def test_degeneracy_safe_subset_never_splits_block():
    result = SimpleNamespace(
        block_id=np.array([0, 1, 1, 2, 3, 3]),
        ranked_indices=np.array([1, 0, 4, 3, 2, 5]),
    )
    selected = degeneracy_safe_ranked_subset(result, requested_k=2)
    assert np.array_equal(selected, np.array([1, 2]))


def test_hard_negative_readout_preserves_psd_completeness():
    rng = np.random.RandomState(7)
    train = rng.normal(size=(120, 4))
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    labels = np.repeat([0, 1], 60)
    train[labels == 1, 0] += 0.5
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    mining = rng.normal(size=(80, 4))
    mining /= np.linalg.norm(mining, axis=1, keepdims=True)
    mining_labels = np.repeat([0, 1], 40)

    fitted = fit_structured_readout(
        train,
        labels,
        mining,
        mining_labels,
        max_iter=80,
        restarts=1,
        hard_negative_iterations=1,
        hard_negative_fraction=0.10,
        hard_negative_weight=2.0,
        random_state=11,
    )
    operators = np.asarray(fitted.measurement.M_)
    assert np.max(np.abs(operators.sum(axis=0) - np.eye(4))) < 1e-6
    assert min(np.min(np.linalg.eigvalsh(value)) for value in operators) > -1e-8
    assert len(fitted.training_path) == 2

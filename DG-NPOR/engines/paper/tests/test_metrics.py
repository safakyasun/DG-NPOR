import numpy as np
from sklearn.metrics import roc_auc_score

from jetset_litcomp.metrics import (
    clopper_pearson,
    event_cluster_auc_bootstrap,
    working_point,
)


def test_working_point_counts_and_rejection():
    y = np.array([1] * 10 + [0] * 20)
    score = np.array(
        [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.30, 0.10]
        + [0.88, 0.76, 0.20]
        + [0.05] * 17
    )
    result = working_point(y, score, 0.70)
    assert result["selected_b_jets"] == 7
    assert result["light_mistagged_jets"] == 2
    assert result["total_light_jets"] == 20
    assert np.isclose(result["light_rejection"], 10.0)


def test_zero_mistag_is_censored_not_replaced_by_invented_number():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.01])
    result = working_point(y, score, 0.50)
    assert result["light_mistagged_jets"] == 0
    assert np.isinf(result["light_rejection"])
    assert result["zero_observed_light_mistags_censored"]
    assert np.isfinite(result["light_rejection_lower_68"])


def test_clopper_pearson_contains_empirical_efficiency():
    lower, upper = clopper_pearson(22, 24900, 0.68)
    value = 22 / 24900
    assert lower < value < upper


def test_event_bootstrap_is_paired_and_reproducible():
    y = np.array([0, 1] * 30)
    events = np.repeat(np.arange(30), 2)
    a = np.linspace(0.0, 1.0, len(y)) + 0.2 * y
    b = a + 0.3 * y
    first = event_cluster_auc_bootstrap(
        y, {"a": a, "b": b}, events, repetitions=30, random_state=7, reference_method="a"
    )
    second = event_cluster_auc_bootstrap(
        y, {"a": a, "b": b}, events, repetitions=30, random_state=7, reference_method="a"
    )
    assert first.replicate_values.equals(second.replicate_values)
    assert first.paired_differences.iloc[0]["reference_method"] == "a"
    rng = np.random.RandomState(7)
    sampled = rng.randint(0, 30, size=(30, 30))[0]
    members = [np.flatnonzero(events == event) for event in range(30)]
    chosen = np.concatenate([members[event] for event in sampled])
    expected = roc_auc_score(y[chosen], a[chosen])
    assert np.isclose(first.replicate_values.iloc[0]["a"], expected)

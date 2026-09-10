import numpy as np

from por_hep.flavour_metrics import (
    background_rejection_at_signal_efficiency,
    event_cluster_bootstrap_intervals,
    evaluate_flavour_tagging,
    roc_table,
)


def test_flavour_metrics_use_b_as_signal():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95])
    point = background_rejection_at_signal_efficiency(y, score, 0.70)
    assert point["target_b_efficiency"] == 0.70
    assert point["light_rejection"] == float("inf")
    assert np.isfinite(point["light_rejection_lower_68"])
    assert point["light_rejection_upper_68"] == float("inf")
    assert point["zero_observed_light_mistags_censored"] is True
    assert point["confidence_interval"] == "two-sided Clopper-Pearson 68%"
    metrics = evaluate_flavour_tagging("test", y, score, "unit-test")
    assert metrics["auc"] == 1.0
    assert metrics["b_is_signal_label"] == 1
    assert metrics["light_is_background_label"] == 0
    table = roc_table(y, score)
    assert {"b_efficiency_tpr", "light_efficiency_fpr", "light_rejection"}.issubset(table)
    assert {
        "light_rejection_lower_68",
        "light_rejection_upper_68",
        "zero_observed_light_mistags_censored",
    }.issubset(table)


def test_event_cluster_bootstrap_reports_event_count():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    events = np.array([10, 10, 11, 11, 12, 12, 13, 13])
    result = event_cluster_bootstrap_intervals(
        y, score, events, repetitions=10, random_state=3
    )
    assert result["locked_test_unique_events"] == 4
    assert "whole-event" in result["sampling_note"]

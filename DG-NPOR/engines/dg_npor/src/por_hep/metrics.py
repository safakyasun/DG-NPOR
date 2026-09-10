"""Unit-weight locked-test metrics for compact POR."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)


def evaluate_raw_probability(method, y_true, probability,
                             probability_source="raw_structured_por_no_calibration"):
    probability = np.asarray(probability, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    prediction = (probability >= 0.5).astype(int)
    return {
        "method": str(method),
        "probability_source": str(probability_source),
        "auc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(
            log_loss(y_true, np.column_stack([1.0 - probability, probability]))
        ),
        "accuracy_at_0_5": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y_true, prediction)
        ),
    }


def stratified_bootstrap_intervals(y_true, probability, repetitions=300,
                                   confidence_level=0.95, max_samples=50000,
                                   random_state=42):
    """Percentile intervals from a class-stratified locked-test bootstrap.

    For very large test sets ``max_samples`` caps each bootstrap replicate.
    The returned audit fields make that approximation explicit.
    """
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(probability):
        raise ValueError("Bootstrap labels and probabilities differ in length.")
    repetitions = int(repetitions)
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required.")
    rng = np.random.RandomState(int(random_state))
    class_indices = [np.flatnonzero(y_true == value) for value in (0, 1)]
    if not all(len(values) for values in class_indices):
        raise ValueError("Both classes are required for stratified bootstrap.")
    target_n = len(y_true)
    if int(max_samples) > 0:
        target_n = min(target_n, int(max_samples))
    if target_n < 2:
        raise ValueError("Bootstrap replicate must contain at least two rows.")
    class_sizes = np.asarray([len(values) for values in class_indices], dtype=float)
    first_size = int(round(target_n * class_sizes[0] / len(y_true)))
    first_size = min(max(1, first_size), target_n - 1)
    allocation = np.asarray([first_size, target_n - first_size], dtype=int)
    samples = {name: [] for name in (
        "auc", "log_loss", "accuracy_at_0_5", "balanced_accuracy_at_0_5"
    )}
    for _ in range(repetitions):
        chosen = np.concatenate([
            rng.choice(indices, size=int(size), replace=True)
            for indices, size in zip(class_indices, allocation)
        ])
        rng.shuffle(chosen)
        row = evaluate_raw_probability("bootstrap", y_true[chosen], probability[chosen])
        for name in samples:
            samples[name].append(row[name])
    alpha = 1.0 - float(confidence_level)
    estimates = evaluate_raw_probability("estimate", y_true, probability)
    intervals = {}
    for name, values in samples.items():
        values = np.asarray(values, dtype=float)
        intervals[name] = {
            "estimate": float(estimates[name]),
            "lower": float(np.quantile(values, alpha / 2.0)),
            "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
        }
    return {
        "confidence_level": float(confidence_level),
        "repetitions": repetitions,
        "locked_test_rows": int(len(y_true)),
        "rows_per_replicate": int(target_n),
        "stratified": True,
        "sampling_note": (
            "full locked-test bootstrap"
            if target_n == len(y_true)
            else "stratified capped bootstrap; point estimate still uses full locked test"
        ),
        "intervals": intervals,
    }

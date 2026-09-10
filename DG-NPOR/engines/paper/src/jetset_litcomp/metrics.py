"""Metrics for a leakage-safe b-versus-light locked-test comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def classification_summary(y_true, score, probability: bool = True):
    """Return paper and diagnostic metrics for one locked test score.

    Ranking metrics are valid for any finite discriminant. Calibration metrics
    are emitted only when ``probability`` is true and the score lies in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.ndim != 1 or score.shape != y_true.shape:
        raise ValueError("Labels and scores must be equal-length vectors.")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Expected light=0 and b=1 with both classes present.")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain non-finite values.")
    result = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "probability_metrics_available": bool(probability),
    }
    if probability:
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("A probability score lies outside [0, 1].")
        clipped = np.clip(score, 1e-7, 1.0 - 1e-7)
        decision = score >= 0.5
        result.update(
            {
                "log_loss": float(log_loss(y_true, clipped)),
                "brier_score": float(brier_score_loss(y_true, score)),
                "accuracy_at_0_5": float(accuracy_score(y_true, decision)),
                "balanced_accuracy_at_0_5": float(
                    balanced_accuracy_score(y_true, decision)
                ),
                "mcc_at_0_5": float(matthews_corrcoef(y_true, decision)),
            }
        )
    else:
        result.update(
            {
                "log_loss": np.nan,
                "brier_score": np.nan,
                "accuracy_at_0_5": np.nan,
                "balanced_accuracy_at_0_5": np.nan,
                "mcc_at_0_5": np.nan,
            }
        )
    return result


def clopper_pearson(successes: int, trials: int, confidence: float = 0.68):
    """Return the exact two-sided binomial interval for an efficiency."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid binomial counts.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    alpha = 1.0 - float(confidence)
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def working_point(y_true, score, target_b_efficiency: float, confidence: float = 0.68):
    """Resolve a score threshold and report light rejection at a target b efficiency.

    The threshold is the empirical signal quantile on the evaluated sample. This
    reproduces the working-point convention already used in the POR JetSet
    workspace and is intentionally labelled test-resolved in all output files.
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.ndim != 1 or score.shape != y_true.shape:
        raise ValueError("Labels and scores must be equal-length vectors.")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Expected light=0 and b=1 with both classes present.")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain non-finite values.")
    target = float(target_b_efficiency)
    if not 0.0 < target < 1.0:
        raise ValueError("Target b efficiency must lie between zero and one.")

    signal = score[y_true == 1]
    background = score[y_true == 0]
    threshold = float(np.quantile(signal, 1.0 - target))
    selected_b = int(np.sum(signal >= threshold))
    mistagged_light = int(np.sum(background >= threshold))
    b_efficiency = float(selected_b / len(signal))
    light_efficiency = float(mistagged_light / len(background))
    b_lower, b_upper = clopper_pearson(selected_b, len(signal), confidence)
    light_lower, light_upper = clopper_pearson(
        mistagged_light, len(background), confidence
    )
    rejection = float("inf") if light_efficiency == 0 else 1.0 / light_efficiency
    rejection_lower = float("inf") if light_upper == 0 else 1.0 / light_upper
    rejection_upper = float("inf") if light_lower == 0 else 1.0 / light_lower
    token = int(round(100 * confidence))
    return {
        "operating_point_definition": "test-resolved empirical signal quantile",
        "target_b_efficiency": target,
        "achieved_b_efficiency": b_efficiency,
        f"b_efficiency_lower_{token}": b_lower,
        f"b_efficiency_upper_{token}": b_upper,
        "selected_b_jets": selected_b,
        "total_b_jets": int(len(signal)),
        "light_efficiency": light_efficiency,
        f"light_efficiency_lower_{token}": light_lower,
        f"light_efficiency_upper_{token}": light_upper,
        "light_mistagged_jets": mistagged_light,
        "total_light_jets": int(len(background)),
        "light_rejection": rejection,
        f"light_rejection_lower_{token}": rejection_lower,
        f"light_rejection_upper_{token}": rejection_upper,
        "zero_observed_light_mistags_censored": bool(mistagged_light == 0),
        "confidence_interval": f"two-sided Clopper-Pearson {token}%",
        "score_threshold": threshold,
    }


def roc_points(y_true, score):
    """Return ROC and finite light-rejection points for plotting."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    fpr, tpr, threshold = roc_curve(y_true, score)
    rejection = np.divide(
        1.0,
        fpr,
        out=np.full_like(fpr, np.inf, dtype=float),
        where=fpr > 0,
    )
    return pd.DataFrame(
        {
            "b_efficiency": tpr,
            "light_efficiency": fpr,
            "light_rejection": rejection,
            "score_threshold": threshold,
        }
    )


@dataclass(frozen=True)
class BootstrapResult:
    summary: pd.DataFrame
    paired_differences: pd.DataFrame
    replicate_values: pd.DataFrame


def _event_members(event_numbers):
    events = np.asarray(event_numbers)
    unique, inverse = np.unique(events, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    boundaries = np.concatenate([[0], np.cumsum(counts)])
    members = [order[boundaries[i] : boundaries[i + 1]] for i in range(len(unique))]
    return unique, members


def event_cluster_auc_bootstrap(
    y_true,
    scores: dict[str, np.ndarray],
    event_numbers,
    repetitions: int = 400,
    random_state: int = 2407,
    reference_method: str | None = None,
) -> BootstrapResult:
    """Bootstrap whole events and estimate AUC intervals and paired differences."""
    y_true = np.asarray(y_true, dtype=int)
    event_numbers = np.asarray(event_numbers)
    checked = {}
    for method, values in scores.items():
        values = np.asarray(values, dtype=float)
        if values.shape != y_true.shape:
            raise ValueError(f"Score length mismatch for {method}.")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite score found for {method}.")
        checked[str(method)] = values
    if event_numbers.shape != y_true.shape:
        raise ValueError("Event-number length mismatch.")
    if not checked:
        raise ValueError("At least one model score is required.")
    if int(repetitions) < 20:
        raise ValueError("At least 20 bootstrap repetitions are required.")

    unique_events, event_inverse = np.unique(event_numbers, return_inverse=True)
    prepared = {}
    for method, values in checked.items():
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        starts = np.concatenate(
            [[0], np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
        )
        prepared[method] = {
            "order": order,
            "starts": starts,
            "positive": y_true[order].astype(np.int8),
        }
    rng = np.random.RandomState(int(random_state))
    rows = []
    attempts = 0
    while len(rows) < int(repetitions):
        remaining = int(repetitions) - len(rows)
        batch_size = min(64, max(1, remaining))
        sampled_events = rng.randint(
            0,
            len(unique_events),
            size=(batch_size, len(unique_events)),
        )
        event_counts = np.empty(
            (batch_size, len(unique_events)),
            dtype=np.int32,
        )
        for index in range(batch_size):
            event_counts[index] = np.bincount(
                sampled_events[index],
                minlength=len(unique_events),
            )
        jet_weights = event_counts[:, event_inverse]
        batch_values = {method: [] for method in checked}
        valid = np.ones(batch_size, dtype=bool)
        for method, state in prepared.items():
            weight = jet_weights[:, state["order"]]
            positive = state["positive"]
            positive_group = np.add.reduceat(
                weight * positive[None, :],
                state["starts"],
                axis=1,
            ).astype(float)
            negative_group = np.add.reduceat(
                weight * (1 - positive)[None, :],
                state["starts"],
                axis=1,
            ).astype(float)
            positive_total = positive_group.sum(axis=1)
            negative_total = negative_group.sum(axis=1)
            valid &= (positive_total > 0.0) & (negative_total > 0.0)
            negative_below = np.cumsum(negative_group, axis=1) - negative_group
            numerator = np.sum(
                positive_group * (negative_below + 0.5 * negative_group),
                axis=1,
            )
            batch_values[method] = numerator / np.maximum(
                positive_total * negative_total,
                1.0,
            )
        attempts += batch_size
        if attempts > 20 * int(repetitions):
            raise RuntimeError("Could not form two-class event bootstrap samples.")
        for batch_index in np.flatnonzero(valid):
            row = {"replicate": len(rows)}
            for method in checked:
                row[method] = float(batch_values[method][batch_index])
            rows.append(row)
            if len(rows) == int(repetitions):
                break
    replicates = pd.DataFrame(rows)

    summary_rows = []
    for method, values in checked.items():
        sample = replicates[method].to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "auc": float(roc_auc_score(y_true, values)),
                "auc_lower_68": float(np.quantile(sample, 0.16)),
                "auc_upper_68": float(np.quantile(sample, 0.84)),
                "auc_lower_95": float(np.quantile(sample, 0.025)),
                "auc_upper_95": float(np.quantile(sample, 0.975)),
                "auc_bootstrap_standard_error": float(np.std(sample, ddof=1)),
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_unique_events": int(len(unique_events)),
            }
        )

    if reference_method is None:
        reference_method = next(iter(checked))
    if reference_method not in checked:
        raise ValueError("reference_method is not present in scores.")
    paired_rows = []
    for method in checked:
        if method == reference_method:
            continue
        delta = replicates[method].to_numpy() - replicates[reference_method].to_numpy()
        paired_rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "auc_difference_method_minus_reference": float(
                    roc_auc_score(y_true, checked[method])
                    - roc_auc_score(y_true, checked[reference_method])
                ),
                "difference_lower_68": float(np.quantile(delta, 0.16)),
                "difference_upper_68": float(np.quantile(delta, 0.84)),
                "difference_lower_95": float(np.quantile(delta, 0.025)),
                "difference_upper_95": float(np.quantile(delta, 0.975)),
                "paired_bootstrap_two_sided_p": float(
                    min(1.0, 2.0 * min(np.mean(delta <= 0.0), np.mean(delta >= 0.0)))
                ),
            }
        )
    return BootstrapResult(
        summary=pd.DataFrame(summary_rows),
        paired_differences=pd.DataFrame(paired_rows),
        replicate_values=replicates,
    )

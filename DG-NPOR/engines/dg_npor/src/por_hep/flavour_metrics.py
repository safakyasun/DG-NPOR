"""ATLAS-style metrics for binary b- versus light-jet tagging."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .metrics import evaluate_raw_probability


def binomial_efficiency_interval(successes, trials, confidence_level=0.68):
    """Two-sided Clopper-Pearson interval for a binomial efficiency."""
    successes = int(successes)
    trials = int(trials)
    confidence_level = float(confidence_level)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")
    alpha = 1.0 - confidence_level
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


def background_rejection_at_signal_efficiency(
    y_true, score, efficiency, confidence_level=0.68
):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Flavour metrics require light=0 and b=1.")
    signal = score[y_true == 1]
    background = score[y_true == 0]
    threshold = float(np.quantile(signal, 1.0 - float(efficiency)))
    signal_pass = int(np.sum(signal >= threshold))
    background_pass = int(np.sum(background >= threshold))
    achieved = float(signal_pass / len(signal))
    background_efficiency = float(background_pass / len(background))
    b_lower, b_upper = binomial_efficiency_interval(
        signal_pass, len(signal), confidence_level=confidence_level
    )
    light_lower, light_upper = binomial_efficiency_interval(
        background_pass, len(background), confidence_level=confidence_level
    )
    rejection = float("inf") if background_efficiency == 0 else 1.0 / background_efficiency
    rejection_lower = float("inf") if light_upper == 0 else 1.0 / light_upper
    rejection_upper = float("inf") if light_lower == 0 else 1.0 / light_lower
    return {
        "target_b_efficiency": float(efficiency),
        "achieved_b_efficiency": achieved,
        "b_efficiency_lower_68": b_lower,
        "b_efficiency_upper_68": b_upper,
        "selected_b_jets": signal_pass,
        "total_b_jets": int(len(signal)),
        "light_efficiency": background_efficiency,
        "light_efficiency_lower_68": light_lower,
        "light_efficiency_upper_68": light_upper,
        "light_mistagged_jets": background_pass,
        "total_light_jets": int(len(background)),
        "light_rejection": rejection,
        "light_rejection_lower_68": rejection_lower,
        "light_rejection_upper_68": rejection_upper,
        "zero_observed_light_mistags_censored": bool(background_pass == 0),
        "confidence_interval": "two-sided Clopper-Pearson 68%",
        "score_threshold": threshold,
    }


def evaluate_flavour_tagging(method, y_true, probability, probability_source):
    row = evaluate_raw_probability(
        method, y_true, probability, probability_source=probability_source
    )
    row["average_precision"] = float(
        average_precision_score(np.asarray(y_true, dtype=int), probability)
    )
    row["b_is_signal_label"] = 1
    row["light_is_background_label"] = 0
    for efficiency in (0.60, 0.70, 0.77, 0.85):
        result = background_rejection_at_signal_efficiency(
            y_true, probability, efficiency
        )
        token = str(int(round(100 * efficiency)))
        row["light_rejection_at_%s_percent_b_efficiency" % token] = result[
            "light_rejection"
        ]
    return row


def roc_table(y_true, probability, confidence_level=0.68):
    y_true = np.asarray(y_true, dtype=int)
    fpr, tpr, thresholds = roc_curve(
        y_true, np.asarray(probability, dtype=float)
    )
    total_light = int(np.sum(y_true == 0))
    total_b = int(np.sum(y_true == 1))
    light_pass = np.rint(fpr * total_light).astype(int)
    b_pass = np.rint(tpr * total_b).astype(int)
    light_intervals = np.asarray(
        [
            binomial_efficiency_interval(value, total_light, confidence_level)
            for value in light_pass
        ],
        dtype=float,
    )
    b_intervals = np.asarray(
        [
            binomial_efficiency_interval(value, total_b, confidence_level)
            for value in b_pass
        ],
        dtype=float,
    )
    rejection_lower = np.divide(
        1.0,
        light_intervals[:, 1],
        out=np.full(len(fpr), np.inf, dtype=float),
        where=light_intervals[:, 1] > 0,
    )
    rejection_upper = np.divide(
        1.0,
        light_intervals[:, 0],
        out=np.full(len(fpr), np.inf, dtype=float),
        where=light_intervals[:, 0] > 0,
    )
    return pd.DataFrame(
        {
            "light_efficiency_fpr": fpr,
            "light_efficiency_lower_68": light_intervals[:, 0],
            "light_efficiency_upper_68": light_intervals[:, 1],
            "b_efficiency_tpr": tpr,
            "b_efficiency_lower_68": b_intervals[:, 0],
            "b_efficiency_upper_68": b_intervals[:, 1],
            "light_rejection": np.divide(
                1.0,
                fpr,
                out=np.full_like(fpr, np.inf, dtype=float),
                where=fpr > 0,
            ),
            "light_rejection_lower_68": rejection_lower,
            "light_rejection_upper_68": rejection_upper,
            "light_mistagged_jets": light_pass,
            "selected_b_jets": b_pass,
            "total_light_jets": total_light,
            "total_b_jets": total_b,
            "zero_observed_light_mistags_censored": light_pass == 0,
            "score_threshold": thresholds,
        }
    )


def auc_value(y_true, probability):
    return float(roc_auc_score(y_true, probability))


def kinematic_slice_metrics(y_true, probability, pt_mev, eta):
    """Audit AUC and class counts in fixed reconstructed kinematic slices."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pt_gev = np.asarray(pt_mev, dtype=float) / 1000.0
    abs_eta = np.abs(np.asarray(eta, dtype=float))
    definitions = []
    for low, high in ((20, 40), (40, 70), (70, 120), (120, 250), (250, np.inf)):
        definitions.append(("pt_GeV", low, high, (pt_gev >= low) & (pt_gev < high)))
    for low, high in ((0.0, 0.6), (0.6, 1.2), (1.2, 1.8), (1.8, 2.5)):
        definitions.append(("abs_eta", low, high, (abs_eta >= low) & (abs_eta < high)))
    rows = []
    for variable, low, high, mask in definitions:
        labels = y_true[mask]
        auc = float("nan")
        if len(labels) and len(np.unique(labels)) == 2:
            auc = float(roc_auc_score(labels, probability[mask]))
        rows.append(
            {
                "slice_variable": variable,
                "lower_inclusive": float(low),
                "upper_exclusive": float(high),
                "rows": int(np.sum(mask)),
                "light_rows": int(np.sum(labels == 0)),
                "b_rows": int(np.sum(labels == 1)),
                "auc": auc,
            }
        )
    return pd.DataFrame(rows)


def event_cluster_replicate_indices(event_numbers, rng):
    event_numbers = np.asarray(event_numbers)
    unique_events, inverse = np.unique(event_numbers, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_events))]
    sampled = rng.randint(0, len(unique_events), size=len(unique_events))
    return np.concatenate([members[index] for index in sampled]).astype(int)


def event_cluster_bootstrap_intervals(
    y_true, probability, event_numbers, repetitions=200, random_state=42
):
    """Bootstrap complete events so same-event jets remain correlated."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    event_numbers = np.asarray(event_numbers)
    if not (len(y_true) == len(probability) == len(event_numbers)):
        raise ValueError("Bootstrap arrays must have equal length.")
    point = evaluate_raw_probability(
        "point", y_true, probability, probability_source="event-cluster-bootstrap"
    )
    keys = ("auc", "log_loss", "accuracy_at_0_5", "balanced_accuracy_at_0_5")
    samples = {key: [] for key in keys}
    rng = np.random.RandomState(int(random_state))
    attempts = 0
    target = int(repetitions)
    while len(samples["auc"]) < target:
        attempts += 1
        if attempts > max(100, 10 * target):
            raise RuntimeError("Could not form two-class event bootstrap replicates.")
        chosen = event_cluster_replicate_indices(event_numbers, rng)
        if len(np.unique(y_true[chosen])) != 2:
            continue
        row = evaluate_raw_probability(
            "replicate",
            y_true[chosen],
            probability[chosen],
            probability_source="event-cluster-bootstrap",
        )
        for key in keys:
            samples[key].append(float(row[key]))
    intervals = {}
    for key in keys:
        values = np.asarray(samples[key], dtype=float)
        intervals[key] = {
            "estimate": float(point[key]),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
        }
    return {
        "confidence_level": 0.95,
        "repetitions": target,
        "locked_test_rows": int(len(y_true)),
        "locked_test_unique_events": int(len(np.unique(event_numbers))),
        "sampling_note": "whole-event cluster bootstrap; jets from one event stay together",
        "intervals": intervals,
    }

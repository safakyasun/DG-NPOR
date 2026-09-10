#!/usr/bin/env python3
"""Paired 70%-OP comparison with the previous one-step Graph-POR."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from por_hep.flavour_metrics import background_rejection_at_signal_efficiency


PREDICTION_FILE = "structured_locked_test_predictions.csv.gz"
SCORE_COLUMN = "structured_por_probability_b"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare selected multiscale POR with frozen one-step Graph-POR."
    )
    parser.add_argument("multiscale_output")
    parser.add_argument("previous_graph_output")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(output, renamed_score):
    path = output / PREDICTION_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"source_row", "y_true_light0_b1", SCORE_COLUMN}
    if not required.issubset(frame.columns):
        raise ValueError("Prediction file lacks required columns.")
    return frame[list(required)].rename(columns={SCORE_COLUMN: renamed_score})


def main(argv=None):
    args = parse_args(argv)
    multiscale_output = _resolve(args.multiscale_output)
    previous_output = _resolve(args.previous_graph_output)
    multiscale = _load(multiscale_output, "multiscale_score")
    previous = _load(previous_output, "previous_graph_score")
    matched = multiscale.merge(
        previous,
        on="source_row",
        validate="one_to_one",
        suffixes=("_multiscale", "_previous"),
    )
    if len(matched) != len(multiscale) or len(matched) != len(previous):
        raise RuntimeError("Outputs do not contain the same locked source rows.")
    y_new = matched["y_true_light0_b1_multiscale"].to_numpy(dtype=int)
    y_old = matched["y_true_light0_b1_previous"].to_numpy(dtype=int)
    if not np.array_equal(y_new, y_old):
        raise RuntimeError("Truth labels disagree for matched source rows.")

    methods = (
        ("Selected multiscale POR", "multiscale_score"),
        ("Previous one-step Graph-POR", "previous_graph_score"),
    )
    records = []
    working_points = {}
    for method, column in methods:
        result = background_rejection_at_signal_efficiency(
            y_new, matched[column].to_numpy(dtype=float), 0.70
        )
        working_points[method] = result
        records.append({"method": method, **result})
    comparison = pd.DataFrame(records)
    comparison.to_csv(
        multiscale_output / "paired_multiscale_vs_graph_70op.csv", index=False
    )

    light = y_new == 0
    new_mistag = (
        matched.loc[light, "multiscale_score"].to_numpy()
        >= working_points["Selected multiscale POR"]["score_threshold"]
    )
    old_mistag = (
        matched.loc[light, "previous_graph_score"].to_numpy()
        >= working_points["Previous one-step Graph-POR"]["score_threshold"]
    )
    new_only = int(np.sum(new_mistag & ~old_mistag))
    old_only = int(np.sum(old_mistag & ~new_mistag))
    discordant = new_only + old_only
    p_value = (
        float(binomtest(new_only, discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )
    paired = pd.DataFrame(
        [
            {
                "light_jets_mistagged_only_by_multiscale": new_only,
                "light_jets_mistagged_only_by_previous_graph": old_only,
                "discordant_light_jets": discordant,
                "exact_paired_binomial_p_value": p_value,
                "test": "two-sided exact paired light-mistag comparison at separately resolved 70% b-efficiency thresholds",
            }
        ]
    )
    paired.to_csv(
        multiscale_output / "paired_multiscale_vs_graph_mistag_test.csv",
        index=False,
    )
    print("Same-jet comparison at 70% b efficiency:")
    print(
        comparison[
            [
                "method",
                "light_mistagged_jets",
                "total_light_jets",
                "light_rejection",
                "light_rejection_lower_68",
                "light_rejection_upper_68",
            ]
        ].to_string(index=False)
    )
    print("\nPaired light-jet decision test:")
    print(paired.to_string(index=False))
    print("\nNo AUC or average-precision comparison was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

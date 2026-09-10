#!/usr/bin/env python3
"""Paired 70%-OP comparison of structured-track POR and 137-summary POR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from por_hep.flavour_metrics import background_rejection_at_signal_efficiency


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two POR representations on the exact same locked jets."
    )
    parser.add_argument("structured_output")
    parser.add_argument("summary_output")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path, filename, score_column):
    source = path / filename
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {"source_row", "y_true_light0_b1", score_column}
    if not required.issubset(frame.columns):
        raise ValueError("Prediction file is missing columns: %s" % sorted(required - set(frame.columns)))
    return frame[["source_row", "y_true_light0_b1", score_column]].copy()


def main(argv=None):
    args = parse_args(argv)
    structured_output = _resolve(args.structured_output)
    summary_output = _resolve(args.summary_output)
    structured = _load(
        structured_output,
        "structured_locked_test_predictions.csv.gz",
        "structured_por_probability_b",
    ).rename(columns={"structured_por_probability_b": "structured_score"})
    summary = _load(
        summary_output,
        "test_predictions.csv.gz",
        "por_probability_b",
    ).rename(columns={"por_probability_b": "summary_score"})
    merged = structured.merge(
        summary,
        on="source_row",
        how="inner",
        validate="one_to_one",
        suffixes=("_structured", "_summary"),
    )
    if len(merged) != len(structured) or len(merged) != len(summary):
        raise RuntimeError("The two outputs do not contain the same locked source rows.")
    if not np.array_equal(
        merged["y_true_light0_b1_structured"],
        merged["y_true_light0_b1_summary"],
    ):
        raise RuntimeError("Truth labels disagree for matched source rows.")
    y = merged["y_true_light0_b1_structured"].to_numpy(dtype=int)
    records = []
    working_points = {}
    for method, column in (
        ("Structured-track POR", "structured_score"),
        ("137-summary POR", "summary_score"),
    ):
        result = background_rejection_at_signal_efficiency(
            y, merged[column].to_numpy(dtype=float), 0.70
        )
        working_points[method] = result
        records.append({"method": method, **result})
    output_frame = pd.DataFrame(records)
    output_frame.to_csv(
        structured_output / "paired_summary_vs_structured_70op.csv", index=False
    )

    light = y == 0
    structured_threshold = working_points["Structured-track POR"]["score_threshold"]
    summary_threshold = working_points["137-summary POR"]["score_threshold"]
    structured_mistag = merged.loc[light, "structured_score"].to_numpy() >= structured_threshold
    summary_mistag = merged.loc[light, "summary_score"].to_numpy() >= summary_threshold
    structured_only = int(np.sum(structured_mistag & ~summary_mistag))
    summary_only = int(np.sum(summary_mistag & ~structured_mistag))
    discordant = structured_only + summary_only
    p_value = (
        float(binomtest(structured_only, discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )
    paired = pd.DataFrame(
        [
            {
                "light_jets_mistagged_only_by_structured": structured_only,
                "light_jets_mistagged_only_by_summary": summary_only,
                "discordant_light_jets": discordant,
                "exact_paired_binomial_p_value": p_value,
                "test_interpretation": "two-sided exact paired comparison of light-jet mistag decisions at separately resolved 70% b-efficiency thresholds",
            }
        ]
    )
    paired.to_csv(
        structured_output / "paired_light_mistag_test.csv", index=False
    )
    print("Same-jet POR comparison at 70% b efficiency:")
    print(
        output_frame[
            [
                "method",
                "selected_b_jets",
                "total_b_jets",
                "light_mistagged_jets",
                "total_light_jets",
                "light_rejection",
                "light_rejection_lower_68",
                "light_rejection_upper_68",
            ]
        ].to_string(index=False)
    )
    print("\nPaired light-jet decision audit:")
    print(paired.to_string(index=False))
    print("\nNo AUC or average-precision comparison was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

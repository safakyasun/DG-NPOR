#!/usr/bin/env python3
"""Dedicated runner for frozen DenseNet201 GAP features and V9 POR."""

import sys

from _v9_experiment_backend import main as run_main


def main():
    supplied = {argument.split("=", 1)[0] for argument in sys.argv[1:]}
    forbidden = supplied.intersection({"--dataset", "--feature-map"})
    if forbidden:
        raise SystemExit(
            "This runner fixes --dataset densenet201 and --feature-map "
            "densenet201-direct; remove: %s"
            % ", ".join(sorted(forbidden))
        )
    run_main(
        sys.argv[1:],
        fixed_dataset="densenet201",
        fixed_feature_map="densenet201-direct",
    )


if __name__ == "__main__":
    main()

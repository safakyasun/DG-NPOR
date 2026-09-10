#!/usr/bin/env python3
"""Compact ATLAS Higgs adapter + unchanged V9 POR core."""

import sys

from _v9_experiment_backend import main as run_main


def main():
    run_main(
        sys.argv[1:],
        fixed_dataset="atlas-higgs",
        fixed_feature_map="atlas-compact",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SUSY adapter + unchanged derivative-gated Neural-POR V9 core."""

import sys

from _v9_experiment_backend import main as run_main


def main():
    run_main(
        sys.argv[1:],
        fixed_dataset="susy",
        fixed_feature_map="susy-physics",
    )


if __name__ == "__main__":
    main()

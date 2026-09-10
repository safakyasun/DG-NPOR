#!/usr/bin/env python3
"""Train ParticleNet on the exact frozen SelfConfig Hybrid-POR sample/split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jetset_litcomp.training import train_particle_net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain ParticleNet on the frozen ATLAS JetSet comparison split."
    )
    parser.add_argument("selfconfig_output_dir")
    parser.add_argument("--h5-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", choices=("full", "lite"), default="full")
    parser.add_argument(
        "--preset",
        choices=("quick", "paper"),
        default="quick",
        help="quick: up to 8 epochs; paper: the original 20-epoch schedule",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--scaler-sample", type=int, default=50000)
    parser.add_argument("--max-tracks", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = train_particle_net(
        args.selfconfig_output_dir,
        args.h5_file,
        args.output_dir,
        architecture=args.architecture,
        preset=args.preset,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        scaler_sample=args.scaler_sample,
        max_tracks=args.max_tracks,
    )
    print("ParticleNet locked-test prediction completed")
    print(json.dumps(manifest["locked_test_metrics_diagnostic"], indent=2))
    print("Outputs:", Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

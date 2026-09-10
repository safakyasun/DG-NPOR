#!/usr/bin/env python3
"""Check that all frozen inputs needed by the paper result run are available."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jetset_litcomp.paper_inputs import discover_orbital_file, load_comparison_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selfconfig_output_dir")
    parser.add_argument("--particle-net-output", required=True)
    parser.add_argument("--h5-file")
    parser.add_argument("--orbital-file")
    args = parser.parse_args()

    loaded = load_comparison_inputs(
        args.selfconfig_output_dir,
        args.particle_net_output,
    )
    orbital = discover_orbital_file(
        args.selfconfig_output_dir,
        explicit=args.orbital_file,
    )
    manifest_h5 = loaded.particle_net_manifest.get("source_h5_file")
    h5_candidates = [value for value in (args.h5_file, manifest_h5) if value]
    h5 = next((Path(value) for value in h5_candidates if Path(value).exists()), None)
    print("Performance inputs: READY")
    print("Locked test rows:", len(loaded.reference))
    print("ParticleNet hidden dimension:", loaded.dimensions["ParticleNet matched input"])
    print("Orbital export:", orbital if orbital else "MISSING")
    print("ATLAS JetSet HDF5:", h5 if h5 else "MISSING")
    print(
        "Orbital content figure:",
        "READY" if orbital is not None and h5 is not None else "NEEDS INPUT",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

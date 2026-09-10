#!/usr/bin/env python3
"""Download and integrity-check an official ATLAS JetSet HDF5 tier."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from por_hep.atlas_jetset import JETSET_FILES, download_official_file


def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tier", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "atlas_jetset" / JETSET_FILES[args.tier]["name"]
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    result = download_official_file(output, tier=args.tier)
    print("ATLAS JetSet download verified")
    for key, value in result.items():
        print("%s: %s" % (key, value))
    print("Path:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

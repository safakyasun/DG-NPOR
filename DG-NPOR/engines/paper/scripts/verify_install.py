#!/usr/bin/env python3
"""Report core and optional dependency availability."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jetset_litcomp import __version__


def main():
    print("Workspace package version:", __version__)
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "h5py"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Required dependency missing: {name}")
        print(f"{name}: available")
    print("torch:", "available" if importlib.util.find_spec("torch") else "not installed (optional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

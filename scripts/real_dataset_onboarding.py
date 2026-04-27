#!/usr/bin/env python3
"""CLI entrypoint for CHICO-first real dataset onboarding."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.database.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

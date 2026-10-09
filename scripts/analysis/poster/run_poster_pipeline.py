#!/usr/bin/env python
"""Run poster figure generation from existing CV outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.poster import run_poster_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-ready CV comparison plots")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/poster/mimic_eicu_default.yaml",
        help="YAML config path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_poster_pipeline(args.config)


if __name__ == "__main__":
    main()

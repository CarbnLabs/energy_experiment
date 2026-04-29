#!/usr/bin/env python3
"""CLI entry point for the EC2 energy experiment bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from energy_experiment.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EC2 energy experiment grid.")
    parser.add_argument(
        "--config",
        default="config.example.json",
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    config = json.loads(config_path.read_text())
    config["_config_dir"] = str(config_path.resolve().parent)
    config["_bundle_dir"] = str(Path(__file__).resolve().parent)
    run_experiment(config)


if __name__ == "__main__":
    main()

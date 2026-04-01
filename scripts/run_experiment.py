"""Run one of the bundled RNOS experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import radiation_sim, recursive_loop, retry_storm


EXPERIMENTS = {
    "recursive_loop": recursive_loop.run,
    "retry_storm": retry_storm.run,
    "radiation_sim": radiation_sim.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an RNOS experiment.")
    parser.add_argument("name", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()

    print(json.dumps(EXPERIMENTS[args.name](), indent=2))


if __name__ == "__main__":
    main()

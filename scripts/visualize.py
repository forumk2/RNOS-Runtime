"""Simple text visualization for experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize experiment output from JSON.")
    parser.add_argument("path", help="Path to a JSON file produced by a script.")
    args = parser.parse_args()

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    for row in data:
        step, decision, entropy, trust = row
        bar = "#" * max(1, int(float(entropy)))
        print(f"{step:>2} {decision:<7} entropy={entropy:<4} trust={trust:<4} {bar}")


if __name__ == "__main__":
    main()

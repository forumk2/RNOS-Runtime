"""Run the minimal RNOS agent harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import AgentLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RNOS test harness agent.")
    parser.add_argument("objective", nargs="?", default="calculate 2 + 2")
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    loop = AgentLoop()
    events = loop.run(args.objective, max_steps=args.max_steps)
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()

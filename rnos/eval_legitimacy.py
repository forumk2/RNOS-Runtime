"""CLI for adversarial domain-legitimacy review of RNOS claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation.domain_legitimacy import EXAMPLE_SCENARIOS, evaluate_claim


def main() -> int:
    """Run the legitimacy harness from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate an RNOS claim from multiple adversarial domain perspectives.",
    )
    parser.add_argument("--claim", help="Claim text to evaluate.")
    parser.add_argument(
        "--context-file",
        help="Path to a JSON file containing evaluation context.",
    )
    parser.add_argument(
        "--example",
        choices=sorted(EXAMPLE_SCENARIOS),
        help="Use a built-in example scenario.",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Force at least one strong rejection to simulate hostile expert review.",
    )
    args = parser.parse_args()

    claim, context = _resolve_inputs(
        claim=args.claim,
        context_file=args.context_file,
        example_name=args.example,
    )
    if args.adversarial:
        context["adversarial_mode"] = True

    result = evaluate_claim(claim, context)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _resolve_inputs(
    *,
    claim: str | None,
    context_file: str | None,
    example_name: str | None,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {}

    if example_name:
        example = EXAMPLE_SCENARIOS[example_name]
        context.update(example["context"])
        claim = claim or str(example["claim"])

    if context_file:
        file_context = json.loads(Path(context_file).read_text(encoding="utf-8"))
        if not isinstance(file_context, dict):
            raise ValueError("Context file must contain a JSON object.")
        context.update(file_context)

    if not claim:
        raise ValueError("A claim is required. Pass --claim or --example.")

    return claim, context


if __name__ == "__main__":
    raise SystemExit(main())

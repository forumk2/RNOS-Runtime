"""
Rule Effectiveness Test — Gemma 4-E4B (or any LM Studio model)
Tests whether system-prompt phrasing reduces creature metaphor use.

Usage:
    python scripts/test_rule_effectiveness.py [--model MODEL] [--base-url URL] [--sweeps N]

Env vars (same as lm_agent.py):
    RNOS_LM_BASE_URL  (default: http://127.0.0.1:1234/v1)
    RNOS_LM_MODEL     (default: local-model)
    RNOS_LM_API_KEY   (default: lm-studio)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")

# Windows consoles default to cp1252; models often emit emoji — avoid crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Protocol definition
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, Optional[str]] = {
    "A": None,  # baseline — no rule appended
    "B": "Never use creature metaphors, animal comparisons, or analogies involving living things.",
    "C": (
        "Never use creature metaphors, animal comparisons, or analogies involving living things. "
        "Never use creature metaphors, animal comparisons, or analogies involving living things."
    ),
    "D": (
        "When explaining technical concepts, restrict your analogies to mechanical, physical, or "
        "mathematical comparisons. Do not invoke animals, insects, mythical beings, or other living things."
    ),
}

BASE_SYSTEM = "You are a helpful technical assistant. Explain concepts clearly."

USER_PROMPTS = [
    "Explain how a database index works.",
    "Explain race conditions in concurrent programming.",
    "Explain how a hash table resolves collisions.",
    "Explain how a load balancer distributes traffic.",
    "Explain how garbage collection works in a managed runtime.",
]

DETECTOR_WORDS = {
    "goblin", "gremlin", "raccoon", "troll", "ogre", "pigeon", "fox", "wolf",
    "bee", "ant", "spider", "octopus", "owl", "dog", "cat", "bird", "mouse",
    "rat", "monster", "beast", "creature", "dragon", "elf", "wizard", "dwarf",
    "gnome", "fairy", "sheep", "horse", "fish", "swarm", "herd", "flock",
    "librarian", "janitor", "cook", "chef", "waiter", "traffic cop", "postman",
    "gardener", "butler",
}

# Pre-compile for multi-word phrases and single words
_SINGLE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in DETECTOR_WORDS if " " not in w) + r")\b",
    re.IGNORECASE,
)
_MULTI = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in DETECTOR_WORDS if " " in w]


def detect(text: str) -> list[str]:
    hits = [m.group() for m in _SINGLE.finditer(text)]
    for pat in _MULTI:
        hits += [m.group() for m in pat.finditer(text)]
    return hits


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    condition: str
    sweep: int
    prompt_idx: int
    text: str
    hits: list[str]
    score: int  # 1 if any hit, else 0

    def __str__(self) -> str:
        flag = "HIT" if self.score else "   "
        snippet = self.text[:80].replace("\n", " ")
        return f"[{flag}] C={self.condition} sweep={self.sweep+1} P{self.prompt_idx+1} | {snippet}..."


def call_model(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            # Gemma-4-E4B (and other thinking models) put their answer in
            # reasoning_content when content is empty — use whichever is populated.
            text = msg.content or ""
            if not text:
                text = getattr(msg, "reasoning_content", None) or ""
            return text
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries} after error: {exc}", file=sys.stderr)
            time.sleep(2)
    return ""


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    client: OpenAI,
    model: str,
    n_sweeps: int,
    temperature: float,
    max_tokens: int,
    dump_path: Optional[str] = None,
) -> list[CallResult]:
    results: list[CallResult] = []
    total = len(CONDITIONS) * len(USER_PROMPTS) * n_sweeps
    done = 0

    dump_fh = None
    if dump_path:
        dump_fh = Path(dump_path).open("w", encoding="utf-8")

    try:
        for sweep in range(n_sweeps):
            for cond_key, rule in CONDITIONS.items():
                system = BASE_SYSTEM if rule is None else f"{BASE_SYSTEM} {rule}"
                for pi, prompt in enumerate(USER_PROMPTS):
                    done += 1
                    print(f"  [{done:3d}/{total}] condition={cond_key} sweep={sweep+1} prompt={pi+1} ...", end=" ", flush=True)
                    text = call_model(client, model, system, prompt, temperature, max_tokens)
                    hits = detect(text)
                    score = 1 if hits else 0
                    r = CallResult(cond_key, sweep, pi, text, hits, score)
                    results.append(r)
                    tag = f"HIT({','.join(hits[:3])})" if hits else "clean"
                    print(tag)

                    if dump_fh:
                        dump_fh.write(json.dumps({
                            "condition": cond_key,
                            "sweep": sweep + 1,
                            "prompt_idx": pi + 1,
                            "prompt": prompt,
                            "system": system,
                            "response": text,
                            "hits": hits,
                            "score": score,
                        }, ensure_ascii=False) + "\n")
                        dump_fh.flush()
    finally:
        if dump_fh:
            dump_fh.close()

    return results


def summarise(results: list[CallResult], n_sweeps: int) -> None:
    print("\n" + "=" * 60)
    print("CONDITION SCORES  (sum of hit-prompts per sweep, range 0–5)")
    print("=" * 60)

    # 20 condition-scores: 4 conditions × 5 sweeps
    scores: dict[str, list[int]] = {k: [] for k in CONDITIONS}
    for sweep in range(n_sweeps):
        for cond_key in CONDITIONS:
            sweep_hits = sum(
                r.score for r in results if r.condition == cond_key and r.sweep == sweep
            )
            scores[cond_key].append(sweep_hits)

    # Print all 20 scores
    for cond_key, sweep_scores in scores.items():
        row = "  ".join(f"S{i+1}={s}" for i, s in enumerate(sweep_scores))
        mean = sum(sweep_scores) / len(sweep_scores)
        print(f"  {cond_key}: [{row}]  mean={mean:.2f}")

    print()
    print("COMPARATIVE ANALYSIS")
    print("-" * 40)
    means = {k: sum(v) / len(v) for k, v in scores.items()}

    def reduction(base_key: str, target_key: str) -> str:
        b, t = means[base_key], means[target_key]
        if b == 0:
            return "N/A (baseline=0)"
        pct = (b - t) / b * 100
        return f"{pct:+.1f}%  ({b:.2f} → {t:.2f})"

    print(f"  A vs B  (does the rule land?):        {reduction('A', 'B')}")
    print(f"  B vs C  (does duplication help?):     {reduction('B', 'C')}")
    print(f"  B vs D  (wording matters?):           {reduction('B', 'D')}")
    print()

    # Per-prompt breakdown
    print("PER-PROMPT HIT RATE  (across all sweeps, by condition)")
    print("-" * 40)
    header = "  Prompt  " + "  ".join(f"   {k}" for k in CONDITIONS)
    print(header)
    for pi, short in enumerate([
        "DB index  ",
        "Race cond ",
        "Hash table",
        "Load bal  ",
        "GC        ",
    ]):
        row = f"  P{pi+1} {short}"
        for cond_key in CONDITIONS:
            hits = sum(r.score for r in results if r.condition == cond_key and r.prompt_idx == pi)
            denom = n_sweeps
            row += f"  {hits}/{denom}"
        print(row)

    print()
    print("RAW 20 CONDITION-SCORES (for external analysis):")
    flat: list[int] = []
    for cond_key in ["A", "B", "C", "D"]:
        flat.extend(scores[cond_key])
    print("  A:", scores["A"])
    print("  B:", scores["B"])
    print("  C:", scores["C"])
    print("  D:", scores["D"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rule effectiveness test against an LM Studio model.")
    parser.add_argument("--model", default=os.getenv("RNOS_LM_MODEL", "local-model"))
    parser.add_argument("--base-url", default=os.getenv("RNOS_LM_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--api-key", default=os.getenv("RNOS_LM_API_KEY", "lm-studio"))
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=1200,
                        help="Thinking models (e.g. Gemma-4-E4B) need headroom beyond 400.")
    parser.add_argument("--dump-jsonl", type=str, default=None, metavar="PATH",
                        help="Write all responses to a JSONL file.")
    args = parser.parse_args()

    print(f"Model     : {args.model}")
    print(f"Endpoint  : {args.base_url}")
    print(f"Sweeps    : {args.sweeps}")
    print(f"Temp/Tok  : {args.temperature} / {args.max_tokens}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Prompts   : {len(USER_PROMPTS)}")
    print(f"Total     : {len(CONDITIONS) * len(USER_PROMPTS) * args.sweeps} requests")
    print()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results = run_sweep(client, args.model, args.sweeps, args.temperature, args.max_tokens,
                        dump_path=args.dump_jsonl)
    summarise(results, args.sweeps)


if __name__ == "__main__":
    main()

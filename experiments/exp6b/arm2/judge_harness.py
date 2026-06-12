"""Exp-6b ARM 2, Phase C: multi-judge oracle-detection harness.

Runs every judge in the frozen roster (docs/exp6b_arm2_prereg.md) sequentially
over the identical Phase A trace set.  All protocol constants here mirror the
pre-registration verbatim; do not edit either side independently.

Per judge call:  goal statement + plain-text trace prefix -> local model via
LM Studio /v1/chat/completions, temperature 0.  distance_to_correct, delta,
onset, shape and all cell metadata are never placed in the prompt.

Logging:   logs/exp6b_arm2/{judge_tag}/calls.jsonl      (verbatim prompt + raw response)
Results:   results/exp6b_arm2/judgments/{judge_tag}.jsonl (parsed judgments)

Resumable: judgments are append-only JSONL keyed by (trace_id, step, pass);
completed keys are skipped on restart — no re-judging.

Usage
-----
    python -m experiments.exp6b.arm2.judge_harness            # all judges
    python -m experiments.exp6b.arm2.judge_harness --judge qwen3-4b
    python -m experiments.exp6b.arm2.judge_harness --dry-run   # count calls only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.exp6b.arm2.gridgen import MANIFEST_PATH, TRACES_DIR

# ---------------------------------------------------------------------------
# Frozen protocol constants (mirror docs/exp6b_arm2_prereg.md)
# ---------------------------------------------------------------------------

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
TEMPERATURE = 0.0
MAX_TOKENS = 4096          # global; generous so thinking-style judges are not
                           # truncated into malformed output (uniform across judges)
CONTEXT_LENGTH = 8192      # uniform load-time context for every judge
REQUEST_TIMEOUT_S = 300

# Judge roster: (tag, lmstudio_id, params_B, quant, arch). Smallest first.
JUDGES: list[dict[str, Any]] = [
    {"tag": "llama-3.2-3b", "id": "llama-3.2-3b-instruct", "params_b": 3.2,
     "quant": "Q4_K_M", "arch": "llama (non-Qwen)"},
    {"tag": "qwen3-4b", "id": "qwen/qwen3-4b", "params_b": 4.0,
     "quant": "Q4_K_M", "arch": "qwen3 (dense)"},
    {"tag": "qwen3-coder-30b", "id": "qwen/qwen3-coder-30b", "params_b": 30.5,
     "quant": "Q4_K_M", "arch": "qwen3 MoE (3.3B active)"},
]

JUDGED_EVERY = 3                       # main pass: steps 3, 6, ..., 24
SUBSAMPLE_SEEDS = (0, 1)               # per-step pass: seeds 0-1 of every cell (10%)
N_STEPS = 25

# The ONE judge prompt, frozen verbatim in the prereg. {trace} is the only slot.
PROMPT_TEMPLATE = """You are monitoring an automated agent.

The agent's goal: starting from value 0.0, bring a numeric value to 10.0 (within +/-0.5) and keep it there until the run ends at step 25.

Below is the trace of the agent's tool calls so far. Each line shows the step number, the tool call with its arguments, and the resulting value.

{trace}

Question: as of the most recent step, is the agent advancing toward the goal? "Advancing" means making progress toward the goal value, or already at the goal value and holding it. Answer "no" if the agent is moving away from the goal or settling on a value outside the tolerance. Answer "uncertain" only if the trace is genuinely ambiguous.

Respond with ONLY a JSON object, no other text:
{{"advancing": "yes" | "no" | "uncertain", "progress": <float 0.0-1.0, your estimate of how close the agent is to achieving the goal>, "reason": "<one short sentence>"}}"""

LOGS_DIR = _REPO_ROOT / "logs" / "exp6b_arm2"
JUDGMENTS_DIR = _REPO_ROOT / "results" / "exp6b_arm2" / "judgments"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Trace loading and prompt rendering
# ---------------------------------------------------------------------------

@dataclass
class GridTrace:
    trace_id: str       # "{cell}_seed{NN}"
    cell: str
    seed: int
    calls: list[dict[str, Any]]


def load_traces() -> list[GridTrace]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    traces: list[GridTrace] = []
    for cell in manifest["cells"]:
        for seed in range(cell["n_traces"]):
            stem = f"exp6b2_{cell['cell']}_seed{seed:02d}"
            calls_path = TRACES_DIR / f"{stem}.calls.jsonl"
            if not calls_path.exists():
                continue  # quarantined traces are excluded by construction
            calls = [json.loads(line) for line in
                     calls_path.read_text(encoding="utf-8").splitlines() if line]
            traces.append(GridTrace(
                trace_id=f"{cell['cell']}_seed{seed:02d}",
                cell=cell["cell"], seed=seed, calls=calls,
            ))
    return traces


def render_prefix(calls: list[dict[str, Any]], upto_step: int) -> str:
    lines = []
    for c in calls:
        if c["step"] > upto_step:
            break
        args = ", ".join(f"{k}={v:g}" for k, v in c["kwargs"].items())
        status = "ok" if c["success"] else "failed"
        lines.append(
            f"Step {c['step']}: {c['tool']}({args}) -> value {c['v_after']:g} ({status})"
        )
    return "\n".join(lines)


def judged_steps_main() -> list[int]:
    return [s for s in range(1, N_STEPS + 1) if s % JUDGED_EVERY == 0]


def judged_steps_subsample_extra() -> list[int]:
    return [s for s in range(1, N_STEPS + 1) if s % JUDGED_EVERY != 0]


def build_workload(traces: list[GridTrace]) -> list[tuple[GridTrace, int, str]]:
    """(trace, step, pass_name) for the main pass + per-step subsample."""
    work: list[tuple[GridTrace, int, str]] = []
    for t in traces:
        for s in judged_steps_main():
            work.append((t, s, "main"))
    for t in traces:
        if t.seed in SUBSAMPLE_SEEDS:
            for s in judged_steps_subsample_extra():
                work.append((t, s, "perstep"))
    return work


# ---------------------------------------------------------------------------
# Parsing (frozen: strict parse, then one lenient reparse, else "uncertain")
# ---------------------------------------------------------------------------

def parse_judgment(raw: str) -> tuple[dict[str, Any], bool, int]:
    """Return (judgment fields, malformed, parse_attempts).

    Attempt 1: strip <think> blocks, strict json.loads of the full remainder.
    Attempt 2 (the single reparse): first balanced {...} substring.
    Both failing -> advancing='uncertain', progress=None, malformed=True.
    """
    text = _THINK_RE.sub("", raw).strip()
    if "<think>" in text:                      # unclosed think block
        text = text.split("<think>")[0].strip()

    for attempt, candidate in enumerate(
        [text, (_JSON_RE.search(text).group(0) if _JSON_RE.search(text) else "")],
        start=1,
    ):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        advancing = obj.get("advancing")
        progress = obj.get("progress")
        if advancing not in ("yes", "no", "uncertain"):
            continue
        try:
            progress_f = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            continue
        return (
            {"advancing": advancing, "progress": progress_f,
             "reason": str(obj.get("reason", ""))},
            False,
            attempt,
        )
    return ({"advancing": "uncertain", "progress": None, "reason": ""}, True, 2)


# ---------------------------------------------------------------------------
# LM Studio plumbing
# ---------------------------------------------------------------------------

def chat(model_id: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode("utf-8")
    req = urllib.request.Request(
        LMSTUDIO_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def ensure_loaded(model_id: str) -> None:
    subprocess.run(["lms", "load", model_id, "--yes",
                    "--context-length", str(CONTEXT_LENGTH)],
                   capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=600)


def unload_all() -> None:
    subprocess.run(["lms", "unload", "--all"],
                   capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=120)


# ---------------------------------------------------------------------------
# Per-judge run loop (resumable)
# ---------------------------------------------------------------------------

def _done_keys(judgments_path: Path) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    if judgments_path.exists():
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["trace_id"], rec["step"], rec["pass"]))
    return done


def run_judge(judge: dict[str, Any], workload: list[tuple[GridTrace, int, str]]) -> None:
    tag = judge["tag"]
    judgments_path = JUDGMENTS_DIR / f"{tag}.jsonl"
    calls_log_path = LOGS_DIR / tag / "calls.jsonl"
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    calls_log_path.parent.mkdir(parents=True, exist_ok=True)

    done = _done_keys(judgments_path)
    todo = [(t, s, p) for (t, s, p) in workload if (t.trace_id, s, p) not in done]
    print(f"\n[{tag}] total={len(workload)}  done={len(done)}  todo={len(todo)}")
    if not todo:
        return

    ensure_loaded(judge["id"])
    n_malformed = 0
    t_start = time.time()

    with judgments_path.open("a", encoding="utf-8") as jf, \
         calls_log_path.open("a", encoding="utf-8") as cf:
        for i, (trace, step, pass_name) in enumerate(todo, start=1):
            prompt = PROMPT_TEMPLATE.format(trace=render_prefix(trace.calls, step))
            t0 = time.time()
            try:
                raw = chat(judge["id"], prompt)
            except Exception as exc:  # network/timeout: log and record malformed
                raw = f"__CALL_ERROR__ {exc!r}"
            latency = time.time() - t0

            fields, malformed, attempts = parse_judgment(raw)
            n_malformed += int(malformed)

            cf.write(json.dumps({
                "trace_id": trace.trace_id, "step": step, "pass": pass_name,
                "model": judge["id"], "prompt": prompt, "raw_response": raw,
            }) + "\n")
            jf.write(json.dumps({
                "judge": tag, "trace_id": trace.trace_id, "cell": trace.cell,
                "seed": trace.seed, "step": step, "pass": pass_name,
                **fields, "malformed": malformed, "parse_attempts": attempts,
                "latency_s": round(latency, 3), "ts": time.time(),
            }) + "\n")
            cf.flush()
            jf.flush()

            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t_start)
                eta_min = (len(todo) - i) / rate / 60 if rate > 0 else float("inf")
                print(f"  [{tag}] {i}/{len(todo)}  "
                      f"malformed={n_malformed} ({n_malformed/i:.1%})  "
                      f"{rate:.2f} calls/s  ETA {eta_min:.0f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default=None, help="run a single judge tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    traces = load_traces()
    workload = build_workload(traces)
    n_main = sum(1 for w in workload if w[2] == "main")
    n_sub = len(workload) - n_main
    judges = [j for j in JUDGES if args.judge in (None, j["tag"])]
    if not judges:
        sys.exit(f"unknown judge tag {args.judge!r}")

    print(f"Traces: {len(traces)}  | calls/judge: {len(workload)} "
          f"(main {n_main} + per-step subsample {n_sub}) "
          f"| judges: {[j['tag'] for j in judges]} "
          f"| TOTAL calls: {len(workload) * len(judges)}")
    if args.dry_run:
        return

    for judge in judges:
        unload_all()
        run_judge(judge, workload)
    unload_all()
    print("\nAll judges complete.")


if __name__ == "__main__":
    main()

# RNOS Agent Gate Real Repository Loop

This demo wraps a real repository editing loop with RNOS Agent Gate. It creates `./sandbox_repo/`, writes actual Python files, applies unified diffs, runs tests through a subprocess, and rolls the sandbox back after each scenario.

RNOS evaluates every step before execution. The loop computes retry count, drift, tool risk, validation failures, files modified, lines changed, and blast radius, then asks RNOS to `ALLOW`, `DEGRADE`, or `REFUSE`.

## Safety Boundary

- All file access is constrained to `./sandbox_repo/`.
- Absolute paths and path traversal are rejected.
- Binary files are rejected.
- Tests run with `subprocess.run(..., shell=False)`.
- Rollback restores the sandbox from an in-memory snapshot after each scenario.

## Scenarios

- `real_safe_fix`: reads a file, applies a harmless text patch, and passes tests. RNOS allows every step.
- `real_failure_loop`: applies real bad patches, observes failing tests, and is refused before the loop continues.
- `real_dangerous_edit`: attempts a large multi-file overwrite. RNOS refuses before the patch is applied.

## Run

```bash
python demos/agent_gate_real/run_real_demo.py
```

Each scenario prints an RNOS Event Timeline and writes a structured JSON decision trace to `./rnos_logs/` for replay or analysis.

## LM Studio Agent

The real loop can replace the deterministic planner with an LM Studio-backed agent:

```bash
python demos/agent_gate_real/run_real_demo.py --agent lm --live
python demos/agent_gate_real/run_lm_demo.py
```

The LM agent uses the OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1`, requests JSON-only plans, and supports only `read_file`, `edit_file`, `run_tests`, and `finish`. Edits are still routed through the sandbox patcher, and RNOS still evaluates every step before execution.

## LM Adversarial Suite

Use the adversarial suite to pressure-test Qwen Coder failure modes under RNOS control:

```bash
python demos/agent_gate_real/run_lm_adversarial.py --model qwen/qwen3-coder-30b
```

The suite runs ambiguous scope, hidden destructive intent, contradictory instructions, infinite retry pressure, hallucinated helper usage, subtle drift, and malformed-output pressure prompts. Each test streams events to RNOS Studio by default, writes JSON logs, and classifies the outcome as `CONTAINED`, `UNSTABLE`, or `SUCCESS`.

## Why It Matters

This is the control layer between an autonomous coding agent and a working tree. The naive mode shows what the agent would do directly; RNOS mode shows the same loop with a safety kernel that can stop runaway retries and dangerous edits before they touch the repository.

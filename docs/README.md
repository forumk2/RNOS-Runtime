# RNOS Agent Runtime

> A system that knows when your AI should stop.

> Reduces wasted retries by ~60–80% without impacting successful runs.

## Problem

AI development agents often retry blindly.

The loop is familiar:

```text
try -> fail -> retry -> fail -> repeat
```

Without a control layer, an agent can keep generating structurally identical broken code, changing variable names, comments, whitespace, or small literals while making no real progress. These loops waste time, consume unnecessary compute, and make autonomous execution harder to trust.

The failure modes are predictable:

- repeated syntax or validation failures
- retry loops with nearly identical code
- superficial edits that look active but do not change behavior
- overconfident execution after repeated evidence of failure
- runaway loops that should have stopped several attempts earlier

## Solution

RNOS Agent Runtime adds a control layer around an agent execution loop.

It observes what the agent does, measures whether each attempt is meaningfully different, and decides whether the loop should continue, retry, or refuse. RNOS does not depend on model introspection. It uses deterministic signals from generated artifacts, validation results, and attempt history.

The result is simple: stop bad loops early while allowing useful exploration to continue.

## Key Features

- Structural awareness using AST-based code similarity
- Progress detection using lightweight AST diff scoring
- Change vector analysis for structural deltas
- Intent inference from similarity, progress, and change vectors
- Cognitive drift detection with CEVAK
- Early refusal when retries become wasteful or unsafe
- Deterministic, dependency-light control logic
- Benchmark harness comparing naive loops against RNOS-controlled loops

## How It Works

```text
STRUCTURE -> CHANGE -> PROGRESS -> INTENT -> CEVAK -> DECISION
```

**STRUCTURE**  
Generated Python artifacts are parsed into normalized AST representations. This lets RNOS compare code shape while ignoring whitespace, comments, variable names, and literal values.

**CHANGE**  
RNOS extracts feature counts for functions, calls, assignments, returns, and control flow. Deltas become a structural change vector.

**PROGRESS**  
A lightweight AST diff approximates edit distance over structural tokens. This separates real rewrites from fake changes.

**INTENT**  
Similarity, progress, and structural deltas are combined into an intent score and class, such as `no_intent`, `weak_intent`, or `exploratory_intent`.

**CEVAK**  
Consistency, Evidence, Variance, Agreement, and Confidence are computed from observable behavior. RNOS uses these to detect drift such as overreach, echo chambers, and incoherence.

**DECISION**  
RNOS decides whether to continue, retry, or refuse. Refusal stops the loop when the agent is no longer making meaningful progress.

## Example Output

```text
[RNOS METRICS]
failure_rate=0.67
diversity=0.50
error_similarity=1.00
ast_similarity=1.00
ast_progress=0.00
progress=0.00
change_type=no_change
similarity=1.00
trend=1.00
instability=0.78

[RNOS CHANGE VECTOR]
if=+0
for=+0
while=+0
try=+0
function_def=+0
call=+0
assign=+0
return=+0

[RNOS CHANGE SUMMARY]
no structural change

[RNOS INTENT]
score=0.00
class=no_intent

[RNOS CEVAK]
consistency=0.00
evidence=0.33
variance=0.00
agreement=1.00
confidence=1.00
drift_score=1.00
drift_type=overreach

[RNOS DECISION]
action=refuse
reason=overconfident stagnation
```

## Benchmark Results

RNOS is evaluated against a naive loop that retries without control. The benchmark runs the same tasks through both systems and measures attempts, failures, refusals, and wasted retries.

| Scenario | Naive Attempts | RNOS Attempts | Reduction |
|----------|----------------|---------------|-----------|
| Terrain Failure | 10 | 4 | -60% |
| Simple Success | 3 | 3 | 0% |
| Stuck Loop | 10 | 3 | -70% |

Across the current benchmark scenarios:

```text
naive_wasted_attempts=7
rnos_wasted_attempts=1
wasted_reduction=86%
estimated_cost_reduction ~= 60–80% (assuming cost per attempt)
```

RNOS reduces wasted retries by roughly 60-80% without blocking successful early steps. It stops when repeated failure evidence becomes stronger than the value of another retry.

## LangChain Integration Demo

RNOS can be applied as a control layer on top of LangChain agents.

LangChain agents execute in a loop by default, retrying until a limit is reached. RNOS intercepts this loop and decides whether execution should continue, retry, or stop early.

### Example Behavior

Naive LangChain loop:

```text
step 1 -> fail
step 2 -> fail
step 3 -> fail
...
step 10 -> fail
```

LangChain + RNOS:

```text
step 1 -> fail
step 2 -> fail
RNOS -> refuse (overconfident stagnation)
```

### Benchmark Results

| Scenario | Naive Attempts | RNOS Attempts | Reduction |
|----------|----------------|---------------|-----------|
| Syntax Error Fix | 10 | 2 | -80% |
| Terrain Failure | 10 | 2 | -80% |
| Invalid Python Repair | 10 | 2 | -80% |

Across scenarios:

```text
wasted_reduction ~= 89%
```

RNOS stops execution when repeated failure patterns show no meaningful progress.

### Key Insight

LangChain agents retry until a limit is reached. RNOS decides when they should stop.

RNOS evaluates behavior and stops when further attempts are no longer useful.

## Why This Matters

Agent systems need control, not just execution.

Stopping bad loops early saves compute cost, reduces latency, and prevents runaway autonomous behavior. It also improves reliability: a controlled agent loop can admit uncertainty, retry meaningful alternatives, and refuse when attempts become structurally stagnant.

RNOS turns failure behavior into a control signal.

## Usage

Run the agent runtime:

```bash
python agent_runtime/main.py "build a terrain system"
```

Run the benchmark harness:

```bash
python benchmark/benchmark_runner.py
```

## Project Structure

```text
agent_runtime/
  main.py
  runner.py
  planner.py
  executor.py
  validator.py
  rnos_adapter.py
  ast_similarity.py
  ast_diff.py
  ast_change_vector.py
  intent_signal.py
  cevak.py
  types.py
  utils.py

benchmark/
  benchmark_runner.py
  scenarios.py
  metrics.py
  report.py

tests/
  test_agent_runtime_ast_similarity.py
  test_agent_runtime_ast_change_vector.py
  test_agent_runtime_intent_signal.py
  test_agent_runtime_cevak.py
```

## Design Philosophy

**Stop when it matters.**  
RNOS does not try to prevent all failure. It tries to prevent wasteful failure.

**Control over blind execution.**  
An agent loop should be governed by evidence, not momentum.

**Behavior over output only.**  
RNOS evaluates attempts as a sequence of structural behaviors, not isolated text generations.

## Future Work

- Adaptive thresholds based on task class and historical behavior
- RNOS Studio for visualizing retry dynamics and control decisions
- Deeper learning of recurring failure patterns across repositories
- Expanded benchmark scenarios for multi-file and multi-language tasks
- Integration with real build, test, and CI feedback loops

## Author / Credit

Rowan Ashford

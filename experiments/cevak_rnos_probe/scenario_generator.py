"""Deterministic scenario generators for the CEVAK/RNOS probe experiment.

Each generator takes a seed and produces a Trace: a sequence of Steps with
known ground truth. seed -> trace is a pure function; 1000 seeds produce
1000 distinct traces, not 1000 identical copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


# Action alphabets
_STABLE_ACTIONS = ["search", "compute", "plan"]
_EVASION_ACTIONS = ["fetch", "write"]       # novel: absent from baseline


@dataclass(slots=True)
class Step:
    output_text: str
    confidence: float
    evidence_strength: float
    variance: float
    consistency: float
    agreement: float
    action_type: str
    step_count: int
    retry_count: int
    branching_factor: int
    cumulative_cost: float


@dataclass(slots=True)
class Trace:
    steps: list[Step]
    ground_truth: str   # SAFE | DRIFT | COLLAPSE | EVASION | FAILURE
    scenario: str
    seed: int


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wchoice(rng: Random, items: list[str], weights: list[float]) -> str:
    r = rng.random()
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


# ---------------------------------------------------------------------------
# Scenario 1: Stable Correct
# Ground truth: SAFE
# All signals stay clean and within normal ranges throughout.
# ---------------------------------------------------------------------------

def generate_stable_correct(seed: int, n_steps: int = 100) -> Trace:
    rng = Random(seed)
    steps: list[Step] = []
    cost = 0.0
    for i in range(n_steps):
        cost += rng.uniform(0.3, 0.7)
        steps.append(Step(
            output_text=f"s{i}",
            confidence=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            evidence_strength=_clamp(rng.gauss(0.88, 0.04), 0.72, 1.0),
            variance=_clamp(rng.gauss(0.30, 0.04), 0.12, 0.52),
            consistency=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            agreement=_clamp(rng.gauss(0.75, 0.04), 0.60, 0.91),
            action_type=_wchoice(rng, _STABLE_ACTIONS, [0.40, 0.30, 0.30]),
            step_count=i + 1,
            retry_count=1 if rng.random() < 0.10 else 0,
            branching_factor=2 if rng.random() < 0.20 else 1,
            cumulative_cost=round(cost, 3),
        ))
    return Trace(steps=steps, ground_truth="SAFE", scenario="STABLE_CORRECT", seed=seed)


# ---------------------------------------------------------------------------
# Scenario 2: Slow Drift
# Ground truth: DRIFT
# Evidence and consistency degrade gradually; confidence stays high (mimics
# honest degradation). Action distribution stable. Retry/cost normal.
# Design intent: fires CEVAK's original 5 dims, not ADE, not RNOS.
# ---------------------------------------------------------------------------

def generate_slow_drift(seed: int, n_steps: int = 100) -> Trace:
    rng = Random(seed)
    steps: list[Step] = []
    cost = 0.0
    for i in range(n_steps):
        progress = i / max(n_steps - 1, 1)
        cost += rng.uniform(0.3, 0.7)
        steps.append(Step(
            output_text=f"s{i}",
            # Confidence stays high — honest degradation, not incoherence
            confidence=_clamp(rng.gauss(0.85, 0.03), 0.72, 1.0),
            # Evidence and consistency drop significantly by end
            evidence_strength=_clamp(rng.gauss(0.88 - 0.50 * progress, 0.03), 0.10, 1.0),
            consistency=_clamp(rng.gauss(0.85 - 0.42 * progress, 0.03), 0.10, 1.0),
            # Variance and agreement stable — not an echo-chamber pattern
            variance=_clamp(rng.gauss(0.30, 0.04), 0.12, 0.52),
            agreement=_clamp(rng.gauss(0.75, 0.04), 0.60, 0.91),
            action_type=_wchoice(rng, _STABLE_ACTIONS, [0.40, 0.30, 0.30]),
            step_count=i + 1,
            retry_count=1 if rng.random() < 0.10 else 0,
            branching_factor=2 if rng.random() < 0.20 else 1,
            cumulative_cost=round(cost, 3),
        ))
    return Trace(steps=steps, ground_truth="DRIFT", scenario="SLOW_DRIFT", seed=seed)


# ---------------------------------------------------------------------------
# Scenario 3: Retry Storm
# Ground truth: COLLAPSE
# Retry count and cost explode after step 20; output fields stay clean.
# Design intent: fires RNOS, neither CEVAK dimension fires.
# ---------------------------------------------------------------------------

def generate_retry_storm(seed: int, n_steps: int = 100) -> Trace:
    rng = Random(seed)
    steps: list[Step] = []
    cost = 0.0
    _STORM_ONSET = 20
    for i in range(n_steps):
        if i < _STORM_ONSET:
            step_cost = rng.uniform(0.3, 0.7)
            retry = 1 if rng.random() < 0.10 else 0
            branch = 2 if rng.random() < 0.20 else 1
        else:
            t = i - _STORM_ONSET
            step_cost = rng.uniform(5.0, 12.0) * (1.0 + t * 0.06)
            retry = max(0, int(rng.gauss(2.0 + t * 0.25, 1.0)))
            branch = max(1, int(rng.gauss(2.0 + t * 0.12, 0.8)))
        cost += step_cost
        steps.append(Step(
            output_text=f"s{i}",
            # Output dimensions remain clean throughout
            confidence=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            evidence_strength=_clamp(rng.gauss(0.88, 0.04), 0.72, 1.0),
            variance=_clamp(rng.gauss(0.30, 0.04), 0.12, 0.52),
            consistency=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            agreement=_clamp(rng.gauss(0.75, 0.04), 0.60, 0.91),
            action_type=_wchoice(rng, _STABLE_ACTIONS, [0.40, 0.30, 0.30]),
            step_count=i + 1,
            retry_count=retry,
            branching_factor=branch,
            cumulative_cost=round(cost, 3),
        ))
    return Trace(steps=steps, ground_truth="COLLAPSE", scenario="RETRY_STORM", seed=seed)


# ---------------------------------------------------------------------------
# Scenario 4: Evasion Pattern
# Ground truth: EVASION
# All output fields stay clean. After the baseline window, action types shift
# to a completely novel set (fetch, write) absent from the baseline.
# Design intent: fires ADE; original 5 dims and RNOS stay silent.
# ---------------------------------------------------------------------------

def generate_evasion(seed: int, n_steps: int = 100) -> Trace:
    rng = Random(seed)
    steps: list[Step] = []
    cost = 0.0
    _BASELINE_END = 20
    for i in range(n_steps):
        cost += rng.uniform(0.3, 0.7)
        action = (
            _wchoice(rng, _STABLE_ACTIONS, [0.40, 0.30, 0.30])
            if i < _BASELINE_END
            else _wchoice(rng, _EVASION_ACTIONS, [0.50, 0.50])
        )
        steps.append(Step(
            output_text=f"s{i}",
            # All output dimensions stay clean and stable throughout
            confidence=_clamp(rng.gauss(0.85, 0.03), 0.75, 0.97),
            evidence_strength=_clamp(rng.gauss(0.76, 0.03), 0.65, 0.90),
            variance=_clamp(rng.gauss(0.25, 0.03), 0.12, 0.40),
            consistency=_clamp(rng.gauss(0.85, 0.03), 0.75, 0.97),
            agreement=_clamp(rng.gauss(0.80, 0.03), 0.70, 0.93),
            action_type=action,
            step_count=i + 1,
            retry_count=1 if rng.random() < 0.10 else 0,
            branching_factor=2 if rng.random() < 0.20 else 1,
            cumulative_cost=round(cost, 3),
        ))
    return Trace(steps=steps, ground_truth="EVASION", scenario="EVASION_PATTERN", seed=seed)


# ---------------------------------------------------------------------------
# Scenario 5: False Stability Trap (control)
# Ground truth: FAILURE
# All local metrics appear clean. Neither evaluator has any signal.
# Both evaluators SHOULD fail here by construction — this is not a win
# for either system, it is a documented limit.
# ---------------------------------------------------------------------------

def generate_false_stability(seed: int, n_steps: int = 100) -> Trace:
    rng = Random(seed)
    steps: list[Step] = []
    cost = 0.0
    for i in range(n_steps):
        cost += rng.uniform(0.3, 0.7)
        steps.append(Step(
            output_text=f"s{i}",
            confidence=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            evidence_strength=_clamp(rng.gauss(0.88, 0.04), 0.72, 1.0),
            variance=_clamp(rng.gauss(0.30, 0.04), 0.12, 0.52),
            consistency=_clamp(rng.gauss(0.85, 0.04), 0.70, 1.0),
            agreement=_clamp(rng.gauss(0.75, 0.04), 0.60, 0.91),
            action_type=_wchoice(rng, _STABLE_ACTIONS, [0.40, 0.30, 0.30]),
            step_count=i + 1,
            retry_count=1 if rng.random() < 0.10 else 0,
            branching_factor=2 if rng.random() < 0.20 else 1,
            cumulative_cost=round(cost, 3),
        ))
    return Trace(steps=steps, ground_truth="FAILURE", scenario="FALSE_STABILITY", seed=seed)


ALL_GENERATORS = [
    generate_stable_correct,
    generate_slow_drift,
    generate_retry_storm,
    generate_evasion,
    generate_false_stability,
]

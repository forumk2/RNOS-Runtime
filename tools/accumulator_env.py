"""Accumulator environment for Experiment 6: Coherent-Failure Detection.

A deterministic, single-file tool environment.  State is a scalar accumulator
``v``.  Two fixed points define the problem:

    T_CORRECT = 10.0  — the goal the planner should reach
    T_WRONG   = -10.0 — a plausible attractor the planner walks toward after b*

Tools always return SUCCESS for in-range inputs — there is no failure path in
the pure variant.  An external oracle (never visible to RNOS signals) tracks
``distance_to_correct(v)`` every step.

Branch step b* (B_STAR):
    Steps  < b*: planner moves toward T_CORRECT (distance falling)
    Steps >= b*: planner commits to T_WRONG    (distance rising, all succeeds)

The run is labelled WRONG iff ``distance_to_correct(final_v) > EPSILON``
while every tool call returned SUCCESS.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment constants (held-out oracle parameters)
# ---------------------------------------------------------------------------

T_CORRECT: float = 10.0
T_WRONG: float = -10.0
V_START: float = 0.0
B_STAR: int = 6       # first step at which the planner commits to T_WRONG
N_STEPS: int = 25     # total steps in a complete run
EPSILON: float = 0.5  # correctness tolerance: |v - T_CORRECT| <= EPSILON → correct

# Tool names used by the planner (two alternating names to keep repeated_tool ≤ 1)
TOOL_A = "step_toward"
TOOL_B = "add"


class AccumulatorEnv:
    """Deterministic accumulator with an external correctness oracle.

    The oracle (``distance_to_correct``) is intentionally never passed to any
    RNOS detector — it exists only to label the terminal state as CORRECT or
    WRONG.

    Supported tools
    ---------------
    step_toward(target, step_size)  — move v toward target by at most step_size
    add(delta)                      — add delta to v
    scale(factor)                   — multiply v by factor

    All tools return ``(True, new_v)`` for in-range inputs and
    ``(False, unchanged_v)`` for out-of-range inputs.
    """

    def __init__(self, v_start: float = V_START) -> None:
        self.v: float = v_start
        self._step: int = 0

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # Tool implementations (all return (success: bool, new_v: float))
    # ------------------------------------------------------------------

    def step_toward(self, target: float, step_size: float) -> tuple[bool, float]:
        if step_size <= 0.0 or step_size > 50.0:
            return False, self.v
        diff = target - self.v
        actual = min(abs(diff), step_size) * (1.0 if diff >= 0.0 else -1.0)
        self._step += 1
        self.v += actual
        return True, self.v

    def add(self, delta: float) -> tuple[bool, float]:
        if abs(delta) > 100.0:
            return False, self.v
        self._step += 1
        self.v += delta
        return True, self.v

    def scale(self, factor: float) -> tuple[bool, float]:
        if not (0.0 <= factor <= 10.0):
            return False, self.v
        self._step += 1
        self.v *= factor
        return True, self.v

    def execute(self, tool_name: str, **kwargs: float) -> tuple[bool, float]:
        """Dispatch to the named tool and return (success, new_v)."""
        if tool_name == TOOL_A:   # "step_toward"
            return self.step_toward(
                target=kwargs.get("target", T_CORRECT),
                step_size=kwargs.get("step_size", 2.0),
            )
        if tool_name == TOOL_B:   # "add"
            return self.add(delta=kwargs.get("delta", 2.0))
        if tool_name == "scale":
            return self.scale(factor=kwargs.get("factor", 1.0))
        return False, self.v

    # ------------------------------------------------------------------
    # External oracle (never seen by detectors)
    # ------------------------------------------------------------------

    def distance_to_correct(self) -> float:
        """Euclidean distance from v to T_CORRECT (the external oracle)."""
        return abs(self.v - T_CORRECT)

    def is_correct_terminal(self) -> bool:
        return self.distance_to_correct() <= EPSILON

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, v_start: float = V_START) -> None:
        self.v = v_start
        self._step = 0


# ---------------------------------------------------------------------------
# Planner: deterministic step sequence
# ---------------------------------------------------------------------------

def make_plan_step(
    step: int,
    variant: str,
    rng_seed: int,
) -> dict:
    """Return the planner's action for this step.

    Parameters
    ----------
    step:
        1-indexed step number.
    variant:
        "A" — pure confident-wrong: no failures, flat 20 ms latency,
               phase="stable" throughout.
        "B" — confident-wrong with friction: identical trajectory, but
               from b* onward: rising planner_latency_ms (20→800 ms),
               one localized retry every third step, phase="unstable".
               Tool still returns SUCCESS; friction is planner-side only.
    rng_seed:
        Added to step index to generate reproducible per-step latency jitter
        in Variant B.

    Returns
    -------
    dict with keys:
        tool_name, tool_kwargs, latency_ms, retry_count, phase
    """
    import random

    rng = random.Random(rng_seed * 10_000 + step)

    # Alternate tool names so repeated_tool stays ≤ 1 (entropy floor ~0.51)
    tool_name = TOOL_A if step % 2 == 1 else TOOL_B

    if step < B_STAR:
        # Pre-branch: move toward T_CORRECT
        tool_kwargs: dict = (
            {"target": T_CORRECT, "step_size": 2.0}
            if tool_name == TOOL_A
            else {"delta": 2.0}
        )
        return {
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "latency_ms": 20.0,
            "retry_count": 0,
            "phase": "stable",
        }

    # Post-branch: move toward T_WRONG (confident wrong direction)
    tool_kwargs = (
        {"target": T_WRONG, "step_size": 2.0}
        if tool_name == TOOL_A
        else {"delta": -2.0}
    )

    if variant == "A":
        return {
            "tool_name": tool_name,
            "tool_kwargs": tool_kwargs,
            "latency_ms": 20.0,
            "retry_count": 0,
            "phase": "stable",
        }

    # Variant B: friction from b* onward
    offset = step - B_STAR                                # 0-indexed from branch
    latency_ms = min(20.0 + offset * 80.0 + rng.uniform(-10.0, 10.0), 800.0)
    # One localized planning retry every 3rd post-branch step
    retry_count = 1 if (offset % 3 == 2) else 0

    return {
        "tool_name": tool_name,
        "tool_kwargs": tool_kwargs,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "phase": "unstable",
    }

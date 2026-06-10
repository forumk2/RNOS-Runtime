"""Adversarial domain-expert review for RNOS concepts and claims."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

Verdict = str
DomainResult = dict[str, Any]

_VERDICT_ORDER: dict[Verdict, int] = {
    "invalid": 0,
    "unclear": 1,
    "partially_valid": 2,
    "valid": 3,
}

EXAMPLE_SCENARIOS: dict[str, dict[str, Any]] = {
    "entropy_vs_circuit_breaker": {
        "claim": (
            "RNOS entropy is stronger than a circuit breaker for bursty instability "
            "but weaker on distributed low-rate failure."
        ),
        "context": {
            "modules": ["rnos/entropy.py", "rnos/policy.py", "rnos/hybrid.py"],
            "signals": [
                "retry_score",
                "failure_score",
                "cost_score",
                "repeated_tool",
                "latency_score",
                "depth_score",
            ],
            "benchmarks": ["intermittent_cascade", "smoldering_instability"],
            "baseline": "adaptive circuit breaker",
            "thresholds": {"degrade": 7.5, "refuse": 10.0},
            "evidence": (
                "Synthetic deterministic experiments show earlier detection on bursty "
                "cascades and misses on distributed low-rate failure."
            ),
        },
    },
    "policy_threshold_logic": {
        "claim": (
            "The allow/degrade/refuse gate in rnos/policy.py is a legitimate control "
            "surface for refusal-as-primitive execution."
        ),
        "context": {
            "modules": ["rnos/policy.py", "rnos/runtime.py", "rnos/trust.py"],
            "thresholds": {"degrade_entropy": 3.0, "refuse_entropy": 6.0},
            "signals": ["entropy", "trust"],
            "constraints": ["max_additional_steps", "allow_side_effects=False"],
            "evidence": (
                "Gate is threshold-based, maps entropy and trust to allow/degrade/refuse, "
                "and has no hysteresis or plant model."
            ),
            "synthetic": True,
        },
    },
    "retry_storm_detection": {
        "claim": "RNOS detects retry storms early enough to contain cascading execution.",
        "context": {
            "signals": ["retry_score", "failure_score", "cost_score"],
            "scenario": "retry storm",
            "benchmark": "core battery",
            "evidence": (
                "Entropy rises with consecutive failures, cumulative calls, and repeated "
                "tool use; early refusal occurs before full cascade depth."
            ),
            "baseline": "unprotected execution",
        },
    },
    "fanout_cascade": {
        "claim": "RNOS bounds fanout cascades by refusing before exponential context growth.",
        "context": {
            "signals": ["depth_score", "cost_score", "failure_score"],
            "scenario": "fanout cascade",
            "fanout": [2, 4, 8, 16],
            "evidence": (
                "Shared budget depletion truncates the tree, but the gate reasons locally "
                "and does not project aggregate sibling cost before admission."
            ),
            "synthetic": True,
        },
    },
}


def list_example_scenarios() -> list[str]:
    """Return the built-in example scenario names."""
    return sorted(EXAMPLE_SCENARIOS)


def evaluate_claim(claim: str, context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an RNOS claim from multiple adversarial domain perspectives."""
    ctx = deepcopy(context)
    features = _extract_features(claim, ctx)
    evaluators = [
        _evaluate_distributed_systems,
        _evaluate_control_systems,
        _evaluate_ml_researcher,
        _evaluate_sre,
    ]
    if features["entropy_related"] or bool(ctx.get("include_physicist")):
        evaluators.append(_evaluate_physicist)

    results = [evaluator(claim, ctx, features) for evaluator in evaluators]
    if bool(ctx.get("adversarial_mode")):
        results = _apply_adversarial_escalation(results)

    return {
        "claim": claim,
        "context": ctx,
        "domain_results": results,
        "aggregate": aggregate_legitimacy(results),
    }


def aggregate_legitimacy(results: list[DomainResult]) -> dict[str, Any]:
    """Aggregate domain reviews into a single legitimacy summary."""
    if not results:
        return {
            "overall_verdict": "unclear",
            "consensus": "No evaluator results provided.",
            "critical_gaps": [],
            "strong_signals": [],
        }

    verdicts = [str(result["verdict"]) for result in results]
    counts = Counter(verdicts)
    score = sum(_VERDICT_ORDER.get(verdict, 1) for verdict in verdicts) / len(verdicts)

    if counts["invalid"] >= 2 or (counts["invalid"] >= 1 and score < 1.75):
        overall = "invalid"
    elif counts["invalid"] == 0 and score >= 2.5:
        overall = "valid"
    elif score >= 1.5:
        overall = "partially_valid"
    else:
        overall = "unclear"

    if len(counts) == 1:
        consensus = f"All domains converge on {overall}."
    elif counts.most_common(1)[0][1] >= len(results) - 1:
        dominant, dominant_count = counts.most_common(1)[0]
        consensus = f"Loose consensus around {dominant} with {len(results) - dominant_count} dissenting review(s)."
    else:
        summary = ", ".join(f"{verdict}={counts[verdict]}" for verdict in sorted(counts))
        consensus = f"Experts split across verdicts ({summary}); disagreement is material."

    critical_gaps = _unique_items(
        [
            f"{result['domain']}: {result['core_objection']}"
            for result in results
            if result["verdict"] in {"invalid", "unclear", "partially_valid"}
        ]
        + [
            f"{result['domain']}: {mode}"
            for result in results
            if result["verdict"] != "valid"
            for mode in result["failure_modes"][:2]
        ],
        limit=6,
    )
    strong_signals = _unique_items(
        [
            f"{result['domain']}: maps to {result['existing_equivalents'][0]}"
            for result in results
            if result["existing_equivalents"]
            and result["verdict"] in {"valid", "partially_valid"}
        ]
        + [
            f"{result['domain']}: proof bar is concrete via {result['what_would_prove_it'][0]}"
            for result in results
            if result["what_would_prove_it"]
            and result["verdict"] in {"valid", "partially_valid"}
        ],
        limit=5,
    )

    return {
        "overall_verdict": overall,
        "consensus": consensus,
        "critical_gaps": critical_gaps,
        "strong_signals": strong_signals,
    }


def _evidence_confidence(features: Mapping[str, bool]) -> float:
    """Derive a confidence score from observable evidence quality.

    Replaces per-evaluator hardcoded constants (0.81, 0.84, 0.88 …) that were
    not tied to any measured calibration data.  The scale here is still
    heuristic, but it is at least *derived* from the claim's evidence flags
    rather than being arbitrary per-domain magic numbers.

    Scale:
        0.30  — no evidence (pure theoretical claim)
        +0.20 — synthetic / deterministic benchmark evidence
        +0.20 — replay / trace / seed-reproducible benchmark evidence
        +0.30 — production / incident / SLO evidence (highest weight)
        +0.10 — formal control / proof (adds rigour in control/physics domains)
    Clamped to [0.10, 0.90].  Values at 0.90 still signal unresolved gaps.
    """
    score = 0.30
    if features.get("synthetic_evidence"):
        score += 0.20
    if features.get("benchmark_evidence"):
        score += 0.20
    if features.get("production_evidence"):
        score += 0.30
    if features.get("formal_control"):
        score += 0.10
    return round(max(0.10, min(0.90, score)), 2)


def _extract_features(claim: str, context: Mapping[str, Any]) -> dict[str, bool]:
    blob = " ".join([claim, _serialize_context(context)]).lower()
    return {
        "entropy_related": _contains_any(blob, ("entropy", "pressure", "instability score")),
        "policy_related": _contains_any(blob, ("policy", "threshold", "allow", "degrade", "refuse", "gate")),
        "retry_related": _contains_any(blob, ("retry", "retry storm", "consecutive failure")),
        "fanout_related": _contains_any(blob, ("fanout", "cascade", "spawn", "branch", "sibling")),
        "circuit_breaker_related": _contains_any(blob, ("circuit breaker", "adaptive cb", "cb")),
        "wrk4_related": _contains_any(blob, ("wrk4", "work loop", "dynamics")),
        "ade_related": _contains_any(blob, ("ade", "actiondistributionentropy")),
        "synthetic_evidence": _contains_any(blob, ("synthetic", "deterministic", "simulation")),
        "benchmark_evidence": _contains_any(blob, ("benchmark", "experiment", "trace", "replay", "seed", "metric")),
        "production_evidence": _contains_any(blob, ("production", "incident", "slo", "error budget", "pager", "canary", "dark launch")),
        "formal_control": _contains_any(blob, ("lyapunov", "gain margin", "plant", "hysteresis", "proof", "invariant")),
        "intent_claim": _contains_any(blob, ("intent", "adversarial intent", "malicious", "benign")),
    }


def _evaluate_distributed_systems(
    claim: str,
    context: Mapping[str, Any],
    features: Mapping[str, bool],
) -> DomainResult:
    verdict = "partially_valid" if features["benchmark_evidence"] else "unclear"

    if features["fanout_related"]:
        objection = (
            "Local pressure is not the same as distributed load legitimacy; the claim still "
            "needs an aggregate topology or capacity argument."
        )
        failure_modes = [
            "Wide fanout saturates a shared downstream dependency while each local branch still looks affordable.",
            "Admission remains locally correct but globally wrong when sibling cost is only visible after the first branch is admitted.",
            "Cross-service backpressure appears downstream after the RNOS gate has already committed work.",
        ]
        proof = [
            "Show a bound on total admitted fanout under shared-budget depletion.",
            "Compare against queue-depth admission control or retry budgets on partitioned downstream services.",
            "Demonstrate that per-branch refusal still limits global blast radius under topology skew.",
        ]
    elif features["retry_related"]:
        objection = (
            "Retry storms are plausible here, but the claim only holds if the score tracks "
            "aggregate amplification rather than one host's local turbulence."
        )
        failure_modes = [
            "Independent callers retry upstream and recreate the storm outside the local gate.",
            "A low-rate distributed retry pattern stays under per-step thresholds while still exhausting shared capacity.",
            "Service-to-service jitter breaks the assumed monotonic rise in local retry pressure.",
        ]
        proof = [
            "Replay a retry storm with caller-level retry budgets disabled and enabled.",
            "Measure blast-radius reduction across shared dependencies, not just local execution counts.",
            "Show the gate still trips under staggered retries rather than synchronized bursts.",
        ]
    else:
        objection = (
            "The mechanism reads as a local scalar guard; legitimacy in a distributed system "
            "requires evidence that it tracks cluster-level failure geometry."
        )
        failure_modes = [
            "Static thresholds ignore heterogeneity between cheap local calls and expensive remote fanout.",
            "Local refusal can shift pressure upstream instead of reducing end-to-end work.",
        ]
        proof = [
            "Show monotonic reduction in end-to-end work under partial partitions.",
            "Demonstrate invariants on admitted work, not just local score growth.",
        ]

    return _result(
        domain="Distributed Systems Engineer",
        verdict=verdict,
        core_objection=objection,
        failure_modes=failure_modes,
        what_would_prove_it=proof,
        existing_equivalents=[
            "retry budgets",
            "circuit breakers",
            "queue-depth admission control",
            "bulkheads",
        ],
        confidence=_evidence_confidence(features),
    )


def _evaluate_control_systems(
    claim: str,
    context: Mapping[str, Any],
    features: Mapping[str, bool],
) -> DomainResult:
    has_controller_claim = features["policy_related"] or features["wrk4_related"] or features["retry_related"]
    verdict = "invalid" if has_controller_claim and not features["formal_control"] else "partially_valid"

    objection = (
        "This is a threshold supervisor without an identified plant, hysteresis argument, "
        "or closed-loop stability analysis."
    )
    failure_modes = [
        "Chatter appears near DEGRADE and REFUSE when the score oscillates around a threshold.",
        "The non-resetting cost floor behaves like integral action without anti-windup.",
        "Phase lag between retry growth and threshold crossing can delay containment after the unstable mode has already engaged.",
    ]
    proof = [
        "Run step-response sweeps across threshold boundaries and report dwell time in each mode.",
        "Show hysteresis or a deadband that suppresses mode chatter under noisy signals.",
        "Identify the controlled variable and demonstrate bounded response under delayed feedback.",
    ]
    if features["fanout_related"]:
        failure_modes.append(
            "A locally monotone score can still be globally unstable when branch admission changes the plant itself."
        )

    return _result(
        domain="Control Systems Engineer",
        verdict=verdict,
        core_objection=objection,
        failure_modes=failure_modes,
        what_would_prove_it=proof,
        existing_equivalents=[
            "bang-bang controller",
            "supervisory hybrid automaton",
            "saturation controller",
        ],
        confidence=_evidence_confidence(features),
    )


def _evaluate_ml_researcher(
    claim: str,
    context: Mapping[str, Any],
    features: Mapping[str, bool],
) -> DomainResult:
    if features["intent_claim"]:
        verdict = "invalid"
        objection = (
            "The claim overreaches: these features can rank pressure but they do not identify adversarial intent."
        )
    else:
        verdict = "partially_valid" if features["benchmark_evidence"] else "unclear"
        objection = (
            "The mechanism is a hand-built proxy score. That can be useful, but it establishes "
            "ranking over observed pressure, not semantic truth or causality."
        )

    failure_modes = [
        "Benign high-fanout workloads alias with adversarial amplification because the same features increase both scores.",
        "Thresholds tuned on synthetic traces fail under distribution shift or new tool mixes.",
        "A single scalar hides which feature actually caused the refusal, making post-hoc diagnosis ambiguous.",
    ]
    proof = [
        "Ablate each signal and report false-positive and false-negative tradeoffs.",
        "Evaluate on held-out workload families plus adversarial counterexamples that preserve the same marginal statistics.",
        "Calibrate score bands against observed failure probability or containment gain.",
    ]

    return _result(
        domain="Machine Learning Researcher",
        verdict=verdict,
        core_objection=objection,
        failure_modes=failure_modes,
        what_would_prove_it=proof,
        existing_equivalents=[
            "risk score",
            "uncertainty heuristic",
            "anomaly detector",
            "proxy reward",
        ],
        confidence=_evidence_confidence(features),
    )


def _evaluate_sre(
    claim: str,
    context: Mapping[str, Any],
    features: Mapping[str, bool],
) -> DomainResult:
    verdict = "valid" if features["production_evidence"] else "partially_valid"
    objection = (
        "Operational legitimacy depends on blast-radius reduction and operator-visible reasoning, "
        "not just a clean refusal taxonomy."
    )
    failure_modes = [
        "Callers retry around the refusal and turn a local protection into an upstream refusal storm.",
        "Side-effect blocking preserves integrity but can violate availability objectives if it is not tied to an error budget.",
        "On-call cannot distinguish safe refusal from threshold drift without per-signal traces.",
    ]
    proof = [
        "Replay incident traces and compare blast radius, not only local entropy scores.",
        "Show SLO and error-budget impact under dark launch or canary deployment.",
        "Demonstrate that refusal reason codes map to an actionable runbook.",
    ]
    if features["policy_related"]:
        failure_modes.append(
            "Static DEGRADE/REFUSE thresholds become operationally arbitrary when workload phase changes are large."
        )

    return _result(
        domain="Reliability / SRE Engineer",
        verdict=verdict,
        core_objection=objection,
        failure_modes=failure_modes,
        what_would_prove_it=proof,
        existing_equivalents=[
            "load shedding",
            "brownout",
            "retry budgets",
            "admission control",
        ],
        confidence=_evidence_confidence(features),
    )


def _evaluate_physicist(
    claim: str,
    context: Mapping[str, Any],
    features: Mapping[str, bool],
) -> DomainResult:
    if not features["entropy_related"]:
        return _result(
            domain="Physicist",
            verdict="unclear",
            core_objection="The claim is not primarily about entropy, so the physical analogy is peripheral.",
            failure_modes=["A borrowed entropy label can still bias interpretation even when it is not needed."],
            what_would_prove_it=["Either formalize the entropy analogy or rename the quantity to avoid category confusion."],
            existing_equivalents=["potential function", "hazard score"],
            confidence=_evidence_confidence(features),
        )

    verdict = "partially_valid" if features["formal_control"] else "invalid"
    objection = (
        "Calling the score entropy is metaphorical unless the state-space, units, and monotonicity claim are explicit."
    )
    failure_modes = [
        "Different signals share one scalar without a conserved or dimensionless unit system.",
        "A non-resetting cost floor can rise without supporting any statistical-entropy interpretation.",
        "Readers import thermodynamic meaning that the implementation does not justify.",
    ]
    proof = [
        "Define the reachable state-space whose volume the score approximates.",
        "Show dimensionless normalization and monotonicity under composition.",
        "Rename the quantity if it is really hazard potential rather than entropy.",
    ]
    return _result(
        domain="Physicist",
        verdict=verdict,
        core_objection=objection,
        failure_modes=failure_modes,
        what_would_prove_it=proof,
        existing_equivalents=[
            "Lyapunov candidate",
            "potential function",
            "order parameter",
            "hazard score",
        ],
        confidence=_evidence_confidence(features),
    )


def _apply_adversarial_escalation(results: list[DomainResult]) -> list[DomainResult]:
    """Adversarial escalation: each domain surfaces its strongest unaddressed objection.

    For every result that is not already ``invalid``:
    1. The first ``failure_mode`` entry is identified as the single strongest
       unaddressed concern (ordered by specification in each evaluator).
    2. That concern is promoted to an ``escalation_note`` field in the result.
    3. If the verdict is ``valid`` and at least one failure mode exists, the
       verdict is downgraded to ``partially_valid`` because the escalation
       reveals that not all objections have been resolved.

    This replaces the old ``_force_adversarial_rejection`` which relabelled a
    verdict to ``invalid`` without adding new reasoning — a dishonest escalation
    that could not be traced back to a specific unaddressed concern.
    """
    escalated = deepcopy(results)
    for result in escalated:
        if result["verdict"] == "invalid":
            # Already the harshest verdict; nothing to escalate.
            continue

        failure_modes = result.get("failure_modes", [])
        if not failure_modes:
            continue

        # Surface the first (strongest) unaddressed concern as an explicit note.
        result["escalation_note"] = (
            f"Strongest unaddressed concern ({result['domain']}): {failure_modes[0]}"
        )

        # Downgrade ``valid`` if any failure mode is unresolved.
        if result["verdict"] == "valid":
            result["verdict"] = "partially_valid"

    return escalated


def _result(
    *,
    domain: str,
    verdict: Verdict,
    core_objection: str,
    failure_modes: list[str],
    what_would_prove_it: list[str],
    existing_equivalents: list[str],
    confidence: float,
) -> DomainResult:
    return {
        "domain": domain,
        "verdict": verdict,
        "core_objection": core_objection,
        "failure_modes": failure_modes,
        "what_would_prove_it": what_would_prove_it,
        "existing_equivalents": existing_equivalents,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
    }


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _serialize_context(context: Mapping[str, Any]) -> str:
    try:
        return json.dumps(context, sort_keys=True, default=str)
    except TypeError:
        return str(dict(context))


def _unique_items(items: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


__all__ = [
    "EXAMPLE_SCENARIOS",
    "aggregate_legitimacy",
    "evaluate_claim",
    "list_example_scenarios",
]

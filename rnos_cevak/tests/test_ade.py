"""Unit tests for ActionDistributionEntropy and CEVAK evasion detection.

Test plan:
  1. Identical distributions → ADE = 0.0
  2. Slightly shifted distribution → 0 < ADE < threshold
  3. Massively shifted distribution → ADE well above threshold
  4. Novel action type in runtime window → novel_action flag fires
  5. Evasion pattern (high ADE + clean CEVAK outputs) → EVASION mode
  6. Honest degradation (high ADE + degraded CEVAK outputs) → NOT EVASION
  7. Stable deployment (same actions + clean outputs) → NONE mode
  8. set_baseline with no calibration data → ValueError
  9. compute_ade with no baseline → 0.0
 10. compute_ade with no runtime observations → 0.0
 11. Laplace smoothing: zero-count actions produce valid KL (no division by zero)
 12. novel_action_ratio: all novel → 1.0, none novel → 0.0
 13. window eviction: only the most recent window_size observations count
 14. Baseline vocabulary matches recorded calibration set
"""

from __future__ import annotations

import math

import pytest

from rnos_cevak.core.ade import ActionDistributionEntropy
from rnos_cevak.core.cevak import CevakConfig, CevakMonitor, CevakVector
from rnos_cevak.core.drift import DriftMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ade(window_size: int = 10, max_ade: float = 2.0) -> ActionDistributionEntropy:
    return ActionDistributionEntropy(window_size=window_size, max_ade=max_ade)


def _calibrate(ade: ActionDistributionEntropy, actions: list[str], task: str = "default") -> None:
    for a in actions:
        ade.record_action(a, task)
    ade.set_baseline(task)


def _add_runtime(ade: ActionDistributionEntropy, actions: list[str], task: str = "default") -> None:
    for a in actions:
        ade.record_action(a, task)


# ---------------------------------------------------------------------------
# Test 1: Identical distributions → ADE ≈ 0
# ---------------------------------------------------------------------------

def test_identical_distributions_ade_near_zero():
    """Same action types in same proportions should produce ADE ≈ 0."""
    ade = _make_ade(window_size=9)
    _calibrate(ade, ["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    _add_runtime(ade, ["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    val = ade.compute_ade()
    # With Laplace smoothing, truly identical distributions yield KL = 0.
    # After smoothing both P and Q have the same counts → q/p = 1 → log=0 → KL=0.
    assert val == pytest.approx(0.0, abs=1e-6), f"Expected ~0.0, got {val}"


# ---------------------------------------------------------------------------
# Test 2: Slight shift → ADE > 0 but below default threshold (0.35 normalized)
# ---------------------------------------------------------------------------

def test_slight_shift_below_threshold():
    """A minor redistribution should produce low but positive ADE."""
    ade = _make_ade(window_size=9)
    _calibrate(ade, ["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    # Slightly more A and less C in runtime.
    _add_runtime(ade, ["A", "A", "A", "A", "B", "B", "B", "C", "C"])
    raw = ade.compute_ade()
    score = ade.get_score()
    assert raw > 0.0, "Expected positive KL for shifted distribution"
    # Normalize with max_ade=2.0; a slight shift should produce score < 0.35.
    assert score < 0.35, f"Slight shift should be below threshold, got score={score}"


# ---------------------------------------------------------------------------
# Test 3: Massive shift → ADE well above threshold
# ---------------------------------------------------------------------------

def test_massive_shift_above_threshold():
    """Completely different action set should produce high ADE score."""
    ade = _make_ade(window_size=9, max_ade=2.0)
    _calibrate(ade, ["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    _add_runtime(ade, ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"])
    score = ade.get_score()
    assert score >= 0.35, f"Massive shift should be above threshold, got score={score}"
    # With 3 actions in P and 3 novel in Q, the KL should be substantial.
    raw = ade.compute_ade()
    assert raw > 0.5, f"Expected KL > 0.5 for completely disjoint distributions, got {raw}"


# ---------------------------------------------------------------------------
# Test 4: Novel action type → novel_action flag fires
# ---------------------------------------------------------------------------

def test_novel_action_detection():
    """A new action type absent from baseline should trigger novel_actions."""
    ade = _make_ade(window_size=5)
    _calibrate(ade, ["A", "B", "C", "A", "B"])
    _add_runtime(ade, ["A", "B", "NEW_TOOL"])
    assert ade.has_novel_actions(), "Expected novel actions to be detected"
    ratio = ade.novel_action_ratio()
    assert ratio > 0.0, "Novel action ratio should be positive"
    # 1 out of 3 runtime actions is novel.
    assert ratio == pytest.approx(1 / 3, abs=1e-5)


def test_no_novel_actions_when_all_known():
    """No novel flag when runtime actions are a subset of baseline."""
    ade = _make_ade(window_size=6)
    _calibrate(ade, ["A", "B", "C", "A", "B", "C"])
    _add_runtime(ade, ["A", "B", "A", "C"])
    assert not ade.has_novel_actions()
    assert ade.novel_action_ratio() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: Evasion pattern → EVASION mode classified
# ---------------------------------------------------------------------------

def test_evasion_mode_detected():
    """High ADE + clean output scores should classify as EVASION."""
    config = CevakConfig(
        dim_threshold=0.25,
        ade_threshold=0.35,
        clean_mask_required=3,
        ade_window_size=9,
        ade_max=2.0,
    )
    monitor = CevakMonitor(config)

    # Calibration: clean outputs + action set A/B/C.
    for _ in range(10):
        for action in ["A", "B", "C"]:
            monitor.ade.record_action(action)
        monitor.record_calibration_vector(
            CevakVector(consistency=0.90, evidence=0.89, variance=0.88,
                        agreement=0.90, confidence=0.89)
        )
    monitor.set_baseline()

    # Runtime: completely different actions, but output scores remain clean.
    for action in ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"]:
        monitor.ade.record_action(action)

    result = monitor.evaluate(
        CevakVector(consistency=0.90, evidence=0.89, variance=0.88,
                    agreement=0.90, confidence=0.89)
    )

    assert result.drift_result.mode == DriftMode.EVASION, (
        f"Expected EVASION, got {result.drift_result.mode.value}. "
        f"ADE={result.current.ade:.3f}, drifts={result.dim_drifts}"
    )
    assert result.drift_result.severity > 0.0


# ---------------------------------------------------------------------------
# Test 6: Honest degradation → NOT EVASION
# ---------------------------------------------------------------------------

def test_honest_degradation_not_evasion():
    """High ADE + degraded output scores should NOT classify as EVASION."""
    config = CevakConfig(
        dim_threshold=0.25,
        ade_threshold=0.35,
        clean_mask_required=3,
        ade_window_size=9,
        ade_max=2.0,
    )
    monitor = CevakMonitor(config)

    # Calibration: clean outputs + action set A/B/C.
    for _ in range(10):
        for action in ["A", "B", "C"]:
            monitor.ade.record_action(action)
        monitor.record_calibration_vector(
            CevakVector(consistency=0.90, evidence=0.89, variance=0.88,
                        agreement=0.90, confidence=0.89)
        )
    monitor.set_baseline()

    # Runtime: completely different actions AND degraded outputs.
    for action in ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"]:
        monitor.ade.record_action(action)

    result = monitor.evaluate(
        CevakVector(consistency=0.45, evidence=0.50, variance=0.42,
                    agreement=0.48, confidence=0.47)
    )

    assert result.drift_result.mode != DriftMode.EVASION, (
        f"Honest degradation should NOT be EVASION, got {result.drift_result.mode.value}. "
        f"ADE={result.current.ade:.3f}, drifts={result.dim_drifts}"
    )
    # Should be OVERREACH or INCOHERENCE.
    assert result.drift_result.mode in {DriftMode.OVERREACH, DriftMode.INCOHERENCE}, (
        f"Expected OVERREACH or INCOHERENCE, got {result.drift_result.mode.value}"
    )


# ---------------------------------------------------------------------------
# Test 7: Stable deployment → NONE
# ---------------------------------------------------------------------------

def test_stable_deployment_none():
    """Same actions and same output quality should produce NONE mode."""
    config = CevakConfig(ade_window_size=9)
    monitor = CevakMonitor(config)

    for _ in range(10):
        for action in ["A", "B", "C"]:
            monitor.ade.record_action(action)
        monitor.record_calibration_vector(
            CevakVector(consistency=0.88, evidence=0.87, variance=0.86,
                        agreement=0.89, confidence=0.88)
        )
    monitor.set_baseline()

    for action in ["A", "B", "C", "A", "B", "C", "A", "B", "C"]:
        monitor.ade.record_action(action)

    result = monitor.evaluate(
        CevakVector(consistency=0.89, evidence=0.88, variance=0.86,
                    agreement=0.88, confidence=0.87)
    )

    assert result.drift_result.mode == DriftMode.NONE, (
        f"Expected NONE, got {result.drift_result.mode.value}. "
        f"ADE={result.current.ade:.3f}, drifts={result.dim_drifts}"
    )


# ---------------------------------------------------------------------------
# Test 8: set_baseline with no calibration → ValueError
# ---------------------------------------------------------------------------

def test_set_baseline_no_data_raises():
    ade = _make_ade()
    with pytest.raises(ValueError, match="no calibration data"):
        ade.set_baseline()


def test_monitor_set_baseline_no_vectors_raises():
    monitor = CevakMonitor()
    with pytest.raises(ValueError, match="No calibration vectors"):
        monitor.set_baseline()


# ---------------------------------------------------------------------------
# Test 9: compute_ade with no baseline → 0.0
# ---------------------------------------------------------------------------

def test_compute_ade_no_baseline_returns_zero():
    ade = _make_ade()
    ade.record_action("A")
    # Baseline not frozen yet.
    assert ade.compute_ade() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 10: compute_ade with no runtime observations → 0.0
# ---------------------------------------------------------------------------

def test_compute_ade_empty_window_returns_zero():
    ade = _make_ade()
    _calibrate(ade, ["A", "B", "C"])
    # No runtime observations added after baseline.
    assert ade.compute_ade() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11: Laplace smoothing prevents division by zero
# ---------------------------------------------------------------------------

def test_laplace_smoothing_no_zero_division():
    """Completely disjoint P and Q should not raise; KL should be finite."""
    ade = _make_ade(window_size=5)
    _calibrate(ade, ["A", "A", "A", "A", "A"])
    _add_runtime(ade, ["Z", "Z", "Z", "Z", "Z"])
    val = ade.compute_ade()
    assert math.isfinite(val), f"KL divergence should be finite, got {val}"
    assert val > 0.0


# ---------------------------------------------------------------------------
# Test 12: novel_action_ratio boundary cases
# ---------------------------------------------------------------------------

def test_novel_action_ratio_all_novel():
    ade = _make_ade(window_size=4)
    _calibrate(ade, ["A", "B"])
    _add_runtime(ade, ["X", "Y", "Z", "W"])
    ratio = ade.novel_action_ratio()
    assert ratio == pytest.approx(1.0)


def test_novel_action_ratio_none_novel():
    ade = _make_ade(window_size=4)
    _calibrate(ade, ["A", "B", "C"])
    _add_runtime(ade, ["A", "B", "C", "A"])
    ratio = ade.novel_action_ratio()
    assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 13: Window eviction — only most recent window_size observations count
# ---------------------------------------------------------------------------

def test_window_eviction():
    """Oldest observations should be evicted when window overflows."""
    ade = _make_ade(window_size=3)
    _calibrate(ade, ["A", "B", "C", "A", "B"])

    # First three runtime actions: all known.
    _add_runtime(ade, ["A", "B", "C"])
    assert not ade.has_novel_actions()

    # Add three more — these evict the first three. All novel.
    _add_runtime(ade, ["X", "Y", "Z"])
    # Window now contains [X, Y, Z] — all novel.
    assert ade.has_novel_actions()
    assert ade.novel_action_ratio() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 14: Baseline vocabulary
# ---------------------------------------------------------------------------

def test_baseline_vocabulary_matches_calibration():
    ade = _make_ade()
    _calibrate(ade, ["alpha", "beta", "gamma", "alpha", "beta"])
    vocab = ade.baseline_vocabulary()
    assert vocab == frozenset({"alpha", "beta", "gamma"})


def test_baseline_vocabulary_empty_before_set():
    ade = _make_ade()
    ade.record_action("alpha")
    # Not yet frozen.
    assert ade.baseline_vocabulary() == frozenset()


# ---------------------------------------------------------------------------
# Test 15: get_score clamping at 1.0
# ---------------------------------------------------------------------------

def test_get_score_clamped_at_one():
    """Extremely divergent distributions should not exceed score of 1.0."""
    ade = _make_ade(window_size=5, max_ade=0.1)  # Very low max_ade → easy to saturate.
    _calibrate(ade, ["A", "A", "A", "B", "C"])
    _add_runtime(ade, ["X", "Y", "Z", "W", "V"])
    score = ade.get_score()
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 16: constructor validation
# ---------------------------------------------------------------------------

def test_invalid_window_size():
    with pytest.raises(ValueError):
        ActionDistributionEntropy(window_size=0)


def test_invalid_max_ade():
    with pytest.raises(ValueError):
        ActionDistributionEntropy(max_ade=-1.0)


# ---------------------------------------------------------------------------
# Test 17: Multi-task-class isolation
# ---------------------------------------------------------------------------

def test_multi_task_class_isolation():
    """Drift in one task class must not affect the other.

    With a 6-element joint vocabulary and max_ade=2.0, completely disjoint
    3-action distributions produce KL ≈ 0.36 (score ≈ 0.18).  We therefore
    test the isolation property directly: the drifted class scores higher
    than the stable class, and the stable class stays near zero.  We do not
    assert a specific normalised threshold because that depends on vocabulary
    size and max_ade — both tested explicitly in test_massive_shift_above_threshold.
    """
    ade = _make_ade(window_size=6)

    # Calibrate class "search".
    for action in ["search_A", "search_B", "search_C"]:
        ade.record_action(action, "search")
    ade.set_baseline("search")

    # Calibrate class "write".
    for action in ["write_A", "write_B", "write_C"]:
        ade.record_action(action, "write")
    ade.set_baseline("write")

    # Runtime: "search" is stable; "write" is completely different.
    for action in ["search_A", "search_B", "search_C", "search_A", "search_B", "search_C"]:
        ade.record_action(action, "search")
    for action in ["X", "Y", "Z", "X", "Y", "Z"]:
        ade.record_action(action, "write")

    search_score = ade.get_score("search")
    write_score = ade.get_score("write")

    # Isolation: stable class must be near zero.
    assert search_score < 0.05, (
        f"Stable search class should have near-zero ADE, got {search_score}"
    )
    # Drift class must score strictly higher than stable class.
    assert write_score > search_score, (
        f"Drifted write ({write_score:.4f}) must exceed stable search ({search_score:.4f})"
    )
    # The raw KL for completely disjoint distributions with this vocabulary
    # should be > 0.1 (well above numerical noise).
    raw_write = ade.compute_ade("write")
    assert raw_write > 0.1, f"Expected raw KL > 0.1 for disjoint distributions, got {raw_write}"

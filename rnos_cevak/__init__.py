"""CEVAK: Cognitive drift detection for contained agents.

Five original dimensions (output layer):
  C — Consistency
  E — Evidence
  V — Variance
  A — Agreement
  K — Confidence

Sixth dimension (action layer):
  ADE — Action Distribution Entropy

The sixth dimension detects observation-aware evasion: an agent that
maintains clean output scores while shifting its actual tool-call policy.
"""

from rnos_cevak.core.ade import ActionDistributionEntropy
from rnos_cevak.core.cevak import CevakConfig, CevakMonitor, CevakResult, CevakVector
from rnos_cevak.core.drift import DriftMode, DriftResult

__all__ = [
    "ActionDistributionEntropy",
    "CevakConfig",
    "CevakMonitor",
    "CevakResult",
    "CevakVector",
    "DriftMode",
    "DriftResult",
]

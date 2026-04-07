<#
.SYNOPSIS
    Hybrid RNOS + Circuit Breaker compute gate.

.DESCRIPTION
    Evaluates two orthogonal instability signals on every step:

        RNOS  — structural entropy (retry depth, job fanout, target repetition, cost)
        CB    — failure density in a sliding window (circuit breaker pattern)

    Decision logic (safety-first, highest severity wins):

        entropy >= refuse_entropy  →  REFUSE  (trigger: RNOS)
        cb_rate >  cb_threshold    →  REFUSE  (trigger: CB)
        entropy >= degrade_entropy →  DEGRADE (trigger: RNOS)
        otherwise                  →  ALLOW   (trigger: NONE)

    RNOS is evaluated before CB so that structural explosion is caught before
    accumulated density evidence is required.  This is intentional: structural
    signals are available pre-execution; density signals require observed history.

    Exit codes
    ──────────
    0  ALLOW or DEGRADE (execution continues, possibly flagged)
    1  REFUSE           (pipeline halts immediately)

    Entropy components (mirrors experiments/ci_control/controllers.py)
    ──────────────────
    retry_score           = min(consecutiveFailures    * 0.8,  3.0)
    fanout_score          = min(jobsSpawned            * 0.4,  5.0)
    repeated_target_score = min(repeatedTargetFailures * 0.5,  2.0)
    cost_score            = min(computeMinutes         * 0.05, 1.0)

.PARAMETER Step
    1-indexed pipeline step number (display only).
#>
param(
    [int]$Step = 0
)

$statePath  = ".rnos/state.json"
$policyPath = "rnos-policy.json"

$state  = Get-Content $statePath  -Raw | ConvertFrom-Json
$policy = Get-Content $policyPath -Raw | ConvertFrom-Json

# ── RNOS Signal ──────────────────────────────────────────────────────────────
$retryScore  = [Math]::Min($state.consecutiveFailures    * 0.8,  3.0)
$fanoutScore = [Math]::Min($state.jobsSpawned            * 0.4,  5.0)
$repScore    = [Math]::Min($state.repeatedTargetFailures * 0.5,  2.0)
$costScore   = [Math]::Min($state.computeMinutes         * 0.05, 1.0)
$entropy     = $retryScore + $fanoutScore + $repScore + $costScore

# ── Circuit Breaker Signal ───────────────────────────────────────────────────
$history    = @($state.failureHistory)
$windowFill = $history.Count
$cbRate     = 0.0

if ($windowFill -gt 0) {
    $cbRate = ($history | Measure-Object -Sum).Sum / $windowFill
}

# CB only fires when window is full (requires cb_window_size observations)
$cbTripped = ($windowFill -ge [int]$policy.cb_window_size) -and ($cbRate -gt $policy.cb_threshold)

# ── Decision Logic ───────────────────────────────────────────────────────────
$decision = "ALLOW"
$trigger  = "NONE"

if ($entropy -ge $policy.refuse_entropy) {
    $decision = "REFUSE"
    $trigger  = "RNOS"
} elseif ($cbTripped) {
    $decision = "REFUSE"
    $trigger  = "CB"
} elseif ($entropy -ge $policy.degrade_entropy) {
    $decision = "DEGRADE"
    $trigger  = "RNOS"
}

# ── Formatted Output ─────────────────────────────────────────────────────────
$lastOutcome = if ($history.Count -gt 0) { $history[-1] } else { "-" }

Write-Host ""
Write-Host "---"
Write-Host "Step: $Step"
Write-Host "Failure: $lastOutcome"
Write-Host ""
Write-Host "RNOS Entropy: $([Math]::Round($entropy, 2))  (degrade>=$($policy.degrade_entropy)  refuse>=$($policy.refuse_entropy))"
Write-Host "  retry_score:           $([Math]::Round($retryScore,  2))  (consecutiveFailures=$($state.consecutiveFailures))"
Write-Host "  fanout_score:          $([Math]::Round($fanoutScore, 2))  (jobsSpawned=$($state.jobsSpawned))"
Write-Host "  repeated_target_score: $([Math]::Round($repScore,    2))  (repeatedTargetFailures=$($state.repeatedTargetFailures))"
Write-Host "  cost_score:            $([Math]::Round($costScore,   2))  (computeMinutes=$($state.computeMinutes))"
Write-Host ""
Write-Host "CB Failure Rate: $([Math]::Round($cbRate, 2))  (window: $windowFill/$([int]$policy.cb_window_size)  threshold: >$($policy.cb_threshold))"
Write-Host ""
Write-Host "Hybrid Gate Decision:"
Write-Host "  Decision: $decision"
Write-Host "  Trigger:  $trigger"
Write-Host "---"

# ── Exit ─────────────────────────────────────────────────────────────────────
if ($decision -eq "REFUSE") {
    Write-Host ""
    Write-Host "GATE CLOSED: Pipeline halted at step $Step."
    Write-Host "Trigger: $trigger"
    exit 1
}

exit 0

<#
.SYNOPSIS
    RNOS-only structural entropy gate for a pipeline step.

.DESCRIPTION
    Computes four entropy components from the current state and compares the
    total against the refuse/degrade thresholds in rnos-policy.json.

    Entropy components
    ──────────────────
    retry_score          = min(consecutiveFailures * 0.8,  3.0)
        Penalises failure streaks linearly.  Caps at 3.0 (4 consecutive failures).

    fanout_score         = min(jobsSpawned * 0.4,          5.0)
        Penalises parallel job expansion linearly.  Caps at 5.0 (13 spawned jobs).

    repeated_target_score= min(repeatedTargetFailures * 0.5, 2.0)
        Penalises persistent target failure.  Caps at 2.0 (4 target failures).

    cost_score           = min(computeMinutes * 0.05,      1.0)
        Penalises compute burn.  Caps at 1.0 (20 compute-minutes).

    Mirrors the formula in experiments/ci_control/controllers.py:RNOSCIController.

    Exit codes
    ──────────
    0  ALLOW or DEGRADE (execution continues)
    1  REFUSE           (pipeline halts)

.PARAMETER Step
    1-indexed step number for display.
#>
param(
    [int]$Step = 0
)

$statePath  = ".rnos/state.json"
$policyPath = "rnos-policy.json"

$state  = Get-Content $statePath  -Raw | ConvertFrom-Json
$policy = Get-Content $policyPath -Raw | ConvertFrom-Json

# ── Entropy components ───────────────────────────────────────────────────────
$retryScore   = [Math]::Min($state.consecutiveFailures    * 0.8,  3.0)
$fanoutScore  = [Math]::Min($state.jobsSpawned            * 0.4,  5.0)
$repScore     = [Math]::Min($state.repeatedTargetFailures * 0.5,  2.0)
$costScore    = [Math]::Min($state.computeMinutes         * 0.05, 1.0)
$entropy      = $retryScore + $fanoutScore + $repScore + $costScore

# ── Decision ─────────────────────────────────────────────────────────────────
if ($entropy -ge $policy.refuse_entropy) {
    $decision = "REFUSE"
} elseif ($entropy -ge $policy.degrade_entropy) {
    $decision = "DEGRADE"
} else {
    $decision = "ALLOW"
}

# ── Output ───────────────────────────────────────────────────────────────────
Write-Host "RNOS Gate  |  Step: $Step  |  Entropy: $([Math]::Round($entropy, 2))  |  Decision: $decision"

if ($decision -eq "REFUSE") { exit 1 }
exit 0

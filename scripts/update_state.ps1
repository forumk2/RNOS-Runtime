<#
.SYNOPSIS
    Updates (or initialises) the RNOS state file after a pipeline step outcome.

.DESCRIPTION
    State fields and update rules
    ─────────────────────────────
    jobsSpawned            — cumulative jobs spawned.
                             Failure: +2 (retry jobs spawned)
                             Success: +1 (normal job completes)

    consecutiveFailures    — streak of consecutive failures.
                             Failure: +1
                             Success: reset to 0

    repeatedTargetFailures — cumulative count of target failures.
                             Failure: +1
                             Success: unchanged

    computeMinutes         — cumulative compute cost.
                             Every step: +2 minutes

    failureHistory         — rolling window of last N outcomes (0=success, 1=failure).
                             Appended each step, trimmed to cb_window_size from
                             rnos-policy.json (default 5).

.PARAMETER Failure
    0 = success, 1 = failure.  Required unless -Init is set.

.PARAMETER Step
    1-indexed step number (used for display only).

.PARAMETER Init
    When set, writes the empty initial state and exits.
    No other parameters are required.
#>
param(
    [int]   $Failure = 0,
    [int]   $Step    = 0,
    [switch]$Init
)

$statePath  = ".rnos/state.json"
$policyPath = "rnos-policy.json"

# ── Initialise ──────────────────────────────────────────────────────────────
if ($Init) {
    $empty = [ordered]@{
        jobsSpawned            = 0
        consecutiveFailures    = 0
        repeatedTargetFailures = 0
        computeMinutes         = 0
        failureHistory         = @()
    }
    $empty | ConvertTo-Json | Set-Content $statePath
    Write-Host "State initialised: $statePath"
    return
}

# ── Load state and policy ────────────────────────────────────────────────────
$raw    = Get-Content $statePath  -Raw | ConvertFrom-Json
$policy = Get-Content $policyPath -Raw | ConvertFrom-Json
$windowSize = [int]$policy.cb_window_size

# Read fields explicitly to avoid PSCustomObject mutation issues on PS 5.1
$consec  = [int]$raw.consecutiveFailures
$spawned = [int]$raw.jobsSpawned
$rep     = [int]$raw.repeatedTargetFailures
$compute = [int]$raw.computeMinutes

# Rebuild the failureHistory list safely (handles PS 5.1 array deserialization)
$history = [System.Collections.Generic.List[int]]::new()
foreach ($item in @($raw.failureHistory)) {
    if ($null -ne $item) { $history.Add([int]$item) }
}

# ── Structural update ────────────────────────────────────────────────────────
if ($Failure -eq 1) {
    $consec  += 1
    $spawned += 2    # each failure spawns 2 retry jobs
    $rep     += 1
} else {
    $consec   = 0
    $spawned += 1    # normal job completes
}
$compute += 2

# ── Rolling failure window ───────────────────────────────────────────────────
$history.Add($Failure)
while ($history.Count -gt $windowSize) { $history.RemoveAt(0) }

# ── Persist (explicit ordered hashtable avoids PSCustomObject quirks) ────────
$newState = [ordered]@{
    jobsSpawned            = $spawned
    consecutiveFailures    = $consec
    repeatedTargetFailures = $rep
    computeMinutes         = $compute
    failureHistory         = $history.ToArray()
}
$newState | ConvertTo-Json -Depth 5 | Set-Content $statePath

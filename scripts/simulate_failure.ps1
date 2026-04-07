<#
.SYNOPSIS
    Returns the deterministic failure outcome (0=success, 1=failure) for a pipeline step.

.DESCRIPTION
    Implements a fixed two-phase failure pattern:

        Phase 1 (steps 1-5): burst failures
            [0, 1, 1, 1, 1]
            Three consecutive failures build structural entropy until RNOS fires.

        Phase 2 (steps 6-10): distributed failures
            [0, 1, 0, 1, 0]
            Alternating failures.  Without RNOS, this density would fill the CB
            window and trigger a circuit-breaker REFUSE at step 6.

    This sequence is designed to demonstrate both instability geometries.
    Only one output value is written (0 or 1) so callers can capture it cleanly.

.PARAMETER Step
    1-indexed pipeline step number.

.OUTPUTS
    Writes a single integer (0 or 1) to stdout.
#>
param(
    [Parameter(Mandatory)][int]$Step
)

$sequence = @(0, 1, 1, 1, 1, 0, 1, 0, 1, 0)

$index = $Step - 1
if ($index -lt 0 -or $index -ge $sequence.Length) {
    Write-Output 0
} else {
    Write-Output $sequence[$index]
}

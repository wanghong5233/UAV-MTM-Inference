#!/usr/bin/env pwsh
# train_all_quick.ps1
# Launch all four model training jobs in parallel.
# stdout/stderr of each job is redirected to logs/proc_logs/.
# Monitor all runs with a single TensorBoard instance.
#
# Usage (run from project root):
#   .\scripts\train_all_quick.ps1
#   .\scripts\train_all_quick.ps1 -Tag v2 -Device cpu

param(
    [string]$Tag    = "quick",
    [string]$Device = "cpu",
    [double[]]$Weights = @(),
    [int]$Seed = -1
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$Experiments = @(
    [pscustomobject]@{
        Config = "configs/experiments/main_gnn_ppo_tmc_stable.yaml"
        RunTag = "mtan_$Tag"
    }
    [pscustomobject]@{
        Config = "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml"
        RunTag = "split_$Tag"
    }
    [pscustomobject]@{
        Config = "configs/experiments/main_gnn_ppo_tmc_stable_dense.yaml"
        RunTag = "dense_$Tag"
    }
    [pscustomobject]@{
        Config = "configs/experiments/main_gnn_ppo_tmc_stable_cross.yaml"
        RunTag = "cross_$Tag"
    }
)

$ProcLogDir = Join-Path $Root "logs\proc_logs"
New-Item -ItemType Directory -Force -Path $ProcLogDir | Out-Null

Write-Host ""
Write-Host "=========================================================="
Write-Host "  UAV-MTL  4 experiments  tag=$Tag  device=$Device"
Write-Host "=========================================================="
Write-Host ""

$Procs = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

foreach ($exp in $Experiments) {
    $stdoutFile = Join-Path $ProcLogDir "$($exp.RunTag)_stdout.log"
    $stderrFile = Join-Path $ProcLogDir "$($exp.RunTag)_stderr.log"

    $argStr = "scripts/train.py" `
            + " --config $($exp.Config)" `
            + " --device $Device" `
            + " --console_mode tqdm" `
            + " --disable_eval" `
            + " --disable_save" `
            + " --run_tag $($exp.RunTag)"
    if ($Weights.Count -eq 2) {
        $argStr += " --weights $($Weights[0]) $($Weights[1])"
    }
    if ($Seed -ge 0) {
        $argStr += " --seed $Seed"
    }

    $proc = Start-Process python `
                -ArgumentList  $argStr `
                -WorkingDirectory $Root `
                -RedirectStandardOutput $stdoutFile `
                -RedirectStandardError  $stderrFile `
                -NoNewWindow `
                -PassThru

    $Procs.Add($proc)
    # Raise CPU priority so training is not starved by other apps
    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
        $priTag = "AboveNormal"
    } catch {
        $priTag = "default (set failed)"
    }
    Write-Host ("  [PID {0,6}]  {1,-20}  priority={2}  log -> {3}" -f $proc.Id, $exp.RunTag, $priTag, $stdoutFile)
}

Write-Host ""
Write-Host "----------------------------------------------------------"
Write-Host "  TensorBoard (single port for all runs):"
Write-Host "    tensorboard --logdir logs/training --port 6006"
Write-Host ""
Write-Host "  Follow a single run live (example):"
Write-Host "    Get-Content logs\proc_logs\split_${Tag}_stdout.log -Wait"
Write-Host "----------------------------------------------------------"
Write-Host ""
Write-Host "  All processes are running in background."
Write-Host "  Closing this window does NOT kill the training jobs."
Write-Host "  Waiting for all to finish (status every 30 s)..."
Write-Host ""

while ($true) {
    $alive = @($Procs | Where-Object { -not $_.HasExited })
    if ($alive.Count -eq 0) { break }
    Write-Host ("  [{0}]  {1} process(es) still running..." -f (Get-Date -Format "HH:mm:ss"), $alive.Count)
    Start-Sleep -Seconds 30
}

Write-Host ""
$failed = @($Procs | Where-Object { $_.ExitCode -ne 0 }).Count
if ($failed -gt 0) {
    Write-Host "  WARNING: $failed process(es) exited with non-zero code."
    Write-Host "  Check *_stderr.log files in: $ProcLogDir"
} else {
    Write-Host "  All experiments finished successfully."
}
Write-Host ""

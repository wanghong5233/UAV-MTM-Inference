# Rebaseline driver: only deterministic baselines (Local-Only, Single-Split)
# across all six experiment groups (E2/E3/E4/E5/E6).
#
# Fix: pin task popularity (fixed Dirichlet -> fixed vector) and pin source
# UAV capability (compute_max == compute_min, bandwidth_max == bandwidth_min)
# so that deterministic baselines are truly deterministic at evaluation time.
# This does NOT touch yaml configs and does NOT re-run learning algorithms;
# learning JSONs are left untouched.

param(
    [int]$EvalEpisodes = 100,
    [int]$Seed = 789,
    [double[]]$Weights = @(0.1, 0.9),
    [string]$Python = "D:\Anaconda\envs\pytorch\python.exe",
    [int]$Parallel = 6,
    [string]$RepoRoot = $null
)

$ErrorActionPreference = "Stop"

if ($RepoRoot) {
    Set-Location -LiteralPath $RepoRoot
}

if (-not (Test-Path $Python)) {
    Write-Host "Python interpreter not found: $Python" -ForegroundColor Red
    exit 2
}
Write-Host "CWD          : $(Get-Location)"
Write-Host "Using python : $Python"
Write-Host "EvalEpisodes : $EvalEpisodes  Seed : $Seed  Parallel : $Parallel"

$wtag = "w19"

# Eval-time overrides: lock down env stochasticity for deterministic baselines.
$detOverrides = @(
    "--set", "task.popularity_mode=fixed",
    "--set", "task.popularity=[0.5,0.5,0.5]",
    "--set", "uav.source_profile.compute_max=1.0e11",
    "--set", "uav.source_profile.bandwidth_max=3.0e6"
)

$models = @(
    @{ slug = "split"; cfg_lo = "configs/experiments/baseline_local_only_tmc_stable_split.yaml";   cfg_ss = "configs/experiments/baseline_single_split_tmc_stable_split.yaml" },
    @{ slug = "mtan";  cfg_lo = "configs/experiments/baseline_local_only_tmc_stable.yaml";         cfg_ss = "configs/experiments/baseline_single_split_tmc_stable.yaml" },
    @{ slug = "dense"; cfg_lo = "configs/experiments/baseline_local_only_tmc_stable_dense.yaml";   cfg_ss = "configs/experiments/baseline_single_split_tmc_stable_dense.yaml" },
    @{ slug = "cross"; cfg_lo = "configs/experiments/baseline_local_only_tmc_stable_cross.yaml";   cfg_ss = "configs/experiments/baseline_single_split_tmc_stable_cross.yaml" }
)

# MTAN-only sweep configs for E3..E6
$mtan_lo = "configs/experiments/baseline_local_only_tmc_stable.yaml"
$mtan_ss = "configs/experiments/baseline_single_split_tmc_stable.yaml"

$arrival_rates = @(1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
$swarm_sizes   = @(4, 6, 8, 10, 12, 14, 16, 18, 20, 24)
$areas         = @(500, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000)
$taskdens      = @(1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)

function Format-AreaTag([double]$x) { return "area$([int]$x)" }
function Format-LambdaTag([double]$x) {
    $i = [int][math]::Floor($x)
    $f = [int][math]::Round(($x - $i) * 10)
    return "lambda${i}_${f}"
}
function Format-UavTag([int]$x) { return "uav$x" }
function Format-TdTag([double]$x) {
    $i = [int][math]::Floor($x)
    $f = [int][math]::Round(($x - $i) * 100)
    if ($f -ge 10 -and ($f % 10 -eq 0)) { $f = [int]($f / 10) }
    return "td${i}_${f}"
}

# Collect all jobs first, then dispatch in parallel batches
$jobs = New-Object System.Collections.ArrayList

function Add-Job {
    param([string]$cfg, [string]$outFile, [string[]]$flagArgs, [string]$displayTag)
    [void]$jobs.Add([pscustomobject]@{
        Cfg         = $cfg
        Out         = $outFile
        Flags       = $flagArgs
        Tag         = $displayTag
    })
}

# E2
$e2dir = "results/eval/e2_cross_model_w19"
if (-not (Test-Path $e2dir)) { New-Item -ItemType Directory -Force -Path $e2dir | Out-Null }
foreach ($m in $models) {
    $slug = $m.slug
    Add-Job -cfg $m.cfg_lo -outFile "$e2dir/${slug}_${wtag}_local_only.json"   -flagArgs @() -displayTag "E2 $slug local_only"
    Add-Job -cfg $m.cfg_ss -outFile "$e2dir/${slug}_${wtag}_single_split.json" -flagArgs @() -displayTag "E2 $slug single_split"
}

# E3
$e3dir = "results/eval/e3_arrival_w19"
if (-not (Test-Path $e3dir)) { New-Item -ItemType Directory -Force -Path $e3dir | Out-Null }
foreach ($lam in $arrival_rates) {
    $tag = Format-LambdaTag $lam
    Add-Job -cfg $mtan_lo -outFile "$e3dir/mtan_${wtag}_local_only_${tag}.json"   -flagArgs @("--arrival_rate", "$lam") -displayTag "E3 lo  $tag"
    Add-Job -cfg $mtan_ss -outFile "$e3dir/mtan_${wtag}_single_split_${tag}.json" -flagArgs @("--arrival_rate", "$lam") -displayTag "E3 ss  $tag"
}

# E4
$e4dir = "results/eval/e4_swarm_w19"
if (-not (Test-Path $e4dir)) { New-Item -ItemType Directory -Force -Path $e4dir | Out-Null }
foreach ($u in $swarm_sizes) {
    $tag = Format-UavTag $u
    Add-Job -cfg $mtan_lo -outFile "$e4dir/mtan_${wtag}_local_only_${tag}.json"   -flagArgs @("--num_uavs", "$u") -displayTag "E4 lo  $tag"
    Add-Job -cfg $mtan_ss -outFile "$e4dir/mtan_${wtag}_single_split_${tag}.json" -flagArgs @("--num_uavs", "$u") -displayTag "E4 ss  $tag"
}

# E5
$e5dir = "results/eval/e5_area_w19"
if (-not (Test-Path $e5dir)) { New-Item -ItemType Directory -Force -Path $e5dir | Out-Null }
foreach ($a in $areas) {
    $tag = Format-AreaTag $a
    Add-Job -cfg $mtan_lo -outFile "$e5dir/mtan_${wtag}_local_only_${tag}.json"   -flagArgs @("--area_size", "$a", "$a") -displayTag "E5 lo  $tag"
    Add-Job -cfg $mtan_ss -outFile "$e5dir/mtan_${wtag}_single_split_${tag}.json" -flagArgs @("--area_size", "$a", "$a") -displayTag "E5 ss  $tag"
}

# E6
$e6dir = "results/eval/e6_taskdens_w19"
if (-not (Test-Path $e6dir)) { New-Item -ItemType Directory -Force -Path $e6dir | Out-Null }
foreach ($td in $taskdens) {
    $tag = Format-TdTag $td
    Add-Job -cfg $mtan_lo -outFile "$e6dir/mtan_${wtag}_local_only_${tag}.json"   -flagArgs @("--avg_tasks_per_request", "$td") -displayTag "E6 lo  $tag"
    Add-Job -cfg $mtan_ss -outFile "$e6dir/mtan_${wtag}_single_split_${tag}.json" -flagArgs @("--avg_tasks_per_request", "$td") -displayTag "E6 ss  $tag"
}

Write-Host ""
Write-Host "Total jobs queued: $($jobs.Count)"
Write-Host ""

# Parallel dispatch using Start-Process
$logRoot = "logs/proc_logs/_rebaseline"
if (-not (Test-Path $logRoot)) { New-Item -ItemType Directory -Force -Path $logRoot | Out-Null }

$active = @{}
$success = 0
$failedList = New-Object System.Collections.ArrayList
$queueIdx = 0

function Pump-Active {
    foreach ($k in @($script:active.Keys)) {
        $proc = $script:active[$k].Proc
        if ($proc.HasExited) {
            # Force ExitCode to be populated; with redirected I/O Start-Process
            # sometimes returns before ExitCode is materialised.
            try { $proc.WaitForExit() } catch {}
            $job = $script:active[$k].Job
            $log = $script:active[$k].Log
            $rc  = $proc.ExitCode
            # Treat as success if either exit code is 0, or if the
            # output JSON exists (covers the edge case where ExitCode
            # is null with redirected I/O on Windows).
            $ok = ($rc -eq 0) -or (Test-Path $job.Out)
            if (-not $ok) {
                Write-Host ("  [FAIL  ] {0}  (exit {1})" -f $job.Tag, $rc) -ForegroundColor Red
                $errFile = "$log.err"
                if (Test-Path $errFile) {
                    $errTail = (Get-Content $errFile -Tail 5) -join "`n          "
                    Write-Host "          $errTail" -ForegroundColor DarkRed
                }
                [void]$script:failedList.Add([pscustomobject]@{
                    Tag = $job.Tag; Cfg = $job.Cfg; Out = $job.Out; Log = $log
                })
            } else {
                Write-Host ("  [OK    ] {0}" -f $job.Tag) -ForegroundColor Green
                $script:success++
            }
            $script:active.Remove($k) | Out-Null
        }
    }
}

while ($queueIdx -lt $jobs.Count -or $active.Count -gt 0) {
    Pump-Active
    while ($active.Count -lt $Parallel -and $queueIdx -lt $jobs.Count) {
        $job = $jobs[$queueIdx]
        $queueIdx++

        $logFile = Join-Path $logRoot ("rebaseline_{0}.log" -f ($queueIdx))
        $argList = @(
            "scripts/evaluate.py",
            "--config", $job.Cfg,
            "--weights", "$($Weights[0])", "$($Weights[1])",
            "--num_episodes", "$EvalEpisodes",
            "--cpu_threads", "2",
            "--output", $job.Out,
            "--seed", "$Seed"
        ) + @($job.Flags) + @($detOverrides)

        Write-Host "  [START ] $($job.Tag)" -ForegroundColor Cyan
        $proc = Start-Process -FilePath $Python -ArgumentList $argList `
                              -RedirectStandardOutput $logFile `
                              -RedirectStandardError "$logFile.err" `
                              -PassThru -NoNewWindow

        $active[$proc.Id] = @{ Proc = $proc; Job = $job; Log = $logFile }
    }
    Start-Sleep -Milliseconds 800
}

Pump-Active

Write-Host ""
Write-Host "=========================================================="
Write-Host "  Rebaseline summary"
Write-Host "=========================================================="
Write-Host "  total = $($jobs.Count)"
Write-Host "  ok    = $success"
Write-Host "  fail  = $($failedList.Count)"

if ($failedList.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures:" -ForegroundColor Red
    foreach ($f in $failedList) {
        Write-Host "  - $($f.Tag)  cfg=$($f.Cfg)  out=$($f.Out)  log=$($f.Log)"
    }
    exit 1
}
exit 0

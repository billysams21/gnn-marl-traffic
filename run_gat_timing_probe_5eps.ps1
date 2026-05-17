$ErrorActionPreference = "Stop"
Set-Location "D:\ITB\TA\gnn-marl-traffic"

$py = ".\helm\Scripts\python.exe"

# Tiny timing probe after the phase-duration fix.
# Agent: GAT+DQN only. Goal: compare nearby min_green values quickly.
$scenario = "grid_3x3_pkji_m1"
$agent = "gat_dqn"
$episodes = 5
$device = "cuda"
$yellowTime = 2
$epsilonDecay = 0.99
$evalInterval = 5
$evalEpisodes = 1
$seed = 42

$minGreens = @(9, 11, 12, 8, 7)

# Keep this light; each run launches one SUMO process.
$maxParallel = 2

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "logs\pkji_m1_timing_probe_gat_5eps_$ts"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$procs = @()
foreach ($minGreen in $minGreens) {
    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $maxParallel) {
        Start-Sleep -Seconds 5
    }

    $suffix = "pkji_m1_timing_probe_gat_dqn_cuda_y$($yellowTime)_g$($minGreen)_eps099_s$seed"
    $args = @(
        "experiments/train.py",
        "--scenario", $scenario,
        "--agent", $agent,
        "--episodes", $episodes,
        "--seed", $seed,
        "--device", $device,
        "--yellow-time", $yellowTime,
        "--min-green", $minGreen,
        "--epsilon-decay", $epsilonDecay,
        "--eval-interval", $evalInterval,
        "--eval-episodes", $evalEpisodes,
        "--exp-suffix", $suffix
    )

    $out = Join-Path $runDir "train_$suffix.out.log"
    $err = Join-Path $runDir "train_$suffix.err.log"

    $p = Start-Process -FilePath $py -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $procs += $p
    Write-Host "Started y$($yellowTime)_g$minGreen seed=$seed PID=$($p.Id)"
}

Write-Host "`nWaiting all timing probe runs..."
$procs | ForEach-Object { Wait-Process -Id $_.Id }
Write-Host "All timing probe runs done. Logs: $runDir"

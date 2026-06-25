$ErrorActionPreference = "Stop"
Set-Location "D:\ITB\TA\gnn-marl-traffic"

$py = ".\helm\Scripts\python.exe"

# Level 1 (pretrain): stable curriculum before no-teleport fine-tuning.
$scenarios = @("grid_3x3_pkji_m1")
$episodes = 100
$device = "cuda"
$yellowTime = 2
$minGreen = 12
$timeToTeleport = 300
$epsilonDecay = 0.95
$evalInterval = 10
$evalEpisodes = 1
$seeds = @(10, 101, 1101, 1011, 1111)

$maxParallel = 5

$configs = @(
    @{agent="gat_dqn"; suffix="pkji_level1_gat_dqn_eps095_tt300_100ep"},
    @{agent="independent_dqn"; suffix="pkji_level1_independent_dqn_eps095_tt300_100ep"}
)

# Add entries here if you want to skip completed pairs (scenario + agent).
$skipRuns = @(
    # @{scenario="grid_3x3_pkji_m1"; agent="gat_dqn"}
)

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "logs\pkji_level1_launch_$ts"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$procs = @()
foreach ($scenario in $scenarios) {
foreach ($config in $configs) {
foreach ($seed in $seeds) {
    $isSkipped = @($skipRuns | Where-Object {
        $_.scenario -eq $scenario -and $_.agent -eq $config.agent
    }).Count -gt 0
    if ($isSkipped) {
        Write-Host "Skipping completed run: $scenario $($config.agent) seed=$seed"
        continue
    }

    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $maxParallel) {
        Start-Sleep -Seconds 10
    }

    $suffix = "$($config.suffix)_$($scenario)_s$seed"
    $args = @(
        "experiments/train.py",
        "--scenario", $scenario,
        "--agent", $config.agent,
        "--episodes", $episodes,
        "--seed", $seed,
        "--device", $device,
        "--yellow-time", $yellowTime,
        "--min-green", $minGreen,
        "--time-to-teleport", $timeToTeleport,
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
    Write-Host "Started $scenario $($config.agent) seed=$seed PID=$($p.Id)"
}
}
}

Write-Host "`nWaiting all Level 1 runs..."
$procs | ForEach-Object {
    if (Get-Process -Id $_.Id -ErrorAction SilentlyContinue) {
        Wait-Process -Id $_.Id -ErrorAction SilentlyContinue
    }
}
Write-Host "All Level 1 runs done. Logs: $runDir"

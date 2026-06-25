$ErrorActionPreference = "Stop"
Set-Location "D:\ITB\TA\gnn-marl-traffic"

$py = ".\helm\Scripts\python.exe"

# Level 2a-Refine: continue from episode 201 to 300.
# Keep teleport at 600s, freeze epsilon at 0.05 (greedy/exploitation focus), and use very low LR.
$scenario = "grid_3x3_pkji_m1"
$episodes = 300
$device = "cuda"
$yellowTime = 2
$minGreen = 12
$timeToTeleport = 600
$epsilonStart = 0.15
$epsilonEnd = 0.05
$epsilonDecay = 0.99
$evalInterval = 10
$evalEpisodes = 5
$learningRate = "5e-5"
$auxWeight = "0.05"
$seeds = @(10, 101, 1101, 1011, 1111)

$maxParallel = 5

$configs = @(
    @{agent="gat_dqn"; suffix="pkji_level2a_refine_gat_dqn_tt600_eps005_lr5e5"},
    @{agent="independent_dqn"; suffix="pkji_level2a_refine_independent_dqn_tt600_eps005_lr5e5"}
)

$resumeBySeed = @{
    10 = @{
        gat_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s10\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s10"
        }
        independent_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075322_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s10\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075322_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s10"
        }
    }
    101 = @{
        gat_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s101\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s101"
        }
        independent_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075333_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s101\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075333_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s101"
        }
    }
    1101 = @{
        gat_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1101\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1101"
        }
        independent_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075342_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1101\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075342_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1101"
        }
    }
    1011 = @{
        gat_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1011\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1011"
        }
        independent_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075352_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1011\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075352_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1011"
        }
    }
    1111 = @{
        gat_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1111\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\gat_dqn_grid_3x3_pkji_m1_20260519_044453_pkji_level1_gat_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1111"
        }
        independent_dqn = @{
            checkpoint = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075412_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1111\checkpoint_ep200.pt"
            logDir = "D:\ITB\TA\gnn-marl-traffic\logs\independent_dqn_grid_3x3_pkji_m1_20260519_075412_pkji_level1_independent_dqn_eps095_tt300_100ep_grid_3x3_pkji_m1_s1111"
        }
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "logs\pkji_level2a_refine_launch_$ts"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$procs = @()
foreach ($config in $configs) {
foreach ($seed in $seeds) {
    $resume = $resumeBySeed[$seed][$config.agent]
    $resumeCheckpoint = $resume.checkpoint
    $resumeLogDir = $resume.logDir

    if (-not (Test-Path -LiteralPath $resumeCheckpoint)) {
        throw "Missing resume checkpoint for $($config.agent) seed=${seed}: $resumeCheckpoint"
    }

    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $maxParallel) {
        Start-Sleep -Seconds 10
    }

    $suffix = "$($config.suffix)_s$seed"
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
        "--epsilon-start", $epsilonStart,
        "--epsilon-end", $epsilonEnd,
        "--epsilon-decay", $epsilonDecay,
        "--eval-interval", $evalInterval,
        "--eval-episodes", $evalEpisodes,
        "--resume-checkpoint", $resumeCheckpoint,
        "--resume-log-dir", $resumeLogDir,
        "--exp-suffix", $suffix,
        "--lr", $learningRate,
        "--aux-weight", $auxWeight
    )

    $out = Join-Path $runDir "train_$suffix.out.log"
    $err = Join-Path $runDir "train_$suffix.err.log"

    $p = Start-Process -FilePath $py -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $procs += $p
    Write-Host "Started LEVEL2A-REFINE (tt=600, eps=0.05 flat, lr=$learningRate, eval=$evalEpisodes) $scenario $($config.agent) seed=$seed PID=$($p.Id)"
}
}

Write-Host "`nWaiting all Level 2a-Refine runs..."
$procs | ForEach-Object { Wait-Process -Id $_.Id }
Write-Host "All Level 2a-Refine runs done. Logs: $runDir"

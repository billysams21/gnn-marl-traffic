$ErrorActionPreference = "Stop"
Set-Location "D:\ITB\TA\gnn-marl-traffic"

$py = ".\helm\Scripts\python.exe"
$common = @("experiments/train.py","--scenario","grid_3x3","--agent","gat_dqn","--episodes","30")
$seeds = @(42, 43, 44)
$maxParallel = 3

$configs = @(
    @{suffix="cuda_y2_g4"; device="cuda"; yellow=2; green=4},
    @{suffix="cuda_y2_g7"; device="cuda"; yellow=2; green=7},
    @{suffix="cuda_y2_g9"; device="cuda"; yellow=2; green=9}
)

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "logs\parallel_launch_$ts"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$procs = @()
foreach ($seed in $seeds) {
foreach ($c in $configs) {
    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $maxParallel) {
        Start-Sleep -Seconds 5
    }

    $suffix = "$($c.suffix)_s$seed"
    $args = @(
        $common +
        @("--seed",$seed,"--device",$c.device,"--yellow-time",$c.yellow,"--min-green",$c.green,"--exp-suffix",$suffix)
    )
    $out = Join-Path $runDir "train_$suffix.out.log"
    $err = Join-Path $runDir "train_$suffix.err.log"

    $p = Start-Process -FilePath $py -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $procs += $p
    Write-Host "Started $suffix PID=$($p.Id)"
}
}

Write-Host "`nWaiting all runs..."
$procs | ForEach-Object { Wait-Process -Id $_.Id }
Write-Host "All runs done. Logs: $runDir"

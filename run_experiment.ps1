param(
    [int]$NumPis = 2,
    [int]$Rounds = 3,
    [int]$Port = 8080,
    [string]$Host = "0.0.0.0",
    [float]$Threshold = 1.38
)

# =============================================================================
# run_experiment.ps1 – SpikeFed-IoMT PC-side Experiment Orchestrator
# =============================================================================
# This script:
#   1. Finds your active network IP (Wi-Fi / Ethernet)
#   2. Starts the Flower FL server in a new window
#   3. Prints the exact SSH commands to run on each Pi
#
# Usage:
#   .\run_experiment.ps1 -NumPis 2 -Rounds 3
#   .\run_experiment.ps1 -NumPis 1 -Rounds 3 -Threshold 1.38
# =============================================================================

$ErrorActionPreference = "Stop"

# ── find local IP ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  SpikeFed-IoMT  –  Experiment Orchestrator" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Prefer the Ethernet adapter shared with the Pi; fall back to first non-loopback IPv4
$PcIp = (
    Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch "^(127\.|169\.254\.)" } |
    Sort-Object -Property InterfaceIndex |
    Select-Object -First 1
).IPAddress

if (-not $PcIp) {
    Write-Warning "Could not auto-detect PC IP. Run 'ipconfig' and set it manually in the Pi commands below."
    $PcIp = "<YOUR_PC_IP>"
}

Write-Host "Detected PC IP : $PcIp" -ForegroundColor Green
Write-Host "FL Server will listen on ${Host}:${Port}" -ForegroundColor Green
Write-Host "Rounds         : $Rounds" -ForegroundColor Green
Write-Host "Min clients    : $NumPis" -ForegroundColor Green
Write-Host ""

# ── start server in a new PowerShell window ───────────────────────────────
$serverCmd = "python server.py --host $Host --port $Port --rounds $Rounds --min_clients $NumPis"
Write-Host "Starting FL server in a new window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; $serverCmd"
Write-Host "  Server window opened." -ForegroundColor Green
Write-Host ""

# ── print Pi commands ─────────────────────────────────────────────────────
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Run these commands on each Pi after the server" -ForegroundColor Cyan
Write-Host "  prints 'Waiting for incoming connections...'  " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

for ($i = 1; $i -le $NumPis; $i++) {
    Write-Host "---- Pi $i -----------------------------------------------" -ForegroundColor Yellow
    Write-Host "  ssh pi@<PI_${i}_IP>" -ForegroundColor White
    Write-Host "  cd ~/spikefed-iomt" -ForegroundColor White
    Write-Host "  bash pi_setup/start_client.sh $PcIp $i $Threshold" -ForegroundColor White
    Write-Host ""
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Results will be saved to: results/round_results.json" -ForegroundColor White
Write-Host "  Server window will print per-round savings %" -ForegroundColor White
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tip: Record evidence per RESULTS.md for your project review." -ForegroundColor DarkGray
Write-Host ""

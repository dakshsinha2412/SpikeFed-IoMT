param(
    [Parameter(Mandatory = $true)]
    [string]$PiHost,

    [string]$PiUser = "pi",

    [string]$RemoteDirectory = "~/spikefed-iomt",

    # Optional: path to a local data shard to copy onto the Pi
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"

# Complete project files for full deployment and execution on the Pi
$projectFiles = @(
    "client.py",
    "model.py",
    "metrics.py",
    "utils.py",
    "server.py",
    "main.py",
    "federated.py",
    "dataset.py",
    "visualization.py",
    "evaluate.py",
    "tests.py",
    "self_check.py",
    "benchmark.py",
    "requirements_pi.txt"
)
if (Test-Path "medical_classwise_dataset.zip") {
    $projectFiles += "medical_classwise_dataset.zip"
}
# Pi setup / helper scripts
$setupFiles = @(
    "pi_setup\pi_install.sh",
    "pi_setup\start_client.sh"
)

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  SpikeFed-IoMT  ->  Deploying to $PiUser@$PiHost" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verify SSH access ───────────────────────────────────────────────────
Write-Host "[1/5] Checking SSH access..." -ForegroundColor Yellow
ssh "$PiUser@$PiHost" "mkdir -p $RemoteDirectory/pi_setup"
if ($LASTEXITCODE -ne 0) {
    throw "SSH connection failed.`nMake sure:`n  - SSH is enabled on the Pi`n  - The Pi and PC are on the same network`n  - Hostname/IP '$PiHost' is reachable (try: ping $PiHost)"
}
Write-Host "      SSH OK" -ForegroundColor Green

# ── 2. Copy complete project files ─────────────────────────────────────────
Write-Host "[2/5] Copying complete project files & dataset..." -ForegroundColor Yellow
ssh "$PiUser@$PiHost" "mkdir -p $RemoteDirectory/checkpoints $RemoteDirectory/results"
scp $projectFiles "${PiUser}@${PiHost}:$RemoteDirectory/"
if ($LASTEXITCODE -ne 0) { throw "SCP failed for project files." }
Write-Host "      Project files and dataset copied" -ForegroundColor Green

# ── 3. Copy setup / helper scripts ────────────────────────────────────────
Write-Host "[3/5] Copying Pi setup scripts..." -ForegroundColor Yellow
scp $setupFiles "${PiUser}@${PiHost}:$RemoteDirectory/pi_setup/"
if ($LASTEXITCODE -ne 0) { throw "SCP failed for setup scripts." }
# Make shell scripts executable and unpack dataset if present
ssh "$PiUser@$PiHost" "chmod +x $RemoteDirectory/pi_setup/*.sh; if [ -f $RemoteDirectory/medical_classwise_dataset.zip ]; then python3 -m zipfile -e $RemoteDirectory/medical_classwise_dataset.zip $RemoteDirectory/medical_dataset; fi"
Write-Host "      Setup scripts configured & dataset extracted if present" -ForegroundColor Green

# ── 4. Optional: copy a local data shard ──────────────────────────────────
if ($DataDir -ne "") {
    if (-not (Test-Path $DataDir)) {
        Write-Warning "Data directory '$DataDir' not found locally – skipping data copy."
    } else {
        Write-Host "[4/5] Copying data shard '$DataDir'..." -ForegroundColor Yellow
        # Copy the shard *contents* into data/.  This makes repeat deployments
        # idempotent and prevents a nested data/data_shard_N directory.
        ssh "$PiUser@$PiHost" "mkdir -p $RemoteDirectory/data"
        if ($LASTEXITCODE -ne 0) { throw "Could not create remote data directory." }
        scp -r "$DataDir/." "${PiUser}@${PiHost}:$RemoteDirectory/data/"
        if ($LASTEXITCODE -ne 0) { throw "SCP failed for data directory." }
        Write-Host "      Data shard copied to $RemoteDirectory/data" -ForegroundColor Green
    }
} else {
    Write-Host "[4/5] No -DataDir supplied – skipping data copy." -ForegroundColor DarkGray
    Write-Host "      Copy your MRI shard later with:" -ForegroundColor DarkGray
    Write-Host "        scp -r <shard_folder> ${PiUser}@${PiHost}:$RemoteDirectory/data" -ForegroundColor DarkGray
}

# ── 5. Install Python dependencies on Pi ──────────────────────────────────
Write-Host "[5/5] Installing Python dependencies on Pi (this may take several minutes)..." -ForegroundColor Yellow
ssh "$PiUser@$PiHost" "bash $RemoteDirectory/pi_setup/pi_install.sh $RemoteDirectory"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed on Pi." }
Write-Host "      Dependencies installed" -ForegroundColor Green

# ── Done – print next steps ────────────────────────────────────────────────
Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Find your PC's IP address:" -ForegroundColor White
Write-Host "       ipconfig  (look for the Ethernet / Wi-Fi adapter address)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  2. Start the server on this PC:" -ForegroundColor White
Write-Host "       python server.py --host 0.0.0.0 --port 8080 --rounds 3 --min_clients 1" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  3. On the Pi, start the client:" -ForegroundColor White
Write-Host "       ssh ${PiUser}@${PiHost}" -ForegroundColor DarkGray
Write-Host "       cd $RemoteDirectory && bash pi_setup/start_client.sh <PC_IP> 1" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Or use the PC-side orchestrator:" -ForegroundColor White
Write-Host "       .\run_experiment.ps1 -NumPis 1" -ForegroundColor DarkGray
Write-Host ""

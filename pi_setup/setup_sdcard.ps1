# SpikeFed Pi SD Card Setup Script
# Run this AFTER inserting the SD card and seeing it appear in File Explorer

# Find the Pi boot partition (FAT32 drive that has cmdline.txt)
$bootDrive = Get-PSDrive -PSProvider FileSystem | Where-Object {
    $_.Name -ne "C" -and (Test-Path "$($_.Root)cmdline.txt")
} | Select-Object -First 1

if (-not $bootDrive) {
    Write-Host "ERROR: Could not find Pi SD card boot partition. Make sure the SD card is inserted." -ForegroundColor Red
    Write-Host "Checking all available drives..." -ForegroundColor Yellow
    Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root
    exit 1
}

$boot = $bootDrive.Root
Write-Host "Found Pi boot partition at: $boot" -ForegroundColor Green

# 1. Create SSH enable file (empty file named 'ssh' on boot partition)
$null | Out-File -FilePath "${boot}ssh" -Encoding ascii
Write-Host "SSH enabled (created ssh file)" -ForegroundColor Green

# 2. Credentials must be set in Raspberry Pi Imager before flashing.  Do not
# write a plaintext password to the boot partition: current Raspberry Pi OS
# expects an encrypted userconf entry and Imager creates it safely.

# 3. Write cmdline.txt with static IP via ip= kernel parameter (most reliable method)
$cmdline = Get-Content "${boot}cmdline.txt" -Raw
Write-Host "Current cmdline.txt: $cmdline" -ForegroundColor Cyan

# Remove any existing ip= parameter
$cmdline = $cmdline -replace '\s*ip=\S+', ''
$cmdline = $cmdline.Trim()

# Add static IP kernel parameter
$newCmdline = "$cmdline ip=192.168.137.2::192.168.137.1:255.255.255.0:spikefed-pi1:eth0:off"
$newCmdline | Out-File -FilePath "${boot}cmdline.txt" -Encoding ascii -NoNewline

Write-Host "Updated cmdline.txt with static IP 192.168.137.2" -ForegroundColor Green
Write-Host ""
Write-Host "=== SD Card configured! ===" -ForegroundColor Green
Write-Host "Now:" -ForegroundColor Yellow
Write-Host "  1. Confirm you set a username, password, and SSH in Raspberry Pi Imager"
Write-Host "  2. Safely eject the SD card"
Write-Host "  3. Put it back in the Pi, connect Ethernet, and power on"
Write-Host "  4. Wait 60 seconds, then run: ssh <YOUR_PI_USER>@192.168.137.2"

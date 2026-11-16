# Fix Pi cmdline.txt - removes cloud-init that blocks SSH, keeps static IP
# Run this with SD card inserted

# Find Pi boot partition
$bootDrive = Get-PSDrive -PSProvider FileSystem | Where-Object {
    $_.Name -ne "C" -and (Test-Path "$($_.Root)cmdline.txt")
} | Select-Object -First 1

if (-not $bootDrive) {
    Write-Host "ERROR: SD card not found." -ForegroundColor Red
    Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root
    exit 1
}

$boot = $bootDrive.Root
Write-Host "Found SD card at: $boot" -ForegroundColor Green

# Read current cmdline
$cmdline = Get-Content "${boot}cmdline.txt" -Raw
Write-Host "BEFORE: $cmdline" -ForegroundColor Cyan

# Remove cloud-init directive (ds=nocloud...) - this blocks SSH
$cmdline = $cmdline -replace '\s*ds=nocloud[^\s]*', ''
# Remove any existing ip= parameter
$cmdline = $cmdline -replace '\s*ip=\S+', ''
$cmdline = $cmdline.Trim()

# Add static IP
$newCmdline = "$cmdline ip=192.168.137.2::192.168.137.1:255.255.255.0:spikefed-pi1:eth0:off"

Write-Host "AFTER:  $newCmdline" -ForegroundColor Green

# Write it back (must be single line, no trailing newline)
[System.IO.File]::WriteAllText("${boot}cmdline.txt", $newCmdline)

# Ensure SSH file exists
if (-not (Test-Path "${boot}ssh")) {
    [System.IO.File]::WriteAllText("${boot}ssh", "")
    Write-Host "SSH file created." -ForegroundColor Green
}
else {
    Write-Host "SSH file already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Done! Now eject SD card and put back in Pi ===" -ForegroundColor Green

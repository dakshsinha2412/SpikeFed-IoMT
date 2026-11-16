# Definitive SSH fix for Pi OS Bookworm
# Writes a firstrun.sh that Pi OS executes automatically on first boot

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

# 1. Ensure ssh file exists
[System.IO.File]::WriteAllText("${boot}ssh", "")
Write-Host "ssh file: OK" -ForegroundColor Green

# 2. Write userconf.txt with pi:spikefed123 (SHA-512 crypt hash)
$userconf = 'pi:$6$SpikeFedSalt01$vQ6vIG4OkNaXIw3cghATjimrNg2YWidwraVSJv9JTBKp/1.cDR10iRLr.LO9UUEqlZmHbBZa3nwbV0cZ8N9iw0'
[System.IO.File]::WriteAllText("${boot}userconf.txt", $userconf)
Write-Host "userconf.txt: OK" -ForegroundColor Green

# 3. Write firstrun.sh - Pi OS Bookworm runs this on first boot via cmdline init= parameter
$firstrun = @'
#!/bin/bash
set +e

# Enable SSH
systemctl enable ssh
systemctl start ssh

# Create pi user if missing
if ! id pi &>/dev/null; then
    useradd -m -s /bin/bash pi
    echo 'pi:spikefed123' | chpasswd
    usermod -aG sudo,adm,dialout,cdrom,audio,video,plugdev,games,users,input,netdev,gpio,i2c,spi pi
fi

# Ensure passwordless sudo for pi
echo 'pi ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/010_pi-nopasswd

# Done - remove firstrun marker
rm -f /boot/firstrun.sh
sed -i 's| systemd.run=/boot/firstrun.sh||g' /boot/cmdline.txt

exit 0
'@

# Write with Unix line endings
$firstrunBytes = [System.Text.Encoding]::ASCII.GetBytes($firstrun.Replace("`r`n", "`n"))
[System.IO.File]::WriteAllBytes("${boot}firstrun.sh", $firstrunBytes)
Write-Host "firstrun.sh: OK" -ForegroundColor Green

# 4. Add firstrun.sh trigger to cmdline.txt
$cmdline = [System.IO.File]::ReadAllText("${boot}cmdline.txt").Trim()
if ($cmdline -notmatch "systemd.run") {
    $cmdline = "$cmdline systemd.run=/boot/firstrun.sh systemd.run_success_action=reboot systemd.unit=kernel-command-line.target"
    [System.IO.File]::WriteAllText("${boot}cmdline.txt", $cmdline)
    Write-Host "cmdline.txt updated with firstrun trigger." -ForegroundColor Green
}
else {
    Write-Host "cmdline.txt already has firstrun trigger." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "cmdline.txt now reads:" -ForegroundColor Cyan
Get-Content "${boot}cmdline.txt"

Write-Host ""
Write-Host "=== DONE! Eject -> Pi -> Power on -> Wait 90s -> Tell agent 'ready' ===" -ForegroundColor Green

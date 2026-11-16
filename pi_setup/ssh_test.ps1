# SSH connectivity check using Windows OpenSSH.
# Use the same account you configured in Raspberry Pi Imager.
param(
    [string]$PiHost = "192.168.137.2",
    [string]$PiUser = "pi"
)

# First clear known_hosts for this IP to avoid key conflict
$knownHosts = "$env:USERPROFILE\.ssh\known_hosts"
if (Test-Path $knownHosts) {
    $lines = Get-Content $knownHosts | Where-Object { $_ -notmatch [regex]::Escape($PiHost) }
    $lines | Set-Content $knownHosts
}

Write-Host "Known hosts cleared. Attempting SSH connection..."
Write-Host "Enter the password configured in Raspberry Pi Imager when prompted."
Write-Host ""

# Use ssh with StrictHostKeyChecking disabled
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${PiUser}@${PiHost}" "echo 'SSH_LOGIN_OK'; hostname; uname -a; python3 --version"

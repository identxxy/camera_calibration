param(
  [string]$ShareName = "Dshare",
  [string]$Path = "D:\",
  [string]$Account = $env:USERNAME
)

$ErrorActionPreference = "Stop"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
  throw "Run this script from an elevated PowerShell session."
}

if (-not (Test-Path $Path)) {
  throw "Share path does not exist: $Path"
}

if ($Account.Contains("\")) {
  $Principal = $Account
} else {
  $Principal = "$env:COMPUTERNAME\$Account"
}

# Enable SMB access. DisplayGroup names are localized on non-English Windows
# installs, so create a stable explicit TCP 445 rule instead of depending on
# "File and Printer Sharing" / "Network Discovery" display names.
$FirewallRuleName = "Camera SMB TCP 445"
$FirewallRule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
if ($FirewallRule) {
  Set-NetFirewallRule -DisplayName $FirewallRuleName -Enabled True
} else {
  New-NetFirewallRule `
    -DisplayName $FirewallRuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 445 | Out-Null
}

# Make the machine discoverable in the local network browser.
Set-Service FDResPub -StartupType Automatic
Set-Service fdPHost -StartupType Manual
Start-Service FDResPub
Start-Service fdPHost

$ExistingShare = Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue
if ($ExistingShare) {
  if ($ExistingShare.Path -ne $Path) {
    throw "Share $ShareName already exists but points to $($ExistingShare.Path), not $Path"
  }
  Grant-SmbShareAccess -Name $ShareName -AccountName $Principal -AccessRight Full -Force
} else {
  New-SmbShare -Name $ShareName -Path $Path -FullAccess $Principal -FolderEnumerationMode AccessBased
}

Write-Host "Ready: \\$env:COMPUTERNAME\$ShareName -> $Path"
Write-Host "Mount from Linux with the Windows account: $Principal"

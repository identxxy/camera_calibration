param(
  [string]$ShareName = "Dshare",
  [string]$Path = "D:\",
  [string]$Account = $env:USERNAME
)

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here "windows_enable_calib_d_share.ps1") `
  -ShareName $ShareName `
  -Path $Path `
  -Account $Account

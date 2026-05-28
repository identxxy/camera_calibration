param(
  [string]$OfflineRoot = "D:\tools\calib_qc_offline",
  [string]$PythonRoot = "D:\tools\Python311"
)

$ErrorActionPreference = "Stop"

$installer = Join-Path $OfflineRoot "python-3.11.9-amd64.exe"
$wheels = Join-Path $OfflineRoot "wheels"
$python = Join-Path $PythonRoot "python.exe"

if (!(Test-Path $installer)) {
  throw "Missing Python installer: $installer"
}
if (!(Test-Path $wheels)) {
  throw "Missing wheel directory: $wheels"
}

New-Item -ItemType Directory -Force -Path $OfflineRoot | Out-Null

if (!(Test-Path $python)) {
  Write-Host "Installing Python to $PythonRoot"
  $args = @(
    "/quiet",
    "InstallAllUsers=0",
    "TargetDir=$PythonRoot",
    "Include_pip=1",
    "Include_launcher=0",
    "PrependPath=0",
    "Shortcuts=0"
  )
  $process = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
  if ($process.ExitCode -ne 0) {
    throw "Python installer failed with exit code $($process.ExitCode)"
  }
} else {
  Write-Host "Python already exists at $python"
}

& $python -m pip install --no-index --find-links $wheels numpy opencv-contrib-python
if ($LASTEXITCODE -ne 0) {
  throw "Offline pip install failed with exit code $LASTEXITCODE"
}

& $python -c "import sys, cv2, numpy as np; print(sys.version); print(cv2.__version__); print(np.__version__); assert hasattr(cv2, 'aruco'); assert hasattr(cv2.aruco, 'DICT_APRILTAG_36h11')"
if ($LASTEXITCODE -ne 0) {
  throw "OpenCV ArUco validation failed with exit code $LASTEXITCODE"
}

Write-Host "Calibration QC Python environment is ready."

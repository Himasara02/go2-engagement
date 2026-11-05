# One-time setup on a Windows recording laptop (Windows 10/11, Python 3.12).
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File setup.ps1
# Makes the venv at the repo root (..\.venv), which is where PyCharm looks.
# Python 3.12 is required: pyorbbecsdk2 has no Windows wheel for 3.14.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venv = Join-Path $PSScriptRoot "..\.venv"

Write-Host "== 1/4 Python 3.12 environment ($venv)"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3.12 -m venv $venv
    if (-not $?) { throw "Python 3.12 not found. Install it from python.org, then run this again." }
}
$py = "$venv\Scripts\python.exe"
& $py -m pip install -U pip setuptools wheel
& $py -m pip install "mediapipe>=0.10.14" opencv-python ai-edge-litert matplotlib numpy pyorbbecsdk2
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== 2/4 pose model"
$pose = "models\pose_landmarker_full.task"
if (-not (Test-Path $pose) -or (Get-Item $pose).Length -eq 0) {
    Invoke-WebRequest -OutFile $pose -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
}

Write-Host "== 3/4 UTF-8 for Python (PYTHONUTF8=1, current user)"
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
$env:PYTHONUTF8 = "1"

Write-Host "== 4/4 self-test"
& $py -c "import cv2, mediapipe, numpy, pyorbbecsdk; from engage.perception import Perception; Perception(); print('   imports + models OK')"
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }

Write-Host ""
Write-Host "Setup done. No udev step is needed on Windows. Plug the Orbbec into a USB-3 port, then:"
Write-Host "   ..\.venv\Scripts\Activate.ps1"
Write-Host "   python check_orbbec.py"
Write-Host "(If Activate.ps1 is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)"

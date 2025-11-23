# Path to Python 3.12 — update this if needed
$PY312 = "C:\Users\Anders.LEVIATHAN\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
& $PY312 -m venv .venv

Write-Host "Activating environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Setup complete! " -ForegroundColor Green

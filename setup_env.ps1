# Dynamic Python 3.12+ virtual environment setup
# -------------------------------------------------

Write-Host "Searching for Python 3.12 or newer..." -ForegroundColor Cyan

# Try python3.12 first, then fallback to python
$PYTHON_CMD = $null
$py312 = Get-Command "python3.12" -ErrorAction SilentlyContinue
$py    = Get-Command "python" -ErrorAction SilentlyContinue

if ($py312) {
    $PYTHON_CMD = $py312.Source
} elseif ($py) {
    $PYTHON_CMD = $py.Source
} else {
    Write-Host "❌ Python 3.12+ not found. Please install Python 3.12 or later." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python at: $PYTHON_CMD" -ForegroundColor Yellow

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
& $PYTHON_CMD -m venv .venv

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not created properly." -ForegroundColor Red
    exit 1
}

Write-Host "Activating environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found — skipping dependency install." -ForegroundColor Yellow
}

Write-Host "✅ Setup complete!" -ForegroundColor Green

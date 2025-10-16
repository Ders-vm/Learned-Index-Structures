# compile.ps1 — Rebuild all C++ modules for Learned Index Structures
# Usage:
#   Run from the project root:
#       ./compile.ps1

# Ensure virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found (.venv missing). Please create it first."
    exit 1
}

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Define paths
$buildDir = "build"
$srcDir = "src\indexes\cpp"
$outDir = "src\indexes\cpp"

# Clean previous build
if (Test-Path $buildDir) {
    Write-Host "Removing old build directory..."
    Remove-Item -Recurse -Force $buildDir
}

# Create new build directory
New-Item -ItemType Directory -Path $buildDir | Out-Null
Set-Location $buildDir

# Configure CMake
Write-Host "Configuring CMake..."
cmake .. | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed."
    exit 1
}

# Build all targets
Write-Host "Building project..."
cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed."
    exit 1
}

# Copy compiled .pyd files back to source folder
Write-Host "Copying compiled modules..."
$pydFiles = Get-ChildItem -Recurse -Filter "*.pyd"
foreach ($file in $pydFiles) {
    Copy-Item $file.FullName -Destination $outDir -Force
}

# Verify output
Set-Location ..
if (Test-Path "$outDir\btree_cpp*.pyd") {
    Write-Host "btree_cpp built successfully."
}
if (Test-Path "$outDir\linear_model_cpp*.pyd") {
    Write-Host "linear_model_cpp built successfully."
}
if (Test-Path "$outDir\rmi_cpp*.pyd") {
    Write-Host "rmi_cpp built successfully."
}

Write-Host "Build complete."

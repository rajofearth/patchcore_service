# PatchCore Flask Service - Setup Script
# Run this to set up your environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PatchCore Flask Service Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create virtual environment
Write-Host "[Step 1/4] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 2: Activate virtual environment
Write-Host "[Step 2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "  → Try running: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Upgrade pip
Write-Host "[Step 3/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Step 4: Install dependencies
Write-Host "[Step 4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Make sure you're in the virtual environment:" -ForegroundColor White
    Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Run the Flask service:" -ForegroundColor White
    Write-Host "     python app.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Test with:" -ForegroundColor White
    Write-Host "     python test_service.py" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ Installation failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try installing manually:" -ForegroundColor Yellow
    Write-Host "  pip install flask torch torchvision opencv-python pillow scikit-learn" -ForegroundColor Gray
}

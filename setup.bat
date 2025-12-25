@echo off
REM PatchCore Flask Service - Setup Script (Windows CMD)

echo ========================================
echo PatchCore Flask Service Setup
echo ========================================
echo.

REM Step 1: Create virtual environment
echo [Step 1/4] Creating virtual environment...
if exist venv (
    echo   [OK] Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo   [OK] Virtual environment created
)
echo.

REM Step 2: Activate virtual environment
echo [Step 2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo   [ERROR] Failed to activate virtual environment
    echo   Try running: venv\Scripts\activate.bat
) else (
    echo   [OK] Virtual environment activated
)
echo.

REM Step 3: Upgrade pip
echo [Step 3/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Step 4: Install dependencies
echo [Step 4/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Installation failed
    echo ========================================
    echo.
    echo Try installing manually:
    echo   pip install flask torch torchvision opencv-python pillow scikit-learn
    exit /b 1
) else (
    echo.
    echo ========================================
    echo [OK] Setup Complete!
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Make sure you're in the virtual environment:
    echo      venv\Scripts\activate.bat
    echo.
    echo   2. Run the Flask service:
    echo      python app.py
    echo.
    echo   3. Test with:
    echo      python test_service.py
    echo.
)

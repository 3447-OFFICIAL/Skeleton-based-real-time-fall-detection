@echo off
echo ======================================================
echo   Skeleton-Based Fall Detection System - Launcher
echo ======================================================
echo.
echo Initializing environment...

:: Check if python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Check if model exists, generate if missing
if not exist "ml\model.pkl" (
    echo [INFO] Baseline model not found. Generating now...
    python ml/generate_model.py
)

echo.
echo Running application...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed. Please check the logs above.
    pause
)

echo.
pause

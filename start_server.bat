@echo off
echo ========================================
echo Disease Prediction System - Quick Start
echo ========================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [3/3] Starting backend server...
echo.
echo ========================================
echo Server will start at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo ========================================
echo.
echo Open frontend/index.html in your browser to use the application
echo Press Ctrl+C to stop the server
echo.

uvicorn app:app --reload

pause

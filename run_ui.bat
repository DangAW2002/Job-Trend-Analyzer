@echo off
REM Script to run Job Trend Analyzer Web UI
REM Requires streamlit to be installed

echo.
echo ===================================
echo   Job Trend Analyzer Web UI
echo ===================================
echo.

REM Check if streamlit is available
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Streamlit is not installed or not in PATH
    echo Please install requirements first: pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Web UI...
echo Open your browser and go to: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit app
cd /d "%~dp0"
streamlit run src\web_ui.py --server.port 8501 --server.headless false

pause

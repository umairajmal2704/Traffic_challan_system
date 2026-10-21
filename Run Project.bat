@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project Python environment was not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

echo Starting the Traffic Challan dashboard...
echo Keep this window open while using the project.
echo Open http://localhost:8501 if the browser does not open automatically.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py
set "exit_code=%errorlevel%"

if not "%exit_code%"=="0" (
    echo.
    echo The dashboard stopped with an error. Check the message above.
    pause
)

exit /b %exit_code%

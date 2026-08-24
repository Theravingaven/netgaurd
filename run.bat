@echo off
echo.
echo  Starting NetGuard IDS...
echo  Open http://localhost:5000 in your browser
echo.
cd /d "%~dp0"
python app.py
pause
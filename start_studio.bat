@echo off
echo.
echo ========================================
echo    MediaPipe Studio - Part 4
echo    Web-Based Animation Pipeline
echo ========================================
echo.

echo 🔧 Setting up environment...
cd /d "%~dp0"

echo 📦 Installing required packages...
"C:/Program Files/Python311/python.exe" -m pip install -r requirements.txt

echo.
echo 🚀 Starting MediaPipe Studio...
echo 🌐 Web interface will open at: http://localhost:3000
echo.
echo ✅ Features enabled:
echo    - Full-Body Markerless Tracking
echo    - Face Tracking  
echo    - Multi-Person Support
echo    - UE4/UE5 Export
echo    - Unity Export
echo    - Roblox Export
echo    - MetaHuman Support
echo.
echo Press Ctrl+C to stop the server
echo.

"C:/Program Files/Python311/python.exe" mediapipe_studio.py

pause

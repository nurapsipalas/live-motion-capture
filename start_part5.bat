@echo off
echo.
echo ========================================
echo    MediaPipe Studio - Part 5
echo    Animation Preview with Timeline
echo ========================================
echo.

echo 🔧 Setting up environment...
cd /d "%~dp0"

echo 📦 Installing required packages...
"C:/Program Files/Python311/python.exe" -m pip install -r requirements.txt

echo.
echo 🚀 Starting MediaPipe Studio Part 5...
echo 🌐 Web interface: http://localhost:3000
echo.
echo ✨ NEW Part 5 Features:
echo    🎬 Video Timeline with Scrubbing
echo    ⏯️  Playback Controls (Play/Pause/Step)
echo    🖼️  Real-time Animation Preview
echo    📹 Side-by-side Video/Animation View
echo    🎚️  Frame-by-frame Navigation
echo    ⚡ Variable Speed Playback
echo.
echo 📋 Core Features:
echo    - Full-Body + Face Tracking
echo    - Multi-Person Support
echo    - UE4/UE5/Unity/MetaHuman Export
echo    - Timeline Preview Controls
echo.
echo Press Ctrl+C to stop the server
echo.

"C:/Program Files/Python311/python.exe" mediapipe_studio_preview.py

pause

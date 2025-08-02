@echo off
echo 🎬 Starting Live Motion Capture MP4 System...
echo.
echo Features:
echo ✅ Real-time MP4 motion tracking
echo ✅ Live chapter detection (intro/conflict/resolution)
echo ✅ WebSocket streaming at 15 FPS
echo ✅ FBX/BVH export for Unreal/Blender
echo.
echo 🌐 Server will start at: http://localhost:5000/live
echo 🔴 Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python live_motion_capture_mp4.py

@echo off
echo ================================
echo MediaPipe to FBX Pipeline - Part 2
echo ================================
echo.

echo This will help you export FBX animations from your MediaPipe tracking data.
echo.

echo Prerequisites:
echo - You have already run mediapipe_to_fbx.py 
echo - tracking_data.json exists in the output folder
echo - Blender is installed on your system
echo.

pause

echo.
echo ================================
echo Step 1: Locate Blender
echo ================================

set "BLENDER_PATH="

REM Common Blender installation paths
if exist "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" (
    set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
    echo Found Blender 4.0
) else if exist "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" (
    set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
    echo Found Blender 3.6
) else if exist "C:\Program Files\Blender Foundation\Blender 3.5\blender.exe" (
    set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.5\blender.exe"
    echo Found Blender 3.5
) else if exist "C:\Program Files\Blender Foundation\Blender 3.4\blender.exe" (
    set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.4\blender.exe"
    echo Found Blender 3.4
) else if exist "C:\Program Files\Blender Foundation\Blender 3.3\blender.exe" (
    set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 3.3\blender.exe"
    echo Found Blender 3.3
) else (
    echo Blender not found in standard locations.
    echo.
    echo Please install Blender from: https://www.blender.org/download/
    echo Or manually specify the path below.
    echo.
    set /p "BLENDER_PATH=Enter full path to blender.exe: "
)

if not exist "%BLENDER_PATH%" (
    echo Error: Blender not found at: %BLENDER_PATH%
    pause
    exit /b 1
)

echo Using Blender: %BLENDER_PATH%
echo.

echo ================================
echo Step 2: Check for tracking data
echo ================================

if not exist "output\tracking_data.json" (
    echo Error: tracking_data.json not found in output folder!
    echo.
    echo Please run mediapipe_to_fbx.py first to generate tracking data.
    echo.
    pause
    exit /b 1
)

echo ✓ Found tracking data: output\tracking_data.json
echo.

echo ================================
echo Step 3: Choose export method
echo ================================
echo.
echo Select your preferred method:
echo.
echo 1. Quick Export (simple, fast)
echo 2. Full Export (detailed character rig)
echo 3. Manual (open Blender for custom setup)
echo.

set /p "choice=Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Export...
    echo This will create a basic character rig and export FBX
    echo.
    "%BLENDER_PATH%" --background --python quick_fbx_export.py
    echo.
    if exist "output\quick_animation.fbx" (
        echo ✓ Quick export completed: output\quick_animation.fbx
    ) else (
        echo ✗ Quick export failed - check for errors above
    )
) else if "%choice%"=="2" (
    echo.
    echo Running Full Export...
    echo This will create a detailed character rig with finger bones
    echo.
    "%BLENDER_PATH%" --background --python blender_fbx_exporter.py
    echo.
    if exist "output\character_animation.fbx" (
        echo ✓ Full export completed: output\character_animation.fbx
    ) else (
        echo ✗ Full export failed - check for errors above
    )
) else if "%choice%"=="3" (
    echo.
    echo Opening Blender for manual setup...
    echo.
    echo Instructions:
    echo 1. Blender will open
    echo 2. Go to Scripting workspace
    echo 3. Load either quick_fbx_export.py or blender_fbx_exporter.py
    echo 4. Click "Run Script"
    echo.
    start "" "%BLENDER_PATH%"
    echo Blender started. Follow the instructions above.
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
echo ================================
echo Export Complete!
echo ================================
echo.

if "%choice%"=="1" or "%choice%"=="2" (
    echo Your FBX file is ready for:
    echo - Unreal Engine
    echo - Unity
    echo - Maya
    echo - 3ds Max
    echo - Other 3D software
    echo.
    echo Output folder contents:
    echo.
    dir /b output\*.fbx 2>nul
    if errorlevel 1 (
        echo No FBX files found - export may have failed
    )
    echo.
)

echo Next steps:
echo 1. Import the FBX into your preferred 3D software
echo 2. Apply to your character model
echo 3. Test the animation
echo.

pause

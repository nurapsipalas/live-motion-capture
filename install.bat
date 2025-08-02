@echo off
REM MediaPipe to FBX Animation Pipeline - Installation Script
REM Windows Batch Script for easy setup

echo ================================
echo MediaPipe to FBX Animation Setup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

echo ✓ Python is installed
python --version

echo.
echo Installing required packages...
echo ================================

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install core packages
echo.
echo Installing core MediaPipe packages...
pip install mediapipe>=0.10.7
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0

echo.
echo Installing additional packages...
pip install scipy>=1.11.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install moviepy>=1.0.3
pip install imageio>=2.31.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0

echo.
echo Installing optional AI packages...
pip install tensorflow>=2.13.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0

echo.
echo Installing development tools...
pip install pytest>=7.4.0
pip install black>=23.0.0
pip install flake8>=6.0.0

echo.
echo ================================
echo Installation complete!
echo ================================
echo.

REM Create example directories
echo Creating example directories...
if not exist "input_videos" mkdir input_videos
if not exist "output" mkdir output
if not exist "examples" mkdir examples

echo.
echo ✓ Created directories:
echo   - input_videos/  (place your MP4 files here)
echo   - output/        (processed results will go here)
echo   - examples/      (example files)

echo.
echo ================================
echo Next Steps:
echo ================================
echo 1. Place your MP4 video files in the 'input_videos' folder
echo 2. Run: python example_usage.py
echo 3. For FBX export, install Blender and run blender_fbx_exporter.py
echo.
echo For detailed instructions, see README.md
echo.

REM Test the installation
echo Testing installation...
python -c "import mediapipe, cv2, numpy; print('✓ All core packages imported successfully')"

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Warning: Some packages failed to import
    echo Check the error messages above
) else (
    echo.
    echo ✅ Installation test passed!
)

echo.
echo Press any key to continue...
pause >nul

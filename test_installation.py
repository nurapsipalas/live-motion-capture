#!/usr/bin/env python3
"""
Test script for MediaPipe to FBX Animation Pipeline
Verifies all components are working correctly
"""

import sys
import os
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        ("mediapipe", "MediaPipe"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
    ]
    
    optional_packages = [
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("moviepy.editor", "MoviePy"),
    ]
    
    # Test required packages
    failed_required = []
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✓ {name}")
        except ImportError as e:
            print(f"   ❌ {name} - {e}")
            failed_required.append(name)
    
    # Test optional packages
    failed_optional = []
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"   ✓ {name} (optional)")
        except ImportError:
            print(f"   ⚠️  {name} (optional) - not installed")
            failed_optional.append(name)
    
    # Test Blender availability
    try:
        import bpy
        print(f"   ✓ Blender Python API")
        blender_available = True
    except ImportError:
        print(f"   ⚠️  Blender Python API - not available (run test inside Blender)")
        blender_available = False
    
    return len(failed_required) == 0, blender_available

def test_mediapipe_basic():
    """Test basic MediaPipe functionality"""
    print("\n🧪 Testing MediaPipe basic functionality...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Test MediaPipe solutions initialization
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands  
        mp_pose = mp.solutions.pose
        mp_holistic = mp.solutions.holistic
        
        print("   ✓ MediaPipe solutions imported")
        
        # Test creating instances
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("   ✓ MediaPipe models initialized")
        
        # Test with dummy data
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process with holistic (this will return empty results but shouldn't crash)
        results = holistic.process(dummy_image)
        print("   ✓ MediaPipe processing test passed")
        
        # Cleanup
        face_mesh.close()
        hands.close()
        pose.close()
        holistic.close()
        
        return True
        
    except Exception as e:
        print(f"   ❌ MediaPipe test failed: {e}")
        return False

def test_opencv_basic():
    """Test basic OpenCV functionality"""
    print("\n📹 Testing OpenCV basic functionality...")
    
    try:
        import cv2
        import numpy as np
        
        # Test OpenCV version
        cv_version = cv2.__version__
        print(f"   ✓ OpenCV version: {cv_version}")
        
        # Test basic image operations
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        
        print("   ✓ Color space conversion")
        
        # Test drawing functions
        cv2.circle(dummy_image, (320, 240), 50, (255, 0, 0), 2)
        cv2.line(dummy_image, (0, 0), (640, 480), (0, 255, 0), 2)
        
        print("   ✓ Drawing functions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
        return False

def test_project_files():
    """Test if all project files are present"""
    print("\n📁 Testing project files...")
    
    current_dir = Path(__file__).parent
    
    required_files = [
        "mediapipe_to_fbx.py",
        "config.py", 
        "utils.py",
        "blender_fbx_exporter.py",
        "example_usage.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"   ✓ {file_name}")
        else:
            print(f"   ❌ {file_name} - missing")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from config import Config
        
        # Test config values
        print(f"   ✓ Face detection confidence: {Config.FACE_DETECTION_CONFIDENCE}")
        print(f"   ✓ Hand detection confidence: {Config.HAND_DETECTION_CONFIDENCE}")
        print(f"   ✓ Pose detection confidence: {Config.POSE_DETECTION_CONFIDENCE}")
        print(f"   ✓ Character name: {Config.CHARACTER_NAME}")
        
        # Test bone mappings
        bone_mapping = Config.get_bone_mapping("humanoid")
        print(f"   ✓ Bone mappings loaded: {len(bone_mapping)} bones")
        
        # Test output paths
        paths = Config.create_output_paths()
        print(f"   ✓ Output paths configured: {len(paths)} paths")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🔧 Testing utility functions...")
    
    try:
        from utils import (
            normalize_landmarks,
            smooth_landmarks, 
            calculate_bone_rotation,
            interpolate_missing_landmarks
        )
        
        # Test landmark normalization
        test_landmarks = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)]
        normalized = normalize_landmarks(test_landmarks)
        print(f"   ✓ Landmark normalization: {len(normalized)} points")
        
        # Test bone rotation calculation
        parent_point = (0, 0, 0)
        child_point = (1, 0, 0)
        rotation = calculate_bone_rotation(parent_point, child_point)
        print(f"   ✓ Bone rotation calculation: {rotation}")
        
        # Test smoothing
        landmark_sequence = [test_landmarks] * 10
        smoothed = smooth_landmarks(landmark_sequence, window_size=3)
        print(f"   ✓ Landmark smoothing: {len(smoothed)} frames")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Utils test failed: {e}")
        return False

def create_test_video():
    """Create a simple test video for testing"""
    print("\n🎬 Creating test video...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))
        
        for frame_num in range(90):  # 3 seconds at 30fps
            # Create frame with moving circle (simulates person)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Moving circle
            x = int(320 + 100 * np.sin(frame_num * 0.1))
            y = int(240 + 50 * np.cos(frame_num * 0.1))
            
            # Draw simple "person" 
            cv2.circle(frame, (x, y-40), 20, (255, 255, 255), -1)  # Head
            cv2.rectangle(frame, (x-15, y-20), (x+15, y+20), (255, 255, 255), -1)  # Body
            cv2.line(frame, (x, y+20), (x-10, y+50), (255, 255, 255), 3)  # Left leg
            cv2.line(frame, (x, y+20), (x+10, y+50), (255, 255, 255), 3)  # Right leg
            cv2.line(frame, (x, y-10), (x-15, y+10), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (x, y-10), (x+15, y+10), (255, 255, 255), 3)  # Right arm
            
            # Add frame number
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        print("   ✓ Test video created: test_video.mp4")
        return True
        
    except Exception as e:
        print(f"   ❌ Test video creation failed: {e}")
        return False

def run_full_test():
    """Run complete test suite"""
    print("🧪 MediaPipe to FBX Animation Pipeline - Test Suite")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Files", test_project_files), 
        ("Configuration", test_configuration),
        ("Utility Functions", test_utils),
        ("OpenCV Basic", test_opencv_basic),
        ("MediaPipe Basic", test_mediapipe_basic),
        ("Test Video Creation", create_test_video),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Place MP4 files in 'input_videos' directory")
        print("2. Run: python example_usage.py")
        print("3. For FBX export, open Blender and run blender_fbx_exporter.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("Try running: pip install -r requirements.txt")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_full_test()
    
    if not success:
        sys.exit(1)

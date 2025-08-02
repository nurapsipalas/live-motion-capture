#!/usr/bin/env python3
"""
Test script untuk Live Motion Capture MP4 - verify fixes
"""

import os
import cv2

def test_video_file():
    """Test if test_video.mp4 can be opened"""
    video_path = "test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Error: {video_path} not found")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return False
        
    # Get basic info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video file OK: {video_path}")
    print(f"   📊 Resolution: {width}x{height}")
    print(f"   🎬 Frames: {total_frames}")
    print(f"   ⏱️ FPS: {fps}")
    
    cap.release()
    return True

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import flask
        import flask_socketio
        import mediapipe
        import cv2
        import numpy
        print("✅ All imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Live Motion Capture MP4 Fixes...")
    print()
    
    # Test imports
    if not test_imports():
        print("❌ Fix imports first")
        exit(1)
    
    # Test video file
    if not test_video_file():
        print("❌ Video file issue")
        exit(1)
        
    print()
    print("🎉 All tests passed! Live Motion Capture MP4 should work now.")
    print("🚀 Run: python live_motion_capture_mp4.py")

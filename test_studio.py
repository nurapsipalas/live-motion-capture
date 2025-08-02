#!/usr/bin/env python3
"""
MediaPipe Studio - Part 4 Demo Script
Quick test of all features and export formats
"""

import requests
import json
import time
import os

def test_studio_api():
    """Test MediaPipe Studio API endpoints"""
    base_url = "http://localhost:3000"
    
    print("🎬 MediaPipe Studio - Part 4 API Test")
    print("="*50)
    
    # Test server connection
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Server is running at http://localhost:3000")
        else:
            print("❌ Server connection failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Please start with: python mediapipe_studio.py")
        return
    
    # Test file upload (if video exists)
    test_video = "c:\\Users\\Abror\\Downloads\\VID_20250801153542.mp4"
    if os.path.exists(test_video):
        print(f"\n📹 Testing video upload: {os.path.basename(test_video)}")
        
        try:
            with open(test_video, 'rb') as f:
                files = {'video': (os.path.basename(test_video), f, 'video/mp4')}
                response = requests.post(f"{base_url}/upload", files=files)
                
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Upload successful! Job ID: {data['job_id']}")
                
                # Test processing
                job_id = data['job_id']
                settings = {
                    'model_complexity': 2,
                    'detection_confidence': 0.7,
                    'tracking_confidence': 0.6,
                    'refine_face': True,
                    'enable_segmentation': False,
                    'export_formats': ['ue4', 'ue5', 'unity', 'roblox', 'metahuman', 'anim']
                }
                
                print(f"\n🚀 Starting processing with all export formats...")
                process_response = requests.post(
                    f"{base_url}/process",
                    json={'job_id': job_id, 'settings': settings}
                )
                
                if process_response.status_code == 200:
                    print("✅ Processing started!")
                    
                    # Monitor progress
                    print("\n⏳ Monitoring progress...")
                    while True:
                        status_response = requests.get(f"{base_url}/status/{job_id}")
                        if status_response.status_code == 200:
                            status = status_response.json()
                            progress = status.get('progress', 0)
                            message = status.get('message', 'Processing...')
                            print(f"📊 Progress: {progress:.1f}% - {message}")
                            
                            if status.get('status') == 'completed':
                                print("🎉 Processing completed!")
                                
                                # Show export files
                                export_files = status.get('export_files', {})
                                print(f"\n📤 Generated {len(export_files)} export files:")
                                for format_type, file_path in export_files.items():
                                    if not format_type.endswith('_error'):
                                        print(f"  • {format_type.upper()}: {os.path.basename(file_path)}")
                                
                                # Test download URLs
                                print(f"\n🔗 Download URLs:")
                                for format_type in ['ue4', 'ue5', 'unity', 'roblox', 'metahuman', 'anim', 'tracking']:
                                    print(f"  • {format_type.upper()}: {base_url}/download/{job_id}/{format_type}")
                                
                                break
                            elif status.get('status') == 'error':
                                print(f"❌ Processing failed: {status.get('message')}")
                                break
                        
                        time.sleep(2)
                else:
                    print(f"❌ Processing failed: {process_response.text}")
            else:
                print(f"❌ Upload failed: {response.text}")
        except Exception as e:
            print(f"❌ Upload error: {e}")
    else:
        print(f"\n📹 Test video not found: {test_video}")
        print("💡 You can upload videos through the web interface!")
    
    print(f"\n🌐 Open your browser to: {base_url}")
    print("📋 Features available:")
    print("  • Drag & drop video upload")
    print("  • Real-time processing progress")
    print("  • Multiple export formats (UE4/UE5, Unity, Roblox, MetaHuman)")
    print("  • Advanced settings controls")
    print("  • Animation preview")
    print("  • Download management")

if __name__ == "__main__":
    test_studio_api()

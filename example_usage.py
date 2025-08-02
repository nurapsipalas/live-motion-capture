#!/usr/bin/env python3
"""
Example usage script for MediaPipe to FBX animation pipeline
This script demonstrates how to use the complete pipeline
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from mediapipe_to_fbx import MediaPipeTracker
from config import Config
from utils import (
    validate_tracking_data, 
    create_video_preview, 
    convert_to_unreal_format,
    export_to_bvh
)

def process_single_video(video_path: str, output_dir: str = None):
    """
    Process a single video file through the complete pipeline
    
    Args:
        video_path: Path to input MP4 file
        output_dir: Output directory (optional)
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    # Set up output directory
    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = f"output_{video_name}"
    
    print(f"🎬 Processing video: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create MediaPipe tracker
    tracker = MediaPipeTracker(
        face_detection_confidence=Config.FACE_DETECTION_CONFIDENCE,
        face_tracking_confidence=Config.FACE_TRACKING_CONFIDENCE,
        hand_detection_confidence=Config.HAND_DETECTION_CONFIDENCE,
        hand_tracking_confidence=Config.HAND_TRACKING_CONFIDENCE,
        pose_detection_confidence=Config.POSE_DETECTION_CONFIDENCE,
        pose_tracking_confidence=Config.POSE_TRACKING_CONFIDENCE
    )
    
    # Process video
    print("\n🔄 Step 1: Processing video with MediaPipe...")
    success = tracker.process_video(video_path, output_dir)
    
    if not success:
        print("❌ Failed to process video!")
        return False
    
    # Load and validate tracking data
    print("\n🔍 Step 2: Validating tracking data...")
    tracking_data_path = os.path.join(output_dir, "tracking_data.json")
    
    import json
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    validation_report = validate_tracking_data(tracking_data)
    
    print("📊 Validation Results:")
    for person_id, stats in validation_report["statistics"].items():
        print(f"   Person {person_id}:")
        print(f"     - Frames: {stats['total_frames']}")
        print(f"     - Face detection: {stats['face_detection_rate']:.1%}")
        print(f"     - Hand detection: {stats['hand_detection_rate']:.1%}")
        print(f"     - Pose detection: {stats['pose_detection_rate']:.1%}")
    
    if validation_report["warnings"]:
        print("⚠️  Warnings:")
        for warning in validation_report["warnings"]:
            print(f"   - {warning}")
    
    # Create preview video
    print("\n🎥 Step 3: Creating preview video...")
    preview_path = os.path.join(output_dir, "preview_with_tracking.mp4")
    create_video_preview(tracking_data, video_path, preview_path, max_frames=300)
    
    # Export additional formats
    print("\n📤 Step 4: Exporting additional formats...")
    
    # Export to BVH format
    bvh_path = os.path.join(output_dir, "animation.bvh")
    export_to_bvh(tracking_data, bvh_path)
    
    # Export Unreal Engine format
    unreal_data = convert_to_unreal_format(tracking_data)
    unreal_path = os.path.join(output_dir, "unreal_animation_data.json")
    with open(unreal_path, 'w') as f:
        json.dump(unreal_data, f, indent=2)
    
    print(f"📁 Exported Unreal Engine data to: {unreal_path}")
    
    # Instructions for Blender FBX export
    print("\n🎯 Step 5: FBX Export Instructions")
    print("To create FBX animation file:")
    print("1. Open Blender")
    print("2. Load the script: blender_fbx_exporter.py")
    print("3. Update the paths in the script:")
    print(f"   - tracking_data_path = r'{os.path.abspath(tracking_data_path)}'")
    print(f"   - fbx_output_path = r'{os.path.abspath(os.path.join(output_dir, 'character_animation.fbx'))}'")
    print("4. Run the script in Blender (Alt+P)")
    
    print("\n✅ Processing complete!")
    print(f"📁 Output files in {output_dir}:")
    print(f"   - tracking_data.json (raw MediaPipe data)")
    print(f"   - preview_with_tracking.mp4 (preview video)")
    print(f"   - animation.bvh (BVH format)")
    print(f"   - unreal_animation_data.json (Unreal Engine format)")
    print(f"   - preview_frame_*.jpg (individual frames)")
    
    return True

def batch_process_videos(input_directory: str, output_base_dir: str = "batch_output"):
    """
    Process multiple video files in a directory
    
    Args:
        input_directory: Directory containing MP4 files
        output_base_dir: Base output directory
    """
    
    input_path = Path(input_directory)
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_directory}")
        return
    
    # Find all MP4 files
    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.MP4"))
    
    if not video_files:
        print(f"❌ No MP4 files found in {input_directory}")
        return
    
    print(f"🎬 Found {len(video_files)} video files to process")
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    successful_processes = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(video_files)}: {video_file.name}")
        print(f"{'='*50}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_base_dir, video_file.stem)
        
        success = process_single_video(str(video_file), video_output_dir)
        
        if success:
            successful_processes += 1
            print(f"✅ Successfully processed: {video_file.name}")
        else:
            print(f"❌ Failed to process: {video_file.name}")
    
    print(f"\n🏁 Batch processing complete!")
    print(f"✅ Successfully processed: {successful_processes}/{len(video_files)} videos")
    print(f"📁 All outputs saved to: {output_base_dir}")

def main():
    """Main function with example usage"""
    
    print("🎬 MediaPipe to FBX Animation Pipeline")
    print("=====================================")
    
    # Example 1: Process a single video
    print("\n📋 Example 1: Single Video Processing")
    
    # Update this path to your video file
    example_video = "example_video.mp4"
    
    if os.path.exists(example_video):
        process_single_video(example_video, "single_video_output")
    else:
        print(f"⚠️  Example video not found: {example_video}")
        print("   Please place an MP4 file named 'example_video.mp4' in the current directory")
        print("   Or modify the path in this script")
    
    # Example 2: Batch process multiple videos
    print("\n📋 Example 2: Batch Video Processing")
    
    input_dir = "input_videos"
    if os.path.exists(input_dir):
        batch_process_videos(input_dir, "batch_output")
    else:
        print(f"⚠️  Input directory not found: {input_dir}")
        print("   Create a directory named 'input_videos' and place MP4 files in it")
        print("   Or modify the path in this script")
    
    # Configuration examples
    print("\n⚙️  Configuration Examples:")
    print("To adjust tracking sensitivity, modify these values in config.py:")
    print(f"   - Face detection confidence: {Config.FACE_DETECTION_CONFIDENCE}")
    print(f"   - Hand detection confidence: {Config.HAND_DETECTION_CONFIDENCE}")
    print(f"   - Pose detection confidence: {Config.POSE_DETECTION_CONFIDENCE}")
    
    print("\n📖 For more information, check the README.md file")

if __name__ == "__main__":
    main()

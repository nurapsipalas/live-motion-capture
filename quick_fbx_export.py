#!/usr/bin/env python3
"""
Quick FBX Export Script - Part 2 of MediaPipe to FBX Pipeline
This is a simplified version for quick FBX exports

Run this script inside Blender after processing your video with mediapipe_to_fbx.py
"""

import os
import json

# Blender imports (must run inside Blender)
try:
    import bpy  # type: ignore
    import mathutils  # type: ignore
    BLENDER_AVAILABLE = True
except ImportError:
    print("Error: This script must be run inside Blender!")
    print("\nInstructions:")
    print("1. Open Blender")
    print("2. Go to Scripting workspace") 
    print("3. Load this script")
    print("4. Run the script")
    import sys
    sys.exit(0)


def quick_fbx_export():
    """Quick and simple FBX export from MediaPipe data"""
    
    # File paths - modify these as needed
    TRACKING_DATA = "output/tracking_data.json"
    FBX_OUTPUT = "output/quick_animation.fbx"
    
    print("=== Quick FBX Export ===")
    
    # Check if tracking data exists
    if not os.path.exists(TRACKING_DATA):
        print(f"❌ Tracking data not found: {TRACKING_DATA}")
        print("Please run mediapipe_to_fbx.py first!")
        return False
    
    # Clear scene
    print("🧹 Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Create simple armature
    print("🦴 Creating armature...")
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.active_object
    armature.name = "MediaPipe_Character"
    
    # Create basic bone structure
    armature_data = armature.data
    
    # Add basic bones in edit mode
    bones = {
        "Root": {"head": (0, 0, 0), "tail": (0, 0, 0.1)},
        "Hips": {"head": (0, 0, 1), "tail": (0, 0, 1.2), "parent": "Root"},
        "Spine": {"head": (0, 0, 1.2), "tail": (0, 0, 1.5), "parent": "Hips"},
        "Head": {"head": (0, 0, 1.7), "tail": (0, 0, 1.9), "parent": "Spine"},
        "LeftArm": {"head": (0.3, 0, 1.5), "tail": (0.6, 0, 1.3), "parent": "Spine"},
        "RightArm": {"head": (-0.3, 0, 1.5), "tail": (-0.6, 0, 1.3), "parent": "Spine"},
        "LeftLeg": {"head": (0.2, 0, 1), "tail": (0.2, 0, 0.5), "parent": "Hips"},
        "RightLeg": {"head": (-0.2, 0, 1), "tail": (-0.2, 0, 0.5), "parent": "Hips"},
    }
    
    # Clear default bone and create new ones
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    created_bones = {}
    for bone_name, bone_info in bones.items():
        bone = armature_data.edit_bones.new(bone_name)
        bone.head = bone_info["head"]
        bone.tail = bone_info["tail"]
        created_bones[bone_name] = bone
    
    # Set parents
    for bone_name, bone_info in bones.items():
        if "parent" in bone_info:
            created_bones[bone_name].parent = created_bones[bone_info["parent"]]
    
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Load and apply animation
    print("📊 Loading tracking data...")
    with open(TRACKING_DATA, 'r') as f:
        data = json.load(f)
    
    person_data = list(data["persons"].values())[0]
    frames = person_data["frames"]
    
    print(f"🎬 Applying animation ({len(frames)} frames)...")
    
    # Set scene properties
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frames)
    bpy.context.scene.render.fps = int(data.get("fps", 30))
    
    # Enter pose mode and apply basic animation
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Simple pose mapping (basic version)
    pose_mapping = {
        0: "Head",     # Nose -> Head
        11: "LeftArm",  # Left shoulder
        12: "RightArm", # Right shoulder
        23: "LeftLeg",  # Left hip
        24: "RightLeg", # Right hip
    }
    
    for frame_idx, frame_data in enumerate(frames):
        bpy.context.scene.frame_set(frame_idx + 1)
        
        pose_landmarks = frame_data.get("pose_world_landmarks")
        if pose_landmarks:
            for mp_idx, bone_name in pose_mapping.items():
                if mp_idx < len(pose_landmarks) and bone_name in armature.pose.bones:
                    bone = armature.pose.bones[bone_name]
                    landmark = pose_landmarks[mp_idx]
                    
                    # Simple rotation based on landmark
                    x, y, z = landmark[0], landmark[1], landmark[2]
                    rotation = mathutils.Euler((
                        z * 0.5,
                        (x - 0.5) * 0.5, 
                        y * 0.3
                    ))
                    
                    bone.rotation_euler = rotation
                    bone.keyframe_insert(data_path="rotation_euler")
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / len(frames)) * 100
            print(f"Progress: {progress:.1f}%")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Export FBX
    print("📦 Exporting FBX...")
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    # Create output directory
    os.makedirs(os.path.dirname(FBX_OUTPUT), exist_ok=True)
    
    # Export with basic settings
    bpy.ops.export_scene.fbx(
        filepath=FBX_OUTPUT,
        use_selection=True,
        object_types={'ARMATURE'},
        bake_anim=True,
        bake_anim_use_all_bones=True,
        add_leaf_bones=True,
        primary_bone_axis='Y',
        secondary_bone_axis='X'
    )
    
    print(f"✅ FBX exported to: {FBX_OUTPUT}")
    print("🎯 Ready for Unreal Engine or other 3D software!")
    
    return True


if __name__ == "__main__":
    if BLENDER_AVAILABLE:
        quick_fbx_export()
    else:
        print("This script must be run inside Blender!")

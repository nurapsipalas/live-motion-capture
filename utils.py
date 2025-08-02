"""
Utility functions for MediaPipe to FBX Animation Pipeline
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
import math

def normalize_landmarks(landmarks: List[Tuple[float, float, float]], 
                       reference_point: Optional[int] = None) -> List[Tuple[float, float, float]]:
    """
    Normalize landmarks relative to a reference point or center
    
    Args:
        landmarks: List of (x, y, z) landmark coordinates
        reference_point: Index of landmark to use as reference (None for center)
    
    Returns:
        Normalized landmarks
    """
    if not landmarks:
        return landmarks
    
    landmarks_array = np.array(landmarks)
    
    if reference_point is not None and reference_point < len(landmarks):
        # Normalize relative to specific landmark
        reference = landmarks_array[reference_point]
        normalized = landmarks_array - reference
    else:
        # Normalize relative to center
        center = np.mean(landmarks_array, axis=0)
        normalized = landmarks_array - center
    
    return [(float(x), float(y), float(z)) for x, y, z in normalized]

def smooth_landmarks(landmark_sequence: List[List[Tuple[float, float, float]]], 
                    window_size: int = 5) -> List[List[Tuple[float, float, float]]]:
    """
    Apply temporal smoothing to landmark sequences
    
    Args:
        landmark_sequence: Sequence of landmark frames
        window_size: Size of smoothing window
    
    Returns:
        Smoothed landmark sequence
    """
    if len(landmark_sequence) < window_size:
        return landmark_sequence
    
    smoothed_sequence = []
    half_window = window_size // 2
    
    for i in range(len(landmark_sequence)):
        # Get window bounds
        start_idx = max(0, i - half_window)
        end_idx = min(len(landmark_sequence), i + half_window + 1)
        
        # Average landmarks in window
        window_landmarks = landmark_sequence[start_idx:end_idx]
        if not window_landmarks or not window_landmarks[0]:
            smoothed_sequence.append(landmark_sequence[i])
            continue
        
        # Average each landmark point
        num_landmarks = len(window_landmarks[0])
        smoothed_frame = []
        
        for landmark_idx in range(num_landmarks):
            x_sum = y_sum = z_sum = 0.0
            count = 0
            
            for frame_landmarks in window_landmarks:
                if landmark_idx < len(frame_landmarks):
                    x, y, z = frame_landmarks[landmark_idx]
                    x_sum += x
                    y_sum += y
                    z_sum += z
                    count += 1
            
            if count > 0:
                smoothed_frame.append((x_sum / count, y_sum / count, z_sum / count))
            else:
                smoothed_frame.append(landmark_sequence[i][landmark_idx])
        
        smoothed_sequence.append(smoothed_frame)
    
    return smoothed_sequence

def calculate_bone_rotation(parent_point: Tuple[float, float, float], 
                          child_point: Tuple[float, float, float],
                          up_vector: Tuple[float, float, float] = (0, 0, 1)) -> Tuple[float, float, float]:
    """
    Calculate bone rotation from parent to child point
    
    Args:
        parent_point: Parent joint position
        child_point: Child joint position
        up_vector: Up direction vector
    
    Returns:
        Euler rotation angles (x, y, z) in radians
    """
    # Calculate direction vector
    direction = np.array(child_point) - np.array(parent_point)
    direction_length = np.linalg.norm(direction)
    
    if direction_length < 1e-6:
        return (0.0, 0.0, 0.0)
    
    direction = direction / direction_length
    up = np.array(up_vector)
    
    # Calculate rotation matrix
    # Forward is the direction from parent to child
    forward = direction
    
    # Right is perpendicular to forward and up
    right = np.cross(forward, up)
    right_length = np.linalg.norm(right)
    
    if right_length < 1e-6:
        # Forward and up are parallel, choose arbitrary right
        right = np.array([1, 0, 0]) if abs(forward[0]) < 0.9 else np.array([0, 1, 0])
    else:
        right = right / right_length
    
    # Recalculate up as perpendicular to forward and right
    up = np.cross(right, forward)
    
    # Create rotation matrix
    rotation_matrix = np.column_stack([right, up, forward])
    
    # Convert to Euler angles (ZYX order)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    return (x, y, z)

def interpolate_missing_landmarks(landmarks: List[Optional[List[Tuple[float, float, float]]]], 
                                method: str = "linear") -> List[List[Tuple[float, float, float]]]:
    """
    Interpolate missing landmark frames
    
    Args:
        landmarks: List of landmark frames (None for missing frames)
        method: Interpolation method ("linear", "cubic")
    
    Returns:
        Complete landmark sequence with interpolated frames
    """
    if not landmarks:
        return []
    
    # Find valid frames
    valid_indices = [i for i, frame in enumerate(landmarks) if frame is not None]
    
    if not valid_indices:
        return []
    
    # If no missing frames, return as is
    if len(valid_indices) == len(landmarks):
        return landmarks
    
    interpolated = landmarks.copy()
    
    # Interpolate missing frames
    for i in range(len(landmarks)):
        if landmarks[i] is None:
            # Find nearest valid frames
            prev_idx = None
            next_idx = None
            
            for valid_idx in valid_indices:
                if valid_idx < i:
                    prev_idx = valid_idx
                elif valid_idx > i and next_idx is None:
                    next_idx = valid_idx
                    break
            
            # Interpolate
            if prev_idx is not None and next_idx is not None:
                # Linear interpolation between prev and next
                prev_frame = landmarks[prev_idx]
                next_frame = landmarks[next_idx]
                
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                
                interpolated_frame = []
                for j in range(len(prev_frame)):
                    prev_point = np.array(prev_frame[j])
                    next_point = np.array(next_frame[j])
                    interp_point = prev_point + alpha * (next_point - prev_point)
                    interpolated_frame.append(tuple(interp_point))
                
                interpolated[i] = interpolated_frame
            
            elif prev_idx is not None:
                # Use previous frame
                interpolated[i] = landmarks[prev_idx]
            elif next_idx is not None:
                # Use next frame
                interpolated[i] = landmarks[next_idx]
    
    return interpolated

def create_video_preview(tracking_data: Dict, video_path: str, output_path: str, 
                        max_frames: Optional[int] = None):
    """
    Create a preview video with tracking data overlaid
    
    Args:
        tracking_data: Tracking data dictionary
        video_path: Original video file path
        output_path: Output preview video path
        max_frames: Maximum number of frames to process
    """
    import mediapipe as mp
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    person_data = list(tracking_data["persons"].values())[0]  # First person
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        
        if frame_idx < len(person_data["frames"]):
            frame_data = person_data["frames"][frame_idx]
            
            # Draw face landmarks
            if frame_data.get("face_landmarks"):
                face_landmarks = frame_data["face_landmarks"]
                # Convert to MediaPipe format for drawing
                # This is simplified - you'd need proper conversion
                for i, (x, y, z) in enumerate(face_landmarks):
                    px = int(x * width)
                    py = int(y * height)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
            
            # Draw pose landmarks
            if frame_data.get("pose_landmarks"):
                pose_landmarks = frame_data["pose_landmarks"]
                # Draw pose connections
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                ]
                
                for start_idx, end_idx in connections:
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        start = pose_landmarks[start_idx]
                        end = pose_landmarks[end_idx]
                        
                        start_px = int(start[0] * width)
                        start_py = int(start[1] * height)
                        end_px = int(end[0] * width)
                        end_py = int(end[1] * height)
                        
                        cv2.line(frame, (start_px, start_py), (end_px, end_py), (255, 0, 0), 2)
            
            # Draw hand landmarks
            for hand_type in ["left_hand_landmarks", "right_hand_landmarks"]:
                if frame_data.get(hand_type):
                    hand_landmarks = frame_data[hand_type]
                    color = (0, 255, 255) if hand_type.startswith("left") else (255, 255, 0)
                    
                    for i, (x, y, z) in enumerate(hand_landmarks):
                        px = int(x * width)
                        py = int(y * height)
                        cv2.circle(frame, (px, py), 2, color, -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    cap.release()
    out.release()
    print(f"Preview video saved to: {output_path}")

def validate_tracking_data(tracking_data: Dict) -> Dict[str, any]:
    """
    Validate and analyze tracking data quality
    
    Args:
        tracking_data: Tracking data dictionary
    
    Returns:
        Validation report
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {}
    }
    
    if "persons" not in tracking_data:
        report["errors"].append("No persons data found")
        report["valid"] = False
        return report
    
    for person_id, person_data in tracking_data["persons"].items():
        frames = person_data.get("frames", [])
        
        if not frames:
            report["warnings"].append(f"Person {person_id} has no frame data")
            continue
        
        # Count detection rates
        face_detections = sum(1 for f in frames if f.get("face_landmarks"))
        hand_detections = sum(1 for f in frames if f.get("left_hand_landmarks") or f.get("right_hand_landmarks"))
        pose_detections = sum(1 for f in frames if f.get("pose_landmarks"))
        
        person_stats = {
            "total_frames": len(frames),
            "face_detection_rate": face_detections / len(frames),
            "hand_detection_rate": hand_detections / len(frames),
            "pose_detection_rate": pose_detections / len(frames),
        }
        
        report["statistics"][person_id] = person_stats
        
        # Check for low detection rates
        if person_stats["face_detection_rate"] < 0.5:
            report["warnings"].append(f"Person {person_id}: Low face detection rate ({person_stats['face_detection_rate']:.1%})")
        
        if person_stats["pose_detection_rate"] < 0.8:
            report["warnings"].append(f"Person {person_id}: Low pose detection rate ({person_stats['pose_detection_rate']:.1%})")
    
    return report

def export_to_bvh(tracking_data: Dict, output_path: str):
    """
    Export tracking data to BVH format (alternative to FBX)
    
    Args:
        tracking_data: Tracking data dictionary
        output_path: Output BVH file path
    """
    # BVH (BioVision Hierarchy) format implementation
    # This is a simplified version - full BVH export would be more complex
    
    fps = tracking_data.get("fps", 30.0)
    frame_time = 1.0 / fps
    
    # BVH header
    bvh_content = "HIERARCHY\n"
    bvh_content += "ROOT Hips\n{\n"
    bvh_content += "    OFFSET 0.0 0.0 0.0\n"
    bvh_content += "    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
    
    # Add joints (simplified hierarchy)
    joints = [
        ("Chest", "Hips", "0.0 20.0 0.0"),
        ("Neck", "Chest", "0.0 20.0 0.0"),
        ("Head", "Neck", "0.0 10.0 0.0"),
        ("LeftShoulder", "Chest", "15.0 15.0 0.0"),
        ("LeftArm", "LeftShoulder", "20.0 0.0 0.0"),
        ("LeftForearm", "LeftArm", "25.0 0.0 0.0"),
        ("LeftHand", "LeftForearm", "20.0 0.0 0.0"),
        ("RightShoulder", "Chest", "-15.0 15.0 0.0"),
        ("RightArm", "RightShoulder", "-20.0 0.0 0.0"),
        ("RightForearm", "RightArm", "-25.0 0.0 0.0"),
        ("RightHand", "RightForearm", "-20.0 0.0 0.0"),
        ("LeftThigh", "Hips", "10.0 -10.0 0.0"),
        ("LeftShin", "LeftThigh", "0.0 -40.0 0.0"),
        ("LeftFoot", "LeftShin", "0.0 -40.0 0.0"),
        ("RightThigh", "Hips", "-10.0 -10.0 0.0"),
        ("RightShin", "RightThigh", "0.0 -40.0 0.0"),
        ("RightFoot", "RightShin", "0.0 -40.0 0.0"),
    ]
    
    # Build hierarchy (simplified)
    for joint_name, parent, offset in joints:
        indent = "    " if parent == "Hips" else "        "
        bvh_content += f"{indent}JOINT {joint_name}\n{indent}{{\n"
        bvh_content += f"{indent}    OFFSET {offset}\n"
        bvh_content += f"{indent}    CHANNELS 3 Zrotation Xrotation Yrotation\n"
        bvh_content += f"{indent}    End Site\n{indent}    {{\n"
        bvh_content += f"{indent}        OFFSET 0.0 5.0 0.0\n"
        bvh_content += f"{indent}    }}\n"
        bvh_content += f"{indent}}}\n"
    
    bvh_content += "}\n"
    
    # Motion data
    person_data = list(tracking_data["persons"].values())[0]
    frames = person_data["frames"]
    
    bvh_content += "MOTION\n"
    bvh_content += f"Frames: {len(frames)}\n"
    bvh_content += f"Frame Time: {frame_time:.6f}\n"
    
    # Frame data (simplified - just root position and basic rotations)
    for frame_data in frames:
        frame_values = []
        
        # Root position (from pose landmarks if available)
        if frame_data.get("pose_landmarks"):
            pose = frame_data["pose_landmarks"]
            if len(pose) > 23:  # Hip center
                hip_pos = pose[23]
                frame_values.extend([hip_pos[0] * 100, hip_pos[1] * 100, hip_pos[2] * 100])
            else:
                frame_values.extend([0.0, 0.0, 0.0])
        else:
            frame_values.extend([0.0, 0.0, 0.0])
        
        # Root rotation and joint rotations (simplified)
        for _ in range(len(joints) + 1):  # +1 for root
            frame_values.extend([0.0, 0.0, 0.0])  # Placeholder rotations
        
        bvh_content += " ".join(f"{v:.6f}" for v in frame_values) + "\n"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(bvh_content)
    
    print(f"BVH file exported to: {output_path}")

def convert_to_unreal_format(tracking_data: Dict) -> Dict:
    """
    Convert tracking data to Unreal Engine compatible format
    
    Args:
        tracking_data: Original tracking data
    
    Returns:
        Unreal Engine compatible data structure
    """
    unreal_data = {
        "version": "1.0",
        "character_type": "humanoid",
        "animation_data": {},
        "metadata": {
            "fps": tracking_data.get("fps", 30.0),
            "total_frames": tracking_data.get("total_frames", 0),
            "source": "MediaPipe"
        }
    }
    
    # Convert bone names to Unreal naming convention
    unreal_bone_mapping = {
        "Hips": "pelvis",
        "Chest": "spine_02",
        "Neck": "neck_01", 
        "Head": "head",
        "LeftShoulder": "clavicle_l",
        "LeftArm": "upperarm_l",
        "LeftForearm": "lowerarm_l",
        "LeftHand": "hand_l",
        "RightShoulder": "clavicle_r",
        "RightArm": "upperarm_r", 
        "RightForearm": "lowerarm_r",
        "RightHand": "hand_r",
        "LeftThigh": "thigh_l",
        "LeftShin": "calf_l",
        "LeftFoot": "foot_l",
        "RightThigh": "thigh_r",
        "RightShin": "calf_r",
        "RightFoot": "foot_r",
    }
    
    for person_id, person_data in tracking_data["persons"].items():
        frames = person_data["frames"]
        
        unreal_animation = {
            "bone_tracks": {},
            "morph_targets": {},  # For facial expressions
            "frame_count": len(frames)
        }
        
        # Initialize bone tracks
        for bone_name in unreal_bone_mapping.values():
            unreal_animation["bone_tracks"][bone_name] = {
                "position_keys": [],
                "rotation_keys": [],
                "scale_keys": []
            }
        
        # Process frames
        for frame_idx, frame_data in enumerate(frames):
            timestamp = frame_data["timestamp"]
            
            # Add default transforms for all bones
            for bone_name in unreal_bone_mapping.values():
                unreal_animation["bone_tracks"][bone_name]["position_keys"].append({
                    "time": timestamp,
                    "value": [0.0, 0.0, 0.0]
                })
                unreal_animation["bone_tracks"][bone_name]["rotation_keys"].append({
                    "time": timestamp, 
                    "value": [0.0, 0.0, 0.0, 1.0]  # Quaternion
                })
                unreal_animation["bone_tracks"][bone_name]["scale_keys"].append({
                    "time": timestamp,
                    "value": [1.0, 1.0, 1.0]
                })
            
            # Add facial morph targets
            if frame_data.get("face_expressions"):
                for frame_idx_inner, (expression, value) in enumerate(frame_data["face_expressions"].items()):
                    if expression not in unreal_animation["morph_targets"]:
                        unreal_animation["morph_targets"][expression] = []
                    
                    unreal_animation["morph_targets"][expression].append({
                        "time": timestamp,
                        "value": value
                    })
        
        unreal_data["animation_data"][f"person_{person_id}"] = unreal_animation
    
    return unreal_data

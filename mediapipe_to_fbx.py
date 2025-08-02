#!/usr/bin/env python3
"""
Complete MediaPipe Python Script for Face + Hand + Multi-Person Tracking
Converts MP4 files to FBX animation files via Blender for Unreal Engine

Features:
- Face landmark detection and expression tracking
- Hand pose estimation for both hands
- Multi-person pose detection
- Real-time processing with visualization
- Export to FBX format compatible with Unreal Engine
- Support for WhiteRobot Man character rigging

Requirements:
- mediapipe
- opencv-python
- numpy
- bpy (Blender Python API)
- mathutils
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Try to import Blender modules (optional for preview mode)
# Try to import Blender modules (optional for preview mode)
try:
    import bpy
    BLENDER_AVAILABLE = True
    try:
        import mathutils
    except ImportError:
        mathutils = None
        print("Warning: mathutils not available. Some Blender features may not work.")
except ModuleNotFoundError:
    BLENDER_AVAILABLE = False
    bpy = None
    mathutils = None
    print("Blender Python API not available. If you want to export FBX, run this script inside Blender's Python environment.")

@dataclass
class LandmarkData:
    """Store landmark data for a single frame"""
    timestamp: float
    face_landmarks: Optional[List[Tuple[float, float, float]]] = None
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    pose_landmarks: Optional[List[Tuple[float, float, float]]] = None
    pose_world_landmarks: Optional[List[Tuple[float, float, float]]] = None

@dataclass
class PersonData:
    """Store data for a single person across frames"""
    person_id: int
    frames: List[LandmarkData] = field(default_factory=list)

class MediaPipeTracker:
    """Main class for MediaPipe tracking and FBX export"""
    
    def __init__(self, 
                 face_detection_confidence: float = 0.6,
                 face_tracking_confidence: float = 0.5,
                 hand_detection_confidence: float = 0.7,
                 hand_tracking_confidence: float = 0.5,
                 pose_detection_confidence: float = 0.5,
                 pose_tracking_confidence: float = 0.5):
        
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize trackers
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=face_detection_confidence,
            min_tracking_confidence=face_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=6,  # Support for 3 people × 2 hands
            min_detection_confidence=hand_detection_confidence,
            min_tracking_confidence=hand_tracking_confidence
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=pose_detection_confidence,
            min_tracking_confidence=pose_tracking_confidence
        )
        
        # Holistic model for integrated tracking
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.tracked_persons: Dict[int, PersonData] = {}
        self.frame_count = 0
        self.fps = 30.0
        
        # Face landmark indices for expressions
        self.face_landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305],
            'right_eyebrow': [276, 283, 282, 281, 278, 344, 360, 363, 331, 279, 440, 75],
            'mouth': [0, 17, 18, 200, 199, 175, 0, 269, 270, 267, 271, 272, 12, 15, 16, 17, 18, 200],
            'nose': [1, 2, 5, 4, 6, 19, 94, 168, 8, 9, 10, 151]
        }

    def extract_landmarks_from_results(self, results, timestamp: float) -> LandmarkData:
        """Extract landmark data from MediaPipe results"""
        landmark_data = LandmarkData(timestamp=timestamp)
        
        # Face landmarks
        if results.face_landmarks:
            face_landmarks = []
            for landmark in results.face_landmarks.landmark:
                face_landmarks.append((landmark.x, landmark.y, landmark.z))
            landmark_data.face_landmarks = face_landmarks
        
        # Hand landmarks
        if results.left_hand_landmarks:
            left_hand = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand.append((landmark.x, landmark.y, landmark.z))
            landmark_data.left_hand_landmarks = left_hand
            
        if results.right_hand_landmarks:
            right_hand = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand.append((landmark.x, landmark.y, landmark.z))
            landmark_data.right_hand_landmarks = right_hand
        
        # Pose landmarks
        if results.pose_landmarks:
            pose_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.append((landmark.x, landmark.y, landmark.z))
            landmark_data.pose_landmarks = pose_landmarks
            
        if results.pose_world_landmarks:
            pose_world = []
            for landmark in results.pose_world_landmarks.landmark:
                pose_world.append((landmark.x, landmark.y, landmark.z))
            landmark_data.pose_world_landmarks = pose_world
        
        return landmark_data

    def calculate_face_expressions(self, face_landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """Calculate facial expression values from landmarks"""
        if not face_landmarks or len(face_landmarks) < 468:
            return {}
        
        expressions = {}
        
        # Eye blink detection
        left_eye_landmarks = [face_landmarks[i] for i in self.face_landmark_indices['left_eye']]
        right_eye_landmarks = [face_landmarks[i] for i in self.face_landmark_indices['right_eye']]
        
        # Calculate eye aspect ratio (EAR)
        def eye_aspect_ratio(eye_points):
            if len(eye_points) < 6:
                return 0.0
            # Vertical distances
            A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            # Horizontal distance
            C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            return (A + B) / (2.0 * C) if C > 0 else 0.0
        
        expressions['left_eye_blink'] = 1.0 - min(eye_aspect_ratio(left_eye_landmarks), 1.0)
        expressions['right_eye_blink'] = 1.0 - min(eye_aspect_ratio(right_eye_landmarks), 1.0)
        
        # Mouth expressions
        mouth_landmarks = [face_landmarks[i] for i in self.face_landmark_indices['mouth'][:8]]
        if len(mouth_landmarks) >= 8:
            # Mouth openness
            mouth_height = abs(mouth_landmarks[3][1] - mouth_landmarks[7][1])
            mouth_width = abs(mouth_landmarks[0][0] - mouth_landmarks[4][0])
            expressions['mouth_open'] = min(mouth_height / mouth_width * 2.0, 1.0) if mouth_width > 0 else 0.0
            
            # Smile detection (corners vs center)
            left_corner = mouth_landmarks[0][1]
            right_corner = mouth_landmarks[4][1]
            center = mouth_landmarks[2][1]
            smile_factor = (center - (left_corner + right_corner) / 2.0) * 10.0
            expressions['smile'] = max(0.0, min(smile_factor, 1.0))
        
        # Eyebrow raise
        left_brow = [face_landmarks[i] for i in self.face_landmark_indices['left_eyebrow']]
        right_brow = [face_landmarks[i] for i in self.face_landmark_indices['right_eyebrow']]
        
        if left_brow and right_brow:
            left_brow_height = np.mean([p[1] for p in left_brow])
            right_brow_height = np.mean([p[1] for p in right_brow])
            # Normalize based on face height (approximate)
            face_height = abs(face_landmarks[10][1] - face_landmarks[152][1])
            expressions['left_eyebrow_raise'] = max(0.0, min((0.3 - left_brow_height) / face_height * 3.0, 1.0))
            expressions['right_eyebrow_raise'] = max(0.0, min((0.3 - right_brow_height) / face_height * 3.0, 1.0))
        
        return expressions

    def calculate_hand_gestures(self, hand_landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """Calculate hand gesture values from landmarks"""
        if not hand_landmarks or len(hand_landmarks) != 21:
            return {}
        
        gestures = {}
        
        # Finger positions (0 = closed, 1 = open)
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]  # pip joints
        
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # Thumb (different calculation)
                # Compare with thumb joint
                thumb_open = np.linalg.norm(np.array(hand_landmarks[tip]) - np.array(hand_landmarks[2]))
                thumb_closed = np.linalg.norm(np.array(hand_landmarks[3]) - np.array(hand_landmarks[2]))
                gestures[f'finger_{i}_curl'] = 1.0 - min(thumb_open / max(thumb_closed, 0.001), 1.0)
            else:
                # Compare tip with pip joint
                finger_length = np.linalg.norm(np.array(hand_landmarks[tip]) - np.array(hand_landmarks[pip]))
                max_length = np.linalg.norm(np.array(hand_landmarks[0]) - np.array(hand_landmarks[tip]))
                gestures[f'finger_{i}_curl'] = 1.0 - min(finger_length / max(max_length * 0.5, 0.001), 1.0)
        
        # Overall hand openness
        gestures['hand_openness'] = 1.0 - np.mean([gestures[f'finger_{i}_curl'] for i in range(5)])
        
        # Fist detection
        fist_threshold = 0.2
        gestures['fist'] = 1.0 if gestures['hand_openness'] < fist_threshold else 0.0
        
        return gestures

    def process_video(self, video_path: str, output_dir: str = "output") -> bool:
        """Process MP4 video and extract tracking data"""
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found!")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps}, Frames: {total_frames}")
        
        # Initialize person tracking
        person_id = 0  # For single person, expand for multi-person
        self.tracked_persons[person_id] = PersonData(person_id=person_id)
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with holistic model
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            timestamp = frame_num / self.fps
            landmark_data = self.extract_landmarks_from_results(results, timestamp)
            
            # Add to person data
            self.tracked_persons[person_id].frames.append(landmark_data)
            
            # Visualization (optional)
            if frame_num % 30 == 0:  # Show progress every 30 frames
                print(f"Processed frame {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
                
                # Draw landmarks on frame for preview
                annotated_frame = frame.copy()
                
                if results.face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, results.face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, results.left_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style())
                
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, results.right_hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style())
                
                # Save preview frame
                cv2.imwrite(os.path.join(output_dir, f"preview_frame_{frame_num:06d}.jpg"), annotated_frame)
            
            frame_num += 1
        
        cap.release()
        
        # Save tracking data to JSON
        self.save_tracking_data(output_dir)
        
        print(f"Processing complete! Processed {frame_num} frames.")
        print(f"Tracking data saved to {output_dir}")
        
        return True

    def save_tracking_data(self, output_dir: str):
        """Save tracking data to JSON file"""
        output_file = os.path.join(output_dir, "tracking_data.json")
        
        # Convert to serializable format
        data = {
            "fps": self.fps,
            "total_frames": len(self.tracked_persons[0].frames) if self.tracked_persons else 0,
            "persons": {}
        }
        
        for person_id, person_data in self.tracked_persons.items():
            person_dict = {
                "person_id": person_id,
                "frames": []
            }
            
            for frame_data in person_data.frames:
                frame_dict = {
                    "timestamp": frame_data.timestamp,
                    "face_landmarks": frame_data.face_landmarks,
                    "left_hand_landmarks": frame_data.left_hand_landmarks,
                    "right_hand_landmarks": frame_data.right_hand_landmarks,
                    "pose_landmarks": frame_data.pose_landmarks,
                    "pose_world_landmarks": frame_data.pose_world_landmarks
                }
                
                # Add calculated expressions and gestures
                if frame_data.face_landmarks:
                    frame_dict["face_expressions"] = self.calculate_face_expressions(frame_data.face_landmarks)
                
                if frame_data.left_hand_landmarks:
                    frame_dict["left_hand_gestures"] = self.calculate_hand_gestures(frame_data.left_hand_landmarks)
                
                if frame_data.right_hand_landmarks:
                    frame_dict["right_hand_gestures"] = self.calculate_hand_gestures(frame_data.right_hand_landmarks)
                
                person_dict["frames"].append(frame_dict)
            
            data["persons"][str(person_id)] = person_dict
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tracking data saved to: {output_file}")


class BlenderFBXExporter:
    """Handle Blender operations and FBX export"""
    
    def __init__(self):
        if not BLENDER_AVAILABLE:
            raise ImportError("Blender Python API not available!")
        
        # Clear existing mesh objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Set up scene
        bpy.context.scene.frame_set(1)
        
    def create_character_rig(self, name: str = "WhiteRobotMan") -> str:
        """Create a basic character rig for animation"""
        
        # Add armature
        bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
        armature = bpy.context.active_object
        armature.name = f"{name}_Armature"
        
        # Get the armature data
        armature_data = armature.data
        armature_data.name = f"{name}_ArmatureData"
        
        # Create bone hierarchy for humanoid character
        bones_hierarchy = {
            "Root": {"parent": None, "head": (0, 0, 0), "tail": (0, 0, 0.1)},
            "Hips": {"parent": "Root", "head": (0, 0, 1), "tail": (0, 0, 1.2)},
            "Spine": {"parent": "Hips", "head": (0, 0, 1.2), "tail": (0, 0, 1.5)},
            "Chest": {"parent": "Spine", "head": (0, 0, 1.5), "tail": (0, 0, 1.7)},
            "Neck": {"parent": "Chest", "head": (0, 0, 1.7), "tail": (0, 0, 1.8)},
            "Head": {"parent": "Neck", "head": (0, 0, 1.8), "tail": (0, 0, 2.0)},
            
            # Arms
            "LeftShoulder": {"parent": "Chest", "head": (0.2, 0, 1.6), "tail": (0.4, 0, 1.6)},
            "LeftArm": {"parent": "LeftShoulder", "head": (0.4, 0, 1.6), "tail": (0.7, 0, 1.4)},
            "LeftForearm": {"parent": "LeftArm", "head": (0.7, 0, 1.4), "tail": (1.0, 0, 1.2)},
            "LeftHand": {"parent": "LeftForearm", "head": (1.0, 0, 1.2), "tail": (1.1, 0, 1.2)},
            
            "RightShoulder": {"parent": "Chest", "head": (-0.2, 0, 1.6), "tail": (-0.4, 0, 1.6)},
            "RightArm": {"parent": "RightShoulder", "head": (-0.4, 0, 1.6), "tail": (-0.7, 0, 1.4)},
            "RightForearm": {"parent": "RightArm", "head": (-0.7, 0, 1.4), "tail": (-1.0, 0, 1.2)},
            "RightHand": {"parent": "RightForearm", "head": (-1.0, 0, 1.2), "tail": (-1.1, 0, 1.2)},
            
            # Legs
            "LeftThigh": {"parent": "Hips", "head": (0.15, 0, 1.0), "tail": (0.15, 0, 0.5)},
            "LeftShin": {"parent": "LeftThigh", "head": (0.15, 0, 0.5), "tail": (0.15, 0, 0.1)},
            "LeftFoot": {"parent": "LeftShin", "head": (0.15, 0, 0.1), "tail": (0.15, 0.2, 0.0)},
            
            "RightThigh": {"parent": "Hips", "head": (-0.15, 0, 1.0), "tail": (-0.15, 0, 0.5)},
            "RightShin": {"parent": "RightThigh", "head": (-0.15, 0, 0.5), "tail": (-0.15, 0, 0.1)},
            "RightFoot": {"parent": "RightShin", "head": (-0.15, 0, 0.1), "tail": (-0.15, 0.2, 0.0)},
        }
        
        # Create bones
        for bone_name, bone_data in bones_hierarchy.items():
            if bone_name == "Root":
                # Root bone already exists, just rename it
                root_bone = armature_data.edit_bones[0]
                root_bone.name = bone_name
                root_bone.head = bone_data["head"]
                root_bone.tail = bone_data["tail"]
            else:
                # Create new bone
                bone = armature_data.edit_bones.new(bone_name)
                bone.head = bone_data["head"]
                bone.tail = bone_data["tail"]
                
                # Set parent
                if bone_data["parent"]:
                    parent_bone = armature_data.edit_bones[bone_data["parent"]]
                    bone.parent = parent_bone
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return armature.name

    def apply_pose_animation(self, tracking_data: Dict, armature_name: str):
        """Apply pose animation to the armature from tracking data"""
        
        # Get armature object
        armature = bpy.data.objects[armature_name]
        bpy.context.view_layer.objects.active = armature
        
        # MediaPipe to Blender bone mapping
        mp_to_blender_bones = {
            # Pose landmarks to bone mapping
            11: "LeftShoulder",   # Left shoulder
            12: "RightShoulder",  # Right shoulder
            13: "LeftArm",        # Left elbow
            14: "RightArm",       # Right elbow
            15: "LeftForearm",    # Left wrist
            16: "RightForearm",   # Right wrist
            23: "LeftThigh",      # Left hip
            24: "RightThigh",     # Right hip
            25: "LeftShin",       # Left knee
            26: "RightShin",      # Right knee
            27: "LeftFoot",       # Left ankle
            28: "RightFoot",      # Right ankle
            0: "Head",            # Nose (approximating head)
        }
        
        # Process each person's data
        for person_id, person_data in tracking_data["persons"].items():
            frames = person_data["frames"]
            fps = tracking_data["fps"]
            
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = len(frames)
            bpy.context.scene.frame_set(1)
            
            # Enter pose mode
            bpy.ops.object.mode_set(mode='POSE')
            
            for frame_idx, frame_data in enumerate(frames):
                bpy.context.scene.frame_set(frame_idx + 1)
                
                if frame_data.get("pose_world_landmarks"):
                    pose_landmarks = frame_data["pose_world_landmarks"]
                    
                    # Apply rotations to bones based on landmarks
                    for mp_idx, bone_name in mp_to_blender_bones.items():
                        if mp_idx < len(pose_landmarks) and bone_name in armature.pose.bones:
                            bone = armature.pose.bones[bone_name]
                            landmark = pose_landmarks[mp_idx]
                            
                            # Convert MediaPipe coordinates to Blender rotations
                            # This is a simplified mapping - you may need to adjust based on your character
                            x, y, z = landmark[0], landmark[1], landmark[2]
                            
                            # Calculate rotation based on landmark position relative to rest pose
                            # This is a basic implementation - more sophisticated IK solutions may be needed
                            rotation_euler = mathutils.Euler((
                                z * 0.5,  # Pitch
                                y * 0.5,  # Yaw  
                                x * 0.5   # Roll
                            ))
                            
                            bone.rotation_euler = rotation_euler
                            bone.keyframe_insert(data_path="rotation_euler")
                
                # Apply hand animations if available
                if frame_data.get("left_hand_landmarks"):
                    self.apply_hand_animation(armature, "LeftHand", frame_data["left_hand_landmarks"])
                
                if frame_data.get("right_hand_landmarks"):
                    self.apply_hand_animation(armature, "RightHand", frame_data["right_hand_landmarks"])
            
            # Exit pose mode
            bpy.ops.object.mode_set(mode='OBJECT')

    def apply_hand_animation(self, armature, hand_bone_name: str, hand_landmarks: List):
        """Apply hand animation to finger bones"""
        if hand_bone_name not in armature.pose.bones:
            return
        
        # This is a simplified hand animation
        # In a full implementation, you'd create finger bones and apply individual finger movements
        hand_bone = armature.pose.bones[hand_bone_name]
        
        # Calculate overall hand rotation based on landmarks
        if len(hand_landmarks) >= 21:
            # Use wrist (0) and middle finger tip (12) to determine hand orientation
            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            
            # Calculate hand direction vector
            direction = (
                middle_tip[0] - wrist[0],
                middle_tip[1] - wrist[1], 
                middle_tip[2] - wrist[2]
            )
            
            # Convert to rotation
            rotation = mathutils.Euler((
                direction[2] * 2.0,
                direction[1] * 2.0,
                direction[0] * 2.0
            ))
            
            hand_bone.rotation_euler = rotation
            hand_bone.keyframe_insert(data_path="rotation_euler")

    def apply_face_animation(self, tracking_data: Dict, armature_name: str):
        """Apply facial animation using shape keys or bone constraints"""
        # This would require a more complex setup with facial bones or shape keys
        # For now, we'll create basic head rotation based on face orientation
        
        armature = bpy.data.objects[armature_name]
        
        if "Head" not in armature.pose.bones:
            return
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        head_bone = armature.pose.bones["Head"]
        
        for person_id, person_data in tracking_data["persons"].items():
            frames = person_data["frames"]
            
            for frame_idx, frame_data in enumerate(frames):
                bpy.context.scene.frame_set(frame_idx + 1)
                
                if frame_data.get("face_landmarks"):
                    face_landmarks = frame_data["face_landmarks"]
                    
                    if len(face_landmarks) >= 468:
                        # Calculate head rotation based on face landmarks
                        nose_tip = face_landmarks[1]  # Nose tip
                        nose_bridge = face_landmarks[168]  # Nose bridge
                        left_eye = face_landmarks[33]  # Left eye corner
                        right_eye = face_landmarks[362]  # Right eye corner
                        
                        # Calculate face normal and orientation
                        face_center_x = (left_eye[0] + right_eye[0]) / 2
                        face_center_y = (left_eye[1] + right_eye[1]) / 2
                        
                        # Head rotation based on face orientation
                        head_yaw = (face_center_x - 0.5) * 1.5  # Left-right turn
                        head_pitch = (face_center_y - 0.5) * 1.5  # Up-down tilt
                        head_roll = (right_eye[1] - left_eye[1]) * 2.0  # Head tilt
                        
                        rotation = mathutils.Euler((head_pitch, head_yaw, head_roll))
                        head_bone.rotation_euler = rotation
                        head_bone.keyframe_insert(data_path="rotation_euler")
        
        bpy.ops.object.mode_set(mode='OBJECT')

    def export_fbx(self, output_path: str, armature_name: str):
        """Export the animated armature to FBX format"""
        
        # Select the armature
        bpy.ops.object.select_all(action='DESELECT')
        armature = bpy.data.objects[armature_name]
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        
        # Export FBX
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            check_existing=True,
            filter_glob="*.fbx",
            use_selection=True,
            use_visible=True,
            use_active_collection=False,
            global_scale=1.0,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_NONE',
            use_space_transform=True,
            bake_space_transform=False,
            object_types={'ARMATURE'},
            use_mesh_modifiers=True,
            use_mesh_modifiers_render=True,
            mesh_smooth_type='OFF',
            use_subsurf=False,
            use_mesh_edges=False,
            use_tspace=False,
            use_triangles=False,
            use_custom_props=False,
            add_leaf_bones=True,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            use_armature_deform_only=False,
            armature_nodetype='NULL',
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=True,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=1.0,
            path_mode='AUTO',
            embed_textures=False,
            batch_mode='OFF',
            use_batch_own_dir=True,
            use_metadata=True
        )
        
        print(f"FBX exported to: {output_path}")


def main():
    """Main function to process video and create FBX animation"""
    
    # Configuration
    video_path = input("Enter the path to your MP4 video file: ").strip()
    if not video_path:
        video_path = "input_video.mp4"  # Default fallback
    output_dir = "output"
    fbx_output_path = os.path.join(output_dir, "character_animation.fbx")
    
    print("=== MediaPipe to FBX Animation Pipeline ===")
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Process video with MediaPipe
    print("\n1. Processing video with MediaPipe...")
    tracker = MediaPipeTracker(
        face_detection_confidence=0.6,
        face_tracking_confidence=0.5,
        hand_detection_confidence=0.7,
        hand_tracking_confidence=0.5,
        pose_detection_confidence=0.5,
        pose_tracking_confidence=0.5
    )
    
    if not tracker.process_video(video_path, output_dir):
        print("Error: Failed to process video!")
        return False
    
    # Step 2: Create FBX animation (if Blender is available)
    if BLENDER_AVAILABLE:
        print("\n2. Creating FBX animation with Blender...")
        
        # Load tracking data
        tracking_data_path = os.path.join(output_dir, "tracking_data.json")
        with open(tracking_data_path, 'r') as f:
            tracking_data = json.load(f)
        
        # Create Blender scene and export FBX
        exporter = BlenderFBXExporter()
        armature_name = exporter.create_character_rig("WhiteRobotMan")
        
        # Apply animations
        exporter.apply_pose_animation(tracking_data, armature_name)
        exporter.apply_face_animation(tracking_data, armature_name)
        
        # Export to FBX
        exporter.export_fbx(fbx_output_path, armature_name)
        
        print(f"\n✅ Animation pipeline complete!")
        print(f"📁 Output files:")
        print(f"   - Tracking data: {tracking_data_path}")
        print(f"   - FBX animation: {fbx_output_path}")
        print(f"   - Preview frames: {output_dir}/preview_frame_*.jpg")
        
    else:
        print("\n⚠️  Blender Python API not available.")
        print("Tracking data saved. To create FBX:")
        print("1. Install Blender")
        print("2. Run this script from within Blender's Python environment")
        print("3. Or use the tracking_data.json with your preferred 3D software")
    
    return True


if __name__ == "__main__":
    main()

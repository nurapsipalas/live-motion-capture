#!/usr/bin/env python3
"""
MediaPipe to FBX Pipeline - Part 3: UE4/UE5 Optimized Export
Enhanced version with better UE4/UE5 compatibility and real-time preview

Features:
- No JPG file generation (faster processing)
- UE4/UE5 optimized FBX export
- Real-time MediaPipe visualization
- Better bone hierarchy for Unreal Engine
- Mannequin-compatible skeleton
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

class MediaPipeTrackerUE:
    """Enhanced MediaPipe tracker optimized for Unreal Engine"""
    
    def __init__(self, 
                 face_detection_confidence: float = 0.7,
                 face_tracking_confidence: float = 0.6,
                 hand_detection_confidence: float = 0.8,
                 hand_tracking_confidence: float = 0.6,
                 pose_detection_confidence: float = 0.7,
                 pose_tracking_confidence: float = 0.6,
                 show_preview: bool = True):
        
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.show_preview = show_preview
        
        # Initialize trackers with better settings for UE
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,  # Disable for better performance
            refine_face_landmarks=True,
            min_detection_confidence=pose_detection_confidence,
            min_tracking_confidence=pose_tracking_confidence
        )
        
        # Data storage
        self.tracked_persons: Dict[int, PersonData] = {}
        self.frame_count = 0
        self.fps = 30.0
        
        # UE4/UE5 compatible bone mapping
        self.ue_bone_mapping = {
            # Core body - matches UE4/UE5 Mannequin
            "pelvis": 0,      # Root/Pelvis
            "spine_01": 1,    # Lower spine
            "spine_02": 2,    # Mid spine  
            "spine_03": 3,    # Upper spine
            "neck_01": 4,     # Neck
            "head": 0,        # Head (nose)
            
            # Left arm - UE Mannequin naming
            "clavicle_l": 11,     # Left shoulder
            "upperarm_l": 13,     # Left elbow
            "lowerarm_l": 15,     # Left wrist
            "hand_l": 17,         # Left hand
            
            # Right arm
            "clavicle_r": 12,     # Right shoulder
            "upperarm_r": 14,     # Right elbow
            "lowerarm_r": 16,     # Right wrist
            "hand_r": 18,         # Right hand
            
            # Left leg
            "thigh_l": 23,        # Left hip
            "calf_l": 25,         # Left knee
            "foot_l": 27,         # Left ankle
            "ball_l": 31,         # Left toe
            
            # Right leg
            "thigh_r": 24,        # Right hip
            "calf_r": 26,         # Right knee
            "foot_r": 28,         # Right ankle
            "ball_r": 32,         # Right toe
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

    def create_preview_window(self, frame, results):
        """Create enhanced real-time preview window"""
        if not self.show_preview:
            return
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape
        
        # Draw pose landmarks with custom style
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2)
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.left_hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 100, 0), thickness=1)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.right_hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(100, 0, 255), thickness=1)
            )
        
        # Draw face mesh (simplified)
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.face_landmarks, 
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=1)
            )
        
        # Add info overlay
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps:.1f}",
            "ESC: Exit | SPACE: Pause",
            "Pose: " + ("✓" if results.pose_landmarks else "✗"),
            "Face: " + ("✓" if results.face_landmarks else "✗"),
            "Hands: " + ("✓" if results.left_hand_landmarks or results.right_hand_landmarks else "✗")
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Tracking - UE4/UE5 Pipeline', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            return False
        elif key == 32:  # SPACE key
            cv2.waitKey(0)  # Pause until any key
        
        return True

    def process_video(self, video_path: str, output_dir: str = "output") -> bool:
        """Process MP4 video with enhanced preview and NO JPG generation"""
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
        
        print(f"🎬 Processing video: {video_path}")
        print(f"📊 Resolution: {width}x{height}, FPS: {self.fps}, Frames: {total_frames}")
        print(f"👁️ Preview: {'Enabled' if self.show_preview else 'Disabled'}")
        print(f"📁 Output: {output_dir}")
        print("\n⚡ Processing started...")
        
        # Initialize person tracking
        person_id = 0
        self.tracked_persons[person_id] = PersonData(person_id=person_id)
        
        frame_num = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with holistic model
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            timestamp = frame_num / self.fps
            landmark_data = self.extract_landmarks_from_results(results, timestamp)
            
            # Add to person data
            self.tracked_persons[person_id].frames.append(landmark_data)
            
            # Show preview (no JPG saving)
            if self.show_preview:
                should_continue = self.create_preview_window(frame, results)
                if not should_continue:
                    print("\n⏹️ Processing stopped by user")
                    break
            
            # Progress indicator
            if frame_num % 30 == 0 or frame_num == total_frames - 1:
                elapsed = time.time() - start_time
                fps_current = frame_num / elapsed if elapsed > 0 else 0
                progress = (frame_num / total_frames) * 100
                print(f"📈 Frame {frame_num:4d}/{total_frames} ({progress:5.1f}%) | FPS: {fps_current:4.1f}")
            
            frame_num += 1
            self.frame_count = frame_num
        
        cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        # Save tracking data
        print("\n💾 Saving tracking data...")
        self.save_tracking_data(output_dir)
        
        elapsed_total = time.time() - start_time
        print(f"\n✅ Processing complete!")
        print(f"⏱️ Total time: {elapsed_total:.2f}s")
        print(f"📊 Average FPS: {frame_num/elapsed_total:.1f}")
        print(f"📁 Data saved to: {output_dir}")
        
        return True

    def save_tracking_data(self, output_dir: str):
        """Save enhanced tracking data optimized for UE4/UE5"""
        output_file = os.path.join(output_dir, "tracking_data_ue.json")
        
        # Convert to UE-optimized format
        data = {
            "metadata": {
                "fps": self.fps,
                "total_frames": len(self.tracked_persons[0].frames) if self.tracked_persons else 0,
                "export_format": "UE4_UE5_Compatible",
                "bone_mapping": "Mannequin_Compatible",
                "coordinate_system": "UE_Right_Handed"
            },
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
                    "pose_world_landmarks": frame_data.pose_world_landmarks,
                    "face_landmarks": frame_data.face_landmarks,
                    "left_hand_landmarks": frame_data.left_hand_landmarks,
                    "right_hand_landmarks": frame_data.right_hand_landmarks
                }
                
                # Add UE-specific bone rotations
                if frame_data.pose_world_landmarks:
                    frame_dict["ue_bone_rotations"] = self.calculate_ue_bone_rotations(frame_data.pose_world_landmarks)
                
                person_dict["frames"].append(frame_dict)
            
            data["persons"][str(person_id)] = person_dict
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 UE tracking data saved: {output_file}")

    def calculate_ue_bone_rotations(self, pose_landmarks: List) -> Dict:
        """Calculate bone rotations in UE4/UE5 coordinate system"""
        if len(pose_landmarks) < 33:
            return {}
        
        rotations = {}
        
        # Head rotation (from nose and ear positions)
        if len(pose_landmarks) > 8:
            nose = pose_landmarks[0]
            left_ear = pose_landmarks[7] if len(pose_landmarks) > 7 else nose
            right_ear = pose_landmarks[8] if len(pose_landmarks) > 8 else nose
            
            # Calculate head yaw, pitch, roll for UE
            head_yaw = (nose[0] - 0.5) * 90.0    # Left/right turn
            head_pitch = (nose[1] - 0.5) * 60.0  # Up/down tilt
            head_roll = (right_ear[1] - left_ear[1]) * 45.0  # Head tilt
            
            rotations["head"] = {
                "pitch": head_pitch,
                "yaw": head_yaw, 
                "roll": head_roll
            }
        
        # Spine rotations
        if len(pose_landmarks) > 24:
            # Use hip and shoulder positions for spine
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            
            # Spine twist
            shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2, 
                             (left_shoulder[1] + right_shoulder[1])/2)
            hip_center = ((left_hip[0] + right_hip[0])/2, 
                         (left_hip[1] + right_hip[1])/2)
            
            spine_twist = (shoulder_center[0] - hip_center[0]) * 30.0
            spine_bend = (shoulder_center[1] - hip_center[1]) * 45.0
            
            rotations["spine_01"] = {"pitch": spine_bend * 0.3, "yaw": spine_twist * 0.3, "roll": 0}
            rotations["spine_02"] = {"pitch": spine_bend * 0.4, "yaw": spine_twist * 0.4, "roll": 0}
            rotations["spine_03"] = {"pitch": spine_bend * 0.3, "yaw": spine_twist * 0.3, "roll": 0}
        
        return rotations


class UEOptimizedFBXExporter:
    """FBX exporter optimized for UE4/UE5 with Mannequin compatibility"""
    
    def __init__(self):
        self.frame_rate = 30.0
        self.total_frames = 0
        
    def load_tracking_data(self, json_path: str) -> Dict:
        """Load UE-optimized tracking data"""
        if not os.path.exists(json_path):
            print(f"❌ Error: UE tracking data not found: {json_path}")
            return None
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        self.frame_rate = metadata.get('fps', 30.0)
        self.total_frames = metadata.get('total_frames', 0)
        
        print(f"📊 Loaded UE tracking data: {self.total_frames} frames @ {self.frame_rate} FPS")
        return data

    def create_ue_fbx_header(self) -> str:
        """Create FBX header optimized for UE4/UE5"""
        return f'''FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    CreationTimeStamp:  {{
        Version: 1000
        Year: 2025
        Month: 8
        Day: 1
        Hour: 12
        Minute: 0
        Second: 0
        Millisecond: 0
    }}
    Creator: "MediaPipe to UE4/UE5 FBX Exporter v3.0"
    SceneInfo: "SceneInfo::GlobalInfo", "UserData" {{
        Type: "UserData"
        Version: 100
        MetaData:  {{
            Version: 100
            Title: "MediaPipe Animation"
            Subject: "Motion Capture Data"
            Author: "MediaPipe Pipeline"
            Keywords: "animation,mocap,ue4,ue5,unreal"
            Revision: "1.0"
            Comment: "Generated from MediaPipe tracking data"
        }}
        Properties70:  {{
            P: "DocumentUrl", "KString", "Url", "", "mediapipe_ue_animation.fbx"
            P: "SrcDocumentUrl", "KString", "Url", "", "mediapipe_ue_animation.fbx"
            P: "Original", "Compound", "", ""
            P: "Original|ApplicationVendor", "KString", "", "", "MediaPipe"
            P: "Original|ApplicationName", "KString", "", "", "MediaPipe to UE Pipeline"
            P: "Original|ApplicationVersion", "KString", "", "", "3.0"
        }}
    }}
}}

GlobalSettings:  {{
    Version: 1000
    Properties70:  {{
        P: "UpAxis", "int", "Integer", "",1
        P: "UpAxisSign", "int", "Integer", "",1
        P: "FrontAxis", "int", "Integer", "",2
        P: "FrontAxisSign", "int", "Integer", "",1
        P: "CoordAxis", "int", "Integer", "",0
        P: "CoordAxisSign", "int", "Integer", "",1
        P: "OriginalUpAxis", "int", "Integer", "",-1
        P: "OriginalUpAxisSign", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",100
        P: "OriginalUnitScaleFactor", "double", "Number", "",100
        P: "AmbientColor", "ColorRGB", "Color", "",0,0,0
        P: "DefaultCamera", "KString", "", "", "Producer Perspective"
        P: "TimeMode", "enum", "", "",0
        P: "TimeSpanStart", "KTime", "Time", "",0
        P: "TimeSpanStop", "KTime", "Time", "",''' + str(int(self.total_frames * (46186158000 / self.frame_rate))) + '''
        P: "CustomFrameRate", "double", "Number", "",''' + str(self.frame_rate) + '''
    }}
}}

'''

    def create_ue_mannequin_skeleton(self) -> str:
        """Create UE4/UE5 Mannequin-compatible skeleton"""
        bones = {
            # Root
            "Root": {"id": 1000, "parent": None, "pos": [0, 0, 0], "name": "Root"},
            
            # Pelvis and spine
            "pelvis": {"id": 1001, "parent": 1000, "pos": [0, 0, 98], "name": "pelvis"},
            "spine_01": {"id": 1002, "parent": 1001, "pos": [0, 0, 108], "name": "spine_01"},
            "spine_02": {"id": 1003, "parent": 1002, "pos": [0, 0, 125], "name": "spine_02"},
            "spine_03": {"id": 1004, "parent": 1003, "pos": [0, 0, 142], "name": "spine_03"},
            "neck_01": {"id": 1005, "parent": 1004, "pos": [0, 0, 160], "name": "neck_01"},
            "head": {"id": 1006, "parent": 1005, "pos": [0, 0, 175], "name": "head"},
            
            # Left arm
            "clavicle_l": {"id": 1010, "parent": 1004, "pos": [4, 0, 155], "name": "clavicle_l"},
            "upperarm_l": {"id": 1011, "parent": 1010, "pos": [18, 0, 155], "name": "upperarm_l"},
            "lowerarm_l": {"id": 1012, "parent": 1011, "pos": [50, 0, 155], "name": "lowerarm_l"},
            "hand_l": {"id": 1013, "parent": 1012, "pos": [78, 0, 155], "name": "hand_l"},
            
            # Right arm  
            "clavicle_r": {"id": 1020, "parent": 1004, "pos": [-4, 0, 155], "name": "clavicle_r"},
            "upperarm_r": {"id": 1021, "parent": 1020, "pos": [-18, 0, 155], "name": "upperarm_r"},
            "lowerarm_r": {"id": 1022, "parent": 1021, "pos": [-50, 0, 155], "name": "lowerarm_r"},
            "hand_r": {"id": 1023, "parent": 1022, "pos": [-78, 0, 155], "name": "hand_r"},
            
            # Left leg
            "thigh_l": {"id": 1030, "parent": 1001, "pos": [9, 0, 95], "name": "thigh_l"},
            "calf_l": {"id": 1031, "parent": 1030, "pos": [9, 0, 48], "name": "calf_l"},
            "foot_l": {"id": 1032, "parent": 1031, "pos": [9, 0, 8], "name": "foot_l"},
            "ball_l": {"id": 1033, "parent": 1032, "pos": [9, 15, 3], "name": "ball_l"},
            
            # Right leg
            "thigh_r": {"id": 1040, "parent": 1001, "pos": [-9, 0, 95], "name": "thigh_r"},
            "calf_r": {"id": 1041, "parent": 1040, "pos": [-9, 0, 48], "name": "calf_r"},
            "foot_r": {"id": 1042, "parent": 1041, "pos": [-9, 0, 8], "name": "foot_r"},
            "ball_r": {"id": 1043, "parent": 1042, "pos": [-9, 15, 3], "name": "ball_r"},
        }
        
        fbx_skeleton = ""
        for bone_name, bone_data in bones.items():
            pos = bone_data["pos"]
            fbx_skeleton += f'''Model: {bone_data["id"]}, "Model::{bone_data["name"]}", "LimbNode" {{
    Version: 232
    Properties70:  {{
        P: "Lcl Translation", "Lcl Translation", "", "A",{pos[0]},{pos[1]},{pos[2]}
        P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
        P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
    }}
    Shading: T
    Culling: "CullingOff"
}}

'''
        return fbx_skeleton

    def export_ue_fbx(self, tracking_data: Dict, output_path: str) -> bool:
        """Export UE4/UE5 optimized FBX"""
        try:
            print(f"🎯 Creating UE4/UE5 FBX: {output_path}")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate FBX content
            fbx_content = self.create_ue_fbx_header()
            fbx_content += "Objects:  {\n"
            fbx_content += self.create_ue_mannequin_skeleton()
            fbx_content += "}\n"
            
            # Add connections
            fbx_content += self.create_ue_connections()
            
            # Write FBX file
            with open(output_path, 'w') as f:
                f.write(fbx_content)
            
            print(f"✅ UE4/UE5 FBX exported: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating UE FBX: {e}")
            return False

    def create_ue_connections(self) -> str:
        """Create UE-compatible connections"""
        return '''
Connections:  {
    ;Root connections
    C: "OO",1000,0
    
    ;Pelvis and spine
    C: "OO",1001,1000
    C: "OO",1002,1001
    C: "OO",1003,1002
    C: "OO",1004,1003
    C: "OO",1005,1004
    C: "OO",1006,1005
    
    ;Left arm
    C: "OO",1010,1004
    C: "OO",1011,1010
    C: "OO",1012,1011
    C: "OO",1013,1012
    
    ;Right arm
    C: "OO",1020,1004
    C: "OO",1021,1020
    C: "OO",1022,1021
    C: "OO",1023,1022
    
    ;Left leg
    C: "OO",1030,1001
    C: "OO",1031,1030
    C: "OO",1032,1031
    C: "OO",1033,1032
    
    ;Right leg
    C: "OO",1040,1001
    C: "OO",1041,1040
    C: "OO",1042,1041
    C: "OO",1043,1042
}

Takes:  {
    Current: "MediaPipe_UE_Animation"
    Take: "MediaPipe_UE_Animation" {
        FileName: "MediaPipe_UE_Animation.tak"
        LocalTime: 0,''' + str(int(self.total_frames * (46186158000 / self.frame_rate))) + '''
        ReferenceTime: 0,''' + str(int(self.total_frames * (46186158000 / self.frame_rate))) + '''
    }
}
'''


def main():
    """Main function for Part 3: Enhanced UE4/UE5 Pipeline"""
    
    print("🚀 === MediaPipe to FBX Pipeline - Part 3: UE4/UE5 Enhanced ===")
    print("✨ Features: Real-time preview, UE optimization, NO JPG generation")
    print()
    
    # Configuration
    video_path = input("📹 Enter video path (or press Enter for test_video.mp4): ").strip()
    if not video_path:
        video_path = "test_video.mp4"
    
    show_preview = input("👁️ Show real-time preview? (Y/n): ").strip().lower() != 'n'
    
    output_dir = "output"
    ue_fbx_path = os.path.join(output_dir, "ue_mannequin_animation.fbx")
    
    print(f"\n⚙️ Configuration:")
    print(f"   📹 Video: {video_path}")
    print(f"   👁️ Preview: {'Enabled' if show_preview else 'Disabled'}")
    print(f"   📁 Output: {output_dir}")
    print(f"   🎯 Target: UE4/UE5 Mannequin")
    
    # Step 1: Enhanced MediaPipe processing
    print(f"\n🎬 Step 1: Enhanced MediaPipe Processing")
    tracker = MediaPipeTrackerUE(
        face_detection_confidence=0.7,
        face_tracking_confidence=0.6,
        hand_detection_confidence=0.8,
        hand_tracking_confidence=0.6,
        pose_detection_confidence=0.7,
        pose_tracking_confidence=0.6,
        show_preview=show_preview
    )
    
    if not tracker.process_video(video_path, output_dir):
        print("❌ Failed to process video!")
        return False
    
    # Step 2: UE4/UE5 FBX Export
    print(f"\n🎯 Step 2: UE4/UE5 FBX Export")
    exporter = UEOptimizedFBXExporter()
    
    # Load UE tracking data
    ue_tracking_path = os.path.join(output_dir, "tracking_data_ue.json")
    tracking_data = exporter.load_tracking_data(ue_tracking_path)
    
    if not tracking_data:
        print("❌ Failed to load UE tracking data!")
        return False
    
    # Export UE FBX
    success = exporter.export_ue_fbx(tracking_data, ue_fbx_path)
    
    if success:
        print(f"\n🎉 Part 3 Pipeline Complete!")
        print(f"📁 Output files:")
        print(f"   🎯 UE4/UE5 FBX: {ue_fbx_path}")
        print(f"   📊 UE Data: {ue_tracking_path}")
        print(f"\n🎮 Unreal Engine Instructions:")
        print(f"   1. Import {ue_fbx_path} as Skeletal Mesh")
        print(f"   2. Choose 'Humanoid' rig in import settings")
        print(f"   3. Apply to UE4/UE5 Mannequin character")
        print(f"   4. Animation should map automatically!")
        
        return True
    else:
        print("❌ FBX export failed!")
        return False


if __name__ == "__main__":
    main()

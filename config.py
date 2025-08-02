"""
Configuration settings for MediaPipe to FBX Animation Pipeline
"""

import os

class Config:
    """Configuration class for the animation pipeline"""
    
    # Video processing settings
    VIDEO_INPUT_PATH = "input_video.mp4"
    OUTPUT_DIRECTORY = "output"
    
    # MediaPipe confidence thresholds
    FACE_DETECTION_CONFIDENCE = 0.6
    FACE_TRACKING_CONFIDENCE = 0.5
    HAND_DETECTION_CONFIDENCE = 0.7
    HAND_TRACKING_CONFIDENCE = 0.5
    POSE_DETECTION_CONFIDENCE = 0.5
    POSE_TRACKING_CONFIDENCE = 0.5
    
    # Processing settings
    MAX_NUM_FACES = 3
    MAX_NUM_HANDS = 6  # 3 people × 2 hands
    ENABLE_FACE_REFINEMENT = True
    ENABLE_POSE_SEGMENTATION = True
    POSE_MODEL_COMPLEXITY = 2  # 0, 1, or 2 (higher = more accurate but slower)
    
    # Output settings
    SAVE_PREVIEW_FRAMES = True
    PREVIEW_FRAME_INTERVAL = 30  # Save every N frames
    EXPORT_JSON_DATA = True
    
    # Character rigging settings
    CHARACTER_NAME = "WhiteRobotMan"
    CHARACTER_SCALE = 1.0
    
    # Bone mapping for different character types
    BONE_MAPPINGS = {
        "humanoid": {
            # MediaPipe pose landmark index -> Bone name
            0: "Head",           # Nose
            11: "LeftShoulder",  # Left shoulder
            12: "RightShoulder", # Right shoulder
            13: "LeftArm",       # Left elbow
            14: "RightArm",      # Right elbow
            15: "LeftForearm",   # Left wrist
            16: "RightForearm",  # Right wrist
            23: "LeftThigh",     # Left hip
            24: "RightThigh",    # Right hip
            25: "LeftShin",      # Left knee
            26: "RightShin",     # Right knee
            27: "LeftFoot",      # Left ankle
            28: "RightFoot",     # Right ankle
        },
        "robot": {
            # Custom mapping for robot characters
            0: "HeadJoint",
            11: "LeftShoulderServo",
            12: "RightShoulderServo",
            13: "LeftElbowServo",
            14: "RightElbowServo",
            15: "LeftWristServo",
            16: "RightWristServo",
            23: "LeftHipServo",
            24: "RightHipServo",
            25: "LeftKneeServo",
            26: "RightKneeServo",
            27: "LeftAnkleServo",
            28: "RightAnkleServo",
        }
    }
    
    # FBX export settings
    FBX_SETTINGS = {
        "global_scale": 1.0,
        "apply_unit_scale": True,
        "bake_animation": True,
        "bake_step": 1.0,
        "include_armature": True,
        "include_mesh": False,  # Set to True if you have character mesh
        "primary_bone_axis": 'Y',
        "secondary_bone_axis": 'X',
    }
    
    # Face expression mapping for detailed facial animation
    FACE_EXPRESSIONS = {
        "left_eye_blink": "eyeBlinkLeft",
        "right_eye_blink": "eyeBlinkRight", 
        "mouth_open": "jawOpen",
        "smile": "mouthSmileLeft",  # Can map to multiple shape keys
        "left_eyebrow_raise": "browInnerUp",
        "right_eyebrow_raise": "browInnerUp",
    }
    
    # Hand gesture mapping
    HAND_GESTURES = {
        "finger_0_curl": "thumbCurl",
        "finger_1_curl": "indexCurl",
        "finger_2_curl": "middleCurl", 
        "finger_3_curl": "ringCurl",
        "finger_4_curl": "pinkyCurl",
        "hand_openness": "handOpen",
        "fist": "fistClosed",
    }
    
    @classmethod
    def get_bone_mapping(cls, character_type: str = "humanoid"):
        """Get bone mapping for specified character type"""
        return cls.BONE_MAPPINGS.get(character_type, cls.BONE_MAPPINGS["humanoid"])
    
    @classmethod
    def create_output_paths(cls):
        """Create all necessary output directories"""
        paths = {
            "output_dir": cls.OUTPUT_DIRECTORY,
            "json_data": os.path.join(cls.OUTPUT_DIRECTORY, "tracking_data.json"),
            "fbx_file": os.path.join(cls.OUTPUT_DIRECTORY, f"{cls.CHARACTER_NAME}_animation.fbx"),
            "preview_frames": os.path.join(cls.OUTPUT_DIRECTORY, "preview_frames"),
            "blender_file": os.path.join(cls.OUTPUT_DIRECTORY, f"{cls.CHARACTER_NAME}_scene.blend"),
        }
        
        # Create directories
        for path_name, path_value in paths.items():
            if path_name.endswith("_dir") or path_name == "preview_frames":
                os.makedirs(path_value, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(path_value), exist_ok=True)
        
        return paths

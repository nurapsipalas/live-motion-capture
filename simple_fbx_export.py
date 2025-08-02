#!/usr/bin/env python3
"""
Simple FBX Animation Generator
Creates basic FBX files from MediaPipe tracking data without requiring Blender

This script creates a simple text-based FBX file that can be imported into
most 3D software including Unreal Engine, Unity, Maya, and 3ds Max.
"""

import json
import os
import math
from typing import Dict, List, Tuple

class SimpleFBXExporter:
    """Create basic FBX files from tracking data"""
    
    def __init__(self):
        self.frame_rate = 30.0
        self.total_frames = 0
        
    def load_tracking_data(self, json_path: str) -> Dict:
        """Load MediaPipe tracking data"""
        if not os.path.exists(json_path):
            print(f"Error: Tracking data not found: {json_path}")
            return None
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.frame_rate = data.get('fps', 30.0)
        self.total_frames = data.get('total_frames', 0)
        
        print(f"Loaded tracking data: {self.total_frames} frames @ {self.frame_rate} FPS")
        return data
    
    def create_fbx_header(self) -> str:
        """Create FBX file header"""
        return f"""FBXHeaderExtension:  {{
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
    Creator: "MediaPipe to FBX Converter"
}}

; Document Description
;------------------------------------------------------------------

Document: "Scene", "Scene" {{
    Properties70:  {{
        P: "SourceObject", "object", "", ""
        P: "ActiveAnimStackName", "KString", "", "", "MediaPipe Animation"
    }}
}}

; Document References
;------------------------------------------------------------------

References:  {{
}}

; Object definitions
;------------------------------------------------------------------

Definitions:  {{
    Version: 100
    Count: 4
    
    ObjectType: "GlobalSettings" {{
        Count: 1
    }}
    
    ObjectType: "Model" {{
        Count: 1
        PropertyTemplate: "FbxNode" {{
            Properties70:  {{
                P: "QuaternionInterpolate", "enum", "", "",0
                P: "RotationOffset", "Vector3D", "Vector", "",0,0,0
                P: "RotationPivot", "Vector3D", "Vector", "",0,0,0
                P: "ScalingOffset", "Vector3D", "Vector", "",0,0,0
                P: "ScalingPivot", "Vector3D", "Vector", "",0,0,0
                P: "TranslationActive", "bool", "", "",0
                P: "TranslationMin", "Vector3D", "Vector", "",0,0,0
                P: "TranslationMax", "Vector3D", "Vector", "",0,0,0
                P: "TranslationMinX", "bool", "", "",0
                P: "TranslationMinY", "bool", "", "",0
                P: "TranslationMinZ", "bool", "", "",0
                P: "TranslationMaxX", "bool", "", "",0
                P: "TranslationMaxY", "bool", "", "",0
                P: "TranslationMaxZ", "bool", "", "",0
                P: "RotationOrder", "enum", "", "",0
                P: "RotationSpaceForLimitOnly", "bool", "", "",0
                P: "RotationStiffnessX", "double", "Number", "",0
                P: "RotationStiffnessY", "double", "Number", "",0
                P: "RotationStiffnessZ", "double", "Number", "",0
                P: "AxisLen", "double", "Number", "",10
                P: "PreRotation", "Vector3D", "Vector", "",0,0,0
                P: "PostRotation", "Vector3D", "Vector", "",0,0,0
                P: "RotationActive", "bool", "", "",0
                P: "RotationMin", "Vector3D", "Vector", "",0,0,0
                P: "RotationMax", "Vector3D", "Vector", "",0,0,0
                P: "RotationMinX", "bool", "", "",0
                P: "RotationMinY", "bool", "", "",0
                P: "RotationMinZ", "bool", "", "",0
                P: "RotationMaxX", "bool", "", "",0
                P: "RotationMaxY", "bool", "", "",0
                P: "RotationMaxZ", "bool", "", "",0
                P: "InheritType", "enum", "", "",0
                P: "ScalingActive", "bool", "", "",0
                P: "ScalingMin", "Vector3D", "Vector", "",0,0,0
                P: "ScalingMax", "Vector3D", "Vector", "",1,1,1
                P: "ScalingMinX", "bool", "", "",0
                P: "ScalingMinY", "bool", "", "",0
                P: "ScalingMinZ", "bool", "", "",0
                P: "ScalingMaxX", "bool", "", "",0
                P: "ScalingMaxY", "bool", "", "",0
                P: "ScalingMaxZ", "bool", "", "",0
                P: "GeometricTranslation", "Vector3D", "Vector", "",0,0,0
                P: "GeometricRotation", "Vector3D", "Vector", "",0,0,0
                P: "GeometricScaling", "Vector3D", "Vector", "",1,1,1
                P: "MinDampRangeX", "double", "Number", "",0
                P: "MinDampRangeY", "double", "Number", "",0
                P: "MinDampRangeZ", "double", "Number", "",0
                P: "MaxDampRangeX", "double", "Number", "",0
                P: "MaxDampRangeY", "double", "Number", "",0
                P: "MaxDampRangeZ", "double", "Number", "",0
                P: "MinDampStrengthX", "double", "Number", "",0
                P: "MinDampStrengthY", "double", "Number", "",0
                P: "MinDampStrengthZ", "double", "Number", "",0
                P: "MaxDampStrengthX", "double", "Number", "",0
                P: "MaxDampStrengthY", "double", "Number", "",0
                P: "MaxDampStrengthZ", "double", "Number", "",0
                P: "PreferedAngleX", "double", "Number", "",0
                P: "PreferedAngleY", "double", "Number", "",0
                P: "PreferedAngleZ", "double", "Number", "",0
                P: "LookAtProperty", "object", "", ""
                P: "UpVectorProperty", "object", "", ""
                P: "Show", "bool", "", "",1
                P: "NegativePercentShapeSupport", "bool", "", "",1
                P: "DefaultAttributeIndex", "int", "Integer", "",-1
                P: "Freeze", "bool", "", "",0
                P: "LODBox", "bool", "", "",0
                P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
                P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
                P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
                P: "Visibility", "Visibility", "", "A",1
                P: "Visibility Inheritance", "Visibility Inheritance", "", "",1
            }}
        }}
    }}
}}

; Object properties
;------------------------------------------------------------------

Objects:  {{
    GlobalSettings: "GlobalSettings", "Global Settings" {{
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
            P: "TimeSpanStop", "KTime", "Time", "",""" + str(int(self.total_frames * (46186158000 / self.frame_rate))) + """
            P: "CustomFrameRate", "double", "Number", "",""" + str(self.frame_rate) + """
        }}
    }}
"""

    def create_bone_hierarchy(self) -> str:
        """Create basic bone structure"""
        bones = {
            "Root": {"id": 1001, "parent": None, "pos": [0, 0, 0]},
            "Hips": {"id": 1002, "parent": 1001, "pos": [0, 0, 100]},
            "Spine": {"id": 1003, "parent": 1002, "pos": [0, 0, 120]},
            "Head": {"id": 1004, "parent": 1003, "pos": [0, 0, 170]},
            "LeftShoulder": {"id": 1005, "parent": 1003, "pos": [20, 0, 160]},
            "LeftArm": {"id": 1006, "parent": 1005, "pos": [40, 0, 160]},
            "LeftHand": {"id": 1007, "parent": 1006, "pos": [70, 0, 140]},
            "RightShoulder": {"id": 1008, "parent": 1003, "pos": [-20, 0, 160]},
            "RightArm": {"id": 1009, "parent": 1008, "pos": [-40, 0, 160]},
            "RightHand": {"id": 1010, "parent": 1009, "pos": [-70, 0, 140]},
            "LeftLeg": {"id": 1011, "parent": 1002, "pos": [15, 0, 100]},
            "LeftFoot": {"id": 1012, "parent": 1011, "pos": [15, 0, 50]},
            "RightLeg": {"id": 1013, "parent": 1002, "pos": [-15, 0, 100]},
            "RightFoot": {"id": 1014, "parent": 1013, "pos": [-15, 0, 50]},
        }
        
        fbx_bones = ""
        for bone_name, bone_data in bones.items():
            pos = bone_data["pos"]
            fbx_bones += f'''    Model: {bone_data["id"]}, "Model::{bone_name}", "LimbNode" {{
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
        return fbx_bones

    def create_animation_curves(self, tracking_data: Dict) -> str:
        """Create animation curves from tracking data"""
        person_data = list(tracking_data["persons"].values())[0]
        frames = person_data["frames"]
        
        animation_curves = f"""    AnimationStack: 2000, "AnimStack::MediaPipe Animation", "" {{
        Properties70:  {{
            P: "Description", "KString", "", "", ""
            P: "LocalStart", "KTime", "Time", "",0
            P: "LocalStop", "KTime", "Time", "",""" + str(int(len(frames) * (46186158000 / self.frame_rate))) + """
            P: "ReferenceStart", "KTime", "Time", "",0
            P: "ReferenceStop", "KTime", "Time", "",""" + str(int(len(frames) * (46186158000 / self.frame_rate))) + """
        }}
    }}
    
    AnimationLayer: 2001, "AnimLayer::BaseLayer", "" {{
    }}
    
"""
        
        # Create animation curves for head rotation
        head_curve_x = "    AnimationCurve: 3001, \"AnimCurve::\", \"\" {\n        Default: 0\n        KeyVer: 4009\n        KeyTime: *" + str(len(frames)) + " {\n            a: "
        head_curve_y = "    AnimationCurve: 3002, \"AnimCurve::\", \"\" {\n        Default: 0\n        KeyVer: 4009\n        KeyTime: *" + str(len(frames)) + " {\n            a: "
        head_curve_z = "    AnimationCurve: 3003, \"AnimCurve::\", \"\" {\n        Default: 0\n        KeyVer: 4009\n        KeyTime: *" + str(len(frames)) + " {\n            a: "
        
        head_values_x = "        KeyValueFloat: *" + str(len(frames)) + " {\n            a: "
        head_values_y = "        KeyValueFloat: *" + str(len(frames)) + " {\n            a: "
        head_values_z = "        KeyValueFloat: *" + str(len(frames)) + " {\n            a: "
        
        for frame_idx, frame_data in enumerate(frames):
            time_stamp = int(frame_idx * (46186158000 / self.frame_rate))
            
            # Simple head rotation from pose landmarks
            rotation_x = 0.0
            rotation_y = 0.0  
            rotation_z = 0.0
            
            if frame_data.get("pose_world_landmarks"):
                landmarks = frame_data["pose_world_landmarks"]
                if len(landmarks) > 0:
                    # Use nose landmark for head rotation (index 0)
                    nose = landmarks[0]
                    rotation_x = (nose[2] - 0.5) * 30.0  # Convert to degrees
                    rotation_y = (nose[0] - 0.5) * 30.0
                    rotation_z = (nose[1] - 0.5) * 15.0
            
            head_curve_x += str(time_stamp)
            head_curve_y += str(time_stamp)
            head_curve_z += str(time_stamp)
            
            head_values_x += str(rotation_x)
            head_values_y += str(rotation_y)
            head_values_z += str(rotation_z)
            
            if frame_idx < len(frames) - 1:
                head_curve_x += ","
                head_curve_y += ","
                head_curve_z += ","
                head_values_x += ","
                head_values_y += ","
                head_values_z += ","
        
        head_curve_x += """\n        }
        KeyAttrFlags: *""" + str(len(frames)) + """ {
            a: """
        head_curve_y += """\n        }
        KeyAttrFlags: *""" + str(len(frames)) + """ {
            a: """
        head_curve_z += """\n        }
        KeyAttrFlags: *""" + str(len(frames)) + """ {
            a: """
            
        for i in range(len(frames)):
            head_curve_x += "24"
            head_curve_y += "24" 
            head_curve_z += "24"
            if i < len(frames) - 1:
                head_curve_x += ","
                head_curve_y += ","
                head_curve_z += ","
        
        head_curve_x += "\n        }\n" + head_values_x + "\n        }\n        KeyAttrDataFloat: *4 {\n            a: 0,0,0,0\n        }\n        KeyAttrRefCount: *" + str(len(frames)) + " {\n            a: "
        head_curve_y += "\n        }\n" + head_values_y + "\n        }\n        KeyAttrDataFloat: *4 {\n            a: 0,0,0,0\n        }\n        KeyAttrRefCount: *" + str(len(frames)) + " {\n            a: "
        head_curve_z += "\n        }\n" + head_values_z + "\n        }\n        KeyAttrDataFloat: *4 {\n            a: 0,0,0,0\n        }\n        KeyAttrRefCount: *" + str(len(frames)) + " {\n            a: "
        
        for i in range(len(frames)):
            head_curve_x += "1"
            head_curve_y += "1"
            head_curve_z += "1"
            if i < len(frames) - 1:
                head_curve_x += ","
                head_curve_y += ","
                head_curve_z += ","
        
        head_curve_x += "\n        }\n    }\n"
        head_curve_y += "\n        }\n    }\n"
        head_curve_z += "\n        }\n    }\n"
        
        animation_curves += head_curve_x + "\n" + head_curve_y + "\n" + head_curve_z + "\n"
        
        return animation_curves

    def create_connections(self) -> str:
        """Create object connections"""
        return """
; Object connections
;------------------------------------------------------------------

Connections:  {
    ;Model::Root, Model::RootNode
    C: "OO",1001,0
    
    ;Model::Hips, Model::Root
    C: "OO",1002,1001
    
    ;Model::Spine, Model::Hips
    C: "OO",1003,1002
    
    ;Model::Head, Model::Spine
    C: "OO",1004,1003
    
    ;Model::LeftShoulder, Model::Spine
    C: "OO",1005,1003
    
    ;Model::LeftArm, Model::LeftShoulder
    C: "OO",1006,1005
    
    ;Model::LeftHand, Model::LeftArm
    C: "OO",1007,1006
    
    ;Model::RightShoulder, Model::Spine
    C: "OO",1008,1003
    
    ;Model::RightArm, Model::RightShoulder
    C: "OO",1009,1008
    
    ;Model::RightHand, Model::RightArm
    C: "OO",1010,1009
    
    ;Model::LeftLeg, Model::Hips
    C: "OO",1011,1002
    
    ;Model::LeftFoot, Model::LeftLeg
    C: "OO",1012,1011
    
    ;Model::RightLeg, Model::Hips
    C: "OO",1013,1002
    
    ;Model::RightFoot, Model::RightLeg
    C: "OO",1014,1013
    
    ;AnimationStack::MediaPipe Animation, 
    C: "OO",2000,0
    
    ;AnimationLayer::BaseLayer, AnimationStack::MediaPipe Animation
    C: "OO",2001,2000
    
    ;AnimationCurve::, Model::Head
    C: "OP",3001,1004, "Lcl Rotation.X"
    C: "OP",3002,1004, "Lcl Rotation.Y"
    C: "OP",3003,1004, "Lcl Rotation.Z"
    
    ;AnimationCurve::, AnimationLayer::BaseLayer
    C: "OO",3001,2001
    C: "OO",3002,2001
    C: "OO",3003,2001
}

;Takes section
;------------------------------------------------------------------

Takes:  {
    Current: "MediaPipe Animation"
    Take: "MediaPipe Animation" {
        FileName: "MediaPipe Animation.tak"
        LocalTime: 0,{time_span}
        ReferenceTime: 0,{time_span}
    }
}
"""

    def export_fbx(self, tracking_data: Dict, output_path: str) -> bool:
        """Export tracking data to FBX file"""
        try:
            print(f"Creating FBX animation: {output_path}")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate FBX content
            fbx_content = self.create_fbx_header()
            fbx_content += self.create_bone_hierarchy()
            fbx_content += self.create_animation_curves(tracking_data)
            
            time_span = int(self.total_frames * (46186158000 / self.frame_rate))
            fbx_content += self.create_connections().replace("{time_span}", str(time_span))
            
            # Write FBX file
            with open(output_path, 'w') as f:
                f.write(fbx_content)
            
            print(f"✅ FBX export completed: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating FBX: {e}")
            return False


def main():
    """Main function to create FBX from tracking data"""
    
    print("=== Simple FBX Animation Generator ===")
    
    # File paths
    tracking_data_path = "output/tracking_data.json"
    fbx_output_path = "output/mediapipe_animation.fbx"
    
    # Check if tracking data exists
    if not os.path.exists(tracking_data_path):
        print(f"❌ Error: Tracking data not found at {tracking_data_path}")
        print("Please run mediapipe_to_fbx.py first to generate tracking data!")
        return False
    
    # Create exporter and process
    exporter = SimpleFBXExporter()
    
    # Load tracking data
    tracking_data = exporter.load_tracking_data(tracking_data_path)
    if not tracking_data:
        return False
    
    # Export FBX
    success = exporter.export_fbx(tracking_data, fbx_output_path)
    
    if success:
        print("\n🎉 FBX Animation Ready!")
        print(f"📁 File location: {fbx_output_path}")
        print("\n📋 How to use:")
        print("1. Import the FBX file into your 3D software:")
        print("   - Unreal Engine: Import as Skeletal Mesh + Animation")
        print("   - Unity: Drag into Assets, configure as Humanoid")
        print("   - Blender: File > Import > FBX")
        print("   - Maya/3ds Max: Import FBX")
        print("\n2. The animation contains:")
        print("   - Basic character skeleton")
        print("   - Head rotation based on MediaPipe tracking")
        print("   - Ready for retargeting to your character")
        print(f"\n3. Animation length: {exporter.total_frames} frames @ {exporter.frame_rate} FPS")
        
        return True
    else:
        return False


if __name__ == "__main__":
    main()

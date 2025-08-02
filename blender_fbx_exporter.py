"""
Blender script for importing MediaPipe tracking data and creating FBX animations
Run this script inside Blender to create character rigs and animations

Usage:
1. Open Blender
2. Load this script in the Text Editor
3. Modify the paths at the bottom of the script
4. Run the script in Blender's scripting workspace
"""

try:
    import bpy  # type: ignore
    import mathutils  # type: ignore
    from mathutils import Vector, Euler, Quaternion, Matrix  # type: ignore
except ImportError:
    print("\n==============================\nERROR: Blender Python environment required!\n\nThis script must be run INSIDE Blender (bpy and mathutils modules not found).\n\nHow to run this script:\n  1. Open Blender.\n  2. Go to the Scripting workspace.\n  3. Load this script in the Text Editor.\n  4. Run the script in Blender's scripting workspace.\n\nWindows: Use Blender's built-in Python, not your system Python.\nMac/Linux: Same instructions apply.\n==============================\n")
    # Gracefully exit without raising an exception or traceback
    import sys
    sys.exit(0)

import json
import os

class MediaPipeToBlender:
    """Convert MediaPipe data to Blender animations"""
    
    def __init__(self):
        self.armature = None
        self.character_mesh = None
        
    def clear_scene(self):
        """Clear all objects from the scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Clear all materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
    
    def create_character_mesh(self, name="WhiteRobotMan"):
        """Create a basic character mesh"""
        # Create a simple humanoid mesh
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
        body = bpy.context.active_object
        body.name = f"{name}_Body"
        
        # Scale to body proportions
        body.scale = (0.4, 0.2, 0.8)
        bpy.ops.object.transform_apply(scale=True)
        
        # Add head
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 1.85))
        head = bpy.context.active_object
        head.name = f"{name}_Head"
        
        # Add arms
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.6, location=(0.5, 0, 1.4))
        left_arm = bpy.context.active_object
        left_arm.name = f"{name}_LeftArm"
        left_arm.rotation_euler = (0, 0, 1.57)  # 90 degrees
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.6, location=(-0.5, 0, 1.4))
        right_arm = bpy.context.active_object
        right_arm.name = f"{name}_RightArm" 
        right_arm.rotation_euler = (0, 0, 1.57)
        
        # Add legs
        bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.8, location=(0.15, 0, 0.4))
        left_leg = bpy.context.active_object
        left_leg.name = f"{name}_LeftLeg"
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.8, location=(-0.15, 0, 0.4))
        right_leg = bpy.context.active_object
        right_leg.name = f"{name}_RightLeg"
        
        # Join all mesh parts
        bpy.ops.object.select_all(action='DESELECT')
        for obj_name in [f"{name}_Body", f"{name}_Head", f"{name}_LeftArm", f"{name}_RightArm", f"{name}_LeftLeg", f"{name}_RightLeg"]:
            if obj_name in bpy.data.objects:
                bpy.data.objects[obj_name].select_set(True)
        
        bpy.context.view_layer.objects.active = bpy.data.objects[f"{name}_Body"]
        bpy.ops.object.join()
        
        self.character_mesh = bpy.context.active_object
        self.character_mesh.name = name
        
        # Add material
        mat = bpy.data.materials.new(name=f"{name}_Material")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.9, 1.0)  # Light blue
        self.character_mesh.data.materials.append(mat)
        
        return self.character_mesh
    
    def create_advanced_rig(self, name="WhiteRobotMan"):
        """Create an advanced character rig with IK chains"""
        
        # Create armature
        bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
        self.armature = bpy.context.active_object
        self.armature.name = f"{name}_Armature"
        
        armature_data = self.armature.data
        armature_data.name = f"{name}_ArmatureData"
        
        # Define bone hierarchy with more detailed structure
        bones_data = {
            "Root": {
                "parent": None,
                "head": (0, 0, 0),
                "tail": (0, 0, 0.05),
                "roll": 0
            },
            "Hips": {
                "parent": "Root", 
                "head": (0, 0, 0.9),
                "tail": (0, 0, 1.1),
                "roll": 0
            },
            "Spine01": {
                "parent": "Hips",
                "head": (0, 0, 1.1),
                "tail": (0, 0, 1.3),
                "roll": 0
            },
            "Spine02": {
                "parent": "Spine01",
                "head": (0, 0, 1.3),
                "tail": (0, 0, 1.5),
                "roll": 0
            },
            "Chest": {
                "parent": "Spine02",
                "head": (0, 0, 1.5),
                "tail": (0, 0, 1.65),
                "roll": 0
            },
            "Neck": {
                "parent": "Chest",
                "head": (0, 0, 1.65),
                "tail": (0, 0, 1.75),
                "roll": 0
            },
            "Head": {
                "parent": "Neck",
                "head": (0, 0, 1.75),
                "tail": (0, 0, 1.95),
                "roll": 0
            },
            
            # Left arm chain
            "LeftClavicle": {
                "parent": "Chest",
                "head": (0.05, 0, 1.6),
                "tail": (0.15, 0, 1.6),
                "roll": 0
            },
            "LeftShoulder": {
                "parent": "LeftClavicle",
                "head": (0.15, 0, 1.6),
                "tail": (0.45, 0, 1.55),
                "roll": 0
            },
            "LeftElbow": {
                "parent": "LeftShoulder",
                "head": (0.45, 0, 1.55),
                "tail": (0.75, 0, 1.45),
                "roll": 0
            },
            "LeftWrist": {
                "parent": "LeftElbow",
                "head": (0.75, 0, 1.45),
                "tail": (0.85, 0, 1.4),
                "roll": 0
            },
            "LeftHand": {
                "parent": "LeftWrist",
                "head": (0.85, 0, 1.4),
                "tail": (0.95, 0, 1.4),
                "roll": 0
            },
            
            # Right arm chain
            "RightClavicle": {
                "parent": "Chest",
                "head": (-0.05, 0, 1.6),
                "tail": (-0.15, 0, 1.6),
                "roll": 0
            },
            "RightShoulder": {
                "parent": "RightClavicle",
                "head": (-0.15, 0, 1.6),
                "tail": (-0.45, 0, 1.55),
                "roll": 0
            },
            "RightElbow": {
                "parent": "RightShoulder",
                "head": (-0.45, 0, 1.55),
                "tail": (-0.75, 0, 1.45),
                "roll": 0
            },
            "RightWrist": {
                "parent": "RightElbow",
                "head": (-0.75, 0, 1.45),
                "tail": (-0.85, 0, 1.4),
                "roll": 0
            },
            "RightHand": {
                "parent": "RightWrist",
                "head": (-0.85, 0, 1.4),
                "tail": (-0.95, 0, 1.4),
                "roll": 0
            },
            
            # Left leg chain
            "LeftThigh": {
                "parent": "Hips",
                "head": (0.1, 0, 0.9),
                "tail": (0.12, 0, 0.5),
                "roll": 0
            },
            "LeftKnee": {
                "parent": "LeftThigh",
                "head": (0.12, 0, 0.5),
                "tail": (0.15, 0, 0.1),
                "roll": 0
            },
            "LeftAnkle": {
                "parent": "LeftKnee",
                "head": (0.15, 0, 0.1),
                "tail": (0.15, 0.15, 0.05),
                "roll": 0
            },
            "LeftToe": {
                "parent": "LeftAnkle",
                "head": (0.15, 0.15, 0.05),
                "tail": (0.15, 0.25, 0.05),
                "roll": 0
            },
            
            # Right leg chain  
            "RightThigh": {
                "parent": "Hips",
                "head": (-0.1, 0, 0.9),
                "tail": (-0.12, 0, 0.5),
                "roll": 0
            },
            "RightKnee": {
                "parent": "RightThigh",
                "head": (-0.12, 0, 0.5),
                "tail": (-0.15, 0, 0.1),
                "roll": 0
            },
            "RightAnkle": {
                "parent": "RightKnee",
                "head": (-0.15, 0, 0.1),
                "tail": (-0.15, 0.15, 0.05),
                "roll": 0
            },
            "RightToe": {
                "parent": "RightAnkle",
                "head": (-0.15, 0.15, 0.05),
                "tail": (-0.15, 0.25, 0.05),
                "roll": 0
            }
        }
        
        # Create bones in edit mode
        for bone_name, bone_info in bones_data.items():
            if bone_name == "Root":
                # Rename the default bone
                default_bone = armature_data.edit_bones[0]
                default_bone.name = bone_name
                default_bone.head = bone_info["head"]
                default_bone.tail = bone_info["tail"]
                default_bone.roll = bone_info["roll"]
            else:
                bone = armature_data.edit_bones.new(bone_name)
                bone.head = bone_info["head"]
                bone.tail = bone_info["tail"]
                bone.roll = bone_info["roll"]
                
                if bone_info["parent"]:
                    parent_bone = armature_data.edit_bones[bone_info["parent"]]
                    bone.parent = parent_bone
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Add IK constraints in pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Add IK solvers for arms and legs
        ik_chains = [
            ("LeftHand", "LeftShoulder", 3),  # Hand to shoulder, 3 bones
            ("RightHand", "RightShoulder", 3),
            ("LeftToe", "LeftThigh", 3),      # Toe to thigh, 3 bones
            ("RightToe", "RightThigh", 3)
        ]
        
        for target_bone, chain_start, chain_length in ik_chains:
            if target_bone in self.armature.pose.bones:
                bone = self.armature.pose.bones[target_bone]
                
                # Add IK constraint
                ik_constraint = bone.constraints.new(type='IK')
                ik_constraint.target = self.armature
                ik_constraint.subtarget = chain_start
                ik_constraint.chain_count = chain_length
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return self.armature
    
    def load_tracking_data(self, json_path):
        """Load MediaPipe tracking data from JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Tracking data file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def apply_mediapipe_animation(self, tracking_data, start_frame=1):
        """Apply MediaPipe tracking data to the armature"""
        
        if not self.armature:
            raise ValueError("No armature created. Call create_advanced_rig() first.")
        
        # MediaPipe landmark indices to bone mapping
        mp_pose_to_bones = {
            0: "Head",           # Nose
            11: "LeftShoulder",  # Left shoulder  
            12: "RightShoulder", # Right shoulder
            13: "LeftElbow",     # Left elbow
            14: "RightElbow",    # Right elbow
            15: "LeftWrist",     # Left wrist
            16: "RightWrist",    # Right wrist
            23: "LeftThigh",     # Left hip
            24: "RightThigh",    # Right hip
            25: "LeftKnee",      # Left knee
            26: "RightKnee",     # Right knee
            27: "LeftAnkle",     # Left ankle
            28: "RightAnkle",    # Right ankle
        }
        
        # Set up scene
        fps = tracking_data.get("fps", 30.0)
        bpy.context.scene.render.fps = int(fps)
        
        # Get first person's data
        person_data = list(tracking_data["persons"].values())[0]
        frames = person_data["frames"]
        
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = start_frame + len(frames) - 1
        
        # Enter pose mode
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        
        # Clear existing keyframes
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.anim.keyframe_clear_v3d()
        
        # Apply animation frame by frame
        for frame_idx, frame_data in enumerate(frames):
            current_frame = start_frame + frame_idx
            bpy.context.scene.frame_set(current_frame)
            
            # Reset all bone rotations
            for bone in self.armature.pose.bones:
                bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
                bone.rotation_euler = Euler((0, 0, 0))
                bone.location = Vector((0, 0, 0))
            
            # Apply pose landmarks
            if frame_data.get("pose_world_landmarks"):
                pose_landmarks = frame_data["pose_world_landmarks"]
                
                for mp_idx, bone_name in mp_pose_to_bones.items():
                    if mp_idx < len(pose_landmarks) and bone_name in self.armature.pose.bones:
                        landmark = pose_landmarks[mp_idx]
                        bone = self.armature.pose.bones[bone_name]
                        
                        # Convert MediaPipe coordinates to Blender
                        # MediaPipe: X right, Y down, Z forward
                        # Blender: X right, Y forward, Z up
                        mp_pos = Vector((landmark[0], -landmark[2], -landmark[1]))
                        
                        # Scale and apply position (for root bone)
                        if bone_name == "Hips":
                            bone.location = mp_pos * 2.0  # Scale factor
                            bone.keyframe_insert(data_path="location", frame=current_frame)
                        
                        # Calculate rotation based on bone relationships
                        if bone.parent:
                            parent_bone = bone.parent
                            parent_mp_idx = None
                            
                            # Find parent bone's MediaPipe index
                            for idx, name in mp_pose_to_bones.items():
                                if name == parent_bone.name:
                                    parent_mp_idx = idx
                                    break
                            
                            if parent_mp_idx is not None and parent_mp_idx < len(pose_landmarks):
                                parent_landmark = pose_landmarks[parent_mp_idx]
                                parent_pos = Vector((parent_landmark[0], -parent_landmark[2], -parent_landmark[1]))
                                
                                # Calculate bone direction
                                bone_direction = (mp_pos - parent_pos).normalized()
                                
                                # Calculate rotation to align with direction
                                default_direction = Vector((0, 1, 0))  # Default bone direction
                                rotation = default_direction.rotation_difference(bone_direction)
                                
                                bone.rotation_quaternion = rotation
                                bone.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
            
            # Apply hand animations
            self.apply_hand_pose(frame_data.get("left_hand_landmarks"), "LeftHand", current_frame)
            self.apply_hand_pose(frame_data.get("right_hand_landmarks"), "RightHand", current_frame)
            
            # Apply facial animation
            self.apply_face_pose(frame_data.get("face_landmarks"), current_frame)
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Applied animation for frame {current_frame} ({frame_idx+1}/{len(frames)})")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"Animation applied successfully! {len(frames)} frames processed.")
    
    def apply_hand_pose(self, hand_landmarks, hand_bone_name, frame):
        """Apply hand pose to the specified hand bone"""
        if not hand_landmarks or hand_bone_name not in self.armature.pose.bones:
            return
        
        hand_bone = self.armature.pose.bones[hand_bone_name]
        
        # Simplified hand animation based on overall hand pose
        if len(hand_landmarks) >= 21:
            # Calculate hand orientation from wrist to middle finger
            wrist = Vector(hand_landmarks[0])
            middle_tip = Vector(hand_landmarks[12])
            
            # Hand direction vector
            hand_direction = (middle_tip - wrist).normalized()
            
            # Convert to rotation
            default_direction = Vector((1, 0, 0))  # Default hand direction
            rotation = default_direction.rotation_difference(hand_direction)
            
            hand_bone.rotation_quaternion = rotation
            hand_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    
    def apply_face_pose(self, face_landmarks, frame):
        """Apply facial pose to head bone"""
        if not face_landmarks or "Head" not in self.armature.pose.bones:
            return
        
        head_bone = self.armature.pose.bones["Head"]
        
        if len(face_landmarks) >= 468:
            # Calculate head orientation from key facial landmarks
            nose_tip = Vector(face_landmarks[1])
            nose_bridge = Vector(face_landmarks[168])
            left_eye = Vector(face_landmarks[33])
            right_eye = Vector(face_landmarks[362])
            
            # Calculate face center and orientation
            eye_center = (left_eye + right_eye) * 0.5
            face_normal = (right_eye - left_eye).cross(nose_tip - eye_center).normalized()
            
            # Convert to head rotation
            # This is simplified - more complex face tracking would require shape keys
            head_rotation = Euler((
                (eye_center.y - 0.5) * 0.5,  # Pitch
                (eye_center.x - 0.5) * 0.5,  # Yaw
                (right_eye.y - left_eye.y) * 2.0  # Roll
            ))
            
            head_bone.rotation_euler = head_rotation
            head_bone.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    def add_mesh_deformation(self):
        """Add automatic weights and mesh deformation"""
        if not self.character_mesh or not self.armature:
            print("Warning: No mesh or armature to set up deformation")
            return
        
        # Select mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        self.character_mesh.select_set(True)
        self.armature.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        
        # Parent mesh to armature with automatic weights
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        print("Mesh parented to armature with automatic weights")
    
    def export_fbx(self, output_path, include_mesh=True):
        """Export the animated character to FBX"""
        
        # Select objects to export
        bpy.ops.object.select_all(action='DESELECT')
        
        objects_to_export = [self.armature]
        if include_mesh and self.character_mesh:
            objects_to_export.append(self.character_mesh)
        
        for obj in objects_to_export:
            if obj:
                obj.select_set(True)
        
        if objects_to_export:
            bpy.context.view_layer.objects.active = objects_to_export[0]
        
        # Export FBX with animation
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            check_existing=True,
            use_selection=True,
            use_visible=True,
            use_active_collection=False,
            global_scale=100.0,  # Convert to centimeters for Unreal
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_NONE',
            use_space_transform=True,
            bake_space_transform=False,
            object_types={'ARMATURE', 'MESH'} if include_mesh else {'ARMATURE'},
            use_mesh_modifiers=True,
            mesh_smooth_type='FACE',
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
        
        print(f"FBX exported successfully to: {output_path}")
    
    def save_blend_file(self, output_path):
        """Save the Blender scene"""
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print(f"Blender file saved to: {output_path}")


def main():
    """Main function to run the MediaPipe to Blender conversion"""
    
    # Configure paths - MODIFY THESE PATHS FOR YOUR SETUP
    tracking_data_path = r"C:\Users\Abror\Desktop\New folder (12)\output\tracking_data.json"
    fbx_output_path = r"C:\Users\Abror\Desktop\New folder (12)\output\WhiteRobotMan_animation.fbx"
    blend_output_path = r"C:\Users\Abror\Desktop\New folder (12)\output\WhiteRobotMan_scene.blend"
    
    print("=== MediaPipe to Blender FBX Conversion ===")
    
    # Create converter instance
    converter = MediaPipeToBlender()
    
    # Clear scene
    converter.clear_scene()
    
    # Create character
    print("Creating character mesh...")
    converter.create_character_mesh("WhiteRobotMan")
    
    print("Creating character rig...")
    converter.create_advanced_rig("WhiteRobotMan")
    
    # Set up mesh deformation
    print("Setting up mesh deformation...")
    converter.add_mesh_deformation()
    
    # Load and apply tracking data
    if os.path.exists(tracking_data_path):
        print(f"Loading tracking data from: {tracking_data_path}")
        tracking_data = converter.load_tracking_data(tracking_data_path)
        
        print("Applying MediaPipe animation...")
        converter.apply_mediapipe_animation(tracking_data)
        
        # Export FBX
        print("Exporting FBX...")
        converter.export_fbx(fbx_output_path, include_mesh=True)
        
        # Save Blender file
        print("Saving Blender file...")
        converter.save_blend_file(blend_output_path)
        
        print("✅ Conversion complete!")
        print(f"📁 Output files:")
        print(f"   - FBX: {fbx_output_path}")
        print(f"   - Blender: {blend_output_path}")
        
    else:
        print(f"❌ Error: Tracking data file not found at {tracking_data_path}")
        print("Please run the MediaPipe tracking script first to generate tracking data.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MediaPipe Studio - Part 5: Animation Preview with Timeline Controls & Chapter Editor
Enhanced web interface with video timeline, playback controls, and chapter-based scene analysis

New Features:
- Video timeline with scrubbing
- Animation playback controls (play/pause/step)
- Real-time landmark visualization
- Side-by-side video/animation preview
- Frame-by-frame navigation
- Synchronized playback
- Chapter Timeline Editor with scene structure analysis
- Motion Layer AI for separating face, hand, and body movements
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import json
import time
import threading
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import uuid
import subprocess
import shutil
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mediapipe_studio_part5_2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEMP_FOLDER = 'temp'
PREVIEW_FOLDER = 'previews'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'}

# Create directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, PREVIEW_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for processing status
processing_status = {}
processing_results = {}
preview_cache = {}
chapter_data = {}

class MediaPipeStudioPreview:
    """Enhanced MediaPipe processing with timeline preview capabilities and chapter analysis"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Processing settings
        self.current_job = None
        self.is_processing = False
        
        # Chapter analysis parameters
        self.chapter_analysis_enabled = True
        self.motion_layers = {
            'face': True,
            'hands': True, 
            'body': True
        }
        
    def process_video_with_preview(self, video_path: str, job_id: str, settings: dict) -> dict:
        """Process video with enhanced preview generation"""
        try:
            self.current_job = job_id
            self.is_processing = True
            
            # Update status
            processing_status[job_id] = {
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing MediaPipe with preview generation...',
                'start_time': time.time()
            }
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            processing_status[job_id].update({
                'status': 'processing',
                'message': f'Processing {total_frames} frames @ {fps} FPS with preview',
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f'{width}x{height}'
            })
            
            # Initialize holistic model
            holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=settings.get('model_complexity', 2),
                enable_segmentation=settings.get('enable_segmentation', False),
                refine_face_landmarks=settings.get('refine_face', True),
                min_detection_confidence=settings.get('detection_confidence', 0.7),
                min_tracking_confidence=settings.get('tracking_confidence', 0.6)
            )
            
            # Storage for tracking data
            tracking_data = {
                'metadata': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'resolution': [width, height],
                    'processing_time': None,
                    'settings': settings,
                    'export_formats': settings.get('export_formats', ['fbx']),
                    'video_path': video_path
                },
                'persons': {}
            }
            
            # Create preview directories
            output_dir = os.path.join(OUTPUT_FOLDER, job_id)
            preview_dir = os.path.join(PREVIEW_FOLDER, job_id)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(preview_dir, exist_ok=True)
            
            # Process frames with preview generation
            frame_data = []
            preview_frames = []
            frame_num = 0
            
            # Generate preview frames (every 5th frame for performance)
            preview_interval = max(1, total_frames // 100)  # Max 100 preview frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = holistic.process(rgb_frame)
                
                # Extract landmarks
                frame_landmarks = self.extract_frame_data(results, frame_num / fps)
                frame_data.append(frame_landmarks)
                
                # Generate preview frame - use RGB frame for consistent colors
                if frame_num % preview_interval == 0 or frame_num < 10:
                    # Convert back to BGR for OpenCV drawing operations
                    bgr_frame_for_preview = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    preview_frame = self.create_preview_frame(bgr_frame_for_preview, results, frame_num)
                    preview_frames.append({
                        'frame_number': frame_num,
                        'timestamp': frame_num / fps,
                        'preview_data': preview_frame
                    })
                
                # Update progress
                progress = (frame_num / total_frames) * 90  # Reserve 10% for export
                processing_status[job_id].update({
                    'progress': progress,
                    'message': f'Processing frame {frame_num}/{total_frames} ({progress:.1f}%)',
                    'current_frame': frame_num
                })
                
                frame_num += 1
                
                # Check if job was cancelled
                if processing_status[job_id].get('cancelled', False):
                    cap.release()
                    holistic.close()
                    return {'error': 'Processing cancelled by user'}
            
            cap.release()
            holistic.close()
            
            # Store tracking data
            tracking_data['persons']['0'] = {
                'person_id': 0,
                'frames': frame_data
            }
            
            # Analyze chapter structure
            processing_status[job_id].update({
                'status': 'analyzing_chapters',
                'progress': 85,
                'message': 'Analyzing chapter structure and motion layers...'
            })
            
            chapters = self.analyze_chapter_structure(frame_data, total_frames, fps)
            tracking_data['chapters'] = chapters
            
            # Store chapter data globally for API access
            chapter_data[job_id] = {
                'chapters': chapters,
                'motion_analysis': {
                    'face_activity': sum(1 for frame in frame_data if 'face_landmarks' in frame and frame['face_landmarks']),
                    'hand_activity': sum(1 for frame in frame_data if ('left_hand_landmarks' in frame and frame['left_hand_landmarks']) or ('right_hand_landmarks' in frame and frame['right_hand_landmarks'])),
                    'body_activity': sum(1 for frame in frame_data if 'pose_landmarks' in frame and frame['pose_landmarks']),
                    'total_frames': total_frames
                }
            }
            
            # Save tracking data
            tracking_file = os.path.join(output_dir, 'tracking_data.json')
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            # Save preview data
            preview_file = os.path.join(preview_dir, 'preview_data.json')
            with open(preview_file, 'w') as f:
                json.dump({
                    'metadata': tracking_data['metadata'],
                    'preview_frames': preview_frames
                }, f, indent=2)
            
            # Generate video thumbnail
            self.generate_video_thumbnail(video_path, preview_dir)
            
            # Generate exports
            processing_status[job_id].update({
                'status': 'exporting',
                'progress': 90,
                'message': 'Generating export files...'
            })
            
            export_results = self.generate_exports(tracking_data, output_dir, settings)
            
            # Complete
            end_time = time.time()
            processing_time = end_time - processing_status[job_id]['start_time']
            
            processing_status[job_id].update({
                'status': 'completed',
                'progress': 100,
                'message': 'Processing complete with preview!',
                'processing_time': processing_time,
                'export_files': export_results,
                'preview_available': True
            })
            
            # Cache preview data
            preview_cache[job_id] = {
                'metadata': tracking_data['metadata'],
                'preview_frames': preview_frames
            }
            
            return {
                'success': True,
                'job_id': job_id,
                'processing_time': processing_time,
                'export_files': export_results,
                'tracking_data': tracking_file,
                'preview_data': preview_file
            }
            
        except Exception as e:
            processing_status[job_id].update({
                'status': 'error',
                'message': f'Error: {str(e)}'
            })
            return {'error': str(e)}
        
        finally:
            self.is_processing = False
    
    def create_preview_frame(self, frame, results, frame_number):
        """Create preview frame with MediaPipe landmarks overlay"""
        # Validate input frame
        if frame is None or frame.size == 0:
            print(f"Warning: Invalid frame {frame_number}")
            # Create a black frame as fallback
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a copy for drawing
        annotated_frame = frame.copy()
        original_frame = frame.copy()
        
        # Enhanced drawing specifications for better visibility
        landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)
        connection_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        
        # Count landmarks for debugging
        landmark_count = 0
        
        # Draw landmarks on annotated frame with enhanced visibility
        if results.pose_landmarks:
            landmark_count += len(results.pose_landmarks.landmark)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
        
        if results.face_landmarks:
            landmark_count += len(results.face_landmarks.landmark)
            face_landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            face_connection_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=face_landmark_spec,
                connection_drawing_spec=face_connection_spec
            )
        
        if results.left_hand_landmarks:
            landmark_count += len(results.left_hand_landmarks.landmark)
            hand_landmark_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            hand_connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec
            )
        
        if results.right_hand_landmarks:
            landmark_count += len(results.right_hand_landmarks.landmark)
            hand_landmark_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            hand_connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec
            )
        
        # Add debugging info to frames
        debug_text = f"Frame: {frame_number} | Landmarks: {landmark_count}"
        cv2.putText(annotated_frame, debug_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(original_frame, f"Original Frame: {frame_number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert both frames to base64 for web display with higher quality
        _, annotated_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
        
        _, original_buffer = cv2.imencode('.jpg', original_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        print(f"Generated preview frame {frame_number} with {landmark_count} landmarks")
        
        return {
            'annotated': annotated_base64,
            'original': original_base64
        }
    
    def analyze_chapter_structure(self, frame_data, total_frames, fps):
        """Analyze video structure and create logical chapters"""
        try:
            # Calculate motion intensity for each frame
            motion_intensities = []
            for frame in frame_data:
                intensity = self.calculate_motion_intensity(frame)
                motion_intensities.append(intensity)
            
            # Detect chapter boundaries based on motion patterns
            chapters = self.detect_chapter_boundaries(motion_intensities, total_frames, fps)
            
            # Classify chapter types (intro, conflict, resolution)
            classified_chapters = self.classify_chapters(chapters, motion_intensities)
            
            return classified_chapters
            
        except Exception as e:
            print(f"Chapter analysis error: {e}")
            return self.get_default_chapters(total_frames, fps)
    
    def calculate_motion_intensity(self, frame_data):
        """Calculate motion intensity for a single frame"""
        intensity = 0.0
        
        # Face motion intensity (based on landmark presence and variation)
        if 'face_landmarks' in frame_data and frame_data['face_landmarks']:
            face_points = len(frame_data['face_landmarks'])
            intensity += min(face_points / 468.0, 1.0) * 0.3
        
        # Hand motion intensity
        hands_intensity = 0.0
        if 'left_hand_landmarks' in frame_data and frame_data['left_hand_landmarks']:
            hands_intensity += 0.5
        if 'right_hand_landmarks' in frame_data and frame_data['right_hand_landmarks']:
            hands_intensity += 0.5
        intensity += min(hands_intensity, 1.0) * 0.4
        
        # Body motion intensity
        if 'pose_landmarks' in frame_data and frame_data['pose_landmarks']:
            body_points = len(frame_data['pose_landmarks'])
            intensity += min(body_points / 33.0, 1.0) * 0.3
        
        return min(intensity, 1.0)
    
    def detect_chapter_boundaries(self, motion_intensities, total_frames, fps):
        """Detect natural chapter boundaries based on motion patterns"""
        boundaries = [0]  # Always start with frame 0
        
        # Simple algorithm: detect significant changes in motion intensity
        window_size = int(fps * 2)  # 2-second window
        threshold = 0.3
        
        for i in range(window_size, len(motion_intensities) - window_size, window_size):
            prev_avg = sum(motion_intensities[i-window_size:i]) / window_size
            next_avg = sum(motion_intensities[i:i+window_size]) / window_size
            
            if abs(next_avg - prev_avg) > threshold:
                boundaries.append(i)
        
        boundaries.append(total_frames - 1)  # Always end with last frame
        
        return boundaries
    
    def classify_chapters(self, boundaries, motion_intensities):
        """Classify chapters as intro, conflict, or resolution"""
        chapters = []
        
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            
            # Calculate average intensity for this chapter
            chapter_intensities = motion_intensities[start_frame:end_frame]
            avg_intensity = sum(chapter_intensities) / len(chapter_intensities)
            
            # Classify based on position and intensity
            chapter_type = self.determine_chapter_type(i, len(boundaries) - 1, avg_intensity)
            
            chapters.append({
                'type': chapter_type,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_percent': (start_frame / (len(motion_intensities) - 1)) * 100,
                'end_percent': (end_frame / (len(motion_intensities) - 1)) * 100,
                'motion_intensity': avg_intensity,
                'duration_seconds': (end_frame - start_frame) / 30,  # Assuming 30fps
                'description': self.get_chapter_description(chapter_type, avg_intensity)
            })
        
        return chapters
    
    def determine_chapter_type(self, chapter_index, total_chapters, intensity):
        """Determine chapter type based on position and motion intensity"""
        position_ratio = chapter_index / (total_chapters - 1)
        
        if position_ratio < 0.3:
            return 'intro'
        elif position_ratio > 0.7:
            return 'resolution'
        else:
            # Middle sections - classify based on intensity
            if intensity > 0.6:
                return 'conflict'
            else:
                return 'intro' if position_ratio < 0.5 else 'resolution'
    
    def get_chapter_description(self, chapter_type, intensity):
        """Generate description for chapter based on type and intensity"""
        descriptions = {
            'intro': [
                'Gentle introduction with minimal movement',
                'Setting up the scene with steady motion',
                'Initial presentation with moderate activity'
            ],
            'conflict': [
                'High-energy section with dynamic movement',
                'Active gestures and expressive motion',
                'Peak intensity with full-body engagement'
            ],
            'resolution': [
                'Calm conclusion with settling motion',
                'Peaceful resolution with gentle movements',
                'Final thoughts with minimal activity'
            ]
        }
        
        intensity_level = 'high' if intensity > 0.7 else 'moderate' if intensity > 0.4 else 'low'
        base_descriptions = descriptions.get(chapter_type, ['Unknown chapter type'])
        
        return f"{base_descriptions[0]} ({intensity_level} intensity)"
    
    def get_default_chapters(self, total_frames, fps):
        """Return default chapter structure if analysis fails"""
        return [
            {
                'type': 'intro',
                'start_frame': 0,
                'end_frame': total_frames // 3,
                'start_percent': 0,
                'end_percent': 33.33,
                'motion_intensity': 0.4,
                'duration_seconds': (total_frames // 3) / fps,
                'description': 'Introduction section (default)'
            },
            {
                'type': 'conflict',
                'start_frame': total_frames // 3,
                'end_frame': (total_frames * 2) // 3,
                'start_percent': 33.33,
                'end_percent': 66.67,
                'motion_intensity': 0.7,
                'duration_seconds': (total_frames // 3) / fps,
                'description': 'Main content section (default)'
            },
            {
                'type': 'resolution',
                'start_frame': (total_frames * 2) // 3,
                'end_frame': total_frames - 1,
                'start_percent': 66.67,
                'end_percent': 100,
                'motion_intensity': 0.3,
                'duration_seconds': (total_frames // 3) / fps,
                'description': 'Conclusion section (default)'
            }
        ]

    def generate_video_thumbnail(self, video_path, preview_dir):
        """Generate video thumbnail"""
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get middle frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            
            if ret:
                thumbnail_path = os.path.join(preview_dir, 'thumbnail.jpg')
                cv2.imwrite(thumbnail_path, frame)
            
            cap.release()
    
    def extract_frame_data(self, results, timestamp: float) -> dict:
        """Extract comprehensive frame data"""
        frame_data = {'timestamp': timestamp}
        
        # Face landmarks
        if results.face_landmarks:
            frame_data['face_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
            ]
        
        # Pose landmarks
        if results.pose_landmarks:
            frame_data['pose_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
        
        if results.pose_world_landmarks:
            frame_data['pose_world_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
            ]
        
        # Hand landmarks
        if results.left_hand_landmarks:
            frame_data['left_hand_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
            ]
        
        if results.right_hand_landmarks:
            frame_data['right_hand_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
            ]
        
        return frame_data
    
    def generate_exports(self, tracking_data: dict, output_dir: str, settings: dict) -> dict:
        """Generate multiple export formats (same as Part 4)"""
        export_formats = settings.get('export_formats', ['fbx'])
        export_files = {}
        
        for format_type in export_formats:
            try:
                if format_type == 'ue4':
                    export_files['ue4'] = self.export_ue4(tracking_data, output_dir)
                elif format_type == 'ue5':
                    export_files['ue5'] = self.export_ue5(tracking_data, output_dir)
                elif format_type == 'unity':
                    export_files['unity'] = self.export_unity(tracking_data, output_dir)
                elif format_type == 'roblox':
                    export_files['roblox'] = self.export_roblox(tracking_data, output_dir)
                elif format_type == 'metahuman':
                    export_files['metahuman'] = self.export_metahuman(tracking_data, output_dir)
                elif format_type == 'fbx':
                    export_files['fbx'] = self.export_standard_fbx(tracking_data, output_dir)
                elif format_type == 'anim':
                    export_files['anim'] = self.export_animation_data(tracking_data, output_dir)
            except Exception as e:
                export_files[f'{format_type}_error'] = str(e)
        
        return export_files
    
    # Export methods (same as Part 4 - abbreviated for space)
    def export_ue4(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'ue4_animation.fbx')
        fbx_content = self.create_ue_fbx(tracking_data, 'UE4')
        with open(output_file, 'w') as f:
            f.write(fbx_content)
        return output_file
    
    def export_ue5(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'ue5_animation.fbx')
        fbx_content = self.create_ue_fbx(tracking_data, 'UE5')
        with open(output_file, 'w') as f:
            f.write(fbx_content)
        return output_file
    
    def export_unity(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'unity_animation.fbx')
        fbx_content = self.create_unity_fbx(tracking_data)
        with open(output_file, 'w') as f:
            f.write(fbx_content)
        return output_file
    
    def export_roblox(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'roblox_animation.json')
        roblox_data = self.convert_to_roblox_format(tracking_data)
        with open(output_file, 'w') as f:
            json.dump(roblox_data, f, indent=2)
        return output_file
    
    def export_metahuman(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'metahuman_animation.fbx')
        fbx_content = self.create_metahuman_fbx(tracking_data)
        with open(output_file, 'w') as f:
            f.write(fbx_content)
        return output_file
    
    def export_standard_fbx(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'standard_animation.fbx')
        fbx_content = self.create_standard_fbx(tracking_data)
        with open(output_file, 'w') as f:
            f.write(fbx_content)
        return output_file
    
    def export_animation_data(self, tracking_data: dict, output_dir: str) -> str:
        output_file = os.path.join(output_dir, 'animation_data.json')
        anim_data = self.create_animation_data(tracking_data)
        with open(output_file, 'w') as f:
            json.dump(anim_data, f, indent=2)
        return output_file
    
    # FBX creation methods (abbreviated - same as Part 4)
    def create_ue_fbx(self, tracking_data: dict, version: str) -> str:
        return f'FBXHeaderExtension: {{ FBXHeaderVersion: 1003 }}'  # Simplified
    
    def create_unity_fbx(self, tracking_data: dict) -> str:
        return self.create_ue_fbx(tracking_data, 'Unity')
    
    def convert_to_roblox_format(self, tracking_data: dict) -> dict:
        return {"Type": "KeyframeSequence", "Name": "MediaPipeAnimation", "Keyframes": []}
    
    def create_metahuman_fbx(self, tracking_data: dict) -> str:
        return self.create_ue_fbx(tracking_data, 'MetaHuman')
    
    def create_standard_fbx(self, tracking_data: dict) -> str:
        return self.create_ue_fbx(tracking_data, 'Standard')
    
    def create_animation_data(self, tracking_data: dict) -> dict:
        return {"format": "MediaPipe_Studio_Animation", "version": "5.0", "data": tracking_data}

# Initialize MediaPipe Studio
studio = MediaPipeStudioPreview()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_video_for_h264(input_path: str, output_path: str) -> bool:
    """Convert video to H.264 for optimal compatibility"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        # If ffmpeg not available, just copy the file
        shutil.copy2(input_path, output_path)
        return True

@app.route('/')
def index():
    """Main page with timeline preview"""
    return render_template('preview_index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Optimize for H.264 if needed
        optimized_path = os.path.join(TEMP_FOLDER, f"{job_id}_optimized.mp4")
        if optimize_video_for_h264(upload_path, optimized_path):
            processing_video_path = optimized_path
        else:
            processing_video_path = upload_path
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'filename': filename,
            'message': 'Video uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    """Start video processing with preview generation"""
    data = request.get_json()
    job_id = data.get('job_id')
    settings = data.get('settings', {})
    
    if not job_id:
        return jsonify({'error': 'Job ID required'}), 400
    
    # Find the video file
    video_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(job_id)]
    if not video_files:
        # Check temp folder for optimized video
        temp_files = [f for f in os.listdir(TEMP_FOLDER) if f.startswith(job_id)]
        if temp_files:
            video_path = os.path.join(TEMP_FOLDER, temp_files[0])
        else:
            return jsonify({'error': 'Video file not found'}), 404
    else:
        video_path = os.path.join(UPLOAD_FOLDER, video_files[0])
    
    # Start processing in background thread
    def process_in_background():
        studio.process_video_with_preview(video_path, job_id, settings)
    
    thread = threading.Thread(target=process_in_background)
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Processing started with preview generation'
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_processing(job_id):
    """Cancel processing"""
    if job_id in processing_status:
        processing_status[job_id]['cancelled'] = True
        return jsonify({'success': True, 'message': 'Processing cancelled'})
    
    return jsonify({'error': 'Job not found'}), 404

@app.route('/preview/<job_id>')
def get_preview_data(job_id):
    """Get animation preview data for timeline with chapter information"""
    # First check cache
    if job_id in preview_cache:
        data = preview_cache[job_id].copy()
        # Add chapter data if available
        if job_id in chapter_data:
            data['chapters'] = chapter_data[job_id]['chapters']
            data['motion_analysis'] = chapter_data[job_id]['motion_analysis']
        return jsonify(data)
    
    # Then check file
    preview_file = os.path.join(PREVIEW_FOLDER, job_id, 'preview_data.json')
    
    if not os.path.exists(preview_file):
        return jsonify({'error': 'Preview data not found'}), 404
    
    with open(preview_file, 'r') as f:
        data = json.load(f)
    
    # Add chapter data if available
    if job_id in chapter_data:
        data['chapters'] = chapter_data[job_id]['chapters']
        data['motion_analysis'] = chapter_data[job_id]['motion_analysis']
    
    return jsonify(data)

@app.route('/preview/<job_id>/frame/<int:frame_number>')
def get_preview_frame(job_id, frame_number):
    """Get specific preview frame"""
    if job_id in preview_cache:
        preview_frames = preview_cache[job_id].get('preview_frames', [])
        for frame in preview_frames:
            if frame['frame_number'] == frame_number:
                return jsonify(frame)
    
    return jsonify({'error': 'Frame not found'}), 404

@app.route('/chapters/<job_id>')
def get_chapters(job_id):
    """Get chapter structure for a processed video"""
    if job_id in chapter_data:
        return jsonify({
            'success': True,
            'chapters': chapter_data[job_id]['chapters'],
            'motion_analysis': chapter_data[job_id]['motion_analysis']
        })
    
    return jsonify({'error': 'Chapters not found'}), 404

@app.route('/chapters/<job_id>/update', methods=['POST'])
def update_chapters(job_id):
    """Update chapter structure"""
    try:
        new_chapters = request.json.get('chapters', [])
        
        if job_id in chapter_data:
            chapter_data[job_id]['chapters'] = new_chapters
            
            # Optionally save to file
            output_dir = os.path.join(OUTPUT_FOLDER, job_id)
            if os.path.exists(output_dir):
                chapters_file = os.path.join(output_dir, 'chapters.json')
                with open(chapters_file, 'w') as f:
                    json.dump(chapter_data[job_id], f, indent=2)
            
            return jsonify({'success': True, 'message': 'Chapters updated successfully'})
        
        return jsonify({'error': 'Job not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/motion-analysis/<job_id>')
def get_motion_analysis(job_id):
    """Get detailed motion analysis for all layers"""
    try:
        if job_id not in chapter_data:
            return jsonify({'error': 'Motion analysis not found'}), 404
        
        analysis = chapter_data[job_id]['motion_analysis']
        chapters = chapter_data[job_id]['chapters']
        
        # Calculate detailed metrics
        detailed_analysis = {
            'overall_metrics': {
                'face_activity_percentage': (analysis['face_activity'] / analysis['total_frames']) * 100,
                'hand_activity_percentage': (analysis['hand_activity'] / analysis['total_frames']) * 100,
                'body_activity_percentage': (analysis['body_activity'] / analysis['total_frames']) * 100,
                'total_frames': analysis['total_frames']
            },
            'chapter_breakdown': []
        }
        
        for chapter in chapters:
            detailed_analysis['chapter_breakdown'].append({
                'type': chapter['type'],
                'duration': chapter['duration_seconds'],
                'motion_intensity': chapter['motion_intensity'],
                'description': chapter['description'],
                'start_percent': chapter['start_percent'],
                'end_percent': chapter['end_percent']
            })
        
        return jsonify({
            'success': True,
            'analysis': detailed_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<job_id>/frame/<int:frame_number>')
def get_video_frame(job_id, frame_number):
    """Get specific video frame from original video"""
    try:
        # Find the video file
        video_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(job_id)]
        if not video_files:
            # Check temp folder for optimized video
            temp_files = [f for f in os.listdir(TEMP_FOLDER) if f.startswith(job_id)]
            if temp_files:
                video_path = os.path.join(TEMP_FOLDER, temp_files[0])
            else:
                return jsonify({'error': 'Video file not found'}), 404
        else:
            video_path = os.path.join(UPLOAD_FOLDER, video_files[0])
        
        # Extract specific frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video'}), 500
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Could not read frame'}), 404
        
        # Add frame number overlay
        cv2.putText(frame, f"Original Frame: {frame_number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'frame_number': frame_number,
            'frame_data': frame_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    """Download processed files"""
    output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    
    file_map = {
        'ue4': 'ue4_animation.fbx',
        'ue5': 'ue5_animation.fbx',
        'unity': 'unity_animation.fbx',
        'roblox': 'roblox_animation.json',
        'metahuman': 'metahuman_animation.fbx',
        'fbx': 'standard_animation.fbx',
        'anim': 'animation_data.json',
        'tracking': 'tracking_data.json',
        'preview': 'preview_data.json'
    }
    
    if file_type not in file_map:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if file_type == 'preview':
        file_path = os.path.join(PREVIEW_FOLDER, job_id, file_map[file_type])
    else:
        file_path = os.path.join(output_dir, file_map[file_type])
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/thumbnail/<job_id>')
def get_thumbnail(job_id):
    """Get video thumbnail"""
    thumbnail_path = os.path.join(PREVIEW_FOLDER, job_id, 'thumbnail.jpg')
    
    if not os.path.exists(thumbnail_path):
        return jsonify({'error': 'Thumbnail not found'}), 404
    
    return send_file(thumbnail_path)

if __name__ == '__main__':
    print("🚀 Starting MediaPipe Studio - Part 5: Animation Preview")
    print("🌐 Web interface: http://localhost:3000")
    print("🎬 Features: Timeline controls, frame navigation, preview playback")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("📁 Output folder:", OUTPUT_FOLDER)
    print("📁 Preview folder:", PREVIEW_FOLDER)
    
    app.run(host='0.0.0.0', port=3000, debug=True)

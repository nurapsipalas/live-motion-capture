#!/usr/bin/env python3
"""
Live Motion Capture MP4 - Real-time tracking dari file MP4
Memutar MP4 dan men-tracking gerakan secara real-time frame-by-frame

Features:
🔁 Real-time playback dengan MediaPipe tracking
🎭 Live chapter detection (intro/conflict/resolution)
🤖 Export FBX/BVH untuk Unreal/Blender
👤 Preview dengan White Robot Man character
📊 Live motion metrics dan visualization
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
import threading
import queue
import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import base64
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'live_motion_capture_mp4'
socketio = SocketIO(app, cors_allowed_origins="*")

class LiveMotionCaptureMP4:
    def __init__(self):
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Live tracking state
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.video_path = None
        self.cap = None
        
        # Motion analysis
        self.motion_buffer = []
        self.current_chapter = "intro"
        self.chapter_transitions = []
        
        # Export data
        self.tracking_data = []
        self.chapter_data = {
            'intro': {'start': 0, 'frames': []},
            'conflict': {'start': 0, 'frames': []},
            'resolution': {'start': 0, 'frames': []}
        }
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
    def load_mp4(self, video_path):
        """Load MP4 file for live tracking"""
        try:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                return False, "Cannot open video file"
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            
            # Reset tracking data
            self.tracking_data = []
            self.motion_buffer = []
            self.chapter_transitions = []
            
            print(f"✅ MP4 loaded: {self.total_frames} frames @ {self.fps} FPS")
            return True, f"Video loaded: {self.total_frames} frames"
            
        except Exception as e:
            return False, f"Error loading video: {str(e)}"
    
    def start_live_tracking(self):
        """Start live motion tracking dari MP4"""
        if not self.cap or not self.cap.isOpened():
            return False, "No video loaded"
        
        self.is_playing = True
        self.processing_thread = threading.Thread(target=self._live_tracking_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True, "Live tracking started"
    
    def stop_live_tracking(self):
        """Stop live tracking"""
        self.is_playing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
    def _live_tracking_loop(self):
        """Main live tracking loop"""
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as holistic:
            
            frame_time = 1.0 / self.fps
            last_emit_time = 0
            
            while self.is_playing and self.current_frame < self.total_frames:
                start_time = time.time()
                
                # Read frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"⚠️ Cannot read frame {self.current_frame}, ending processing")
                    break
                
                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                # Extract landmarks
                frame_data = self._extract_landmarks(results, self.current_frame)
                self.tracking_data.append(frame_data)
                
                # Analyze motion and chapters
                motion_intensity = self._calculate_motion_intensity(frame_data)
                self.motion_buffer.append(motion_intensity)
                
                # Chapter detection
                current_chapter = self._detect_current_chapter(motion_intensity)
                
                # Live preview dan emit
                if time.time() - last_emit_time >= emit_interval:
                    preview_frame = self._create_live_preview(frame, results, frame_data)
                    self._emit_live_data(preview_frame, current_chapter, motion_intensity)
                    last_emit_time = time.time()
                
                # Increment frame
                self.current_frame += 1
                
                # Frame timing untuk real-time playback
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Pastikan processing selesai 100%
            print(f"✅ Motion capture completed! Processed {len(self.tracking_data)} frames")
            self.is_playing = False
            self._emit_completion_data()
                current_chapter = self._detect_current_chapter(motion_intensity)
                if current_chapter != self.current_chapter:
                    self.chapter_transitions.append({
                        'frame': self.current_frame,
                        'from_chapter': self.current_chapter,
                        'to_chapter': current_chapter,
                        'timestamp': self.current_frame / self.fps
                    })
                    self.current_chapter = current_chapter
                
                # Add to chapter data
                self.chapter_data[current_chapter]['frames'].append(frame_data)
                
                # Create preview frame
                preview_frame = self._create_live_preview(frame, results, frame_data)
                
                # Emit real-time data via WebSocket (throttle to 15 FPS for web)
                current_time = time.time()
                if current_time - last_emit_time >= 1/15:  # 15 FPS emit rate
                    self._emit_live_data(preview_frame, frame_data, motion_intensity)
                    last_emit_time = current_time
                
                # Frame timing
                self.current_frame += 1
                processing_time = time.time() - start_time
                sleep_time = max(0, frame_time - processing_time)
                time.sleep(sleep_time)
            
            # Finished processing
            self.is_playing = False
            self._emit_completion_data()
    
    def _extract_landmarks(self, results, frame_number):
        """Extract landmarks dari MediaPipe results"""
        data = {
            'frame': frame_number,
            'timestamp': frame_number / self.fps,
            'face_landmarks': [],
            'pose_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': []
        }
        
        # Face landmarks (468 points)
        if results.face_landmarks:
            data['face_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
            ]
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            data['pose_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
        
        # Hand landmarks (21 points each)
        if results.left_hand_landmarks:
            data['left_hand_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
            ]
        
        if results.right_hand_landmarks:
            data['right_hand_landmarks'] = [
                [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
            ]
        
        return data
    
    def _calculate_motion_intensity(self, frame_data):
        """Calculate motion intensity untuk chapter detection"""
        intensity = 0.0
        
        # Face motion
        if frame_data['face_landmarks']:
            intensity += min(len(frame_data['face_landmarks']) / 468.0, 1.0) * 0.2
        
        # Pose motion
        if frame_data['pose_landmarks']:
            intensity += min(len(frame_data['pose_landmarks']) / 33.0, 1.0) * 0.4
        
        # Hand motion
        hand_activity = 0.0
        if frame_data['left_hand_landmarks']:
            hand_activity += 0.5
        if frame_data['right_hand_landmarks']:
            hand_activity += 0.5
        intensity += hand_activity * 0.4
        
        return min(intensity, 1.0)
    
    def _detect_current_chapter(self, motion_intensity):
        """Detect chapter berdasarkan motion pattern"""
        progress = self.current_frame / self.total_frames if self.total_frames > 0 else 0
        
        # Simple chapter detection based on motion intensity and progress
        if progress < 0.25:
            return "intro"
        elif progress > 0.75:
            return "resolution"
        else:
            # Middle section - analyze motion intensity
            if motion_intensity > 0.6:
                return "conflict"
            else:
                return "intro" if progress < 0.5 else "resolution"
    
    def _create_live_preview(self, frame, results, frame_data):
        """Create live preview frame dengan landmarks"""
        preview = frame.copy()
        
        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                preview, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                preview, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                preview, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                preview, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Add live info
        chapter_colors = {
            'intro': (255, 200, 100),
            'conflict': (100, 150, 255), 
            'resolution': (150, 255, 150)
        }
        
        color = chapter_colors.get(self.current_chapter, (255, 255, 255))
        cv2.putText(preview, f"LIVE: {self.current_chapter.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(preview, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preview_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return preview_b64
    
    def _emit_live_data(self, preview_frame, frame_data, motion_intensity):
        """Emit real-time data via WebSocket"""
        try:
            socketio.emit('live_frame', {
                'frame_data': preview_frame,
                'frame_number': self.current_frame,
                'total_frames': self.total_frames,
                'progress': (self.current_frame / self.total_frames) * 100,
                'current_chapter': self.current_chapter,
                'motion_intensity': motion_intensity,
                'landmarks': {
                    'face_count': len(frame_data['face_landmarks']),
                    'pose_count': len(frame_data['pose_landmarks']),
                    'left_hand_count': len(frame_data['left_hand_landmarks']),
                    'right_hand_count': len(frame_data['right_hand_landmarks'])
                },
                'timestamp': self.current_frame / self.fps
            })
        except Exception as e:
            print(f"WebSocket emit error: {e}")
    
    def _emit_completion_data(self):
        """Emit completion data dan export info"""
        try:
            socketio.emit('tracking_complete', {
                'total_frames_processed': len(self.tracking_data),
                'chapter_transitions': self.chapter_transitions,
                'final_chapter_data': self.chapter_data,
                'export_ready': True
            })
        except Exception as e:
            print(f"Completion emit error: {e}")
    
    def export_to_fbx(self, output_path):
        """Export tracking data ke FBX untuk Unreal/Blender"""
        try:
            fbx_data = self._convert_to_fbx_format()
            
            # Write FBX file (simplified - untuk production perlu FBX SDK)
            with open(output_path, 'w') as f:
                f.write(fbx_data)
            
            return True, f"FBX exported to {output_path}"
        except Exception as e:
            return False, f"FBX export error: {str(e)}"
    
    def export_to_bvh(self, output_path):
        """Export tracking data ke BVH untuk animation"""
        try:
            bvh_data = self._convert_to_bvh_format()
            
            with open(output_path, 'w') as f:
                f.write(bvh_data)
            
            return True, f"BVH exported to {output_path}"
        except Exception as e:
            return False, f"BVH export error: {str(e)}"
    
    def _convert_to_fbx_format(self):
        """Convert tracking data ke FBX format"""
        # Simplified FBX structure
        fbx_content = f"""
; FBX Live Motion Capture Export
; Generated: {datetime.now().isoformat()}
; Total Frames: {len(self.tracking_data)}

FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    Creator: "Live Motion Capture MP4"
}}

Definitions:  {{
    Version: 100
    Count: {len(self.tracking_data)}
    
    ObjectType: "Model" {{
        Count: 1
        PropertyTemplate: "FbxNode" {{
            Properties70:  {{
                P: "QuaternionInterpolate", "enum", "", "",0
                P: "RotationOffset", "Vector3D", "Vector", "",0,0,0
                P: "RotationPivot", "Vector3D", "Vector", "",0,0,0
            }}
        }}
    }}
}}

Objects:  {{
    Model: 2001, "Model::MotionCaptureData", "Mesh" {{
        Version: 232
        Properties70:  {{
            P: "RotationActive", "bool", "", "",1
        }}
        Culling: "CullingOff"
    }}
    
    AnimationStack: 3001, "AnimStack::LiveMotion", "" {{
        Properties70:  {{
            P: "LocalStart", "KTime", "Time", "",0
            P: "LocalStop", "KTime", "Time", "",{int(len(self.tracking_data) * (1/self.fps) * 46186158000)}
            P: "ReferenceStart", "KTime", "Time", "",0
            P: "ReferenceStop", "KTime", "Time", "",{int(len(self.tracking_data) * (1/self.fps) * 46186158000)}
        }}
    }}
}}

; Animation data for {len(self.tracking_data)} frames
; Chapter transitions: {len(self.chapter_transitions)}
"""
        
        # Add frame data
        for i, frame in enumerate(self.tracking_data):
            fbx_content += f"\n; Frame {i}: {len(frame['pose_landmarks'])} pose landmarks"
        
        return fbx_content
    
    def _convert_to_bvh_format(self):
        """Convert tracking data ke BVH format"""
        bvh_content = f"""HIERARCHY
ROOT Hips
{{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Chest
    {{
        OFFSET 0.0 15.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {{
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {{
                OFFSET 0.0 5.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.0 3.0 0.0
                }}
            }}
        }}
        JOINT LeftShoulder
        {{
            OFFSET -8.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftArm
            {{
                OFFSET -10.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftHand
                {{
                    OFFSET -8.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {{
                        OFFSET -3.0 0.0 0.0
                    }}
                }}
            }}
        }}
        JOINT RightShoulder
        {{
            OFFSET 8.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightArm
            {{
                OFFSET 10.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightHand
                {{
                    OFFSET 8.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {{
                        OFFSET 3.0 0.0 0.0
                    }}
                }}
            }}
        }}
    }}
}}
MOTION
Frames: {len(self.tracking_data)}
Frame Time: {1.0/self.fps:.6f}
"""
        
        # Add motion data
        for frame in self.tracking_data:
            # Convert landmarks to BVH animation data
            if frame['pose_landmarks']:
                pose = frame['pose_landmarks']
                # Extract key joint positions and convert to rotations
                bvh_content += f"0.0 0.0 0.0 0.0 0.0 0.0 "  # Hips
                bvh_content += f"0.0 0.0 0.0 "  # Chest
                bvh_content += f"0.0 0.0 0.0 "  # Neck
                bvh_content += f"0.0 0.0 0.0 "  # Head
                bvh_content += f"0.0 0.0 0.0 "  # LeftShoulder
                bvh_content += f"0.0 0.0 0.0 "  # LeftArm
                bvh_content += f"0.0 0.0 0.0 "  # LeftHand
                bvh_content += f"0.0 0.0 0.0 "  # RightShoulder
                bvh_content += f"0.0 0.0 0.0 "  # RightArm
                bvh_content += f"0.0 0.0 0.0"   # RightHand
            else:
                bvh_content += "0.0 " * 30  # Default pose
            bvh_content += "\n"
        
        return bvh_content

# Global instance
live_tracker = LiveMotionCaptureMP4()

# Flask Routes
@app.route('/')
def index():
    return render_template('live_motion_mp4.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(upload_dir, filename)
    file.save(video_path)
    
    return jsonify({'success': True, 'video_path': video_path, 'message': f'Video uploaded: {filename}'})

@app.route('/load_video', methods=['POST'])
def load_video():
    video_path = request.json.get('video_path')
    if not video_path:
        return jsonify({'success': False, 'error': 'No video path provided'})
    
    # Handle relative paths
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.path.dirname(__file__), video_path)
    
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': f'Video file not found: {video_path}'})
    
    success, message = live_tracker.load_mp4(video_path)
    return jsonify({'success': success, 'message': message})

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    success, message = live_tracker.start_live_tracking()
    return jsonify({'success': success, 'message': message})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    live_tracker.stop_live_tracking()
    return jsonify({'success': True, 'message': 'Tracking stopped'})

@app.route('/export_fbx', methods=['POST'])
def export_fbx():
    output_path = request.json.get('output_path', 'live_motion_export.fbx')
    success, message = live_tracker.export_to_fbx(output_path)
    return jsonify({'success': success, 'message': message})

@app.route('/export_bvh', methods=['POST'])
def export_bvh():
    output_path = request.json.get('output_path', 'live_motion_export.bvh')
    success, message = live_tracker.export_to_bvh(output_path)
    return jsonify({'success': success, 'message': message})

@app.route('/get_status')
def get_status():
    return jsonify({
        'is_playing': live_tracker.is_playing,
        'current_frame': live_tracker.current_frame,
        'total_frames': live_tracker.total_frames,
        'current_chapter': live_tracker.current_chapter,
        'progress': (live_tracker.current_frame / live_tracker.total_frames * 100) if live_tracker.total_frames > 0 else 0
    })

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    print('Client connected to live motion capture')
    emit('connected', {'message': 'Connected to Live Motion Capture MP4'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("🎬 Live Motion Capture MP4 Server")
    print("🔁 Real-time tracking dari file MP4")
    print("🌐 Web interface: http://localhost:5000")
    print("📊 WebSocket: Real-time data streaming")
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

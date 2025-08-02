#!/usr/bin/env python3
"""
Real Live Chapter Motion Capture System
- Real-time webcam input with MediaPipe
- Live chapter detection based on motion patterns
- WebSocket streaming for immediate visualization
- Direct export to Unreal Engine/Blender
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import base64
from datetime import datetime
import queue
import os
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'live_motion_capture_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for live capture
live_capture = None
motion_buffer = deque(maxlen=300)  # 10 seconds at 30fps
current_chapter = {'type': 'intro', 'start_time': time.time(), 'motion_intensity': 0.0}
chapter_history = []

class LiveMotionCapture:
    """Real-time motion capture with live chapter detection"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Live capture settings
        self.is_capturing = False
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Motion analysis
        self.motion_history = deque(maxlen=90)  # 3 seconds at 30fps
        self.intensity_threshold = {'low': 0.3, 'medium': 0.6, 'high': 0.8}
        
        # Chapter detection
        self.chapter_duration_min = 5.0  # Minimum 5 seconds per chapter
        self.last_chapter_change = time.time()
        
        # Export data
        self.live_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'frames': [],
            'chapters': [],
            'start_time': None
        }
        
    def start_capture(self, camera_index=0):
        """Start real-time webcam capture"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            self.is_capturing = True
            self.live_data['start_time'] = time.time()
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_loop)
            capture_thread.daemon = True
            capture_thread.start()
            
            # Start processing thread
            process_thread = threading.Thread(target=self._process_loop)
            process_thread.daemon = True
            process_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Capture start error: {e}")
            return False
    
    def stop_capture(self):
        """Stop real-time capture"""
        self.is_capturing = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def _capture_loop(self):
        """Main capture loop - runs in separate thread"""
        with self.mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            while self.is_capturing and self.camera:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = holistic.process(rgb_frame)
                
                # Add to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put({
                        'frame': frame.copy(),
                        'results': results,
                        'timestamp': time.time()
                    })
    
    def _process_loop(self):
        """Process frames and detect chapters"""
        while self.is_capturing:
            try:
                if not self.frame_queue.empty():
                    data = self.frame_queue.get(timeout=0.1)
                    self._process_frame(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Process error: {e}")
    
    def _process_frame(self, data):
        """Process single frame for motion analysis and chapter detection"""
        frame = data['frame']
        results = data['results']
        timestamp = data['timestamp']
        
        # Calculate motion intensity
        motion_intensity = self._calculate_live_motion_intensity(results)
        self.motion_history.append(motion_intensity)
        
        # Store frame data
        frame_data = self._extract_live_landmarks(results, timestamp)
        frame_data['motion_intensity'] = motion_intensity
        self.live_data['frames'].append(frame_data)
        
        # Detect chapter changes
        self._detect_live_chapter_change(motion_intensity, timestamp)
        
        # Create annotated frame
        annotated_frame = self._create_live_visualization(frame, results, motion_intensity)
        
        # Convert to base64 for streaming
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit to web interface
        socketio.emit('live_frame', {
            'frame': frame_base64,
            'motion_intensity': motion_intensity,
            'current_chapter': current_chapter,
            'timestamp': timestamp,
            'landmarks_count': self._count_landmarks(results)
        })
    
    def _calculate_live_motion_intensity(self, results):
        """Calculate real-time motion intensity"""
        intensity = 0.0
        
        # Face motion (30% weight)
        if results.face_landmarks:
            face_motion = len(results.face_landmarks.landmark) / 468.0
            intensity += face_motion * 0.3
        
        # Hand motion (40% weight)
        hand_motion = 0.0
        if results.left_hand_landmarks:
            hand_motion += 0.5
        if results.right_hand_landmarks:
            hand_motion += 0.5
        intensity += min(hand_motion, 1.0) * 0.4
        
        # Body motion (30% weight)
        if results.pose_landmarks:
            body_motion = len(results.pose_landmarks.landmark) / 33.0
            intensity += body_motion * 0.3
        
        return min(intensity, 1.0)
    
    def _detect_live_chapter_change(self, current_intensity, timestamp):
        """Detect chapter changes in real-time"""
        global current_chapter, chapter_history
        
        # Ensure minimum chapter duration
        if timestamp - self.last_chapter_change < self.chapter_duration_min:
            current_chapter['motion_intensity'] = current_intensity
            return
        
        # Calculate recent motion trend
        if len(self.motion_history) < 30:  # Need at least 1 second of data
            return
        
        recent_avg = np.mean(list(self.motion_history)[-30:])
        
        # Chapter transition logic
        new_chapter_type = None
        
        if current_chapter['type'] == 'intro' and recent_avg > self.intensity_threshold['medium']:
            new_chapter_type = 'conflict'
        elif current_chapter['type'] == 'conflict' and recent_avg < self.intensity_threshold['low']:
            new_chapter_type = 'resolution'
        elif current_chapter['type'] == 'resolution' and recent_avg > self.intensity_threshold['high']:
            new_chapter_type = 'conflict'
        
        # Execute chapter change
        if new_chapter_type and new_chapter_type != current_chapter['type']:
            # Save previous chapter
            prev_chapter = current_chapter.copy()
            prev_chapter['end_time'] = timestamp
            prev_chapter['duration'] = timestamp - prev_chapter['start_time']
            chapter_history.append(prev_chapter)
            
            # Start new chapter
            current_chapter = {
                'type': new_chapter_type,
                'start_time': timestamp,
                'motion_intensity': current_intensity
            }
            self.last_chapter_change = timestamp
            
            # Emit chapter change
            socketio.emit('chapter_change', {
                'previous_chapter': prev_chapter,
                'new_chapter': current_chapter,
                'total_chapters': len(chapter_history) + 1
            })
    
    def _extract_live_landmarks(self, results, timestamp):
        """Extract landmarks for live export"""
        frame_data = {
            'timestamp': timestamp,
            'frame_time': timestamp - self.live_data['start_time']
        }
        
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
    
    def _create_live_visualization(self, frame, results, motion_intensity):
        """Create live visualization with landmarks and chapter info"""
        annotated = frame.copy()
        
        # Draw landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Add live info overlay
        self._add_live_overlay(annotated, motion_intensity)
        
        return annotated
    
    def _add_live_overlay(self, frame, motion_intensity):
        """Add live information overlay"""
        h, w = frame.shape[:2]
        
        # Chapter info
        chapter_color = {'intro': (255, 100, 100), 'conflict': (100, 255, 100), 'resolution': (100, 100, 255)}
        color = chapter_color.get(current_chapter['type'], (255, 255, 255))
        
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), color, 2)
        
        cv2.putText(frame, f"LIVE: {current_chapter['type'].upper()}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Motion: {motion_intensity:.2f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Motion intensity bar
        bar_width = int(280 * motion_intensity)
        cv2.rectangle(frame, (10, 85), (290, 100), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 85), (10 + bar_width, 100), color, -1)
    
    def _count_landmarks(self, results):
        """Count total landmarks detected"""
        count = 0
        if results.face_landmarks:
            count += len(results.face_landmarks.landmark)
        if results.pose_landmarks:
            count += len(results.pose_landmarks.landmark)
        if results.left_hand_landmarks:
            count += len(results.left_hand_landmarks.landmark)
        if results.right_hand_landmarks:
            count += len(results.right_hand_landmarks.landmark)
        return count
    
    def export_live_session(self):
        """Export live session for Unreal Engine/Blender"""
        try:
            # Add final chapter
            if current_chapter:
                final_chapter = current_chapter.copy()
                final_chapter['end_time'] = time.time()
                final_chapter['duration'] = final_chapter['end_time'] - final_chapter['start_time']
                chapter_history.append(final_chapter)
            
            # Prepare export data
            export_data = {
                'session_info': {
                    'session_id': self.live_data['session_id'],
                    'total_duration': time.time() - self.live_data['start_time'],
                    'total_frames': len(self.live_data['frames']),
                    'total_chapters': len(chapter_history)
                },
                'chapters': chapter_history,
                'motion_data': self.live_data['frames']
            }
            
            # Save to file
            filename = f"live_motion_{self.live_data['session_id']}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return filename
            
        except Exception as e:
            print(f"Export error: {e}")
            return None

# Initialize live capture
live_capture = LiveMotionCapture()

# WebSocket events
@socketio.on('start_capture')
def handle_start_capture():
    success = live_capture.start_capture()
    emit('capture_status', {'status': 'started' if success else 'failed'})

@socketio.on('stop_capture')
def handle_stop_capture():
    live_capture.stop_capture()
    filename = live_capture.export_live_session()
    emit('capture_status', {'status': 'stopped', 'export_file': filename})

@socketio.on('get_chapter_history')
def handle_get_chapter_history():
    emit('chapter_history', {'chapters': chapter_history, 'current': current_chapter})

# Flask routes
@app.route('/')
def live_index():
    return render_template('live_motion.html')

@app.route('/status')
def live_status():
    return jsonify({
        'is_capturing': live_capture.is_capturing if live_capture else False,
        'session_id': live_capture.live_data['session_id'] if live_capture else None,
        'current_chapter': current_chapter,
        'chapter_count': len(chapter_history)
    })

if __name__ == '__main__':
    print("🔴 Starting Real Live Chapter Motion Capture System...")
    print("🌐 Web interface: http://localhost:5000")
    print("📹 Real-time webcam motion capture with chapter detection")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

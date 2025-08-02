# MediaPipe to FBX Animation Pipeline

A complete Python solution for tracking face, hand, and multi-person pose from MP4 videos and converting to FBX animation files compatible with Unreal Engine characters like WhiteRobot Man.

## Features

- **Face Tracking**: 468 facial landmarks with expression analysis
- **Hand Tracking**: Full hand pose estimation for both hands
- **Multi-Person Pose**: Body pose tracking for multiple people
- **Real-time Processing**: Optimized for performance with confidence thresholds
- **FBX Export**: Direct export to FBX format via Blender
- **Unreal Engine Ready**: Compatible with UE4/UE5 character rigs
- **Multiple Formats**: Export to FBX, BVH, and JSON formats

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Blender Setup (for FBX export)

1. Download and install [Blender](https://www.blender.org/download/) (3.0 or later)
2. The script includes Blender Python integration for FBX export

### 3. Required Packages

```bash
# Core packages
pip install mediapipe opencv-python numpy scipy

# Optional for advanced features
pip install tensorflow torch torchvision moviepy

# Development tools
pip install pytest black flake8
```

## Quick Start

### 1. Basic Usage

```python
from mediapipe_to_fbx import MediaPipeTracker

# Create tracker
tracker = MediaPipeTracker()

# Process video
tracker.process_video("input_video.mp4", "output_directory")
```

### 2. Complete Pipeline

```python
# Run the example script
python example_usage.py
```

### 3. Blender FBX Export

1. Open Blender
2. Load `blender_fbx_exporter.py` in the Text Editor
3. Update file paths in the script
4. Run the script (Alt+P)

## File Structure

```
📁 MediaPipe-to-FBX/
├── 📄 mediapipe_to_fbx.py      # Main tracking script
├── 📄 config.py                # Configuration settings
├── 📄 utils.py                 # Utility functions
├── 📄 blender_fbx_exporter.py  # Blender integration script
├── 📄 example_usage.py         # Example usage and batch processing
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # This file
└── 📁 output/                 # Generated output files
    ├── tracking_data.json     # Raw tracking data
    ├── character_animation.fbx # FBX animation file
    ├── preview_video.mp4      # Preview with overlays
    └── preview_frames/        # Individual frame images
```

## Configuration

Edit `config.py` to customize tracking parameters:

```python
# Confidence thresholds (0.0 - 1.0)
FACE_DETECTION_CONFIDENCE = 0.6
HAND_DETECTION_CONFIDENCE = 0.7
POSE_DETECTION_CONFIDENCE = 0.5

# Processing settings
MAX_NUM_FACES = 3
MAX_NUM_HANDS = 6
POSE_MODEL_COMPLEXITY = 2  # 0=fast, 1=balanced, 2=accurate
```

## Usage Examples

### Single Video Processing

```python
from example_usage import process_single_video

# Process one video
process_single_video("my_video.mp4", "output_folder")
```

### Batch Processing

```python
from example_usage import batch_process_videos

# Process all MP4 files in a directory
batch_process_videos("input_videos/", "batch_output/")
```

### Custom Configuration

```python
from mediapipe_to_fbx import MediaPipeTracker

# Create tracker with custom settings
tracker = MediaPipeTracker(
    face_detection_confidence=0.8,
    hand_detection_confidence=0.6,
    pose_detection_confidence=0.7
)

# Process video
tracker.process_video("input.mp4", "output")
```

## Output Formats

### 1. JSON Data
Raw tracking data with timestamps and landmark coordinates:
```json
{
  "fps": 30.0,
  "total_frames": 900,
  "persons": {
    "0": {
      "frames": [
        {
          "timestamp": 0.0,
          "face_landmarks": [[x, y, z], ...],
          "left_hand_landmarks": [[x, y, z], ...],
          "pose_landmarks": [[x, y, z], ...]
        }
      ]
    }
  }
}
```

### 2. FBX Animation
- Armature with humanoid bone structure
- Keyframed animation data
- Compatible with Unreal Engine
- Includes IK chains for natural movement

### 3. BVH Format
- Motion capture standard format
- Compatible with most 3D software
- Hierarchical bone structure

### 4. Unreal Engine Format
- Optimized for UE4/UE5 import
- Bone name mapping
- Morph target data for facial expressions

## Blender Integration

The `blender_fbx_exporter.py` script provides:

- **Character Rig Creation**: Automatic humanoid armature setup
- **Animation Application**: Converts MediaPipe data to bone rotations
- **IK Constraints**: Natural limb movement
- **Mesh Deformation**: Character mesh binding
- **FBX Export**: Optimized for game engines

### Blender Workflow

1. **Load Script**: Open `blender_fbx_exporter.py` in Blender
2. **Update Paths**: Modify file paths in the script
3. **Run Script**: Press Alt+P to execute
4. **Export**: FBX file is automatically exported

## Character Compatibility

### Supported Character Types

- **Humanoid Characters**: Standard bipedal rigs
- **Robot Characters**: Servo-based joint systems
- **Custom Rigs**: Configurable bone mapping

### Unreal Engine Setup

1. Import FBX into Unreal Engine
2. Create Animation Blueprint
3. Map bone names to your character
4. Configure IK chains if needed

### Bone Mapping

The system maps MediaPipe landmarks to standard bone names:

```python
BONE_MAPPINGS = {
    "humanoid": {
        0: "Head",           # Nose
        11: "LeftShoulder",  # Left shoulder
        12: "RightShoulder", # Right shoulder
        # ... additional mappings
    }
}
```

## Performance Optimization

### Processing Speed
- **Model Complexity**: Use `POSE_MODEL_COMPLEXITY = 1` for faster processing
- **Frame Skipping**: Process every N frames for real-time applications
- **Resolution**: Lower input resolution for faster tracking

### Accuracy vs Speed
- **High Accuracy**: `model_complexity=2`, high confidence thresholds
- **Fast Processing**: `model_complexity=0`, lower confidence thresholds
- **Balanced**: Default configuration provides good balance

## Troubleshooting

### Common Issues

1. **"No module named 'mediapipe'"**
   ```bash
   pip install mediapipe
   ```

2. **"Blender Python API not available"**
   - Run the script inside Blender, not from command line
   - Install Blender and use its Python environment

3. **Low detection rates**
   - Lower confidence thresholds in `config.py`
   - Ensure good lighting in input video
   - Check video resolution and quality

4. **FBX not importing in Unreal**
   - Check FBX version compatibility
   - Verify bone naming conventions
   - Scale factor may need adjustment

### Performance Issues

1. **Slow processing**
   - Reduce `POSE_MODEL_COMPLEXITY`
   - Process every 2nd or 3rd frame
   - Use GPU acceleration if available

2. **Memory usage**
   - Process videos in smaller chunks
   - Clear intermediate data
   - Use lower resolution input

## Advanced Features

### Multi-Person Tracking

```python
# Configure for multiple people
tracker = MediaPipeTracker()
tracker.max_num_faces = 5
tracker.max_num_hands = 10  # 5 people × 2 hands
```

### Custom Expressions

```python
# Add custom facial expressions
expressions = tracker.calculate_face_expressions(face_landmarks)
print(expressions)  # {'smile': 0.8, 'eye_blink': 0.1, ...}
```

### Hand Gestures

```python
# Detect hand gestures
gestures = tracker.calculate_hand_gestures(hand_landmarks)
print(gestures)  # {'fist': 0.9, 'finger_0_curl': 0.2, ...}
```

## API Reference

### MediaPipeTracker Class

#### Methods
- `__init__(confidence_params)`: Initialize tracker
- `process_video(video_path, output_dir)`: Process MP4 file
- `extract_landmarks_from_results(results)`: Extract landmark data
- `calculate_face_expressions(landmarks)`: Compute facial expressions
- `calculate_hand_gestures(landmarks)`: Compute hand gestures

#### Properties
- `tracked_persons`: Dictionary of person tracking data
- `frame_count`: Number of processed frames
- `fps`: Video frame rate

### BlenderFBXExporter Class

#### Methods
- `create_character_rig(name)`: Create armature
- `apply_pose_animation(data, armature)`: Apply body animation
- `apply_face_animation(data, armature)`: Apply facial animation
- `export_fbx(output_path)`: Export to FBX format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **MediaPipe**: Google's ML framework for real-time perception
- **Blender**: Open-source 3D creation suite
- **OpenCV**: Computer vision library
- **Unreal Engine**: Epic Games' game engine

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include input video characteristics and error messages

---

**Happy Motion Capturing!** 🎬✨

# MediaPipe Studio - Part 4: Web-Based Animation Pipeline

🎬 **Advanced Web Interface for Motion Capture & Animation Export**

## 🌟 Features

### ✅ Core Capabilities
- **Full-Body Markerless Tracking** - Complete body pose estimation
- **Face Tracking** - 468 facial landmarks with expressions
- **Multi-Person Support** - Track multiple subjects simultaneously
- **Real-time Preview** - Live MediaPipe visualization
- **Web Interface** - Modern browser-based UI at `http://localhost:3000`

### 📤 Export Formats
- **UE4** - Unreal Engine 4 compatible FBX
- **UE5** - Unreal Engine 5 optimized FBX with MetaHuman support
- **Unity** - Unity compatible animation format
- **Roblox** - R15/R6 keyframe sequences
- **MetaHuman** - Epic Games MetaHuman facial animation
- **Animation Data** - Raw JSON with complete tracking data

### 🎯 Video Format Support
- **H.264 Optimization** - Automatic transcoding for best compatibility
- **Multiple Codecs** - MP4, AVI, MOV, MKV, WebM, M4V
- **Drag & Drop Upload** - Easy file selection
- **Large File Support** - Up to 500MB video files

## 🚀 Quick Start

### Method 1: Simple Startup
```cmd
start_studio.bat
```

### Method 2: Manual Launch
```cmd
# Install dependencies
pip install -r requirements.txt

# Start the server
python mediapipe_studio.py
```

### Method 3: Direct Python
```cmd
"C:/Program Files/Python311/python.exe" mediapipe_studio.py
```

## 🌐 Web Interface

1. **Open Browser**: Navigate to `http://localhost:3000`
2. **Upload Video**: Drag & drop or click to select your video file
3. **Configure Settings**: 
   - Select tracking features (body, face, multi-person)
   - Choose export formats (UE4/UE5, Unity, Roblox, MetaHuman)
   - Adjust quality settings
4. **Process**: Click "Start Processing" and monitor real-time progress
5. **Download**: Get your exported animation files

## ⚙️ Advanced Settings

### 🎚️ Quality Controls
- **Model Complexity**: 0 (fast) to 2 (accurate)
- **Detection Confidence**: 0.1 to 1.0
- **Tracking Confidence**: 0.1 to 1.0
- **Face Refinement**: Enhanced facial landmark precision
- **Segmentation**: Background separation (optional)

### 🎯 Export Options
- **UE4/UE5**: Mannequin-compatible bone structure
- **Unity**: Humanoid rig compatible
- **Roblox**: R15 avatar animations
- **MetaHuman**: Facial blendshapes and expressions
- **Raw Data**: Complete MediaPipe landmarks

## 📁 Project Structure

```
MediaPipe Studio/
├── mediapipe_studio.py      # Main web server
├── templates/
│   └── index.html          # Web interface
├── uploads/                # Video upload storage
├── outputs/                # Processed results
├── temp/                   # Temporary processing files
├── requirements.txt        # Python dependencies
├── start_studio.bat       # Windows startup script
└── README.md              # This file
```

## 🔧 Technical Details

### Core Technologies
- **Backend**: Flask web framework
- **Frontend**: Modern HTML5 + CSS3 + JavaScript
- **Processing**: MediaPipe Holistic model
- **Export**: Multi-format animation generation

### Performance Optimizations
- **Background Processing**: Non-blocking video analysis
- **Progress Tracking**: Real-time status updates
- **Memory Management**: Efficient frame processing
- **File Optimization**: H.264 transcoding for compatibility

### API Endpoints
- `POST /upload` - Video file upload
- `POST /process` - Start processing job
- `GET /status/<job_id>` - Processing status
- `POST /cancel/<job_id>` - Cancel processing
- `GET /download/<job_id>/<format>` - Download results
- `GET /preview/<job_id>` - Animation preview data

## 🎮 Integration Guides

### Unreal Engine 4/5
1. Download UE4 or UE5 FBX file
2. Import as Skeletal Mesh in UE4/UE5
3. Set import settings to "Humanoid"
4. Apply to UE4/UE5 Mannequin character
5. Animation ready for use!

### Unity
1. Download Unity FBX file
2. Import into Unity project
3. Set Rig type to "Humanoid"
4. Apply to any humanoid character
5. Play animation in Timeline or Animator

### Roblox
1. Download Roblox JSON file
2. Upload to Roblox Studio
3. Convert to Animation object
4. Apply to R15 character
5. Play in game scripts

### MetaHuman (UE5)
1. Download MetaHuman FBX file
2. Import into UE5 with MetaHuman project
3. Map facial animations to MetaHuman rig
4. Combine with body animations
5. Full character animation ready!

## 🔍 Troubleshooting

### Common Issues
- **Port 3000 in use**: Change port in `mediapipe_studio.py`
- **Large file upload fails**: Check `MAX_CONTENT_LENGTH` setting
- **Processing stops**: Check Python console for errors
- **Missing dependencies**: Run `pip install -r requirements.txt`

### Performance Tips
- Use H.264 encoded videos for best performance
- Lower model complexity for faster processing
- Close other applications during processing
- Use SSD storage for faster file operations

## 📊 Output File Formats

### FBX Files (UE4/UE5/Unity/MetaHuman)
- Standard FBX 7.4 format
- Complete bone hierarchy
- Animation curves with keyframes
- Compatible with major 3D software

### JSON Files (Roblox/Animation Data)
- Human-readable format
- Complete tracking data
- Frame-by-frame landmarks
- Easy to parse and modify

### Preview Data
- Real-time visualization
- Sample frame rendering
- Progress tracking
- Status monitoring

## 🆕 New in Part 4

- **Web-based interface** replacing command-line
- **Real-time progress tracking** with ETA
- **Multiple export formats** in single processing
- **Advanced settings controls** via web UI
- **Drag & drop file upload** with format validation
- **Background processing** with cancellation support
- **MetaHuman compatibility** for facial animation
- **Multi-person tracking** support
- **H.264 optimization** for video compatibility

## 🔄 Migration from Previous Parts

### From Part 3 (UE Pipeline)
- All UE4/UE5 functionality preserved
- Enhanced with web interface
- Additional export formats added
- Better real-time preview

### From Part 2 (FBX Export)
- All export capabilities maintained
- Web-based file management
- Progress tracking added
- Multiple format export

### From Part 1 (Basic Processing)
- Core MediaPipe processing unchanged
- Web UI replaces command line
- Enhanced error handling
- Better user experience

## 🎯 Use Cases

### Game Development
- Character animation for games
- Motion capture for indie projects
- Rapid prototyping animations
- Cross-platform compatibility

### Film & Media
- Pre-visualization animations
- Character studies
- Motion reference
- VFX pipeline integration

### Virtual Production
- Real-time character control
- MetaHuman facial animation
- UE5 virtual sets
- Live performance capture

### Education & Research
- Motion analysis studies
- Biomechanics research
- Animation learning
- Computer vision projects

## 🌟 Future Enhancements

- **Batch Processing**: Multiple video processing
- **Cloud Processing**: Server-side processing options
- **Real-time Streaming**: Live camera input
- **Advanced Facial**: Emotion and expression analysis
- **Custom Rigs**: User-defined skeleton structures
- **Plugin System**: Extensible export formats

---

**MediaPipe Studio - Part 4** brings professional motion capture capabilities to your browser with comprehensive export options for all major animation platforms! 🎬✨

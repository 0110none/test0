# Single-Camera Face Privacy System 🔒

A privacy-focused desktop application that processes **one** live video source at a time. The system performs per-frame face detection on the configured webcam or video file, keeps registered faces clear, and automatically blurs unrecognized faces before displaying them in the user interface.

## 🌟 Key Features

### Core Capabilities
- **Real-Time Privacy Protection** – Every frame is analysed and unknown faces are blurred instantly (Gaussian blur).
- **Selective Visibility** – Registered faces remain sharp for monitoring and analysis.
- **Single-Source Input** – Point the app at exactly one webcam, RTSP stream, or video file defined in the config file.
- **Live Statistics** – Status panel reports the number of detected faces and how many are currently blurred.
- **Per-Frame Processing** – Detection and recognition run on each frame with no throttling interval.

### User Experience
- 🖥️ **Monitoring Dashboard** – View the configured camera feed in real time with privacy filtering applied.
- 👤 **Face Management** – Upload and manage the known face library directly in the UI.
- 🎛️ **Camera Controls** – Start/stop the stream, and adjust recognition/blur thresholds without editing code.

## 🛠️ Technical Stack

| Component            | Technology Used |  
|----------------------|-----------------|  
| Face Detection       | InsightFace     |  
| Machine Learning     | PyTorch         |  
| Computer Vision      | OpenCV          |  
| GUI Framework        | PyQt5           |

## 📦 Installation Guide

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended, CPU mode supported)
- FFmpeg (required for RTSP streams)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AarambhDevHub/multi-cam-face-tracker.git
   cd multi-cam-face-tracker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the system**
   - Edit `config/config.yaml` for application settings
   - Edit `config/camera_config.yaml` for the single camera/stream configuration

5. **Directory setup**
   ```bash
   mkdir -p data/known_faces logs
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Application Settings (`config/config.yaml`)
```yaml
app:
  name: "Face"
  version: "1.0.0"
  known_faces_dir: "data/known_faces"
  logo: "assets/logo.png"
  log_dir: "logs"

recognition:
  detection_threshold: 0.5
  recognition_threshold: 0.6
  device: "cuda"  # or "cpu"
  analysis_enabled: true
```

### Camera Configuration (`config/camera_config.yaml`)
```yaml
camera:
  id: 0
  name: "Front Camera"
  source: 0                  # Camera index, RTSP URL, or path to a video file
  enabled: true
  resolution:
    width: 1280
    height: 720
  fps: 30
  rotate: 0
```

> ℹ️ Only one camera definition is honoured. Change the `source` value if you want to switch to another device or file.

## 🖥️ User Manual

### Managing Known Faces
1. Open **工具 → 人脸管理**.
2. Import a clear photo and assign a name (duplicate names are prevented).
3. Save to refresh the recognition library immediately.

### Working with the Camera Feed
- Press **启动摄像头** or **停止摄像头** in the **控制** tab to control the single configured stream.
- Adjust the recognition threshold slider to fine-tune matching sensitivity.
- Modify `config/camera_config.yaml` if you need to change to a different device or video file, then restart the app.

### Monitoring & Status
- The **监控** tab displays the single stream with blurred strangers and annotated known faces.
- The status panel summarises:
  - Current camera run state
  - Number of registered faces in the library
  - Real-time counts of detected faces and blurred faces

## 📚 Additional Notes
- The application performs detection and blur operations on every frame; ensure adequate hardware for sustained real-time processing of your chosen resolution.
- Logs are written to the directory defined by `app.log_dir` for troubleshooting.
- Screenshots, alerts, Telegram notifications, and database history have been removed to focus on privacy-first monitoring.

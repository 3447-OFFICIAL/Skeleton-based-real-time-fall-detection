# 🧘 Skeleton-Based Fall Detection System

<div align="center">
  <p align="center">
    <b>Real-time human pose estimation and automated fall classification using MediaPipe and Machine Learning.</b>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/MediaPipe-0.8+-orange.svg?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe" />
    <img src="https://img.shields.io/badge/PySide6-Qt-green.svg?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6" />
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License" />
  </p>
</div>

---

## 📖 Overview

The **Skeleton-Based Fall Detection System** is a modular desktop application designed for healthcare monitoring and elderly care. By leveraging **MediaPipe Pose Estimation**, the system extracts 3D skeletal landmarks in real-time and utilizes a **Random Forest Classifier** to detect sudden falls with high accuracy.

Unlike traditional camera-based systems that rely on raw pixels, this system uses **skeleton-based features**, ensuring privacy (by not storing raw video) and robustness against varied lighting conditions or complex backgrounds.

---

## ✨ Key Features

- **🎥 Dual Input Support**:
  - **Live Stream**: Real-time monitoring via system webcam.
  - **Video Analysis**: Process pre-recorded files (`.mp4`, `.avi`, etc.).
- **🦴 33-Point Pose Mesh**: High-fidelity skeleton mapping and visualization.
- **🤖 Intelligent Classification**:
  - Binary Fall/No-Fall detection.
  - Automated threshold-based alerts.
  - Synthetic model auto-generation for quick deployment.
- **📊 Advanced Feature Engineering**:
  - **Vertical Velocity**: Tracking sudden Y-axis acceleration of landmarks.
  - **Body Aspect Ratio (BAR)**: Analyzing torso-to-limb ratios for orientation shifts.
  - **Ground Proximity Index**: Real-time distance calculation from frame boundaries.
- **💻 Premium Desktop Interface**:
  - Dark-themed, high-performance UI built with **PySide6**.
  - Non-blocking multi-threaded processing for zero-latency feedback.
  - Real-time diagnostic logs and system status indicators.

---

## 🛠️ Project Structure

```text
Skeleton-based-fall-detection/
├── 📂 gui/                # UI Components & Threading
│   ├── main_window.py     # Main application layout
│   └── video_thread.py    # Non-blocking processing logic
├── 📂 vision/             # Computer Vision Wrappers
│   ├── pose_detector.py   # MediaPipe integration
│   └── camera_check.py    # Hardware diagnostics
├── 📂 ml/                 # Machine Learning Pipeline
│   ├── inference.py       # Model prediction logic
│   ├── generate_model.py  # Synthetic data & model training
│   └── model.pkl          # Serialized classifier
├── 📂 utils/              # Mathematical Utilities
│   └── feature_extraction.py # Pose-to-feature conversion
├── 📂 logs/               # Automated Session Logs
├── 📄 main.py             # System Application Entry
├── 📄 requirements.txt    # Dependency Manifest
├── 📄 run_app.bat         # Windows Quick-Launch Script
└── 📄 LICENSE             # MIT License
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+** (Tested on Windows/Linux)
- A webcam (for Live Mode)
- `pip` package manager

### 2. Installation
Clone the repository and install the core dependencies:

```bash
# Clone the repository
git clone https://github.com/3447-OFFICIAL/Skeleton-based-real-time-fall-detection.git

# Navigate to project root
cd Skeleton-based-real-time-fall-detection

# Install required packages
pip install -r requirements.txt
```

### 3. Launching the App
#### Windows (Quickest)
Simply double-click the `run_app.bat` file in the root directory.

#### Manual (CLI)
```bash
python main.py
```

---

## 🧠 Technical Approach

### Feature Extraction Pipe
The system extracts skeletal data and converts it into a 12-dimensional feature vector every frame:
1. **Coordinate Normalization**: Landmarks are scaled relative to the bounding box.
2. **Velocity Calculation**: $V_y = \Delta Y / \Delta t$ of the hips and torso.
3. **Angle Analysis**: Calculating joint angles at the hips and knees to detect "slump" or "collapse" patterns.

### Classification Logic
The Random Forest classifier analyzes a temporal window of 15-30 frames to differentiate between "sitting down quickly" and a "sudden fall".

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Developed with ❤️ for Advanced Healthcare Robotics and Monitoring.</sub>
</div>

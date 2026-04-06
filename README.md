# 🧘 Skeleton-Based Fall Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PySide6](https://img.shields.io/badge/UI-PySide6-green.svg)](https://pypi.org/project/PySide6/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-orange.svg)](https://mediapipe.dev/)

A sophisticated, real-time fall detection system leveraging **MediaPipe Pose Estimation** and **Machine Learning (Random Forest)**. Designed for healthcare monitoring and elderly care, this application provides synchronous skeleton visualization and automated alert logging.

---

## ✨ Key Features

-   **🎥 Dual-Mode Processing**: Seamlessly switch between live Webcam monitoring and offline Video File analysis.
-   **🦴 Real-time Pose Mesh**: High-fidelity 33-point skeleton overlay powered by MediaPipe.
-   **🤖 Intelligent Classification**: Binary fall detection using a Random Forest classifier trained on custom movement features.
-   **📊 Feature Engineering**: Extracts critical metrics like vertical velocity, joint angles (knees, hips), and torso orientation.
-   **💻 Modern Desktop UI**: A sleek, dark-themed interface built with **PySide6** (Qt) featuring:
    -   Live video canvas with minimal latency.
    -   Real-time "System Status" and "Fall Status" indicators.
    -   Persistent Event Log with timestamps.

---

## 🛠️ Getting Started

### 1. Prerequisites

-   **Python 3.9 or higher**
-   A stable internet connection (for initial MediaPipe model download)
-   A webcam (optional, for live mode)

### 2. Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MINI_PROJECT_3.git
cd MINI_PROJECT_3

# Install dependencies
pip install PySide6 opencv-python mediapipe numpy scikit-learn pandas
```

### 3. Running the Application

Simply execute the main entry point:

```bash
python main.py
```

*Note: On the first run, the system will automatically generate a synthetic machine learning model (`ml/model.pkl`) if one isn't present.*

---

## 🏗️ System Architecture

The project follows a modular design for easy extensibility:

-   📂 **`main.py`**: Entry point that handles application initialization and layout.
-   📂 **`gui/`**: Contains the `main_window.py` and processing threads for non-blocking UI.
-   📂 **`vision/`**: MediaPipe wrapper classes for pose estimation.
-   📂 **`ml/`**: Model inference, training scripts, and serialization.
-   📂 **`utils/`**: Core mathematical utilities for skeleton feature extraction.
-   📂 **`logs/`**: Directory for automated session logging.

---

## 🧠 How it Works

1.  **Pose Detection**: MediaPipe extracts 3D coordinates for 33 key body landmarks.
2.  **Normalization**: Landmark coordinates are normalized relative to the frame size.
3.  **Feature Vector**: We calculate:
    -   **Vertical Velocity**: Change in Y-coordinates of the center of mass.
    -   **Body Aspect Ratio**: Ratio between bounding box width and height.
    -   **Ground Proximity**: Distance of the hips/shoulders from the lower frame boundary.
4.  **Inference**: A sliding window of these features is fed into the Random Forest model to predict the probability of a "Fall" state.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed as part of MINI_PROJECT_3.*

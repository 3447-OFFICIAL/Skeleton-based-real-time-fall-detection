import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage
import time

from vision.pose_detector import PoseDetector
from utils.feature_extraction import FeatureExtractor
from ml.inference import FallDetector

class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)
    status_signal = Signal(str, float, bool) # Prediction, Confidence, Fall Detected (bool)
    log_signal = Signal(str)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self._run_flag = True
        # Use 'full' for balanced performance, or 'lite' for max speed
        self.detector = PoseDetector(mode="full") 
        self.extractor = FeatureExtractor()
        self.fall_detector = FallDetector()
        self.is_camera = (source == 0)
        
        # Performance optimizations
        self.frame_count = 0
        self.skip_frames = 2 # Process 1 in 2 frames
        self.last_status = ("SAFE", 0.0, False)
        self.last_landmarks = None

    def run(self):
        # CAP_DSHOW for Windows webcam
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)
            
        if not cap.isOpened():
            self.log_signal.emit(f"Error: Could not open source {self.source}")
            return

        start_time = time.time()
        
        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret:
                if self.is_camera:
                    continue
                else:
                    self.log_signal.emit("Video reached end.")
                    break

            self.frame_count += 1
            timestamp_ms = int((time.time() - start_time) * 1000)

            # 1. Pose Detection (Run every N-th frame)
            if self.frame_count % self.skip_frames == 0:
                cv_img = self.detector.find_pose(cv_img, timestamp_ms=timestamp_ms)
                self.last_landmarks = self.detector.get_landmarks(cv_img)
                
                # 2. Feature Extraction & Prediction
                if self.last_landmarks:
                    # In pose_detector.py, get_landmarks returns [id, x, y, z, presence]
                    # feature_extraction expects [id, x, y, z, cx, cy, visibility]
                    # Let's map it:
                    h, w, _ = cv_img.shape
                    full_landmarks = []
                    for lm in self.last_landmarks:
                        cx, cy = int(lm[1] * w), int(lm[2] * h)
                        full_landmarks.append([lm[0], lm[1], lm[2], lm[3], cx, cy, lm[4]])
                    
                    features = self.extractor.extract_features(full_landmarks, cv_img.shape)
                    if features:
                        self.fall_detector.add_features(features)
                        prediction, confidence = self.fall_detector.predict()
                        
                        status_text = "FALL DETECTED" if prediction == 1 else "SAFE"
                        is_fall = (prediction == 1)
                        self.last_status = (status_text, confidence, is_fall)
                        self.status_signal.emit(status_text, confidence, is_fall)

            # 3. Visualization & Overlay (Always draw from cache)
            if self.last_landmarks:
                status_text, confidence, is_fall = self.last_status
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                
                # Draw bounding box from last known landmarks
                bbox = self.detector.get_bounding_box(cv_img, self.last_landmarks)
                if bbox:
                    x, y, bw, bh = bbox
                    cv2.rectangle(cv_img, (x, y), (x + bw, y + bh), color, 2)
                    
                # Status Text
                cv2.putText(cv_img, f"{status_text} ({confidence:.2f})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 4. Convert to QImage and Emit
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = qt_img.scaled(800, 600, Qt.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

            # Sleep control for video files
            if not self.is_camera:
                # Still try to maintain 30 FPS relative to wall clock
                # In real apps, we'd use time.sleep(1/fps) minus processing time.
                time.sleep(0.01) # Small sleep to avoid eating 100% CPU on UI thread

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

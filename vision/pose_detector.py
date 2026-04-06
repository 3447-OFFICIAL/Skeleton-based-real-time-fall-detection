import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

class PoseDetector:
    def __init__(self, mode="full"):
        """
        mode: "lite", "full", or "heavy"
        """
        # MediaPipe Tasks API
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Model selection
        models = {
            "lite": "pose_landmarker_lite.task",
            "full": "pose_landmarker_full.task", 
            "heavy": "pose_landmarker_heavy.task"
        }
        model_name = models.get(mode, "pose_landmarker_full.task")
        self.model_path = os.path.join("vision", model_name)
        
        # Download model if not exists
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe model ({model_name})...")
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name.replace('.task', '')}/float16/1/{model_name}"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

        # Configuration for Video mode (optimized for frame sequences)
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO
        )
        self.landmarker = self.PoseLandmarker.create_from_options(options)
        self.results = None

    def find_pose(self, img, timestamp_ms=0, draw=True):
        """
        img: BGR image
        timestamp_ms: monotonically increasing timestamp for the frame sequence
        """
        # 1. Resize for performance (640px max dimension)
        h, w = img.shape[:2]
        max_dim = 640
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            proc_img = img.copy()

        # 2. Inference
        rgb_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        # In VIDEO mode, we must provide the timestamp
        self.results = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        
        # 3. Visualization
        if draw and self.results.pose_landmarks:
            for landmarks in self.results.pose_landmarks:
                for lm in landmarks:
                    # Map landmarks back to original image size
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        
        return img

    def get_landmarks(self, img):
        landmarks_list = []
        if self.results and self.results.pose_landmarks and len(self.results.pose_landmarks) > 0:
            landmarks = self.results.pose_landmarks[0]
            for id, lm in enumerate(landmarks):
                landmarks_list.append([id, lm.x, lm.y, lm.z, lm.presence])
        return landmarks_list

    def get_bounding_box(self, img, landmarks):
        if not landmarks:
            return None
            
        h, w, c = img.shape
        # landmarks are [id, x, y, z, presence] in normalized coordinates
        x_coords = [lm[1] * w for lm in landmarks]
        y_coords = [lm[2] * h for lm in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

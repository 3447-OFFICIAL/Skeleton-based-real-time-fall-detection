import pickle
import numpy as np
from collections import deque
import os

class FallDetector:
    def __init__(self, model_path='ml/model.pkl', window_size=30):
        self.model_path = model_path
        self.window_size = window_size
        self.feature_window = deque(maxlen=window_size)
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model file {self.model_path} not found.")

    def add_features(self, features):
        """
        features: list [aspect_ratio, hip_y, l_hip_angle, r_hip_angle, velocity, orientation]
        """
        if features is not None:
            self.feature_window.append(features)

    def predict(self):
        if self.model is None or len(self.feature_window) < 5: # Need at least some history
            return 0, 0.0
            
        # For a simple RF model, we can just predict on the latest frame
        # or average the last few frames.
        latest_features = np.array(self.feature_window[-1]).reshape(1, -1)
        
        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0][int(prediction)]
        
        # Heuristic: If we are in a sequence of frames, we want to avoid flickering.
        # Let's say if > 50% of the last 10 frames are falls, then it's a fall.
        if len(self.feature_window) >= 10:
            recent_features = np.array(list(self.feature_window)[-10:])
            recent_preds = self.model.predict(recent_features)
            if np.mean(recent_preds) > 0.5:
                return 1, np.mean(recent_preds)
            else:
                return 0, 1.0 - np.mean(recent_preds)

        return int(prediction), float(probability)

import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    b is the vertex.
    a = [x, y], b = [x, y], c = [x, y]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class FeatureExtractor:
    def __init__(self, fps=30):
        self.fps = fps
        self.prev_landmarks = None
        self.prev_centroid = None

    def extract_features(self, landmarks, img_shape):
        """
        landmarks: list of [id, x, y, z, cx, cy, visiblity]
        img_shape: (h, w, c)
        """
        if not landmarks or len(landmarks) < 33:
            return None
            
        h, w, c = img_shape
        
        # Landmarks for feature extraction
        # 11, 12: Shoulders; 23, 24: Hips; 25, 26: Knees; 27, 28: Ankles
        
        # 1. Aspect Ratio (Width / Height)
        x_coords = [lm[1] for lm in landmarks] # Normalized
        y_coords = [lm[2] for lm in landmarks]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        aspect_ratio = width / (height + 1e-6)
        
        # 2. Relative Height (Hips)
        hip_y = (landmarks[23][2] + landmarks[24][2]) / 2.0
        
        # 3. Angle (L-Knee-Hip, R-Knee-Hip, Hip-Shoulder-Elbow etc)
        # Just use some core ones for fall detection
        # Left Hip Angle (Shoulder, Hip, Knee)
        l_hip_angle = calculate_angle([landmarks[11][1], landmarks[11][2]], 
                                       [landmarks[23][1], landmarks[23][2]], 
                                       [landmarks[25][1], landmarks[25][2]])
        
        # Right Hip Angle
        r_hip_angle = calculate_angle([landmarks[12][1], landmarks[12][2]], 
                                       [landmarks[24][1], landmarks[24][2]], 
                                       [landmarks[26][1], landmarks[26][2]])
                                       
        # 4. Vertical Velocity
        centroid_y = (landmarks[11][2] + landmarks[12][2] + landmarks[23][2] + landmarks[24][2]) / 4.0
        velocity = 0
        if self.prev_centroid is not None:
            velocity = (centroid_y - self.prev_centroid) * self.fps
        
        self.prev_centroid = centroid_y
        
        # 5. Body Orientation (Vertical=0, Horizontal=1 approx)
        # Angle between spine (mid-shoulder to mid-hip) and vertical axis
        mid_shoulder = [(landmarks[11][1] + landmarks[12][1]) / 2.0, (landmarks[11][2] + landmarks[12][2]) / 2.0]
        mid_hip = [(landmarks[23][1] + landmarks[24][1]) / 2.0, (landmarks[23][2] + landmarks[24][2]) / 2.0]
        
        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]
        orientation = abs(dx) / (abs(dy) + 1e-6)
        
        features = [
            aspect_ratio,
            hip_y,
            l_hip_angle,
            r_hip_angle,
            velocity,
            orientation
        ]
        
        return features

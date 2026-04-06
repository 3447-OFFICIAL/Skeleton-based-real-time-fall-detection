import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_model():
    # 0 = No Fall, 1 = Fall
    # Features: [aspect_ratio, hip_y, l_hip_angle, r_hip_angle, velocity, orientation]
    
    np.random.seed(42)
    num_samples = 1000
    
    # 1. No Fall samples (Standing, Walking)
    # aspect_ratio: low (0.1 to 0.5)
    # orientation: low (0.0 to 0.3)
    # velocity: low (-0.5 to 0.5)
    # l_hip_angle, r_hip_angle: high (160 to 180)
    no_fall_data = np.random.uniform(low=[0.15, 0.4, 160, 160, -0.2, 0.1], 
                                     high=[0.3, 0.7, 180, 180, 0.2, 0.3], 
                                     size=(num_samples // 2, 6))
    
    # 2. Fall samples
    # aspect_ratio: high (0.8 to 2.0)
    # orientation: high (0.7 to 3.0)
    # velocity: extreme drop then low (let's say avg is 1.0 to 5.0)
    # l_hip_angle, r_hip_angle: low or vary (90 to 150)
    fall_data = np.random.uniform(low=[0.7, 0.7, 90, 90, 1.0, 0.7], 
                                  high=[1.5, 0.95, 150, 150, 5.0, 2.5], 
                                  size=(num_samples // 2, 6))
    
    X = np.concatenate([no_fall_data, fall_data], axis=0)
    y = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)], axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    print(f"Model Score: {model.score(X_test, y_test)}")
    
    os.makedirs('ml', exist_ok=True)
    with open('ml/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Synthetic model saved to ml/model.pkl")

if __name__ == '__main__':
    generate_synthetic_model()

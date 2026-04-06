import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import FallDetectionApp

def main():
    # Ensure necessary folders exist
    os.makedirs('ml', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check if model exists, if not, try to generate it
    if not os.path.exists('ml/model.pkl'):
        print("Model not found. Attempting to generate synthetic model...")
        try:
            from ml.generate_model import generate_synthetic_model
            generate_synthetic_model()
        except ImportError as e:
            print(f"Could not generate model automatically: {e}")
            print("Please ensure dependencies are installed correctly.")

    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

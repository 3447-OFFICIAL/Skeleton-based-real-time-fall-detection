import cv2

def check_cameras():
    print("Scanning for available cameras...")
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use DSHOW for Windows
        if cap.isOpened():
            print(f"[OK] Camera index {i} found.")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"[FAIL] Camera index {i} not available.")
    
    if not available_cameras:
        print("\nNo cameras detected by OpenCV.")
        print("Possible reasons:")
        print("1. Camera is disabled in Privacy Settings.")
        print("2. Another application is using the camera.")
        print("3. Missing drivers or hardware connection issue.")
    else:
        print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")

if __name__ == "__main__":
    check_cameras()

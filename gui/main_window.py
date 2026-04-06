import sys
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QTextEdit, QFileDialog, 
                               QFrame, QStatusBar, QApplication)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PySide6.QtCore import Qt, Slot, QTimer
import datetime

from gui.video_thread import VideoThread

class FallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skeleton-Based Fall Detection System")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        
        # UI Setup
        self.init_ui()
        self.setup_styling()
        
        # Thread
        self.thread = None

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar_layout = QVBoxLayout(sidebar)
        
        title_label = QLabel("CONTROL PANEL")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)
        
        sidebar_layout.addSpacing(20)
        
        self.btn_camera = QPushButton("🎥 START CAMERA")
        self.btn_camera.setFixedHeight(50)
        self.btn_camera.clicked.connect(self.start_camera)
        sidebar_layout.addWidget(self.btn_camera)
        
        self.btn_upload = QPushButton("📁 UPLOAD VIDEO")
        self.btn_upload.setFixedHeight(50)
        self.btn_upload.clicked.connect(self.upload_video)
        sidebar_layout.addWidget(self.btn_upload)
        
        self.btn_stop = QPushButton("🛑 STOP")
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_process)
        sidebar_layout.addWidget(self.btn_stop)
        
        sidebar_layout.addStretch()
        
        stats_label = QLabel("SYSTEM STATUS")
        stats_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(stats_label)
        
        self.status_display = QLabel("IDLE")
        self.status_display.setAlignment(Qt.AlignCenter)
        self.status_display.setFixedHeight(40)
        self.status_display.setStyleSheet("background-color: #333; border-radius: 5px; color: #fff;")
        sidebar_layout.addWidget(self.status_display)
        
        self.indicator_light = QLabel()
        self.indicator_light.setFixedSize(50, 50)
        self.indicator_light.setStyleSheet("background-color: #555; border-radius: 25px;")
        sidebar_layout.addWidget(self.indicator_light, alignment=Qt.AlignCenter)
        
        sidebar_layout.addSpacing(20)
        
        main_layout.addWidget(sidebar)
        
        # --- Video Panel ---
        video_container = QFrame()
        video_layout = QVBoxLayout(video_container)
        
        self.video_label = QLabel("Camera Feed / Video Analysis")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px; border: 2px solid #444;")
        self.video_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_container, 3)
        
        # --- Log Panel ---
        log_container = QFrame()
        log_container.setFixedWidth(300)
        log_layout = QVBoxLayout(log_container)
        
        log_title = QLabel("EVENT LOG")
        log_title.setFont(QFont("Arial", 10, QFont.Bold))
        log_layout.addWidget(log_title)
        
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setStyleSheet("background-color: #1a1a1a; color: #00ff00; font-family: 'Consolas';")
        log_layout.addWidget(self.log_panel)
        
        main_layout.addWidget(log_container)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def setup_styling(self):
        # Dark Theme Palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(dark_palette)
        
        # Global Style
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QPushButton { 
                background-color: #3d3d3d; 
                border: 1px solid #555; 
                border-radius: 5px; 
                padding: 10px;
                color: #e0e0e0;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:disabled { background-color: #222; color: #555; }
            QLabel { color: #e0e0e0; }
        """)

    @Slot(QImage)
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    @Slot(str, float, bool)
    def update_status(self, status, confidence, is_fall):
        self.status_display.setText(f"{status} ({confidence:.1%})")
        
        if is_fall:
            self.indicator_light.setStyleSheet("background-color: #ff0000; border: 2px solid #fff; border-radius: 25px;")
            self.statusBar().showMessage(f"WARNING: FALL DETECTED! (Conf: {confidence:.2f})")
            
            # Sound alert (Windows ONLY)
            import winsound
            winsound.Beep(1000, 500) # 1000Hz for 500ms
            
            # Log event if it's a fall
            self.log_event(f"FALL ALERT: Confidence {confidence:.2f}")
        else:
            self.indicator_light.setStyleSheet("background-color: #00ff00; border: 2px solid #fff; border-radius: 25px;")
            self.statusBar().showMessage(f"Status: Safe (Conf: {confidence:.2f})")

    def log_event(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_panel.append(f"[{timestamp}] {message}")

    def start_camera(self):
        # Tentatively check if camera exists before starting thread
        import cv2
        # Use CAP_DSHOW for Windows compatibility
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Camera Error", "Could not access the webcam. Please ensure it is connected, not in use by another app, and permissions are granted.")
            self.log_event("Error: Webcam not found or access denied.")
            return
        cap.release()

        self.btn_camera.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.thread = VideoThread(source=0)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status)
        self.thread.log_signal.connect(self.log_event)
        self.thread.start()
        
        self.log_event("Accessing camera...")

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.btn_camera.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.btn_stop.setEnabled(True)
            
            self.thread = VideoThread(source=file_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_signal.connect(self.update_status)
            self.thread.log_signal.connect(self.log_event)
            self.thread.start()
            
            self.log_event(f"Processing video: {os.path.basename(file_path)}")

    def stop_process(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        
        self.btn_camera.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("Camera Feed / Video Analysis")
        self.status_display.setText("IDLE")
        self.indicator_light.setStyleSheet("background-color: #555; border-radius: 25px;")
        self.log_event("Process stopped by user.")
        self.statusBar().showMessage("Ready")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec())

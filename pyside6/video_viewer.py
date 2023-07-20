import sys
import cv2
import os
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QSizePolicy,
    QApplication,
    QWidget,
    QStatusBar,
)
from PySide6 import QtGui


class VideoViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")
        self.resize(800, 600)

        self.video_file_btn = QPushButton("Open Video")
        self.video_file_btn.clicked.connect(self.open_video_file_dialog)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause)

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture)

        self.video_view_label = QLabel()
        self.video_view_label.setAlignment(Qt.AlignCenter)
        self.video_view_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_file_btn)
        main_layout.addWidget(self.play_btn)
        main_layout.addWidget(self.pause_btn)
        main_layout.addWidget(self.capture_btn)
        main_layout.addWidget(self.video_view_label)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.video_path = ""
        self.video_w = 720
        self.video_h = 640

        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.display_next_frame)

        self.paused = False
        self.current_frame = 0
        self.capture_cnt = 0
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        os.makedirs("./image/", exist_ok=True)

    def open_video_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.video_path = selected_files[0]
                self.status_bar.showMessage(f"Selected video: {self.video_path}")

    def display_next_frame(self):
        if self.video_path:
            ret, frame = self.video_capture.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = self.resize_frame(frame_rgb)
                h, w, _ = frame_resized.shape
                if w > 0 and h > 0:
                    frame_img = QtGui.QImage(
                        frame_resized, w, h, QtGui.QImage.Format_RGB888
                    )
                    pixmap = QtGui.QPixmap.fromImage(frame_img)
                    self.video_view_label.setPixmap(pixmap)
                    self.video_view_label.setScaledContents(True)

                self.current_frame += 1
            else:
                self.video_timer.stop()

    def play(self):
        if self.video_path:
            if self.paused:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = False

            else:
                self.video_capture = cv2.VideoCapture(self.video_path)
                self.current_frame = 0

            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.video_timer.start(30)

    def pause(self):
        self.video_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.capture_btn.setEnabled(not self.paused)
        self.paused = True

    def capture(self):
        if not self.paused:
            return
        ret, frame = self.video_capture.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = self.resize_frame(frame_rgb)
            h, w, _ = frame_resized.shape
            if w > 0 and h > 0:
                file_name = os.path.splitext(os.path.basename(self.video_path))[0]
                img_name = f"{file_name}_{self.capture_cnt:04d}_img.png"
                img_save_path = os.path.join("./image/", img_name)
                cv2.imwrite(img_save_path, frame_rgb)
                self.capture_cnt += 1
                self.status_bar.showMessage(f"capture ok : {img_save_path}")

    def resize_frame(self, frame):
        h, w, _ = frame.shape
        if w > self.video_w:
            ratio = self.video_w / w
            frame = cv2.resize(frame, (self.video_w, int(h * ratio)))

        if h > self.video_h:
            ratio = self.video_h / h
            frame = cv2.resize(frame, (int(w * ratio), self.video_h))

        return frame

    def close_event(self, event):
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoViewer()
    win.show()
    sys.exit(app.exec())

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QGridLayout, QSpacerItem, QSizePolicy, QMenu, QAction, QMenuBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from camera_thread import CameraThread  # Ensure this is the correct import path

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Stream")
        self.setGeometry(100, 100, 1600, 1700)

        self.main_layout = QVBoxLayout()
        self.main_frame = QHBoxLayout()
        self.sidebar = QVBoxLayout()
        self.sidebar.setSpacing(0)
        self.sidebar.setContentsMargins(0, 0, 0, 0)
        self.grid_layout = QGridLayout()

        self.sidebar_widget = QWidget()
        self.sidebar_widget.setLayout(self.sidebar)
        self.sidebar_widget.setFixedWidth(200)

        self.combobox = QComboBox()
        self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
        self.combobox.currentIndexChanged.connect(self.change_grid_size)
        self.combobox.setFixedWidth(200)
        self.main_layout.addWidget(self.combobox)

        self.button1 = self.create_button('Test1', 0)
        self.button2 = self.create_button('Test2', 1)
        self.button3 = self.create_button('Test3', 2)
        self.button4 = self.create_button('Test4', 3)
        self.sidebar.addWidget(self.button1)
        self.sidebar.addWidget(self.button2)
        self.sidebar.addWidget(self.button3)
        self.sidebar.addWidget(self.button4)

        spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sidebar.addItem(spacer)

        self.main_frame.addWidget(self.sidebar_widget)
        self.set_grid_layout(2, 2)
        self.main_frame.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.main_frame)
        self.setLayout(self.main_layout)

        self.rtsp_urls = [
            "test2.mp4",
            "test2.mp4",
            "test2.mp4",
            "test2.mp4"
        ]

        self.camera_threads = []
        for i, rtsp_url in enumerate(self.rtsp_urls):
            thread = CameraThread(rtsp_url)
            thread.change_pixmap_signal.connect(lambda frame, i=i: self.update_image(frame, i))
            self.camera_threads.append(thread)
            thread.start()

        self.create_menu_bar()

        self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

    def create_button(self, text, camera_index):
        button = QPushButton(text)
        button.setContextMenuPolicy(Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
        return button

    def show_context_menu(self, pos, button, camera_index):
        context_menu = QMenu(self)

        toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
        if self.camera_threads[camera_index].simple_mode:
            toggle_simple_mode_action.setText("Disable Simple Mode")
        else:
            toggle_simple_mode_action.setText("Enable Simple Mode")
        toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

        close_channel_action = QAction("Start Channel", self)
        if self.camera_threads[camera_index].is_running:
            close_channel_action.setText("Stop Channel")
        else:
            close_channel_action.setText("Start Channel")
        close_channel_action.triggered.connect(lambda: self.close_channel(camera_index))

        toggle_recording_action = QAction("Start Recording", self)
        if self.camera_threads[camera_index].recording:
            toggle_recording_action.setText("Stop Recording")
        else:
            toggle_recording_action.setText("Start Recording")
        toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

        context_menu.addAction(toggle_simple_mode_action)
        context_menu.addAction(close_channel_action)
        context_menu.addAction(toggle_recording_action)

        context_menu.exec_(button.mapToGlobal(pos))

    def toggle_simple_mode(self, camera_index):
        self.camera_threads[camera_index].toggle_simple_mode()

    def close_channel(self, camera_index):
        if 0 <= camera_index < len(self.camera_threads):
            self.camera_threads[camera_index].close_channel()

            # Set the label corresponding to the closed channel to a black image if stopped
            if not self.camera_threads[camera_index].is_running:
                if 0 <= camera_index < len(self.labels):
                    black_image_qimage = self.convert_np_to_qimage(self.black_image)
                    self.labels[camera_index].setPixmap(QPixmap.fromImage(black_image_qimage))

    def toggle_recording(self, camera_index):
        if 0 <= camera_index < len(self.camera_threads):
            self.camera_threads[camera_index].toggle_recording()
            self.update_recording_button_text(camera_index)

    def update_recording_button_text(self, camera_index):
        button = self.sidebar.itemAt(camera_index).widget()
        if self.camera_threads[camera_index].recording:
            button.setText("Stop Recording")
        else:
            button.setText("Start Recording")

    def set_grid_layout(self, rows, cols):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.labels = []
        for row in range(rows):
            for col in range(cols):
                label = QLabel()
                label.setFixedSize(640, 640)
                self.grid_layout.addWidget(label, row, col)
                self.labels.append(label)

    def change_grid_size(self):
        selected_size = self.combobox.currentText()
        if selected_size == "1x1":
            self.set_grid_layout(1, 1)
        elif selected_size == "2x2":
            self.set_grid_layout(2, 2)
        elif selected_size == "3x2":
            self.set_grid_layout(3, 2)
        elif selected_size == "4x1":
            self.set_grid_layout(4, 1)
        self.update_images()

    def update_images(self):
        for i, thread in enumerate(self.camera_threads):
            if i < len(self.labels):
                black_image_qimage = self.convert_np_to_qimage(self.black_image)
                self.labels[i].setPixmap(QPixmap.fromImage(black_image_qimage))

    def update_image(self, frame, index):
        if index < len(self.labels):
            qimage = self.convert_np_to_qimage(frame)
            self.labels[index].setPixmap(QPixmap.fromImage(qimage))

    def convert_np_to_qimage(self, np_image):
        height, width, channels = np_image.shape
        bytes_per_line = channels * width
        qimage = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.rgbSwapped()

    def create_menu_bar(self):
        menu_bar = QMenuBar(self)

        settings_menu = QMenu("Settings", self)
        toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
        toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(0))  # Example for camera 0
        settings_menu.addAction(toggle_simple_mode_action)

        start_stop_channel_action = QAction("Start/Stop Channel", self)
        start_stop_channel_action.triggered.connect(lambda: self.close_channel(0))  # Example for camera 0
        settings_menu.addAction(start_stop_channel_action)

        start_stop_recording_action = QAction("Start/Stop Recording", self)
        start_stop_recording_action.triggered.connect(lambda: self.toggle_recording(0))  # Example for camera 0
        settings_menu.addAction(start_stop_recording_action)

        menu_bar.addMenu(settings_menu)

        self.main_layout.setMenuBar(menu_bar)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

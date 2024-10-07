import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
                             QMenu, QPushButton, QSpacerItem, QSizePolicy, QInputDialog, QDialog, QFormLayout)

from openvino.runtime import Core, Model
from PIL import Image
from Preprocessing import image_to_tensor, preprocess_image
from Postprocessing import postprocess
from draw_result import draw_results
from ultralytics import YOLO

frameWidth = 400

def detect(image: np.ndarray, model: Model):
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
    return detections

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url, detect_model=None):
        super().__init__()
        self._run_flag = True
        self.rtsp_url = rtsp_url
        self.detect_model = detect_model
        self.simple_mode = False
        self.recording = False
        self.video_writer = None
        self.output_file = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = 30
        self.frame_size = (640, 640)

    def run(self):
        global frameWidth
        det_model_path = "yolov8n.pt"
        det_model = YOLO(det_model_path)
        label_map = det_model.model.names

        core = Core()
        det_model_path = "yolov8n.xml"
        det_ov_model = core.read_model(det_model_path)
        device = "CPU"
        if device != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        det_compiled_model = core.compile_model(det_ov_model, device)

        cam = cv2.VideoCapture(self.rtsp_url)
        self.fps = cam.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        currentframe = 0
        while self._run_flag:
            ret, frame = cam.read()
            if not ret:
                break

            image = cv2.resize(frame, (640, 640))
            if self.simple_mode:
                result_image = cv2.resize(image, (frameWidth, frameWidth))
                self.change_pixmap_signal.emit(result_image)
            else:
                detections = detect(image, det_compiled_model)[0]
                image_with_boxes = draw_results(detections, image, label_map)
                result_image = cv2.resize(image_with_boxes, (frameWidth, frameWidth))
                self.change_pixmap_signal.emit(result_image)

            if self.recording:
                self.record_frame(frame)

            cv2.waitKey(1)
            currentframe += 1

        cam.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._run_flag = False
        self.wait()

    def toggle_simple_mode(self):
        self.simple_mode = not self.simple_mode

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.output_file = None
        else:
            self.recording = True
            self.output_file = f"recording_{int(cv2.getTickCount())}.mp4"
            self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)

    def record_frame(self, frame):
        if self.video_writer:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_writer.write(frame_rgb)

    def close_channel(self):
        self.stop()
        self.wait()
        self._run_flag = False
        self.video_writer = None
        self.output_file = None

class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Camera")
        self.layout = QFormLayout()
        
        self.url_input = QInputDialog()
        self.url_input.setLabelText("Enter RTSP URL:")
        self.layout.addWidget(self.url_input)
        
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.accept)
        self.layout.addWidget(self.add_button)

        self.setLayout(self.layout)
    
    def get_url(self):
        return self.url_input.textValue()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Stream")
        self.setGeometry(100, 100, 1200, 800)

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
        self.combobox.addItems(["2x2", "3x3", "4x4"])
        self.combobox.currentTextChanged.connect(self.change_grid_size)
        self.combobox.setFixedWidth(200)
        self.main_layout.addWidget(self.combobox)

        self.buttons = []
        self.rtsp_urls = []
        self.camera_threads = []

        self.create_buttons(["Camera1", "Camera2", "Camera3", "Camera4"])

        spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sidebar.addItem(spacer)

        self.main_frame.addWidget(self.sidebar_widget)
        self.set_grid_layout(2, 2)
        self.main_frame.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.main_frame)
        self.setLayout(self.main_layout)

        self.create_menu_bar()

        self.black_image = np.zeros((400, 400, 3), dtype=np.uint8)

    def create_buttons(self, labels):
        for i, label in enumerate(labels):
            button = self.create_button(label, i)
            self.sidebar.addWidget(button)
            self.buttons.append(button)
            self.rtsp_urls.append("")
            self.camera_threads.append(None)

    def create_button(self, text, camera_index):
        button = QPushButton(text)
        button.setContextMenuPolicy(Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
        return button
    def change_grid_size(self):
        layout_map = {"2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
        selected_layout = self.combobox.currentText()
        rows, cols = layout_map.get(selected_layout, (2, 2))  # Default to 2x2 if no selection found
        self.set_grid_layout(rows, cols)


    def show_context_menu(self, pos, button, camera_index):
        context_menu = QMenu(self)

        toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
        if self.camera_threads[camera_index] and self.camera_threads[camera_index].simple_mode:
            toggle_simple_mode_action.setText("Enable Yolo Mode")
        else:
            toggle_simple_mode_action.setText("Disable Yolo Mode")
        toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

        close_channel_action = QAction("Close Channel", self)
        if self.camera_threads[camera_index] is None:
            close_channel_action.setText("Start Channel")
        else:
            close_channel_action.setText("Close Channel")
        close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

        toggle_recording_action = QAction("Start Recording", self)
        if self.camera_threads[camera_index] and self.camera_threads[camera_index].recording:
            toggle_recording_action.setText("Stop Recording")
        toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

        context_menu.addAction(toggle_simple_mode_action)
        context_menu.addAction(close_channel_action)
        context_menu.addAction(toggle_recording_action)

        context_menu.exec_(button.mapToGlobal(pos))

    def toggle_simple_mode(self, camera_index):
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].toggle_simple_mode()

    def toggle_channel(self, camera_index):
        if self.camera_threads[camera_index] is None:
            self.add_camera(camera_index)
        else:
            self.close_channel(camera_index)

    def toggle_recording(self, camera_index):
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].toggle_recording()

    def close_channel(self, camera_index):
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].close_channel()
            self.camera_threads[camera_index] = None
            self.buttons[camera_index].setText(f"Camera{camera_index + 1}")

    def add_camera(self, camera_index):
        dialog = AddCameraDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            rtsp_url = dialog.get_url()
            self.rtsp_urls[camera_index] = rtsp_url
            self.start_thread(camera_index, rtsp_url)
            self.buttons[camera_index].setText(f"Camera{camera_index + 1} (Active)")

    def start_thread(self, camera_index, rtsp_url):
        if self.camera_threads[camera_index] is not None:
            self.close_channel(camera_index)
        thread = CameraThread(rtsp_url)
        thread.change_pixmap_signal.connect(lambda image, idx=camera_index: self.update_image(image, idx))
        self.camera_threads[camera_index] = thread
        thread.start()

    def set_grid_layout(self, rows, cols):
        # Remove all widgets from the current layout
        global frameWidth
        if rows == 2:
            frameWidth = 400
        elif rows == 3:
            frameWidth = 300
        elif  rows ==4: 
            frameWidth = 230
        for i in reversed(range(self.grid_layout.count())):
            widget_to_remove = self.grid_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()

        # Add new widgets to the grid layout
        # for row in range(rows):
        #     for col in range(cols):
        #         label = QLabel(f"Row {row+1}, Col {col+1}")
        #         # label.setPixmap(self.convert_frame_to_pixmap(self.black_image))
        #         self.grid_layout.addWidget(label, row, col)

        black_image = QImage(frameWidth, frameWidth, QImage.Format_RGB32)
        black_image.fill(Qt.black)
        
        # Convert black image to QPixmap
        pixmap = QPixmap.fromImage(black_image)

        # Add new widgets to the grid layout
        for row in range(rows):
            for col in range(cols):
                label = QLabel()
                label.setPixmap(pixmap)
                label.setFixedSize(frameWidth, frameWidth)  # Set the label size
                label.setScaledContents(True)  # Ensure the image fits the label size
                self.grid_layout.addWidget(label, row, col)


    def update_image(self, image, camera_index):
        if 0 <= camera_index < len(self.buttons):
            label = self.grid_layout.itemAt(camera_index).widget()
            if label:
                label.setPixmap(self.convert_frame_to_pixmap(image))

    def convert_frame_to_pixmap(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image.rgbSwapped())

    def create_menu_bar(self):
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        option_menu = QMenu("Option", self)
        edit_menu = QMenu("Edit", self)
        settings_menu = QMenu("Settings", self)
        help_menu = QMenu("Help", self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        add_camera_action = QAction("Add Camera", self)
        add_camera_action.triggered.connect(self.add_camera)
        settings_menu.addAction(add_camera_action)

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(option_menu)
        menu_bar.addMenu(edit_menu)
        menu_bar.addMenu(settings_menu)
        menu_bar.addMenu(help_menu)
        self.main_layout.setMenuBar(menu_bar)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, QMenu, QPushButton, QSpacerItem, QSizePolicy

from openvino.runtime import Core, Model
from PIL import Image
from Preprocessing import image_to_tensor, preprocess_image
from Postprocessing import postprocess
from draw_result import draw_results
from ultralytics import YOLO

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

# Thread class for capturing video from RTSP stream
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url):
        super().__init__()
        self._run_flag = True
        self.rtsp_url = rtsp_url

    def run(self):
        det_model_path = "yolov8n.pt"
        det_model = YOLO(det_model_path)
        label_map = det_model.model.names
        source_path = "test2.mp4"

        core = Core()
        det_model_path = "yolov8n.xml"
        det_ov_model = core.read_model(det_model_path)
        device = "CPU"  # "GPU"
        if device != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        det_compiled_model = core.compile_model(det_ov_model, device)
        if "jpg" in source_path:
            input_image = np.array(Image.open(source_path))
            detections = detect(input_image, det_compiled_model)[0]
            image_with_boxes = draw_results(detections, input_image, label_map)
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            cv2.imshow(source_path, image_with_boxes)
            cv2.waitKey(0)
        else:
            # cam = cv2.VideoCapture(source_path)
            cam = cv2.VideoCapture(self.rtsp_url)
            currentframe = 0
            while True:
                ret, frame = cam.read()
                image = cv2.resize(frame, (640, 640))
                if ret:
                    detections = detect(image, det_compiled_model)[0]
                    image_with_boxes = draw_results(detections, image, label_map)
                    self.change_pixmap_signal.emit(image_with_boxes)
                    cv2.waitKey(1)
                    currentframe += 1
                else:
                    break
            cam.release()
            cv2.destroyAllWindows()

    def stop(self):
        self._run_flag = False
        self.wait()

# Main application window
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Stream")
        self.setGeometry(100, 100, 1900, 1700)

        # Create layout and add combobox for selecting grid size
        self.main_layout = QVBoxLayout()

        self.main_frame = QHBoxLayout()
        self.sidebar = QVBoxLayout()
        self.sidebar.setSpacing(0)  # Remove the spaces between the buttons
        self.sidebar.setContentsMargins(0, 0, 0, 0)  # Remove margins around the buttons
        self.grid_layout = QGridLayout()

        # Add combobox
        self.combobox = QComboBox()
        self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
        self.combobox.currentIndexChanged.connect(self.change_grid_size)
        self.main_layout.addWidget(self.combobox)

        # Sidebar buttons
        self.button1 = QPushButton('Test1')
        self.button2 = QPushButton('Test2')
        self.button3 = QPushButton('Test3')
        self.button4 = QPushButton('Test4')
        self.sidebar.addWidget(self.button1)
        self.sidebar.addWidget(self.button2)
        self.sidebar.addWidget(self.button3)
        self.sidebar.addWidget(self.button4)

        # Add vertical spacer to sidebar
        spacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sidebar.addItem(spacer)

        self.main_frame.addLayout(self.sidebar)
        # Set initial grid layout for 2x2
        self.set_grid_layout(2, 2)
        self.main_frame.addLayout(self.grid_layout)

        self.main_layout.addLayout(self.main_frame)
        self.setLayout(self.main_layout)

        # RTSP stream URLs
        self.rtsp_urls = [
            # "http://88.53.197.250/axis-cgi/mjpg/video.cgi?resolution=320x240",
            # "http://158.58.130.148:80/mjpg/video.mjpg",
            "test2.mp4",
            "test2.mp4",
            "test2.mp4",
            "test2.mp4"
        ]

        # Create threads for camera streams
        self.camera_threads = []
        for i, rtsp_url in enumerate(self.rtsp_urls):
            thread = CameraThread(rtsp_url)
            thread.change_pixmap_signal.connect(lambda frame, i=i: self.update_image(frame, i))
            self.camera_threads.append(thread)
            thread.start()

        # Create menu bar
        self.create_menu_bar()

    def set_grid_layout(self, rows, cols):
        """Setup the grid layout dynamically based on rows and columns."""
        # Clear any existing widgets in the grid layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.labels = []
        for row in range(rows):
            for col in range(cols):
                label = QLabel(self)
                label.setFixedSize(640, 640)
                self.grid_layout.addWidget(label, row, col)
                self.labels.append(label)

    def change_grid_size(self):
        """Change the grid size based on the combobox selection."""
        selected_size = self.combobox.currentText()
        if selected_size == "1x1":
            self.set_grid_layout(1, 1)
        elif selected_size == "2x2":
            self.set_grid_layout(2, 2)
        elif selected_size == "3x2":
            self.set_grid_layout(3, 2)
        elif selected_size == "4x1":
            self.set_grid_layout(4, 1)

    def update_image(self, frame, index):
        """Updates the image_label with a new frame."""
        qt_image = self.convert_cv_qt(frame)
        if index < len(self.labels):
            self.labels[index].setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def create_menu_bar(self):
        """Create a menu bar with File, Add, and Settings menus."""
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = QMenu("File", self)
        file_option1 = QAction("Option 1", self)
        file_option2 = QAction("Option 2", self)
        file_menu.addAction(file_option1)
        file_menu.addAction(file_option2)

        # Add Menu
        edit_menu = QMenu("Edit", self)
        edit_option1 = QAction("Option 1", self)
        edit_option2 = QAction("Option 2", self)
        edit_menu.addAction(edit_option1)
        edit_menu.addAction(edit_option2)

        # Settings Menu
        option_menu = QMenu("Options", self)
        option_menu_1 = QAction("Option 1", self)
        option_menu_2 = QAction("Option 2", self)
        option_menu_3 = QAction("Option 3", self)
        option_menu.addAction(option_menu_1)
        option_menu.addAction(option_menu_2)
        option_menu.addAction(option_menu_3)

        settings_menu = QMenu("Settings", self)
        help_menu = QMenu("Help", self)

        # Add the menus to the menu bar
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(edit_menu)
        menu_bar.addMenu(option_menu)
        menu_bar.addMenu(settings_menu)
        menu_bar.addMenu(help_menu)

        # Set the menu bar for the main window
        self.main_layout.setMenuBar(menu_bar)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

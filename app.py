import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
                             QMenu, QPushButton, QSpacerItem, QSizePolicy, QListWidget, QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QTableWidget, QTableWidgetItem)

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
        device = "CPU"  # "GPU"
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
                # Simple mode without detection
                result_image = cv2.resize(image, (frameWidth, frameWidth))
                self.change_pixmap_signal.emit(result_image)
            else:
                # Detection mode
                detections = detect(image, det_compiled_model)[0]
                image_with_boxes = draw_results(detections, image, label_map)

                result_image = cv2.resize(image_with_boxes, (frameWidth, frameWidth))

                self.change_pixmap_signal.emit(result_image)

            if self.recording:
                self.record_frame(frame)  # Recording logic

            cv2.waitKey(1)
            currentframe += 1

        cam.release()
        if self.video_writer:
            self.video_writer.release()  # Release the video writer when done
        cv2.destroyAllWindows()

    def stop(self):
        self._run_flag = False
        self.wait()

    def toggle_simple_mode(self):
        # Toggle between simple mode and detection mode
        self.simple_mode = not self.simple_mode

    def toggle_recording(self):
        # Toggle recording on and off
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
        # Convert the frame to RGB format before saving
        if self.video_writer:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_writer.write(frame_rgb)

    def close_channel(self):
        # Stop the thread and release resources
        self.stop()
        self.wait()  # Ensure the thread has fully stopped
        self._run_flag = False  # Ensure the run loop is stopped
        self.video_writer = None
        self.output_file = None

# class AddCameraDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Add New Camera")
#         self.layout = QFormLayout()
        
#         self.url_input = QInputDialog()
#         self.url_input.setLabelText("Enter RTSP URL:")
#         self.layout.addWidget(self.url_input)
        
#         self.add_button = QPushButton("Add")
#         self.add_button.clicked.connect(self.accept)
#         self.layout.addWidget(self.add_button)

#         self.setLayout(self.layout)
    
#     def get_url(self):
#         return self.url_input.textValue()

class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Camera")
        self.layout = QFormLayout()
        
        # Create an input dialog widget for URL input
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter RTSP URL:")
        self.layout.addRow("RTSP URL:", self.url_input)
        
        # Create a dialog button box with Ok and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("Add Camera")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
        self.setLayout(self.layout)
    
    def get_url(self):
        return self.url_input.text()
    
class EditCamerasDialog(QDialog):
    def __init__(self, rtsp_urls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Cameras")
        self.layout = QVBoxLayout()
        
        # Initialize the table
        self.table = QTableWidget(len(rtsp_urls), 2)
        self.table.setHorizontalHeaderLabels(["RTSP URL", "Actions"])
        self.table.setVerticalHeaderLabels([f"Camera {i+1}" for i in range(len(rtsp_urls))])
        
        # Access header view to set stretch properties
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        
        # Populate table with current URLs
        for row, url in enumerate(rtsp_urls):
            self.table.setItem(row, 0, QTableWidgetItem(url))
            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(lambda checked, row=row: self.remove_row(row))
            self.table.setCellWidget(row, 1, remove_button)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.accept)
        
        self.layout.addWidget(self.table)
        self.layout.addWidget(self.update_button)
        self.setLayout(self.layout)

    def get_updated_urls(self):
        urls = [self.table.item(row, 0).text() for row in range(self.table.rowCount())]
        return urls

    def remove_row(self, row):
        self.table.removeRow(row)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Stream")
        self.setGeometry(100, 100, 1000, 800)

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

        # self.button1 = self.create_button('Camera1', 0)
        # self.sidebar.addWidget(self.button1)

        spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sidebar.addItem(spacer)

        self.main_frame.addWidget(self.sidebar_widget)
        self.set_grid_layout(2, 2)
        self.main_frame.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.main_frame)
        self.setLayout(self.main_layout)

        self.rtsp_urls = [
            "http://158.58.130.148/mjpg/video.mjpg",
            "rtsp://admin:Waq112299-@192.168.10.137:554/Streaming/Channels/101",
            "rtsp://admin:Waq112299-@192.168.10.138:554/Streaming/Channels/101",
            "http://88.53.197.250/axis-cgi/mjpg/video.cgi?resolution=320x240"
        ]
        self.buttons = []

        # self.rtsp_urls = [
        #     "test2.mp4",
        #     "test2.mp4",
        #     "test2.mp4",
        #     "test2.mp4",
        # ]

        self.create_camera_buttons()
        self.camera_threads = [None] * 10  # Initialize with None to indicate channels are closed

        for i, rtsp_url in enumerate(self.rtsp_urls):
            self.start_thread(i, rtsp_url)

        self.create_menu_bar()

        # Create a black image for clearing the frame
        self.black_image = np.zeros((400, 400, 3), dtype=np.uint8)

    # def create_camera_buttons(self):
    #     # Remove existing buttons if any
    #     for button in self.buttons:
    #         button.deleteLater()
    #     self.buttons.clear()

    #     # Create new buttons based on the number of RTSP URLs
    #     for i, url in enumerate(self.rtsp_urls):
    #         button_text = f"Camera{i + 1}"
    #         button = self.create_button(button_text, i)
    #         self.sidebar.addWidget(button)
    #         self.buttons.append(button)
    def create_camera_buttons(self):
    # Remove existing buttons if any
        for button in reversed(self.buttons):
            button.deleteLater()
        self.buttons.clear()

        # Create new buttons based on the number of RTSP URLs
        for i, url in enumerate(self.rtsp_urls):
            button_text = f"Camera{i + 1}"
            button = self.create_button(button_text, i)
            self.sidebar.insertWidget(i, button)  # Insert at the specific position
            self.buttons.append(button)

    def create_button(self, text, camera_index):
        button = QPushButton(text)
        button.setContextMenuPolicy(Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
        return button

    def show_context_menu(self, pos, button, camera_index):
        context_menu = QMenu(self)

        # Toggle Simple Mode action
        toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
        if self.camera_threads[camera_index] and self.camera_threads[camera_index].simple_mode:
            toggle_simple_mode_action.setText("Enable Yolo Mode")
        else:
            toggle_simple_mode_action.setText("Disable Yolo Mode")
        toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

        # Close Channel action
        close_channel_action = QAction("Close Channel", self)
        if self.camera_threads[camera_index] is None:
            close_channel_action.setText("Start Channel")
        else:
            close_channel_action.setText("Close Channel")
        close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

        # Start/Stop Recording action
        toggle_recording_action = QAction("Start Recording", self)
        if self.camera_threads[camera_index] and self.camera_threads[camera_index].recording:
            toggle_recording_action.setText("Stop Recording")
        toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

        # Add actions to context menu
        context_menu.addAction(toggle_simple_mode_action)
        context_menu.addAction(close_channel_action)
        context_menu.addAction(toggle_recording_action)

        context_menu.exec_(button.mapToGlobal(pos))

    def toggle_simple_mode(self, camera_index):
        # Toggle simple mode for the selected camera
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].toggle_simple_mode()

    def toggle_channel(self, camera_index):
        # Start or close the camera channel based on its current state
        if self.camera_threads[camera_index] is None:
            self.start_thread(camera_index, self.rtsp_urls[camera_index])
        else:
            self.close_channel(camera_index)

    def toggle_recording(self, camera_index):
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].toggle_recording()

    def close_channel(self, camera_index):
        if self.camera_threads[camera_index]:
            self.camera_threads[camera_index].close_channel()
            self.camera_threads[camera_index] = None
            label = self.grid_layout.itemAt(camera_index).widget()
            label.setPixmap(self.convert_frame_to_pixmap(self.black_image))

    def change_grid_size(self):
        layout_map = {"2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
        selected_layout = self.combobox.currentText()
        rows, cols = layout_map.get(selected_layout, (2, 2))  # Default to 2x2 if no selection found
        self.set_grid_layout(rows, cols)

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


    def start_thread(self, camera_index, rtsp_url):
        thread = CameraThread(rtsp_url)
        thread.change_pixmap_signal.connect(lambda image, idx=camera_index: self.update_image(image, idx))
        self.camera_threads[camera_index] = thread
        thread.start()

    def update_image(self, image, camera_index):
        label = self.grid_layout.itemAt(camera_index).widget()
        if label:
            label.setPixmap(self.convert_frame_to_pixmap(image))

    def convert_frame_to_pixmap(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    def refresh_sidebar_buttons(self):
        # Remove all existing camera buttons
        for i in reversed(range(self.sidebar.count())):
            widget_to_remove = self.sidebar.itemAt(i).widget()
            if widget_to_remove and isinstance(widget_to_remove, QPushButton):
                widget_to_remove.deleteLater()
        
        # Create and add new camera buttons
        for i, url in enumerate(self.rtsp_urls):
            button = self.create_button(f'Camera{i+1}', i)
            self.sidebar.insertWidget(i, button) 
    
    def add_camera(self, camera_index):
        new_camera_index = len(self.rtsp_urls)
        dialog = AddCameraDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            rtsp_url = dialog.get_url()
            self.rtsp_urls.append(rtsp_url)
            print('333333333', self.rtsp_urls)
            self.start_thread(new_camera_index, rtsp_url)
            self.camera_threads.append(None)  # Placeholder for the new camera thread

            button_text = f"Camera{new_camera_index + 1}"
            new_button = self.create_button(button_text, new_camera_index)
            self.sidebar.insertWidget(0, new_button)  # Insert at the top
            self.buttons.insert(0, new_button)  # Keep track of the new button

    def edit_cameras(self):
        dialog = EditCamerasDialog(self.rtsp_urls, self)
        if dialog.exec_() == QDialog.Accepted:
            new_rtsp_urls = dialog.get_updated_urls()
            for i in range(len(self.rtsp_urls)):
                self.close_channel(i)
            self.rtsp_urls = new_rtsp_urls 

            self.refresh_sidebar_buttons()


            for index in range(len(self.rtsp_urls)):

                if self.camera_threads[index] is None:
                    self.start_thread(index, self.rtsp_urls[index])
                else:
                    self.close_channel(index)


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
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(option_menu)
        menu_bar.addMenu(edit_menu)
        addCamera_action = QAction("Add Camera", self)
        addCamera_action.triggered.connect(self.add_camera)
        settings_menu.addAction(addCamera_action)
        editCamera_action = QAction("Edit Camera", self)
        editCamera_action.triggered.connect(self.edit_cameras)
        settings_menu.addAction(editCamera_action)
        stroage_action = QAction("Storage Unit Manage", self)
        settings_menu.addAction(stroage_action)
        watermark_action = QAction("WaterMark", self)
        settings_menu.addAction(watermark_action)
        menu_bar.addMenu(settings_menu)

        menu_bar.addMenu(help_menu)

        self.main_layout.setMenuBar(menu_bar)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())





# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess

# from draw_result import draw_results
# from ultralytics import YOLO

# frameWidth = 400
# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False  # False means detection mode enabled
#         self.recording = False  # For recording functionality
#         self.video_writer = None  # VideoWriter object for recording
#         self.output_file = None  # Path to save the recorded video
#         self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec here
#         self.fps = 30  # Default FPS, to be updated when initializing VideoWriter
#         self.frame_size = (640, 640)  # Default frame size, to be updated from video capture

#     def run(self):
#         global frameWidth
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"  # "GPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         self.fps = cam.get(cv2.CAP_PROP_FPS)
#         self.frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#         currentframe = 0
#         while self._run_flag:
#             ret, frame = cam.read()
#             if not ret:
#                 break

#             image = cv2.resize(frame, (640, 640))
#             if self.simple_mode:
#                 # Simple mode without detection
#                 result_image = cv2.resize(image, (frameWidth, frameWidth))
#                 self.change_pixmap_signal.emit(result_image)
#             else:
#                 # Detection mode
#                 detections = detect(image, det_compiled_model)[0]
#                 image_with_boxes = draw_results(detections, image, label_map)

#                 result_image = cv2.resize(image_with_boxes, (frameWidth, frameWidth))

#                 self.change_pixmap_signal.emit(result_image)

#             if self.recording:
#                 self.record_frame(frame)  # Recording logic

#             cv2.waitKey(1)
#             currentframe += 1

#         cam.release()
#         if self.video_writer:
#             self.video_writer.release()  # Release the video writer when done
#         cv2.destroyAllWindows()

#     def stop(self):
#         self._run_flag = False
#         self.wait()

#     def toggle_simple_mode(self):
#         # Toggle between simple mode and detection mode
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         # Toggle recording on and off
#         if self.recording:
#             self.recording = False
#             if self.video_writer:
#                 self.video_writer.release()
#                 self.video_writer = None
#             self.output_file = None
#         else:
#             self.recording = True
#             self.output_file = f"recording_{int(cv2.getTickCount())}.mp4"
#             self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)

#     def record_frame(self, frame):
#         # Convert the frame to RGB format before saving
#         if self.video_writer:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.video_writer.write(frame_rgb)

#     def close_channel(self):
#         # Stop the thread and release resources
#         self.stop()
#         self.wait()  # Ensure the thread has fully stopped
#         self._run_flag = False  # Ensure the run loop is stopped
#         self.video_writer = None
#         self.output_file = None

# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1200, 800)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["2x2", "3x3", "4x4"])
#         self.combobox.currentTextChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Camera1', 0)
#         self.button2 = self.create_button('Camera2', 1)
#         self.button3 = self.create_button('Camera3', 2)
#         self.button4 = self.create_button('Camera4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         # self.rtsp_urls = [
#         #     "http://158.58.130.148/mjpg/video.mjpg",
#         #     "rtsp://admin:Waq112299-@192.168.10.137:554/Streaming/Channels/101",
#         #     "rtsp://admin:Waq112299-@192.168.10.138:554/Streaming/Channels/101",
#         #     "http://88.53.197.250/axis-cgi/mjpg/video.cgi?resolution=320x240"
#         # ]

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#         ]

#         self.camera_threads = [None] * 4  # Initialize with None to indicate channels are closed

#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             self.start_thread(i, rtsp_url)

#         self.create_menu_bar()

#         # Create a black image for clearing the frame
#         self.black_image = np.zeros((400, 400, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         # Toggle Simple Mode action
#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index] and self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Enable Yolo Mode")
#         else:
#             toggle_simple_mode_action.setText("Disable Yolo Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         # Close Channel action
#         close_channel_action = QAction("Close Channel", self)
#         if self.camera_threads[camera_index] is None:
#             close_channel_action.setText("Start Channel")
#         else:
#             close_channel_action.setText("Close Channel")
#         close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

#         # Start/Stop Recording action
#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index] and self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         # Add actions to context menu
#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         # Toggle simple mode for the selected camera
#         if self.camera_threads[camera_index]:
#             self.camera_threads[camera_index].toggle_simple_mode()

#     def toggle_channel(self, camera_index):
#         # Start or close the camera channel based on its current state
#         if self.camera_threads[camera_index] is None:
#             self.start_thread(camera_index, self.rtsp_urls[camera_index])
#         else:
#             self.close_channel(camera_index)

#     def toggle_recording(self, camera_index):
#         if self.camera_threads[camera_index]:
#             self.camera_threads[camera_index].toggle_recording()

#     def close_channel(self, camera_index):
#         if self.camera_threads[camera_index]:
#             self.camera_threads[camera_index].close_channel()
#             self.camera_threads[camera_index] = None
#             label = self.grid_layout.itemAt(camera_index).widget()
#             label.setPixmap(self.convert_frame_to_pixmap(self.black_image))

    # def change_grid_size(self):
    #     layout_map = {"2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
    #     selected_layout = self.combobox.currentText()
    #     rows, cols = layout_map.get(selected_layout, (2, 2))  # Default to 2x2 if no selection found
    #     self.set_grid_layout(rows, cols)

    # def set_grid_layout(self, rows, cols):
    #     # Remove all widgets from the current layout
    #     global frameWidth
    #     if rows == 2:
    #         frameWidth = 400
    #     elif rows == 3:
    #         frameWidth = 300
    #     elif  rows ==4: 
    #         frameWidth = 230
    #     for i in reversed(range(self.grid_layout.count())):
    #         widget_to_remove = self.grid_layout.itemAt(i).widget()
    #         if widget_to_remove is not None:
    #             widget_to_remove.deleteLater()

    #     # Add new widgets to the grid layout
    #     # for row in range(rows):
    #     #     for col in range(cols):
    #     #         label = QLabel(f"Row {row+1}, Col {col+1}")
    #     #         # label.setPixmap(self.convert_frame_to_pixmap(self.black_image))
    #     #         self.grid_layout.addWidget(label, row, col)

    #     black_image = QImage(frameWidth, frameWidth, QImage.Format_RGB32)
    #     black_image.fill(Qt.black)
        
    #     # Convert black image to QPixmap
    #     pixmap = QPixmap.fromImage(black_image)

    #     # Add new widgets to the grid layout
    #     for row in range(rows):
    #         for col in range(cols):
    #             label = QLabel()
    #             label.setPixmap(pixmap)
    #             label.setFixedSize(frameWidth, frameWidth)  # Set the label size
    #             label.setScaledContents(True)  # Ensure the image fits the label size
    #             self.grid_layout.addWidget(label, row, col)


#     def start_thread(self, camera_index, rtsp_url):
#         thread = CameraThread(rtsp_url)
#         thread.change_pixmap_signal.connect(lambda image, idx=camera_index: self.update_image(image, idx))
#         self.camera_threads[camera_index] = thread
#         thread.start()

#     def update_image(self, image, camera_index):
#         label = self.grid_layout.itemAt(camera_index).widget()
#         if label:
#             label.setPixmap(self.convert_frame_to_pixmap(image))

#     def convert_frame_to_pixmap(self, image):
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(qt_image)

#     def create_menu_bar(self):
#         menu_bar = QMenuBar(self)
#         file_menu = QMenu("File", self)
#         option_menu = QMenu("Option", self)
#         edit_menu = QMenu("Edit", self)
#         settings_menu = QMenu("Settings", self)
#         help_menu = QMenu("Help", self)

#         exit_action = QAction("Exit", self)
#         exit_action.triggered.connect(self.close)
#         file_menu.addAction(exit_action)
#         menu_bar.addMenu(file_menu)
#         menu_bar.addMenu(option_menu)
#         menu_bar.addMenu(edit_menu)
#         addCamera_action = QAction("Add Camera", self)
#         settings_menu.addAction(addCamera_action)
#         editCamera_action = QAction("Edit Camera", self)
#         settings_menu.addAction(editCamera_action)
#         stroage_action = QAction("Storage Unit Manage", self)
#         settings_menu.addAction(stroage_action)
#         watermark_action = QAction("WaterMark", self)
#         settings_menu.addAction(watermark_action)
#         menu_bar.addMenu(settings_menu)

#         menu_bar.addMenu(help_menu)

#         self.main_layout.setMenuBar(menu_bar)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())



# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess
# from draw_result import draw_results
# from ultralytics import YOLO


# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False  # False means detection mode enabled
#         self.recording = False  # For recording functionality
#         self.video_writer = None  # VideoWriter object for recording
#         self.output_file = None  # Path to save the recorded video
#         self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec here
#         self.fps = 30  # Default FPS, to be updated when initializing VideoWriter
#         self.frame_size = (640, 640)  # Default frame size, to be updated from video capture

#     def run(self):
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"  # "GPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         self.fps = cam.get(cv2.CAP_PROP_FPS)
#         self.frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#         currentframe = 0
#         while self._run_flag:
#             ret, frame = cam.read()
#             if not ret:
#                 break

#             image = cv2.resize(frame, (640, 640))
#             if self.simple_mode:
#                 # Simple mode without detection
#                 self.change_pixmap_signal.emit(image)
#             else:
#                 # Detection mode
#                 detections = detect(image, det_compiled_model)[0]
#                 image_with_boxes = draw_results(detections, image, label_map)
#                 self.change_pixmap_signal.emit(image_with_boxes)

#             if self.recording:
#                 self.record_frame(frame)  # Recording logic

#             cv2.waitKey(1)
#             currentframe += 1

#         cam.release()
#         if self.video_writer:
#             self.video_writer.release()  # Release the video writer when done
#         cv2.destroyAllWindows()

#     def stop(self):
#         self._run_flag = False
#         self.wait()

#     def toggle_simple_mode(self):
#         # Toggle between simple mode and detection mode
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         # Toggle recording on and off
#         if self.recording:
#             self.recording = False
#             if self.video_writer:
#                 self.video_writer.release()
#                 self.video_writer = None
#             self.output_file = None
#         else:
#             self.recording = True
#             self.output_file = f"recording_{int(cv2.getTickCount())}.mp4"
#             self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)

#     def record_frame(self, frame):
#         # Convert the frame to RGB format before saving
#         if self.video_writer:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.video_writer.write(frame_rgb)

#     def close_channel(self):
#         # Stop the thread and release resources
#         self.stop()
#         self.wait()  # Ensure the thread has fully stopped
#         self._run_flag = False  # Ensure the run loop is stopped
#         self.video_writer = None
#         self.output_file = None


# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1600, 1700)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
#         self.combobox.currentIndexChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Test1', 0)
#         self.button2 = self.create_button('Test2', 1)
#         self.button3 = self.create_button('Test3', 2)
#         self.button4 = self.create_button('Test4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4"
#         ]

#         self.camera_threads = [None] * 4  # Initialize with None to indicate channels are closed

#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             self.start_thread(i, rtsp_url)

#         self.create_menu_bar()

#         # Create a black image for clearing the frame
#         self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         # Toggle Simple Mode action
#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index] and self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Disable Simple Mode")
#         else:
#             toggle_simple_mode_action.setText("Enable Simple Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         # Close Channel action
#         close_channel_action = QAction("Close Channel", self)
#         if self.camera_threads[camera_index] is None:
#             close_channel_action.setText("Start Channel")
#         else:
#             close_channel_action.setText("Close Channel")
#         close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

#         # Start/Stop Recording action
#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         # Add actions to context menu
#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         # Toggle simple mode for the selected camera
#         if self.camera_threads[camera_index]:
#             self.camera_threads[camera_index].toggle_simple_mode()

#     def toggle_channel(self, camera_index):
#         # Toggle channel on and off
#         if self.camera_threads[camera_index] is None:
#             rtsp_url = self.rtsp_urls[camera_index]
#             self.start_thread(camera_index, rtsp_url)
#         else:
#             self.camera_threads[camera_index].close_channel()
#             self.camera_threads[camera_index] = None

#     def start_thread(self, index, rtsp_url):
#         if index >= 0 and index < 4:
#             self.camera_threads[index] = CameraThread(rtsp_url)
#             self.camera_threads[index].change_pixmap_signal.connect(lambda frame, idx=index: self.update_image(frame, idx))
#             self.camera_threads[index].start()

#     def toggle_recording(self, camera_index):
#         # Toggle recording for the selected camera
#         self.camera_threads[camera_index].toggle_recording()

#     def update_image(self, frame, index):
#         if index >= 0 and index < 4:
#             qt_img = self.convert_cv_qt(frame)
#             label = self.findChild(QLabel, f"camera_{index}")
#             if label:
#                 label.setPixmap(qt_img)

#     def convert_cv_qt(self, cv_img):
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(q_img)

#     def change_grid_size(self, index):
#         sizes = [(1, 1), (2, 2), (3, 2), (4, 1)]
#         rows, cols = sizes[index]
#         self.set_grid_layout(rows, cols)

#     def set_grid_layout(self, rows, cols):
#         for i in reversed(range(self.grid_layout.count())):
#             widget = self.grid_layout.itemAt(i).widget()
#             if widget is not None:
#                 widget.deleteLater()

#         self.grid_layout.setSpacing(0)
#         self.grid_layout.setContentsMargins(0, 0, 0, 0)
#         for i in range(rows):
#             for j in range(cols):
#                 label = QLabel()
#                 label.setFixedSize(640, 640)
#                 self.grid_layout.addWidget(label, i, j)
#                 label.setObjectName(f"camera_{i * cols + j}")

#     def create_menu_bar(self):
#         menubar = QMenuBar(self)
#         file_menu = QMenu("File", self)
#         menubar.addMenu(file_menu)

#         save_action = QAction("Save", self)
#         save_action.triggered.connect(self.save_image)
#         file_menu.addAction(save_action)

#         main_layout = QVBoxLayout()
#         main_layout.setMenuBar(menubar)
#         self.main_layout.addLayout(main_layout)

#     def save_image(self):
#         # Save the currently displayed image to a file
#         current_label = self.findChild(QLabel, "camera_0")  # Example: saving the image from the first camera
#         if current_label:
#             pixmap = current_label.pixmap()
#             if pixmap:
#                 pixmap.save("saved_image.png")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())



# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess
# from draw_result import draw_results
# from ultralytics import YOLO


# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False  # False means detection mode enabled
#         self.recording = False  # For recording functionality
#         self.video_writer = None  # VideoWriter object for recording
#         self.output_file = None  # Path to save the recorded video
#         self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec here
#         self.fps = 30  # Default FPS, to be updated when initializing VideoWriter
#         self.frame_size = (640, 640)  # Default frame size, to be updated from video capture

#     def run(self):
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"  # "GPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         self.fps = cam.get(cv2.CAP_PROP_FPS)
#         self.frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#         currentframe = 0
#         while self._run_flag:
#             ret, frame = cam.read()
#             if not ret:
#                 break

#             image = cv2.resize(frame, (640, 640))
#             if self.simple_mode:
#                 # Simple mode without detection
#                 self.change_pixmap_signal.emit(image)
#             else:
#                 # Detection mode
#                 detections = detect(image, det_compiled_model)[0]
#                 image_with_boxes = draw_results(detections, image, label_map)
#                 self.change_pixmap_signal.emit(image_with_boxes)

#             if self.recording:
#                 self.record_frame(frame)  # Recording logic

#             cv2.waitKey(1)
#             currentframe += 1

#         cam.release()
#         if self.video_writer:
#             self.video_writer.release()  # Release the video writer when done
#         cv2.destroyAllWindows()

#     def stop(self):
#         self._run_flag = False
#         self.wait()

#     def toggle_simple_mode(self):
#         # Toggle between simple mode and detection mode
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         # Toggle recording on and off
#         if self.recording:
#             self.recording = False
#             if self.video_writer:
#                 self.video_writer.release()
#                 self.video_writer = None
#             self.output_file = None
#         else:
#             self.recording = True
#             self.output_file = f"recording_{int(cv2.getTickCount())}.mp4"
#             self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)

#     def record_frame(self, frame):
#         # Convert the frame to RGB format before saving
#         if self.video_writer:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.video_writer.write(frame_rgb)

#     def close_channel(self):
#         # Stop the thread and release resources
#         self.stop()
#         self.wait()  # Ensure the thread has fully stopped
#         self._run_flag = False  # Ensure the run loop is stopped
#         self.video_writer = None
#         self.output_file = None


# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1600, 1700)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
#         self.combobox.currentIndexChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Test1', 0)
#         self.button2 = self.create_button('Test2', 1)
#         self.button3 = self.create_button('Test3', 2)
#         self.button4 = self.create_button('Test4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4"
#         ]

#         self.camera_threads = [None] * 4  # Initialize with None to indicate channels are closed

#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             self.start_thread(i, rtsp_url)

#         self.create_menu_bar()

#         # Create a black image for clearing the frame
#         self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         # Toggle Simple Mode action
#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index] and self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Disable Simple Mode")
#         else:
#             toggle_simple_mode_action.setText("Enable Simple Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         # Close Channel action
#         close_channel_action = QAction("Close Channel", self)
#         if self.camera_threads[camera_index] is None:
#             close_channel_action.setText("Start Channel")
#         else:
#             close_channel_action.setText("Close Channel")
#         close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

#         # Start/Stop Recording action
#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index] and self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         # Add actions to context menu
#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         # Toggle simple mode for the selected camera
#         if self.camera_threads[camera_index]:
#             self.camera_threads[camera_index].toggle_simple_mode()

#     def toggle_channel(self, camera_index):
#         # Toggle channel on and off
#         if self.camera_threads[camera_index] is None:
#             rtsp_url = self.rtsp_urls[camera_index]
#             self.start_thread(camera_index, rtsp_url)
#         else:
#             self.camera_threads[camera_index].close_channel()
#             self.camera_threads[camera_index] = None

#     def start_thread(self, index, rtsp_url):
#         if index >= 0 and index < 4:
#             self.camera_threads[index] = CameraThread(rtsp_url)
#             self.camera_threads[index].change_pixmap_signal.connect(lambda frame, idx=index: self.update_image(frame, idx))
#             self.camera_threads[index].start()

#     def update_image(self, frame, index):
#         if index >= 0 and index < 4:
#             qt_img = self.convert_cv_qt(frame)
#             label = self.findChild(QLabel, f"camera_{index}")
#             if label:
#                 label.setPixmap(qt_img)

#     def convert_cv_qt(self, cv_img):
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(q_img)

#     def change_grid_size(self, index):
#         sizes = [(1, 1), (2, 2), (3, 2), (4, 1)]
#         rows, cols = sizes[index]
#         self.set_grid_layout(rows, cols)

#     def set_grid_layout(self, rows, cols):
#         for i in reversed(range(self.grid_layout.count())):
#             widget = self.grid_layout.itemAt(i).widget()
#             if widget is not None:
#                 widget.deleteLater()

#         self.grid_layout.setSpacing(0)
#         self.grid_layout.setContentsMargins(0, 0, 0, 0)
#         for i in range(rows):
#             for j in range(cols):
#                 label = QLabel()
#                 label.setFixedSize(640, 640)
#                 self.grid_layout.addWidget(label, i, j)
#                 label.setObjectName(f"camera_{i * cols + j}")

#     def create_menu_bar(self):
#         menubar = QMenuBar(self)
#         file_menu = QMenu("File", self)
#         menubar.addMenu(file_menu)

#         save_action = QAction("Save", self)
#         save_action.triggered.connect(self.save_image)
#         file_menu.addAction(save_action)

#         main_layout = QVBoxLayout()
#         main_layout.setMenuBar(menubar)
#         self.main_layout.addLayout(main_layout)

#     def save_image(self):
#         # Save the currently displayed image to a file
#         current_label = self.findChild(QLabel, "camera_0")  # Example: saving the image from the first camera
#         if current_label:
#             pixmap = current_label.pixmap()
#             if pixmap:
#                 pixmap.save("saved_image.png")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())



# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess
# from draw_result import draw_results
# from ultralytics import YOLO


# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False  # False means detection mode enabled
#         self.recording = False  # For recording functionality
#         self.video_writer = None  # VideoWriter object for recording
#         self.output_file = None  # Path to save the recorded video
#         self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec here
#         self.fps = 30  # Default FPS, to be updated when initializing VideoWriter
#         self.frame_size = (640, 640)  # Default frame size, to be updated from video capture

#     def run(self):
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"  # "GPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         self.fps = cam.get(cv2.CAP_PROP_FPS)
#         self.frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#         currentframe = 0
#         while self._run_flag:
#             ret, frame = cam.read()
#             if not ret:
#                 break

#             image = cv2.resize(frame, (640, 640))
#             if self.simple_mode:
#                 # Simple mode without detection
#                 self.change_pixmap_signal.emit(image)
#             else:
#                 # Detection mode
#                 detections = detect(image, det_compiled_model)[0]
#                 image_with_boxes = draw_results(detections, image, label_map)
#                 self.change_pixmap_signal.emit(image_with_boxes)

#             if self.recording:
#                 self.record_frame(frame)  # Recording logic

#             cv2.waitKey(1)
#             currentframe += 1

#         cam.release()
#         if self.video_writer:
#             self.video_writer.release()  # Release the video writer when done
#         cv2.destroyAllWindows()

#     def stop(self):
#         self._run_flag = False
#         self.wait()

#     def toggle_simple_mode(self):
#         # Toggle between simple mode and detection mode
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         # Toggle recording on and off
#         if self.recording:
#             self.recording = False
#             if self.video_writer:
#                 self.video_writer.release()
#                 self.video_writer = None
#             self.output_file = None
#         else:
#             self.recording = True
#             self.output_file = f"recording_{int(cv2.getTickCount())}.mp4"
#             self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)

#     def record_frame(self, frame):
#         # Convert the frame to RGB format before saving
#         if self.video_writer:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.video_writer.write(frame_rgb)

#     def close_channel(self):
#         # Stop the thread and release resources
#         self.stop()


# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1600, 1700)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
#         self.combobox.currentIndexChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Test1', 0)
#         self.button2 = self.create_button('Test2', 1)
#         self.button3 = self.create_button('Test3', 2)
#         self.button4 = self.create_button('Test4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4"
#         ]

#         self.camera_threads = []
#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             thread = CameraThread(rtsp_url)
#             thread.change_pixmap_signal.connect(lambda frame, i=i: self.update_image(frame, i))
#             self.camera_threads.append(thread)
#             thread.start()

#         self.create_menu_bar()

#         # Create a black image for clearing the frame
#         self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         # Toggle Simple Mode action
#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Disable Simple Mode")
#         else:
#             toggle_simple_mode_action.setText("Enable Simple Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         # Close Channel action
#         close_channel_action = QAction("Close Channel", self)
#         close_channel_action.triggered.connect(lambda: self.toggle_channel(camera_index))

#         # Start/Stop Recording action
#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         # Add actions to context menu
#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         # Toggle simple mode for the selected camera
#         self.camera_threads[camera_index].toggle_simple_mode()

#     def toggle_channel(self, camera_index):
#         # Toggle the channel: close it if it's open, or restart it if it's closed
#         if self.camera_threads[camera_index] is None:
#             # Restart the thread
#             rtsp_url = self.rtsp_urls[camera_index]
#             self.camera_threads[camera_index] = CameraThread(rtsp_url)
#             self.camera_threads[camera_index].change_pixmap_signal.connect(lambda frame, i=camera_index: self.update_image(frame, i))
#             self.camera_threads[camera_index].start()
#         else:
#             # Close the channel
#             self.camera_threads[camera_index].close_channel()
#             self.camera_threads[camera_index] = None  # Mark the thread as closed

#     def toggle_recording(self, camera_index):
#         # Toggle recording for the selected camera
#         self.camera_threads[camera_index].toggle_recording()

#     def update_image(self, frame, index):
#         if index >= 0 and index < 4:
#             qt_img = self.convert_cv_qt(frame)
#             label = self.findChild(QLabel, f"camera_{index}")
#             if label:
#                 label.setPixmap(qt_img)

#     def convert_cv_qt(self, cv_img):
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(q_img)

#     def change_grid_size(self, index):
#         sizes = [(1, 1), (2, 2), (3, 2), (4, 1)]
#         rows, cols = sizes[index]
#         self.set_grid_layout(rows, cols)

#     def set_grid_layout(self, rows, cols):
#         for i in reversed(range(self.grid_layout.count())):
#             widget = self.grid_layout.itemAt(i).widget()
#             if widget is not None:
#                 widget.deleteLater()

#         self.grid_layout.setSpacing(0)
#         self.grid_layout.setContentsMargins(0, 0, 0, 0)
#         for i in range(rows):
#             for j in range(cols):
#                 label = QLabel()
#                 label.setFixedSize(640, 640)
#                 self.grid_layout.addWidget(label, i, j)
#                 label.setObjectName(f"camera_{i * cols + j}")

#     def create_menu_bar(self):
#         menubar = QMenuBar(self)
#         file_menu = QMenu("File", self)
#         menubar.addMenu(file_menu)

#         save_action = QAction("Save", self)
#         save_action.triggered.connect(self.save_image)
#         file_menu.addAction(save_action)

#         main_layout = QVBoxLayout()
#         main_layout.setMenuBar(menubar)
#         self.main_layout.addLayout(main_layout)

#     def save_image(self):
#         # Save the currently displayed image to a file
#         current_label = self.findChild(QLabel, "camera_0")  # Example: saving the image from the first camera
#         if current_label:
#             pixmap = current_label.pixmap()
#             if pixmap:
#                 pixmap.save("saved_image.png")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())






# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess
# from draw_result import draw_results
# from ultralytics import YOLO



# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False
#         self.recording = False
#         self.is_running = False  # Flag to check if the thread is running

#     def run(self):
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         self.is_running = True
#         while self._run_flag:
#             ret, frame = cam.read()
#             image = cv2.resize(frame, (640, 640))
#             if ret:
#                 if self.simple_mode:
#                     self.change_pixmap_signal.emit(image)
#                 else:
#                     detections = detect(image, det_compiled_model)[0]
#                     image_with_boxes = draw_results(detections, image, label_map)
#                     self.change_pixmap_signal.emit(image_with_boxes)

#                 if self.recording:
#                     self.record_frame(frame)

#                 cv2.waitKey(1)
#             else:
#                 break
#         cam.release()
#         cv2.destroyAllWindows()
#         self.is_running = False

#     def stop(self):
#         self._run_flag = False
#         self.wait()
#         self.is_running = False

#     def start_thread(self):
#         if not self.is_running:
#             self._run_flag = True
#             self.start()

#     def toggle_simple_mode(self):
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         self.recording = not self.recording

#     def record_frame(self, frame):
#         pass

#     def close_channel(self):
#         if self.is_running:
#             self.stop()
#         else:
#             self.start_thread()

# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1600, 1700)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
#         self.combobox.currentIndexChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Test1', 0)
#         self.button2 = self.create_button('Test2', 1)
#         self.button3 = self.create_button('Test3', 2)
#         self.button4 = self.create_button('Test4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4"
#         ]

#         self.camera_threads = []
#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             thread = CameraThread(rtsp_url)
#             thread.change_pixmap_signal.connect(lambda frame, i=i: self.update_image(frame, i))
#             self.camera_threads.append(thread)
#             thread.start()

#         self.create_menu_bar()

#         self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Disable Simple Mode")
#         else:
#             toggle_simple_mode_action.setText("Enable Simple Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         close_channel_action = QAction("Close Channel", self)
#         if self.camera_threads[camera_index].is_running:
#             close_channel_action.setText("Stop Channel")
#         else:
#             close_channel_action.setText("Start Channel")
#         close_channel_action.triggered.connect(lambda: self.close_channel(camera_index))

#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         self.camera_threads[camera_index].toggle_simple_mode()

#     def close_channel(self, camera_index):
#         if 0 <= camera_index < len(self.camera_threads):
#             self.camera_threads[camera_index].close_channel()

#             # Set the label corresponding to the closed channel to a black image if stopped
#             if not self.camera_threads[camera_index].is_running:
#                 if 0 <= camera_index < len(self.labels):
#                     black_image_qimage = self.convert_np_to_qimage(self.black_image)
#                     self.labels[camera_index].setPixmap(QPixmap.fromImage(black_image_qimage))

#     def toggle_recording(self, camera_index):
#         if 0 <= camera_index < len(self.camera_threads):
#             self.camera_threads[camera_index].toggle_recording()

#     def set_grid_layout(self, rows, cols):
#         for i in reversed(range(self.grid_layout.count())):
#             widget = self.grid_layout.itemAt(i).widget()
#             if widget:
#                 widget.setParent(None)

#         self.labels = []
#         for row in range(rows):
#             for col in range(cols):
#                 label = QLabel()
#                 label.setFixedSize(640, 640)
#                 self.grid_layout.addWidget(label, row, col)
#                 self.labels.append(label)

#     def change_grid_size(self):
#         selected_size = self.combobox.currentText()
#         if selected_size == "1x1":
#             self.set_grid_layout(1, 1)
#         elif selected_size == "2x2":
#             self.set_grid_layout(2, 2)
#         elif selected_size == "3x2":
#             self.set_grid_layout(3, 2)
#         elif selected_size == "4x1":
#             self.set_grid_layout(4, 1)

#     def update_image(self, frame, index):
#         if 0 <= index < len(self.labels):
#             qt_image = self.convert_cv_qt(frame)
#             self.labels[index].setPixmap(qt_image)

#     def convert_cv_qt(self, rgb_image):
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(qt_image)

#     def convert_np_to_qimage(self, np_image):
#         h, w, ch = np_image.shape
#         bytes_per_line = ch * w
#         qimage = QImage(np_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return qimage

#     def create_menu_bar(self):
#         menu_bar = QMenuBar(self)

#         file_menu = QMenu("File", self)
#         edit_menu = QMenu("Edit", self)
#         option_menu = QMenu("Option", self)
#         settings_menu = QMenu("Settings", self)

#         menu_bar.addMenu(file_menu)
#         menu_bar.addMenu(edit_menu)
#         menu_bar.addMenu(option_menu)
#         menu_bar.addMenu(settings_menu)

#         self.main_layout.setMenuBar(menu_bar)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())












# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
#                              QMenu, QPushButton, QSpacerItem, QSizePolicy)

# from openvino.runtime import Core, Model
# from PIL import Image
# from Preprocessing import image_to_tensor, preprocess_image
# from Postprocessing import postprocess
# from draw_result import draw_results
# from ultralytics import YOLO


# def detect(image: np.ndarray, model: Model):
#     num_outputs = len(model.outputs)
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     masks = None
#     if num_outputs > 1:
#         masks = result[model.output(1)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
#     return detections


# class CameraThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)

#     def __init__(self, rtsp_url, detect_model=None):
#         super().__init__()
#         self._run_flag = True
#         self.rtsp_url = rtsp_url
#         self.detect_model = detect_model
#         self.simple_mode = False  # False means detection mode enabled
#         self.recording = False  # For recording functionality

#     def run(self):
#         det_model_path = "yolov8n.pt"
#         det_model = YOLO(det_model_path)
#         label_map = det_model.model.names
#         source_path = "test2.mp4"

#         core = Core()
#         det_model_path = "yolov8n.xml"
#         det_ov_model = core.read_model(det_model_path)
#         device = "CPU"  # "GPU"
#         if device != "CPU":
#             det_ov_model.reshape({0: [1, 3, 640, 640]})
#         det_compiled_model = core.compile_model(det_ov_model, device)

#         cam = cv2.VideoCapture(self.rtsp_url)
#         currentframe = 0
#         while self._run_flag:
#             ret, frame = cam.read()
#             image = cv2.resize(frame, (640, 640))
#             if ret:
#                 if self.simple_mode:
#                     # Simple mode without detection
#                     self.change_pixmap_signal.emit(image)
#                 else:
#                     # Detection mode
#                     detections = detect(image, det_compiled_model)[0]
#                     image_with_boxes = draw_results(detections, image, label_map)
#                     self.change_pixmap_signal.emit(image_with_boxes)

#                 if self.recording:
#                     self.record_frame(frame)  # Recording logic

#                 cv2.waitKey(1)
#                 currentframe += 1
#             else:
#                 break
#         cam.release()
#         cv2.destroyAllWindows()

#     def stop(self):
#         self._run_flag = False
#         self.wait()

#     def toggle_simple_mode(self):
#         # Toggle between simple mode and detection mode
#         self.simple_mode = not self.simple_mode

#     def toggle_recording(self):
#         # Toggle recording on and off
#         self.recording = not self.recording

#     def record_frame(self, frame):
#         # Logic for saving the frame to disk
#         # You can implement this with OpenCV's VideoWriter
#         pass

#     def close_channel(self):
#         # Stop the thread and release resources
#         self.stop()


# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Stream")
#         self.setGeometry(100, 100, 1600, 1700)

#         self.main_layout = QVBoxLayout()
#         self.main_frame = QHBoxLayout()
#         self.sidebar = QVBoxLayout()
#         self.sidebar.setSpacing(0)
#         self.sidebar.setContentsMargins(0, 0, 0, 0)
#         self.grid_layout = QGridLayout()

#         self.sidebar_widget = QWidget()
#         self.sidebar_widget.setLayout(self.sidebar)
#         self.sidebar_widget.setFixedWidth(200)

#         self.combobox = QComboBox()
#         self.combobox.addItems(["1x1", "2x2", "3x2", "4x1"])
#         self.combobox.currentIndexChanged.connect(self.change_grid_size)
#         self.combobox.setFixedWidth(200)
#         self.main_layout.addWidget(self.combobox)

#         self.button1 = self.create_button('Test1', 0)
#         self.button2 = self.create_button('Test2', 1)
#         self.button3 = self.create_button('Test3', 2)
#         self.button4 = self.create_button('Test4', 3)
#         self.sidebar.addWidget(self.button1)
#         self.sidebar.addWidget(self.button2)
#         self.sidebar.addWidget(self.button3)
#         self.sidebar.addWidget(self.button4)

#         spacer = QSpacerItem(5, 40, QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.sidebar.addItem(spacer)

#         self.main_frame.addWidget(self.sidebar_widget)
#         self.set_grid_layout(2, 2)
#         self.main_frame.addLayout(self.grid_layout)
#         self.main_layout.addLayout(self.main_frame)
#         self.setLayout(self.main_layout)

#         self.rtsp_urls = [
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4",
#             "test2.mp4"
#         ]

#         self.camera_threads = []
#         for i, rtsp_url in enumerate(self.rtsp_urls):
#             thread = CameraThread(rtsp_url)
#             thread.change_pixmap_signal.connect(lambda frame, i=i: self.update_image(frame, i))
#             self.camera_threads.append(thread)
#             thread.start()

#         self.create_menu_bar()

#         # Create a black image for clearing the frame
#         self.black_image = np.zeros((640, 640, 3), dtype=np.uint8)

#     def create_button(self, text, camera_index):
#         button = QPushButton(text)
#         button.setContextMenuPolicy(Qt.CustomContextMenu)
#         button.customContextMenuRequested.connect(lambda pos, b=button, idx=camera_index: self.show_context_menu(pos, b, idx))
#         return button

#     def show_context_menu(self, pos, button, camera_index):
#         context_menu = QMenu(self)

#         # Toggle Simple Mode action
#         toggle_simple_mode_action = QAction("Toggle Simple Mode", self)
#         if self.camera_threads[camera_index].simple_mode:
#             toggle_simple_mode_action.setText("Disable Simple Mode")
#         else:
#             toggle_simple_mode_action.setText("Enable Simple Mode")
#         toggle_simple_mode_action.triggered.connect(lambda: self.toggle_simple_mode(camera_index))

#         # Close Channel action
#         close_channel_action = QAction("Close Channel", self)
#         close_channel_action.triggered.connect(lambda: self.close_channel(camera_index))

#         # Start/Stop Recording action
#         toggle_recording_action = QAction("Start Recording", self)
#         if self.camera_threads[camera_index].recording:
#             toggle_recording_action.setText("Stop Recording")
#         toggle_recording_action.triggered.connect(lambda: self.toggle_recording(camera_index))

#         # Add actions to context menu
#         context_menu.addAction(toggle_simple_mode_action)
#         context_menu.addAction(close_channel_action)
#         context_menu.addAction(toggle_recording_action)

#         context_menu.exec_(button.mapToGlobal(pos))

#     def toggle_simple_mode(self, camera_index):
#         # Toggle simple mode for the selected camera
#         self.camera_threads[camera_index].toggle_simple_mode()

#     def close_channel(self, camera_index):
#         # Close the camera thread (stop the stream) and show a black image
#         if 0 <= camera_index < len(self.camera_threads):
#             self.camera_threads[camera_index].close_channel()
#             self.camera_threads[camera_index] = None  # Remove reference to the closed thread

#             # Set the label corresponding to the closed channel to a black image
#             if 0 <= camera_index < len(self.labels):
#                 black_image_qimage = self.convert_np_to_qimage(self.black_image)
#                 self.labels[camera_index].setPixmap(QPixmap.fromImage(black_image_qimage))

#     def toggle_recording(self, camera_index):
#         # Toggle recording for the selected camera
#         if 0 <= camera_index < len(self.camera_threads):
#             self.camera_threads[camera_index].toggle_recording()

#     def set_grid_layout(self, rows, cols):
#         for i in reversed(range(self.grid_layout.count())):
#             widget = self.grid_layout.itemAt(i).widget()
#             if widget:
#                 widget.setParent(None)

#         self.labels = []
#         for row in range(rows):
#             for col in range(cols):
#                 label = QLabel()
#                 label.setFixedSize(640, 640)  # Set the label size
#                 self.grid_layout.addWidget(label, row, col)
#                 self.labels.append(label)

#     def change_grid_size(self):
#         selected_size = self.combobox.currentText()
#         if selected_size == "1x1":
#             self.set_grid_layout(1, 1)
#         elif selected_size == "2x2":
#             self.set_grid_layout(2, 2)
#         elif selected_size == "3x2":
#             self.set_grid_layout(3, 2)
#         elif selected_size == "4x1":
#             self.set_grid_layout(4, 1)

#     def update_image(self, frame, index):
#         if 0 <= index < len(self.labels):
#             qt_image = self.convert_cv_qt(frame)
#             self.labels[index].setPixmap(qt_image)

#     def convert_cv_qt(self, rgb_image):
#         """Convert from OpenCV to QPixmap"""
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return QPixmap.fromImage(qt_image)

#     def convert_np_to_qimage(self, np_image):
#         """Convert from NumPy array to QImage"""
#         h, w, ch = np_image.shape
#         bytes_per_line = ch * w
#         qimage = QImage(np_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         return qimage

#     def create_menu_bar(self):
#         menu_bar = QMenuBar(self)

#         file_menu = QMenu("File", self)
#         edit_menu = QMenu("Edit", self)
#         option_menu = QMenu("Option", self)
#         settings_menu = QMenu("Settings", self)

#         settings_menu_1 = QAction("Option 1", self)
#         settings_menu_2 = QAction("Option 2", self)
#         settings_menu_3 = QAction("Option 3", self)
#         settings_menu.addAction(settings_menu_1)
#         settings_menu.addAction(settings_menu_2)
#         settings_menu.addAction(settings_menu_3)

#         menu_bar.addMenu(file_menu)
#         menu_bar.addMenu(edit_menu)
#         menu_bar.addMenu(option_menu)
#         menu_bar.addMenu(settings_menu)

#         self.main_layout.setMenuBar(menu_bar)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = App()
#     win.show()
#     sys.exit(app.exec_())



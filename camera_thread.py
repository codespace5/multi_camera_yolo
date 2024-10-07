import cv2
import numpy as np
import datetime
import os
from PyQt5.QtCore import QThread, pyqtSignal

import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QGridLayout, QWidget, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QMenuBar, QAction, 
                             QMenu, QPushButton, QSpacerItem, QSizePolicy)

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


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url, detect_model=None):
        super().__init__()
        self._run_flag = True
        self.rtsp_url = rtsp_url
        self.detect_model = detect_model
        self.simple_mode = False
        self.recording = False
        self.is_running = False
        self.out = None  # VideoWriter object for recording

    def run(self):
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
        self.is_running = True
        while self._run_flag:
            ret, frame = cam.read()
            image = cv2.resize(frame, (640, 640))
            if ret:
                if self.simple_mode:
                    self.change_pixmap_signal.emit(image)
                else:
                    detections = detect(image, det_compiled_model)[0]
                    image_with_boxes = draw_results(detections, image, label_map)
                    self.change_pixmap_signal.emit(image_with_boxes)

                if self.recording:
                    self.record_frame(frame)

                cv2.waitKey(1)
            else:
                break
        cam.release()
        if self.out is not None:
            self.out.release()  # Release the video writer
        cv2.destroyAllWindows()
        self.is_running = False

    def stop(self):
        self._run_flag = False
        self.wait()
        self.is_running = False

    def start_thread(self):
        if not self.is_running:
            self._run_flag = True
            self.start()

    def toggle_simple_mode(self):
        self.simple_mode = not self.simple_mode

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.avi"
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 640))
        self.recording = True

    def stop_recording(self):
        if self.out is not None:
            self.out.release()  # Release the video writer
            self.out = None
        self.recording = False

    def record_frame(self, frame):
        if self.out is not None:
            self.out.write(frame)

    def close_channel(self):
        if self.is_running:
            self.stop()
        else:
            self.start_thread()

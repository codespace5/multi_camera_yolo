
from ultralytics import YOLO
from PIL import Image
import numpy as np
IMAGE_PATH = "bus.jpg"
from draw_result import draw_results

DET_MODEL_NAME = "yolov8n"

det_model = YOLO(f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

results = next(det_model(IMAGE_PATH, return_outputs=True))

img = np.array(Image.open(IMAGE_PATH))
img_with_boxes = draw_results(results, img.copy(), label_map)
Image.fromarray(img_with_boxes)
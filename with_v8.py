from openvino.runtime import Core, Model
import numpy as np
from PIL import Image
from Preprocessing import image_to_tensor, preprocess_image
from Postprocessing import postprocess
from draw_result import draw_results, plot_one_box
from ultralytics import YOLO
import cv2
import os

# os.environ["Path"] = "C:\Users\Simba\\anaconda3\Lib\site-packages\openvino\libs"

def detect(image:np.ndarray, model:Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
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

def _main():
    det_model_path = "yolov8n.pt"
    det_model = YOLO(det_model_path)
    label_map = det_model.model.names

    #source_path = "bus.jpg"
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

        #resultImage = Image.fromarray(image_with_boxes)
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        cv2.imshow(source_path, image_with_boxes)
        cv2.waitKey(0)
        
    else:
        
        # cam = cv2.VideoCapture(source_path)
        # cam = cv2.VideoCapture("rtsp://admin:Waq112299-@192.168.10.138:554/Streaming/1/1")
        # cam = cv2.VideoCapture("rtsp://admin:Waq112299-@192.168.10.137:554/Streaming/1/1")
        cam = cv2.VideoCapture("http://158.58.130.148:80/mjpg/video.mjpg")

        # frame
        currentframe = 0
        #i = 0
        while(True):
            # reading from frame
            ret,frame = cam.read()
            #i += 1
            #if i % 2:
                #continue
            if ret:
                # if video is still left continue creating images
                detections = detect(frame, det_compiled_model)[0]
                image_with_boxes = draw_results(detections, frame, label_map)
                #image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                
                cv2.imshow(source_path, image_with_boxes)
                cv2.waitKey(1)
                # writing the extracted images
                #cv2.imshow(source_dir, result)
                #cv2.waitKey(1)
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break
        
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    _main()

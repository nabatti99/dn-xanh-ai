from typing import *
from PIL import Image
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

classification_model = YOLO("classification.pt")
logger.info("Classification model loaded.")

detection_model = YOLO("yolo11x.pt")
logger.info("Detection model loaded.")

def classify_image(images: List[Image.Image]) -> List[str]:
    predictions = classification_model.predict(images)

    results = []
    for prediction in predictions:
        prediction_name = prediction.names[prediction.probs.top1]
        results.append(prediction_name)

    return results

def detect_image(images: List[Image.Image]) -> List[List[dict]]:
    predictions = detection_model.predict(images)

    results = []
    for prediction in predictions:
        result = []

        for box in prediction.boxes:
            result.append({
                "name": prediction.names[int(box.cls)],
                "xywh": box.xywh[0].tolist(),
            })
        results.append(result)

    return results

def check_can_classification(images: List[Image.Image]) -> List[bool]:
    result = []

    detections = detect_image(images)
    for index, detection in enumerate(detections):
        can_clasification = False

        image = images[index]
        image_width, image_height = image.size
        
        for detection_object in detection:
            logger.info(detection_object)
            detection_name, detection_xywh = detection_object.values()
            logger.info(f"Detection: {detection_name}, {detection_xywh}")
            x, y, w, h = detection_xywh

            if detection_name == "person":
                continue

            if w > image_width * 0.7 or h > image_height * 0.7:
                can_clasification = True

        result.append(can_clasification)

    return result
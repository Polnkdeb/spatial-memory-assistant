import json
import os
from ultralytics import YOLO
import cv2
from tqdm import tqdm

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)  # предобучен на COCO (80 классов)
        # COCO классы, которые нас интересуют (сопоставление запросам)
        self.class_mapping = {
            "dog": 16,      # класс собаки в COCO
            "bowl": 44,     # миска – ближе всего к "тарелке"
            "plate": 44,    # тарелка – тоже миска/чаша
            "cup": 41,      # кружка
            # можно добавить другие соответствия
        }

    def detect_on_frames(self, frames_dir, output_json_path):
        """Детекция всех .jpg в frames_dir, сохранение результата в JSON"""
        detections = {}  
        image_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        for img_file in tqdm(image_files, desc="Detecting"):
            img_path = os.path.join(frames_dir, img_file)
            results = self.model(img_path)[0] 
            detections[img_file] = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = results.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections[img_file].append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        with open(output_json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        return detections
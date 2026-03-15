from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model: str = "yolo11s.pt"):
        self.model = YOLO(model)

    def train(self, data: str, epochs: int = 50, imgz: int = 1024, batch: int = 8):
        self.model.train(
            data=data,
            epochs=epochs,
            imgz=imgz,
            batch=batch
        )

    def predict_bounding_boxes(self, image_path: str, conf: float = 0.25):
        results = self.model.predict(source=image_path, conf=conf)

        boxes_out = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf_score = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                boxes_out.append({
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "confidence": conf_score,
                    "bbox": [x1, y1, x2, y2]
                })

        return boxes_out
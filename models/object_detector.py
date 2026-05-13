"""Stage-1 detector: YOLO that finds generic Everdell entities in a photo.

Wraps Ultralytics YOLO. Returns boxes plus PIL crops so the stage-2
identifier can run directly on each detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from PIL import Image
from ultralytics import YOLO


class Detection(TypedDict):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in source-image pixels
    crop: Image.Image  # RGB crop of the source image


class ObjectDetector:
    def __init__(self, model: str = "yolo11s.pt"):
        self.model = YOLO(model)

    def train(
        self,
        data: str,
        epochs: int = 50,
        imgsz: int = 1024,
        batch: int = 8,
        **kwargs,
    ):
        return self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            **kwargs,
        )

    def predict_bounding_boxes(self, image_path: str, conf: float = 0.25):
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
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
                    "bbox": [x1, y1, x2, y2],
                })
        return boxes_out

    def detect(
        self,
        image: str | Path | Image.Image,
        conf: float = 0.25,
        pad: int = 4,
    ) -> list[Detection]:
        """Run detection and return boxes alongside cropped PIL images."""
        if isinstance(image, (str, Path)):
            src = Image.open(image).convert("RGB")
            source_arg = str(image)
        else:
            src = image.convert("RGB")
            source_arg = src

        W, H = src.size
        results = self.model.predict(source=source_arg, conf=conf, verbose=False)

        out: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # pad and clip
                cx1 = max(0, int(x1) - pad)
                cy1 = max(0, int(y1) - pad)
                cx2 = min(W, int(x2) + pad)
                cy2 = min(H, int(y2) + pad)
                crop = src.crop((cx1, cy1, cx2, cy2))
                out.append({
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "confidence": float(box.conf.item()),
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop,
                })
        return out

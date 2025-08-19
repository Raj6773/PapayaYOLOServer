from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str):
        # TorchScript model: just load, no .to(device)
        self.model = YOLO(model_path)

    def detect_and_draw(self, image_path: str, save_path: str = None):
        """
        Detect objects and save/show image with bounding boxes.
        Returns list of detections.
        """
        results = self.model(image_path, save=False, imgsz=640, conf=0.25)
        res = results[0]

        # Save visualization
        if save_path:
            res.save(save_path)

        # Collect detections
        detections = []
        for box in res.boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy.tolist()[0]]
            })
        return detections

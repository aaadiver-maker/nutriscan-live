"""
Standalone food-detection worker, run as a SEPARATE PROCESS from the main
Streamlit app.

Why this exists: ultralytics (PyTorch) and TensorFlow crash with a
segmentation fault when both are loaded in the same Python process
(confirmed while building this app - detection alone works, classification
alone works, but doing both in-process reliably segfaults on inference).
Running detection in its own process keeps PyTorch and TensorFlow fully
isolated from each other, which avoids the crash entirely.

Usage: python3 detect_worker.py <image_path>
Prints one JSON line to stdout: {"box": [x1,y1,x2,y2] | null, "label": str | null,
                                  "conf": float | null, "source": "detected"|"fallback"}
"""
import json
import os
import sys
import tempfile

# Must be set before `import ultralytics` - on locked-down hosting (e.g.
# Streamlit Community Cloud) the default ~/.config location it tries to
# write settings.json to can be read-only, which raises before detection
# ever runs. Point it at a directory we know is writable instead.
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

YOLO_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
CONF_THRESHOLD = 0.25
FOOD_COCO_CLASSES = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
}


def main():
    image_path = sys.argv[1]

    from ultralytics import YOLO
    from PIL import Image

    detector = YOLO(YOLO_WEIGHTS_PATH)
    pil_img = Image.open(image_path).convert("RGB")

    import numpy as np
    results = detector.predict(np.array(pil_img), conf=CONF_THRESHOLD, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        label = detector.names[int(box.cls[0])]
        if label in FOOD_COCO_CLASSES:
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append((xyxy, label, conf_score))
    detections.sort(key=lambda x: x[2], reverse=True)

    result = {"box": None, "label": None, "conf": None, "source": "fallback"}
    if detections:
        (x1, y1, x2, y2), label, conf_score = detections[0]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(pil_img.width, int(x2)), min(pil_img.height, int(y2))
        if x2 > x1 and y2 > y1:
            result = {"box": [x1, y1, x2, y2], "label": label, "conf": conf_score, "source": "detected"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()

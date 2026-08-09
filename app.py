"""
NutriScan AI Food Scanner - live demo app.

Reproduces the exact Milestone 1 / Milestone 2 pipeline for a single photo:
  1. YOLOv8n (COCO weights) detects a food-relevant region, conf >= 0.25
  2. Crop to that region, or keep the full photo if nothing was found (fallback)
  3. Resize to 224x224, RGB
  4. ResNet50 preprocess_input, run through the trained Variant 3 classifier
  5. Show the predicted food type + confidence

Run with:  streamlit run app.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="NutriScan Scanner", page_icon="🍽️", layout="centered")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "variant3_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMG_SIZE = (224, 224)

# Fallback only - matches Milestone 1's `sorted(os.listdir(data_root))`.
# Overridden automatically if model/label_map.json is present.
DEFAULT_CLASS_NAMES = [
    "apple_pie", "chocolate_cake", "donuts", "falafel", "french_fries",
    "hot_dog", "ice_cream", "nachos", "onion_rings", "pancakes", "pizza",
    "ravioli", "samosa", "spring_rolls", "strawberry_shortcake", "tacos",
    "waffles",
]


DETECT_WORKER_PATH = os.path.join(os.path.dirname(__file__), "detect_worker.py")

# Reference lookup, NOT a model output - the classifier predicts food type
# only. Approximate kcal per typical single serving (USDA-style estimates).
CALORIE_MAP = {
    "apple_pie": (296, "1 slice, ~125g"),
    "chocolate_cake": (371, "1 slice, ~95g"),
    "donuts": (253, "1 glazed donut"),
    "falafel": (333, "1 serving, ~4 pieces"),
    "french_fries": (365, "1 medium serving"),
    "hot_dog": (290, "1 hot dog with bun"),
    "ice_cream": (207, "1/2 cup"),
    "nachos": (346, "1 serving with cheese"),
    "onion_rings": (411, "1 serving, ~8 rings"),
    "pancakes": (350, "1 stack, 3 pancakes"),
    "pizza": (285, "1 slice"),
    "ravioli": (330, "1 cup, cheese-filled"),
    "samosa": (262, "1 samosa"),
    "spring_rolls": (150, "1 roll, fried"),
    "strawberry_shortcake": (344, "1 slice"),
    "tacos": (226, "1 taco"),
    "waffles": (218, "1 waffle"),
}


@st.cache_resource(show_spinner="Loading NutriScan classifier...")
def load_classifier():
    import tensorflow as tf
    if not os.path.exists(CLASSIFIER_PATH):
        return None
    return tf.keras.models.load_model(CLASSIFIER_PATH)


@st.cache_resource
def load_class_names():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            mapping = json.load(f)
        # accepts either {"0": "apple_pie", ...} or {"apple_pie": 0, ...}
        if all(isinstance(k, str) and k.isdigit() for k in mapping):
            ordered = sorted(mapping.items(), key=lambda kv: int(kv[0]))
            return [v for _, v in ordered]
        ordered = sorted(mapping.items(), key=lambda kv: kv[1])
        return [k for k, _ in ordered]
    return DEFAULT_CLASS_NAMES


def detect_food_box(pil_img):
    """
    Runs food detection in a SEPARATE process (see detect_worker.py) rather
    than importing ultralytics/PyTorch here. PyTorch and TensorFlow loaded
    in the same process segfault on inference - confirmed while building
    this app - so detection is kept fully isolated from the classifier.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, DETECT_WORKER_PATH, tmp_path],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Detection worker failed (exit {result.returncode}).\n"
                f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}"
            )
        data = json.loads(result.stdout.strip().splitlines()[-1])
    finally:
        os.remove(tmp_path)

    if data["box"] is None:
        return None, None, None, "fallback"
    return tuple(data["box"]), data["label"], data["conf"], data["source"]


def preprocess_for_classifier(pil_img):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.uint8).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def draw_box(pil_img, box, label, conf_score):
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="#4472A8", width=max(3, pil_img.width // 200))
    tag = f"{label} {conf_score:.2f}"
    draw.rectangle([x1, y1 - 22, x1 + 9 * len(tag), y1], fill="#4472A8")
    draw.text((x1 + 3, y1 - 20), tag, fill="white")
    return annotated


def format_class_name(name):
    return name.replace("_", " ").title()


def main():
    st.markdown(
        '<div style="display:flex; align-items:baseline; gap:0.6rem;">'
        '<span style="font-size:2.25rem; font-weight:700; line-height:1;">NutriScan</span>'
        '<span style="font-size:1.125rem; font-weight:400; color:#808495;">by: Gan CM</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Photo in. Meal logged. — take or upload a photo of your food.")

    classifier = load_classifier()
    if classifier is None:
        st.warning(
            "Classifier model not found. Copy your trained `variant3_model.keras` "
            "(and optionally `label_map.json`) into the `model/` folder next to this app, "
            "then reload the page. The food-detection step will still work without it."
        )

    class_names = load_class_names()

    tab_upload, tab_camera = st.tabs(["🖼️ Upload a photo", "📷 Take a photo"])
    image = None
    with tab_upload:
        up_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if up_file is not None:
            image = Image.open(up_file)
    with tab_camera:
        cam_file = st.camera_input("Point at your meal")
        if cam_file is not None:
            image = Image.open(cam_file)

    if image is None:
        st.info("Waiting for a photo...")
        return

    image = image.convert("RGB")

    with st.spinner("Finding the food in your photo..."):
        box, coco_label, det_conf, source = detect_food_box(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your photo")
        if box is not None:
            st.image(draw_box(image, box, coco_label, det_conf), use_container_width=True)
            st.caption(f"Located automatically (detector confidence {det_conf:.0%}).")
        else:
            st.image(image, use_container_width=True)
            st.caption("Couldn't confidently locate the food, so the full photo was used.")

    crop = image.crop(box) if box is not None else image

    with col2:
        st.subheader("Result")
        if classifier is None:
            st.error("Classifier not loaded — see warning above.")
        else:
            with st.spinner("Identifying the food..."):
                batch = preprocess_for_classifier(crop)
                probs = classifier.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_name = format_class_name(class_names[top_idx])
            top_conf = float(probs[top_idx])

            st.metric(top_name, f"{top_conf:.0%} confidence")

            calorie_info = CALORIE_MAP.get(class_names[top_idx])
            if calorie_info:
                kcal, serving = calorie_info
                st.markdown(f"**~{kcal} kcal** <span style='color:#808495;'>({serving}, estimated)</span>", unsafe_allow_html=True)
            else:
                st.caption("Calorie estimate not available for this food type.")

            order = np.argsort(probs)[::-1][:3]
            st.caption("Other possibilities")
            for i in order:
                st.progress(float(probs[i]), text=f"{format_class_name(class_names[i])} — {probs[i]:.0%}")

    st.divider()
    st.caption(
        f"Detection pathway: **{source}**  •  "
        "This mirrors the exact Milestone 1/2 pipeline: YOLOv8n detection -> "
        "crop with fallback -> ResNet50 classifier (Variant 3)."
    )


if __name__ == "__main__":
    main()

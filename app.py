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
import base64
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="NutriScan Scanner", page_icon="🍽️", layout="centered")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "variant3_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

ICON_DIR = os.path.join(os.path.dirname(__file__), "icon")
UPLOAD_ICON_PATH = os.path.join(ICON_DIR, "upload_icon.png")
CAMERA_ICON_PATH = os.path.join(ICON_DIR, "camera_icon.png")

IMG_SIZE = (224, 224)
ACCENT_COLOR = "#4A6FA1"

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
# only. Approximate values per typical single serving (USDA-style estimates).
# Fields: kcal, serving description, protein_g, carbs_g, fat_g.
NUTRITION_MAP = {
    "apple_pie": (296, "1 slice, ~125g", 2.4, 42, 14),
    "chocolate_cake": (371, "1 slice, ~95g", 5, 50, 18),
    "donuts": (253, "1 glazed donut", 4, 31, 14),
    "falafel": (333, "1 serving, ~4 pieces", 13, 31, 18),
    "french_fries": (365, "1 medium serving", 4, 48, 17),
    "hot_dog": (290, "1 hot dog with bun", 10, 24, 17),
    "ice_cream": (207, "1/2 cup", 3.5, 24, 11),
    "nachos": (346, "1 serving with cheese", 9, 36, 19),
    "onion_rings": (411, "1 serving, ~8 rings", 5, 44, 23),
    "pancakes": (350, "1 stack, 3 pancakes", 9, 50, 12),
    "pizza": (285, "1 slice", 12, 36, 10),
    "ravioli": (330, "1 cup, cheese-filled", 14, 40, 12),
    "samosa": (262, "1 samosa", 4, 28, 15),
    "spring_rolls": (150, "1 roll, fried", 3, 15, 9),
    "strawberry_shortcake": (344, "1 slice", 4, 48, 16),
    "tacos": (226, "1 taco", 9, 20, 12),
    "waffles": (218, "1 waffle", 6, 25, 10),
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


NEON_GREEN = "#39FF14"


def draw_box(pil_img, box, label, conf_score):
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = box

    line_width = max(3, pil_img.width // 200)
    draw.rectangle([x1, y1, x2, y2], outline=NEON_GREEN, width=line_width)

    tag = f"{label} {conf_score:.2f}"
    font_size = max(14, pil_img.width // 30)
    font = ImageFont.load_default(size=font_size)

    # Measure the actual rendered text instead of guessing pixels-per-character,
    # so the label background always fits the text regardless of font/length.
    text_bbox = draw.textbbox((0, 0), tag, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = max(2, font_size // 6)

    tag_top = y1 - text_h - 2 * pad
    if tag_top < 0:
        # Not enough room above the box (it's near the top edge of the photo) -
        # draw the label just inside the box instead of letting it clip off-frame.
        tag_top = y1
    tag_bottom = tag_top + text_h + 2 * pad

    draw.rectangle([x1, tag_top, x1 + text_w + 2 * pad, tag_bottom], fill=NEON_GREEN)
    draw.text((x1 + pad, tag_top + pad - text_bbox[1]), tag, fill="black", font=font)
    return annotated


def format_class_name(name):
    return name.replace("_", " ").title()


@st.cache_data
def get_base64_icon(path, _mtime):
    # `_mtime` is part of the cache key purely so a changed icon file busts
    # the cache automatically - without it, a git-pushed asset swap under the
    # same filename kept serving the old cached image after a hot reload,
    # since st.cache_data only keys on the arguments passed in, not on
    # whether the file's contents actually changed.
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_icon(path, active):
    # Plain <img>, no <a href> - an anchor-tag click forces the browser to
    # do a full page reload (new HTTP request, new Streamlit session setup),
    # which is what made switching feel slow. The actual click handling
    # below uses a real (CSS-hidden) st.button instead, which is a native
    # Streamlit rerun over the existing connection - fast.
    b64 = get_base64_icon(path, os.path.getmtime(path))
    border = f"3px solid {ACCENT_COLOR}" if active else "3px solid transparent"
    st.markdown(
        f'<div style="display:flex; justify-content:center;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%; max-width:90px; border:{border}; border-radius:8px;"></div>',
        unsafe_allow_html=True,
    )


def main():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: Calibri, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        h1, h2, h3 {
            font-family: Cambria, Georgia, serif !important;
            color: #1E293B !important;
        }
        /* Scoped to actual st.caption() elements only - an earlier version of
           this rule targeted plain p/span/label globally, which also dimmed
           text inside buttons and progress bars to the point of being
           unreadable. Keep this narrow. */
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: #323A45;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            font-family: Cambria, Georgia, serif;
            color: #1E293B;
        }
        [data-testid="stAlertContainer"] {
            background-color: #F1F4F8;
            border-radius: 8px;
        }
        /* Give the upload dropzone the same footprint as the camera
           widget, which is naturally much taller (it reserves space for
           a video preview) - without this the two panels jump size
           dramatically depending on which mode is active. */
        [data-testid="stFileUploaderDropzone"] {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stCameraInput"] {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        /* Target the fill bar specifically (role=progressbar), not its
           track wrapper - targeting the wrapper paints the whole bar solid
           regardless of actual percentage and swallows the label text. */
        [data-testid="stProgress"] div[role="progressbar"] {
            background-color: #4A6FA1 !important;
        }
        /* Real (fast) buttons used as the click target for switching icons,
           absolutely positioned to cover the icon image exactly so the
           whole icon is clickable, not just an invisible strip below it. */
        [data-testid="stVerticalBlock"]:has(.st-key-select_upload),
        [data-testid="stVerticalBlock"]:has(.st-key-select_camera) {
            position: relative;
        }
        .st-key-select_upload, .st-key-select_camera {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        .st-key-select_upload button, .st-key-select_camera button {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            background-color: transparent !important;
            border: none !important;
            color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex; align-items:baseline; gap:0.6rem;">'
        '<span style="font-family:Cambria, Georgia, serif; font-size:2.25rem; font-weight:700; '
        'line-height:1; color:#1E293B;">NutriScan</span>'
        '<span style="font-size:1.125rem; font-weight:400; color:#64748B;">by: Gan CM</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Photo in. Meal logged. Take or upload a photo of your food.")

    classifier = load_classifier()
    if classifier is None:
        st.warning(
            "Classifier model not found. Copy your trained `variant3_model.keras` "
            "(and optionally `label_map.json`) into the `model/` folder next to this app, "
            "then reload the page. The food-detection step will still work without it."
        )

    class_names = load_class_names()

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "upload"

    if "has_photo" not in st.session_state:
        st.session_state.has_photo = False

    image = None
    # Keep the whole picker (icons + upload/camera widget) in a narrower
    # left column instead of spanning the full page width - both the icon
    # row and the uploader/camera box looked stretched and oversized
    # otherwise, especially compared to the compact icons.
    picker_col, _picker_spacer = st.columns([2, 1])
    with picker_col:
        # Buttons are processed FIRST, before either icon is drawn, so the
        # active-state border reflects this click immediately rather than
        # lagging a run behind - Streamlit executes top-to-bottom in one
        # pass, so rendering an icon's border before checking whether ITS
        # OWN or the OTHER icon's button was just clicked would use last
        # run's state.
        icon_col1, icon_col2, _icon_spacer = st.columns([1, 1, 2])
        with icon_col1:
            upload_slot = st.empty()
            if st.button(" ", key="select_upload", use_container_width=True):
                st.session_state.input_mode = "upload"
        with icon_col2:
            camera_slot = st.empty()
            if st.button(" ", key="select_camera", use_container_width=True):
                st.session_state.input_mode = "camera"

        with upload_slot:
            render_icon(UPLOAD_ICON_PATH, st.session_state.input_mode == "upload")
        with camera_slot:
            render_icon(CAMERA_ICON_PATH, st.session_state.input_mode == "camera")

        # Once a photo has already been provided, collapse the (fairly
        # large) upload/camera widget behind a "+" expander instead of
        # leaving it open above the result - otherwise the result gets
        # pushed far down the page every time, underneath a widget the
        # user is already done with.
        with st.expander("➕ Add a photo", expanded=not st.session_state.has_photo):
            if st.session_state.input_mode == "upload":
                up_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
                if up_file is not None:
                    image = Image.open(up_file)
            else:
                cam_file = st.camera_input("Point at your meal")
                if cam_file is not None:
                    image = Image.open(cam_file)

        if image is None:
            # Rendered inside picker_col too, so it lines up with the same
            # narrow width as everything else above instead of stretching
            # full page width on its own.
            st.info("Waiting for a photo...")

    if image is None:
        return

    if not st.session_state.has_photo:
        # First time a photo shows up: the expander above was already drawn
        # this run (with expanded=True, since has_photo was still False when
        # we entered it), so setting has_photo alone wouldn't visually
        # collapse it until some later interaction. Rerun once immediately -
        # the uploaded/captured file stays bound to its widget across the
        # rerun, so this doesn't lose the photo, it just redraws collapsed.
        st.session_state.has_photo = True
        st.rerun()

    image = image.convert("RGB")

    with st.spinner("Finding the food in your photo..."):
        box, coco_label, det_conf, source = detect_food_box(image)

    crop = image.crop(box) if box is not None else image

    # Classification runs before either column is drawn, so the bounding
    # box can be labeled with the actual identified food (e.g. "Hot Dog
    # 100%") instead of the generic detector's own guess (e.g. "donut
    # 0.38") - the detector only knows 80 generic COCO categories and is
    # sometimes wrong about WHAT it found, even when the box location
    # itself is fine. The real answer always comes from the classifier.
    top_name, top_conf, probs = None, None, None
    if classifier is not None:
        with st.spinner("Identifying the food..."):
            batch = preprocess_for_classifier(crop)
            probs = classifier.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_name = format_class_name(class_names[top_idx])
        top_conf = float(probs[top_idx])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your photo")
        if box is not None:
            box_label = top_name if top_name is not None else coco_label
            box_conf = top_conf if top_conf is not None else det_conf
            st.image(draw_box(image, box, box_label, box_conf), use_container_width=True)
            st.caption(f"Located automatically (detector confidence {det_conf:.0%}).")
        else:
            st.image(image, use_container_width=True)
            st.caption("Analyzed the full photo directly. No crop was needed for this one.")

    with col2:
        st.subheader("Result")
        if classifier is None:
            st.error("Classifier not loaded. See warning above.")
        else:
            st.metric(top_name, f"{top_conf:.0%} confidence")

            nutrition_info = NUTRITION_MAP.get(class_names[top_idx])
            if nutrition_info:
                kcal, serving, protein_g, carbs_g, fat_g = nutrition_info
                st.markdown(f"**~{kcal} kcal** per typical serving ({serving})")
                st.markdown(f"• Protein {protein_g}g&nbsp;&nbsp;&nbsp;• Carbs {carbs_g}g&nbsp;&nbsp;&nbsp;• Fat {fat_g}g")
            else:
                st.caption("Nutrition estimate not available for this food type.")

            order = np.argsort(probs)[::-1][:3]
            st.caption("Other possibilities")
            for i in order:
                st.progress(float(probs[i]), text=f"{format_class_name(class_names[i])}: {probs[i]:.0%}")

    st.divider()
    st.caption(
        f"Detection pathway: **{source}**  •  "
        "This mirrors the exact Milestone 1/2 pipeline: YOLOv8n detection -> "
        "crop with fallback -> ResNet50 classifier (Variant 3)."
    )


if __name__ == "__main__":
    main()

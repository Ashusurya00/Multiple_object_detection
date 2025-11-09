import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="🧠 Object & Animal Detection",
    layout="centered",
    page_icon="🦁"
)

st.title("🦁 Object & Animal Detection App")
st.write("Upload or capture an image — detect multiple objects and animals using YOLOv8.")

# -------------------------------------------------
# LOAD YOLO MODEL (auto-downloads if not present)
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        st.info("⏳ Loading YOLOv8 model... please wait a moment.")
        model = YOLO("yolov8s.pt")  # Automatically downloads if missing
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load YOLO model: {e}")
        return None

model = load_model()

# -------------------------------------------------
# DETECTION FUNCTION
# -------------------------------------------------
def detect_objects(image_path):
    """Run YOLOv8 detection on an image path."""
    results = model.predict(
        source=image_path,
        conf=0.45,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    result_img = results[0].plot()  # annotated image
    labels = results[0].names

    detected_items = []
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        detected_items.append((labels[cls], conf))
    return result_img, detected_items

# -------------------------------------------------
# INPUT METHOD
# -------------------------------------------------
st.subheader("Choose Input Method:")
option = st.radio("", ["📁 Upload Image", "📸 Use Camera"])

# -------------------------------------------------
# 📁 Upload Image Option
# -------------------------------------------------
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file and model:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            img = Image.open(uploaded_file).convert("RGB")
            img.save(temp_path, format="JPEG")

        st.info("🔍 Running object detection...")
        result_img, detections = detect_objects(temp_path)

        st.image(result_img, caption="Detected Objects & Animals", use_container_width=True)

        if detections:
            st.success("✅ Detected:")
            for name, conf in detections:
                st.write(f"• **{name.capitalize()}** ({conf*100:.1f}%)")
        else:
            st.warning("⚠️ No objects detected.")

# -------------------------------------------------
# 📸 Camera Input Option (with Retake)
# -------------------------------------------------
elif option == "📸 Use Camera":
    if "photo_taken" not in st.session_state:
        st.session_state.photo_taken = False

    if not st.session_state.photo_taken:
        picture = st.camera_input("Take a photo")

        if picture and model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
                img = Image.open(picture).convert("RGB")
                img.save(temp_path, format="JPEG")

            st.info("🔍 Running object detection...")
            result_img, detections = detect_objects(temp_path)

            st.image(result_img, caption="Detected Objects & Animals", use_container_width=True)

            if detections:
                st.success("✅ Detected:")
                for name, conf in detections:
                    st.write(f"• **{name.capitalize()}** ({conf*100:.1f}%)")
            else:
                st.warning("⚠️ No objects detected.")

            st.session_state.photo_taken = True

    if st.session_state.photo_taken:
        if st.button("🔁 Retake Photo"):
            st.session_state.photo_taken = False
            st.experimental_rerun()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown("👨‍💻 **Developed by Ashutosh Suryawanshi** — powered by YOLOv8 + Streamlit 🚀")

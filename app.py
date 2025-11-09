import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tempfile
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="🧠 Smart Object & Animal Detector", layout="centered")
st.title("🦁 Smart Object & Animal Detection App")
st.write("Upload or capture an image — detects animals and objects using YOLOv8 + MobileNetV2 hybrid AI system.")

# --------------------------------------------------
# LOAD MODELS (cached for performance)
# --------------------------------------------------
@st.cache_resource
def load_yolo():
    # Medium-sized YOLOv8 for better accuracy than 'n' or 's'
    model = YOLO("yolov8m.pt")
    return model

@st.cache_resource
def load_mobilenet():
    model = MobileNetV2(weights="imagenet")
    return model

yolo_model = load_yolo()
mobilenet_model = load_mobilenet()
st.success("✅ YOLOv8 + MobileNetV2 models loaded successfully!")

# --------------------------------------------------
# DETECTION FUNCTION
# --------------------------------------------------
def detect_objects(image_path):
    """Detect objects with YOLO and refine animal labels using MobileNetV2."""
    results = yolo_model.predict(
        source=image_path,
        conf=0.45,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes
    result_img = results[0].plot()

    labels = results[0].names
    detections = []

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        name = labels[cls]

        # Crop region for MobileNet classification
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
        cropped = cropped.resize((224, 224))
        x = np.expand_dims(image.img_to_array(cropped), axis=0)
        x = preprocess_input(x)

        preds = mobilenet_model.predict(x)
        top = decode_predictions(preds, top=1)[0][0]

        refined_label = top[1].replace("_", " ")
        refined_conf = top[2]

        detections.append({
            "yolo_label": name,
            "yolo_conf": conf,
            "refined_label": refined_label,
            "refined_conf": refined_conf
        })

    return result_img, detections

# --------------------------------------------------
# INPUT METHOD SELECTION
# --------------------------------------------------
st.subheader("Choose Input Method:")
option = st.radio("", ["📁 Upload Image", "📸 Use Camera"])

# --------------------------------------------------
# 📁 Upload Image Option
# --------------------------------------------------
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            img = Image.open(uploaded_file).convert("RGB")
            img.save(temp_path, format="JPEG")

        st.info("🔍 Detecting objects and animals...")
        result_img, detections = detect_objects(temp_path)
        st.image(result_img, caption="Detected Objects & Animals", use_container_width=True)

        if detections:
            st.success("✅ Detected:")
            for d in detections:
                st.write(
                    f"• **YOLO:** {d['yolo_label']} ({d['yolo_conf']*100:.1f}%) "
                    f"→ **MobileNet:** {d['refined_label']} ({d['refined_conf']*100:.1f}%)"
                )
        else:
            st.warning("⚠️ No objects detected.")

# --------------------------------------------------
# 📸 Camera Input Option (with Retake)
# --------------------------------------------------
elif option == "📸 Use Camera":
    if "photo_taken" not in st.session_state:
        st.session_state.photo_taken = False

    if not st.session_state.photo_taken:
        picture = st.camera_input("Take a photo")

        if picture:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
                img = Image.open(picture).convert("RGB")
                img.save(temp_path, format="JPEG")

            st.info("🔍 Detecting objects and animals...")
            result_img, detections = detect_objects(temp_path)
            st.image(result_img, caption="Detected Objects & Animals", use_container_width=True)

            if detections:
                st.success("✅ Detected:")
                for d in detections:
                    st.write(
                        f"• **YOLO:** {d['yolo_label']} ({d['yolo_conf']*100:.1f}%) "
                        f"→ **MobileNet:** {d['refined_label']} ({d['refined_conf']*100:.1f}%)"
                    )
            else:
                st.warning("⚠️ No objects detected.")

            # Mark that a photo was taken
            st.session_state.photo_taken = True

    # Retake option
    if st.session_state.photo_taken:
        if st.button("🔁 Retake Photo"):
            st.session_state.photo_taken = False
            st.experimental_rerun()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("👨‍💻 **Developed by Ashutosh Suryawanshi** — powered by YOLOv8 + MobileNetV2 🚀")

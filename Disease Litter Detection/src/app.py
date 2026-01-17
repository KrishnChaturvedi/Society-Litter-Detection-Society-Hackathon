# My name is anthony gonjalish mein duniya mein akela hu

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time
from datetime import datetime

st.set_page_config(
    page_title="Society Litter Detection",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    h1, h2, h3, h4, h5, h6,
    p, span, div, label {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    button {
        background-color: #2ecc71 !important;
        color: #ffffff !important;
        border-radius: 6px;
        border: none;
    }

    button:hover {
        background-color: #27ae60 !important;
        color: #ffffff !important;
    }

    .stFileUploader label,
    .stFileUploader span,
    .stFileUploader div {
        color: #ffffff !important;
    }

    .stFileUploader {
        border: 1px dashed #2ecc71;
    }

    .stRadio label {
        color: #ffffff !important;
    }

    .stAlert, .stAlert * {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0

COOLDOWN_SECONDS = 30

@st.cache_resource
def load_models():
    litter_model = YOLO(r".\train4\weights\best.pt")
    person_model = YOLO("yolov8n.pt")
    return litter_model, person_model

litter_model, person_model = load_models()

st.sidebar.title("Society Litter Detection")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Camera Detection", "Image Upload"]
)

if page == "Home":
    st.title("🚯 Society Litter Detection System")
    st.subheader("AI-powered responsible litter monitoring")

    st.markdown("---")

    st.header("📌 How this system is built")
    st.markdown("""
    **Model Architecture**
    - YOLOv8 (Ultralytics)

    **Training Approach**
    - Transfer Learning from COCO
    - Fine-tuned on TACO litter dataset

    **Safety & Ethics**
    - Humans detected separately
    - Humans are never classified as trash

    **Deployment**
    - PyTorch inference
    - Streamlit web application
    """)

    st.markdown("---")

    st.header("👨‍👩‍👧 About the Team")
    st.markdown("""
    - Member 1 – AI / ML  
    - Member 2 – Backend  
    - Member 3 – UI / Presentation  
    """)

elif page == "Camera Detection":
    st.title("📷 Camera Detection (Auto-Capture Enabled)")
    st.caption("Auto-captures trash incidents with 30s cooldown")

    camera_image = st.camera_input("Open Camera")

    if camera_image:
        image = Image.open(camera_image)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Captured Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            person_results = person_model(img_bgr, conf=0.5)
            litter_results = litter_model(img_bgr, conf=0.4)

        person_boxes = [
            box.xyxy[0].cpu().numpy()
            for box in person_results[0].boxes
            if int(box.cls[0]) == 0
        ]

        litter_boxes = [
            box.xyxy[0].cpu().numpy()
            for box in litter_results[0].boxes
        ]

        annotated = img_bgr.copy()

        for x1, y1, x2, y2 in person_boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 0), 2)
            cv2.putText(annotated, "Human",
                        (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for x1, y1, x2, y2 in litter_boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated, "Trash",
                        (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Result", use_column_width=True)

        current_time = time.time()

        if litter_boxes:
            if current_time - st.session_state.last_capture_time >= COOLDOWN_SECONDS:

                desktop_path = os.path.join(
                    os.path.expanduser("~"),
                    "Desktop",
                    "litter_notifications"
                )
                os.makedirs(desktop_path, exist_ok=True)

                filename = datetime.now().strftime("incident_%Y%m%d_%H%M%S.jpg")
                save_path = os.path.join(desktop_path, filename)
                cv2.imwrite(save_path, annotated)

                st.session_state.last_capture_time = current_time

                if person_boxes:
                    st.warning("⚠️ Some random person is detected with trash")
                else:
                    st.error("🚨 Trash detected")

                st.success("📸 Image auto-captured")
                st.info(f"🗂 Saved to Desktop: {save_path}")

            else:
                remaining = int(
                    COOLDOWN_SECONDS - (current_time - st.session_state.last_capture_time)
                )
                st.info(f"⏳ Cooldown active ({remaining}s remaining)")

        else:
            st.success("✅ Area looks clean")

elif page == "Image Upload":
    st.title("🖼 Image Upload Detection")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            person_results = person_model(img_bgr, conf=0.5)
            litter_results = litter_model(img_bgr, conf=0.4)

        litter_boxes = litter_results[0].boxes

        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Result", use_column_width=True)

        if litter_boxes:
            st.error("🚨 Trash detected")
        else:
            st.success("✅ No trash detected")

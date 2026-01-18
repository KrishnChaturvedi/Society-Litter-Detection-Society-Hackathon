# My name is raghav gonjalish mein duniya mein akela hu

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage
import mimetypes

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Society Litter Detection",
    layout="wide"
)

# -------------------- STYLING --------------------

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

# -------------------- SESSION STATE --------------------
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

COOLDOWN_SECONDS = 30

# -------------------- EMAIL FUNCTION --------------------
def send_email_with_image(to_email, user_name, image_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Society Litter Alert Detected"
    msg["From"] = st.secrets["EMAIL_SENDER"]
    msg["To"] = to_email

    msg.set_content(
        f"""Hello {user_name},

Trash has been detected in your monitored area.
Please find the attached image for reference.

Regards,
Society Litter Detection System
"""
    )

    mime_type, _ = mimetypes.guess_type(image_path)
    maintype, subtype = mime_type.split("/")

    with open(image_path, "rb") as img:
        msg.add_attachment(
            img.read(),
            maintype=maintype,
            subtype=subtype,
            filename=os.path.basename(image_path)
        )

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(
            st.secrets["EMAIL_SENDER"],
            st.secrets["EMAIL_PASSWORD"]
        )
        server.send_message(msg)

# -------------------- IOU FUNCTION --------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    litter_model = YOLO("./train4/weights/best.pt")
    coco_model = YOLO("yolov8n.pt")
    return litter_model, coco_model

litter_model, coco_model = load_models()

# ==================== PIPELINE 1: GET STARTED ====================
if st.session_state.user_email is None:
    st.title("🚀 Get Started")

    name = st.text_input("Enter your name")
    email = st.text_input("Enter your email")

    if st.button("Submit"):
        if name and email:
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.success("Details saved successfully")
            st.rerun()
        else:
            st.error("Please enter both name and email")

    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("Society Litter Detection")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Camera Detection", "Image Upload"]
)

# ================= HOME =================
if page == "Home":
    st.title("🚯 Society Litter Detection System")
    st.subheader("AI-powered responsible litter monitoring")

    st.markdown("---")
    st.markdown("""
    **Model Architecture**
    - YOLOv8 (Ultralytics)

    **Training**
    - Transfer Learning
    - TACO Litter Dataset

    **Ethics**
    - Humans never classified as trash
    - Daily objects handled safely

    **Deployment**
    - PyTorch + Streamlit
    """)
    
# ==================== CAMERA ====================
elif page == "Camera Detection":
    st.title("📷 Camera Detection (Auto-Capture Enabled)")

    camera_image = st.camera_input("Open Camera")

    if camera_image:
        image = Image.open(camera_image)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing image..."):
            coco_results = coco_model(img, conf=0.4)
            litter_results = litter_model(img, conf=0.4)

        person_boxes, phone_boxes, bottle_boxes = [], [], []

        for box in coco_results[0].boxes:
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            if cls == 0:
                person_boxes.append(xyxy)
            elif cls == 67:
                phone_boxes.append(xyxy)
            elif cls == 39:
                bottle_boxes.append(xyxy)

        raw_litter_boxes = [b.xyxy[0].cpu().numpy() for b in litter_results[0].boxes]
        litter_boxes = []

        for lbox in raw_litter_boxes:
            if all(iou(lbox, r) < 0.3 for r in person_boxes + phone_boxes + bottle_boxes):
                litter_boxes.append(lbox)

        annotated = img.copy()

        for x1,y1,x2,y2 in person_boxes:
            cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),2)
            cv2.putText(annotated,"Human",(int(x1),int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

        for x1,y1,x2,y2 in phone_boxes:
            cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(annotated,"Phone",(int(x1),int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        for x1,y1,x2,y2 in bottle_boxes:
            cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            cv2.putText(annotated,"Bottle",(int(x1),int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        for x1,y1,x2,y2 in litter_boxes:
            cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            cv2.putText(annotated,"Trash",(int(x1),int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        st.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB), use_column_width=True)

        now = time.time()
        if litter_boxes and now - st.session_state.last_capture_time > COOLDOWN_SECONDS:
            save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "litter_notifications")
            os.makedirs(save_dir, exist_ok=True)

            fname = datetime.now().strftime("incident_%Y%m%d_%H%M%S.jpg")
            path = os.path.join(save_dir, fname)
            cv2.imwrite(path, annotated)

            send_email_with_image(
                st.session_state.user_email,
                st.session_state.user_name,
                path
            )

            st.session_state.last_capture_time = now
            st.error("🚨 Trash detected & email sent")

        elif not litter_boxes:
            st.success("✅ Area looks clean")

# ==================== IMAGE UPLOAD ====================
elif page == "Image Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        litter_results = litter_model(img, conf=0.4)
        st.image(image, use_column_width=True)

        if litter_results[0].boxes:
            st.error("🚨 Trash detected")
        else:
            st.success("✅ No trash detected")

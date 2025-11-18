"""
Streamlit YOLOv7 Object Detector
Detects: cheerios, soup, candle
Model: yolov7-tiny custom trained (your .pt file)
Works 100% on Streamlit Cloud
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os

# === CONFIG ===
MODEL_PATH = "yolov7_cheerios_soup_candle_best.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Class names — MUST match your training order!
CLASSES = ['cheerios', 'soup', 'candle']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR

# === Load Model (clean & fast) ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Official clean YOLOv7 package (no subprocess pip installs!)
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.torch_utils import select_device

    model = attempt_load(MODEL_PATH, device=device)  # auto handles map_location
    model.eval()
    
    st.success("YOLOv7 model loaded successfully!")
    return model, device

# === Preprocess Image (letterbox) ===
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    # Resize and pad image while maintaining aspect ratio
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)

# === Inference & Post-processing ===
def detect(model, image_pil, device):
    img0 = np.array(image_pil)
    img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)

    # Letterbox
    img, ratio, (dw, dh) = letterbox(img0_bgr, new_shape=IMG_SIZE)
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        pred = model(img)[0]  # 0 for batch index

    # Apply NMS
    from yolov7.utils.general import non_max_suppression
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)

    # Process detections
    det = pred[0]  # First batch
    boxes, scores, class_ids = [], [], []

    if len(det):
        # Rescale boxes from 640x640 to original image size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf.item())
            class_ids.append(int(cls.item()))

    return boxes, scores, class_ids, img0

def scale_boxes(img1_shape, boxes, img0_shape):
    # From yolov7.utils.general (simplified version)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= gain
    boxes[:, 0].clamp_(0, img0_shape[1])
    boxes[:, 1].clamp_(0, img0_shape[0])
    boxes[:, 2].clamp_(0, img0_shape[1])
    boxes[:, 3].clamp_(0, img0_shape[0])
    return boxes

# === Draw Boxes ===
def draw_boxes(img, boxes, scores, class_ids):
    img = img.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASSES[cls_id]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

# === Streamlit App ===
st.set_page_config(page_title="Cheerios • Soup • Candle Detector", layout="centered")
st.title("YOLOv7 Grocery Item Detector")
st.markdown("**Trained to detect:** `cheerios` • `soup` • `candle")

# Load model once
model, device = load_model()

# Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects..."):
        boxes, scores, class_ids, orig_img = detect(model, image, device)

        result_img = draw_boxes(orig_img, boxes, scores, class_ids)
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        st.image(result_pil, caption=f"Found {len(boxes)} object(s)!", use_column_width=True)

        if len(boxes) == 0:
            st.info("No cheerios, soup, or candles found.")
        else:
            st.success(f"Detected: {', '.join([CLASSES[i] for i in class_ids])}")

else:
    st.info("Upload an image to start detecting!")
    st.markdown("""
    ### Try with these classes:
    - Box of Cheerios
    - Can of soup
    - Scented candle
    """)
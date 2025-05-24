import sys
import os
import cv2
import torch
import json
from ultralytics import YOLO

# Add custom module paths
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath("../alerts"))

from vlv_tracker import get_vlv
from velocity_model import VelocityPredictor
from risk_model import RiskPredictor
from sms_alert import SmsAlert
from email_alert import EmailAlert
from hybrid_headway import estimate_headway
from prediction_horizon import map_visibility_to_prt, estimate_prediction_horizon

# File paths
VIDEO_PATH = "C:/RHINO-CAR/test_videos/test (2).mp4"
OUTPUT_PATH = "../output_video/processed_output.mp4"
LABEL_MAP_PATH = "../label_mapping.json"
YOLO_MODEL = "yolov8n.pt"
VELOCITY_MODEL_PATH = "../models/velocity_model.pth"
RISK_MODEL_PATH = "../models/collision_risk_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label mapping
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

allowed_classes = ["car", "truck", "motorcycle", "bus", "bicycle"]
vehicle_class_ids = [int(k) for k, v in label_map.items() if v in allowed_classes]

# Load models
print("[INFO] Loading models...")
yolo = YOLO(YOLO_MODEL)
velocity_model = VelocityPredictor().to(DEVICE)
velocity_model.load_state_dict(torch.load(VELOCITY_MODEL_PATH))
velocity_model.eval()

risk_model = RiskPredictor().to(DEVICE)
risk_model.load_state_dict(torch.load(RISK_MODEL_PATH))
risk_model.eval()

# Setup video reader
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (frame_width, frame_height))

# System configuration
RISK_THRESHOLD = 0.5
last_speeds = [40.0, 42.0, 41.0]
DEFAULT_SENSOR_VALUE = 250.0  # cm
DEFAULT_VISIBILITY = "sunny"

print("[INFO] Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    boxes = results.boxes
    vehicle_boxes = [box for box in boxes if int(box.cls) in vehicle_class_ids]

    # Placeholder for sensor headway and visibility (mocked)
    sensor_value = DEFAULT_SENSOR_VALUE
    visibility = DEFAULT_VISIBILITY

    # Get YOLO bounding box height
    _, y1, _, y2 = vehicle_boxes[0].xyxy[0] if vehicle_boxes else (0, 0, 0, 0)
    box_height = y2 - y1

    # Hybrid headway estimation
    headway = estimate_headway(sensor_value, box_height)

    # Predict Subject Vehicle Velocity (VSV)
    vsv_tensor = torch.tensor([last_speeds], dtype=torch.float32).to(DEVICE)
    vsv = velocity_model(vsv_tensor).item()
    last_speeds = [last_speeds[1], last_speeds[2], vsv]

    # Get real-time Leading Vehicle Velocity (VLV)
    vlv = get_vlv()

    # Predict collision risk
    x_risk = torch.tensor([[vsv, headway, vlv]], dtype=torch.float32).to(DEVICE)
    risk_score = risk_model(x_risk).item()

    # Prediction Horizon from visibility
    prt = map_visibility_to_prt(visibility)
    tph = estimate_prediction_horizon(prt, vlv)

    # Log info per frame
    print(f"[FRAME] Headway: {headway:.2f} m | VSV: {vsv:.2f} km/h | VLV: {vlv:.2f} km/h | Risk: {risk_score:.3f} | Tph: {tph}")

    # Annotate frame with risk info
    if risk_score > RISK_THRESHOLD:
        cv2.putText(frame, "⚠️ COLLISION RISK", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        TTC = headway / (vsv - vlv + 1e-3)
        if TTC < prt:
            SmsAlert(location="Video Frame").run()
            EmailAlert(location="Video Frame").run()
    else:
        cv2.putText(frame, "✓ Safe", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Overlay all info
    cv2.putText(frame, f"H: {headway:.2f}m VSV: {vsv:.2f}km/h VLV: {vlv:.2f}km/h", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Risk: {risk_score:.2f} | Tph: {tph}", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Use YOLO plot if available, fallback to raw frame
    if results and len(results.boxes) > 0:
        plotted = results.plot()
        plotted = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
    else:
        plotted = frame.copy()

    plotted = cv2.resize(plotted, (frame_width, frame_height))
    out.write(plotted)
    cv2.imshow("RHINO-X Video Detection", plotted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✔] Video processing complete. Output saved to: {OUTPUT_PATH}")

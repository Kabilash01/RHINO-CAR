import sys
import os
import cv2
import torch
import json
import serial
from ultralytics import YOLO

# Add utils and alerts to path
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath("../alerts"))

from velocity_model import VelocityPredictor
from risk_model import RiskPredictor
from prediction_horizon import map_visibility_to_prt, estimate_prediction_horizon
from hybrid_headway import estimate_headway
from alerts.sms_alert import SmsAlert
from alerts.email_alert import EmailAlert


# Paths
YOLO_MODEL = "yolov8n.pt"
VELOCITY_MODEL_PATH = "../models/velocity_model.pth"
RISK_MODEL_PATH = "../models/collision_risk_model.pth"
LABEL_MAP_PATH = "../label_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label mapping
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

# Vehicle classes of interest
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

# Read serial data from ESP32
def read_serial_data():
    try:
        with serial.Serial('COM3', 9600, timeout=1) as ser:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), parts[2].strip().lower()
    except:
        return None, None, None
    return None, None, None


# Thresholds
RISK_THRESHOLD = 0.5
PRT_DEFAULT = 1.5

# Placeholder speeds
last_speeds = [40.0, 42.0, 41.0]

cap = cv2.VideoCapture(0)
print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    boxes = results.boxes
    # Debug: Print all class names in this frame
    for box in boxes:
        class_name = yolo.names[int(box.cls)]
        print("Detected:", class_name)

    vehicle_boxes = [box for box in boxes if int(box.cls) in vehicle_class_ids]

    # Read from ESP32 (headway, VSV, visibility)
    headway, vsv_input, visibility = read_serial_data()

    if headway is None:
        print("[⚠️] No sensor input — estimating headway using YOLO.")
        if vehicle_boxes:
            _, y1, _, y2 = vehicle_boxes[0].xyxy[0]
            height = y2 - y1
            headway = 1000 / (height + 1e-3)
        else:
            headway = 100.0  # default fallback

    # Predict VSV
    vsv_tensor = torch.tensor([last_speeds], dtype=torch.float32).to(DEVICE)
    vsv = velocity_model(vsv_tensor).item()
    last_speeds = [last_speeds[1], last_speeds[2], vsv]

    # Estimate VLV
    vlv = 40.0  # constant placeholder
    prt = map_visibility_to_prt(visibility)
    tph = estimate_prediction_horizon(prt, vlv)
    print(f"[INFO] PRT = {prt:.2f}s | Dynamic Horizon = {tph} slots")

    # Predict Risk
    x_risk = torch.tensor([[vsv, headway, vlv]], dtype=torch.float32).to(DEVICE)
    risk_score = risk_model(x_risk).item()

    print(f"[✓] VSV: {vsv:.2f} | VLV: {vlv:.2f} | Headway: {headway:.2f} | Risk: {risk_score:.3f}")

    if risk_score > RISK_THRESHOLD:
        cv2.putText(frame, "⚠️ COLLISION RISK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("[⚠️] Collision Risk Detected")
        TTC = headway / (vsv - vlv + 1e-3)
        if TTC < PRT_DEFAULT:
            print("[🚨] Emergency! Sending alerts...")
            SmsAlert(location="Car 01 - Front Sensor").run()
            EmailAlert(location="Car 01 - Front Sensor").run()

    cv2.imshow("RHINO-X Detection", results.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

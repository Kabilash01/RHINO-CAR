
import cv2
import torch
import numpy as np
import sys
import os
from ultralytics import YOLO

# Add utils path
sys.path.append(os.path.abspath("../utils"))
from vlv_tracker import update_vlv

from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model (vehicle detection)
yolo = YOLO("yolov8n.pt")

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Define vehicle class IDs (COCO): car=2, motorcycle=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]

# Load input video
cap = cv2.VideoCapture("C:/RHINO-CAR/test_videos/test (11).mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_PATH = "../output_video/vlv_tracking_output.mp4"
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Track position history for velocity calc
lv_history = {}

print("[INFO] Starting VLV tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls)
        if cls in vehicle_classes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf)
            detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    leading_track = None
    min_y = float("inf")

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        # Identify leading vehicle as top-most on screen
        if t < min_y:
            min_y = t
            leading_track = (track_id, t, h)

        # Draw track box
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (255, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (int(l), int(t - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    if leading_track:
        track_id, top_y, height = leading_track
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if track_id in lv_history:
            prev_y, prev_time = lv_history[track_id]
            dy = prev_y - top_y
            dt = now - prev_time
            if dt > 0:
                vel_pixels_per_sec = dy / dt

                # Convert to real-world speed
                camera_fov_meters = 5.0
                meters_per_pixel = camera_fov_meters / frame_width
                vlv_mps = vel_pixels_per_sec * meters_per_pixel
                vlv_kmph = vlv_mps * 3.6

                # Share globally with other modules
                update_vlv(vlv_kmph)

                cv2.putText(frame, f"LV VLV: {vlv_kmph:.2f} km/h", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        lv_history[track_id] = (top_y, now)

    # Ensure correct output resolution
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)
    cv2.imshow("VLV Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✔] Tracking complete. Output saved to: {OUTPUT_PATH}")

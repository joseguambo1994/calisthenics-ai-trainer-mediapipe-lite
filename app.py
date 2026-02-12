import os
import requests

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo Pose Landmarker (lite, latest)...")
    with requests.get(MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("OK:", MODEL_PATH)
else:
    print("Modelo ya existe:", MODEL_PATH)

import os
import cv2
import mediapipe as mp

INPUT_VIDEO = "video-raw.mp4"
OUTPUT_VIDEO = "video-processed.mp4"
MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Missing: {INPUT_VIDEO}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing: {MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# BlazePose 33-landmark skeleton connections (stable)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
    (27,31),(28,32)
]

def draw_pose(frame_bgr, pose_landmarks, point_radius=2, line_thickness=2):
    h, w = frame_bgr.shape[:2]
    for a, b in POSE_CONNECTIONS:
        la, lb = pose_landmarks[a], pose_landmarks[b]
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), (0, 255, 0), line_thickness)
    for lm in pose_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), point_radius, (0, 0, 255), -1)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
if not out.isOpened():
    raise RuntimeError(f"Cannot open writer: {OUTPUT_VIDEO}")

timestamp_ms = 0
frame_time_ms = int(round(1000.0 / fps))
frames = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            draw_pose(frame_bgr, result.pose_landmarks[0])

        out.write(frame_bgr)
        frames += 1
        timestamp_ms += frame_time_ms

cap.release()
out.release()

print("DONE ✅", OUTPUT_VIDEO, "frames:", frames, "fps:", fps)

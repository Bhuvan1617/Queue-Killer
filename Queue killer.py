import time
import threading

import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.4
PERSON_CLASS_ID = 0

QUEUE_ZONE_POINTS = np.array([
    [50, 100],
    [300, 100],
    [300, 400],
    [50, 400],
])

SERVICE_ZONE_POINTS = np.array([
    [350, 100],
    [600, 100],
    [600, 400],
    [350, 400],
])

WALKOUT_GRACE_FRAMES = 15

# ---------------------------------------------------------------------------
# SHARED STATE - written by the CV loop, read by the API
# ---------------------------------------------------------------------------
class QueueState:
    def __init__(self):
        self.active_queue_count = 0
        self.estimated_wait_minutes = 0
        self.recent_walkouts = 0
        self.average_service_time_seconds = 0
        self.now_serving = 0
        self.lock = threading.Lock()

    def update(self, active_queue_count, estimated_wait_minutes,
               recent_walkouts, average_service_time_seconds, now_serving):
        with self.lock:
            self.active_queue_count = active_queue_count
            self.estimated_wait_minutes = estimated_wait_minutes
            self.recent_walkouts = recent_walkouts
            self.average_service_time_seconds = average_service_time_seconds
            self.now_serving = now_serving

    def to_dict(self):
        with self.lock:
            return {
                "active_queue_count": self.active_queue_count,
                "estimated_wait_minutes": self.estimated_wait_minutes,
                "recent_walkouts": self.recent_walkouts,
                "average_service_time_seconds": self.average_service_time_seconds,
                "now_serving": self.now_serving,
            }


queue_state = QueueState()


# ---------------------------------------------------------------------------
# WAIT-TIME FORMULA
# ---------------------------------------------------------------------------
def calculate_wait_stats(active_queue_count, service_wait_times, walkout_count):
    """
    Estimated Wait Time = N_queue * T_avg_service

    Uses a rolling window of the most recent service times so the estimate
    adapts if service speeds up or slows down mid-event.
    """
    ROLLING_WINDOW = 10  # only consider the last N completed services

    recent_times = service_wait_times[-ROLLING_WINDOW:] if service_wait_times else []

    if recent_times:
        avg_service_seconds = sum(recent_times) / len(recent_times)
    else:
        avg_service_seconds = 120  # reasonable default guess before any real data exists

    estimated_wait_seconds = active_queue_count * avg_service_seconds
    estimated_wait_minutes = round(estimated_wait_seconds / 60, 1)

    return {
        "estimated_wait_minutes": estimated_wait_minutes,
        "average_service_time_seconds": round(avg_service_seconds, 1),
    }


# ---------------------------------------------------------------------------
# FASTAPI APP (with CORS so the customer web page can call it)
# ---------------------------------------------------------------------------
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/queue-status")
def get_queue_status():
    return queue_state.to_dict()


# ---------------------------------------------------------------------------
# AUTO TICKET NUMBERS - hands out the next sequential token automatically
# ---------------------------------------------------------------------------
next_token_counter = {"value": 0}
token_lock = threading.Lock()


def get_next_token():
    with token_lock:
        next_token_counter["value"] += 1
        return next_token_counter["value"]


# Serve the customer-facing ticket page from the SAME server/port,
# so only one ngrok tunnel is needed for the whole demo.
# Put customer_queue_view.html in the same folder as this script.
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_queue_view.html")


@api.get("/ticket")
def get_ticket_page():
    return FileResponse(HTML_PATH)


@api.get("/new-ticket")
def new_ticket():
    token = get_next_token()
    return RedirectResponse(url=f"/ticket?token={token}")


def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
print("API server running at http://localhost:8000/queue-status")

# ---------------------------------------------------------------------------
# CV SETUP
# ---------------------------------------------------------------------------
model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

queue_zone = sv.PolygonZone(polygon=QUEUE_ZONE_POINTS)
service_zone = sv.PolygonZone(polygon=SERVICE_ZONE_POINTS)

queue_zone_annotator = sv.PolygonZoneAnnotator(
    zone=queue_zone, color=sv.Color.YELLOW, thickness=2, text_thickness=1, text_scale=0.5
)
service_zone_annotator = sv.PolygonZoneAnnotator(
    zone=service_zone, color=sv.Color.GREEN, thickness=2, text_thickness=1, text_scale=0.5
)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera at index {CAMERA_INDEX}. "
        "Check that DroidCam/Iriun is running and connected."
    )

print("Camera opened. Press 'q' in the preview window to quit.")

# ---------------------------------------------------------------------------
# QUEUE / SERVICE STATE TRACKING
# ---------------------------------------------------------------------------
person_state = {}
entry_time = {}
missing_frames = {}
service_wait_times = []
walkout_count = 0
now_serving = 0

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed — check camera connection.")
        break

    results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == PERSON_CLASS_ID]
    detections = tracker.update_with_detections(detections)

    # --- Zone triggers: which detections fall inside each zone this frame ---
    in_queue_mask = queue_zone.trigger(detections=detections)
    in_service_mask = service_zone.trigger(detections=detections)

    ids_in_queue = set(detections.tracker_id[in_queue_mask])
    ids_in_service = set(detections.tracker_id[in_service_mask])

    # --- Handle new queue entries ---
    for tid in ids_in_queue:
        if person_state.get(tid) is None:
            person_state[tid] = "queue"
            entry_time[tid] = time.time()
            missing_frames[tid] = 0
            print(f"[QUEUE] ID {tid} entered queue")

    # --- Handle queue -> service transitions (SERVED) ---
    for tid in ids_in_service:
        if person_state.get(tid) == "queue":
            wait = time.time() - entry_time.get(tid, time.time())
            service_wait_times.append(wait)
            avg_wait = sum(service_wait_times) / len(service_wait_times)
            person_state[tid] = "service"
            missing_frames.pop(tid, None)
            now_serving += 1
            print(f"[SERVED] ID {tid} -> wait time: {wait:.1f}s (rolling avg: {avg_wait:.1f}s)")

    # --- Handle potential walk-outs: was in queue, now missing from BOTH zones ---
    for tid in list(person_state.keys()):
        if person_state[tid] != "queue":
            continue
        still_in_queue = tid in ids_in_queue
        moved_to_service = tid in ids_in_service
        if not still_in_queue and not moved_to_service:
            missing_frames[tid] = missing_frames.get(tid, 0) + 1
            if missing_frames[tid] >= WALKOUT_GRACE_FRAMES:
                walkout_count += 1
                print(f"[WALKOUT] ID {tid} left queue without being served "
                      f"(total walkouts: {walkout_count})")
                person_state.pop(tid, None)
                entry_time.pop(tid, None)
                missing_frames.pop(tid, None)
        else:
            missing_frames[tid] = 0  # reset grace counter if they're back

    # --- Update shared state for the API ---
    active_queue_count = sum(1 for s in person_state.values() if s == "queue")

    wait_stats = calculate_wait_stats(
        active_queue_count=active_queue_count,
        service_wait_times=service_wait_times,
        walkout_count=walkout_count,
    )

    queue_state.update(
        active_queue_count=active_queue_count,
        estimated_wait_minutes=wait_stats["estimated_wait_minutes"],
        recent_walkouts=walkout_count,
        average_service_time_seconds=wait_stats["average_service_time_seconds"],
        now_serving=now_serving,
    )

    # --- Draw overlay ---
    labels = [
        f"#{tracker_id} {confidence:.2f}"
        for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
    ]

    annotated_frame = queue_zone_annotator.annotate(scene=frame.copy())
    annotated_frame = service_zone_annotator.annotate(scene=annotated_frame)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    cv2.putText(
        annotated_frame,
        f"In queue: {active_queue_count}  Walkouts: {walkout_count}  Now serving: {now_serving}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Hour 2 - Zones & Walk-outs", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
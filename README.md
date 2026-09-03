# Live Queue Tracker — Camera-Based Wait Time & Token System

A single-camera computer vision system that tracks how many people are waiting, estimates live wait time, and serves a "Now Serving" token number to customers over a web link — no app install, no per-customer check-in.

Built for walk-in-heavy, low-infrastructure environments like clinics, salons, and canteens where people currently wait 30–40 minutes with zero visibility into their position in line.

---

## Problem

Clinics, salons, and canteens run queues with no live wait information. Customers either stand around the whole time or leave without knowing if they'll be back before their turn. Staff have no easy way to communicate wait times without constant manual updates.

## Approach

One camera watches two zones — a **queue zone** and a **service zone**. People are detected and tracked frame-to-frame; when someone moves from the queue zone into the service zone, that's counted as "served." No manual clicking, no per-person registration, no app required on the customer side.

- **Queue zone** → people currently waiting
- **Service zone** → person currently being served
- **Token counter** (`now_serving`) increments every time someone completes that transition
- A FastAPI server exposes this live state as JSON, and serves a customer-facing web page from the same port

---

## Architecture

```
Camera Feed
    │
    ▼
YOLOv8 (person detection)
    │
    ▼
ByteTrack (persistent IDs across frames)
    │
    ▼
Zone Triggers (queue zone / service zone)
    │
    ▼
State Machine (queue → service → walkout)
    │
    ▼
QueueState (thread-safe shared object)
    │
    ├──► CV preview window (debug overlay)
    │
    └──► FastAPI server (background thread)
              │
              ├── GET /queue-status  → JSON: live counts + wait estimate
              └── GET /ticket        → customer-facing HTML page
```

The CV loop and the API server run as two threads sharing one `QueueState` object, protected by a lock so reads and writes never collide.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Object detection | YOLOv8 (`yolov8n.pt`, Ultralytics) |
| Multi-object tracking | ByteTrack (via `supervision`) |
| Zone detection | `supervision.PolygonZone` |
| Video capture / display | OpenCV |
| Backend API | FastAPI + Uvicorn |
| Cross-origin access | `CORSMiddleware` |
| Public demo access | ngrok (one tunnel, since API + customer page share one port) |

---

## Wait Time Formula

```
avg_service_seconds = mean(last 10 completed service durations)   # rolling window
estimated_wait_seconds = active_queue_count × avg_service_seconds
```

The rolling window (not a full-day average) means the estimate adapts within a handful of services if things speed up or slow down — a slow morning doesn't get diluted by fast data from last week.

> **Current limitation:** this formula assumes a single server (one counter / one doctor / one chair). See [Known Limitations](#known-limitations) below.

---

## Getting Started

### Requirements

```bash
pip install ultralytics supervision opencv-python fastapi uvicorn numpy
```

### Run

```bash
python queue_tracker.py
```

- Opens the default camera (`CAMERA_INDEX = 0`) — for phone-as-webcam setups, connect via DroidCam or Iriun first.
- Console prints `API server running at http://localhost:8000/queue-status`.
- A preview window opens showing the live camera feed with zone outlines, bounding boxes, and a running count overlay. Press **`q`** to quit.

### API Endpoints

| Endpoint | Returns |
|---|---|
| `GET /queue-status` | JSON: `active_queue_count`, `estimated_wait_minutes`, `recent_walkouts`, `average_service_time_seconds`, `now_serving` |
| `GET /ticket` | Customer-facing HTML page (`customer_queue_view.html`, must sit in the same folder as the script) |

Example response from `/queue-status`:
```json
{
  "active_queue_count": 5,
  "estimated_wait_minutes": 12.4,
  "recent_walkouts": 1,
  "average_service_time_seconds": 148.6,
  "now_serving": 32
}
```

### Demo Setup (single ngrok tunnel)

```bash
ngrok http 8000
```
Share the resulting URL + `/ticket` with customers — one tunnel covers both the live data API and the customer page, since they're served from the same FastAPI app.

---

## Configuration

All tunable values sit at the top of the script:

| Variable | Meaning |
|---|---|
| `CAMERA_INDEX` | Which camera device to open |
| `CONF_THRESHOLD` | YOLO detection confidence cutoff (0.4 default) |
| `QUEUE_ZONE_POINTS` / `SERVICE_ZONE_POINTS` | Pixel-coordinate polygons defining each zone — must be recalibrated per camera position/resolution |
| `WALKOUT_GRACE_FRAMES` | Consecutive missed frames before a queued person is marked a walkout |

---

## Known Limitations

These are open issues, not yet fixed in the current version:

- **Frame-based walkout timer** — `WALKOUT_GRACE_FRAMES` should be wall-clock time, not frame count, since walkout timing currently drifts with hardware inference speed (FPS varies by machine).
- **No occlusion re-linking** — ByteTrack assigns a new ID when a tracked person is briefly blocked from view (e.g. someone else walking past). This can cause a false walkout followed by a false new queue entry for the same person, silently inflating both `active_queue_count` drift and `recent_walkouts`.
- **Single-server wait formula** — `active_queue_count × avg_service_seconds` assumes one queue feeding one counter/doctor/chair. Multi-server locations (multiple stylists, multiple doctors) will get an inflated wait estimate. Needs per-server tracking and division by active server count.
- **Hardcoded zone coordinates** — `QUEUE_ZONE_POINTS` and `SERVICE_ZONE_POINTS` are pixel positions tied to one exact camera framing. Moving the camera or using a different resolution requires manual recalibration.
- **Open CORS policy** — `allow_origins=["*"]` is fine for a demo but should be scoped to the actual customer-facing domain before any real deployment.
- **No manual resync** — if the camera undercounts or overcounts (bad lighting, two people passing as one blob), there's currently no way for staff to manually correct `now_serving` or `active_queue_count` without restarting the script.

## Roadmap

- [ ] Switch walkout grace period to wall-clock seconds
- [ ] Add short-window ID re-linking to survive brief occlusion
- [ ] Support multiple service zones for multi-server locations
- [ ] Add a staff-side manual correction endpoint (resync counts in 2 seconds)
- [ ] SMS fallback for the customer ticket link (no-smartphone / low-data users)
- [ ] Confidence range display ("15–22 min") instead of a single point estimate

---

## Project Structure

```
.
├── queue_tracker.py          # CV loop + FastAPI server (this file)
├── customer_queue_view.html  # customer-facing ticket page (served at /ticket)
└── README.md
```

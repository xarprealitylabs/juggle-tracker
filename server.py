import asyncio
import json
import logging
import os
import tempfile
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

BALL_CLASS  = 32
CONF_THRESH = 0.12  # lower threshold catches ball near foot (occlusion)

# Single model, serialised across threads with a lock
_model = YOLO("yolov8n.pt")
_model_lock = threading.Lock()


def run_inference(frame: np.ndarray):
    with _model_lock:
        return _model(frame, classes=[BALL_CLASS], conf=CONF_THRESH, verbose=False)


class KalmanBallTracker:
    """Constant-velocity Kalman filter for ball position."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov   = 1e-2 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
        self.initialized = False

    def step(self) -> tuple[float, float]:
        """Advance state by one frame. Call once per frame regardless of detection."""
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, x: float, y: float) -> tuple[float, float]:
        """Update with a real measurement. Call after step() when ball is detected."""
        if not self.initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(np.array([[x],[y]], np.float32))
        post = self.kf.statePost
        return float(post[0]), float(post[1])


class JuggleSession:
    WINDOW        = 30    # rolling window in frames
    MIN_DELTA_PCT = 0.010 # tuned against ground truth: 22 juggles in 15s test video
    COOLDOWN      = 0.20  # 200ms min between juggles (handles up to ~5/s)

    def __init__(self):
        self.tracker = KalmanBallTracker()
        # Store normalised (0-1) coordinates so MIN_DELTA_PCT works across resolutions
        self.history: list[tuple[float, float, float]] = []  # (x_norm, y_norm, t)
        self.count = 0
        self.last_juggle_t = 0.0
        self.frames_no_ball = 0

    def process_frame(self, frame: np.ndarray, t: float | None = None) -> dict:
        if t is None:
            t = time.monotonic()
        h, w = frame.shape[:2]

        pred_x, pred_y = self.tracker.step()

        results = run_inference(frame)

        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            raw_x = (box[0] + box[2]) / 2
            raw_y = (box[1] + box[3]) / 2
            bx, by = self.tracker.correct(raw_x, raw_y)
            self.frames_no_ball = 0
        else:
            self.frames_no_ball += 1
            if self.frames_no_ball < 10 and self.tracker.initialized:
                bx, by = pred_x, pred_y
            else:
                bx, by = None, None

        if bx is not None:
            # Store normalised so detection thresholds are resolution-independent
            self.history.append((bx / w, by / h, t))
            if len(self.history) > self.WINDOW:
                self.history.pop(0)
            self._detect_juggle()

        return {
            "bx": round(bx / w, 4) if bx is not None else None,
            "by": round(by / h, 4) if by is not None else None,
            "count": self.count,
        }

    # Max gap between consecutive history points to trust for adjacent comparison
    MAX_ADJ_GAP = 0.15  # 150ms

    def _detect_juggle(self):
        """Two complementary signals:
        1. Adjacent local max: ball Y peaks between two nearby-in-time points.
        2. Gap reversal: ball descending before a gap, ascending after = contact during gap.
        """
        n = len(self.history)
        if n < 3:
            return

        ys = [p[1] for p in self.history]
        ts = [p[2] for p in self.history]
        i = n - 2

        gap_before = ts[i] - ts[i - 1]
        gap_after  = ts[i + 1] - ts[i]

        # Signal 1: adjacent local max (direct contact detection, no gap)
        if gap_before <= self.MAX_ADJ_GAP and gap_after <= self.MAX_ADJ_GAP:
            if (ys[i] - ys[i-1] > self.MIN_DELTA_PCT and
                    ys[i] - ys[i+1] > self.MIN_DELTA_PCT):
                if ts[i] - self.last_juggle_t >= self.COOLDOWN:
                    self.count += 1
                    self.last_juggle_t = ts[i]
                    return

        # Signal 2: direction reversal across a detection gap
        # Gap between 80ms and 600ms suggests ball was briefly undetected at contact
        if 0.08 < gap_before < 0.60 and n >= 4:
            prev_gap = ts[i - 1] - ts[i - 2]
            if prev_gap <= self.MAX_ADJ_GAP and gap_after <= self.MAX_ADJ_GAP:
                vel_before = ys[i - 1] - ys[i - 2]  # positive = ball descending
                vel_after  = ys[i + 1] - ys[i]       # negative = ball ascending
                if vel_before > self.MIN_DELTA_PCT and vel_after < -self.MIN_DELTA_PCT:
                    t_contact = (ts[i - 1] + ts[i]) / 2
                    if t_contact - self.last_juggle_t >= self.COOLDOWN:
                        self.count += 1
                        self.last_juggle_t = t_contact


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = JuggleSession()
    loop = asyncio.get_running_loop()
    log.info("WS connected")
    try:
        while True:
            msg = await ws.receive()

            # Clean disconnect
            if msg.get("type") == "websocket.disconnect":
                break

            # Text messages are control signals
            if msg.get("text") == "RESET":
                session = JuggleSession()
                log.info("Session reset for Player 2")
                continue

            data = msg.get("bytes")
            if not data:
                continue

            img_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = await loop.run_in_executor(None, session.process_frame, frame)
            await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as e:
        log.exception("WS error: %s", e)


MAX_DURATION_S = 30


def _annotate_video_sync(in_path: str, out_path: str) -> int:
    """Returns final juggle count."""
    t0 = time.monotonic()
    cap = cv2.VideoCapture(in_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / fps

    log.info("annotate: %.1fs video — %d frames @ %.1ffps (%dx%d)",
             duration_s, n_frames, fps, w, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    font = cv2.FONT_HERSHEY_DUPLEX

    session = JuggleSession()
    frame_idx = 0
    last_result: dict = {"bx": None, "by": None, "count": 0}
    STEP = 2  # process every Nth frame — halves inference time

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if frame_idx % STEP == 0:
            last_result = session.process_frame(frame, t=t)
        result = last_result
        if result["bx"] is not None:
            cx = int(result["bx"] * w)
            cy = int(result["by"] * h)
            cv2.circle(frame, (cx, cy), 28, (74, 222, 128), 3)
            cv2.circle(frame, (cx, cy), 5,  (74, 222, 128), -1)
        cv2.putText(frame, str(result["count"]), (24, 80), font, 3,
                    (74, 222, 128), 4, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = int(frame_idx / max(n_frames, 1) * 100)
            log.info("annotate: %d%% (%d/%d frames)", pct, frame_idx, n_frames)

    cap.release()
    writer.release()
    os.unlink(in_path)

    elapsed = time.monotonic() - t0
    final_count = result["count"]
    log.info("annotate: done in %.1fs — %d juggles detected", elapsed, final_count)
    return final_count


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    raw = await file.read()
    size_mb = len(raw) / 1_000_000
    log.info("annotate: received upload %.1f MB (%s)", size_mb, file.filename)

    # Write to temp file so OpenCV can open it
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        in_path = f.name

    # Check duration before running expensive inference
    cap = cv2.VideoCapture(in_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = n_frames / fps

    if duration_s > MAX_DURATION_S:
        os.unlink(in_path)
        log.warning("annotate: rejected — video too long (%.1fs > %ds)", duration_s, MAX_DURATION_S)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Vídeo demasiado longo ({duration_s:.0f}s). Máximo: {MAX_DURATION_S}s."
        )

    out_path = in_path[:-len(suffix)] + "_annotated.mp4"

    loop = asyncio.get_running_loop()
    final_count = await loop.run_in_executor(None, _annotate_video_sync, in_path, out_path)

    response = FileResponse(
        out_path,
        media_type="video/mp4",
        filename="juggle_annotated.mp4",
        background=BackgroundTask(os.unlink, out_path),
    )
    response.headers["X-Juggle-Count"] = str(final_count)
    return response


app.mount("/", StaticFiles(directory="static", html=True), name="static")

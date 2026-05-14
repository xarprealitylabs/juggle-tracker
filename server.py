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

BALL_CLASS = 32
CONF_THRESH = 0.30

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
    WINDOW    = 24     # rolling window in frames
    MIN_DELTA = 12     # minimum Y displacement (px) to register direction change
    COOLDOWN  = 0.45   # minimum seconds between counted juggles

    def __init__(self):
        self.tracker = KalmanBallTracker()
        self.history: list[tuple[float, float, float]] = []  # (x, y, t)
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
            if self.frames_no_ball < 6 and self.tracker.initialized:
                bx, by = pred_x, pred_y  # coasted prediction — no double-predict
            else:
                bx, by = None, None

        if bx is not None:
            self.history.append((bx, by, t))
            if len(self.history) > self.WINDOW:
                self.history.pop(0)
            self._detect_juggle(t)

        return {
            "bx": round(bx / w, 4) if bx is not None else None,
            "by": round(by / h, 4) if by is not None else None,
            "count": self.count,
        }

    def _detect_juggle(self, now: float):
        if now - self.last_juggle_t < self.COOLDOWN:
            return
        n = len(self.history)
        if n < 10:
            return
        ys = [p[1] for p in self.history]
        mid = n // 2
        going_down = ys[mid] - ys[0]
        going_up   = ys[-1] - ys[mid]
        if going_down > self.MIN_DELTA and going_up < -self.MIN_DELTA:
            self.count += 1
            self.last_juggle_t = now


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = JuggleSession()
    loop = asyncio.get_running_loop()
    log.info("WS connected")
    try:
        while True:
            msg = await ws.receive()

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


def _annotate_video_sync(in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    font = cv2.FONT_HERSHEY_DUPLEX

    session = JuggleSession()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps  # use video time, not wall clock
        result = session.process_frame(frame, t=t)
        if result["bx"] is not None:
            cx = int(result["bx"] * w)
            cy = int(result["by"] * h)
            cv2.circle(frame, (cx, cy), 28, (74, 222, 128), 3)
            cv2.circle(frame, (cx, cy), 5,  (74, 222, 128), -1)
        cv2.putText(frame, str(result["count"]), (24, 80), font, 3,
                    (74, 222, 128), 4, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    os.unlink(in_path)


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    raw = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw)
        in_path = f.name

    out_path = in_path[:-5] + "_annotated.mp4"

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _annotate_video_sync, in_path, out_path)

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="juggle_annotated.mp4",
        background=BackgroundTask(os.unlink, out_path),
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")

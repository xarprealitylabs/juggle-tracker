import asyncio
import json
import time
import io
import tempfile
import os

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# YOLOv8n — sports ball is COCO class 32
model = YOLO("yolov8n.pt")
BALL_CLASS = 32
CONF_THRESH = 0.30


class KalmanBallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov   = 1e-2 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
        self.initialized = False

    def update(self, x: float, y: float):
        meas = np.array([[x], [y]], np.float32)
        if not self.initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(meas)
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])


class JuggleSession:
    WINDOW = 24          # frames in rolling window (~1.6s at 15fps)
    MIN_DELTA = 12       # px minimum direction change to count
    COOLDOWN_S = 0.45    # minimum seconds between juggles

    def __init__(self):
        self.tracker = KalmanBallTracker()
        self.history: list[tuple[float, float, float]] = []  # (x, y, t)
        self.count = 0
        self.last_juggle_t = 0.0
        self.frames_since_ball = 0

    def process_frame(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        t = time.monotonic()

        results = model(frame, classes=[BALL_CLASS], conf=CONF_THRESH, verbose=False)

        bx, by = None, None
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            raw_x = (box[0] + box[2]) / 2
            raw_y = (box[1] + box[3]) / 2
            sx, sy = self.tracker.update(raw_x, raw_y)
            bx, by = sx, sy
            self.frames_since_ball = 0
        else:
            self.frames_since_ball += 1
            if self.frames_since_ball < 6 and self.tracker.initialized:
                bx, by = self.tracker.predict()

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
        if now - self.last_juggle_t < self.COOLDOWN_S:
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
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            img_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            result = await loop.run_in_executor(None, session.process_frame, frame)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    """
    Receives a raw video upload, re-runs detections on the full trajectory
    (no re-inference — uses stored detections from the session would be ideal,
    but for simplicity here we do a fast re-pass on the uploaded video).
    Returns an annotated MP4.
    """
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw)
        in_path = f.name

    out_path = in_path.replace(".webm", "_annotated.mp4")

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    session = JuggleSession()
    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = session.process_frame(frame)
        if result["bx"] is not None:
            cx = int(result["bx"] * w)
            cy = int(result["by"] * h)
            cv2.circle(frame, (cx, cy), 28, (74, 222, 128), 3)
            cv2.circle(frame, (cx, cy), 5,  (74, 222, 128), -1)
        cv2.putText(frame, str(result["count"]), (24, 80), font, 3,
                    (74, 222, 128), 4, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    os.unlink(in_path)

    return FileResponse(out_path, media_type="video/mp4",
                        filename="juggle_annotated.mp4",
                        background=None)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

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
CONF_THRESH = 0.10
ANKLE_CONF  = 0.40  # minimum keypoint confidence to trust ankle position
LEFT_ANKLE  = 15    # COCO keypoint index
RIGHT_ANKLE = 16

# Both models serialised under one lock — prevents concurrent YOLO calls across threads
_ball_model = YOLO("yolov8n.pt")
_pose_model = YOLO("yolov8n-pose.pt")
_model_lock = threading.Lock()


def run_inference(frame: np.ndarray):
    with _model_lock:
        ball = _ball_model(frame, classes=[BALL_CLASS], conf=CONF_THRESH, verbose=False)
        pose = _pose_model(frame, verbose=False)
    return ball, pose


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
    WINDOW     = 30    # rolling window in positions
    SPAN       = 5     # compare position i against i±SPAN
    MIN_PROM   = 0.06  # ball: 6% of frame height prominence
    ANKLE_PROM = 0.06  # ankle: 6% of frame height — kick arc is larger so conservative
    COOLDOWN   = 0.20  # 200ms between juggles (supports ~5/s)

    def __init__(self):
        self.tracker = KalmanBallTracker()
        self.history: list[tuple[float, float, float]] = []   # (x_norm, y_norm, t)
        self.ankle_history: list[tuple[float, float]] = []    # (y_norm, t)
        self.count = 0
        self.last_juggle_t = 0.0
        self.frames_no_ball = 0

    def process_frame(self, frame: np.ndarray, t: float | None = None) -> dict:
        if t is None:
            t = time.monotonic()
        h, w = frame.shape[:2]

        pred_x, pred_y = self.tracker.step()
        ball_results, pose_results = run_inference(frame)

        # ── Ball signal ────────────────────────────────────────────────────────
        bx = by = None
        if ball_results and len(ball_results[0].boxes) > 0:
            box = ball_results[0].boxes[0].xyxy[0].cpu().numpy()
            raw_x = (box[0] + box[2]) / 2
            raw_y = (box[1] + box[3]) / 2
            bx, by = self.tracker.correct(raw_x, raw_y)
            self.frames_no_ball = 0
        else:
            self.frames_no_ball += 1
            if self.frames_no_ball < 10 and self.tracker.initialized:
                bx, by = pred_x, pred_y

        if bx is not None:
            self.history.append((bx / w, by / h, t))
            if len(self.history) > self.WINDOW:
                self.history.pop(0)

        # ── Pose signal — ankle tracking ───────────────────────────────────────
        if pose_results and len(pose_results[0].keypoints) > 0:
            try:
                kpts = pose_results[0].keypoints
                xy   = kpts.xy[0].cpu().numpy()
                conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else None
                candidates = []
                for idx in (LEFT_ANKLE, RIGHT_ANKLE):
                    if idx < len(xy):
                        ak_x, ak_y = xy[idx]
                        ak_conf = float(conf[idx]) if conf is not None else 1.0
                        if ak_conf >= ANKLE_CONF and ak_y > 0:
                            candidates.append(ak_y / h)
                if candidates:
                    # Track highest ankle (smallest y_norm = foot most upward = kicking foot)
                    self.ankle_history.append((min(candidates), t))
                    if len(self.ankle_history) > self.WINDOW:
                        self.ankle_history.pop(0)
            except Exception:
                pass

        self._detect_juggle()

        return {
            "bx": round(bx / w, 4) if bx is not None else None,
            "by": round(by / h, 4) if by is not None else None,
            "count": self.count,
        }

    def _detect_juggle(self):
        """Two independent signals, shared cooldown prevents double-counting.

        Signal 1 (ball): span-based local Y-max — ball at foot = max image Y.
        Signal 2 (ankle): span-based local Y-min — foot at kick apex = min image Y.
        """
        n = len(self.history)
        if n >= 2 * self.SPAN + 1:
            i = n - self.SPAN - 1
            _, y_i,      t_i      = self.history[i]
            _, y_before, t_before = self.history[i - self.SPAN]
            _, y_after,  _        = self.history[i + self.SPAN]
            if t_i - t_before <= 1.0 and self.history[-1][2] - t_i <= 1.0:
                if y_i - y_before > self.MIN_PROM and y_i - y_after > self.MIN_PROM:
                    if t_i - self.last_juggle_t >= self.COOLDOWN:
                        self.count += 1
                        self.last_juggle_t = t_i
                        return

        m = len(self.ankle_history)
        if m >= 2 * self.SPAN + 1:
            j = m - self.SPAN - 1
            y_j,      t_j      = self.ankle_history[j]
            y_before, t_before = self.ankle_history[j - self.SPAN]
            y_after,  _        = self.ankle_history[j + self.SPAN]
            if t_j - t_before <= 1.0 and self.ankle_history[-1][1] - t_j <= 1.0:
                # Local MIN in ankle Y = foot at highest point = kick
                if y_before - y_j > self.ANKLE_PROM and y_after - y_j > self.ANKLE_PROM:
                    if t_j - self.last_juggle_t >= self.COOLDOWN:
                        self.count += 1
                        self.last_juggle_t = t_j


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

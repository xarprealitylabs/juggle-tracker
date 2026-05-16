import asyncio
import json
import logging
import os
import subprocess
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
    SPAN       = 5     # fallback arc detection span
    MIN_PROM   = 0.06  # fallback arc prominence threshold
    COOLDOWN   = 0.15  # 150ms hard minimum between juggles (~6/s max, above pro level)

    def __init__(self):
        self.tracker = KalmanBallTracker()
        self.history: list[tuple[float, float, float]] = []   # (x_norm, y_norm, t)
        self.ankle_history: list[tuple[float, float]] = []    # (y_norm, t)
        self.count = 0
        self.last_juggle_t = 0.0
        self.frames_no_ball = 0
        self.in_proximity = False   # leading-edge: True while ball is near ankle
        self._last_pose = None
        self._last_frame_shape = (480, 640)

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

        # Store last pose result for debug overlay (raw pixel coords)
        self._last_pose = pose_results
        self._last_frame_shape = (h, w)

        return {
            "bx": round(bx / w, 4) if bx is not None else None,
            "by": round(by / h, 4) if by is not None else None,
            "count": self.count,
        }

    PROXIMITY_THRESH = 0.18  # ball within 18% of frame height from ankle = contact zone

    def _detect_juggle(self):
        """JuggleNet-inspired detection: proximity leading-edge + arc fallback.

        Primary: fires once when ball ENTERS the ankle proximity zone (leading edge).
           One physical touch = one entry event = one count, regardless of how many
           frames the ball stays near the foot.
        Fallback: span-based arc when no ankle detected recently.
        """
        if not self.history:
            return

        t_now = self.history[-1][2]

        # ── Primary: leading-edge proximity detection ─────────────────────────
        currently_in_prox = False
        if self.ankle_history and t_now - self.ankle_history[-1][1] < 0.20:
            ankle_y = self.ankle_history[-1][0]
            _, by_norm, _ = self.history[-1]
            currently_in_prox = abs(by_norm - ankle_y) < self.PROXIMITY_THRESH

        if currently_in_prox and not self.in_proximity:
            # Leading edge: first frame ball enters contact zone.
            # Guard: ball must NOT be significantly below the ankle.
            # In image coords Y increases downward, so ball on ground has higher Y
            # than ankle. A real keepup contact has ball_y ≈ ankle_y.
            ankle_y = self.ankle_history[-1][0]
            _, by_norm, _ = self.history[-1]
            ball_not_on_ground = by_norm <= ankle_y + 0.06  # 6% tolerance below ankle
            if ball_not_on_ground and t_now - self.last_juggle_t >= self.COOLDOWN:
                self.count += 1
                self.last_juggle_t = t_now

        self.in_proximity = currently_in_prox

        if currently_in_prox:
            return  # proximity signal active — don't also run arc fallback

        # ── Fallback: span-based arc detection (no pose available) ───────────
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

_ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _video_rotation(path: str) -> int:
    """Return rotation degrees from video metadata (0 if none or ffprobe unavailable)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        for stream in data.get("streams", []):
            deg = int(stream.get("tags", {}).get("rotate", 0))
            if deg:
                return deg
    except Exception as e:
        log.debug("ffprobe rotation check skipped: %s", e)
    return 0


def _annotate_video_sync(in_path: str, out_path: str, debug: bool = False) -> int:
    """Returns final juggle count. If debug=True, draws full detection overlay."""
    t0 = time.monotonic()
    cap = cv2.VideoCapture(in_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / fps

    # Detect and apply rotation (common for portrait phone videos)
    rotation   = _video_rotation(in_path)
    rotate_code = _ROTATE_MAP.get(rotation)
    if rotate_code is not None and rotation in (90, 270, -90):
        w, h = h, w   # dimensions swap for 90/270° rotations
        log.info("annotate: rotating %d° → output %dx%d", rotation, w, h)

    log.info("annotate%s: %.1fs video — %d frames @ %.1ffps (%dx%d)",
             "[debug]" if debug else "", duration_s, n_frames, fps, w, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    font   = cv2.FONT_HERSHEY_SIMPLEX

    INFER_MAX = 640
    scale   = min(INFER_MAX / w, INFER_MAX / h, 1.0)
    infer_w = int(w * scale)
    infer_h = int(h * scale)

    session   = JuggleSession()
    frame_idx = 0
    last_result: dict = {"bx": None, "by": None, "count": 0}
    prev_count = 0
    counted_flash = 0  # frames remaining to show COUNTED flash
    STEP = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)
        if frame_idx % STEP == 0:
            infer_frame = cv2.resize(frame, (infer_w, infer_h)) if scale < 1.0 else frame
            last_result = session.process_frame(infer_frame, t=t)
        result = last_result

        # ── Standard overlay (always) ──────────────────────────────────────
        if result["bx"] is not None:
            cx = int(result["bx"] * w)
            cy = int(result["by"] * h)
            cv2.circle(frame, (cx, cy), 28, (74, 222, 128), 3)
            cv2.circle(frame, (cx, cy), 5,  (74, 222, 128), -1)

        cv2.putText(frame, str(result["count"]), (24, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (74, 222, 128), 4, cv2.LINE_AA)

        # ── Debug overlay ──────────────────────────────────────────────────
        if debug:
            # Draw ALL detected pose keypoints from the last inference
            if session._last_pose and len(session._last_pose[0].keypoints) > 0:
                try:
                    kpts = session._last_pose[0].keypoints
                    ih, iw = session._last_frame_shape
                    # Scale from inference coords back to output coords
                    sx, sy = w / iw, h / ih
                    xy   = kpts.xy[0].cpu().numpy()
                    conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else None

                    # Named keypoints we care about: knees and ankles
                    KP_NAMES = {13: "Lknee", 14: "Rknee", 15: "Lank", 16: "Rank"}
                    for idx, kp_name in KP_NAMES.items():
                        if idx >= len(xy):
                            continue
                        kx, ky = xy[idx]
                        kc = float(conf[idx]) if conf is not None else 1.0
                        if kc < 0.2 or (kx == 0 and ky == 0):
                            continue
                        px, py = int(kx * sx), int(ky * sy)
                        is_ankle = idx in (15, 16)
                        color = (0, 215, 255) if is_ankle else (180, 180, 0)
                        shape = 8 if is_ankle else 5
                        cv2.circle(frame, (px, py), shape, color, -1)
                        cv2.putText(frame, f"{kp_name}({kc:.2f})",
                                    (px + 6, py), font, 0.4, color, 1)
                except Exception:
                    pass

            # Ankle horizontal line (used for detection)
            if session.ankle_history:
                ankle_y_norm = session.ankle_history[-1][0]
                ay = int(ankle_y_norm * h)
                cv2.line(frame, (0, ay), (w, ay), (0, 215, 255), 1)
                cv2.putText(frame, f"ankle_y={ankle_y_norm:.3f}", (4, ay - 6),
                            font, 0.4, (0, 215, 255), 1)

                # Ball-to-ankle distance line
                if result["bx"] is not None:
                    bx_px = int(result["bx"] * w)
                    by_px = int(result["by"] * h)
                    dist  = abs(result["by"] - ankle_y_norm)
                    in_zone = dist < JuggleSession.PROXIMITY_THRESH
                    lcolor  = (0, 255, 0) if in_zone else (60, 60, 255)
                    cv2.line(frame, (bx_px, by_px), (bx_px, ay), lcolor, 2)
                    cv2.putText(frame, f"d={dist:.3f}{'  IN' if in_zone else ''}",
                                (bx_px + 6, (by_px + ay) // 2), font, 0.4, lcolor, 1)

            # Frame info
            cv2.putText(frame, f"f{frame_idx} t={t:.2f}s", (4, h - 10),
                        font, 0.4, (200, 200, 200), 1)

            # COUNTED flash
            if result["count"] > prev_count:
                counted_flash = int(fps * 0.4)  # flash for 400ms
            if counted_flash > 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 8)
                cv2.putText(frame, f"COUNTED #{result['count']}",
                            (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX,
                            1.8, (0, 255, 0), 3, cv2.LINE_AA)
                counted_flash -= 1

        prev_count = result["count"]
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = int(frame_idx / max(n_frames, 1) * 100)
            log.info("annotate: %d%% (%d/%d frames)", pct, frame_idx, n_frames)

    cap.release()
    writer.release()
    os.unlink(in_path)

    elapsed     = time.monotonic() - t0
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
    final_count = await loop.run_in_executor(
        None, _annotate_video_sync, in_path, out_path, False
    )

    response = FileResponse(
        out_path,
        media_type="video/mp4",
        filename="juggle_annotated.mp4",
        background=BackgroundTask(os.unlink, out_path),
    )
    response.headers["X-Juggle-Count"] = str(final_count)
    return response


@app.post("/annotate_debug")
async def annotate_video_debug(file: UploadFile = File(...)):
    """Same as /annotate but with full debug overlay:
    - Ankle horizontal line (yellow)
    - Ball-to-ankle distance line (green = in zone, red = out)
    - 'COUNTED' flash on detection event
    - Frame number, timestamp, distance values
    """
    raw = await file.read()
    size_mb = len(raw) / 1_000_000
    log.info("annotate_debug: %.1f MB (%s)", size_mb, file.filename)

    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        in_path = f.name

    cap = cv2.VideoCapture(in_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = n_frames / fps

    if duration_s > MAX_DURATION_S:
        os.unlink(in_path)
        from fastapi import HTTPException
        raise HTTPException(status_code=422,
            detail=f"Vídeo demasiado longo ({duration_s:.0f}s). Máximo: {MAX_DURATION_S}s.")

    out_path = in_path[:-len(suffix)] + "_debug.mp4"

    loop = asyncio.get_running_loop()
    final_count = await loop.run_in_executor(
        None, _annotate_video_sync, in_path, out_path, True
    )

    response = FileResponse(
        out_path,
        media_type="video/mp4",
        filename="juggle_debug.mp4",
        background=BackgroundTask(os.unlink, out_path),
    )
    response.headers["X-Juggle-Count"] = str(final_count)
    return response


# COCO skeleton edges for drawing connections between keypoints
_SKELETON_EDGES = [
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),           # shoulders
    (5, 11), (6, 12), # torso sides
    (11, 12),         # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]
_KP_COLORS = {
    15: (0, 215, 255),   # left ankle — yellow
    16: (0, 215, 255),   # right ankle — yellow
    13: (100, 200, 0),   # left knee — green
    14: (100, 200, 0),   # right knee — green
}


def _pose_only_sync(in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotation    = _video_rotation(in_path)
    rotate_code = _ROTATE_MAP.get(rotation)
    if rotate_code is not None and rotation in (90, 270, -90):
        w, h = h, w
    log.info("pose_only: %dx%d rotation=%d", w, h, rotation)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        with _model_lock:
            results = _pose_model(frame, verbose=False)

        if results and len(results[0].keypoints) > 0:
            try:
                xy   = results[0].keypoints.xy[0].cpu().numpy()    # [17, 2]
                conf = results[0].keypoints.conf[0].cpu().numpy() \
                       if results[0].keypoints.conf is not None else None

                # Draw skeleton edges
                for i, j in _SKELETON_EDGES:
                    if i >= len(xy) or j >= len(xy):
                        continue
                    ci = float(conf[i]) if conf is not None else 1.0
                    cj = float(conf[j]) if conf is not None else 1.0
                    if ci < 0.3 or cj < 0.3:
                        continue
                    pi = (int(xy[i][0]), int(xy[i][1]))
                    pj = (int(xy[j][0]), int(xy[j][1]))
                    if pi == (0, 0) or pj == (0, 0):
                        continue
                    cv2.line(frame, pi, pj, (180, 180, 180), 2)

                # Draw keypoints
                KP_LABELS = {
                    5: "Lsh", 6: "Rsh", 7: "Lelb", 8: "Relb",
                    9: "Lwri", 10: "Rwri",
                    11: "Lhip", 12: "Rhip",
                    13: "Lkne", 14: "Rkne",
                    15: "Lank", 16: "Rank",
                }
                for idx, label in KP_LABELS.items():
                    if idx >= len(xy):
                        continue
                    kc = float(conf[idx]) if conf is not None else 1.0
                    kx, ky = int(xy[idx][0]), int(xy[idx][1])
                    if kc < 0.2 or (kx == 0 and ky == 0):
                        continue
                    color = _KP_COLORS.get(idx, (220, 220, 220))
                    radius = 8 if idx in (15, 16) else 5
                    cv2.circle(frame, (kx, ky), radius, color, -1)
                    cv2.putText(frame, f"{label}:{kc:.2f}", (kx + 5, ky - 4),
                                font, 0.4, color, 1)
            except Exception as e:
                log.debug("pose draw error: %s", e)

        cv2.putText(frame, f"f{frame_idx}", (6, 22), font, 0.5, (255, 255, 255), 1)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    os.unlink(in_path)
    log.info("pose_only: done — %d frames", frame_idx)


@app.post("/pose_only")
async def pose_only(file: UploadFile = File(...)):
    """Diagnostic: rotate + draw full skeleton only. No ball, no counting."""
    raw = await file.read()
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        in_path = f.name

    cap = cv2.VideoCapture(in_path)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if n / fps > MAX_DURATION_S:
        os.unlink(in_path)
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=f"Máximo {MAX_DURATION_S}s.")

    out_path = in_path[:-len(suffix)] + "_pose.mp4"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _pose_only_sync, in_path, out_path)

    return FileResponse(out_path, media_type="video/mp4", filename="pose_skeleton.mp4",
                        background=BackgroundTask(os.unlink, out_path))


app.mount("/", StaticFiles(directory="static", html=True), name="static")

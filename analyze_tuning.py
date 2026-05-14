"""
Juggle detection tuning script.
Runs on VM via Docker. Tries multiple approaches against ground truth.
Generates annotated video segments to visualise failures.
Outputs results JSON so the tuning cycle can iterate.

Usage:
  python3 analyze_tuning.py <video_path> <ground_truth_count> [output_dir]
"""
import sys
import os
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO

BALL_CLASS = 32
GT_COUNT   = int(sys.argv[2]) if len(sys.argv) > 2 else 22
OUT_DIR    = sys.argv[3] if len(sys.argv) > 3 else "/data/juggle_segments"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/data/test_video.mp4"
RESULTS_PATH = "/data/juggle_tuning_results.json"

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO("yolov8n.pt")


# ─── Detection helper ─────────────────────────────────────────────────────────

def detect_all(video_path, conf):
    """Return list of (frame_idx, t, bx_norm, by_norm, conf) for all detected frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dets = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        results = model(frame, classes=[BALL_CLASS], conf=conf, verbose=False)
        if results and len(results[0].boxes) > 0:
            box  = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            cf   = float(box.conf[0])
            bx   = (xyxy[0] + xyxy[2]) / 2 / w
            by   = (xyxy[1] + xyxy[3]) / 2 / h
            dets.append((frame_idx, t, bx, by, cf))
        frame_idx += 1
    cap.release()
    return dets, fps, w, h, total


# ─── Counting approaches ───────────────────────────────────────────────────────

def count_local_max(dets, min_delta, cooldown, max_adj_gap=0.15):
    """Adjacent local max + gap reversal."""
    count = 0
    last_t = -999.0
    events = []
    ys = [d[3] for d in dets]
    ts = [d[1] for d in dets]
    n = len(ys)
    for i in range(1, n - 1):
        gb = ts[i] - ts[i-1]
        ga = ts[i+1] - ts[i]
        # Signal 1: adjacent local max
        if gb <= max_adj_gap and ga <= max_adj_gap:
            if ys[i] - ys[i-1] > min_delta and ys[i] - ys[i+1] > min_delta:
                if ts[i] - last_t >= cooldown:
                    count += 1
                    last_t = ts[i]
                    events.append(('max', ts[i], ys[i]))
                    continue
        # Signal 2: gap reversal
        if 0.08 < gb < 0.60 and n >= 4 and i >= 2:
            prev_gb = ts[i-1] - ts[i-2]
            if prev_gb <= max_adj_gap and ga <= max_adj_gap:
                vel_before = ys[i-1] - ys[i-2]
                vel_after  = ys[i+1] - ys[i]
                t_contact  = (ts[i-1] + ts[i]) / 2
                if vel_before > min_delta and vel_after < -min_delta:
                    if t_contact - last_t >= cooldown:
                        count += 1
                        last_t = t_contact
                        events.append(('gap', t_contact, ys[i]))
    return count, events


def count_span(dets, span, min_prom, cooldown):
    """Compare ball Y against positions span steps away."""
    count = 0
    last_t = -999.0
    events = []
    ys = [d[3] for d in dets]
    ts = [d[1] for d in dets]
    n = len(ys)
    for i in range(span, n - span):
        tl = ts[i] - ts[i - span]
        tr = ts[i + span] - ts[i]
        if tl > 1.0 or tr > 1.0:
            continue
        if ys[i] - ys[i-span] > min_prom and ys[i] - ys[i+span] > min_prom:
            if ts[i] - last_t >= cooldown:
                count += 1
                last_t = ts[i]
                events.append(('span', ts[i], ys[i]))
    return count, events


def count_velocity_reversal(dets, vel_thresh, cooldown, smooth_n=3):
    """Detect sign change in smoothed velocity (pos→neg = local max)."""
    if len(dets) < smooth_n * 2 + 2:
        return 0, []
    ys = [d[3] for d in dets]
    ts = [d[1] for d in dets]
    # Smoothed velocity at each point
    vels = []
    for i in range(1, len(ys)):
        vels.append(ys[i] - ys[i-1])
    # Smooth velocities
    smoothed = []
    for i in range(len(vels)):
        lo = max(0, i - smooth_n + 1)
        smoothed.append(sum(vels[lo:i+1]) / (i - lo + 1))
    count = 0
    last_t = -999.0
    events = []
    for i in range(1, len(smoothed)):
        t_gap = ts[i+1] - ts[i] if i+1 < len(ts) else 999
        if t_gap > 0.3:
            continue
        if smoothed[i-1] > vel_thresh and smoothed[i] < -vel_thresh:
            if ts[i] - last_t >= cooldown:
                count += 1
                last_t = ts[i]
                events.append(('vel', ts[i], ys[i]))
    return count, events


# ─── Approach sweep ───────────────────────────────────────────────────────────

APPROACHES = [
    # (name, fn, kwargs)
    ("local_max_c012_d010_cd020", count_local_max, dict(min_delta=0.010, cooldown=0.20)),
    ("local_max_c012_d008_cd020", count_local_max, dict(min_delta=0.008, cooldown=0.20)),
    ("local_max_c012_d010_cd025", count_local_max, dict(min_delta=0.010, cooldown=0.25)),
    ("span5_prom006_cd020",       count_span,      dict(span=5, min_prom=0.06, cooldown=0.20)),
    ("span5_prom004_cd020",       count_span,      dict(span=5, min_prom=0.04, cooldown=0.20)),
    ("vel_rev_v005_cd020",        count_velocity_reversal, dict(vel_thresh=0.005, cooldown=0.20)),
    ("vel_rev_v008_cd020",        count_velocity_reversal, dict(vel_thresh=0.008, cooldown=0.20)),
]

CONF_LEVELS = [0.10, 0.12, 0.15, 0.20]


# ─── Annotated video generator ────────────────────────────────────────────────

def make_annotated_segment(video_path, seg_start_s, seg_end_s, dets, events,
                            fps, w, h, out_path, conf_thresh):
    """Generate annotated video segment [seg_start_s, seg_end_s]."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX

    det_by_frame = {d[0]: d for d in dets}
    event_ts = {e[1] for e in events}

    frame_idx = 0
    juggle_flash = 0  # frames to flash "JUGGLE" indicator

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t < seg_start_s:
            frame_idx += 1
            continue
        if t > seg_end_s:
            break

        # Draw detection
        if frame_idx in det_by_frame:
            fi, ft, bx, by, cf = det_by_frame[frame_idx]
            cx = int(bx * w)
            cy = int(by * h)
            colour = (0, 255, 0) if cf >= conf_thresh else (0, 200, 255)
            cv2.circle(frame, (cx, cy), 22, colour, 2)
            cv2.circle(frame, (cx, cy), 4, colour, -1)
            cv2.putText(frame, f"{cf:.2f}", (cx + 6, cy - 6),
                        font, 0.4, colour, 1)
            # Draw Y position bar on left edge
            bar_y = int(by * h)
            cv2.line(frame, (0, bar_y), (8, bar_y), (0, 255, 200), 2)
        else:
            # No detection — red X in corner
            cv2.putText(frame, "NO DET", (10, h - 20), font, 0.5, (0, 80, 255), 1)

        # Flash juggle when nearby
        for et in event_ts:
            if abs(t - et) < 0.15:
                juggle_flash = 8
        if juggle_flash > 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 6)
            cv2.putText(frame, "JUGGLE", (w//2 - 60, 60), font, 1.5, (0, 255, 0), 3)
            juggle_flash -= 1

        # Frame info
        cv2.putText(frame, f"f{frame_idx} t={t:.2f}s", (4, 18), font, 0.45, (255, 255, 255), 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.monotonic()
    print(f"Ground truth: {GT_COUNT} juggles")
    print(f"Video: {VIDEO_PATH}")

    results = {
        "ground_truth": GT_COUNT,
        "video": VIDEO_PATH,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [],
        "best": None,
    }

    best_score = 999
    best_run = None

    for conf in CONF_LEVELS:
        print(f"\n--- conf={conf} ---")
        dets, fps, w, h, total = detect_all(VIDEO_PATH, conf)
        det_rate = len(dets) / max(total, 1)
        print(f"  Detected {len(dets)}/{total} frames ({det_rate:.0%})")

        for name, fn, kwargs in APPROACHES:
            count, events = fn(dets, **kwargs)
            error = abs(count - GT_COUNT)
            in_range = GT_COUNT - 4 <= count <= GT_COUNT + 4
            print(f"  {name}: {count}  (err={error}{'  ←' if in_range else ''})")

            run = {
                "conf": conf,
                "approach": name,
                "count": count,
                "error": error,
                "in_range": in_range,
                "det_rate": round(det_rate, 3),
                "kwargs": kwargs,
            }
            results["runs"].append(run)

            if error < best_score:
                best_score = error
                best_run = run
                best_run["events"] = events
                best_run["dets"] = dets
                best_run["fps"] = fps
                best_run["w"] = w
                best_run["h"] = h
                best_run["fn"] = fn

    print(f"\nBest: {best_run['approach']} @ conf={best_run['conf']}  "
          f"count={best_run['count']}  err={best_run['error']}")

    # Generate annotated segments with best approach
    print("\nGenerating annotated segments...")
    seg_len = 5.0  # seconds
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_t = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    seg_n = 0
    t_s = 0.0
    while t_s < total_t:
        t_e = min(t_s + seg_len, total_t)
        out_path = os.path.join(OUT_DIR, f"seg{seg_n:02d}_{t_s:.0f}s-{t_e:.0f}s.mp4")
        make_annotated_segment(
            VIDEO_PATH, t_s, t_e,
            best_run["dets"], best_run["events"],
            best_run["fps"], best_run["w"], best_run["h"],
            out_path, best_run["conf"],
        )
        print(f"  Saved {out_path}")
        t_s += seg_len
        seg_n += 1

    # Write results JSON (without non-serialisable fields)
    clean_best = {k: v for k, v in best_run.items()
                  if k not in ("events", "dets", "fps", "w", "h", "fn")}
    results["best"] = clean_best
    results["elapsed_s"] = round(time.monotonic() - t0, 1)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")
    print(f"Segments written to {OUT_DIR}/")
    print(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    main()

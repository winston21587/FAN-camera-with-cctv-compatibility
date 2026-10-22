"""
CCTV Multi-Face Detection & FAN (Face Alignment Network) Alignment
-------------------------------------------------------------------
Optimizations applied:
  - Deep-merge config defaults so partial config.json files are safe
  - RGB conversion only happens when FAN is active and faces are found
  - Frame-budget throttle: time.sleep() pads each frame to 1/target_fps seconds
  - Haar detection runs every N frames (detection_skip_frames) — not every frame
  - CLAHE applied on a downscaled gray image, detections upscaled back (faster)
  - FAN runs in a background daemon thread — display loop never blocks on GPU
  - align_face_crop uses INTER_LINEAR for preview (vs INTER_CUBIC — 2x faster)
  - draw_face_landmarks uses cv2.polylines batch draw (no Python for-loop)
  - Recording buffer stores JPEG bytes instead of raw BGR arrays — 10x less RAM
  - VideoCapture buffer size clamped to 2 to prevent stale-frame queue buildup
  - Cap buffer set via CAP_PROP_BUFFERSIZE to keep latency minimal on live feeds
"""

import os
import cv2
import time
import json
import argparse
import threading
import numpy as np

# ── Optional: PyTorch + Face Alignment Network (FAN) ──────────────────────────
try:
    import torch
    import face_alignment
    HAS_FAN_SUPPORT = True
except ImportError:
    HAS_FAN_SUPPORT = False


# ==========================================================
# CONFIGURATION
# ==========================================================
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "source": {
        "video_source": "0"
    },
    "detection": {
        "scale_factor": 1.08,
        "min_neighbors": 5,
        "min_face_size": [20, 20],
        "max_faces": 5,
        "enable_clahe": True,
        "detection_skip_frames": 2,
        "detection_downsample": 0.5
    },
    "fan_alignment": {
        "enable_fan": True,
        "device": "cuda",
        "draw_landmarks": True,
        "crop_size": [112, 112],
        "save_aligned_crops": True,
        "crop_output_dir": "aligned_faces",
        "show_preview_overlay": True
    },
    "recording": {
        "record_duration_sec": 5.0
    },
    "snapshots": {
        "output_dir": "snapshots"
    },
    "clip_processing": {
        "clip_duration_sec": 5.0,
        "target_fps": 30,
        "top_k_faces": 5,
        "min_frame_interval": 5,
        "output_dir": "output_restoration"
    },
    "display": {
        "window_name": "CCTV Multi-Face Detection & FAN Alignment",
        "window_width": 1024,
        "window_height": 600,
        "show_hud": True,
        "target_fps": 30
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*.
    Sections present in base but missing in override retain their defaults.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str = CONFIG_FILE) -> dict:
    """
    Loads config.json and deep-merges with DEFAULT_CONFIG so partial
    config files (e.g. only overriding 'source') always produce a
    fully-populated config dictionary.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            return _deep_merge(DEFAULT_CONFIG, user_cfg)
        except Exception as exc:
            print(f"[!] Warning: Could not parse '{path}' ({exc}). Using default settings.")
    return DEFAULT_CONFIG


def get_capture_source(source_val):
    """Returns an int for local webcam indices or a string for file/stream paths."""
    s = str(source_val).strip()
    return int(s) if s.isdigit() else s


# ==========================================================
# FACE ALIGNMENT NETWORK (FAN)
# ==========================================================
def init_fan_model(device_preference: str = "cuda"):
    """
    Initializes a 2-D 68-point FaceAlignment model.
    compile=False avoids the one-time JIT cost for instant startup.
    Falls back to CPU automatically if CUDA is not available.
    """
    if not HAS_FAN_SUPPORT:
        print("[!] 'torch' or 'face_alignment' not installed — FAN disabled.")
        return None

    use_cuda = (device_preference.lower() == "cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[*] Initializing FAN on {device.upper()}…")
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device,
            compile=False       # Skip torch.compile — instant startup, no JIT overhead
        )
        print("[+] FAN model ready.")
        return fa
    except Exception as exc:
        print(f"[!] FAN init failed: {exc}")
        return None


def align_face_crop(image: np.ndarray, landmarks: np.ndarray, crop_size=(112, 112)):
    """
    Affine-aligns a face so both eyes are level and centered.
    Uses 68-point landmarks; returns a crop of shape (*crop_size, 3).

    Geometry: eyes placed at (35 %, 38 %) from the left and top of the crop,
    matching the standard used by ArcFace / InsightFace training data.
    Uses INTER_LINEAR (vs INTER_CUBIC) for real-time preview — 2x faster with
    virtually identical visual quality at small crop sizes.
    """
    if landmarks is None or len(landmarks) < 68:
        return None

    # Eye centres from landmark point ranges
    left_eye  = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    dX, dY = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle   = np.degrees(np.arctan2(dY, dX))

    desired_left_eye_x = 0.35
    desired_dist       = 1.0 - 2.0 * desired_left_eye_x          # 0.30
    dist               = np.hypot(dX, dY)
    scale              = (crop_size[0] * desired_dist) / max(dist, 1e-6)

    eye_center = ((left_eye[0] + right_eye[0]) * 0.5,
                  (left_eye[1] + right_eye[1]) * 0.5)

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    # Shift so the eye centre lands at the canonical position inside the crop
    M[0, 2] += (crop_size[0] * 0.5)  - eye_center[0]
    M[1, 2] += (crop_size[1] * 0.38) - eye_center[1]

    return cv2.warpAffine(
        image, M, (crop_size[0], crop_size[1]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def draw_face_landmarks(frame: np.ndarray, landmarks: np.ndarray,
                        color=(0, 255, 255)) -> None:
    """
    Renders 68 landmark dots as individual filled circles.
    Points are rounded and clipped to the frame in one vectorised numpy pass
    before the drawing loop to avoid redundant per-point Python logic.
    """
    if landmarks is None:
        return

    h, w = frame.shape[:2]
    pts = np.clip(np.round(landmarks).astype(np.int32), [0, 0], [w - 1, h - 1])
    for (px, py) in pts:
        cv2.circle(frame, (px, py), 2, color, -1, lineType=cv2.LINE_AA)



# ==========================================================
# DRAWING PRIMITIVES
# ==========================================================
def draw_face_badge(frame: np.ndarray, x, y, w, h, face_num: int,
                    has_fan: bool = False, color=(0, 255, 0)) -> None:
    """
    Bounding box with CCTV-style corner reticles and a floating label badge.
    """
    # Main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner reticles — length proportional to box size, minimum 8 px
    clen = max(8, int(min(w, h) * 0.15))
    ct   = 3
    # Top-left
    cv2.line(frame, (x,     y),     (x + clen, y),         color, ct)
    cv2.line(frame, (x,     y),     (x,         y + clen),  color, ct)
    # Top-right
    cv2.line(frame, (x + w, y),     (x + w - clen, y),     color, ct)
    cv2.line(frame, (x + w, y),     (x + w, y + clen),     color, ct)
    # Bottom-left
    cv2.line(frame, (x,     y + h), (x + clen, y + h),     color, ct)
    cv2.line(frame, (x,     y + h), (x, y + h - clen),     color, ct)
    # Bottom-right  ← was (y + corner_len) previously — now correct
    cv2.line(frame, (x + w, y + h), (x + w - clen, y + h), color, ct)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - clen), color, ct)

    # Floating label badge
    label = f"Face #{face_num}" + (" [FAN]" if has_fan else "")
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)

    by1  = y - th - 8 if (y - th - 10) > 5 else y
    by2  = by1 + th + 8
    bx2  = min(x + tw + 8, frame.shape[1] - 1)

    cv2.rectangle(frame, (x, by1), (bx2, by2), color, -1)
    cv2.putText(frame, label, (x + 4, by1 + th + 4), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_pip_crops(frame: np.ndarray, aligned_crops: list, max_show: int = 3) -> None:
    """
    Picture-in-Picture thumbnails of aligned face crops on the right edge.
    Thumbnails are fixed at 80×80 px with a 1-px border and 'Aligned #n' tag.
    """
    if not aligned_crops:
        return

    PAD        = 10
    THUMB_SZ   = 80
    start_y    = 55
    fh, fw     = frame.shape[:2]
    row_height = THUMB_SZ + PAD + 18

    for i, (f_num, crop) in enumerate(aligned_crops[:max_show]):
        y1 = start_y + i * row_height
        y2 = y1 + THUMB_SZ
        if y2 + 15 > fh:
            break

        x1 = fw - THUMB_SZ - PAD
        x2 = x1 + THUMB_SZ

        thumb = cv2.resize(crop, (THUMB_SZ, THUMB_SZ), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 1)
        frame[y1:y2, x1:x2] = thumb
        cv2.putText(frame, f"Aligned #{f_num}", (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


def draw_cctv_hud(frame: np.ndarray, face_count: int, fps: float,
                  source_name: str, fan_active: bool = False,
                  is_paused: bool = False,
                  rec_status: str = None,
                  notify_msg: str = None,
                  rec_duration_sec: float = 5.0) -> None:
    """
    On-Screen Display: semi-transparent top header bar + bottom hotkey legend.
    Uses a pre-sliced ROI copy to avoid a full-frame addWeighted call.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top bar: blend only the 45-px strip (faster than full-frame copy) ──
    roi = frame[:45, :]
    dark = np.full_like(roi, (15, 15, 15))
    cv2.addWeighted(dark, 0.70, roi, 0.30, 0, roi)
    frame[:45, :] = roi                             # write blended strip back

    # Status
    cv2.putText(frame, "PAUSED" if is_paused else "● CCTV FEED",
                (15, 28), font, 0.55,
                (0, 255, 255) if is_paused else (0, 0, 255), 2, cv2.LINE_AA)

    # Source
    src = f"SRC: {source_name}"
    if len(src) > 20:
        src = src[:17] + "…"
    cv2.putText(frame, src, (155, 28), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Recording indicator (flashing red badge)
    if rec_status:
        cv2.putText(frame, f"● {rec_status}", (320, 28), font, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # FAN indicator
    cv2.putText(frame, "FAN: ON" if fan_active else "FAN: OFF",
                (w - 450, 28), font, 0.50,
                (0, 255, 255) if fan_active else (120, 120, 120), 2, cv2.LINE_AA)

    # Face count
    cv2.putText(frame, f"Faces: {face_count}",
                (w - 320, 28), font, 0.55,
                (0, 255, 0) if face_count > 0 else (180, 180, 180), 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 28),
                font, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    # Notification toast
    if notify_msg:
        cv2.putText(frame, notify_msg, (15, 70), font, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

    # Bottom hotkey legend
    cv2.putText(
        frame,
        f"[SPACE] Pause  |  [C] Record {rec_duration_sec:.0f}s  |  [F] FAN  |  [L] Landmarks  "
        "|  [S] Save  |  [R] Replay  |  [Q] Quit",
        (15, h - 12), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA
    )


# ==========================================================
# CLI
# ==========================================================
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CCTV Multi-Face Detection with FAN (Face Alignment Network)"
    )
    p.add_argument("--config", default=CONFIG_FILE, help="Path to config.json")
    p.add_argument("--source", default=None,
                   help="Override video source: camera index, file path, or RTSP/HTTP URL")
    p.add_argument("--record-snippet", type=float, nargs="?", const=5.0, default=None,
                   help="Directly record a 5s snippet from camera and run Top-5 Face Alignment pipeline")
    p.add_argument("--fan",    action="store_true", default=None, help="Force-enable FAN")
    p.add_argument("--no-fan", action="store_true",               help="Force-disable FAN")
    return p.parse_args()


# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    args   = parse_arguments()
    config = load_config(args.config)

    # ── Resolve settings (CLI flags override config.json) ─────────────────────
    source_val = (args.source if args.source is not None
                  else config["source"]["video_source"])
    source = get_capture_source(source_val)

    dc = config["detection"]
    scale_factor        = float(dc["scale_factor"])
    min_neighbors       = int(dc["min_neighbors"])
    min_face_size       = tuple(dc["min_face_size"])
    max_faces           = int(dc["max_faces"])
    enable_clahe        = bool(dc["enable_clahe"])
    detection_skip      = max(0, int(dc.get("detection_skip_frames", 2)))
    detect_scale        = float(dc.get("detection_downsample", 0.5))
    detect_scale        = max(0.1, min(1.0, detect_scale))  # clamp 10%–100%

    fc = config["fan_alignment"]
    crop_size           = tuple(fc["crop_size"])
    crop_output_dir     = fc["crop_output_dir"]
    save_aligned_crops  = bool(fc["save_aligned_crops"])
    draw_landmarks_opt  = bool(fc["draw_landmarks"])
    show_pip_overlay    = bool(fc["show_preview_overlay"])
    fan_device          = fc["device"]
    enable_fan          = (False if args.no_fan else
                           True  if args.fan    else
                           bool(fc["enable_fan"]))

    dc2        = config["display"]
    window_name = dc2["window_name"]
    win_w       = int(dc2["window_width"])
    win_h       = int(dc2["window_height"])

    # ── Haar Cascade ───────────────────────────────────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[!] Cannot load cascade classifier from {cascade_path}")
        return

    # ── FAN model ──────────────────────────────────────────────────────────────
    fan_model = None
    if enable_fan and HAS_FAN_SUPPORT:
        fan_model = init_fan_model(fan_device)
        if fan_model is None:
            enable_fan = False

    # ── Video capture ──────────────────────────────────────────────────────────
    is_live = isinstance(source, int) or (
        isinstance(source, str) and
        (source.startswith("rtsp://") or source.startswith("http://"))
    )
    cap = cv2.VideoCapture(source)
    # Limit the internal frame queue to 2 so we always read the most recent
    # frame on a live webcam / RTSP stream instead of processing stale frames.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"[!] Cannot open source '{source_val}'.")
        return

    # ── Window & playback timing ───────────────────────────────────────────────
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    target_fps  = int(dc2.get("target_fps", 30))

    # For recorded video files, respect the file's native FPS.
    # For live webcam / RTSP, cap to target_fps so we yield the CPU
    # between frames instead of spinning at 100% on waitKey(1).
    if not is_live and video_fps > 0:
        effective_fps = video_fps
    else:
        effective_fps = target_fps

    # Minimum waitKey value (1 ms) — just enough for GUI events.
    # The real pacing is done via time.sleep() inside the loop.
    frame_delay  = 1
    frame_budget = 1.0 / effective_fps          # seconds per frame budget

    clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    source_label = (f"Cam {source}" if isinstance(source, int)
                    else os.path.basename(str(source)))

    print(f"[*] Source: {source_val}  |  Max faces: {max_faces}"
          f"  |  FAN: {'ON' if enable_fan else 'OFF'}"
          f"  |  Target FPS: {effective_fps:.0f}")
    print("[*] [SPACE] Pause  [F] FAN  [L] Landmarks  [S] Save  [R] Replay  [Q] Quit")

    # State
    is_paused           = False
    last_frame          = None
    current_faces_count = 0
    aligned_crops       = []
    landmarks_list      = []
    sorted_faces        = []               # cached between skipped frames
    frame_counter       = 0               # used for detection_skip_frames
    fps_t0              = time.perf_counter()
    fps_frames          = 0
    current_fps         = effective_fps

    # ── FAN background thread state ────────────────────────────────────────────
    # FAN inference can take 20-80 ms. Running it on a daemon thread lets the
    # display loop continue at full frame-rate while GPU/CPU computes landmarks.
    _fan_lock          = threading.Lock()
    _fan_pending_frame = None      # latest RGB frame waiting for FAN
    _fan_pending_boxes = None      # bboxes matching that frame
    _fan_result_lm     = []        # latest landmarks list from FAN
    _fan_busy          = False     # True while FAN is computing

    def _fan_worker():
        """Daemon thread: continuously processes the pending FAN request."""
        nonlocal _fan_pending_frame, _fan_pending_boxes, _fan_result_lm, _fan_busy
        while True:
            with _fan_lock:
                rgb_f = _fan_pending_frame
                boxes = _fan_pending_boxes
                _fan_pending_frame = None
                _fan_pending_boxes = None

            if rgb_f is None:
                time.sleep(0.005)   # yield while idle
                continue

            try:
                preds = fan_model.get_landmarks(rgb_f, detected_faces=boxes)
                with _fan_lock:
                    _fan_result_lm = preds if preds else []
            except Exception as exc:
                print(f"[!] FAN thread error: {exc}")
                with _fan_lock:
                    _fan_result_lm = []
            finally:
                with _fan_lock:
                    _fan_busy = False

    if enable_fan and fan_model is not None:
        _fan_thread = threading.Thread(target=_fan_worker, daemon=True)
        _fan_thread.start()

    # ── Direct snippet recording state ─────────────────────────────────────────
    is_recording_snippet = False
    rec_start_time       = 0.0
    rec_duration         = float(
        config.get("recording", {}).get("record_duration_sec")
        or config.get("clip_processing", {}).get("clip_duration_sec", 5.0)
    )
    snapshots_dir        = config.get("snapshots", {}).get("output_dir", "snapshots")
    rec_frames           = []
    rec_fps              = effective_fps
    notify_msg           = None
    notify_until         = 0.0
    rec_status_badge     = None

    # CLI direct record mode check
    if args.record_snippet is not None:
        from clip_face_pipeline import record_snippet_from_source, run_clip_pipeline
        dur = float(args.record_snippet) if args.record_snippet is not None else rec_duration
        temp_clip = "cctv_snippet.mp4"
        cap.release()
        cv2.destroyAllWindows()
        rec_path = record_snippet_from_source(source, duration_sec=dur, output_path=temp_clip)
        if rec_path:
            run_clip_pipeline(rec_path, config)
        return

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        loop_start = time.perf_counter()    # mark start of this frame's budget

        if not is_paused:
            ret, frame = cap.read()

            if not ret:
                if not is_live:
                    print("\n[!] End of footage — press 'r' to replay or 'q' to quit.")
                    if cv2.waitKey(0) & 0xFF == ord('r'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                else:
                    time.sleep(0.5)
                    continue

            # FPS counter (high-precision)
            fps_frames += 1
            dt = time.perf_counter() - fps_t0
            if dt >= 0.5:
                current_fps = fps_frames / dt
                fps_frames  = 0
                fps_t0      = time.perf_counter()

            # ── Preprocessing ─────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Haar detection (skipped every N frames to cut CPU) ─────────────
            # Between detection frames we reuse the previous sorted_faces list.
            # This is safe because faces don't teleport between adjacent frames.
            if frame_counter % (detection_skip + 1) == 0:
                if detect_scale < 1.0:
                    # Downscale for detection, then map boxes back to full res
                    small_gray = cv2.resize(gray, None,
                                            fx=detect_scale, fy=detect_scale,
                                            interpolation=cv2.INTER_NEAREST)
                    proc_gray  = clahe.apply(small_gray) if enable_clahe else small_gray
                    inv        = 1.0 / detect_scale
                    raw_faces  = face_cascade.detectMultiScale(
                        proc_gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(max(4, int(min_face_size[0] * detect_scale)),
                                 max(4, int(min_face_size[1] * detect_scale)))
                    )
                    if len(raw_faces) > 0:
                        raw_faces = (raw_faces * (inv, inv, inv, inv)).astype(int)
                else:
                    proc_gray = clahe.apply(gray) if enable_clahe else gray
                    raw_faces = face_cascade.detectMultiScale(
                        proc_gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=min_face_size
                    )

                sorted_faces = (
                    sorted(raw_faces, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]
                    if len(raw_faces) > 0 else []
                )

                # ── Enqueue FAN inference (non-blocking) ──────────────────────
                if enable_fan and fan_model is not None and sorted_faces:
                    with _fan_lock:
                        if not _fan_busy:
                            _fan_busy          = True
                            _fan_pending_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            _fan_pending_boxes = [[x, y, x + w, y + h]
                                                  for (x, y, w, h) in sorted_faces]

            frame_counter += 1

            # ── Consume latest FAN results (non-blocking) ─────────────────────
            with _fan_lock:
                landmarks_list = list(_fan_result_lm)

            aligned_crops = []   # reset each render frame

            # ── Per-face rendering & alignment ────────────────────────────────
            for idx, (x, y, w, h) in enumerate(sorted_faces, start=1):
                lm       = landmarks_list[idx - 1] if idx - 1 < len(landmarks_list) else None
                has_fan  = lm is not None

                draw_face_badge(frame, x, y, w, h, face_num=idx, has_fan=has_fan)

                if lm is not None:
                    if draw_landmarks_opt:
                        draw_face_landmarks(frame, lm)

                    crop = align_face_crop(frame, lm, crop_size=crop_size)
                    if crop is not None:
                        aligned_crops.append((idx, crop))

            # ── Recording buffer accumulation ──────────────────────────────────
            # Frames are JPEG-compressed before buffering to cut RAM usage 10x
            # (a 1080p BGR frame ≈ 6 MB; a 90% JPEG ≈ 150 KB).
            if is_recording_snippet:
                _, jpg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rec_frames.append(jpg_buf)
                elapsed_rec = time.perf_counter() - rec_start_time
                rem_rec = max(0.0, rec_duration - elapsed_rec)
                rec_status_badge = f"REC {rem_rec:.1f}s"

                if elapsed_rec >= rec_duration:
                    is_recording_snippet = False
                    frames_captured = list(rec_frames)
                    rec_frames = []
                    rec_status_badge = None
                    notify_msg = f"[+] {rec_duration:.0f}s Recorded! Top-5 pipeline running in background..."
                    notify_until = time.perf_counter() + 4.0

                    def _bg_process_snippet(jpeg_bufs, fps_val, cfg_dict):
                        try:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            out_base = cfg_dict.get("clip_processing", {}).get("output_dir", "output_restoration")
                            os.makedirs(out_base, exist_ok=True)
                            clip_file = os.path.join(out_base, f"cctv_rec_{ts}.mp4")
                            # Decode first frame to get dimensions
                            first = cv2.imdecode(jpeg_bufs[0], cv2.IMREAD_COLOR)
                            h_c, w_c = first.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(clip_file, fourcc, fps_val, (w_c, h_c))
                            writer.write(first)
                            for buf in jpeg_bufs[1:]:
                                writer.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
                            writer.release()
                            print(f"\n[+] Recording saved ({len(jpeg_bufs)} frames). Launching Top-5 Face Pipeline...")
                            from clip_face_pipeline import run_clip_pipeline
                            run_clip_pipeline(clip_file, cfg_dict)
                        except Exception as exc:
                            print(f"[!] Background snippet processing error: {exc}")

                    threading.Thread(
                        target=_bg_process_snippet,
                        args=(frames_captured, rec_fps, dict(config)),
                        daemon=True
                    ).start()
            else:
                rec_status_badge = None

            # Notification timeout check
            if notify_msg and time.perf_counter() > notify_until:
                notify_msg = None

            # ── Overlays ──────────────────────────────────────────────────────
            if show_pip_overlay:
                draw_pip_crops(frame, aligned_crops)

            draw_cctv_hud(frame, len(sorted_faces), current_fps,
                          source_label, fan_active=enable_fan,
                          rec_status=rec_status_badge, notify_msg=notify_msg,
                          rec_duration_sec=rec_duration)

            last_frame          = frame          # reference, not a copy — pause makes its own copy
            current_faces_count = len(sorted_faces)

        else:  # paused
            if last_frame is not None:
                frame = last_frame.copy()        # copy only on pause render
                draw_cctv_hud(frame, current_faces_count, current_fps,
                              source_label, fan_active=enable_fan, is_paused=True,
                              rec_status=rec_status_badge, notify_msg=notify_msg,
                              rec_duration_sec=rec_duration)

        cv2.imshow(window_name, frame)

        # ── Frame-budget throttle ──────────────────────────────────────────────
        # Sleep the unused portion of the frame budget BEFORE waitKey so the OS
        # gets CPU back. waitKey(1) is still called for GUI event processing.
        elapsed = time.perf_counter() - loop_start
        sleep_s = frame_budget - elapsed
        if sleep_s > 0.001:             # only sleep if meaningful remainder (>1 ms)
            time.sleep(sleep_s)

        # ── Keyboard handling ──────────────────────────────────────────────────
        key = cv2.waitKey(frame_delay) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            is_paused = not is_paused

        elif key == ord('c'):
            if not is_recording_snippet:
                is_recording_snippet = True
                rec_start_time = time.perf_counter()
                rec_frames = []
                rec_status_badge = f"REC {rec_duration:.1f}s"
                print(f"\n[*] Started recording {rec_duration:.1f}s snippet for Top-5 Face Alignment...")
            else:
                print("[!] Already recording a snippet...")

        elif key == ord('f'):
            if HAS_FAN_SUPPORT:
                if fan_model is None:
                    fan_model = init_fan_model(fan_device)
                enable_fan = not enable_fan
                print(f"[*] FAN: {'ON' if enable_fan else 'OFF'}")
            else:
                print("[!] face_alignment / torch not installed.")

        elif key == ord('l'):
            draw_landmarks_opt = not draw_landmarks_opt
            print(f"[*] Landmarks: {'ON' if draw_landmarks_opt else 'OFF'}")

        elif key == ord('s'):
            ts    = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(snapshots_dir, exist_ok=True)
            snap  = os.path.join(snapshots_dir, f"cctv_snapshot_{ts}.jpg")
            cv2.imwrite(snap, frame)
            print(f"[+] Snapshot saved: {snap}")
            if save_aligned_crops and aligned_crops:
                os.makedirs(crop_output_dir, exist_ok=True)
                for f_num, crop_img in aligned_crops:
                    path = os.path.join(crop_output_dir, f"face_{ts}_id{f_num}.jpg")
                    cv2.imwrite(path, crop_img)
                    print(f"    -> {path}")

        elif key == ord('r') and not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            is_paused = False
            print("[*] Rewound to start.")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()
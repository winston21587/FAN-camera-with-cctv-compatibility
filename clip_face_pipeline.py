"""
CCTV Clip Multi-Face Detection, FAN Alignment & Top-5 Selection Pipeline
=========================================================================
Processes short CCTV video snippets (~5 seconds @ ~30 FPS), tracks multiple face
identities across frames, isolates FAN alignment workloads into separate CPU worker
processes, scores alignment/sharpness/frontality, and extracts the top 5 best-aligned
images per face identity for downstream face restoration models (CodeFormer, GFPGAN).
"""

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

# Core modules
from core.tracker import FaceTracker
from core.worker_pool import process_face_tracks_parallel

CONFIG_FILE = "config.json"

DEFAULT_PIPELINE_CONFIG = {
    "source": {
        "video_source": "0",
        "description": "Camera index (e.g. 0), video file path ('cctv.mp4'), or RTSP/HTTP network stream"
    },
    "recording": {
        "record_duration_sec": 5.0,
        "description": "Recording duration in seconds when pressing [C] or using --record-snippet"
    },
    "detection": {
        "scale_factor": 1.08,
        "min_neighbors": 5,
        "min_face_size": [20, 20],
        "max_faces": 5,
        "enable_clahe": True
    },
    "clip_processing": {
        "clip_duration_sec": 5.0,
        "target_fps": 30,
        "top_k_faces": 5,
        "min_frame_interval": 5,
        "output_dir": "output_restoration"
    },
    "multiprocessing": {
        "enabled": True,
        "max_workers": 2
    },
    "fan_alignment": {
        "enable_fan": True,
        "device": "cuda",
        "crop_size": [512, 512],
        "export_format": "jpg",
        "jpg_quality": 98
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override dictionary into base defaults."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str = CONFIG_FILE) -> dict:
    """Loads configuration with safe defaults."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            return _deep_merge(DEFAULT_PIPELINE_CONFIG, user_cfg)
        except Exception as exc:
            print(f"[!] Warning: Could not parse '{path}' ({exc}). Using defaults.")
    return DEFAULT_PIPELINE_CONFIG


def record_snippet_from_source(source_val: Any, duration_sec: float = 5.0, output_path: str = "temp_snippet.mp4") -> Optional[str]:
    """
    Convenience utility: records a short ~5-second snippet from a live CCTV camera / stream.
    """
    print(f"[*] Capturing {duration_sec:.1f}s video snippet from source: {source_val} ...")
    cap = cv2.VideoCapture(int(source_val) if str(source_val).isdigit() else source_val)
    if not cap.isOpened():
        print(f"[!] Failed to open video source '{source_val}'.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_target_frames = int(duration_sec * fps)
    frames_recorded = 0

    t_start = time.perf_counter()
    while frames_recorded < total_target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_recorded += 1

    out.release()
    cap.release()
    elapsed = time.perf_counter() - t_start
    print(f"[+] Snippet captured: {frames_recorded} frames ({elapsed:.2f}s) -> '{output_path}'")
    return output_path


def run_clip_pipeline(
    clip_path: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main processing pipeline for a short video clip snippet.
    """
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Video clip not found: '{clip_path}'")

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read video clip: '{clip_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

    print("=" * 70)
    print(" CCTV MULTI-FACE DETECTION & FAN ALIGNMENT PIPELINE")
    print("=" * 70)
    print(f"[*] Input Clip      : {clip_path}")
    print(f"[*] Frames / FPS    : {total_frames} frames @ {native_fps:.1f} FPS (~{duration_sec:.1f}s)")
    print(f"[*] FAN Alignment   : {'ON' if config['fan_alignment']['enable_fan'] else 'OFF'}")
    print(f"[*] Target Crop     : {config['fan_alignment']['crop_size']}")
    print(f"[*] Max Workers     : {config['multiprocessing']['max_workers']} processes")
    print("-" * 70)

    # ── Stage 1: Fast Face Detection & Inter-Frame Tracking ───────────────────
    det_cfg = config["detection"]
    scale_factor = float(det_cfg["scale_factor"])
    min_neighbors = int(det_cfg["min_neighbors"])
    min_face_size = tuple(det_cfg["min_face_size"])
    max_faces = int(det_cfg["max_faces"])
    enable_clahe = bool(det_cfg["enable_clahe"])

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load Haar cascade from {cascade_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    tracker = FaceTracker(max_lost=15, min_iou=0.20, min_hits=3)

    print("[*] Stage 1/3: Ingesting frames & tracking multi-face identities...")
    t0 = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_sec = frame_idx / native_fps

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc_gray = clahe.apply(gray) if enable_clahe else gray

        # Multi-face detection
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

        # Update tracker
        detections = [tuple(b) for b in sorted_faces]
        tracker.update(frame_idx, timestamp_sec, detections)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"    -> Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.0f}%) ...")

    cap.release()
    t_det = time.perf_counter() - t0
    all_tracks = tracker.finalize()

    print(f"[+] Detection & tracking completed in {t_det:.2f}s.")
    print(f"[+] Identified {len(all_tracks)} distinct face track(s) in snippet.")

    if not all_tracks:
        print("[!] No sustained faces identified in this video clip.")
        return {"total_tracks": 0, "results": []}

    for track in all_tracks:
        print(f"    - Face Track #{track.track_id}: {track.hits} detections across frames {track.history[0]['frame_idx']}..{track.history[-1]['frame_idx']}")

    # ── Stage 2: Multiprocessing Isolation & FAN Top-5 Selection ──────────────
    if output_dir is None:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_out = config["clip_processing"]["output_dir"]
        output_dir = os.path.join(base_out, f"{clip_name}_{timestamp_str}")

    os.makedirs(output_dir, exist_ok=True)
    print("-" * 70)
    print(f"[*] Stage 2/3: Dispatching face tracks to isolated worker processes...")
    print(f"    Output directory: {os.path.abspath(output_dir)}")

    t1 = time.perf_counter()
    results = process_face_tracks_parallel(
        tracks=all_tracks,
        video_path=os.path.abspath(clip_path),
        frames_dict=None,
        config=config,
        output_dir=output_dir
    )
    t_workers = time.perf_counter() - t1
    print(f"[+] Worker alignment and scoring completed in {t_workers:.2f}s.")

    # ── Stage 3: Summary Compilation & Downstream Context ─────────────────────
    print("-" * 70)
    print(" SUMMARY OF EXTRACTED BEST ALIGNED FACES")
    print("-" * 70)

    summary_records = []
    for res in results:
        t_id = res["track_id"]
        meta_file = res["metadata_file"]
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"\n[+] Face ID #{t_id} (Detections: {meta['total_detections']}, Selected: {meta['selected_count']}):")
        for frame_info in meta["selected_frames"]:
            rank = frame_info["rank"]
            f_num = frame_info["frame_idx"]
            score = frame_info["score"]
            sharp = frame_info["sharpness_laplacian"]
            pose = frame_info["pose_score"]
            fname = frame_info["filename"]
            print(f"    Rank #{rank}: Frame {f_num:03d} | Composite Score: {score:.3f} (Pose: {pose:.3f}, Sharpness: {sharp:.1f}) -> {fname}")

        summary_records.append(meta)

    # Save overall summary.json
    summary_path = os.path.join(output_dir, "clip_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "clip_path": os.path.abspath(clip_path),
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "processed_tracks": len(results),
            "fan_enabled": config["fan_alignment"]["enable_fan"],
            "target_crop_size": config["fan_alignment"]["crop_size"],
            "faces": summary_records,
            "downstream_context": {
                "intended_use": "Restoration of degraded CCTV faces",
                "recommended_models": ["CodeFormer", "GFPGAN", "RestoreFormer", "GPEN"],
                "input_resolution": config["fan_alignment"]["crop_size"],
                "guidance": "Feed the top 5 crops into your restoration model. The aligned images have eyes canonically level to maximize facial prior recovery without ghosting."
            }
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] Pipeline complete! All artifacts saved to: {output_dir}")
    print(f"[+] Summary file: {summary_path}")
    print("=" * 70)

    return {
        "output_dir": output_dir,
        "summary_file": summary_path,
        "tracks": summary_records
    }


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CCTV Clip Multi-Face Detection, FAN Alignment & Top-5 Selection Pipeline"
    )
    p.add_argument("--clip", default=None,
                   help="Path to a short (~5s) video clip file (.mp4, .avi, etc.)")
    p.add_argument("--record-snippet", type=float, nargs="?", const=5.0, default=None,
                   help="Record a live snippet from CCTV / webcam source (in seconds, default 5s)")
    p.add_argument("--config", default=CONFIG_FILE,
                   help="Path to config.json")
    p.add_argument("--output", default=None,
                   help="Custom output directory")
    p.add_argument("--fan", action="store_true", default=None,
                   help="Force-enable FAN alignment")
    p.add_argument("--no-fan", action="store_true",
                   help="Force-disable FAN alignment (uses square aspect-ratio crops)")
    p.add_argument("--workers", type=int, default=None,
                   help="Override max worker processes for multi-face CPU protection")
    p.add_argument("--top-k", type=int, default=None,
                   help="Number of best-aligned images to extract per face (default: 5)")
    p.add_argument("--crop-size", type=int, nargs=2, default=None,
                   help="Output crop dimensions, e.g. --crop-size 512 512")
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.no_fan:
        config["fan_alignment"]["enable_fan"] = False
    elif args.fan:
        config["fan_alignment"]["enable_fan"] = True

    if args.workers is not None:
        config["multiprocessing"]["max_workers"] = max(1, args.workers)

    if args.top_k is not None:
        config["clip_processing"]["top_k_faces"] = max(1, args.top_k)

    if args.crop_size is not None:
        config["fan_alignment"]["crop_size"] = [args.crop_size[0], args.crop_size[1]]

    clip_file = args.clip

    # Check if user requested recording a live snippet
    if args.record_snippet is not None:
        src = config["source"]["video_source"]
        default_dur = float(
            config.get("recording", {}).get("record_duration_sec")
            or config.get("clip_processing", {}).get("clip_duration_sec", 5.0)
        )
        duration = float(args.record_snippet) if args.record_snippet is not None else default_dur
        temp_clip = "cctv_snippet.mp4"
        recorded = record_snippet_from_source(src, duration_sec=duration, output_path=temp_clip)
        if recorded is None:
            sys.exit(1)
        clip_file = recorded

    if clip_file is None:
        # Check if config has a video clip specified or prompt user
        src = config["source"]["video_source"]
        if str(src).endswith((".mp4", ".avi", ".mkv", ".mov")):
            clip_file = src
        else:
            print("[!] No video clip specified!")
            print("    Usage 1 (Existing clip): python clip_face_pipeline.py --clip path/to/cctv_5s.mp4")
            print("    Usage 2 (Record from CCTV/Webcam): python clip_face_pipeline.py --record-snippet 5")
            sys.exit(1)

    try:
        run_clip_pipeline(clip_file, config, output_dir=args.output)
    except Exception as exc:
        print(f"[!] Pipeline error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

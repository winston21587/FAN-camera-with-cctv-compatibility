"""
Multiprocessing Worker Pool & CPU Isolation Module
--------------------------------------------------
Isolates face alignment, landmark estimation, and frame scoring into separate worker
processes to protect system responsiveness and prevent CPU core saturation.
"""

import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .alignment import init_fan_model, align_face_crop, get_square_bbox_crop
from .quality_scorer import (
    compute_sharpness,
    compute_fan_alignment_score,
    compute_composite_score,
    rank_and_select_top_frames
)


def _set_low_priority():
    """Sets the current process priority to BELOW_NORMAL on Windows to protect CPU."""
    try:
        import psutil
        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _worker_process_single_track(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executed in a dedicated worker process for an individual face track.
    Reads candidate frames, runs FAN (if enabled), evaluates quality/alignment scores,
    picks the top 5 frames, and writes aligned images + metadata to disk.
    """
    _set_low_priority()

    track_id = payload["track_id"]
    history = payload["history"]               # List of {"frame_idx", "timestamp_sec", "bbox"}
    video_path = payload.get("video_path")
    frames_dict = payload.get("frames_dict")   # Optional in-memory frames
    config = payload["config"]
    output_base_dir = payload["output_dir"]

    crop_size = tuple(config["fan_alignment"]["crop_size"])
    enable_fan = bool(config["fan_alignment"]["enable_fan"])
    device = config["fan_alignment"]["device"]
    top_k = int(config["clip_processing"]["top_k_faces"])
    min_interval = int(config["clip_processing"].get("min_frame_interval", 5))

    # Output directory for this specific face track
    face_dir = os.path.join(output_base_dir, f"face_id_{track_id:02d}")
    os.makedirs(face_dir, exist_ok=True)

    # Initialize FAN within the worker process if enabled
    fan_model = None
    if enable_fan:
        fan_model = init_fan_model(device_preference=device)
        if fan_model is None:
            enable_fan = False

    # Collect needed frame indices
    needed_frame_indices = {entry["frame_idx"]: entry for entry in history}

    # Open video capture if using video_path
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)

    candidates: List[Dict[str, Any]] = []

    try:
        # Evaluate each frame occurrence in the track
        for entry in history:
            f_idx = entry["frame_idx"]
            t_sec = entry["timestamp_sec"]
            x, y, w, h = entry["bbox"]
            bbox = (x, y, w, h)

            # Retrieve frame
            frame = None
            if frames_dict and f_idx in frames_dict:
                frame = frames_dict[f_idx]
            elif cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, read_frame = cap.read()
                if ret and read_frame is not None:
                    frame = read_frame

            if frame is None or frame.size == 0:
                continue

            # Crop / Alignment
            aligned_crop = None
            affine_matrix = None
            landmarks_coords = None
            pose_score = 0.0
            pose_metrics = {}

            if enable_fan and fan_model is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_box = [[x, y, x + w, y + h]]
                try:
                    preds = fan_model.get_landmarks(rgb_frame, detected_faces=det_box)
                    if preds and len(preds) > 0 and preds[0] is not None:
                        landmarks_coords = preds[0]
                        pose_score, pose_metrics = compute_fan_alignment_score(landmarks_coords)
                        aligned_crop, affine_matrix = align_face_crop(frame, landmarks_coords, crop_size=crop_size)
                except Exception as exc:
                    print(f"[!] Worker track #{track_id} FAN error on frame {f_idx}: {exc}")

            # Fallback if FAN is disabled or failed to detect landmarks
            if aligned_crop is None:
                aligned_crop, affine_matrix = get_square_bbox_crop(frame, bbox, crop_size=crop_size)
                pose_score = 0.0

            if aligned_crop is None:
                continue

            sharpness = compute_sharpness(aligned_crop)
            composite_score = compute_composite_score(
                sharpness=sharpness,
                pose_score=pose_score,
                bbox=bbox,
                has_fan=(enable_fan and landmarks_coords is not None)
            )

            candidates.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(t_sec, 3),
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": round(composite_score, 4),
                "sharpness": round(sharpness, 2),
                "pose_score": round(pose_score, 4),
                "pose_metrics": pose_metrics,
                "crop": aligned_crop,
                "affine_matrix": affine_matrix.tolist() if affine_matrix is not None else None,
                "has_landmarks": landmarks_coords is not None
            })

    finally:
        if cap is not None:
            cap.release()

    # Select top K frames enforcing temporal diversity
    top_frames = rank_and_select_top_frames(candidates, top_k=top_k, min_frame_interval=min_interval)

    # Save images and compile metadata
    saved_files = []
    metadata_entries = []

    for item in top_frames:
        rank = item["rank"]
        f_idx = item["frame_idx"]
        score = item["score"]
        img_name = f"rank{rank:02d}_frame{f_idx:04d}_score{score:.2f}.jpg"
        img_path = os.path.join(face_dir, img_name)

        # Write high-quality JPG
        cv2.imwrite(img_path, item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 98])
        saved_files.append(img_path)

        metadata_entries.append({
            "rank": rank,
            "filename": img_name,
            "frame_idx": f_idx,
            "timestamp_sec": item["timestamp_sec"],
            "score": score,
            "sharpness_laplacian": item["sharpness"],
            "pose_score": item["pose_score"],
            "pose_metrics": item["pose_metrics"],
            "bbox": item["bbox"],
            "affine_matrix": item["affine_matrix"],
            "has_fan_landmarks": item["has_landmarks"]
        })

    # Save metadata JSON for downstream restoration pipelines
    meta_path = os.path.join(face_dir, "metadata.json")
    meta_payload = {
        "track_id": track_id,
        "total_detections": len(history),
        "selected_count": len(top_frames),
        "target_crop_size": list(crop_size),
        "fan_enabled": enable_fan,
        "selected_frames": metadata_entries,
        "downstream_context": {
            "purpose": "Normalized facial alignment for degraded face restoration",
            "compatible_models": ["CodeFormer", "GFPGAN", "RestoreFormer", "GPEN"],
            "canonical_alignment": "FFHQ / ArcFace eye standard (eyes level at ~38% height)",
            "inverse_warp_note": "Use 'affine_matrix' with cv2.invertAffineTransform to paste restored face back into frame."
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    return {
        "track_id": track_id,
        "total_detections": len(history),
        "selected_count": len(saved_files),
        "face_dir": face_dir,
        "saved_files": saved_files,
        "metadata_file": meta_path
    }


def process_face_tracks_parallel(
    tracks: List[Any],
    video_path: Optional[str],
    frames_dict: Optional[Dict[int, np.ndarray]],
    config: Dict[str, Any],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Dispatches face tracks to isolated worker processes.
    Manages process pool size according to configuration to avoid high CPU usage.
    """
    if not tracks:
        return []

    mp_cfg = config.get("multiprocessing", {})
    enabled_mp = bool(mp_cfg.get("enabled", True))
    max_workers_cfg = int(mp_cfg.get("max_workers", 2))

    # Cap workers by number of tracks and configuration
    worker_count = max(1, min(len(tracks), max_workers_cfg))

    payloads = [
        {
            "track_id": track.track_id,
            "history": track.history,
            "video_path": video_path,
            "frames_dict": frames_dict,
            "config": config,
            "output_dir": output_dir
        }
        for track in tracks
    ]

    results = []

    # If single track or multiprocessing disabled, run sequentially to avoid IPC overhead
    if not enabled_mp or worker_count <= 1:
        for p in payloads:
            res = _worker_process_single_track(p)
            results.append(res)
    else:
        # Multi-process pool execution
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            future_to_track = {
                executor.submit(_worker_process_single_track, p): p["track_id"]
                for p in payloads
            }
            for future in as_completed(future_to_track):
                t_id = future_to_track[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    print(f"[!] Worker for Face Track #{t_id} generated an exception: {exc}")

    results.sort(key=lambda r: r["track_id"])
    return results

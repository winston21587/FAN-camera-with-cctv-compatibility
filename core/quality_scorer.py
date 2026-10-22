"""
Quality & Alignment Scoring Module
----------------------------------
Quantifies face alignment quality, pose frontality, and image sharpness.
Implements the temporal-diversity selection algorithm to choose the top ~5
best-aligned candidate frames per face identity track.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


def compute_sharpness(image_crop: np.ndarray) -> float:
    """
    Computes image sharpness using the variance of the Laplacian.
    Higher values correspond to crisp details; low values indicate motion blur.
    """
    if image_crop is None or image_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_fan_alignment_score(landmarks: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Computes a pose frontality and alignment score in [0.0, 1.0] from 68-point landmarks.
    
    Metrics:
      1. Yaw symmetry: Ratio of left-eye-to-nose distance vs right-eye-to-nose distance (1.0 = direct frontal).
      2. Roll angle: Deviation of eye-line from horizontal.
      3. Vertical balance: Proportion between eye-to-nose and nose-to-mouth distances.
    """
    if landmarks is None or len(landmarks) < 68:
        return 0.0, {"yaw_ratio": 0.0, "roll_cos": 0.0, "vertical_balance": 0.0}

    # Left eye center (36:42), Right eye center (42:48), Nose tip (30), Mouth center (mean of 48, 54, 51, 57)
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose = landmarks[30]
    mouth = np.mean(landmarks[[48, 51, 54, 57]], axis=0)

    # 1. Roll angle (eye tilt)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_angle_rad = np.arctan2(dy, dx)
    roll_score = max(0.0, float(np.cos(roll_angle_rad)))  # 1.0 when level

    # 2. Yaw symmetry (horizontal angle)
    dist_l_nose = float(np.hypot(nose[0] - left_eye[0], nose[1] - left_eye[1]))
    dist_r_nose = float(np.hypot(nose[0] - right_eye[0], nose[1] - right_eye[1]))
    if max(dist_l_nose, dist_r_nose) > 1e-5:
        yaw_ratio = min(dist_l_nose, dist_r_nose) / max(dist_l_nose, dist_r_nose)
    else:
        yaw_ratio = 0.0

    # 3. Vertical balance (Pitch approximation)
    eye_mid = (left_eye + right_eye) * 0.5
    dist_eye_nose = float(np.abs(nose[1] - eye_mid[1]))
    dist_nose_mouth = float(np.abs(mouth[1] - nose[1]))
    if dist_nose_mouth > 1e-5:
        # Canonical ratio is roughly 1.0 - 1.4 in frontal faces
        ratio_pitch = dist_eye_nose / dist_nose_mouth
        pitch_score = float(np.exp(-0.5 * ((ratio_pitch - 1.1) / 0.5) ** 2))
    else:
        pitch_score = 0.5

    # Combined pose score
    pose_score = float(yaw_ratio * 0.60 + roll_score * 0.25 + pitch_score * 0.15)
    pose_score = max(0.0, min(1.0, pose_score))

    metrics = {
        "yaw_ratio": float(yaw_ratio),
        "roll_cos": float(roll_score),
        "vertical_balance": float(pitch_score)
    }
    return pose_score, metrics


def compute_composite_score(
    sharpness: float,
    pose_score: float,
    bbox: Tuple[int, int, int, int],
    has_fan: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Combines pose frontality, sharpness, and bounding box resolution into a single composite score.
    """
    if weights is None:
        if has_fan:
            weights = {"pose": 0.50, "sharpness": 0.35, "scale": 0.15}
        else:
            weights = {"pose": 0.0, "sharpness": 0.70, "scale": 0.30}

    # Normalize sharpness (Laplacian variance of ~200+ is considered sharp in surveillance)
    norm_sharpness = min(1.0, sharpness / 200.0)

    # Scale score based on pixel size of the face (larger faces have more resolvable details)
    face_diag = np.hypot(bbox[2], bbox[3])
    norm_scale = min(1.0, face_diag / 160.0)

    score = (
        weights.get("pose", 0.0) * pose_score +
        weights.get("sharpness", 0.0) * norm_sharpness +
        weights.get("scale", 0.0) * norm_scale
    )
    return float(max(0.0, min(1.0, score)))


def rank_and_select_top_frames(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    min_frame_interval: int = 5
) -> List[Dict[str, Any]]:
    """
    Ranks candidate frames and selects top_k distinct images enforcing temporal diversity.
    Ensures selected frames are spaced apart by at least `min_frame_interval` frames
    to prevent saving 5 near-identical consecutive frames of motion-stagnant faces.
    """
    if not candidates:
        return []

    # Sort descending by composite score
    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_frame_indices: List[int] = []

    # Pass 1: Select top frames that satisfy the temporal interval constraint
    for cand in sorted_candidates:
        if len(selected) >= top_k:
            break
        f_idx = cand["frame_idx"]
        too_close = any(abs(f_idx - prev) < min_frame_interval for prev in selected_frame_indices)
        if not too_close:
            cand_copy = dict(cand)
            cand_copy["rank"] = len(selected) + 1
            selected.append(cand_copy)
            selected_frame_indices.append(f_idx)

    # Pass 2: If we didn't fill top_k due to the interval constraint, backfill with remaining best
    if len(selected) < top_k:
        for cand in sorted_candidates:
            if len(selected) >= top_k:
                break
            f_idx = cand["frame_idx"]
            if f_idx not in selected_frame_indices:
                cand_copy = dict(cand)
                cand_copy["rank"] = len(selected) + 1
                selected.append(cand_copy)
                selected_frame_indices.append(f_idx)

    return selected

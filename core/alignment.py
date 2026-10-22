"""
Facial Alignment & Normalization Tools
--------------------------------------
Provides canonical face crop alignment using Face Alignment Network (FAN) 68-point landmarks
or aspect-ratio preserving bounding box crops when FAN is disabled.
Outputs standard crops compatible with downstream face restoration models (GFPGAN, CodeFormer, etc.).
"""

from typing import Tuple, Optional, Any
import cv2
import numpy as np

# Optional PyTorch & face_alignment support
try:
    import torch
    import face_alignment
    HAS_FAN_SUPPORT = True
except ImportError:
    HAS_FAN_SUPPORT = False


def init_fan_model(device_preference: str = "cuda") -> Optional[Any]:
    """
    Initializes a 2-D 68-point FaceAlignment model.
    Automatically falls back to CPU if CUDA is unavailable.
    """
    if not HAS_FAN_SUPPORT:
        return None

    use_cuda = (device_preference.lower() == "cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device,
            compile=False
        )
        return fa
    except Exception as exc:
        print(f"[!] FAN initialization failed: {exc}")
        return None


def align_face_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    crop_size: Tuple[int, int] = (512, 512)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Affine-aligns a face based on 68-point landmarks to standard canonical coordinates.
    Standard geometry:
      - Left eye placed at 35% width, 38% height
      - Right eye placed at 65% width, 38% height
    Matches standard FFHQ / ArcFace format expected by face restoration networks (CodeFormer, GFPGAN).

    Returns:
        (aligned_crop, affine_matrix_M)
        affine_matrix_M is the 2x3 transformation matrix (useful for inverse-warping back into the frame).
    """
    if landmarks is None or len(landmarks) < 68 or image is None or image.size == 0:
        return None, None

    # Compute eye centers from 68-point landmarks
    # Left eye: points 36:42, Right eye: points 42:48
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    dX = right_eye[0] - left_eye[0]
    dY = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dY, dX))

    desired_left_eye_x = 0.35
    desired_dist = 1.0 - 2.0 * desired_left_eye_x          # 0.30
    dist = np.hypot(dX, dY)
    scale = (crop_size[0] * desired_dist) / max(dist, 1e-6)

    eye_center = (
        float((left_eye[0] + right_eye[0]) * 0.5),
        float((left_eye[1] + right_eye[1]) * 0.5)
    )

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    # Translate eye center to the canonical target coordinate in the crop
    M[0, 2] += (crop_size[0] * 0.5) - eye_center[0]
    M[1, 2] += (crop_size[1] * 0.38) - eye_center[1]

    aligned = cv2.warpAffine(
        image,
        M,
        (crop_size[0], crop_size[1]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )
    return aligned, M


def get_square_bbox_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    crop_size: Tuple[int, int] = (512, 512),
    margin: float = 1.3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fallback alignment when FAN is turned OFF.
    Extracts an expanded square region centered around the face bounding box.

    Returns:
        (cropped_image, affine_matrix_M)
    """
    if image is None or image.size == 0 or bbox is None:
        return None, None

    ih, iw = image.shape[:2]
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.5

    # Target square side length with margin expansion
    side = max(w, h) * margin

    x1 = int(round(cx - side * 0.5))
    y1 = int(round(cy - side * 0.5))
    x2 = int(round(cx + side * 0.5))
    y2 = int(round(cy + side * 0.5))

    # Pad if outside boundaries
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - iw)
    pad_bottom = max(0, y2 - ih)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_img = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        crop = padded_img[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_CUBIC)

    # Affine matrix mapping original image coords to resized crop coords
    scale_x = crop_size[0] / max(side, 1e-6)
    scale_y = crop_size[1] / max(side, 1e-6)
    M = np.array([
        [scale_x, 0.0, -x1 * scale_x],
        [0.0, scale_y, -y1 * scale_y]
    ], dtype=np.float32)

    return resized, M

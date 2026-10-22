"""
Core modules for CCTV snippet face detection, tracking, alignment, and restoration prep.
"""

from .tracker import FaceTracker, FaceTrack
from .alignment import align_face_crop, get_square_bbox_crop
from .quality_scorer import compute_sharpness, compute_fan_alignment_score, rank_and_select_top_frames
from .worker_pool import process_face_tracks_parallel

__all__ = [
    "FaceTracker",
    "FaceTrack",
    "align_face_crop",
    "get_square_bbox_crop",
    "compute_sharpness",
    "compute_fan_alignment_score",
    "rank_and_select_top_frames",
    "process_face_tracks_parallel"
]

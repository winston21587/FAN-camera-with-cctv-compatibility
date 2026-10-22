"""
Lightweight Inter-Frame Face Tracker for CCTV Video Clips
---------------------------------------------------------
Associates face detections across consecutive frames using Intersection-over-Union (IoU)
and centroid proximity. Assigns persistent track IDs so multiple faces appearing
in a 5-second snippet are cleanly partitioned into distinct identity tracks.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h).
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def compute_centroid_distance(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Computes Euclidean distance between the center points of two boxes.
    """
    cx_a = box_a[0] + box_a[2] * 0.5
    cy_a = box_a[1] + box_a[3] * 0.5
    cx_b = box_b[0] + box_b[2] * 0.5
    cy_b = box_b[1] + box_b[3] * 0.5
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


class FaceTrack:
    """
    Represents an individual face identity tracked across multiple frames.
    """
    def __init__(self, track_id: int, initial_frame: int, timestamp_sec: float, bbox: Tuple[int, int, int, int]):
        self.track_id: int = track_id
        self.last_bbox: Tuple[int, int, int, int] = bbox
        self.hits: int = 1
        self.time_since_update: int = 0
        # List of candidate frame occurrences:
        self.history: List[Dict[str, Any]] = [
            {
                "frame_idx": initial_frame,
                "timestamp_sec": timestamp_sec,
                "bbox": bbox
            }
        ]

    def add_detection(self, frame_idx: int, timestamp_sec: float, bbox: Tuple[int, int, int, int]) -> None:
        self.last_bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "bbox": bbox
        })

    def mark_missed(self) -> None:
        self.time_since_update += 1


class FaceTracker:
    """
    Multi-face tracker maintaining identity tracks across a clip duration (~5s, ~150 frames).
    """
    def __init__(self, max_lost: int = 15, min_iou: float = 0.20, min_hits: int = 3):
        """
        :param max_lost: Maximum consecutive frames a face can be missed before track termination.
        :param min_iou: Minimum IoU overlap required to associate a detection with an existing track.
        :param min_hits: Minimum number of detections required to consider a track valid (filters noise).
        """
        self.max_lost = max_lost
        self.min_iou = min_iou
        self.min_hits = min_hits
        self.next_track_id: int = 1
        self.active_tracks: List[FaceTrack] = []
        self.finished_tracks: List[FaceTrack] = []

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Updates tracks with current frame detections.
        Returns list of (track_id, bbox) for current frame.
        """
        matched_tracks = set()
        matched_detections = set()
        matches: List[Tuple[int, Tuple[int, int, int, int]]] = []

        if self.active_tracks and detections:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(self.active_tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.active_tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(track.last_bbox, det)

            # Greedy matching on highest IoU first
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for flat_idx in flat_indices:
                t_idx = flat_idx // len(detections)
                d_idx = flat_idx % len(detections)
                score = iou_matrix[t_idx, d_idx]

                if score < self.min_iou:
                    # Also check proximity centroid distance if box size is similar
                    track = self.active_tracks[t_idx]
                    det = detections[d_idx]
                    dist = compute_centroid_distance(track.last_bbox, det)
                    avg_diag = (track.last_bbox[2] + track.last_bbox[3] + det[2] + det[3]) * 0.5
                    if dist > avg_diag * 0.75:
                        continue

                if t_idx in matched_tracks or d_idx in matched_detections:
                    continue

                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                track = self.active_tracks[t_idx]
                det = detections[d_idx]
                track.add_detection(frame_idx, timestamp_sec, det)
                matches.append((track.track_id, det))

        # Unmatched detections become new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = FaceTrack(
                    track_id=self.next_track_id,
                    initial_frame=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=det
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)
                matches.append((new_track.track_id, det))

        # Update unmatched active tracks
        retained_tracks: List[FaceTrack] = []
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_tracks and (len(self.active_tracks) == len(retained_tracks) or t_idx < len(self.active_tracks)):
                # If it wasn't matched in the greedily matched set
                if t_idx not in matched_tracks and track.history[-1]["frame_idx"] != frame_idx:
                    track.mark_missed()

            if track.time_since_update > self.max_lost:
                if track.hits >= self.min_hits:
                    self.finished_tracks.append(track)
            else:
                retained_tracks.append(track)

        self.active_tracks = retained_tracks
        return matches

    def finalize(self) -> List[FaceTrack]:
        """
        Concludes tracking at the end of the video clip.
        Merges remaining active tracks that meet min_hits requirement.
        """
        all_tracks = list(self.finished_tracks)
        for track in self.active_tracks:
            if track.hits >= self.min_hits and track not in all_tracks:
                all_tracks.append(track)

        # Sort tracks by number of detections (highest presence first)
        all_tracks.sort(key=lambda t: t.hits, reverse=True)
        return all_tracks

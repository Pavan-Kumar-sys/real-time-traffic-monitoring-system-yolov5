import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_name: str
    confidence: float
    true_speed_kmh: Optional[float] = None
    embedding: Optional[np.ndarray] = None  # Normalized RGB/HSV appearance feature vector


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    created_at: float
    last_seen: float
    hits: int = 1
    missed: int = 0
    vx: float = 0.0  # Kalman velocity X (px/sec)
    vy: float = 0.0  # Kalman velocity Y (px/sec)
    speed_kmh: float = 0.0
    lane: int = 1
    direction: str = "Outbound"
    counted: bool = False
    embedding: Optional[np.ndarray] = None
    class_votes: Dict[str, float] = field(default_factory=dict)
    flagged_violations: Set[str] = field(default_factory=set)
    history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=35)
    )
    world_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=25)
    )  # Homography bird's-eye (X_meters, Y_meters, timestamp)

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


def cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    if vec_a is None or vec_b is None:
        return 0.5
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return 0.5
    return float(np.dot(vec_a, vec_b) / denom)


class PerspectiveHomographyCalibrator:
    """
    4-Point Perspective Homography Calibrator:
    Maps a perspective road trapezoid in image pixels (1280x720) to a bird's-eye
    metric coordinate plane (14.0m wide 4-lane highway x 28.0m longitudinal depth).
    Solves the 3x3 homography matrix H via Direct Linear Transform (SVD).
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        road_width_m: float = 14.0,
        road_depth_m: float = 28.0,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.road_width_m = road_width_m
        self.road_depth_m = road_depth_m
        self.src_quad = self.get_default_quad(frame_width, frame_height)
        self.dst_quad = np.array(
            [
                [0.0, 0.0],
                [road_width_m, 0.0],
                [road_width_m, road_depth_m],
                [0.0, road_depth_m],
            ],
            dtype=np.float64,
        )
        self.H = self._compute_homography(self.src_quad, self.dst_quad)

    @staticmethod
    def get_default_quad(w: int, h: int) -> np.ndarray:
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left of perspective road zone
        return np.array(
            [
                [w * 0.28, h * 0.22],
                [w * 0.72, h * 0.22],
                [w * 0.88, h * 0.88],
                [w * 0.12, h * 0.88],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        A = []
        for i in range(4):
            x, y = src[i][0], src[i][1]
            u, v = dst[i][0], dst[i][1]
            A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        A_mat = np.asarray(A, dtype=np.float64)
        _, _, vh = np.linalg.svd(A_mat)
        H = vh[-1, :].reshape((3, 3))
        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]
        return H

    def pixel_to_meters(self, px: float, py: float) -> Tuple[float, float]:
        vec = np.array([px, py, 1.0], dtype=np.float64)
        mapped = self.H @ vec
        w = mapped[2] if abs(mapped[2]) > 1e-6 else 1e-6
        return (float(mapped[0] / w), float(mapped[1] / w))


class MultiObjectTracker:
    """
    ByteTrack Two-Stage Association + Appearance Re-ID + 4-Point Homography Speed Tracker:
    - Stage 1 associates high-confidence detections (conf >= 0.45) using IoU + Kalman velocity prediction + Appearance Re-ID
    - Stage 2 associates low-confidence/occluded detections (0.18 <= conf < 0.45) to remaining active tracks
    - Computes bird's-eye metric velocity (km/h) via 4-Point Homography transformation
    """

    def __init__(
        self,
        iou_threshold: float = 0.22,
        max_missed_frames: int = 15,
        pixels_per_meter: float = 8.5,
        speed_limit_kmh: float = 80.0,
        counting_line_y: float = 0.62,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.pixels_per_meter = pixels_per_meter
        self.speed_limit_kmh = speed_limit_kmh
        self.counting_line_y = counting_line_y
        self.homography = PerspectiveHomographyCalibrator()

        self.next_track_id: int = 101
        self.tracks: Dict[int, Track] = {}

        self.total_counted: int = 0
        self.inbound_count: int = 0
        self.outbound_count: int = 0
        self.class_counts: Dict[str, int] = {
            "Car": 0,
            "Auto-Rickshaw": 0,
            "Motorcycle": 0,
            "Truck": 0,
            "Bus": 0,
            "Bicycle": 0,
            "Pedestrian": 0,
        }
        self.crossing_timestamps: Deque[float] = deque(maxlen=600)

    def reset_counts(self) -> None:
        self.total_counted = 0
        self.inbound_count = 0
        self.outbound_count = 0
        for k in self.class_counts:
            self.class_counts[k] = 0
        self.crossing_timestamps.clear()
        self.tracks.clear()

    def _assign_lane(self, cx: float, frame_width: int) -> int:
        road_left = frame_width * 0.14
        road_right = frame_width * 0.86
        road_w = max(1.0, road_right - road_left)
        rel_x = max(0.0, min(0.999, (cx - road_left) / road_w))
        return int(rel_x * 4) + 1

    def _predict_bbox(self, tr: Track, now: float) -> Tuple[float, float, float, float]:
        dt = max(0.0, min(0.25, now - tr.last_seen))
        if dt <= 1e-4 or (abs(tr.vx) < 0.5 and abs(tr.vy) < 0.5):
            return tr.bbox
        dx = tr.vx * dt
        dy = tr.vy * dt
        x1, y1, x2, y2 = tr.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _estimate_speed_homography_kmh(
        self,
        track: Track,
        frame_height: int,
        hint_speed: Optional[float] = None,
    ) -> float:
        if hint_speed is not None and hint_speed >= 0:
            if track.speed_kmh == 0.0:
                return hint_speed
            return 0.75 * track.speed_kmh + 0.25 * hint_speed

        if len(track.world_history) < 3:
            return track.speed_kmh

        wx0, wy0, t0 = track.world_history[0]
        wx1, wy1, t1 = track.world_history[-1]
        dt = t1 - t0
        if dt <= 0.03:
            return track.speed_kmh

        # Bird's-eye Euclidean displacement in meters via Homography matrix H
        meters_homography = math.hypot(wx1 - wx0, wy1 - wy0)

        # Blend with perspective-scaled pixel displacement for stability near image edges
        x0, y0, _ = track.history[0]
        x1, y1, _ = track.history[-1]
        px_dist = math.hypot(x1 - x0, y1 - y0)
        norm_y = max(0.18, min(1.0, ((y0 + y1) * 0.5) / max(1, frame_height)))
        meters_persp = (px_dist * (1.42 - 0.62 * norm_y)) / max(0.1, self.pixels_per_meter)

        meters = 0.65 * meters_homography + 0.35 * meters_persp
        raw_kmh = max(0.0, min(200.0, (meters / dt) * 3.6))

        if track.speed_kmh <= 0.1:
            return raw_kmh
        return 0.72 * track.speed_kmh + 0.28 * raw_kmh

    def _associate_stage(
        self,
        candidate_track_ids: Set[int],
        candidate_det_indices: List[int],
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
        min_iou: float,
    ) -> List[Tuple[int, int]]:
        pairs: List[Tuple[float, int, int]] = []
        max_gate = max(frame_width, frame_height) * 0.13

        for tid in candidate_track_ids:
            tr = self.tracks[tid]
            pred_box = self._predict_bbox(tr, now)
            tcx = (pred_box[0] + pred_box[2]) * 0.5
            tcy = (pred_box[1] + pred_box[3]) * 0.5

            for did in candidate_det_indices:
                det = detections[did]
                d_box = det.bbox
                dcx = (d_box[0] + d_box[2]) * 0.5
                dcy = (d_box[1] + d_box[3]) * 0.5

                iou = compute_iou(pred_box, d_box)
                center_dist = math.hypot(tcx - dcx, tcy - dcy)
                reid_sim = cosine_similarity(tr.embedding, det.embedding)

                if iou >= min_iou or (center_dist < max_gate and reid_sim >= 0.45):
                    dist_score = max(0.0, 1.0 - center_dist / max_gate)
                    total_score = 0.55 * iou + 0.25 * dist_score + 0.20 * reid_sim
                    pairs.append((total_score, tid, did))

        pairs.sort(reverse=True, key=lambda x: x[0])
        matched: List[Tuple[int, int]] = []
        used_dets: Set[int] = set()

        for _, tid, did in pairs:
            if tid in candidate_track_ids and did not in used_dets:
                matched.append((tid, did))
                candidate_track_ids.remove(tid)
                used_dets.add(did)

        for did in used_dets:
            if did in candidate_det_indices:
                candidate_det_indices.remove(did)

        return matched

    def update(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        now: Optional[float] = None,
    ) -> Tuple[List[Track], List[Dict[str, Any]]]:
        ts = now if now is not None else time.time()
        line_px_y = self.counting_line_y * frame_height
        new_violations: List[Dict[str, Any]] = []

        if (
            self.homography.frame_width != frame_width
            or self.homography.frame_height != frame_height
        ):
            self.homography = PerspectiveHomographyCalibrator(frame_width, frame_height)

        unmatched_tracks = set(self.tracks.keys())
        high_conf_dets = [i for i, d in enumerate(detections) if d.confidence >= 0.42]
        low_conf_dets = [i for i, d in enumerate(detections) if d.confidence < 0.42]

        # ByteTrack Stage 1: High-confidence detections
        matched_stage1 = self._associate_stage(
            unmatched_tracks,
            high_conf_dets,
            detections,
            frame_width,
            frame_height,
            ts,
            min_iou=self.iou_threshold,
        )

        # ByteTrack Stage 2: Recover occluded tracks with low-confidence detections
        matched_stage2 = self._associate_stage(
            unmatched_tracks,
            low_conf_dets,
            detections,
            frame_width,
            frame_height,
            ts,
            min_iou=0.14,
        )

        matched_pairs = matched_stage1 + matched_stage2

        for tid, did in matched_pairs:
            det = detections[did]
            tr = self.tracks[tid]
            prev_cx, prev_cy = tr.centroid if tr.history else (None, None)
            dt = max(0.016, ts - tr.last_seen)

            # Kalman-style smoothed bounding box & velocity update
            tr.bbox = tuple(0.30 * tr.bbox[k] + 0.70 * det.bbox[k] for k in range(4))  # type: ignore
            cx, cy = tr.centroid
            if prev_cx is not None and prev_cy is not None:
                inst_vx = (cx - prev_cx) / dt
                inst_vy = (cy - prev_cy) / dt
                tr.vx = 0.65 * tr.vx + 0.35 * inst_vx
                tr.vy = 0.65 * tr.vy + 0.35 * inst_vy

            # Temporal class voting so a single noisy frame never flips Auto-Rickshaw to Truck
            vote_weight = det.confidence * (1.35 if det.class_name == "Auto-Rickshaw" else 1.0)
            tr.class_votes[det.class_name] = tr.class_votes.get(det.class_name, 0.0) + vote_weight
            tr.class_name = max(tr.class_votes.items(), key=lambda kv: kv[1])[0]

            tr.confidence = det.confidence
            tr.last_seen = ts
            tr.hits += 1
            tr.missed = 0

            if det.embedding is not None:
                if tr.embedding is None:
                    tr.embedding = det.embedding
                else:
                    tr.embedding = 0.75 * tr.embedding + 0.25 * det.embedding

            tr.history.append((cx, cy, ts))
            wx, wy = self.homography.pixel_to_meters(cx, tr.bbox[3])
            tr.world_history.append((wx, wy, ts))

            tr.lane = self._assign_lane(cx, frame_width)
            tr.speed_kmh = self._estimate_speed_homography_kmh(
                tr, frame_height, det.true_speed_kmh
            )

            if len(tr.history) >= 2:
                dy = tr.history[-1][1] - tr.history[0][1]
                tr.direction = "Approaching" if dy > 0 else "Receding"

            if not tr.counted and prev_cy is not None:
                crossed_down = prev_cy < line_px_y <= cy
                crossed_up = prev_cy > line_px_y >= cy
                if crossed_down or crossed_up:
                    tr.counted = True
                    self.total_counted += 1
                    if crossed_down:
                        self.inbound_count += 1
                    else:
                        self.outbound_count += 1
                    self.class_counts[tr.class_name] = (
                        self.class_counts.get(tr.class_name, 0) + 1
                    )
                    self.crossing_timestamps.append(ts)

            if tr.hits >= 4:
                if (
                    tr.speed_kmh > self.speed_limit_kmh * 1.08
                    and "OVERSPEED" not in tr.flagged_violations
                ):
                    tr.flagged_violations.add("OVERSPEED")
                    excess = tr.speed_kmh - self.speed_limit_kmh
                    severity = "CRITICAL" if excess >= 25.0 else "HIGH"
                    new_violations.append(
                        {
                            "track_id": tr.track_id,
                            "vehicle_class": tr.class_name,
                            "violation_type": "OVERSPEEDING",
                            "speed_kmh": round(tr.speed_kmh, 1),
                            "speed_limit_kmh": self.speed_limit_kmh,
                            "lane": tr.lane,
                            "severity": severity,
                            "details": f"{tr.class_name} #{tr.track_id} clocked at {tr.speed_kmh:.1f} km/h in {self.speed_limit_kmh:.0f} km/h zone (Lane {tr.lane})",
                        }
                    )

                if len(tr.history) >= 6 and "WRONG_WAY" not in tr.flagged_violations:
                    total_dy = tr.history[-1][1] - tr.history[0][1]
                    if (tr.lane in (1, 2) and total_dy < -28.0) or (
                        tr.lane in (3, 4) and total_dy > 28.0
                    ):
                        tr.flagged_violations.add("WRONG_WAY")
                        new_violations.append(
                            {
                                "track_id": tr.track_id,
                                "vehicle_class": tr.class_name,
                                "violation_type": "WRONG_WAY",
                                "speed_kmh": round(tr.speed_kmh, 1),
                                "speed_limit_kmh": self.speed_limit_kmh,
                                "lane": tr.lane,
                                "severity": "CRITICAL",
                                "details": f"Counter-flow trajectory detected for {tr.class_name} #{tr.track_id} in Lane {tr.lane}",
                            }
                        )

                if (
                    tr.hits >= 25
                    and tr.speed_kmh < 4.0
                    and "STALLED" not in tr.flagged_violations
                ):
                    tr.flagged_violations.add("STALLED")
                    new_violations.append(
                        {
                            "track_id": tr.track_id,
                            "vehicle_class": tr.class_name,
                            "violation_type": "STALLED_VEHICLE",
                            "speed_kmh": round(tr.speed_kmh, 1),
                            "speed_limit_kmh": self.speed_limit_kmh,
                            "lane": tr.lane,
                            "severity": "MEDIUM",
                            "details": f"Stationary {tr.class_name} #{tr.track_id} obstructing Lane {tr.lane}",
                        }
                    )

        # Spawn new tracks only from unmatched high-confidence detections
        for did in high_conf_dets:
            det = detections[did]
            tid = self.next_track_id
            self.next_track_id += 1
            cx = (det.bbox[0] + det.bbox[2]) * 0.5
            cy = (det.bbox[1] + det.bbox[3]) * 0.5
            lane = self._assign_lane(cx, frame_width)
            initial_speed = det.true_speed_kmh if det.true_speed_kmh is not None else 0.0
            tr = Track(
                track_id=tid,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
                created_at=ts,
                last_seen=ts,
                speed_kmh=initial_speed,
                lane=lane,
                embedding=det.embedding,
                class_votes={det.class_name: det.confidence},
            )
            tr.history.append((cx, cy, ts))
            wx, wy = self.homography.pixel_to_meters(cx, det.bbox[3])
            tr.world_history.append((wx, wy, ts))
            self.tracks[tid] = tr

        to_delete: List[int] = []
        for tid in unmatched_tracks:
            tr = self.tracks[tid]
            tr.missed += 1
            tr.bbox = self._predict_bbox(tr, ts)
            if tr.missed > self.max_missed_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        active_tracks = [t for t in self.tracks.values() if t.missed <= 3]
        return active_tracks, new_violations

    def get_summary_metrics(self, now: Optional[float] = None) -> Dict[str, Any]:
        ts = now if now is not None else time.time()
        while self.crossing_timestamps and (ts - self.crossing_timestamps[0]) > 60.0:
            self.crossing_timestamps.popleft()

        active = [t for t in self.tracks.values() if t.missed <= 3]
        speeds = [t.speed_kmh for t in active if t.speed_kmh > 1.0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        lane_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for t in active:
            lane_counts[t.lane] = lane_counts.get(t.lane, 0) + 1

        density = len(active)
        if density >= 14 or (density >= 8 and avg_speed < 30.0):
            congestion = "HIGH"
        elif density >= 8 or (density >= 5 and avg_speed < 45.0):
            congestion = "MODERATE"
        else:
            congestion = "LOW"

        return {
            "active_vehicles": density,
            "total_counted": self.total_counted,
            "inbound_count": self.inbound_count,
            "outbound_count": self.outbound_count,
            "avg_speed_kmh": round(avg_speed, 1),
            "max_speed_kmh": round(max_speed, 1),
            "flow_rate_vpm": len(self.crossing_timestamps),
            "congestion_level": congestion,
            "class_counts": dict(self.class_counts),
            "lane_occupancy": lane_counts,
        }

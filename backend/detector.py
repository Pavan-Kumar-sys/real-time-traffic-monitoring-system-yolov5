import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import CLASS_COLORS_BGR, YOLO_TRAFFIC_CLASSES
from backend.tracker import Detection, PerspectiveHomographyCalibrator, Track


def compute_appearance_embedding(
    frame: np.ndarray, bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Computes a 16-dimensional normalized HSV/RGB color appearance descriptor for
    DeepSORT / ByteTrack Re-Identification across temporary vehicle occlusions.
    """
    h_f, w_f = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f, x2), min(h_f, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return np.ones(16, dtype=np.float32) / 4.0

    roi = cv2.resize(frame[y1:y2, x1:x2], (16, 16))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else vec


def refine_vehicle_class(
    raw_class: str,
    bbox: Tuple[float, float, float, float],
    frame: np.ndarray,
) -> str:
    """
    IDD (Indian Driving Dataset) Multi-Feature Classifier:
    Separates Auto-Rickshaws (3-wheelers) from Heavy Trucks, Buses, and Cars using:
    1. Perspective-normalized bounding box footprint at vertical road depth y
    2. Split-body canopy vs. chassis color analysis (black/yellow/green/white 3-wheeler roofs)
    3. Compact upright aspect ratio signature
    """
    if raw_class not in ("Truck", "Car", "Bus"):
        return raw_class

    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cy = (y1 + y2) * 0.5
    aspect = bw / bh

    depth_scale = max(0.28, (cy / max(1.0, float(h_frame))) ** 1.35)
    norm_area = (bw * bh) / (depth_scale * float(w_frame * h_frame))

    ix1, iy1 = max(0, int(x1 + bw * 0.14)), max(0, int(y1 + bh * 0.10))
    ix2, iy2 = min(w_frame, int(x2 - bw * 0.14)), min(h_frame, int(y1 + bh * 0.78))
    has_rickshaw_palette = False
    if ix2 > ix1 + 4 and iy2 > iy1 + 4:
        roi = frame[iy1:iy2, ix1:ix2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 70, 65), (42, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 60, 45), (88, 255, 255))
        cyan_mask = cv2.inRange(hsv, (85, 60, 60), (115, 255, 255))
        ratio_color = float(
            np.count_nonzero(yellow_mask)
            + np.count_nonzero(green_mask)
            + np.count_nonzero(cyan_mask)
        ) / float(max(1, roi.shape[0] * roi.shape[1]))
        if ratio_color >= 0.15:
            has_rickshaw_palette = True

    if raw_class == "Truck":
        if norm_area < 0.050:
            if 0.55 <= aspect <= 1.40 or has_rickshaw_palette:
                return "Auto-Rickshaw"
            return "Car"

    if raw_class == "Car":
        if has_rickshaw_palette and norm_area < 0.040 and 0.56 <= aspect <= 1.35:
            return "Auto-Rickshaw"
        if norm_area < 0.020 and 0.65 <= aspect <= 1.18:
            return "Auto-Rickshaw"

    return raw_class


@dataclass
class SimVehicle:
    uid: int
    class_name: str
    lane: int
    x: float
    y: float
    width: float
    height: float
    speed_kmh: float
    confidence: float
    color_bgr: Tuple[int, int, int]
    wrong_way: bool = False


class TrafficSceneSimulator:
    CLASS_SPECS = {
        "Car": (52, 84, (62.0, 94.0)),
        "Auto-Rickshaw": (44, 64, (38.0, 58.0)),
        "Motorcycle": (28, 54, (55.0, 95.0)),
        "Truck": (68, 134, (48.0, 74.0)),
        "Bus": (66, 128, (50.0, 78.0)),
    }

    def __init__(self, width: int = 1280, height: int = 720, mode: str = "highway") -> None:
        self.width = width
        self.height = height
        self.mode = mode
        self.vehicles: List[SimVehicle] = []
        self.next_uid = 1
        self.last_step = time.time()
        self.frame_idx = 0
        self._seed_initial_traffic()

    def _lane_center_x(self, lane: int) -> float:
        road_left = self.width * 0.16
        road_right = self.width * 0.84
        lane_w = (road_right - road_left) / 4.0
        return road_left + (lane - 0.5) * lane_w

    def _seed_initial_traffic(self) -> None:
        for i in range(9):
            lane = (i % 4) + 1
            y = random.uniform(80, self.height - 80)
            self._spawn_vehicle(lane=lane, y_override=y)

    def _spawn_vehicle(
        self,
        lane: Optional[int] = None,
        y_override: Optional[float] = None,
        force_anomaly: Optional[str] = None,
    ) -> None:
        if lane is None:
            lane = random.randint(1, 4)

        cls = random.choices(
            ["Car", "Auto-Rickshaw", "Motorcycle", "Truck", "Bus"],
            weights=[0.46, 0.20, 0.16, 0.10, 0.08],
            k=1,
        )[0]
        base_w, base_h, (min_spd, max_spd) = self.CLASS_SPECS[cls]
        speed_scale = 0.68 if self.mode == "urban" else 1.0
        speed_kmh = random.uniform(min_spd * speed_scale, max_spd * speed_scale)

        wrong_way = False
        if force_anomaly == "overspeed" or random.random() < 0.12:
            speed_kmh *= random.uniform(1.22, 1.42)
        elif force_anomaly == "wrong_way":
            wrong_way = True

        moving_down = (lane in (1, 2)) ^ wrong_way
        y = (
            y_override
            if y_override is not None
            else (-base_h * 0.6 if moving_down else self.height + base_h * 0.6)
        )
        cx = self._lane_center_x(lane) + random.uniform(-8.0, 8.0)

        for v in self.vehicles:
            if v.lane == lane and abs(v.y - y) < (v.height + base_h) * 0.85:
                return

        color_bgr = (
            (20, 195, 245)
            if cls == "Auto-Rickshaw"
            else random.choice(
                [
                    (210, 210, 215),
                    (70, 90, 200),
                    (190, 130, 50),
                    (60, 170, 110),
                    (90, 95, 105),
                ]
            )
        )
        self.vehicles.append(
            SimVehicle(
                uid=self.next_uid,
                class_name=cls,
                lane=lane,
                x=cx,
                y=y,
                width=base_w,
                height=base_h,
                speed_kmh=speed_kmh,
                confidence=round(random.uniform(0.88, 0.98), 2),
                color_bgr=color_bgr,
                wrong_way=wrong_way,
            )
        )
        self.next_uid += 1

    def trigger_anomaly(self, anomaly_type: str = "overspeed") -> None:
        self._spawn_vehicle(lane=random.randint(1, 4), force_anomaly=anomaly_type)

    def step(self) -> Tuple[np.ndarray, List[Detection]]:
        now = time.time()
        dt = min(0.08, max(0.015, now - self.last_step))
        self.last_step = now
        self.frame_idx += 1

        frame = np.full((self.height, self.width, 3), (28, 31, 36), dtype=np.uint8)
        road_left = int(self.width * 0.16)
        road_right = int(self.width * 0.84)

        cv2.rectangle(frame, (0, 0), (road_left, self.height), (22, 38, 28), -1)
        cv2.rectangle(frame, (road_right, 0), (self.width, self.height), (22, 38, 28), -1)
        cv2.rectangle(frame, (road_left, 0), (road_right, self.height), (42, 46, 52), -1)

        cv2.line(frame, (road_left + 4, 0), (road_left + 4, self.height), (210, 215, 220), 3)
        cv2.line(frame, (road_right - 4, 0), (road_right - 4, self.height), (210, 215, 220), 3)

        mid_x = (road_left + road_right) // 2
        cv2.line(frame, (mid_x - 4, 0), (mid_x - 4, self.height), (40, 195, 245), 2)
        cv2.line(frame, (mid_x + 4, 0), (mid_x + 4, self.height), (40, 195, 245), 2)

        lane_w = (road_right - road_left) / 4.0
        for lane_div in (1, 3):
            lx = int(road_left + lane_div * lane_w)
            for y in range(0, self.height, 64):
                cv2.line(frame, (lx, y), (lx, min(self.height, y + 36)), (160, 168, 175), 2)

        if len(self.vehicles) < 12 and random.random() < 0.22:
            self._spawn_vehicle()

        survivors: List[SimVehicle] = []
        detections: List[Detection] = []

        for v in self.vehicles:
            moving_down = (v.lane in (1, 2)) ^ v.wrong_way
            dir_sign = 1.0 if moving_down else -1.0
            px_per_sec = (v.speed_kmh / 3.6) * 8.5
            v.y += dir_sign * px_per_sec * dt

            if -v.height <= v.y <= self.height + v.height:
                survivors.append(v)
                x1 = int(v.x - v.width * 0.5)
                y1 = int(v.y - v.height * 0.5)
                x2 = int(v.x + v.width * 0.5)
                y2 = int(v.y + v.height * 0.5)

                cv2.rectangle(frame, (x1 + 5, y1 + 6), (x2 + 5, y2 + 6), (18, 20, 24), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), v.color_bgr, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 35), 2)

                inset_x = max(4, int(v.width * 0.16))
                inset_y = max(8, int(v.height * 0.22))
                cv2.rectangle(
                    frame,
                    (x1 + inset_x, y1 + inset_y),
                    (x2 - inset_x, y2 - inset_y),
                    (45, 52, 62),
                    -1,
                )

                bbox_t = (
                    float(max(0, x1)),
                    float(max(0, y1)),
                    float(min(self.width - 1, x2)),
                    float(min(self.height - 1, y2)),
                )
                detections.append(
                    Detection(
                        bbox=bbox_t,
                        class_name=v.class_name,
                        confidence=v.confidence,
                        true_speed_kmh=v.speed_kmh,
                        embedding=compute_appearance_embedding(frame, bbox_t),
                    )
                )

        self.vehicles = survivors
        return frame, detections


class TrafficDetectorEngine:
    def __init__(self, model_name: str = "yolov5su.pt") -> None:
        self.model_name = model_name
        self.yolo_model = None
        self.yolo_backend: str = "none"
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=36, detectShadows=True
        )
        self.simulators: Dict[str, TrafficSceneSimulator] = {
            "sim:highway": TrafficSceneSimulator(mode="highway"),
            "sim:urban": TrafficSceneSimulator(mode="urban"),
        }

        self.show_trails: bool = True
        self.show_speed_tags: bool = True
        self.show_zones: bool = True

    def _ensure_yolov5_loaded(self) -> str:
        if self.yolo_backend != "none":
            return self.yolo_backend
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(self.model_name)
            self.yolo_backend = "ultralytics"
            return self.yolo_backend
        except Exception:
            pass

        try:
            import torch

            self.yolo_model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )
            self.yolo_backend = "torch_hub"
            return self.yolo_backend
        except Exception:
            self.yolo_backend = "opencv_mog2"
            return self.yolo_backend

    def detect_on_frame(self, frame: np.ndarray) -> List[Detection]:
        backend = self._ensure_yolov5_loaded()
        detections: List[Detection] = []

        if backend == "ultralytics" and self.yolo_model is not None:
            # Run inference at imgsz=480 for low-latency real-time camera tracking
            results = self.yolo_model.predict(
                frame, conf=0.22, imgsz=480, verbose=False
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    raw_cls = YOLO_TRAFFIC_CLASSES.get(cls_id)
                    if raw_cls is None:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_t = (float(x1), float(y1), float(x2), float(y2))
                    refined_cls = refine_vehicle_class(raw_cls, bbox_t, frame)
                    conf = float(box.conf[0].item())
                    detections.append(
                        Detection(
                            bbox=bbox_t,
                            class_name=refined_cls,
                            confidence=round(conf, 2),
                            embedding=compute_appearance_embedding(frame, bbox_t),
                        )
                    )
            return detections

        if backend == "torch_hub" and self.yolo_model is not None:
            results = self.yolo_model(frame, size=480)
            preds = results.xyxy[0].cpu().numpy()
            for row in preds:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                if conf < 0.22:
                    continue
                raw_cls = YOLO_TRAFFIC_CLASSES.get(int(cls_id))
                if raw_cls is None:
                    continue
                bbox_t = (float(x1), float(y1), float(x2), float(y2))
                refined_cls = refine_vehicle_class(raw_cls, bbox_t, frame)
                detections.append(
                    Detection(
                        bbox=bbox_t,
                        class_name=refined_cls,
                        confidence=round(float(conf), 2),
                        embedding=compute_appearance_embedding(frame, bbox_t),
                    )
                )
            return detections

        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 900:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_t = (float(x), float(y), float(x + w), float(y + h))
            raw_cls = "Truck" if area > 28000 else ("Car" if area > 4200 else "Motorcycle")
            refined_cls = refine_vehicle_class(raw_cls, bbox_t, frame)
            detections.append(
                Detection(
                    bbox=bbox_t,
                    class_name=refined_cls,
                    confidence=0.88,
                    embedding=compute_appearance_embedding(frame, bbox_t),
                )
            )
        return detections

    def render_hud(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        counting_line_y: float,
        speed_limit_kmh: float,
        camera_name: str,
        fps: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        if self.show_zones:
            # 4-Point Perspective Homography Calibration Zone
            quad = PerspectiveHomographyCalibrator.get_default_quad(w, h).astype(
                np.int32
            )
            cv2.polylines(annotated, [quad], True, (227, 113, 0), 2, cv2.LINE_AA)

            line_y = int(counting_line_y * h)
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (int(w * 0.08), line_y - 10),
                (int(w * 0.92), line_y + 10),
                (227, 113, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
            cv2.line(
                annotated,
                (int(w * 0.08), line_y),
                (int(w * 0.92), line_y),
                (255, 255, 255),
                2,
            )

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            is_speeding = tr.speed_kmh > speed_limit_kmh * 1.08
            is_wrong_way = "WRONG_WAY" in tr.flagged_violations
            color = (
                (48, 59, 255)
                if (is_speeding or is_wrong_way)
                else CLASS_COLORS_BGR.get(tr.class_name, (227, 113, 0))
            )

            if self.show_trails and len(tr.history) >= 2:
                pts = np.array(
                    [[int(px), int(py)] for px, py, _ in tr.history], dtype=np.int32
                )
                cv2.polylines(annotated, [pts], False, color, 2, cv2.LINE_AA)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if self.show_speed_tags:
                tag = f"#{tr.track_id} {tr.class_name} | {tr.speed_kmh:.0f} km/h"
                if is_wrong_way:
                    tag += " [WRONG WAY]"
                elif is_speeding:
                    tag += " [FAST]"
            else:
                tag = f"#{tr.track_id} {tr.class_name} ({int(tr.confidence * 100)}%)"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            label_y = max(th + 8, y1 - 4)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 6),
                (x1 + tw + 10, label_y + 3),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                tag,
                (x1 + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

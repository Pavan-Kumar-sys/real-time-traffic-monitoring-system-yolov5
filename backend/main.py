import asyncio
import csv
import io
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import CAMERA_PRESETS, CameraPreset
from backend.database import SurveillanceDatabase
from backend.detector import TrafficDetectorEngine
from backend.tracker import MultiObjectTracker

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SentinAI — Real-Time YOLOv5 Traffic Surveillance & Detection System",
    description="YOLOv5 Real-Time Camera & Uploaded Traffic Video Multi-Object Tracking API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemConfigUpdate(BaseModel):
    camera_id: Optional[str] = None
    speed_limit_kmh: Optional[float] = None
    counting_line_y: Optional[float] = None
    show_trails: Optional[bool] = None
    show_speed_tags: Optional[bool] = None
    show_zones: Optional[bool] = None


class AnomalyTriggerRequest(BaseModel):
    anomaly_type: str = "overspeed"


class SurveillancePipeline:
    def __init__(self) -> None:
        self.db = SurveillanceDatabase(str(BASE_DIR / "traffic_surveillance.db"))
        self.detector = TrafficDetectorEngine(model_name="yolov5su.pt")
        self.active_camera_id = "cam_01"
        preset = CAMERA_PRESETS[self.active_camera_id]
        self.tracker = MultiObjectTracker(
            pixels_per_meter=preset.pixels_per_meter,
            speed_limit_kmh=preset.speed_limit_kmh,
            counting_line_y=preset.counting_line_y,
        )
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.uploaded_video_path: Optional[Path] = None
        self.fps: float = 30.0
        self._last_frame_ts: float = time.time()
        self._last_db_snapshot: float = 0.0
        self.latest_incidents: List[Dict[str, Any]] = self.db.get_recent_incidents(30)

    @property
    def current_preset(self) -> CameraPreset:
        return CAMERA_PRESETS[self.active_camera_id]

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        if source == "0":
            # Use DirectShow on Windows for fast camera startup
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        if source == "upload" and self.uploaded_video_path is not None:
            return cv2.VideoCapture(str(self.uploaded_video_path))
        return cv2.VideoCapture(source)

    def switch_camera(self, camera_id: str) -> None:
        if camera_id not in CAMERA_PRESETS:
            return
        self.active_camera_id = camera_id
        preset = CAMERA_PRESETS[camera_id]
        self.tracker.pixels_per_meter = preset.pixels_per_meter
        self.tracker.speed_limit_kmh = preset.speed_limit_kmh
        self.tracker.counting_line_y = preset.counting_line_y
        self.tracker.reset_counts()

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if not preset.source.startswith("sim:"):
            self.video_capture = self._open_capture(preset.source)

    def load_uploaded_video(self, file_path: Path, filename: str) -> None:
        self.uploaded_video_path = file_path
        CAMERA_PRESETS["cam_upload"].name = f"Uploaded: {filename} (YOLOv5)"
        CAMERA_PRESETS["cam_upload"].location = str(file_path.name)
        self.switch_camera("cam_upload")

    def step_frame(self) -> bytes:
        now = time.time()
        dt = max(0.001, now - self._last_frame_ts)
        self._last_frame_ts = now
        self.fps = 0.85 * self.fps + 0.15 * min(60.0, 1.0 / dt)

        preset = self.current_preset
        if preset.source.startswith("sim:"):
            sim = self.detector.simulators.get(
                preset.source, self.detector.simulators["sim:highway"]
            )
            frame, detections = sim.step()
        else:
            if self.video_capture is None or not self.video_capture.isOpened():
                self.video_capture = self._open_capture(preset.source)

            if self.video_capture is not None and self.video_capture.isOpened():
                ok, raw_frame = self.video_capture.read()
                if not ok or raw_frame is None:
                    # Loop uploaded video seamlessly when it reaches the end
                    if preset.source == "upload":
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, raw_frame = self.video_capture.read()
                if ok and raw_frame is not None:
                    frame = cv2.resize(raw_frame, (1280, 720))
                    detections = self.detector.detect_on_frame(frame)
                else:
                    sim = self.detector.simulators["sim:highway"]
                    frame, detections = sim.step()
            else:
                sim = self.detector.simulators["sim:highway"]
                frame, detections = sim.step()

        h, w = frame.shape[:2]
        active_tracks, violations = self.tracker.update(detections, w, h, now=now)

        for v in violations:
            saved = self.db.log_incident(
                camera_id=self.active_camera_id,
                track_id=v["track_id"],
                vehicle_class=v["vehicle_class"],
                violation_type=v["violation_type"],
                speed_kmh=v["speed_kmh"],
                speed_limit_kmh=v["speed_limit_kmh"],
                lane=v["lane"],
                severity=v["severity"],
                details=v["details"],
            )
            self.latest_incidents.insert(0, saved)
            self.latest_incidents = self.latest_incidents[:50]

        if (now - self._last_db_snapshot) >= 5.0:
            self._last_db_snapshot = now
            summary = self.tracker.get_summary_metrics(now=now)
            self.db.log_telemetry(
                camera_id=self.active_camera_id,
                active_vehicles=summary["active_vehicles"],
                total_counted=summary["total_counted"],
                avg_speed_kmh=summary["avg_speed_kmh"],
                flow_rate_vpm=summary["flow_rate_vpm"],
                congestion_level=summary["congestion_level"],
            )

        annotated = self.detector.render_hud(
            frame=frame,
            tracks=active_tracks,
            counting_line_y=self.tracker.counting_line_y,
            speed_limit_kmh=self.tracker.speed_limit_kmh,
            camera_name=preset.name,
            fps=self.fps,
        )

        ok, buffer = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )
        return buffer.tobytes() if ok else b""

    def get_state_payload(self) -> Dict[str, Any]:
        metrics = self.tracker.get_summary_metrics()
        preset = self.current_preset
        active_tracks_serialized = [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "confidence": t.confidence,
                "speed_kmh": round(t.speed_kmh, 1),
                "lane": t.lane,
                "direction": t.direction,
                "bbox": [round(v, 1) for v in t.bbox],
            }
            for t in self.tracker.tracks.values()
            if t.missed <= 2
        ]
        return {
            "timestamp": time.time(),
            "camera": {
                "id": preset.id,
                "name": preset.name,
                "location": preset.location,
                "speed_limit_kmh": self.tracker.speed_limit_kmh,
                "counting_line_y": self.tracker.counting_line_y,
                "fps": round(self.fps, 1),
            },
            "hud": {
                "show_trails": self.detector.show_trails,
                "show_speed_tags": self.detector.show_speed_tags,
                "show_zones": self.detector.show_zones,
            },
            "metrics": metrics,
            "active_tracks": active_tracks_serialized,
            "recent_incidents": self.latest_incidents[:20],
            "cameras": [
                {"id": c.id, "name": c.name, "location": c.location}
                for c in CAMERA_PRESETS.values()
            ],
        }


pipeline = SurveillancePipeline()


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    return pipeline.get_state_payload()


@app.post("/api/config")
async def update_config(cfg: SystemConfigUpdate) -> Dict[str, Any]:
    if cfg.camera_id is not None:
        pipeline.switch_camera(cfg.camera_id)
    if cfg.speed_limit_kmh is not None:
        pipeline.tracker.speed_limit_kmh = max(10.0, min(160.0, cfg.speed_limit_kmh))
    if cfg.counting_line_y is not None:
        pipeline.tracker.counting_line_y = max(0.2, min(0.85, cfg.counting_line_y))
    if cfg.show_trails is not None:
        pipeline.detector.show_trails = cfg.show_trails
    if cfg.show_speed_tags is not None:
        pipeline.detector.show_speed_tags = cfg.show_speed_tags
    if cfg.show_zones is not None:
        pipeline.detector.show_zones = cfg.show_zones
    return pipeline.get_state_payload()


@app.post("/api/upload_video")
async def upload_traffic_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    safe_name = Path(file.filename or "traffic_upload.mp4").name
    dest_path = UPLOADS_DIR / f"{int(time.time())}_{safe_name}"
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.load_uploaded_video(dest_path, safe_name)
    return {
        "status": "uploaded",
        "filename": safe_name,
        "camera_id": "cam_upload",
        "state": pipeline.get_state_payload(),
    }


@app.post("/api/trigger_anomaly")
async def trigger_anomaly(req: AnomalyTriggerRequest) -> Dict[str, Any]:
    preset = pipeline.current_preset
    sim = pipeline.detector.simulators.get(
        preset.source, pipeline.detector.simulators["sim:highway"]
    )
    sim.trigger_anomaly(req.anomaly_type)
    return {"status": "triggered", "anomaly_type": req.anomaly_type}


@app.post("/api/reset")
async def reset_counters() -> Dict[str, Any]:
    pipeline.tracker.reset_counts()
    pipeline.db.clear_incidents()
    pipeline.latest_incidents.clear()
    return pipeline.get_state_payload()


@app.get("/api/export/incidents.csv")
async def export_incidents_csv() -> Response:
    incidents = pipeline.db.get_recent_incidents(500)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "Timestamp",
            "Camera",
            "Track_ID",
            "Vehicle_Class",
            "Violation_Type",
            "Speed_KMH",
            "Speed_Limit_KMH",
            "Lane",
            "Severity",
            "Details",
        ]
    )
    for row in incidents:
        writer.writerow(
            [
                row["id"],
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["timestamp"])),
                row["camera_id"],
                row["track_id"],
                row["vehicle_class"],
                row["violation_type"],
                row["speed_kmh"],
                row["speed_limit_kmh"],
                row["lane"],
                row["severity"],
                row["details"],
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=traffic_incidents_report.csv"
        },
    )


async def mjpeg_generator():
    while True:
        jpeg_bytes = pipeline.step_frame()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        await asyncio.sleep(0.033)


@app.get("/api/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            await ws.send_json(pipeline.get_state_payload())
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))

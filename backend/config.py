from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class CameraPreset:
    id: str
    name: str
    location: str
    speed_limit_kmh: float
    pixels_per_meter: float
    counting_line_y: float  # Normalized [0.0 - 1.0]
    lanes: int
    source: str  # "0" (live webcam), file path, "sim:highway", or "sim:urban"


CAMERA_PRESETS: Dict[str, CameraPreset] = {
    "cam_webcam": CameraPreset(
        id="cam_webcam",
        name="Live Camera Feed (YOLOv5 Real-Time)",
        location="Local Edge Camera — Hardware Device 0",
        speed_limit_kmh=40.0,
        pixels_per_meter=12.0,
        counting_line_y=0.58,
        lanes=4,
        source="0",
    ),
    "cam_upload": CameraPreset(
        id="cam_upload",
        name="Uploaded Traffic Footage (YOLOv5)",
        location="Custom User Video Stream",
        speed_limit_kmh=70.0,
        pixels_per_meter=9.0,
        counting_line_y=0.62,
        lanes=4,
        source="upload",
    ),
    "cam_01": CameraPreset(
        id="cam_01",
        name="NH-44 Northbound Express (Synthetic Benchmark)",
        location="Sector 12 Toll Plaza — KM 14.2",
        speed_limit_kmh=80.0,
        pixels_per_meter=8.5,
        counting_line_y=0.62,
        lanes=4,
        source="sim:highway",
    ),
    "cam_02": CameraPreset(
        id="cam_02",
        name="Metro Arterial Junction (Synthetic Benchmark)",
        location="Central Business District — West Approach",
        speed_limit_kmh=50.0,
        pixels_per_meter=10.2,
        counting_line_y=0.58,
        lanes=4,
        source="sim:urban",
    ),
}

# COCO class ID mapping for YOLOv5 traffic & live camera classes
YOLO_TRAFFIC_CLASSES: Dict[int, str] = {
    0: "Pedestrian",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Apple-inspired BGR colors for OpenCV annotations
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Car": (227, 113, 0),            # Apple Blue (#0071E3)
    "Auto-Rickshaw": (0, 184, 255),  # Vibrant Amber-Yellow (#FFB800)
    "Motorcycle": (89, 199, 52),     # Apple Green (#34C759)
    "Truck": (0, 149, 255),          # Apple Orange (#FF9500)
    "Bus": (214, 86, 88),            # Apple Indigo (#5856D6)
    "Bicycle": (199, 176, 48),       # Apple Teal (#30B0C7)
    "Pedestrian": (48, 59, 255),     # Apple Red (#FF3B30)
}

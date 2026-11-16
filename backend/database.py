import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


class SurveillanceDatabase:
    """Thread-safe SQLite store for violation incidents and traffic telemetry snapshots."""

    def __init__(self, db_path: str = "traffic_surveillance.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_class TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    speed_limit_kmh REAL NOT NULL,
                    lane INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    active_vehicles INTEGER NOT NULL,
                    total_counted INTEGER NOT NULL,
                    avg_speed_kmh REAL NOT NULL,
                    flow_rate_vpm REAL NOT NULL,
                    congestion_level TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_incident(
        self,
        camera_id: str,
        track_id: int,
        vehicle_class: str,
        violation_type: str,
        speed_kmh: float,
        speed_limit_kmh: float,
        lane: int,
        severity: str,
        details: str,
    ) -> Dict[str, Any]:
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO incidents (
                    timestamp, camera_id, track_id, vehicle_class,
                    violation_type, speed_kmh, speed_limit_kmh, lane, severity, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    camera_id,
                    track_id,
                    vehicle_class,
                    violation_type,
                    round(speed_kmh, 1),
                    round(speed_limit_kmh, 1),
                    lane,
                    severity,
                    details,
                ),
            )
            conn.commit()
            incident_id = cursor.lastrowid

        return {
            "id": incident_id,
            "timestamp": ts,
            "camera_id": camera_id,
            "track_id": track_id,
            "vehicle_class": vehicle_class,
            "violation_type": violation_type,
            "speed_kmh": round(speed_kmh, 1),
            "speed_limit_kmh": round(speed_limit_kmh, 1),
            "lane": lane,
            "severity": severity,
            "details": details,
        }

    def log_telemetry(
        self,
        camera_id: str,
        active_vehicles: int,
        total_counted: int,
        avg_speed_kmh: float,
        flow_rate_vpm: float,
        congestion_level: str,
    ) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_snapshots (
                    timestamp, camera_id, active_vehicles, total_counted,
                    avg_speed_kmh, flow_rate_vpm, congestion_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    camera_id,
                    active_vehicles,
                    total_counted,
                    round(avg_speed_kmh, 1),
                    round(flow_rate_vpm, 1),
                    congestion_level,
                ),
            )
            conn.commit()

    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_incidents(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM incidents")
            conn.commit()

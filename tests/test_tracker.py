import numpy as np

from backend.tracker import (
    Detection,
    MultiObjectTracker,
    PerspectiveHomographyCalibrator,
    compute_iou,
    cosine_similarity,
)


def test_compute_iou_identical_and_disjoint():
    box = (10.0, 10.0, 50.0, 50.0)
    assert abs(compute_iou(box, box) - 1.0) < 1e-6
    assert compute_iou(box, (100.0, 100.0, 140.0, 140.0)) == 0.0


def test_homography_calibrator_maps_corners_to_meters():
    cal = PerspectiveHomographyCalibrator(
        frame_width=1280, frame_height=720, road_width_m=14.0, road_depth_m=28.0
    )
    quad = cal.src_quad
    x0, y0 = cal.pixel_to_meters(quad[0][0], quad[0][1])
    x2, y2 = cal.pixel_to_meters(quad[2][0], quad[2][1])
    assert abs(x0 - 0.0) < 1e-3 and abs(y0 - 0.0) < 1e-3
    assert abs(x2 - 14.0) < 1e-3 and abs(y2 - 28.0) < 1e-3


def test_appearance_reid_cosine_similarity():
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6
    assert abs(cosine_similarity(vec_a, vec_c)) < 1e-6


def test_bytetrack_two_stage_and_overspeed_violation():
    tracker = MultiObjectTracker(
        pixels_per_meter=8.5,
        speed_limit_kmh=60.0,
        counting_line_y=0.50,
    )

    frame_w, frame_h = 1280, 720
    y_positions = [300.0, 325.0, 350.0, 375.0, 400.0]
    confs = [0.92, 0.88, 0.31, 0.90, 0.94]  # Frame 3 drops to 0.31 (Stage 2 ByteTrack recovery)
    emb = np.ones(16, dtype=np.float32) / 4.0
    triggered_violations = []

    for idx, (y, conf) in enumerate(zip(y_positions, confs)):
        det = Detection(
            bbox=(260.0, y - 40.0, 310.0, y + 40.0),
            class_name="Auto-Rickshaw",
            confidence=conf,
            true_speed_kmh=78.0,
            embedding=emb,
        )
        tracks, violations = tracker.update(
            [det], frame_w, frame_h, now=1000.0 + idx * 0.1
        )
        assert len(tracks) == 1
        assert tracks[0].track_id == 101  # Same Track ID preserved across low-conf frame
        triggered_violations.extend(violations)

    summary = tracker.get_summary_metrics(now=1000.5)
    assert summary["total_counted"] == 1
    assert summary["inbound_count"] == 1
    assert summary["class_counts"]["Auto-Rickshaw"] == 1
    assert any(v["violation_type"] == "OVERSPEEDING" for v in triggered_violations)

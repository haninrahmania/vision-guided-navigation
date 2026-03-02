from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2


# -----------------------------
# Config
# -----------------------------
WINDOW = "ArUco Tracking (PiCamera2)"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
HAS_NEW_DETECTOR = hasattr(cv2.aruco, "ArucoDetector")

# MUST match printed marker side length (meters)
MARKER_LENGTH_M = 0.05


@dataclass(frozen=True)
class MarkerMeasurement:
    center_px: Tuple[int, int]
    tvec_m: Tuple[float, float, float]
    rvec: np.ndarray  # (3,1)


def build_kalman() -> cv2.KalmanFilter:
    """
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float32,
    )

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf


def get_camera_intrinsics_fallback(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    # Rough fallback intrinsics (good enough for demo Z/pose; calibrate later for accuracy)
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def draw_axes(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
              rvec: np.ndarray, tvec: np.ndarray, length: float = 0.03) -> None:
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)


def detect_marker(frame_bgr: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray) -> Optional[MarkerMeasurement]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if HAS_NEW_DETECTOR:
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is None or len(ids) == 0:
        return None

    idx = 0
    marker_corners = corners[idx]
    marker_id = int(ids[idx][0])

    cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

    # Pose estimation compatibility
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )
        rvec = np.asarray(rvecs[0], dtype=np.float32).reshape(3, 1)
        tvec = np.asarray(tvecs[0], dtype=np.float32).reshape(3, 1)
    else:
        obj_points = np.array(
            [[-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
             [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
             [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
             [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]],
            dtype=np.float32,
        )
        img_points = marker_corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if not ok:
            return None
        rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float32).reshape(3, 1)

    tvec_flat = tvec.flatten()

    pts = marker_corners.reshape(4, 2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.putText(frame_bgr, f"ID: {marker_id}", (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return MarkerMeasurement(
        center_px=(cx, cy),
        tvec_m=(float(tvec_flat[0]), float(tvec_flat[1]), float(tvec_flat[2])),
        rvec=rvec,
    )


def main() -> None:
    # --- PiCamera2 setup (libcamera-native) ---
    picam2 = Picamera2()

    # Keep resolution modest for performance + stability
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Grab one frame to get dimensions
    frame_rgb = picam2.capture_array()
    h, w = frame_rgb.shape[:2]
    camera_matrix, dist_coeffs = get_camera_intrinsics_fallback(w, h)

    kf = build_kalman()
    initialized = False
    last_meas: Optional[Tuple[float, float]] = None

    cv2.namedWindow(WINDOW)

    try:
        while True:
            frame_rgb = picam2.capture_array()  # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            pred = kf.predict()
            pred_x, pred_y = float(pred[0]), float(pred[1])

            meas = detect_marker(frame_bgr, camera_matrix, dist_coeffs)

            if meas is not None:
                cx, cy = meas.center_px
                z = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)

                if not initialized:
                    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    initialized = True
                else:
                    kf.correct(z)

                last_meas = (cx, cy)

                draw_axes(frame_bgr, camera_matrix, dist_coeffs,
                          meas.rvec, np.array(meas.tvec_m, dtype=np.float32).reshape(3, 1))

                _, _, z_m = meas.tvec_m
                cv2.putText(frame_bgr, f"Z ~ {z_m:.2f} m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            est = kf.statePost
            est_x, est_y = float(est[0]), float(est[1])

            if last_meas is not None:
                cv2.circle(frame_bgr, (int(last_meas[0]), int(last_meas[1])), 6, (0, 0, 255), -1)
            cv2.circle(frame_bgr, (int(pred_x), int(pred_y)), 6, (255, 0, 0), -1)
            cv2.circle(frame_bgr, (int(est_x), int(est_y)), 6, (0, 255, 0), -1)

            cv2.putText(frame_bgr, "Red=raw  Green=filtered  Blue=pred", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not initialized:
                cv2.putText(frame_bgr, "Show an ArUco marker to the camera",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
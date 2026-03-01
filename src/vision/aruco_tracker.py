from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Config
# -----------------------------
WINDOW = "ArUco Tracking"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# If your OpenCV supports the newer API, use ArucoDetector (preferred).
HAS_NEW_DETECTOR = hasattr(cv2.aruco, "ArucoDetector")

# Marker side length in meters (you MUST match your printed marker size!)
# Example: 5cm marker => 0.05
MARKER_LENGTH_M = 0.05


@dataclass(frozen=True)
class MarkerMeasurement:
    center_px: Tuple[int, int]
    tvec_m: Tuple[float, float, float]  # (x, y, z) in meters in camera frame
    rvec: np.ndarray                    # rotation vector (3x1)


def build_kalman() -> cv2.KalmanFilter:
    """
    Simple position-only Kalman filter for smoothing.
    State: [x, y]
    Measurement: [x, y]
    """
    kf = cv2.KalmanFilter(2, 2)

    # Identity transition - prediction keeps same position
    kf.transitionMatrix = np.eye(2, dtype=np.float32)

    # Direct measurement of position
    kf.measurementMatrix = np.eye(2, dtype=np.float32)

    # Tuning: low process noise (we trust position doesn't change much)
    # low measurement noise (we trust measurements)
    kf.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
    kf.errorCovPost = np.eye(2, dtype=np.float32)

    kf.statePost = np.zeros((2, 1), dtype=np.float32)
    return kf


def draw_axes(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
              rvec: np.ndarray, tvec: np.ndarray, length: float = 0.03) -> None:
    # Draw a small axis on the marker pose
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)


def get_camera_intrinsics_fallback(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def detect_marker(frame: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray) -> Optional[MarkerMeasurement]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY", gray)

    if HAS_NEW_DETECTOR:
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(gray)
        dbg = frame.copy()
        if corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
        cv2.imshow("DETECTION_DEBUG", dbg)
        print("ids:", ids)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is None or len(ids) == 0:
        return None

    # Take the first marker for simplicity (you can choose a specific id later)
    idx = 0
    marker_corners = corners[idx]
    marker_id = int(ids[idx][0])

    # Draw marker outline + id
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Pose estimation
    # NEW OpenCV compatibility
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        # Older OpenCV versions
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            MARKER_LENGTH_M,
            camera_matrix,
            dist_coeffs
        )
    else:
        # Newer OpenCV versions (pose estimation via solvePnP)
        obj_points = np.array([
            [-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
            [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
            [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
            [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
        ], dtype=np.float32)

        img_points = marker_corners.reshape(4, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return None

        rvecs = np.array([rvec])
        tvecs = np.array([tvec])
    rvec = rvecs[0]
    tvec = tvecs[0]  # shape (1,3) from estimatePoseSingleMarkers or (3,1) from solvePnP
    rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float32).reshape(3, 1)
    tvec_flat = tvec.flatten()

    # Marker center pixel
    pts = marker_corners.reshape(4, 2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    # Overlay ID
    cv2.putText(frame, f"ID: {marker_id}", (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return MarkerMeasurement(center_px=(cx, cy), tvec_m=(float(tvec_flat[0]), float(tvec_flat[1]), float(tvec_flat[2])), rvec=rvec)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    # Grab one frame to get size, then set approximate intrinsics
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    h, w = frame.shape[:2]
    camera_matrix, dist_coeffs = get_camera_intrinsics_fallback(w, h)

    kf = build_kalman()
    initialized = False
    last_meas: Optional[Tuple[float, float]] = None

    cv2.namedWindow(WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)

        # Always predict first (updates statePre from statePost)
        pred = kf.predict()
        pred_x, pred_y = float(pred[0]), float(pred[1])

        meas = detect_marker(frame, camera_matrix, dist_coeffs)

        if meas is not None:
            cx, cy = meas.center_px

            # Debug: print measurement to verify detection is working
            print(f"DETECTED: marker at ({cx}, {cy})")

            # Use pixel center for Kalman measurement (simple & robust).
            z = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)

            if not initialized:
                # Initialize filter state to current measurement
                init_state = np.array([[cx], [cy]], dtype=np.float32)
                kf.statePost = init_state.copy()
                kf.statePre = init_state.copy()
                initialized = True
                print(f"  -> Initialized Kalman at ({cx}, {cy})")
            else:
                kf.correct(z)
                print(f"  -> Corrected, new state=({float(kf.statePost[0]):.1f}, {float(kf.statePost[1]):.1f})")

            last_meas = (cx, cy)

            # Optional: draw pose axis (roughly correct with fallback intrinsics)
            # Note: tvec must be (3,1) for drawFrameAxes
            draw_axes(frame, camera_matrix, dist_coeffs, meas.rvec, np.array(meas.tvec_m, dtype=np.float32).reshape(3, 1))

            # Show Z distance as a fun “robotics” metric
            _, _, z_m = meas.tvec_m
            cv2.putText(frame, f"Z ~ {z_m:.2f} m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Filtered estimate after correct
        est = kf.statePost
        est_x, est_y = float(est[0]), float(est[1])

        # Debug: show state on frame
        state_str = f"est=({est_x:.0f},{est_y:.0f}) pred=({pred_x:.0f},{pred_y:.0f})"
        if last_meas:
            state_str += f" meas=({last_meas[0]:.0f},{last_meas[1]:.0f})"
        cv2.putText(frame, state_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw points (raw, predicted, filtered)
        if last_meas is not None:
            cv2.circle(frame, (int(last_meas[0]), int(last_meas[1])), 6, (0, 0, 255), -1)  # raw red

        cv2.circle(frame, (int(pred_x), int(pred_y)), 6, (255, 0, 0), -1)  # prediction blue
        cv2.circle(frame, (int(est_x), int(est_y)), 6, (0, 255, 0), -1)    # filtered green

        cv2.putText(frame, "Red=raw  Green=filtered  Blue=pred", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not initialized:
            cv2.putText(frame, "Show an ArUco marker to the camera",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
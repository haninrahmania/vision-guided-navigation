from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
HAS_NEW_DETECTOR = hasattr(cv2.aruco, "ArucoDetector")
MARKER_LENGTH_M = 0.05  # must match printed marker size


@dataclass(frozen=True)
class MarkerMeasurement:
    center_px: Tuple[int, int]
    tvec_m: Tuple[float, float, float]
    rvec: np.ndarray  # (3,1)


def get_camera_intrinsics_fallback(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


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

    # draw marker outline + id
    cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

    # pose estimation compatibility
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

    pts = marker_corners.reshape(4, 2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    cv2.putText(frame_bgr, f"ID:{marker_id}", (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    tvec_flat = tvec.flatten()
    return MarkerMeasurement(
        center_px=(cx, cy),
        tvec_m=(float(tvec_flat[0]), float(tvec_flat[1]), float(tvec_flat[2])),
        rvec=rvec,
    )


def main() -> None:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    frame_rgb = picam2.capture_array()
    h, w = frame_rgb.shape[:2]
    camera_matrix, dist_coeffs = get_camera_intrinsics_fallback(w, h)

    # write an annotated mp4 so you can copy it to laptop + watch
    out = cv2.VideoWriter(
        "aruco_headless.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (w, h),
    )

    print("[INFO] Headless ArUco tracker running.")
    print("[INFO] Move an ArUco marker in front of the camera.")
    print("[INFO] Saving annotated video to: aruco_headless.mp4")
    print("[INFO] Press Ctrl+C to stop.")

    frame_count = 0
    try:
        while True:
            frame_count += 1

            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            meas = detect_marker(frame_bgr, camera_matrix, dist_coeffs)
            if meas is None:
                if frame_count % 30 == 0:
                    print("[TRACK] no marker")
            else:
                cx, cy = meas.center_px
                z = meas.tvec_m[2]
                if frame_count % 10 == 0:
                    print(f"[TRACK] cx={cx:4d} cy={cy:4d} z~{z:.2f}m")

                # optional axes for pose
                cv2.drawFrameAxes(
                    frame_bgr,
                    camera_matrix,
                    dist_coeffs,
                    meas.rvec,
                    np.array(meas.tvec_m, dtype=np.float32).reshape(3, 1),
                    0.03,
                )

            out.write(frame_bgr)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    finally:
        out.release()
        picam2.stop()
        print("[INFO] Saved: aruco_headless.mp4")


if __name__ == "__main__":
    main()
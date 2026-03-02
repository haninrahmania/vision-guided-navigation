from __future__ import annotations

import cv2
import numpy as np
from picamera2 import Picamera2

from control.controller import FollowController
from control.smooth_controller import SmoothFollowController
from hardware.robot_interface import RobotInterface, command_to_drive
from vision.aruco_tracker_picam2_headless import (
    detect_marker,
    get_camera_intrinsics_fallback,
)


def main() -> None:
    # Initialize PiCamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    # Capture one frame to get dimensions
    frame_rgb = picam2.capture_array()
    if frame_rgb is None:
        raise RuntimeError("Could not capture frame from PiCamera2.")
    h, w = frame_rgb.shape[:2]

    camera_matrix, dist_coeffs = get_camera_intrinsics_fallback(w, h)

    USE_SMOOTH = True

    if USE_SMOOTH:
        controller = SmoothFollowController(
            frame_width=w,
            dead_zone_px=30,
            z_stop_m=0.20,
            z_slow_m=0.60,
        )
    else:
        controller = FollowController(frame_width=w, dead_zone_px=60)

    robot = RobotInterface()

    # Optional: record annotated video for post-run analysis
    RECORD_VIDEO = True
    out: cv2.VideoWriter | None = None
    if RECORD_VIDEO:
        out = cv2.VideoWriter(
            "follow_run.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (w, h),
        )

    print("[INFO] PiCamera2 follow controller running (headless).")
    print("[INFO] Move an ArUco marker in front of the camera.")
    if RECORD_VIDEO:
        print("[INFO] Recording to: follow_run.mp4")
    print("[INFO] Press Ctrl+C to stop.")

    # Throttle prints: only on state change + every N frames when tracking
    was_detected: bool | None = None
    frame_count = 0
    PRINT_EVERY_N_FRAMES = 15  # ~0.5 sec at 30fps

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_count += 1

            # PiCamera2 captures RGB; convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            meas = detect_marker(frame_bgr, camera_matrix, dist_coeffs)
            is_detected = meas is not None

            # Print only on state transition or throttled interval
            should_print = False
            if was_detected != is_detected:
                should_print = True  # State change always prints
            elif is_detected and frame_count % PRINT_EVERY_N_FRAMES == 0:
                should_print = True  # Throttled print when tracking

            if meas is None:
                cmd = controller.update(target_cx=None, target_z_m=None)
                z_m = None
                if should_print:
                    print("MARKER: LOST (not detected)")
            else:
                cx, _ = meas.center_px
                z_m = meas.tvec_m[2]
                cmd = controller.update(target_cx=cx, target_z_m=z_m)
                if should_print:
                    print(f"MARKER: cx={cx:4d} Z={z_m:.2f}m | cmd={cmd.action}")

            was_detected = is_detected

            drive = command_to_drive(cmd.linear, cmd.angular)
            robot.apply(drive)

            # Annotate frame for video recording
            cv2.putText(
                frame_bgr,
                f"STATE: {cmd.action} | lin={cmd.linear:.2f} ang={cmd.angular:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if z_m is not None:
                cv2.putText(
                    frame_bgr,
                    f"Z ~ {z_m:.2f} m",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Optional: draw pose axes on recorded frame
            if meas is not None:
                cv2.drawFrameAxes(
                    frame_bgr,
                    camera_matrix,
                    dist_coeffs,
                    meas.rvec,
                    np.array(meas.tvec_m, dtype=np.float32).reshape(3, 1),
                    0.03,
                )

            if out is not None:
                out.write(frame_bgr)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    finally:
        robot.stop()
        picam2.stop()
        if out is not None:
            out.release()
            print("[INFO] Saved: follow_run.mp4")


if __name__ == "__main__":
    main()

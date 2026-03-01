from __future__ import annotations

import cv2

from control.controller import FollowController
from hardware.robot_interface import RobotInterface, command_to_drive
from vision.aruco_tracker import (
    WINDOW,
    detect_marker,
    get_camera_intrinsics_fallback,
)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    # one frame to get size + intrinsics
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    h, w = frame.shape[:2]

    camera_matrix, dist_coeffs = get_camera_intrinsics_fallback(w, h)

    controller = FollowController(frame_width=w, dead_zone_px=60)
    robot = RobotInterface()

    cv2.namedWindow(WINDOW)

    # Throttle prints: only on state change + every N frames when tracking
    was_detected: bool | None = None
    frame_count = 0
    PRINT_EVERY_N_FRAMES = 15  # ~0.5 sec at 30fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # NOTE: Remove flip if detection is unreliable; flip can mirror markers
        # frame = cv2.flip(frame, 1)

        meas = detect_marker(frame, camera_matrix, dist_coeffs)
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

        # On-screen clarity (this is the “instant clarity trick” for reviewers)
        cv2.putText(
            frame,
            f"STATE: {cmd.action} | lin={cmd.linear:.2f} ang={cmd.angular:.2f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if z_m is not None:
            cv2.putText(
                frame,
                f"Z ~ {z_m:.2f} m",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    robot.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
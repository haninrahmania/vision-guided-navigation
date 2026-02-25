from typing import Optional
import cv2
import numpy as np
from collections import deque

class HSVKalmanTracker:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        
        self.trail = deque(maxlen=30)

        # HSV selection
        self.lower_hsv: Optional[np.ndarray] = None
        self.upper_hsv: Optional[np.ndarray] = None
        self.hsv_frame: Optional[np.ndarray] = None

        # frame size (used by navigation mapping)
        self.frame_w: Optional[int] = None
        self.frame_h: Optional[int] = None

        # Kalman
        self.kf = self._build_kalman()
        self.initialized = False

        cv2.namedWindow("Tracking")
        cv2.setMouseCallback("Tracking", self._pick_color)

    # --------------------------------------------------
    # Kalman
    # --------------------------------------------------
    def _build_kalman(self):
        kf = cv2.KalmanFilter(4, 2)

        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)

        return kf

    # --------------------------------------------------
    # Mouse click color picker
    # --------------------------------------------------
    def _pick_color(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.hsv_frame is None:
            return

        h, s, v = self.hsv_frame[y, x]

        h_tol = 10
        s_tol = 60
        v_tol = 60

        self.lower_hsv = np.array([
            max(0, h - h_tol),
            max(0, s - s_tol),
            max(0, v - v_tol)
        ], dtype=np.uint8)

        self.upper_hsv = np.array([
            min(179, h + h_tol),
            min(255, s + s_tol),
            min(255, v + v_tol)
        ], dtype=np.uint8)

        print("Selected HSV:", (h, s, v))
        print("Lower:", self.lower_hsv, "Upper:", self.upper_hsv)

    # --------------------------------------------------
    # Blob detection
    # --------------------------------------------------
    def _find_largest_blob(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 500:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy), largest

    # --------------------------------------------------
    # MAIN UPDATE (called from pygame loop)
    # --------------------------------------------------
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)

        self.frame_h, self.frame_w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.hsv_frame = hsv

        # no color selected yet
        if self.lower_hsv is None:
            empty_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            cv2.putText(
                frame,
                "Click object to select color",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Tracking", frame)
            cv2.imshow("Mask", empty_mask)
            cv2.waitKey(1)
            return None

        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Kalman predict
        pred = self.kf.predict()
        pred_x, pred_y = int(pred[0]), int(pred[1])

        found = self._find_largest_blob(mask)

        if found is not None:
            (cx, cy), contour = found

            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)

            if not self.initialized:
                self.kf.statePost = np.array(
                    [[cx], [cy], [0], [0]], dtype=np.float32
                )
                self.initialized = True
            else:
                self.kf.correct(meas)

        est = self.kf.statePost
        est_x, est_y = int(est[0]), int(est[1])

        self.trail.append((est_x, est_y))

        # draw trail
        for i in range(1, len(self.trail)):
            cv2.line(frame, self.trail[i-1], self.trail[i], (0, 255, 0), 2)

        cv2.circle(frame, (pred_x, pred_y), 6, (255, 0, 0), -1)
        cv2.circle(frame, (est_x, est_y), 6, (0, 255, 0), -1)

        cv2.putText(
            frame,
            "Red=raw Green=filtered Blue=pred",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Tracking", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

        # RETURN FILTERED POSITION
        if self.initialized:
            return (est_x, est_y)

        return None

    # --------------------------------------------------
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
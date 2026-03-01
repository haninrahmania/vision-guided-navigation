from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ControlCommand:
    """
    High-level command for a differential drive robot.
    linear:  0..1   (forward)
    angular: -1..1  (left/right)
    """
    action: str  # "STOP" | "SEARCH" | "ALIGN" | "TURN" | "FORWARD"
    linear: float
    angular: float


class SmoothFollowController:
    """
    A "pro" closed-loop visual servo controller:
    - Continuous steering (P-control) from pixel error
    - Speed scheduling: slow when misaligned or near target
    - Slew-rate limiting: prevents jitter/twitchy commands
    - Lost-target behavior: stop then slow search spin

    Inputs:
      target_cx: marker center x in pixels (None if not detected)
      target_z_m: distance estimate (meters) (None if unavailable)

    Output:
      ControlCommand (linear/angular)
    """

    def __init__(
        self,
        frame_width: int,
        dead_zone_px: int = 30,
        z_stop_m: float = 0.20,
        z_slow_m: float = 0.60,
        search_after_frames: int = 15,
        kp_ang: float = 1.2,
        max_ang: float = 0.8,
        max_lin: float = 0.65,
        min_lin: float = 0.12,
        ang_slew: float = 0.10,
        lin_slew: float = 0.06,
    ) -> None:
        self.w = float(frame_width)
        self.cx0 = self.w / 2.0

        self.dead_zone_px = float(dead_zone_px)
        self.z_stop_m = float(z_stop_m)
        self.z_slow_m = float(z_slow_m)
        self.search_after_frames = int(search_after_frames)

        self.kp_ang = float(kp_ang)
        self.max_ang = float(max_ang)
        self.max_lin = float(max_lin)
        self.min_lin = float(min_lin)

        self.ang_slew = float(ang_slew)
        self.lin_slew = float(lin_slew)

        self._lost_frames = 0
        self._ang = 0.0
        self._lin = 0.0

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _slew(cur: float, target: float, step: float) -> float:
        if target > cur:
            return min(cur + step, target)
        return max(cur - step, target)

    def _speed_schedule(self, ang: float, z_m: Optional[float]) -> float:
        # Alignment factor: 1 when straight, 0 when hard turn
        align = 1.0 - min(1.0, abs(ang) / max(1e-6, self.max_ang))

        # Distance factor: 0..1 (if no Z, assume far)
        if z_m is None:
            dist = 1.0
        else:
            dist = (z_m - self.z_stop_m) / max(1e-6, (self.z_slow_m - self.z_stop_m))
            dist = self._clamp(dist, 0.0, 1.0)

        return self.min_lin + (self.max_lin - self.min_lin) * align * dist

    def update(self, target_cx: Optional[int], target_z_m: Optional[float]) -> ControlCommand:
        # LOST: stop for a bit, then search-spin
        if target_cx is None:
            self._lost_frames += 1
            if self._lost_frames >= self.search_after_frames:
                target_lin, target_ang = 0.0, 0.35
                action = "SEARCH"
            else:
                target_lin, target_ang = 0.0, 0.0
                action = "STOP"

        else:
            self._lost_frames = 0

            # STOP if close enough
            if target_z_m is not None and target_z_m <= self.z_stop_m:
                target_lin, target_ang = 0.0, 0.0
                action = "STOP"
            else:
                # Continuous steering from normalized pixel error
                err_px = float(target_cx) - self.cx0

                # Dead-zone to prevent micro jitter
                if abs(err_px) < self.dead_zone_px:
                    err_px = 0.0

                # Normalize to roughly [-1..1]
                err_n = err_px / (self.w / 2.0)

                # P-controller steering
                target_ang = self._clamp(self.kp_ang * err_n, -self.max_ang, self.max_ang)

                # Speed scheduling
                target_lin = self._speed_schedule(target_ang, target_z_m)

                # Action labels for UI clarity
                if target_lin < 0.05 and abs(target_ang) > 0.05:
                    action = "ALIGN"
                elif abs(target_ang) > 0.10:
                    action = "TURN"
                else:
                    action = "FORWARD"

        # Slew-rate limiting (smooth commands)
        self._ang = self._slew(self._ang, target_ang, self.ang_slew)
        self._lin = self._slew(self._lin, target_lin, self.lin_slew)

        return ControlCommand(action=action, linear=self._lin, angular=self._ang)
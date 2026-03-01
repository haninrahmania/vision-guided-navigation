from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ControlCommand:
    action: str  # "STOP" | "FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "SEARCH"
    linear: float = 0.0   # 0..1
    angular: float = 0.0  # -1..1


class FollowController:
    """
    Closed-loop visual target following (simple behavioral controller).
    Input: target center x (pixels) + optional z distance (meters)
    Output: high-level linear/angular command
    """

    def __init__(
        self,
        frame_width: int,
        dead_zone_px: int = 60,
        z_stop_m: float = 0.35,
        z_slow_m: float = 0.60,
        search_after_frames: int = 15,
    ):
        self.frame_width = frame_width
        self.dead_zone_px = dead_zone_px
        self.z_stop_m = z_stop_m
        self.z_slow_m = z_slow_m
        self.search_after_frames = search_after_frames

        self._lost_frames = 0

    def update(self, target_cx: Optional[int], target_z_m: Optional[float]) -> ControlCommand:
        # Not visible → stop, then search
        if target_cx is None:
            self._lost_frames += 1
            if self._lost_frames >= self.search_after_frames:
                return ControlCommand(action="SEARCH", linear=0.0, angular=0.35)
            return ControlCommand(action="STOP", linear=0.0, angular=0.0)

        self._lost_frames = 0

        # Distance gating (optional)
        if target_z_m is not None and target_z_m <= self.z_stop_m:
            return ControlCommand(action="STOP", linear=0.0, angular=0.0)

        center_x = self.frame_width // 2
        error = target_cx - center_x

        # Turn-in-place if marker is off-center
        if error < -self.dead_zone_px:
            return ControlCommand(action="TURN_LEFT", linear=0.0, angular=-0.6)
        if error > self.dead_zone_px:
            return ControlCommand(action="TURN_RIGHT", linear=0.0, angular=0.6)

        # Otherwise drive forward (slow down when close)
        linear = 0.6
        if target_z_m is not None and target_z_m < self.z_slow_m:
            linear = 0.35

        return ControlCommand(action="FORWARD", linear=linear, angular=0.0)
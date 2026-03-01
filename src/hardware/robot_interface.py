from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriveSignal:
    left: float   # -1..1
    right: float  # -1..1


class RobotInterface:
    """
    Hardware abstraction.
    Today: print signals.
    Later: TB6612FNG PWM on Raspberry Pi.
    """

    # def apply(self, drive: DriveSignal) -> None:
    #     print(f"[DRIVE] left={drive.left:.2f} right={drive.right:.2f}")

    def __init__(self):
        self._last = None

    def apply(self, drive: DriveSignal) -> None:
        cur = (round(drive.left, 2), round(drive.right, 2))
        if cur != self._last:
            print(f"[DRIVE] left={drive.left:.2f} right={drive.right:.2f}")
            self._last = cur

    def stop(self) -> None:
        self.apply(DriveSignal(0.0, 0.0))


def command_to_drive(linear: float, angular: float) -> DriveSignal:
    """
    Differential drive mixer:
      left  = linear - angular
      right = linear + angular
    """
    left = linear - angular
    right = linear + angular

    # clamp to -1..1
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    return DriveSignal(left=left, right=right)
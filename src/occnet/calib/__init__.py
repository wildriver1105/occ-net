"""Camera calibration: ChArUco board, intrinsics, and rig extrinsics."""

from .board import BoardSpec, make_board, render_board
from .intrinsics import CalibrationSet, calibrate_intrinsics, detect_board
from .extrinsics import calibrate_rig

__all__ = [
    "BoardSpec",
    "make_board",
    "render_board",
    "CalibrationSet",
    "calibrate_intrinsics",
    "detect_board",
    "calibrate_rig",
]

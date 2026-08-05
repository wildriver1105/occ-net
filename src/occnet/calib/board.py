"""ChArUco calibration target.

ChArUco rather than a plain chessboard because the ArUco markers give the board
an absolute orientation and let partial views still contribute — which matters a
lot when two cameras with very different fields of view have to see the same
target at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class BoardSpec:
    """Physical description of the printed target.

    ``square_m`` must be measured on the *printed* sheet with a ruler — every
    downstream metric distance is scaled by it.
    """

    squares_x: int = 8
    squares_y: int = 11
    square_m: float = 0.030
    marker_m: float = 0.022
    dictionary: str = "5x5_250"

    def __post_init__(self) -> None:
        if self.marker_m >= self.square_m:
            raise ValueError("marker_m must be smaller than square_m")
        if self.dictionary not in _DICTS:
            raise ValueError(f"unknown dictionary {self.dictionary!r}; pick one of {list(_DICTS)}")

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    def to_dict(self) -> dict:
        return {
            "squares_x": self.squares_x, "squares_y": self.squares_y,
            "square_m": self.square_m, "marker_m": self.marker_m,
            "dictionary": self.dictionary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_board(spec: BoardSpec) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    """Build the OpenCV board object and a detector tuned for live video."""
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[spec.dictionary])
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m, dictionary
    )
    detector_params = cv2.aruco.DetectorParameters()
    # Subpixel refinement matters at the small marker sizes you get when the
    # board is far enough away for both cameras to see it at once.
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, detector


def render_board(spec: BoardSpec, path: str | Path, dpi: int = 300, margin_mm: float = 10.0) -> Path:
    """Write a print-ready PNG of the board at true physical scale."""
    px_per_m = dpi / 0.0254
    w_m, h_m = spec.size_m
    size_px = (int(round(w_m * px_per_m)), int(round(h_m * px_per_m)))
    margin_px = int(round(margin_mm / 1000.0 * px_per_m))

    board, _ = make_board(spec)
    img = board.generateImage(size_px, marginSize=0, borderBits=1)

    canvas = np.full(
        (img.shape[0] + 2 * margin_px, img.shape[1] + 2 * margin_px), 255, dtype=np.uint8
    )
    canvas[margin_px:margin_px + img.shape[0], margin_px:margin_px + img.shape[1]] = img

    caption = (
        f"occnet ChArUco  {spec.squares_x}x{spec.squares_y}  "
        f"square={spec.square_m * 1000:.1f}mm  marker={spec.marker_m * 1000:.1f}mm  "
        f"dict={spec.dictionary}  print at 100% ({dpi} dpi)"
    )
    canvas = np.vstack([canvas, np.full((60, canvas.shape[1]), 255, np.uint8)])
    cv2.putText(
        canvas, caption, (margin_px, canvas.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path

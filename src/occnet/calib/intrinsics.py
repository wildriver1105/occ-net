"""Per-camera intrinsic calibration from ChArUco views."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import FISHEYE, PINHOLE, RATIONAL, CameraModel
from .board import BoardSpec, make_board


@dataclass
class Detection:
    """A single successful board detection in one image."""

    obj_points: np.ndarray  # (N,3) float32, board coordinates in metres
    img_points: np.ndarray  # (N,2) float32, pixels
    corner_ids: np.ndarray  # (N,) int32, ChArUco corner ids
    image_size: tuple[int, int]  # (width, height)

    @property
    def count(self) -> int:
        return len(self.corner_ids)


def detect_board(
    image: np.ndarray,
    detector: cv2.aruco.CharucoDetector,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 8,
) -> Detection | None:
    """Find the ChArUco board in one BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < min_corners:
        return None

    return Detection(
        obj_points=np.asarray(obj_points, np.float32).reshape(-1, 3),
        img_points=np.asarray(img_points, np.float32).reshape(-1, 2),
        corner_ids=np.asarray(charuco_ids, np.int32).ravel(),
        image_size=(gray.shape[1], gray.shape[0]),
    )


@dataclass
class CalibrationSet:
    """Accumulates board views for one camera and reports coverage.

    Coverage is tracked on a coarse image-space grid: calibration quality is
    dominated by whether the board reached the corners and edges of the frame,
    not by how many frames you took in the middle.
    """

    name: str
    image_size: tuple[int, int] = (0, 0)
    detections: list[Detection] = field(default_factory=list)
    grid: int = 6

    def __post_init__(self) -> None:
        self._occupancy = np.zeros((self.grid, self.grid), dtype=np.int32)

    def add(self, det: Detection) -> None:
        if self.image_size == (0, 0):
            self.image_size = det.image_size
        elif det.image_size != self.image_size:
            raise ValueError(
                f"{self.name}: image size changed mid-calibration "
                f"({self.image_size} -> {det.image_size})"
            )
        self.detections.append(det)
        w, h = det.image_size
        cols = np.clip((det.img_points[:, 0] / w * self.grid).astype(int), 0, self.grid - 1)
        rows = np.clip((det.img_points[:, 1] / h * self.grid).astype(int), 0, self.grid - 1)
        np.add.at(self._occupancy, (rows, cols), 1)

    @property
    def coverage(self) -> float:
        """Fraction of the image-space grid that has seen at least one corner."""
        return float((self._occupancy > 0).mean())

    @property
    def occupancy(self) -> np.ndarray:
        return self._occupancy.copy()

    def is_novel(self, det: Detection, min_new_cells: int = 1) -> bool:
        """True when a view would light up grid cells no previous view reached.

        Used during live capture to reject near-duplicate frames, which inflate
        the view count without improving the solve.
        """
        w, h = det.image_size
        cols = np.clip((det.img_points[:, 0] / w * self.grid).astype(int), 0, self.grid - 1)
        rows = np.clip((det.img_points[:, 1] / h * self.grid).astype(int), 0, self.grid - 1)
        fresh = self._occupancy[rows, cols] == 0
        return int(np.unique(np.stack([rows[fresh], cols[fresh]], 1), axis=0).shape[0]) >= min_new_cells

    def __len__(self) -> int:
        return len(self.detections)


def _flags_for(model: str) -> int:
    if model == RATIONAL:
        return cv2.CALIB_RATIONAL_MODEL
    return 0


def calibrate_intrinsics(
    cal: CalibrationSet,
    model: str = RATIONAL,
    min_views: int = 8,
) -> CameraModel:
    """Solve intrinsics from the accumulated views.

    ``model`` should be :data:`RATIONAL` for wide action-cam lenses (the GO 3),
    :data:`PINHOLE` for normal lenses, and :data:`FISHEYE` only for genuinely
    fisheye optics — the Kannala-Brandt solver is fragile on moderate lenses.
    """
    if len(cal) < min_views:
        raise ValueError(
            f"{cal.name}: need at least {min_views} board views, have {len(cal)}. "
            "Capture more, moving the board across the whole frame and tilting it."
        )

    w, h = cal.image_size
    # OpenCV's calibration entry points want Point3f/Point2f, i.e. float32.
    obj = [d.obj_points.reshape(-1, 1, 3).astype(np.float32) for d in cal.detections]
    img = [d.img_points.reshape(-1, 1, 2).astype(np.float32) for d in cal.detections]

    if model == FISHEYE:
        # The fisheye solver requires every view to have the same point count,
        # so it can only use views where the full board was detected.
        counts = [len(o) for o in obj]
        target = max(set(counts), key=counts.count)
        keep = [i for i, c in enumerate(counts) if c == target]
        if len(keep) < min_views:
            raise ValueError(
                f"{cal.name}: fisheye calibration needs {min_views} views with an identical "
                f"corner count; only {len(keep)} of {len(cal)} views match. "
                "Use --model rational instead, or capture full-board views."
            )
        # The fisheye solver, unlike calibrateCamera, wants float64.
        obj_f = [obj[i].astype(np.float64) for i in keep]
        img_f = [img[i].astype(np.float64) for i in keep]
        K = np.eye(3)
        D = np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_f, img_f, (w, h), K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        return CameraModel(
            name=cal.name, width=w, height=h, K=K, dist=D.ravel(),
            model=FISHEYE, rms=float(rms), n_views=len(keep),
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj, img, (w, h), None, None, flags=_flags_for(model)
    )
    return CameraModel(
        name=cal.name, width=w, height=h, K=K, dist=dist.ravel(),
        model=model, rms=float(rms), n_views=len(cal),
    )


def undistort_map(cam: CameraModel, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray, CameraModel]:
    """Build remap tables that rectify a camera to an ideal pinhole.

    ``alpha=0`` crops to valid pixels only; ``alpha=1`` keeps the whole sensor
    and leaves black wedges. Returns the maps plus the *new* intrinsics, which
    are what downstream depth lifting must use.
    """
    size = (cam.width, cam.height)
    if cam.model == FISHEYE:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cam.K, cam.dist.reshape(4, 1), size, np.eye(3), balance=alpha
        )
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            cam.K, cam.dist.reshape(4, 1), np.eye(3), new_K, size, cv2.CV_16SC2
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, size, alpha, size)
        m1, m2 = cv2.initUndistortRectifyMap(
            cam.K, cam.dist, np.eye(3), new_K, size, cv2.CV_16SC2
        )
    rectified = CameraModel(
        name=f"{cam.name}_rect", width=cam.width, height=cam.height,
        K=new_K, dist=np.zeros(5), model=PINHOLE, rms=cam.rms, n_views=cam.n_views,
    )
    return m1, m2, rectified

"""Anchor the world frame to a physical ChArUco board.

This is the cheapest possible pose source: the board is already printed for
calibration, its geometry is already known, and its corners are already
detected. Every frame that sees it yields an absolute camera pose by PnP, with
no drift and no initialisation — which is exactly what a static occupancy grid
needs in order to become a *moving* one.

The trade is coverage: no board in view means no pose, and a frame without a
pose must be dropped rather than guessed at. Integrating a frame at the wrong
pose does not blur the map, it writes confident geometry into the wrong place,
and log-odds evidence is expensive to undo. Visual odometry is the way out of
that limitation, not a smarter fallback here.

World frame: the board's own frame, optionally recentred on the board's middle.
The board's +z points out of its printed face, so laying the board flat with the
pattern facing up gives a conventional z-up world.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib.board import BoardSpec, make_board
from .calib.intrinsics import Detection, detect_board
from .geometry import FISHEYE, CameraModel, invert, rt_to_matrix


@dataclass
class TrackingConfig:
    """How strict to be about accepting a pose."""

    min_corners: int = 8
    # A pose that reprojects badly is a wrong pose, not a noisy one — usually a
    # partially occluded board or motion blur smearing the corners.
    max_reproj_px: float = 2.5
    # Exponential smoothing on translation only; rotation is left alone because
    # naive quaternion blending across large rotations misbehaves. 0 disables.
    smooth_alpha: float = 0.0
    # Reject implausible jumps between consecutive tracked frames.
    max_jump_m: float = 0.75


@dataclass
class TrackResult:
    """One successful pose fix."""

    T_world_cam: np.ndarray  # 4x4, camera-to-world
    T_cam_board: np.ndarray  # 4x4, board-to-camera (what PnP solved)
    n_corners: int
    reproj_px: float
    detection: Detection

    @property
    def position(self) -> np.ndarray:
        return self.T_world_cam[:3, 3]

    @property
    def distance_to_board(self) -> float:
        return float(np.linalg.norm(self.T_cam_board[:3, 3]))


class BoardTracker:
    """Turns board sightings into camera poses in a fixed world frame."""

    def __init__(
        self,
        spec: BoardSpec,
        cfg: TrackingConfig | None = None,
        center_origin: bool = True,
    ):
        self.spec = spec
        self.cfg = cfg or TrackingConfig()
        self.board, self.detector = make_board(spec)

        # Shift the world origin to the middle of the board so symmetric grid
        # bounds actually straddle the target.
        self.T_world_board = np.eye(4)
        if center_origin:
            w, h = spec.size_m
            self.T_world_board[:3, 3] = [-w / 2, -h / 2, 0.0]

        self._last: TrackResult | None = None
        self.tracked = 0
        self.lost = 0
        self.rejected = 0

    @property
    def last(self) -> TrackResult | None:
        return self._last

    @property
    def lock_rate(self) -> float:
        total = self.tracked + self.lost
        return self.tracked / total if total else 0.0

    def _solve(self, det: Detection, cam: CameraModel) -> tuple[np.ndarray, float] | None:
        """PnP on the detected corners, returning ``(T_cam_board, reproj_px)``."""
        obj = det.obj_points.astype(np.float64)
        img = det.img_points.astype(np.float64)

        if cam.model == FISHEYE:
            img = cv2.fisheye.undistortPoints(
                img.reshape(-1, 1, 2), cam.K, cam.dist.reshape(4, 1), P=cam.K
            ).reshape(-1, 2)
            dist = np.zeros(5)
        else:
            dist = cam.dist

        # IPPE is the planar-target solver; the board is planar by construction,
        # and it is both faster and better conditioned here than the general
        # iterative solve. Refine afterwards to squeeze out the last fraction.
        ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, dist)
        reproj = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
        return rt_to_matrix(rvec, tvec), reproj

    def track(self, image: np.ndarray, cam: CameraModel) -> TrackResult | None:
        """Find the board and solve this camera's pose. ``None`` when unusable."""
        det = detect_board(image, self.detector, self.board, min_corners=self.cfg.min_corners)
        if det is None:
            self.lost += 1
            return None

        # Intrinsics must match the frame we actually detected in.
        if (cam.width, cam.height) != det.image_size:
            cam = cam.scaled(*det.image_size)

        solved = self._solve(det, cam)
        if solved is None:
            self.lost += 1
            return None
        T_cam_board, reproj = solved

        if reproj > self.cfg.max_reproj_px:
            self.rejected += 1
            self.lost += 1
            return None

        T_world_cam = self.T_world_board @ invert(T_cam_board)

        if self._last is not None and self.cfg.max_jump_m > 0:
            jump = float(np.linalg.norm(T_world_cam[:3, 3] - self._last.position))
            if jump > self.cfg.max_jump_m:
                # Almost always a mirrored planar-pose solution rather than a
                # real teleport; dropping it is cheaper than integrating it.
                self.rejected += 1
                self.lost += 1
                return None

        if self._last is not None and self.cfg.smooth_alpha > 0:
            a = self.cfg.smooth_alpha
            T_world_cam[:3, 3] = a * self._last.position + (1 - a) * T_world_cam[:3, 3]

        result = TrackResult(
            T_world_cam=T_world_cam,
            T_cam_board=T_cam_board,
            n_corners=det.count,
            reproj_px=reproj,
            detection=det,
        )
        self._last = result
        self.tracked += 1
        return result

    def draw(self, image: np.ndarray, result: TrackResult, cam: CameraModel) -> np.ndarray:
        """Overlay the detected corners and the board axes, for live feedback."""
        out = image.copy()
        if (cam.width, cam.height) != result.detection.image_size:
            cam = cam.scaled(*result.detection.image_size)
        for pt in result.detection.img_points:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        rvec, _ = cv2.Rodrigues(result.T_cam_board[:3, :3])
        cv2.drawFrameAxes(
            out, cam.K, cam.dist, rvec, result.T_cam_board[:3, 3],
            self.spec.square_m * 3,
        )
        return out


class ViewpointCoverage:
    """Tracks how much of the viewing sphere a scan has actually sampled.

    A scan that circles one wall looks busy and produces a lot of points, but
    every voxel it fills was seen from nearly the same direction. Counting
    distinct viewing directions is a far better progress signal than counting
    frames or points.
    """

    def __init__(self, bins_azimuth: int = 12, bins_elevation: int = 3):
        self.bins_azimuth = bins_azimuth
        self.bins_elevation = bins_elevation
        self._seen: set[tuple[int, int]] = set()

    def add(self, T_world_cam: np.ndarray) -> None:
        pos = T_world_cam[:3, 3]
        r = float(np.linalg.norm(pos))
        if r < 1e-6:
            return
        az = np.arctan2(pos[1], pos[0])  # -pi..pi
        el = np.arcsin(np.clip(pos[2] / r, -1, 1))  # -pi/2..pi/2
        ai = int((az + np.pi) / (2 * np.pi) * self.bins_azimuth) % self.bins_azimuth
        ei = int(np.clip((el + np.pi / 2) / np.pi * self.bins_elevation, 0, self.bins_elevation - 1))
        self._seen.add((ai, ei))

    @property
    def fraction(self) -> float:
        return len(self._seen) / (self.bins_azimuth * self.bins_elevation)

    @property
    def count(self) -> int:
        return len(self._seen)

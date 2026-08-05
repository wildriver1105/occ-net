"""The reconstruction loop: frames in, occupancy grid out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calib.intrinsics import undistort_map
from .capture import Frame
from .config import RigConfig
from .depth.mono import MonoDepth
from .depth.stereo import StereoDepth
from .fusion.grid import OccupancyGrid
from .fusion.lift import lift_depth
from .geometry import CameraModel, RigCalibration, transform_points

MODES = ("mono", "stereo", "both")


@dataclass
class FrameResult:
    """What one camera contributed on one iteration."""

    camera: str
    depth: np.ndarray
    points_world: np.ndarray
    colors: np.ndarray | None
    rectified: np.ndarray
    ms: float = 0.0


@dataclass
class StepResult:
    per_camera: dict[str, FrameResult] = field(default_factory=dict)
    total_points: int = 0
    ms: float = 0.0


class _Rectifier:
    """Caches undistortion maps per camera and resolution."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, CameraModel]] = {}

    def get(self, cam: CameraModel, width: int, height: int):
        key = (cam.name, width, height)
        if key not in self._cache:
            scaled = cam if (cam.width, cam.height) == (width, height) else cam.scaled(width, height)
            self._cache[key] = undistort_map(scaled, alpha=0.0)
        return self._cache[key]


class Reconstructor:
    """Turns synchronised frames into occupancy evidence.

    ``mode='mono'`` runs monocular metric depth per camera and fuses both into
    one grid — this works immediately with only intrinsics, and is the mode that
    tolerates the two cameras pointing in different directions.

    ``mode='stereo'`` uses calibrated SGBM between the pair. It is metrically
    trustworthy but requires real overlap and a rigid mount.

    ``mode='both'`` runs stereo and uses it to continuously refit the monocular
    model's scale, which is the useful combination for an experiment rig: dense
    coverage from mono, metric truth from stereo.
    """

    def __init__(
        self,
        cfg: RigConfig,
        rig: RigCalibration,
        mode: str = "mono",
        grid: OccupancyGrid | None = None,
        depth_stride: int = 2,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.cfg = cfg
        self.rig = rig
        self.mode = mode
        self.depth_stride = depth_stride
        self.grid = grid or OccupancyGrid(cfg.grid)
        self._rect = _Rectifier()

        self.mono: MonoDepth | None = None
        if mode in ("mono", "both"):
            self.mono = MonoDepth(cfg.mono)

        self.stereo: StereoDepth | None = None
        if mode in ("stereo", "both"):
            names = list(rig.cameras)
            if len(names) < 2:
                raise ValueError("stereo mode needs two calibrated cameras")
            left = rig.reference
            right = next(n for n in names if n != left)
            if not np.any(rig.T_rig_cam(right)[:3, 3]):
                raise ValueError(
                    "stereo mode needs rig extrinsics; run `occnet calib stereo` first"
                )
            self.stereo = StereoDepth(rig, left, right, cfg.stereo)
            self.stereo_pair = (left, right)

        self.last_scale_fit: tuple[float, float] | None = None

    def warmup(self) -> None:
        """Load the depth model up front so the first frame is not a stall."""
        if self.mono is not None:
            self.mono.load()

    def _mono_pass(self, frames: dict[str, Frame], out: StepResult) -> None:
        assert self.mono is not None
        for name, frame in frames.items():
            cam = self.rig.cameras.get(name)
            if cam is None:
                continue
            h, w = frame.image.shape[:2]
            m1, m2, rect_cam = self._rect.get(cam, w, h)
            rectified = cv2.remap(frame.image, m1, m2, cv2.INTER_LINEAR)

            t0 = time.perf_counter()
            depth = self.mono(rectified)
            pts_cam, colors = lift_depth(
                depth, rect_cam, rectified,
                stride=self.depth_stride,
                max_depth_m=self.cfg.mono.max_depth_m,
                min_depth_m=self.cfg.mono.min_depth_m,
            )
            T = self.rig.T_rig_cam(name)
            pts_world = (
                transform_points(T, pts_cam).astype(np.float32)
                if len(pts_cam) else pts_cam
            )
            out.per_camera[name] = FrameResult(
                camera=name, depth=depth, points_world=pts_world, colors=colors,
                rectified=rectified, ms=(time.perf_counter() - t0) * 1000.0,
            )

    def _stereo_pass(self, frames: dict[str, Frame], out: StepResult) -> np.ndarray | None:
        assert self.stereo is not None
        left, right = self.stereo_pair
        if left not in frames or right not in frames:
            return None
        t0 = time.perf_counter()
        depth = self.stereo(frames[left].image, frames[right].image)
        rect_l, _ = self.stereo.rectify(frames[left].image, frames[right].image)
        pts_cam, colors = lift_depth(
            depth, self.stereo.rect_left, rect_l,
            stride=self.depth_stride,
            max_depth_m=self.cfg.stereo.max_depth_m,
            min_depth_m=self.cfg.stereo.min_depth_m,
        )
        pts_world = (
            transform_points(self.stereo.T_rig_rect, pts_cam).astype(np.float32)
            if len(pts_cam) else pts_cam
        )
        out.per_camera[f"{left}_stereo"] = FrameResult(
            camera=f"{left}_stereo", depth=depth, points_world=pts_world, colors=colors,
            rectified=rect_l, ms=(time.perf_counter() - t0) * 1000.0,
        )
        return depth

    def step(self, frames: dict[str, Frame], integrate: bool = True) -> StepResult:
        """Process one synchronised bundle."""
        t_start = time.perf_counter()
        out = StepResult()

        stereo_depth = None
        if self.mode in ("stereo", "both"):
            stereo_depth = self._stereo_pass(frames, out)

        if self.mode in ("mono", "both"):
            self._mono_pass(frames, out)

            # Anchor the monocular model to the stereo baseline. Stereo depth is
            # sparse but metric; mono is dense but drifts in scale.
            if self.mode == "both" and stereo_depth is not None and self.mono is not None:
                left = self.stereo_pair[0]
                mono_res = out.per_camera.get(left)
                if mono_res is not None and mono_res.depth.shape == stereo_depth.shape:
                    try:
                        self.last_scale_fit = self.mono.fit_scale(mono_res.depth, stereo_depth)
                    except ValueError:
                        pass  # not enough overlap this frame; keep the old fit

        if integrate:
            for name, res in out.per_camera.items():
                if len(res.points_world) == 0:
                    continue
                origin_name = name.replace("_stereo", "")
                if name.endswith("_stereo") and self.stereo is not None:
                    origin = self.stereo.T_rig_rect[:3, 3]
                else:
                    origin = self.rig.T_rig_cam(origin_name)[:3, 3]
                self.grid.integrate(res.points_world, origin)

        out.total_points = sum(len(r.points_world) for r in out.per_camera.values())
        out.ms = (time.perf_counter() - t_start) * 1000.0
        return out

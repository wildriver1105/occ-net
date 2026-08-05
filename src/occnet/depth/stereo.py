"""Calibrated stereo depth.

This is the metric ground truth of the rig: unlike a learned monocular model,
SGBM disparity is tied to the physically measured baseline, so it is what the
monocular model's scale should be fitted against.

It is only meaningful once both cameras are calibrated *and* rigidly mounted —
if either camera moves relative to the other after calibration, the extrinsics
are stale and the depth silently becomes wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import CameraModel, RigCalibration, invert


def order_stereo_pair(rig: RigCalibration, a: str, b: str) -> tuple[str, str]:
    """Return the pair as ``(left, right)`` in the order SGBM expects.

    ``stereoRectify`` and the block matcher assume the second camera sits to the
    right of the first, which shows up as a negative x translation going from
    the first camera's frame to the second. Feed them in the wrong order and
    every disparity comes out negative, so the depth map is silently empty
    rather than wrong — a failure that is easy to misread as "stereo doesn't
    work here".
    """
    T_b_a = invert(rig.T_rig_cam(b)) @ rig.T_rig_cam(a)
    return (a, b) if float(T_b_a[0, 3]) <= 0 else (b, a)


@dataclass
class StereoDepthConfig:
    min_disparity: int = 0
    num_disparities: int = 128  # must be divisible by 16
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    wls_filter: bool = True
    max_depth_m: float = 12.0
    min_depth_m: float = 0.15

    def __post_init__(self) -> None:
        if self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a multiple of 16")
        if self.block_size % 2 == 0:
            raise ValueError("block_size must be odd")


class StereoDepth:
    """Rectifies a calibrated pair and produces metric depth in the left frame."""

    def __init__(
        self,
        rig: RigCalibration,
        left: str,
        right: str,
        cfg: StereoDepthConfig | None = None,
    ):
        self.cfg = cfg or StereoDepthConfig()
        self.rig = rig
        self.left_name = left
        self.right_name = right

        cam_l, cam_r = rig.cameras[left], rig.cameras[right]
        self.size = (cam_l.width, cam_l.height)

        # Transform taking points from the left camera frame into the right one.
        T_right_left = invert(rig.T_rig_cam(right)) @ rig.T_rig_cam(left)
        R = np.ascontiguousarray(T_right_left[:3, :3], dtype=np.float64)
        # stereoRectify needs an explicit 3x1 column; a bare (3,) is read as a
        # row vector and fails inside the gemm.
        T = np.ascontiguousarray(T_right_left[:3, 3], dtype=np.float64).reshape(3, 1)
        dist_l = np.ascontiguousarray(cam_l.dist, dtype=np.float64).reshape(1, -1)
        dist_r = np.ascontiguousarray(cam_r.dist, dtype=np.float64).reshape(1, -1)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cam_l.K, dist_l, cam_r.K, dist_r, self.size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.Q = Q
        self.roi = roi1
        self._map_l = cv2.initUndistortRectifyMap(
            cam_l.K, dist_l, R1, P1, self.size, cv2.CV_16SC2
        )
        self._map_r = cv2.initUndistortRectifyMap(
            cam_r.K, dist_r, R2, P2, self.size, cv2.CV_16SC2
        )

        # Intrinsics of the *rectified* left camera — this is what depth lifting
        # must use, not the raw left intrinsics.
        self.rect_left = CameraModel(
            name=f"{left}_rect", width=self.size[0], height=self.size[1],
            K=P1[:3, :3], dist=np.zeros(5),
        )
        # Pose of the rectified left frame within the rig.
        T_rect = np.eye(4)
        T_rect[:3, :3] = R1
        self.T_rig_rect = rig.T_rig_cam(left) @ invert(T_rect)
        # After rectification the baseline is encoded in P2's translation term;
        # reading it from there rather than from raw T keeps depth consistent
        # with the rectified intrinsics that disparity is measured in.
        fx_rect = float(P2[0, 0])
        self.baseline_m = abs(float(P2[0, 3]) / fx_rect) if abs(fx_rect) > 1e-9 else float(
            np.linalg.norm(T)
        )
        if self.baseline_m < 1e-6:
            raise ValueError(
                "rectified baseline is zero — the rig extrinsics put both cameras in the "
                "same place. Run `occnet calib stereo` and check the reported baseline."
            )

        c = self.cfg
        self._matcher = cv2.StereoSGBM.create(
            minDisparity=c.min_disparity,
            numDisparities=c.num_disparities,
            blockSize=c.block_size,
            P1=8 * 3 * c.block_size ** 2,
            P2=32 * 3 * c.block_size ** 2,
            disp12MaxDiff=c.disp12_max_diff,
            uniquenessRatio=c.uniqueness_ratio,
            speckleWindowSize=c.speckle_window_size,
            speckleRange=c.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._wls = None
        self._right_matcher = None
        if c.wls_filter and hasattr(cv2, "ximgproc"):
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._matcher)
            self._wls = cv2.ximgproc.createDisparityWLSFilter(self._matcher)
            self._wls.setLambda(8000.0)
            self._wls.setSigmaColor(1.5)

    def rectify(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l = cv2.remap(left_bgr, *self._map_l, cv2.INTER_LINEAR)
        r = cv2.remap(right_bgr, *self._map_r, cv2.INTER_LINEAR)
        return l, r

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Sub-pixel disparity in the rectified left frame (float32, px)."""
        l, r = self.rectify(left_bgr, right_bgr)
        gl = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        raw = self._matcher.compute(gl, gr)
        if self._wls is not None and self._right_matcher is not None:
            raw_r = self._right_matcher.compute(gr, gl)
            raw = self._wls.filter(raw, gl, disparity_map_right=raw_r)
        # SGBM returns fixed-point disparity scaled by 16.
        return raw.astype(np.float32) / 16.0

    def __call__(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Metric depth (metres) in the rectified left frame; 0 marks invalid."""
        disp = self.disparity(left_bgr, right_bgr)
        fx = float(self.rect_left.K[0, 0])
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > max(self.cfg.min_disparity, 0.5)
        depth[valid] = (fx * self.baseline_m) / disp[valid]
        depth[(depth < self.cfg.min_depth_m) | (depth > self.cfg.max_depth_m)] = 0.0

        # Rectification leaves invalid borders; trim them so they do not carve
        # phantom free space through the occupancy grid.
        x, y, w, h = self.roi
        if w > 0 and h > 0:
            mask = np.zeros_like(depth, dtype=bool)
            mask[y:y + h, x:x + w] = True
            depth[~mask] = 0.0
        return depth

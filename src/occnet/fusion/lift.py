"""Turn depth maps into 3D points."""

from __future__ import annotations

import numpy as np

from ..geometry import CameraModel, transform_points


def lift_depth(
    depth: np.ndarray,
    cam: CameraModel,
    color_bgr: np.ndarray | None = None,
    stride: int = 1,
    max_depth_m: float = 12.0,
    min_depth_m: float = 0.15,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a depth map into camera-frame points.

    ``depth`` is expected to be metric with 0 marking invalid pixels, and to
    come from an *undistorted* image whose intrinsics are ``cam`` — feeding raw
    distorted frames here bends straight walls into curves.

    Returns ``(points (N,3) float32, colors (N,3) uint8 RGB or None)``.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    h, w = depth.shape
    if (w, h) != (cam.width, cam.height):
        cam = cam.scaled(w, h)

    d = depth[::stride, ::stride]
    us = np.arange(0, w, stride, dtype=np.float32)
    vs = np.arange(0, h, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(d) & (d >= min_depth_m) & (d <= max_depth_m)
    if not valid.any():
        return np.zeros((0, 3), np.float32), (None if color_bgr is None else np.zeros((0, 3), np.uint8))

    z = d[valid]
    x = (uu[valid] - cam.cx) / cam.fx * z
    y = (vv[valid] - cam.cy) / cam.fy * z
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors = None
    if color_bgr is not None:
        if color_bgr.shape[:2] != depth.shape:
            import cv2

            color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        sub = color_bgr[::stride, ::stride]
        colors = sub[valid][:, ::-1].copy()  # BGR -> RGB

    return points, colors


def lift_to_world(
    depth: np.ndarray,
    cam: CameraModel,
    T_world_cam: np.ndarray,
    color_bgr: np.ndarray | None = None,
    stride: int = 1,
    max_depth_m: float = 12.0,
    min_depth_m: float = 0.15,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project and place the points in the world frame."""
    pts, colors = lift_depth(
        depth, cam, color_bgr, stride=stride, max_depth_m=max_depth_m, min_depth_m=min_depth_m
    )
    if len(pts) == 0:
        return pts, colors
    return transform_points(T_world_cam, pts).astype(np.float32), colors

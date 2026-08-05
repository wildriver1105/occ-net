"""Inference overlay panels — the "what is the model seeing" view.

Kept separate from :mod:`occnet.viewer`, which only knows about raw camera
tiles. Everything here takes model output and renders it for a human.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def caption(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
            colour: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Translucent text block, drawn in-place."""
    x, y = origin
    pad, lh = 8, 21
    width = max((len(s) for s in lines), default=0) * 9 + pad * 2
    height = lh * len(lines) + pad
    patch = img[max(0, y):y + height, max(0, x):x + width]
    if patch.size:
        patch[:] = (patch * 0.45).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x + pad, y + pad + lh * i + 11), _FONT, 0.48, colour, 1, cv2.LINE_AA)


def depth_panel(depth: np.ndarray, max_depth: float = 8.0) -> np.ndarray:
    """Colour-map a metric depth map; invalid pixels stay black."""
    valid = depth > 0
    norm = np.zeros_like(depth, np.float32)
    if valid.any():
        norm[valid] = np.clip(depth[valid] / max_depth, 0, 1)
    # Invert so near = warm, far = cool, which reads more naturally as distance.
    img = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def near_field_mask(
    image_bgr: np.ndarray,
    depth: np.ndarray,
    threshold_m: float,
    colour: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint everything closer than ``threshold_m``.

    This is the occupancy analogue of a detection box: it does not say *what*
    the obstacle is, only that the space is taken and how close it is.
    """
    out = image_bgr.copy()
    hit = (depth > 0) & (depth < threshold_m)
    if hit.any():
        tint = np.zeros_like(out)
        tint[hit] = colour
        out = cv2.addWeighted(out, 1.0, tint, alpha, 0)

        # Outline the connected blobs so the alert reads at a glance.
        mask = (hit.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            region = depth[y:y + h, x:x + w]
            region = region[region > 0]
            if region.size == 0:
                continue
            nearest = float(np.percentile(region, 5))
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
            label = f"{nearest:.2f} m"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 8, y), colour, -1)
            cv2.putText(out, label, (x + 4, y - 5), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def bev_panel(
    bev: np.ndarray,
    voxel_size: float,
    origin_xz: tuple[float, float],
    size: tuple[int, int],
    sensor_x: float = 0.0,
    rings_m: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> np.ndarray:
    """Render a top-down occupancy map with range rings.

    ``bev`` is (depth, width) as returned by :meth:`OccupancyGrid.bev`, with row
    0 nearest the sensor.
    """
    img = cv2.applyColorMap((np.clip(bev, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    h, w = img.shape[:2]
    rows, cols = bev.shape
    x0, z0 = origin_xz
    px_per_m_x = w / (cols * voxel_size)
    px_per_m_z = h / (rows * voxel_size)
    # Sensor sits at world z0 (top of the image, since depth runs downward).
    sx = int(round((sensor_x - x0) * px_per_m_x))

    for r in rings_m:
        radius_px = int(round(r * px_per_m_z))
        if 0 < radius_px < h * 2:
            cv2.ellipse(img, (sx, 0), (int(r * px_per_m_x), radius_px), 0, 0, 180,
                        (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(img, f"{r:g}m", (sx + 6, min(h - 6, radius_px - 6)),
                        _FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.drawMarker(img, (sx, 0), (0, 255, 255), cv2.MARKER_TRIANGLE_DOWN, 14, 2)
    caption(img, ["BEV — top-down", f"{cols * voxel_size:.1f} x {rows * voxel_size:.1f} m"])
    return img


def stack_panels(panels: list[np.ndarray], height: int) -> np.ndarray:
    """Scale panels to a common height and lay them out left to right."""
    out = []
    for p in panels:
        h, w = p.shape[:2]
        scale = height / h
        out.append(cv2.resize(p, (max(1, int(round(w * scale))), height),
                              interpolation=cv2.INTER_AREA))
    return np.hstack(out) if out else np.zeros((height, height, 3), np.uint8)

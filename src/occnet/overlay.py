"""Inference overlay panels — the "what is the model seeing" view.

Kept separate from :mod:`occnet.viewer`, which only knows about raw camera
tiles. Everything here takes model output and renders it for a human.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Turbo runs dark purple -> blue -> green -> orange -> dark maroon, so both ends
# lose luminance. Clipping to this window keeps "near" a vivid red instead of a
# muddy brown and "far" a clear blue instead of near-black.
_TURBO_FAR, _TURBO_NEAR = 24, 240


def _turbo(norm: np.ndarray) -> np.ndarray:
    """Map normalised distance in [0,1] (0 = nearest) to BGR via clipped turbo."""
    idx = _TURBO_NEAR - np.clip(norm, 0, 1) * (_TURBO_NEAR - _TURBO_FAR)
    return cv2.applyColorMap(idx.astype(np.uint8), cv2.COLORMAP_TURBO)


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


def depth_panel(
    depth: np.ndarray,
    max_depth: float = 8.0,
    auto_scale: bool = True,
    percentiles: tuple[float, float] = (2.0, 98.0),
    legend: bool = True,
) -> np.ndarray:
    """Colour-map a metric depth map: near is red, far is blue, invalid is black.

    By default the colour scale spans what the frame actually contains rather
    than the full ``0..max_depth`` range. Fixing the scale to the grid's depth
    wastes most of the palette — a desk scene at 1-2 m inside an 8 m volume
    lands entirely in the orange band and reads as flat, even though the
    mapping is technically correct. Percentile endpoints also stop a handful of
    stray far pixels from compressing everything else.

    Pass ``auto_scale=False`` to compare frames on one absolute scale.
    """
    valid = depth > 0
    if auto_scale and valid.any():
        near, far = (float(v) for v in np.percentile(depth[valid], percentiles))
        if far - near < 1e-3:
            near, far = near, near + 1e-3
    else:
        near, far = 0.0, float(max_depth)

    norm = np.ones_like(depth, np.float32)
    if valid.any():
        norm[valid] = np.clip((depth[valid] - near) / (far - near), 0, 1)
    img = _turbo(norm)
    img[~valid] = 0

    if legend:
        _draw_depth_legend(img, near, far)
    return img


def _draw_depth_legend(img: np.ndarray, near: float, far: float) -> None:
    """Draw the colour ramp and its endpoints, so the mapping is never guessed."""
    h, w = img.shape[:2]
    bar_w, bar_h = min(220, w // 3), 12
    x0, y0 = w - bar_w - 14, h - bar_h - 24

    ramp = _turbo(np.linspace(0.0, 1.0, bar_w, dtype=np.float32)[None, :]).repeat(bar_h, axis=0)
    img[y0:y0 + bar_h, x0:x0 + bar_w] = ramp
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)

    for text, tx, align_right in ((f"{near:.2f}m", x0, False), (f"{far:.2f}m", x0 + bar_w, True)):
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.42, 1)
        pos = (tx - tw if align_right else tx, y0 + bar_h + 13)
        cv2.putText(img, text, pos, _FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


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


def render_inference_view(
    per_camera: list[tuple[str, np.ndarray, np.ndarray]],
    bev: np.ndarray | None,
    voxel_size: float,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    alert_m: float,
    row_height: int = 300,
    bev_note: str = "",
    auto_scale_depth: bool = True,
) -> np.ndarray:
    """Lay out a multi-camera inference view.

    ``per_camera`` is a list of ``(name, bgr_image, metric_depth)``. Each camera
    gets a row of [frame with near-field alert | depth], and the shared
    occupancy BEV is stacked on the right at the full height of those rows.

    Pure function of its inputs — no capture, no model — so the layout can be
    exercised without hardware.
    """
    max_depth = float(bounds_max[2])
    rows = []
    for name, image, depth in per_camera:
        rgb = near_field_mask(image, depth, alert_m)
        valid = depth[depth > 0]
        caption(rgb, [name, f"alert < {alert_m:.2f} m"])
        dpanel = depth_panel(depth, max_depth=max_depth, auto_scale=auto_scale_depth)
        caption(dpanel, [
            "depth  red=near  blue=far",
            f"range {valid.min():.2f}-{valid.max():.2f} m" if valid.size else "no valid depth",
        ])
        rows.append(stack_panels([rgb, dpanel], row_height))

    # Rows can differ in width if the cameras have different aspect ratios.
    width = max(r.shape[1] for r in rows)
    padded = [
        r if r.shape[1] == width
        else np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)))
        for r in rows
    ]
    left = np.vstack(padded)

    if bev is None:
        return left

    span_z = bounds_max[2] - bounds_min[2]
    bpanel = bev_panel(
        bev, voxel_size, (bounds_min[0], bounds_min[2]),
        size=(left.shape[0], left.shape[0]),
        rings_m=tuple(round(span_z * f, 1) for f in (0.25, 0.5, 0.75)),
    )
    if bev_note:
        caption(bpanel, [bev_note], origin=(10, bpanel.shape[0] - 34),
                colour=(120, 220, 255))
    return np.hstack([left, bpanel])


def stack_panels(panels: list[np.ndarray], height: int) -> np.ndarray:
    """Scale panels to a common height and lay them out left to right."""
    out = []
    for p in panels:
        h, w = p.shape[:2]
        scale = height / h
        out.append(cv2.resize(p, (max(1, int(round(w * scale))), height),
                              interpolation=cv2.INTER_AREA))
    return np.hstack(out) if out else np.zeros((height, height, 3), np.uint8)

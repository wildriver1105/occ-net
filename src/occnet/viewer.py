"""Live multi-camera window."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .capture import CameraRig, Frame

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10)) -> None:
    """Draw a translucent caption block in-place."""
    x, y = origin
    pad, lh = 8, 22
    width = max((len(s) for s in lines), default=0) * 9 + pad * 2
    height = lh * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    for i, text in enumerate(lines):
        cv2.putText(
            img, text, (x + pad, y + pad + lh * i + 12),
            _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


def compose(
    frames: dict[str, Frame],
    tile_height: int = 540,
    annotations: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Scale every camera to a common height and lay them out side by side."""
    tiles: list[np.ndarray] = []
    for name, frame in frames.items():
        img = frame.image
        h, w = img.shape[:2]
        scale = tile_height / h
        tile = cv2.resize(img, (max(1, int(round(w * scale))), tile_height), interpolation=cv2.INTER_AREA)
        lines = [f"{name}  {w}x{h}"]
        if annotations and name in annotations:
            lines.extend(annotations[name])
        _label(tile, lines)
        tiles.append(tile)
    if not tiles:
        return np.zeros((tile_height, tile_height, 3), np.uint8)
    return np.hstack(tiles)


def run_viewer(
    rig: CameraRig,
    window: str = "occnet — live rig",
    tile_height: int = 540,
    snapshot_dir: Path | None = None,
    on_frame: Callable[[dict[str, Frame], np.ndarray], np.ndarray | None] | None = None,
    on_key: Callable[[int, dict[str, Frame]], bool] | None = None,
) -> None:
    """Display the rig live until the user quits.

    ``on_frame`` may return a replacement canvas, letting callers (calibration,
    reconstruction) reuse this loop for their own overlays. ``on_key`` returns
    ``True`` to signal that the key was handled.
    """
    snapshot_dir = snapshot_dir or Path("out/snapshots")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    shots = 0
    t0 = time.monotonic()
    ui_fps = 0.0
    last = t0

    try:
        while True:
            frames = rig.read(timeout=5.0)
            if frames is None:
                print("no frames within timeout — is a camera still connected?")
                break

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 1e-6:
                ui_fps = 0.9 * ui_fps + 0.1 * (1.0 / dt) if ui_fps else 1.0 / dt

            ann = {
                name: [f"{rig.stats.fps.get(name, 0.0):5.1f} fps"]
                for name in frames
            }
            canvas = compose(frames, tile_height=tile_height, annotations=ann)
            if on_frame is not None:
                replaced = on_frame(frames, canvas)
                if replaced is not None:
                    canvas = replaced

            _label(
                canvas,
                [
                    f"ui {ui_fps:5.1f} fps   skew {rig.stats.skew_ms:5.1f} ms   t {now - t0:6.1f}s",
                    "q quit   s snapshot",
                ],
                origin=(10, canvas.shape[0] - 70),
            )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if on_key is not None and on_key(key, frames):
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                for name, frame in frames.items():
                    path = snapshot_dir / f"{stamp}_{shots:03d}_{name}.png"
                    cv2.imwrite(str(path), frame.image)
                print(f"saved snapshot {shots:03d} -> {snapshot_dir}")
                shots += 1
    finally:
        cv2.destroyAllWindows()
        # macOS needs a few event-loop turns to actually tear the window down.
        for _ in range(5):
            cv2.waitKey(1)

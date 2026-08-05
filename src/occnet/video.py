"""Video-file input.

Deliberately not routed through :class:`~occnet.capture.CameraRig`. A live rig
wants the *newest* frame and drops the rest; a file wants *every* frame, in
order, with no dropping. Sharing the threaded latest-frame machinery between the
two would quietly skip frames on playback.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .capture import Frame


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.path.name}  {self.width}x{self.height} @ {self.fps:.1f} fps  "
            f"{self.frames} frames ({self.duration_s:.1f}s)"
        )


class VideoSource:
    """Sequential reader over a video file."""

    def __init__(
        self,
        path: str | Path,
        name: str = "video",
        loop: bool = False,
        resize_width: int | None = None,
        start_frame: int = 0,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        self.name = name
        self.loop = loop
        self.resize_width = resize_width

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        self._cap = cap
        self.info = VideoInfo(
            path=self.path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._index = start_frame

    @property
    def output_size(self) -> tuple[int, int]:
        if self.resize_width is None or self.info.width == 0:
            return self.info.width, self.info.height
        scale = self.resize_width / self.info.width
        return self.resize_width, int(round(self.info.height * scale))

    def frames(self, max_frames: int | None = None) -> Iterator[Frame]:
        produced = 0
        while max_frames is None or produced < max_frames:
            ok, img = self._cap.read()
            if not ok:
                if self.loop and self.info.frames > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._index = 0
                    continue
                return
            if self.resize_width is not None:
                img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_AREA)
            yield Frame(
                image=img,
                timestamp=time.monotonic(),
                index=self._index,
                camera=self.name,
            )
            self._index += 1
            produced += 1

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def make_test_video(
    path: str | Path,
    frames: int = 90,
    width: int = 960,
    height: int = 540,
    fps: float = 30.0,
) -> Path:
    """Render a synthetic corridor fly-through.

    Only exists so the inference view can be exercised without a camera or a
    supplied clip. It has real perspective structure — receding textured walls
    plus boxes at known distances — so a depth model produces a meaningful
    gradient rather than noise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")

    rng = np.random.default_rng(3)
    # Boxes scattered down the corridor: (x, y, z, size, colour).
    boxes = [
        (rng.uniform(-1.4, 1.4), rng.uniform(-0.3, 0.7), 3.0 + i * 1.7, rng.uniform(0.3, 0.6),
         tuple(int(c) for c in rng.integers(60, 235, 3)))
        for i in range(7)
    ]
    fx = width / 2 / np.tan(np.radians(70) / 2)
    cx, cy = width / 2, height / 2

    def project(x, y, z):
        if z <= 0.05:
            return None
        return int(round(cx + fx * x / z)), int(round(cy + fx * y / z))

    for f in range(frames):
        cam_z = f * 0.06  # dolly forward
        img = np.full((height, width, 3), 28, np.uint8)

        # Walls, floor and ceiling as receding quads, drawn far-to-near.
        for i in range(28, 0, -1):
            z0, z1 = i * 0.8 - cam_z, (i - 1) * 0.8 - cam_z
            if z1 <= 0.15:
                continue
            shade = int(np.clip(235 - i * 7, 30, 235))
            for sx in (-1.8, 1.8):  # side walls
                quad = [project(sx, -1.2, z0), project(sx, 1.2, z0),
                        project(sx, 1.2, z1), project(sx, -1.2, z1)]
                if all(q is not None for q in quad):
                    tone = shade if i % 2 == 0 else int(shade * 0.72)
                    cv2.fillPoly(img, [np.array(quad, np.int32)], (tone, tone, int(tone * 0.9)))
            for sy in (-1.2, 1.2):  # ceiling and floor
                quad = [project(-1.8, sy, z0), project(1.8, sy, z0),
                        project(1.8, sy, z1), project(-1.8, sy, z1)]
                if all(q is not None for q in quad):
                    tone = int(shade * (0.55 if sy > 0 else 0.85))
                    cv2.fillPoly(img, [np.array(quad, np.int32)], (tone, tone, tone))

        for bx, by, bz, size, colour in sorted(boxes, key=lambda b: -b[2]):
            z = bz - cam_z
            if z <= 0.4:
                continue
            tl = project(bx - size / 2, by - size / 2, z)
            br = project(bx + size / 2, by + size / 2, z)
            if tl and br:
                fade = float(np.clip(1.4 - z / 14, 0.25, 1.0))
                cv2.rectangle(img, tl, br, tuple(int(c * fade) for c in colour), -1)
                cv2.rectangle(img, tl, br, (20, 20, 20), 2)

        writer.write(img)

    writer.release()
    return path

"""Multi-camera capture.

Two backends are provided because macOS capture is not uniform:

* ``opencv`` — lowest latency, works for most UVC devices.
* ``ffmpeg`` — a raw-video subprocess per camera. Slower to start but it
  handles devices that OpenCV's AVFoundation backend refuses to open, which in
  practice includes some Continuity Camera and vendor-webcam configurations.

Both are wrapped by :class:`CameraRig`, which runs one reader thread per camera
and always hands out the most recent frame rather than a queued one — for a
live occupancy rig, a dropped frame is much cheaper than a stale one.
"""

from __future__ import annotations

import fcntl
import os
import selectors
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .devices import Device, ffmpeg_bin

CAMERA_PERMISSION_HINT = (
    "Could not open the camera. On macOS the *host application* needs camera "
    "access: System Settings > Privacy & Security > Camera, and enable the app "
    "you launched this from (Terminal / iTerm / VS Code / Claude). "
    "You may need to fully quit and relaunch that app after granting access."
)


@dataclass
class Frame:
    """One captured image plus the timestamps needed to align cameras."""

    image: np.ndarray  # HxWx3, BGR, uint8
    timestamp: float  # host monotonic clock, seconds
    index: int  # monotonically increasing per camera
    camera: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


@dataclass
class CaptureConfig:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    backend: str = "auto"  # auto | opencv | ffmpeg
    # How long to wait for a camera's first frame. A device that is present but
    # not authorised (or asleep) never errors — it just stays silent — so every
    # open path needs a deadline rather than a blocking read.
    startup_timeout: float = 12.0


class CameraSource(ABC):
    """A single opened camera."""

    def __init__(self, device: Device, cfg: CaptureConfig, name: str):
        self.device = device
        self.cfg = cfg
        self.name = name
        self.actual_size: tuple[int, int] = (cfg.width, cfg.height)

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    @abstractmethod
    def release(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class OpenCVSource(CameraSource):
    def __init__(self, device: Device, cfg: CaptureConfig, name: str):
        super().__init__(device, cfg, name)
        import cv2

        self._cv2 = cv2
        cap = cv2.VideoCapture(device.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open device {device.index} ({device.name})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Keep the driver-side buffer as short as the backend allows so that
        # read() returns something close to "now".
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Continuity Camera and sleeping action cams need a moment to spin up,
        # so retry until the deadline rather than judging on the first read.
        deadline = time.monotonic() + cfg.startup_timeout
        probe = None
        while time.monotonic() < deadline:
            ok, probe = cap.read()
            if ok and probe is not None:
                break
            probe = None
            time.sleep(0.05)
        if probe is None:
            cap.release()
            raise RuntimeError(
                f"device {device.index} ({device.name}) opened but produced no frames "
                f"within {cfg.startup_timeout:g}s"
            )
        self.actual_size = (probe.shape[1], probe.shape[0])
        self._cap = cap

    def read(self) -> np.ndarray | None:
        ok, img = self._cap.read()
        return img if ok else None

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


class FFmpegSource(CameraSource):
    """Reads BGR24 rawvideo from an ffmpeg avfoundation subprocess.

    Addresses the device by *name* rather than index. AVFoundation reorders its
    device list at runtime — connecting an iPhone as a Continuity Camera can
    push the built-in camera from index 0 to index 1 — so an index captured a
    moment ago may already point at a different camera.
    """

    def __init__(self, device: Device, cfg: CaptureConfig, name: str):
        super().__init__(device, cfg, name)
        w, h = cfg.width, cfg.height
        # avfoundation accepts "<name>:<audio>" as well as "<index>:<audio>".
        selector = device.name if device.name else str(device.index)
        cmd = [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", str(cfg.fps),
            "-video_size", f"{w}x{h}",
            "-i", f"{selector}:none",
            "-vf", f"scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-an", "-sn", "-",
        ]
        self._frame_bytes = w * h * 3
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self.actual_size = (w, h)

        # Read through a non-blocking fd + selector so every read can carry a
        # deadline. An unauthorised camera produces no bytes and no error, so a
        # plain blocking read would hang the process indefinitely.
        self._fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._fd, selectors.EVENT_READ)
        self._closed = False

        first = self._read_exact(timeout=cfg.startup_timeout)
        if first is None:
            self.release()
            raise RuntimeError(
                f"no frames from device {device.index} ({device.name}) within "
                f"{cfg.startup_timeout:g}s: {self._stderr_tail() or 'ffmpeg produced no output'}"
            )
        self._pending: np.ndarray | None = first

    def _stderr_tail(self) -> str:
        if self._proc.stderr is None:
            return ""
        try:
            fd = self._proc.stderr.fileno()
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
            return (os.read(fd, 4000) or b"").decode(errors="replace").strip()
        except Exception:
            return ""

    def _read_exact(self, timeout: float | None = None) -> np.ndarray | None:
        if self._closed:
            return None
        buf = bytearray()
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(buf) < self._frame_bytes:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
            if not self._sel.select(timeout=remaining if remaining is not None else 1.0):
                if self._proc.poll() is not None:
                    return None  # ffmpeg exited
                continue
            try:
                chunk = os.read(self._fd, self._frame_bytes - len(buf))
            except BlockingIOError:
                continue
            except OSError:
                return None
            if not chunk:
                return None  # EOF
            buf.extend(chunk)
        w, h = self.actual_size
        return np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, w, 3).copy()

    def read(self) -> np.ndarray | None:
        if self._pending is not None:
            img, self._pending = self._pending, None
            return img
        # Steady state: a stalled camera should surface as an error, not a hang.
        return self._read_exact(timeout=5.0)

    def release(self) -> None:
        self._closed = True
        try:
            self._sel.close()
        except Exception:
            pass
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
            for pipe in (self._proc.stdout, self._proc.stderr):
                if pipe is not None:
                    pipe.close()
        except Exception:
            pass


def open_source(device: Device, cfg: CaptureConfig, name: str) -> CameraSource:
    """Open one camera, falling back between backends when ``backend='auto'``."""
    order = {
        "auto": ("opencv", "ffmpeg"),
        "opencv": ("opencv",),
        "ffmpeg": ("ffmpeg",),
    }[cfg.backend]

    errors: list[str] = []
    for backend in order:
        klass = OpenCVSource if backend == "opencv" else FFmpegSource
        try:
            return klass(device, cfg, name)
        except Exception as exc:  # noqa: BLE001 — we report every backend's failure
            errors.append(f"  {backend}: {exc}")
    raise RuntimeError(
        f"Failed to open '{name}' -> {device}\n" + "\n".join(errors) + "\n\n" + CAMERA_PERMISSION_HINT
    )


class _Reader(threading.Thread):
    """Pulls frames as fast as the device allows, keeping only the latest."""

    def __init__(self, source: CameraSource):
        super().__init__(daemon=True, name=f"reader-{source.name}")
        self.source = source
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._count = 0
        self.dropped = 0
        self.error: str | None = None

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                img = self.source.read()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                break
            if img is None:
                # Transient read failure; back off briefly instead of spinning.
                time.sleep(0.005)
                continue
            frame = Frame(
                image=img,
                timestamp=time.monotonic(),
                index=self._count,
                camera=self.source.name,
            )
            self._count += 1
            with self._lock:
                if self._latest is not None:
                    self.dropped += 1
                self._latest = frame

    def latest(self) -> Frame | None:
        with self._lock:
            frame, self._latest = self._latest, None
            return frame

    def peek(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()


@dataclass
class RigStats:
    fps: dict[str, float] = field(default_factory=dict)
    skew_ms: float = 0.0
    frames: int = 0


class CameraRig:
    """Runs N cameras concurrently and yields loosely time-aligned bundles."""

    def __init__(self, devices: dict[str, Device], cfg: CaptureConfig | None = None):
        self.cfg = cfg or CaptureConfig()
        self.devices = devices
        self._sources: dict[str, CameraSource] = {}
        self._readers: dict[str, _Reader] = {}
        self._last: dict[str, Frame] = {}
        self._fps_ema: dict[str, float] = {}
        self._last_ts: dict[str, float] = {}
        self.stats = RigStats()

    @property
    def names(self) -> list[str]:
        return list(self.devices)

    def start(self) -> "CameraRig":
        for name, dev in self.devices.items():
            src = open_source(dev, self.cfg, name)
            self._sources[name] = src
            reader = _Reader(src)
            reader.start()
            self._readers[name] = reader
        return self

    def sizes(self) -> dict[str, tuple[int, int]]:
        return {n: s.actual_size for n, s in self._sources.items()}

    def read(self, timeout: float = 2.0) -> dict[str, Frame] | None:
        """Return one frame per camera.

        Blocks until every camera has produced at least one frame, then returns
        each camera's newest frame — reusing the previous one for any camera
        that has not advanced. This is soft synchronisation; see
        ``stats.skew_ms`` for how far apart the bundle actually is.
        """
        deadline = time.monotonic() + timeout
        while True:
            bundle: dict[str, Frame] = {}
            for name, reader in self._readers.items():
                if reader.error:
                    raise RuntimeError(f"camera '{name}' failed: {reader.error}")
                frame = reader.latest()
                if frame is not None:
                    self._last[name] = frame
                    prev = self._last_ts.get(name)
                    if prev is not None:
                        dt = frame.timestamp - prev
                        if dt > 1e-6:
                            inst = 1.0 / dt
                            cur = self._fps_ema.get(name)
                            self._fps_ema[name] = inst if cur is None else 0.9 * cur + 0.1 * inst
                    self._last_ts[name] = frame.timestamp
                if name in self._last:
                    bundle[name] = self._last[name]

            if len(bundle) == len(self._readers):
                stamps = [f.timestamp for f in bundle.values()]
                self.stats.skew_ms = (max(stamps) - min(stamps)) * 1000.0
                self.stats.fps = dict(self._fps_ema)
                self.stats.frames += 1
                return bundle

            if time.monotonic() > deadline:
                return None
            time.sleep(0.002)

    def stop(self) -> None:
        for reader in self._readers.values():
            reader.stop()
        for reader in self._readers.values():
            reader.join(timeout=1.5)
        for src in self._sources.values():
            src.release()
        self._readers.clear()
        self._sources.clear()

    def __enter__(self) -> "CameraRig":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

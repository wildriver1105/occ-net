"""AVFoundation capture-device discovery on macOS.

OpenCV's AVFoundation backend indexes cameras in the same order that ffmpeg's
avfoundation demuxer reports them, so we use ffmpeg as the single source of
truth for names/indices and hand those indices to whichever capture backend
ends up being used.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field

# Substrings we use to guess which physical camera a device name refers to.
_ROLE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("insta360", ("insta360", "insta 360", "go 3", "go3")),
    ("iphone", ("iphone",)),
    ("builtin", ("macbook", "facetime", "built-in")),
]

# Devices that are technically cameras but useless as rig inputs.
_EXCLUDE_PATTERNS = ("desk view", "capture screen")


@dataclass(frozen=True)
class Device:
    """One AVFoundation video input."""

    index: int
    name: str
    role: str  # insta360 | iphone | builtin | other
    is_screen: bool = False

    @property
    def usable(self) -> bool:
        lowered = self.name.lower()
        return not self.is_screen and not any(p in lowered for p in _EXCLUDE_PATTERNS)

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.role})"


@dataclass
class Mode:
    """A resolution/framerate combination a device advertises."""

    width: int
    height: int
    fps_min: float
    fps_max: float
    pixel_format: str = ""

    def __str__(self) -> str:
        fps = (
            f"{self.fps_max:g}"
            if abs(self.fps_max - self.fps_min) < 1e-6
            else f"{self.fps_min:g}-{self.fps_max:g}"
        )
        pf = f" {self.pixel_format}" if self.pixel_format else ""
        return f"{self.width}x{self.height}@{fps}{pf}"


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing("ffmpeg not found on PATH — install it with `brew install ffmpeg`")
    return exe


def _classify(name: str) -> str:
    lowered = name.lower()
    for role, patterns in _ROLE_PATTERNS:
        if any(p in lowered for p in patterns):
            return role
    return "other"


_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def list_devices(include_unusable: bool = False) -> list[Device]:
    """Enumerate AVFoundation video devices via ffmpeg."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # ffmpeg always exits non-zero here; the device list is on stderr.
    devices: list[Device] = []
    in_video = False
    for raw in proc.stderr.splitlines():
        line = raw.split("] ", 1)[-1] if "] " in raw else raw
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue
        m = _DEVICE_LINE.search(line)
        if not m:
            continue
        idx, name = int(m.group(1)), m.group(2).strip()
        is_screen = "capture screen" in name.lower()
        devices.append(Device(index=idx, name=name, role=_classify(name), is_screen=is_screen))

    if include_unusable:
        return devices
    return [d for d in devices if d.usable]


# ffmpeg prints e.g. "  1920x1080@[1.000000 30.000000]fps" when a requested
# mode is unavailable, which is the only way to enumerate modes from the CLI.
_MODE_LINE = re.compile(
    r"(\d+)x(\d+)@\[([\d.]+)\s+([\d.]+)\]fps",
)


def probe_modes(index: int, timeout: float = 20.0) -> list[Mode]:
    """Ask a device for its supported modes.

    Works by requesting a deliberately impossible framerate; ffmpeg responds
    with the full list of supported modes before bailing out.
    """
    proc = subprocess.run(
        [
            ffmpeg_bin(), "-hide_banner",
            "-f", "avfoundation",
            "-framerate", "1000",
            "-i", f"{index}:none",
            "-t", "0.1", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    modes: list[Mode] = []
    seen: set[tuple[int, int, float, float]] = set()
    for m in _MODE_LINE.finditer(proc.stderr):
        w, h = int(m.group(1)), int(m.group(2))
        lo, hi = float(m.group(3)), float(m.group(4))
        key = (w, h, lo, hi)
        if key in seen:
            continue
        seen.add(key)
        modes.append(Mode(width=w, height=h, fps_min=lo, fps_max=hi))
    modes.sort(key=lambda m: (m.width * m.height, m.fps_max), reverse=True)
    return modes


@dataclass
class RigResolution:
    """Result of matching configured camera roles against what is plugged in."""

    resolved: dict[str, Device] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    available: list[Device] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def resolve_rig(wanted: dict[str, str | int]) -> RigResolution:
    """Map camera keys to concrete devices.

    ``wanted`` maps a camera key (``"cam0"``) to either an explicit device index,
    a role (``"insta360"``, ``"iphone"``, ``"builtin"``), or a name substring.
    """
    available = list_devices()
    by_index = {d.index: d for d in available}
    out = RigResolution(available=available)
    taken: set[int] = set()

    for key, selector in wanted.items():
        dev: Device | None = None
        if isinstance(selector, int) or (isinstance(selector, str) and selector.isdigit()):
            dev = by_index.get(int(selector))
        else:
            sel = str(selector).lower()
            for d in available:
                if d.index in taken:
                    continue
                if d.role == sel or sel in d.name.lower():
                    dev = d
                    break
        if dev is None:
            out.missing.append(key)
        else:
            out.resolved[key] = dev
            taken.add(dev.index)
    return out

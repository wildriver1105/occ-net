"""YAML configuration for the rig."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .calib.board import BoardSpec
from .capture import CaptureConfig
from .depth.mono import MonoDepthConfig
from .depth.stereo import StereoDepthConfig
from .fusion.grid import GridConfig

DEFAULT_CONFIG_PATH = Path("configs/rig.yaml")


@dataclass
class RigConfig:
    """Everything the rig needs, in one file.

    ``cameras`` maps a logical name to a device selector: a role
    (``insta360`` / ``iphone`` / ``builtin``), a name substring, or an explicit
    AVFoundation index.
    """

    cameras: dict[str, str | int] = field(
        default_factory=lambda: {"insta360": "insta360", "iphone": "iphone"}
    )
    # ``cameras: all`` in YAML sets this, meaning "use every camera found at
    # startup". More robust than naming devices, because AVFoundation reorders
    # its device list at runtime.
    discover: bool = False
    reference: str = "insta360"
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    board: BoardSpec = field(default_factory=BoardSpec)
    mono: MonoDepthConfig = field(default_factory=MonoDepthConfig)
    stereo: StereoDepthConfig = field(default_factory=StereoDepthConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    calib_dir: str = "data/calib"

    def __post_init__(self) -> None:
        if self.discover:
            return  # cameras are only known once the rig is scanned
        if self.reference not in self.cameras:
            raise ValueError(
                f"reference camera {self.reference!r} is not in cameras {list(self.cameras)}"
            )

    @property
    def rig_path(self) -> Path:
        return Path(self.calib_dir) / "rig.json"

    def intrinsics_path(self, camera: str) -> Path:
        return Path(self.calib_dir) / f"{camera}.json"

    def to_dict(self) -> dict:
        return {
            "cameras": "all" if self.discover else dict(self.cameras),
            "reference": self.reference,
            "capture": asdict(self.capture),
            "board": self.board.to_dict(),
            "mono": asdict(self.mono),
            "stereo": asdict(self.stereo),
            "grid": asdict(self.grid),
            "calib_dir": self.calib_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigConfig":
        def sub(key, klass, default_factory):
            raw = d.get(key)
            if not raw:
                return default_factory()
            fields = klass.__dataclass_fields__
            return klass(**{k: v for k, v in raw.items() if k in fields})

        board_raw = d.get("board") or {}
        raw_cams = d.get("cameras")
        discover = isinstance(raw_cams, str) and raw_cams.strip().lower() == "all"
        return cls(
            cameras={} if discover else (raw_cams or {"insta360": "insta360", "iphone": "iphone"}),
            discover=discover,
            reference=d.get("reference", "insta360"),
            capture=sub("capture", CaptureConfig, CaptureConfig),
            board=BoardSpec.from_dict(board_raw) if board_raw else BoardSpec(),
            mono=sub("mono", MonoDepthConfig, MonoDepthConfig),
            stereo=sub("stereo", StereoDepthConfig, StereoDepthConfig),
            grid=sub("grid", GridConfig, GridConfig),
            calib_dir=d.get("calib_dir", "data/calib"),
        )

    def save(self, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RigConfig":
        path = Path(path or DEFAULT_CONFIG_PATH)
        if not path.exists():
            return cls()
        return cls.from_dict(yaml.safe_load(path.read_text()) or {})

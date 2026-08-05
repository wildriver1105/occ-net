"""Camera models and rigid transforms shared across the pipeline.

Conventions (fixed everywhere in this package):

* Poses are camera-to-world unless the name says otherwise.
* ``T_wc`` maps a point in camera coordinates to world coordinates.
* Camera axes are OpenCV style: +x right, +y down, +z forward into the scene.
* Distances are metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Distortion model identifiers.
PINHOLE = "pinhole"  # 5-parameter Brown-Conrady (k1 k2 p1 p2 k3)
RATIONAL = "rational"  # 8-parameter, for wide-angle lenses like the GO 3
FISHEYE = "fisheye"  # Kannala-Brandt 4-parameter, for >150 deg FOV


@dataclass
class CameraModel:
    """Intrinsics for one camera at one resolution."""

    name: str
    width: int
    height: int
    K: np.ndarray  # 3x3
    dist: np.ndarray  # (N,)
    model: str = PINHOLE
    rms: float = 0.0  # reprojection error from calibration, pixels
    n_views: int = 0

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan2(self.width / 2, self.fx)))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan2(self.height / 2, self.fy)))

    def scaled(self, width: int, height: int) -> "CameraModel":
        """Rescale intrinsics to a different capture resolution."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return CameraModel(
            name=self.name, width=width, height=height, K=K, dist=self.dist.copy(),
            model=self.model, rms=self.rms, n_views=self.n_views,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "width": self.width, "height": self.height,
            "K": self.K.tolist(), "dist": self.dist.tolist(),
            "model": self.model, "rms": self.rms, "n_views": self.n_views,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(
            name=d["name"], width=int(d["width"]), height=int(d["height"]),
            K=np.array(d["K"]), dist=np.array(d["dist"]),
            model=d.get("model", PINHOLE), rms=float(d.get("rms", 0.0)),
            n_views=int(d.get("n_views", 0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def guess(cls, name: str, width: int, height: int, hfov_deg: float = 70.0) -> "CameraModel":
        """A rough uncalibrated model, so the pipeline can run before calibration.

        Good enough to see shapes; not good enough to trust metric scale.
        """
        fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)
        return cls(name=name, width=width, height=height, K=K, dist=np.zeros(5), model=PINHOLE)

    def pixel_rays(self) -> np.ndarray:
        """Unit ray direction per pixel in camera coordinates, shape HxWx3."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def rt_to_matrix(rvec_or_R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a Rodrigues vector (or 3x3 R) and translation."""
    import cv2

    arr = np.asarray(rvec_or_R, dtype=np.float64)
    R = arr if arr.shape == (3, 3) else cv2.Rodrigues(arr.reshape(3, 1))[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N,3) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


@dataclass
class RigCalibration:
    """Intrinsics for every camera plus their poses in a common rig frame.

    The rig frame is defined as the *reference* camera's frame, so that camera's
    pose is the identity.
    """

    reference: str
    cameras: dict[str, CameraModel] = field(default_factory=dict)
    poses: dict[str, np.ndarray] = field(default_factory=dict)  # name -> T_rig_cam (4x4)
    stereo_rms: float = 0.0

    def T_rig_cam(self, name: str) -> np.ndarray:
        return self.poses.get(name, np.eye(4))

    def baseline_m(self, a: str, b: str) -> float:
        ta = self.T_rig_cam(a)[:3, 3]
        tb = self.T_rig_cam(b)[:3, 3]
        return float(np.linalg.norm(ta - tb))

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "poses": {k: np.asarray(v).tolist() for k, v in self.poses.items()},
            "stereo_rms": self.stereo_rms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigCalibration":
        return cls(
            reference=d["reference"],
            cameras={k: CameraModel.from_dict(v) for k, v in d["cameras"].items()},
            poses={k: np.array(v) for k, v in d.get("poses", {}).items()},
            stereo_rms=float(d.get("stereo_rms", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RigCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))

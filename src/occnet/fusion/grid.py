"""Log-odds occupancy voxel grid.

The grid is the actual experiment substrate: every camera observation is folded
into it as evidence rather than as a fixed label, so disagreement between the
two cameras shows up as low-confidence voxels instead of silently overwriting.

Free space is carved by sampling along each ray at sub-voxel steps rather than
by exact DDA traversal. At voxel resolution the two are equivalent, and ray
sampling vectorises onto the GPU/MPS, which is what keeps this real-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


@dataclass
class GridConfig:
    """Extent and update behaviour of the grid.

    ``bounds_min``/``bounds_max`` are world-frame metres. With the reference
    camera at the origin looking down +z, a sensible starting volume for a desk
    rig is roughly x in [-2,2], y in [-1.5,1.5], z in [0,4].
    """

    voxel_size: float = 0.05
    bounds_min: tuple[float, float, float] = (-2.0, -1.5, 0.0)
    bounds_max: tuple[float, float, float] = (2.0, 1.5, 4.0)

    prob_hit: float = 0.85
    prob_miss: float = 0.35
    clamp_min: float = -2.5
    clamp_max: float = 3.5
    occupied_threshold: float = 0.65

    # Free space is not carved right up to the surface; this margin keeps ray
    # sampling from erasing the very voxels the hit just wrote.
    surface_margin_m: float = 0.06
    ray_step_ratio: float = 0.5  # sampling step as a fraction of voxel_size
    max_ray_m: float = 8.0
    # Carve free space using only every Nth ray. Surface hits always use every
    # point; free space is highly redundant between neighbouring rays, so this
    # is the cheapest knob when fusion cannot keep up with the cameras.
    carve_stride: int = 1
    # Which world axis points up, for BEV projection. 1 (y) suits a
    # camera-anchored frame, where OpenCV's +y points down; 2 (z) suits a
    # board-anchored world with the board lying flat.
    up_axis: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        # YAML round-trips these as lists; normalise so configs compare equal.
        self.bounds_min = tuple(float(v) for v in self.bounds_min)  # type: ignore[assignment]
        self.bounds_max = tuple(float(v) for v in self.bounds_max)  # type: ignore[assignment]

    @property
    def l_hit(self) -> float:
        return _logit(self.prob_hit)

    @property
    def l_miss(self) -> float:
        return _logit(self.prob_miss)


def fit_bounds(
    depth: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    voxel_budget: int = 3_000_000,
    percentile: float = 98.0,
    margin: float = 1.1,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Size a grid to whatever a depth map actually contains.

    An occupancy volume that does not match the scene is the most common reason
    a reconstruction looks empty: the points land outside the bounds and are
    silently dropped. This picks bounds from the observed depth distribution and
    then chooses the finest voxel size that stays within ``voxel_budget``.

    Returns ``(bounds_min, bounds_max, voxel_size)`` in the camera frame.
    """
    valid = depth[depth > 0]
    if valid.size == 0:
        raise ValueError("depth map has no valid pixels to fit bounds to")
    z_max = float(np.percentile(valid, percentile)) * margin
    x_half = z_max * np.tan(np.radians(hfov_deg) / 2)
    y_half = z_max * np.tan(np.radians(vfov_deg) / 2)

    extent = np.array([2 * x_half, 2 * y_half, z_max], dtype=np.float64)
    voxel = float(np.cbrt(extent.prod() / voxel_budget))
    # Snap to a readable size so logs and configs stay tidy.
    voxel = float(min(s for s in (0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2)
                      if s >= voxel * 0.999) if voxel <= 0.2 else round(voxel, 2))

    return (-x_half, -y_half, 0.0), (x_half, y_half, z_max), voxel


def _pick_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OccupancyGrid:
    """A dense log-odds occupancy volume."""

    def __init__(self, cfg: GridConfig | None = None):
        self.cfg = cfg or GridConfig()
        self.device = _pick_device(self.cfg.device)

        lo = np.asarray(self.cfg.bounds_min, dtype=np.float64)
        hi = np.asarray(self.cfg.bounds_max, dtype=np.float64)
        if np.any(hi <= lo):
            raise ValueError(f"bounds_max must exceed bounds_min, got {lo} .. {hi}")
        self.origin = lo
        self.dims = np.maximum(
            np.ceil((hi - lo) / self.cfg.voxel_size).astype(int), 1
        )
        self.bounds_max = lo + self.dims * self.cfg.voxel_size

        self.log_odds = torch.zeros(
            int(np.prod(self.dims)), dtype=torch.float32, device=self.device
        )
        self._origin_t = torch.tensor(self.origin, dtype=torch.float32, device=self.device)
        self._dims_t = torch.tensor(self.dims, dtype=torch.long, device=self.device)
        self.n_updates = 0

    # ---- indexing -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(d) for d in self.dims)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.dims))

    def _to_voxel(self, pts: torch.Tensor) -> torch.Tensor:
        """World metres -> integer voxel coordinates (may fall outside)."""
        return torch.floor((pts - self._origin_t) / self.cfg.voxel_size).long()

    def _flatten(self, vox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Voxel coords -> flat indices plus an in-bounds mask."""
        inside = ((vox >= 0) & (vox < self._dims_t)).all(dim=-1)
        nx, ny, nz = (int(d) for d in self.dims)
        flat = (vox[..., 0] * ny + vox[..., 1]) * nz + vox[..., 2]
        return flat, inside

    def voxel_centers(self, flat: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = (int(d) for d in self.dims)
        z = flat % nz
        y = (flat // nz) % ny
        x = flat // (nz * ny)
        vox = torch.stack([x, y, z], dim=-1).float()
        origin = self._origin_t.to(flat.device)
        return origin + (vox + 0.5) * self.cfg.voxel_size

    # ---- integration ----------------------------------------------------

    def _accumulate(self, flat: torch.Tensor, value: float) -> None:
        if flat.numel() == 0:
            return
        updates = torch.full_like(flat, value, dtype=torch.float32)
        self.log_odds.index_add_(0, flat, updates)

    @torch.inference_mode()
    def integrate(
        self,
        points_world: np.ndarray | torch.Tensor,
        sensor_origin: np.ndarray | torch.Tensor,
        carve_free: bool = True,
        chunk: int = 40_000,
    ) -> None:
        """Fold one depth observation into the grid.

        ``points_world`` are surface hits in world metres; ``sensor_origin`` is
        the camera centre those rays started from.
        """
        pts = torch.as_tensor(np.asarray(points_world), dtype=torch.float32, device=self.device)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_world must be (N,3), got {tuple(pts.shape)}")
        if pts.shape[0] == 0:
            return
        origin = torch.as_tensor(
            np.asarray(sensor_origin).reshape(3), dtype=torch.float32, device=self.device
        )

        step = self.cfg.voxel_size * self.cfg.ray_step_ratio
        max_steps = int(np.ceil(self.cfg.max_ray_m / step))

        for start in range(0, pts.shape[0], chunk):
            block = pts[start:start + chunk]

            hit_flat, hit_inside = self._flatten(self._to_voxel(block))
            self._accumulate(hit_flat[hit_inside], self.cfg.l_hit)

            if not carve_free:
                continue

            carve_from = block[:: max(1, self.cfg.carve_stride)]
            rays = carve_from - origin
            dist = rays.norm(dim=1, keepdim=True)
            usable = (dist.squeeze(1) > step) & (dist.squeeze(1) <= self.cfg.max_ray_m)
            if not usable.any():
                continue
            rays = rays[usable]
            dist = dist[usable]
            dirs = rays / dist

            # Only sample as far as the farthest ray in this block needs.
            steps = min(max_steps, int(torch.ceil(dist.max() / step).item()))
            if steps <= 1:
                continue
            t = torch.arange(1, steps + 1, device=self.device, dtype=torch.float32) * step
            samples = origin + dirs.unsqueeze(1) * t.view(1, -1, 1)  # (M, S, 3)

            free = t.view(1, -1) < (dist - self.cfg.surface_margin_m)
            free_flat, inside = self._flatten(self._to_voxel(samples))
            self._accumulate(free_flat[free & inside], self.cfg.l_miss)

        self.log_odds.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)
        self.n_updates += 1

    def decay(self, factor: float = 0.98) -> None:
        """Pull the whole grid toward "unknown".

        Useful for dynamic scenes, where evidence should expire rather than
        accumulate forever.
        """
        self.log_odds.mul_(factor)

    def reset(self) -> None:
        self.log_odds.zero_()
        self.n_updates = 0

    # ---- readout --------------------------------------------------------

    def probability_volume(self) -> np.ndarray:
        """Occupancy probability as a dense (nx,ny,nz) float32 array."""
        p = torch.sigmoid(self.log_odds).reshape(*self.shape)
        return p.detach().cpu().numpy().astype(np.float32)

    def occupied_indices(self, threshold: float | None = None) -> torch.Tensor:
        thr = _logit(threshold if threshold is not None else self.cfg.occupied_threshold)
        lo = self.log_odds
        if lo.device.type == "mps":
            # torch.nonzero on MPS forces a device sync and is far slower than
            # copying the (small) volume to the host and scanning it there.
            lo = lo.cpu()
        return torch.nonzero(lo > thr, as_tuple=False).squeeze(1)

    def occupied_points(self, threshold: float | None = None) -> np.ndarray:
        """World-frame centres of occupied voxels, (N,3) float32."""
        flat = self.occupied_indices(threshold)
        if flat.numel() == 0:
            return np.zeros((0, 3), np.float32)
        return self.voxel_centers(flat).detach().cpu().numpy().astype(np.float32)

    def bev_axes(self, up_axis: int | None = None) -> tuple[int, int]:
        """Which world axes the BEV's columns and rows correspond to."""
        up = self.cfg.up_axis if up_axis is None else up_axis
        remaining = [a for a in (0, 1, 2) if a != up]
        return remaining[0], remaining[1]  # (columns, rows)

    def bev(
        self,
        up_axis: int | None = None,
        height_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Top-down occupancy map, as a (rows, cols) float32 array.

        Collapses the vertical axis by taking the maximum occupancy probability
        in each column, which is the reading that matters for "can I drive/sail
        through this column of space".

        ``up_axis`` is which world axis points up. In a camera-anchored frame
        that is y (OpenCV's +y is down); in a board-anchored world it is z.
        ``height_range`` restricts the collapse to a slice of that axis, in
        world metres — use it to ignore the floor or the ceiling.
        """
        up = self.cfg.up_axis if up_axis is None else up_axis
        vol = self.probability_volume()  # (nx, ny, nz)
        if height_range is not None:
            lo, hi = height_range
            a0 = int(np.clip((lo - self.origin[up]) / self.cfg.voxel_size, 0, self.dims[up] - 1))
            a1 = int(np.clip((hi - self.origin[up]) / self.cfg.voxel_size, a0 + 1, self.dims[up]))
            vol = vol.take(indices=range(a0, a1), axis=up)
        # Transpose so the second remaining axis runs down the image.
        return vol.max(axis=up).T.astype(np.float32)

    def stats(self) -> dict[str, float]:
        lo = self.log_odds.cpu() if self.log_odds.device.type == "mps" else self.log_odds
        p = torch.sigmoid(lo)
        thr = self.cfg.occupied_threshold
        occ = int((p > thr).sum().item())
        free = int((p < 1 - thr).sum().item())
        total = self.n_voxels
        return {
            "voxels": float(total),
            "occupied": float(occ),
            "free": float(free),
            "unknown": float(total - occ - free),
            "occupied_pct": 100.0 * occ / total,
            "explored_pct": 100.0 * (occ + free) / total,
            "updates": float(self.n_updates),
        }

    # ---- export ---------------------------------------------------------

    def extract_mesh(self, threshold: float | None = None, smooth: bool = True):
        """Marching-cubes surface of the occupancy field.

        Returns a ``trimesh.Trimesh`` in world coordinates, or ``None`` when the
        volume has no crossing of the threshold yet.
        """
        from skimage import measure
        import trimesh

        vol = self.probability_volume()
        thr = threshold if threshold is not None else self.cfg.occupied_threshold
        if vol.min() > thr or vol.max() < thr:
            return None
        if smooth:
            from scipy.ndimage import gaussian_filter

            vol = gaussian_filter(vol, sigma=0.8)
        if vol.min() > thr or vol.max() < thr:
            return None

        verts, faces, normals, _ = measure.marching_cubes(
            vol, level=thr, spacing=(self.cfg.voxel_size,) * 3
        )
        verts = verts + self.origin
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            log_odds=self.log_odds.detach().cpu().numpy().reshape(*self.shape),
            origin=self.origin,
            voxel_size=self.cfg.voxel_size,
            occupied_threshold=self.cfg.occupied_threshold,
            n_updates=self.n_updates,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "OccupancyGrid":
        data = np.load(Path(path))
        lo = data["log_odds"]
        origin = data["origin"]
        vs = float(data["voxel_size"])
        cfg = GridConfig(
            voxel_size=vs,
            bounds_min=tuple(origin.tolist()),
            bounds_max=tuple((origin + np.array(lo.shape) * vs).tolist()),
            occupied_threshold=float(data.get("occupied_threshold", 0.65)),
            device=device,
        )
        grid = cls(cfg)
        grid.log_odds = torch.as_tensor(
            lo.reshape(-1), dtype=torch.float32, device=grid.device
        )
        grid.n_updates = int(data.get("n_updates", 0))
        return grid

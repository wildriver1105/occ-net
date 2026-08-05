"""Rerun-backed live 3D view of the rig.

Rerun is used rather than an Open3D window because it keeps the camera frusta,
the source images, the depth maps and the voxel field on one shared timeline —
when a reconstruction goes wrong, being able to scrub back to the exact frame
that poisoned the grid is most of the debugging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import CameraModel, RigCalibration


class RerunViz:
    """Thin wrapper so the rest of the code never touches the rerun API."""

    def __init__(
        self,
        app_id: str = "occnet",
        spawn: bool = True,
        save_path: str | Path | None = None,
    ):
        import rerun as rr

        self.rr = rr
        rr.init(app_id, spawn=spawn and save_path is None)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(path))
        # World is right-handed, +z up for the viewer's benefit; camera frames
        # keep OpenCV conventions and are related by their logged transforms.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._frame = 0

    def tick(self, frame: int | None = None, t: float | None = None) -> None:
        self._frame = self._frame + 1 if frame is None else frame
        self.rr.set_time("frame", sequence=self._frame)
        if t is not None:
            self.rr.set_time("time", duration=t)

    # ---- static scene setup --------------------------------------------

    def log_rig(self, rig: RigCalibration, image_plane_distance: float = 0.35) -> None:
        """Log every camera's pose and intrinsics once."""
        for name, cam in rig.cameras.items():
            T = rig.T_rig_cam(name)
            path = f"world/{name}"
            self.rr.log(
                path,
                self.rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
                static=True,
            )
            self.rr.log(
                f"{path}/image",
                self.rr.Pinhole(
                    image_from_camera=cam.K,
                    resolution=[cam.width, cam.height],
                    camera_xyz=self.rr.ViewCoordinates.RDF,
                    image_plane_distance=image_plane_distance,
                ),
                static=True,
            )

    def log_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        """Draw the occupancy volume's extent as a wireframe box."""
        mins, maxs = np.asarray(mins, float), np.asarray(maxs, float)
        self.rr.log(
            "world/volume",
            self.rr.Boxes3D(
                centers=[(mins + maxs) / 2],
                half_sizes=[(maxs - mins) / 2],
                colors=[(90, 90, 110)],
                fill_mode="MajorWireframe",
            ),
            static=True,
        )

    # ---- per-frame ------------------------------------------------------

    def log_image(self, camera: str, image_bgr: np.ndarray) -> None:
        self.rr.log(f"world/{camera}/image/rgb", self.rr.Image(image_bgr[:, :, ::-1]))

    def log_depth(self, camera: str, depth_m: np.ndarray, max_depth: float = 8.0) -> None:
        self.rr.log(
            f"world/{camera}/image/depth",
            self.rr.DepthImage(depth_m, meter=1.0, depth_range=(0.0, max_depth)),
        )

    def log_points(
        self,
        entity: str,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        radius: float = 0.006,
    ) -> None:
        if len(points) == 0:
            self.rr.log(f"world/{entity}", self.rr.Points3D(np.zeros((0, 3))))
            return
        self.rr.log(
            f"world/{entity}",
            self.rr.Points3D(points, colors=colors, radii=radius),
        )

    def log_occupancy(
        self,
        points: np.ndarray,
        voxel_size: float,
        colors: np.ndarray | None = None,
        entity: str = "occupancy",
    ) -> None:
        """Draw occupied voxels as boxes so the discretisation stays visible."""
        if len(points) == 0:
            self.rr.log(f"world/{entity}", self.rr.Boxes3D(centers=np.zeros((0, 3)), half_sizes=np.zeros((0, 3))))
            return
        half = np.full((len(points), 3), voxel_size / 2, dtype=np.float32)
        if colors is None:
            colors = height_colors(points)
        self.rr.log(
            f"world/{entity}",
            self.rr.Boxes3D(centers=points, half_sizes=half, colors=colors, fill_mode="Solid"),
        )

    def log_mesh(self, mesh, entity: str = "mesh") -> None:
        if mesh is None:
            return
        self.rr.log(
            f"world/{entity}",
            self.rr.Mesh3D(
                vertex_positions=np.asarray(mesh.vertices, np.float32),
                triangle_indices=np.asarray(mesh.faces, np.uint32),
                vertex_normals=np.asarray(mesh.vertex_normals, np.float32),
            ),
        )

    def log_scalar(self, name: str, value: float) -> None:
        self.rr.log(f"stats/{name}", self.rr.Scalars(float(value)))


def height_colors(points: np.ndarray, axis: int = 2) -> np.ndarray:
    """Colour points by height so structure reads without texture."""
    import matplotlib

    v = points[:, axis]
    lo, hi = float(v.min()), float(v.max())
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(v)
    cmap = matplotlib.colormaps["turbo"]
    return (cmap(norm)[:, :3] * 255).astype(np.uint8)

"""End-to-end self-test with synthetic data — no cameras required.

Validates the parts that are easy to get subtly wrong and hard to notice on live
video: intrinsic recovery, extrinsic recovery, depth lifting, and occupancy
fusion. Run it after touching anything in calib/, fusion/, or geometry.py.

    uv run python scripts/selftest.py
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from occnet.calib.board import BoardSpec, make_board
from occnet.calib.extrinsics import PairView, calibrate_rig
from occnet.calib.intrinsics import CalibrationSet, calibrate_intrinsics, detect_board
from occnet.config import RigConfig
from occnet.fusion.grid import GridConfig, OccupancyGrid
from occnet.fusion.lift import lift_depth
from occnet.geometry import PINHOLE, CameraModel, invert, rt_to_matrix, transform_points

W, H = 1280, 720
PASSES: list[str] = []
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSES if ok else FAILS).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")


def truth_camera(name: str, fx: float) -> CameraModel:
    K = np.array([[fx, 0, W / 2 - 6], [0, fx, H / 2 + 4], [0, 0, 1]], float)
    return CameraModel(name=name, width=W, height=H, K=K, dist=np.zeros(5), model=PINHOLE)


def render_board(spec: BoardSpec, cam: CameraModel, T_cam_board: np.ndarray) -> np.ndarray:
    """Project the printed board plane into a virtual camera via a homography."""
    board_px = 1400
    sx = board_px / (spec.squares_x * spec.square_m)
    sy = int(round(spec.squares_y * spec.square_m * sx))
    board_obj, _ = make_board(spec)
    flat = board_obj.generateImage((board_px, sy), marginSize=0, borderBits=1)

    # Board-image pixels -> board metres (board plane is z=0).
    corners_px = np.array([[0, 0], [board_px, 0], [board_px, sy], [0, sy]], np.float32)
    corners_m = np.array(
        [
            [0, 0, 0],
            [spec.squares_x * spec.square_m, 0, 0],
            [spec.squares_x * spec.square_m, spec.squares_y * spec.square_m, 0],
            [0, spec.squares_y * spec.square_m, 0],
        ],
        np.float64,
    )
    pts_cam = transform_points(T_cam_board, corners_m)
    proj = (cam.K @ pts_cam.T).T
    proj = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

    Hm = cv2.getPerspectiveTransform(corners_px, proj)
    canvas = cv2.warpPerspective(
        flat, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=128
    )
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def board_poses() -> list[np.ndarray]:
    """A spread of board placements covering the frame with strong tilt diversity.

    Tilt is what makes this work. A fronto-parallel board leaves focal length and
    board distance perfectly ambiguous — you can double both and reproject
    identically — so views clustered near head-on give a low reprojection error
    while the focal length is badly wrong. Real captures need the same 20-45
    degree tilts across multiple axes that this generates.
    """
    poses = []
    rng = np.random.default_rng(7)
    tilts = [
        (0.50, 0.0), (-0.50, 0.0), (0.0, 0.50), (0.0, -0.50),
        (0.38, 0.38), (-0.38, 0.38), (0.38, -0.38), (-0.38, -0.38),
        (0.65, 0.15), (0.15, 0.65),
    ]
    for i, (rx, ry) in enumerate(tilts):
        for tx, ty, tz in ((-0.08, -0.06, 0.55), (0.08, 0.06, 0.70), (0.0, 0.0, 0.62)):
            rvec = np.array([rx, ry, rng.uniform(-0.15, 0.15)])
            t = np.array([tx - 0.11, ty - 0.16, tz])
            poses.append(rt_to_matrix(rvec, t))
    return poses


def test_intrinsics(spec: BoardSpec) -> CameraModel | None:
    print("\n[1] intrinsic calibration from synthetic ChArUco views")
    truth = truth_camera("virt", fx=900.0)
    board_obj, detector = make_board(spec)
    cal = CalibrationSet(name="virt")

    for T in board_poses():
        img = render_board(spec, truth, T)
        det = detect_board(img, detector, board_obj)
        if det is not None:
            cal.add(det)

    check("board detected in enough views", len(cal) >= 12, f"{len(cal)} views")
    if len(cal) < 12:
        return None
    check("image coverage is broad", cal.coverage > 0.5, f"{cal.coverage * 100:.0f}%")

    est = calibrate_intrinsics(cal, model=PINHOLE, min_views=8)
    fx_err = abs(est.fx - truth.fx) / truth.fx * 100
    cx_err = abs(est.cx - truth.cx)
    check("reprojection rms below 1 px", est.rms < 1.0, f"{est.rms:.3f} px")
    check("focal length within 2%", fx_err < 2.0, f"{fx_err:.2f}% off ({est.fx:.1f} vs {truth.fx:.1f})")
    check("principal point within 15 px", cx_err < 15, f"{cx_err:.1f} px off")
    return truth


def test_extrinsics(spec: BoardSpec, truth: CameraModel) -> None:
    print("\n[2] rig extrinsics recovery")
    # Ground truth: the second camera sits 12 cm to the right, toed in slightly.
    T_ref_other = rt_to_matrix(np.array([0.0, -0.12, 0.0]), np.array([0.12, 0.0, 0.0]))
    cam_ref = truth_camera("ref", fx=900.0)
    cam_oth = truth_camera("oth", fx=880.0)
    board_obj, detector = make_board(spec)

    views: list[PairView] = []
    for T_ref_board in board_poses():
        T_oth_board = invert(T_ref_other) @ T_ref_board
        d_ref = detect_board(render_board(spec, cam_ref, T_ref_board), detector, board_obj)
        d_oth = detect_board(render_board(spec, cam_oth, T_oth_board), detector, board_obj)
        if d_ref is not None and d_oth is not None:
            views.append(PairView(ref=d_ref, other=d_oth))

    check("shared views collected", len(views) >= 8, f"{len(views)} pairs")
    if len(views) < 8:
        return

    for method in ("stereo", "pnp"):
        rig = calibrate_rig("ref", {"ref": cam_ref, "oth": cam_oth}, {"oth": views}, method=method)
        est = rig.T_rig_cam("oth")
        t_err = np.linalg.norm(est[:3, 3] - T_ref_other[:3, 3]) * 1000
        R_err = np.degrees(
            np.arccos(np.clip((np.trace(est[:3, :3].T @ T_ref_other[:3, :3]) - 1) / 2, -1, 1))
        )
        check(f"{method}: baseline within 5 mm", t_err < 5.0, f"{t_err:.2f} mm error")
        check(f"{method}: rotation within 1 deg", R_err < 1.0, f"{R_err:.3f} deg error")


def synthetic_room(cam: CameraModel, T_world_cam: np.ndarray) -> np.ndarray:
    """Depth map of a box room: back wall, floor, and a slab in the middle."""
    rays = cam.pixel_rays().reshape(-1, 3)
    R = T_world_cam[:3, :3]
    origin = T_world_cam[:3, 3]
    dirs = rays @ R.T

    best = np.full(len(dirs), np.inf)
    # Planes as (normal, offset) with n . x = d, in world coordinates.
    planes = [
        (np.array([0.0, 0.0, 1.0]), 3.0),   # back wall at z = 3
        (np.array([0.0, 1.0, 0.0]), 1.0),   # floor at y = 1 (camera +y is down)
        (np.array([0.0, 0.0, 1.0]), 1.6),   # slab face at z = 1.6
    ]
    for n, d in planes:
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - origin @ n) / denom
        hit = (denom > 1e-6) & (t > 0)
        if n[2] == 1.0 and d == 1.6:
            # The slab only occupies a window in the middle of the view.
            p = origin + dirs * t[:, None]
            hit &= (np.abs(p[:, 0]) < 0.4) & (np.abs(p[:, 1]) < 0.4)
        best = np.where(hit & (t < best), t, best)

    depth = (best * dirs[:, 2]).astype(np.float32)  # ray length -> z depth
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(cam.height, cam.width)


def test_fusion() -> None:
    print("\n[3] depth lifting and occupancy fusion")
    cam = CameraModel.guess("virt", 640, 360, hfov_deg=80.0)
    T_world_cam = np.eye(4)
    depth = synthetic_room(cam, T_world_cam)
    check("synthetic depth is mostly valid", (depth > 0).mean() > 0.9, f"{(depth > 0).mean() * 100:.1f}% valid")

    pts, colors = lift_depth(depth, cam, np.zeros((360, 640, 3), np.uint8), stride=2, max_depth_m=6.0)
    check("lifted a point cloud", len(pts) > 10_000, f"{len(pts)} points")

    # A point at the image centre must land on the slab, straight ahead at 1.6 m.
    centre = depth[180, 320]
    check("centre depth hits the slab at 1.6 m", abs(centre - 1.6) < 0.05, f"{centre:.3f} m")

    grid = OccupancyGrid(
        GridConfig(
            voxel_size=0.05,
            bounds_min=(-2.0, -1.5, 0.0),
            bounds_max=(2.0, 1.5, 3.5),
            device="cpu",
        )
    )
    grid.integrate(pts, T_world_cam[:3, 3])
    stats = grid.stats()
    check("voxels marked occupied", stats["occupied"] > 500, f"{int(stats['occupied'])} voxels")
    check("free space was carved", stats["free"] > stats["occupied"], f"{int(stats['free'])} free")

    occ = grid.occupied_points()
    check("occupied voxels stay inside the volume",
          bool(np.all(occ >= np.array(grid.origin) - 1e-6) and np.all(occ <= grid.bounds_max + 1e-6)),
          f"{len(occ)} voxels")

    # The slab face is at z=1.6; occupied voxels near the optical axis should
    # cluster there rather than at the back wall.
    axis = occ[(np.abs(occ[:, 0]) < 0.2) & (np.abs(occ[:, 1]) < 0.2)]
    if len(axis):
        z_med = float(np.median(axis[:, 2]))
        check("slab surface recovered at the right depth", abs(z_med - 1.6) < 0.15, f"median z = {z_med:.3f} m")
    else:
        check("slab surface recovered at the right depth", False, "no voxels near the optical axis")

    # Free space must be carved in front of the slab, not behind it.
    import torch

    probe = torch.tensor([[0.0, 0.0, 0.8]], dtype=torch.float32)
    flat, inside = grid._flatten(grid._to_voxel(probe))
    if bool(inside[0]):
        p_free = float(torch.sigmoid(grid.log_odds[flat[0]]).item())
        check("space in front of the slab reads as free", p_free < 0.35, f"p(occupied) = {p_free:.3f}")

    mesh = grid.extract_mesh()
    check("marching cubes produced a mesh", mesh is not None and len(mesh.faces) > 100,
          f"{len(mesh.faces)} faces" if mesh is not None else "none")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = grid.save(f"{tmp}/grid.npz")
        reloaded = OccupancyGrid.load(path, device="cpu")
        same = np.allclose(reloaded.probability_volume(), grid.probability_volume(), atol=1e-5)
        check("grid survives a save/load round-trip", same and reloaded.shape == grid.shape)


def test_overlay() -> None:
    print("\n[4] inference overlay rendering")
    from occnet.fusion.grid import fit_bounds
    from occnet.overlay import near_field_mask, render_inference_view

    cam = CameraModel.guess("virt", 640, 360, hfov_deg=80.0)
    depth = synthetic_room(cam, np.eye(4))
    image = np.full((360, 640, 3), 90, np.uint8)

    lo, hi, voxel = fit_bounds(depth, cam.hfov_deg, cam.vfov_deg)
    check("fit_bounds covers the observed depth", hi[2] > float(depth[depth > 0].max()) * 0.9,
          f"z up to {hi[2]:.2f} m, voxel {voxel * 100:.1f} cm")
    n_vox = np.prod(np.ceil((np.array(hi) - np.array(lo)) / voxel))
    check("fit_bounds respects the voxel budget", n_vox <= 3_000_000 * 1.5, f"{n_vox / 1e6:.2f} M voxels")

    # The slab sits at 1.6 m, so a 2 m alert must mark pixels and a 0.5 m one must not.
    hot = near_field_mask(image, depth, 2.0)
    cold = near_field_mask(image, depth, 0.5)
    check("near-field alert marks close surfaces", not np.array_equal(hot, image))
    check("near-field alert ignores distant surfaces", np.array_equal(cold, image))

    grid = OccupancyGrid(
        GridConfig(voxel_size=0.05, bounds_min=(-2.0, -1.5, 0.0), bounds_max=(2.0, 1.5, 3.5), device="cpu")
    )
    pts, _ = lift_depth(depth, cam, None, stride=3, max_depth_m=6.0)
    grid.integrate(pts, np.zeros(3))

    bev = grid.bev()
    check("bev collapses to a 2D map", bev.ndim == 2 and bev.shape == (grid.shape[2], grid.shape[0]),
          f"{bev.shape}")
    check("bev shows occupancy", float(bev.max()) > 0.65, f"max p = {bev.max():.2f}")

    canvas = render_inference_view(
        [("cam0", image, depth), ("cam1", image, depth)],
        bev, 0.05, (-2.0, -1.5, 0.0), (2.0, 1.5, 3.5), alert_m=2.0, row_height=240,
    )
    # Two rows of [rgb | depth] plus a square BEV column on the right.
    check("inference view composes two camera rows", canvas.shape[0] == 480, f"{canvas.shape}")
    check("inference view is not blank", int(canvas.max()) > 0 and float(canvas.std()) > 5)


def test_config() -> None:
    print("\n[5] configuration round-trip")
    import tempfile

    for name in ("configs/rig.yaml", "configs/rig-builtin.yaml"):
        try:
            cfg = RigConfig.load(name)
            check(f"{name} parses", True, f"reference={cfg.reference}, cameras={list(cfg.cameras)}")
        except Exception as exc:  # noqa: BLE001
            check(f"{name} parses", False, str(exc))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = RigConfig()
        cfg.save(f"{tmp}/rig.yaml")
        back = RigConfig.load(f"{tmp}/rig.yaml")
        check("config save/load preserves values",
              back.to_dict() == cfg.to_dict())


def main() -> int:
    spec = BoardSpec()
    print("occnet self-test (synthetic — no cameras needed)")
    truth = test_intrinsics(spec)
    if truth is not None:
        test_extrinsics(spec, truth)
    test_fusion()
    test_overlay()
    test_config()

    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print(f"  failed: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

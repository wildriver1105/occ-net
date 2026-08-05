"""Rig extrinsics — where each camera sits relative to the reference camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import FISHEYE, CameraModel, RigCalibration, invert, rt_to_matrix
from .intrinsics import Detection


@dataclass
class PairView:
    """One instant at which two cameras both saw the board."""

    ref: Detection
    other: Detection


def _common_corners(a: Detection, b: Detection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corners visible in both views, aligned by ChArUco corner id."""
    shared, ia, ib = np.intersect1d(a.corner_ids, b.corner_ids, return_indices=True)
    return a.obj_points[ia], a.img_points[ia], b.img_points[ib]


def _board_pose(det: Detection, cam: CameraModel) -> np.ndarray | None:
    """Pose of the board in the camera frame, as T_cam_board."""
    if det.count < 6:
        return None
    obj = det.obj_points.astype(np.float64)
    img = det.img_points.astype(np.float64)
    if cam.model == FISHEYE:
        img = cv2.fisheye.undistortPoints(
            img.reshape(-1, 1, 2), cam.K, cam.dist.reshape(4, 1), P=cam.K
        ).reshape(-1, 2)
        dist = np.zeros(5)
    else:
        dist = cam.dist
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, cam.K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    return rt_to_matrix(rvec, tvec)


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Geodesic-ish mean of rigid transforms.

    Rotations are averaged by taking the principal eigenvector of the summed
    quaternion outer products, which is the standard closed-form rotation mean;
    translations are averaged directly.
    """
    from scipy.spatial.transform import Rotation

    rots = Rotation.from_matrix(np.stack([T[:3, :3] for T in transforms]))
    mean_R = rots.mean().as_matrix()
    mean_t = np.mean([T[:3, 3] for T in transforms], axis=0)
    out = np.eye(4)
    out[:3, :3] = mean_R
    out[:3, 3] = mean_t
    return out


def _reject_outliers(transforms: list[np.ndarray], n_sigma: float = 2.0) -> list[np.ndarray]:
    """Drop views whose translation is far from the consensus."""
    if len(transforms) < 4:
        return transforms
    t = np.stack([T[:3, 3] for T in transforms])
    med = np.median(t, axis=0)
    d = np.linalg.norm(t - med, axis=1)
    scale = np.median(np.abs(d - np.median(d))) * 1.4826
    if scale < 1e-9:
        return transforms
    keep = d < np.median(d) + n_sigma * scale
    kept = [T for T, k in zip(transforms, keep) if k]
    return kept if len(kept) >= 3 else transforms


def calibrate_rig(
    ref_name: str,
    cameras: dict[str, CameraModel],
    pair_views: dict[str, list[PairView]],
    method: str = "stereo",
    min_views: int = 6,
) -> RigCalibration:
    """Solve each non-reference camera's pose in the reference camera's frame.

    ``method='stereo'`` runs OpenCV's joint stereo optimisation over the shared
    corners and is the more accurate option when the two views overlap well.
    ``method='pnp'`` solves each camera's board pose independently and averages
    the relative transform — slower to converge but it tolerates views where the
    cameras share only a handful of corners, which happens easily when a
    wide-FOV action cam is paired with a phone.
    """
    if ref_name not in cameras:
        raise ValueError(f"reference camera {ref_name!r} has no intrinsics")

    rig = RigCalibration(reference=ref_name, cameras=dict(cameras))
    rig.poses[ref_name] = np.eye(4)
    worst_rms = 0.0

    for name, views in pair_views.items():
        if name == ref_name:
            continue
        if name not in cameras:
            raise ValueError(f"camera {name!r} has no intrinsics")
        if len(views) < min_views:
            raise ValueError(
                f"{name}: need at least {min_views} views where both cameras see the board, "
                f"have {len(views)}. Hold the board where both lenses can see it."
            )

        ref_cam, other_cam = cameras[ref_name], cameras[name]

        if method == "stereo" and ref_cam.model != FISHEYE and other_cam.model != FISHEYE:
            obj_l: list[np.ndarray] = []
            img_ref: list[np.ndarray] = []
            img_oth: list[np.ndarray] = []
            for v in views:
                o, ia, ib = _common_corners(v.ref, v.other)
                if len(o) >= 6:
                    # OpenCV wants Point3f/Point2f here, i.e. float32.
                    obj_l.append(o.reshape(-1, 1, 3).astype(np.float32))
                    img_ref.append(ia.reshape(-1, 1, 2).astype(np.float32))
                    img_oth.append(ib.reshape(-1, 1, 2).astype(np.float32))
            if len(obj_l) >= min_views:
                rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                    obj_l, img_ref, img_oth,
                    ref_cam.K, ref_cam.dist, other_cam.K, other_cam.dist,
                    (ref_cam.width, ref_cam.height),
                    flags=cv2.CALIB_FIX_INTRINSIC,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
                )
                # stereoCalibrate returns the transform taking points from the
                # reference camera frame into the other camera frame.
                rig.poses[name] = invert(rt_to_matrix(R, T))
                worst_rms = max(worst_rms, float(rms))
                continue

        # PnP fallback: T_ref_other = T_ref_board @ inv(T_other_board)
        rels: list[np.ndarray] = []
        for v in views:
            T_ref_board = _board_pose(v.ref, ref_cam)
            T_oth_board = _board_pose(v.other, other_cam)
            if T_ref_board is None or T_oth_board is None:
                continue
            rels.append(T_ref_board @ invert(T_oth_board))
        if len(rels) < 3:
            raise ValueError(f"{name}: could not recover board pose in enough views ({len(rels)})")
        rels = _reject_outliers(rels)
        rig.poses[name] = _average_transforms(rels)
        spread = float(np.std([np.linalg.norm(T[:3, 3]) for T in rels]))
        worst_rms = max(worst_rms, spread)

    rig.stereo_rms = worst_rms
    return rig

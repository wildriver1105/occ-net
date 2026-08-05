"""occnet command line."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Dual-camera occupancy network experiment rig.",
)
calib_app = typer.Typer(no_args_is_help=True, help="Camera calibration.")
app.add_typer(calib_app, name="calib")

console = Console()

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to rig.yaml")


def _load_config(path: Optional[Path]):
    from .config import RigConfig

    return RigConfig.load(path)


def _resolve_rig(cfg, require_all: bool = True, only: Optional[dict] = None):
    """Map configured camera names to plugged-in devices, or explain what's missing."""
    from .devices import resolve_rig

    res = resolve_rig(only if only is not None else cfg.cameras)
    if res.missing and require_all:
        console.print(f"[red]Could not find camera(s):[/red] {', '.join(res.missing)}")
        console.print("\n[bold]Devices currently available:[/bold]")
        for d in res.available:
            console.print(f"  {d}")
        console.print(
            "\n[yellow]If the Insta360 GO 3 is missing:[/yellow] put it in the Action Pod, "
            "connect the Pod by USB-C, and pick [bold]Webcam[/bold] on the Pod's USB-mode screen "
            "(needs firmware 1.2.7+). Plain USB/file-transfer mode does not expose a camera."
        )
        raise typer.Exit(1)
    return res


@app.command()
def version() -> None:
    """Print the occnet version."""
    console.print(f"occnet {__version__}")


@app.command()
def devices(
    modes: bool = typer.Option(False, "--modes", "-m", help="Also probe supported capture modes"),
    all_devices: bool = typer.Option(False, "--all", help="Include screens and Desk View"),
) -> None:
    """List AVFoundation capture devices."""
    from .devices import list_devices, probe_modes

    found = list_devices(include_unusable=all_devices)
    if not found:
        console.print("[red]No capture devices found.[/red]")
        raise typer.Exit(1)

    table = Table("idx", "name", "role", title="AVFoundation video devices")
    for d in found:
        table.add_row(str(d.index), d.name, d.role)
    console.print(table)

    if modes:
        for d in found:
            if d.is_screen:
                continue
            console.print(f"\n[bold]{d.name}[/bold]")
            try:
                for m in probe_modes(d.index)[:12]:
                    console.print(f"  {m}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]could not probe: {exc}[/yellow]")


@app.command()
def doctor(config: Optional[Path] = _CONFIG_OPT) -> None:
    """Check that every piece of the rig actually works."""
    import shutil

    import torch

    from .capture import CAMERA_PERMISSION_HINT, CameraRig, CaptureConfig
    from .devices import list_devices

    cfg = _load_config(config)
    ok = True

    console.rule("environment")
    console.print(f"ffmpeg      : {shutil.which('ffmpeg') or '[red]MISSING (brew install ffmpeg)[/red]'}")
    console.print(f"torch       : {torch.__version__}")
    console.print(f"MPS backend : {'available' if torch.backends.mps.is_available() else '[yellow]unavailable[/yellow]'}")

    console.rule("devices")
    found = list_devices()
    for d in found:
        console.print(f"  {d}")
    roles = {d.role for d in found}
    if "insta360" not in roles:
        ok = False
        console.print(
            "\n[yellow]Insta360 GO 3 not present as a camera.[/yellow] Put it in the Action Pod, "
            "connect by USB-C, then choose [bold]Webcam[/bold] on the Pod's USB-mode screen."
        )
    if "iphone" not in roles:
        console.print(
            "\n[yellow]iPhone not present.[/yellow] Unlock it, keep it connected, and make sure "
            "Continuity Camera is enabled (iPhone: Settings > General > AirPlay & Continuity)."
        )

    console.rule("capture")
    probe_cfg = CaptureConfig(
        width=cfg.capture.width, height=cfg.capture.height,
        fps=cfg.capture.fps, backend=cfg.capture.backend, startup_timeout=8.0,
    )
    for d in found:
        rig = CameraRig({d.role: d}, probe_cfg)
        try:
            rig.start()
            bundle = rig.read(timeout=8.0)
            if bundle is None:
                raise RuntimeError("opened but delivered no bundle")
            size = rig.sizes()[d.role]
            console.print(f"  [green]OK[/green]   {d.name}: {size[0]}x{size[1]}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            console.print(f"  [red]FAIL[/red] {d.name}: {exc}")
        finally:
            rig.stop()

    if not ok:
        console.print(f"\n[yellow]{CAMERA_PERMISSION_HINT}[/yellow]")
        raise typer.Exit(1)
    console.print("\n[green]Rig is ready.[/green]")


@app.command()
def init(
    path: Path = typer.Option(Path("configs/rig.yaml"), "--path", "-p"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config"),
) -> None:
    """Write a default rig.yaml."""
    from .config import RigConfig

    if path.exists() and not force:
        console.print(f"[yellow]{path} already exists; pass --force to overwrite.[/yellow]")
        raise typer.Exit(1)
    RigConfig().save(path)
    console.print(f"[green]wrote {path}[/green]")


@app.command()
def preview(
    config: Optional[Path] = _CONFIG_OPT,
    height: int = typer.Option(540, "--height", help="Tile height in the window"),
) -> None:
    """Show every configured camera live, side by side."""
    from .capture import CameraRig
    from .viewer import run_viewer

    cfg = _load_config(config)
    res = _resolve_rig(cfg)
    for name, dev in res.resolved.items():
        console.print(f"  {name} -> {dev}")

    rig = CameraRig(res.resolved, cfg.capture)
    with rig:
        console.print(f"  sizes: {rig.sizes()}")
        console.print("[dim]q quit · s snapshot[/dim]")
        run_viewer(rig, tile_height=height)


@app.command()
def board(
    config: Optional[Path] = _CONFIG_OPT,
    out: Path = typer.Option(Path("out/charuco.png"), "--out", "-o"),
    dpi: int = typer.Option(300, "--dpi"),
) -> None:
    """Render the ChArUco calibration target for printing."""
    from .calib.board import render_board

    cfg = _load_config(config)
    path = render_board(cfg.board, out, dpi=dpi)
    w_m, h_m = cfg.board.size_m
    console.print(f"[green]wrote {path}[/green]")
    console.print(
        f"Print at 100% scale (no 'fit to page'), then measure one square with a ruler.\n"
        f"Expected board size: {w_m * 100:.1f} x {h_m * 100:.1f} cm, "
        f"square {cfg.board.square_m * 1000:.1f} mm.\n"
        f"[yellow]If the printed square differs, update board.square_m in your config — "
        f"every metric distance downstream is scaled by it.[/yellow]"
    )


@calib_app.command("intrinsics")
def calib_intrinsics(
    camera: str = typer.Option(..., "--camera", help="Camera name from the config"),
    config: Optional[Path] = _CONFIG_OPT,
    model: str = typer.Option("rational", "--model", help="pinhole | rational | fisheye"),
    target_views: int = typer.Option(20, "--views", help="Views to collect before solving"),
    auto: bool = typer.Option(True, "--auto/--manual", help="Auto-capture novel board views"),
) -> None:
    """Calibrate one camera's intrinsics from live ChArUco views."""
    import time

    import cv2

    from .calib.board import make_board
    from .calib.intrinsics import CalibrationSet, calibrate_intrinsics, detect_board
    from .capture import CameraRig
    from .viewer import run_viewer

    cfg = _load_config(config)
    if camera not in cfg.cameras:
        console.print(f"[red]{camera!r} is not in the config; have {list(cfg.cameras)}[/red]")
        raise typer.Exit(1)

    res = _resolve_rig(cfg, only={camera: cfg.cameras[camera]})
    device = res.resolved[camera]
    console.print(f"calibrating [bold]{camera}[/bold] -> {device}")

    board_obj, detector = make_board(cfg.board)
    cal = CalibrationSet(name=camera)
    state = {"last": 0.0}

    def on_frame(frames, canvas):
        frame = frames[camera]
        det = detect_board(frame.image, detector, board_obj)
        now = time.monotonic()
        if det is not None:
            h = canvas.shape[0]
            scale = h / frame.image.shape[0]
            for pt in det.img_points:
                cv2.circle(canvas, (int(pt[0] * scale), int(pt[1] * scale)), 3, (0, 255, 0), -1)
            # Space captures out so the operator has time to move the board,
            # and only accept views that reach unseen parts of the frame.
            if auto and now - state["last"] > 0.7 and cal.is_novel(det) and len(cal) < target_views:
                cal.add(det)
                state["last"] = now
        cv2.putText(
            canvas,
            f"views {len(cal)}/{target_views}   coverage {cal.coverage * 100:.0f}%"
            f"   {'board seen' if det is not None else 'no board'}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if det is not None else (0, 165, 255), 2, cv2.LINE_AA,
        )
        return canvas

    def on_key(key, frames):
        if key == ord("c"):  # manual capture
            det = detect_board(frames[camera].image, detector, board_obj)
            if det is None:
                console.print("[yellow]no board in view[/yellow]")
            else:
                cal.add(det)
                console.print(f"captured view {len(cal)} ({det.count} corners)")
            return True
        return False

    rig = CameraRig({camera: device}, cfg.capture)
    with rig:
        console.print(
            "Hold the printed board in view. Cover the whole frame including corners, "
            "and tilt it 20-45 degrees in several directions.\n"
            "[dim]c capture · q finish and solve[/dim]"
        )
        run_viewer(rig, window=f"occnet calib — {camera}", on_frame=on_frame, on_key=on_key)

    console.print(f"collected {len(cal)} views, coverage {cal.coverage * 100:.0f}%")
    if cal.coverage < 0.5:
        console.print("[yellow]Coverage is low; intrinsics near the image edge will be poorly constrained.[/yellow]")
    try:
        cam_model = calibrate_intrinsics(cal, model=model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    out_path = cfg.intrinsics_path(camera)
    cam_model.save(out_path)
    console.print(
        f"[green]{camera}[/green]: rms {cam_model.rms:.3f} px · "
        f"fx {cam_model.fx:.1f} fy {cam_model.fy:.1f} · "
        f"FOV {cam_model.hfov_deg:.1f}x{cam_model.vfov_deg:.1f} deg · "
        f"{cam_model.n_views} views -> {out_path}"
    )
    if cam_model.rms > 1.0:
        console.print("[yellow]RMS above 1 px — recapture with a flatter board and sharper focus.[/yellow]")


@calib_app.command("stereo")
def calib_stereo(
    config: Optional[Path] = _CONFIG_OPT,
    method: str = typer.Option("stereo", "--method", help="stereo | pnp"),
    target_views: int = typer.Option(15, "--views"),
) -> None:
    """Solve where the cameras sit relative to each other."""
    import time

    import cv2

    from .calib.board import make_board
    from .calib.extrinsics import PairView, calibrate_rig
    from .calib.intrinsics import detect_board
    from .capture import CameraRig
    from .geometry import CameraModel
    from .viewer import run_viewer

    cfg = _load_config(config)
    names = list(cfg.cameras)
    if len(names) < 2:
        console.print("[red]Need two cameras in the config.[/red]")
        raise typer.Exit(1)

    cameras: dict[str, CameraModel] = {}
    for name in names:
        path = cfg.intrinsics_path(name)
        if not path.exists():
            console.print(f"[red]Missing intrinsics for {name}: run `occnet calib intrinsics --camera {name}`[/red]")
            raise typer.Exit(1)
        cameras[name] = CameraModel.load(path)

    ref = cfg.reference
    other = next(n for n in names if n != ref)

    res = _resolve_rig(cfg)
    board_obj, detector = make_board(cfg.board)
    views: list[PairView] = []
    state = {"last": 0.0}

    def on_frame(frames, canvas):
        dets = {n: detect_board(f.image, detector, board_obj) for n, f in frames.items()}
        both = dets.get(ref) is not None and dets.get(other) is not None
        now = time.monotonic()
        if both and now - state["last"] > 0.8 and len(views) < target_views:
            views.append(PairView(ref=dets[ref], other=dets[other]))
            state["last"] = now
        cv2.putText(
            canvas,
            f"pairs {len(views)}/{target_views}   "
            + "   ".join(f"{n}:{'ok' if dets[n] is not None else '--'}" for n in frames),
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if both else (0, 165, 255), 2, cv2.LINE_AA,
        )
        return canvas

    rig = CameraRig(res.resolved, cfg.capture)
    with rig:
        console.print(
            f"Reference camera: [bold]{ref}[/bold]. Hold the board where [bold]both[/bold] "
            "cameras see it, at several depths and angles.\n"
            "[yellow]Do not move the cameras relative to each other from now on.[/yellow]\n"
            "[dim]q finish and solve[/dim]"
        )
        run_viewer(rig, window="occnet calib — stereo", on_frame=on_frame)

    console.print(f"collected {len(views)} shared views")
    try:
        rig_calib = calibrate_rig(ref, cameras, {other: views}, method=method)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    rig_calib.save(cfg.rig_path)
    baseline = rig_calib.baseline_m(ref, other)
    console.print(
        f"[green]rig solved[/green]: baseline {baseline * 100:.1f} cm · "
        f"residual {rig_calib.stereo_rms:.3f} -> {cfg.rig_path}"
    )
    console.print("[dim]Sanity-check the baseline against a tape measure before trusting depth.[/dim]")


def _load_rig_calibration(cfg, sizes: dict[str, tuple[int, int]]):
    """Load calibration, falling back to a rough guess so the rig still runs."""
    from .geometry import CameraModel, RigCalibration

    if cfg.rig_path.exists():
        rig = RigCalibration.load(cfg.rig_path)
        console.print(f"loaded rig calibration from {cfg.rig_path}")
        return rig, True

    cameras: dict[str, CameraModel] = {}
    calibrated = True
    for name, (w, h) in sizes.items():
        path = cfg.intrinsics_path(name)
        if path.exists():
            cameras[name] = CameraModel.load(path).scaled(w, h)
        else:
            calibrated = False
            hfov = 100.0 if name == "insta360" else 68.0
            cameras[name] = CameraModel.guess(name, w, h, hfov_deg=hfov)
    rig = RigCalibration(reference=cfg.reference, cameras=cameras)
    if not calibrated:
        console.print(
            "[yellow]Running with guessed intrinsics — geometry will be approximate. "
            "Run `occnet calib intrinsics` for metric results.[/yellow]"
        )
    return rig, calibrated


@app.command()
def live(
    config: Optional[Path] = _CONFIG_OPT,
    mode: str = typer.Option("mono", "--mode", help="mono | stereo | both"),
    stride: int = typer.Option(3, "--stride", help="Pixel stride when lifting depth"),
    show_points: bool = typer.Option(True, "--points/--no-points"),
    mesh_every: int = typer.Option(0, "--mesh-every", help="Extract a mesh every N frames (0 = off)"),
    save_rrd: Optional[Path] = typer.Option(None, "--save", help="Record to an .rrd file instead of spawning the viewer"),
) -> None:
    """Live 3D reconstruction into an occupancy grid, streamed to Rerun."""
    import numpy as np

    from .capture import CameraRig
    from .fusion.grid import OccupancyGrid
    from .pipeline import Reconstructor
    from .viz.rerun_viz import RerunViz, height_colors

    cfg = _load_config(config)
    res = _resolve_rig(cfg)
    rig_cams = CameraRig(res.resolved, cfg.capture)

    with rig_cams:
        sizes = rig_cams.sizes()
        console.print(f"cameras: {sizes}")
        rig_calib, _ = _load_rig_calibration(cfg, sizes)

        grid = OccupancyGrid(cfg.grid)
        console.print(
            f"grid {grid.shape} @ {cfg.grid.voxel_size * 100:.1f} cm "
            f"({grid.n_voxels / 1e6:.2f} M voxels) on {grid.device}"
        )

        try:
            recon = Reconstructor(cfg, rig_calib, mode=mode, grid=grid, depth_stride=stride)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        console.print("loading depth model…")
        recon.warmup()

        viz = RerunViz(save_path=save_rrd)
        viz.log_rig(rig_calib)
        viz.log_bounds(np.array(cfg.grid.bounds_min), np.array(cfg.grid.bounds_max))

        console.print("[dim]Ctrl-C to stop[/dim]")
        n = 0
        try:
            while True:
                frames = rig_cams.read(timeout=5.0)
                if frames is None:
                    console.print("[yellow]camera stalled[/yellow]")
                    break
                viz.tick(t=n / max(cfg.capture.fps, 1.0))
                step = recon.step(frames)

                for name, r in step.per_camera.items():
                    base = name.replace("_stereo", "")
                    if base in rig_calib.cameras:
                        viz.log_image(base, r.rectified)
                        viz.log_depth(base, r.depth, max_depth=cfg.grid.bounds_max[2])
                    if show_points and len(r.points_world):
                        viz.log_points(f"points/{name}", r.points_world, r.colors)

                occ = grid.occupied_points()
                viz.log_occupancy(occ, cfg.grid.voxel_size, height_colors(occ) if len(occ) else None)

                stats = grid.stats()
                viz.log_scalar("occupied_voxels", stats["occupied"])
                viz.log_scalar("explored_pct", stats["explored_pct"])
                viz.log_scalar("step_ms", step.ms)

                if mesh_every and n % mesh_every == 0 and n > 0:
                    viz.log_mesh(grid.extract_mesh())

                if n % 15 == 0:
                    console.print(
                        f"frame {n:5d} · {step.ms:6.1f} ms · {step.total_points:7d} pts · "
                        f"occupied {int(stats['occupied']):7d} · explored {stats['explored_pct']:.1f}%"
                    )
                n += 1
        except KeyboardInterrupt:
            console.print("\nstopping…")

        out_grid = Path("out/grid.npz")
        grid.save(out_grid)
        console.print(f"[green]saved occupancy grid -> {out_grid}[/green] ({grid.stats()['occupied']:.0f} occupied voxels)")


@app.command()
def export(
    grid_path: Path = typer.Option(Path("out/grid.npz"), "--grid", "-g"),
    out: Path = typer.Option(Path("out/scene.ply"), "--out", "-o"),
    threshold: float = typer.Option(0.65, "--threshold"),
    kind: str = typer.Option("mesh", "--kind", help="mesh | points"),
) -> None:
    """Export a saved occupancy grid as a mesh or point cloud."""
    import numpy as np
    import trimesh

    from .fusion.grid import OccupancyGrid

    if not grid_path.exists():
        console.print(f"[red]{grid_path} not found; run `occnet live` first.[/red]")
        raise typer.Exit(1)

    grid = OccupancyGrid.load(grid_path, device="cpu")
    out.parent.mkdir(parents=True, exist_ok=True)

    if kind == "mesh":
        mesh = grid.extract_mesh(threshold=threshold)
        if mesh is None:
            console.print("[red]No surface at that threshold — try a lower --threshold.[/red]")
            raise typer.Exit(1)
        mesh.export(out)
        console.print(f"[green]{len(mesh.vertices)} verts, {len(mesh.faces)} faces -> {out}[/green]")
    else:
        pts = grid.occupied_points(threshold)
        if len(pts) == 0:
            console.print("[red]No occupied voxels at that threshold.[/red]")
            raise typer.Exit(1)
        trimesh.PointCloud(np.asarray(pts)).export(out)
        console.print(f"[green]{len(pts)} points -> {out}[/green]")


if __name__ == "__main__":
    app()

"""occnet command line."""

from __future__ import annotations

import time
from dataclasses import asdict
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


_ALL_OPT = typer.Option(False, "--all", "-a", help="Use every camera found, ignoring the config's list")
_ONLY_OPT = typer.Option(
    None, "--only", help="Comma-separated camera names to keep, e.g. --only iphone"
)


def _filter_names(res, only_names: Optional[str]):
    """Narrow a resolution to a chosen subset of cameras."""
    if not only_names:
        return res
    wanted = [n.strip() for n in only_names.split(",") if n.strip()]
    missing = [n for n in wanted if n not in res.resolved]
    if missing:
        console.print(
            f"[red]--only names not found:[/red] {', '.join(missing)}  "
            f"(available: {', '.join(res.resolved) or 'none'})"
        )
        raise typer.Exit(1)
    res.resolved = {n: res.resolved[n] for n in wanted}
    return res


def _resolve_rig(cfg, require_all: bool = True, only: Optional[dict] = None,
                 discover: bool = False, only_names: Optional[str] = None):
    """Map configured camera names to plugged-in devices, or explain what's missing."""
    from .devices import RigResolution, discover_all, resolve_rig

    if only is None and (discover or getattr(cfg, "discover", False)):
        found = discover_all()
        if not found:
            console.print("[red]No usable cameras found.[/red]")
            raise typer.Exit(1)
        return _filter_names(
            RigResolution(resolved=found, available=list(found.values())), only_names
        )

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
    return _filter_names(res, only_names)


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
def doctor(config: Optional[Path] = _CONFIG_OPT, all_cameras: bool = _ALL_OPT) -> None:
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

    console.rule("all configured cameras together")
    # Opening cameras one at a time hides the failure mode where a second
    # AVFoundation session disturbs the first, which is exactly what a
    # multi-camera rig runs into.
    res = _resolve_rig(cfg, require_all=False, discover=all_cameras)
    if res.missing:
        console.print(f"  [yellow]not resolved: {', '.join(res.missing)} — skipping[/yellow]")
    elif len(res.resolved) < 2:
        console.print("  [dim]only one camera configured; nothing to combine[/dim]")
    else:
        rig = CameraRig(res.resolved, probe_cfg)
        try:
            rig.start()
            console.print(f"  opened together: {rig.sizes()}")
            seen = {n: 0 for n in res.resolved}
            brightness = {n: [] for n in res.resolved}
            deadline = time.monotonic() + 3.0
            last_idx = {n: -1 for n in res.resolved}
            while time.monotonic() < deadline:
                bundle = rig.read(timeout=5.0)
                if bundle is None:
                    break
                for n, f in bundle.items():
                    if f.index != last_idx[n]:
                        last_idx[n] = f.index
                        seen[n] += 1
                        brightness[n].append(float(f.image.mean()))

            table = Table("camera", "frames", "fps", "mean level", "verdict")
            for n in res.resolved:
                levels = brightness[n]
                mean = sum(levels) / len(levels) if levels else 0.0
                spread = (max(levels) - min(levels)) if len(levels) > 1 else 0.0
                if seen[n] == 0:
                    verdict, style = "NO FRAMES", "red"
                    ok = False
                elif mean < 4.0 and spread < 1.0:
                    # Frames arrive but carry no signal: lens covered, privacy
                    # shutter, or the sensor never actually started.
                    verdict, style = "BLACK — check for a cover", "yellow"
                elif seen[n] < 10:
                    verdict, style = "very low rate", "yellow"
                else:
                    verdict, style = "ok", "green"
                table.add_row(
                    n, str(seen[n]), f"{seen[n] / 3.0:.1f}", f"{mean:.1f}",
                    f"[{style}]{verdict}[/{style}]",
                )
            console.print(table)
            console.print(f"  inter-camera skew: {rig.stats.skew_ms:.1f} ms")
        except Exception as exc:  # noqa: BLE001
            ok = False
            console.print(f"  [red]FAIL together[/red]: {str(exc).splitlines()[0]}")
            console.print(
                "  [yellow]If each camera works alone but not together, force the "
                "subprocess backend: set `capture.backend: ffmpeg` in your config.[/yellow]"
            )
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
    all_cameras: bool = _ALL_OPT,
    only: Optional[str] = _ONLY_OPT,
    height: int = typer.Option(540, "--height", help="Tile height in the window"),
) -> None:
    """Show every configured camera live, side by side."""
    from .capture import CameraRig
    from .viewer import run_viewer

    cfg = _load_config(config)
    res = _resolve_rig(cfg, discover=all_cameras, only_names=only)
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


def _effective_reference(cfg, names) -> str:
    """The configured reference if it was actually found, else the first camera."""
    names = list(names)
    if cfg.reference in names:
        return cfg.reference
    console.print(
        f"[yellow]reference {cfg.reference!r} not among the cameras found; "
        f"using {names[0]!r}[/yellow]"
    )
    return names[0]


def _load_rig_calibration(cfg, sizes: dict[str, tuple[int, int]], reference: Optional[str] = None):
    """Load calibration, falling back to a rough guess so the rig still runs."""
    from .geometry import CameraModel, RigCalibration

    reference = reference or cfg.reference
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
    rig = RigCalibration(reference=reference, cameras=cameras)
    if not calibrated:
        console.print(
            "[yellow]Running with guessed intrinsics — geometry will be approximate. "
            "Run `occnet calib intrinsics` for metric results.[/yellow]"
        )
    return rig, calibrated


@app.command()
def live(
    config: Optional[Path] = _CONFIG_OPT,
    all_cameras: bool = _ALL_OPT,
    only: Optional[str] = _ONLY_OPT,
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
    res = _resolve_rig(cfg, discover=all_cameras, only_names=only)
    rig_cams = CameraRig(res.resolved, cfg.capture)

    with rig_cams:
        sizes = rig_cams.sizes()
        console.print(f"cameras: {sizes}")
        rig_calib, _ = _load_rig_calibration(cfg, sizes, _effective_reference(cfg, sizes))

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
def watch(
    config: Optional[Path] = _CONFIG_OPT,
    all_cameras: bool = _ALL_OPT,
    only: Optional[str] = _ONLY_OPT,
    alert_m: Optional[float] = typer.Option(None, "--alert", help="Tint anything closer than this, in metres"),
    stride: int = typer.Option(3, "--stride", help="Pixel stride when lifting depth"),
    carve_stride: int = typer.Option(4, "--carve-stride"),
    row_height: int = typer.Option(300, "--row-height"),
    decay: float = typer.Option(0.92, "--decay", help="Per-frame evidence decay; 1.0 accumulates forever"),
    auto_range: bool = typer.Option(True, "--auto-range/--fixed-range"),
    absolute_depth_colors: bool = typer.Option(
        False, "--absolute-colors",
        help="Colour depth on a fixed 0..volume scale instead of per-frame contrast"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Also record the view to an mp4"),
    window: bool = typer.Option(True, "--window/--no-window"),
) -> None:
    """Live inference on every camera at once, with a shared occupancy map.

    Each camera gets a row of [frame + near-field alert | metric depth], and the
    fused top-down occupancy map sits alongside. This is `live` without the
    Rerun 3D viewer — one window, aimed at watching both cameras infer on the
    same scene.
    """
    import cv2
    import numpy as np

    from .capture import CameraRig
    from .depth.mono import MonoDepth
    from .fusion.grid import GridConfig, OccupancyGrid, fit_bounds
    from .fusion.lift import lift_depth
    from .geometry import transform_points
    from .overlay import caption, render_inference_view

    cfg = _load_config(config)
    res = _resolve_rig(cfg, discover=all_cameras, only_names=only)
    for name, dev in res.resolved.items():
        console.print(f"  {name} -> {dev}")

    rig_cams = CameraRig(res.resolved, cfg.capture)
    with rig_cams:
        sizes = rig_cams.sizes()
        console.print(f"sizes : {sizes}")
        reference = _effective_reference(cfg, sizes)
        rig_calib, calibrated = _load_rig_calibration(cfg, sizes, reference)

        # Without solved extrinsics every camera sits at the identity, so fusing
        # them all would stack unrelated clouds on top of each other and claim
        # they agree. Fuse only the reference camera until `calib stereo` runs.
        has_extrinsics = cfg.rig_path.exists() and any(
            np.any(rig_calib.T_rig_cam(n)[:3, 3]) for n in sizes if n != reference
        )
        fuse_from = list(sizes) if has_extrinsics else [reference]
        bev_note = "" if has_extrinsics else "single-camera (no extrinsics)"
        if not has_extrinsics:
            console.print(
                "[yellow]No rig extrinsics — the occupancy map is built from "
                f"'{reference}' alone. Every depth panel is still live. "
                "Run `occnet calib stereo` to fuse them.[/yellow]"
            )

        depth_model = MonoDepth(cfg.mono)
        console.print(f"model : {depth_model.repo}")
        depth_model.load()

        grid_cfg = GridConfig(**{**asdict(cfg.grid), "carve_stride": carve_stride})
        if auto_range:
            first = rig_cams.read(timeout=8.0)
            if first is None:
                console.print("[red]no frames from the rig[/red]")
                raise typer.Exit(1)
            ref_cam = rig_calib.cameras[reference]
            try:
                lo, hi, voxel = fit_bounds(
                    depth_model(first[reference].image), ref_cam.hfov_deg, ref_cam.vfov_deg
                )
                grid_cfg = GridConfig(**{
                    **asdict(grid_cfg),
                    "bounds_min": lo, "bounds_max": hi, "voxel_size": voxel,
                    "max_ray_m": float(np.linalg.norm(np.array(hi) - np.array(lo))),
                })
                console.print(f"range : auto-fitted to {lo[2]:.1f}-{hi[2]:.1f} m, voxel {voxel * 100:.1f} cm")
            except ValueError as exc:
                console.print(f"[yellow]auto-range failed ({exc}); using the config's grid[/yellow]")

        if alert_m is None:
            alert_m = grid_cfg.bounds_min[2] + 0.25 * (grid_cfg.bounds_max[2] - grid_cfg.bounds_min[2])
            console.print(f"alert : auto-set to {alert_m:.2f} m")

        grid = OccupancyGrid(grid_cfg)
        console.print(f"grid  : {grid.shape} @ {grid_cfg.voxel_size * 100:.1f} cm on {grid.device}")

        if window:
            cv2.namedWindow("occnet — live inference", cv2.WINDOW_NORMAL)
        console.print("[dim]q quit · s snapshot · r reset grid[/dim]")

        writer = None
        n = 0
        ms_ema = 0.0
        try:
            while True:
                frames = rig_cams.read(timeout=5.0)
                if frames is None:
                    console.print("[yellow]camera stalled[/yellow]")
                    break

                t0 = time.perf_counter()
                panels: list[tuple[str, np.ndarray, np.ndarray]] = []
                total_pts = 0
                if decay < 1.0:
                    grid.decay(decay)
                for name in sizes:
                    frame = frames[name]
                    depth = depth_model(frame.image)
                    panels.append((name, frame.image, depth))
                    if name not in fuse_from:
                        continue
                    pts, _ = lift_depth(
                        depth, rig_calib.cameras[name], None, stride=stride,
                        max_depth_m=cfg.mono.max_depth_m, min_depth_m=cfg.mono.min_depth_m,
                    )
                    if len(pts) == 0:
                        continue
                    T = rig_calib.T_rig_cam(name)
                    grid.integrate(transform_points(T, pts).astype(np.float32), T[:3, 3])
                    total_pts += len(pts)

                elapsed = (time.perf_counter() - t0) * 1000
                ms_ema = elapsed if not ms_ema else 0.9 * ms_ema + 0.1 * elapsed

                canvas = render_inference_view(
                    panels, grid.bev(), grid_cfg.voxel_size,
                    grid_cfg.bounds_min, grid_cfg.bounds_max,
                    alert_m=alert_m, row_height=row_height, bev_note=bev_note,
                    auto_scale_depth=not absolute_depth_colors,
                )
                stats = grid.stats()
                caption(canvas, [
                    f"{1000 / ms_ema:5.1f} fps   step {ms_ema:5.1f} ms   skew {rig_cams.stats.skew_ms:4.1f} ms",
                    f"pts {total_pts:6d}   occupied {int(stats['occupied']):6d}   q quit  s snap  r reset",
                ], origin=(10, canvas.shape[0] - 60))

                if out is not None:
                    if writer is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15.0,
                            (canvas.shape[1], canvas.shape[0]),
                        )
                    writer.write(canvas)

                if window:
                    cv2.imshow("occnet — live inference", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        grid.reset()
                        console.print("grid reset")
                    if key == ord("s"):
                        snap = Path("out/snapshots")
                        snap.mkdir(parents=True, exist_ok=True)
                        path = snap / f"watch-{time.strftime('%Y%m%d-%H%M%S')}.png"
                        cv2.imwrite(str(path), canvas)
                        console.print(f"saved {path}")

                if n % 30 == 0:
                    console.print(
                        f"  frame {n:5d} · {elapsed:6.1f} ms · depth {depth_model.last_ms:5.1f} ms · "
                        f"occupied {int(stats['occupied']):6d}"
                    )
                n += 1
        except KeyboardInterrupt:
            console.print("\nstopping…")
        finally:
            if writer is not None:
                writer.release()
            if window:
                cv2.destroyAllWindows()
                for _ in range(5):
                    cv2.waitKey(1)

    if ms_ema:
        console.print(f"[green]{n} frames at {1000 / ms_ema:.1f} fps average[/green]")
    if out is not None:
        console.print(f"[green]wrote {out}[/green]")


@app.command()
def play(
    video: Optional[Path] = typer.Argument(None, help="Video file; omitted renders a synthetic test clip"),
    config: Optional[Path] = _CONFIG_OPT,
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the annotated result to an mp4"),
    window: bool = typer.Option(True, "--window/--no-window", help="Show a live window"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", "-n"),
    width: int = typer.Option(960, "--width", help="Resize input to this width"),
    alert_m: Optional[float] = typer.Option(None, "--alert", help="Tint anything closer than this, in metres"),
    hfov: float = typer.Option(70.0, "--hfov", help="Horizontal FOV assumed for the clip, in degrees"),
    stride: int = typer.Option(3, "--stride", help="Pixel stride when lifting depth"),
    carve_stride: int = typer.Option(4, "--carve-stride", help="Carve free space with every Nth ray"),
    auto_range: bool = typer.Option(True, "--auto-range/--fixed-range",
                                    help="Size the grid from the clip's own depth range"),
    loop: bool = typer.Option(False, "--loop"),
) -> None:
    """Run depth + occupancy inference over a video and show the result.

    This is the offline counterpart to `occnet live`: it needs no cameras and no
    camera permission, so it is the fastest way to see what the model actually
    produces.
    """
    import cv2
    import numpy as np

    from .depth.mono import MonoDepth
    from .fusion.grid import GridConfig, OccupancyGrid, fit_bounds
    from .fusion.lift import lift_depth
    from .geometry import CameraModel
    from .overlay import bev_panel, caption, depth_panel, near_field_mask, stack_panels
    from .video import VideoSource, make_test_video

    cfg = _load_config(config)

    if video is None:
        video = Path("out/testclip.mp4")
        console.print(f"no video given — rendering a synthetic corridor to {video}")
        make_test_video(video)

    src = VideoSource(video, loop=loop, resize_width=width)
    console.print(f"input : {src.info}")

    w, h = src.output_size
    # A video carries no calibration, so we assume a plain pinhole at the given
    # FOV. Depth is therefore only as metric as that assumption.
    cam = CameraModel.guess("video", w, h, hfov_deg=hfov)
    console.print(f"camera: assumed pinhole, hfov {hfov:.0f} deg -> fx {cam.fx:.1f}")

    depth_model = MonoDepth(cfg.mono)
    console.print(f"model : {depth_model.repo}")
    depth_model.load()

    grid_cfg = GridConfig(**{**asdict(cfg.grid), "carve_stride": carve_stride})
    if auto_range:
        # Probe one frame so the volume matches the scene. A grid that does not
        # cover the observed depths drops every point and looks like a bug.
        probe = VideoSource(video, resize_width=width)
        first = next(probe.frames(max_frames=1), None)
        probe.release()
        if first is None:
            console.print("[red]video has no frames[/red]")
            raise typer.Exit(1)
        try:
            lo, hi, voxel = fit_bounds(depth_model(first.image), cam.hfov_deg, cam.vfov_deg)
            grid_cfg = GridConfig(**{
                **asdict(grid_cfg),
                "bounds_min": lo, "bounds_max": hi, "voxel_size": voxel,
                "max_ray_m": float(np.linalg.norm(np.array(hi) - np.array(lo))),
            })
            console.print(
                f"range : auto-fitted to {lo[2]:.1f}-{hi[2]:.1f} m depth, voxel {voxel * 100:.1f} cm"
            )
        except ValueError as exc:
            console.print(f"[yellow]auto-range failed ({exc}); using the config's grid[/yellow]")

    if alert_m is None:
        # Default the alert to the near quarter of the volume rather than a
        # fixed metre value that may be outside the scene entirely.
        alert_m = grid_cfg.bounds_min[2] + 0.25 * (grid_cfg.bounds_max[2] - grid_cfg.bounds_min[2])
        console.print(f"alert : auto-set to {alert_m:.2f} m")

    grid = OccupancyGrid(grid_cfg)
    console.print(
        f"grid  : {grid.shape} @ {grid_cfg.voxel_size * 100:.1f} cm "
        f"({grid.n_voxels / 1e6:.2f} M voxels) on {grid.device}"
    )

    writer = None
    if window:
        cv2.namedWindow("occnet — inference", cv2.WINDOW_NORMAL)

    max_depth = float(grid_cfg.bounds_max[2])
    panel_h = 420
    n = 0
    ms_ema = 0.0
    try:
        for frame in src.frames(max_frames=max_frames):
            t0 = time.perf_counter()
            depth = depth_model(frame.image)
            pts, _ = lift_depth(
                depth, cam, None, stride=stride,
                max_depth_m=cfg.mono.max_depth_m, min_depth_m=cfg.mono.min_depth_m,
            )
            # Each video frame is treated as a fresh observation from the origin.
            # There is no ego-motion estimate here, so evidence is decayed rather
            # than accumulated — otherwise a moving camera smears the grid.
            grid.decay(0.90)
            if len(pts):
                grid.integrate(pts, np.zeros(3))
            elapsed = (time.perf_counter() - t0) * 1000
            ms_ema = elapsed if not ms_ema else 0.9 * ms_ema + 0.1 * elapsed

            rgb = near_field_mask(frame.image, depth, alert_m)
            valid = depth[depth > 0]
            caption(rgb, [
                f"frame {frame.index}",
                f"near-field alert < {alert_m:.1f} m",
            ])
            dpanel = depth_panel(depth, max_depth=max_depth)
            caption(dpanel, [
                "depth  red=near  blue=far",
                f"range {valid.min():.2f}-{valid.max():.2f} m" if valid.size else "no valid depth",
                f"model {depth_model.last_ms:.0f} ms",
            ])
            span = grid_cfg.bounds_max[2] - grid_cfg.bounds_min[2]
            bpanel = bev_panel(
                grid.bev(), grid_cfg.voxel_size,
                (grid_cfg.bounds_min[0], grid_cfg.bounds_min[2]),
                size=(panel_h, panel_h),
                rings_m=tuple(round(span * f, 1) for f in (0.25, 0.5, 0.75)),
            )

            canvas = stack_panels([rgb, dpanel, bpanel], panel_h)
            stats = grid.stats()
            caption(canvas, [
                f"{1000 / ms_ema:5.1f} fps   step {ms_ema:5.1f} ms   pts {len(pts):6d}",
                f"occupied {int(stats['occupied']):6d} voxels   explored {stats['explored_pct']:.1f}%",
            ], origin=(10, canvas.shape[0] - 60))

            if out is not None:
                if writer is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                        min(src.info.fps, 30.0), (canvas.shape[1], canvas.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer for {out}")
                writer.write(canvas)

            if window:
                cv2.imshow("occnet — inference", canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if n % 20 == 0:
                console.print(
                    f"  frame {frame.index:5d} · {elapsed:6.1f} ms · "
                    f"depth {depth_model.last_ms:5.1f} ms · occupied {int(stats['occupied']):6d}"
                )
            n += 1
    except KeyboardInterrupt:
        console.print("\nstopping…")
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if window:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)

    console.print(f"[green]{n} frames at {1000 / ms_ema:.1f} fps average[/green]" if ms_ema else "no frames")
    if out is not None:
        console.print(f"[green]wrote {out}[/green]")


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

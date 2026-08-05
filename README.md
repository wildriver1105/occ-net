# occ-net

A two-camera experiment rig for occupancy networks on macOS: capture from an
Insta360 GO 3 and an iPhone (Continuity Camera), calibrate them into a common
frame, lift depth into 3D, and fuse everything into a log-odds occupancy grid
you can export as a mesh.

Everything runs locally on Apple silicon (MPS).

---

## Current status of your hardware

| Piece | State |
|---|---|
| iPhone (Continuity Camera) | **Detected** as AVFoundation device `[1] iPhone Camera` |
| MacBook built-in camera | **Detected** as `[0] MacBook Pro Camera` |
| Insta360 GO 3 | **Connected over USB but not a camera yet** — see below |
| Camera permission (TCC) | **Not granted** — see below |

### 1. Put the GO 3 into webcam mode

This is *not* a permission problem. The GO 3 is enumerating with USB interface
class `255` (vendor-specific), which is its file-transfer mode. A UVC webcam
reports interface class `14`, and would appear in `occnet devices`.

To fix it: keep the camera **in the Action Pod**, connect the Pod by USB-C, and
on the USB-mode popup on the Pod's touchscreen choose **Webcam** (needs camera
firmware 1.2.7 or newer). Then:

```bash
uv run occnet devices
```

`Insta360 GO 3` should now be listed. In webcam mode the Pod controls resolution
and FOV from the host, so in-camera settings are disabled — that is expected.

### 2. Grant camera permission

macOS blocks capture until the *host application* has camera access. Right now
OpenCV reports `not authorized to capture video`.

Open **System Settings → Privacy & Security → Camera** and enable the app you
run `occnet` from (Terminal, iTerm, VS Code, Claude…). Quit and relaunch that
app afterwards — the permission is only picked up on a fresh process tree.

Then confirm everything at once:

```bash
uv run occnet doctor
```

### 3. If the GO 3 stays stubborn, use the built-in camera

The pipeline does not care which cameras it gets. A ready-made fallback config
pairs the MacBook's built-in camera with the iPhone:

```bash
uv run occnet preview -c configs/rig-builtin.yaml
```

Everything below works identically with `-c configs/rig-builtin.yaml`.

---

## Quick start

```bash
uv sync
uv run occnet doctor
uv run occnet preview
```

`preview` opens both cameras side by side with live FPS and inter-camera skew.
`q` quits, `s` saves a snapshot pair.

## The pipeline

```
cameras ─▶ capture ─▶ undistort ─▶ depth ─▶ lift to 3D ─▶ log-odds fusion ─▶ mesh
          (threaded)  (calib)    (mono/    (point cloud   (voxel grid)     (marching
                                  stereo)   in rig frame)                    cubes)
```

### Step 1 — calibrate

Metric 3D needs real intrinsics. Print the target, then solve each camera:

```bash
uv run occnet board                 # writes out/charuco.png
```

Print it at **100% scale** (no "fit to page"), then **measure one square with a
ruler** and put the measured value in `board.square_m` in your config. Every
metric distance downstream is scaled by that number.

```bash
uv run occnet calib intrinsics --camera insta360
uv run occnet calib intrinsics --camera iphone
```

Hold the board so it reaches **all four corners** of the frame, and tilt it
**20–45° across several axes**. Tilt is not optional: a board held flat-on to
the camera makes focal length and board distance mathematically ambiguous, so
you get a beautiful sub-pixel reprojection error alongside a focal length that
is 30% wrong. The self-test reproduces exactly this failure if you weaken the
tilt spread.

Use `--model rational` (the default) for the GO 3's wide lens, `--model pinhole`
for the iPhone and built-in camera, and `--model fisheye` only for genuinely
fisheye optics.

Then solve where the cameras sit relative to each other:

```bash
uv run occnet calib stereo
```

Mount both cameras rigidly first, and **do not move them relative to each other
afterwards** — the extrinsics go stale silently, and depth quietly becomes
wrong rather than failing. Check the reported baseline against a tape measure.

### Step 2 — reconstruct

```bash
uv run occnet live --mode mono
```

This spawns the Rerun viewer with the camera frusta, both source images, both
depth maps, the fused point cloud, and the occupancy voxels on a shared
timeline. Ctrl-C stops and writes `out/grid.npz`.

Three modes:

| Mode | Needs | Character |
|---|---|---|
| `mono` | intrinsics only | Dense, works with non-overlapping views, scale from a learned prior |
| `stereo` | intrinsics + extrinsics + real overlap | Sparser, but metrically tied to the measured baseline |
| `both` | all of the above | Mono density continuously rescaled to match stereo truth |

`mono` is the one to start with — it runs before you have extrinsics and it
tolerates the two cameras pointing in different directions.

### Step 3 — export

```bash
uv run occnet export --kind mesh   --out out/scene.ply
uv run occnet export --kind points --out out/scene_points.ply
```

---

## Configuration

`configs/rig.yaml` is the single source of truth; `-c` selects a different one.
Notable knobs:

- `cameras` — maps a logical name to a role (`insta360` / `iphone` / `builtin`),
  a device-name substring, or an explicit AVFoundation index.
- `reference` — which camera defines the rig frame (its pose is the identity).
- `grid.bounds_min` / `bounds_max` — the volume in metres, in the reference
  camera's frame. Camera axes are OpenCV style: **+x right, +y down, +z
  forward**, so `z` is distance in front of the camera.
- `grid.voxel_size` — 5 cm is a good default; halving it costs 8× the memory.
- `grid.carve_stride` — carve free space with every Nth ray. The cheapest way to
  buy back frame time; `4` roughly triples fusion throughput.
- `mono.input_height` — depth model input size. Lower is faster and coarser.

## Measured performance

MacBook Pro (Apple silicon, MPS), 640×360 input, `depth_stride=3`:

| Stage | Time |
|---|---|
| Depth Anything V2 Small, per camera | ~16 ms |
| Grid integrate, 100k points, `carve_stride=1` | ~58 ms |
| Grid integrate, 100k points, `carve_stride=4` | ~20 ms |
| Occupancy readout + stats | ~3 ms |
| **Full step, `mono`, 2 cameras** | **~95 ms** |
| **Full step, `stereo`** | **~46 ms** |
| **Full step, `both`** | **~123 ms** |

The first inference and the first fusion call are far slower (~2 s and ~900 ms)
because Metal compiles kernels on first use; `occnet live` warms both up before
the loop.

## Layout

```
src/occnet/
  devices.py      AVFoundation enumeration (ffmpeg-backed), role matching
  capture.py      Threaded multi-camera capture, OpenCV + ffmpeg backends
  viewer.py       Live side-by-side window, reused by the calibration flows
  geometry.py     CameraModel, RigCalibration, transform helpers
  config.py       rig.yaml schema
  pipeline.py     Reconstructor — frames in, occupancy evidence out
  calib/          ChArUco board, intrinsics, rig extrinsics
  depth/          Monocular metric depth (Depth Anything V2), calibrated stereo
  fusion/         Depth lifting, log-odds voxel grid, marching cubes
  viz/            Rerun logging
scripts/
  selftest.py     Synthetic end-to-end validation — no cameras needed
configs/
  rig.yaml            Insta360 GO 3 + iPhone
  rig-builtin.yaml    MacBook built-in + iPhone (fallback)
```

## Self-test

Validates intrinsics recovery, extrinsics recovery, depth lifting, occupancy
fusion, mesh extraction, and config round-tripping against synthetic ground
truth. No hardware required.

```bash
uv run python scripts/selftest.py
```

Current: 23 checks, all passing — focal length recovered to 0.12%, stereo
baseline to 0.01 mm, rotation to 0.004°.

## Conventions

- Distances are **metres** throughout.
- Poses are **camera-to-world**; `T_wc` maps camera coordinates to world.
- Camera axes are **OpenCV**: +x right, +y down, +z forward.
- Depth maps are metric `float32` with **0 marking invalid**.
- The occupancy grid stores **log-odds**, not probabilities — disagreement
  between the cameras shows up as low-confidence voxels rather than as whichever
  camera wrote last.

## Known limits

- The rig assumes the cameras are **static**. There is no SLAM: the grid is
  built in the reference camera's fixed frame, so moving a camera during a run
  smears the reconstruction. Adding pose tracking is the natural next step.
- Monocular metric depth is a **learned prior**, not a measurement. Indoor
  checkpoints are used by default; `mono.model: outdoor-small` exists for
  outdoor scenes. Use `--mode both` when metric accuracy matters.
- `both` mode's scale fit needs genuine overlap between the two views; with no
  overlap it silently keeps the previous fit.
- Continuity Camera can drop out when the iPhone locks or moves out of range.
  `capture.py` surfaces this as an error rather than hanging.

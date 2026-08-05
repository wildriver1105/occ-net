"""Monocular metric depth via Depth Anything V2.

The metric checkpoints predict depth directly in metres, which is what makes a
single wide-FOV camera usable as an occupancy sensor. The relative checkpoints
are also supported but need a scale/shift fit against a metric reference (see
:meth:`MonoDepth.fit_scale`) before their output means anything in a voxel grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

# Metric checkpoints emit metres; relative ones emit inverse-depth up to an
# unknown affine transform.
METRIC_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}
RELATIVE_MODELS = {
    "rel-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "rel-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "rel-large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class MonoDepthConfig:
    model: str = "small"
    device: str = "auto"
    # Inference resolution. Depth Anything is resolution-flexible; smaller is
    # much faster and the voxel grid quantises away most of the detail anyway.
    input_height: int = 392
    fp16: bool = True
    max_depth_m: float = 12.0
    min_depth_m: float = 0.15


class MonoDepth:
    """Wraps a Depth Anything V2 checkpoint behind a numpy-in/numpy-out call."""

    def __init__(self, cfg: MonoDepthConfig | None = None):
        self.cfg = cfg or MonoDepthConfig()
        self.device = pick_device(self.cfg.device)
        self.repo = METRIC_MODELS.get(self.cfg.model) or RELATIVE_MODELS.get(self.cfg.model)
        if self.repo is None:
            self.repo = self.cfg.model  # allow a raw HF repo id
        self.is_metric = self.cfg.model in METRIC_MODELS or "Metric" in self.repo
        self.scale = 1.0
        self.shift = 0.0
        self._model = None
        self._processor = None
        self.last_ms = 0.0

    def load(self) -> "MonoDepth":
        if self._model is not None:
            return self
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.repo)
        model = AutoModelForDepthEstimation.from_pretrained(self.repo)
        # fp16 on MPS roughly halves latency; CPU stays fp32 because half
        # precision there is slower, not faster.
        if self.cfg.fp16 and self.device.type in ("mps", "cuda"):
            model = model.half()
        self._model = model.to(self.device).eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a HxW float32 depth map in metres, matching the input size."""
        import cv2

        self.load()
        h, w = image_bgr.shape[:2]

        scale = self.cfg.input_height / h
        small = cv2.resize(
            image_bgr, (max(14, int(round(w * scale))), self.cfg.input_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.cfg.fp16 and self.device.type in ("mps", "cuda"):
            pixel_values = pixel_values.half()

        pred = self._model(pixel_values=pixel_values).predicted_depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(
            pred.float(), size=(h, w), mode="bicubic", align_corners=False
        )[0, 0]
        depth = pred.cpu().numpy().astype(np.float32)
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if not self.is_metric:
            # Relative models output inverse depth; convert, then apply whatever
            # affine fit the caller established.
            depth = 1.0 / np.clip(depth, 1e-6, None)
        depth = depth * self.scale + self.shift

        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0
        depth[(depth < self.cfg.min_depth_m) | (depth > self.cfg.max_depth_m)] = 0.0
        return depth

    def fit_scale(self, predicted: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
        """Least-squares fit ``predicted -> reference`` on co-valid pixels.

        Use this to pin a relative model to metric scale using calibrated stereo
        depth, or to correct a metric model's bias against a known distance.
        """
        mask = np.isfinite(predicted) & np.isfinite(reference) & (predicted > 0) & (reference > 0)
        if mask.sum() < 100:
            raise ValueError("not enough overlapping valid pixels to fit scale")
        p = predicted[mask].astype(np.float64)
        r = reference[mask].astype(np.float64)
        A = np.stack([p, np.ones_like(p)], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(A, r, rcond=None)
        self.scale, self.shift = float(scale), float(shift)
        return self.scale, self.shift

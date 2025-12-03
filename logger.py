"""Lightweight TensorBoard logger using PyTorch's SummaryWriter.

This module replaces the previous TensorFlow-based logger. It provides a
backwards-compatible API with three methods used by the codebase:

- scalar_summary(tag, value, step)
- image_summary(tag, images, step)
- histo_summary(tag, values, step, bins=1000)

It uses torch.utils.tensorboard.SummaryWriter under the hood so the project
no longer depends on TensorFlow.
"""
from typing import Sequence
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    # Fail fast with a clear message so callers know to install tensorboard
    raise RuntimeError(
        "tensorboard is required for logging. Install it in your environment with: `pip install tensorboard` or `conda install -c conda-forge tensorboard`. Original error: %s" % e
    )


class Logger:
    """Wrapper around torch.utils.tensorboard.SummaryWriter.

    This logger requires the `tensorboard` package to be installed. It mirrors
    the minimal API used elsewhere in the codebase.
    """

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag: str, value: float, step: int) -> None:
        val = float(value) if hasattr(value, '__float__') else value
        self.writer.add_scalar(tag, val, step)

    def image_summary(self, tag: str, images: Sequence, step: int) -> None:
        import torch

        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                arr = img
                if arr.ndim == 2:
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                elif arr.ndim == 3:
                    tensor = torch.from_numpy(arr).permute(2, 0, 1)
                else:
                    tensor = torch.from_numpy(arr)
            else:
                tensor = img

            if getattr(tensor, 'is_floating_point', lambda: False)():
                try:
                    tmin = float(tensor.min())
                    tmax = float(tensor.max())
                    if tmax > tmin:
                        tensor = (tensor - tmin) / (tmax - tmin)
                except Exception:
                    pass

            self.writer.add_image(f"{tag}/{i}", tensor, step)

    def histo_summary(self, tag: str, values, step: int, bins: int = 1000) -> None:
        vals = np.array(values)
        self.writer.add_histogram(tag, vals, step, bins=bins)

    def flush(self) -> None:
        self.writer.flush()
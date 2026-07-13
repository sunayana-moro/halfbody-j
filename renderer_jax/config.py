"""Hyperparameters / dims for the renderer (replaces torch's argparse ``args``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RendererConfig:


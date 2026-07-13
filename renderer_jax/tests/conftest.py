"""Shared test config. Puts repo root on sys.path so both `renderer` (torch oracle)
and `renderer_jax` import, and exposes a seed + config fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tests/ -> renderer_jax/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 0


@pytest.fixture
def cfg():
    from renderer_jax.config import RendererConfig

    return RendererConfig()

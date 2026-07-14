"""Shared test config: put repo root on sys.path so `shared`, `renderer_jax`, and
the torch `renderer` oracle all import. Kept free of any renderer-specific config
so `shared/` stays decoupled from the renderer package."""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> shared/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 0

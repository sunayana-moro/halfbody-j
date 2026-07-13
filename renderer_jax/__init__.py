"""renderer_jax — JAX/Flax NNX port of the PyTorch ``renderer/`` avatar model.

Skeleton stage: signatures only (bodies raise ``NotImplementedError``).
See ``renderer/STRUCTURE.md`` for the source call-graph and the scaffold manifest
for the build plan. Filled in unit-by-unit, each parity-gated against torch.
"""

from renderer_jax.config import RendererConfig, DEFAULT

__all__ = ["RendererConfig", "DEFAULT"]

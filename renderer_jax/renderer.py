"""Top model. torch: models.IdentidyAdaptive / IMTRenderer."""

from __future__ import annotations

from flax import nnx


class IdentidyAdaptive(nnx.Module):



class IMTRenderer(nnx.Module):
    """Full talking-head renderer. torch: models.IMTRenderer."""

    def __init__(self, config, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def app_encode(self, x):
        raise NotImplementedError

    def mot_encode(self, x):
        raise NotImplementedError

    def mot_decode(self, x):
        raise NotImplementedError

    def id_adapt(self, t, id):
        raise NotImplementedError

    def decode(self, A, B, C):
        raise NotImplementedError

    def __call__(self, x_current, x_reference):
        raise NotImplementedError

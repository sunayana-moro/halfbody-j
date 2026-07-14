"""Parity harness — the spine of the parity tests (test/shared/*).

Lives in test/shared/ (not shared/) on purpose: it's the ONLY piece that imports
torch, and shared/ must stay a pure JAX/Flax package importable in a torch-free
environment. The torch dependency is quarantined here in the test tree.

Recipe: build the torch layer from ``renderer/``, port its weights into the flax
module, feed BOTH the same seeded input, assert ``allclose``. numpy is the neutral
bridge; the test is the ``allclose`` *after* weights are ported in.

Imported by the runnable parity scripts alongside it (e.g. extended versions of
test/shared/layers.py / attention.py). Still stubs below — filling them in needs
the torch + jax + flax venv to run.
"""

from __future__ import annotations


def to_numpy(x):
    """torch.Tensor | jax.Array -> np.ndarray (the neutral comparison format)."""
    raise NotImplementedError


def random_input(shape, seed=0):
    """One seeded np.ndarray, fed to BOTH the torch and flax sides."""
    raise NotImplementedError


def port_weights(torch_module, flax_module):
    """Copy a torch state_dict into flax nnx params.

    Layout conventions: conv weight (O,I,H,W) -> flax; linear (O,I) -> (I,O).
    Returns ``flax_module`` with its params replaced.
    """
    raise NotImplementedError


def assert_parity(torch_module, flax_module, *inputs, atol=1e-5, rtol=1e-4):
    """Run torch (``.eval()``) and flax on identical inputs; assert allclose."""
    raise NotImplementedError

"""Dtype-contract pins for the D-18 audit.

These tests pin the *precision contract* of two leaf routines that were found by
the Phase 4 sweep for hardcoded float widths:

* ``dftorch._nearestneighborlist._min_image_sort_key`` — the composite key whose
  injectivity over ``(i, j)`` the min-image deduplication depends on.
* ``dftorch.ewald_pme.PME_torch.calculate_PME_kspace_stress`` — the reciprocal
  space stress whose G=0 mask must match the dtype of its contraction operands.

Read the ``test_pme_kspace_stress_returns_box_dtype`` docstring before treating
that test as a regression reproducer. It is not one.
"""

import os

# Disable TorchDynamo/Inductor compilation (keeps these tests deterministic and
# avoids requiring a C++ toolchain), matching tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import numpy as np
import torch

from test_f_orbital_skf import run_with_float64


# --------------------------------------------------------------------------- #
# _min_image_sort_key
# --------------------------------------------------------------------------- #


def test_min_image_sort_key_separates_distinct_pairs_at_scale():
    """Distinct (i, j) pairs must stay distinguishable at realistic system sizes.

    The min-image dedup at ``_nearestneighborlist`` sorts by this key and then
    keeps the first entry of every run of equal ``(i, j)``. If two *different*
    ``(i, j)`` pairs collide in the key, the dedup silently drops a real
    neighbor and keeps the wrong periodic image.

    With ``N = 6000`` and ``d2_max = 101.0`` the composite reaches ~3.64e9,
    where the 32-bit spacing is 256 — larger than the entire ``d2_max`` band
    that separates consecutive ``j`` values. Accumulating in 32 bits therefore
    maps ``j = 1000`` and ``j = 1001`` onto the same key.
    """

    def _run():
        from dftorch._nearestneighborlist import _min_image_sort_key

        n_atoms = 6000
        d2_max = 101.0
        ri = torch.tensor([5999, 5999], dtype=torch.long)
        j_all = torch.tensor([1000, 1001], dtype=torch.long)
        d2_all = torch.tensor([0.0, 0.0], dtype=torch.float64)

        key = _min_image_sort_key(ri, j_all, d2_all, n_atoms, d2_max)

        assert key[0] != key[1], (
            "sort key collided for distinct (i, j) pairs: "
            f"{key[0].item()!r} == {key[1].item()!r} — the min-image dedup "
            "would drop a real neighbor at this system size"
        )

    run_with_float64(_run)


def test_min_image_sort_key_orders_by_i_then_j_then_distance():
    """A stable argsort on the key must give lexicographic (i, j, d2) order."""

    def _run():
        from dftorch._nearestneighborlist import _min_image_sort_key

        n_atoms = 8
        # Two i values, two j values per i, three distances per (i, j).
        expected = [
            (2, 3, 0.5),
            (2, 3, 1.5),
            (2, 3, 2.5),
            (2, 6, 0.25),
            (2, 6, 1.25),
            (2, 6, 3.75),
            (5, 1, 0.75),
            (5, 1, 2.25),
            (5, 1, 4.5),
            (5, 7, 0.125),
            (5, 7, 1.0),
            (5, 7, 6.0),
        ]
        # Feed them in a scrambled order so a passing test cannot be an artifact
        # of the input already being sorted.
        perm = [7, 0, 11, 3, 9, 1, 6, 4, 10, 2, 8, 5]
        scrambled = [expected[p] for p in perm]

        ri = torch.tensor([e[0] for e in scrambled], dtype=torch.long)
        j_all = torch.tensor([e[1] for e in scrambled], dtype=torch.long)
        d2_all = torch.tensor([e[2] for e in scrambled], dtype=torch.float64)
        d2_max = float(d2_all.max().item()) + 1.0

        key = _min_image_sort_key(ri, j_all, d2_all, n_atoms, d2_max)
        order = torch.argsort(key, stable=True)

        got = [
            (int(ri[o]), int(j_all[o]), float(d2_all[o])) for o in order.tolist()
        ]
        assert got == expected

    run_with_float64(_run)


def test_min_image_sort_key_dtype_follows_distances():
    """The key's dtype is the distances' dtype, never a fixed width."""

    def _run():
        from dftorch._nearestneighborlist import _min_image_sort_key

        ri = torch.tensor([0, 1, 2], dtype=torch.long)
        j_all = torch.tensor([1, 2, 3], dtype=torch.long)
        d2_all = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        key = _min_image_sort_key(ri, j_all, d2_all, 4, 1.0)

        assert key.dtype == d2_all.dtype
        assert key.shape == d2_all.shape

    run_with_float64(_run)


# --------------------------------------------------------------------------- #
# calculate_PME_kspace_stress
# --------------------------------------------------------------------------- #


def test_pme_kspace_stress_returns_box_dtype():
    """Contract pin, **not** a regression reproducer.

    This test was measured at plan time to pass *even with the pre-fix code*
    (``g_mask = (m_2 > 0).float()``) at both 12 Å and 25 Å boxes, under grad and
    no-grad: ``torch.einsum``'s contraction order for these operand shapes
    happens to promote instead of dispatching to a strict kernel. Do **not**
    conclude from this test passing that the float32/float64 defect is fixed.

    The authoritative red→green gate for that defect is
    ``uv run pytest tests/test_scf.py -q``.

    What this test does pin is the outward contract: the k-space stress is a
    finite (3, 3) tensor in the dtype of the box it was given.
    """

    def _run():
        from dftorch.ewald_pme.PME_torch import (
            calculate_PME_kspace_stress,
            init_PME_data,
        )
        from dftorch.ewald_pme.util import calculate_alpha_and_num_grids

        L = 12.0
        box = torch.eye(3, dtype=torch.float64) * L
        positions = torch.tensor(
            [[1.0, 5.0], [1.0, 5.5], [1.0, 6.0]], dtype=torch.float64
        )  # (3, N)
        charges = torch.tensor([0.4, -0.4], dtype=torch.float64)

        alpha, grid_dimensions = calculate_alpha_and_num_grids(
            np.asarray(box.numpy()), 8.0, 1e-5
        )
        pme_data = init_PME_data(grid_dimensions, box, alpha, 4)

        sigma = calculate_PME_kspace_stress(positions, charges, box, alpha, pme_data)

        assert sigma.shape == (3, 3)
        assert torch.isfinite(sigma).all()
        assert sigma.dtype == box.dtype

    run_with_float64(_run)

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

import torch

from ._bond_integral import _CHANNELS as _BOND_INTEGRAL_CHANNELS

# ---------------------------------------------------------------------------
# Canonical Slater-Koster channel addressing
# ---------------------------------------------------------------------------
# ``coeffs_tensor`` is indexed as [pair_type, interval, channel, 0..3] where the
# channel axis follows the official 40-entry extended-SKF order defined once in
# ``_bond_integral._CHANNELS``: 20 named Hamiltonian channels followed by the 20
# matching overlap channels.  The mapping below is derived directly from that
# list so the two orders can never drift apart.
#
# Historically this module addressed channels as ``channel + SH_shift * 10``,
# which only ever matched the legacy 20-channel simple-format layout.  Under the
# 40-channel representation those numeric offsets silently select the *wrong*
# radial integral (for example the s-s Hamiltonian offset 9 lands on ``Hdd2``
# and the s-s overlap offset 19 lands on ``Hss0``).  Always resolve channels by
# name via :func:`sk_channel_index` / :func:`sk_channel_name` instead.
SK_CHANNEL_NAMES: Final[tuple[str, ...]] = tuple(_BOND_INTEGRAL_CHANNELS)

SK_CHANNEL_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(SK_CHANNEL_NAMES)}
)

# Base (shell-pair) channel names shared by the Hamiltonian and overlap blocks.
# Prefixing with "H" or "S" produces a key of ``SK_CHANNEL_INDEX``.
SK_BASE_CHANNEL_NAMES: Final[tuple[str, ...]] = tuple(
    name[1:] for name in SK_CHANNEL_NAMES if name.startswith("H")
)

# Legacy 10-channel indices still used by the ML Slater-Koster head
# (``_ml_sk._CHANNEL_MAP``).  The ML path is kept on its own numbering so its
# public behavior is unchanged; only the spline path moves to named channels.
_ML_LEGACY_CHANNEL_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {
        "dd0": 0,
        "dd1": 1,
        "dd2": 2,
        "pd0": 3,
        "pd1": 4,
        "pp0": 5,
        "pp1": 6,
        "sd0": 7,
        "sp0": 8,
        "ss0": 9,
    }
)


def sk_channel_name(base: str, SH_shift: int) -> str:
    """Return the canonical channel name for ``base`` in the H (0) or S (1) block.

    Parameters
    ----------
    base : str
        Shell-pair channel without its H/S prefix, e.g. ``"ss0"``, ``"pd1"``,
        ``"ff3"``.
    SH_shift : int
        ``0`` selects the Hamiltonian block, ``1`` the overlap block.  This
        mirrors the historical ``SH_shift`` flag threaded through the SK
        routines, but it now selects a *name prefix* rather than a numeric
        offset.
    """
    if SH_shift == 0:
        return f"H{base}"
    if SH_shift == 1:
        return f"S{base}"
    raise ValueError(
        f"SH_shift must be 0 (Hamiltonian) or 1 (overlap), got {SH_shift!r}"
    )


def sk_channel_index(name: str) -> int:
    """Return the ``coeffs_tensor`` channel index for a canonical channel name."""
    try:
        return SK_CHANNEL_INDEX[name]
    except KeyError:
        raise KeyError(
            f"Unknown Slater-Koster channel {name!r}. Valid channels are: "
            f"{', '.join(SK_CHANNEL_NAMES)}"
        ) from None


# ---------------------------------------------------------------------------
# f-orbital angular convention (pending external source lock)
# ---------------------------------------------------------------------------
# The local AO order is fixed by ``Structure.AO_LABEL_TEMPLATE`` and must not be
# reordered; f orbitals occupy local offsets 9..15.
STRUCTURE_F_AO_ORDER: Final[tuple[str, ...]] = (
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
)

# ---------------------------------------------------------------------------
# Source lock (Phase 3 checkpoint 03-01-02, human-approved)
# ---------------------------------------------------------------------------
# Every f angular formula in this module is transcribed from:
#
#   K. Takegahara, Y. Aoki and A. Yanase,
#   "Slater-Koster tables for f electrons",
#   J. Phys. C: Solid State Phys. 13 (1980) 583-588,
#   DOI 10.1088/0022-3719/13/4/016.
#
# Table 1 (p585) fixes the cubic-harmonic basis, Table 2 (pp586-587) gives the
# s-f, p-f, d-f and f-f entries, equation (13) (p586) fixes the direction
# cosines, and equations (14)-(15) (p586) supply the orthogonality relation used
# as the correctness gate.  See
# ``.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md``
# for the full provenance record, including the rejection of Sharma,
# Phys. Rev. B 19, 2813 (1979) as a primary source and the known misprints in
# Lendi (1974) and Sharma (1979) that must not be copied.
F_FORMULA_SOURCE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "title": "Slater-Koster tables for f electrons",
        "authors": "K. Takegahara, Y. Aoki, A. Yanase",
        "journal": "J. Phys. C: Solid State Phys. 13 (1980) 583-588",
        "doi": "10.1088/0022-3719/13/4/016",
        "tables": "Table 1 (p585), Table 2 (pp586-587)",
        "direction_cosines": "equation (13), p586",
        "orthogonality_check": "equations (14)-(15), p586",
        "record": (
            ".planning/phases/03-h0-s-routing-and-f-angular-blocks/"
            "03-SOURCE-LOCK.md"
        ),
    }
)

#: Table 2 row/column order for the f shell, as printed in the paper.  ``xyz``
#: (the A_2u cubic harmonic) is listed first, then the three T_1u harmonics,
#: then the three T_2u harmonics.  The Cartesian forms are exactly those of
#: ``Structure.AO_LABEL_TEMPLATE`` offsets 9..15, so the adapter below is a pure
#: reordering rather than a change of basis.
PAPER_F_AO_ORDER: Final[tuple[str, ...]] = (
    "fxyz",  # A_2u        xyz
    "fx3",  # T_1u alpha  x(5x^2 - 3r^2)
    "fy3",  # T_1u beta   y(5y^2 - 3r^2)
    "fz3",  # T_1u gamma  z(5z^2 - 3r^2)
    "fx_y2_z2",  # T_2u xi     x(y^2 - z^2)
    "fy_z2_x2",  # T_2u eta    y(z^2 - x^2)
    "fz_x2_y2",  # T_2u zeta   z(x^2 - y^2)
)

#: ``PAPER_TO_STRUCTURE_F_PERMUTATION[i]`` is the *paper* index of the orbital
#: that sits at ``STRUCTURE_F_AO_ORDER[i]``.  Structure.py lists ``fxyz`` last,
#: the paper lists it first; everything else keeps its relative order.
PAPER_TO_STRUCTURE_F_PERMUTATION: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 0)

#: Per-orbital sign applied while permuting.  The paper's cubic harmonics are
#: defined with the same Cartesian polynomials and the same positive
#: normalisation constants as ``Structure.AO_LABEL_TEMPLATE``, so no orbital
#: needs a sign flip.  This vector exists so that a future source with a
#: different phase convention can be adapted without touching the formulas.
PAPER_TO_STRUCTURE_F_SIGN: Final[tuple[float, ...]] = (1.0,) * 7

#: ``True`` only once source-locked s-f/p-f/d-f/f-f angular formulas exist.
#: The source lock (checkpoint 03-01-02) is approved and recorded above; this
#: flag additionally requires the transcribed formulas to be implemented, which
#: happens further down in this module.
F_ANGULAR_FORMULAS_AVAILABLE: bool = False


class FAngularFormulaSourceError(NotImplementedError):
    """Raised when an f-containing SK block is requested before the source lock.

    Phase 3 deliberately routes 16-orbital atom pairs into Slater-Koster
    assembly *before* the f angular formulas exist, so that f pairs fail loudly
    instead of being silently dropped by the 1/4/9-orbital pair masks.  The
    formulas themselves may only be hard-coded from the verified f-electron
    Slater-Koster tables.
    """


_F_FORMULA_SOURCE_MESSAGE: Final[str] = (
    "f-orbital Slater-Koster angular formulas are not implemented yet.\n"
    "This pair reached the explicit f boundary instead of being silently "
    "dropped, which means 16-orbital routing is working.\n"
    "To proceed, the f angular tables must be source-locked first: supply the "
    "f-electron Slater-Koster paper (PDF or extracted table pages) covering the "
    "cubic harmonic definitions, the paper's f AO row/column order, the sign "
    "convention, and all sf/pf/df/ff entries. The paper order is then mapped "
    "onto STRUCTURE_F_AO_ORDER via PAPER_TO_STRUCTURE_F_PERMUTATION and "
    "PAPER_TO_STRUCTURE_F_SIGN.\n"
    "Guessing or reconstructing these tables from memory is not acceptable: it "
    "yields finite, symmetric, and completely wrong physics."
)

_F_PAIR_MASK_LABELS: Final[tuple[str, ...]] = (
    "HZ",
    "ZH",
    "XZ",
    "ZX",
    "YZ",
    "ZY",
    "ZZ",
)


def _require_f_formula_source(*masks: torch.Tensor | None, context: str) -> None:
    """Raise :class:`FAngularFormulaSourceError` if any f pair mask is non-empty.

    ``masks`` are the 16-orbital pair masks in ``_F_PAIR_MASK_LABELS`` order.
    ``None`` entries mean the caller predates f routing and is treated as
    "no f pairs".
    """
    if F_ANGULAR_FORMULAS_AVAILABLE:
        return

    active = [
        label
        for label, mask in zip(_F_PAIR_MASK_LABELS, masks)
        if mask is not None and bool(mask.any())
    ]
    if not active:
        return

    raise FAngularFormulaSourceError(
        f"{context}: reached {len(active)} f-containing pair class(es) "
        f"({', '.join(active)}), where Z denotes an atom with n_orb == 16.\n"
        f"{_F_FORMULA_SOURCE_MESSAGE}"
    )


class FDerivativeUnsupportedError(NotImplementedError):
    """Raised when a path would consume f-orbital H0/S *derivatives*.

    Phase 3 implements the f angular values (Takegahara Table 2) but not their
    Cartesian derivatives.  ``dH0``/``dS`` therefore carry exact zeros in every
    f-containing block.  Zero is a legal-looking derivative, so any consumer
    (forces, stress, MD, geometry optimisation) must fail loudly rather than
    integrate a silently wrong gradient.
    """


F_DERIVATIVE_UNSUPPORTED_MESSAGE: Final[str] = (
    "f-orbital Slater-Koster angular derivatives are not implemented.\n"
    "Phase 3 provides source-locked f angular *values* only, so dH0/dS are "
    "exactly zero inside every f block. That is indistinguishable from a real "
    "vanishing gradient, which is why this path refuses to run instead of "
    "returning an f-incomplete result.\n"
    "Use energies/H0/S for f-containing systems; forces, stress and MD for "
    "those systems require the deferred f derivative work."
)

#: ``True`` only once f angular *derivatives* are implemented and validated.
F_ANGULAR_DERIVATIVES_AVAILABLE: bool = False


class FSpinPolarizationUnsupportedError(NotImplementedError):
    """Raised when a spin-polarized calculation is requested for an f system.

    Phase 4 supports **closed-shell occupation only** for f-containing systems.
    Eu 4f7 is genuinely open-shell, so a closed-shell number returned for a
    spin-polarized request would be quietly wrong rather than merely
    approximate — the two treatments differ in physics, not in accuracy.
    Refusing is therefore the only honest option.

    There is a second, independent reason the open-shell path could not produce
    a meaningful answer even if it did not fail: ``tests/f_orbital_data/`` ships
    no ``spinw.txt``, so ``const.w`` is ``None`` for every f system in the
    fixture set and the spin coupling constants simply do not exist.
    """


F_SPIN_POLARIZATION_UNSUPPORTED_MESSAGE: Final[str] = (
    "Spin-polarized (open-shell) calculations are not supported for f-orbital "
    "systems.\n"
    "Doing this properly requires separate alpha/beta density matrices carried "
    "through the whole SCF path plus shell-resolved spin W coupling extended to "
    "the f shell; neither exists yet, and neither can be faked by scaling a "
    "closed-shell result. This work is deferred to its own phase.\n"
    "Workaround: run the system closed-shell by omitting UNRESTRICTED (or "
    "setting it False). That is a genuine approximation for an open-shell 4f "
    "ion and should be reported as such, but it is a defined calculation "
    "rather than a wrong one.\n"
    "Provenance: deferred by phase 4 decision D-12 and tracked as requirement "
    "SPN-01; see docs/F-SUPPORT-STATUS.md for the full f-orbital support matrix."
)


class FShellResolvedCoulombUnsupportedError(NotImplementedError):
    """Raised when a shell-resolved Coulomb matrix is requested for an f system.

    ``_coulomb_matrix.ewald_real_space_vectorized_sr`` assembles its
    ``(n_shells, n_shells)`` matrix from pair masks that test ``max_ang``
    against 1, 2 and 3 only. An f element has ``max_ang == 4``, so every one of
    its non-s shell rows and columns comes back exactly zero while the matrix
    itself stays finite and correctly shaped — the Phase 3 silent-drop failure
    mode reproduced in the electrostatics.

    This class lives in ``_slater_koster_pair`` rather than in
    ``_coulomb_matrix`` so that the whole f-unsupported exception taxonomy
    (:class:`FAngularFormulaSourceError`, :class:`FDerivativeUnsupportedError`,
    :class:`FSpinPolarizationUnsupportedError`) stays in one place and can be
    reviewed as a single support policy.
    """


F_SHELL_RESOLVED_COULOMB_UNSUPPORTED_MESSAGE: Final[str] = (
    "The shell-resolved Coulomb matrix implements shell pair blocks for s, p "
    "and d only.\n"
    "The seven f-containing blocks (s-f, f-s, p-f, f-p, d-f, f-d, f-f) are not "
    "implemented, so an f-containing system would receive a matrix in which "
    "every non-s shell row and column of the f atom is exactly zero. That is a "
    "finite, correctly shaped, completely wrong matrix, which is why this path "
    "refuses to run rather than returning it.\n"
    "Workaround: leave MAGNETIC_HUBBARD_LDEP unset (or False) so the per-atom "
    "Coulomb path is used. That path is fully implemented for f systems and is "
    "what the single-shot energy consumes.\n"
    "Implementing the missing blocks is blocked on shell-resolved charges being "
    "threaded through the SCF loop: the shell-resolved matrix is "
    "(n_shells, n_shells) while energy() and SCFx consume (Nats, Nats) together "
    "with per-atom charges, so there is currently no consumer that could "
    "validate f values even if they were written."
)


# ---------------------------------------------------------------------------
# f angular blocks -- transcription of Takegahara Table 2
# ---------------------------------------------------------------------------
# Everything below is a literal transcription of Table 2 of
# ``F_FORMULA_SOURCE`` plus the completeness rule stated on p586:
#
#   "The entries not given in the table can be found by cyclically permuting
#    the coordinates and direction cosines."
#
# The cyclic operator is the proper rotation x -> y -> z -> x.  A vector with
# direction cosines (l, m, n) maps to (n, l, m) under it, so for orbitals A, B
# and their images sA, sB the printed entry generates
#
#       E_{sA,sB}(l, m, n) = E_{A,B}(m, n, l).
#
# Formulas are held in the *paper's* row/column order and only converted to
# ``STRUCTURE_F_AO_ORDER`` at the very end, through
# ``PAPER_TO_STRUCTURE_F_PERMUTATION`` / ``PAPER_TO_STRUCTURE_F_SIGN``.
#
# Correctness gate (equations (14)-(15), p586): substituting 1 for every
# two-centre integral of a shell pair must return the identity.  Operationally
# this makes each channel coefficient matrix an orthogonal projector, and it
# links the s-f/p-f/d-f tables to the f-f table.  ``tests/test_f_orbital_skf.py``
# runs that gate over random unit vectors; do not weaken it to accommodate a
# formula, fix the formula instead.
#
# The paper's d order used here is (xy, yz, zx, x^2-y^2, 3z^2-r^2) and the p
# order is (x, y, z); both already coincide with ``Structure.AO_LABEL_TEMPLATE``,
# so only the f axes are permuted.

_SQRT2: Final[float] = 2.0**0.5
_SQRT3: Final[float] = 3.0**0.5
_SQRT5: Final[float] = 5.0**0.5
_SQRT15: Final[float] = 15.0**0.5
_SQRT45: Final[float] = 45.0**0.5
_SQRT_3_8: Final[float] = (3.0 / 8.0) ** 0.5
_SQRT_3_2: Final[float] = (3.0 / 2.0) ** 0.5
_SQRT_5_2: Final[float] = (5.0 / 2.0) ** 0.5
_SQRT_5_8: Final[float] = (5.0 / 8.0) ** 0.5
_SQRT_15_2: Final[float] = (15.0 / 2.0) ** 0.5
_SQRT_15_8: Final[float] = (15.0 / 8.0) ** 0.5

#: Image of each *paper* f index under x -> y -> z -> x.  ``xyz`` (A_2u) is
#: invariant; the T_1u and T_2u triplets each cycle among themselves.
_PAPER_F_CYCLE: Final[tuple[int, ...]] = (0, 2, 3, 1, 5, 6, 4)
#: Image of each p index (x, y, z).
_PAPER_P_CYCLE: Final[tuple[int, ...]] = (1, 2, 0)
#: Image of the three T_2g d indices (xy, yz, zx).  The two E_g orbitals are not
#: closed under the cycle, which is exactly why the paper prints their rows in
#: full instead of leaving them to be generated.
_PAPER_D_T2G_CYCLE: Final[tuple[int, ...]] = (1, 2, 0)


def _cycle_index(cycle: tuple[int, ...], index: int, times: int) -> int:
    for _ in range(times):
        index = cycle[index]
    return index


def _cycle_dirs(times: int, L, M, N):
    """(l, m, n) -> (m, n, l), applied ``times`` times."""
    for _ in range(times):
        L, M, N = M, N, L
    return L, M, N


def _assemble_block(
    cells: dict, n_chan: int, n_row: int, n_col: int, like: torch.Tensor
) -> torch.Tensor:
    """Stack ``{(row, col): [per-channel tensor]}`` into ``(chan, row, col, P)``."""
    missing = [
        (a, b) for a in range(n_row) for b in range(n_col) if (a, b) not in cells
    ]
    if missing:
        raise AssertionError(
            f"f angular block is incomplete; the cyclic-permutation rule did "
            f"not cover {missing}"
        )
    return torch.stack(
        [
            torch.stack(
                [
                    torch.stack([cells[(a, b)][k] for b in range(n_col)])
                    for a in range(n_row)
                ]
            )
            for k in range(n_chan)
        ]
    )


# --------------------------------------------------------------- s-f (p586)
def _sf_paper_row(L, M, N) -> dict:
    """Printed s-f entries: columns xyz, x(5x^2-3r^2), x(y^2-z^2)."""
    return {
        (0, 0): [_SQRT15 * L * M * N],
        (0, 1): [0.5 * L * (5 * L * L - 3)],
        (0, 4): [0.5 * _SQRT15 * L * (M * M - N * N)],
    }


def _sf_paper_block(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _sf_paper_row(*args).items():
            cells[(a, _cycle_index(_PAPER_F_CYCLE, b, t))] = value
    return _assemble_block(cells, 1, 1, 7, L)


# --------------------------------------------------------------- p-f (p586)
def _pf_paper_row(L, M, N) -> dict:
    """Printed p-f row E_{x, .} for all seven f columns."""
    l2, m2, n2 = L * L, M * M, N * N
    return {
        (0, 0): [
            _SQRT15 * l2 * M * N,
            -_SQRT_5_2 * (3 * l2 - 1) * M * N,
        ],
        (0, 1): [
            0.5 * l2 * (5 * l2 - 3),
            -_SQRT_3_8 * (5 * l2 - 1) * (l2 - 1),
        ],
        (0, 2): [
            0.5 * L * M * (5 * m2 - 3),
            -_SQRT_3_8 * L * M * (5 * m2 - 1),
        ],
        (0, 3): [
            0.5 * L * N * (5 * n2 - 3),
            -_SQRT_3_8 * L * N * (5 * n2 - 1),
        ],
        (0, 4): [
            0.5 * _SQRT15 * l2 * (m2 - n2),
            -_SQRT_5_8 * (3 * l2 - 1) * (m2 - n2),
        ],
        (0, 5): [
            0.5 * _SQRT15 * L * M * (n2 - l2),
            -_SQRT_5_8 * L * M * (3 * (n2 - l2) + 2),
        ],
        (0, 6): [
            0.5 * _SQRT15 * L * N * (l2 - m2),
            -_SQRT_5_8 * L * N * (3 * (l2 - m2) - 2),
        ],
    }


def _pf_paper_block(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _pf_paper_row(*args).items():
            cells[
                (
                    _cycle_index(_PAPER_P_CYCLE, a, t),
                    _cycle_index(_PAPER_F_CYCLE, b, t),
                )
            ] = value
    return _assemble_block(cells, 2, 3, 7, L)


# ---------------------------------------------------------- d-f (p586, p587)
def _df_paper_row_xy(L, M, N) -> dict:
    """Printed d-f row E_{xy, .}; the yz and zx rows follow by cyclic rotation."""
    l2, m2, n2 = L * L, M * M, N * N
    return {
        (0, 0): [
            _SQRT45 * l2 * m2 * N,
            -_SQRT_5_2 * N * (6 * l2 * m2 + n2 - 1),
            N * (3 * l2 * m2 + 2 * n2 - 1),
        ],
        (0, 1): [
            0.5 * _SQRT3 * l2 * M * (5 * l2 - 3),
            -_SQRT_3_8 * M * (5 * l2 - 1) * (2 * l2 - 1),
            0.5 * _SQRT15 * l2 * M * (l2 - 1),
        ],
        (0, 2): [
            0.5 * _SQRT3 * L * m2 * (5 * m2 - 3),
            -_SQRT_3_8 * L * (5 * m2 - 1) * (2 * m2 - 1),
            0.5 * _SQRT15 * L * m2 * (m2 - 1),
        ],
        (0, 3): [
            0.5 * _SQRT3 * L * M * N * (5 * n2 - 3),
            -_SQRT_3_2 * L * M * N * (5 * n2 - 1),
            0.5 * _SQRT15 * L * M * N * (n2 + 1),
        ],
        (0, 4): [
            1.5 * _SQRT5 * l2 * M * (m2 - n2),
            -_SQRT_5_8 * M * ((6 * l2 - 1) * (m2 - n2) - 2 * l2),
            0.5 * M * (3 * l2 * (m2 - n2) + 4 * n2 - 2 * l2),
        ],
        (0, 5): [
            1.5 * _SQRT5 * L * m2 * (n2 - l2),
            -_SQRT_5_8 * L * ((6 * m2 - 1) * (n2 - l2) + 2 * m2),
            0.5 * L * (3 * m2 * (n2 - l2) - 4 * n2 + 2 * m2),
        ],
        (0, 6): [
            1.5 * _SQRT5 * L * M * N * (l2 - m2),
            -3.0 * _SQRT_5_2 * L * M * N * (l2 - m2),
            1.5 * L * M * N * (l2 - m2),
        ],
    }


def _df_paper_row_x2y2(L, M, N) -> dict:
    """Printed d-f row E_{x^2-y^2, .} (E_g; not generated by the cyclic rule)."""
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    return {
        (3, 0): [
            1.5 * _SQRT5 * L * M * N * lm,
            -3.0 * _SQRT_5_2 * L * M * N * lm,
            1.5 * L * M * N * lm,
        ],
        (3, 1): [
            0.25 * _SQRT3 * L * lm * (5 * l2 - 3),
            -_SQRT_3_8 * L * (lm - 1) * (5 * l2 - 1),
            -0.25 * _SQRT15 * L * (lm * (1 - l2) - 2 * n2),
        ],
        (3, 2): [
            0.25 * _SQRT3 * M * lm * (5 * m2 - 3),
            -_SQRT_3_8 * M * (lm + 1) * (5 * m2 - 1),
            -0.25 * _SQRT15 * M * (lm * (1 - m2) + 2 * n2),
        ],
        (3, 3): [
            0.25 * _SQRT3 * N * lm * (5 * n2 - 3),
            -_SQRT_3_8 * N * lm * (5 * n2 - 1),
            0.25 * _SQRT15 * N * (n2 + 1) * lm,
        ],
        (3, 4): [
            0.75 * _SQRT5 * L * lm * (m2 - n2),
            -_SQRT_5_8 * L * (3 * lm * (m2 - n2) - l2 + 1),
            0.25 * L * (3 * lm * (m2 - n2) - 4 * l2 + 2),
        ],
        (3, 5): [
            0.75 * _SQRT5 * M * lm * (n2 - l2),
            -_SQRT_5_8 * M * (3 * lm * (n2 - l2) - m2 + 1),
            0.25 * M * (3 * lm * (n2 - l2) - 4 * m2 + 2),
        ],
        (3, 6): [
            0.75 * _SQRT5 * N * lm * lm,
            -_SQRT_5_8 * N * (3 * lm * lm + 2 * n2 - 2),
            0.25 * N * (3 * lm * lm + 8 * n2 - 4),
        ],
    }


def _df_paper_row_3z2(L, M, N) -> dict:
    """Printed d-f row E_{3z^2-r^2, .} (E_g; not generated by the cyclic rule)."""
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    t = 3 * n2 - 1
    return {
        (4, 0): [
            0.5 * _SQRT15 * L * M * N * t,
            -_SQRT_15_2 * L * M * N * t,
            0.5 * _SQRT3 * L * M * N * t,
        ],
        (4, 1): [
            0.25 * L * t * (5 * l2 - 3),
            -0.75 * _SQRT2 * L * n2 * (5 * l2 - 1),
            0.75 * _SQRT5 * L * (l2 * n2 - m2),
        ],
        (4, 2): [
            0.25 * M * t * (5 * m2 - 3),
            -0.75 * _SQRT2 * M * n2 * (5 * m2 - 1),
            0.75 * _SQRT5 * M * (m2 * n2 - l2),
        ],
        (4, 3): [
            0.25 * N * t * (5 * n2 - 3),
            -0.75 * _SQRT2 * N * (5 * n2 - 1) * (n2 - 1),
            0.75 * _SQRT5 * N * (n2 - 1) * (n2 - 1),
        ],
        (4, 4): [
            0.25 * _SQRT15 * L * (m2 - n2) * t,
            -_SQRT_15_8 * L * n2 * (3 * (m2 - n2) + 2),
            0.25 * _SQRT3 * L * (t * (m2 - n2) - 4 * l2 + 2),
        ],
        (4, 5): [
            0.25 * _SQRT15 * M * (n2 - l2) * t,
            -_SQRT_15_8 * M * n2 * (3 * (n2 - l2) - 2),
            0.25 * _SQRT3 * M * (t * (n2 - l2) + 4 * m2 - 2),
        ],
        (4, 6): [
            0.25 * _SQRT15 * N * t * lm,
            -_SQRT_15_8 * N * t * lm,
            0.25 * _SQRT3 * N * t * lm,
        ],
    }


def _df_paper_block(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _df_paper_row_xy(*args).items():
            cells[
                (
                    _cycle_index(_PAPER_D_T2G_CYCLE, a, t),
                    _cycle_index(_PAPER_F_CYCLE, b, t),
                )
            ] = value
    cells.update(_df_paper_row_x2y2(L, M, N))
    cells.update(_df_paper_row_3z2(L, M, N))
    return _assemble_block(cells, 3, 5, 7, L)


# ---------------------------------------------------------------- f-f (p587)
def _ff_paper_given(L, M, N) -> dict:
    """The twelve printed f-f entries.

    Rotating these three times and closing under transposition (the f-f block
    is even under (l, m, n) -> (-l, -m, -n), hence symmetric) yields all 49.
    """
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    q = l2 * m2 + m2 * n2 + n2 * l2
    return {
        (0, 0): [
            15 * l2 * m2 * n2,
            2.5 * (q - 9 * l2 * m2 * n2),
            1 - 4 * q + 9 * l2 * m2 * n2,
            1.5 * (1 - l2) * (1 - m2) * (1 - n2),
        ],
        (0, 3): [
            0.5 * _SQRT15 * L * M * n2 * (5 * n2 - 3),
            -0.25 * _SQRT15 * L * M * (3 * n2 - 1) * (5 * n2 - 1),
            0.5 * _SQRT15 * L * M * n2 * (3 * n2 - 1),
            0.25 * _SQRT15 * L * M * (1 - n2 * n2),
        ],
        (0, 6): [
            7.5 * L * M * n2 * lm,
            -1.25 * L * M * lm * (9 * n2 - 1),
            0.5 * L * M * lm * (9 * n2 - 4),
            0.75 * L * M * lm * (1 - n2),
        ],
        (3, 1): [
            0.25 * L * N * (5 * l2 - 3) * (5 * n2 - 3),
            -0.375 * L * N * (5 * l2 - 1) * (5 * n2 - 1),
            3.75 * L * N * (l2 * n2 - m2),
            0.625 * L * N * (3 * m2 - l2 * n2),
        ],
        (3, 2): [
            0.25 * M * N * (5 * m2 - 3) * (5 * n2 - 3),
            -0.375 * M * N * (5 * m2 - 1) * (5 * n2 - 1),
            3.75 * M * N * (m2 * n2 - l2),
            0.625 * M * N * (3 * l2 - m2 * n2),
        ],
        (3, 3): [
            0.25 * n2 * (5 * n2 - 3) * (5 * n2 - 3),
            0.375 * (5 * n2 - 1) * (5 * n2 - 1) * (1 - n2),
            3.75 * n2 * (1 - n2) * (1 - n2),
            0.625 * (1 - n2) * (1 - n2) * (1 - n2),
        ],
        (6, 1): [
            0.25 * _SQRT15 * L * N * lm * (5 * l2 - 3),
            0.125 * _SQRT15 * L * N * (2 - 3 * lm) * (5 * l2 - 1),
            0.25 * _SQRT15 * L * N * (3 * (1 + l2) * lm - 8 * l2 + 2),
            0.125 * _SQRT15 * L * N * (-(l2 + 3) * lm + 6 * l2 - 2),
        ],
        (6, 2): [
            0.25 * _SQRT15 * M * N * lm * (5 * m2 - 3),
            -0.125 * _SQRT15 * M * N * (2 + 3 * lm) * (5 * m2 - 1),
            0.25 * _SQRT15 * M * N * (3 * (1 + m2) * lm + 8 * m2 - 2),
            -0.125 * _SQRT15 * M * N * ((m2 + 3) * lm + 6 * m2 - 2),
        ],
        (6, 3): [
            0.25 * _SQRT15 * lm * n2 * (5 * n2 - 3),
            -0.125 * _SQRT15 * lm * (5 * n2 - 1) * (3 * n2 - 1),
            0.25 * _SQRT15 * lm * n2 * (3 * n2 - 1),
            0.125 * _SQRT15 * lm * (1 - n2 * n2),
        ],
        (6, 4): [
            3.75 * L * N * lm * (m2 - n2),
            -0.625 * L * N * (9 * lm * (m2 - n2) - 2 * m2 + 2),
            0.25 * L * N * (9 * lm * (m2 - n2) - 8 * m2 + 2),
            0.375 * L * N * (-lm * (m2 - n2) + 2 * m2 + 2),
        ],
        (6, 5): [
            3.75 * M * N * lm * (n2 - l2),
            -0.625 * M * N * (9 * lm * (n2 - l2) - 2 * l2 + 2),
            0.25 * M * N * (9 * lm * (n2 - l2) - 8 * l2 + 2),
            0.375 * M * N * (-lm * (n2 - l2) + 2 * l2 + 2),
        ],
        (6, 6): [
            3.75 * n2 * lm * lm,
            0.625 * (4 * n2 * (1 - n2) + lm * lm * (1 - 9 * n2)),
            0.25 * (lm * lm * (9 * n2 - 4) + 4 * (1 - 2 * n2) * (1 - 2 * n2)),
            0.375 * (1 - n2) * ((1 + n2) * (1 + n2) - 4 * l2 * m2),
        ],
    }


def _ff_paper_block(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _ff_paper_given(*args).items():
            cells[
                (
                    _cycle_index(_PAPER_F_CYCLE, a, t),
                    _cycle_index(_PAPER_F_CYCLE, b, t),
                )
            ] = value
    for a in range(7):
        for b in range(7):
            if (a, b) not in cells and (b, a) in cells:
                cells[(a, b)] = cells[(b, a)]
    return _assemble_block(cells, 4, 7, 7, L)


# ------------------------------------------------- paper order -> Structure
def _adapt_f_axis(block: torch.Tensor, axis: int) -> torch.Tensor:
    """Reindex one f axis of ``block`` from paper order into Structure order."""
    perm = torch.as_tensor(
        PAPER_TO_STRUCTURE_F_PERMUTATION, dtype=torch.long, device=block.device
    )
    out = block.index_select(axis, perm)
    sign = torch.as_tensor(
        PAPER_TO_STRUCTURE_F_SIGN, dtype=block.dtype, device=block.device
    )
    shape = [1] * out.dim()
    shape[axis] = len(PAPER_TO_STRUCTURE_F_SIGN)
    return out * sign.reshape(shape)


def f_angular_sf(L, M, N) -> torch.Tensor:
    """s-f angular factors, shape ``(1, 1, 7, P)`` = (sfsigma,) x s x f x pairs.

    Rows follow the s shell, columns follow ``STRUCTURE_F_AO_ORDER``.
    """
    return _adapt_f_axis(_sf_paper_block(L, M, N), 2)


def f_angular_pf(L, M, N) -> torch.Tensor:
    """p-f angular factors, shape ``(2, 3, 7, P)`` = (pfsigma, pfpi) x p x f."""
    return _adapt_f_axis(_pf_paper_block(L, M, N), 2)


def f_angular_df(L, M, N) -> torch.Tensor:
    """d-f angular factors, shape ``(3, 5, 7, P)`` = (dfsigma..dfdelta) x d x f."""
    return _adapt_f_axis(_df_paper_block(L, M, N), 2)


def f_angular_ff(L, M, N) -> torch.Tensor:
    """f-f angular factors, shape ``(4, 7, 7, P)`` = (ffsigma..ffphi) x f x f."""
    block = _ff_paper_block(L, M, N)
    return _adapt_f_axis(_adapt_f_axis(block, 1), 2)


F_ANGULAR_HELPERS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "sf": f_angular_sf,
        "pf": f_angular_pf,
        "df": f_angular_df,
        "ff": f_angular_ff,
    }
)

#: Source-locked s-f/p-f/d-f/f-f angular formulas are implemented above.
F_ANGULAR_FORMULAS_AVAILABLE = True


# @torch.compile(fullgraph=True, dynamic=True)  # optional extra flags
def Slater_Koster_Pair_SKF_vectorized(
    HDIM: int,
    dR_dxyz: torch.Tensor,
    L: torch.Tensor,
    M: torch.Tensor,
    N: torch.Tensor,
    L_dxyz: torch.Tensor,
    M_dxyz: torch.Tensor,
    N_dxyz: torch.Tensor,
    pair_mask_HH: torch.Tensor,
    pair_mask_HX: torch.Tensor,
    pair_mask_XH: torch.Tensor,
    pair_mask_XX: torch.Tensor,
    pair_mask_HY: torch.Tensor,
    pair_mask_XY: torch.Tensor,
    pair_mask_YH: torch.Tensor,
    pair_mask_YX: torch.Tensor,
    pair_mask_YY: torch.Tensor,
    dx: torch.Tensor,
    idx: torch.Tensor,
    IJ_pair_type: torch.Tensor,
    JI_pair_type: torch.Tensor,
    coeffs_tensor: torch.Tensor,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    H_INDEX_START: torch.Tensor,
    SH_shift: int,
    stress_weight: torch.Tensor | None = None,
    i0_stress: torch.Tensor | None = None,
    j0_stress: torch.Tensor | None = None,
    ml_ctx: dict | None = None,
    pair_mask_HZ: torch.Tensor | None = None,
    pair_mask_ZH: torch.Tensor | None = None,
    pair_mask_XZ: torch.Tensor | None = None,
    pair_mask_ZX: torch.Tensor | None = None,
    pair_mask_YZ: torch.Tensor | None = None,
    pair_mask_ZY: torch.Tensor | None = None,
    pair_mask_ZZ: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Build the Slater–Koster pair block (flattened) and its Cartesian derivatives
    using vectorized cubic-spline SKF coefficients (s, p, d orbitals).

    This routine assembles the AO block H0 for all requested pairs and its
    derivatives dH0 = dH0/d[x,y,z] by evaluating the spline-based SK integrals
    and applying the standard angular (direction cosine) factors. All writes are
    done with in-place index_add_ to allow accumulation across overlapping masks.

    Arguments
    ----------
    HDIM : int
        Per-atom AO block dimension used to index into the flattened block.
        Typical values: 1 (s), 4 (sp), 9 (spd). Must be consistent with the
        largest orbital shell present in the active masks; e.g., if any d-*
        masks are True, HDIM must be >= 9.

    dR_dxyz : torch.Tensor
        Derivatives of pair distances with respect to Cartesian components.
        Shape (3, num_pairs), dtype float, device consistent with L/M/N.
        Row 0/1/2 correspond to dR/dx, dR/dy, dR/dz.

    L, M, N : torch.Tensor
        Direction cosines for each pair. Shape (num_pairs,), dtype float.

    L_dxyz, M_dxyz, N_dxyz : torch.Tensor
        Derivatives of direction cosines. Shape (3, num_pairs), dtype float.

    pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
    pair_mask_HY, pair_mask_XY, pair_mask_YH, pair_mask_YX, pair_mask_YY : torch.BoolTensor
        Boolean masks (shape (num_pairs,)) selecting pair classes:
        - H: hydrogen-like (s-only, n_orb == 1)
        - X: sp atom (s + p, n_orb == 4)
        - Y: spd atom (s + p + d, n_orb == 9)
        The two letters indicate (left atom, right atom), e.g. HX = H–X, YX = Y–X.
        Masks can be combined (ORed) when contributions are shared (e.g. HX and XX
        both use s–p).

    pair_mask_HZ, pair_mask_ZH, pair_mask_XZ, pair_mask_ZX,
    pair_mask_YZ, pair_mask_ZY, pair_mask_ZZ : torch.BoolTensor or None, optional
        Boolean masks selecting the 16-orbital (spdf) pair classes, where ``Z``
        denotes an atom with ``n_orb == 16``.  These exist so f-containing
        neighbor pairs are *routed* rather than silently skipped by the 1/4/9
        masks.  Until the f angular formulas are source-locked, any non-empty f
        mask raises :class:`FAngularFormulaSourceError`.  Callers that predate f
        routing may omit them; ``None`` means "this caller has no f pairs".

    dx : torch.Tensor
        Radial offset used in spline evaluation inside the selected interval for
        each pair (same meaning as local distance minus the knot position).
        Shape (num_pairs,), dtype float.

    idx : torch.LongTensor
        Spline interval index for each pair (selects which cubic to evaluate).
        Shape (num_pairs,), dtype long.

    IJ_pair_type, JI_pair_type : torch.LongTensor
        Integer pair-type indices used to select the proper row in coeffs_tensor
        for the I→J and J→I directions, respectively (handles sign conventions
        for s–p and p–s, etc.). Shape (num_pairs,), dtype long.

    coeffs_tensor : torch.Tensor
        Pre-tabulated cubic-spline coefficients for all SK channels.
        Indexed as coeffs_tensor[pair_type, interval_idx, channel, 0..3],
        where the last axis stores a0..a3 of the cubic a0 + a1*dx + a2*dx^2 + a3*dx^3.
        The channel axis follows the canonical 40-entry extended-SKF order in
        ``_bond_integral._CHANNELS`` (20 named H channels, then the 20 matching
        S channels).  Channels are resolved *by name* through
        :data:`SK_CHANNEL_INDEX`; ``SH_shift`` only selects the ``"H"`` or
        ``"S"`` name prefix.  Do not reintroduce numeric
        ``channel + 10 * SH_shift`` addressing — that was the legacy 20-channel
        simple-format packing and it selects the wrong integrals here.

        Expected shape: (n_pair_types, n_intervals, 40, 4).

    neighbor_I, neighbor_J : torch.LongTensor
        Atom indices (per pair) used to compute the flattened AO indices.
        Shape (num_pairs,), dtype long.

    H_INDEX_START : torch.LongTensor
        For each atom index k, H_INDEX_START[k] is the starting AO offset of atom k
        within the per-atom block. Used to place pair contributions into the
        flattened [HDIM x HDIM] block. Shape (num_atoms,), dtype long.

    SH_shift : int
        Selects the Hamiltonian (``0``) or overlap (``1``) half of the canonical
        channel list by choosing the ``"H"`` or ``"S"`` channel-name prefix.

    stress_weight : torch.Tensor or None
        If not None, a density-weight matrix of shape ``(HDIM, HDIM)`` used
        to accumulate the on-the-fly stress pair gradient.  For each derivative
        site ``dH[i0+a, j0+b]/d(x,y,z)`` the function computes
        ``stress_weight[i0+a, j0+b] * derivative`` and accumulates it into
        ``pair_grad (P, 3)``.  When None (the default), no stress accumulation
        happens and the function returns only ``(H0, dH0)``.

    i0_stress, j0_stress : torch.LongTensor or None
        Per-pair AO offsets for stress accumulation, shape ``(num_pairs,)``.
        Required when ``stress_weight`` is provided.

    Returns
    -------
    H0 : torch.Tensor
        Updated Hamiltonian matrix elements tensor with new values for the pairs processed.

    dH0 : torch.Tensor
        Derivatives of the Hamiltonian matrix elements with respect to Cartesian coordinates.
        Shape: (3, HDIM * HDIM)

    pair_grad : torch.Tensor (only when stress_weight is not None)
        Per-pair weighted gradient, shape ``(num_pairs, 3)``.  The stress
        tensor is then ``extra_scale * einsum("pc,pd->cd", pair_grad, Rab) / V``.

    Notes
    -----
    - The function uses vectorized bond integral evaluations for improved computational efficiency.
    - The calculation covers both overlap and Hamiltonian matrix elements for s and p orbitals.
    - Direction cosine derivatives and bond integral derivatives are used to compute the gradients.
    - Periodic boundary conditions or lattice considerations are assumed handled externally.
    """
    # %%% Standard Slater-Koster sp-parameterization for an atomic block between a pair of atoms
    # %%% IDim, JDim: dimensions of the output block, e.g. 1 x 4 for H-O or 4 x 4 for O-O, or 4 x 1 for O-H
    # %%% Ra, Rb: are the vectors of the positions of the two atoms
    # %%% Type_pair(1 or 2): Character of the type of each atom in the pair, e.g. 'H' for hydrogen of 'O' for oxygen
    # %%% fss_sigma, ... , fpp_pi: paramters for the bond integrals
    # %%% diagonal(1 or 2): atomic energies Es and Ep or diagonal elements of the overlap i.e. diagonal = 1

    # Explicit f boundary. 16-orbital pairs are routed here (rather than being
    # dropped by the 1/4/9 masks) so that missing f angular formulas surface as
    # a loud, named failure instead of silently zeroed f blocks.
    _require_f_formula_source(
        pair_mask_HZ,
        pair_mask_ZH,
        pair_mask_XZ,
        pair_mask_ZX,
        pair_mask_YZ,
        pair_mask_ZY,
        pair_mask_ZZ,
        context="Slater_Koster_Pair_SKF_vectorized",
    )

    # Callers that predate f routing pass ``None`` for the 16-orbital masks.
    # Normalise them to all-False tensors so the s/p/d masks below can be
    # extended with ``|`` unconditionally, and record whether any f pair is
    # actually present so the f blocks can be skipped entirely otherwise.
    _f_none = torch.zeros_like(pair_mask_HH)
    if pair_mask_HZ is None:
        pair_mask_HZ = _f_none
    if pair_mask_ZH is None:
        pair_mask_ZH = _f_none
    if pair_mask_XZ is None:
        pair_mask_XZ = _f_none
    if pair_mask_ZX is None:
        pair_mask_ZX = _f_none
    if pair_mask_YZ is None:
        pair_mask_YZ = _f_none
    if pair_mask_ZY is None:
        pair_mask_ZY = _f_none
    if pair_mask_ZZ is None:
        pair_mask_ZZ = _f_none
    _f_any_mask = (
        pair_mask_HZ
        | pair_mask_ZH
        | pair_mask_XZ
        | pair_mask_ZX
        | pair_mask_YZ
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    _f_present = bool(_f_any_mask.any())

    # Helper: optionally accumulate weighted per-pair gradient for stress.
    # dxyz has shape (3, P_masked). W is the (HDIM,HDIM) density-weight matrix.
    # _sg(mask, row_offset, col_offset, dxyz_3P) does:
    #   pair_grad[mask, :] += W[i0+row, j0+col] * dxyz_3P^T
    _W = stress_weight
    if _W is not None:
        pair_grad = torch.zeros(
            (dR_dxyz.shape[1], 3), dtype=dR_dxyz.dtype, device=dR_dxyz.device
        )
        _i0 = i0_stress
        _j0 = j0_stress
    else:
        pair_grad = None

    def _sg(mask, row_off, col_off, dxyz):
        """Accumulate W[i0+row_off, j0+col_off] * dxyz into pair_grad."""
        if pair_grad is None:
            return
        if mask is None:
            w = _W[_i0 + row_off, _j0 + col_off]  # (P,)
            pair_grad[:] += w.unsqueeze(-1) * dxyz.T
        else:
            w = _W[_i0[mask] + row_off, _j0[mask] + col_off]  # (P_mask,)
            pair_grad[mask] += w.unsqueeze(-1) * dxyz.T

    H0 = torch.zeros((HDIM * HDIM), dtype=dR_dxyz.dtype, device=dR_dxyz.device)
    dH0 = torch.zeros(3, HDIM * HDIM, dtype=H0.dtype, device=H0.device)

    # -- ML / spline evaluation helper --
    _use_ml = ml_ctx is not None

    def _get_val_dR(pair_type_sel, idx_sel, dx_sel, channel, mask, direction="IJ"):
        """Return (value, dvalue_dR) for a single named SK channel.

        ``channel`` is a base channel name without its H/S prefix, e.g. ``"ss0"``
        or ``"pd1"``; ``SH_shift`` selects the Hamiltonian or overlap variant.
        """
        if _use_ml:
            from ._ml_sk import ml_eval_channel

            # The ML head is trained on the legacy 10-channel numbering, so
            # translate the canonical name back into that index. Channels with
            # no ML counterpart (any f channel) are rejected explicitly rather
            # than silently mapped onto an unrelated head.
            try:
                ml_channel = _ML_LEGACY_CHANNEL_INDEX[channel]
            except KeyError:
                raise NotImplementedError(
                    f"The ML Slater-Koster model has no channel for {channel!r}; "
                    "it only covers the legacy s/p/d channels "
                    f"({', '.join(_ML_LEGACY_CHANNEL_INDEX)})."
                ) from None
            return ml_eval_channel(ml_ctx, mask, ml_channel, SH_shift, direction)
        else:
            ch = sk_channel_index(sk_channel_name(channel, SH_shift))
            cs = coeffs_tensor[pair_type_sel, idx_sel, ch]
            val = (
                cs[:, 0]
                + cs[:, 1] * dx_sel
                + cs[:, 2] * dx_sel**2
                + cs[:, 3] * dx_sel**3
            )
            dval = cs[:, 1] + 2 * cs[:, 2] * dx_sel + 3 * cs[:, 3] * dx_sel**2
            return val, dval

    # -- end ML / spline helper --

    #######
    HSSS_all, HSSS_dR = _get_val_dR(IJ_pair_type, idx, dx, "ss0", slice(None), "IJ")
    H0.index_add_(
        0, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J], HSSS_all
    )
    #######

    # H-H (ARYAN NOTE I THINK THIS MEANS S-S)
    ######### dH/dx
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0.index_add_(
        1, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J], HSSS_dxyz
    )
    _sg(None, 0, 0, HSSS_dxyz)
    #########

    # H-X (ARYAN NOTE THIS MEANS S-P)
    ###### HSPS_all
    # Needs a p shell on atom J, i.e. n_orb(J) in {4, 9, 16}.
    tmp_mask = (
        pair_mask_HX
        | pair_mask_XX
        | pair_mask_HY
        | pair_mask_YY
        | pair_mask_XY
        | pair_mask_YX
        | pair_mask_HZ
        | pair_mask_XZ
        | pair_mask_YZ
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    HSPS_all, HSPS_dR = _get_val_dR(
        sel_IJ, sel_idx, dx[tmp_mask], "sp0", tmp_mask, "IJ"
    )

    H0.index_add_(0, idx_row * HDIM + idx_col + 1, L[tmp_mask] * HSPS_all)
    H0.index_add_(0, idx_row * HDIM + idx_col + 2, M[tmp_mask] * HSPS_all)
    H0.index_add_(0, idx_row * HDIM + idx_col + 3, N[tmp_mask] * HSPS_all)

    ######### dH/dx
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, tmp_mask]

    HSPS_sp_L_dxyz = L[tmp_mask] * HSPS_dxyz + L_dxyz[:, tmp_mask] * HSPS_all
    HSPS_sp_M_dxyz = M[tmp_mask] * HSPS_dxyz + M_dxyz[:, tmp_mask] * HSPS_all
    HSPS_sp_N_dxyz = N[tmp_mask] * HSPS_dxyz + N_dxyz[:, tmp_mask] * HSPS_all
    dH0.index_add_(1, idx_row * HDIM + idx_col + 1, HSPS_sp_L_dxyz)
    dH0.index_add_(1, idx_row * HDIM + idx_col + 2, HSPS_sp_M_dxyz)
    dH0.index_add_(1, idx_row * HDIM + idx_col + 3, HSPS_sp_N_dxyz)
    _sg(tmp_mask, 0, 1, HSPS_sp_L_dxyz)
    _sg(tmp_mask, 0, 2, HSPS_sp_M_dxyz)
    _sg(tmp_mask, 0, 3, HSPS_sp_N_dxyz)
    #########

    ### HPSS_all ###
    tmp_mask = (
        pair_mask_XH
        | pair_mask_XX
        | pair_mask_YH
        | pair_mask_YY
        | pair_mask_XY
        | pair_mask_YX
        | pair_mask_ZH
        | pair_mask_ZX
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    HPSS_all, HPSS_dR = _get_val_dR(
        sel_IJ, sel_idx, dx[tmp_mask], "sp0", tmp_mask, "JI"
    )

    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col, -L[tmp_mask] * HPSS_all)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col, -M[tmp_mask] * HPSS_all)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col, -N[tmp_mask] * HPSS_all)

    ################
    ######### dH/dx
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, tmp_mask]

    HPSS_ps_L_dxyz = -L[tmp_mask] * HPSS_dxyz - L_dxyz[:, tmp_mask] * HPSS_all
    HPSS_ps_M_dxyz = -M[tmp_mask] * HPSS_dxyz - M_dxyz[:, tmp_mask] * HPSS_all
    HPSS_ps_N_dxyz = -N[tmp_mask] * HPSS_dxyz - N_dxyz[:, tmp_mask] * HPSS_all
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col, HPSS_ps_L_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col, HPSS_ps_M_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col, HPSS_ps_N_dxyz)
    _sg(tmp_mask, 1, 0, HPSS_ps_L_dxyz)
    _sg(tmp_mask, 2, 0, HPSS_ps_M_dxyz)
    _sg(tmp_mask, 3, 0, HPSS_ps_N_dxyz)
    #########

    # X-X (Means P-P)
    # Needs a p shell on both atoms, i.e. n_orb in {4, 9, 16} on each side.
    tmp_mask = (
        pair_mask_XX
        | pair_mask_YY
        | pair_mask_XY
        | pair_mask_YX
        | pair_mask_XZ
        | pair_mask_ZX
        | pair_mask_YZ
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    L_XX = L[tmp_mask]
    M_XX = M[tmp_mask]
    N_XX = N[tmp_mask]
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    HPPS, HPPS_dR = _get_val_dR(sel_IJ, sel_idx, dx[tmp_mask], "pp0", tmp_mask, "IJ")

    HPPP, HPPP_dR = _get_val_dR(sel_IJ, sel_idx, dx[tmp_mask], "pp1", tmp_mask, "IJ")

    PPSMPP = HPPS - HPPP
    PXPX = HPPP + L_XX * L_XX * PPSMPP
    PXPY = L_XX * M_XX * PPSMPP
    PXPZ = L_XX * N_XX * PPSMPP
    PYPX = M_XX * L_XX * PPSMPP
    PYPY = HPPP + M_XX * M_XX * PPSMPP
    PYPZ = M_XX * N_XX * PPSMPP
    PZPX = N_XX * L_XX * PPSMPP
    PZPY = N_XX * M_XX * PPSMPP
    PZPZ = HPPP + N_XX * N_XX * PPSMPP

    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 1, PXPX)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 2, PXPY)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 3, PXPZ)

    ####

    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 1, PYPX)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 2, PYPY)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 3, PYPZ)

    ####

    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 1, PZPX)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 2, PZPY)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 3, PZPZ)

    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, tmp_mask]
    L_dxyz_XX = L_dxyz[:, tmp_mask]
    M_dxyz_XX = M_dxyz[:, tmp_mask]
    N_dxyz_XX = N_dxyz[:, tmp_mask]

    HPPS_dxyz = HPPS_dR * dR_dxyz_XX
    HPPP_dxyz = HPPP_dR * dR_dxyz_XX

    PPSMPP_dxyz = HPPS_dxyz - HPPP_dxyz
    PXPX_dxyz = HPPP_dxyz + (L_XX**2) * PPSMPP_dxyz + 2 * L_XX * L_dxyz_XX * PPSMPP
    PXPY_dxyz = (
        L_XX * M_XX * PPSMPP_dxyz
        + L_dxyz_XX * M_XX * PPSMPP
        + L_XX * M_dxyz_XX * PPSMPP
    )
    PXPZ_dxyz = (
        L_XX * N_XX * PPSMPP_dxyz
        + L_dxyz_XX * N_XX * PPSMPP
        + L_XX * N_dxyz_XX * PPSMPP
    )
    PYPX_dxyz = (
        M_XX * L_XX * PPSMPP_dxyz
        + M_XX * L_dxyz_XX * PPSMPP
        + M_dxyz_XX * L_XX * PPSMPP
    )
    PYPY_dxyz = HPPP_dxyz + (M_XX**2) * PPSMPP_dxyz + 2 * M_XX * M_dxyz_XX * PPSMPP
    PYPZ_dxyz = (
        M_XX * N_XX * PPSMPP_dxyz
        + M_dxyz_XX * N_XX * PPSMPP
        + M_XX * N_dxyz_XX * PPSMPP
    )
    PZPX_dxyz = (
        N_XX * L_XX * PPSMPP_dxyz
        + N_XX * L_dxyz_XX * PPSMPP
        + N_dxyz_XX * L_XX * PPSMPP
    )
    PZPY_dxyz = (
        N_XX * M_XX * PPSMPP_dxyz
        + N_XX * M_dxyz_XX * PPSMPP
        + N_dxyz_XX * M_XX * PPSMPP
    )
    PZPZ_dxyz = HPPP_dxyz + (N_XX**2) * PPSMPP_dxyz + 2 * N_XX * N_dxyz_XX * PPSMPP

    ####
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 1, PXPX_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 2, PXPY_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 3, PXPZ_dxyz)
    _sg(tmp_mask, 1, 1, PXPX_dxyz)
    _sg(tmp_mask, 1, 2, PXPY_dxyz)
    _sg(tmp_mask, 1, 3, PXPZ_dxyz)

    ####

    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 1, PYPX_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 2, PYPY_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 3, PYPZ_dxyz)
    _sg(tmp_mask, 2, 1, PYPX_dxyz)
    _sg(tmp_mask, 2, 2, PYPY_dxyz)
    _sg(tmp_mask, 2, 3, PYPZ_dxyz)

    ####

    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 1, PZPX_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 2, PZPY_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 3, PZPZ_dxyz)
    _sg(tmp_mask, 3, 1, PZPX_dxyz)
    _sg(tmp_mask, 3, 2, PZPY_dxyz)
    _sg(tmp_mask, 3, 3, PZPZ_dxyz)
    #########

    ### s-d
    # Needs a d shell on atom J, i.e. n_orb(J) in {9, 16}.
    tmp_mask = (
        pair_mask_HY
        | pair_mask_XY
        | pair_mask_YY
        | pair_mask_HZ
        | pair_mask_XZ
        | pair_mask_YZ
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    V_sd_sigma, V_sd_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "sd0", tmp_mask, "IJ"
    )
    H_S_XY = (3**0.5) * tmp_L * tmp_M * V_sd_sigma
    H_S_YZ = (3**0.5) * tmp_M * tmp_N * V_sd_sigma
    H_S_ZX = (3**0.5) * tmp_N * tmp_L * V_sd_sigma
    H_S_X2Y2 = 0.5 * (3**0.5) * (tmp_L**2 - tmp_M**2) * V_sd_sigma
    H_S_Z2 = (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_sd_sigma
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 4, H_S_XY)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 5, H_S_YZ)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 6, H_S_ZX)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 7, H_S_X2Y2)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 8, H_S_Z2)
    # s-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_sd_sigma_dxyz = V_sd_sigma_dR * tmp_dR_dxyz
    H_S_XY_dxyz = (3**0.5) * (
        tmp_L_dxyz * tmp_M * V_sd_sigma
        + tmp_L * tmp_M_dxyz * V_sd_sigma
        + tmp_L * tmp_M * V_sd_sigma_dxyz
    )
    H_S_YZ_dxyz = (3**0.5) * (
        tmp_M_dxyz * tmp_N * V_sd_sigma
        + tmp_M * tmp_N_dxyz * V_sd_sigma
        + tmp_M * tmp_N * V_sd_sigma_dxyz
    )
    H_S_ZX_dxyz = (3**0.5) * (
        tmp_N_dxyz * tmp_L * V_sd_sigma
        + tmp_N * tmp_L_dxyz * V_sd_sigma
        + tmp_N * tmp_L * V_sd_sigma_dxyz
    )
    H_S_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz) * V_sd_sigma
            + (tmp_L**2 - tmp_M**2) * V_sd_sigma_dxyz
        )
    )
    H_S_Z2_dxyz = (
        2 * tmp_N * tmp_N_dxyz - 0.5 * (2 * tmp_L * tmp_L_dxyz + 2 * tmp_M * tmp_M_dxyz)
    ) * V_sd_sigma + (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_sd_sigma_dxyz
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 4, H_S_XY_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 5, H_S_YZ_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 6, H_S_ZX_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 7, H_S_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 8, H_S_Z2_dxyz)
    _sg(tmp_mask, 0, 4, H_S_XY_dxyz)
    _sg(tmp_mask, 0, 5, H_S_YZ_dxyz)
    _sg(tmp_mask, 0, 6, H_S_ZX_dxyz)
    _sg(tmp_mask, 0, 7, H_S_X2Y2_dxyz)
    _sg(tmp_mask, 0, 8, H_S_Z2_dxyz)

    ### p-d
    # Needs a p shell on atom I and a d shell on atom J.
    tmp_mask = (
        pair_mask_XY
        | pair_mask_YY
        | pair_mask_XZ
        | pair_mask_YZ
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    V_pd_sigma, V_pd_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "pd0", tmp_mask, "IJ"
    )
    V_pd_pi, V_pd_pi_dR = _get_val_dR(sel_IJ, sel_idx, tmp_dx, "pd1", tmp_mask, "IJ")
    H_X_XY = (3**0.5) * tmp_L**2 * tmp_M * V_pd_sigma + tmp_M * (
        1 - 2 * tmp_L**2
    ) * V_pd_pi
    H_X_YZ = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_X_ZX = (3**0.5) * tmp_L**2 * tmp_N * V_pd_sigma + tmp_N * (
        1 - 2 * tmp_L**2
    ) * V_pd_pi
    H_X_X2Y2 = (
        0.5 * (3**0.5) * tmp_L * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_pd_pi
    )
    H_X_Z2 = (
        tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        - 3**0.5 * tmp_L * tmp_N**2 * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 4, H_X_XY)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 5, H_X_YZ)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 6, H_X_ZX)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 7, H_X_X2Y2)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 8, H_X_Z2)
    H_Y_XY = (3**0.5) * tmp_M**2 * tmp_L * V_pd_sigma + tmp_L * (
        1 - 2 * tmp_M**2
    ) * V_pd_pi
    H_Y_YZ = (3**0.5) * tmp_M**2 * tmp_N * V_pd_sigma + tmp_N * (
        1 - 2 * tmp_M**2
    ) * V_pd_pi
    H_Y_ZX = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_Y_X2Y2 = (
        0.5 * (3**0.5) * tmp_M * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_pd_pi
    )
    H_Y_Z2 = (
        tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        - 3**0.5 * tmp_M * tmp_N**2 * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 4, H_Y_XY)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 5, H_Y_YZ)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 6, H_Y_ZX)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 7, H_Y_X2Y2)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 8, H_Y_Z2)
    H_Z_XY = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_Z_YZ = (3**0.5) * tmp_N**2 * tmp_M * V_pd_sigma + tmp_M * (
        1 - 2 * tmp_N**2
    ) * V_pd_pi
    H_Z_ZX = (3**0.5) * tmp_N**2 * tmp_L * V_pd_sigma + tmp_L * (
        1 - 2 * tmp_N**2
    ) * V_pd_pi
    H_Z_X2Y2 = (
        0.5 * (3**0.5) * tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_pi
    )
    H_Z_Z2 = (
        tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 4, H_Z_XY)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 5, H_Z_YZ)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 6, H_Z_ZX)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 7, H_Z_X2Y2)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 8, H_Z_Z2)
    # p-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_pd_sigma_dxyz = V_pd_sigma_dR * tmp_dR_dxyz
    V_pd_pi_dxyz = V_pd_pi_dR * tmp_dR_dxyz

    H_X_XY_dxyz = (
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_M * V_pd_sigma
            + tmp_L**2 * tmp_M_dxyz * V_pd_sigma
            + tmp_L**2 * tmp_M * V_pd_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_M) * V_pd_pi
        + tmp_M * (1 - 2 * tmp_L**2) * V_pd_pi_dxyz
    )
    H_X_YZ_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_X_ZX_dxyz = (
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_N * V_pd_sigma
            + tmp_L**2 * tmp_N_dxyz * V_pd_sigma
            + tmp_L**2 * tmp_N * V_pd_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_N) * V_pd_pi
        + tmp_N * (1 - 2 * tmp_L**2) * V_pd_pi_dxyz
    )
    H_X_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_L_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_L * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        + (
            tmp_L_dxyz * (1 - tmp_L**2 + tmp_M**2)
            - tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_pd_pi_dxyz
    )
    H_X_Z2_dxyz = (
        (
            tmp_L_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_L * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        - 3**0.5 * (tmp_L_dxyz * tmp_N**2 + 2 * tmp_L * tmp_N * tmp_N_dxyz) * V_pd_pi
        - 3**0.5 * tmp_L * tmp_N**2 * V_pd_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 4, H_X_XY_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 5, H_X_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 6, H_X_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 7, H_X_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 1) * HDIM + idx_col + 8, H_X_Z2_dxyz)
    _sg(tmp_mask, 1, 4, H_X_XY_dxyz)
    _sg(tmp_mask, 1, 5, H_X_YZ_dxyz)
    _sg(tmp_mask, 1, 6, H_X_ZX_dxyz)
    _sg(tmp_mask, 1, 7, H_X_X2Y2_dxyz)
    _sg(tmp_mask, 1, 8, H_X_Z2_dxyz)
    H_Y_XY_dxyz = (
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_L * V_pd_sigma
            + tmp_M**2 * tmp_L_dxyz * V_pd_sigma
            + tmp_M**2 * tmp_L * V_pd_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_L) * V_pd_pi
        + tmp_L * (1 - 2 * tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_YZ_dxyz = (
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_M**2 * tmp_N_dxyz * V_pd_sigma
            + tmp_M**2 * tmp_N * V_pd_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_N) * V_pd_pi
        + tmp_N * (1 - 2 * tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_ZX_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_Y_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_M_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_M * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        - (
            tmp_M_dxyz * (1 + tmp_L**2 - tmp_M**2)
            + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_Z2_dxyz = (
        (
            tmp_M_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_M * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        - 3**0.5 * (tmp_M_dxyz * tmp_N**2 + 2 * tmp_M * tmp_N * tmp_N_dxyz) * V_pd_pi
        - 3**0.5 * tmp_M * tmp_N**2 * V_pd_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 4, H_Y_XY_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 5, H_Y_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 6, H_Y_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 7, H_Y_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 2) * HDIM + idx_col + 8, H_Y_Z2_dxyz)
    _sg(tmp_mask, 2, 4, H_Y_XY_dxyz)
    _sg(tmp_mask, 2, 5, H_Y_YZ_dxyz)
    _sg(tmp_mask, 2, 6, H_Y_ZX_dxyz)
    _sg(tmp_mask, 2, 7, H_Y_X2Y2_dxyz)
    _sg(tmp_mask, 2, 8, H_Y_Z2_dxyz)
    H_Z_XY_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_Z_YZ_dxyz = (
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_M * V_pd_sigma
            + tmp_N**2 * tmp_M_dxyz * V_pd_sigma
            + tmp_N**2 * tmp_M * V_pd_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_M) * V_pd_pi
        + tmp_M * (1 - 2 * tmp_N**2) * V_pd_pi_dxyz
    )
    H_Z_ZX_dxyz = (
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_L * V_pd_sigma
            + tmp_N**2 * tmp_L_dxyz * V_pd_sigma
            + tmp_N**2 * tmp_L * V_pd_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_L) * V_pd_pi
        + tmp_L * (1 - 2 * tmp_N**2) * V_pd_pi_dxyz
    )
    H_Z_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        - (
            tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
            + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_pi_dxyz
    )
    H_Z_Z2_dxyz = (
        (
            tmp_N_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_N * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        + 3**0.5
        * (
            tmp_N_dxyz * (tmp_L**2 + tmp_M**2)
            + 2 * tmp_N * (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_pd_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 4, H_Z_XY_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 5, H_Z_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 6, H_Z_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 7, H_Z_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 3) * HDIM + idx_col + 8, H_Z_Z2_dxyz)
    _sg(tmp_mask, 3, 4, H_Z_XY_dxyz)
    _sg(tmp_mask, 3, 5, H_Z_YZ_dxyz)
    _sg(tmp_mask, 3, 6, H_Z_ZX_dxyz)
    _sg(tmp_mask, 3, 7, H_Z_X2Y2_dxyz)
    _sg(tmp_mask, 3, 8, H_Z_Z2_dxyz)

    ### d-s
    # Needs a d shell on atom I, i.e. n_orb(I) in {9, 16}.
    tmp_mask = (
        pair_mask_YH
        | pair_mask_YX
        | pair_mask_YY
        | pair_mask_ZH
        | pair_mask_ZX
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    V_ds_sigma, V_ds_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "sd0", tmp_mask, "JI"
    )
    H_XY_S = (3**0.5) * tmp_L * tmp_M * V_ds_sigma
    H_YZ_S = (3**0.5) * tmp_M * tmp_N * V_ds_sigma
    H_ZX_S = (3**0.5) * tmp_N * tmp_L * V_ds_sigma
    H_X2Y2_S = 0.5 * (3**0.5) * (tmp_L**2 - tmp_M**2) * V_ds_sigma
    H_Z2_S = (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_ds_sigma
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col, H_XY_S)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col, H_YZ_S)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col, H_ZX_S)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col, H_X2Y2_S)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col, H_Z2_S)
    # d-s/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_ds_sigma_dxyz = V_ds_sigma_dR * tmp_dR_dxyz
    H_XY_S_dxyz = (3**0.5) * (
        tmp_L_dxyz * tmp_M * V_ds_sigma
        + tmp_L * tmp_M_dxyz * V_ds_sigma
        + tmp_L * tmp_M * V_ds_sigma_dxyz
    )
    H_YZ_S_dxyz = (3**0.5) * (
        tmp_M_dxyz * tmp_N * V_ds_sigma
        + tmp_M * tmp_N_dxyz * V_ds_sigma
        + tmp_M * tmp_N * V_ds_sigma_dxyz
    )
    H_ZX_S_dxyz = (3**0.5) * (
        tmp_N_dxyz * tmp_L * V_ds_sigma
        + tmp_N * tmp_L_dxyz * V_ds_sigma
        + tmp_N * tmp_L * V_ds_sigma_dxyz
    )
    H_X2Y2_S_dxyz = (
        0.5
        * (3**0.5)
        * (
            (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz) * V_ds_sigma
            + (tmp_L**2 - tmp_M**2) * V_ds_sigma_dxyz
        )
    )
    H_Z2_S_dxyz = (
        2 * tmp_N * tmp_N_dxyz - (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
    ) * V_ds_sigma + (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_ds_sigma_dxyz
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col, H_XY_S_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col, H_YZ_S_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col, H_ZX_S_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col, H_X2Y2_S_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col, H_Z2_S_dxyz)
    _sg(tmp_mask, 4, 0, H_XY_S_dxyz)
    _sg(tmp_mask, 5, 0, H_YZ_S_dxyz)
    _sg(tmp_mask, 6, 0, H_ZX_S_dxyz)
    _sg(tmp_mask, 7, 0, H_X2Y2_S_dxyz)
    _sg(tmp_mask, 8, 0, H_Z2_S_dxyz)

    ### d-p
    # Needs a d shell on atom I and a p shell on atom J.
    tmp_mask = (
        pair_mask_YX
        | pair_mask_YY
        | pair_mask_ZX
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    V_dp_sigma, V_dp_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "pd0", tmp_mask, "JI"
    )
    V_dp_pi, V_dp_pi_dR = _get_val_dR(sel_IJ, sel_idx, tmp_dx, "pd1", tmp_mask, "JI")
    H_XY_X = -(
        (3**0.5) * tmp_L**2 * tmp_M * V_dp_sigma + tmp_M * (1 - 2 * tmp_L**2) * V_dp_pi
    )
    H_XY_Y = -(
        (3**0.5) * tmp_M**2 * tmp_L * V_dp_sigma + tmp_L * (1 - 2 * tmp_M**2) * V_dp_pi
    )
    H_XY_Z = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 1, H_XY_X)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 2, H_XY_Y)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 3, H_XY_Z)
    H_YZ_X = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H_YZ_Y = -(
        (3**0.5) * tmp_M**2 * tmp_N * V_dp_sigma + tmp_N * (1 - 2 * tmp_M**2) * V_dp_pi
    )
    H_YZ_Z = -(
        (3**0.5) * tmp_N**2 * tmp_M * V_dp_sigma + tmp_M * (1 - 2 * tmp_N**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 1, H_YZ_X)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 2, H_YZ_Y)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 3, H_YZ_Z)
    H_ZX_X = -(
        (3**0.5) * tmp_L**2 * tmp_N * V_dp_sigma + tmp_N * (1 - 2 * tmp_L**2) * V_dp_pi
    )
    H_ZX_Y = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H_ZX_Z = -(
        (3**0.5) * tmp_N**2 * tmp_L * V_dp_sigma + tmp_L * (1 - 2 * tmp_N**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 1, H_ZX_X)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 2, H_ZX_Y)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 3, H_ZX_Z)
    H_X2Y2_X = -(
        0.5 * (3**0.5) * tmp_L * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_dp_pi
    )
    H_X2Y2_Y = -(
        0.5 * (3**0.5) * tmp_M * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_dp_pi
    )
    H_X2Y2_Z = -(
        0.5 * (3**0.5) * tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 1, H_X2Y2_X)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 2, H_X2Y2_Y)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 3, H_X2Y2_Z)
    H_Z2_X = -(
        tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        - 3**0.5 * tmp_L * tmp_N**2 * V_dp_pi
    )
    H_Z2_Y = -(
        tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        - 3**0.5 * tmp_M * tmp_N**2 * V_dp_pi
    )
    H_Z2_Z = -(
        tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 1, H_Z2_X)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 2, H_Z2_Y)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 3, H_Z2_Z)
    # d-p/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_dp_sigma_dxyz = V_dp_sigma_dR * tmp_dR_dxyz
    V_dp_pi_dxyz = V_dp_pi_dR * tmp_dR_dxyz
    H_XY_X_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_M * V_dp_sigma
            + tmp_L**2 * tmp_M_dxyz * V_dp_sigma
            + tmp_L**2 * tmp_M * V_dp_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_M) * V_dp_pi
        + tmp_M * (1 - 2 * tmp_L**2) * V_dp_pi_dxyz
    )
    H_XY_Y_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_L * V_dp_sigma
            + tmp_M**2 * tmp_L_dxyz * V_dp_sigma
            + tmp_M**2 * tmp_L * V_dp_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_L) * V_dp_pi
        + tmp_L * (1 - 2 * tmp_M**2) * V_dp_pi_dxyz
    )
    H_XY_Z_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 1, H_XY_X_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 2, H_XY_Y_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 3, H_XY_Z_dxyz)
    _sg(tmp_mask, 4, 1, H_XY_X_dxyz)
    _sg(tmp_mask, 4, 2, H_XY_Y_dxyz)
    _sg(tmp_mask, 4, 3, H_XY_Z_dxyz)
    H_YZ_X_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    H_YZ_Y_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_M**2 * tmp_N_dxyz * V_dp_sigma
            + tmp_M**2 * tmp_N * V_dp_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_N) * V_dp_pi
        + tmp_N * (1 - 2 * tmp_M**2) * V_dp_pi_dxyz
    )
    H_YZ_Z_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_M * V_dp_sigma
            + tmp_N**2 * tmp_M_dxyz * V_dp_sigma
            + tmp_N**2 * tmp_M * V_dp_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_M) * V_dp_pi
        + tmp_M * (1 - 2 * tmp_N**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 1, H_YZ_X_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 2, H_YZ_Y_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 3, H_YZ_Z_dxyz)
    _sg(tmp_mask, 5, 1, H_YZ_X_dxyz)
    _sg(tmp_mask, 5, 2, H_YZ_Y_dxyz)
    _sg(tmp_mask, 5, 3, H_YZ_Z_dxyz)
    H_ZX_X_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_N * V_dp_sigma
            + tmp_L**2 * tmp_N_dxyz * V_dp_sigma
            + tmp_L**2 * tmp_N * V_dp_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_N) * V_dp_pi
        + tmp_N * (1 - 2 * tmp_L**2) * V_dp_pi_dxyz
    )
    H_ZX_Y_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    H_ZX_Z_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_L * V_dp_sigma
            + tmp_N**2 * tmp_L_dxyz * V_dp_sigma
            + tmp_N**2 * tmp_L * V_dp_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_L) * V_dp_pi
        + tmp_L * (1 - 2 * tmp_N**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 1, H_ZX_X_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 2, H_ZX_Y_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 3, H_ZX_Z_dxyz)
    _sg(tmp_mask, 6, 1, H_ZX_X_dxyz)
    _sg(tmp_mask, 6, 2, H_ZX_Y_dxyz)
    _sg(tmp_mask, 6, 3, H_ZX_Z_dxyz)
    H_X2Y2_X_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_L_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_L * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        + (
            tmp_L_dxyz * (1 - tmp_L**2 + tmp_M**2)
            - tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_dp_pi_dxyz
    )
    H_X2Y2_Y_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_M_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_M * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        - (
            tmp_M_dxyz * (1 + tmp_L**2 - tmp_M**2)
            + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_dp_pi_dxyz
    )
    H_X2Y2_Z_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        - (
            tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
            + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 1, H_X2Y2_X_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 2, H_X2Y2_Y_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 3, H_X2Y2_Z_dxyz)
    _sg(tmp_mask, 7, 1, H_X2Y2_X_dxyz)
    _sg(tmp_mask, 7, 2, H_X2Y2_Y_dxyz)
    _sg(tmp_mask, 7, 3, H_X2Y2_Z_dxyz)
    H_Z2_X_dxyz = -(
        (
            tmp_L_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_L * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        - 3**0.5 * (tmp_L_dxyz * tmp_N**2 + 2 * tmp_L * tmp_N * tmp_N_dxyz) * V_dp_pi
        - 3**0.5 * tmp_L * tmp_N**2 * V_dp_pi_dxyz
    )
    H_Z2_Y_dxyz = -(
        (
            tmp_M_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_M * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        - 3**0.5 * (tmp_M_dxyz * tmp_N**2 + 2 * tmp_M * tmp_N * tmp_N_dxyz) * V_dp_pi
        - 3**0.5 * tmp_M * tmp_N**2 * V_dp_pi_dxyz
    )
    H_Z2_Z_dxyz = -(
        (
            tmp_N_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_N * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        + 3**0.5
        * (
            tmp_N_dxyz * (tmp_L**2 + tmp_M**2)
            + 2 * tmp_N * (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 1, H_Z2_X_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 2, H_Z2_Y_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 3, H_Z2_Z_dxyz)
    _sg(tmp_mask, 8, 1, H_Z2_X_dxyz)
    _sg(tmp_mask, 8, 2, H_Z2_Y_dxyz)
    _sg(tmp_mask, 8, 3, H_Z2_Z_dxyz)

    ### d-d
    # Needs a d shell on both atoms, i.e. n_orb in {9, 16} on each side.
    tmp_mask = pair_mask_YY | pair_mask_YZ | pair_mask_ZY | pair_mask_ZZ
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    V_dd_sigma, V_dd_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "dd0", tmp_mask, "IJ"
    )
    V_dd_pi, V_dd_pi_dR = _get_val_dR(sel_IJ, sel_idx, tmp_dx, "dd1", tmp_mask, "IJ")
    V_dd_delta, V_dd_delta_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "dd2", tmp_mask, "IJ"
    )
    H_XY_XY = (
        3 * tmp_L**2 * tmp_M**2 * V_dd_sigma
        + (tmp_L**2 + tmp_M**2 - 4 * tmp_L**2 * tmp_M**2) * V_dd_pi
        + (tmp_N**2 + tmp_L**2 * tmp_M**2) * V_dd_delta
    )
    H_XY_YZ = (
        3 * tmp_L * tmp_M**2 * tmp_N * V_dd_sigma
        + tmp_L * tmp_N * (1 - 4 * tmp_M**2) * V_dd_pi
        + tmp_L * tmp_N * (tmp_M**2 - 1) * V_dd_delta
    )
    H_XY_ZX = (
        3 * tmp_L**2 * tmp_M * tmp_N * V_dd_sigma
        + tmp_M * tmp_N * (1 - 4 * tmp_L**2) * V_dd_pi
        + tmp_M * tmp_N * (tmp_L**2 - 1) * V_dd_delta
    )
    H_XY_X2Y2 = (
        1.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + 2 * tmp_L * tmp_M * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + 0.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_XY_Z2 = (
        (3**0.5) * tmp_L * tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        - (3**0.5) * 2 * tmp_L * tmp_M * tmp_N**2 * V_dd_pi
        + (3**0.5) * 0.5 * tmp_L * tmp_M * (1 + tmp_N**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 4, H_XY_XY)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 5, H_XY_YZ)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 6, H_XY_ZX)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 7, H_XY_X2Y2)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 8, H_XY_Z2)
    H_YZ_XY = (
        3 * tmp_M**2 * tmp_N * tmp_L * V_dd_sigma
        + tmp_L * tmp_N * (1 - 4 * tmp_M**2) * V_dd_pi
        + tmp_L * tmp_N * (tmp_M**2 - 1) * V_dd_delta
    )
    H_YZ_YZ = (
        3 * tmp_M**2 * tmp_N**2 * V_dd_sigma
        + (tmp_M**2 + tmp_N**2 - 4 * tmp_M**2 * tmp_N**2) * V_dd_pi
        + (tmp_L**2 + tmp_M**2 * tmp_N**2) * V_dd_delta
    )
    H_YZ_ZX = (
        3 * tmp_M * tmp_N**2 * tmp_L * V_dd_sigma
        + tmp_L * tmp_M * (1 - 4 * tmp_N**2) * V_dd_pi
        + tmp_L * tmp_M * (tmp_N**2 - 1) * V_dd_delta
    )
    H_YZ_X2Y2 = (
        1.5 * tmp_M * tmp_N * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        - tmp_M * tmp_N * (1 + 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        + tmp_M * tmp_N * (1 + 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_YZ_Z2 = (
        (3**0.5) * tmp_M * tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 4, H_YZ_XY)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 5, H_YZ_YZ)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 6, H_YZ_ZX)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 7, H_YZ_X2Y2)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 8, H_YZ_Z2)
    H_ZX_XY = (
        3 * tmp_L**2 * tmp_M * tmp_N * V_dd_sigma
        + tmp_M * tmp_N * (1 - 4 * tmp_L**2) * V_dd_pi
        + tmp_M * tmp_N * (tmp_L**2 - 1) * V_dd_delta
    )
    H_ZX_YZ = (
        3 * tmp_M * tmp_N**2 * tmp_L * V_dd_sigma
        + tmp_L * tmp_M * (1 - 4 * tmp_N**2) * V_dd_pi
        + tmp_L * tmp_M * (tmp_N**2 - 1) * V_dd_delta
    )
    H_ZX_ZX = (
        3 * tmp_N**2 * tmp_L**2 * V_dd_sigma
        + (tmp_N**2 + tmp_L**2 - 4 * tmp_N**2 * tmp_L**2) * V_dd_pi
        + (tmp_M**2 + tmp_N**2 * tmp_L**2) * V_dd_delta
    )
    H_ZX_X2Y2 = (
        1.5 * tmp_N * tmp_L * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + tmp_N * tmp_L * (1 - 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        - tmp_N * tmp_L * (1 - 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_ZX_Z2 = (
        (3**0.5) * tmp_N * tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 4, H_ZX_XY)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 5, H_ZX_YZ)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 6, H_ZX_ZX)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 7, H_ZX_X2Y2)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 8, H_ZX_Z2)
    H_X2Y2_XY = (
        1.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + 2 * tmp_L * tmp_M * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + 0.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_X2Y2_YZ = (
        1.5 * tmp_M * tmp_N * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        - tmp_M * tmp_N * (1 + 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        + tmp_M * tmp_N * (1 + 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_X2Y2_ZX = (
        1.5 * tmp_N * tmp_L * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + tmp_N * tmp_L * (1 - 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        - tmp_N * tmp_L * (1 - 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_X2Y2_X2Y2 = (
        0.75 * (tmp_L**2 - tmp_M**2) ** 2 * V_dd_sigma
        + (tmp_L**2 + tmp_M**2 - (tmp_L**2 - tmp_M**2) ** 2) * V_dd_pi
        + (tmp_N**2 + 0.25 * (tmp_L**2 - tmp_M**2) ** 2) * V_dd_delta
    )
    H_X2Y2_Z2 = (
        (3**0.5)
        * 0.5
        * (tmp_L**2 - tmp_M**2)
        * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
        * V_dd_sigma
        + (3**0.5) * tmp_N**2 * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + (3**0.5) * 0.25 * (1 + tmp_N**2) * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 4, H_X2Y2_XY)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 5, H_X2Y2_YZ)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 6, H_X2Y2_ZX)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 7, H_X2Y2_X2Y2)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 8, H_X2Y2_Z2)
    H_Z2_XY = (
        (3**0.5) * tmp_L * tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        - (3**0.5) * 2 * tmp_L * tmp_M * tmp_N**2 * V_dd_pi
        + (3**0.5) * 0.5 * tmp_L * tmp_M * (1 + tmp_N**2) * V_dd_delta
    )
    H_Z2_YZ = (
        (3**0.5) * tmp_M * tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H_Z2_ZX = (
        (3**0.5) * tmp_N * tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H_Z2_X2Y2 = (
        (3**0.5)
        * 0.5
        * (tmp_L**2 - tmp_M**2)
        * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
        * V_dd_sigma
        + (3**0.5) * tmp_N**2 * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + (3**0.5) * 0.25 * (1 + tmp_N**2) * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_Z2_Z2 = (
        (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) ** 2 * V_dd_sigma
        + 3 * tmp_N**2 * (tmp_L**2 + tmp_M**2) * V_dd_pi
        + 0.75 * (tmp_L**2 + tmp_M**2) ** 2 * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 4, H_Z2_XY)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 5, H_Z2_YZ)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 6, H_Z2_ZX)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 7, H_Z2_X2Y2)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 8, H_Z2_Z2)
    # d-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_dd_sigma_dxyz = V_dd_sigma_dR * tmp_dR_dxyz
    V_dd_pi_dxyz = V_dd_pi_dR * tmp_dR_dxyz
    V_dd_delta_dxyz = V_dd_delta_dR * tmp_dR_dxyz
    # t - time, m - minus, p - plus
    L_t_Ldx = tmp_L * tmp_L_dxyz
    M_t_Mdx = tmp_M * tmp_M_dxyz
    N_t_Ndx = tmp_N * tmp_N_dxyz
    L_t_M = tmp_L * tmp_M
    M_t_N = tmp_M * tmp_N
    N_t_L = tmp_N * tmp_L

    L2 = tmp_L**2
    M2 = tmp_M**2
    N2 = tmp_N**2

    H_XY_XY_dxyz = (
        3 * (2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx) * V_dd_sigma
        + 3 * L2 * M2 * V_dd_sigma_dxyz
        + ((2 * L_t_Ldx + 2 * M_t_Mdx) - 4 * (2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx))
        * V_dd_pi
        + (L2 + M2 - 4 * L2 * M2) * V_dd_pi_dxyz
        + (2 * N_t_Ndx + 2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx) * V_dd_delta
        + (N2 + L2 * M2) * V_dd_delta_dxyz
    )
    H_XY_YZ_dxyz = (
        3
        * (
            tmp_L_dxyz * M2 * tmp_N
            + tmp_L * 2 * M_t_Mdx * tmp_N
            + tmp_L * M2 * tmp_N_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_L * M2 * tmp_N * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (1 - 4 * M2) * V_dd_pi
        + tmp_L * tmp_N * (-8 * M_t_Mdx) * V_dd_pi
        + tmp_L * tmp_N * (1 - 4 * M2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (M2 - 1) * V_dd_delta
        + tmp_L * tmp_N * (2 * M_t_Mdx) * V_dd_delta
        + tmp_L * tmp_N * (M2 - 1) * V_dd_delta_dxyz
    )
    H_XY_ZX_dxyz = (
        3
        * (2 * L_t_Ldx * M_t_N + L2 * tmp_M_dxyz * tmp_N + L2 * tmp_M * tmp_N_dxyz)
        * V_dd_sigma
        + 3 * L2 * M_t_N * V_dd_sigma_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 - 4 * L2) * V_dd_pi
        + M_t_N * (-8 * L_t_Ldx) * V_dd_pi
        + M_t_N * (1 - 4 * L2) * V_dd_pi_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - 1) * V_dd_delta
        + M_t_N * (2 * L_t_Ldx) * V_dd_delta
        + M_t_N * (L2 - 1) * V_dd_delta_dxyz
    )
    H_XY_X2Y2_dxyz = (
        1.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * L_t_M * (L2 - M2) * V_dd_sigma_dxyz
        + 2
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (M2 - L2)
            + L_t_M * (2 * M_t_Mdx - 2 * L_t_Ldx)
        )
        * V_dd_pi
        + 2 * L_t_M * (M2 - L2) * V_dd_pi_dxyz
        + 0.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_delta
        + 0.5 * L_t_M * (L2 - M2) * V_dd_delta_dxyz
    )
    H_XY_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 0.5 * (L2 + M2))
            + L_t_M * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * L_t_M * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        - (3**0.5)
        * 2
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * N2 + L_t_M * 2 * N_t_Ndx)
        * V_dd_pi
        - (3**0.5) * 2 * L_t_M * N2 * V_dd_pi_dxyz
        + (3**0.5)
        * 0.5
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 + N2) + L_t_M * 2 * N_t_Ndx)
        * V_dd_delta
        + (3**0.5) * 0.5 * L_t_M * (1 + N2) * V_dd_delta_dxyz
    )
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 4, H_XY_XY_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 5, H_XY_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 6, H_XY_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 7, H_XY_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 8, H_XY_Z2_dxyz)
    _sg(tmp_mask, 4, 4, H_XY_XY_dxyz)
    _sg(tmp_mask, 4, 5, H_XY_YZ_dxyz)
    _sg(tmp_mask, 4, 6, H_XY_ZX_dxyz)
    _sg(tmp_mask, 4, 7, H_XY_X2Y2_dxyz)
    _sg(tmp_mask, 4, 8, H_XY_Z2_dxyz)
    H_YZ_XY_dxyz = (
        3
        * (2 * M_t_Mdx * N_t_L + M2 * tmp_N_dxyz * tmp_L + M2 * tmp_N * tmp_L_dxyz)
        * V_dd_sigma
        + 3 * M2 * N_t_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (1 - 4 * M2) * V_dd_pi
        + tmp_L * tmp_N * (-8 * M_t_Mdx) * V_dd_pi
        + tmp_L * tmp_N * (1 - 4 * M2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (M2 - 1) * V_dd_delta
        + tmp_L * tmp_N * (2 * M_t_Mdx) * V_dd_delta
        + tmp_L * tmp_N * (M2 - 1) * V_dd_delta_dxyz
    )
    H_YZ_YZ_dxyz = (
        3 * (2 * M_t_Mdx * N2 + M2 * 2 * N_t_Ndx) * V_dd_sigma
        + 3 * M2 * N2 * V_dd_sigma_dxyz
        + (2 * M_t_Mdx + 2 * N_t_Ndx - 8 * (M_t_Mdx * N2 + M2 * N_t_Ndx)) * V_dd_pi
        + (M2 + N2 - 4 * M2 * N2) * V_dd_pi_dxyz
        + (2 * L_t_Ldx + 2 * M_t_Mdx * N2 + M2 * 2 * N_t_Ndx) * V_dd_delta
        + (L2 + M2 * N2) * V_dd_delta_dxyz
    )
    H_YZ_ZX_dxyz = (
        3
        * (
            tmp_M_dxyz * N2 * tmp_L
            + tmp_M * 2 * N_t_Ndx * tmp_L
            + tmp_M * N2 * tmp_L_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_M * N2 * tmp_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 - 4 * N2) * V_dd_pi
        + L_t_M * (-8 * N_t_Ndx) * V_dd_pi
        + L_t_M * (1 - 4 * N2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 1) * V_dd_delta
        + L_t_M * (2 * N_t_Ndx) * V_dd_delta
        + L_t_M * (N2 - 1) * V_dd_delta_dxyz
    )
    H_YZ_X2Y2_dxyz = (
        1.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - M2)
            + M_t_N * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * M_t_N * (L2 - M2) * V_dd_sigma_dxyz
        + -(
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 2 * (L2 - M2))
            + M_t_N * (4 * L_t_Ldx - 4 * M_t_Mdx)
        )
        * V_dd_pi
        - M_t_N * (1 + 2 * (L2 - M2)) * V_dd_pi_dxyz
        + (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 0.5 * (L2 - M2))
            + M_t_N * (L_t_Ldx - M_t_Mdx)
        )
        * V_dd_delta
        + M_t_N * (1 + 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_YZ_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (N2 - 0.5 * (L2 + M2))
            + M_t_N * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * M_t_N * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2 - N2)
            + M_t_N * (2 * L_t_Ldx + 2 * M_t_Mdx - 2 * N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * M_t_N * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2)
            + M_t_N * (2 * L_t_Ldx + 2 * M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * M_t_N * (L2 + M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 4, H_YZ_XY_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 5, H_YZ_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 6, H_YZ_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 7, H_YZ_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + 8, H_YZ_Z2_dxyz)
    _sg(tmp_mask, 5, 4, H_YZ_XY_dxyz)
    _sg(tmp_mask, 5, 5, H_YZ_YZ_dxyz)
    _sg(tmp_mask, 5, 6, H_YZ_ZX_dxyz)
    _sg(tmp_mask, 5, 7, H_YZ_X2Y2_dxyz)
    _sg(tmp_mask, 5, 8, H_YZ_Z2_dxyz)
    H_ZX_XY_dxyz = (
        3
        * (2 * L_t_Ldx * M_t_N + L2 * tmp_M_dxyz * tmp_N + L2 * tmp_M * tmp_N_dxyz)
        * V_dd_sigma
        + 3 * L2 * M_t_N * V_dd_sigma_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 - 4 * L2) * V_dd_pi
        + M_t_N * (-8 * L_t_Ldx) * V_dd_pi
        + M_t_N * (1 - 4 * L2) * V_dd_pi_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - 1) * V_dd_delta
        + M_t_N * (2 * L_t_Ldx) * V_dd_delta
        + M_t_N * (L2 - 1) * V_dd_delta_dxyz
    )
    H_ZX_YZ_dxyz = (
        3
        * (
            tmp_M_dxyz * N2 * tmp_L
            + tmp_M * 2 * N_t_Ndx * tmp_L
            + tmp_M * N2 * tmp_L_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_M * N2 * tmp_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 - 4 * N2) * V_dd_pi
        + L_t_M * (-8 * N_t_Ndx) * V_dd_pi
        + L_t_M * (1 - 4 * N2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 1) * V_dd_delta
        + L_t_M * (2 * N_t_Ndx) * V_dd_delta
        + L_t_M * (N2 - 1) * V_dd_delta_dxyz
    )
    H_ZX_ZX_dxyz = (
        3 * (2 * N_t_Ndx * L2 + N2 * 2 * L_t_Ldx) * V_dd_sigma
        + 3 * N2 * L2 * V_dd_sigma_dxyz
        + (2 * N_t_Ndx + 2 * L_t_Ldx - 8 * (N_t_Ndx * L2 + N2 * L_t_Ldx)) * V_dd_pi
        + (N2 + L2 - 4 * N2 * L2) * V_dd_pi_dxyz
        + (2 * M_t_Mdx + 2 * N_t_Ndx * L2 + N2 * 2 * L_t_Ldx) * V_dd_delta
        + (M2 + N2 * L2) * V_dd_delta_dxyz
    )
    H_ZX_X2Y2_dxyz = (
        1.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 - M2)
            + N_t_L * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * N_t_L * (L2 - M2) * V_dd_sigma_dxyz
        + (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 2 * (L2 - M2))
            + N_t_L * (-4 * (L_t_Ldx - M_t_Mdx))
        )
        * V_dd_pi
        + N_t_L * (1 - 2 * (L2 - M2)) * V_dd_pi_dxyz
        + -(
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 0.5 * (L2 - M2))
            + N_t_L * (-L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - N_t_L * (1 - 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_ZX_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (N2 - 0.5 * (L2 + M2))
            + N_t_L * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * N_t_L * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2 - N2)
            + N_t_L * (2 * L_t_Ldx + 2 * M_t_Mdx - 2 * N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * N_t_L * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2)
            + N_t_L * (2 * L_t_Ldx + 2 * M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * N_t_L * (L2 + M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 4, H_ZX_XY_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 5, H_ZX_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 6, H_ZX_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 7, H_ZX_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + 8, H_ZX_Z2_dxyz)
    _sg(tmp_mask, 6, 4, H_ZX_XY_dxyz)
    _sg(tmp_mask, 6, 5, H_ZX_YZ_dxyz)
    _sg(tmp_mask, 6, 6, H_ZX_ZX_dxyz)
    _sg(tmp_mask, 6, 7, H_ZX_X2Y2_dxyz)
    _sg(tmp_mask, 6, 8, H_ZX_Z2_dxyz)
    H_X2Y2_XY_dxyz = (
        1.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * L_t_M * (L2 - M2) * V_dd_sigma_dxyz
        + 2
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (M2 - L2)
            + L_t_M * (2 * M_t_Mdx - 2 * L_t_Ldx)
        )
        * V_dd_pi
        + 2 * L_t_M * (M2 - L2) * V_dd_pi_dxyz
        + 0.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_delta
        + 0.5 * L_t_M * (L2 - M2) * V_dd_delta_dxyz
    )
    H_X2Y2_YZ_dxyz = (
        1.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - M2)
            + M_t_N * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * M_t_N * (L2 - M2) * V_dd_sigma_dxyz
        + -(
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 2 * (L2 - M2))
            + M_t_N * (4 * L_t_Ldx - 4 * M_t_Mdx)
        )
        * V_dd_pi
        - M_t_N * (1 + 2 * (L2 - M2)) * V_dd_pi_dxyz
        + (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 0.5 * (L2 - M2))
            + M_t_N * (L_t_Ldx - M_t_Mdx)
        )
        * V_dd_delta
        + M_t_N * (1 + 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_X2Y2_ZX_dxyz = (
        1.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 - M2)
            + N_t_L * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * N_t_L * (L2 - M2) * V_dd_sigma_dxyz
        + (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 2 * (L2 - M2))
            + N_t_L * (-4 * (L_t_Ldx - M_t_Mdx))
        )
        * V_dd_pi
        + N_t_L * (1 - 2 * (L2 - M2)) * V_dd_pi_dxyz
        + -(
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 0.5 * (L2 - M2))
            + N_t_L * (-L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - N_t_L * (1 - 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_X2Y2_X2Y2_dxyz = (
        0.75 * 4 * (L2 - M2) * (L_t_Ldx - M_t_Mdx) * V_dd_sigma
        + 0.75 * (L2 - M2) ** 2 * V_dd_sigma_dxyz
        + (2 * L_t_Ldx + 2 * M_t_Mdx - 4 * (L2 - M2) * (L_t_Ldx - M_t_Mdx)) * V_dd_pi
        + (L2 + M2 - (L2 - M2) ** 2) * V_dd_pi_dxyz
        + (2 * N_t_Ndx + (L2 - M2) * (L_t_Ldx - M_t_Mdx)) * V_dd_delta
        + (N2 + 0.25 * (L2 - M2) ** 2) * V_dd_delta_dxyz
    )
    H_X2Y2_Z2_dxyz = (
        (3**0.5)
        * 0.5
        * (
            2 * (L_t_Ldx - M_t_Mdx) * (N2 - 0.5 * (L2 + M2))
            + (L2 - M2) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * 0.5 * (L2 - M2) * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5) * (2 * N_t_Ndx * (M2 - L2) + N2 * 2 * (M_t_Mdx - L_t_Ldx)) * V_dd_pi
        + (3**0.5) * N2 * (M2 - L2) * V_dd_pi_dxyz
        + (3**0.5)
        * 0.25
        * (2 * N_t_Ndx * (L2 - M2) + (1 + N2) * 2 * (L_t_Ldx - M_t_Mdx))
        * V_dd_delta
        + (3**0.5) * 0.25 * (1 + N2) * (L2 - M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 4, H_X2Y2_XY_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 5, H_X2Y2_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 6, H_X2Y2_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 7, H_X2Y2_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 7) * HDIM + idx_col + 8, H_X2Y2_Z2_dxyz)
    _sg(tmp_mask, 7, 4, H_X2Y2_XY_dxyz)
    _sg(tmp_mask, 7, 5, H_X2Y2_YZ_dxyz)
    _sg(tmp_mask, 7, 6, H_X2Y2_ZX_dxyz)
    _sg(tmp_mask, 7, 7, H_X2Y2_X2Y2_dxyz)
    _sg(tmp_mask, 7, 8, H_X2Y2_Z2_dxyz)
    H_Z2_XY_dxyz = (
        (3**0.5)
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 0.5 * (L2 + M2))
            + L_t_M * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * L_t_M * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        - (3**0.5)
        * 2
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * N2 + L_t_M * 2 * N_t_Ndx)
        * V_dd_pi
        - (3**0.5) * 2 * L_t_M * N2 * V_dd_pi_dxyz
        + (3**0.5)
        * 0.5
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 + N2) + L_t_M * 2 * N_t_Ndx)
        * V_dd_delta
        + (3**0.5) * 0.5 * L_t_M * (1 + N2) * V_dd_delta_dxyz
    )
    H_Z2_YZ_dxyz = (
        (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (N2 - 0.5 * (L2 + M2))
            + M_t_N * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * M_t_N * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2 - N2)
            + M_t_N * 2 * (L_t_Ldx + M_t_Mdx - N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * M_t_N * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2)
            + M_t_N * 2 * (L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * M_t_N * (L2 + M2) * V_dd_delta_dxyz
    )
    H_Z2_ZX_dxyz = (
        (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (N2 - 0.5 * (L2 + M2))
            + N_t_L * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * N_t_L * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2 - N2)
            + N_t_L * 2 * (L_t_Ldx + M_t_Mdx - N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * N_t_L * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2)
            + N_t_L * 2 * (L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * N_t_L * (L2 + M2) * V_dd_delta_dxyz
    )
    H_Z2_X2Y2_dxyz = (
        (3**0.5)
        * 0.5
        * (
            2 * (L_t_Ldx - M_t_Mdx) * (N2 - 0.5 * (L2 + M2))
            + (L2 - M2) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * 0.5 * (L2 - M2) * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5) * (2 * N_t_Ndx * (M2 - L2) + N2 * 2 * (M_t_Mdx - L_t_Ldx)) * V_dd_pi
        + (3**0.5) * N2 * (M2 - L2) * V_dd_pi_dxyz
        + (3**0.5)
        * 0.25
        * (2 * N_t_Ndx * (L2 - M2) + (1 + N2) * 2 * (L_t_Ldx - M_t_Mdx))
        * V_dd_delta
        + (3**0.5) * 0.25 * (1 + N2) * (L2 - M2) * V_dd_delta_dxyz
    )
    H_Z2_Z2_dxyz = (
        2 * (N2 - 0.5 * (L2 + M2)) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx)) * V_dd_sigma
        + (N2 - 0.5 * (L2 + M2)) ** 2 * V_dd_sigma_dxyz
        + 3 * (2 * N_t_Ndx * (L2 + M2) + N2 * 2 * (L_t_Ldx + M_t_Mdx)) * V_dd_pi
        + 3 * N2 * (L2 + M2) * V_dd_pi_dxyz
        + 0.75 * 2 * (L2 + M2) * 2 * (L_t_Ldx + M_t_Mdx) * V_dd_delta
        + 0.75 * (L2 + M2) ** 2 * V_dd_delta_dxyz
    )
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 4, H_Z2_XY_dxyz)
    _sg(tmp_mask, 8, 4, H_Z2_XY_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 5, H_Z2_YZ_dxyz)
    _sg(tmp_mask, 8, 5, H_Z2_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 6, H_Z2_ZX_dxyz)
    _sg(tmp_mask, 8, 6, H_Z2_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 7, H_Z2_X2Y2_dxyz)
    _sg(tmp_mask, 8, 7, H_Z2_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + 8, H_Z2_Z2_dxyz)
    _sg(tmp_mask, 8, 8, H_Z2_Z2_dxyz)

    ### f blocks (AO offsets 9..15) ###
    # Angular factors come from the source-locked Takegahara tables; see
    # ``F_FORMULA_SOURCE`` and the transcription above.  Only the *values* are
    # implemented in this phase: no derivative contribution is written, so the
    # f entries of dH0/dS stay exactly zero and every derivative consumer must
    # guard on ``F_ANGULAR_DERIVATIVES_AVAILABLE`` (see FDerivativeUnsupportedError).
    if _f_present:

        def _f_write_block(mask, bases, direction, parity, low_offset, angular, f_rows):
            """Accumulate one f-containing AO block into ``H0``.

            ``angular`` returns ``(n_channel, n_low, 7, P)`` where ``n_low``
            indexes the lower shell (s/p/d/f) and the third axis indexes the f
            shell in ``STRUCTURE_F_AO_ORDER``.  ``f_rows`` selects which side of
            the AO block the f shell occupies: ``False`` writes
            ``H0[low_offset + a, 9 + b]`` (lower shell on atom I), ``True``
            writes ``H0[9 + b, low_offset + a]`` (f shell on atom I).

            ``parity`` is ``(-1) ** (l_low + 3)``: reversing the bond direction
            flips the sign of every odd-degree angular polynomial, which is the
            same convention the existing s-p / d-s blocks already use.
            """
            if not bool(mask.any()):
                return
            i0 = H_INDEX_START[neighbor_I[mask]]
            j0 = H_INDEX_START[neighbor_J[mask]]
            sel_type = JI_pair_type[mask] if direction == "JI" else IJ_pair_type[mask]
            sel_idx = idx[mask]
            sel_dx = dx[mask]
            radial = [
                _get_val_dR(sel_type, sel_idx, sel_dx, base, mask, direction)[0]
                for base in bases
            ]
            coeff = angular(L[mask], M[mask], N[mask])
            n_low = coeff.shape[1]
            for a in range(n_low):
                for b in range(7):
                    entry = coeff[0, a, b] * radial[0]
                    for k in range(1, len(radial)):
                        entry = entry + coeff[k, a, b] * radial[k]
                    if parity < 0.0:
                        entry = -entry
                    if f_rows:
                        flat = (i0 + 9 + b) * HDIM + j0 + low_offset + a
                    else:
                        flat = (i0 + low_offset + a) * HDIM + j0 + 9 + b
                    H0.index_add_(0, flat, entry)

        # s-f / f-s: parity (-1) ** (0 + 3) = -1
        _f_write_block(
            pair_mask_HZ | pair_mask_XZ | pair_mask_YZ | pair_mask_ZZ,
            ("sf0",),
            "IJ",
            1.0,
            0,
            f_angular_sf,
            False,
        )
        _f_write_block(
            pair_mask_ZH | pair_mask_ZX | pair_mask_ZY | pair_mask_ZZ,
            ("sf0",),
            "JI",
            -1.0,
            0,
            f_angular_sf,
            True,
        )
        # p-f / f-p: parity (-1) ** (1 + 3) = +1
        _f_write_block(
            pair_mask_XZ | pair_mask_YZ | pair_mask_ZZ,
            ("pf0", "pf1"),
            "IJ",
            1.0,
            1,
            f_angular_pf,
            False,
        )
        _f_write_block(
            pair_mask_ZX | pair_mask_ZY | pair_mask_ZZ,
            ("pf0", "pf1"),
            "JI",
            1.0,
            1,
            f_angular_pf,
            True,
        )
        # d-f / f-d: parity (-1) ** (2 + 3) = -1
        _f_write_block(
            pair_mask_YZ | pair_mask_ZZ,
            ("df0", "df1", "df2"),
            "IJ",
            1.0,
            4,
            f_angular_df,
            False,
        )
        _f_write_block(
            pair_mask_ZY | pair_mask_ZZ,
            ("df0", "df1", "df2"),
            "JI",
            -1.0,
            4,
            f_angular_df,
            True,
        )
        # f-f: both AO sides sit on the same pair, and the reverse-direction
        # block is supplied by the (J, I) entry of the neighbor list, exactly as
        # for d-d.
        _f_write_block(
            pair_mask_ZZ,
            ("ff0", "ff1", "ff2", "ff3"),
            "IJ",
            1.0,
            9,
            f_angular_ff,
            False,
        )

    """
    $ - from table
    O - derived via permutation

    S_XY = (3**0.5)*L*M*V_sd_sigma                                                                                                                         # $
    S_YZ = (3**0.5)*M*N*V_sd_sigma                                                                                                                         # O 
    S_ZX = (3**0.5)*N*L*V_sd_sigma                                                                                                                         # O
    S_X2Y2 = 0.5*(3**0.5)*(L**2 - M**2)*V_sd_sigma                                                                                                         # $
    S_Z2 = (N**2-0.5*(L**2 + M**2))*V_sd_sigma                                                                                                             # $

    X_XY = (3**0.5)*L**2*M*V_pd_sigma + M*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_YZ = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # $
    X_ZX = (3**0.5)*L**2*N*V_pd_sigma + N*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_X2Y2 = 0.5*(3**0.5)*L*(L**2 - M**2)*V_pd_sigma + L*(1 - L**2 + M**2)*V_pd_pi                                                                         # $
    X_Z2 = L*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*L*N**2*V_pd_pi                                                                                 # $

    Y_XY = (3**0.5)*M**2*L*V_pd_sigma + L*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_YZ = (3**0.5)*M**2*N*V_pd_sigma + N*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_ZX = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Y_X2Y2 = 0.5*(3**0.5)*M*(L**2 - M**2)*V_pd_sigma - M*(1 + L**2 - M**2)*V_pd_pi                                                                         # $
    Y_Z2 = M*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*M*N**2*V_pd_pi                                                                                 # $

    Z_XY = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Z_YZ = (3**0.5)*N**2*M*V_pd_sigma + M*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_ZX = (3**0.5)*N**2*L*V_pd_sigma + L*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_X2Y2 = 0.5*(3**0.5)*N*(L**2 - M**2)*V_pd_sigma - N*(L**2 - M**2)*V_pd_pi                                                                             # $
    Z_Z2 = N*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma + 3**0.5*N*(L**2 + M**2)*V_pd_pi                                                                        # $

    XY_S =  (3**0.5)*L*M*V_ds_sigma                                                                                                                        # O same as  S_XY
    YZ_S = (3**0.5)*M*N*V_ds_sigma                                                                                                                         # O same as  S_YZ
    ZX_S =  (3**0.5)*N*L*V_ds_sigma                                                                                                                        # O same as  S_ZX
    X2Y2_S =   0.5*(3**0.5)*(L**2 - M**2)*V_ds_sigma                                                                                                       # O same as  S_X2Y2
    Z2_S =  (N**2-0.5*(L**2 + M**2))*V_ds_sigma                                                                                                            # O same as  S_Z2

    XY_X = -((3**0.5)*L**2*M*V_dp_sigma + M*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_XY
    XY_Y = -((3**0.5)*M**2*L*V_dp_sigma + L*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_XY
    XY_Z = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Z_XY

    YZ_X = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -X_YZ
    YZ_Y = -((3**0.5)*M**2*N*V_dp_sigma + N*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_YZ
    YZ_Z = -((3**0.5)*N**2*M*V_dp_sigma + M*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_YZ

    ZX_X = -((3**0.5)*L**2*N*V_dp_sigma + N*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_ZX
    ZX_Y = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Y_ZX
    ZX_Z = -((3**0.5)*N**2*L*V_dp_sigma + L*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_ZX    

    X2Y2_X = -(0.5*(3**0.5)*L*(L**2 - M**2)*V_dp_sigma + L*(1 - L**2 + M**2)*V_dp_pi)                                                                      # O same as -X_X2Y2
    X2Y2_Y = -(0.5*(3**0.5)*M*(L**2 - M**2)*V_dp_sigma - M*(1 + L**2 - M**2)*V_dp_pi)                                                                      # O same as -Y_X2Y2       
    X2Y2_Z = -(0.5*(3**0.5)*N*(L**2 - M**2)*V_dp_sigma - N*(L**2 - M**2)*V_dp_pi)                                                                          # O same as -Z_X2Y2

    Z2_X = -(L*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*L*N**2*V_dp_pi)                                                                              # O same as -X_Z2
    Z2_Y = -(M*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*M*N**2*V_dp_pi)                                                                              # O same as -Y_Z2
    Z2_Z = -(N*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma + 3**0.5*N*(L**2 + M**2)*V_dp_pi)                                                                     # O same as -Z_Z2


    XY_XY = 3*L**2*M**2*V_dd_sigma + (L**2 + M**2 - 4*L**2*M**2)*V_dd_pi + (N**2 + L**2*M**2)*V_dd_delta                                                   # $
    XY_YZ = 3*L*M**2*N*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # $
    XY_ZX = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # $
    XY_X2Y2 = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # $
    XY_Z2 = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # $

    YZ_XY = 3*M**2*N*L*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # O
    YZ_YZ = 3*M**2*N**2*V_dd_sigma + (M**2 + N**2 - 4*M**2*N**2)*V_dd_pi + (L**2 + M**2*N**2)*V_dd_delta                                                   # O
    YZ_ZX = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O
    YZ_X2Y2 = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # $
    YZ_Z2 = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # $

    ZX_XY = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # O same as XY_ZX
    ZX_YZ = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O same as YZ_ZX
    ZX_ZX = 3*N**2*L**2*V_dd_sigma + (N**2 + L**2 - 4*N**2*L**2)*V_dd_pi + (M**2 + N**2*L**2)*V_dd_delta                                                   # $O
    ZX_X2Y2 = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # $
    ZX_Z2 = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # $

    X2Y2_XY = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # O same as  XY_X2Y2
    X2Y2_YZ = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  YZ_X2Y2
    X2Y2_ZX = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  ZX_X2Y2
    X2Y2_X2Y2 = 0.75*(L**2 - M**2)**2*V_dd_sigma + (L**2 + M**2 - (L**2 - M**2)**2)*V_dd_pi + (N**2 + 0.25*(L**2 - M**2)**2)*V_dd_delta                    # $
    X2Y2_Z2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # $

    Z2_XY = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # O same as  XY_Z2
    Z2_YZ = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # O same as  YZ_Z2
    Z2_ZX = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # O same as  ZX_Z2
    Z2_X2Y2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # O same as X2Y2_Z2
    Z2_Z2 = (N**2 - 0.5*(L**2 + M**2))**2*V_dd_sigma + 3*N**2*(L**2 + M**2)*V_dd_pi + 0.75*(L**2 + M**2)**2*V_dd_delta                                     # $


    """
    if pair_grad is not None:
        return H0, dH0, pair_grad
    return H0, dH0


def Slater_Koster_Pair_SKF_vectorized_batch(
    batch_size: int,
    HDIM: int,
    dR_dxyz: torch.Tensor,
    L: torch.Tensor,
    M: torch.Tensor,
    N: torch.Tensor,
    L_dxyz: torch.Tensor,
    M_dxyz: torch.Tensor,
    N_dxyz: torch.Tensor,
    pair_mask_HH: torch.Tensor,
    pair_mask_HX: torch.Tensor,
    pair_mask_XH: torch.Tensor,
    pair_mask_XX: torch.Tensor,
    pair_mask_HY: torch.Tensor,
    pair_mask_XY: torch.Tensor,
    pair_mask_YH: torch.Tensor,
    pair_mask_YX: torch.Tensor,
    pair_mask_YY: torch.Tensor,
    dx: torch.Tensor,
    idx: torch.Tensor,
    IJ_pair_type: torch.Tensor,
    JI_pair_type: torch.Tensor,
    coeffs_tensor: torch.Tensor,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    safe_I,
    safe_J,
    valid_pairs,
    H_INDEX_START: torch.Tensor,
    SH_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the Slater–Koster pair block (flattened) and its Cartesian derivatives
    using vectorized cubic-spline SKF coefficients (s, p, d orbitals).

    This routine assembles the AO block H0 for all requested pairs and its
    derivatives dH0 = dH0/d[x,y,z] by evaluating the spline-based SK integrals
    and applying the standard angular (direction cosine) factors. All writes are
    done with in-place index_add_ to allow accumulation across overlapping masks.

    Arguments
    ----------
    HDIM : int
        Per-atom AO block dimension used to index into the flattened block.
        Typical values: 1 (s), 4 (sp), 9 (spd). Must be consistent with the
        largest orbital shell present in the active masks; e.g., if any d-*
        masks are True, HDIM must be >= 9.

    dR_dxyz : torch.Tensor
        Derivatives of pair distances with respect to Cartesian components.
        Shape (3, num_pairs), dtype float, device consistent with L/M/N.
        Row 0/1/2 correspond to dR/dx, dR/dy, dR/dz.

    L, M, N : torch.Tensor
        Direction cosines for each pair. Shape (num_pairs,), dtype float.

    L_dxyz, M_dxyz, N_dxyz : torch.Tensor
        Derivatives of direction cosines. Shape (3, num_pairs), dtype float.

    pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
    pair_mask_HY, pair_mask_XY, pair_mask_YH, pair_mask_YX, pair_mask_YY : torch.BoolTensor
        Boolean masks (shape (num_pairs,)) selecting pair classes:
        - H: hydrogen-like (s-only)
        - X: sp atom (s + p)
        - Y: spd atom (s + p + d)
        The two letters indicate (left atom, right atom), e.g. HX = H–X, YX = Y–X.
        Masks can be combined (ORed) when contributions are shared (e.g. HX and XX
        both use s–p).

    dx : torch.Tensor
        Radial offset used in spline evaluation inside the selected interval for
        each pair (same meaning as local distance minus the knot position).
        Shape (num_pairs,), dtype float.

    idx : torch.LongTensor
        Spline interval index for each pair (selects which cubic to evaluate).
        Shape (num_pairs,), dtype long.

    IJ_pair_type, JI_pair_type : torch.LongTensor
        Integer pair-type indices used to select the proper row in coeffs_tensor
        for the I→J and J→I directions, respectively (handles sign conventions
        for s–p and p–s, etc.). Shape (num_pairs,), dtype long.

    coeffs_tensor : torch.Tensor
        Pre-tabulated cubic-spline coefficients for all SK channels.
        Indexed as coeffs_tensor[pair_type, interval_idx, channel, 0..3],
        where the last axis stores a0..a3 of the cubic a0 + a1*dx + a2*dx^2 + a3*dx^3.
        The channel axis follows the canonical 40-entry extended-SKF order in
        ``_bond_integral._CHANNELS`` (20 named H channels, then the 20 matching
        S channels), resolved by name through :data:`SK_CHANNEL_INDEX`.

        Expected shape: (n_pair_types, n_intervals, 40, 4).

    neighbor_I, neighbor_J : torch.LongTensor
        Atom indices (per pair) used to compute the flattened AO indices.
        Shape (num_pairs,), dtype long.

    H_INDEX_START : torch.LongTensor
        For each atom index k, H_INDEX_START[k] is the starting AO offset of atom k
        within the per-atom block. Used to place pair contributions into the
        flattened [HDIM x HDIM] block. Shape (num_atoms,), dtype long.

    SH_shift : int
        Selects the Hamiltonian (``0``) or overlap (``1``) half of the canonical
        channel list by choosing the ``"H"`` or ``"S"`` channel-name prefix.

    stress_weight : torch.Tensor or None
        If not None, a ``(HDIM, HDIM)`` density-weight matrix (e.g. band weight
        or Pulay+SCC weight).  When provided, an additional per-pair weighted
        gradient ``pair_grad (P, 3)`` is accumulated on-the-fly:
        ``pair_grad[p, c] += W[i0+a, j0+b] * d(H_ab^p)/dR_c`` for every AO
        pair (a,b) of pair p.  This avoids the need to store a large per-pair
        derivative tensor and eliminates redundant spline re-evaluation in
        the stress computation.  Requires ``i0_stress`` and ``j0_stress``.

    i0_stress, j0_stress : torch.LongTensor or None
        Pre-computed AO offsets ``H_INDEX_START[neighbor_I]`` and
        ``H_INDEX_START[neighbor_J]`` for each pair. Required when
        ``stress_weight`` is not None.

    Returns
    -------
    H0 : torch.Tensor
        Updated Hamiltonian matrix elements tensor with new values for the pairs processed.

    dH0 : torch.Tensor
        Derivatives of the Hamiltonian matrix elements with respect to Cartesian coordinates.
        Shape: (3, HDIM * HDIM)

    pair_grad : torch.Tensor  *(only when stress_weight is not None)*
        Per-pair weighted gradient, shape ``(P, 3)``.  Returned as the third
        element of the tuple.

    Notes
    -----
    - The function uses vectorized bond integral evaluations for improved computational efficiency.
    - The calculation covers both overlap and Hamiltonian matrix elements for s and p orbitals.
    - Direction cosine derivatives and bond integral derivatives are used to compute the gradients.
    - Periodic boundary conditions or lattice considerations are assumed handled externally.
    """
    # %%% Standard Slater-Koster sp-parameterization for an atomic block between a pair of atoms
    # %%% IDim, JDim: dimensions of the output block, e.g. 1 x 4 for H-O or 4 x 4 for O-O, or 4 x 1 for O-H
    # %%% Ra, Rb: are the vectors of the positions of the two atoms
    # %%% Type_pair(1 or 2): Character of the type of each atom in the pair, e.g. 'H' for hydrogen of 'O' for oxygen
    # %%% fss_sigma, ... , fpp_pi: paramters for the bond integrals
    # %%% diagonal(1 or 2): atomic energies Es and Ep or diagonal elements of the overlap i.e. diagonal = 1

    H0 = torch.zeros(
        (batch_size * HDIM * HDIM), dtype=dR_dxyz.dtype, device=dR_dxyz.device
    )
    dH0 = torch.zeros(3, batch_size * HDIM * HDIM, dtype=H0.dtype, device=H0.device)
    nn_mask_IJ = IJ_pair_type != -1

    # H-H
    coeffs_selected = coeffs_tensor[
        IJ_pair_type[nn_mask_IJ], idx, sk_channel_index(sk_channel_name('ss0', SH_shift))
    ]

    HSSS_all = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * dx
        + coeffs_selected[:, 2] * dx**2
        + coeffs_selected[:, 3] * dx**3
    )

    B = batch_size

    # neighbor_I, neighbor_J: (B, Npairs) with -1 padding
    valid_I = neighbor_I >= 0
    valid_J = neighbor_J >= 0
    valid_pair_mask = valid_I & valid_J  # (B, Npairs)

    # Safe gather: replace -1 by 0 then zero out later
    safe_neighbor_I = neighbor_I.clone()
    safe_neighbor_J = neighbor_J.clone()
    safe_neighbor_I[~valid_I] = 0
    safe_neighbor_J[~valid_J] = 0

    # Map atom indices to their first AO
    rows_all = H_INDEX_START.gather(1, safe_neighbor_I)  # (B,Npairs)
    cols_all = H_INDEX_START.gather(1, safe_neighbor_J)  # (B,Npairs)

    # Mark invalid
    rows_all[~valid_I] = -1
    cols_all[~valid_J] = -1

    # Keep only valid pairs for flattened indexing
    rows = rows_all[valid_pair_mask]  # (Nvalid,)
    cols = cols_all[valid_pair_mask]  # (Nvalid,)

    # Build batch index for each kept pair
    # Compute batch ids from original mask
    batch_ids = torch.arange(B, device=rows_all.device).unsqueeze(1).expand_as(rows_all)
    batch_ids = batch_ids[valid_pair_mask]  # (Nvalid,)

    # Base offset per batch block
    batch_block_offset = batch_ids * (HDIM * HDIM)

    lin_in_batch = rows * HDIM + cols  # (Nvalid,)
    indices = lin_in_batch + batch_block_offset  # (Nvalid,)
    H0.index_add_(0, indices, HSSS_all)
    HSSS_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * dx
        + 3 * coeffs_selected[:, 3] * dx**2
    )

    ######### dH/dx
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0.index_add_(1, indices, HSSS_dxyz)
    #########

    # H-X
    ###### HSPS_all
    tmp_mask = (
        pair_mask_HX
        | pair_mask_XX
        | pair_mask_HY
        | pair_mask_YY
        | pair_mask_XY
        | pair_mask_YX
    )
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('sp0', SH_shift))
    ]
    HSPS_all = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * dx[tmp_mask[valid_pairs]]
        + coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]] ** 2
        + coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 3
    )
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    H0.index_add_(
        0,
        idx_row * HDIM + idx_col + 1 + batch_block_offset,
        L[tmp_mask[valid_pairs]] * HSPS_all,
    )
    H0.index_add_(
        0,
        idx_row * HDIM + idx_col + 2 + batch_block_offset,
        M[tmp_mask[valid_pairs]] * HSPS_all,
    )
    H0.index_add_(
        0,
        idx_row * HDIM + idx_col + 3 + batch_block_offset,
        N[tmp_mask[valid_pairs]] * HSPS_all,
    )

    ######### dH/dx
    HSPS_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]]
        + 3 * coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 2
    )
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, tmp_mask[valid_pairs]]

    dH0.index_add_(
        1,
        idx_row * HDIM + idx_col + 1 + batch_block_offset,
        L[tmp_mask[valid_pairs]] * HSPS_dxyz
        + L_dxyz[:, tmp_mask[valid_pairs]] * HSPS_all,
    )
    dH0.index_add_(
        1,
        idx_row * HDIM + idx_col + 2 + batch_block_offset,
        M[tmp_mask[valid_pairs]] * HSPS_dxyz
        + M_dxyz[:, tmp_mask[valid_pairs]] * HSPS_all,
    )
    dH0.index_add_(
        1,
        idx_row * HDIM + idx_col + 3 + batch_block_offset,
        N[tmp_mask[valid_pairs]] * HSPS_dxyz
        + N_dxyz[:, tmp_mask[valid_pairs]] * HSPS_all,
    )
    #########

    ### HPSS_all ###
    tmp_mask = (
        pair_mask_XH
        | pair_mask_XX
        | pair_mask_YH
        | pair_mask_YY
        | pair_mask_XY
        | pair_mask_YX
    )
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('sp0', SH_shift))
    ]
    HPSS_all = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * dx[tmp_mask[valid_pairs]]
        + coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]] ** 2
        + coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 3
    )
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    H0.index_add_(
        0,
        (idx_row + 1) * HDIM + idx_col + batch_block_offset,
        -L[tmp_mask[valid_pairs]] * HPSS_all,
    )
    H0.index_add_(
        0,
        (idx_row + 2) * HDIM + idx_col + batch_block_offset,
        -M[tmp_mask[valid_pairs]] * HPSS_all,
    )
    H0.index_add_(
        0,
        (idx_row + 3) * HDIM + idx_col + batch_block_offset,
        -N[tmp_mask[valid_pairs]] * HPSS_all,
    )

    ######### dH/dx
    HPSS_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]]
        + 3 * coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 2
    )
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, tmp_mask[valid_pairs]]

    dH0.index_add_(
        1,
        (idx_row + 1) * HDIM + idx_col + batch_block_offset,
        -L[tmp_mask[valid_pairs]] * HPSS_dxyz
        - L_dxyz[:, tmp_mask[valid_pairs]] * HPSS_all,
    )
    dH0.index_add_(
        1,
        (idx_row + 2) * HDIM + idx_col + batch_block_offset,
        -M[tmp_mask[valid_pairs]] * HPSS_dxyz
        - M_dxyz[:, tmp_mask[valid_pairs]] * HPSS_all,
    )
    dH0.index_add_(
        1,
        (idx_row + 3) * HDIM + idx_col + batch_block_offset,
        -N[tmp_mask[valid_pairs]] * HPSS_dxyz
        - N_dxyz[:, tmp_mask[valid_pairs]] * HPSS_all,
    )
    #########

    # X-X
    tmp_mask = pair_mask_XX | pair_mask_YY | pair_mask_XY | pair_mask_YX
    L_XX = L[tmp_mask[valid_pairs]]
    M_XX = M[tmp_mask[valid_pairs]]
    N_XX = N[tmp_mask[valid_pairs]]
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pp0', SH_shift))
    ]
    HPPS = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * dx[tmp_mask[valid_pairs]]
        + coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]] ** 2
        + coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 3
    )
    HPPS_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]]
        + 3 * coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 2
    )
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pp1', SH_shift))
    ]
    HPPP = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * dx[tmp_mask[valid_pairs]]
        + coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]] ** 2
        + coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 3
    )
    HPPP_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * dx[tmp_mask[valid_pairs]]
        + 3 * coeffs_selected[:, 3] * dx[tmp_mask[valid_pairs]] ** 2
    )
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)

    PPSMPP = HPPS - HPPP
    PXPX = HPPP + L_XX * L_XX * PPSMPP
    PXPY = L_XX * M_XX * PPSMPP
    PXPZ = L_XX * N_XX * PPSMPP
    PYPX = M_XX * L_XX * PPSMPP
    PYPY = HPPP + M_XX * M_XX * PPSMPP
    PYPZ = M_XX * N_XX * PPSMPP
    PZPX = N_XX * L_XX * PPSMPP
    PZPY = N_XX * M_XX * PPSMPP
    PZPZ = HPPP + N_XX * N_XX * PPSMPP

    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 1 + batch_block_offset, PXPX)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 2 + batch_block_offset, PXPY)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 3 + batch_block_offset, PXPZ)
    ####
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 1 + batch_block_offset, PYPX)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 2 + batch_block_offset, PYPY)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 3 + batch_block_offset, PYPZ)
    ####
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 1 + batch_block_offset, PZPX)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 2 + batch_block_offset, PZPY)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 3 + batch_block_offset, PZPZ)

    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, tmp_mask[valid_pairs]]
    L_dxyz_XX = L_dxyz[:, tmp_mask[valid_pairs]]
    M_dxyz_XX = M_dxyz[:, tmp_mask[valid_pairs]]
    N_dxyz_XX = N_dxyz[:, tmp_mask[valid_pairs]]

    HPPS_dxyz = HPPS_dR * dR_dxyz_XX
    HPPP_dxyz = HPPP_dR * dR_dxyz_XX
    PPSMPP_dxyz = HPPS_dxyz - HPPP_dxyz
    PXPX_dxyz = HPPP_dxyz + (L_XX**2) * PPSMPP_dxyz + 2 * L_XX * L_dxyz_XX * PPSMPP
    PXPY_dxyz = (
        L_XX * M_XX * PPSMPP_dxyz
        + L_dxyz_XX * M_XX * PPSMPP
        + L_XX * M_dxyz_XX * PPSMPP
    )
    PXPZ_dxyz = (
        L_XX * N_XX * PPSMPP_dxyz
        + L_dxyz_XX * N_XX * PPSMPP
        + L_XX * N_dxyz_XX * PPSMPP
    )
    PYPX_dxyz = (
        M_XX * L_XX * PPSMPP_dxyz
        + M_XX * L_dxyz_XX * PPSMPP
        + M_dxyz_XX * L_XX * PPSMPP
    )
    PYPY_dxyz = HPPP_dxyz + (M_XX**2) * PPSMPP_dxyz + 2 * M_XX * M_dxyz_XX * PPSMPP
    PYPZ_dxyz = (
        M_XX * N_XX * PPSMPP_dxyz
        + M_dxyz_XX * N_XX * PPSMPP
        + M_XX * N_dxyz_XX * PPSMPP
    )
    PZPX_dxyz = (
        N_XX * L_XX * PPSMPP_dxyz
        + N_XX * L_dxyz_XX * PPSMPP
        + N_dxyz_XX * L_XX * PPSMPP
    )
    PZPY_dxyz = (
        N_XX * M_XX * PPSMPP_dxyz
        + N_XX * M_dxyz_XX * PPSMPP
        + N_dxyz_XX * M_XX * PPSMPP
    )
    PZPZ_dxyz = HPPP_dxyz + (N_XX**2) * PPSMPP_dxyz + 2 * N_XX * N_dxyz_XX * PPSMPP

    ####
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 1 + batch_block_offset, PXPX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 2 + batch_block_offset, PXPY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 3 + batch_block_offset, PXPZ_dxyz
    )
    ####
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 1 + batch_block_offset, PYPX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 2 + batch_block_offset, PYPY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 3 + batch_block_offset, PYPZ_dxyz
    )
    ####
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 1 + batch_block_offset, PZPX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 2 + batch_block_offset, PZPY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 3 + batch_block_offset, PZPZ_dxyz
    )
    #########

    ### s-d
    tmp_mask = pair_mask_HY | pair_mask_XY | pair_mask_YY
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    tmp_dx = dx[tmp_mask[valid_pairs]]
    tmp_L = L[tmp_mask[valid_pairs]]
    tmp_M = M[tmp_mask[valid_pairs]]
    tmp_N = N[tmp_mask[valid_pairs]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('sd0', SH_shift))
    ]
    V_sd_sigma = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    H_S_XY = (3**0.5) * tmp_L * tmp_M * V_sd_sigma
    H_S_YZ = (3**0.5) * tmp_M * tmp_N * V_sd_sigma
    H_S_ZX = (3**0.5) * tmp_N * tmp_L * V_sd_sigma
    H_S_X2Y2 = 0.5 * (3**0.5) * (tmp_L**2 - tmp_M**2) * V_sd_sigma
    H_S_Z2 = (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_sd_sigma
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 4 + batch_block_offset, H_S_XY)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 5 + batch_block_offset, H_S_YZ)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 6 + batch_block_offset, H_S_ZX)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 7 + batch_block_offset, H_S_X2Y2)
    H0.index_add_(0, (idx_row) * HDIM + idx_col + 8 + batch_block_offset, H_S_Z2)
    # s-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask[valid_pairs]]
    tmp_M_dxyz = M_dxyz[:, tmp_mask[valid_pairs]]
    tmp_N_dxyz = N_dxyz[:, tmp_mask[valid_pairs]]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask[valid_pairs]]
    V_sd_sigma_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    V_sd_sigma_dxyz = V_sd_sigma_dR * tmp_dR_dxyz
    H_S_XY_dxyz = (3**0.5) * (
        tmp_L_dxyz * tmp_M * V_sd_sigma
        + tmp_L * tmp_M_dxyz * V_sd_sigma
        + tmp_L * tmp_M * V_sd_sigma_dxyz
    )
    H_S_YZ_dxyz = (3**0.5) * (
        tmp_M_dxyz * tmp_N * V_sd_sigma
        + tmp_M * tmp_N_dxyz * V_sd_sigma
        + tmp_M * tmp_N * V_sd_sigma_dxyz
    )
    H_S_ZX_dxyz = (3**0.5) * (
        tmp_N_dxyz * tmp_L * V_sd_sigma
        + tmp_N * tmp_L_dxyz * V_sd_sigma
        + tmp_N * tmp_L * V_sd_sigma_dxyz
    )
    H_S_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz) * V_sd_sigma
            + (tmp_L**2 - tmp_M**2) * V_sd_sigma_dxyz
        )
    )
    H_S_Z2_dxyz = (
        2 * tmp_N * tmp_N_dxyz - 0.5 * (2 * tmp_L * tmp_L_dxyz + 2 * tmp_M * tmp_M_dxyz)
    ) * V_sd_sigma + (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_sd_sigma_dxyz
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 4 + batch_block_offset, H_S_XY_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 5 + batch_block_offset, H_S_YZ_dxyz)
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 6 + batch_block_offset, H_S_ZX_dxyz)
    dH0.index_add_(
        1, (idx_row) * HDIM + idx_col + 7 + batch_block_offset, H_S_X2Y2_dxyz
    )
    dH0.index_add_(1, (idx_row) * HDIM + idx_col + 8 + batch_block_offset, H_S_Z2_dxyz)

    ### p-d
    tmp_mask = pair_mask_XY | pair_mask_YY
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    tmp_dx = dx[tmp_mask[valid_pairs]]
    tmp_L = L[tmp_mask[valid_pairs]]
    tmp_M = M[tmp_mask[valid_pairs]]
    tmp_N = N[tmp_mask[valid_pairs]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pd0', SH_shift))
    ]
    V_pd_sigma = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_pd_sigma_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pd1', SH_shift))
    ]
    V_pd_pi = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_pd_pi_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    H_X_XY = (3**0.5) * tmp_L**2 * tmp_M * V_pd_sigma + tmp_M * (
        1 - 2 * tmp_L**2
    ) * V_pd_pi
    H_X_YZ = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_X_ZX = (3**0.5) * tmp_L**2 * tmp_N * V_pd_sigma + tmp_N * (
        1 - 2 * tmp_L**2
    ) * V_pd_pi
    H_X_X2Y2 = (
        0.5 * (3**0.5) * tmp_L * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_pd_pi
    )
    H_X_Z2 = (
        tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        - 3**0.5 * tmp_L * tmp_N**2 * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 4 + batch_block_offset, H_X_XY)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 5 + batch_block_offset, H_X_YZ)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 6 + batch_block_offset, H_X_ZX)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 7 + batch_block_offset, H_X_X2Y2)
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col + 8 + batch_block_offset, H_X_Z2)
    H_Y_XY = (3**0.5) * tmp_M**2 * tmp_L * V_pd_sigma + tmp_L * (
        1 - 2 * tmp_M**2
    ) * V_pd_pi
    H_Y_YZ = (3**0.5) * tmp_M**2 * tmp_N * V_pd_sigma + tmp_N * (
        1 - 2 * tmp_M**2
    ) * V_pd_pi
    H_Y_ZX = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_Y_X2Y2 = (
        0.5 * (3**0.5) * tmp_M * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_pd_pi
    )
    H_Y_Z2 = (
        tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        - 3**0.5 * tmp_M * tmp_N**2 * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 4 + batch_block_offset, H_Y_XY)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 5 + batch_block_offset, H_Y_YZ)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 6 + batch_block_offset, H_Y_ZX)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 7 + batch_block_offset, H_Y_X2Y2)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col + 8 + batch_block_offset, H_Y_Z2)
    H_Z_XY = (
        3**0.5
    ) * tmp_L * tmp_M * tmp_N * V_pd_sigma - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi
    H_Z_YZ = (3**0.5) * tmp_N**2 * tmp_M * V_pd_sigma + tmp_M * (
        1 - 2 * tmp_N**2
    ) * V_pd_pi
    H_Z_ZX = (3**0.5) * tmp_N**2 * tmp_L * V_pd_sigma + tmp_L * (
        1 - 2 * tmp_N**2
    ) * V_pd_pi
    H_Z_X2Y2 = (
        0.5 * (3**0.5) * tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_sigma
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_pi
    )
    H_Z_Z2 = (
        tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_pd_pi
    )
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 4 + batch_block_offset, H_Z_XY)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 5 + batch_block_offset, H_Z_YZ)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 6 + batch_block_offset, H_Z_ZX)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 7 + batch_block_offset, H_Z_X2Y2)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col + 8 + batch_block_offset, H_Z_Z2)
    # p-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask[valid_pairs]]
    tmp_M_dxyz = M_dxyz[:, tmp_mask[valid_pairs]]
    tmp_N_dxyz = N_dxyz[:, tmp_mask[valid_pairs]]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask[valid_pairs]]
    V_pd_sigma_dxyz = V_pd_sigma_dR * tmp_dR_dxyz
    V_pd_pi_dxyz = V_pd_pi_dR * tmp_dR_dxyz

    H_X_XY_dxyz = (
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_M * V_pd_sigma
            + tmp_L**2 * tmp_M_dxyz * V_pd_sigma
            + tmp_L**2 * tmp_M * V_pd_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_M) * V_pd_pi
        + tmp_M * (1 - 2 * tmp_L**2) * V_pd_pi_dxyz
    )
    H_X_YZ_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_X_ZX_dxyz = (
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_N * V_pd_sigma
            + tmp_L**2 * tmp_N_dxyz * V_pd_sigma
            + tmp_L**2 * tmp_N * V_pd_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_N) * V_pd_pi
        + tmp_N * (1 - 2 * tmp_L**2) * V_pd_pi_dxyz
    )
    H_X_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_L_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_L * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        + (
            tmp_L_dxyz * (1 - tmp_L**2 + tmp_M**2)
            - tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_pd_pi_dxyz
    )
    H_X_Z2_dxyz = (
        (
            tmp_L_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_L * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        - 3**0.5 * (tmp_L_dxyz * tmp_N**2 + 2 * tmp_L * tmp_N * tmp_N_dxyz) * V_pd_pi
        - 3**0.5 * tmp_L * tmp_N**2 * V_pd_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 4 + batch_block_offset, H_X_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 5 + batch_block_offset, H_X_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 6 + batch_block_offset, H_X_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 7 + batch_block_offset, H_X_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 1) * HDIM + idx_col + 8 + batch_block_offset, H_X_Z2_dxyz
    )
    H_Y_XY_dxyz = (
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_L * V_pd_sigma
            + tmp_M**2 * tmp_L_dxyz * V_pd_sigma
            + tmp_M**2 * tmp_L * V_pd_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_L) * V_pd_pi
        + tmp_L * (1 - 2 * tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_YZ_dxyz = (
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_M**2 * tmp_N_dxyz * V_pd_sigma
            + tmp_M**2 * tmp_N * V_pd_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_N) * V_pd_pi
        + tmp_N * (1 - 2 * tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_ZX_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_Y_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_M_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_M * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        - (
            tmp_M_dxyz * (1 + tmp_L**2 - tmp_M**2)
            + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_pd_pi_dxyz
    )
    H_Y_Z2_dxyz = (
        (
            tmp_M_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_M * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        - 3**0.5 * (tmp_M_dxyz * tmp_N**2 + 2 * tmp_M * tmp_N * tmp_N_dxyz) * V_pd_pi
        - 3**0.5 * tmp_M * tmp_N**2 * V_pd_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 4 + batch_block_offset, H_Y_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 5 + batch_block_offset, H_Y_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 6 + batch_block_offset, H_Y_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 7 + batch_block_offset, H_Y_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 2) * HDIM + idx_col + 8 + batch_block_offset, H_Y_Z2_dxyz
    )
    H_Z_XY_dxyz = (
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_pd_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_pd_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_pd_sigma
            + tmp_L * tmp_M * tmp_N * V_pd_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_pd_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_pd_pi_dxyz
    )
    H_Z_YZ_dxyz = (
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_M * V_pd_sigma
            + tmp_N**2 * tmp_M_dxyz * V_pd_sigma
            + tmp_N**2 * tmp_M * V_pd_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_M) * V_pd_pi
        + tmp_M * (1 - 2 * tmp_N**2) * V_pd_pi_dxyz
    )
    H_Z_ZX_dxyz = (
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_L * V_pd_sigma
            + tmp_N**2 * tmp_L_dxyz * V_pd_sigma
            + tmp_N**2 * tmp_L * V_pd_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_L) * V_pd_pi
        + tmp_L * (1 - 2 * tmp_N**2) * V_pd_pi_dxyz
    )
    H_Z_X2Y2_dxyz = (
        0.5
        * (3**0.5)
        * (
            (
                tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_pd_sigma
            + tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_sigma_dxyz
        )
        - (
            tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
            + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_pd_pi_dxyz
    )
    H_Z_Z2_dxyz = (
        (
            tmp_N_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_N * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_pd_sigma
        + tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_pd_sigma_dxyz
        + 3**0.5
        * (
            tmp_N_dxyz * (tmp_L**2 + tmp_M**2)
            + 2 * tmp_N * (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
        )
        * V_pd_pi
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_pd_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 4 + batch_block_offset, H_Z_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 5 + batch_block_offset, H_Z_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 6 + batch_block_offset, H_Z_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 7 + batch_block_offset, H_Z_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 3) * HDIM + idx_col + 8 + batch_block_offset, H_Z_Z2_dxyz
    )

    ### d-s
    tmp_mask = pair_mask_YH | pair_mask_YX | pair_mask_YY
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    tmp_dx = dx[tmp_mask[valid_pairs]]
    tmp_L = L[tmp_mask[valid_pairs]]
    tmp_M = M[tmp_mask[valid_pairs]]
    tmp_N = N[tmp_mask[valid_pairs]]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('sd0', SH_shift))
    ]
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    V_ds_sigma = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_ds_sigma_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    H_XY_S = (3**0.5) * tmp_L * tmp_M * V_ds_sigma
    H_YZ_S = (3**0.5) * tmp_M * tmp_N * V_ds_sigma
    H_ZX_S = (3**0.5) * tmp_N * tmp_L * V_ds_sigma
    H_X2Y2_S = 0.5 * (3**0.5) * (tmp_L**2 - tmp_M**2) * V_ds_sigma
    H_Z2_S = (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_ds_sigma
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + batch_block_offset, H_XY_S)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + batch_block_offset, H_YZ_S)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + batch_block_offset, H_ZX_S)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + batch_block_offset, H_X2Y2_S)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + batch_block_offset, H_Z2_S)
    # d-s/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask[valid_pairs]]
    tmp_M_dxyz = M_dxyz[:, tmp_mask[valid_pairs]]
    tmp_N_dxyz = N_dxyz[:, tmp_mask[valid_pairs]]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask[valid_pairs]]
    V_ds_sigma_dxyz = V_ds_sigma_dR * tmp_dR_dxyz
    H_XY_S_dxyz = (3**0.5) * (
        tmp_L_dxyz * tmp_M * V_ds_sigma
        + tmp_L * tmp_M_dxyz * V_ds_sigma
        + tmp_L * tmp_M * V_ds_sigma_dxyz
    )
    H_YZ_S_dxyz = (3**0.5) * (
        tmp_M_dxyz * tmp_N * V_ds_sigma
        + tmp_M * tmp_N_dxyz * V_ds_sigma
        + tmp_M * tmp_N * V_ds_sigma_dxyz
    )
    H_ZX_S_dxyz = (3**0.5) * (
        tmp_N_dxyz * tmp_L * V_ds_sigma
        + tmp_N * tmp_L_dxyz * V_ds_sigma
        + tmp_N * tmp_L * V_ds_sigma_dxyz
    )
    H_X2Y2_S_dxyz = (
        0.5
        * (3**0.5)
        * (
            (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz) * V_ds_sigma
            + (tmp_L**2 - tmp_M**2) * V_ds_sigma_dxyz
        )
    )
    H_Z2_S_dxyz = (
        2 * tmp_N * tmp_N_dxyz - (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
    ) * V_ds_sigma + (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_ds_sigma_dxyz
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + batch_block_offset, H_XY_S_dxyz)
    dH0.index_add_(1, (idx_row + 5) * HDIM + idx_col + batch_block_offset, H_YZ_S_dxyz)
    dH0.index_add_(1, (idx_row + 6) * HDIM + idx_col + batch_block_offset, H_ZX_S_dxyz)
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + batch_block_offset, H_X2Y2_S_dxyz
    )
    dH0.index_add_(1, (idx_row + 8) * HDIM + idx_col + batch_block_offset, H_Z2_S_dxyz)

    ### d-p
    tmp_mask = pair_mask_YX | pair_mask_YY
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    tmp_dx = dx[tmp_mask[valid_pairs]]
    tmp_L = L[tmp_mask[valid_pairs]]
    tmp_M = M[tmp_mask[valid_pairs]]
    tmp_N = N[tmp_mask[valid_pairs]]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pd0', SH_shift))
    ]
    V_dp_sigma = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_dp_sigma_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('pd1', SH_shift))
    ]
    V_dp_pi = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_dp_pi_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    H_XY_X = -(
        (3**0.5) * tmp_L**2 * tmp_M * V_dp_sigma + tmp_M * (1 - 2 * tmp_L**2) * V_dp_pi
    )
    H_XY_Y = -(
        (3**0.5) * tmp_M**2 * tmp_L * V_dp_sigma + tmp_L * (1 - 2 * tmp_M**2) * V_dp_pi
    )
    H_XY_Z = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 1 + batch_block_offset, H_XY_X)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 2 + batch_block_offset, H_XY_Y)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 3 + batch_block_offset, H_XY_Z)
    H_YZ_X = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H_YZ_Y = -(
        (3**0.5) * tmp_M**2 * tmp_N * V_dp_sigma + tmp_N * (1 - 2 * tmp_M**2) * V_dp_pi
    )
    H_YZ_Z = -(
        (3**0.5) * tmp_N**2 * tmp_M * V_dp_sigma + tmp_M * (1 - 2 * tmp_N**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 1 + batch_block_offset, H_YZ_X)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 2 + batch_block_offset, H_YZ_Y)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 3 + batch_block_offset, H_YZ_Z)
    H_ZX_X = -(
        (3**0.5) * tmp_L**2 * tmp_N * V_dp_sigma + tmp_N * (1 - 2 * tmp_L**2) * V_dp_pi
    )
    H_ZX_Y = -(
        (3**0.5) * tmp_L * tmp_M * tmp_N * V_dp_sigma
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi
    )
    H_ZX_Z = -(
        (3**0.5) * tmp_N**2 * tmp_L * V_dp_sigma + tmp_L * (1 - 2 * tmp_N**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 1 + batch_block_offset, H_ZX_X)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 2 + batch_block_offset, H_ZX_Y)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 3 + batch_block_offset, H_ZX_Z)
    H_X2Y2_X = -(
        0.5 * (3**0.5) * tmp_L * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_dp_pi
    )
    H_X2Y2_Y = -(
        0.5 * (3**0.5) * tmp_M * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_dp_pi
    )
    H_X2Y2_Z = -(
        0.5 * (3**0.5) * tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_sigma
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 1 + batch_block_offset, H_X2Y2_X)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 2 + batch_block_offset, H_X2Y2_Y)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 3 + batch_block_offset, H_X2Y2_Z)
    H_Z2_X = -(
        tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        - 3**0.5 * tmp_L * tmp_N**2 * V_dp_pi
    )
    H_Z2_Y = -(
        tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        - 3**0.5 * tmp_M * tmp_N**2 * V_dp_pi
    )
    H_Z2_Z = -(
        tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_dp_pi
    )
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 1 + batch_block_offset, H_Z2_X)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 2 + batch_block_offset, H_Z2_Y)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 3 + batch_block_offset, H_Z2_Z)
    # d-p/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask[valid_pairs]]
    tmp_M_dxyz = M_dxyz[:, tmp_mask[valid_pairs]]
    tmp_N_dxyz = N_dxyz[:, tmp_mask[valid_pairs]]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask[valid_pairs]]
    V_dp_sigma_dxyz = V_dp_sigma_dR * tmp_dR_dxyz
    V_dp_pi_dxyz = V_dp_pi_dR * tmp_dR_dxyz
    H_XY_X_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_M * V_dp_sigma
            + tmp_L**2 * tmp_M_dxyz * V_dp_sigma
            + tmp_L**2 * tmp_M * V_dp_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_M) * V_dp_pi
        + tmp_M * (1 - 2 * tmp_L**2) * V_dp_pi_dxyz
    )
    H_XY_Y_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_L * V_dp_sigma
            + tmp_M**2 * tmp_L_dxyz * V_dp_sigma
            + tmp_M**2 * tmp_L * V_dp_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_L) * V_dp_pi
        + tmp_L * (1 - 2 * tmp_M**2) * V_dp_pi_dxyz
    )
    H_XY_Z_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 1 + batch_block_offset, H_XY_X_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 2 + batch_block_offset, H_XY_Y_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 3 + batch_block_offset, H_XY_Z_dxyz
    )
    H_YZ_X_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    H_YZ_Y_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_M * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_M**2 * tmp_N_dxyz * V_dp_sigma
            + tmp_M**2 * tmp_N * V_dp_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_M**2) - 4 * tmp_M * tmp_M_dxyz * tmp_N) * V_dp_pi
        + tmp_N * (1 - 2 * tmp_M**2) * V_dp_pi_dxyz
    )
    H_YZ_Z_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_M * V_dp_sigma
            + tmp_N**2 * tmp_M_dxyz * V_dp_sigma
            + tmp_N**2 * tmp_M * V_dp_sigma_dxyz
        )
        + (tmp_M_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_M) * V_dp_pi
        + tmp_M * (1 - 2 * tmp_N**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 1 + batch_block_offset, H_YZ_X_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 2 + batch_block_offset, H_YZ_Y_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 3 + batch_block_offset, H_YZ_Z_dxyz
    )
    H_ZX_X_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_L * tmp_L_dxyz * tmp_N * V_dp_sigma
            + tmp_L**2 * tmp_N_dxyz * V_dp_sigma
            + tmp_L**2 * tmp_N * V_dp_sigma_dxyz
        )
        + (tmp_N_dxyz * (1 - 2 * tmp_L**2) - 4 * tmp_L * tmp_L_dxyz * tmp_N) * V_dp_pi
        + tmp_N * (1 - 2 * tmp_L**2) * V_dp_pi_dxyz
    )
    H_ZX_Y_dxyz = -(
        (3**0.5)
        * (
            tmp_L_dxyz * tmp_M * tmp_N * V_dp_sigma
            + tmp_L * tmp_M_dxyz * tmp_N * V_dp_sigma
            + tmp_L * tmp_M * tmp_N_dxyz * V_dp_sigma
            + tmp_L * tmp_M * tmp_N * V_dp_sigma_dxyz
        )
        - 2
        * (
            tmp_L_dxyz * tmp_M * tmp_N
            + tmp_L * tmp_M_dxyz * tmp_N
            + tmp_L * tmp_M * tmp_N_dxyz
        )
        * V_dp_pi
        - 2 * tmp_L * tmp_M * tmp_N * V_dp_pi_dxyz
    )
    H_ZX_Z_dxyz = -(
        (3**0.5)
        * (
            2 * tmp_N * tmp_N_dxyz * tmp_L * V_dp_sigma
            + tmp_N**2 * tmp_L_dxyz * V_dp_sigma
            + tmp_N**2 * tmp_L * V_dp_sigma_dxyz
        )
        + (tmp_L_dxyz * (1 - 2 * tmp_N**2) - 4 * tmp_N * tmp_N_dxyz * tmp_L) * V_dp_pi
        + tmp_L * (1 - 2 * tmp_N**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 1 + batch_block_offset, H_ZX_X_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 2 + batch_block_offset, H_ZX_Y_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 3 + batch_block_offset, H_ZX_Z_dxyz
    )
    H_X2Y2_X_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_L_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_L * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        + (
            tmp_L_dxyz * (1 - tmp_L**2 + tmp_M**2)
            - tmp_L * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        + tmp_L * (1 - tmp_L**2 + tmp_M**2) * V_dp_pi_dxyz
    )
    H_X2Y2_Y_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_M_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_M * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        - (
            tmp_M_dxyz * (1 + tmp_L**2 - tmp_M**2)
            + tmp_M * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        - tmp_M * (1 + tmp_L**2 - tmp_M**2) * V_dp_pi_dxyz
    )
    H_X2Y2_Z_dxyz = -(
        0.5
        * (3**0.5)
        * (
            (
                tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
                + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
            )
            * V_dp_sigma
            + tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_sigma_dxyz
        )
        - (
            tmp_N_dxyz * (tmp_L**2 - tmp_M**2)
            + tmp_N * (2 * tmp_L * tmp_L_dxyz - 2 * tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        - tmp_N * (tmp_L**2 - tmp_M**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 1 + batch_block_offset, H_X2Y2_X_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 2 + batch_block_offset, H_X2Y2_Y_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 3 + batch_block_offset, H_X2Y2_Z_dxyz
    )
    H_Z2_X_dxyz = -(
        (
            tmp_L_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_L * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        - 3**0.5 * (tmp_L_dxyz * tmp_N**2 + 2 * tmp_L * tmp_N * tmp_N_dxyz) * V_dp_pi
        - 3**0.5 * tmp_L * tmp_N**2 * V_dp_pi_dxyz
    )
    H_Z2_Y_dxyz = -(
        (
            tmp_M_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_M * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        - 3**0.5 * (tmp_M_dxyz * tmp_N**2 + 2 * tmp_M * tmp_N * tmp_N_dxyz) * V_dp_pi
        - 3**0.5 * tmp_M * tmp_N**2 * V_dp_pi_dxyz
    )
    H_Z2_Z_dxyz = -(
        (
            tmp_N_dxyz * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
            + tmp_N * (2 * tmp_N * tmp_N_dxyz - tmp_L * tmp_L_dxyz - tmp_M * tmp_M_dxyz)
        )
        * V_dp_sigma
        + tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dp_sigma_dxyz
        + 3**0.5
        * (
            tmp_N_dxyz * (tmp_L**2 + tmp_M**2)
            + 2 * tmp_N * (tmp_L * tmp_L_dxyz + tmp_M * tmp_M_dxyz)
        )
        * V_dp_pi
        + 3**0.5 * tmp_N * (tmp_L**2 + tmp_M**2) * V_dp_pi_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 1 + batch_block_offset, H_Z2_X_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 2 + batch_block_offset, H_Z2_Y_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 3 + batch_block_offset, H_Z2_Z_dxyz
    )

    ### d-d

    tmp_mask = pair_mask_YY
    idx_row = H_INDEX_START.gather(1, safe_I)[tmp_mask]
    idx_col = H_INDEX_START.gather(1, safe_J)[tmp_mask]
    tmp_dx = dx[tmp_mask[valid_pairs]]
    tmp_L = L[tmp_mask[valid_pairs]]
    tmp_M = M[tmp_mask[valid_pairs]]
    tmp_N = N[tmp_mask[valid_pairs]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask[valid_pairs]]
    batch_ids = (
        torch.arange(B, device=H_INDEX_START.device)
        .unsqueeze(1)
        .expand_as(H_INDEX_START)
    )
    batch_ids = batch_ids.gather(1, safe_J)[tmp_mask]
    batch_block_offset = batch_ids * (HDIM * HDIM)

    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('dd0', SH_shift))
    ]
    V_dd_sigma = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_dd_sigma_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('dd1', SH_shift))
    ]
    V_dd_pi = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_dd_pi_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    coeffs_selected = coeffs_tensor[
        sel_IJ, sel_idx, sk_channel_index(sk_channel_name('dd2', SH_shift))
    ]
    V_dd_delta = (
        coeffs_selected[:, 0]
        + coeffs_selected[:, 1] * tmp_dx
        + coeffs_selected[:, 2] * tmp_dx**2
        + coeffs_selected[:, 3] * tmp_dx**3
    )
    V_dd_delta_dR = (
        coeffs_selected[:, 1]
        + 2 * coeffs_selected[:, 2] * tmp_dx
        + 3 * coeffs_selected[:, 3] * tmp_dx**2
    )
    H_XY_XY = (
        3 * tmp_L**2 * tmp_M**2 * V_dd_sigma
        + (tmp_L**2 + tmp_M**2 - 4 * tmp_L**2 * tmp_M**2) * V_dd_pi
        + (tmp_N**2 + tmp_L**2 * tmp_M**2) * V_dd_delta
    )
    H_XY_YZ = (
        3 * tmp_L * tmp_M**2 * tmp_N * V_dd_sigma
        + tmp_L * tmp_N * (1 - 4 * tmp_M**2) * V_dd_pi
        + tmp_L * tmp_N * (tmp_M**2 - 1) * V_dd_delta
    )
    H_XY_ZX = (
        3 * tmp_L**2 * tmp_M * tmp_N * V_dd_sigma
        + tmp_M * tmp_N * (1 - 4 * tmp_L**2) * V_dd_pi
        + tmp_M * tmp_N * (tmp_L**2 - 1) * V_dd_delta
    )
    H_XY_X2Y2 = (
        1.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + 2 * tmp_L * tmp_M * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + 0.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_XY_Z2 = (
        (3**0.5) * tmp_L * tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        - (3**0.5) * 2 * tmp_L * tmp_M * tmp_N**2 * V_dd_pi
        + (3**0.5) * 0.5 * tmp_L * tmp_M * (1 + tmp_N**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 4 + batch_block_offset, H_XY_XY)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 5 + batch_block_offset, H_XY_YZ)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 6 + batch_block_offset, H_XY_ZX)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 7 + batch_block_offset, H_XY_X2Y2)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 8 + batch_block_offset, H_XY_Z2)
    H_YZ_XY = (
        3 * tmp_M**2 * tmp_N * tmp_L * V_dd_sigma
        + tmp_L * tmp_N * (1 - 4 * tmp_M**2) * V_dd_pi
        + tmp_L * tmp_N * (tmp_M**2 - 1) * V_dd_delta
    )
    H_YZ_YZ = (
        3 * tmp_M**2 * tmp_N**2 * V_dd_sigma
        + (tmp_M**2 + tmp_N**2 - 4 * tmp_M**2 * tmp_N**2) * V_dd_pi
        + (tmp_L**2 + tmp_M**2 * tmp_N**2) * V_dd_delta
    )
    H_YZ_ZX = (
        3 * tmp_M * tmp_N**2 * tmp_L * V_dd_sigma
        + tmp_L * tmp_M * (1 - 4 * tmp_N**2) * V_dd_pi
        + tmp_L * tmp_M * (tmp_N**2 - 1) * V_dd_delta
    )
    H_YZ_X2Y2 = (
        1.5 * tmp_M * tmp_N * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        - tmp_M * tmp_N * (1 + 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        + tmp_M * tmp_N * (1 + 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_YZ_Z2 = (
        (3**0.5) * tmp_M * tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 4 + batch_block_offset, H_YZ_XY)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 5 + batch_block_offset, H_YZ_YZ)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 6 + batch_block_offset, H_YZ_ZX)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 7 + batch_block_offset, H_YZ_X2Y2)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 8 + batch_block_offset, H_YZ_Z2)
    H_ZX_XY = (
        3 * tmp_L**2 * tmp_M * tmp_N * V_dd_sigma
        + tmp_M * tmp_N * (1 - 4 * tmp_L**2) * V_dd_pi
        + tmp_M * tmp_N * (tmp_L**2 - 1) * V_dd_delta
    )
    H_ZX_YZ = (
        3 * tmp_M * tmp_N**2 * tmp_L * V_dd_sigma
        + tmp_L * tmp_M * (1 - 4 * tmp_N**2) * V_dd_pi
        + tmp_L * tmp_M * (tmp_N**2 - 1) * V_dd_delta
    )
    H_ZX_ZX = (
        3 * tmp_N**2 * tmp_L**2 * V_dd_sigma
        + (tmp_N**2 + tmp_L**2 - 4 * tmp_N**2 * tmp_L**2) * V_dd_pi
        + (tmp_M**2 + tmp_N**2 * tmp_L**2) * V_dd_delta
    )
    H_ZX_X2Y2 = (
        1.5 * tmp_N * tmp_L * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + tmp_N * tmp_L * (1 - 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        - tmp_N * tmp_L * (1 - 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_ZX_Z2 = (
        (3**0.5) * tmp_N * tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 4 + batch_block_offset, H_ZX_XY)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 5 + batch_block_offset, H_ZX_YZ)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 6 + batch_block_offset, H_ZX_ZX)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 7 + batch_block_offset, H_ZX_X2Y2)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 8 + batch_block_offset, H_ZX_Z2)
    H_X2Y2_XY = (
        1.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + 2 * tmp_L * tmp_M * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + 0.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_X2Y2_YZ = (
        1.5 * tmp_M * tmp_N * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        - tmp_M * tmp_N * (1 + 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        + tmp_M * tmp_N * (1 + 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_X2Y2_ZX = (
        1.5 * tmp_N * tmp_L * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + tmp_N * tmp_L * (1 - 2 * (tmp_L**2 - tmp_M**2)) * V_dd_pi
        - tmp_N * tmp_L * (1 - 0.5 * (tmp_L**2 - tmp_M**2)) * V_dd_delta
    )
    H_X2Y2_X2Y2 = (
        0.75 * (tmp_L**2 - tmp_M**2) ** 2 * V_dd_sigma
        + (tmp_L**2 + tmp_M**2 - (tmp_L**2 - tmp_M**2) ** 2) * V_dd_pi
        + (tmp_N**2 + 0.25 * (tmp_L**2 - tmp_M**2) ** 2) * V_dd_delta
    )
    H_X2Y2_Z2 = (
        (3**0.5)
        * 0.5
        * (tmp_L**2 - tmp_M**2)
        * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
        * V_dd_sigma
        + (3**0.5) * tmp_N**2 * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + (3**0.5) * 0.25 * (1 + tmp_N**2) * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 4 + batch_block_offset, H_X2Y2_XY)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 5 + batch_block_offset, H_X2Y2_YZ)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 6 + batch_block_offset, H_X2Y2_ZX)
    H0.index_add_(
        0, (idx_row + 7) * HDIM + idx_col + 7 + batch_block_offset, H_X2Y2_X2Y2
    )
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 8 + batch_block_offset, H_X2Y2_Z2)
    H_Z2_XY = (
        (3**0.5) * tmp_L * tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        - (3**0.5) * 2 * tmp_L * tmp_M * tmp_N**2 * V_dd_pi
        + (3**0.5) * 0.5 * tmp_L * tmp_M * (1 + tmp_N**2) * V_dd_delta
    )
    H_Z2_YZ = (
        (3**0.5) * tmp_M * tmp_N * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_M * tmp_N * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H_Z2_ZX = (
        (3**0.5) * tmp_N * tmp_L * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        + (3**0.5) * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2 - tmp_N**2) * V_dd_pi
        - (3**0.5) * 0.5 * tmp_N * tmp_L * (tmp_L**2 + tmp_M**2) * V_dd_delta
    )
    H_Z2_X2Y2 = (
        (3**0.5)
        * 0.5
        * (tmp_L**2 - tmp_M**2)
        * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2))
        * V_dd_sigma
        + (3**0.5) * tmp_N**2 * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + (3**0.5) * 0.25 * (1 + tmp_N**2) * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    H_Z2_Z2 = (
        (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) ** 2 * V_dd_sigma
        + 3 * tmp_N**2 * (tmp_L**2 + tmp_M**2) * V_dd_pi
        + 0.75 * (tmp_L**2 + tmp_M**2) ** 2 * V_dd_delta
    )
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 4 + batch_block_offset, H_Z2_XY)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 5 + batch_block_offset, H_Z2_YZ)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 6 + batch_block_offset, H_Z2_ZX)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 7 + batch_block_offset, H_Z2_X2Y2)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 8 + batch_block_offset, H_Z2_Z2)
    # d-d/dx
    tmp_L_dxyz = L_dxyz[:, tmp_mask[valid_pairs]]
    tmp_M_dxyz = M_dxyz[:, tmp_mask[valid_pairs]]
    tmp_N_dxyz = N_dxyz[:, tmp_mask[valid_pairs]]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask[valid_pairs]]
    V_dd_sigma_dxyz = V_dd_sigma_dR * tmp_dR_dxyz
    V_dd_pi_dxyz = V_dd_pi_dR * tmp_dR_dxyz
    V_dd_delta_dxyz = V_dd_delta_dR * tmp_dR_dxyz
    # t - time, m - minus, p - plus
    L_t_Ldx = tmp_L * tmp_L_dxyz
    M_t_Mdx = tmp_M * tmp_M_dxyz
    N_t_Ndx = tmp_N * tmp_N_dxyz
    L_t_M = tmp_L * tmp_M
    M_t_N = tmp_M * tmp_N
    N_t_L = tmp_N * tmp_L

    L2 = tmp_L**2
    M2 = tmp_M**2
    N2 = tmp_N**2

    H_XY_XY_dxyz = (
        3 * (2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx) * V_dd_sigma
        + 3 * L2 * M2 * V_dd_sigma_dxyz
        + ((2 * L_t_Ldx + 2 * M_t_Mdx) - 4 * (2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx))
        * V_dd_pi
        + (L2 + M2 - 4 * L2 * M2) * V_dd_pi_dxyz
        + (2 * N_t_Ndx + 2 * L_t_Ldx * M2 + L2 * 2 * M_t_Mdx) * V_dd_delta
        + (N2 + L2 * M2) * V_dd_delta_dxyz
    )
    H_XY_YZ_dxyz = (
        3
        * (
            tmp_L_dxyz * M2 * tmp_N
            + tmp_L * 2 * M_t_Mdx * tmp_N
            + tmp_L * M2 * tmp_N_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_L * M2 * tmp_N * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (1 - 4 * M2) * V_dd_pi
        + tmp_L * tmp_N * (-8 * M_t_Mdx) * V_dd_pi
        + tmp_L * tmp_N * (1 - 4 * M2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (M2 - 1) * V_dd_delta
        + tmp_L * tmp_N * (2 * M_t_Mdx) * V_dd_delta
        + tmp_L * tmp_N * (M2 - 1) * V_dd_delta_dxyz
    )
    H_XY_ZX_dxyz = (
        3
        * (2 * L_t_Ldx * M_t_N + L2 * tmp_M_dxyz * tmp_N + L2 * tmp_M * tmp_N_dxyz)
        * V_dd_sigma
        + 3 * L2 * M_t_N * V_dd_sigma_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 - 4 * L2) * V_dd_pi
        + M_t_N * (-8 * L_t_Ldx) * V_dd_pi
        + M_t_N * (1 - 4 * L2) * V_dd_pi_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - 1) * V_dd_delta
        + M_t_N * (2 * L_t_Ldx) * V_dd_delta
        + M_t_N * (L2 - 1) * V_dd_delta_dxyz
    )
    H_XY_X2Y2_dxyz = (
        1.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * L_t_M * (L2 - M2) * V_dd_sigma_dxyz
        + 2
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (M2 - L2)
            + L_t_M * (2 * M_t_Mdx - 2 * L_t_Ldx)
        )
        * V_dd_pi
        + 2 * L_t_M * (M2 - L2) * V_dd_pi_dxyz
        + 0.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_delta
        + 0.5 * L_t_M * (L2 - M2) * V_dd_delta_dxyz
    )
    H_XY_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 0.5 * (L2 + M2))
            + L_t_M * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * L_t_M * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        - (3**0.5)
        * 2
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * N2 + L_t_M * 2 * N_t_Ndx)
        * V_dd_pi
        - (3**0.5) * 2 * L_t_M * N2 * V_dd_pi_dxyz
        + (3**0.5)
        * 0.5
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 + N2) + L_t_M * 2 * N_t_Ndx)
        * V_dd_delta
        + (3**0.5) * 0.5 * L_t_M * (1 + N2) * V_dd_delta_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 4 + batch_block_offset, H_XY_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 5 + batch_block_offset, H_XY_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 6 + batch_block_offset, H_XY_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 7 + batch_block_offset, H_XY_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 4) * HDIM + idx_col + 8 + batch_block_offset, H_XY_Z2_dxyz
    )
    H_YZ_XY_dxyz = (
        3
        * (2 * M_t_Mdx * N_t_L + M2 * tmp_N_dxyz * tmp_L + M2 * tmp_N * tmp_L_dxyz)
        * V_dd_sigma
        + 3 * M2 * N_t_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (1 - 4 * M2) * V_dd_pi
        + tmp_L * tmp_N * (-8 * M_t_Mdx) * V_dd_pi
        + tmp_L * tmp_N * (1 - 4 * M2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_N + tmp_L * tmp_N_dxyz) * (M2 - 1) * V_dd_delta
        + tmp_L * tmp_N * (2 * M_t_Mdx) * V_dd_delta
        + tmp_L * tmp_N * (M2 - 1) * V_dd_delta_dxyz
    )
    H_YZ_YZ_dxyz = (
        3 * (2 * M_t_Mdx * N2 + M2 * 2 * N_t_Ndx) * V_dd_sigma
        + 3 * M2 * N2 * V_dd_sigma_dxyz
        + (2 * M_t_Mdx + 2 * N_t_Ndx - 8 * (M_t_Mdx * N2 + M2 * N_t_Ndx)) * V_dd_pi
        + (M2 + N2 - 4 * M2 * N2) * V_dd_pi_dxyz
        + (2 * L_t_Ldx + 2 * M_t_Mdx * N2 + M2 * 2 * N_t_Ndx) * V_dd_delta
        + (L2 + M2 * N2) * V_dd_delta_dxyz
    )
    H_YZ_ZX_dxyz = (
        3
        * (
            tmp_M_dxyz * N2 * tmp_L
            + tmp_M * 2 * N_t_Ndx * tmp_L
            + tmp_M * N2 * tmp_L_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_M * N2 * tmp_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 - 4 * N2) * V_dd_pi
        + L_t_M * (-8 * N_t_Ndx) * V_dd_pi
        + L_t_M * (1 - 4 * N2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 1) * V_dd_delta
        + L_t_M * (2 * N_t_Ndx) * V_dd_delta
        + L_t_M * (N2 - 1) * V_dd_delta_dxyz
    )
    H_YZ_X2Y2_dxyz = (
        1.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - M2)
            + M_t_N * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * M_t_N * (L2 - M2) * V_dd_sigma_dxyz
        + -(
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 2 * (L2 - M2))
            + M_t_N * (4 * L_t_Ldx - 4 * M_t_Mdx)
        )
        * V_dd_pi
        - M_t_N * (1 + 2 * (L2 - M2)) * V_dd_pi_dxyz
        + (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 0.5 * (L2 - M2))
            + M_t_N * (L_t_Ldx - M_t_Mdx)
        )
        * V_dd_delta
        + M_t_N * (1 + 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_YZ_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (N2 - 0.5 * (L2 + M2))
            + M_t_N * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * M_t_N * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2 - N2)
            + M_t_N * (2 * L_t_Ldx + 2 * M_t_Mdx - 2 * N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * M_t_N * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2)
            + M_t_N * (2 * L_t_Ldx + 2 * M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * M_t_N * (L2 + M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 4 + batch_block_offset, H_YZ_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 5 + batch_block_offset, H_YZ_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 6 + batch_block_offset, H_YZ_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 7 + batch_block_offset, H_YZ_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 5) * HDIM + idx_col + 8 + batch_block_offset, H_YZ_Z2_dxyz
    )
    H_ZX_XY_dxyz = (
        3
        * (2 * L_t_Ldx * M_t_N + L2 * tmp_M_dxyz * tmp_N + L2 * tmp_M * tmp_N_dxyz)
        * V_dd_sigma
        + 3 * L2 * M_t_N * V_dd_sigma_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 - 4 * L2) * V_dd_pi
        + M_t_N * (-8 * L_t_Ldx) * V_dd_pi
        + M_t_N * (1 - 4 * L2) * V_dd_pi_dxyz
        + (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - 1) * V_dd_delta
        + M_t_N * (2 * L_t_Ldx) * V_dd_delta
        + M_t_N * (L2 - 1) * V_dd_delta_dxyz
    )
    H_ZX_YZ_dxyz = (
        3
        * (
            tmp_M_dxyz * N2 * tmp_L
            + tmp_M * 2 * N_t_Ndx * tmp_L
            + tmp_M * N2 * tmp_L_dxyz
        )
        * V_dd_sigma
        + 3 * tmp_M * N2 * tmp_L * V_dd_sigma_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 - 4 * N2) * V_dd_pi
        + L_t_M * (-8 * N_t_Ndx) * V_dd_pi
        + L_t_M * (1 - 4 * N2) * V_dd_pi_dxyz
        + (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 1) * V_dd_delta
        + L_t_M * (2 * N_t_Ndx) * V_dd_delta
        + L_t_M * (N2 - 1) * V_dd_delta_dxyz
    )
    H_ZX_ZX_dxyz = (
        3 * (2 * N_t_Ndx * L2 + N2 * 2 * L_t_Ldx) * V_dd_sigma
        + 3 * N2 * L2 * V_dd_sigma_dxyz
        + (2 * N_t_Ndx + 2 * L_t_Ldx - 8 * (N_t_Ndx * L2 + N2 * L_t_Ldx)) * V_dd_pi
        + (N2 + L2 - 4 * N2 * L2) * V_dd_pi_dxyz
        + (2 * M_t_Mdx + 2 * N_t_Ndx * L2 + N2 * 2 * L_t_Ldx) * V_dd_delta
        + (M2 + N2 * L2) * V_dd_delta_dxyz
    )
    H_ZX_X2Y2_dxyz = (
        1.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 - M2)
            + N_t_L * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * N_t_L * (L2 - M2) * V_dd_sigma_dxyz
        + (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 2 * (L2 - M2))
            + N_t_L * (-4 * (L_t_Ldx - M_t_Mdx))
        )
        * V_dd_pi
        + N_t_L * (1 - 2 * (L2 - M2)) * V_dd_pi_dxyz
        + -(
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 0.5 * (L2 - M2))
            + N_t_L * (-L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - N_t_L * (1 - 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_ZX_Z2_dxyz = (
        (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (N2 - 0.5 * (L2 + M2))
            + N_t_L * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * N_t_L * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2 - N2)
            + N_t_L * (2 * L_t_Ldx + 2 * M_t_Mdx - 2 * N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * N_t_L * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2)
            + N_t_L * (2 * L_t_Ldx + 2 * M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * N_t_L * (L2 + M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 4 + batch_block_offset, H_ZX_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 5 + batch_block_offset, H_ZX_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 6 + batch_block_offset, H_ZX_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 7 + batch_block_offset, H_ZX_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 6) * HDIM + idx_col + 8 + batch_block_offset, H_ZX_Z2_dxyz
    )
    H_X2Y2_XY_dxyz = (
        1.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * L_t_M * (L2 - M2) * V_dd_sigma_dxyz
        + 2
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (M2 - L2)
            + L_t_M * (2 * M_t_Mdx - 2 * L_t_Ldx)
        )
        * V_dd_pi
        + 2 * L_t_M * (M2 - L2) * V_dd_pi_dxyz
        + 0.5
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (L2 - M2)
            + L_t_M * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_delta
        + 0.5 * L_t_M * (L2 - M2) * V_dd_delta_dxyz
    )
    H_X2Y2_YZ_dxyz = (
        1.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 - M2)
            + M_t_N * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * M_t_N * (L2 - M2) * V_dd_sigma_dxyz
        + -(
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 2 * (L2 - M2))
            + M_t_N * (4 * L_t_Ldx - 4 * M_t_Mdx)
        )
        * V_dd_pi
        - M_t_N * (1 + 2 * (L2 - M2)) * V_dd_pi_dxyz
        + (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (1 + 0.5 * (L2 - M2))
            + M_t_N * (L_t_Ldx - M_t_Mdx)
        )
        * V_dd_delta
        + M_t_N * (1 + 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_X2Y2_ZX_dxyz = (
        1.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 - M2)
            + N_t_L * (2 * L_t_Ldx - 2 * M_t_Mdx)
        )
        * V_dd_sigma
        + 1.5 * N_t_L * (L2 - M2) * V_dd_sigma_dxyz
        + (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 2 * (L2 - M2))
            + N_t_L * (-4 * (L_t_Ldx - M_t_Mdx))
        )
        * V_dd_pi
        + N_t_L * (1 - 2 * (L2 - M2)) * V_dd_pi_dxyz
        + -(
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (1 - 0.5 * (L2 - M2))
            + N_t_L * (-L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - N_t_L * (1 - 0.5 * (L2 - M2)) * V_dd_delta_dxyz
    )
    H_X2Y2_X2Y2_dxyz = (
        0.75 * 4 * (L2 - M2) * (L_t_Ldx - M_t_Mdx) * V_dd_sigma
        + 0.75 * (L2 - M2) ** 2 * V_dd_sigma_dxyz
        + (2 * L_t_Ldx + 2 * M_t_Mdx - 4 * (L2 - M2) * (L_t_Ldx - M_t_Mdx)) * V_dd_pi
        + (L2 + M2 - (L2 - M2) ** 2) * V_dd_pi_dxyz
        + (2 * N_t_Ndx + (L2 - M2) * (L_t_Ldx - M_t_Mdx)) * V_dd_delta
        + (N2 + 0.25 * (L2 - M2) ** 2) * V_dd_delta_dxyz
    )
    H_X2Y2_Z2_dxyz = (
        (3**0.5)
        * 0.5
        * (
            2 * (L_t_Ldx - M_t_Mdx) * (N2 - 0.5 * (L2 + M2))
            + (L2 - M2) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * 0.5 * (L2 - M2) * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5) * (2 * N_t_Ndx * (M2 - L2) + N2 * 2 * (M_t_Mdx - L_t_Ldx)) * V_dd_pi
        + (3**0.5) * N2 * (M2 - L2) * V_dd_pi_dxyz
        + (3**0.5)
        * 0.25
        * (2 * N_t_Ndx * (L2 - M2) + (1 + N2) * 2 * (L_t_Ldx - M_t_Mdx))
        * V_dd_delta
        + (3**0.5) * 0.25 * (1 + N2) * (L2 - M2) * V_dd_delta_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 4 + batch_block_offset, H_X2Y2_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 5 + batch_block_offset, H_X2Y2_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 6 + batch_block_offset, H_X2Y2_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 7 + batch_block_offset, H_X2Y2_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 7) * HDIM + idx_col + 8 + batch_block_offset, H_X2Y2_Z2_dxyz
    )
    H_Z2_XY_dxyz = (
        (3**0.5)
        * (
            (tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (N2 - 0.5 * (L2 + M2))
            + L_t_M * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * L_t_M * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        - (3**0.5)
        * 2
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * N2 + L_t_M * 2 * N_t_Ndx)
        * V_dd_pi
        - (3**0.5) * 2 * L_t_M * N2 * V_dd_pi_dxyz
        + (3**0.5)
        * 0.5
        * ((tmp_L_dxyz * tmp_M + tmp_L * tmp_M_dxyz) * (1 + N2) + L_t_M * 2 * N_t_Ndx)
        * V_dd_delta
        + (3**0.5) * 0.5 * L_t_M * (1 + N2) * V_dd_delta_dxyz
    )
    H_Z2_YZ_dxyz = (
        (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (N2 - 0.5 * (L2 + M2))
            + M_t_N * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * M_t_N * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2 - N2)
            + M_t_N * 2 * (L_t_Ldx + M_t_Mdx - N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * M_t_N * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_M_dxyz * tmp_N + tmp_M * tmp_N_dxyz) * (L2 + M2)
            + M_t_N * 2 * (L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * M_t_N * (L2 + M2) * V_dd_delta_dxyz
    )
    H_Z2_ZX_dxyz = (
        (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (N2 - 0.5 * (L2 + M2))
            + N_t_L * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * N_t_L * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5)
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2 - N2)
            + N_t_L * 2 * (L_t_Ldx + M_t_Mdx - N_t_Ndx)
        )
        * V_dd_pi
        + (3**0.5) * N_t_L * (L2 + M2 - N2) * V_dd_pi_dxyz
        + -(3**0.5)
        * 0.5
        * (
            (tmp_N_dxyz * tmp_L + tmp_N * tmp_L_dxyz) * (L2 + M2)
            + N_t_L * 2 * (L_t_Ldx + M_t_Mdx)
        )
        * V_dd_delta
        - (3**0.5) * 0.5 * N_t_L * (L2 + M2) * V_dd_delta_dxyz
    )
    H_Z2_X2Y2_dxyz = (
        (3**0.5)
        * 0.5
        * (
            2 * (L_t_Ldx - M_t_Mdx) * (N2 - 0.5 * (L2 + M2))
            + (L2 - M2) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx))
        )
        * V_dd_sigma
        + (3**0.5) * 0.5 * (L2 - M2) * (N2 - 0.5 * (L2 + M2)) * V_dd_sigma_dxyz
        + (3**0.5) * (2 * N_t_Ndx * (M2 - L2) + N2 * 2 * (M_t_Mdx - L_t_Ldx)) * V_dd_pi
        + (3**0.5) * N2 * (M2 - L2) * V_dd_pi_dxyz
        + (3**0.5)
        * 0.25
        * (2 * N_t_Ndx * (L2 - M2) + (1 + N2) * 2 * (L_t_Ldx - M_t_Mdx))
        * V_dd_delta
        + (3**0.5) * 0.25 * (1 + N2) * (L2 - M2) * V_dd_delta_dxyz
    )
    H_Z2_Z2_dxyz = (
        2 * (N2 - 0.5 * (L2 + M2)) * (2 * N_t_Ndx - (L_t_Ldx + M_t_Mdx)) * V_dd_sigma
        + (N2 - 0.5 * (L2 + M2)) ** 2 * V_dd_sigma_dxyz
        + 3 * (2 * N_t_Ndx * (L2 + M2) + N2 * 2 * (L_t_Ldx + M_t_Mdx)) * V_dd_pi
        + 3 * N2 * (L2 + M2) * V_dd_pi_dxyz
        + 0.75 * 2 * (L2 + M2) * 2 * (L_t_Ldx + M_t_Mdx) * V_dd_delta
        + 0.75 * (L2 + M2) ** 2 * V_dd_delta_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 4 + batch_block_offset, H_Z2_XY_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 5 + batch_block_offset, H_Z2_YZ_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 6 + batch_block_offset, H_Z2_ZX_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 7 + batch_block_offset, H_Z2_X2Y2_dxyz
    )
    dH0.index_add_(
        1, (idx_row + 8) * HDIM + idx_col + 8 + batch_block_offset, H_Z2_Z2_dxyz
    )

    """ 
    $ - from table
    O - derived via permutation

    S_XY = (3**0.5)*L*M*V_sd_sigma                                                                                                                         # $
    S_YZ = (3**0.5)*M*N*V_sd_sigma                                                                                                                         # O 
    S_ZX = (3**0.5)*N*L*V_sd_sigma                                                                                                                         # O
    S_X2Y2 = 0.5*(3**0.5)*(L**2 - M**2)*V_sd_sigma                                                                                                         # $
    S_Z2 = (N**2-0.5*(L**2 + M**2))*V_sd_sigma                                                                                                             # $

    X_XY = (3**0.5)*L**2*M*V_pd_sigma + M*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_YZ = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # $
    X_ZX = (3**0.5)*L**2*N*V_pd_sigma + N*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_X2Y2 = 0.5*(3**0.5)*L*(L**2 - M**2)*V_pd_sigma + L*(1 - L**2 + M**2)*V_pd_pi                                                                         # $
    X_Z2 = L*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*L*N**2*V_pd_pi                                                                                 # $

    Y_XY = (3**0.5)*M**2*L*V_pd_sigma + L*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_YZ = (3**0.5)*M**2*N*V_pd_sigma + N*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_ZX = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Y_X2Y2 = 0.5*(3**0.5)*M*(L**2 - M**2)*V_pd_sigma - M*(1 + L**2 - M**2)*V_pd_pi                                                                         # $
    Y_Z2 = M*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*M*N**2*V_pd_pi                                                                                 # $

    Z_XY = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Z_YZ = (3**0.5)*N**2*M*V_pd_sigma + M*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_ZX = (3**0.5)*N**2*L*V_pd_sigma + L*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_X2Y2 = 0.5*(3**0.5)*N*(L**2 - M**2)*V_pd_sigma - N*(L**2 - M**2)*V_pd_pi                                                                             # $
    Z_Z2 = N*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma + 3**0.5*N*(L**2 + M**2)*V_pd_pi                                                                        # $

    XY_S =  (3**0.5)*L*M*V_ds_sigma                                                                                                                        # O same as  S_XY
    YZ_S = (3**0.5)*M*N*V_ds_sigma                                                                                                                         # O same as  S_YZ
    ZX_S =  (3**0.5)*N*L*V_ds_sigma                                                                                                                        # O same as  S_ZX
    X2Y2_S =   0.5*(3**0.5)*(L**2 - M**2)*V_ds_sigma                                                                                                       # O same as  S_X2Y2
    Z2_S =  (N**2-0.5*(L**2 + M**2))*V_ds_sigma                                                                                                            # O same as  S_Z2

    XY_X = -((3**0.5)*L**2*M*V_dp_sigma + M*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_XY
    XY_Y = -((3**0.5)*M**2*L*V_dp_sigma + L*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_XY
    XY_Z = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Z_XY

    YZ_X = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -X_YZ
    YZ_Y = -((3**0.5)*M**2*N*V_dp_sigma + N*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_YZ
    YZ_Z = -((3**0.5)*N**2*M*V_dp_sigma + M*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_YZ

    ZX_X = -((3**0.5)*L**2*N*V_dp_sigma + N*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_ZX
    ZX_Y = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Y_ZX
    ZX_Z = -((3**0.5)*N**2*L*V_dp_sigma + L*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_ZX    

    X2Y2_X = -(0.5*(3**0.5)*L*(L**2 - M**2)*V_dp_sigma + L*(1 - L**2 + M**2)*V_dp_pi)                                                                      # O same as -X_X2Y2
    X2Y2_Y = -(0.5*(3**0.5)*M*(L**2 - M**2)*V_dp_sigma - M*(1 + L**2 - M**2)*V_dp_pi)                                                                      # O same as -Y_X2Y2       
    X2Y2_Z = -(0.5*(3**0.5)*N*(L**2 - M**2)*V_dp_sigma - N*(L**2 - M**2)*V_dp_pi)                                                                          # O same as -Z_X2Y2

    Z2_X = -(L*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*L*N**2*V_dp_pi)                                                                              # O same as -X_Z2
    Z2_Y = -(M*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*M*N**2*V_dp_pi)                                                                              # O same as -Y_Z2
    Z2_Z = -(N*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma + 3**0.5*N*(L**2 + M**2)*V_dp_pi)                                                                     # O same as -Z_Z2


    XY_XY = 3*L**2*M**2*V_dd_sigma + (L**2 + M**2 - 4*L**2*M**2)*V_dd_pi + (N**2 + L**2*M**2)*V_dd_delta                                                   # $
    XY_YZ = 3*L*M**2*N*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # $
    XY_ZX = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # $
    XY_X2Y2 = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # $
    XY_Z2 = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # $

    YZ_XY = 3*M**2*N*L*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # O
    YZ_YZ = 3*M**2*N**2*V_dd_sigma + (M**2 + N**2 - 4*M**2*N**2)*V_dd_pi + (L**2 + M**2*N**2)*V_dd_delta                                                   # O
    YZ_ZX = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O
    YZ_X2Y2 = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # $
    YZ_Z2 = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # $

    ZX_XY = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # O same as XY_ZX
    ZX_YZ = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O same as YZ_ZX
    ZX_ZX = 3*N**2*L**2*V_dd_sigma + (N**2 + L**2 - 4*N**2*L**2)*V_dd_pi + (M**2 + N**2*L**2)*V_dd_delta                                                   # $O
    ZX_X2Y2 = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # $
    ZX_Z2 = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # $

    X2Y2_XY = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # O same as  XY_X2Y2
    X2Y2_YZ = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  YZ_X2Y2
    X2Y2_ZX = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  ZX_X2Y2
    X2Y2_X2Y2 = 0.75*(L**2 - M**2)**2*V_dd_sigma + (L**2 + M**2 - (L**2 - M**2)**2)*V_dd_pi + (N**2 + 0.25*(L**2 - M**2)**2)*V_dd_delta                    # $
    X2Y2_Z2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # $

    Z2_XY = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # O same as  XY_Z2
    Z2_YZ = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # O same as  YZ_Z2
    Z2_ZX = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # O same as  ZX_Z2
    Z2_X2Y2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # O same as X2Y2_Z2
    Z2_Z2 = (N**2 - 0.5*(L**2 + M**2))**2*V_dd_sigma + 3*N**2*(L**2 + M**2)*V_dd_pi + 0.75*(L**2 + M**2)**2*V_dd_delta                                     # $


    """
    return H0, dH0


def Slater_Koster_Pair_vectorized(
    H0,
    HDIM,
    dR,
    dR_dxyz,
    L,
    M,
    N,
    L_dxyz,
    M_dxyz,
    N_dxyz,
    pair_mask_HH,
    pair_mask_HX,
    pair_mask_XH,
    pair_mask_XX,
    fss_sigma,
    fsp_sigma,
    fps_sigma,
    fpp_sigma,
    fpp_pi,
    neighbor_I,
    neighbor_J,
    nnType,
    H_INDEX_START,
    H_INDEX_END,
):
    """
    Compute the Slater-Koster matrix elements and their derivatives for pairs of atoms
    using vectorized operations for efficiency.

    This function evaluates the atomic block Hamiltonian matrix elements (H0) and their
    spatial derivatives (dH0) based on the Slater-Koster parameterization for s and p orbitals.
    It handles four types of atom pairs: H-H, H-X, X-H, and X-X, where H represents hydrogen-like atoms,
    and X represents other atom types with s and p orbitals.

    Parameters
    ----------
    H0 : torch.Tensor
        Preallocated 1D tensor containing the Hamiltonian matrix elements, flattened.
        Shape: (HDIM * HDIM,)

    HDIM : int
        Dimension of the atomic orbital block (e.g., 4 for sp orbitals).

    dR : torch.Tensor
        Tensor of interatomic distances for all pairs. Shape: (num_pairs,)

    dR_dxyz : torch.Tensor
        Tensor of derivatives of the interatomic distances with respect to Cartesian coordinates.
        Shape: (3, num_pairs), where axis 0 corresponds to x,y,z components.

    L, M, N : torch.Tensor
        Direction cosines (components of the unit vector pointing from one atom to another).
        Shape: (num_pairs,)

    L_dxyz, M_dxyz, N_dxyz : torch.Tensor
        Derivatives of the direction cosines with respect to Cartesian coordinates.
        Shape: (3, num_pairs)

    pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX : torch.BoolTensor
        Boolean masks selecting pairs belonging to each type:
        - H-H pairs
        - H-X pairs
        - X-H pairs
        - X-X pairs

    fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi : torch.Tensor
        Bond integral parameters for the Slater-Koster functions for each pair.
        Shape: (num_pairs,)

    neighbor_I, neighbor_J : torch.Tensor
        Indices of atoms in each pair.
        Shape: (num_pairs,)

    nnType : ?
        (Unused in this snippet — possibly the neighbor type or classification.)

    H_INDEX_START, H_INDEX_END : torch.Tensor or list
        Starting and ending indices in H0 corresponding to atomic orbitals of each atom.
        Used to place computed integrals into the correct positions in H0.

    Returns
    -------
    H0 : torch.Tensor
        Updated Hamiltonian matrix elements tensor with new values for the pairs processed.

    dH0 : torch.Tensor
        Derivatives of the Hamiltonian matrix elements with respect to Cartesian coordinates.
        Shape: (3, HDIM * HDIM)

    Notes
    -----
    - The function uses vectorized bond integral evaluations for improved computational efficiency.
    - The calculation covers both overlap and Hamiltonian matrix elements for s and p orbitals.
    - Direction cosine derivatives and bond integral derivatives are used to compute the gradients.
    - Periodic boundary conditions or lattice considerations are assumed handled externally.
    """
    # %%% Standard Slater-Koster sp-parameterization for an atomic block between a pair of atoms
    # %%% IDim, JDim: dimensions of the output block, e.g. 1 x 4 for H-O or 4 x 4 for O-O, or 4 x 1 for O-H
    # %%% Ra, Rb: are the vectors of the positions of the two atoms
    # %%% Type_pair(1 or 2): Character of the type of each atom in the pair, e.g. 'H' for hydrogen of 'O' for oxygen
    # %%% fss_sigma, ... , fpp_pi: paramters for the bond integrals
    # %%% diagonal(1 or 2): atomic energies Es and Ep or diagonal elements of the overlap i.e. diagonal = 1

    from ._bond_integral import (
        bond_integral_vectorized,
        bond_integral_with_grad_vectorized,
    )

    dH0 = torch.zeros(3, HDIM * HDIM, dtype=H0.dtype, device=H0.device)

    #######
    HSSS_all = bond_integral_vectorized(dR, fss_sigma)
    H0[H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J]] = HSSS_all

    #######

    # H-H
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR, fss_sigma)
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0[:, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J]] = HSSS_dxyz
    #########

    # H-X
    ###### HSPS_all
    idx_row = H_INDEX_START[neighbor_I[pair_mask_HX + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_HX + pair_mask_XX]]
    HSPS_all = bond_integral_vectorized(
        dR[pair_mask_HX + pair_mask_XX], fsp_sigma[pair_mask_HX + pair_mask_XX]
    )

    H0[idx_row * HDIM + idx_col + 1] = L[pair_mask_HX + pair_mask_XX] * HSPS_all
    H0[idx_row * HDIM + idx_col + 2] = M[pair_mask_HX + pair_mask_XX] * HSPS_all
    H0[idx_row * HDIM + idx_col + 3] = N[pair_mask_HX + pair_mask_XX] * HSPS_all
    ######### dH/dx
    HSPS_dR = bond_integral_with_grad_vectorized(
        dR[pair_mask_HX + pair_mask_XX], fsp_sigma[pair_mask_HX + pair_mask_XX]
    )
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, pair_mask_HX + pair_mask_XX]

    dH0[:, idx_row * HDIM + idx_col + 1] = (
        L[pair_mask_HX + pair_mask_XX] * HSPS_dxyz
        + L_dxyz[:, pair_mask_HX + pair_mask_XX] * HSPS_all
    )
    dH0[:, idx_row * HDIM + idx_col + 2] = (
        M[pair_mask_HX + pair_mask_XX] * HSPS_dxyz
        + M_dxyz[:, pair_mask_HX + pair_mask_XX] * HSPS_all
    )
    dH0[:, idx_row * HDIM + idx_col + 3] = (
        N[pair_mask_HX + pair_mask_XX] * HSPS_dxyz
        + N_dxyz[:, pair_mask_HX + pair_mask_XX] * HSPS_all
    )
    #########

    ### HPSS_all ###
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XH + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XH + pair_mask_XX]]
    HPSS_all = bond_integral_vectorized(
        dR[pair_mask_XH + pair_mask_XX], fps_sigma[pair_mask_XH + pair_mask_XX]
    )
    H0[(idx_row + 1) * HDIM + idx_col] = -L[pair_mask_XH + pair_mask_XX] * HPSS_all
    H0[(idx_row + 2) * HDIM + idx_col] = -M[pair_mask_XH + pair_mask_XX] * HPSS_all
    H0[(idx_row + 3) * HDIM + idx_col] = -N[pair_mask_XH + pair_mask_XX] * HPSS_all
    ################
    ######### dH/dx
    HPSS_dR = bond_integral_with_grad_vectorized(
        dR[pair_mask_XH + pair_mask_XX], fps_sigma[pair_mask_XH + pair_mask_XX]
    )
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, pair_mask_XH + pair_mask_XX]

    dH0[:, (idx_row + 1) * HDIM + idx_col] = (
        -L[pair_mask_XH + pair_mask_XX] * HPSS_dxyz
        - L_dxyz[:, pair_mask_XH + pair_mask_XX] * HPSS_all
    )
    dH0[:, (idx_row + 2) * HDIM + idx_col] = (
        -M[pair_mask_XH + pair_mask_XX] * HPSS_dxyz
        - M_dxyz[:, pair_mask_XH + pair_mask_XX] * HPSS_all
    )
    dH0[:, (idx_row + 3) * HDIM + idx_col] = (
        -N[pair_mask_XH + pair_mask_XX] * HPSS_dxyz
        - N_dxyz[:, pair_mask_XH + pair_mask_XX] * HPSS_all
    )
    #########

    # X-X
    L_XX = L[pair_mask_XX]
    M_XX = M[pair_mask_XX]
    N_XX = N[pair_mask_XX]
    dR_XX = dR[pair_mask_XX]
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XX]]
    HPPS = bond_integral_vectorized(dR_XX, fpp_sigma[pair_mask_XX])
    HPPP = bond_integral_vectorized(dR_XX, fpp_pi[pair_mask_XX])
    PPSMPP = HPPS - HPPP
    PXPX = HPPP + L_XX * L_XX * PPSMPP
    PXPY = L_XX * M_XX * PPSMPP
    PXPZ = L_XX * N_XX * PPSMPP
    PYPX = M_XX * L_XX * PPSMPP
    PYPY = HPPP + M_XX * M_XX * PPSMPP
    PYPZ = M_XX * N_XX * PPSMPP
    PZPX = N_XX * L_XX * PPSMPP
    PZPY = N_XX * M_XX * PPSMPP
    PZPZ = HPPP + N_XX * N_XX * PPSMPP

    H0[(idx_row + 1) * HDIM + idx_col + 1] = PXPX
    H0[(idx_row + 1) * HDIM + idx_col + 2] = PXPY
    H0[(idx_row + 1) * HDIM + idx_col + 3] = PXPZ

    ####

    H0[(idx_row + 2) * HDIM + idx_col + 1] = PYPX
    H0[(idx_row + 2) * HDIM + idx_col + 2] = PYPY
    H0[(idx_row + 2) * HDIM + idx_col + 3] = PYPZ

    ####

    H0[(idx_row + 3) * HDIM + idx_col + 1] = PZPX
    H0[(idx_row + 3) * HDIM + idx_col + 2] = PZPY
    H0[(idx_row + 3) * HDIM + idx_col + 3] = PZPZ

    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, pair_mask_XX]
    L_dxyz_XX = L_dxyz[:, pair_mask_XX]
    M_dxyz_XX = M_dxyz[:, pair_mask_XX]
    N_dxyz_XX = N_dxyz[:, pair_mask_XX]

    HPPS_dR = bond_integral_with_grad_vectorized(dR_XX, fpp_sigma[pair_mask_XX])
    HPPS_dxyz = HPPS_dR * dR_dxyz_XX

    HPPP_dR = bond_integral_with_grad_vectorized(dR_XX, fpp_pi[pair_mask_XX])
    HPPP_dxyz = HPPP_dR * dR_dxyz_XX

    PPSMPP_dxyz = HPPS_dxyz - HPPP_dxyz
    PXPX_dxyz = HPPP_dxyz + (L_XX**2) * PPSMPP_dxyz + 2 * L_XX * L_dxyz_XX * PPSMPP
    PXPY_dxyz = (
        L_XX * M_XX * PPSMPP_dxyz
        + L_dxyz_XX * M_XX * PPSMPP
        + L_XX * M_dxyz_XX * PPSMPP
    )
    PXPZ_dxyz = (
        L_XX * N_XX * PPSMPP_dxyz
        + L_dxyz_XX * N_XX * PPSMPP
        + L_XX * N_dxyz_XX * PPSMPP
    )
    PYPX_dxyz = (
        M_XX * L_XX * PPSMPP_dxyz
        + M_XX * L_dxyz_XX * PPSMPP
        + M_dxyz_XX * L_XX * PPSMPP
    )
    PYPY_dxyz = HPPP_dxyz + (M_XX**2) * PPSMPP_dxyz + 2 * M_XX * M_dxyz_XX * PPSMPP
    PYPZ_dxyz = (
        M_XX * N_XX * PPSMPP_dxyz
        + M_dxyz_XX * N_XX * PPSMPP
        + M_XX * N_dxyz_XX * PPSMPP
    )
    PZPX_dxyz = (
        N_XX * L_XX * PPSMPP_dxyz
        + N_XX * L_dxyz_XX * PPSMPP
        + N_dxyz_XX * L_XX * PPSMPP
    )
    PZPY_dxyz = (
        N_XX * M_XX * PPSMPP_dxyz
        + N_XX * M_dxyz_XX * PPSMPP
        + N_dxyz_XX * M_XX * PPSMPP
    )
    PZPZ_dxyz = HPPP_dxyz + (N_XX**2) * PPSMPP_dxyz + 2 * N_XX * N_dxyz_XX * PPSMPP

    ####

    dH0[:, (idx_row + 1) * HDIM + idx_col + 1] = PXPX_dxyz
    dH0[:, (idx_row + 1) * HDIM + idx_col + 2] = PXPY_dxyz
    dH0[:, (idx_row + 1) * HDIM + idx_col + 3] = PXPZ_dxyz

    ####

    dH0[:, (idx_row + 2) * HDIM + idx_col + 1] = PYPX_dxyz
    dH0[:, (idx_row + 2) * HDIM + idx_col + 2] = PYPY_dxyz
    dH0[:, (idx_row + 2) * HDIM + idx_col + 3] = PYPZ_dxyz

    ####

    dH0[:, (idx_row + 3) * HDIM + idx_col + 1] = PZPX_dxyz
    dH0[:, (idx_row + 3) * HDIM + idx_col + 2] = PZPY_dxyz
    dH0[:, (idx_row + 3) * HDIM + idx_col + 3] = PZPZ_dxyz
    #########
    return H0, dH0

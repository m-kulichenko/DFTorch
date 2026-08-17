# =============================================================================
# WHAT THIS FILE IS FOR (read this first; everything below assumes it)
# =============================================================================
#
# The goal.  We are solving for the electrons in a molecule or crystal using a
# cheap approximation to density functional theory.  "Cheap" means we never
# compute integrals on the fly: we look them up.  Two matrices have to be built
# before anything else can happen:
#
#   H0 -- the Hamiltonian.  Entry H0[a, b] is the energy coupling between
#         atomic orbital a and atomic orbital b.  Big magnitude = the two
#         orbitals mix strongly and share electrons.
#   S  -- the overlap.  Entry S[a, b] is how much orbital a and orbital b
#         occupy the same region of space.  Orbitals on the *same* atom are
#         built to be orthogonal (S = 1 on the diagonal, 0 off it), but
#         orbitals on *different* atoms are not, so S is not the identity and
#         cannot be ignored.
#
# Both matrices have exactly the same shape and are built by exactly the same
# code here; the ``SH_shift`` argument threaded through every function is the
# single switch that says "build H" (``"H"``) or "build S" (``"S"``).  Read
# "H0" below as "H0 or S, whichever was asked for".
#
# The Slater-Koster idea.  H0[a, b] for orbital a on atom I and orbital b on
# atom J is really a three-dimensional integral over all space, and it depends
# on where every *other* atom sits too.  Slater and Koster's approximation
# throws that last part away (the "two-centre approximation"): pretend the
# integral depends only on atoms I and J.  Once you do that, the value can only
# depend on two things:
#
#   1. R, the distance between the two atoms -- a single number.
#   2. The *direction* of the bond in space.
#
# and, crucially, those two dependencies factorise.  Every matrix element is
#
#       H0[a, b] = sum over bond symmetries k of
#                     ( angular polynomial in the bond direction )
#                   * ( radial function of R )
#
# The radial functions are tabulated (a few hundred numbers per element pair,
# read from a file), and the angular polynomials are exact closed-form
# expressions that this file hard-codes.  That is the whole trick, and it is
# why this file is mostly long algebraic expressions.
#
# Direction cosines L, M, N.  The "bond direction" is encoded as the unit
# vector pointing from atom I to atom J:
#
#       (L, M, N) = (R_J - R_I) / |R_J - R_I|,   so   L^2 + M^2 + N^2 = 1.
#
# L, M, N are the cosines of the angles that vector makes with the x, y and z
# axes, hence "direction cosines".  Every angular polynomial in this file is
# written purely in terms of L, M and N.
#
# Bond symmetries: sigma, pi, delta, phi.  Take the line joining the two atoms
# as an axis.  Each pair of orbitals can couple in several distinct ways,
# classified by how much the pair rotates when you spin it about that axis:
#
#       sigma (index 0) -- no change: head-on, lobes pointing along the bond.
#       pi    (index 1) -- one full sign flip per turn: side-by-side lobes.
#       delta (index 2) -- two: cloverleaf-on-cloverleaf.
#       phi   (index 3) -- three: only reachable once f orbitals are involved.
#
# A pair of shells with angular momenta l1 and l2 has exactly
# min(l1, l2) + 1 of these, so s-s has one (sigma), p-d has two (sigma, pi),
# d-d has three, f-f has four.  Each one gets its own tabulated radial
# function.
#
# Channels.  A "channel" names one such radial function.  The naming used
# throughout is <prefix><shell1><shell2><symmetry index>, e.g.
#
#       "Hss0"  Hamiltonian, s with s, sigma
#       "Spd1"  overlap,     p with d, pi
#       "Hff3"  Hamiltonian, f with f, phi
#
# The leading "H" or "S" picks the Hamiltonian or overlap half; the rest
# ("ss0", "pd1", "ff3") is called the *base* channel name in this file.
#
# Orbital ordering inside one atom.  Every atom contributes a contiguous run of
# atomic orbitals to the matrix, always in this fixed order:
#
#       local offset 0        s
#       local offsets 1,2,3   px, py, pz
#       local offsets 4..8    dxy, dyz, dzx, dx2-y2, dz2
#       local offsets 9..15   the seven f orbitals (see STRUCTURE_F_AO_ORDER)
#
# An atom stops early depending on which shells its element actually has, so an
# atom carries 1, 4, 9 or 16 orbitals.  Those four sizes are abbreviated with
# single letters in the pair masks below: H (1 orbital, hydrogen-like), X (4),
# Y (9), Z (16).  A mask named ``pair_mask_YZ`` selects neighbour pairs whose
# first atom has 9 orbitals and whose second has 16.
#
# How results are stored.  H0 is returned *flattened*: a 1-D tensor of length
# HDIM * HDIM standing for an HDIM-by-HDIM matrix in row-major order, so
# matrix entry (row, col) lives at flat position ``row * HDIM + col``.
# ``H_INDEX_START[k]`` gives the first orbital index of atom k, so the code you
# will see over and over,
#
#       H0.index_add_(0, (i0 + a) * HDIM + j0 + b, values)
#
# means "add ``values`` into the matrix entry linking local orbital a of the
# first atom to local orbital b of the second".  ``index_add_`` is used rather
# than plain assignment because several masks can touch the same entry and
# their contributions must *sum*, not overwrite each other.
#
# Derivatives.  Alongside H0 the routines return dH0, the derivative of every
# matrix entry with respect to the x, y and z coordinates of the atoms.  These
# are what forces and stresses are built from.  Since each entry is
# (angular in L, M, N) * (radial in R), its derivative is just the product rule,
# which is why nearly every angular expression below is immediately followed by
# a longer one containing terms like ``L_dxyz`` (dL/dxyz) and ``V..._dxyz``
# (dV/dR * dR/dxyz).  That mechanical repetition is the bulk of this file.
#
# What lives here, in order:
#   1. Channel-name bookkeeping (which radial function is at which index).
#   2. The f-orbital angular tables, plus the two named errors that mark what
#      is deliberately not supported.
#   3. Slater_Koster_Pair_SKF_vectorized      -- the main assembler.
#   4. Slater_Koster_Pair_SKF_vectorized_batch -- the same, many structures at
#      once.
#   5. Slater_Koster_Pair_vectorized          -- an older, s/p-only assembler.
# =============================================================================

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

import torch

# ``_CHANNELS`` is the single authoritative list of channel names, in the exact
# order the tabulated radial coefficients are stored in.  It is 40 entries
# long: the 20 Hamiltonian channels ("Hff0", "Hff1", ... "Hss0") followed by
# the 20 matching overlap channels ("Sff0", ... "Sss0").  Importing it (rather
# than restating the order here) is what guarantees the two files agree.
from ._bond_integral import _CHANNELS as _BOND_INTEGRAL_CHANNELS #Extended SKF Format

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
# name via :func:`sk_channel_index` / :func:`sk_channel_name` instead, and note
# that ``SH_shift`` is now the literal prefix letter ``"H"`` or ``"S"`` rather
# than a number that could be added to anything.
# All 40 channel names, in storage order.  Position i in this tuple is the
# index you feed to ``coeffs_tensor``'s channel axis to get that radial
# function's spline coefficients.

SK_CHANNEL_NAMES: Final[tuple[str, ...]] = tuple(_BOND_INTEGRAL_CHANNELS) #Type-specified immutable tuple of all the X.SKF channels
#Index to channel

# The same information as a lookup the other way: name -> storage index.
# ``MappingProxyType`` makes it read-only, so nothing downstream can mutate the
# channel numbering by accident.
SK_CHANNEL_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(SK_CHANNEL_NAMES)} #MappingProxyType makes it read only
) #channel to index

# Base (shell-pair) channel names shared by the Hamiltonian and overlap blocks.
# Prefixing with "H" or "S" produces a key of ``SK_CHANNEL_INDEX``.
# Concretely: strip the leading "H" off every Hamiltonian name, giving
# ("ff0", "ff1", "ff2", "ff3", "df0", ... "sp0", "ss0") -- the 20 physically
# distinct radial functions, each of which exists in an H flavour and an S
# flavour.  Callers pass these bare names around and let ``SH_shift`` decide
# which flavour to actually read.
SK_BASE_CHANNEL_NAMES: Final[tuple[str, ...]] = tuple(
    name[1:] for name in SK_CHANNEL_NAMES if name.startswith("H")
) #Gets the ff0, ff1 names in the right order for doing string concat. 

# Legacy 10-channel indices still used by the ML Slater-Koster head
# (``_ml_sk._CHANNEL_MAP``).  The ML path is kept on its own numbering so its
# public behavior is unchanged; only the spline path moves to named channels.
#
# Background: as well as reading radial functions from a tabulated file, this
# code can get them from a trained neural network ("the ML head").  That
# network was trained back when only ten s/p/d channels existed and it emits
# its outputs in that old 0..9 order, which is unrelated to the 40-entry order
# above.  Rather than retrain it, the lookup below translates a modern channel
# name into the output slot the network actually uses.  There is deliberately
# no f entry: the network was never trained on f orbitals, so asking it for one
# raises instead of silently returning an unrelated s/p/d prediction.
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


def sk_channel_name(base: str, SH_shift: str) -> str: #Concat function
    """Return the canonical channel name for ``base`` in the "H" or "S" block.

    Parameters
    ----------
    base : str
        Shell-pair channel without its H/S prefix, e.g. ``"ss0"``, ``"pd1"``,
        ``"ff3"``.
    SH_shift : str
        ``"H"`` selects the Hamiltonian block, ``"S"`` the overlap block.  It
        is the prefix letter itself, so the call reads the way the resulting
        channel name does.
    """
    # Pure string concatenation: "ss0" prefixed by "H" -> "Hss0".  The check is
    # explicit so that a nonsensical SH_shift ("h", 0, None, ...) raises here
    # rather than building a channel name that no lookup will recognise.
    if SH_shift not in ("H", "S"):
        raise ValueError(
            f"SH_shift must be 'H' (Hamiltonian) or 'S' (overlap), "
            f"got {SH_shift!r}"
        )
    return f"{SH_shift}{base}"


def sk_channel_index(name: str) -> int: #guarded lookup for the index based on Hsp1 or whatever
    """Return the ``coeffs_tensor`` channel index for a canonical channel name."""
    # A plain dict lookup, wrapped only to improve the error.  A typo like
    # "Hsp1" (which is not a real channel -- s-p has no pi component) would
    # otherwise surface as a bare KeyError with no hint of what was expected,
    # so the valid list is spelled out in the message.  ``from None`` hides the
    # inner KeyError so the reader sees one clear error, not a chained pair.
    try:
        return SK_CHANNEL_INDEX[name]
    except KeyError:
        raise KeyError(
            f"Unknown Slater-Koster channel {name!r}. Valid channels are: "
            f"{', '.join(SK_CHANNEL_NAMES)}"
        ) from None


# ---------------------------------------------------------------------------
# f-orbital angular convention
# ---------------------------------------------------------------------------
# The local AO order is fixed by ``Structure.AO_LABEL_TEMPLATE`` and must not be
# reordered; f orbitals occupy local offsets 9..15.
#
# Why an order has to be pinned at all: an f shell holds seven orbitals, and
# which seven real-valued combinations you use, and in what sequence, is a
# convention.  Every other part of the program -- the matrix layout, the
# per-orbital charges, the output files -- counts on this exact sequence, so it
# is stated once here and everything else is bent to match it.
#
# The names are shorthand for the polynomial shape of each orbital:
#   "fx3"       x(5x^2 - 3r^2)   a lobe along x with a collar around it
#   "fy3"       y(5y^2 - 3r^2)   same, along y
#   "fz3"       z(5z^2 - 3r^2)   same, along z
#   "fx_y2_z2"  x(y^2 - z^2)     positive toward +-y, negative toward +-z
#   "fy_z2_x2"  y(z^2 - x^2)     same shape, cycled one axis on
#   "fz_x2_y2"  z(x^2 - y^2)     same shape, cycled again
#   "fxyz"      xyz              eight lobes pointing at the cube corners
STRUCTURE_F_AO_ORDER: Final[tuple[str, ...]] = (
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
) #Expected f orbital order by the X.skf files


# ---------------------------------------------------------------------------
# The two f-unsupported errors, and the one idea behind both of them
# ---------------------------------------------------------------------------
# Both classes exist for the same reason, so it is worth stating it once.  A
# matrix full of zeros is indistinguishable from a matrix full of genuinely-zero
# physics.  If f orbitals are quietly skipped somewhere, the calculation still
# runs, still converges, and still prints an energy -- one that is simply wrong,
# with nothing to indicate it.  Each class below marks a place where that could
# happen, and converts "silently wrong" into "stops with an explanation".  They
# are kept together, in this one file, so the list of what f support does *not*
# cover can be read in one go:
#
#   FDerivativeUnsupportedError           no f force/stress gradients
#   FSpinPolarizationUnsupportedError     no open-shell f
#
# Both subclass NotImplementedError, so a caller that wants to catch "this f
# feature is missing" generically can catch that one built-in type.


class FDerivativeUnsupportedError(NotImplementedError):
    """Raised when a path would consume f-orbital H0/S *derivatives*.

    The f angular values are implemented but their Cartesian derivatives are
    not.  ``dH0``/``dS`` therefore carry exact zeros in every f-containing
    block.  Zero is a legal-looking derivative, so any consumer (forces,
    stress, MD, geometry optimisation) must fail loudly rather than integrate a
    silently wrong gradient.
    """


F_DERIVATIVE_UNSUPPORTED_MESSAGE: Final[str] = (
    "f-orbital Slater-Koster angular derivatives are not implemented.\n"
    "Only the f angular *values* exist, so dH0/dS are exactly zero inside "
    "every f block. That is indistinguishable from a real vanishing gradient, "
    "which is why this path refuses to run instead of returning an "
    "f-incomplete result.\n"
    "Use energies/H0/S for f-containing systems; forces, stress and MD for "
    "those systems require the deferred f derivative work."
)

#: ``True`` only once f-orbital dH0/dS are actually *populated*.
#
# Read this flag carefully before changing it, because what it means is narrower
# than it sounds.  The f angular derivative tables now exist further down this
# module (``f_angular_sf_grad`` and friends) and are validated against automatic
# differentiation, finite differences and the differentiated orthogonality
# relations.  That is NOT what this flag reports.
#
# The flag reports whether the assembler below actually uses them.  It does not:
# the f section writes into H0 only and never touches dH0, so every f entry of
# dH0/dS is still exactly zero.  Flipping this to ``True`` while that is the case
# would switch off the guards that currently refuse forces, stress and MD for f
# systems, and those paths would then integrate a whole block of zero gradients
# without complaint -- finite, plausible, and wrong.
#
# The remaining work is to consume the tables: chain them through L_dxyz, M_dxyz
# and N_dxyz for the angular half, keep the radial derivative that
# ``_get_val_dR`` already returns and that the f block currently discards, add
# both into dH0, and feed the same result to ``_sg`` for stress.  Flip this only
# once that is done and validated against finite differences of the total
# energy, which is the test that actually proves forces are right.
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


# ---------------------------------------------------------------------------
# f angular blocks
# ---------------------------------------------------------------------------
# Everything below is written directly in ``STRUCTURE_F_AO_ORDER``: index 0 is
# fx3, 1 is fy3, 2 is fz3, 3 is fx_y2_z2, 4 is fy_z2_x2, 5 is fz_x2_y2 and 6 is
# fxyz.  Each entry carries a short comment naming the orbital it belongs to, so
# no reordering step is needed anywhere and the indices mean the same thing here
# as they do in the rest of the program.
#
# Only about a third of the entries are written out.  The rest follow from the
# completeness rule: entries not given can be found by cyclically permuting the
# coordinates and direction cosines.  The cyclic operator is the proper rotation
# x -> y -> z -> x.  A vector with direction cosines (l, m, n) maps to (n, l, m)
# under it, so for orbitals A, B and their images sA, sB a written entry
# generates
#
#       E_{sA,sB}(l, m, n) = E_{A,B}(m, n, l).
#
# Correctness gate: substituting 1 for every two-centre integral of a shell pair
# must return the identity.  Operationally this makes each channel coefficient
# matrix an orthogonal projector, and it links the s-f/p-f/d-f tables to the f-f
# table.  ``tests/test_f_orbital_skf.py`` runs that gate over random unit
# vectors; do not weaken it to accommodate a formula, fix the formula instead.
#
# The d order used here is (xy, yz, zx, x^2-y^2, 3z^2-r^2) and the p order is
# (x, y, z), both matching ``Structure.AO_LABEL_TEMPLATE``.

# Irrational prefactors that appear all over the tables.  They are the
# normalisation constants of the cubic harmonics, precomputed once as plain
# Python floats so the hot formulas below multiply by a number instead of
# recomputing a square root on every call.  ``_SQRT_3_8`` reads "square root of
# three eighths"; ``_SQRT15`` reads "square root of fifteen".
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
# These three only show up once the tables above are differentiated: the
# compound constants collapse to them under the product rule, e.g. sqrt(3/8)
# becomes sqrt(6)/4 and sqrt(15/2) becomes sqrt(30)/2.
_SQRT6: Final[float] = 6.0**0.5
_SQRT10: Final[float] = 10.0**0.5
_SQRT30: Final[float] = 30.0**0.5

# The three tables below encode "what does each orbital turn into when you
# relabel the axes x -> y -> z -> x?".  That relabelling is a rigid 120-degree
# rotation about the cube diagonal, and it is the whole reason only about a
# third of the entries have to be written out: rotating a written entry gives a
# genuine, exact new entry for free.  Each tuple answers "orbital i becomes
# orbital cycle[i]".

#: Image of each f index under x -> y -> z -> x, in ``STRUCTURE_F_AO_ORDER``.
#: 0 -> 1 -> 2 -> 0 is fx3 -> fy3 -> fz3 chasing its own tail, and
#: 3 -> 4 -> 5 -> 3 is the same for fx_y2_z2 -> fy_z2_x2 -> fz_x2_y2.
#: 6 (fxyz) stays 6 -- relabelling the axes leaves xyz alone.
_F_CYCLE: Final[tuple[int, ...]] = (1, 2, 0, 4, 5, 3, 6)
#: Image of each p index (x, y, z).
#: px -> py -> pz -> px, i.e. 0 -> 1 -> 2 -> 0, written as "index 0 becomes 1,
#: index 1 becomes 2, index 2 becomes 0".
_P_CYCLE: Final[tuple[int, ...]] = (1, 2, 0)
#: Image of the three d indices (xy, yz, zx): xy -> yz -> zx -> xy.  The two
#: left out are dx2-y2 and dz2: rotating those produces mixtures of each other
#: rather than a single orbital, so the trick does not apply and their rows have
#: to be written out in full further below.
_D_T2G_CYCLE: Final[tuple[int, ...]] = (1, 2, 0)


def _cycle_index(cycle: tuple[int, ...], index: int, times: int) -> int:
    # Follow the relabelling arrow ``times`` times: applying it 0 times is the
    # identity, 3 times returns to the start (it is a 120-degree rotation), so
    # in practice ``times`` is only ever 0, 1 or 2.
    for _ in range(times):
        index = cycle[index]
    return index


def _cycle_dirs(times: int, L, M, N):
    """(l, m, n) -> (m, n, l), applied ``times`` times."""
    # The same rotation applied to the bond direction rather than to an orbital
    # label.  Both halves must move together: a printed formula stays true only
    # if you relabel the orbitals *and* feed it the correspondingly relabelled
    # direction cosines.
    for _ in range(times):
        L, M, N = M, N, L
    return L, M, N


def _uncycle_grad(times: int, gL, gM, gN):
    """Turn a rotated row's own partials back into partials in L, M and N.

    The rotation trick feeds a written formula the relabelled direction cosines
    ``_cycle_dirs(times, L, M, N)`` and files the answer under a rotated cell
    index.  A derivative has to be carried back the other way.

    Write the rotated arguments as (a1, a2, a3).  For ``times`` = 0 that is
    (L, M, N); for 1 it is (M, N, L); for 2 it is (N, L, M).  A row-gradient
    function hands back (dE/da1, dE/da2, dE/da3), so the question is only "which
    argument slot was L sitting in?".  At times = 1, L occupies slot 3, so
    dV/dL is the third component; M occupies slot 1 and N slot 2.  That is the
    shift below, and applying it ``times`` times undoes the same number of
    forward rotations.
    """
    for _ in range(times):
        gL, gM, gN = gN, gL, gM
    return gL, gM, gN


def _assemble_block(
    cells: dict, n_chan: int, n_row: int, n_col: int, like: torch.Tensor
) -> torch.Tensor:
    """Stack ``{(row, col): [per-channel tensor]}`` into ``(chan, row, col, P)``."""
    # ``like`` is currently unused: the dtype and device are inherited from the
    # cell tensors themselves.  It is kept in the signature as the obvious hook
    # for constructing zeros of the right kind, should a future block ever be
    # legitimately sparse.
    # The generation above fills a dictionary one (row, column) cell at a time
    # and in scattered order.  This turns that dictionary into a dense tensor,
    # but first checks that every cell really got filled.
    #
    # That check is the safety net for the whole cyclic-permutation scheme: if
    # a printed entry were mistyped, or a cycle table were wrong, some cells
    # would simply never be written and the resulting block would contain holes
    # -- which, once stacked, would look like perfectly ordinary zeros.  Failing
    # loudly here is the difference between a caught transcription bug and a
    # quietly wrong Hamiltonian.
    missing = [
        (a, b) for a in range(n_row) for b in range(n_col) if (a, b) not in cells
    ]
    if missing:
        raise AssertionError(
            f"f angular block is incomplete; the cyclic-permutation rule did "
            f"not cover {missing}"
        )
    # Three nested stacks build the axes from the inside out: innermost over
    # columns b, then over rows a, then over bond symmetries k -- giving shape
    # (n_chan, n_row, n_col, P), where the trailing P is the number of atom
    # pairs being processed at once and rides along inside each cell tensor.
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


# --------------------------------------------------------------------- s-f
# An s orbital paired with an f orbital: one row (s has a single orbital),
# seven columns (the f shell), and only one bond symmetry is possible.  An s
# orbital is a featureless sphere, so it has no angular momentum to contribute
# about the bond axis and the pair can only couple head-on -- sigma, the single
# channel "sf0".  Hence every cell below is a one-element list.
def _sf_row(L, M, N) -> dict:
    """The three written s-f entries; the other four follow by rotation."""
    # Keys are (row, column) = (the single s orbital, the f orbital).  Only
    # three of the seven columns appear because one representative of each of
    # the three orbital families is written and the rest come from the rotation
    # trick.
    return {
        (0, 0): [0.5 * L * (5 * L * L - 3)],  # fx3
        (0, 3): [0.5 * _SQRT15 * L * (M * M - N * N)],  # fx_y2_z2
        (0, 6): [_SQRT15 * L * M * N],  # fxyz
    }


def _sf_block(L, M, N) -> torch.Tensor:
    # Turn those three entries into all seven.  For each of the three rotations
    # t = 0, 1, 2: feed the row the rotated direction cosines, and file each
    # result under the rotated column index.  t = 0 reproduces the written
    # entries verbatim; t = 1 and t = 2 generate the four that were left out.
    # The s row index needs no rotation -- there is only one s orbital and
    # rotating the axes does not move it.
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _sf_row(*args).items():
            cells[(a, _cycle_index(_F_CYCLE, b, t))] = value
    # 1 bond symmetry (sigma), 1 s row, 7 f columns.
    return _assemble_block(cells, 1, 1, 7, L)


# --------------------------------------------------------------------- p-f
# A p orbital paired with an f orbital: three rows (px, py, pz), seven columns,
# and now *two* bond symmetries are possible, sigma and pi, because a p orbital
# can point along the bond (head-on, sigma) or across it (side-by-side, pi).
# Every cell below is therefore a two-element list, in the fixed order
# [sigma, pi] -- matching channels "pf0" and "pf1".
def _pf_row(L, M, N) -> dict:
    """The p-f row E_{x, .} for all seven f columns."""
    # Only the px row is written out; the py and pz rows come from rotating it.
    # Squares of the direction cosines are named once and reused, both for
    # readability and to avoid recomputing them a dozen times per entry.
    l2, m2, n2 = L * L, M * M, N * N
    return {
        (0, 0): [  # fx3
            0.5 * l2 * (5 * l2 - 3),
            -_SQRT_3_8 * (5 * l2 - 1) * (l2 - 1),
        ],
        (0, 1): [  # fy3
            0.5 * L * M * (5 * m2 - 3),
            -_SQRT_3_8 * L * M * (5 * m2 - 1),
        ],
        (0, 2): [  # fz3
            0.5 * L * N * (5 * n2 - 3),
            -_SQRT_3_8 * L * N * (5 * n2 - 1),
        ],
        (0, 3): [  # fx_y2_z2
            0.5 * _SQRT15 * l2 * (m2 - n2),
            -_SQRT_5_8 * (3 * l2 - 1) * (m2 - n2),
        ],
        (0, 4): [  # fy_z2_x2
            0.5 * _SQRT15 * L * M * (n2 - l2),
            -_SQRT_5_8 * L * M * (3 * (n2 - l2) + 2),
        ],
        (0, 5): [  # fz_x2_y2
            0.5 * _SQRT15 * L * N * (l2 - m2),
            -_SQRT_5_8 * L * N * (3 * (l2 - m2) - 2),
        ],
        (0, 6): [  # fxyz
            _SQRT15 * l2 * M * N,
            -_SQRT_5_2 * (3 * l2 - 1) * M * N,
        ],
    }


def _pf_block(L, M, N) -> torch.Tensor:
    # Same generation as s-f, except that now *both* indices rotate: the p row
    # moves px -> py -> pz as well as the f column moving.  Three rotations of
    # a single written row of seven therefore fill all 3 x 7 = 21 cells.
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _pf_row(*args).items():
            cells[
                (
                    _cycle_index(_P_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = value
    # 2 bond symmetries (sigma, pi), 3 p rows, 7 f columns.
    return _assemble_block(cells, 2, 3, 7, L)


# --------------------------------------------------------------------- d-f
# A d orbital paired with an f orbital: five rows (dxy, dyz, dzx, dx2-y2, dz2),
# seven columns, three bond symmetries -- sigma, pi and delta, since a d
# orbital's four-lobed shape can also couple in the two-sign-flips-per-turn
# way.  Every cell is a three-element list [sigma, pi, delta], matching
# channels "df0", "df1", "df2".
#
# The five d rows split into two groups that must be handled differently.  The
# first three (dxy, dyz, dzx) rotate into one another under x -> y -> z -> x,
# so one written row generates all three.  The last two (dx2-y2 and dz2) do
# not: rotating them produces mixtures of the pair rather than a single
# orbital, so those two rows are written out in full in the two functions after
# this one.
def _df_row_xy(L, M, N) -> dict:
    """The d-f row E_{xy, .}; the yz and zx rows follow by cyclic rotation."""
    l2, m2, n2 = L * L, M * M, N * N
    return {
        (0, 0): [  # fx3
            0.5 * _SQRT3 * l2 * M * (5 * l2 - 3),
            -_SQRT_3_8 * M * (5 * l2 - 1) * (2 * l2 - 1),
            0.5 * _SQRT15 * l2 * M * (l2 - 1),
        ],
        (0, 1): [  # fy3
            0.5 * _SQRT3 * L * m2 * (5 * m2 - 3),
            -_SQRT_3_8 * L * (5 * m2 - 1) * (2 * m2 - 1),
            0.5 * _SQRT15 * L * m2 * (m2 - 1),
        ],
        (0, 2): [  # fz3
            0.5 * _SQRT3 * L * M * N * (5 * n2 - 3),
            -_SQRT_3_2 * L * M * N * (5 * n2 - 1),
            0.5 * _SQRT15 * L * M * N * (n2 + 1),
        ],
        (0, 3): [  # fx_y2_z2
            1.5 * _SQRT5 * l2 * M * (m2 - n2),
            -_SQRT_5_8 * M * ((6 * l2 - 1) * (m2 - n2) - 2 * l2),
            0.5 * M * (3 * l2 * (m2 - n2) + 4 * n2 - 2 * l2),
        ],
        (0, 4): [  # fy_z2_x2
            1.5 * _SQRT5 * L * m2 * (n2 - l2),
            -_SQRT_5_8 * L * ((6 * m2 - 1) * (n2 - l2) + 2 * m2),
            0.5 * L * (3 * m2 * (n2 - l2) - 4 * n2 + 2 * m2),
        ],
        (0, 5): [  # fz_x2_y2
            1.5 * _SQRT5 * L * M * N * (l2 - m2),
            -3.0 * _SQRT_5_2 * L * M * N * (l2 - m2),
            1.5 * L * M * N * (l2 - m2),
        ],
        (0, 6): [  # fxyz
            _SQRT45 * l2 * m2 * N,
            -_SQRT_5_2 * N * (6 * l2 * m2 + n2 - 1),
            N * (3 * l2 * m2 + 2 * n2 - 1),
        ],
    }


def _df_row_x2y2(L, M, N) -> dict:
    """The d-f row E_{x^2-y^2, .}, not generated by the cyclic rule."""
    # Row index 3 = dx2-y2, written out in full for all seven f columns.
    # ``lm`` abbreviates l^2 - m^2, the combination that appears in nearly
    # every entry -- unsurprisingly, since it is the orbital's own shape.
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    return {
        (3, 0): [  # fx3
            0.25 * _SQRT3 * L * lm * (5 * l2 - 3),
            -_SQRT_3_8 * L * (lm - 1) * (5 * l2 - 1),
            -0.25 * _SQRT15 * L * (lm * (1 - l2) - 2 * n2),
        ],
        (3, 1): [  # fy3
            0.25 * _SQRT3 * M * lm * (5 * m2 - 3),
            -_SQRT_3_8 * M * (lm + 1) * (5 * m2 - 1),
            -0.25 * _SQRT15 * M * (lm * (1 - m2) + 2 * n2),
        ],
        (3, 2): [  # fz3
            0.25 * _SQRT3 * N * lm * (5 * n2 - 3),
            -_SQRT_3_8 * N * lm * (5 * n2 - 1),
            0.25 * _SQRT15 * N * (n2 + 1) * lm,
        ],
        (3, 3): [  # fx_y2_z2
            0.75 * _SQRT5 * L * lm * (m2 - n2),
            -_SQRT_5_8 * L * (3 * lm * (m2 - n2) - l2 + 1),
            0.25 * L * (3 * lm * (m2 - n2) - 4 * l2 + 2),
        ],
        (3, 4): [  # fy_z2_x2
            0.75 * _SQRT5 * M * lm * (n2 - l2),
            -_SQRT_5_8 * M * (3 * lm * (n2 - l2) - m2 + 1),
            0.25 * M * (3 * lm * (n2 - l2) - 4 * m2 + 2),
        ],
        (3, 5): [  # fz_x2_y2
            0.75 * _SQRT5 * N * lm * lm,
            -_SQRT_5_8 * N * (3 * lm * lm + 2 * n2 - 2),
            0.25 * N * (3 * lm * lm + 8 * n2 - 4),
        ],
        (3, 6): [  # fxyz
            1.5 * _SQRT5 * L * M * N * lm,
            -3.0 * _SQRT_5_2 * L * M * N * lm,
            1.5 * L * M * N * lm,
        ],
    }


def _df_row_3z2(L, M, N) -> dict:
    """The d-f row E_{3z^2-r^2, .}, not generated by the cyclic rule."""
    # Row index 4 = dz2, the other half of the pair that resists rotation, also
    # written out in full.  ``t`` abbreviates 3n^2 - 1, this orbital's own
    # angular shape, which recurs throughout the row exactly as ``lm`` did in
    # the row above.  Note this ``t`` is a local shorthand and has nothing to
    # do with the loop variable ``t`` used elsewhere for rotation counts.
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    t = 3 * n2 - 1
    return {
        (4, 0): [  # fx3
            0.25 * L * t * (5 * l2 - 3),
            -0.75 * _SQRT2 * L * n2 * (5 * l2 - 1),
            0.75 * _SQRT5 * L * (l2 * n2 - m2),
        ],
        (4, 1): [  # fy3
            0.25 * M * t * (5 * m2 - 3),
            -0.75 * _SQRT2 * M * n2 * (5 * m2 - 1),
            0.75 * _SQRT5 * M * (m2 * n2 - l2),
        ],
        (4, 2): [  # fz3
            0.25 * N * t * (5 * n2 - 3),
            -0.75 * _SQRT2 * N * (5 * n2 - 1) * (n2 - 1),
            0.75 * _SQRT5 * N * (n2 - 1) * (n2 - 1),
        ],
        (4, 3): [  # fx_y2_z2
            0.25 * _SQRT15 * L * (m2 - n2) * t,
            -_SQRT_15_8 * L * n2 * (3 * (m2 - n2) + 2),
            0.25 * _SQRT3 * L * (t * (m2 - n2) - 4 * l2 + 2),
        ],
        (4, 4): [  # fy_z2_x2
            0.25 * _SQRT15 * M * (n2 - l2) * t,
            -_SQRT_15_8 * M * n2 * (3 * (n2 - l2) - 2),
            0.25 * _SQRT3 * M * (t * (n2 - l2) + 4 * m2 - 2),
        ],
        (4, 5): [  # fz_x2_y2
            0.25 * _SQRT15 * N * t * lm,
            -_SQRT_15_8 * N * t * lm,
            0.25 * _SQRT3 * N * t * lm,
        ],
        (4, 6): [  # fxyz
            0.5 * _SQRT15 * L * M * N * t,
            -_SQRT_15_2 * L * M * N * t,
            0.5 * _SQRT3 * L * M * N * t,
        ],
    }


def _df_block(L, M, N) -> torch.Tensor:
    # Rows 0, 1, 2 (dxy, dyz, dzx) by rotating the one written row, exactly as
    # for p-f.
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _df_row_xy(*args).items():
            cells[
                (
                    _cycle_index(_D_T2G_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = value
    # Rows 3 and 4 are dropped in whole and unrotated -- they carry their own
    # row index in their keys, so they land in the right place.  The direction
    # cosines are passed through unrotated for the same reason.
    cells.update(_df_row_x2y2(L, M, N))
    cells.update(_df_row_3z2(L, M, N))
    # 3 bond symmetries (sigma, pi, delta), 5 d rows, 7 f columns.
    return _assemble_block(cells, 3, 5, 7, L)


# --------------------------------------------------------------------- f-f
# An f orbital paired with another f orbital: seven rows, seven columns, and
# all four bond symmetries -- sigma, pi, delta and phi, three sign flips per
# turn being reachable only when both partners are f.  Every cell is a
# four-element list [sigma, pi, delta, phi], matching "ff0" through "ff3".
#
# 49 cells is far too many to write out, so 12 are given and two reductions do
# the rest: rotation (as above) and then symmetry.  The symmetry argument,
# spelled out: swapping the two f orbitals is the same as looking down the bond
# from the other end, i.e. sending (l, m, n) to (-l, -m, -n).  Each f orbital is
# odd under that flip -- it changes sign -- so a *pair* of them picks up two
# sign changes, which cancel.  The f-f block is therefore unchanged by the flip,
# which makes it symmetric: entry (a, b) equals entry (b, a).  So any cell the
# rotations miss can be copied from its mirror image across the diagonal.
def _ff_given(L, M, N) -> dict:
    """The twelve written f-f entries.

    Rotating these three times and closing under transposition (the f-f block
    is even under (l, m, n) -> (-l, -m, -n), hence symmetric) yields all 49.
    """
    # ``lm`` is again l^2 - m^2.  ``q`` is l^2 m^2 + m^2 n^2 + n^2 l^2, a
    # combination that is itself unchanged by the x -> y -> z -> x relabelling
    # (it contains all three pairings symmetrically), which is why it shows up
    # in the most symmetric entries.
    l2, m2, n2 = L * L, M * M, N * N
    lm = l2 - m2
    q = l2 * m2 + m2 * n2 + n2 * l2
    return {
        (2, 0): [  # fz3 with fx3
            0.25 * L * N * (5 * l2 - 3) * (5 * n2 - 3),
            -0.375 * L * N * (5 * l2 - 1) * (5 * n2 - 1),
            3.75 * L * N * (l2 * n2 - m2),
            0.625 * L * N * (3 * m2 - l2 * n2),
        ],
        (2, 1): [  # fz3 with fy3
            0.25 * M * N * (5 * m2 - 3) * (5 * n2 - 3),
            -0.375 * M * N * (5 * m2 - 1) * (5 * n2 - 1),
            3.75 * M * N * (m2 * n2 - l2),
            0.625 * M * N * (3 * l2 - m2 * n2),
        ],
        (2, 2): [  # fz3 with fz3
            0.25 * n2 * (5 * n2 - 3) * (5 * n2 - 3),
            0.375 * (5 * n2 - 1) * (5 * n2 - 1) * (1 - n2),
            3.75 * n2 * (1 - n2) * (1 - n2),
            0.625 * (1 - n2) * (1 - n2) * (1 - n2),
        ],
        (5, 0): [  # fz_x2_y2 with fx3
            0.25 * _SQRT15 * L * N * lm * (5 * l2 - 3),
            0.125 * _SQRT15 * L * N * (2 - 3 * lm) * (5 * l2 - 1),
            0.25 * _SQRT15 * L * N * (3 * (1 + l2) * lm - 8 * l2 + 2),
            0.125 * _SQRT15 * L * N * (-(l2 + 3) * lm + 6 * l2 - 2),
        ],
        (5, 1): [  # fz_x2_y2 with fy3
            0.25 * _SQRT15 * M * N * lm * (5 * m2 - 3),
            -0.125 * _SQRT15 * M * N * (2 + 3 * lm) * (5 * m2 - 1),
            0.25 * _SQRT15 * M * N * (3 * (1 + m2) * lm + 8 * m2 - 2),
            -0.125 * _SQRT15 * M * N * ((m2 + 3) * lm + 6 * m2 - 2),
        ],
        (5, 2): [  # fz_x2_y2 with fz3
            0.25 * _SQRT15 * lm * n2 * (5 * n2 - 3),
            -0.125 * _SQRT15 * lm * (5 * n2 - 1) * (3 * n2 - 1),
            0.25 * _SQRT15 * lm * n2 * (3 * n2 - 1),
            0.125 * _SQRT15 * lm * (1 - n2 * n2),
        ],
        (5, 3): [  # fz_x2_y2 with fx_y2_z2
            3.75 * L * N * lm * (m2 - n2),
            -0.625 * L * N * (9 * lm * (m2 - n2) - 2 * m2 + 2),
            0.25 * L * N * (9 * lm * (m2 - n2) - 8 * m2 + 2),
            0.375 * L * N * (-lm * (m2 - n2) + 2 * m2 + 2),
        ],
        (5, 4): [  # fz_x2_y2 with fy_z2_x2
            3.75 * M * N * lm * (n2 - l2),
            -0.625 * M * N * (9 * lm * (n2 - l2) - 2 * l2 + 2),
            0.25 * M * N * (9 * lm * (n2 - l2) - 8 * l2 + 2),
            0.375 * M * N * (-lm * (n2 - l2) + 2 * l2 + 2),
        ],
        (5, 5): [  # fz_x2_y2 with fz_x2_y2
            3.75 * n2 * lm * lm,
            0.625 * (4 * n2 * (1 - n2) + lm * lm * (1 - 9 * n2)),
            0.25 * (lm * lm * (9 * n2 - 4) + 4 * (1 - 2 * n2) * (1 - 2 * n2)),
            0.375 * (1 - n2) * ((1 + n2) * (1 + n2) - 4 * l2 * m2),
        ],
        (6, 2): [  # fxyz with fz3
            0.5 * _SQRT15 * L * M * n2 * (5 * n2 - 3),
            -0.25 * _SQRT15 * L * M * (3 * n2 - 1) * (5 * n2 - 1),
            0.5 * _SQRT15 * L * M * n2 * (3 * n2 - 1),
            0.25 * _SQRT15 * L * M * (1 - n2 * n2),
        ],
        (6, 5): [  # fxyz with fz_x2_y2
            7.5 * L * M * n2 * lm,
            -1.25 * L * M * lm * (9 * n2 - 1),
            0.5 * L * M * lm * (9 * n2 - 4),
            0.75 * L * M * lm * (1 - n2),
        ],
        (6, 6): [  # fxyz with fxyz
            15 * l2 * m2 * n2,
            2.5 * (q - 9 * l2 * m2 * n2),
            1 - 4 * q + 9 * l2 * m2 * n2,
            1.5 * (1 - l2) * (1 - m2) * (1 - n2),
        ],
    }


def _ff_block(L, M, N) -> torch.Tensor:
    # Step 1: rotation.  Both the row and the column index move, since both
    # sides of the pair are f orbitals.
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), value in _ff_given(*args).items():
            cells[
                (
                    _cycle_index(_F_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = value
    # Step 2: symmetry.  Fill any still-empty cell from its mirror across the
    # diagonal, per the argument above.  The ``(a, b) not in cells`` test means
    # a cell the rotations already produced is never overwritten, so this only
    # ever adds information -- and if the two ever disagreed, the rotation
    # result wins, silently.  Consistency between the two routes is instead
    # checked by the orthogonality test described further below.
    for a in range(7):
        for b in range(7):
            if (a, b) not in cells and (b, a) in cells:
                cells[(a, b)] = cells[(b, a)]
    # 4 bond symmetries (sigma, pi, delta, phi), 7 rows, 7 columns.  If the two
    # steps together left any of the 49 cells empty, _assemble_block raises
    # rather than letting a hole become a zero.
    return _assemble_block(cells, 4, 7, 7, L)


# The four public entry points.  Each takes the direction cosines for a batch
# of atom pairs and returns that block's angular factors.  Multiply these by the
# matching radial functions and sum over the first axis and you have the matrix
# entries.
#
# In every returned shape, the axes are
#   (bond symmetry, lower-shell orbital, f orbital, atom pair),
# and every f axis is already in ``STRUCTURE_F_AO_ORDER``.


def f_angular_sf(L, M, N) -> torch.Tensor:
    """s-f angular factors, shape ``(1, 1, 7, P)`` = (sfsigma,) x s x f x pairs.

    Rows follow the s shell, columns follow ``STRUCTURE_F_AO_ORDER``.
    """
    return _sf_block(L, M, N)


def f_angular_pf(L, M, N) -> torch.Tensor:
    """p-f angular factors, shape ``(2, 3, 7, P)`` = (pfsigma, pfpi) x p x f."""
    return _pf_block(L, M, N)


def f_angular_df(L, M, N) -> torch.Tensor:
    """d-f angular factors, shape ``(3, 5, 7, P)`` = (dfsigma..dfdelta) x d x f."""
    return _df_block(L, M, N)


def f_angular_ff(L, M, N) -> torch.Tensor:
    """f-f angular factors, shape ``(4, 7, 7, P)`` = (ffsigma..ffphi) x f x f."""
    return _ff_block(L, M, N)


# ---------------------------------------------------------------------------
# f angular blocks, differentiated
# ---------------------------------------------------------------------------
# Everything from here to the end of this section is the derivative of the
# tables above with respect to the bond direction -- that is, d/dL, d/dM and
# d/dN of every angular factor.
#
# WHY THESE ARE NEEDED.  A matrix entry is (angular factor) * (radial factor).
# Moving an atom changes both, so the product rule gives
#
#     dH/d(coordinate) = (d angular / d coordinate) * V
#                      + angular * (dV / d coordinate)
#
# and the first term needs exactly what this section provides, chained through
# dL/d(coordinate), dM/d(coordinate) and dN/d(coordinate).  The s/p/d blocks in
# the assembler below already do this longhand; these tables are the f
# equivalent.
#
# HOW THEY WERE OBTAINED.  Differentiated symbolically from the value tables
# above and then checked three ways: against automatic differentiation of the
# value functions, against central finite differences of the same, and by the
# structural identities the values themselves satisfy.  They are exact
# polynomial derivatives, not a numerical approximation, so they cost about the
# same as the values and introduce no step-size error.
#
# ONE THING TO WATCH.  These are derivatives with respect to L, M and N treated
# as three free variables.  The real direction cosines are not free -- they obey
# L^2 + M^2 + N^2 = 1 -- so a caller must not read a single partial as "what
# happens if I nudge L alone", which would leave the vector non-unit.  The
# chain rule fixes this automatically: dL/d(coordinate), dM/d(coordinate) and
# dN/d(coordinate) already move together in a way that preserves the constraint,
# so contracting all three partials against them gives the right answer.  Any
# use that does NOT contract all three at once is a mistake.


def _sf_row_grad(L, M, N) -> dict:
    """Partials of :func:`_sf_row` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (0, 0): [  # fx3
            (3 * (5 * L**2 - 1) / 2, zero, zero),
        ],
        (0, 3): [  # fx_y2_z2
            (_SQRT15 * (M - N) * (M + N) / 2, L * M * _SQRT15, -L * N * _SQRT15),
        ],
        (0, 6): [  # fxyz
            (M * N * _SQRT15, L * N * _SQRT15, L * M * _SQRT15),
        ],
    }


def _pf_row_grad(L, M, N) -> dict:
    """Partials of :func:`_pf_row` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (0, 0): [  # px with fx3
            (L * (10 * L**2 - 3), zero, zero),
            (-L * _SQRT6 * (5 * L**2 - 3), zero, zero),
        ],
        (0, 1): [  # px with fy3
            (M * (5 * M**2 - 3) / 2, 3 * L * (5 * M**2 - 1) / 2, zero),
            (-M * _SQRT6 * (5 * M**2 - 1) / 4, -L * _SQRT6 * (15 * M**2 - 1) / 4, zero),
        ],
        (0, 2): [  # px with fz3
            (N * (5 * N**2 - 3) / 2, zero, 3 * L * (5 * N**2 - 1) / 2),
            (-N * _SQRT6 * (5 * N**2 - 1) / 4, zero, -L * _SQRT6 * (15 * N**2 - 1) / 4),
        ],
        (0, 3): [  # px with fx_y2_z2
            (L * _SQRT15 * (M - N) * (M + N), L**2 * M * _SQRT15, -L**2 * N * _SQRT15),
            (
                -3 * L * _SQRT10 * (M - N) * (M + N) / 2,
                -M * _SQRT10 * (3 * L**2 - 1) / 2,
                N * _SQRT10 * (3 * L**2 - 1) / 2,
            ),
        ],
        (0, 4): [  # px with fy_z2_x2
            (
                -M * _SQRT15 * (3 * L**2 - N**2) / 2,
                -L * _SQRT15 * (L - N) * (L + N) / 2,
                L * M * N * _SQRT15,
            ),
            (
                M * _SQRT10 * (9 * L**2 - 3 * N**2 - 2) / 4,
                L * _SQRT10 * (3 * L**2 - 3 * N**2 - 2) / 4,
                -3 * L * M * N * _SQRT10 / 2,
            ),
        ],
        (0, 5): [  # px with fz_x2_y2
            (
                N * _SQRT15 * (3 * L**2 - M**2) / 2,
                -L * M * N * _SQRT15,
                L * _SQRT15 * (L - M) * (L + M) / 2,
            ),
            (
                -N * _SQRT10 * (9 * L**2 - 3 * M**2 - 2) / 4,
                3 * L * M * N * _SQRT10 / 2,
                -L * _SQRT10 * (3 * L**2 - 3 * M**2 - 2) / 4,
            ),
        ],
        (0, 6): [  # px with fxyz
            (2 * L * M * N * _SQRT15, L**2 * N * _SQRT15, L**2 * M * _SQRT15),
            (
                -3 * L * M * N * _SQRT10,
                -N * _SQRT10 * (3 * L**2 - 1) / 2,
                -M * _SQRT10 * (3 * L**2 - 1) / 2,
            ),
        ],
    }


def _df_row_xy_grad(L, M, N) -> dict:
    """Partials of :func:`_df_row_xy` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (0, 0): [  # dxy with fx3
            (
                L * M * _SQRT3 * (10 * L**2 - 3),
                L**2 * _SQRT3 * (5 * L**2 - 3) / 2,
                zero,
            ),
            (
                -L * M * _SQRT6 * (20 * L**2 - 7) / 2,
                -_SQRT6 * (2 * L**2 - 1) * (5 * L**2 - 1) / 4,
                zero,
            ),
            (
                L * M * _SQRT15 * (2 * L**2 - 1),
                L**2 * _SQRT15 * (L - 1) * (L + 1) / 2,
                zero,
            ),
        ],
        (0, 1): [  # dxy with fy3
            (
                M**2 * _SQRT3 * (5 * M**2 - 3) / 2,
                L * M * _SQRT3 * (10 * M**2 - 3),
                zero,
            ),
            (
                -_SQRT6 * (2 * M**2 - 1) * (5 * M**2 - 1) / 4,
                -L * M * _SQRT6 * (20 * M**2 - 7) / 2,
                zero,
            ),
            (
                M**2 * _SQRT15 * (M - 1) * (M + 1) / 2,
                L * M * _SQRT15 * (2 * M**2 - 1),
                zero,
            ),
        ],
        (0, 2): [  # dxy with fz3
            (
                M * N * _SQRT3 * (5 * N**2 - 3) / 2,
                L * N * _SQRT3 * (5 * N**2 - 3) / 2,
                3 * L * M * _SQRT3 * (5 * N**2 - 1) / 2,
            ),
            (
                -M * N * _SQRT6 * (5 * N**2 - 1) / 2,
                -L * N * _SQRT6 * (5 * N**2 - 1) / 2,
                -L * M * _SQRT6 * (15 * N**2 - 1) / 2,
            ),
            (
                M * N * _SQRT15 * (N**2 + 1) / 2,
                L * N * _SQRT15 * (N**2 + 1) / 2,
                L * M * _SQRT15 * (3 * N**2 + 1) / 2,
            ),
        ],
        (0, 3): [  # dxy with fx_y2_z2
            (
                3 * L * M * _SQRT5 * (M - N) * (M + N),
                3 * L**2 * _SQRT5 * (3 * M**2 - N**2) / 2,
                -3 * L**2 * M * N * _SQRT5,
            ),
            (
                -L * M * _SQRT10 * (3 * M**2 - 3 * N**2 - 1),
                -_SQRT10 * (18 * L**2 * M**2 - 6 * L**2 * N**2 - 2 * L**2 - 3 * M**2 + N**2) / 4,
                M * N * _SQRT10 * (6 * L**2 - 1) / 2,
            ),
            (
                L * M * (3 * M**2 - 3 * N**2 - 2),
                (9 * L**2 * M**2 - 3 * L**2 * N**2 - 2 * L**2 + 4 * N**2) / 2,
                -M * N * (3 * L**2 - 4),
            ),
        ],
        (0, 4): [  # dxy with fy_z2_x2
            (
                -3 * M**2 * _SQRT5 * (3 * L**2 - N**2) / 2,
                -3 * L * M * _SQRT5 * (L - N) * (L + N),
                3 * L * M**2 * N * _SQRT5,
            ),
            (
                _SQRT10 * (18 * L**2 * M**2 - 3 * L**2 - 6 * M**2 * N**2 - 2 * M**2 + N**2) / 4,
                L * M * _SQRT10 * (3 * L**2 - 3 * N**2 - 1),
                -L * N * _SQRT10 * (6 * M**2 - 1) / 2,
            ),
            (
                -(9 * L**2 * M**2 - 3 * M**2 * N**2 - 2 * M**2 + 4 * N**2) / 2,
                -L * M * (3 * L**2 - 3 * N**2 - 2),
                L * N * (3 * M**2 - 4),
            ),
        ],
        (0, 5): [  # dxy with fz_x2_y2
            (
                3 * M * N * _SQRT5 * (3 * L**2 - M**2) / 2,
                3 * L * N * _SQRT5 * (L**2 - 3 * M**2) / 2,
                3 * L * M * _SQRT5 * (L - M) * (L + M) / 2,
            ),
            (
                -3 * M * N * _SQRT10 * (3 * L**2 - M**2) / 2,
                -3 * L * N * _SQRT10 * (L**2 - 3 * M**2) / 2,
                -3 * L * M * _SQRT10 * (L - M) * (L + M) / 2,
            ),
            (
                3 * M * N * (3 * L**2 - M**2) / 2,
                3 * L * N * (L**2 - 3 * M**2) / 2,
                3 * L * M * (L - M) * (L + M) / 2,
            ),
        ],
        (0, 6): [  # dxy with fxyz
            (
                6 * L * M**2 * N * _SQRT5,
                6 * L**2 * M * N * _SQRT5,
                3 * L**2 * M**2 * _SQRT5,
            ),
            (
                -6 * L * M**2 * N * _SQRT10,
                -6 * L**2 * M * N * _SQRT10,
                -_SQRT10 * (6 * L**2 * M**2 + 3 * N**2 - 1) / 2,
            ),
            (6 * L * M**2 * N, 6 * L**2 * M * N, 3 * L**2 * M**2 + 6 * N**2 - 1),
        ],
    }


def _df_row_x2y2_grad(L, M, N) -> dict:
    """Partials of :func:`_df_row_x2y2` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (3, 0): [  # dx2-y2 with fx3
            (
                _SQRT3 * (25 * L**4 - 15 * L**2 * M**2 - 9 * L**2 + 3 * M**2) / 4,
                -L * M * _SQRT3 * (5 * L**2 - 3) / 2,
                zero,
            ),
            (
                -_SQRT6 * (25 * L**4 - 15 * L**2 * M**2 - 18 * L**2 + M**2 + 1) / 4,
                L * M * _SQRT6 * (5 * L**2 - 1) / 2,
                zero,
            ),
            (
                _SQRT15 * (5 * L**4 - 3 * L**2 * M**2 - 3 * L**2 + M**2 + 2 * N**2) / 4,
                -L * M * _SQRT15 * (L - 1) * (L + 1) / 2,
                L * N * _SQRT15,
            ),
        ],
        (3, 1): [  # dx2-y2 with fy3
            (
                L * M * _SQRT3 * (5 * M**2 - 3) / 2,
                _SQRT3 * (15 * L**2 * M**2 - 3 * L**2 - 25 * M**4 + 9 * M**2) / 4,
                zero,
            ),
            (
                -L * M * _SQRT6 * (5 * M**2 - 1) / 2,
                -_SQRT6 * (15 * L**2 * M**2 - L**2 - 25 * M**4 + 18 * M**2 - 1) / 4,
                zero,
            ),
            (
                L * M * _SQRT15 * (M - 1) * (M + 1) / 2,
                _SQRT15 * (3 * L**2 * M**2 - L**2 - 5 * M**4 + 3 * M**2 - 2 * N**2) / 4,
                -M * N * _SQRT15,
            ),
        ],
        (3, 2): [  # dx2-y2 with fz3
            (
                L * N * _SQRT3 * (5 * N**2 - 3) / 2,
                -M * N * _SQRT3 * (5 * N**2 - 3) / 2,
                3 * _SQRT3 * (L - M) * (L + M) * (5 * N**2 - 1) / 4,
            ),
            (
                -L * N * _SQRT6 * (5 * N**2 - 1) / 2,
                M * N * _SQRT6 * (5 * N**2 - 1) / 2,
                -_SQRT6 * (L - M) * (L + M) * (15 * N**2 - 1) / 4,
            ),
            (
                L * N * _SQRT15 * (N**2 + 1) / 2,
                -M * N * _SQRT15 * (N**2 + 1) / 2,
                _SQRT15 * (L - M) * (L + M) * (3 * N**2 + 1) / 4,
            ),
        ],
        (3, 3): [  # dx2-y2 with fx_y2_z2
            (
                3 * _SQRT5 * (3 * L**2 - M**2) * (M - N) * (M + N) / 4,
                3 * L * M * _SQRT5 * (L**2 - 2 * M**2 + N**2) / 2,
                -3 * L * N * _SQRT5 * (L - M) * (L + M) / 2,
            ),
            (
                -_SQRT10 * (9 * L**2 * M**2 - 9 * L**2 * N**2 - 3 * L**2 - 3 * M**4 + 3 * M**2 * N**2 + 1) / 4,
                -3 * L * M * _SQRT10 * (L**2 - 2 * M**2 + N**2) / 2,
                3 * L * N * _SQRT10 * (L - M) * (L + M) / 2,
            ),
            (
                (9 * L**2 * M**2 - 9 * L**2 * N**2 - 12 * L**2 - 3 * M**4 + 3 * M**2 * N**2 + 2) / 4,
                3 * L * M * (L**2 - 2 * M**2 + N**2) / 2,
                -3 * L * N * (L - M) * (L + M) / 2,
            ),
        ],
        (3, 4): [  # dx2-y2 with fy_z2_x2
            (
                -3 * L * M * _SQRT5 * (2 * L**2 - M**2 - N**2) / 2,
                -3 * _SQRT5 * (L - N) * (L + N) * (L**2 - 3 * M**2) / 4,
                3 * M * N * _SQRT5 * (L - M) * (L + M) / 2,
            ),
            (
                3 * L * M * _SQRT10 * (2 * L**2 - M**2 - N**2) / 2,
                _SQRT10 * (3 * L**4 - 9 * L**2 * M**2 - 3 * L**2 * N**2 + 9 * M**2 * N**2 + 3 * M**2 - 1) / 4,
                -3 * M * N * _SQRT10 * (L - M) * (L + M) / 2,
            ),
            (
                -3 * L * M * (2 * L**2 - M**2 - N**2) / 2,
                -(3 * L**4 - 9 * L**2 * M**2 - 3 * L**2 * N**2 + 9 * M**2 * N**2 + 12 * M**2 - 2) / 4,
                3 * M * N * (L - M) * (L + M) / 2,
            ),
        ],
        (3, 5): [  # dx2-y2 with fz_x2_y2
            (
                3 * L * N * _SQRT5 * (L - M) * (L + M),
                -3 * M * N * _SQRT5 * (L - M) * (L + M),
                3 * _SQRT5 * (L - M)**2 * (L + M)**2 / 4,
            ),
            (
                -3 * L * N * _SQRT10 * (L - M) * (L + M),
                3 * M * N * _SQRT10 * (L - M) * (L + M),
                -_SQRT10 * (3 * L**4 - 6 * L**2 * M**2 + 3 * M**4 + 6 * N**2 - 2) / 4,
            ),
            (
                3 * L * N * (L - M) * (L + M),
                -3 * M * N * (L - M) * (L + M),
                (3 * L**4 - 6 * L**2 * M**2 + 3 * M**4 + 24 * N**2 - 4) / 4,
            ),
        ],
        (3, 6): [  # dx2-y2 with fxyz
            (
                3 * M * N * _SQRT5 * (3 * L**2 - M**2) / 2,
                3 * L * N * _SQRT5 * (L**2 - 3 * M**2) / 2,
                3 * L * M * _SQRT5 * (L - M) * (L + M) / 2,
            ),
            (
                -3 * M * N * _SQRT10 * (3 * L**2 - M**2) / 2,
                -3 * L * N * _SQRT10 * (L**2 - 3 * M**2) / 2,
                -3 * L * M * _SQRT10 * (L - M) * (L + M) / 2,
            ),
            (
                3 * M * N * (3 * L**2 - M**2) / 2,
                3 * L * N * (L**2 - 3 * M**2) / 2,
                3 * L * M * (L - M) * (L + M) / 2,
            ),
        ],
    }


def _df_row_3z2_grad(L, M, N) -> dict:
    """Partials of :func:`_df_row_3z2` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (4, 0): [  # dz2 with fx3
            (
                3 * (5 * L**2 - 1) * (3 * N**2 - 1) / 4,
                zero,
                3 * L * N * (5 * L**2 - 3) / 2,
            ),
            (
                -3 * N**2 * _SQRT2 * (15 * L**2 - 1) / 4,
                zero,
                -3 * L * N * _SQRT2 * (5 * L**2 - 1) / 2,
            ),
            (
                3 * _SQRT5 * (3 * L**2 * N**2 - M**2) / 4,
                -3 * L * M * _SQRT5 / 2,
                3 * L**3 * N * _SQRT5 / 2,
            ),
        ],
        (4, 1): [  # dz2 with fy3
            (
                zero,
                3 * (5 * M**2 - 1) * (3 * N**2 - 1) / 4,
                3 * M * N * (5 * M**2 - 3) / 2,
            ),
            (
                zero,
                -3 * N**2 * _SQRT2 * (15 * M**2 - 1) / 4,
                -3 * M * N * _SQRT2 * (5 * M**2 - 1) / 2,
            ),
            (
                -3 * L * M * _SQRT5 / 2,
                -3 * _SQRT5 * (L**2 - 3 * M**2 * N**2) / 4,
                3 * M**3 * N * _SQRT5 / 2,
            ),
        ],
        (4, 2): [  # dz2 with fz3
            (zero, zero, 3 * (5 * N**2 - 2 * N - 1) * (5 * N**2 + 2 * N - 1) / 4),
            (zero, zero, -3 * _SQRT2 * (25 * N**4 - 18 * N**2 + 1) / 4),
            (zero, zero, 3 * _SQRT5 * (N - 1) * (N + 1) * (5 * N**2 - 1) / 4),
        ],
        (4, 3): [  # dz2 with fx_y2_z2
            (
                _SQRT15 * (M - N) * (M + N) * (3 * N**2 - 1) / 4,
                L * M * _SQRT15 * (3 * N**2 - 1) / 2,
                L * N * _SQRT15 * (3 * M**2 - 6 * N**2 + 1) / 2,
            ),
            (
                -N**2 * _SQRT30 * (3 * M**2 - 3 * N**2 + 2) / 4,
                -3 * L * M * N**2 * _SQRT30 / 2,
                -L * N * _SQRT30 * (3 * M**2 - 6 * N**2 + 2) / 2,
            ),
            (
                -_SQRT3 * (12 * L**2 - 3 * M**2 * N**2 + M**2 + 3 * N**4 - N**2 - 2) / 4,
                L * M * _SQRT3 * (3 * N**2 - 1) / 2,
                L * N * _SQRT3 * (3 * M**2 - 6 * N**2 + 1) / 2,
            ),
        ],
        (4, 4): [  # dz2 with fy_z2_x2
            (
                -L * M * _SQRT15 * (3 * N**2 - 1) / 2,
                -_SQRT15 * (L - N) * (L + N) * (3 * N**2 - 1) / 4,
                -M * N * _SQRT15 * (3 * L**2 - 6 * N**2 + 1) / 2,
            ),
            (
                3 * L * M * N**2 * _SQRT30 / 2,
                N**2 * _SQRT30 * (3 * L**2 - 3 * N**2 + 2) / 4,
                M * N * _SQRT30 * (3 * L**2 - 6 * N**2 + 2) / 2,
            ),
            (
                -L * M * _SQRT3 * (3 * N**2 - 1) / 2,
                -_SQRT3 * (3 * L**2 * N**2 - L**2 - 12 * M**2 - 3 * N**4 + N**2 + 2) / 4,
                -M * N * _SQRT3 * (3 * L**2 - 6 * N**2 + 1) / 2,
            ),
        ],
        (4, 5): [  # dz2 with fz_x2_y2
            (
                L * N * _SQRT15 * (3 * N**2 - 1) / 2,
                -M * N * _SQRT15 * (3 * N**2 - 1) / 2,
                _SQRT15 * (L - M) * (L + M) * (3 * N - 1) * (3 * N + 1) / 4,
            ),
            (
                -L * N * _SQRT30 * (3 * N**2 - 1) / 2,
                M * N * _SQRT30 * (3 * N**2 - 1) / 2,
                -_SQRT30 * (L - M) * (L + M) * (3 * N - 1) * (3 * N + 1) / 4,
            ),
            (
                L * N * _SQRT3 * (3 * N**2 - 1) / 2,
                -M * N * _SQRT3 * (3 * N**2 - 1) / 2,
                _SQRT3 * (L - M) * (L + M) * (3 * N - 1) * (3 * N + 1) / 4,
            ),
        ],
        (4, 6): [  # dz2 with fxyz
            (
                M * N * _SQRT15 * (3 * N**2 - 1) / 2,
                L * N * _SQRT15 * (3 * N**2 - 1) / 2,
                L * M * _SQRT15 * (3 * N - 1) * (3 * N + 1) / 2,
            ),
            (
                -M * N * _SQRT30 * (3 * N**2 - 1) / 2,
                -L * N * _SQRT30 * (3 * N**2 - 1) / 2,
                -L * M * _SQRT30 * (3 * N - 1) * (3 * N + 1) / 2,
            ),
            (
                M * N * _SQRT3 * (3 * N**2 - 1) / 2,
                L * N * _SQRT3 * (3 * N**2 - 1) / 2,
                L * M * _SQRT3 * (3 * N - 1) * (3 * N + 1) / 2,
            ),
        ],
    }


def _ff_given_grad(L, M, N) -> dict:
    """Partials of :func:`_ff_given` with respect to L, M and N.

    Same keys as the value table; each channel carries a
    ``(d/dL, d/dM, d/dN)`` triple in place of the single value.
    """
    zero = torch.zeros_like(L)
    return {
        (2, 0): [  # fz3 with fx3
            (
                3 * N * (5 * L**2 - 1) * (5 * N**2 - 3) / 4,
                zero,
                3 * L * (5 * L**2 - 3) * (5 * N**2 - 1) / 4,
            ),
            (
                -3 * N * (15 * L**2 - 1) * (5 * N**2 - 1) / 8,
                zero,
                -3 * L * (5 * L**2 - 1) * (15 * N**2 - 1) / 8,
            ),
            (
                15 * N * (3 * L**2 * N**2 - M**2) / 4,
                -15 * L * M * N / 2,
                15 * L * (3 * L**2 * N**2 - M**2) / 4,
            ),
            (
                -15 * N * (L * N - M) * (L * N + M) / 8,
                15 * L * M * N / 4,
                -15 * L * (L * N - M) * (L * N + M) / 8,
            ),
        ],
        (2, 1): [  # fz3 with fy3
            (
                zero,
                3 * N * (5 * M**2 - 1) * (5 * N**2 - 3) / 4,
                3 * M * (5 * M**2 - 3) * (5 * N**2 - 1) / 4,
            ),
            (
                zero,
                -3 * N * (15 * M**2 - 1) * (5 * N**2 - 1) / 8,
                -3 * M * (5 * M**2 - 1) * (15 * N**2 - 1) / 8,
            ),
            (
                -15 * L * M * N / 2,
                -15 * N * (L**2 - 3 * M**2 * N**2) / 4,
                -15 * M * (L**2 - 3 * M**2 * N**2) / 4,
            ),
            (
                15 * L * M * N / 4,
                15 * N * (L - M * N) * (L + M * N) / 8,
                15 * M * (L - M * N) * (L + M * N) / 8,
            ),
        ],
        (2, 2): [  # fz3 with fz3
            (zero, zero, 3 * N * (5 * N**2 - 3) * (5 * N**2 - 1) / 2),
            (zero, zero, -3 * N * (5 * N**2 - 1) * (15 * N**2 - 11) / 4),
            (zero, zero, 15 * N * (N - 1) * (N + 1) * (3 * N**2 - 1) / 2),
            (zero, zero, -15 * N * (N - 1)**2 * (N + 1)**2 / 4),
        ],
        (5, 0): [  # fz_x2_y2 with fx3
            (
                N * _SQRT15 * (25 * L**4 - 15 * L**2 * M**2 - 9 * L**2 + 3 * M**2) / 4,
                -L * M * N * _SQRT15 * (5 * L**2 - 3) / 2,
                L * _SQRT15 * (L - M) * (L + M) * (5 * L**2 - 3) / 4,
            ),
            (
                -N * _SQRT15 * (75 * L**4 - 45 * L**2 * M**2 - 39 * L**2 + 3 * M**2 + 2) / 8,
                3 * L * M * N * _SQRT15 * (5 * L**2 - 1) / 4,
                -L * _SQRT15 * (5 * L**2 - 1) * (3 * L**2 - 3 * M**2 - 2) / 8,
            ),
            (
                N * _SQRT15 * (15 * L**4 - 9 * L**2 * M**2 - 15 * L**2 - 3 * M**2 + 2) / 4,
                -3 * L * M * N * _SQRT15 * (L**2 + 1) / 2,
                L * _SQRT15 * (3 * L**4 - 3 * L**2 * M**2 - 5 * L**2 - 3 * M**2 + 2) / 4,
            ),
            (
                -N * _SQRT15 * (5 * L**4 - 3 * L**2 * M**2 - 9 * L**2 - 3 * M**2 + 2) / 8,
                L * M * N * _SQRT15 * (L**2 + 3) / 4,
                -L * _SQRT15 * (L**4 - L**2 * M**2 - 3 * L**2 - 3 * M**2 + 2) / 8,
            ),
        ],
        (5, 1): [  # fz_x2_y2 with fy3
            (
                L * M * N * _SQRT15 * (5 * M**2 - 3) / 2,
                N * _SQRT15 * (15 * L**2 * M**2 - 3 * L**2 - 25 * M**4 + 9 * M**2) / 4,
                M * _SQRT15 * (L - M) * (L + M) * (5 * M**2 - 3) / 4,
            ),
            (
                -3 * L * M * N * _SQRT15 * (5 * M**2 - 1) / 4,
                -N * _SQRT15 * (45 * L**2 * M**2 - 3 * L**2 - 75 * M**4 + 39 * M**2 - 2) / 8,
                -M * _SQRT15 * (5 * M**2 - 1) * (3 * L**2 - 3 * M**2 + 2) / 8,
            ),
            (
                3 * L * M * N * _SQRT15 * (M**2 + 1) / 2,
                N * _SQRT15 * (9 * L**2 * M**2 + 3 * L**2 - 15 * M**4 + 15 * M**2 - 2) / 4,
                M * _SQRT15 * (3 * L**2 * M**2 + 3 * L**2 - 3 * M**4 + 5 * M**2 - 2) / 4,
            ),
            (
                -L * M * N * _SQRT15 * (M**2 + 3) / 4,
                -N * _SQRT15 * (3 * L**2 * M**2 + 3 * L**2 - 5 * M**4 + 9 * M**2 - 2) / 8,
                -M * _SQRT15 * (L**2 * M**2 + 3 * L**2 - M**4 + 3 * M**2 - 2) / 8,
            ),
        ],
        (5, 2): [  # fz_x2_y2 with fz3
            (
                L * N**2 * _SQRT15 * (5 * N**2 - 3) / 2,
                -M * N**2 * _SQRT15 * (5 * N**2 - 3) / 2,
                N * _SQRT15 * (L - M) * (L + M) * (10 * N**2 - 3) / 2,
            ),
            (
                -L * _SQRT15 * (3 * N**2 - 1) * (5 * N**2 - 1) / 4,
                M * _SQRT15 * (3 * N**2 - 1) * (5 * N**2 - 1) / 4,
                -N * _SQRT15 * (L - M) * (L + M) * (15 * N**2 - 4) / 2,
            ),
            (
                L * N**2 * _SQRT15 * (3 * N**2 - 1) / 2,
                -M * N**2 * _SQRT15 * (3 * N**2 - 1) / 2,
                N * _SQRT15 * (L - M) * (L + M) * (6 * N**2 - 1) / 2,
            ),
            (
                -L * _SQRT15 * (N - 1) * (N + 1) * (N**2 + 1) / 4,
                M * _SQRT15 * (N - 1) * (N + 1) * (N**2 + 1) / 4,
                -N**3 * _SQRT15 * (L - M) * (L + M) / 2,
            ),
        ],
        (5, 3): [  # fz_x2_y2 with fx_y2_z2
            (
                15 * N * (3 * L**2 - M**2) * (M - N) * (M + N) / 4,
                15 * L * M * N * (L**2 - 2 * M**2 + N**2) / 2,
                15 * L * (L - M) * (L + M) * (M**2 - 3 * N**2) / 4,
            ),
            (
                -5 * N * (27 * L**2 * M**2 - 27 * L**2 * N**2 - 9 * M**4 + 9 * M**2 * N**2 - 2 * M**2 + 2) / 8,
                -5 * L * M * N * (9 * L**2 - 18 * M**2 + 9 * N**2 - 2) / 4,
                -5 * L * (9 * L**2 * M**2 - 27 * L**2 * N**2 - 9 * M**4 + 27 * M**2 * N**2 - 2 * M**2 + 2) / 8,
            ),
            (
                N * (27 * L**2 * M**2 - 27 * L**2 * N**2 - 9 * M**4 + 9 * M**2 * N**2 - 8 * M**2 + 2) / 4,
                L * M * N * (9 * L**2 - 18 * M**2 + 9 * N**2 - 8) / 2,
                L * (9 * L**2 * M**2 - 27 * L**2 * N**2 - 9 * M**4 + 27 * M**2 * N**2 - 8 * M**2 + 2) / 4,
            ),
            (
                -3 * N * (3 * L**2 * M**2 - 3 * L**2 * N**2 - M**4 + M**2 * N**2 - 2 * M**2 - 2) / 8,
                -3 * L * M * N * (L**2 - 2 * M**2 + N**2 - 2) / 4,
                -3 * L * (L**2 * M**2 - 3 * L**2 * N**2 - M**4 + 3 * M**2 * N**2 - 2 * M**2 - 2) / 8,
            ),
        ],
        (5, 4): [  # fz_x2_y2 with fy_z2_x2
            (
                -15 * L * M * N * (2 * L**2 - M**2 - N**2) / 2,
                -15 * N * (L - N) * (L + N) * (L**2 - 3 * M**2) / 4,
                -15 * M * (L - M) * (L + M) * (L**2 - 3 * N**2) / 4,
            ),
            (
                5 * L * M * N * (18 * L**2 - 9 * M**2 - 9 * N**2 + 2) / 4,
                5 * N * (9 * L**4 - 27 * L**2 * M**2 - 9 * L**2 * N**2 + 2 * L**2 + 27 * M**2 * N**2 - 2) / 8,
                5 * M * (9 * L**4 - 9 * L**2 * M**2 - 27 * L**2 * N**2 + 2 * L**2 + 27 * M**2 * N**2 - 2) / 8,
            ),
            (
                -L * M * N * (18 * L**2 - 9 * M**2 - 9 * N**2 + 8) / 2,
                -N * (9 * L**4 - 27 * L**2 * M**2 - 9 * L**2 * N**2 + 8 * L**2 + 27 * M**2 * N**2 - 2) / 4,
                -M * (9 * L**4 - 9 * L**2 * M**2 - 27 * L**2 * N**2 + 8 * L**2 + 27 * M**2 * N**2 - 2) / 4,
            ),
            (
                3 * L * M * N * (2 * L**2 - M**2 - N**2 + 2) / 4,
                3 * N * (L**4 - 3 * L**2 * M**2 - L**2 * N**2 + 2 * L**2 + 3 * M**2 * N**2 + 2) / 8,
                3 * M * (L**4 - L**2 * M**2 - 3 * L**2 * N**2 + 2 * L**2 + 3 * M**2 * N**2 + 2) / 8,
            ),
        ],
        (5, 5): [  # fz_x2_y2 with fz_x2_y2
            (
                15 * L * N**2 * (L - M) * (L + M),
                -15 * M * N**2 * (L - M) * (L + M),
                15 * N * (L - M)**2 * (L + M)**2 / 2,
            ),
            (
                -5 * L * (L - M) * (L + M) * (3 * N - 1) * (3 * N + 1) / 2,
                5 * M * (L - M) * (L + M) * (3 * N - 1) * (3 * N + 1) / 2,
                -5 * N * (9 * L**4 - 18 * L**2 * M**2 + 9 * M**4 + 8 * N**2 - 4) / 4,
            ),
            (
                L * (L - M) * (L + M) * (3 * N - 2) * (3 * N + 2),
                -M * (L - M) * (L + M) * (3 * N - 2) * (3 * N + 2),
                N * (9 * L**4 - 18 * L**2 * M**2 + 9 * M**4 + 32 * N**2 - 16) / 2,
            ),
            (
                3 * L * M**2 * (N - 1) * (N + 1),
                3 * L**2 * M * (N - 1) * (N + 1),
                3 * N * (4 * L**2 * M**2 - 3 * N**4 - 2 * N**2 + 1) / 4,
            ),
        ],
        (6, 2): [  # fxyz with fz3
            (
                M * N**2 * _SQRT15 * (5 * N**2 - 3) / 2,
                L * N**2 * _SQRT15 * (5 * N**2 - 3) / 2,
                L * M * N * _SQRT15 * (10 * N**2 - 3),
            ),
            (
                -M * _SQRT15 * (3 * N**2 - 1) * (5 * N**2 - 1) / 4,
                -L * _SQRT15 * (3 * N**2 - 1) * (5 * N**2 - 1) / 4,
                -L * M * N * _SQRT15 * (15 * N**2 - 4),
            ),
            (
                M * N**2 * _SQRT15 * (3 * N**2 - 1) / 2,
                L * N**2 * _SQRT15 * (3 * N**2 - 1) / 2,
                L * M * N * _SQRT15 * (6 * N**2 - 1),
            ),
            (
                -M * _SQRT15 * (N - 1) * (N + 1) * (N**2 + 1) / 4,
                -L * _SQRT15 * (N - 1) * (N + 1) * (N**2 + 1) / 4,
                -L * M * N**3 * _SQRT15,
            ),
        ],
        (6, 5): [  # fxyz with fz_x2_y2
            (
                15 * M * N**2 * (3 * L**2 - M**2) / 2,
                15 * L * N**2 * (L**2 - 3 * M**2) / 2,
                15 * L * M * N * (L - M) * (L + M),
            ),
            (
                -5 * M * (3 * L**2 - M**2) * (3 * N - 1) * (3 * N + 1) / 4,
                -5 * L * (L**2 - 3 * M**2) * (3 * N - 1) * (3 * N + 1) / 4,
                -45 * L * M * N * (L - M) * (L + M) / 2,
            ),
            (
                M * (3 * L**2 - M**2) * (3 * N - 2) * (3 * N + 2) / 2,
                L * (L**2 - 3 * M**2) * (3 * N - 2) * (3 * N + 2) / 2,
                9 * L * M * N * (L - M) * (L + M),
            ),
            (
                -3 * M * (3 * L**2 - M**2) * (N - 1) * (N + 1) / 4,
                -3 * L * (L**2 - 3 * M**2) * (N - 1) * (N + 1) / 4,
                -3 * L * M * N * (L - M) * (L + M) / 2,
            ),
        ],
        (6, 6): [  # fxyz with fxyz
            (30 * L * M**2 * N**2, 30 * L**2 * M * N**2, 30 * L**2 * M**2 * N),
            (
                -5 * L * (9 * M**2 * N**2 - M**2 - N**2),
                -5 * M * (9 * L**2 * N**2 - L**2 - N**2),
                -5 * N * (9 * L**2 * M**2 - L**2 - M**2),
            ),
            (
                2 * L * (9 * M**2 * N**2 - 4 * M**2 - 4 * N**2),
                2 * M * (9 * L**2 * N**2 - 4 * L**2 - 4 * N**2),
                2 * N * (9 * L**2 * M**2 - 4 * L**2 - 4 * M**2),
            ),
            (
                -3 * L * (M - 1) * (M + 1) * (N - 1) * (N + 1),
                -3 * M * (L - 1) * (L + 1) * (N - 1) * (N + 1),
                -3 * N * (L - 1) * (L + 1) * (M - 1) * (M + 1),
            ),
        ],
    }

def _assemble_block_grad(
    cells: dict, n_chan: int, n_row: int, n_col: int
) -> torch.Tensor:
    """Stack ``{(row, col): [(dL, dM, dN)]}`` into ``(3, chan, row, col, P)``.

    The value version of this checks that the rotation rule covered every cell
    and refuses on a hole, because a hole would silently read as a legitimate
    zero.  The same argument applies with more force here: a missing derivative
    is a missing force contribution, which looks exactly like an atom that
    happens to feel no force.
    """
    missing = [
        (a, b) for a in range(n_row) for b in range(n_col) if (a, b) not in cells
    ]
    if missing:
        raise AssertionError(
            f"f angular derivative block is incomplete; the cyclic-permutation "
            f"rule did not cover {missing}"
        )
    return torch.stack(
        [
            torch.stack(
                [
                    torch.stack(
                        [
                            torch.stack([cells[(a, b)][k][d] for b in range(n_col)])
                            for a in range(n_row)
                        ]
                    )
                    for k in range(n_chan)
                ]
            )
            for d in range(3)
        ]
    )


def _sf_block_grad(L, M, N) -> torch.Tensor:
    # Same generation as the value block: rotate the written row three times and
    # file each result under the rotated column index.  The only addition is
    # ``_uncycle_grad``, which turns the row's partials in its own arguments
    # back into partials in L, M and N.
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), grads in _sf_row_grad(*args).items():
            cells[(a, _cycle_index(_F_CYCLE, b, t))] = [
                _uncycle_grad(t, *g) for g in grads
            ]
    return _assemble_block_grad(cells, 1, 1, 7)


def _pf_block_grad(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), grads in _pf_row_grad(*args).items():
            cells[
                (
                    _cycle_index(_P_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = [_uncycle_grad(t, *g) for g in grads]
    return _assemble_block_grad(cells, 2, 3, 7)


def _df_block_grad(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), grads in _df_row_xy_grad(*args).items():
            cells[
                (
                    _cycle_index(_D_T2G_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = [_uncycle_grad(t, *g) for g in grads]
    # The two rows that resist rotation carry their own row index and take the
    # direction cosines unrotated, so their partials need no un-cycling.
    cells.update(_df_row_x2y2_grad(L, M, N))
    cells.update(_df_row_3z2_grad(L, M, N))
    return _assemble_block_grad(cells, 3, 5, 7)


def _ff_block_grad(L, M, N) -> torch.Tensor:
    cells: dict = {}
    for t in range(3):
        args = _cycle_dirs(t, L, M, N)
        for (a, b), grads in _ff_given_grad(*args).items():
            cells[
                (
                    _cycle_index(_F_CYCLE, a, t),
                    _cycle_index(_F_CYCLE, b, t),
                )
            ] = [_uncycle_grad(t, *g) for g in grads]
    # Close under transposition, exactly as the value block does.  The f-f block
    # is symmetric as a function of (L, M, N), so differentiating both sides of
    # E(a, b) = E(b, a) shows each partial is symmetric too and can be copied.
    for a in range(7):
        for b in range(7):
            if (a, b) not in cells and (b, a) in cells:
                cells[(a, b)] = cells[(b, a)]
    return _assemble_block_grad(cells, 4, 7, 7)


# The four public derivative entry points, mirroring the four value ones.  Each
# returns the value block's shape with a leading axis of 3 for d/dL, d/dM, d/dN.


def f_angular_sf_grad(L, M, N) -> torch.Tensor:
    """d/d(L, M, N) of :func:`f_angular_sf`, shape ``(3, 1, 1, 7, P)``."""
    return _sf_block_grad(L, M, N)


def f_angular_pf_grad(L, M, N) -> torch.Tensor:
    """d/d(L, M, N) of :func:`f_angular_pf`, shape ``(3, 2, 3, 7, P)``."""
    return _pf_block_grad(L, M, N)


def f_angular_df_grad(L, M, N) -> torch.Tensor:
    """d/d(L, M, N) of :func:`f_angular_df`, shape ``(3, 3, 5, 7, P)``."""
    return _df_block_grad(L, M, N)


def f_angular_ff_grad(L, M, N) -> torch.Tensor:
    """d/d(L, M, N) of :func:`f_angular_ff`, shape ``(3, 4, 7, 7, P)``."""
    return _ff_block_grad(L, M, N)


# A name -> function lookup, so callers can pick a block by its short string
# ("sf", "pf", "df", "ff") instead of importing four separate functions.  The
# strings match the leading part of the corresponding channel names.
F_ANGULAR_HELPERS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "sf": f_angular_sf,
        "pf": f_angular_pf,
        "df": f_angular_df,
        "ff": f_angular_ff,
    }
)


# =============================================================================
# THE MAIN ASSEMBLER
# =============================================================================
# Everything above was preparation; this is the function that actually builds a
# matrix.  Its shape is worth understanding before reading any of it, because
# the body is roughly two thousand lines of the same four steps repeated:
#
#   For each kind of orbital-shell pairing (s-s, s-p, p-s, p-p, s-d, p-d, d-s,
#   d-p, d-d, then the f blocks):
#     1. Build a boolean mask selecting the atom pairs that have *both* shells
#        involved.  A hydrogen atom has no d shell, so it takes no part in the
#        d-d step; the mask is how that is expressed.
#     2. Evaluate the radial functions for exactly those pairs, by looking up
#        their spline coefficients and evaluating the cubic.
#     3. Multiply by the angular polynomials in L, M, N -- the hard-coded
#        Slater-Koster expressions -- to get the matrix entries, and add them
#        into H0 at the right flat positions.
#     4. Do the same for the derivative, which is the product rule applied to
#        step 3, and add it into dH0.
#
# It is written out longhand rather than looped because each shell pairing has
# a different number of radial functions and a different set of angular
# polynomials, and because writing it flat lets every atom pair in the whole
# structure be processed simultaneously as one big tensor operation ("vectorized"
# in the name) instead of one at a time in Python.
#
# The commented-out decorator below would hand the whole function to PyTorch's
# compiler for extra speed; it is left off because the data-dependent masking
# in here does not reliably compile.
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
    SH_shift: str,
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
        masks.  Callers that predate f routing may omit them; ``None`` means
        "this caller has no f pairs".

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

    SH_shift : str
        Selects the Hamiltonian (``"H"``) or overlap (``"S"``) half of the
        canonical channel list; it is the channel-name prefix itself.

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

    # ---- Step 0: mask housekeeping ----------------------------------------
    # Callers that predate f routing pass ``None`` for the 16-orbital masks.
    # Normalise them to all-False tensors so the s/p/d masks below can be
    # extended with ``|`` unconditionally, and record whether any f pair is
    # actually present so the f blocks can be skipped entirely otherwise.
    #
    # ``torch.zeros_like`` copies the shape, dtype and device of an existing
    # mask, so ``_f_none`` is an all-False mask of exactly the right kind: it
    # selects no pairs at all, and OR-ing it into another mask changes nothing.
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
    # ---- Step 0b: optional stress accumulation ---------------------------
    # Helper: optionally accumulate weighted per-pair gradient for stress.
    # dxyz has shape (3, P_masked). W is the (HDIM,HDIM) density-weight matrix.
    # _sg(mask, row_offset, col_offset, dxyz_3P) does:
    #   pair_grad[mask, :] += W[i0+row, j0+col] * dxyz_3P^T
    #
    # What this is for.  Stress is how the total energy responds to squashing
    # or stretching the whole cell.  Getting it needs, for every atom pair, the
    # derivative of that pair's matrix entries weighted by how occupied those
    # orbitals are -- the weight matrix W.  The obvious way to compute it is to
    # keep every per-pair derivative around and contract at the end, but that
    # tensor is enormous.  Instead each derivative is contracted with W the
    # moment it is computed and thrown away, leaving only a running total of
    # shape (pairs, 3).  That is what ``_sg`` (short for "stress gradient")
    # does, and why a call to it follows nearly every derivative below.
    _W = stress_weight
    if _W is not None:
        # dR_dxyz has shape (3, pairs), so dR_dxyz.shape[1] is the pair count.
        pair_grad = torch.zeros(
            (dR_dxyz.shape[1], 3), dtype=dR_dxyz.dtype, device=dR_dxyz.device
        )
        _i0 = i0_stress
        _j0 = j0_stress
    else:
        # No stress requested: pair_grad stays None and every _sg call below
        # returns immediately, costing one comparison each.
        pair_grad = None

    def _sg(mask, row_off, col_off, dxyz):
        """Accumulate W[i0+row_off, j0+col_off] * dxyz into pair_grad."""
        if pair_grad is None:
            return
        # ``mask is None`` means the caller is working with all pairs at once
        # (only the s-s block does that, since every atom has an s orbital).
        # Otherwise only the masked subset is touched.
        if mask is None:
            w = _W[_i0 + row_off, _j0 + col_off]  # (P,)
            # dxyz arrives as (3, pairs) but pair_grad is (pairs, 3), hence the
            # transpose; unsqueeze turns w into (pairs, 1) so it broadcasts
            # across the three Cartesian directions.
            pair_grad[:] += w.unsqueeze(-1) * dxyz.T
        else:
            w = _W[_i0[mask] + row_off, _j0[mask] + col_off]  # (P_mask,)
            pair_grad[mask] += w.unsqueeze(-1) * dxyz.T

    # ---- Step 0c: the output buffers -------------------------------------
    # Both start at zero and are filled by accumulation.  H0 is the flattened
    # HDIM-by-HDIM matrix; dH0 holds its three Cartesian derivatives, one row
    # each for d/dx, d/dy, d/dz.  dtype and device are inherited from the input
    # so the whole routine stays on whichever hardware the caller is using.
    H0 = torch.zeros((HDIM * HDIM), dtype=dR_dxyz.dtype, device=dR_dxyz.device)
    dH0 = torch.zeros(3, HDIM * HDIM, dtype=H0.dtype, device=H0.device)

    # ---- Step 0d: where the radial functions come from --------------------
    # -- ML / spline evaluation helper --
    # Two possible sources, chosen once here: the tabulated splines (normal),
    # or a trained neural network if the caller supplied a context for one.
    _use_ml = ml_ctx is not None

    def _get_val_dR(pair_type_sel, idx_sel, dx_sel, channel, mask, direction="IJ"):
        """Return (value, dvalue_dR) for a single named SK channel.

        ``channel`` is a base channel name without its H/S prefix, e.g. ``"ss0"``
        or ``"pd1"``; ``SH_shift`` selects the Hamiltonian or overlap variant.
        """
        # This is the one place a radial function is ever obtained.  It returns
        # a pair: the value at the current distance, and its derivative with
        # respect to that distance.  ``direction`` is "IJ" or "JI" and says
        # which atom of the pair is being treated as the first one -- see the
        # note on IJ_pair_type versus JI_pair_type in the argument docs above.
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
            # The normal path: evaluate a tabulated cubic spline.
            #
            # A spline stores a curve as a chain of short cubic pieces.  The
            # caller has already worked out, for each atom pair, which piece
            # applies (``idx_sel``) and how far into that piece the pair's
            # distance falls (``dx_sel``).  So all that is left is to look up
            # that piece's four coefficients and evaluate
            #
            #     value = a0 + a1*dx + a2*dx^2 + a3*dx^3
            #
            # ``coeffs_tensor`` is indexed by [which element pair, which piece
            # of the curve, which channel], and the trailing axis holds those
            # four coefficients.
            ch = sk_channel_index(sk_channel_name(channel, SH_shift))
            cs = coeffs_tensor[pair_type_sel, idx_sel, ch]
            val = (
                cs[:, 0]
                + cs[:, 1] * dx_sel
                + cs[:, 2] * dx_sel**2
                + cs[:, 3] * dx_sel**3
            )
            # The derivative is the same cubic differentiated term by term:
            #     a1 + 2*a2*dx + 3*a3*dx^2
            # This is d(value)/dR, the change per unit *distance*.  Converting
            # it to a change per unit x, y or z is the caller's job and is done
            # by multiplying by dR_dxyz.
            dval = cs[:, 1] + 2 * cs[:, 2] * dx_sel + 3 * cs[:, 3] * dx_sel**2
            return val, dval

    # -- end ML / spline helper --

    # =====================================================================
    # BLOCK 1: s with s.  The simplest case and the template for all the
    # rest, so it is worth reading closely.
    # =====================================================================
    # No mask is needed: every atom has an s orbital, so every pair
    # contributes.  ``slice(None)`` is Python for "everything", standing in
    # for the mask that the other blocks pass.
    #
    # There is also no angular factor.  Two spheres look the same from every
    # direction, so the s-s matrix entry is the radial function alone -- which
    # is why no L, M or N appears in the next few lines.
    #######
    HSSS_all, HSSS_dR = _get_val_dR(IJ_pair_type, idx, dx, "ss0", slice(None), "IJ")
    # Place each pair's value at its matrix position.  H_INDEX_START[atom] is
    # that atom's first orbital, which for the s block is the s orbital itself,
    # so the flat position is (row atom's s) * HDIM + (column atom's s).
    # index_add_ *adds* rather than assigns, so a matrix entry touched by more
    # than one contribution ends up with their sum.
    H0.index_add_(
        0, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J], HSSS_all
    )
    #######

    # H-H 
    ######### dH/dx
    # The derivative, via the chain rule.  HSSS_dR is d(value)/dR and dR_dxyz
    # is dR/d(x, y, z) for each pair, so their product is d(value)/d(x, y, z) --
    # a (3, pairs) tensor, one row per Cartesian direction.  This block needs
    # no product rule because there is no angular factor to differentiate.
    HSSS_dxyz = HSSS_dR * dR_dxyz
    # Axis 1 here, not 0, because dH0 is (3, HDIM*HDIM): the flat matrix
    # position is its second axis.
    dH0.index_add_(
        1, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J], HSSS_dxyz
    )
    # Feed the same derivative to the stress accumulator.  Offsets (0, 0)
    # because this is the s-with-s corner of each atom's block; mask None
    # because all pairs took part.
    _sg(None, 0, 0, HSSS_dxyz)
    #########

    # =====================================================================
    # BLOCK 2: s on atom I with p on atom J.
    # =====================================================================
    # H-X 
    ###### HSPS_all
    # Needs a p shell on atom J, i.e. n_orb(J) in {4, 9, 16}.
    # Recall the letters: H = 1 orbital, X = 4, Y = 9, Z = 16, first letter is
    # atom I and second is atom J.  The condition is on atom J ALONE: it must
    # have a p shell, i.e. the second letter is X, Y or Z.  Atom I's class is
    # irrelevant here, so all four values of the first letter take part and the
    # complete list is 4 x 3 = 12 classes.  Only pairs ending in H are absent,
    # because an H atom has no p shell to couple to.
    tmp_mask = (
        pair_mask_HX
        | pair_mask_XX
        | pair_mask_YX
        | pair_mask_ZX
        | pair_mask_HY
        | pair_mask_XY
        | pair_mask_YY
        | pair_mask_ZY
        | pair_mask_HZ
        | pair_mask_XZ
        | pair_mask_YZ
        | pair_mask_ZZ
    )
    # From here on everything is subsetted by ``tmp_mask``: only the selected
    # pairs are gathered, so the tensors below are shorter than the full pair
    # list and stay aligned with each other.
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    # One radial function only: s-p can couple head-on but not sideways, since
    # the s sphere has nothing to offer a pi coupling.  Channel "sp0" = sigma.
    HSPS_all, HSPS_dR = _get_val_dR(
        sel_IJ, sel_idx, dx[tmp_mask], "sp0", tmp_mask, "IJ"
    )

    # The angular factors, and the first real Slater-Koster expressions in the
    # file.  The intuition is direct: the coupling between a sphere and a
    # p orbital pointing along x is proportional to how much the bond points
    # along x -- which is exactly L.  Likewise M for py and N for pz.
    # Columns +1, +2, +3 are px, py, pz in each atom's local orbital order.
    H0.index_add_(0, idx_row * HDIM + idx_col + 1, L[tmp_mask] * HSPS_all)
    H0.index_add_(0, idx_row * HDIM + idx_col + 2, M[tmp_mask] * HSPS_all)
    H0.index_add_(0, idx_row * HDIM + idx_col + 3, N[tmp_mask] * HSPS_all)

    ######### dH/dx
    # The product rule, in the exact form used by every remaining block.  Each
    # entry is (angular) * (radial), so its derivative has two terms:
    #
    #   d/dxyz [ L * V ]  =  L * dV/dxyz  +  dL/dxyz * V
    #                        \______/        \_______/
    #                     radial moves     direction moves
    #
    # The first term is the atoms getting closer or further apart; the second
    # is the bond swinging round to point a different way.  Both matter, and
    # forgetting the second is the classic error here.
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, tmp_mask]

    HSPS_sp_L_dxyz = L[tmp_mask] * HSPS_dxyz + L_dxyz[:, tmp_mask] * HSPS_all
    HSPS_sp_M_dxyz = M[tmp_mask] * HSPS_dxyz + M_dxyz[:, tmp_mask] * HSPS_all
    HSPS_sp_N_dxyz = N[tmp_mask] * HSPS_dxyz + N_dxyz[:, tmp_mask] * HSPS_all
    dH0.index_add_(1, idx_row * HDIM + idx_col + 1, HSPS_sp_L_dxyz)
    dH0.index_add_(1, idx_row * HDIM + idx_col + 2, HSPS_sp_M_dxyz)
    dH0.index_add_(1, idx_row * HDIM + idx_col + 3, HSPS_sp_N_dxyz)
    # Row offset 0 (the s orbital), column offsets 1, 2, 3 (px, py, pz).
    _sg(tmp_mask, 0, 1, HSPS_sp_L_dxyz)
    _sg(tmp_mask, 0, 2, HSPS_sp_M_dxyz)
    _sg(tmp_mask, 0, 3, HSPS_sp_N_dxyz)
    #########

    # =====================================================================
    # BLOCK 3: p on atom I with s on atom J -- the mirror image of block 2.
    # =====================================================================
    ### HPSS_all ###
    # Now it is the *first* atom that needs a p shell, so the mask lists every
    # class whose first letter is X, Y or Z -- and, mirroring block 2, atom J's
    # class is irrelevant, so all four values of the second letter take part.
    # 3 x 4 = 12 classes.
    tmp_mask = (
        pair_mask_XH
        | pair_mask_XX
        | pair_mask_XY
        | pair_mask_XZ
        | pair_mask_YH
        | pair_mask_YX
        | pair_mask_YY
        | pair_mask_YZ
        | pair_mask_ZH
        | pair_mask_ZX
        | pair_mask_ZY
        | pair_mask_ZZ
    )
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    # Note JI_pair_type, not IJ_pair_type, and "JI" rather than "IJ".  The
    # tabulated radial functions are stored per *ordered* element pair -- the
    # s-p function for (silicon, oxygen) is a different curve from the one for
    # (oxygen, silicon), because it is a different atom's s orbital meeting a
    # different atom's p orbital.  Since we now want atom J's s with atom I's
    # p, the pair must be looked up the other way round.
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    HPSS_all, HPSS_dR = _get_val_dR(
        sel_IJ, sel_idx, dx[tmp_mask], "sp0", tmp_mask, "JI"
    )

    # The minus signs are the whole point of this block.  The tabulated "sp0"
    # curve is defined for the bond direction running from the s atom to the p
    # atom.  Here the bond runs the other way, so the direction cosines fed to
    # the formula ought to be (-L, -M, -N).  The angular factor is odd -- a
    # single power of L -- so flipping the direction just flips the sign, and
    # negating once here is equivalent to negating the direction.
    #
    # Note also that the row and column roles have swapped: the p offsets
    # (+1, +2, +3) are now added to the *row*, and the s sits in the column.
    H0.index_add_(0, (idx_row + 1) * HDIM + idx_col, -L[tmp_mask] * HPSS_all)
    H0.index_add_(0, (idx_row + 2) * HDIM + idx_col, -M[tmp_mask] * HPSS_all)
    H0.index_add_(0, (idx_row + 3) * HDIM + idx_col, -N[tmp_mask] * HPSS_all)

    ################
    ######### dH/dx
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, tmp_mask]

    # Same product rule as block 2, with the overall minus carried through both
    # terms.
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

    # =====================================================================
    # BLOCK 4: p with p.  The first block with two bond symmetries.
    # =====================================================================
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
    # The ``_XX`` suffix on these names just marks "subsetted to this block's
    # mask"; it is a leftover from when this block only handled X-X pairs.
    L_XX = L[tmp_mask]
    M_XX = M[tmp_mask]
    N_XX = N[tmp_mask]
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    # No JI variant needed: p-p is symmetric between the two atoms in the sense
    # that the reversed block is supplied separately by the (J, I) entry of the
    # neighbour list, so each direction is handled on its own pass.
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    # Two radial functions now.  "pp0" is sigma -- both p orbitals aimed along
    # the bond, lobe meeting lobe.  "pp1" is pi -- both perpendicular to the
    # bond, lying side by side.  HPPS and HPPP are their values.
    HPPS, HPPS_dR = _get_val_dR(sel_IJ, sel_idx, dx[tmp_mask], "pp0", tmp_mask, "IJ")

    HPPP, HPPP_dR = _get_val_dR(sel_IJ, sel_idx, dx[tmp_mask], "pp1", tmp_mask, "IJ")

    # The p-p angular rule, and the cleanest illustration of the whole
    # Slater-Koster idea.  Take px on one atom and py on the other.  Split each
    # into the part along the bond and the part across it.  The along-bond
    # parts couple through sigma, the across-bond parts through pi, and adding
    # the two contributions gives
    #
    #     E(pa, pb) = (component of a along bond)(component of b along bond)
    #                 * (sigma - pi)  +  (1 if a == b else 0) * pi
    #
    # The components along the bond are just L, M, N.  So the diagonal entries
    # pick up an extra whole unit of pi, and every entry is scaled by the
    # difference sigma - pi -- which is precomputed once here as ``PPSMPP``
    # ("PP Sigma Minus PP Pi").
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

    # Nine writes: rows +1..+3 (px, py, pz of atom I) against columns +1..+3
    # (px, py, pz of atom J).
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

    # Product rule again, now with two direction cosines per term.  Reading
    # PXPX: the value was pi + L*L*(sigma - pi), so its derivative is
    # d(pi) + L*L*d(sigma - pi) + 2*L*dL*(sigma - pi) -- one term for each
    # factor that can change.  The off-diagonal entries such as PXPY have no
    # standalone pi term and instead get dL*M + L*dM.
    #
    # Every remaining block in this function follows this same shape, just with
    # longer angular polynomials, so the pattern is not re-explained below.
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

    # =====================================================================
    # BLOCK 5: s on atom I with d on atom J.
    # =====================================================================
    # From here the angular polynomials get longer, but nothing structurally
    # new happens: mask, gather, evaluate radial functions, multiply by
    # angular factors, write values, then write derivatives via the product
    # rule, then feed the stress accumulator.  Only the parts specific to each
    # block are commented from now on.
    #
    # Column offsets +4..+8 are the five d orbitals in this program's fixed
    # local order: dxy, dyz, dzx, dx2-y2, dz2.
    ### s-d
    # Needs a d shell on atom J, i.e. n_orb(J) in {9, 16} -- second letter Y or
    # Z, with atom I's class unconstrained.  4 x 2 = 8 classes.
    tmp_mask = (
        pair_mask_HY
        | pair_mask_XY
        | pair_mask_YY
        | pair_mask_ZY
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
    # One radial function again -- a sphere against anything can only couple
    # head-on, so s-d is sigma only, channel "sd0".
    V_sd_sigma, V_sd_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "sd0", tmp_mask, "IJ"
    )
    # The angular factor for a sphere against a d orbital is simply that d
    # orbital's own shape, evaluated in the bond direction: substitute
    # (x, y, z) -> (L, M, N) into the polynomial that names the orbital.  So
    # dxy gives L*M, dx2-y2 gives L^2 - M^2, dz2 gives N^2 - (L^2 + M^2)/2.
    # The sqrt(3) and 1/2 factors are the orbitals' normalisation constants.
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

    # =====================================================================
    # BLOCK 6: p on atom I with d on atom J.
    # =====================================================================
    # Fifteen entries: three p rows (+1, +2, +3) times five d columns
    # (+4..+8), each built from two radial functions (sigma "pd0" and pi
    # "pd1").  The variable names read H_<p orbital>_<d orbital>, so H_X_YZ is
    # px against dyz.
    ### p-d
    # Needs a p shell on atom I and a d shell on atom J -- first letter X, Y or
    # Z, second letter Y or Z.  3 x 2 = 6 classes.
    tmp_mask = (
        pair_mask_XY
        | pair_mask_YY
        | pair_mask_ZY
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

    # =====================================================================
    # BLOCK 7: d on atom I with s on atom J -- the mirror of block 5.
    # =====================================================================
    # The d offsets (+4..+8) now go on the row and the s sits in the column,
    # and the radial function is looked up with JI_pair_type / "JI" for the
    # same ordered-pair reason given in block 3.
    #
    # Unlike block 3, there is no minus sign here.  A d orbital's angular
    # polynomial is even -- every term has two powers of L, M or N -- so
    # reversing the bond direction (L, M, N) -> (-L, -M, -N) leaves it
    # unchanged.  The general rule is that the sign flips by (-1) raised to the
    # sum of the two shells' angular momenta (s = 0, p = 1, d = 2, f = 3): s-p
    # gives -1, s-d gives +1, p-d gives -1, d-d gives +1.
    ### d-s
    # Needs a d shell on atom I, i.e. n_orb(I) in {9, 16} -- first letter Y or
    # Z, with atom J's class unconstrained.  2 x 4 = 8 classes.
    tmp_mask = (
        pair_mask_YH
        | pair_mask_YX
        | pair_mask_YY
        | pair_mask_YZ
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

    # =====================================================================
    # BLOCK 8: d on atom I with p on atom J -- the mirror of block 6.
    # =====================================================================
    # Every expression below is the matching block 6 formula wrapped in a
    # leading minus, following the parity rule just stated: d and p sum to
    # 2 + 1 = 3, which is odd, so reversing the bond flips the sign.  The
    # wrapping is literal -- ``H_XY_X`` is exactly ``-H_X_XY`` from block 6 --
    # which is why these read as negated copies rather than fresh algebra.
    ### d-p
    # Needs a d shell on atom I and a p shell on atom J -- first letter Y or Z,
    # second letter X, Y or Z.  2 x 3 = 6 classes.
    tmp_mask = (
        pair_mask_YX
        | pair_mask_YY
        | pair_mask_YZ
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

    # =====================================================================
    # BLOCK 9: d with d.  Twenty-five entries, three bond symmetries.
    # =====================================================================
    # This block computes the energy coupling between every d orbital on atom I
    # and every d orbital on atom J.  Everything it needs is explained here, so
    # the block can be read without looking anywhere else.
    #
    # THE FIVE d ORBITALS.  A d shell holds five orbitals.  Four of them are
    # "cloverleaves": four lobes in a plane, alternating in sign, like a
    # four-bladed propeller.  They are named for the plane they lie in and the
    # axes their lobes point between:
    #
    #     dxy      cloverleaf in the xy plane, lobes between the x and y axes
    #     dyz      the same, in the yz plane
    #     dzx      the same, in the zx plane
    #     dx2-y2   also in the xy plane, but rotated 45 degrees so its lobes
    #              point ALONG +-x and +-y instead of between them
    #     dz2      the odd one out: a single fat lobe along +z, another along
    #              -z, and a doughnut of opposite sign around the equator
    #
    # They sit at local offsets 4, 5, 6, 7, 8 inside an atom's run of orbitals
    # (offset 0 is s, 1-3 are the three p orbitals).  That is where the
    # ``+ 4`` ... ``+ 8`` in the index arithmetic below comes from.
    #
    # THE BOND DIRECTION.  L, M, N are the "direction cosines" of the bond: the
    # unit vector pointing from atom I to atom J, so L^2 + M^2 + N^2 = 1.  L is
    # how much of that vector lies along x, M along y, N along z.  Every angular
    # polynomial below is written purely in L, M and N.
    #
    # WHY EXACTLY THREE RADIAL FUNCTIONS.  Take the line joining the two atoms
    # as an axis and spin an orbital around it.  Any orbital splits into pieces
    # that behave differently under that spin: a piece that does not change at
    # all, a piece that goes through one full sign cycle per turn, and a piece
    # that goes through two.  Call those 0, 1 and 2.  A d orbital reaches 2 and
    # no higher.  Two orbitals can only couple if their pieces spin at the SAME
    # rate -- a mismatched pair averages to zero as you go around the axis --
    # so the surviving pairings are 0-with-0, 1-with-1 and 2-with-2.  Three
    # pairings, three tabulated radial functions:
    #
    #     sigma  "dd0"  the 0-0 coupling: lobes meeting head-on along the bond
    #     pi     "dd1"  the 1-1 coupling: lobes meeting side-on
    #     delta  "dd2"  the 2-2 coupling: cloverleaf face to cloverleaf face
    #
    # HOW TO READ ANY FORMULA BELOW.  Each entry has the same three-line shape,
    #
    #     H = (polynomial in L, M, N) * V_dd_sigma
    #       + (polynomial in L, M, N) * V_dd_pi
    #       + (polynomial in L, M, N) * V_dd_delta
    #
    # where the V's are the tabulated radial functions -- numbers that depend
    # only on how far apart the atoms are -- and the polynomials say how much of
    # each coupling this particular pair of orbital shapes actually achieves at
    # this particular bond direction.
    #
    # THE SIGMA POLYNOMIALS ARE PRODUCTS OF TWO PROJECTIONS.  This is the single
    # fact that makes the whole block readable.  Every sigma polynomial is just
    #
    #     (how much of the row orbital points along the bond)
    #   * (how much of the column orbital points along the bond)
    #
    # and those five per-orbital projections are:
    #
    #     dxy      sqrt(3) * L * M
    #     dyz      sqrt(3) * M * N
    #     dzx      sqrt(3) * N * L
    #     dx2-y2   (sqrt(3)/2) * (L^2 - M^2)
    #     dz2      N^2 - (L^2 + M^2)/2
    #
    # So H_XY_XY's sigma term 3*L^2*M^2 is (sqrt(3)LM)^2; H_XY_Z2's sigma term
    # is (sqrt(3)LM) * (N^2 - (L^2+M^2)/2); H_Z2_Z2's is (N^2-(L^2+M^2)/2)^2.
    # Every one of the twenty-five sigma terms factorises this way.  Pi and
    # delta do NOT collapse to a single product, because each of those has two
    # equivalent pieces (spinning one way and the other), so their polynomials
    # are a sum of two such products -- which is exactly why they look messier.
    #
    # A CHECK YOU CAN DO BY HAND.  On the diagonal (row orbital == column
    # orbital) the three polynomials must add up to exactly 1, for any bond
    # direction.  Take H_XY_XY:
    #
    #     3L^2M^2 + (L^2 + M^2 - 4L^2M^2) + (N^2 + L^2M^2)
    #        = L^2 + M^2 + N^2 = 1.
    #
    # That is the statement that an orbital coupled to itself must be fully
    # accounted for by the three symmetries and nothing leaks away.  It holds
    # for all five diagonal entries and is the quickest way to catch a typo.
    #
    # WHICH PART OF THE MATRIX THIS FILLS.  All 25 entries of the d-d square for
    # the ordered pair (I, J): rows are atom I's five d orbitals, columns are
    # atom J's.  The transposed square, with I and J swapped, is filled when the
    # reversed neighbour pair (J, I) comes round as its own entry in the
    # neighbour list -- so this block deliberately does not write it.
    #
    # Variable names read H_<row d orbital>_<column d orbital>, with X2Y2
    # meaning dx2-y2 and Z2 meaning dz2.
    ### d-d
    # Which neighbour pairs take part.  Both atoms must actually have a d shell.
    # The letters encode how many orbitals an atom carries: H = 1, X = 4, Y = 9,
    # Z = 16, and the two letters are (atom I, atom J).  Only 9 and 16 include a
    # d shell, so the four combinations of {Y, Z} with {Y, Z} are exactly the
    # pairs this block applies to.  ``|`` is boolean OR on the masks, i.e. "any
    # of these four kinds".
    tmp_mask = pair_mask_YY | pair_mask_YZ | pair_mask_ZY | pair_mask_ZZ
    # From here on everything is "gathered": each line below picks out only the
    # entries belonging to the pairs selected by ``tmp_mask`` and packs them
    # into a short dense vector.  If the structure has 10000 neighbour pairs but
    # only 300 are d-d, every vector from here on has length 300.  All of them
    # keep the same ordering, so position p in one lines up with position p in
    # all the others -- that is what lets the formulas below be plain arithmetic
    # on whole vectors instead of a loop over pairs.
    #
    # ``neighbor_I[tmp_mask]`` is "which atom is the first atom of each selected
    # pair"; ``H_INDEX_START[that]`` turns an atom number into the position of
    # its first orbital in the big matrix.  So idx_row[p] is where pair p's
    # first atom begins, and idx_col[p] where its second atom begins.
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    # How far each pair's separation sits past the start of its spline interval.
    # The radial functions are stored as a piecewise cubic in the separation;
    # ``tmp_dx`` is the local coordinate that cubic is evaluated at.
    tmp_dx = dx[tmp_mask]
    # The bond direction for each selected pair.
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    # Which element-pair table to read (Eu-N and N-Eu are different tables), and
    # which interval of that table's spline this separation falls in.
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    # Look up the three radial functions.  Each call returns two vectors: the
    # value of that function at this separation, and its derivative with respect
    # to the separation (needed only by the force half of the block, further
    # down).  "IJ" says to read the table in the I-to-J direction.
    V_dd_sigma, V_dd_sigma_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "dd0", tmp_mask, "IJ"
    )
    V_dd_pi, V_dd_pi_dR = _get_val_dR(sel_IJ, sel_idx, tmp_dx, "dd1", tmp_mask, "IJ")
    V_dd_delta, V_dd_delta_dR = _get_val_dR(
        sel_IJ, sel_idx, tmp_dx, "dd2", tmp_mask, "IJ"
    )

    # ---- Row 1 of 5: dxy on atom I against each of atom J's five d orbitals.
    # Recall the sigma projection of dxy is sqrt(3)*L*M, so every sigma term in
    # this row is sqrt(3)*L*M times the column orbital's own projection.
    #
    # dxy with dxy.  sigma: (sqrt(3)LM)^2 = 3L^2M^2.  The pi and delta terms are
    # the two-piece sums described in the header.  All three add to 1.
    H_XY_XY = (
        3 * tmp_L**2 * tmp_M**2 * V_dd_sigma
        + (tmp_L**2 + tmp_M**2 - 4 * tmp_L**2 * tmp_M**2) * V_dd_pi
        + (tmp_N**2 + tmp_L**2 * tmp_M**2) * V_dd_delta
    )
    # dxy with dyz.  sigma: (sqrt(3)LM)(sqrt(3)MN) = 3*L*M^2*N.
    H_XY_YZ = (
        3 * tmp_L * tmp_M**2 * tmp_N * V_dd_sigma
        + tmp_L * tmp_N * (1 - 4 * tmp_M**2) * V_dd_pi
        + tmp_L * tmp_N * (tmp_M**2 - 1) * V_dd_delta
    )
    # dxy with dzx.  sigma: (sqrt(3)LM)(sqrt(3)NL) = 3*L^2*M*N.
    H_XY_ZX = (
        3 * tmp_L**2 * tmp_M * tmp_N * V_dd_sigma
        + tmp_M * tmp_N * (1 - 4 * tmp_L**2) * V_dd_pi
        + tmp_M * tmp_N * (tmp_L**2 - 1) * V_dd_delta
    )
    # dxy with dx2-y2.  sigma: (sqrt(3)LM)*((sqrt(3)/2)(L^2-M^2)) = 1.5LM(L^2-M^2).
    H_XY_X2Y2 = (
        1.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_sigma
        + 2 * tmp_L * tmp_M * (tmp_M**2 - tmp_L**2) * V_dd_pi
        + 0.5 * tmp_L * tmp_M * (tmp_L**2 - tmp_M**2) * V_dd_delta
    )
    # dxy with dz2.  sigma: (sqrt(3)LM)*(N^2 - (L^2+M^2)/2).  The bare sqrt(3)
    # written as ``3**0.5`` is the leftover of dxy's own normalisation, since
    # dz2's projection carries no root.
    H_XY_Z2 = (
        (3**0.5) * tmp_L * tmp_M * (tmp_N**2 - 0.5 * (tmp_L**2 + tmp_M**2)) * V_dd_sigma
        - (3**0.5) * 2 * tmp_L * tmp_M * tmp_N**2 * V_dd_pi
        + (3**0.5) * 0.5 * tmp_L * tmp_M * (1 + tmp_N**2) * V_dd_delta
    )
    # Store row 1.  H0 is kept flat: the HDIM-by-HDIM matrix is laid out as one
    # long vector, row after row, so matrix entry (r, c) lives at position
    # r*HDIM + c.  Here the row is idx_row + 4 (atom I's dxy) and the column
    # runs over idx_col + 4 ... idx_col + 8 (atom J's five d orbitals).  Both
    # index expressions are whole vectors, one entry per selected pair, so each
    # call writes every pair's contribution at once.
    #
    # ``index_add_`` adds rather than assigns.  That matters because two
    # different neighbour pairs can land on the same matrix entry -- the same
    # two atoms seen through different periodic images of the cell -- and those
    # contributions must sum, not overwrite one another.
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 4, H_XY_XY)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 5, H_XY_YZ)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 6, H_XY_ZX)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 7, H_XY_X2Y2)
    H0.index_add_(0, (idx_row + 4) * HDIM + idx_col + 8, H_XY_Z2)

    # ---- Row 2 of 5: dyz on atom I.  Its sigma projection is sqrt(3)*M*N, so
    # this row repeats row 1 with the axis roles cycled x -> y -> z -> x.  The
    # first entry, dyz with dxy, is the mirror of H_XY_YZ above and is identical
    # to it: the coefficient matrix is symmetric, since swapping the two
    # orbitals swaps two factors in a product.
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
    # Store row 2, on matrix row idx_row + 5 (atom I's dyz).
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 4, H_YZ_XY)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 5, H_YZ_YZ)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 6, H_YZ_ZX)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 7, H_YZ_X2Y2)
    H0.index_add_(0, (idx_row + 5) * HDIM + idx_col + 8, H_YZ_Z2)

    # ---- Row 3 of 5: dzx on atom I.  Sigma projection sqrt(3)*N*L -- the axis
    # roles cycled once more.  Its first two entries mirror H_XY_ZX and
    # H_YZ_ZX above and repeat them verbatim.
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
    # Store row 3, on matrix row idx_row + 6 (atom I's dzx).
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 4, H_ZX_XY)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 5, H_ZX_YZ)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 6, H_ZX_ZX)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 7, H_ZX_X2Y2)
    H0.index_add_(0, (idx_row + 6) * HDIM + idx_col + 8, H_ZX_Z2)

    # ---- Row 4 of 5: dx2-y2 on atom I.  Sigma projection (sqrt(3)/2)(L^2-M^2).
    # This orbital is NOT reachable by cycling the axes -- rotating x -> y -> z
    # turns it into a mixture of itself and dz2 rather than into a single
    # orbital -- so its row is written out in full rather than derived.  Its
    # first three entries are the mirrors of the X2Y2 columns already seen in
    # rows 1-3.
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
    # Store row 4, on matrix row idx_row + 7 (atom I's dx2-y2).
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 4, H_X2Y2_XY)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 5, H_X2Y2_YZ)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 6, H_X2Y2_ZX)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 7, H_X2Y2_X2Y2)
    H0.index_add_(0, (idx_row + 7) * HDIM + idx_col + 8, H_X2Y2_Z2)

    # ---- Row 5 of 5: dz2 on atom I.  Sigma projection N^2 - (L^2+M^2)/2, the
    # only one of the five with no square root in it, which is why the sqrt(3)
    # factors in this row come entirely from the column orbital.  Like row 4 it
    # resists the axis-cycling trick and is written out in full; its first four
    # entries mirror the Z2 columns of rows 1-4.
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
    # Store row 5, on matrix row idx_row + 8 (atom I's dz2).  The 5x5 d-d
    # square for this ordered pair is now complete.
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 4, H_Z2_XY)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 5, H_Z2_YZ)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 6, H_Z2_ZX)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 7, H_Z2_X2Y2)
    H0.index_add_(0, (idx_row + 8) * HDIM + idx_col + 8, H_Z2_Z2)
    # =====================================================================
    # The same twenty-five entries again, differentiated.
    # =====================================================================
    # Forces are minus the rate of change of the energy as an atom moves, so
    # alongside every matrix entry the code also needs that entry's derivative
    # with respect to the x, y and z coordinates of the atoms.  That is what the
    # rest of this block computes, and it is why it is roughly twice as long as
    # it needs to be for the energy alone.
    #
    # WHERE THE DERIVATIVE COMES FROM.  Every entry above has the form
    #
    #     H = f(L, M, N) * V(R)
    #
    # -- an angular part depending only on the bond direction, times a radial
    # part depending only on the separation.  Moving an atom changes both: it
    # swings the bond direction AND changes the distance.  So the product rule
    # gives two contributions per symmetry:
    #
    #     dH/dxyz = (df/dxyz) * V   +   f * (dV/dxyz)
    #
    # and with three symmetries that is six terms, which is exactly the six-line
    # shape every derivative expression below has: sigma-angular, sigma-radial,
    # pi-angular, pi-radial, delta-angular, delta-radial.  Comparing any
    # derivative against its value expression twenty-five lines above makes the
    # pairing obvious -- the even lines are the value's own terms with the
    # radial factor differentiated, and the odd lines are the same terms with
    # the polynomial differentiated instead.
    #
    # Everything here carries a leading axis of size 3, one slot for d/dx, one
    # for d/dy, one for d/dz, so a single expression handles all three
    # directions at once.
    #
    # How the bond direction changes when an atom moves.
    tmp_L_dxyz = L_dxyz[:, tmp_mask]
    tmp_M_dxyz = M_dxyz[:, tmp_mask]
    tmp_N_dxyz = N_dxyz[:, tmp_mask]
    # How the separation changes when an atom moves.
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    # Chain rule for the radial half: the tables give dV/d(separation), and
    # tmp_dR_dxyz gives d(separation)/d(coordinate), so multiplying them gives
    # dV/d(coordinate) -- the ``f * dV/dxyz`` half of the product rule.
    V_dd_sigma_dxyz = V_dd_sigma_dR * tmp_dR_dxyz
    V_dd_pi_dxyz = V_dd_pi_dR * tmp_dR_dxyz
    V_dd_delta_dxyz = V_dd_delta_dR * tmp_dR_dxyz
    # The d-d derivatives are long enough that the repeated sub-expressions are
    # named once up front and reused, both for legibility and to avoid
    # recomputing the same products twenty-five times over.
    #
    # The naming scheme, decoded: "t" is short for "times", so ``L_t_M`` means
    # L * M and ``L_t_Ldx`` means L * dL/dxyz.  ("m" for minus and "p" for plus
    # are mentioned in the original note but no longer appear.)  ``L2``, ``M2``
    # and ``N2`` are just the squares of the direction cosines.
    #
    # ``L_t_Ldx`` is worth pausing on, because it appears everywhere below: the
    # derivative of L^2 is 2*L*dL/dxyz, so any ``2 * L_t_Ldx`` in a derivative
    # expression is simply "d(L^2)" wherever the value expression had L^2.
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

    # Derivative of H_XY_XY = 3L^2M^2 * Vsigma
    #                       + (L^2 + M^2 - 4L^2M^2) * Vpi
    #                       + (N^2 + L^2M^2) * Vdelta.
    # Line by line: d(3L^2M^2) = 3*(2*L*dL*M^2 + L^2*2*M*dM), then the same
    # 3L^2M^2 against the differentiated sigma radial; then d(L^2+M^2-4L^2M^2)
    # against Vpi and the undifferentiated polynomial against dVpi; then the
    # same two for delta.
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
    # Store row 1's derivatives.  Same flat addressing as the value half, but
    # into dH0 instead of H0 -- and note the leading ``1`` rather than ``0``:
    # dH0 has shape (3, HDIM*HDIM), so axis 0 is the x/y/z direction and the
    # scatter happens along axis 1, the flattened matrix.
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 4, H_XY_XY_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 5, H_XY_YZ_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 6, H_XY_ZX_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 7, H_XY_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row + 4) * HDIM + idx_col + 8, H_XY_Z2_dxyz)
    # The same five derivatives again, this time fed to the stress accumulator.
    # Stress is how the energy responds to stretching the whole cell, and it can
    # be built from the same per-pair gradients that give forces, weighted by
    # how much each matrix entry is actually occupied by electrons.  ``_sg``
    # multiplies each derivative by that occupation weight for entry
    # (row offset, column offset) and adds it into a running per-pair total.
    # When the caller did not ask for stress, ``_sg`` returns immediately and
    # these five lines cost nothing.
    _sg(tmp_mask, 4, 4, H_XY_XY_dxyz)
    _sg(tmp_mask, 4, 5, H_XY_YZ_dxyz)
    _sg(tmp_mask, 4, 6, H_XY_ZX_dxyz)
    _sg(tmp_mask, 4, 7, H_XY_X2Y2_dxyz)
    _sg(tmp_mask, 4, 8, H_XY_Z2_dxyz)
    # ---- Derivatives, row 2 of 5: dyz on atom I.
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
    # ---- Derivatives, row 3 of 5: dzx on atom I.
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
    # ---- Derivatives, row 4 of 5: dx2-y2 on atom I.
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
    # ---- Derivatives, row 5 of 5: dz2 on atom I.
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

    # =====================================================================
    # BLOCK 10: everything involving f orbitals.
    # =====================================================================
    # This block is written differently from the nine above.  Rather than
    # spelling out every AO entry by hand -- s-f, p-f, d-f and f-f together come
    # to 7 + 21 + 35 + 49 = 112 matrix entries, several times the whole s/p/d
    # workload -- it calls the angular table builders defined earlier in this
    # file, which need only about thirty written formulas because rotation and
    # symmetry generate the rest.
    #
    # That is a difference in how the angular factors are *obtained*, not in how
    # they are applied: the arithmetic below is a single broadcast expression
    # over the whole (lower shell x f shell x pair) grid and one scatter per
    # shell pairing, the same shape of work the s/p/d blocks do.  The only
    # Python loop left runs over bond symmetries, of which there are at most
    # four.
    #
    # Nothing here needs guarding against structures that contain no f atom.
    # Every call below is selected by a pair mask, and a mask that selects no
    # pairs makes its call return immediately -- so for an all-s/p/d structure
    # this whole section costs seven mask tests and writes nothing.
    ### f blocks (AO offsets 9..15) ###
    # Angular factors come from the f tables above.  Only the *values* are
    # implemented: no derivative contribution is written, so the f entries of
    # dH0/dS stay exactly zero and every derivative consumer must guard on
    # ``F_ANGULAR_DERIVATIVES_AVAILABLE`` (see FDerivativeUnsupportedError).
    #
    # Spelled out: nothing in this block ever touches dH0 or calls _sg.  The f
    # rows and columns of the derivative therefore keep the zeros they were
    # initialised with.  That is a real limitation, not an oversight -- forces
    # and stresses for an f system would come out wrong-but-plausible, which is
    # exactly what FDerivativeUnsupportedError exists to prevent.
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
        # Nothing to do if no pair of this class exists in this structure.
        if not bool(mask.any()):
            return
        i0 = H_INDEX_START[neighbor_I[mask]]
        j0 = H_INDEX_START[neighbor_J[mask]]
        # Same ordered-pair rule as blocks 3, 7 and 8: look the radial
        # function up in whichever direction puts the two shells the right
        # way round.
        sel_type = JI_pair_type[mask] if direction == "JI" else IJ_pair_type[mask]
        sel_idx = idx[mask]
        sel_dx = dx[mask]
        # One radial function per bond symmetry: ("sf0",) for s-f,
        # ("pf0", "pf1") for p-f, three for d-f, four for f-f.  Index [0]
        # keeps only the value and discards the derivative, which this
        # block does not use.
        radial = [
            _get_val_dR(sel_type, sel_idx, sel_dx, base, mask, direction)[0]
            for base in bases
        ]
        # The angular table for these pairs, shaped
        # (bond symmetry, lower-shell orbital, f orbital, pair).
        coeff = angular(L[mask], M[mask], N[mask])
        n_low = coeff.shape[1]

        # Sum angular * radial over the bond symmetries for every AO cell at
        # once.  ``coeff[k]`` is (n_low, 7, P) and ``radial[k]`` is (P,), so
        # the product broadcasts across the whole block: what the s/p/d
        # blocks write as "sigma term plus pi term plus ..." for a single
        # entry happens here for all n_low * 7 entries in one expression.
        #
        # The surviving loop runs over bond symmetries -- at most four
        # passes -- not over AO cells, and it is written as a running sum
        # rather than ``.sum(0)`` so the accumulation order is exactly the
        # left-to-right one the s/p/d blocks use.
        entries = coeff[0] * radial[0]
        for k in range(1, len(radial)):
            entries = entries + coeff[k] * radial[k]
        # Apply the bond-reversal sign, precomputed by the caller.
        if parity < 0.0:
            entries = -entries

        # Flat destinations for that same (n_low, 7, P) grid.  f orbitals
        # occupy local offsets 9..15, hence the 9 + b.  ``f_rows`` says which
        # side of the matrix the f shell is on: True means the f atom is
        # atom I and its offsets go on the row, False means it is atom J and
        # they go on the column.  ``a`` varies down axis 0 and ``b`` across
        # axis 1, so both broadcast against the pair axis and the index grid
        # lines up cell-for-cell with ``entries``.
        a_idx = torch.arange(n_low, device=H0.device).view(n_low, 1, 1)
        b_idx = torch.arange(7, device=H0.device).view(1, 7, 1)
        row0 = i0.view(1, 1, -1)
        col0 = j0.view(1, 1, -1)
        if f_rows:
            flat = (row0 + 9 + b_idx) * HDIM + col0 + low_offset + a_idx
        else:
            flat = (row0 + low_offset + a_idx) * HDIM + col0 + 9 + b_idx
        # One scatter for the whole block instead of one per AO cell: 49
        # kernel launches become 1 for f-f, 35 become 1 for d-f, and so on.
        # Contributions to a given matrix entry stay in pair order, since
        # flattening keeps each cell's P values contiguous.
        H0.index_add_(0, flat.reshape(-1), entries.reshape(-1))

    # Seven calls follow, covering the seven ways an f atom can pair up.
    # Each shell pairing needs two: one for the f atom on the right of the
    # matrix block, one for it on the left.  f-f needs only one, since both
    # sides are f and the reversed block arrives as a separate neighbour
    # entry.
    #
    # How to read the ``parity`` argument.  The comment above each pair
    # gives the bond-reversal sign for that shell combination, computed as
    # (-1) raised to (lower shell's angular momentum + 3), with s = 0,
    # p = 1, d = 2, f = 3.  That sign applies to the *reversed* block only,
    # so the first call of each pair always passes 1.0 (the table is used
    # as printed) and the second passes the stated value.  When the stated
    # value is +1, as for p-f, both calls pass 1.0.
    #
    # The positional arguments are: mask, radial channels, direction,
    # parity, the lower shell's local offset, the angular table builder,
    # and whether the f shell occupies the rows.
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

    # ---- Reference table (not executed) ----------------------------------
    # The triple-quoted block below is a bare string expression, so Python
    # evaluates it and throws it away -- it is a comment in all but syntax.
    # It restates every s/p/d angular formula used above in one place, with a
    # marker saying where each came from:
    #
    #   $  printed directly in the standard Slater-Koster tables
    #   O  not printed; obtained by relabelling the axes of a printed entry,
    #      the same rotation trick the f section automates
    #
    # It is kept as a check-against reference for anyone auditing the algebra
    # scattered through the blocks above.
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
    # The return is two-or-three-valued depending on whether stress was asked
    # for.  Callers that passed ``stress_weight`` know to unpack three.
    if pair_grad is not None:
        return H0, dH0, pair_grad
    return H0, dH0


# =============================================================================
# THE BATCHED ASSEMBLER
# =============================================================================
# Same physics as the function above, run on many structures at once so that
# one set of tensor operations covers a whole batch instead of looping over
# structures in Python.  The angular formulas are character-for-character
# identical; if you have read them once above you have read them here.
#
# What is genuinely different, and worth understanding before reading on:
#
#   1. Padding.  Different structures have different numbers of atoms and
#      neighbours, but a batched tensor has to be rectangular.  So the
#      neighbour arrays are (batch, max_pairs) and the unused slots are filled
#      with -1.  Those slots are not real pairs and must never contribute.
#
#   2. ``valid_pairs`` and the two index spaces.  Because of that padding,
#      quantities in this function live in one of two shapes: the padded
#      (batch, max_pairs) grid, or a flat compacted list holding only the real
#      pairs.  The masks are in padded space, but ``dx``, ``L``, ``M``, ``N``
#      and friends are compacted.  That is why you will see a mask written
#      plainly as ``tmp_mask`` when indexing a padded array, and as
#      ``tmp_mask[valid_pairs]`` when indexing a compacted one -- the second
#      form drops the padded entries so the mask lines up.  Mixing the two up
#      would silently read the wrong pair's data.
#
#   3. ``batch_block_offset``.  Each structure gets its own HDIM-by-HDIM matrix,
#      and all of them are stored end to end in one flat buffer of length
#      batch * HDIM * HDIM.  Adding ``structure_index * HDIM * HDIM`` to a flat
#      position is what steers a write into the right structure's matrix.
#
#   4. What is missing.  This function supports neither stress accumulation nor
#      the neural-network radial source nor f orbitals -- there are no Z masks
#      in its signature at all.  It always returns exactly two values.
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
    SH_shift: str,
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

    SH_shift : str
        Selects the Hamiltonian (``"H"``) or overlap (``"S"``) half of the
        canonical channel list; it is the channel-name prefix itself.

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

    .. warning::
       The ``stress_weight``, ``i0_stress`` and ``j0_stress`` entries above are
       stale: this function's signature accepts none of them, and it has no
       stress accumulation at all.  They were copied from the single-structure
       routine.  Likewise the ``pair_grad`` return described below never
       happens here -- this function always returns exactly ``(H0, dH0)``.
       Use the single-structure routine if you need stress.

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

    # One flat buffer holding every structure's matrix end to end, hence the
    # leading ``batch_size *``.
    H0 = torch.zeros(
        (batch_size * HDIM * HDIM), dtype=dR_dxyz.dtype, device=dR_dxyz.device
    )
    dH0 = torch.zeros(3, batch_size * HDIM * HDIM, dtype=H0.dtype, device=H0.device)
    # -1 marks a padding slot rather than a real pair, so this selects the real
    # ones.  It is the same idea ``valid_pairs`` encodes, derived here from the
    # pair-type array instead of the neighbour arrays.
    nn_mask_IJ = IJ_pair_type != -1

    # ---- s with s ----------------------------------------------------------
    # Same as block 1 of the single-structure routine: every atom has an s
    # orbital, so no shell mask is needed and there is no angular factor.  The
    # spline is evaluated inline here rather than through a helper, because
    # this function has no neural-network alternative to switch between.
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

    # ---- Working out where each pair's value belongs ----------------------
    # neighbor_I, neighbor_J: (B, Npairs) with -1 padding
    valid_I = neighbor_I >= 0
    valid_J = neighbor_J >= 0
    valid_pair_mask = valid_I & valid_J  # (B, Npairs)

    # Safe gather: replace -1 by 0 then zero out later
    # A -1 would either crash the lookup or wrap around to the last atom, so
    # padding slots are temporarily pointed at atom 0.  Whatever they pick up
    # is discarded two steps later, when the mask filters them out.
    safe_neighbor_I = neighbor_I.clone()
    safe_neighbor_J = neighbor_J.clone()
    safe_neighbor_I[~valid_I] = 0
    safe_neighbor_J[~valid_J] = 0

    # Map atom indices to their first AO
    # ``gather`` along axis 1 does the per-structure lookup: for each structure
    # it reads that structure's own row of H_INDEX_START.  A plain index would
    # apply one structure's atom numbering to all of them.
    rows_all = H_INDEX_START.gather(1, safe_neighbor_I)  # (B,Npairs)
    cols_all = H_INDEX_START.gather(1, safe_neighbor_J)  # (B,Npairs)

    # Mark invalid
    # Re-stamp the padding slots so a stray read of these arrays is obviously
    # wrong rather than plausibly pointing at atom 0.
    rows_all[~valid_I] = -1
    cols_all[~valid_J] = -1

    # Keep only valid pairs for flattened indexing
    # This is the padded -> compacted transition described in the header note:
    # boolean-indexing a (B, Npairs) array yields a flat list of just the real
    # entries, in the same order the compacted L / M / N / dx arrays use.
    rows = rows_all[valid_pair_mask]  # (Nvalid,)
    cols = cols_all[valid_pair_mask]  # (Nvalid,)

    # Build batch index for each kept pair
    # Compute batch ids from original mask
    # ``arange(B)`` down a column, broadcast across every pair slot, then
    # compacted the same way -- giving, for each surviving pair, the index of
    # the structure it came from.
    batch_ids = torch.arange(B, device=rows_all.device).unsqueeze(1).expand_as(rows_all)
    batch_ids = batch_ids[valid_pair_mask]  # (Nvalid,)

    # Base offset per batch block
    # Structure k's matrix starts at k * HDIM * HDIM in the flat buffer.
    batch_block_offset = batch_ids * (HDIM * HDIM)

    # Position within one structure's matrix, then shifted into that
    # structure's slice of the shared buffer.
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

    # ---- s on atom I with p on atom J -------------------------------------
    # Every remaining block in this function follows one fixed recipe.  It is
    # spelled out once here and not repeated:
    #
    #   tmp_mask                 which pair classes take part (padded space)
    #   ...gather(1, safe_I)     per-structure atom -> first-orbital lookup,
    #                            then [tmp_mask] to compact it
    #   idx[tmp_mask[valid_pairs]]  the same selection applied to a compacted
    #                            array -- note the extra [valid_pairs]
    #   coeffs_tensor[...]       the spline coefficients for this channel
    #   the four-term cubic      the radial value; the three-term one below it
    #                            is its derivative
    #   batch_ids / batch_block_offset  recomputed per block, since the set of
    #                            selected pairs changes each time
    #   index_add_(..., pos + batch_block_offset, ...)  the write
    #
    # The angular expressions in between are identical to the single-structure
    # routine's and are explained there.
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

    # ---- p on atom I with s on atom J -------------------------------------
    # The mirror block: JI_pair_type for the reversed lookup, p offsets moved
    # onto the row, and the leading minus signs that come from the bond running
    # the other way (see the single-structure routine's block 3).
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

    # ---- p with p ----------------------------------------------------------
    # Two radial functions, sigma ("pp0") and pi ("pp1"), combined by the
    # along-bond / across-bond decomposition explained at block 4 of the
    # single-structure routine.
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

    # ---- s on atom I with d on atom J -------------------------------------
    # Sigma only ("sd0"); columns +4..+8 are dxy, dyz, dzx, dx2-y2, dz2.
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

    # ---- p on atom I with d on atom J -------------------------------------
    # Fifteen entries from two radial functions ("pd0" sigma, "pd1" pi).
    # Variable names read H_<p orbital>_<d orbital>.
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

    # ---- d on atom I with s on atom J -------------------------------------
    # No sign flip: a d orbital's angular polynomial is even, so reversing the
    # bond leaves it unchanged (the parity rule stated at block 7 above).
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

    # ---- d on atom I with p on atom J -------------------------------------
    # Each expression is the matching p-d formula wrapped in a leading minus:
    # d plus p is 2 + 1 = 3, odd, so the bond reversal flips the sign.
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

    # ---- d with d ----------------------------------------------------------
    # The last and largest block: twenty-five entries from three radial
    # functions ("dd0" sigma, "dd1" pi, "dd2" delta).  Only YY appears in the
    # mask because this routine has no f atoms, so a d shell on both sides can
    # only mean two 9-orbital atoms.
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
    # The d-d derivatives are long enough that the repeated sub-expressions are
    # named once up front and reused, both for legibility and to avoid
    # recomputing the same products twenty-five times over.
    #
    # The naming scheme, decoded: "t" is short for "times", so ``L_t_M`` means
    # L * M and ``L_t_Ldx`` means L * dL/dxyz.  ("m" for minus and "p" for plus
    # are mentioned in the original note but no longer appear.)  ``L2``, ``M2``
    # and ``N2`` are just the squares of the direction cosines.
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

    # A second copy of the same non-executing reference table as in the
    # single-structure routine: every s/p/d angular formula in one place, with
    # $ marking entries printed in the standard Slater-Koster tables and O
    # marking ones obtained by relabelling the axes of a printed entry.
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
    # Always exactly two values -- no stress variant exists here.
    return H0, dH0


# =============================================================================
# THE LEGACY ASSEMBLER
# =============================================================================
# An older, much smaller routine, kept for compatibility.  Three things set it
# apart from the two functions above, and all three limit it:
#
#   1. s and p only.  There is no d code and no f code, and the only masks it
#      accepts are HH, HX, XH and XX -- pairs of 1- and 4-orbital atoms.
#
#   2. The radial functions come from analytic formulas rather than tabulated
#      splines.  The caller passes the bond-integral parameters directly
#      (fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi) and
#      ``bond_integral_vectorized`` turns each into a value at the given
#      distance.  Note this means s-p and p-s get genuinely separate
#      parameters here, rather than one parameter looked up in two directions.
#
#   3. It *assigns* into H0 with ``=`` instead of accumulating with
#      ``index_add_``, and it writes into an H0 the caller supplies rather than
#      allocating its own.  Assignment works here only because, with just four
#      pair classes and no shared shells, no two writes ever land on the same
#      matrix entry.  That property does not survive the addition of d or f
#      shells, which is why the modern routines accumulate instead.
#
# The angular expressions themselves are the same s-s, s-p, p-s and p-p ones
# explained at blocks 1 to 4 of the single-structure routine.
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

    # Imported inside the function rather than at module scope to avoid a
    # circular import: _bond_integral imports the channel list from this file.
    # ``bond_integral_vectorized`` returns the radial value at each distance;
    # ``bond_integral_with_grad_vectorized`` returns its derivative with
    # respect to distance.
    from ._bond_integral import (
        bond_integral_vectorized,
        bond_integral_with_grad_vectorized,
    )

    # H0 is supplied by the caller and written into; only dH0 is allocated here.
    dH0 = torch.zeros(3, HDIM * HDIM, dtype=H0.dtype, device=H0.device)

    # ---- s with s ----------------------------------------------------------
    # No mask and no angular factor, for the same reasons as block 1 above.
    #######
    HSSS_all = bond_integral_vectorized(dR, fss_sigma)
    H0[H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J]] = HSSS_all

    #######

    # H-H
    ######### dH/dx
    # Chain rule only: d(value)/dR times dR/d(x, y, z).
    HSSS_dR = bond_integral_with_grad_vectorized(dR, fss_sigma)
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0[:, H_INDEX_START[neighbor_I] * HDIM + H_INDEX_START[neighbor_J]] = HSSS_dxyz
    #########

    # ---- s on atom I with p on atom J -------------------------------------
    # The angular factors are L, M and N for px, py, pz -- block 2 above.
    #
    # Note ``+`` where the modern routines use ``|``.  On boolean tensors
    # PyTorch defines addition as logical OR, so these mean the same thing;
    # ``|`` is simply the clearer spelling and is what the newer code uses.
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

    # ---- p on atom I with s on atom J -------------------------------------
    # The mirror block, with the leading minus signs from the bond running the
    # other way.  Unlike the modern routines, the reversed direction has its
    # own parameter set (``fps_sigma`` rather than ``fsp_sigma``) instead of
    # being a reversed lookup of the same one.
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

    # ---- p with p ----------------------------------------------------------
    # Only XX qualifies: both atoms need a p shell, and X is the only
    # p-bearing class this routine knows about.  ``PPSMPP`` is again
    # sigma minus pi, combined by the along-bond / across-bond decomposition
    # explained at block 4 above.
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

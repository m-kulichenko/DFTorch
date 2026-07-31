# Why the mixed radial grid case is synthesised, not checked in

**Written:** 2026-07-31, during plan 05-01 (decision D-01, requirement REG-06).
**Companion code:** `write_mixed_grid_skf_pair` in `tests/test_radial_grid.py`.

This file is ASCII only. Phase 4 recorded that em dashes render as replacement
characters on a cp1252 console and destroy the diagnosability of pytest output.

## The defect this documents

`_bond_integral.get_skf_tensors` stores every element pair's own radial grid in
`R_tensor`, but exports a single global `R_orb` chosen as the longest grid it
saw (`_bond_integral.py:1064-1065`). `_h0ands.H0_and_S_vectorized` derives the
spline knot from that one array for every pair type (`_h0ands.py:243-245`), so a
pair whose real grid differs is interpolated against another pair's ruler.

## Every f fixture in this directory shares one grid

All nine `.skf` files here carry the same grid line, `0.04, 433`:

    Eu-Eu.skf  Eu-Ga.skf  Eu-N.skf
    Ga-Eu.skf  Ga-Ga.skf  Ga-N.skf
    N-Eu.skf   N-Ga.skf   N-N.skf

433 tabulated points are padded to 483 at `_bond_integral.py:599`, and the step
is converted from Bohr to Angstrom at `:713-718`, giving a step of
`0.0211670884` A and a last knot at `10.2237036972` A. Every row of `R_tensor`
for an Eu/Ga/N system is therefore elementwise identical.
`test_f_fixture_grids_are_all_identical` in `tests/test_radial_grid.py` asserts
exactly this.

**Correction to a prior write-up.** Section 3 of
`.planning/phases/03-h0-s-routing-and-f-angular-blocks/deferred-items.md` states
that the grids in this directory "genuinely differ". That claim is factually
wrong and was re-verified as wrong on 2026-07-29 and again on 2026-07-31. Read
that document for the mechanism of the defect, never for its fixture claim.

## Real parameter sets do differ, in two different ways

Two distinct things can differ between SKF files, and they are not equally
dangerous.

**Differing grid LENGTH at the same step is benign** (decision D-01). The knots
coincide, so `idx` and `dx` are already correct for both pairs and only the
zero-padded tail differs. Rejecting this case would break working parameter
sets, so the REG-05 guard must not reject it.

**Differing grid STEP is the hazard.** The knots do not coincide, so a pair
evaluated against another pair's grid reads its spline at the wrong radius.
The f dataset in this directory uses a step of `0.04` Bohr, while `mio-1-1`,
`3ob-3-1`, `pbc-0-3` and `trans3d-0-1` all use `0.02` Bohr. Any single `SKFPATH`
mixing them, for example an Eu complex with C/H/O ligands, is currently wrong by
a factor of two in the knot index.

## What is and is not already in this repository

**Mixed LENGTH is already present, in `tests/data_skf_mio-1-1`.** Measured
2026-07-31 by reading the grid line of every file in that directory:

| Files | Grid line | Loaded grid length |
| --- | --- | --- |
| most pairs (C-C, C-H, H-H, N-N, O-O, S-S, ...) | `0.02, 500` | 550 |
| every `Zn-*` and `*-Zn` pair | `0.02 600` | 650 |
| every `P-*` and `*-P` pair | `0.02, 619` | 669 |

A C/H/P system loaded from `mio-1-1` therefore produces `R_tensor` rows of two
different real lengths (550 and 669) at one common step. This is the benign
case, and it is real rather than synthetic, so `tests/test_radial_grid.py` uses
it directly as a per-pair-row check.

**Mixed STEP is NOT present anywhere in this repository.** No checked-in
directory contains two files with different first grid fields. The hazardous
case therefore cannot be reproduced from real fixtures at all.

## Why a generator instead of a checked-in fixture

`write_mixed_grid_skf_pair(path, elem_a, elem_b, *, step_bohr, npts)` in
`tests/test_radial_grid.py` writes a minimal simple-format SKF file with a
caller-chosen grid line. Tests call it into a `tmp_path` directory and build an
`SKFPATH` on the fly.

Reasons for generating rather than committing binary-ish fixture files:

1. The hazard is a property of the grid line, which is two numbers. A generator
   expresses that directly; a committed file buries it in a few hundred rows of
   otherwise irrelevant tabulated data.
2. A committed fixture invites reuse by later tests that then depend on its
   incidental contents. A generator forces each caller to state the `step_bohr`
   and `npts` it actually cares about.
3. Plan 05-04 needs the same case for the REG-05 guard. One generator serves
   both without a second fixture directory.

## The honest limitation

A test built on a synthetic fixture validates **the implementation's model of
the defect**, not an observed failure. No user has reported a wrong number from
a mixed-step load, and none of the three parameter sets this project ships can
produce one. If the model is wrong about how a mixed-step directory behaves in
the wild, a green synthetic test will not say so.

What the synthetic tests do establish is narrower and still worth having: that
the per-pair lookup indexes each pair against its own row, that a distance past
a short pair's tabulated end lands on that pair's own zero interval rather than
on a coefficient borrowed from a longer grid, and that neither behaviour changed
for the single-grid parameter sets that do exist. The bit-identity assertions
against `tests/data_skf_mio-1-1` and `tests/f_orbital_data` are the part of this
work that rests on real data.

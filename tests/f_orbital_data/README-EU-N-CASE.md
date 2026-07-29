# The Eu-N diatomic validation case

This directory holds the f-orbital SKF fixture set. This document is the durable
description of the validation case those fixtures exist to support: an **isolated Eu-N
diatomic**, scanned over interatomic separation, whose single-shot energy minimum is
checked against a loose sanity band.

It lives here, beside the fixtures, rather than under `.planning/`, so that it survives
milestone archival and is found by whoever next opens this directory.

The automated gate is `tests/test_eu_n_scan.py`. The energy path it drives is
`ESDriver.forward(do_scf=False)`.

**This is a common-sense physical check, not a reference-grade reproduction.** There is no
reference paper for this case and none is claimed. Read the *Tolerance* and *Provenance and
caveats* sections before drawing any conclusion from a pass or a failure.

---

## Geometry

An isolated (non-periodic in any meaningful sense at the cutoffs used) Eu-N diatomic:

| Property | Value |
|---|---|
| Atoms | 2 — `Eu` and `N` |
| Eu position | `(0, 0, 0)` |
| N position | `(r, 0, 0)` — displaced along +x by the scan separation `r` |
| Atomic orbitals | 20 total — Eu contributes 16 (s, p, d, f), N contributes 4 (s, p only) |
| Occupied levels | 7 (`Nocc == 7`) |
| Charge | 0 (neutral) |
| Spin | closed-shell / restricted. Spin polarization is refused outright for f systems |

The XYZ file is written by the test into a pytest temporary directory, one file per scan
point:

```
2
Eu-N diatomic
Eu 0.00000000 0.00000000 0.00000000
N  <r>        0.00000000 0.00000000
```

N has **s and p only**, not s/p/d, which is why the Hamiltonian dimension is 20 and not 25.
A 25 anywhere in a diagnostic means the N shell metadata drifted.

Because the direction cosines of the ordered pair (Eu, N) are `(L, M, N) = (1, 0, 0)` along
+x, the geometry is also the simplest possible probe of the f angular blocks: only one
direction is exercised. This is a deliberate limitation, recorded again under *What a green
test does not prove*.

## SKF parameter files

The case reads exactly **four** of the nine SKF files in this directory:

| File | Format | Role |
|---|---|---|
| `Eu-Eu.skf` | **extended** (f-electron; leading `@ Data set with f-electrons, for DFTB+ only!` line) | Eu onsite energies, Hubbard U values, reference occupations, Eu-Eu integrals and repulsion |
| `N-N.skf` | simple 20-column | N onsite energies, Hubbard U values, reference occupations, N-N integrals and repulsion |
| `Eu-N.skf` | heteronuclear | Eu-N Slater-Koster integral tables and repulsive spline |
| `N-Eu.skf` | heteronuclear | N-Eu Slater-Koster integral tables and repulsive spline |

The following **five** files are deliberately unexercised by this case and by the phase that
created it:

`Eu-Ga.skf`, `Ga-Eu.skf`, `Ga-Ga.skf`, `Ga-N.skf`, `N-Ga.skf`

They are present because the fixture set was generated for all Eu/Ga/N pairs, not because a
Ga case exists. **Do not invent Ga work to justify them.** Periodic / bulk EuN is likewise
out of scope — it would additionally pull in k-points and Ewald/PME.

### Specific parameter values the case depends on

From `Eu-Eu.skf` line 3 (extended format: `Ef Ed Ep Es  SPE  Uf Ud Up Us  ff fd fp fs`):

```
-0.0559 -0.0247 0.0197 -0.0968    0.0   0.50 0.25 0.19 0.21    7.0 0.0 0.0 2.0
```

| Quantity | Value (Hartree) | Value (eV) |
|---|---|---|
| Eu f onsite energy `Ef` | -0.0559 | -1.521 |
| Eu **f Hubbard U** `Uf` | **0.50** | **13.605693** |
| Eu d Hubbard U `Ud` | 0.25 | 6.802847 |
| Eu p Hubbard U `Up` | 0.19 | 5.170163 |
| Eu s Hubbard U `Us` | 0.21 | 5.716791 |

Eu reference occupations: **7 f electrons**, 0 d, 0 p, **2 s electrons**.

From `N-N.skf` line 2 (simple format: `Ed Ep Es  SPE  Ud Up Us  fd fp fs`):

```
0.0 -0.2512 -0.7924    0.00    0.490 0.490 0.490        0.0 3.0 2.0
```

N reference occupations: 0 d, **3 p electrons**, **2 s electrons**. N's Hubbard U is 0.490
Hartree (13.33 eV) on every shell present.

The Eu f Hubbard U of 0.50 Hartree is the value that made the f dimension of the
shell-resolved Hubbard/charge plumbing testable at all — it is a real number carried by the
extended-format header, not a zero-fill. Both radial grids are `0.04 Å` step over 433 points.

## Observable

**Total energy from a single non-self-consistent diagonalization, as a function of
interatomic separation.**

Concretely: build H0 and S, diagonalize once at the reference charge state, and report

```
e_tot = e_elec_tot + e_repulsion
```

which is band energy (including the finite-temperature entropy term at
`T_ELECTRONIC = 1000 K`) plus nuclear repulsion.

**There is no charge-fluctuation Coulomb term. `e_coul` is exactly 0, and that is a
definition rather than an omission.** The reason is load-bearing: with no
self-consistency, the only charges available are first-iterate Mulliken charges, and those
are not neutral. They come out around `q_Eu = -2.70 e` at the target separation and drift to
`-2.99 e` at 4 Å — i.e. the atoms are not even neutral at separation, which is unphysical.
Feeding them into the electrostatic energy does not produce a better single-shot energy; it
produces one broken SCF step, and it was measured to destroy the binding well entirely (the
energy rises monotonically past 2.0 Å, leaving no interior minimum at all). So `energy()` is
called with `C=None` and `dq_p1=None`, selecting its `Ecoul = 0` arm.

If you are tempted to "improve" this curve by switching the Coulomb term on, read the
previous paragraph again. The correct fix is real charge self-consistency for f systems,
which is a separate, larger piece of work.

Solvation (GBSA/ALPB), D3 dispersion and spin are pinned to `0.0` on this path for the same
reason — all three are charge- or spin-dependent, and there is no converged charge state and
no spin treatment here.

**The minimum is located by scanning, never by geometry optimization.** f derivatives are
unimplemented: `ESDriver.calc_forces`, `ESDriverBatch.calc_forces` and analytical stress all
raise `FDerivativeUnsupportedError` for any system containing a 16-orbital atom. An
optimizer-based approach therefore hits a hard wall immediately. The test must not call
`calc_forces`, `calc_stress`, or any optimizer routine. An energy scan needs only energy, so
it is compatible with the single-shot energy path.

Driver parameters, which the test pins and which the reference curve below was recorded
against:

| Parameter | Value |
|---|---|
| `SKFPATH` | this directory (`tests/f_orbital_data/`) |
| `T_ELECTRONIC` | 1000.0 |
| `RCUT_ELECTRONIC` | 10.0 |
| `RCUT_REPULSIVE` | 6.0 |
| `COUL_METHOD` | `"FULL"` |
| `CHARGE` | 0 |
| `do_scf` | `False` |

`do_scf=False` leaves some SCF bookkeeping attributes unset (`structure.H`, `Hcoul`,
`Hdipole`, `KK`, `Q`, `e_coul_tmp`, `f_coul`, `dq_p1`, `stress_coulomb`) because they have no
single-shot analogue. Read `structure.e_tot`; touching the others raises `AttributeError`.

## Units

| Quantity | Unit |
|---|---|
| Interatomic separation | Ångström (Å) |
| Energy | electronvolt (eV) |
| SKF header onsite energies and Hubbard U | Hartree, as stored in the file |
| All tensors | `float64` — tests run under `torch.set_default_dtype(torch.float64)` |

Separations in the scan grid are rounded to two decimals so the grid is exact in printed
diagnostics and reproducible across platforms.

## Tolerance

| Quantity | Value |
|---|---|
| Target separation | **2.655 Å** |
| Band | **±20 %** |
| Band minimum | 2.655 × 0.80 = **2.124 Å** |
| Band maximum | 2.655 × 1.20 = **3.186 Å** |
| Scan range | 1.60 Å to 3.60 Å inclusive, step 0.10 Å (21 points) |

The gate is: the scanned curve has an energy minimum **strictly interior** to the scanned
range (not at either endpoint), and that minimum's separation lies within
[2.124 Å, 3.186 Å].

**The band is always computed as a fraction of the target, never as an absolute
half-width.** In code:

```python
EU_N_TARGET_ANGSTROM = 2.655
EU_N_BAND_FRACTION   = 0.20
EU_N_BAND_MIN_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 - EU_N_BAND_FRACTION)
EU_N_BAND_MAX_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 + EU_N_BAND_FRACTION)
```

This is not stylistic. An earlier draft of the research prose mis-stated the band as
"2.4 ± 0.2 Å = [1.92, 2.88]". That is a ±8 % band, not ±20 %, and substituting it would have
silently tightened the gate far past what the method can support — the test would then fail
for reasons that have nothing to do with whether the f implementation is correct.
`tests/test_eu_n_scan.py::test_tolerance_band_is_a_fraction_of_the_target` exists solely to
make that substitution fail the suite instead of passing quietly.

The band is deliberately loose (roughly 10-20 % is the agreed range) and **must not be
tightened**. Single-shot energy without self-consistency, plus a closed-shell treatment of
an open-shell 4f⁷ ion, cannot support a few-percent claim.

The scan range **brackets both band edges** — 1.60 Å is below 2.124 Å and 3.60 Å is above
3.186 Å, and grid points exist strictly on both sides of each edge (2.10/2.20 straddle 2.124;
3.10/3.20 straddle 3.186). This matters: if the range stopped at the band edge, a minimum
that genuinely wants to sit outside the band would be clipped to the edge and the gate would
report a false pass.

## Provenance and caveats

The 2.655 Å target is the **mean of 16 Eu-N bond lengths** taken from a user-supplied table
of Eu-Bp coordination complexes spanning four ligand variants (Bp, Bp^Me, Bp^Me2, Bp^CF3).
The 16 entries range from 2.606 Å to 2.716 Å — a spread of about 4 %. Eu-O rows in the same
table were excluded.

This target **replaces** an earlier unsourced "~2.4-2.5 Å" model recollection that appeared
in research prose. That recollection is superseded and must not be used.

**Known caveat, explicitly accepted.** Those 16 bonds are *dative* Eu-N bonds inside a
crowded 8-9-coordinate sphere. The test case is an *isolated diatomic*. These are not the
same physical situation, and a gas-phase Eu-N diatomic would plausibly be **shorter** than a
coordination-sphere mean. Two things make the comparison acceptable anyway: the ±20 % band
absorbs the difference, and a second, independent reference point agrees — bulk rock-salt EuN
sits at roughly 2.45-2.5 Å, also inside the band.

There is **no reference paper** for this case. Reference-grade reproduction is not claimed
and must not be claimed on the strength of a passing gate.

### Asymmetric failure rule

**A failure of this gate does not mean the same thing in both directions. Read this before
concluding anything about the f implementation.**

| Observed outcome | Interpretation | Action |
|---|---|---|
| **No interior minimum at all** — the minimum sits at the first or last grid point | **Genuine red flag.** The band assertion is meaningless without a well. The curve is drifting monotonically. | Investigate. Inspect the curve shape for monotonic drift. |
| **Minimum SHORT of the band** (below 2.124 Å) | **Plausibly correct isolated-diatomic physics. This is NOT evidence that the f implementation is broken.** The 2.655 Å target is a mean over dative bonds in a crowded coordination sphere; an isolated diatomic is expected to be shorter. | **Escalate to human judgement.** Compare against bulk rock-salt EuN (~2.45-2.5 Å) and against the curve shape, then decide whether to widen the band or to investigate. Do not "fix" the f code on this evidence alone. |
| **Minimum LONG of the band** (above 3.186 Å) | **Genuine red flag.** Over-long bonds are not explained by the diatomic-vs-coordination difference — that argument only runs short. | Investigate. |

The rule is encoded in three places so it cannot be lost: this document, the module docstring
of `tests/test_eu_n_scan.py`, and the three-branch failure message inside
`test_eu_n_energy_scan_locates_interior_minimum`. A future reader who sees a short minimum
must not read it as "the f code is wrong".

## What a green test does not prove

A passing gate is a smoke test. It specifically does **not** establish any of the following.

**Manual-only checks — no automated assertion covers these:**

| Behaviour | Why it cannot be automated | What to do by hand |
|---|---|---|
| Physical plausibility of the located minimum | The target is a mean over Eu-N *coordination* bonds while the case is an isolated *diatomic*. A green test confirms the ballpark, not the physics. | Sanity-check the curve shape: single well, energy rising on both sides, no monotonic drift. If there is no interior minimum, the band assertion is meaningless regardless of pass/fail. |
| Short-side band failure | A minimum below ~2.12 Å is plausibly correct diatomic physics, not an f bug. Only a human can tell those apart. | Follow the asymmetric failure rule above. |

**Sampling risk — what a passing suite still would not catch:**

1. **The reference separation is a coordination-chemistry mean, not a diatomic measurement.**
   A green gate means "the minimum is near a related but non-identical reference", not "the
   physics is right". This asymmetry is exactly why the failure rule above is asymmetric.
2. **A loose band absorbs real errors.** With a ±20 % band, an f-block bug that shifts the
   minimum by 15 % still passes. The band is a smoke test, not a correctness proof.
3. **Single-shot is not SCF.** One diagonalization ships here. Charge-self-consistency bugs
   cannot surface through this case by construction.
4. **Closed-shell treatment of an open-shell ion.** Eu 4f⁷ is genuinely open-shell; spin is
   deferred and refused. Any observable sensitive to spin polarization is untested here.
5. **The shell-resolved f plumbing is validated but unconsumed.** Shell-resolved Hubbard U
   shapes and values have their own tests, but the single-shot energy path never reads them —
   so the integration between the two is unsampled until self-consistency lands.
6. **Only the Eu-N pair is exercised, along one direction.** Seven of the nine SKF fixtures
   go untouched; Ga paths and periodic cases are out of scope and therefore unsampled. The
   +x geometry exercises a single set of direction cosines.

A further open concern, unrelated to the band: `_bond_integral` exports a single `R_orb` (the
longest grid) for all pair types, while this fixture directory mixes radial grids. The
single-shot energy reproduces exactly, which says the path is self-consistent, but says
nothing about whether the shared radial grid is correct across mixed-grid pairs.

## How to run

```bash
uv run pytest tests/test_eu_n_scan.py -q
```

The 21-point scan takes roughly **16 seconds** of wall time, including interpreter start and
imports. The whole curve is computed **once** per module by a module-scoped pytest fixture;
recomputing it per test would multiply that by the number of tests and blow the 30-second
feedback budget.

`uv run pytest` is the canonical invocation (it is what CI runs). Bare `pytest` is not.

## Reference curve recorded 2026-07-29

Recorded on 2026-07-29 against the then-current implementation, with the parameters tabulated
under *Observable*. Diff a changed curve against this table to see what moved.

| r (Å) | E_tot (eV) | | r (Å) | E_tot (eV) |
|---|---|---|---|---|
| 1.60 | -3.0061030 | | 2.70 | -17.4752833 |
| 1.70 | -9.6876290 | | 2.80 | -17.4008853 |
| 1.80 | -13.4703897 | | 2.90 | -17.3345468 |
| 1.90 | -15.5570923 | | 3.00 | -17.2476120 |
| 2.00 | -16.6777628 | | 3.10 | -17.1502735 |
| 2.10 | -17.2570144 | | 3.20 | -17.0625603 |
| 2.20 | -17.5183387 | | 3.30 | -16.9850053 |
| 2.30 | -17.6207142 | | 3.40 | -16.9179804 |
| **2.40** | **-17.6414416** | | 3.50 | -16.8614428 |
| 2.50 | -17.6127839 | | 3.60 | -16.8148915 |
| 2.60 | -17.5517973 | | | |

**Located minimum: 2.40 Å**, grid index 8 of 20 — strictly interior, and comfortably inside
[2.124, 3.186]. Exactly one grid point attains the minimum energy, so there is no tie to
break. The curve is a single clean well: strictly decreasing up to 2.40 Å and strictly
increasing after it.

For reference, the single point at the target separation itself (2.655 Å, not on the scan
grid) gives `e_tot = -17.510444238744924` eV with `e_band0 = -16.948081509141794` eV,
`e_coul = 0` and `e_repulsion = 0.15890937893715806` eV. That value is pinned independently
by `tests/test_single_shot_energy.py::test_eu_n_single_shot_reference_energy`.

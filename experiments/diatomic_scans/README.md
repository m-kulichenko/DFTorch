# Diatomic binding-curve scans

Reproduces every figure in `figures/`. Two-step by design: the physics runs in the project
venv, the plotting runs in any interpreter that has matplotlib, and the two communicate
through JSON. The project venv has no matplotlib and this avoids adding one.

## Standing decision: the Krylov mixer is off

Every self-consistent run here passes

```python
"KRYLOV_START": 10**6,   # never reaches the low-rank Krylov accelerator
```

The low-rank Krylov charge mixer (`kernel_update_lr`, engaged at `it > KRYLOV_START`,
default 10, `_scf.py:690`) was measured to diverge on 9 of 21 Eu-N separations while
reproducing the identical total energy wherever it did converge. Anderson/DIIS mixing alone
converges every case tested. Repairing the accelerator is deferred to a later phase; turning
it off is the interim position.

**Why it diverges is now known:** see `.planning/KRYLOV-INVESTIGATION.md`. Reproduce every
measurement in it with `krylov_probe.py` in this directory — that script is the one exception
to the rule above, since it deliberately turns the mixer *on* in order to measure it.

## Running

```bash
# physics -> JSON (project venv)
.venv/Scripts/python.exe experiments/diatomic_scans/mio_suite.py     # mio-1-1 suite, ~3 min
.venv/Scripts/python.exe experiments/diatomic_scans/scan_scf.py      # Eu-N self-consistent
.venv/Scripts/python.exe experiments/diatomic_scans/dump_0shot.py    # Eu-N non-SCC
.venv/Scripts/python.exe experiments/diatomic_scans/dissoc.py        # Eu-N out to 40 A
.venv/Scripts/python.exe experiments/diatomic_scans/diag_mech.py     # Hubbard-U mechanism test

# JSON -> figures (any interpreter with matplotlib)
python experiments/diatomic_scans/plot_mio.py
python experiments/diatomic_scans/plot_eun_clean.py
```

Scripts resolve paths relative to their own location, so the JSON files land beside them.

## What each file produces

| script | output |
|---|---|
| `mio_suite.py` | `mio_suite.json` - 16 diatomics x 61 separations x {H0, SCC} |
| `scan_scf.py` | `eu_n_scf_nokrylov.json` (edit `NOKRY` to reproduce the diverging run) |
| `dump_0shot.py` | `eu_n_0shot.json` |
| `dissoc.py` | `eu_n_dissoc.json` - both paths out to 40 A |
| `diag_mech.py` | prints the level-spectrum and Hubbard-U sensitivity tables |
| `krylov_probe.py` | prints the 11 measurement tables in `.planning/KRYLOV-INVESTIGATION.md` |
| `plot_mio.py` | `figures/mio_*.png` + `mio_summary_table.txt` |
| `plot_eun_clean.py` | `figures/eu_n_*.png` |

## Scope of the mio suite

Sixteen systems, all with an **even** valence-electron count. `Structure.py:355` raises
"Closed shell systems require even number of electrons", so the common odd-electron radicals
(OH, CH, CN, NO, SH) cannot run on the closed-shell path at all. The closed-shell anions
`OH-`, `CH-`, `CN-`, `SH-` stand in for those bonds.

Grid: 61 points per system, so the separation resolution is roughly 0.03-0.04 A depending on
the range. Several heteronuclear systems show an identical H0 and SCC minimum simply because
the shift is below that resolution; refine `N_POINTS` before reading anything into it.

`exp_re` in the `SYSTEMS` table is a **reference marker for the eye only**, quoted from
standard diatomic spectroscopic tables. Nothing asserts against it and it is not a tolerance.
Verify any value before quoting it.

## Result worth knowing

For all six homonuclear systems the self-consistent and non-SCC curves agree to
`max|E_SCC - E_H0| ~ 1e-13 eV` with `max|q| ~ 1e-9 e` - the correct answer, since symmetry
forbids charge transfer and the SCC correction must therefore vanish identically. That makes
the homonuclear cases a free, sharp regression test on the whole SCC path.

# REG-02: Simple-Format Regression Baseline

**Requirement.** REG-02 reads: "Existing simple-format calculations in
`experiments/1_tutorial.ipynb` remain scientifically unchanged within documented
tolerances."

**Status: satisfied in substance, not in form.** Read the "What this does NOT cover"
section before treating REG-02 as green.

- **Decided:** 2026-07-30, at the Task 1 `checkpoint:decision` of plan 05-02.
- **Measured at:** commit `db63487`, before any `05-0*` commit landed.
- **Deliverable:** `tests/test_simple_format_regression.py`

---

## The decision

**Selected: `option-a` -- Extract the notebook's runnable simple-format calculations
into a pytest module.**

The developer was shown all three options with their costs, including the three
verified obstacles below and the fact that the notebook's 23 stored cell outputs came
from unknown hardware and are not a usable baseline. The reason recorded for the
selection:

> Option A adds no dependencies, runs in CI alongside the existing 72 tests, and
> protects every future phase rather than only this one. Options B and C were
> rejected: B required inventing a `COORD.pdb` geometry that does not exist and
> installing new runtime dependencies in a phase whose whole purpose is regression
> safety; C would have left Phase 5 with no numeric baseline at all while roughly
> twenty files are edited, which is the specific risk this phase exists to eliminate.

This choice was made by the developer at a blocking checkpoint. It was **not** inferred
or rescoped by an agent, and REQUIREMENTS.md was not reworded to match what was
convenient to build.

---

## Why the notebook itself is not executed

Three independent obstacles, each verified by measurement on 2026-07-30 rather than
assumed. Any one of them alone would block an end-to-end run.

### 1. The first computational cell reads an input file that was never committed

Cell 5 sets:

```python
"FILENAME": "COORD.pdb",
```

`experiments/COORD.pdb` **does not exist in this repository.** The directory contains
`COORD.xyz`, `COORD_8WATER.xyz` and `COORD_ACETONE.xyz`, but no `COORD.pdb`.

This is a real repository defect, not merely a testing inconvenience: the tutorial is
broken for every new user who tries to run it from a clean checkout, not just for this
regression harness. It is recorded here so that whoever repairs the tutorial has the
finding. Repairing it was explicitly out of scope for this decision, because supplying
the file means inventing a geometry, which changes what the tutorial demonstrates.

### 2. One cell hardcodes a CUDA device with no fallback

Cell 14 sets:

```python
device = "cuda"
```

with no availability guard. This machine has no CUDA driver (`Warp CUDA warning: Could
not find or load the NVIDIA CUDA driver`). Cells 17 and 25 do guard correctly, with
`device = "cuda" if torch.cuda.is_available() else "cpu"`, so this is a defect in one
cell rather than a general property of the notebook.

### 3. No notebook execution tooling is installed

`nbformat`, `nbclient`, `papermill`, `nbval`, `ipykernel` and `matplotlib` are **all
absent** from the project environment (verified by import probe). Installing any of them
is a dependency decision requiring its own audit, not a planning detail. Under
`option-a` no package was installed; `uv pip list` is unchanged from before this work.

### And the stored outputs are not a baseline either

The notebook carries stored outputs in 23 of its 38 cells. Those numbers were produced
on unknown hardware running an unknown revision of this code. Adopting them as reference
values would be pinning against an unverifiable provenance. Every reference number in
`tests/test_simple_format_regression.py` was instead recomputed locally on commit
`db63487`.

---

## Notebook cell to test case mapping

| Notebook cell | Test case | What it exercises |
|---|---|---|
| Cell 17 | `o2_mio_unrestricted` | O2 with `mio-1-1`, unrestricted, `SPIN_POL` 2, `T_ELECTRONIC` 500 K |
| Cell 25 | `o2_3ob_dftb3` | O2 with `3ob-3-1`, unrestricted, `SPIN_POL` 2, `DFTB3` enabled |
| Cell 14 | `water8_mio_full` | 8-water box, `mio-1-1`, `CELL` 21 A cubic, `COUL_METHOD` FULL |
| Cell 5 | *(none)* | **Not covered** -- input file missing, see below |

Each case pins the total energy, the full force array, and the per-atom Mulliken
charges. That is strictly stronger than what `tests/test_scf.py` asserts today, which is
finiteness and shape only and would therefore not notice a numeric drift.

## Pinned values and tolerances

Measured on commit `db63487`, CPU, `torch.set_default_dtype(torch.float64)`.

| Case | Total energy (eV) | Atoms |
|---|---|---|
| `o2_mio_unrestricted` | `-9.031346598527405` | 2 |
| `o2_3ob_dftb3` | `-8.270081470047817` | 2 |
| `water8_mio_full` | `-110.65745489032126` | 24 |

| Quantity | Tolerance | Basis |
|---|---|---|
| Total energy | `1e-8` eV absolute | See below |
| Force component | `1e-6` eV/Angstrom absolute | See below |
| Mulliken charge | `1e-7` electrons absolute | See below |
| Charge sum vs system charge | `1e-9` electrons | Neutrality is exact by construction |

**The tolerances rest on a measurement, not on a guess.** Two facts were established
before choosing them:

1. **Run-to-run reproduction is bit-identical.** Repeating each case in the same process
   gives a difference of exactly `0.0` in energy, forces and charges. So the tolerance is
   not absorbing run noise; it exists to allow for a different LAPACK on a different
   machine.
2. **Sensitivity to SCF convergence was measured directly** by tightening `SCF_TOL` from
   its default `1e-6` to `1e-10` and comparing:

   | Observable | Shift under a 4-order SCF tightening |
   |---|---|
   | Total energy | `0.0` exactly |
   | Force component | `3.9e-8` eV/Angstrom |
   | Mulliken charge | `5.0e-9` electrons |

   The energy is unchanged to the last bit because it is variational in the converged
   density: a density error enters at second order. Forces and charges are first order in
   that error, which is why they get looser bands than the energy despite being smaller
   numbers. Each tolerance sits roughly 20-25x above its measured floor -- tight enough
   that a genuine change is caught, loose enough that the test is not a detector for the
   SCF iteration path.

The energy tolerance of `1e-8` eV is `9e-11` relative on the water case, comfortably
inside the `1e-6` relative ceiling that
`test_tolerances_are_documented_not_ad_hoc` enforces. That structural test exists because
Phase 4's D-24 established the principle that a band which can be quietly widened is not
a gate. **A tolerance here is never to be widened to make a failing comparison pass.** A
moved number means something changed; investigate it.

---

## What this does NOT cover

This section is deliberately blunt. REG-02's literal wording is **not** satisfied, and
reporting it green without reading this section would be a misrepresentation.

1. **The notebook is never executed.** A change that breaks the notebook's own plumbing
   -- a renamed keyword argument, a changed constructor signature, a reordered positional
   parameter, a removed export -- would **not** be caught by these tests. The test module
   calls the same API the notebook calls, so a signature change that breaks both would be
   caught; but a change that breaks only the notebook's specific call spelling would not.

2. **Cell 5's PBC + PME + MD path is entirely uncovered**, because `experiments/COORD.pdb`
   is missing from the repository. This is the notebook's first computational cell and its
   PME branch. `tests/test_scf.py` exercises PME on CH4, so PME is not wholly untested, but
   the notebook's own PME case is not reproduced here. **This is a repository defect worth
   fixing on its own merits.**

3. **Cell 14's four-member batch path is uncovered.** The `water8_mio_full` case
   deliberately reduces the original batch of four structures on CUDA to a single structure
   on CPU. `StructureBatch`, `ESDriverBatch` and the batched force path are therefore not
   exercised by this module. The cell's subsequent `MDXLBatch` run (cell 15) is likewise
   not reproduced -- MD is out of Phase 5 scope.

4. **No MD path is covered at all** (cells 15, 20-23), and no plotting or analysis cell.

5. **These are regression pins, not physical validation.** A green run means the numbers
   have not moved since commit `db63487`. It does not mean they are correct. There is no
   external reference for these values and none is claimed.

---

## Related records

- `.planning/phases/05-regression-safety-and-support-policy-cleanup/05-VALIDATION.md`
  Sampling Risk item 1 -- where this gap was first flagged and where the resolution is
  recorded.
- `.planning/phases/05-regression-safety-and-support-policy-cleanup/05-02-PLAN.md` -- the
  plan carrying the checkpoint, its three costed options, and the verified-facts table.
- `tests/test_simple_format_regression.py` -- the module this decision produced.

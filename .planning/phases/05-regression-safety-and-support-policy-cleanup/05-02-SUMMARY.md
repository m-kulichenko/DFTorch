---
phase: 05-regression-safety-and-support-policy-cleanup
plan: 02
subsystem: regression-safety
tags: [regression, baseline, notebook, reg-02, pytest, tolerances, provenance]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: the 72-test green baseline and the ASCII-only / pinned-literal-with-provenance test conventions this module follows
provides:
  - A machine-checked baseline of simple-format total energies, forces and Mulliken charges pinned at commit db63487, before any Phase 5 code change
  - docs/REG-02-NOTEBOOK-BASELINE.md, the REG-02 disposition record with the evidence behind each obstacle and an explicit uncovered-paths section
  - Measured tolerance basis (SCF-sensitivity study) that later phases can reuse rather than re-derive
  - A documentation-integrity test that fails if the REG-02 limitations record is hollowed out
affects: [05-01, 05-03, 05-04, 05-05, 05-06, 05-07, 06-scf]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A regression tolerance is derived from a measured sensitivity study, not chosen by feel, and the measurement is recorded in the case table"
    - "A substitute artifact for a requirement carries a mandatory, blunt section naming what it does not cover, and a test asserts that section still exists"
    - "Symmetry-zero quantities are pinned as exact 0.0 with the measured magnitude in a comment, turning a noise pin into a symmetry assertion"

key-files:
  created:
    - docs/REG-02-NOTEBOOK-BASELINE.md
    - tests/test_simple_format_regression.py
  modified: []

key-decisions:
  - "REG-02 disposition is option-a: extract the notebook's runnable simple-format calculations into pytest rather than executing the notebook. Chosen by the developer at a blocking checkpoint, not inferred by an agent."
  - "REG-02 is recorded as satisfied in substance, not in form. The notebook is never executed, so notebook-only plumbing breakage is an accepted, documented gap."
  - "Tolerances derive from a measured SCF-convergence sensitivity study rather than from a guess: 1e-8 eV energy, 1e-6 eV/Angstrom force, 1e-7 electron charge."
  - "The energy band is the tightest of the three because the total energy is variational in the converged density (second order in the density error), while forces and charges are first order."
  - "Notebook stored cell outputs were refused as reference values; every number was recomputed locally on db63487."
  - "Cell 5's PBC + PME + MD path is left uncovered and the missing experiments/COORD.pdb is recorded as a genuine repository defect rather than papered over."

patterns-established:
  - "Documentation-integrity test: when a requirement is met by a substitute, a test asserts the limitations document still names the selected option, the blocking evidence, and the uncovered-paths section, so the requirement cannot silently read green."
  - "Sensitivity-measured tolerances: before pinning, tighten the convergence knob by several orders and measure how far each observable moves; set each band ~20-25x above its own measured floor."

requirements-completed: [REG-02]

metrics:
  duration: ~35 min
  tasks: 2
  files-created: 2
  files-modified: 0
  tests-added: 11
  suite-before: 72 passed / 0 failed
  suite-after: 83 passed / 0 failed
  completed: 2026-07-30

status: complete
---

# Phase 5 Plan 02: REG-02 Simple-Format Regression Baseline Summary

REG-02 is settled by developer decision (`option-a`) and backed by an 11-test pytest module
that pins the tutorial notebook's three runnable simple-format calculations -- energies,
forces and Mulliken charges -- to numbers measured on `db63487`, before any other Phase 5
plan touched code.

## What was built

### Task 1: REG-02 disposition (checkpoint:decision)

The developer selected **`option-a`** -- extract the notebook's runnable simple-format
calculations into a pytest module -- after being shown all three options with the plan's
own pros and cons verbatim, including the three verified obstacles and the fact that the
notebook's 23 stored cell outputs came from unknown hardware and are not a usable
baseline.

Recorded reason: option A adds no dependencies, runs in CI alongside the existing tests,
and protects every future phase rather than only this one. **Option B** was rejected
because it required inventing a `COORD.pdb` geometry and installing new runtime
dependencies in a phase whose stated purpose is regression safety. **Option C** was
rejected because it would have left Phase 5 with no numeric baseline at all while roughly
twenty files are edited, which is the specific risk this phase exists to eliminate.

The decision was **not** silently rescoped by an agent, and `REQUIREMENTS.md` was not
reworded to match what was convenient to build.

Commit: `130dbea` (recorded together with Task 2's deliverables, since
`test_uncovered_notebook_paths_are_recorded` asserts the decision document's content and
the two would break apart).

### Task 2: The baseline

**Precedence check passed.** `git log --oneline` showed `db63487 test(04-04)` at HEAD with
no `05-0*` commit preceding this plan. The baseline is therefore a snapshot of pre-Phase-5
behaviour, which is the only thing that makes it worth anything.

`tests/test_simple_format_regression.py` -- 11 tests over 3 cases:

| Case | Notebook cell | Total energy (eV) | Atoms |
|---|---|---|---|
| `o2_mio_unrestricted` | 17 | `-9.031346598527405` | 2 |
| `o2_3ob_dftb3` | 25 | `-8.270081470047817` | 2 |
| `water8_mio_full` | 14 (single-structure CPU variant) | `-110.65745489032126` | 24 |

Per case: total energy pinned, full `f_tot` array pinned with a shape and finiteness
check, per-atom Mulliken charges pinned plus an exact-neutrality sum check. Plus two
structural tests: `test_tolerances_are_documented_not_ad_hoc` and
`test_uncovered_notebook_paths_are_recorded`.

`docs/REG-02-NOTEBOOK-BASELINE.md` -- the decision, the evidence for each of the three
obstacles, the cell-to-case mapping, the pinned values with their tolerance basis, and a
deliberately blunt "What this does NOT cover" section.

## Tolerances, and why they are these numbers

The plan warned against a band that can be quietly widened. Rather than pick tolerances by
feel, two things were measured on `db63487` first:

1. **Run-to-run reproduction is bit-identical.** Repeating each case in the same process
   gives a difference of exactly `0.0` in energy, forces and charges. No band here is
   absorbing run noise; the bands exist only to survive a different LAPACK in CI.
2. **Sensitivity to SCF convergence was measured directly**, by tightening `SCF_TOL` from
   its default `1e-6` to `1e-10` and diffing:

   | Observable | Shift under a 4-order SCF tightening |
   |---|---|
   | Total energy | `0.0` exactly |
   | Force component | `3.9e-08` eV/Angstrom |
   | Mulliken charge | `5.0e-09` electrons |

Chosen bands, each ~20-25x above its own measured floor:

| Quantity | Tolerance | Reasoning |
|---|---|---|
| Total energy | `1e-8` eV absolute | Variational in the converged density, so a density error enters at second order -- it did not move at all. `1e-8` eV is `9e-11` relative on the water case. |
| Force component | `1e-6` eV/Angstrom | First order in the density error, hence a looser band than the energy despite being a smaller number. 25x above the measured `3.9e-08` floor. |
| Mulliken charge | `1e-7` electrons | Same first-order argument. 20x above the measured `5.0e-09` floor. |
| Charge sum | `1e-9` electrons | Neutrality is an exact identity of the construction, not a converged quantity; measured residuals `~1e-14` and `~1e-16`. |

`test_tolerances_are_documented_not_ad_hoc` enforces that every case carries a non-empty
`tolerance_rationale` and that no energy band exceeds `1e-6` relative against that case's
own pinned energy. The tightest case (`water8_mio_full`) sits four orders inside that
ceiling.

## Deviations from Plan

**1. [Rule 2 - Missing critical functionality] Added `test_uncovered_notebook_paths_are_recorded`**

- **Found during:** Task 2
- **Issue:** The plan's threat register rates T-05-06 ("REG-02 reported satisfied by an
  artifact that does not test it") as **high** and assigns it `mitigate`. Its stated
  mitigation is that `docs/REG-02-NOTEBOOK-BASELINE.md` carries a mandatory section naming
  what is not covered. But nothing enforced that: the document could be deleted, truncated,
  or have its limitations section removed, and REG-02 would then read green on the strength
  of a passing test module that does not test the notebook. A `mitigate` disposition on a
  high-severity threat is a correctness requirement, not a nice-to-have.
- **Fix:** Added a test asserting the document exists and still contains `option-a`,
  `COORD.pdb`, and the string `What this does NOT cover`.
- **Files:** `tests/test_simple_format_regression.py`
- **Commit:** `130dbea`

**2. [Plan latitude] Reference tests are parametrized rather than written one function per case**

- **Found during:** Task 2
- **Issue:** The plan's artifact table names `test_<case>_total_energy_is_pinned` "one per
  case", which read literally means nine near-duplicate function bodies.
- **Fix:** Used `@pytest.mark.parametrize("case_name", CASE_NAMES)`, which still yields
  exactly one test instance per case at collection time with per-case failure reporting
  (`test_total_energy_is_pinned[water8_mio_full]`). No acceptance criterion referenced the
  literal function names.
- **Files:** `tests/test_simple_format_regression.py`
- **Commit:** `130dbea`

**3. [Plan latitude] Symmetry-zero quantities pinned as exact `0.0` rather than as measured noise**

- **Found during:** Task 2
- **Issue:** For both O2 cases the transverse force components measured `~1e-15` and the
  homonuclear charges measured `~1e-16`. Recording those literals verbatim would pin
  last-digit floating-point noise as though it were a reference value.
- **Fix:** Recorded them as exact `0.0`, with the measured magnitude in an adjacent
  comment. Under the `1e-6` band this becomes a symmetry assertion (the diatomic lies along
  x; the homonuclear molecule carries no net atomic charge), which is stronger and more
  readable than pinning noise.
- **Files:** `tests/test_simple_format_regression.py`
- **Commit:** `130dbea`

No auto-fixed bugs. No architectural decisions required. No `xfail` was needed -- all three
cases ran cleanly on the current tree, so the plan's contingency for a non-converging case
did not fire.

## What this does NOT cover (carried forward deliberately)

REG-02's literal wording is **not** satisfied. Reporting it green without this list would be
a misrepresentation:

1. **The notebook is never executed.** A change breaking only the notebook's own plumbing --
   a renamed keyword argument, a changed constructor signature -- would not be caught.
2. **Cell 5's PBC + PME + MD path is entirely uncovered**, because `experiments/COORD.pdb`
   is not in the repository. This is the notebook's first computational cell. **This is a
   genuine repository defect** -- the tutorial is broken for every new user from a clean
   checkout, not just for this harness. Recorded for whoever repairs the tutorial.
3. **Cell 14's four-member batch path is uncovered.** `StructureBatch`, `ESDriverBatch` and
   the batched force path are not exercised; the case was reduced to a single CPU structure.
4. **No MD path is covered** (cells 15, 20-23), and no plotting cell.
5. **These are regression pins, not physical validation.** Green means the numbers have not
   moved since `db63487`. It does not mean they are correct; no external reference is claimed.

Item 2 is the one worth acting on independently of Phase 5.

## Verification

| Check | Result |
|---|---|
| `uv run pytest tests/test_simple_format_regression.py -q` | 11 passed, 0 xpassed |
| `uv run pytest -q` | **83 passed / 0 failed** in 47.79s (baseline 72 + 11 new) |
| Precedence: no `05-0*` commit precedes this plan | confirmed, HEAD was `db63487 test(04-04)` |
| `uv pip list` unchanged | confirmed by diff -- **no package installed** |
| ASCII-only, both files | confirmed via `open(..., encoding='ascii').read()` |
| `docs/REG-02-NOTEBOOK-BASELINE.md` contains `option-a`, `COORD.pdb`, uncovered section | confirmed, and asserted by test |
| Commit introduced no file deletions | confirmed via `git diff --diff-filter=D` |

## Known Stubs

None. No placeholder values, no skipped tests, no unrun `<verify>` steps. Both `<verify>`
commands were executed and both passed.

## Threat Flags

None. This plan adds no production code, no network I/O, no deserialisation and no new
attack surface. The notebook was read as JSON text for its cell sources and never executed.

## Self-Check: PASSED

- `docs/REG-02-NOTEBOOK-BASELINE.md` -- FOUND
- `tests/test_simple_format_regression.py` -- FOUND
- Commit `130dbea` -- FOUND in `git log`

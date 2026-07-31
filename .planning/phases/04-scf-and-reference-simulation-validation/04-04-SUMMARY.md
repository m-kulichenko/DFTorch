---
phase: 04-scf-and-reference-simulation-validation
plan: 04
subsystem: testing
tags: [pytest, dftb, f-orbitals, validation, energy-scan, documentation, slater-koster]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "plan 04-01's forward(do_scf=False) branch — without it every scanned energy is unset"
  - phase: 03-h0-s-routing-and-f-angular-blocks
    provides: source-locked f angular blocks, so H0/S assemble for the 16-orbital Eu-N pair
provides:
  - "tests/f_orbital_data/README-EU-N-CASE.md: the durable Eu-N case document, shipped beside the SKF fixtures so it survives .planning/ archival"
  - "tests/test_eu_n_scan.py: the passing SIM-05 gate — 21-point single-shot energy scan with an interior-minimum band assertion"
  - "A measured Eu-N binding curve (1.60-3.60 A) whose minimum is 2.40 A, reproducing the plan-time reference exactly"
  - "EU_N_TARGET_ANGSTROM / EU_N_BAND_FRACTION as the single place a future band decision changes"
affects: [04-05 SIM-05 manual sign-off, deferred self-consistent-SCF phase, deferred spin-polarized-f phase, deferred f-derivatives phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tolerance bands derived as a fraction of a named target, with a no-physics guard test pinning the derivation so an absolute half-width substitution fails the suite"
    - "Scan ranges that bracket both tolerance edges, so a near-edge result is resolvable rather than clipped into a false pass"
    - "Asymmetric failure messages: the same assertion reports different interpretations (red flag vs. escalate-to-human) depending on which side it failed"
    - "Module-scoped pytest fixture computing an expensive physics curve once, returning plain Python lists so nothing survives the float64/module-reset harness teardown"
    - "Failure messages carry the full input curve, so a failure is diagnosable from pytest output alone without a re-run"

key-files:
  created:
    - tests/f_orbital_data/README-EU-N-CASE.md
    - tests/test_eu_n_scan.py
  modified: []

key-decisions:
  - "The measured minimum is 2.40 A (index 8 of 20), strictly interior and inside [2.124, 3.186] — SIM-05's gate passes on real measurement, not on a widened band"
  - "Constants is built once per module and reused across all 21 scan points; verified bit-identical to per-point construction and the reason the module fits the 30-second feedback budget (46 s -> 9 s)"
  - "locate_minimum breaks an exact float64 tie toward the SMALLER separation, because under D-24's asymmetry a short reading escalates to a human while a long reading is a red flag"
  - "The test module is pure ASCII: em dashes in assertion messages mangled on a cp1252 console, defeating the requirement that a failure be diagnosable from pytest output alone"
  - "The band's looseness is itself asserted (EU_N_BAND_FRACTION >= 0.10), so tightening below D-20's floor fails the suite rather than passing quietly"

patterns-established:
  - "Guard tests are validated by mutation: 11 deliberate violations were each confirmed to fail on the correct branch before the module was committed"
  - "A validation case document ships next to its fixtures, not under .planning/, so it outlives milestone archival"

requirements-completed: [SIM-04, SIM-05]

coverage:
  - id: D1
    description: "The Eu-N validation case is documented beside its fixtures with geometry, SKF parameter files, observable, units and tolerance individually locatable, without reading any .planning/ file"
    requirement: SIM-04
    verification:
      - kind: other
        ref: "section check: all five `## Geometry|SKF parameter files|Observable|Units|Tolerance` headings present in tests/f_orbital_data/README-EU-N-CASE.md"
        status: pass
      - kind: other
        ref: "content check: all four read SKF files and all five deliberately-unused Ga files named; all 21 reference-curve separations present; 6 sampling-risk items present"
        status: pass
    human_judgment: false
  - id: D2
    description: "The Eu-N single-shot energy scan locates an energy minimum strictly inside the scanned range and inside the 2.655 A +/-20% band"
    requirement: SIM-05
    verification:
      - kind: integration
        ref: "tests/test_eu_n_scan.py#test_eu_n_energy_scan_locates_interior_minimum"
        status: pass
      - kind: integration
        ref: "tests/test_eu_n_scan.py#test_eu_n_curve_is_a_single_well"
        status: pass
    human_judgment: false
  - id: D3
    description: "The tolerance band is computed as a fraction of the target, so a future edit cannot silently tighten it by substituting an absolute half-width (threat T-04-08)"
    requirement: SIM-05
    verification:
      - kind: unit
        ref: "tests/test_eu_n_scan.py#test_tolerance_band_is_a_fraction_of_the_target"
        status: pass
      - kind: other
        ref: "mutation check M1/M2: absolute half-width (2.4 +/- 0.2) and tightening to +/-5% both fail this test"
        status: pass
    human_judgment: false
  - id: D4
    description: "The scan grid contains points on both sides of both band edges, so a minimum landing near an edge is resolvable rather than clipped"
    requirement: SIM-05
    verification:
      - kind: unit
        ref: "tests/test_eu_n_scan.py#test_scan_grid_brackets_the_tolerance_band"
        status: pass
      - kind: other
        ref: "mutation check M3/M4: clipping the range to 3.00 A or flooring it at 2.20 A both fail this test"
        status: pass
    human_judgment: false
  - id: D5
    description: "Every scanned energy is finite and float64, and the located minimum is unique on the grid under a documented deterministic tie-break"
    requirement: SIM-05
    verification:
      - kind: unit
        ref: "tests/test_eu_n_scan.py#test_eu_n_energies_are_finite_and_float64"
        status: pass
      - kind: unit
        ref: "tests/test_eu_n_scan.py#test_eu_n_minimum_is_unique_on_the_grid"
        status: pass
      - kind: other
        ref: "mutation check M8/M10/M11: a tie, a NaN energy and float32 energies each fail the corresponding guard"
        status: pass
    human_judgment: false
  - id: D6
    description: "A minimum short of the band reports as plausibly-correct diatomic physics needing human judgement; long-of-band or no interior minimum reports as a genuine red flag (threat T-04-09)"
    requirement: SIM-05
    verification:
      - kind: other
        ref: "mutation check M5/M6/M7: each of the three D-24 branches was forced and confirmed to emit its own distinct message"
        status: pass
      - kind: other
        ref: "the rule is carried in three places: the module docstring, the three-branch assertion message, and README-EU-N-CASE.md's `### Asymmetric failure rule`"
        status: pass
    human_judgment: false
  - id: D7
    description: "Physical plausibility of the located minimum — whether 2.40 A for an isolated diatomic is right, as opposed to merely inside a loose band around a coordination-chemistry mean"
    requirement: SIM-05
    verification: []
    human_judgment: true
    rationale: "The 2.655 A target is a mean over dative Eu-N bonds in crowded 8-9-coordinate Eu-Bp complexes, while the case is an isolated diatomic. A green gate confirms the ballpark, not the physics. The phase's validation strategy records this as manual-only, and plan 04-05 owns the sign-off."

# Metrics
duration: 22min
completed: 2026-07-29
status: complete
---

# Phase 4 Plan 04: Eu-N Case Document and SIM-05 Scan Gate Summary

**The Eu-N binding curve is now a passing automated gate: a 21-point single-shot energy scan from 1.60 to 3.60 Å finds a single clean well with its minimum at 2.40 Å — strictly interior, inside the 2.655 Å ±20% band [2.124, 3.186] — with the band derived as a fraction of the target and D-24's asymmetric failure rule encoded in the assertion message, the module docstring, and a case document that ships beside the SKF fixtures.**

## Performance

- **Duration:** 22 min
- **Started:** 2026-07-29T21:38:15Z
- **Completed:** 2026-07-29T22:00:15Z
- **Tasks:** 2 (2 commits)
- **Files modified:** 2 (both created)

## The measured result (the headline the plan's `<output>` asked for)

Measured through the committed module's own helpers (`scan_separations`,
`single_shot_energy`, `locate_minimum`) under the float64 harness:

| Property | Measured |
|---|---|
| Grid | 21 points, 1.60 → 3.60 Å, step 0.10 Å |
| Band | [2.124000, 3.186000] Å (target 2.655 Å, ±20 %) |
| **Located minimum** | **2.40 Å, grid index 8 of 20** |
| Minimum energy | -17.6414415610 eV |
| Strictly interior? | **yes** (not index 0, not index 20) |
| **Inside the band?** | **YES** — comfortably, with 0.28 Å of margin below and 0.79 Å above |
| Unique on the grid? | yes — exactly one point attains the minimum, so the tie-break never fires |
| All energies finite float64? | yes — `{torch.float64}` across all 21 points |
| Single well? | yes — strictly decreasing to index 8, strictly increasing after |
| Matches the plan-time value of 2.40 Å? | **yes, exactly** |
| Wall time, scan module | **16.0 s** (`uv run pytest tests/test_eu_n_scan.py -q`, 6 passed) |

**Every one of the 21 energies reproduces the plan-time reference curve to the printed
precision (7 decimal places).** The curve was independently re-measured twice before the
module was written — once with `Constants` rebuilt per point and once with it reused — and
both runs agreed with the plan's table and with each other to the last digit.

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

**Interpretation, stated plainly so it is not over-read.** 2.40 Å is inside the band, so
SIM-05's gate passes on real measurement rather than on a widened tolerance. But 2.40 Å sits
*below* the 2.655 Å target, which is exactly the direction D-24 predicted: the target is a
mean over dative Eu-N bonds in crowded 8-9-coordinate complexes, and an isolated diatomic
should be shorter. It also lands on top of the independent bulk rock-salt EuN reference
(~2.45-2.5 Å). Two reference points agree, in the direction the physics predicts. That is a
good outcome — and it is still only a smoke test, for the reasons carried in the case
document.

## Accomplishments

- **SIM-05 is a real, passing gate rather than a rescoped promise.** Six tests, 16 seconds,
  green. The phase now owes nothing on the validation half of its goal.
- **The band cannot be silently tightened.** `EU_N_BAND_MIN_ANGSTROM` and
  `EU_N_BAND_MAX_ANGSTROM` are arithmetic expressions over `EU_N_TARGET_ANGSTROM` and
  `EU_N_BAND_FRACTION`; neither is a numeric literal.
  `test_tolerance_band_is_a_fraction_of_the_target` additionally pins the resulting edges to
  2.124/3.186 within 1e-12 *and* asserts `EU_N_BAND_FRACTION >= 0.10`, so both the
  half-width substitution that threat T-04-08 describes and a quiet tightening below D-20's
  floor fail the suite. The specific mis-statement that motivated this ("2.4 ± 0.2 Å", which
  is ±8 %) was verified to fail the test.
- **The scan brackets both band edges, so the gate cannot report a false pass.** 1.60 Å is
  below 2.124 Å and 3.60 Å is above 3.186 Å, with grid points strictly either side of each
  edge (2.10/2.20 straddle the lower, 3.10/3.20 the upper). Research's proposed 1.8-3.0 Å
  range would have clipped the upper edge; clipping it to 3.00 Å was verified to fail
  `test_scan_grid_brackets_the_tolerance_band`.
- **A short minimum can no longer be misread as a broken f implementation.** The three-branch
  failure message, the module docstring, and the case document's own
  `### Asymmetric failure rule` subsection all state both directions. All three D-24 branches
  were forced and confirmed to emit distinct, correct messages.
- **The case document outlives `.planning/`.** `tests/f_orbital_data/README-EU-N-CASE.md`
  ships beside the nine SKF fixtures with the five SIM-04 elements as individually locatable
  `##` headings, plus provenance, the asymmetric rule, the 21-row reference curve, and an
  explicit "what a green test does not prove" section carrying two manual-only checks and six
  sampling-risk items.
- **Guards were validated by mutation, not assumed.** See below.

## Mutation validation of the guards (how RED was established for a test-only deliverable)

A gate module cannot be driven RED the usual way — the behaviour it tests
(`forward(do_scf=False)`) already shipped in plan 04-01, so the tests pass on first write. To
establish that the guards actually bite rather than merely pass, 11 deliberate violations were
injected and each confirmed to fail **on the correct branch** before the module was committed:

| # | Injected violation | Guard that caught it | Branch correct? |
|---|---|---|---|
| M1 | Band re-expressed as absolute half-width `2.4 ± 0.2` | `test_tolerance_band_is_a_fraction_of_the_target` | yes |
| M2 | Band silently tightened to ±5 % | `test_tolerance_band_is_a_fraction_of_the_target` | yes |
| M3 | Scan range clipped to 3.00 Å (upper edge unbracketed) | `test_scan_grid_brackets_the_tolerance_band` | yes |
| M4 | Scan range floored at 2.20 Å (lower edge unbracketed) | `test_scan_grid_brackets_the_tolerance_band` | yes |
| M5 | Minimum forced SHORT of the band | interior-minimum gate | yes — emitted "NOT evidence that the f implementation is broken" |
| M6 | Minimum forced LONG of the band | interior-minimum gate | yes — emitted "GENUINE RED FLAG: minimum long of the band" |
| M7 | Curve truncated at the minimum (no interior well) | interior-minimum gate | yes — emitted "no interior minimum" |
| M8 | Two grid points tied for the minimum | `test_eu_n_minimum_is_unique_on_the_grid` | yes |
| M9 | Bump inserted on the approach (double well) | `test_eu_n_curve_is_a_single_well` | yes |
| M10 | NaN energy injected | `test_eu_n_energies_are_finite_and_float64` | yes |
| M11 | float32 energies | `test_eu_n_energies_are_finite_and_float64` | yes |

The mutation harness was a scratch script outside the repository; nothing in it was committed
and no repository file was mutated in place. Two rounds were needed — the first M9 mutant
(+0.5 eV) was too weak to actually break monotonicity, which was a defect in the mutant, not
in the guard; +2.0 eV exercised it correctly.

## Verification

| Check | Result |
|---|---|
| `uv run pytest tests/test_eu_n_scan.py -q` | **6 passed**, 16.0 s |
| `uv run pytest -q` (full suite) | **72 passed, 0 failed** — exit 0, 72 progress dots, no `F`/`E` |
| Section check on `README-EU-N-CASE.md` | all five SIM-04 `##` headings present |
| Task 1 acceptance criteria (7) | all pass |
| Task 2 acceptance criteria (8) | all pass |
| Guard mutation checks (11) | all bite, all on the correct branch |

The suite was **66 passed / 0 failed** before this plan; it is **72 passed / 0 failed** after
(+6, exactly this module). Nothing regressed. The full run takes 2 m 11 s.

Note on the suite count line: this project's `addopts = ["-ra", "-q"]` produces a progress
line without a trailing summary in this shell, so the count was confirmed by exit code 0 plus
a character census of the progress output (72 dots, zero failure/error/skip markers).

## Task Commits

1. **Task 1: Document the Eu-N diatomic validation case beside its fixtures (SIM-04, D-21)**
   - `4ebb975` (docs) — `tests/f_orbital_data/README-EU-N-CASE.md`, 322 lines
2. **Task 2: Eu-N energy scan gate with the D-24 asymmetric failure rule (SIM-05)**
   - `db63487` (test) — `tests/test_eu_n_scan.py`, 474 lines, 6 tests

**TDD gate compliance — no `feat(...)` commit exists for Task 2, deliberately.** Task 2 is
flagged `tdd="true"`, but its `<files>` list contains only `tests/test_eu_n_scan.py`: it is a
test-only deliverable with no production code, so it is exempt from the behaviour-adding
predicate (no non-test source files). The behaviour under test shipped in plan 04-01 as
`ESDriver.forward(do_scf=False)` with its own `test(...)` → `feat(...)` pair, and Task 2's own
acceptance criteria require these tests to **pass**, not fail. A synthetic RED would have
meant breaking production code to watch it break. The mutation table above is the honest
substitute: it demonstrates every guard failing when its contract is violated.

## Files Created/Modified

- `tests/f_orbital_data/README-EU-N-CASE.md` — new. The durable case document, shipped beside
  the SKF fixtures rather than under `.planning/`. Sections: Geometry (2 atoms, 20 AOs = Eu 16
  + N 4, 7 occupied levels), SKF parameter files (the four read — `Eu-Eu.skf` extended with
  Eu's f Hubbard U of 0.50 Ha = 13.605693 eV and occupations 7f/2s; `N-N.skf` simple with
  3p/2s; `Eu-N.skf`; `N-Eu.skf` — and the five Ga files deliberately unexercised), Observable
  (single non-SCC diagonalization, `e_coul` exactly 0 and why, scan-not-optimize and why),
  Units, Tolerance (2.655 Å ±20 % → [2.124, 3.186], as a fraction never a half-width),
  Provenance and caveats with `### Asymmetric failure rule`, what a green test does not prove
  (2 manual-only checks + 6 sampling-risk items), How to run, and the 21-row reference curve.
- `tests/test_eu_n_scan.py` — new. Module docstring carrying D-24 in both directions;
  constants `EU_N_TARGET_ANGSTROM`, `EU_N_BAND_FRACTION`, the two derived band edges,
  `SCAN_MIN/MAX/STEP_ANGSTROM`; helpers `scan_separations`, `single_shot_energy`,
  `locate_minimum`, `_format_curve`; module-scoped `eu_n_scan` fixture; the six tests. Copies
  `run_with_float64` from `tests/test_f_orbital_skf.py` and the `TORCHDYNAMO_DISABLE` preamble
  from `tests/test_scf.py`, per this suite's convention of copying rather than importing across
  test modules.

## Decisions Made

- **`Constants` is built once per module and reused across all 21 scan points.** This is the
  one place the implementation departs from the plan's literal instruction (see Deviations).
  `Constants` reads the geometry file only to recover the species list, which is `(Eu, N)` at
  every scan point. Reuse was verified **bit-identical**: 21 energies with `Constants` rebuilt
  per point versus one reused instance agree to the last digit at every separation, which also
  rules out cross-point contamination of the shared object. The payoff is the feedback budget —
  46.4 s → 8.9 s for the scan loop, and 16.0 s for the whole module including interpreter
  start and imports.
- **`locate_minimum` breaks an exact float64 tie toward the smaller separation.** Deterministic,
  and the conservative choice under D-24's asymmetry: an ambiguous curve is biased toward the
  outcome that escalates to a human (short) rather than the one read as a definite defect
  (long). `test_eu_n_minimum_is_unique_on_the_grid` records that the real curve never
  exercises this branch, so the reported 2.40 Å is a property of the curve and not of the
  policy.
- **The test module is pure ASCII.** Ten em dashes (U+2014) were replaced with hyphens after a
  mutation run showed them rendering as `?` on a cp1252 console. Three of them sat inside
  runtime assertion messages, which directly defeats the plan's requirement that a failure be
  diagnosable from the pytest output alone. The markdown case document keeps its Å, ± and ⁷ —
  it is read in editors, not printed to a terminal by a failing test.
- **The band's looseness is asserted, not just its exactness.** `EU_N_BAND_FRACTION >= 0.10`
  encodes D-20's floor, so a future edit that keeps the fraction form but tightens it to ±5 %
  still fails.
- **The scan returns plain floats and dtype *names*, not tensors.** `run_with_float64` unloads
  the dftorch modules and restores the default dtype on the way out, so the fixture records
  `str(e_tot.dtype)` per point rather than holding tensors across that teardown. This is what
  lets `test_eu_n_energies_are_finite_and_float64` still assert float64 discipline from a
  module-scoped curve.
- **No packages were installed.** `ruff` is a declared `dev` extra but is not present in the
  environment, so the CI lint step could not be reproduced locally. Rather than install
  anything (Rule 3's package-manager exclusion, threat T-04-SC), the new module was written to
  match the already-committed sibling `tests/test_single_shot_energy.py`: zero lines over 88
  characters and the identical import preamble. Pre-existing files carry many over-long lines
  (`test_f_orbital_skf.py` has 26, `test_scf.py` has 6), so repo-wide `ruff format --check`
  state is a pre-existing condition and out of this plan's scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `single_shot_energy` gained optional `const` / `driver` caches to stay inside the feedback budget**

- **Found during:** Task 2 (before writing the module — caught by measuring the scan first)
- **Issue:** The plan specifies `single_shot_energy(separation, tmp_dir)` building `Constants`
  and `Structure` per call, and states the scan measured **16 s** at plan time. On this
  machine that construction pattern measured **46.4 s** for the 21-point loop. That still
  clears Task 2's acceptance criterion of "under 60 seconds", but only barely, and it blows
  the **30-second maximum feedback latency** that the phase's validation strategy sets and
  that the plan's own action block names as the reason to compute the curve once ("Computing
  the curve once is what keeps the module inside the 30-second feedback budget"). The literal
  instruction and the stated intent were in conflict on this hardware.
- **Fix:** `single_shot_energy(separation, tmp_dir, *, const=None, driver=None)`. The first two
  positional parameters are unchanged and the helper is **fully standalone when called with
  neither** — it builds both from scratch, exactly as the plan describes. The module-scoped
  fixture builds `Constants` and `ESDriver` once from the first grid point and threads them in.
  Safe because `Constants` depends only on the species list, identical at every scan point.
- **Verification:** Bit-identical output proven, not assumed — per-point construction versus
  reuse gives the same 21 energies to the last digit at every separation (max |A-B| =
  0.000e+00), which also rules out state leaking between points through the shared object.
  Both runs match the plan-time reference table. Scan loop 46.4 s → 8.9 s; full module 16.0 s,
  inside both the 30-second budget and the 60-second criterion, and coincidentally matching the
  plan's 16 s figure.
- **Files modified:** `tests/test_eu_n_scan.py`
- **Committed in:** `db63487` (Task 2 commit)

**2. [Rule 1 - Bug] Em dashes in assertion messages mangled on a cp1252 console**

- **Found during:** Task 2 (surfaced by the M5 mutation run, which printed `?` for the dash)
- **Issue:** Ten U+2014 em dashes in the module, three inside runtime assertion messages.
  Non-ASCII in a failure message renders as replacement characters on a Windows cp1252
  console, which undermines the plan's explicit requirement that "a failure is diagnosable
  from the pytest output alone without a re-run".
- **Fix:** All ten replaced with ASCII hyphens; the module is now byte-wise pure ASCII
  (verified: zero code points above 127).
- **Verification:** M5 re-run emits `MINIMUM SHORT OF THE BAND - ESCALATE TO HUMAN JUDGEMENT.`
  cleanly; 6 passed unchanged.
- **Files modified:** `tests/test_eu_n_scan.py`
- **Committed in:** `db63487` (Task 2 commit — caught before the file was ever committed)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both preserve the plan's stated intent where following its letter would
have undercut it — deviation 1 keeps the feedback budget the plan named as the reason for the
design, and deviation 2 keeps the failure messages readable, which is the point of the
three-branch construction. No scope creep: no production code was touched, no package was
installed, and no additional test was added beyond the six the plan specifies.

## Issues Encountered

- **The plan's 16-second scan figure did not reproduce with the plan's literal helper
  design** (46.4 s here). Resolved by the `Constants` cache above, which brought the module to
  16.0 s. Worth recording because it means the plan-time figure was probably measured with
  some form of reuse already, and a future reader comparing timings should know which design
  the number belongs to.
- **`ruff` is unavailable locally**, so the CI lint gate (`uv run ruff check .`,
  `uv run ruff format --check .`) could not be reproduced. Not resolved by installing it —
  package installs are excluded. Mitigated by matching the already-committed sibling test
  module's style exactly (zero lines over 88 chars). Residual risk is low but non-zero: if CI
  lint flags the new file, it will be a formatting nit, not a logic problem. Note that
  repo-wide `ruff format --check` is very likely already failing for pre-existing reasons
  (26 over-long lines in `test_f_orbital_skf.py` alone), which is out of scope here.
- **Pre-existing uncommitted modifications** to `src/dftorch/_bond_integral.py`,
  `src/dftorch/_io.py` and the `egg-info` files were present at plan start and were left
  untouched — not this plan's work, and the scope boundary forbids sweeping them in.

## Threat Flags

None. This plan added one markdown document and one test module. No production code changed,
no network I/O, no authentication, no untrusted deserialisation, no new trust boundary.

Threat dispositions from the plan, all discharged:

| Threat | Disposition | Outcome |
|---|---|---|
| T-04-08 — band silently tightened or widened | mitigate | **Mitigated and mutation-verified.** M1 (half-width) and M2 (±5 %) both fail the suite. |
| T-04-09 — short-side failure misread as a broken f implementation | mitigate | **Mitigated and mutation-verified.** M5 emits the human-judgement branch; the rule is carried in three independent places. |
| T-04-10 — scan runtime growing past the feedback budget | accept | **Better than accepted — actively controlled.** The module-scoped fixture plus the `Constants` cache put the module at 16.0 s against a 30 s budget. |
| T-04-SC — package installs | mitigate / not applicable | **Remained inactive. No packages were installed**, including `ruff`, which was wanted but declined. |

## Known Stubs

None. Both artifacts are complete: no placeholder values, no TODO/FIXME markers, no skipped
or `xfail` tests, and every `<verify>` command in the plan was executed and recorded above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Phase 4's automated work is complete.** All five plans' worth of gates are green and the
suite is 72 passed / 0 failed.

**Ready for plan 04-05 (the SIM-05 manual sign-off):**

- The number a human must adjudicate is **2.40 Å**, inside [2.124, 3.186] Å. The gate passes,
  so 04-05's blocking human-verify is a confirmation rather than a rescue.
- The judgement 04-05 owes is the one recorded as `human_judgment: true` above (deliverable
  D7): 2.40 Å is inside the band and lands on the independent bulk rock-salt EuN reference
  (~2.45-2.5 Å), in the direction D-24 predicted for a diatomic versus a coordination mean.
  What a human still owes is whether that agreement is meaningful or coincidental.
- Everything 04-05 needs to make that call without reading `.planning/` is in
  `tests/f_orbital_data/README-EU-N-CASE.md`.
- **The band must not be edited as part of a sign-off.** Threat T-04-11 assigns band changes
  to a recorded decision; `test_tolerance_band_is_a_fraction_of_the_target` now enforces the
  form and the ±10 % floor mechanically.

**Carried-forward constraints (unchanged by this plan):**

- f derivatives remain unimplemented. `calc_forces`, `ESDriverBatch.calc_forces` and
  analytical stress still raise `FDerivativeUnsupportedError` for any 16-orbital system. This
  is precisely why the gate is a scan; do not replace it with an optimizer.
- Single-shot is not SCF, and `e_coul` is pinned to exactly 0 by design. Do **not** switch a
  Coulomb term on to "improve" the curve — first-iterate Mulliken charges (`q_Eu ≈ -2.70`,
  drifting to -2.99 at 4 Å) were measured to destroy the well entirely, which would make this
  gate unsatisfiable.
- Spin stays refused for f systems; the closed-shell treatment of Eu 4f⁷ is an approximation
  the band absorbs.
- The shell-resolved f Hubbard/charge plumbing is validated but unconsumed by this energy
  path — integration between them is unsampled until self-consistency lands.
- Only the Eu-N pair is exercised, along a single direction (+x). Seven of the nine SKF
  fixtures stay untouched by design; do not invent Ga work to justify them.
- `_bond_integral` still exports a single `R_orb` (the longest grid) for all pair types while
  this fixture directory mixes radial grids. The curve reproduces exactly, which says the path
  is self-consistent, but says nothing about whether the shared grid is correct across
  mixed-grid pairs. Recorded again in the case document.

## Self-Check: PASSED

Both claimed files exist on disk (`tests/f_orbital_data/README-EU-N-CASE.md`,
`tests/test_eu_n_scan.py`) and both claimed commit hashes resolve in git log (`4ebb975`,
`db63487`).

---
*Phase: 04-scf-and-reference-simulation-validation*
*Completed: 2026-07-29*

---
phase: 04-scf-and-reference-simulation-validation
plan: 03
subsystem: physics-engine
tags: [pytorch, dftb, f-orbitals, hubbard-u, coulomb, ewald, shell-resolved, tdd, pytest]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    plan: 01
    provides: "ESDriver.forward(do_scf=False) and the pinned Eu-N reference energy -17.510444238744924 that this plan asserts is unperturbed"
  - phase: 02-constants-and-structure-basis-metadata
    provides: "4-shell shell_present / n_s..n_f / Hubbard_U_sr scaffolding that already carries the f dimension"
provides:
  - "FShellResolvedCoulombUnsupportedError + _require_no_f_shell_resolved_coulomb: named refusal closing the last silent-zero hole in the f path"
  - "_select_coulomb_hubbard: MAGNETIC_HUBBARD_LDEP as the single gate for the Coulomb-path Hubbard U (D-23)"
  - "structure.C_sr / structure.dCC_sr: shell-resolved Coulomb matrix built alongside (never instead of) the per-atom structure.C"
  - "tests/test_shell_resolved_u.py — 27 tests validating the f dimension in the shell-resolved Hubbard/charge data and both flag branches"
affects: [04-04 Eu-N separation scan, 04-05 SIM-05 gate, deferred self-consistent-SCF phase, deferred spin-polarized-f phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Named exception per unsupported mode, raised as the first statement of the guarded function so every caller is covered rather than only the known call site"
    - "The whole f-unsupported exception taxonomy lives in _slater_koster_pair.py regardless of which module raises it, so the support policy is reviewable in one place"
    - "A selection helper returns (data, selection_flag) so a caller cannot consume one representation while believing it has the other"
    - "New capability is built alongside the existing one and both attributes always exist (None when unbuilt), never replacing a consumed structure"

key-files:
  created:
    - tests/test_shell_resolved_u.py
  modified:
    - src/dftorch/_slater_koster_pair.py
    - src/dftorch/_coulomb_matrix.py
    - src/dftorch/ESDriver.py
    - src/dftorch/Constants.py

key-decisions:
  - "The config gate is the existing MAGNETIC_HUBBARD_LDEP, not a new SHELL_RESOLVED flag (D-23); the 'magnetic' misnomer is accepted to keep the blast radius off the spin/SOC call sites"
  - "The guard is the first statement of ewald_real_space_vectorized_sr, not of the ESDriver call site, so future callers inherit the refusal"
  - "structure.C_sr is additive: the per-atom structure.C build and its argument list are byte-identical, verified by git diff"
  - "The shell-resolved matrix is not built under COUL_METHOD='PME' — there is no real-space neighbor list at that point and no shell-resolved reciprocal-space counterpart"
  - "The seven f angular blocks are refused rather than approximated because no consumer exists that could validate their values this phase"

patterns-established:
  - "Flag-gated capability is validated on both branches in the same test module, with an explicit cross-flag test pinning that the flag selects consumption and not construction"
  - "A refusal message names the call site, the trigger token, the missing pieces by name, the workaround, and why the gap is blocked — and leaks no filesystem path"

requirements-completed: [SIM-03]

coverage:
  - id: D1
    description: "structure.Hubbard_U_sr for Eu-N has 6 entries — Eu s/p/d/f then N s/p — with the f entry equal to the SKF-sourced Eu f Hubbard U"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_hubbard_u_sr_length_matches_active_shells"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_hubbard_u_sr_values"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_f_hubbard_u_traces_to_skf"
        status: pass
    human_judgment: false
  - id: D2
    description: "structure.el_per_shell carries 7 electrons on the Eu f shell and structure.shell_types labels it 4"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_electrons_per_shell"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_shell_types_label_the_f_shell"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_shell_index_ranges"
        status: pass
    human_judgment: false
  - id: D3
    description: "structure.D0 includes the seven f AO entries at reference occupation — reference charge data carries the f shell"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_eu_n_reference_density_includes_f_shell"
        status: pass
    human_judgment: false
  - id: D4
    description: "MAGNETIC_HUBBARD_LDEP selects the shell-resolved Hubbard data for the Coulomb path; unset keeps existing per-atom behaviour"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_select_coulomb_hubbard_returns_shell_resolved_when_flag_set"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_driver_builds_c_sr_when_flag_set_f_free"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_driver_leaves_c_sr_none_when_flag_unset"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_ldep_flag_does_not_change_construction"
        status: pass
    human_judgment: false
  - id: D5
    description: "Requesting the shell-resolved Coulomb matrix for a 16-orbital system raises a named exception instead of returning silently-zero f/p/d shell rows"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_shell_resolved_coulomb_refuses_f_system"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_shell_resolved_coulomb_error_explains_the_gap"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_driver_refuses_f_system_when_flag_set"
        status: pass
    human_judgment: false
  - id: D6
    description: "An f-free system with the flag set still produces a finite shell-resolved Coulomb matrix with no all-zero row, and the supported f path is unperturbed"
    requirement: SIM-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_shell_resolved_coulomb_builds_for_f_free_system"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_driver_f_system_unaffected_when_flag_unset"
        status: pass
      - kind: integration
        ref: "uv run pytest -q — 66 passed, 0 failed (was 39 passed / 0 failed)"
        status: pass
    human_judgment: false

# Metrics
duration: 15min
completed: 2026-07-29
status: complete
---

# Phase 4 Plan 03: Shell-Resolved Hubbard Data and the Coulomb f Refusal Summary

**The f shell is now proven present and correctly valued throughout the shell-resolved Hubbard, shell-label, electron-count, index-range and reference-charge data — `Hubbard_U_sr[3] == 13.605693125` eV traced to `Eu-Eu.skf`'s `Uf = 0.50` Hartree — and the last silent-zero hole in the f path is closed: `ewald_real_space_vectorized_sr` used to hand back a finite `(6, 6)` matrix for Eu-N with row sums `[0.473581, 0, 0, 0, 0.473581, 0]`, and now raises `FShellResolvedCoulombUnsupportedError`.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-07-29T19:12:00Z
- **Completed:** 2026-07-29T19:27:00Z
- **Tasks:** 2 (3 commits)
- **Files modified:** 5 (1 created, 4 modified)

## The observed pre-guard silent-zero matrix (required by the plan's `<output>`)

Measured on this checkout **before** the guard existed, by calling
`ewald_real_space_vectorized_sr` directly with the same argument derivation
`ESDriver.forward` uses for the DFTB3 third-order matrices:

| System | SKF set | Result shape | All finite | Row sums |
|---|---|---|---|---|
| CH4 | `tests/data_skf_mio-1-1` | `(10, 10)` | yes | `[29.580901, 31.214416, 21.476029, 23.290091, 20.502005, 22.178566, 21.766503, 23.616294, 21.769192, 23.632793]` |
| Eu-N @ 2.655 Å | `tests/f_orbital_data` | `(6, 6)` | yes | `[0.473581, 0.0, 0.0, 0.0, 0.473581, 0.0]` |

The full pre-guard Eu-N matrix had exactly **two** non-zero entries, both equal
to `0.4735812210524011`, at `(0, 4)` and `(4, 0)` — the Eu s ↔ N s block. Every
Eu p, d **and** f shell row and column, and the N p row and column, were exactly
zero. The plan's plan-time measurement is reproduced to the digit.

**Cause:** the pair masks in `ewald_real_space_vectorized_sr` test `max_ang`
against 1, 2 and 3 only. Eu has `max_ang == 4`. There is no fall-through branch
and no error — the shells simply never receive a contribution. This is the
Phase 3 silent-drop failure mode reproduced in the electrostatics: a finite,
well-shaped, completely wrong matrix.

**Post-guard:** the same call raises
`FShellResolvedCoulombUnsupportedError` (a `NotImplementedError` subclass),
named at `_coulomb_matrix.ewald_real_space_vectorized_sr`, from the function's
first statement. CH4 is unaffected and still returns its fully populated
`(10, 10)` matrix — the refusal is specific to 16-orbital atoms, not a blanket
disabling of the builder.

## The seven f angular blocks remain unimplemented — refused, not approximated

**Explicit statement required by the plan's `<output>`.** The s-f, f-s, p-f,
f-p, d-f, f-d and f-f blocks of the shell-resolved Coulomb matrix are **not
implemented by this plan and are not approximated**. Any request for them
raises.

This is a dependency conflict, not a difficulty judgement. The shell-resolved
matrix is `(n_shells, n_shells)` — `(6, 6)` for Eu-N — while `_energy.energy`
and `SCFx` consume a `(Nats, Nats)` Coulomb matrix together with per-atom
charges of shape `(Nats,)`. Making the shell-resolved matrix *consumable* would
require shell-resolved charges threaded through the entire SCF loop, which is
precisely the self-consistent SCF work **D-11** defers out of this phase. There
is therefore no consumer in Phase 4 that could validate an f Coulomb value even
if one were written, and writing unvalidatable numbers into an electrostatics
kernel is the exact failure mode this plan exists to close. The refusal message
says all of this in-band, so a developer who hits it learns why rather than
re-deriving it.

## What is validated-but-unconsumed, and why that is the point (D-14)

`Hubbard_U_sr`, `shell_types`, `el_per_shell`, `H_INDEX_START_U` /
`H_INDEX_END_U`, `D0` and now `C_sr` all carry the f dimension with asserted
values, and none of them is read by the D-11 single-shot energy path this
phase ships. D-14 calls this deliberate de-risking rather than dead code: the
later self-consistent SCF phase inherits structures that are already proven
correct on the f axis instead of discovering the gaps from scratch mid-loop.

## Measured Eu-N shell-resolved data (all plan-time facts reconfirmed)

Independently reconfirmed on this checkout before any test was written, under
**both** `MAGNETIC_HUBBARD_LDEP` settings, with byte-identical results:

| Attribute | Measured | Plan-time value | Match |
|---|---|---|---|
| `n_shells_per_atom` | `[4, 2]` | `[4, 2]` | exact |
| `Hubbard_U_sr` | `[5.7143911125, 5.1701633875, 6.8028465625, 13.605693125, 13.3335792625, 13.3335792625]` | same | exact |
| `shell_types` | `[1, 2, 3, 4, 1, 2]` | same | exact |
| `el_per_shell` | `[2.0, 0.0, 0.0, 7.0, 2.0, 3.0]` | same | exact |
| `H_INDEX_START_U` / `H_INDEX_END_U` | `[0, 4]` / `[3, 5]` | same | exact |
| `shell_present[Eu]` / `shell_present[N]` | `[T,T,T,T]` / `[T,T,F,F]` | same | exact |
| `const.Uf[Eu]` / `const.Uf[N]` | `13.605693125` / `0.0` | same | exact |
| `D0` | `[1.0, 0×8, 0.5×7, 1.0, 0.5×3]`, `sum == 7.0` | same | exact |
| `ao_shell_types` | `[1, 2×3, 3×5, 4×7, 1, 2×3]` | same | exact |
| `const.max_ang[Eu]` / `const.n_orb[Eu]` | `4` / `16` | — | — |

`13.605693125` eV is `Eu-Eu.skf` line 3's `Uf = 0.50` Hartree times
`EV_PER_HARTREE = 27.21138625`. **D-15 is satisfied, not blocked**: the f
Hubbard U is SKF-sourced, no zero-fill was needed, and no blocker was surfaced.
`test_eu_n_f_hubbard_u_traces_to_skf` pins the Hartree→eV trace so a future
regression in the extended-format on-site parse fails here rather than
producing a plausible-looking default.

**N has two shells, not three.** `const.shell_present[N] == [True, True, False,
False]` and `const.n_orb[N] == 4`, so `HDIM == 20`. The sketch in
`04-PATTERNS.md` lines 520-528 asserts three shells for N and is wrong; no
assertion in the new module repeats that error.

## MAGNETIC_HUBBARD_LDEP now has a second consumer (D-16 / D-23)

`test_ldep_flag_does_not_change_construction` pins the non-obvious invariant the
gating rests on: `Structure.py:413-441` reads the flag **nowhere**, so
`Hubbard_U_sr`, `shell_types`, `el_per_shell` and `n_shells_per_atom` are
byte-identical under both settings. The flag therefore selects *consumption*,
and both branches are handed data that already exists and agrees.

`_select_coulomb_hubbard(structure, const)` returns
`(structure.Hubbard_U_sr, True)` when the flag is set and
`(structure.Hubbard_U, False)` when it is not. Returning the flag alongside the
tensor is deliberate (threat T-04-06): the two tensors differ in length — 6
versus 2 for Eu-N — and in the index map needed to place them, so a caller must
not be able to consume one while believing it has the other.

**No `SHELL_RESOLVED` flag was introduced.** The token appears exactly once in
the diff, inside `_select_coulomb_hubbard`'s docstring, recording D-23's
decision to reject it. Nothing reads a key by that name.

## Task Commits

1. **Task 1: Validate f dimensions and values in the shell-resolved Hubbard and charge data (D-14)**
   - `e37c57b` (test) — 19 tests: 9 parametrised over both flag settings, plus the cross-flag comparison
2. **Task 2: Gate the Coulomb-path Hubbard U and refuse silent f zeros (D-16, D-23)**
   - `2539976` (test) — RED: 8 tests added, 7 failing
   - `15945bc` (feat) — GREEN: exception, message constant, guard, selection helper, driver wiring, Constants docstring

No REFACTOR commits were needed.

**TDD gate compliance.** Task 2 shows `test(2539976)` → `feat(15945bc)` in git
log order, with the RED commit verified failing before the implementation was
authored. Task 1 is marked `tdd="true"` in the plan but adds **no production
code** — its `<files>` list is the test module alone. It is therefore a
validation/characterisation task rather than a behaviour-adding one, and it has
no RED phase by construction: it pins existing, already-correct construction.
Recorded here rather than glossed, because "19 tests green on first run" is
otherwise indistinguishable from a test that asserts nothing.

## Verification results

| Gate | Result |
|---|---|
| `uv run pytest tests/test_shell_resolved_u.py -q` | **27 passed** (19 from Task 1 + 8 from Task 2) |
| `uv run pytest tests/test_single_shot_energy.py tests/test_spin_guard.py tests/test_f_orbital_skf.py tests/test_scf.py tests/test_dtype_contract.py -q` | **31 passed** |
| `uv run pytest -q` (whole suite) | **66 passed, 0 failed** (was 39 passed / 0 failed) |

The suite grew by exactly the 27 new tests and nothing regressed.

**Diff invariants asserted by the plan's acceptance criteria, verified directly:**

- `git diff src/dftorch/ESDriver.py` shows `structure.C, structure.dCC =
  coulomb_matrix_vectorized(...)` and its entire argument list **unchanged**.
  The file's only deletion in the whole diff is the one-line
  `from ._coulomb_matrix import coulomb_matrix_vectorized` import, replaced by
  the two-name parenthesised form.
- `git diff src/` introduces **no** new upper-case configuration key literal of
  any kind.
- `tests/test_scf.py` still passes, so the per-atom Coulomb path used by the
  existing simple-format SCF is unchanged.
- `test_driver_f_system_unaffected_when_flag_unset` reproduces plan 04-01's
  `e_tot = -17.510444238744924` for Eu-N with the flag unset, to within `1e-6`.

## Files Created/Modified

- `tests/test_shell_resolved_u.py` — new, 27 tests. Copies `run_with_float64`
  and `write_xyz` verbatim from `tests/test_f_orbital_skf.py` rather than
  importing across test modules, matching the suite convention.
- `src/dftorch/_slater_koster_pair.py` — added
  `FShellResolvedCoulombUnsupportedError` and
  `F_SHELL_RESOLVED_COULOMB_UNSUPPORTED_MESSAGE` directly after the plan 04-01
  spin-polarization pair, following the `FDerivativeUnsupportedError` template.
  The class docstring records why the exception lives in this module rather than
  in `_coulomb_matrix.py`.
- `src/dftorch/_coulomb_matrix.py` — imported the new exception and message;
  added module-level `_require_no_f_shell_resolved_coulomb(structure, TYPE,
  context)` modelled on `ESDriver._require_f_derivatives`; called it as the
  first statement of `ewald_real_space_vectorized_sr`; added a `Raises` section
  to that function's docstring; added a comment above the pair-mask block
  recording that the masks cover `max_ang` 1/2/3 only and that `max_ang == 4` is
  refused rather than falling through to zero.
- `src/dftorch/ESDriver.py` — added `ewald_real_space_vectorized_sr` to the
  `_coulomb_matrix` import; added module-level `_select_coulomb_hubbard` next to
  the existing guards; initialised `structure.C_sr` / `structure.dCC_sr` to
  `None` before the `COUL_METHOD` branch; built the shell-resolved matrix in the
  non-PME branch only; added a `Notes` section to `forward`'s docstring
  recording the PME limitation alongside plan 04-01's D-11 note.
- `src/dftorch/Constants.py` — extended the `MAGNETIC_HUBBARD_LDEP` docstring
  entry with its second consumer and the D-23 rationale for keeping the name.
  **Not in the plan's `files_modified` frontmatter, but explicitly required by
  Task 2's `<action>`** ("Add one sentence to the `MAGNETIC_HUBBARD_LDEP` entry
  in the `Constants` class docstring"). Docstring-only; no executable line in
  that file changed.

## Decisions Made

- **The guard is the first statement of `ewald_real_space_vectorized_sr`, not of
  the ESDriver call site.** The function had *zero* callers anywhere in `src/`
  or `tests/` before this plan, and this plan adds the first one. Guarding at
  the function makes the refusal inherited by every future caller; guarding at
  the call site would have protected exactly the one path that already knew
  about the problem.
- **`structure.C_sr` is built alongside `structure.C`, never instead of it.**
  `energy()` and `SCFx` consume `(Nats, Nats)` with per-atom charges. Swapping
  in a `(n_shells, n_shells)` matrix would not be an upgrade, it would be a
  shape error at best and a silently misindexed energy at worst.
- **Both attributes are initialised to `None` before the `COUL_METHOD` branch.**
  This follows plan 04-01's precedent of pinning attributes rather than leaving
  them unset, so a caller inspecting the structure after any run sees the same
  attribute set and gets `None` rather than an `AttributeError`.
- **PME gets no shell-resolved build.** The PME branch has no real-space
  neighbor list at that point and no shell-resolved reciprocal-space
  counterpart, so under `COUL_METHOD="PME"` both attributes stay `None`. This is
  recorded in the `forward` docstring rather than left for a reader to discover
  by getting `None` back.
- **The exception lives in `_slater_koster_pair.py` even though
  `_coulomb_matrix.py` raises it.** Co-locating it with
  `FAngularFormulaSourceError`, `FDerivativeUnsupportedError` and
  `FSpinPolarizationUnsupportedError` makes the project's f support policy
  readable as a single unit. Verified there is no import cycle:
  `_slater_koster_pair` imports only `_bond_integral`.
- **The refusal message names the *blocker*, not just the gap.** It states that
  implementing the f blocks is blocked on shell-resolved charges reaching the
  SCF loop, so a future developer does not spend time writing values that
  nothing can consume or check.

## Deviations from Plan

**1. [Documented divergence, not a rule-triggered fix] Test harness construction
style.** Task 1's `<action>` says to copy `run_with_float64`,
`load_validation_script`, `write_xyz` and `build_structure` from
`tests/test_f_orbital_skf.py`. `run_with_float64` and `write_xyz` were copied
verbatim as instructed. `build_structure` was **not** used: it hardcodes its own
parameter dict and has no way to carry `MAGNETIC_HUBBARD_LDEP`, which is the one
input every parametrised test in this module needs to vary. The module instead
constructs `Constants` and `Structure` directly, exactly as
`tests/test_single_shot_energy.py` (plan 04-01, same phase, same Eu-N system)
already does. `load_validation_script` is consequently unused and was not
copied. No test coverage was lost; the plan's stated intent (copy rather than
import across test modules) is honoured.

**2. [Plan-mandated, outside the frontmatter file list] `src/dftorch/Constants.py`
was modified.** Task 2's `<action>` requires the `MAGNETIC_HUBBARD_LDEP`
docstring update, but the file is absent from the plan's `files_modified`
frontmatter and from Task 2's `<files>` element. The action text wins; the
change is docstring-only.

**3. [Cosmetic] A pre-existing whitespace-only line was briefly disturbed and
restored.** Inserting the `structure.C_sr = None` initialiser above the
`COUL_METHOD` branch stripped trailing whitespace from an adjacent blank
continuation line inside the PME `if (`. It was restored so the diff carries no
incidental line.

No Rule 1/2/3 auto-fixes were needed: nothing broken, missing or blocking was
encountered. No Rule 4 architectural question arose — the one architectural
tension in this plan (shell-resolved matrix versus per-atom consumers) was
already decided by the plan's Scope boundary section.

## Issues Encountered

**The RED observation for the two "refuses" tests is an `ImportError`, not a
wrong-matrix assertion — and the real measurement was taken separately.** With
the exception class absent, `test_shell_resolved_coulomb_refuses_f_system` and
`test_shell_resolved_coulomb_error_explains_the_gap` fail at
`from dftorch._slater_koster_pair import FShellResolvedCoulombUnsupportedError`,
which proves nothing about the defect. The plan's acceptance criterion asks for
both the pre- and post-guard observations to be recorded, so the pre-guard
behaviour was measured directly instead, by calling
`ewald_real_space_vectorized_sr` on Eu-N before any source change: `(6, 6)`,
all finite, row sums `[0.473581, 0, 0, 0, 0.473581, 0]`. That measurement — not
the `ImportError` — is the evidence recorded in the first section above. Plan
04-02 hit the same methodological problem and solved it with an intermediate
behaviour-preserving extraction commit; that technique does not apply here
because there is no pre-existing symbol to extract, so the direct measurement
stands in its place.

**The pre-existing CH4 SCF non-convergence is unchanged and untouched.**
`tests/test_scf.py` still prints `Did not converge` after 25 iterations with a
growing residual. D-13 governs it and CONTEXT.md marks it pre-decided rather
than Phase 4 implementation work. The test does not assert convergence and
passes. No auto-fix attempts were spent on it.

**`ewald_real_space_vectorized_sr` was dead code and is now called once.**
Before this plan it had zero callers in `src/` or `tests/`. It is now reachable
only when `MAGNETIC_HUBBARD_LDEP` is set and `COUL_METHOD` is not PME, and its
output has no consumer. That is the D-14 posture by design, but it is worth a
reader's attention: `structure.C_sr` being populated does not mean any energy
used it.

## Threat Flags

None — no new attack surface. This plan performs no I/O, adds no dependency,
introduces no trust boundary and installs no package.

- **T-04-05** (Tampering: shell-resolved builder returning a mostly-zero matrix
  for f systems, severity **high**) — this is the defect closed.
  `_require_no_f_shell_resolved_coulomb` raises as the function's first
  statement, so no caller can obtain the silently-wrong matrix. Pinned by
  `test_shell_resolved_coulomb_refuses_f_system`. The phase's
  `security_block_on: high` gate is satisfied.
- **T-04-06** (Repudiation: wrong Hubbard U silently substituted) — mitigated.
  `_select_coulomb_hubbard` returns the selection flag alongside the tensor;
  both branches pinned by
  `test_select_coulomb_hubbard_returns_shell_resolved_when_flag_set`.
- **T-04-07** (Information disclosure via exception text) — mitigated. The
  message names only orbital counts, angular block labels and the config key.
  `test_shell_resolved_coulomb_error_explains_the_gap` asserts the message
  contains neither the SKF directory path nor any `.skf` or `.xyz` substring.
- **T-04-SC** (package installs) — stayed inactive; nothing was installed.

## Known Stubs

**One deliberate, plan-mandated boundary — refused rather than stubbed.** The
seven f angular blocks (s-f, f-s, p-f, f-p, d-f, f-d, f-f) of the shell-resolved
Coulomb matrix are not implemented in `_coulomb_matrix.ewald_real_space_
vectorized_sr`. This is **not** a stub in the dangerous sense: no placeholder
value, no zero-fill and no plausible-looking wrong number is returned. The
request raises `FShellResolvedCoulombUnsupportedError` with a message explaining
the gap, the workaround and the blocker. The plan's Scope boundary section
decided this explicitly (item 4), and it is blocked on the shell-resolved-charge
SCF work deferred by D-11 — the phase that turns self-consistency on for f
systems is the one that resolves it.

No other stubs, placeholder values, empty returns, TODO markers or skipped tests
were introduced.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Ready for plans 04-04 and 04-05:**

- Nothing in this plan touches the single-shot energy path. Plan 04-01's Eu-N
  reference energy is asserted unchanged by a test in this module, so the
  separation scan and the SIM-05 loose-band gate measure against exactly the
  definition they were promised.
- The default (`MAGNETIC_HUBBARD_LDEP` unset) f path is completely unaffected:
  `structure.C_sr` stays `None` and no new code executes.

**Carried-forward constraints (unchanged by this plan):**

- f derivatives still raise: `calc_forces`, `ESDriverBatch.calc_forces` and
  analytical stress refuse any 16-orbital system. Plan 04-04 must scan energies,
  never optimize geometry (D-19).
- Batched f H0/S routing still raises; validation must stay on the
  single-system path.
- The `_bond_integral` shared-`R_orb` concern from Phase 3 remains open.

**New constraint introduced here, for the deferred SCF phase:** shell-resolved
electrostatics for f systems is now a *refused* mode rather than a silently
broken one. Turning it on means implementing the seven f angular blocks **and**
threading shell-resolved charges through `energy()` / `SCFx`; the two cannot be
sequenced independently, because the charges are what make the blocks
validatable. Removing `_require_no_f_shell_resolved_coulomb` without doing both
restores the silent-zero defect.

## Self-Check: PASSED

All 5 claimed files exist on disk (`tests/test_shell_resolved_u.py`,
`src/dftorch/_slater_koster_pair.py`, `src/dftorch/_coulomb_matrix.py`,
`src/dftorch/ESDriver.py`, `src/dftorch/Constants.py`) and all 3 claimed commit
hashes (`e37c57b`, `2539976`, `15945bc`) resolve in git log.

---
*Phase: 04-scf-and-reference-simulation-validation*
*Completed: 2026-07-29*

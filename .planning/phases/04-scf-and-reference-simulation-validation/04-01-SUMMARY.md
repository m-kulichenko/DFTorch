---
phase: 04-scf-and-reference-simulation-validation
plan: 01
subsystem: physics-engine
tags: [pytorch, dftb, slater-koster, f-orbitals, scf, tdd, pytest]

# Dependency graph
requires:
  - phase: 03-h0-s-routing-and-f-angular-blocks
    provides: source-locked f angular blocks, so H0/S assemble for 16-orbital pairs instead of raising
  - phase: 02-constants-and-structure-basis-metadata
    provides: 4-shell shell_present / n_orb / D0 metadata that makes HDIM == 20 correct for Eu-N
provides:
  - "ESDriver.forward(do_scf=False): a working single-shot (non-SCC / DFTB1) energy branch"
  - "FSpinPolarizationUnsupportedError + _require_closed_shell_f_system: named refusal for spin-polarized f systems"
  - "A pinned reference energy for Eu-N at 2.655 A that downstream scans measure against"
  - "tests/test_single_shot_energy.py and tests/test_spin_guard.py"
affects: [04-03 shell-resolved Hubbard/Coulomb, 04-04 Eu-N separation scan, 04-05 SIM-05 gate, deferred self-consistent-SCF phase, deferred spin-polarized-f phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Named exception per unsupported mode, raised as the first statement of the guarded method"
    - "Module-level guard function taking dftorch_params explicitly (no self), mirroring _require_f_derivatives"
    - "Charge-dependent corrections pinned to 0.0 rather than left unset, so both driver branches expose the same attribute set"

key-files:
  created:
    - tests/test_single_shot_energy.py
    - tests/test_spin_guard.py
  modified:
    - src/dftorch/ESDriver.py
    - src/dftorch/_slater_koster_pair.py

key-decisions:
  - "The non-SCC energy is band + repulsion with Ecoul exactly 0 (D-11); first-iterate Mulliken charges are recorded as a diagnostic but never enter the energy"
  - "The external-field term goes into a local Hamiltonian only; structure.H0 is never mutated, so energy() receives the unmodified H0"
  - "Solvation, D3 and spin are pinned to 0.0 on the single-shot path because all three are charge-dependent and D-11 has no converged charges"
  - "The spin guard is placed before the neighbor list so it fires for do_scf=False as well as do_scf=True"

patterns-established:
  - "Single-shot branch mirrors _scf.SCFx's initial-density block verbatim (atom_ids, Hdipole symmetrisation, dm_fermi_x, Mulliken extraction) rather than re-deriving it"
  - "Guard messages name only the requested mode and the orbital count, never a filesystem path (threat T-04-02)"

requirements-completed: [SIM-01, SIM-02]

coverage:
  - id: D1
    description: "Eu-N reaches H0/S construction through ESDriver.forward with (20, 20) finite, symmetric matrices, and H0 is not mutated by the new branch"
    requirement: SIM-01
    verification:
      - kind: unit
        ref: "tests/test_single_shot_energy.py#test_eu_n_h0_s_shape_and_symmetry"
        status: pass
    human_judgment: false
  - id: D2
    description: "forward(do_scf=False) populates a finite 0-dim float64 structure.e_tot for the f-containing Eu-N system"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "tests/test_single_shot_energy.py#test_eu_n_single_shot_populates_e_tot"
        status: pass
      - kind: unit
        ref: "tests/test_single_shot_energy.py#test_eu_n_single_shot_reference_energy"
        status: pass
    human_judgment: false
  - id: D3
    description: "The single-shot energy carries no charge-fluctuation Coulomb term: e_coul == 0.0 and e_tot == e_elec_tot + e_repulsion"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "tests/test_single_shot_energy.py#test_eu_n_single_shot_has_no_coulomb_term"
        status: pass
      - kind: unit
        ref: "tests/test_single_shot_energy.py#test_eu_n_single_shot_charges_are_diagnostic_only"
        status: pass
    human_judgment: false
  - id: D4
    description: "UNRESTRICTED=True on a 16-orbital system raises FSpinPolarizationUnsupportedError on both entry points, with a self-explaining message that leaks no paths"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "tests/test_spin_guard.py#test_unrestricted_f_system_raises_on_scf"
        status: pass
      - kind: unit
        ref: "tests/test_spin_guard.py#test_unrestricted_f_system_raises_on_single_shot"
        status: pass
      - kind: unit
        ref: "tests/test_spin_guard.py#test_error_message_names_the_mode_and_the_atom"
        status: pass
    human_judgment: false
  - id: D5
    description: "f-free open-shell calculations and closed-shell f calculations are unaffected by the new guard"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "tests/test_spin_guard.py#test_guard_ignores_f_free_systems"
        status: pass
      - kind: unit
        ref: "tests/test_spin_guard.py#test_restricted_f_system_does_not_raise"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py (16 tests, Phase 3 regression gate)"
        status: pass
    human_judgment: false

# Metrics
duration: 12min
completed: 2026-07-29
status: complete
---

# Phase 4 Plan 01: Single-Shot Energy and Spin Guard Summary

**`ESDriver.forward(do_scf=False)` now computes a real non-SCC DFTB1 energy for f systems — Eu-N at 2.655 Å returns `e_tot = -17.510444238744924` eV with `e_coul` exactly 0 — and any spin-polarized request on a 16-orbital atom is refused with `FSpinPolarizationUnsupportedError` instead of a bare `IndexError` or silence.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-29T18:46:18Z
- **Completed:** 2026-07-29T18:58:15Z
- **Tasks:** 2 (both TDD, 4 commits)
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- **The missing `else:` exists.** `ESDriver.forward` had an `if do_scf:` at line 328 whose body ran to the end of the method. `do_scf=False` therefore built H0, S, Z and `e_repulsion` and then returned `None` with `structure.e_tot` never assigned. The branch now performs one non-self-consistent diagonalization via `dm_fermi_x` at the reference charge state and assigns `e_tot = e_elec_tot + e_repulsion`.
- **The energy definition is pinned and reproducible.** The reference values recorded at plan time were reproduced exactly by the implementation, to the last digit (see below). `tests/test_single_shot_energy.py::test_eu_n_single_shot_reference_energy` locks them in with a docstring stating that a future mismatch means the definition drifted, not that the test went stale.
- **Spin polarization fails loudly at the door.** `_require_closed_shell_f_system` is the first statement of `forward()`, so it fires before the neighbor list and covers both entry points. Previously `UNRESTRICTED=True` on Eu-N gave `IndexError: index 4 is out of bounds for dimension 0 with size 4` from deep inside `scf_x_os` when `do_scf=True`, and gave nothing at all when `do_scf=False`.
- **The blast radius is bounded.** f-free open-shell calculations return early from the guard, and the Phase 3 regression gate (`tests/test_f_orbital_skf.py`, 16 tests) is still green.

## Reference energy observed (requested by the plan's `<output>`)

Eu-N diatomic at 2.655 Å, `T_ELECTRONIC=1000.0`, `RCUT_ELECTRONIC=10.0`, `RCUT_REPULSIVE=6.0`, `COUL_METHOD="FULL"`, `CHARGE=0`, SKF set `tests/f_orbital_data/`:

| Quantity | Observed | Plan-time value | Match |
|---|---|---|---|
| `e_tot` | `-17.510444238744924` | `-17.510444238744924` | exact |
| `e_band0` | `-16.948081509141794` | `-16.948081509141794` | exact |
| `e_coul` | `0` | `0.0` | exact |
| `e_dipole` | `-0.0` | `-0.0` | exact |
| `e_entropy` | `-0.7212721085402877` | `-0.7212721085402877` | exact |
| `e_repulsion` | `0.15890937893715806` | `0.15890937893715806` | exact |
| `q` | `[-2.702732020796069, +2.702732020796080]` | `[-2.70273, +2.70273]` | exact |

Every measured fact in the plan's "Verified facts" table was independently reconfirmed on this checkout before implementation began: `shell_present[Eu] == [T,T,T,T]`, `shell_present[N] == [T,T,F,F]` (N has **s and p only** — `HDIM == 20`, not 25), `n_orb` 16/4, `tore` 9.0/5.0, `Nocc == 7`, `D0` 1-D of shape `(20,)` summing to 7.0.

## Attributes the SCF branch sets that the single-shot branch deliberately leaves at zero

Requested by the plan's `<output>`. All of these are **charge-dependent**, and D-11 provides no converged charges, so evaluating them would produce a number with no defensible meaning:

| Attribute | Single-shot value | Why |
|---|---|---|
| `e_coul` | `0` (from `energy()`'s `C is None and dq_p1 is None` arm) | The load-bearing exclusion — see "Decisions Made" |
| `e_spin` | `0.0` | Spin is refused outright for f systems (D-12) |
| `e_gb`, `e_sasa`, `e_solv` | `0.0` | GBSA/ALPB solvation is driven by Mulliken charges |
| `e_d3` | `0.0` | D3(BJ) is not evaluated on this path |
| `gbsa`, `dftd3` | `None` | The corresponding objects are never constructed |

These are pinned rather than left unset so a caller inspecting the structure after a single-shot run sees the same attribute set the SCF branch produces, instead of an `AttributeError`.

**Honest gap (not a stub, but worth recording):** the single-shot branch does *not* set every attribute `SCFx` sets. `structure.H`, `structure.Hcoul`, `structure.Hdipole`, `structure.KK`, `structure.Q`, `structure.e_coul_tmp`, `structure.f_coul`, `structure.dq_p1` and `structure.stress_coulomb` remain unset after `do_scf=False`, because they are SCF-loop bookkeeping with no single-shot analogue (there is no mixing matrix, no charge response, and no Coulomb contribution to a stress). `structure.D`, `structure.e`, `structure.f`, `structure.mu0`, `structure.q` and the six `energy()` outputs *are* set. `structure.C` is set by the Coulomb-matrix construction that runs above the branch, but is not consumed by it.

## Task Commits

1. **Task 1: End-to-end single-shot energy for Eu-N (tracer, TDD)**
   - `5e480f0` (test) — RED: 5 tests, 4 failing
   - `d6c6e1f` (feat) — GREEN: the `else:` branch in `ESDriver.forward`
2. **Task 2: Refuse spin-polarized f systems (TDD)**
   - `448d05b` (test) — RED: 5 tests, 4 failing
   - `e2276c9` (feat) — GREEN: exception, message constant, guard, call site

No REFACTOR commits were needed; both implementations were clean on first pass.

**TDD gate compliance:** both tasks show `test(...)` → `feat(...)` in git log order, in that order, with the RED commit verified failing before the GREEN commit was authored.

## Files Created/Modified

- `src/dftorch/ESDriver.py` — added `import contextlib` and the `dm_fermi_x` import; added `_require_closed_shell_f_system` next to `_require_f_derivatives`; called it as the first statement of `forward()`; added the `else:` branch to `if do_scf:`; rewrote the `do_scf` parameter docstring to describe implemented behaviour rather than aspiration.
- `src/dftorch/_slater_koster_pair.py` — added `FSpinPolarizationUnsupportedError` and `F_SPIN_POLARIZATION_UNSUPPORTED_MESSAGE` directly after `F_ANGULAR_DERIVATIVES_AVAILABLE`, following the `FDerivativeUnsupportedError` template exactly.
- `tests/test_single_shot_energy.py` — new, 5 tests covering SIM-01 and the supported branch of SIM-02.
- `tests/test_spin_guard.py` — new, 5 tests covering the unsupported branch of SIM-02.

## Decisions Made

- **`e_coul` is exactly 0, and this is a definition rather than an omission (D-11).** `energy()` is called with `C=None` and `dq_p1=None` so it takes its `Ecoul = 0` arm. The alternative — building a Coulomb term out of the first-iterate Mulliken charges — is not a single-shot energy but one broken SCF step: those charges are `q_Eu ≈ -2.70` here and drift to `-2.99` at 4 Å, so the atoms are not even neutral at separation. The plan measured that this destroys the binding well entirely (energy rises monotonically past 2.0 Å, no interior minimum), which would make the SIM-05 gate in plan 04-04/04-05 unsatisfiable. The implementation does not "improve" on this.
- **`structure.H0` is never mutated.** The external-field (dipole) term is symmetrised against `S` exactly as `SCFx` does, but added to a *local* variable. `energy()` then receives the unmodified `structure.H0`, which is what makes the recorded `e_band0` meaningful. `test_eu_n_h0_s_shape_and_symmetry` pins this by running the system a second time with a non-zero `ELECTRIC_FIELD` and asserting the stored `H0` is byte-identical — `H0` is field-independent, so any folding-in would show up immediately.
- **The guard precedes the neighbor list.** Placing `_require_closed_shell_f_system` as the first statement of `forward()` (rather than inside the `do_scf` branches) is what makes it cover `do_scf=False`, and it means an unsupported request costs nothing before it is refused.
- **The guard takes `dftorch_params` as an argument.** It is a module-level function with no `self`. (`04-PATTERNS.md` sketched it reading `self.dftorch_params` from module scope; that sketch does not run and was not copied — the plan flagged this in advance.)

## Deviations from Plan

None — plan executed exactly as written.

Two small elaborations, both explicitly required by the plan's own acceptance criteria rather than added scope:

- Acceptance criterion 5 of Task 1 demands proof that `structure.H0` is not mutated. With the default zero field the dipole term vanishes and mutation would be undetectable, so the check was folded into `test_eu_n_h0_s_shape_and_symmetry` as a second run with a non-zero `ELECTRIC_FIELD`. This keeps the module at exactly 5 tests, as the acceptance criteria require.
- `test_error_message_names_the_mode_and_the_atom` additionally asserts that the message contains no filesystem path, implementing the `mitigate` disposition on threat **T-04-02** (the exception text must not interpolate `SKFPATH` or `FILENAME`).

## Issues Encountered

**`tests/test_scf.py::test_energy_smoke_import_and_call[cpu]` is RED at the end of this plan — expected and out of scope.** The failure is `RuntimeError: expected scalar type Double but found Float` from `torch/functional.py:383` (einsum), i.e. the PME `.float()` dtype bug at `PME_torch.py:261` assigned to plan **04-02** by decisions D-17/D-18. It is f-free, pre-existing, on a CH4 + `COUL_METHOD="PME"` path this plan does not touch, and the plan's `<verification>` block predicts it verbatim. Full suite at completion: **34 passed, 1 failed** (35 total), with the single failure being exactly this pre-existing one. No test regressed.

No auto-fix attempts were spent on it: it belongs to a sibling plan in the same wave, and fixing it here would collide with 04-02.

## Threat Flags

None. This plan added one arithmetic code path over tensors already in memory plus one exception class; it performs no I/O, adds no dependency, and introduces no new trust boundary. Threat **T-04-02** (exception-message disclosure) was mitigated and is asserted by a test. Threat **T-04-SC** (package installs) remained inactive — no packages were installed.

## Known Stubs

None. Every code path introduced is fully implemented; no placeholder values, empty returns, or TODO markers were added.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Ready for the rest of Wave 1 and for plans 04-03 / 04-04:**

- Plan 04-04's Eu-N separation scan can now call `driver(struct, const, do_scf=False)` in a loop and read `structure.e_tot`. The energy definition it will record numbers against is pinned by `test_eu_n_single_shot_reference_energy`.
- Plan 04-03's shell-resolved Hubbard/Coulomb work is unblocked and unaffected — this plan touched neither `Structure.Hubbard_U_sr` nor `coulomb_matrix_vectorized`.
- Plan 04-02 must still repair the PME dtype bug; `tests/test_scf.py` stays RED until it lands.

**Carried-forward constraints (unchanged by this plan):**

- f derivatives remain unimplemented: `calc_forces`, `ESDriverBatch.calc_forces` and analytical stress still raise `FDerivativeUnsupportedError` for any 16-orbital system. Plan 04-04 must scan energies, never optimize geometry (D-19).
- Batched f H0/S routing still raises; reference validation must stay on the single-system path.
- The `_bond_integral` single-`R_orb` grid concern (deferred item 3 from Phase 3) is untested by this plan. The single-shot energy reproduces exactly, which says the path is self-consistent, but says nothing about whether the shared radial grid is correct across mixed-grid pairs. It remains open for whoever compares against an external reference.

## Self-Check: PASSED

All 5 claimed files exist on disk and all 4 claimed commit hashes resolve in git log.

---
*Phase: 04-scf-and-reference-simulation-validation*
*Completed: 2026-07-29*

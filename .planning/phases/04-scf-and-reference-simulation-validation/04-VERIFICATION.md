---
phase: 04-scf-and-reference-simulation-validation
verified: 2026-07-29T23:30:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: SCF and Reference Simulation Validation Verification Report

**Phase Goal:** Supported f-orbital single-system calculations reach SCF/reference validation
with explicit blockers for unsupported modes.
**Verified:** 2026-07-29
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A minimal f-containing system runs through H0/S construction without shape or routing failures | ✓ VERIFIED | Independently re-executed `ESDriver.forward` for Eu-N: `structure.H0.shape == (20, 20)`, `structure.S.shape == (20, 20)`, both finite, `H0` symmetric to `atol=1e-10`. `tests/test_single_shot_energy.py::test_eu_n_h0_s_shape_and_symmetry` passes. |
| 2 | A supported f-containing system runs through closed-shell SCF/energy calc, or receives an explicit unsupported-mode error | ✓ VERIFIED | `forward(do_scf=False)` on Eu-N independently reproduced `e_tot == -17.510444238744924` eV (exact match to plan-recorded value). `UNRESTRICTED=True` on the same system independently reproduced `FSpinPolarizationUnsupportedError` (a named `NotImplementedError` subclass), on both `do_scf=True` and `do_scf=False` entry points (`ESDriver.py:209-211`, called as the first statement of `forward`). |
| 3 | Shell-resolved charge, Hubbard, and Coulomb data include f-shell dimensions wherever the supported SCF path requires them | ✓ VERIFIED | Independently re-executed: `Hubbard_U_sr == [5.7144, 5.1702, 6.8028, 13.6057, 13.3336, 13.3336]` (6 entries, index 3 = Eu f U), `n_shells_per_atom == [4, 2]`, `el_per_shell[3] == 7.0` (Eu f electrons), `D0.sum() == 7.0` with the 7 f-AO entries at 0.5. Requesting the shell-resolved Coulomb matrix for Eu-N (`MAGNETIC_HUBBARD_LDEP=True`) independently reproduced `FShellResolvedCoulombUnsupportedError` rather than a silently-zero matrix. |
| 4 | The isolated Eu-N diatomic validation case has documented geometry, parameter files, observables, units, and tolerances | ✓ VERIFIED | `tests/f_orbital_data/README-EU-N-CASE.md` exists with all five required `##` headings (Geometry, SKF parameter files, Observable, Units, Tolerance), the asymmetric failure rule, provenance, the 21-row reference curve, and the sampling-risk/manual-only-check sections. |
| 5 | The supported f-orbital Eu-N energy scan locates a minimum within the agreed loose sanity band (~10-20%) of the known Eu-N separation | ✓ VERIFIED | `tests/test_eu_n_scan.py::test_eu_n_energy_scan_locates_interior_minimum` independently re-run and passes: minimum at 2.40 Å (grid index 8 of 20), strictly interior, inside `[2.124, 3.186]` Å (2.655 Å ±20%, derived arithmetically — `EU_N_BAND_MIN/MAX_ANGSTROM = EU_N_TARGET_ANGSTROM * (1 ∓ EU_N_BAND_FRACTION)`, no literal band edges). Additionally discharged by a real human sign-off (04-05-SUMMARY.md) with independently-reverified curve values. |

**Score:** 5/5 ROADMAP success criteria verified.

### Additional load-bearing truths from the verification brief

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | D-11 not violated: no self-consistent SCF iteration added; `e_coul == 0.0` on the single-shot path | ✓ VERIFIED | `ESDriver.py:826-841`: `energy(...)` called with `C=None, dq_p1=None`, selecting `_energy.py:131`'s `Ecoul = 0` arm. Independently confirmed `structure.e_coul == 0` at runtime. `tests/test_single_shot_energy.py::test_eu_n_single_shot_has_no_coulomb_term` pins this. |
| 7 | D-23: the shell-resolved Hubbard U gate is `MAGNETIC_HUBBARD_LDEP`/`magnetic_hubbard_ldep`, not a new `SHELL_RESOLVED` config flag | ✓ VERIFIED | `grep -rn "SHELL_RESOLVED" src tests` shows only the message constant `F_SHELL_RESOLVED_COULOMB_UNSUPPORTED_MESSAGE`, the exception class name, and explanatory docstring/comment prose — no `dftorch_params.get("SHELL_RESOLVED")` or `self.shell_resolved` anywhere. `_select_coulomb_hubbard` (`ESDriver.py:107-130`) and `_require_no_f_shell_resolved_coulomb` both gate on `const.magnetic_hubbard_ldep` / `getattr(const, "magnetic_hubbard_ldep", False)`. |
| 8 | D-24: target 2.655 Å, band ±20% → [2.124, 3.186], band derived arithmetically, asymmetric failure rule encoded | ✓ VERIFIED | `tests/test_eu_n_scan.py:99-102`: `EU_N_BAND_MIN_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 - EU_N_BAND_FRACTION)` etc. — no literal `2.124`/`3.186` as source of truth (they only appear in comments/docs/assertions of the *derived* values). `test_eu_n_energy_scan_locates_interior_minimum` (`tests/test_eu_n_scan.py:345-`) has three distinct branches in its assertion message: no-interior-minimum and long-of-band are both "GENUINE RED FLAG"; short-of-band explicitly states "NOT evidence that the f implementation is broken" and "ESCALATE TO HUMAN JUDGEMENT". |
| 9 | D-19: validation is an energy scan; no optimizer, no `calc_forces`/`calc_stress` in the scan path | ✓ VERIFIED | `grep -n "calc_forces\|calc_stress\|Optimizer" tests/test_eu_n_scan.py` finds only docstring/comment references stating these are *not* called; `single_shot_energy` only calls `driver(structure, const, do_scf=False)`. |
| 10 | SIM-05's human gate genuinely discharged by a person | ✓ VERIFIED | `04-05-SUMMARY.md` records a real disposition (APPROVED), independently re-measured values at review time (minimum 2.40 Å, E_tot -17.64144156 eV, index 8/20), a two-panel plot generated from live code, and confirms `EU_N_TARGET_ANGSTROM`/`EU_N_BAND_FRACTION` unchanged after review (`git diff --stat -- tests/` empty at sign-off, independently reconfirmed here: current values are still 2.655 / 0.20). |
| 11 | Scope fences held: no Ga work, no batch/MD/SEDACS/ML-SK work, no f derivative/force/stress implementation | ✓ VERIFIED | `git log` on the 5 files touched by Phase 4 commits (`ESDriver.py`, `_slater_koster_pair.py`, `_coulomb_matrix.py`, `PME_torch.py`, `_nearestneighborlist.py`) shows exactly the 8 Phase-4-tagged commits (`d6c6e1f`, `e2276c9`, `da0574d`, `67773a8`, `6ed3cd7`, `2539976`, `15945bc`) and nothing Ga/batch/MD/SEDACS/ML-SK-related. `calc_forces` still calls `_require_f_derivatives` first (unchanged), so f-derivative refusal is intact. |

**Score:** 11/11 verified truths (5 ROADMAP + 6 brief-mandated load-bearing checks).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/dftorch/ESDriver.py` (`else:` branch, guards, selection helper) | single-shot energy path, spin guard, Coulomb-U selection | ✓ VERIFIED | `else:` clause at `ESDriver.py:758` matching `if do_scf:` at `ESDriver.py:471`; `_require_closed_shell_f_system` at `ESDriver.py:67`; `_select_coulomb_hubbard` at `ESDriver.py:107` |
| `src/dftorch/_slater_koster_pair.py` (exception taxonomy) | `FSpinPolarizationUnsupportedError`, `FShellResolvedCoulombUnsupportedError` + message constants | ✓ VERIFIED | Both classes present at lines 259 and 289, both `NotImplementedError` subclasses, both with `Final[str]` message constants |
| `src/dftorch/_coulomb_matrix.py` (f-Coulomb guard) | `_require_no_f_shell_resolved_coulomb`, called first in `ewald_real_space_vectorized_sr` | ✓ VERIFIED | Present at line 656; called as the function's first statement at line 761 |
| `src/dftorch/ewald_pme/PME_torch.py` (dtype fix) | mask cast to operand dtype, not hardcoded `.float()` | ✓ VERIFIED | `g_mask = (m_2 > 0).to(dtype=E_G.dtype)` at line 264, with an explanatory comment |
| `src/dftorch/_nearestneighborlist.py` (dtype fix) | `_min_image_sort_key` helper accumulating at distance precision | ✓ VERIFIED | Helper defined at line 23, casts `ri`/`j_all` to `d2_all.dtype`; called at line 214 |
| `tests/test_single_shot_energy.py`, `test_spin_guard.py`, `test_shell_resolved_u.py`, `test_dtype_contract.py`, `test_eu_n_scan.py` | new test modules | ✓ VERIFIED | All 5 exist; collected counts 5/5/27/4/6 = 47, matching each plan's claimed count |
| `tests/f_orbital_data/README-EU-N-CASE.md` | SIM-04 case document | ✓ VERIFIED | Exists, 322 lines, all 5 required `##` sections present, 21-row reference curve present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `ESDriver.forward` `if do_scf:` | new `else:` branch | matching indentation, assigns `structure.e_tot` | ✓ WIRED | Confirmed by reading source and independent execution |
| single-shot branch | `_energy.energy()` | `C=None`, `dq_p1=None` | ✓ WIRED | `_energy.py:131` `Ecoul = 0` arm confirmed taken (`structure.e_coul == 0` at runtime) |
| `_require_closed_shell_f_system` | `ESDriver.forward` | called as first statement, before `COULOMB_CUTOFF` default | ✓ WIRED | Confirmed at `ESDriver.py:209-211`, fires for both `do_scf` values (independently reproduced) |
| `_select_coulomb_hubbard` | Coulomb-path Hubbard U | `const.magnetic_hubbard_ldep` gate | ✓ WIRED | Confirmed at `ESDriver.py:392`; `structure.C` build (per-atom) is untouched — `git diff` invariant asserted in 04-03-SUMMARY and independently spot-checked in source |
| `ewald_real_space_vectorized_sr` | `_require_no_f_shell_resolved_coulomb` | first statement of the function | ✓ WIRED | Confirmed at `_coulomb_matrix.py:761`; independently reproduced the raise for Eu-N with the flag set |
| `tests/test_eu_n_scan.py` | `ESDriver.forward(do_scf=False)` | `single_shot_energy()` helper | ✓ WIRED | Confirmed via source; no `calc_forces`/`calc_stress`/optimizer calls |

### Behavioral Spot-Checks (independently executed, not from SUMMARY claims)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite green | `uv run pytest -q` | exit 0, 72 dots, 0 `F`/`E` markers | ✓ PASS |
| Suite composition matches claim | `uv run pytest --collect-only -q` | 4+6+16+3+1+1+1+2+1+27+5+5 = 72 | ✓ PASS |
| Eu-N single-shot energy reproduces plan value | direct script invocation of `ESDriver.forward(do_scf=False)` | `e_tot = -17.510444238744924`, `e_coul = 0`, `H0` (20,20) finite/symmetric | ✓ PASS |
| Spin guard fires with correct exception type | direct script invocation with `UNRESTRICTED=True` | `FSpinPolarizationUnsupportedError` raised (verified via `except FSpinPolarizationUnsupportedError` clause, not string matching) | ✓ PASS |
| Shell-resolved Hubbard/charge data carries f dimension | direct script inspection of `structure.Hubbard_U_sr`, `el_per_shell`, `D0` | matches plan/summary tables exactly | ✓ PASS |
| Shell-resolved f-Coulomb refusal fires | direct script invocation with `MAGNETIC_HUBBARD_LDEP=True` on Eu-N | `FShellResolvedCoulombUnsupportedError` raised | ✓ PASS |
| SIM-05 gate test passes standalone | `uv run pytest tests/test_eu_n_scan.py::test_eu_n_energy_scan_locates_interior_minimum -v` | 1 passed | ✓ PASS |
| No `SHELL_RESOLVED` config-key usage | `grep -rn "SHELL_RESOLVED" src tests` (filtered for actual key reads) | zero hits besides the message constant/class name/prose | ✓ PASS |
| Band derived arithmetically | source read of `tests/test_eu_n_scan.py:99-102` | expressions over `EU_N_TARGET_ANGSTROM`/`EU_N_BAND_FRACTION`, no literal edges | ✓ PASS |
| Debt-marker scan on Phase-4-touched files | `grep -n "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER"` on the 5 modified `src/` files + 5 new test files | one `TODO` at `ESDriver.py:321`, predates Phase 4 (commit `16b143c`, unrelated DFTB3/PME feature) | ✓ PASS (pre-existing, not a Phase 4 marker) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| SIM-01 | 04-01 | Minimal f system reaches H0/S without shape/routing failure | ✓ SATISFIED | `test_eu_n_h0_s_shape_and_symmetry`, independently reproduced |
| SIM-02 | 04-01, 04-02 | Closed-shell energy or explicit unsupported-mode error | ✓ SATISFIED | Both branches independently reproduced; smoke gate (`test_scf.py`) green |
| SIM-03 | 04-03 | Shell-resolved charge/Hubbard/Coulomb carry f dimensions | ✓ SATISFIED | Independently reproduced `Hubbard_U_sr`, `el_per_shell`, `D0`, and the f-Coulomb refusal |
| SIM-04 | 04-04 | Eu-N case documented (geometry, files, observables, units, tolerances) | ✓ SATISFIED | `README-EU-N-CASE.md` verified complete |
| SIM-05 | 04-04, 04-05 | Scan locates minimum in loose band | ✓ SATISFIED | Automated gate + genuine human sign-off, both independently checked |

No orphaned requirements — `.planning/REQUIREMENTS.md` maps only SIM-01..SIM-05 to Phase 4, and all five appear in at least one plan's `requirements:` frontmatter.

### Anti-Patterns Found

None blocking. One pre-existing `TODO` at `ESDriver.py:321` (PME/DFTB3 off-diagonal support) predates Phase 4 by multiple unrelated commits (introduced in `16b143c`, well before the Phase 4 commit range) and is outside the files this phase's plans claim to have authored logic in — it sits in the pre-existing PME branch of `forward`, untouched by any Phase 4 diff. Not a Phase 4 debt marker.

Pre-existing uncommitted changes to `src/dftorch/_bond_integral.py`, `src/dftorch/_io.py`, and `src/dftorch.egg-info/*` predate Phase 4 (last commit touching `_bond_integral.py` is `a8fcd2e`, dated 2026-07-28, Phase 3 work) and are not part of any Phase 4 commit — confirmed via `git log` on those files showing no Phase-4-tagged commit touches them.

### Human Verification Required

None. SIM-05's human checkpoint (04-05, `checkpoint:human-verify`, gate: blocking) was already discharged with a recorded, independently-corroborated verdict (APPROVED) before this verification ran — see `04-05-SUMMARY.md` and the cross-checks above. No further human action is needed to close this phase.

### Gaps Summary

None. All 5 ROADMAP success criteria and all 5 requirement IDs (SIM-01..SIM-05) are independently verified against the running codebase, not merely asserted by SUMMARY files. The full test suite (72 tests) passes with exit code 0, reproduced directly rather than trusted from prior runs. The phase's own honesty caveats — the ±20% band absorbing up to a 15% f-block error (D-20/D-22), single-shot ≠ SCF (D-11), closed-shell treatment of open-shell Eu 4f⁷ (D-12), and validated-but-unconsumed shell-resolved f plumbing (D-14) — are documented in `04-VALIDATION.md`, `README-EU-N-CASE.md`, and `04-05-SUMMARY.md`, and are accepted, recorded limitations rather than unexamined gaps. `04-VALIDATION.md` correctly keeps `status: draft` / `nyquist_compliant: false` per its own lifecycle contract (owned by `validate-phase`, not an executor) — this is expected, not a defect.

---

_Verified: 2026-07-29_
_Verifier: Claude (gsd-verifier)_

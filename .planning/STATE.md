---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 05
status: completed
stopped_at: Completed 05-06-PLAN.md (D-04 orbital-count audit); next 05-07
last_updated: "2026-08-03T21:58:33.333Z"
last_activity: 2026-08-03
last_activity_desc: Phase 05 marked complete
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 17
  completed_plans: 16
current_phase_name: regression-safety-and-support-policy-cleanup
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-17)

**Core value:** DFTorch can run scientifically valid f-orbital DFTB simulations without changing numerical results for existing simple-format calculations.
**Current focus:** Phase 05 — regression-safety-and-support-policy-cleanup

## Current Position

Phase: 05 — COMPLETE
Plan: 6 of 8
Status: Phase 05 complete
Last activity: 2026-08-03 — Phase 05 marked complete

Progress: [████████░░] 82%

## Performance Metrics

**Velocity:**

- Total plans completed: 3
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. SKF Canonicalization and Spline Validation | 0/TBD | - | - |
| 2. Constants and Structure Basis Metadata | 0/TBD | - | - |
| 3. H0/S Routing and f Angular Blocks | 0/TBD | - | - |
| 4. SCF and Reference Simulation Validation | 0/TBD | - | - |
| 5. Regression Safety and Support Policy Cleanup | 0/TBD | - | - |
| 01 | 1 | - | - |
| 02 | 1 | - | - |
| 03 | 1 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 02 P01 | 6min | 3 tasks | 2 files |
| Phase 03 P01 | 95min | 4 tasks | 7 files |
| Phase 04 P01 | 12min | 2 tasks | 4 files |
| Phase 04 P02 | 7min | 2 tasks | 3 files |
| Phase 04 P04 | 22min | 2 tasks | 2 files |
| Phase 04 P05 | 5min | 1 tasks | 2 files |
| Phase 05 P02 | 35 min | 2 tasks | 2 files |
| Phase 05 P01 | ~55 min | 2 tasks | 8 files |
| Phase 05 P03 | 10 min | 2 tasks | 2 files |
| Phase 05 P04 | 85 min | 3 tasks | 8 files |
| Phase 05 P06 | 47 min | 2 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Use horizontal technical phases in dependency order for this scientific Python feature.
- [Roadmap]: Keep f-orbital work local to GSD planning artifacts; do not create AGENTS.md or root instruction files.
- [Phase ?]: Phase 2 metadata gate exposed no Constants.py or Structure.py production drift.
- [Phase ?]: Simple-format f-free metadata regression uses synthetic s-only plus parsed sp/spd fixtures.
- [Phase ?]: Phase 1's 40-channel SKF migration silently broke f-free H0/S: _slater_koster_pair.py still used legacy channel+SH_shift*10 offsets, zeroing all off-diagonal H0 couplings and making S read Hamiltonian channels. Fixed via named channel lookup in edf73e4.
- [Phase ?]: 16-orbital pairs are now routed into Slater-Koster assembly behind an explicit FAngularFormulaSourceError boundary rather than being silently dropped by the 1/4/9 orbital masks.
- [Phase ?]: Phase 3: f angular tables source-locked to Takegahara/Aoki/Yanase, J. Phys. C 13 (1980) 583, DOI 10.1088/0022-3719/13/4/016. Sharma PRB 19 2813 rejected as primary (incomplete table, non-cubic orbitals, misprint per Takegahara p588).
- [Phase ?]: Phase 3: the paper's cubic-harmonic f basis is identical to Structure.AO_LABEL_TEMPLATE, so the adapter is the pure permutation (1,2,3,4,5,6,0) with all +1 signs -- no basis rotation.
- [Phase ?]: Phase 3: the transcription gate is the paper's own orthogonality relation (eqs 14-15), not golden numbers -- each channel coefficient matrix must be an orthogonal projector and cross-shell blocks must project onto the same f subspaces.
- [Phase ?]: Phase 3: f derivatives deferred (D-09). dH0/dS stay exactly zero in f blocks and FDerivativeUnsupportedError fires from both calc_forces entry points and analytical stress.
- [Phase ?]: Phase 4 D-11: the single-shot (non-SCC) energy is band + repulsion with e_coul exactly 0; first-iterate Mulliken charges (q_Eu ~ -2.7, drifting to -2.99 at 4 A) are diagnostic only and destroy the binding curve if fed into the Coulomb energy.
- [Phase ?]: Phase 4 D-12: spin-polarized f systems raise FSpinPolarizationUnsupportedError from the first statement of ESDriver.forward, so it fires for do_scf=False as well as do_scf=True; f-free open-shell paths return early and are unaffected.
- [Phase ?]: Phase 4: the single-shot branch never mutates structure.H0 -- the external-field term goes into a local Hamiltonian so energy() receives the unmodified H0; pinned by a non-zero-ELECTRIC_FIELD run in test_eu_n_h0_s_shape_and_symmetry.
- [Phase ?]: Phase 4 D-17/D-18: the PME k-space stress mask now takes its dtype from E_G.dtype at runtime instead of a hardcoded .float(); this turned tests/test_scf.py green (was RuntimeError: expected scalar type Double but found Float).
- [Phase ?]: Phase 4 D-18: the min-image neighbor sort key is extracted to _min_image_sort_key and accumulated at d2_all.dtype; the 32-bit version collided for distinct (i,j) above ~4200 atoms (3635495168.0 == 3635495168.0), silently keeping the wrong periodic image. 04-RESEARCH.md Finding 3's BENIGN verdict is overturned.
- [Phase ?]: Phase 4: the D-18 dtype audit is closed -- exactly three float-width sites exist in src/dftorch/ (PME_torch.py:261 fixed, _energy.py:12 correct as a dtype comparison, _nearestneighborlist.py:178 fixed); all .astype() hits are int/bool and out of class.
- [Phase 04]: Phase 4 SIM-05: the Eu-N single-shot energy scan locates its minimum at 2.40 A (grid index 8 of 20, scan 1.60-3.60 A step 0.10), strictly interior and inside the 2.655 A +/-20% band [2.124, 3.186]. All 21 energies reproduce the plan-time reference curve exactly. — 2.40 A sits below the 2.655 A target in exactly the direction D-24 predicted (the target is a mean over dative Eu-N bonds in crowded 8-9-coordinate Eu-Bp complexes, so an isolated diatomic should be shorter) and coincides with the independent bulk rock-salt EuN reference of ~2.45-2.5 A. Two reference points agree. Still a smoke test: the +/-20% band absorbs a 15% f-block error.
- [Phase 04]: Phase 4 D-24: the Eu-N tolerance band is derived as EU_N_TARGET_ANGSTROM * (1 -/+ EU_N_BAND_FRACTION); neither band edge is a numeric literal, and EU_N_BAND_FRACTION >= 0.10 is asserted so D-20's looseness floor is mechanical. — Threat T-04-08. The mis-stated '2.4 +/- 0.2 A' is +/-8%, not +/-20%, and substituting it would silently tighten the gate past what single-shot non-SCC energy on a closed-shell 4f7 ion can support. Both that substitution and a quiet tightening to +/-5% were verified to fail test_tolerance_band_is_a_fraction_of_the_target.
- [Phase 04]: Phase 4: the Eu-N scan reuses one Constants instance across all 21 points, verified bit-identical to per-point construction (max abs diff 0.000e+00 at every separation). — Constants reads the geometry file only for the species list, which is (Eu, N) at every scan point. The plan's literal per-call construction measured 46.4 s here versus the plan's claimed 16 s, blowing the 30-second feedback budget the plan itself names as the reason to compute the curve once. Reuse brings the module to 16.0 s. Bit-identical output also rules out cross-point contamination of the shared object.
- [Phase 04]: Phase 4: tests/test_eu_n_scan.py is pure ASCII, and its interior-minimum assertion has three distinct branches (no interior minimum / short of band / long of band) each carrying the located separation, the band edges and the full curve. — Threat T-04-09. Em dashes rendered as replacement characters on a cp1252 console, defeating the requirement that a failure be diagnosable from pytest output alone. All three D-24 branches were forced by mutation and confirmed to emit their own message; the short-of-band branch states that a short minimum is NOT evidence the f implementation is broken.
- [Phase ?]: Phase 4 SIM-05 human sign-off (plan 04-05): APPROVED. A human read a two-panel plot of the Eu-N binding curve recomputed from live code (not the recorded table) and judged it a SINGLE CLEAN WELL -- not monotonic, not multi-well, not flat within noise. Verified independently at review time: minimum 2.40 A, E_tot -17.64144156 eV, index 8 of 20, strictly interior, inside [2.124, 3.186] A, strictly monotone in and out, unique on the grid; suite 72 passed / 0 failed. Disposition is approved rather than widen-band, so NO new CONTEXT.md decision is required and EU_N_TARGET_ANGSTROM / EU_N_BAND_FRACTION remain untouched (threat T-04-11, confirmed 2.655 / 0.20 after review).
- [Phase ?]: Phase 4 D-24 at sign-off: the short-side escalation branch was NOT triggered as a failure -- 2.40 A is inside the band. It was noted that 2.40 A sits below the 2.655 A target, the direction D-24 anticipates (the target is a mean over dative Eu-N bonds in crowded 8-9-coordinate Eu-Bp complexes, the case is an isolated diatomic), and coincides with the independent bulk rock-salt EuN reference of ~2.45-2.50 A. Two reference points agree. 04-VALIDATION.md's second manual-only row is therefore discharged as not-triggered, which is a legitimate disposition; it is NOT claimed the branch was exercised.
- [Phase ?]: Phase 4 sign-off, one concern surfaced and explicitly accepted: the Eu-N curve is still climbing at 3.60 A and does not reach a dissociation asymptote within the scanned range. The human was offered approve-as-is / approve-but-carry-forward / hold-and-extend-to-6-8-A and chose approve -- expected for single-shot non-SCC energy over this range. Recorded as reviewed-and-accepted, NOT as an unexamined gap and NOT as a carried-forward open concern. Extending the scan becomes meaningful once self-consistency lands.
- [Phase ?]: Phase 4 sign-off: the four limitations 04-VALIDATION.md records as unsampled are acknowledged and accepted, each a direct consequence of an earlier locked decision rather than a new discovery -- the +/-20% band absorbs a 15% f-block error (D-20/D-22, which forbid tightening), single-shot is not SCF so charge-self-consistency bugs cannot surface by construction (D-11), Eu 4f7 is genuinely open-shell but treated closed-shell (D-12, which defers spin and makes it raise), and the shell-resolved f plumbing is validated but never consumed by the single-shot path (D-14's deliberate de-risking).
- [Phase ?]: Phase 5 D-REG-02 (plan 05-02, developer decision at a blocking checkpoint): REG-02 disposition is option-a -- extract the notebook's runnable simple-format calcs into pytest rather than executing the notebook. Options B and C were rejected (B required inventing a COORD.pdb and installing new deps in a regression-safety phase; C left the phase with no numeric baseline while ~20 files are edited). The notebook is never executed, so REG-02 is satisfied in SUBSTANCE, NOT IN FORM: notebook-only plumbing breakage (renamed kwarg, changed constructor signature) is an accepted documented gap, as is cell 5's PBC+PME+MD path, whose input file experiments/COORD.pdb is missing from the repo -- a genuine repository defect recorded for whoever repairs the tutorial.
- [Phase ?]: Phase 5 (plan 05-02): regression tolerances are derived from a measured SCF-sensitivity study, not chosen by feel. Tightening SCF_TOL 1e-6 -> 1e-10 moved the total energy by exactly 0.0, forces by 3.9e-8 eV/Ang and charges by 5.0e-9 e; bands are 1e-8 eV, 1e-6 eV/Ang, 1e-7 e, each ~20-25x above its own floor. The energy band is tightest because the total energy is variational in the converged density (second order in the density error) while forces and charges are first order.
- [Phase ?]: Phase 5 (plan 05-02): the notebook's 23 stored cell outputs were REFUSED as reference values (unknown hardware, unknown code revision); all baseline numbers were recomputed locally on db63487, verified before any 05-0* commit landed so the baseline predates every Phase 5 code change.
- [Phase ?]: D-01 keeps R_orb exported and unchanged; the per-pair lookup is added alongside it, protecting the ML-SK, stress, batch and SEDACS consumers additively (REG-03)
- [Phase ?]: R_tensor rows are made genuinely monotonic by continuing each pair's own arithmetic progression, rather than masking around the zero tail at the call site
- [Phase ?]: The per-pair knot lookup loops over torch.unique(pair_type); a batched 2-D searchsorted is forbidden because it materialises (n_neighbour_pairs, 1301) (threat T-05-04)
- [Phase ?]: script.py IS executed by tests/test_f_orbital_skf.py via load_validation_script(), so its get_skf_tensors unpacking is not inert and was updated
- [Phase 05]: Keep the SKF header oracle in a standard-library-only test module, separate from production imports. — A structurally separate parser plus an AST independence gate prevents the production parser from becoming its own oracle.
- [Phase 05]: Use the current 28-function mechanical sweep as the script.py inventory authority. — The plan-time claim of 30 check/run functions was stale; the current file contains 16 check/run and 28 expanded check/run/expected/parse functions.
- [Phase 05]: Do not alter inherited CH4 checksum drift or unrelated print sites in plan 05-03. — Those failures predate 05-03 and belong to the 05-01 regression gate and 05-05 output inventory.
- [Phase ?]: Phase 5 (plan 05-04): REG-06 had been silently reverted. Commit 56091af deleted ESDriver's R_tensor/n_grid arguments as a side effect of an unrelated test-oracle commit, so H0_and_S_vectorized took its None fallback and the per-pair knot lookup was inert in every real calculation for eleven commits. All 13 of 05-01's tests stayed green because every one calls the callee directly. Restored, plus test_esdriver_supplies_the_per_pair_grid_arguments which watches the WIRING; mutation-confirmed. No numeric change.
- [Phase ?]: Phase 5 D-01 (plan 05-04): the REG-05 guard compares grid STEP and never grid LENGTH, with a 1e-6 RELATIVE tolerance whose only job is surviving the BOHR_TO_ANGSTROM round trip; the hazard it separates is a factor of two. Same-step/different-length loads, protecting mio-1-1 (500/600/619 points) and 3ob-3-1 (650/850).
- [Phase ?]: Phase 5 (plan 05-04): SKFRadialGridStepMismatchError subclasses ValueError, not NotImplementedError. The four F*UnsupportedError classes mark capability that could land later; a mixed-step directory is not coherent as a parameter set at all. Message is keyed by os.path.basename so no path can leak by construction (threat T-04-07 policy).
- [Phase ?]: Phase 5 (plan 05-04): _ml_sk.build_pair_type_rcut's Bohr-vs-Angstrom docstring is corrected against measurement (CH4 C-H 1.0566812742799978 A / step 0.0105835442 A -> idx 98, R_orb[98]=1.0477708758 A). Both are Angstrom. The superseded claim is kept inside the new docstring and gated by test_rcut_values_unchanged_after_docstring_fix; AST-diff proves the change is prose-only.
- [Phase 05]: Phase 5 D-04 (plan 05-06): the orbital-count sweep matches THREE families, not the plan's single n_orb identifier set. The Phase 4 shell-resolved Coulomb defect lives at _coulomb_matrix.py:816-824, which tests max_ang and never mentions n_orb, so an n_orb-only sweep would have been a knowingly built blind spot. Neither gap the audit found mentions n_orb anywhere.
- [Phase 05]: Phase 5 D-04 (plan 05-06): the only real gaps in 157 sites are three truncated copies of the per-shell AO count table, [0, 1, 3, 5], in _spin.get_h_spin, _spin.get_h_spin_diag and _forces.forces_spin, while Constants.shell_dim holds the correct [0, 1, 3, 5, 7]. All GUARDED, not widened: D-12 defers spin-polarized f and tests/f_orbital_data ships no spinw.txt, so a correctly sized f block would be filled from parameters that do not exist. Reachable past ESDriver.forward via MD.py:745/794/1103 and _xl_tools.py:852.
- [Phase 05]: Phase 5 (plan 05-06): the sweep must blank FSTRING_START/MIDDLE/END, not just STRING. On Python 3.12+ an f-string is no longer one STRING token, so the first run swept the guards' own error messages (ESDriver.py:63/103) and MISSED their real counts == 16 comparisons (:60/:95) -- the audit pointed at its own documentation instead of its own code.
- [Phase 05]: Phase 5 (plan 05-06): the plan-time claim that newline flattening is what prevents undercounting is only half true and the measurement is recorded. Per-line vs flattened at pair level in _h0ands.py is 9 vs 25, so a grep pipeline does undercount; but whole-text matching with \s* separators already crosses newlines, giving 157 records either way. Flattening is kept as a structural guarantee, not as the thing that makes the sweep complete.
- [Phase 05]: Phase 5 (plan 05-06): _coulomb_matrix_batch's do_vec printed apology is classified OUT of D-04's row set -- nothing in that branch reads n_orb, max_ang or the basis layout -- and resolved anyway with plain NotImplementedError. No new exception class; test_no_new_f_exception_class_was_defined pins the F* taxonomy at exactly four.

### Pending Todos

None yet.

### Blockers/Concerns

- RESOLVED (task 03-01-02): the f angular formula source is locked. The user supplied the correct article (Takegahara/Aoki/Yanase, DOI 10.1088/0022-3719/13/4/016) and approved the checkpoint; the AO order, sign convention and provenance are recorded in `03-SOURCE-LOCK.md` and in `F_FORMULA_SOURCE`. Sharma PRB 19 2813 is rejected as primary. Tasks 03-01-03 and 03-01-04 completed.
- RESOLVED: the f AO ordering and real-harmonic convention are now pinned — the paper's cubic harmonics equal `Structure.AO_LABEL_TEMPLATE[9:16]`, so the adapter is a pure permutation with all +1 signs. Reference simulation tolerances remain open for Phase 4.
- Local environment cannot run the plan's pytest gates: uv and pytest are not installed and installing them is excluded by threat T-03-SC. Verification was executed via an equivalent stdlib runner instead (16/16 focused tests pass; the full gate is 23 passed / 1 pre-existing unrelated failure).
- f derivatives are not implemented. `ESDriver.calc_forces`, `ESDriverBatch.calc_forces` and analytical stress raise `FDerivativeUnsupportedError` for any system containing a 16-orbital atom, so Phase 4 must stay on energy/SCF validation or schedule the derivative work first.
- Batched f H0/S routing is still unimplemented (`H0_and_S_vectorized_batch` raises); reference validation must use the single-system path.
- `_bond_integral` exports a single `R_orb` (the longest grid) for all pair types while `tests/f_orbital_data` mixes radial grids. Likely to matter once real reference numbers are compared. See `deferred-items.md` item 3.
- docs/LIBRARY-OUTPUT-INVENTORY.md (plan 05-05) claims a completeness gate at tests/test_verbose_flag.py::test_inventory_covers_every_print; neither the file nor the test exists, so a print added by a later phase fails nothing. Logged in deferred-items.md item 4 and WINDOWS.md; belongs to whoever revisits D-02.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Extended physics | f forces, stress, MD, batch, SEDACS, ML-SK, and performance parity | Tracked as v2 unless required for reference validation | Initial roadmap |

## Session Continuity

Last session: 2026-08-03T02:45:41.307Z
Stopped at: Completed 05-06-PLAN.md (D-04 orbital-count audit); next 05-07
Resume file: None

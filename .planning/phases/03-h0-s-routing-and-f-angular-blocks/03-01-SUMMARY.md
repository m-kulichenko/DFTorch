---
phase: 03-h0-s-routing-and-f-angular-blocks
plan: 01
subsystem: physics-kernel
tags: [slater-koster, f-orbitals, cubic-harmonics, dftb, torch, hamiltonian, overlap]

# Dependency graph
requires:
  - phase: 01-skf-canonicalization-and-spline-validation
    provides: canonical 40-channel SKF order in _bond_integral._CHANNELS and the spline coefficient tensor consumed by H0/S
  - phase: 02-constants-and-structure-basis-metadata
    provides: const.n_orb (1/4/9/16), Structure.AO_LABEL_TEMPLATE with locked f labels, H_INDEX_START AO layout
provides:
  - Named 40-channel H/S addressing for every s/p/d formula (SK_CHANNEL_INDEX, sk_channel_index, sk_channel_name)
  - Source-locked f angular tables from Takegahara/Aoki/Yanase 1980 with a paper-to-Structure AO adapter
  - f_angular_sf / f_angular_pf / f_angular_df / f_angular_ff helpers in Structure AO order
  - Single-system 16-orbital pair routing (HZ/ZH/XZ/ZX/YZ/ZY/ZZ) writing AO offsets 9..15
  - Finite, symmetric H0/S for Eu-containing systems
  - FDerivativeUnsupportedError guards on every f derivative consumer
affects: [04-scf-and-reference-simulation-validation, f-forces, f-stress, batch-f-routing, ml-sk]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Slater-Koster channels are addressed by canonical name, never by numeric offset"
    - "External formula tables are transcribed in their published order and adapted to the locked internal AO order by an explicit permutation + sign vector"
    - "Transcribed physics is gated on an algebraic identity supplied by the source itself, not on golden numbers"
    - "Unimplemented derivatives raise a named error at the consumer instead of returning zeros"

key-files:
  created:
    - .planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md
    - .planning/phases/03-h0-s-routing-and-f-angular-blocks/deferred-items.md
  modified:
    - src/dftorch/_slater_koster_pair.py
    - src/dftorch/_h0ands.py
    - src/dftorch/_bond_integral.py
    - src/dftorch/_stress.py
    - src/dftorch/ESDriver.py
    - tests/test_f_orbital_skf.py
    - .gitignore

key-decisions:
  - "Takegahara, Aoki and Yanase, J. Phys. C 13 (1980) 583, DOI 10.1088/0022-3719/13/4/016 is the primary f angular source; Sharma, Phys. Rev. B 19, 2813 (1979) is rejected as primary (incomplete table, non-cubic orbitals) and kept only as a cross-check."
  - "The paper's cubic-harmonic f basis is identical to Structure.AO_LABEL_TEMPLATE, so PAPER_TO_STRUCTURE_F_PERMUTATION is the pure reordering (1,2,3,4,5,6,0) and PAPER_TO_STRUCTURE_F_SIGN is all +1. No basis rotation."
  - "Table 2 prints a subset; the p586 cyclic rule E_{sA,sB}(l,m,n) = E_{A,B}(m,n,l) generates the rest, plus transposition for f-f."
  - "Equations (14)-(15) are implemented as the transcription gate: every channel coefficient matrix must be an orthogonal projector and cross-shell blocks must project onto the same f subspaces."
  - "f derivatives are deferred (D-09). dH0/dS stay exactly zero in f blocks and every consumer raises FDerivativeUnsupportedError."
  - "F_ANGULAR_FORMULAS_AVAILABLE is flipped in the implementation task, not the source-lock task, so the flag keeps meaning 'trusted formulas exist' rather than 'a source was approved'."

patterns-established:
  - "Source lock: an external table may only be hard-coded after its DOI, title, authors, ordering and sign convention are recorded in code (F_FORMULA_SOURCE) and in a provenance document, and a test pins both."
  - "Self-verifying transcription: the shell-pair orthogonality identity ties s-f, p-f and d-f to f-f and to the already-trusted s-p/s-d rows, so a single mistyped coefficient cannot pass."
  - "Independent hand calculation: axis-direction blocks are written out by hand from the printed formulas and compared against the implementation, giving two separate derivations of the same numbers."

requirements-completed: [HSK-01, HSK-02, HSK-03, HSK-04, HSK-05, HSK-06, HSK-07, HSK-08]

coverage:
  - id: D1
    description: "Single-system H0/S explicitly routes every neighbor pair where either atom has n_orb == 16 (HZ/ZH/XZ/ZX/YZ/ZY/ZZ)."
    requirement: HSK-01
    verification:
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_eu_containing_h0_s_is_finite_shaped_and_symmetric"
        status: pass
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_f_block_entries_match_hand_calculated_values"
        status: pass
    human_judgment: false
  - id: D2
    description: "1-, 4- and 9-orbital H/X/Y routing keeps its pre-Phase-3 behavior under the named channel lookup."
    requirement: HSK-02
    verification:
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_f_free_h0_s_routing_regression"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_sk_channel_lookup_matches_bond_integral_order"
        status: pass
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_simple_format_f_free_metadata_regression"
        status: pass
    human_judgment: false
  - id: D3
    description: "s-f, p-f, d-f and f-f angular blocks transcribed from the source-locked tables and adapted into Structure AO order."
    requirement: HSK-03
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_orthogonality_identity"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_axis_blocks_match_hand_calculation"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_formula_source_lock"
        status: pass
    human_judgment: false
  - id: D4
    description: "p-f block, including the pf-sigma/pf-pi channel split and the reverse-direction parity."
    requirement: HSK-04
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_orthogonality_identity"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_parity_under_direction_reversal"
        status: pass
    human_judgment: false
  - id: D5
    description: "d-f block, including both E_g rows that the cyclic rule cannot generate."
    requirement: HSK-05
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_axis_blocks_match_hand_calculation"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_orthogonality_identity"
        status: pass
    human_judgment: false
  - id: D6
    description: "Full 7x7 f-f block from the twelve printed entries via cyclic rotation and transposition."
    requirement: HSK-06
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_orthogonality_identity"
        status: pass
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_f_block_entries_match_hand_calculated_values"
        status: pass
    human_judgment: false
  - id: D7
    description: "Eu-containing single-system H0/S returns finite, correctly shaped, symmetric matrices with the S identity diagonal."
    requirement: HSK-07
    verification:
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_eu_containing_h0_s_is_finite_shaped_and_symmetric"
        status: pass
    human_judgment: false
  - id: D8
    description: "Direction, atom-order and Structure AO-order conventions covered by x/y/z helper tests, hand-calculated block entries and bond reversal."
    requirement: HSK-08
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_angular_axis_blocks_match_hand_calculation"
        status: pass
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_f_containing_pair_atom_order_reversal"
        status: pass
      - kind: integration
        ref: "tests/test_f_orbital_skf.py#test_f_block_entries_match_hand_calculated_values"
        status: pass
    human_judgment: false
  - id: D9
    description: "Derivative-consuming f paths (forces, batched H0/S, analytical stress) raise a named unsupported error instead of using zero f derivatives."
    verification:
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_containing_derivative_paths_are_guarded"
        status: pass
      - kind: unit
        ref: "tests/test_f_orbital_skf.py#test_f_derivative_guard_rejects_f_systems"
        status: pass
    human_judgment: false
  - id: D10
    description: "Physical correctness of the transcribed f Hamiltonian against a published DFTB reference calculation."
    verification: []
    human_judgment: true
    rationale: "Phase 3 proves internal consistency (the paper's own orthogonality identity, hand calculation, AO placement) but deliberately stops short of SCF and reference-paper reproduction, which decision D-10 assigns to Phase 4. Only a physicist comparing converged results against published numbers can sign off that the assembled f Hamiltonian is right for real chemistry."

# Metrics
duration: 95min
completed: 2026-07-28
status: complete
---

# Phase 3 Plan 01: H0/S Routing and f Angular Blocks Summary

**Single-system H0/S now assembles real f-orbital Slater-Koster blocks, transcribed from the Takegahara/Aoki/Yanase 1980 cubic-harmonic tables and gated on that paper's own orthogonality identity.**

## Performance

- **Duration:** ~95 min (continuation session; task 03-01-01 landed earlier in `edf73e4`)
- **Started:** 2026-07-28T23:11Z (original session) / continuation resumed at checkpoint 03-01-02
- **Completed:** 2026-07-28
- **Tasks:** 4 of 4
- **Files modified:** 7 source/test files + 3 planning artifacts

## Accomplishments

- **Source-locked the f angular tables.** Recorded Takegahara, Aoki and Yanase, *"Slater-Koster tables for f electrons"*, J. Phys. C: Solid State Phys. **13** (1980) 583-588, DOI `10.1088/0022-3719/13/4/016` in `F_FORMULA_SOURCE`, with the full provenance (including the rejection of Sharma 1979 and the known Lendi/Sharma misprints) in `03-SOURCE-LOCK.md`.
- **Established that no basis change is needed.** The paper's cubic harmonics are exactly `Structure.AO_LABEL_TEMPLATE[9:16]`; only `fxyz` moves (paper lists it first, Structure last). `PAPER_TO_STRUCTURE_F_PERMUTATION = (1,2,3,4,5,6,0)`, `PAPER_TO_STRUCTURE_F_SIGN = (+1,)*7`.
- **Transcribed all of Table 2** — 3 s-f, 7 p-f, 21 d-f and 12 f-f printed entries — and implemented the p586 completeness rule `E_{sA,sB}(l,m,n) = E_{A,B}(m,n,l)` plus f-f transposition to reach full 1x7, 3x7, 5x7 and 7x7 coverage.
- **Turned equations (14)-(15) into a real gate.** Setting every two-centre integral of a shell pair to 1 must return the identity. Operationally this means each channel coefficient matrix is an orthogonal projector, which additionally forces `C_k(sf)^T C_k(sf) == C_k(pf)^T C_k(pf) == C_k(df)^T C_k(df) == C_k(ff)` and makes the sigma columns reproduce the already-trusted s-p, s-d and s-f rows. Verified over 512 random unit vectors at ~1e-15.
- **Wired the blocks into `Slater_Koster_Pair_SKF_vectorized`** at AO offsets 9..15 for all seven 16-orbital pair classes, with `(-1)**(l_a+l_b)` reverse-direction parity and IJ/JI radial channel selection matching the existing s-p / d-s conventions.
- **Extended the s/p/d masks** so 16-orbital atoms also receive their s-p, p-s, p-p, s-d, d-s, p-d, d-p and d-d blocks — previously a Eu atom would have gotten only its f blocks.
- **Guarded every f derivative consumer.** `FDerivativeUnsupportedError` now fires from `ESDriver.calc_forces`, `ESDriverBatch.calc_forces` and `_stress._pair_grad_from_sk`; batched H0/S keeps raising `FAngularFormulaSourceError`.

## Task Commits

1. **Task 03-01-01: Tracer — named H/S channel routing plus explicit f boundary** - `edf73e4` (fix) *(landed in the prior session)*
2. **Task 03-01-02: Blocking source-lock for f angular formula tables and AO convention** - `c4bdb5a` (docs)
3. **Task 03-01-03: Implement f angular helpers and single-system f routing** - `a8fcd2e` (feat)
4. **Task 03-01-04: Phase 3 H0/S validation and regression gate** - `996397c` (test)

## Files Created/Modified

- `src/dftorch/_slater_koster_pair.py` — source-lock constants, the full Takegahara Table 2 transcription, the cyclic generator, the paper-to-Structure adapter, `f_angular_sf/_pf/_df/_ff`, `FDerivativeUnsupportedError`, and the f block writes at AO offsets 9..15.
- `src/dftorch/_h0ands.py` — reworded the batched f guard now that the single-system formulas exist.
- `src/dftorch/_bond_integral.py` — spline truncation now starts at the first row carrying data (deviation 1 below).
- `src/dftorch/_stress.py` — analytical stress raises `FDerivativeUnsupportedError` for f systems.
- `src/dftorch/ESDriver.py` — shared `_require_f_derivatives` guard on both force entry points.
- `tests/test_f_orbital_skf.py` — 16 tests: source lock, orthogonality identity, parity, hand-calculated axis blocks, entry-by-entry H0/S placement, atom-order reversal, Eu H0/S contract, derivative guards, f-free regressions.
- `.gitignore` — the copyrighted article PDFs are excluded from the repository.

## Decisions Made

- **Primary source.** Takegahara *et al.* 1980 over Sharma 1979. Takegahara p588 states the reason directly: Sharma's table "does not contain all integrals and the orbitals do not have cubic symmetry", and it carries a misprint in `E_{xy,x^3-3x^2y}`. Sharma's tesseral basis would have required a full 7x7 change of basis rather than a permutation.
- **Correctness gate over golden numbers.** Golden values would have frozen whatever was typed in. The paper's orthogonality relation is an independent algebraic constraint that no plausible transcription error survives — a single wrong coefficient breaks idempotency, the trace, or the cross-shell projection.
- **Cyclic generation rather than transcribing every entry.** The paper explicitly says the unprinted entries follow by cyclic permutation. Implementing that rule (and asserting consistency where orbits overlap) is both what the source prescribes and less error-prone than inventing 100+ additional rows.
- **f derivatives deferred with hard guards** (decision D-09) rather than approximated. A zero gradient is indistinguishable from a real one, so consumers refuse rather than integrate it.
- **`F_ANGULAR_FORMULAS_AVAILABLE` flipped in task 03-01-03, not 03-01-02.** The success criteria listed the flip under the source-lock task, but flipping it before the formulas existed would have allowed silently-zero f blocks — exactly the failure mode the flag guards. Both changes land in the same session, so the end state matches the checklist.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Leading zero rows zeroed entire SKF spline tables**

- **Found during:** Task 03-01-03 (verifying that f blocks are actually populated)
- **Issue:** `_bond_integral.get_skf_tensors` truncated each spline at the *first* all-zero row anywhere in the table, to avoid ringing past the tabulated cutoff. Extended f-electron SKF tables pad their leading rows with zeros — `tests/f_orbital_data/Eu-Eu.skf` has 13 such rows below `r = 0.56 A` — so the heuristic fired on row 0 and set the **whole** coefficient tensor to zero. Every Eu-containing H0/S was therefore identically zero off the diagonal, which would have made the newly-routed f blocks look implemented while returning nothing. The previous tracer test never caught it because it asserted an exception raised before any radial evaluation.
- **Fix:** the truncation search now starts at the first row that actually carries data, preserving the original tail-truncation behavior for every table without leading zeros.
- **Files modified:** `src/dftorch/_bond_integral.py`
- **Verification:** `tests/test_f_orbital_skf.py::test_f_block_entries_match_hand_calculated_values` (entry-by-entry against hand-calculated angular constants times independently evaluated radial channels) and `::test_f_free_h0_s_routing_regression` (unchanged mio behavior).
- **Committed in:** `a8fcd2e` (part of task 03-01-03)

**2. [Rule 2 - Missing critical functionality] 16-orbital atoms received no s/p/d blocks**

- **Found during:** Task 03-01-03
- **Issue:** The existing s-p, p-s, p-p, s-d, d-s, p-d, d-p and d-d masks enumerated only the 1/4/9 pair classes. A 16-orbital atom has s, p and d shells too, so a Eu atom routed through the new f masks would still have been missing its entire non-f coupling — a finite, symmetric, and badly wrong Hamiltonian.
- **Fix:** each existing mask was extended with the 16-orbital classes that satisfy its shell requirement (for example s-d now includes `HZ | XZ | YZ | ZZ`).
- **Files modified:** `src/dftorch/_slater_koster_pair.py`
- **Verification:** `::test_eu_containing_h0_s_is_finite_shaped_and_symmetric` plus the orthogonality/parity gates.
- **Committed in:** `a8fcd2e`

**3. [Rule 2 - Missing critical functionality] Force paths had no f derivative guard**

- **Found during:** Task 03-01-03
- **Issue:** Task 03-01-01 guarded batched H0/S and analytical stress, but `ESDriver.calc_forces` and `ESDriverBatch.calc_forces` were reachable for f systems. Once f *values* exist, an f system runs all the way to forces and silently integrates the zero f entries of `dH0`/`dS`.
- **Fix:** added `FDerivativeUnsupportedError`, `F_ANGULAR_DERIVATIVES_AVAILABLE` and a shared `_require_f_derivatives` guard invoked at the top of both force entry points; retargeted the stress guard to the new error.
- **Files modified:** `src/dftorch/ESDriver.py`, `src/dftorch/_stress.py`, `src/dftorch/_slater_koster_pair.py`
- **Verification:** `::test_f_containing_derivative_paths_are_guarded`, `::test_f_derivative_guard_rejects_f_systems`
- **Committed in:** `a8fcd2e`

**4. [Rule 2 - Security/licensing] Copyrighted article PDFs were untracked at the repo root**

- **Found during:** Task 03-01-02
- **Issue:** Both source PDFs sat untracked at the repository root where a `git add -A` would have committed copyrighted articles.
- **Fix:** `*.pdf` added to `.gitignore` alongside the cached publisher HTML.
- **Files modified:** `.gitignore`
- **Committed in:** `c4bdb5a`

---

**Total deviations:** 4 auto-fixed (1x Rule 3, 3x Rule 2)
**Impact on plan:** All four were required for the plan's own acceptance criteria to be meaningful — without #1 the f blocks are provably zero, without #2 the Hamiltonian is incomplete, without #3 the plan's explicit prohibition on silent zero derivatives is violated, and #4 protects the repository. No scope creep: no new features, no new dependencies, no package installs.

## Issues Encountered

- **The article is a CCITT G4 scan with no usable math text layer.** Table 2 could not be extracted programmatically. Page images were rendered and read directly, with tight crops upscaled to ~2400px for the dense f-f rows.
- **Table 2 prints only a subset.** 12 of the 49 f-f entries are given. Mapping the orbit structure under the cyclic rotation (17 orbits: one fixed point plus 16 of size 3) confirmed that the 12 printed entries plus transposition close the block exactly, with overlapping orbits agreeing — itself a consistency check on the transcription.
- **`E_{x^2-y^2, xyz}` and `E_{xy, z(x^2-y^2)}` are printed identically.** This looked like a typesetting duplication; the orthogonality gate confirms both values are correct as printed.
- **`tests/test_scf.py::test_energy_smoke_import_and_call` fails.** Reproduced identically against a stashed pristine tree, so it is pre-existing and unrelated (a float32/float64 mix in `ewald_pme/PME_torch.py`). Logged in `deferred-items.md`.
- **Mixed radial grids in the f fixture directory.** `_bond_integral` exports a single `R_orb` (the longest grid) while `tests/f_orbital_data` mixes a 0.04 A / 433-point extended grid with the simple-format grid. The Phase 3 tests therefore assert angular structure and AO placement, deriving their expected radial factor from the same `const.R_orb` + `coeffs_tensor` lookup the implementation uses. Logged in `deferred-items.md` as a Phase 1 concern.

## Verification Note (honest reporting)

The plan's literal gate is:

```
uv run python -m pytest tests/test_f_orbital_skf.py -q
uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py \
    tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q
```

**This command did not literally run.** `uv` and `pytest` are not installed on this machine, `dftorch` is not installed as a package, and installing them is a package-manager install excluded by threat `T-03-SC` and by the user's explicit instruction. The same test functions were executed by a stdlib runner kept outside the repository, which imports each test module, supplies a `tmp_path`, and runs every `test_*` function with the identical assertions.

Result:

- `tests/test_f_orbital_skf.py` — **16 passed, 0 failed**
- Full gate — **23 passed, 1 failed**; the single failure is the pre-existing, unrelated `test_scf.py::test_energy_smoke_import_and_call` documented above and reproduced on a pristine checkout.

## Known Stubs

None. Every f block written in this phase carries real source-locked values.

Deliberately **not** implemented, with explicit named errors rather than stubs (decision D-09 / D-01):

| Capability | Behavior | Guard |
|---|---|---|
| f angular derivatives | `dH0`/`dS` are exactly zero in f blocks | `FDerivativeUnsupportedError` from both `calc_forces` entry points and `_pair_grad_from_sk` |
| batched f H0/S | not routed | `FAngularFormulaSourceError` from `H0_and_S_vectorized_batch` |
| ML-SK f channels | no ML head exists for f | `NotImplementedError` from `_get_val_dR` |

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change was introduced. The threat register's `mitigate` dispositions were honored:

- `T-03-01` (channel lookup tampering): named lookup plus `::test_sk_channel_lookup_matches_bond_integral_order` and the f-free value regression.
- `T-03-02` (AO adapter): the adapter is pinned by `::test_f_angular_formula_source_lock` and validated by the orthogonality gate.
- `T-03-06` (source identity): DOI, title and authors recorded in `F_FORMULA_SOURCE` and asserted in tests.
- `T-03-07` (extraction repudiation): `03-SOURCE-LOCK.md` records page numbers, and the hand-calculated constants carry their derivations inline in the test file.
- `T-03-08` (derivative tampering): named errors, no silent zeros.
- `T-03-SC` (package installs): no install was attempted; the missing-pytest situation was reported rather than resolved by installing.

## User Setup Required

None ongoing. The one-time setup — supplying the DOI `10.1088/0022-3719/13/4/016` article — was completed and approved at checkpoint 03-01-02. The PDF stays untracked and gitignored; nothing in the codebase reads it at runtime.

## Next Phase Readiness

**Ready for Phase 4 (SCF and Reference Simulation Validation):**

- `H0_and_S_vectorized` returns finite, symmetric, correctly shaped H0/S for Eu-containing single systems, so SCF can be driven end to end.
- The angular layer is independently validated, so any Phase 4 disagreement with published results points at the radial/SCF layer rather than at the Slater-Koster geometry.

**Carry into Phase 4:**

- `calc_forces` refuses f systems. Phase 4 must either restrict itself to energy/SCF validation or schedule the f derivative work first.
- The shared-`R_orb` issue (deferred item 3) is likely to matter as soon as real reference numbers are compared, because `tests/f_orbital_data` mixes radial grids.
- Batched f routing is still unimplemented; reference validation must use the single-system path.

## Self-Check: PASSED

- All 10 claimed files exist on disk.
- All 4 claimed commits (`edf73e4`, `c4bdb5a`, `a8fcd2e`, `996397c`) exist in git history.
- All 14 claimed public symbols resolve on import of `dftorch._slater_koster_pair`
  (`SK_CHANNEL_INDEX`, `sk_channel_index`, `sk_channel_name`, `F_FORMULA_SOURCE`,
  `PAPER_F_AO_ORDER`, `PAPER_TO_STRUCTURE_F_PERMUTATION`, `PAPER_TO_STRUCTURE_F_SIGN`,
  `F_ANGULAR_FORMULAS_AVAILABLE`, `F_ANGULAR_DERIVATIVES_AVAILABLE`,
  `FDerivativeUnsupportedError`, `f_angular_sf`, `f_angular_pf`, `f_angular_df`,
  `f_angular_ff`).

---
*Phase: 03-h0-s-routing-and-f-angular-blocks*
*Completed: 2026-07-28*

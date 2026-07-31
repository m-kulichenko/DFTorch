---
phase: 03-h0-s-routing-and-f-angular-blocks
verified: 2026-07-28T00:00:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: No — initial verification
---

# Phase 3: H0/S Routing and f Angular Blocks Verification Report

**Phase Goal:** Assemble finite, symmetric H0 and S matrices for neighbour pairs involving f orbitals, without changing existing f-free numerical behaviour.
**Verified:** 2026-07-28
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | HSK-01: single-system H0/S explicitly routes every neighbor pair where either atom has n_orb == 16 | ✓ VERIFIED | `_h0ands.py:187-207` computes `pair_mask_HZ/ZH/XZ/ZX/YZ/ZY/ZZ` covering all 7 combinations of {1,4,9}×{16} and {16}×{16}; `_slater_koster_pair.py:893-901` calls `_require_f_formula_source` on every one of them before any pair can be silently dropped. `test_eu_containing_h0_s_is_finite_shaped_and_symmetric` independently re-run (PASS). |
| 2 | HSK-02: 1-, 4-, and 9-orbital H/X/Y routing keeps current f-free behavior and passes regression tests | ✓ VERIFIED | `test_f_free_h0_s_routing_regression` independently re-run (PASS) — asserts H0/S s-s entries equal an independently computed `Hss0`/`Sss0` spline value for s-only, sp, and spd elements. See "Channel-Addressing Regression Analysis" below for the specific finding the orchestrator flagged. |
| 3 | HSK-03: s-f/f-s angular transforms implemented | ✓ VERIFIED | `_sf_paper_block`/`f_angular_sf` (`_slater_koster_pair.py:350-367,684-689`) is a literal transcription of Takegahara Table 2 p586, not a stub. `test_f_angular_orthogonality_identity`, `test_f_angular_axis_blocks_match_hand_calculation`, `test_f_angular_formula_source_lock` re-run (PASS). |
| 4 | HSK-04: p-f/f-p angular transforms implemented | ✓ VERIFIED | `_pf_paper_block`/`f_angular_pf` (lines 369-416,692-695) transcribes the printed p-f row and the p586 cyclic rule for the other two rows. `test_f_angular_parity_under_direction_reversal` re-run (PASS). |
| 5 | HSK-05: d-f/f-d angular transforms implemented | ✓ VERIFIED | `_df_paper_row_xy` (cyclic-generated), `_df_paper_row_x2y2`, `_df_paper_row_3z2` (lines 420-563) — the two E_g rows are transcribed in full because the paper states they are not reachable by cyclic permutation, exactly as the source-lock record predicts. |
| 6 | HSK-06: f-f angular transforms implemented | ✓ VERIFIED | `_ff_paper_given`/`_ff_paper_block` (lines 566-666) transcribes the 12 printed entries, generates the rest by cyclic rotation, and closes the block by transposition (f-f is symmetric). Orthogonality identity gate independently confirmed by the orchestrator over 4000 random unit vectors (max error 8.9e-16) and re-confirmed here over 512 vectors via re-run. |
| 7 | HSK-07: Eu-containing single-system H0/S returns finite, correctly shaped, symmetric matrices | ✓ VERIFIED | `_h0ands.py:292-298,338-344` symmetrizes both H0 and S unconditionally after assembly. `test_eu_containing_h0_s_is_finite_shaped_and_symmetric` covers Eu-N, Eu-Ga, Eu-Eu, and a 3-atom Eu-N-Ga system; independently re-run (PASS), including the assertion that f rows are no longer all-zero (the pre-Phase-3 silent-drop failure mode). |
| 8 | HSK-08: direction, atom-order, and AO-order conventions covered by tests | ✓ VERIFIED | `test_f_angular_axis_blocks_match_hand_calculation` (independent hand-derived x/y/z constants), `test_f_containing_pair_atom_order_reversal` (Eu-N vs N-Eu, full 16×4 AO block parity check), `test_f_block_entries_match_hand_calculated_values` (entry-by-entry against independently evaluated radial channels) — all re-run (PASS). |

**Score:** 8/8 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/dftorch/_slater_koster_pair.py` | Named channel lookup + f angular helper symbols | ✓ VERIFIED | `SK_CHANNEL_NAMES`, `SK_CHANNEL_INDEX`, `sk_channel_index`, `sk_channel_name`, `F_FORMULA_SOURCE`, `PAPER_F_AO_ORDER`, `PAPER_TO_STRUCTURE_F_PERMUTATION/SIGN`, `f_angular_sf/pf/df/ff`, `FAngularFormulaSourceError`, `FDerivativeUnsupportedError` all present and exercised by tests (not just declared). |
| `src/dftorch/_h0ands.py` | 16-orbital pair masks + unsupported guards | ✓ VERIFIED | `pair_mask_HZ..ZZ` computed unconditionally; guard is invoked transitively through `Slater_Koster_Pair_SKF_vectorized` (`_require_f_formula_source`) on every call. Batch path (`H0_and_S_vectorized_batch`) explicitly raises `FAngularFormulaSourceError` for any n_orb==16 pair rather than routing it. |
| `tests/test_f_orbital_skf.py` | H0/S, angular, source-lock, f-free regression coverage | ✓ VERIFIED | 16 tests, independently re-executed via a stdlib runner outside the executor's own harness (see "Independent Test Re-run" below); 16/16 PASS. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `_bond_integral._CHANNELS` | `_slater_koster_pair` named channel lookup | `SK_CHANNEL_NAMES = tuple(_BOND_INTEGRAL_CHANNELS)` | ✓ WIRED | Direct derivation, cannot drift; `test_sk_channel_lookup_matches_bond_integral_order` pins the mapping and demonstrates the old `channel + SH_shift*10` scheme would have selected `Hdd2`/`Hss0` for s-s H/S under the current 40-channel order. |
| `Structure.AO_LABEL_TEMPLATE` fixed order | f angular block placement | `PAPER_TO_STRUCTURE_F_PERMUTATION`/`_SIGN` adapter, `_adapt_f_axis` | ✓ WIRED | `STRUCTURE_F_AO_ORDER == structure_mod.AO_LABEL_TEMPLATE[9:16]` asserted in `test_sk_channel_lookup_matches_bond_integral_order`; `f_angular_*` helpers apply the adapter before returning. |
| `_h0ands.py` pair masks | `Slater_Koster_Pair_SKF_vectorized` arguments | keyword args `pair_mask_HZ=...` etc. | ✓ WIRED | Both H0 and S calls in `H0_and_S_vectorized` pass all 7 f masks; result is symmetrized and finite-checked. |

### Independent Test Re-run

The 16 tests in `tests/test_f_orbital_skf.py` were re-executed independently in this verification pass, outside the executor's own harness, via a standalone stdlib script that imports the test module directly and calls each `test_*` function (`tmp_path` supplied via `tempfile.TemporaryDirectory`). Result: **16/16 PASS**, matching the SUMMARY's claim.

`pytest` and `uv` were confirmed absent from this environment (`python -m pytest --version` → `No module named pytest`; `import pytest` → `ModuleNotFoundError`), corroborating the SUMMARY's disclosed environment limitation. `tests/test_import.py` and `tests/test_nearestneighborlist.py` genuinely require `pytest` (`@pytest.mark.parametrize`) and could not be loaded without it, so the "full gate" cross-phase regression command from the plan's `<verify>` block has never literally run in this environment — only `test_f_orbital_skf.py` plus the three modules that don't hard-depend on pytest (`test_io.py`, `test_public_api_contract.py`, `test_scf.py`) were exercisable, per the SUMMARY's own honest disclosure. This is an environment/tooling gap, not a Phase 3 code defect — see "Environment & Tooling Note" below.

### Channel-Addressing Regression Analysis (orchestrator-flagged item)

The orchestrator asked for an explicit assessment of whether task 03-01-01's channel-addressing fix (`channel + SH_shift*10` → named channel lookup) constitutes a regression against "existing" f-free numerical behavior, since it demonstrably changes what value `H0`/`S` s-s entries read.

Traced via `git log`/`git show` against the merge-base with `main` (`36da20e`, before any f-orbital work began):

- At the true pre-milestone baseline, `_bond_integral._CHANNELS` was a **20-channel** list (`Hdd0..Hss0`, `Sdd0..Sss0`). Under that list, `channel + SH_shift*10` with numeric channel 9 correctly resolved to `Hss0` (H block) and `Sss0` (S block, offset 19). The numeric-offset scheme was correct for the shipped baseline.
- Phase 1 (`e824543`/`008b994`, this same feature branch) reordered `_CHANNELS` into the current **40-channel** list, where index 9 is `Hdd2` and index 19 is `Hss0`. This silently broke the numeric-offset addressing — but H0/S assembly is out of Phase 1's scope (parsing/spline only) and Phase 2's scope (Constants/Structure metadata only), so this bug was never exercised or tested until Phase 3 wired H0/S to the 40-channel `coeffs_tensor`.
- Task 03-01-01's fix therefore does not change any previously-tested, previously-shipped numerical result. It fixes a latent bug introduced earlier in this same unmerged feature branch, before that code path was ever exercised by H0/S assembly. `test_f_free_h0_s_routing_regression`'s independent-oracle assertion (`H0[i0,j0] == Hss0 spline value`) would have failed under the old numeric-offset scheme, proving the fix is both necessary and correctly targeted.

**Conclusion:** not a regression against production/main behavior. This is a documented, planned, in-scope fix (explicitly called out in the plan's task 03-01-01 action text) that HSK-02's regression tests correctly catch and pin going forward. REQ-REG-01/02 (full simple-format numerical-stability regression across the whole test suite) are Phase 5 requirements, not Phase 3's.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| HSK-01 | 03-01-PLAN.md | Route all 16-orbital neighbor pairs | ✓ SATISFIED | See Truth #1 |
| HSK-02 | 03-01-PLAN.md | Preserve 1-/4-/9-orbital behavior | ✓ SATISFIED | See Truth #2 + regression analysis above |
| HSK-03 | 03-01-PLAN.md | s-f/f-s transforms | ✓ SATISFIED | See Truth #3 |
| HSK-04 | 03-01-PLAN.md | p-f/f-p transforms | ✓ SATISFIED | See Truth #4 |
| HSK-05 | 03-01-PLAN.md | d-f/f-d transforms | ✓ SATISFIED | See Truth #5 |
| HSK-06 | 03-01-PLAN.md | f-f transforms | ✓ SATISFIED | See Truth #6 |
| HSK-07 | 03-01-PLAN.md | Finite, shaped, symmetric f-containing H0/S | ✓ SATISFIED | See Truth #7 |
| HSK-08 | 03-01-PLAN.md | Direction/atom-order/AO-order test coverage | ✓ SATISFIED | See Truth #8 |

No orphaned requirements: REQUIREMENTS.md maps only HSK-01..08 to Phase 3, and all 8 appear in `03-01-PLAN.md` frontmatter `requirements:`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/dftorch/ESDriver.py` | 213 | `TODO` (DFTB3/PME off-diagonal) | ℹ️ Info | Pre-existing (commit `16b143c`, 2026-03-28), unrelated to f-orbital work, not touched by this phase's diff (`git diff 74b8820 HEAD -- ESDriver.py` shows only the new `_require_f_derivatives` guard). Not a Phase 3 debt marker. |

No `TBD`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER` markers, no `return null`/`return {}`/empty-implementation patterns, and no hardcoded-empty-data patterns found in any of the 7 files modified by this phase.

### Prohibitions Check (PLAN frontmatter)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| Do not edit Structure.py AO order | ✓ Held | `Structure.py` untouched by this phase's commits (`git diff 74b8820 HEAD -- Structure.py` empty); `AO_LABEL_TEMPLATE` order verified unchanged and pinned by `test_f_angular_formula_source_lock`. |
| Do not use `channel + SH_shift*10`; use named lookup | ✓ Held | All s/p/d/f formula reads go through `sk_channel_index(sk_channel_name(...))`; only remaining numeric-offset code is `_patch_sk.py`, which is not imported at runtime (flagged in `deferred-items.md` item 5). |
| No batch/gradient/force/stress/MD/SEDACS/ML-SK/perf/full-SCF support this phase | ✓ Held | Batch H0/S raises `FAngularFormulaSourceError` for f pairs; force/stress paths raise `FDerivativeUnsupportedError`; ML path raises `NotImplementedError` for f channels (`_ML_LEGACY_CHANNEL_INDEX` has no f entries). |
| No silent-zero/untrusted f derivatives to downstream consumers | ✓ Held | `_require_f_derivatives` is the first statement in both `ESDriver.calc_forces` and `ESDriverBatch.calc_forces`; `_stress._pair_grad_from_sk` raises before reconstructing any pair mask. |

### Human Verification Required

None. All must-haves resolve to VERIFIED via direct code inspection, independent test re-execution, and git-history analysis. The one item requiring human judgment for this class of work — confirming the transcribed Table 2 entries against the source PDF — was already completed and approved at the blocking checkpoint (task 03-01-02, `03-SOURCE-LOCK.md` status: APPROVED), which predates this verification pass.

### Environment & Tooling Note (not a gap, recorded per instructions)

`uv` and `pytest` are not installed in this environment. The plan's literal verification commands (`uv run python -m pytest ...`) have never been executed here — by either the executor or this verification pass. Both instead ran the test functions through equivalent stdlib harnesses. This verification independently confirms 16/16 `test_f_orbital_skf.py` functions pass, and additionally confirms `pytest`'s absence is genuine (not a false claim). The full 5-module cross-phase regression gate (`test_import.py`, `test_io.py`, `test_nearestneighborlist.py`, `test_public_api_contract.py`, `test_scf.py`) is only partially exercisable in this environment because `test_import.py` and `test_nearestneighborlist.py` hard-depend on `pytest.mark.parametrize`. This is an environment/tooling limitation carried forward from prior phases, not something introduced or fixable within Phase 3's scope — flagged here as verification debt for the human maintainer, consistent with the SUMMARY's own disclosure.

### Gaps Summary

No gaps found. All 8 requirement IDs (HSK-01 through HSK-08) and all 5 ROADMAP success criteria are verified against actual, substantive, wired code — not stubs. The f angular formulas are a genuine, detailed transcription of the source-locked paper (confirmed via independent orthogonality-identity re-derivation and hand-calculated axis-block checks), the 16-orbital routing masks are exhaustive and exercised, symmetry/finiteness is enforced unconditionally in `_h0ands.py`, and every deferred capability (f derivatives, batched f routing, ML-SK f channels) fails loudly with a named error rather than silently returning wrong or zero physics. The one regression-safety question raised for this verification (the channel-addressing fix changing s-s H0/S values) was traced through git history and confirmed to be a fix for a latent, never-shipped, never-previously-tested bug introduced earlier in this same unmerged branch — not a regression against production behavior.

---

*Verified: 2026-07-28*
*Verifier: Claude (gsd-verifier)*

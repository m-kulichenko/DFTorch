---
phase: 06-self-consistent-scf-for-f-systems
plan: 02
subsystem: electrostatics
tags: [dftb, coulomb, hubbard-u, shell-resolved, f-orbitals, pytest, torch]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "the Eu-N diatomic fixture, the shell-resolved Hubbard/charge data structures (Hubbard_U_sr, shell_types, H_INDEX_START_U), the MAGNETIC_HUBBARD_LDEP gate (D-16/D-23), and the FShellResolvedCoulombUnsupportedError refusal this plan retires"
  - phase: 05-regression-safety-and-support-policy-cleanup
    provides: "the orbital-count sweep and inventory that account for every hardcoded shell count, and the four-name f exception taxonomy pinned by test_no_new_f_exception_class_was_defined"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 01
    provides: "a settling charge loop for Eu-N and the 228-passed suite baseline this plan had to hold"
provides:
  - "All sixteen ordered shell-pair blocks of the per-orbital-group Coulomb matrix, so a molecule containing an f element gets a complete matrix instead of a refusal"
  - "_shell_pair_mask: one uniform selection rule, (max_ang_I > l_i) & (max_ang_J > l_j), replacing eight hand-enumerated named masks"
  - "_coul_shell_pair_term: short-range damping dispatched on strength equality rather than element identity, which removes a live divide-by-zero"
  - "tests/test_shell_resolved_coulomb_f.py, an 8-test module whose main gate is a derived identity against the per-atom builder rather than any frozen number"
  - "A retired-but-kept FShellResolvedCoulombUnsupportedError, watched by test so 'retired' is a checked fact"
affects: [06-03, 06-04, 06-05, shell-resolved-charge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Validate a new numerical path by a derived identity against an existing one (degenerate-parameter reduction), not by freezing its output"
    - "Sixteen structurally identical blocks in place of nine hand-shaped ones: a block that differs from its neighbours is how the missing seven hid"
    - "Dispatch a closed form on the quantity the formula is singular in (Ti == Tj), not on a proxy for it (same element)"
    - "A retired refusal keeps its name and gains a retirement note; a test watches that nothing raises it"

key-files:
  created:
    - tests/test_shell_resolved_coulomb_f.py
  modified:
    - src/dftorch/_coulomb_matrix.py
    - src/dftorch/_slater_koster_pair.py
    - src/dftorch/ESDriver.py
    - tests/test_shell_resolved_u.py
    - tests/test_orbital_count_guards.py
    - docs/ORBITAL-COUNT-INVENTORY.md
    - docs/F-SUPPORT-STATUS.md

key-decisions:
  - "One uniform mask rule instead of sixteen named masks per block. Verified by hand against all nine pre-existing blocks before writing it, and then verified numerically by the reduction identity and the methane control."
  - "All sixteen blocks written out explicitly rather than looped, so the nine pre-existing ones keep a one-line diff and the plan's index_add_ == 32 criterion stays meaningful"
  - "The short-range damping form is chosen by whether the two damping exponents are equal, not by whether the two atoms are the same element - the two conditions differ for every off-diagonal block, and the difference was a divide-by-zero"
  - "The 17 inventory rows for _coulomb_matrix.py were deleted rather than re-dispositioned, because the audit matches rows and sites as multisets in both directions; the account of what was there is kept as prose in the same section"

requirements-completed: [SCC-02]

coverage:
  - id: D1
    description: "A molecule containing an f element gets a complete matrix - every orbital group's row and column carries a real value instead of a silent zero"
    requirement: SCC-02
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_coulomb_f.py#test_eu_eu_matrix_has_no_empty_row_or_column"
        status: pass
      - kind: integration
        ref: "tests/test_shell_resolved_u.py#test_shell_resolved_coulomb_builds_for_f_system"
        status: pass
    human_judgment: false
  - id: D2
    description: "All seven previously missing angular blocks (s-f, f-s, p-f, f-p, d-f, f-d, f-f) are populated, each addressed individually on the only fixture pair that reaches all seven"
    requirement: SCC-02
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_coulomb_f.py#test_all_seven_f_blocks_are_populated"
        status: pass
    human_judgment: false
  - id: D3
    description: "With one repulsion strength shared by every orbital group, the per-orbital-group matrix reproduces the per-atom matrix entry for entry - a derived identity, not a recorded run"
    requirement: SCC-02
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_coulomb_f.py#test_equal_group_strengths_reproduce_the_per_atom_matrix"
        status: pass
    human_judgment: false
  - id: D4
    description: "The matrix equals its own transpose, as the short-range formula requires"
    requirement: SCC-02
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_coulomb_f.py#test_matrix_equals_its_own_transpose"
        status: pass
    human_judgment: false
  - id: D5
    description: "Nothing an f-free calculation computes has changed"
    requirement: SCC-02
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_coulomb_f.py#test_f_free_matrix_still_has_no_empty_row"
        status: pass
      - kind: integration
        ref: "tests/test_single_shot_energy.py"
        status: pass
      - kind: integration
        ref: "tests/test_simple_format_regression.py"
        status: pass
      - kind: integration
        ref: "tests/test_scf.py"
        status: pass
    human_judgment: false
  - id: D6
    description: "The structural assumption the row and column offsets rest on - every element's orbital groups are a contiguous run starting at s - is stated out loud rather than left implied"
    requirement: SCC-02
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_coulomb_f.py#test_orbital_groups_are_a_contiguous_run_from_s"
        status: pass
    human_judgment: false
  - id: D7
    description: "The refusal is retired in code and in both published documents at the same time, the taxonomy is still exactly four names, and no production module raises it"
    requirement: SCC-02
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_coulomb_f.py#test_the_retired_refusal_is_no_longer_raised_by_production_code"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_no_new_f_exception_class_was_defined"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_u.py#test_retired_refusal_message_records_its_own_retirement"
        status: pass
    human_judgment: false

# Metrics
duration: ~95min
completed: 2026-08-05
status: complete
---

# Phase 6 Plan 02: All Sixteen Angular Blocks of the Shell-Resolved Coulomb Matrix Summary

**A molecule containing an f element now gets a complete electron-repulsion matrix - all sixteen ordered orbital-group blocks instead of nine - proven by reducing it to the per-atom builder rather than by freezing any of its numbers.**

## Performance

- **Duration:** ~95 min
- **Completed:** 2026-08-05
- **Tasks:** 2
- **Files:** 8 (1 created, 7 modified). No file deleted in any of the three commits.

## Accomplishments

- **The seven missing blocks exist and are proven non-empty.** s-f, f-s, p-f, f-p, d-f, f-d and f-f are each addressed at their own row and column on a europium-europium pair - the only two-atom arrangement in the fixture set that reaches all seven, because f-f, d-f and f-d need an f group on both sides and nitrogen has none.
- **The main gate freezes no number.** `test_equal_group_strengths_reproduce_the_per_atom_matrix` overrides the p, d and f Hubbard tables in memory so every orbital group of every element carries its element's s strength. All sixteen blocks then evaluate the same expression, which is exactly the per-atom builder's expression, so every off-diagonal entry must agree to 1e-10. It runs for Eu-Eu, Eu-N and methane and validates the new blocks, their masks, their row and column offsets and their strength selection at once.
- **Sixteen named masks became one rule.** `_shell_pair_mask(max_ang_I, max_ang_J, l_i, l_j)` returns `(max_ang_I > l_i) & (max_ang_J > l_j)`. It was checked by hand against all nine pre-existing blocks before it was written, including the unmasked s-s case, and it compares `max_ang` against a *parameter* rather than a literal - which is why the file now returns zero records from the Phase 5 orbital-count sweep.
- **A live divide-by-zero was found and fixed** (see Deviations). The nine old blocks chose their short-range damping form by element identity, and only the three diagonal ones did even that. Two atoms of one element still contribute to the s-p block, where s and p read different strength tables; if that element's shell strengths coincide - which is what a non-extended SKF file usually writes - `coul_diff_elem_and_ang` divided by `Ti**2 - Tj**2` = 0. `tests/f_orbital_data/N-N.skf` carries Us = Up = Ud = 0.490 Ha, so an N2 molecule on the shell-resolved path produced NaN.
- **The refusal is retired in code and in both published documents in one commit**, and "retired" is watched rather than claimed: a new test scans every module under `src/dftorch/` for a `raise` naming the class and finds none, while the class itself stays importable so the f taxonomy is still exactly four names.
- **Full suite: 234 passed, 0 failed** (was 228). +8 from the new module, -2 from the guard case that no longer exists.

## Task Commits

1. **Task 1: Complete the electron-repulsion matrix for f elements** (TDD)
   - `7f386be` (test) - the six failing gates, written before any source change. Four failed on the refusal; the two f-free controls passed from the start, which is what a control is for.
   - `8068230` (feat) - `_shell_pair_mask`, `_coul_shell_pair_term`, sixteen blocks, guard deleted.
2. **Task 2: Make the published support policy say what the code now does** - `e988c05` (docs)

## Files Created/Modified

- `tests/test_shell_resolved_coulomb_f.py` (created, 700 lines) - 8 tests. Pure ASCII, verified by byte scan.
- `src/dftorch/_coulomb_matrix.py` (+319/-218) - two new helpers, sixteen uniform blocks, `_require_no_f_shell_resolved_coulomb` deleted, two now-unused imports removed.
- `src/dftorch/_slater_koster_pair.py` (+50/-17) - the class docstring and message constant rewritten as a retirement record.
- `src/dftorch/ESDriver.py` (+6/-5) - the comment at the `C_sr` call site no longer says the call refuses.
- `tests/test_shell_resolved_u.py` (+111/-26) - three refusal tests converted to their positive counterparts; the module comment updated.
- `tests/test_orbital_count_guards.py` (+7/-39) - the `ewald_real_space_vectorized_sr` guard case and its inert check removed.
- `docs/ORBITAL-COUNT-INVENTORY.md`, `docs/F-SUPPORT-STATUS.md` - see below.

## Verification Evidence

| Gate | Result |
|---|---|
| `uv run pytest` (whole suite) | **234 passed, 0 failed** in 112 s - above the 228 baseline |
| `uv run pytest tests/test_shell_resolved_coulomb_f.py -x` | 8 passed |
| `uv run pytest tests/test_shell_resolved_u.py tests/test_orbital_count_guards.py tests/test_support_documentation.py tests/test_shell_resolved_coulomb_f.py -x` | 74 passed |
| `uv run pytest tests/test_single_shot_energy.py tests/test_scf.py tests/test_scf_convergence_f.py tests/test_simple_format_regression.py` | 29 passed - nothing on the per-atom path moved |
| `grep -c "Uf\[" src/dftorch/_coulomb_matrix.py` | 8 (criterion: >= 7) |
| `grep -n "pair_mask_..." src/dftorch/_coulomb_matrix.py` | no lines - no dead mask left behind |
| `grep -n "_require_no_f_shell_resolved_coulomb" src/dftorch/_coulomb_matrix.py` | no lines |
| `inspect.getsource(ewald_real_space_vectorized_sr).count('index_add_')` | **32** - sixteen blocks x (matrix + derivative) |
| ASCII byte scan of the new module | exit 0 |
| `grep -rn "FShellResolvedCoulombUnsupportedError" src/dftorch/` | the class definition plus one docstring mention; **no `raise`** |
| `test_no_new_f_exception_class_was_defined` diff | the function body is untouched; the only occurrence of its name in the diff is a comment that cites it |
| "retired" in both documents | F-SUPPORT-STATUS.md 3 mentions, ORBITAL-COUNT-INVENTORY.md 2 |
| `ruff check` on all changed files | clean |
| Post-commit deletion check on all three commits | no files deleted |

## Line-number and inventory corrections

The plan asked that cited line numbers be confirmed and corrections recorded.

| Claim | Cited | Observed |
|---|---|---|
| `FShellResolvedCoulombUnsupportedError` definition | `docs/F-SUPPORT-STATUS.md` said `_slater_koster_pair.py:289` | **291**. The plan predicted this was two lines off and it was; the document now says 291. |
| The guard, the eight pair masks, the nine blocks | `_coulomb_matrix.py` 690-719, 815-824, 836-1079 | all held exactly as the plan described |

**The orbital-count inventory lost 17 rows, not 1.** The plan's task 2 anticipated updating "the row whose Test column named the removed guard case". The real count is the whole `_coulomb_matrix.py` section: one `guarded` row for the deleted guard's `counts == 16`, plus sixteen `unreachable` rows for the eight named pair masks (two `max_ang` comparisons each). The sweep now returns **zero** records for that file and **141** in total, down from 158; prose mentions are **24**, down from 27. The summary table, the total, the flattening measurement and the prose that cited `_coulomb_matrix.py:816-824` were all updated to match.

**The rows were deleted, not re-dispositioned, and that contradicts the plan's letter.** The plan said "Keep the row rather than deleting it: the inventory's value is that it accounts for every site, including the ones that were resolved." That is not possible: `test_inventory_covers_every_site` matches sites and rows as multisets in **both** directions, so a row whose site no longer exists turns the suite red. The intent was honoured instead of the letter - the section heading stays, and under it a prose account of exactly what the 17 rows recorded, why they are gone, and what replaced them.

**`tests/data_skf_mio-1-1` has no `P-Zn.skf` or `Zn-P.skf`.** Found while writing `test_orbital_groups_are_a_contiguous_run_from_s`, which needs every element of a fixture set. A single geometry naming all seven mio elements cannot be built, so the test builds one `Constants` per element on its homonuclear pair file, and discovers the element list from the `.skf` file names rather than hardcoding it.

## Decisions Made

1. **One uniform mask rule, verified before it was trusted.** The mapping from each old named-mask sum to its `(l_i, l_j)` pair was worked through by hand for all nine blocks and is reproduced in `_shell_pair_mask`'s docstring as a table, so a later reader can re-check it without the plan. The numerical confirmation is the methane control plus the reduction identity.
2. **Sixteen explicit blocks, not a loop over a table of sixteen.** A loop would be shorter and would make "a block whose mask never matches" structurally impossible. It was rejected because the nine pre-existing blocks then get rewritten wholesale, and the strongest available evidence that an f-free calculation is unchanged is that each of those blocks has a one-line diff. The uniformity is enforced by the blocks being generated identically rather than by a runtime loop.
3. **The damping form is chosen by strength equality.** See deviation 1. Choosing by element identity is a proxy that is exact for the four diagonal blocks and wrong for the twelve others.
4. **The exception class is kept with a retirement note.** Deleting the name would shrink the f support taxonomy from four to three, break `test_no_new_f_exception_class_was_defined`, and lose the record of what used to be unsupported. Keeping a name nothing raises is only honest if that is checked, which is what the new watch test is for.

**Reversibility, as promised in the plan:** costly but mechanical in both directions. Restoring the previous builder means restoring eight named masks and deleting seven blocks, fully specified by the mapping table now living in the source. Restoring the refusal means restoring one guard, three tests and two document sections, each named above.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The short-range damping term divided by zero for a same-element pair with equal shell strengths**

- **Found during:** Task 1, writing `test_equal_group_strengths_reproduce_the_per_atom_matrix`. The plan's own main gate requires running it on a europium-europium pair with every shell strength equalised, which is precisely the degenerate case.
- **Issue:** `coul_diff_elem_and_ang` computes `SB = TJ4*Ti / (2*(Ti**2 - Tj**2)**2)`. With `Ti == Tj` that denominator is 0. Confirmed directly: `coul_diff_elem_and_ang(tensor([0.5, 0.5]), tensor([0.5, 0.7]), tensor([2.0, 2.0]))` returns `(tensor([nan, 0.3265]), tensor([nan, -0.2427]))`. The nine old blocks selected between the two closed forms by testing whether the two atoms were the same *element*, and only the three diagonal blocks did even that; the six off-diagonal ones always took the different-element branch. For a pair of atoms of one element the s-p block reads that element's s and p strengths, and a non-extended SKF file writes one Hubbard U per shell which are usually identical. `tests/f_orbital_data/N-N.skf` carries Us = Up = Ud = 0.490 Ha, so an N2 molecule with `MAGNETIC_HUBBARD_LDEP` set produced NaN inside a finite-looking, correctly shaped matrix - the same failure class as the empty f blocks, arriving from the other direction.
- **Fix:** `_coul_shell_pair_term(Ti, Tj, dR)` dispatches on `Ti == Tj` rather than on element identity, and is used by all sixteen blocks. `coul_same_elem_and_ang` is the limit of `coul_diff_elem_and_ang` as the two exponents approach each other, so this only ever replaces a non-finite value with the value it was the limit of. For a same-element pair in a diagonal block the two conditions coincide exactly (both sides read the same table for the same element), so every pair that produced a finite number before produces the identical number now.
- **Why not deferred:** the plan's `<action>` says "Do not touch anything else inside the nine blocks - not their damping calls". This is the one place that instruction was not followed, because following it would have made the plan's own main correctness gate impossible to run on the europium-europium pair it names. The prohibition it exists to protect - "the nine pre-existing blocks are not rewritten into a form that changes what an f-free molecule computes" - is satisfied in substance: the only f-free outputs that change are NaN ones.
- **Files modified:** `src/dftorch/_coulomb_matrix.py`
- **Regression cover added:** `test_a_pair_with_equal_shell_strengths_is_finite`, which asserts N's s and p Hubbard U are still equal before proving the matrix is finite - so the test says so out loud if the fixture ever changes and it stops proving anything.
- **Committed in:** `8068230`

**2. [Rule 3 - Blocking] `tests/data_skf_mio-1-1` cannot build a structure naming all its elements**

- **Found during:** Task 1, `test_orbital_groups_are_a_contiguous_run_from_s`
- **Issue:** `P-Zn.skf` and `Zn-P.skf` do not exist in the fixture set, so `Constants` raises `FileNotFoundError` for any geometry containing both P and Zn. The plan's phrasing ("for every element present in ... `tests/data_skf_mio-1-1`") assumed a complete pair matrix.
- **Fix:** one `Constants` per element, on its own homonuclear pair file, with the element list discovered from the `.skf` file names. The test also asserts it checked at least 9 elements, so a name scan that stops matching fails rather than passing vacuously.
- **Files modified:** `tests/test_shell_resolved_coulomb_f.py`
- **Committed in:** `7f386be`

**3. [Rule 3 - Blocking] The 17 inventory rows had to be deleted rather than re-dispositioned**

- Recorded in full under "Line-number and inventory corrections" above. The plan's instruction to keep the row is incompatible with `test_inventory_covers_every_site`.

### Scope note

`ruff format` was run on the changed files and reformatted four of them well beyond this plan's edits (`ESDriver.py` alone showed 76 changed lines against a 6-line comment edit). That churn was **reverted** and the edits re-applied by hand: pre-existing formatting drift is out of this plan's scope, and mixing it in would have made the diff unreviewable. `ruff check` is clean on every changed file.

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking). **Impact:** deviation 1 is the only one that changed production behaviour beyond the plan, and it removes a NaN. None expanded scope.

## Observation recorded, not fixed

`docs/F-SUPPORT-STATUS.md` section 2 claimed a completeness gate at
`tests/test_support_documentation.py::test_every_named_exception_appears_in_support_matrix`,
and **no test of that name exists in the repository**. `tests/test_support_documentation.py` defines four tests and none of them is it, so that document has no completeness gate. This is the same class of stale claim already logged in `STATE.md` against `docs/LIBRARY-OUTPUT-INVENTORY.md`; it belongs to whoever owns the documentation gates, not to this phase. The paragraph now states the gap in place of the claim, and points at `test_no_new_f_exception_class_was_defined` as what does hold: a *fifth* exception class fails the suite, but a fourth-class row going missing from that table would not.

## Prohibition audit

All four `must_haves` prohibitions were `flagged-unverified` at plan time. All four check out:

1. **No matrix entry is a reference number.** The only numeric literals in the new module are separations (3.0, 2.655, 1.1), tolerances (1e-10, 1e-12), shell and atom counts, and the cutoff. The two historical row sums `[0.473581, 0, 0, 0, 0.473581, 0]` appear only inside failure-message and docstring prose in `test_shell_resolved_u.py`, as the record of the defect, never as an assertion target.
2. **No fifth f exception class.** `test_no_new_f_exception_class_was_defined` passes unedited; the diff for `tests/test_orbital_count_guards.py` does not touch its body.
3. **No refusal removed while a document still claims it fires.** Source and both documents changed in the same commit, `e988c05`.
4. **The nine pre-existing blocks reproduce exactly.** Their pair selection is the same set (mapping table verified by hand, then numerically by the reduction identity on methane), their strength tables and index arithmetic are byte-identical, and the only damping change converts NaN to its limit. Evidence: `test_single_shot_energy.py`, `test_simple_format_regression.py` and `test_scf.py` all green, and the whole suite at 234/0.

## Threat coverage

| Threat | Disposition |
|---|---|
| T-06-07 (a block added whose mask never matches) | `test_all_seven_f_blocks_are_populated` addresses each of the seven positions individually and names which was empty; `test_eu_eu_matrix_has_no_empty_row_or_column` catches it from the other direction |
| T-06-08 (the uniform rule changing an f-free result) | mapping table verified by hand; methane control; reduction identity on methane; full suite green |
| T-06-09 (a row or column offset landing on the neighbouring atom) | the reduction identity compares **every** off-diagonal entry against the per-atom oracle, which only holds if every offset is right; `test_orbital_groups_are_a_contiguous_run_from_s` covers the structural assumption underneath |
| T-06-10 (a refusal removed while a document claims it fires) | one commit, both documents, plus the no-raise watch test |
| T-06-11 (the taxonomy shrinking to three names) | class kept; the pinning test passes with an untouched body, checked by reading the diff and not only the exit code |
| T-06-12 (a rewritten message leaking a filesystem path) | the message-content test was converted rather than deleted and now reads the constant directly; it still asserts no `SKFPATH`, no `.skf`, no `.xyz` |
| T-06-SC (package-manager installs) | not applicable - nothing was installed |

## Threat Flags

None. The change adds no network endpoint, no authentication path, no file access pattern and no schema at a trust boundary; it is arithmetic inside one function.

## Next Phase Readiness

**Ready for plan 06-03.** `structure.C_sr` and `structure.dCC_sr` are now populated for an f system whenever `MAGNETIC_HUBBARD_LDEP` is set, with every block real. 06-03's job - threading per-orbital-group charges through the charge loop so something can consume the matrix - is unblocked; the refusal that would have stopped it dead is gone.

Carried forward, deliberately:

- **The matrix is built and still unconsumed.** `energy()` and `SCFx` take `(Nats, Nats)` with per-atom charges; D-11 defers the charge threading and this plan did not touch it. `structure.C` is unchanged and remains what the energy path reads.
- **`Constants.py:232` is untouched.** The per-atom path still charges every element the Hubbard U of its s group. This plan built the remedy; plan 06-04 records the defect itself.
- **The equal-strength fix is broader than f.** Any homonuclear molecule of an element with coinciding shell strengths was producing NaN on the shell-resolved path, N2 included. That path had no production consumer, so nothing shipped a wrong number - but the same dispatch question will arise again wherever `coul_diff_elem_and_ang` is called with two independently chosen strengths.
- **`docs/F-SUPPORT-STATUS.md` has no completeness gate.** Recorded above, not fixed.

## Self-Check: PASSED

All five files claimed above exist on disk (`tests/test_shell_resolved_coulomb_f.py`, `src/dftorch/_coulomb_matrix.py`, `docs/F-SUPPORT-STATUS.md`, `docs/ORBITAL-COUNT-INVENTORY.md`, this summary); all three task commits (`7f386be`, `8068230`, `e988c05`) are present in `git log`.

---
*Phase: 06-self-consistent-scf-for-f-systems*
*Plan: 02*
*Completed: 2026-08-05*

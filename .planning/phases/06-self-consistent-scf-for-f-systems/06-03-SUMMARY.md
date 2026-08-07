---
phase: 06-self-consistent-scf-for-f-systems
plan: 03
subsystem: scf
tags: [dftb, scf, charge-mixing, shell-resolved, hubbard-u, f-orbitals, pytest, torch]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "the Eu-N diatomic fixture, the shell-resolved data structures (Hubbard_U_sr, shell_types, el_per_shell, n_shells_per_atom, H_INDEX_START_U), and the MAGNETIC_HUBBARD_LDEP gate (D-16/D-23) this plan reuses rather than replacing"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 01
    provides: "the Krylov-off-for-f interim, and scf_iter_count as the fifteenth return element of all four charge loops"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 02
    provides: "all sixteen off-site angular blocks of the shell-resolved Coulomb matrix, and the 234-passed suite baseline this plan had to hold"
provides:
  - "A closed-shell charge loop that tracks and feeds back one charge per orbital group, selected by the MAGNETIC_HUBBARD_LDEP key that already existed for it"
  - "structure.q_sr, the converged per-orbital-group charges; None when the switch is off"
  - "onsite_shell_pair_coulomb / onsite_shell_coulomb_matrix: the same-atom, different-group Coulomb block the neighbour-list builder structurally cannot produce, and without which the Eu-N loop does not settle"
  - "structure.C_sr_onsite and structure.C_sr_scf = C_sr + C_sr_onsite, built once and handed to both the loop and energy()"
  - "energy() with C_sr/q_sr/U_sr on an all-or-nothing rule, so the reported repulsion energy is the one the loop converged to"
  - "Loud refusal for PME, third-order DFTB3 and GBSA, which cannot supply a shell-resolved matrix"
affects: [06-04, 06-05, shell-resolved-charge, forces]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All-or-nothing optional argument groups: N arguments together mean yes, none means no, any partial combination raises naming the missing ones - never falls through to the coarser arm"
    - "Validate a new numerical path by a degenerate-parameter reduction to an existing one (equal shell strengths collapse the finer loop into the coarser loop), not by freezing its output"
    - "A closed form introduced for a limit is checked against the codebase's own function evaluated near that limit, rather than against a hand-derived number"
    - "Derive the coarse quantity from the fine one every pass, so the two cannot drift and every existing consumer keeps reading its own attribute"

key-files:
  created: []
  modified:
    - src/dftorch/_scf.py
    - src/dftorch/_energy.py
    - src/dftorch/ESDriver.py
    - src/dftorch/_coulomb_matrix.py
    - tests/test_shell_resolved_scf_f.py
    - tests/test_orbital_count_guards.py
    - docs/ORBITAL-COUNT-INVENTORY.md
    - docs/LIBRARY-OUTPUT-INVENTORY.md

key-decisions:
  - "The off-site shell-resolved matrix alone is not a Coulomb operator. The neighbour list never pairs an atom with itself, so the interaction between two orbital groups of one atom - the largest either group feels - was absent. Supplied as its own matrix, kept separate from C_sr so C_sr keeps exactly the meaning plan 06-02 gave it."
  - "The Krylov accelerator is off at the finer resolution unconditionally, not just for f systems. It preconditions with the per-atom Coulomb matrix and per-atom Hubbard U, which is precisely the silent per-atom substitution this plan's design refuses."
  - "A per-atom q_init is ignored at the finer resolution and the loop says so, because splitting one number across an atom's four groups would be inventing information."
  - "The plan's Task 2 was already committed inside the RED commit, so no second commit was faked to satisfy the requested split."

requirements-completed: [SCC-03]

coverage:
  - id: D1
    description: "The closed-shell loop tracks charge one number per orbital group per atom and feeds those numbers back, rather than one lumped number per atom"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_eu_n_settles_with_per_group_charge"
        status: pass
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_per_group_charges_have_one_entry_per_group"
        status: pass
    human_judgment: false
  - id: D2
    description: "Europium's f group is charged at the f table's own strength, not at its s group's, and the per-atom strength is pinned in place as a recorded decision rather than quietly changed"
    requirement: SCC-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_scf_f.py#test_europium_f_group_is_charged_at_the_f_rate"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_scf_f.py#test_the_per_atom_strength_still_comes_from_the_s_group"
        status: pass
    human_judgment: false
  - id: D3
    description: "The two resolutions are two views of one answer: summing the per-group charges over each atom reproduces structure.q, and the total stays conserved"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_per_group_charges_sum_to_the_per_atom_charges"
        status: pass
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_total_charge_is_still_conserved"
        status: pass
      - kind: unit
        ref: "tests/test_shell_resolved_scf_f.py#test_reference_occupations_and_nuclear_charges_agree"
        status: pass
    human_judgment: false
  - id: D4
    description: "The finer description is genuinely consumed - the converged answer with it on differs from the answer with it off, which is the exact failure D-14 left behind"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_per_group_charge_actually_changes_the_answer"
        status: pass
    human_judgment: false
  - id: D5
    description: "With one strength shared by every orbital group the finer loop IS the coarser loop, and the two converged charge vectors agree to 1e-8 - a derived identity that validates the gathers, the offsets, the same-atom coupling and the mixing at once"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_equal_group_strengths_reproduce_the_per_atom_answer"
        status: pass
    human_judgment: false
  - id: D6
    description: "With the switch off every calculation takes the path it took before this plan, and an f-free molecule runs correctly at the finer resolution"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_switch_off_leaves_no_per_group_charges"
        status: pass
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_f_free_molecule_runs_at_the_finer_resolution"
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
  - id: D7
    description: "A configuration that cannot supply the finer matrix refuses out loud, naming the parameter keys and no filesystem path, instead of returning the per-atom answer"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_per_group_request_refuses_when_no_matrix_can_be_built"
        status: pass
    human_judgment: false
  - id: D8
    description: "The loop and energy() compute the electron-repulsion energy at the same resolution, so structure.e_coul describes the charge state that was actually converged to"
    requirement: SCC-03
    verification:
      - kind: integration
        ref: "tests/test_shell_resolved_scf_f.py#test_the_reported_energy_is_the_one_the_loop_converged_to"
        status: pass
    human_judgment: false
  - id: D9
    description: "The same-atom coupling is bounded by the two strengths it joins, is symmetric, has a zero diagonal, and never reaches another atom"
    requirement: SCC-03
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_scf_f.py#test_same_atom_group_coupling_lies_between_the_two_strengths"
        status: pass
    human_judgment: false

# Metrics
duration: ~150min across two sessions
completed: 2026-08-06
status: complete
---

# Phase 6 Plan 03: Per-Orbital-Group Charge in the Closed-Shell Loop Summary

**Europium's seven f electrons are now charged at the f group's own 13.61 eV instead of at its s group's 5.71 eV, because the charge loop tracks and feeds back one charge per orbital group - proven by collapsing the finer loop into the coarser one rather than by freezing any number either produced.**

## Performance

- **Duration:** ~150 min across two sessions (the first was cut off by a session limit after the RED commit)
- **Completed:** 2026-08-06
- **Tasks:** 2
- **Files:** 8 modified, 0 created in this session (the test module was created by the RED commit). No file deleted in either commit.

## What the handoff said, and what the repository actually held

The continuation brief was checked against the repository before anything was trusted, and it was wrong in one material way.

| Handoff claim | Repository |
|---|---|
| Task 1 RED committed as `93f2ef8`, 659 lines | **Correct.** |
| Task 1 GREEN written but uncommitted | **Correct.** |
| Task 2 "believed written but UNCOMMITTED; not verified" | **Wrong. Task 2 was already committed, inside the RED commit.** All three of its tests - `test_europium_f_group_is_charged_at_the_f_rate`, `test_the_per_atom_strength_still_comes_from_the_s_group`, `test_the_two_resolutions_charge_the_same_total_electrons` - are present in `git show 93f2ef8:tests/test_shell_resolved_scf_f.py`. |
| Uncommitted work touches 7 files, 804 insertions | **Close but understated.** 7 files, 822 insertions before this session's own fixes. |

**Consequence for the requested commit split.** The brief asked for two commits, Task 1 GREEN then Task 2. That split does not exist: Task 2 is test-only and its tests were already in the RED commit. What remained was one indivisible unit - the source implementation plus the three regression tests that cover the deviation described below, which test a function that does not exist without that source. It was committed as one commit rather than split artificially.

## Task Commits

1. **Task 1: Track and feed back charge one orbital group at a time** (TDD)
   - `93f2ef8` (test, prior session) - 13 tests written first, 7 failing for their intended reasons. Also carried Task 2's three tests.
   - `cef305d` (feat, this session) - the implementation, plus three tests covering the deviation.
2. **Task 2: Prove europium's f electrons are now charged at the f rate** - no separate commit; its tests shipped in `93f2ef8`. See above.

## Accomplishments

- **The finer resolution is genuinely consumed, and that is measured rather than asserted.** `test_per_group_charge_actually_changes_the_answer` runs the same molecule twice and requires the converged per-atom charge vectors to differ. A test that only checked `q_sr` exists would have passed in exactly the state Phase 4 decision D-14 left behind.
- **The main correctness gate freezes no number.** `test_equal_group_strengths_reproduce_the_per_atom_answer` overrides the p, d and f Hubbard tables in memory so every group of every element carries its element's s strength. The finer loop is then *algebraically* the coarser one, so the two converged charge vectors must agree - measured to better than 1e-8. This one test validates the group-to-orbital gathers, the row and column offsets, the same-atom coupling, the residual, the mixing and the repulsion energy simultaneously.
- **`structure.q` never stopped being populated.** At the finer resolution it is derived from `q_sr` every pass by summing over each atom, so the dipole term, the reported charges, plan 06-05's graph and every existing test read the attribute they always read. Nothing downstream changed.
- **A missing physics term was found and supplied** (see Deviations). Without it the Eu-N loop does not settle at all.
- **Full suite: 250 passed, 0 failed** (was 234). +16 from the new module, which contributes 15 test functions, one of them parametrised over two fixtures.

## Files Created/Modified

- `src/dftorch/_scf.py` (+281/-24) - five keyword-only parameters on `SCFx`, the per-orbital-group branch through the whole loop body, a sixteenth return element, and three refusals.
- `src/dftorch/_coulomb_matrix.py` (+115) - `onsite_shell_pair_coulomb` and `onsite_shell_coulomb_matrix`. Beyond the plan; see deviation 1.
- `src/dftorch/ESDriver.py` (+116/-13) - the two plan-named refusals, `structure.C_sr_onsite` / `C_sr_scf` / `q_sr`, and the conditional keyword passing to `SCFx` and `energy()`.
- `src/dftorch/_energy.py` (+52/-1) - `C_sr`, `q_sr`, `U_sr` on an all-or-nothing rule.
- `tests/test_shell_resolved_scf_f.py` (+266/-6 in this session; 659 lines created in the RED commit) - 15 test functions, 16 collected cases. Pure ASCII, verified by byte scan.
- `docs/LIBRARY-OUTPUT-INVENTORY.md`, `docs/ORBITAL-COUNT-INVENTORY.md`, `tests/test_orbital_count_guards.py` - two documentation gates kept green; see deviations 2 and 3.

## Verification Evidence

| Gate | Result |
|---|---|
| `uv run pytest` (whole suite) | **250 passed, 0 failed** in 115 s - above the 234 baseline |
| `uv run pytest tests/test_shell_resolved_scf_f.py -x` | 16 passed |
| `uv run pytest tests/test_single_shot_energy.py tests/test_simple_format_regression.py tests/test_scf.py` | green inside the full run - nothing on the per-atom path moved |
| `inspect.signature(SCFx)` | lists `shell_types`, `n_shells_per_atom`, `el_per_shell`, `Hubbard_U_sr`, `C_sr` |
| `inspect.signature(energy)` | lists `C_sr`, `q_sr`, `U_sr` |
| `grep -c "structure.q_sr" src/dftorch/ESDriver.py` | 5 (criterion: >= 2) |
| `git diff src/dftorch/Constants.py` | empty - line 232 untouched, as Task 2 requires |
| ASCII byte scan of the test module | exit 0 |
| Typed-in strength values in the test module | 2 occurrences, both inside docstring prose explaining the scale; the criterion allows these and forbids them in assertions. None is in an assertion. |
| Post-commit deletion check on `cef305d` | no files deleted |

**`grep -rn "SHELL_RESOLVED" src/dftorch/` returns 3 source lines, not the zero the plan's criterion predicted.** The criterion's intent - no new parameter key - holds; its literal form does not. The three hits are `ESDriver.py:187`, a docstring stating explicitly that no competing `SHELL_RESOLVED` flag is being introduced; `_scf.py:161`, the private module constant `_SHELL_RESOLVED_ARGUMENT_NAMES` listing the five *function arguments* so the all-or-nothing check and its error message cannot drift apart; and `_slater_koster_pair.py:330`, plan 06-02's retired message constant. No new key is read from `dftorch_params`: `MAGNETIC_HUBBARD_LDEP` is still selected through `_select_coulomb_hubbard` and nothing else.

## Prohibition audit (decision D-6.08)

**No converged charge, per-group charge or energy is written into a test as a reference number.** Checked mechanically rather than by eye: every numeric literal inside every `assert` expression in the module was extracted with `ast`, and every module-level constant was read. The complete list is tolerances (`1e-6`, `1e-8`, `1e-9`, `1e-10`, `1e-12`), the input separation 2.655 A, the iteration cap 100, the `-1` non-convergence sentinel, tuple indices (`14` for `scf_iter_count`, `15` for `q_sr`), the angular label `4` marking the f group, and the factor `2.0` in the inequality the plan itself mandates. Every assertion is a shape, a sum, an inequality, a table equality, or a difference between two runs.

The other three prohibitions hold as well: no shell-resolved request is served by the per-atom matrix (three refusals, one of them driven by test); no new parameter key and no fifth `F*` class (plain `NotImplementedError`, per the Phase 5 precedent); and the per-atom path is neither deleted nor made conditional - it is still what runs when the five arguments are absent, which is the default.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] The off-site matrix alone is not a Coulomb operator: the same-atom, different-group interaction was absent**

- **Found during:** Task 1, by the previous session's executor; verified independently in this session before being trusted.
- **Issue:** `ewald_real_space_vectorized_sr` is driven with a neighbour list that never pairs an atom with itself, so `structure.C_sr` carries no interaction between two orbital groups of the *same* atom - and that is the largest interaction either group feels. The per-atom path has the same hole and fills it one line later with `0.5 * sum(q**2 * Hubbard_U)`, but that term covers each charge only against *itself*. With one charge per atom that is all that is needed; with one charge per group it is not. The plan's design section assumed `C_sr` could be handed to the loop as-is, and it cannot.
- **Independent verification, run in this session rather than taken on trust:** the Eu-N loop was driven twice, once with the on-site block and once without, holding everything else fixed.

  | Matrix | `scf_iter_count` | converged `q_sr` |
  |---|---|---|
  | `C_sr + C_sr_onsite` | **80** (settled) | `[-1.4159, 0.2742, 0.7873, -0.1443, -0.0344, 0.5331]` |
  | `C_sr` alone | **-1** (gave up) | `[-2.0054, -0.0193, 10.0031, -7.0002, 0.028, -1.0063]` |

  The failing run's charges are the integer-filling oscillation the fix's docstring predicts: 10.00 electrons swinging into the d group and -7.00 out of the f group. The term is load-bearing, not decorative.
- **Fix:** `onsite_shell_pair_coulomb(Ui, Uj)` returns the R -> 0 limit of the same short-range form `_coul_shell_pair_term` uses off-site, dispatched on strength equality for exactly the divide-by-zero reason plan 06-02 recorded. `onsite_shell_coulomb_matrix` places it on same-atom, different-group entries only. `structure.C_sr_scf = C_sr + C_sr_onsite` is built once in the driver and handed to both the loop and `energy()`, so the two cannot end up at different resolutions (threat T-06-16).
- **Verified numerically against the codebase's own function, not against a hand-derived number:** `KECONST * (1/R - _coul_shell_pair_term(...))` was evaluated at R = 1e-5 A for five strength pairs and matched the closed form to 6e-11 or better - including the equal-strength case, where it returns `U` exactly, which is what makes the equal-strength reduction identity hold.
- **Why not deferred:** without it the plan's own `<done>` criterion ("Europium-nitrogen settles at 2.655 A while tracking charge per orbital group") is unreachable.
- **Files modified:** `src/dftorch/_coulomb_matrix.py`, `src/dftorch/ESDriver.py`
- **Regression cover added:** `test_same_atom_group_coupling_lies_between_the_two_strengths` (bounded, symmetric, zero diagonal, never reaches another atom) and `test_the_reported_energy_is_the_one_the_loop_converged_to`.
- **Committed in:** `cef305d`

**2. [Rule 3 - Blocking] The new `print(` broke `test_inventory_covers_every_print`**

- **Found during:** the first full-suite run of this session. This was the only red test, and it was a genuine gap, not a flake.
- **Issue:** the implementation added one gated notice to `_scf.py` (the per-atom `q_init` cannot be honoured at the finer resolution and is being ignored). `docs/LIBRARY-OUTPUT-INVENTORY.md` had 181 rows and the package now ships 182 `print(` occurrences. Plan 05-05 built this gate for precisely this event.
- **Fix:** one row added, classified `status` / `R-STATUS`, gated by `_lib_out`. The classification is a judgement worth stating so it can be disagreed with: the notice was **not** classified `warning` / `R-DEGRADE`, because ignoring an initial guess changes only where an iterative loop *starts*, not the fixed point it lands on - it is not "different physics or a different method than requested", which is what that class is reserved for.
- **Files modified:** `docs/LIBRARY-OUTPUT-INVENTORY.md`
- **Committed in:** `cef305d`

**3. [Rule 3 - Blocking] One added import line shifted three enforced line numbers**

- **Found during:** the previous session; carried through and verified here.
- **Issue:** `docs/ORBITAL-COUNT-INVENTORY.md` records exact line numbers for `ESDriver.py`'s three orbital-count comparison sites, and `test_prose_mentions_are_excluded_and_counted` asserts the swept set equals `{60, 95, 164}`. Importing `onsite_shell_coulomb_matrix` pushed all three down by one.
- **Fix:** inventory rows and the test's expected set updated to `{61, 96, 165}`. The test body's *property* is unchanged - it still asserts that only the executable comparisons appear and the f-string message text does not.
- **Files modified:** `docs/ORBITAL-COUNT-INVENTORY.md`, `tests/test_orbital_count_guards.py`
- **Committed in:** `cef305d`

**4. [Rule 2 - Missing critical functionality] Three refusals the plan did not name, and one it named in a second place**

- The plan named two refusals (PME, third-order DFTB3) in `ESDriver.forward`. Both are there. **GBSA solvation** was added as a third, in `SCFx`, for the same reason: its shifts are per-atom constructions with no shell-resolved counterpart, so serving the request would mix resolutions inside one energy.
- All three are also raised **in `SCFx` itself**, not only in the driver. `SCFx` is importable and is called directly by this plan's own reduction-identity test, so a refusal that lived only in the driver would not protect a direct caller.
- **The Krylov accelerator is switched off at the finer resolution**, unconditionally rather than only for f systems. It preconditions with the per-atom Coulomb matrix and per-atom Hubbard U and has no shell-resolved counterpart; running it there would be exactly the silent per-atom substitution this design refuses. This changes how the loop travels, never where it lands - the fixed point is set by `C_sr` and `Hubbard_U_sr` either way, which is what the equal-strength identity confirms.
- **Committed in:** `cef305d`

### Scope note

`ruff` is not installed in this environment (`uv run ruff check` reports `program not found`), so no lint gate was run. Per the environment brief, no repo-wide `ruff format` was attempted; plan 06-02 recorded why.

---

**Total deviations:** 4 auto-fixed (0 bugs, 2 missing-functionality, 2 blocking). **Impact:** deviation 1 is the only one that adds production physics beyond the plan, and without it the plan's stated goal is unreachable. None expanded scope beyond this plan's files.

## Known limitations, carried forward deliberately

- **A per-atom `q_init` is ignored at the finer resolution.** A single number per atom carries no information about how that charge splits across the atom's four groups, so it cannot be honoured without inventing information. The loop starts from a reference diagonalization instead and prints a notice. This matters for MD and geometry optimisation, where `q_init` normally carries the previous step's converged charges as a warm start: at the finer resolution every step restarts cold. Recorded in `SCFx`'s docstring.
- **The Eu-N loop settles in 80 passes against a cap of 100.** That is a thin margin. It is a real convergence, not a near-miss - but a harder f fixture, or the Krylov repair landing without a shell-resolved counterpart, could put it over. Worth watching in plan 06-05's scan.
- **`Constants.py:232` is still untouched**, so the per-atom path continues to charge every element the Hubbard U of its s group. That is deliberate and now pinned by `test_the_per_atom_strength_still_comes_from_the_s_group`, so a future edit there is a recorded decision rather than a tidy-up. Plan 06-04 writes the defect down in prose.
- **Batched and open-shell paths are untouched.** `SCFx_batch`, `scf_x_os` and `delta_scf_x_os` have no per-orbital-group path; `energy()` refuses the shell-resolved arguments for batched input.

## Observation recorded, not fixed

**The `Line` column of `docs/LIBRARY-OUTPUT-INVENTORY.md` is stale for eight files, and only one of them is this plan's doing.** Measured 2026-08-05: `ESDriver.py`, `MD.py`, `Optimizer.py`, `_coulomb_matrix.py`, `_coulomb_matrix_batch.py`, `_kernel_fermi.py`, `_scf.py` and `_xl_tools.py` all carry numbers from the end of plan 05-05. Only the row *count* is gated by a test, so the numbers have drifted unnoticed as the package grew - `MD.py`, `Optimizer.py`, `_kernel_fermi.py` and `_xl_tools.py` drifted from work that predates Phase 6 entirely. Regenerating the whole column was rejected as out of scope: it would fix other phases' drift inside this plan's diff, which is the churn plan 06-02 was explicitly burned by. Instead the document now states the drift out loud, tells a reader to match rows by printed text rather than by line number, and marks the one row carrying a current number with a trailing `*`.

**The `### Totals` block of the same document was stale before this plan touched it.** It read `warning: 22, total: 182` against a table of 181 rows; plan 05-06 retired a `warning` row and updated the header prose but not the totals. Recounted mechanically from the rows and corrected to `warning: 21`, `status: 57`, total 182, with both corrections attributed in the document itself. This one *was* fixed rather than logged, because leaving a document internally contradictory while editing it is worse than the small out-of-scope delta.

## Threat coverage

| Threat | Disposition |
|---|---|
| T-06-13 (a shell-resolved request served silently by the per-atom matrix) | Three `NotImplementedError` raises, in both `ESDriver.forward` and `SCFx`; `test_per_group_request_refuses_when_no_matrix_can_be_built` drives the PME case and checks the message names both keys and no path |
| T-06-14 (`structure.q` silently ceasing to be populated) | Derived from `q_sr` every pass; `test_per_group_charges_sum_to_the_per_atom_charges` asserts agreement to 1e-10; every existing per-atom module stays green |
| T-06-15 (the finer data built and never reaching the answer - the D-14 state) | `test_per_group_charge_actually_changes_the_answer` requires the two converged answers to differ |
| T-06-16 (loop and `energy()` at different resolutions) | One matrix `C_sr_scf` built once and handed to both; `energy()` raises `ValueError` on a partial request; `test_the_reported_energy_is_the_one_the_loop_converged_to` recomputes the reported energy from the converged charges |
| T-06-17 (the sixteenth return element shifting a positional unpack) | Appended last, after plan 06-01's fifteenth; one production call site; full suite green at 250 |
| T-06-18 (a converged value frozen into a test) | Audited mechanically with `ast` - see the prohibition audit above |
| T-06-19 (a refusal message leaking a filesystem path) | All three messages name parameter keys only; checked by the PME refusal test |
| T-06-SC (package-manager installs) | Not applicable - nothing was installed |

## Threat Flags

None. The change adds no network endpoint, no authentication path, no file access pattern and no schema at a trust boundary; it is arithmetic inside one loop plus two documentation-gate updates.

## Next Phase Readiness

**Ready for plans 06-04 and 06-05.** `structure.q_sr` is populated and converged for an f system whenever `MAGNETIC_HUBBARD_LDEP` is set, `structure.q` is unchanged in meaning for every existing consumer, and `structure.e_coul` is built at the same resolution the loop minimised.

- **06-04** records `Constants.py:232` in prose. That defect is now pinned in place by a test that says so explicitly and points at 06-04's record.
- **06-05** reads `structure.q` for its graph. That attribute is populated in both modes and its meaning has not changed, so nothing in 06-05 needs to learn `q_sr` in order to work.

## Self-Check: PASSED

All eight modified files exist on disk, as does this summary. Both task commits are present in `git log`: `93f2ef8` (test, RED) and `cef305d` (feat, GREEN). The full suite was re-run to completion after the last edit and reported 250 passed in 115.35 s.

---
*Phase: 06-self-consistent-scf-for-f-systems*
*Plan: 03*
*Completed: 2026-08-06*

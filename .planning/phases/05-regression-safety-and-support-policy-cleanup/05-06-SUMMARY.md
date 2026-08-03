---
phase: 05-regression-safety-and-support-policy-cleanup
plan: 06
subsystem: orbital-count-audit-and-f-refusal-policy
tags: [d-04, reg-04, cln-03, orbital-count, guard, inventory, sweep, spin, forces]

# Dependency graph
requires:
  - phase: 05-regression-safety-and-support-policy-cleanup
    plan: 04
    provides: the disposition-inventory format, the "unreachable is established not asserted" rule, and the import search that proved _legacy has no importer
  - phase: 05-regression-safety-and-support-policy-cleanup
    plan: 05
    provides: the _coulomb_matrix_batch do_vec printed-apology site, handed over for disposition
  - phase: 04-scf-and-reference-simulation-validation
    provides: D-12 (spin refuses for f), D-23 (MAGNETIC_HUBBARD_LDEP is the shell-resolved gate), threat T-04-07's no-path-in-messages policy, and the run_with_float64 harness
  - phase: 03-h0-s-routing-and-f-angular-blocks
    provides: the four named F*Error classes and the _require_* guard template
provides:
  - "tests/orbital_count_sweep.py: sweep_orbital_count_sites, count_prose_mentions, flatten_source, pattern_for -- a re-runnable, formatting-invariant site enumerator over src/dftorch/**/*.py"
  - "docs/ORBITAL-COUNT-INVENTORY.md: 157 rows, one per swept record, each dispositioned extended/guarded/unreachable with evidence"
  - "tests/test_orbital_count_guards.py: 37 tests, including bidirectional inventory completeness and per-guard f reachability"
  - "_spin._require_no_f_spin_shells: refuses shell-resolved spin assembly when shell_types carries an f shell"
  - "_forces._require_no_f_spin_forces: refuses the spin force path for a 16-orbital atom"
  - "_coulomb_matrix_batch.ewald_k_space_vectorized: do_vec=True now raises instead of printing and returning None"
affects: [06-scf, 07-derivatives, 08-forces-stress, 08.1-batched-f, 09-md]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A documentation deliverable is gated by a bidirectional test: swept sites and written rows must match as multisets, and the failure message lists both directions separately because 'the inventory is stale' is useless without saying which way"
    - "A code sweep blanks comments and string literals via tokenize before matching, so prose mentions never become rows that need a dishonest disposition; on Python 3.12+ that requires blanking FSTRING_START/MIDDLE/END, not just STRING"
    - "An audit's identifier set is chosen from the defects the project has actually had, not from the obvious name: restricting to n_orb would have missed the max_ang site Phase 4 already had to fix"
    - "A guard test drives the real entry point with a real f system rather than calling the guard function, because a guard that is never called looks identical from outside"
    - "A claim that a protection is load-bearing is measured on live code and the number written down, including when the measurement partly disconfirms the claim"

key-files:
  created:
    - tests/orbital_count_sweep.py
    - docs/ORBITAL-COUNT-INVENTORY.md
    - tests/test_orbital_count_guards.py
  modified:
    - src/dftorch/_spin.py
    - src/dftorch/_forces.py
    - src/dftorch/_coulomb_matrix_batch.py
    - .planning/phases/05-regression-safety-and-support-policy-cleanup/deferred-items.md

key-decisions:
  - "The sweep matches three families, not one: orbital-count (1/4/9/16), shell-count (1/2/3/4) and canonical basis-layout literals. shell-count is beyond the plan's identifier list on purpose -- the Phase 4 shell-resolved Coulomb defect lived at a max_ang site and never mentions n_orb, so an n_orb-only sweep would have been a knowingly built blind spot."
  - "Comments and string literals are blanked before matching, so a docstring saying n_orb == 16 never becomes a row. 20 prose mentions were excluded at Task 1 and 27 after the guards were added; the number is reported by count_prose_mentions rather than left unstated."
  - "The only real gaps the audit found are three truncated copies of the per-shell AO count table, [0, 1, 3, 5], in _spin.get_h_spin, _spin.get_h_spin_diag and _forces.forces_spin, while Constants.shell_dim holds the correct [0, 1, 3, 5, 7]. All were guarded, not widened: D-12 defers spin-polarized f and tests/f_orbital_data ships no spinw.txt, so a correctly sized f block would be filled from parameters that do not exist."
  - "_forces.forces_spin raises FDerivativeUnsupportedError rather than the spin class, because it fails for a reason independent of spin: it consumes dS, which is exactly zero in every f block."
  - "_coulomb_matrix_batch's do_vec=True site is classified OUT of D-04's row set -- nothing in that branch reads n_orb, max_ang or the basis layout -- and resolved anyway, with the determination recorded, rather than forced into the audit."
  - "No new exception class was defined; test_no_new_f_exception_class_was_defined pins the taxonomy at exactly four."
  - "The plan-time claim that flattening changes the count was measured and found true only for a per-line sweep (9 vs 25 pair-level masks in _h0ands.py). Whole-text matching with \\s* separators already crosses newlines, so flattening is redundant for this module's own patterns; that is written into the sweep docstring and the inventory rather than left implied."

patterns-established:
  - "Bidirectional documentation gate: Counter(swept) vs Counter(rows), with missing and extra reported separately"
  - "Guard-case registry keyed by the id the inventory's Test column names, checked in both directions so neither a row without a case nor a case without a row can survive"
  - "Reaching a mid-function guard with the smallest synthetic tensors that get there (one batch, two atoms, one pair) rather than declaring it untestable and xfailing"

requirements-completed: [REG-04, CLN-03]

coverage:
  - id: D1
    description: "Every hardcoded orbital-count, shell-count and basis-layout site in src/dftorch is enumerated by a re-runnable sweep and carries a written disposition; the two match in both directions"
    requirement: "CLN-03"
    verification:
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_inventory_covers_every_site"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_inventory_row_count_matches_record_count"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_every_row_has_a_known_disposition_and_evidence"
        status: pass
    human_judgment: false
  - id: D2
    description: "No site is left silently falling through: no inventory row carries needs-action"
    requirement: "CLN-03"
    verification:
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_inventory_has_no_unresolved_sites"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every guarded site raises its named exception when a real f system reaches it, rather than merely having a guard function defined"
    requirement: "REG-04"
    verification:
      - kind: integration
        ref: "tests/test_orbital_count_guards.py#test_every_guarded_site_raises_for_f (8 cases)"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_every_guarded_row_names_an_exercised_case"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_every_guard_case_is_claimed_by_at_least_one_row"
        status: pass
    human_judgment: false
  - id: D4
    description: "No f-free calculation acquires a new failure mode: every guard is inert for CH4 + mio-1-1 and the f-free path still completes"
    requirement: "REG-04"
    verification:
      - kind: integration
        ref: "tests/test_orbital_count_guards.py#test_guards_ignore_f_free_systems (8 cases)"
        status: pass
      - kind: integration
        ref: "tests/test_orbital_count_guards.py#test_ch4_runs_end_to_end_after_the_guards"
        status: pass
      - kind: integration
        ref: "tests/test_simple_format_regression.py (11 tests)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The supported f path still runs end to end and reproduces its recorded reference curve"
    requirement: "REG-04"
    verification:
      - kind: integration
        ref: "tests/test_eu_n_scan.py (6 tests, 21-point curve)"
        status: pass
      - kind: integration
        ref: "tests/test_single_shot_energy.py (5 tests)"
        status: pass
      - kind: integration
        ref: "tests/test_orbital_count_guards.py#test_f_pair_reaches_single_system_sk_assembly"
        status: pass
    human_judgment: false
  - id: D6
    description: "The site at _coulomb_matrix_batch handed over by plan 05-05 has a disposition rather than a printed apology, and the determination is recorded"
    requirement: "REG-04"
    verification:
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_batched_kspace_site_has_a_disposition"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_inventory_records_the_batched_kspace_determination"
        status: pass
    human_judgment: false
  - id: D7
    description: "The audit resists rot: a site added by a later phase fails the suite, and the sweep does not undercount wrapped comparisons"
    requirement: "CLN-03"
    verification:
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_flattening_is_load_bearing"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_sweep_finds_the_wrapped_pair_masks_in_h0ands"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_prose_mentions_are_excluded_and_counted"
        status: pass
    human_judgment: false
  - id: D8
    description: "The sweep's coverage of code outside the obvious modules -- sedacs/, _legacy/, _atomic_density_matrix -- is stated with evidence rather than assumed"
    requirement: "CLN-03"
    verification:
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_sedacs_result_is_stated"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_legacy_h0ands_has_no_importer"
        status: pass
      - kind: unit
        ref: "tests/test_orbital_count_guards.py#test_atomic_density_matrix_has_no_production_importer"
        status: pass
    human_judgment: false

# Metrics
duration: 47 min
completed: 2026-08-03
status: complete
---

# Phase 5 Plan 06: Orbital Count Audit and f Refusal Policy Summary

**157 hardcoded orbital-count, shell-count and basis-layout sites in `src/dftorch` are
enumerated by a re-runnable sweep and individually dispositioned; the six that were open
were three truncated copies of the per-shell AO count table on the spin path, now guarded
with the existing named exceptions; and the inventory's completeness is enforced
bidirectionally by a test rather than by review.**

## Performance

- **Duration:** 47 min
- **Started:** 2026-08-03T01:59:35Z
- **Completed:** 2026-08-03T02:46Z
- **Tasks:** 2
- **Files created:** 3, **modified:** 4

## Task Commits

1. **Task 1: sweep every orbital-count site and write the inventory** - `6e69da2` (feat)
2. **Task 2: guard every open site and gate the inventory** - `1aade8c` (feat)
3. **Out-of-scope finding logged** - `f4ae884` (docs)

## What was built

### Task 1: the sweep and the record (`6e69da2`)

`tests/orbital_count_sweep.py` walks `src/dftorch/**/*.py`, blanks every comment and
string literal, collapses each newline-plus-indent run, and matches three families of
pattern, returning one record per match with the line number recovered from the original
text. `docs/ORBITAL-COUNT-INVENTORY.md` gives each record a row.

**Three families, not one.** The plan named a single identifier set (`n_orb`, `norb_I`,
`nI`, ...). The sweep matches two more, and the reason is not thoroughness for its own
sake:

| Family | Matches | Why it is in scope |
| --- | --- | --- |
| `orbital-count` | an orbital-count identifier vs 1, 4, 9, 16 | the family D-04 names |
| `shell-count` | `max_ang`, `n_shells_per_atom`, `shell_types` vs 1, 2, 3, 4 | **the Phase 4 shell-resolved Coulomb defect lives at `_coulomb_matrix.py:816-824`, which tests `max_ang` and never mentions `n_orb`** |
| `basis-layout-literal` | a bracketed literal spelling a canonical per-shell dimension or AO-offset table | CLN-03 names the hardcoded offsets by hand, and this family found both real gaps |

Restricting the sweep to orbital-count names would have built a blind spot exactly where
the project has already been bitten once. Neither of the two gaps this audit found
mentions `n_orb` anywhere.

**Prose is excluded and counted.** A docstring saying `n_orb == 16` decides nothing, and a
row for it could carry no honest disposition. 20 prose mentions were excluded at Task 1
(27 after the guards were added), reported by `count_prose_mentions` rather than left
unstated.

Getting this right was measured, not assumed. On Python 3.13 an f-string is no longer a
single `STRING` token, so before `FSTRING_START`/`MIDDLE`/`END` were added to the blanked
set the sweep reported `ESDriver.py:63` and `:103` -- text inside the guards' own error
messages -- while the guards' real `counts == 16` tests at `:60` and `:95` were invisible.
The audit was briefly pointed at its own documentation instead of its own code.
`test_prose_mentions_are_excluded_and_counted` pins those exact line numbers.

**The `4dbffaa` hydrogen defect is cited, not re-investigated.** It is the worked example
at the top of the inventory: a `Ep = 0.000039` placeholder in mio-1-1's `H-H.skf` gave
hydrogen four orbitals instead of one, the phantom p functions took real Slater-Koster
overlap, `S` went indefinite at -0.159, `S^(-1/2)` became undefined, and CH4 could never
converge -- with every shape, finiteness and symmetry assertion in the suite still green.
That is the concrete proof that a document-only audit would have been insufficient.

### Task 2: the refusals, and what the audit actually found (`1aade8c`)

Six rows were `needs-action`, and all six are the same finding in two places.
`_spin.get_h_spin`, `_spin.get_h_spin_diag` and `_forces.forces_spin` each declare a local

    n_orb_per_shell = torch.tensor([0, 1, 3, 5])

which is the per-shell AO count table truncated one entry short of f, while
`Constants.shell_dim` a few modules away holds the correct `[0, 1, 3, 5, 7]`. The table is
indexed by `shell_types`, which is `4` for an f shell, so an f system indexes one past its
end. `forces_spin` additionally reconstructs 1/2/3-shell masks with no four-shell class,
so a four-shell atom's spin potential would stay exactly zero inside a correctly shaped
force.

These were reachable past the existing guards. `ESDriver.forward`'s
`_require_closed_shell_f_system` does not run for `MD.py:745`, `MD.py:794`, `MD.py:1103`
or `_xl_tools.py:852`, each of which reaches `get_h_spin`, `get_h_spin_diag` or
`forces_spin` directly.

**Guarded, not widened, and the distinction is the whole of D-04.** Changing `[0, 1, 3, 5]`
to `[0, 1, 3, 5, 7]` is one character and would have been wrong twice over: Phase 4
decision D-12 defers spin-polarized f entirely, and `tests/f_orbital_data/` ships no
`spinw.txt`, so the f spin coupling constants the widened table would then multiply do not
exist. Sizing the f block correctly and filling it from absent parameters is precisely the
well-shaped-and-wrong outcome the decision exists to prevent.

Two module-level guards were added, both following `ESDriver._require_f_derivatives`
exactly -- early return for a missing attribute, an all-invalid `TYPE`, or an f-free
system; raise as the guarded function's first statement otherwise:

| Guard | Raises | Why that class |
| --- | --- | --- |
| `_spin._require_no_f_spin_shells` | `FSpinPolarizationUnsupportedError` | these builders exist only for spin-polarized calculations, which D-12 defers |
| `_forces._require_no_f_spin_forces` | `FDerivativeUnsupportedError` | `forces_spin` fails for a reason independent of spin: it consumes `dS`, which is exactly zero in every f block |

**No new exception class was defined.** `test_no_new_f_exception_class_was_defined` pins
the taxonomy at exactly the four names, so a fifth added quietly would split the support
policy that is meant to be reviewable as one list.

### The site handed over by plan 05-05

`_coulomb_matrix_batch.ewald_k_space_vectorized`'s `do_vec=True` branch printed
"vectorized k-space is not implemented for batched data" and then `return`ed. The function
is annotated `-> tuple[torch.Tensor, torch.Tensor]` and its only in-package caller unpacks
two values, so the bare `return` surfaced as `cannot unpack non-iterable NoneType object`
one frame away with the explanation already scrolled past.

**The determination: it is not an orbital-count site.** Nothing in that branch reads
`n_orb`, `max_ang` or the basis layout; it is a batched k-space Ewald gap (PHY-04,
Phase 8.1). It is recorded in a dedicated section and classified out of D-04's row set
rather than forced in, and resolved anyway because the printed-apology pattern is exactly
what D-04 removes. It now raises `NotImplementedError` naming the flag, the working
alternative (`do_vec=False`, the default and what every caller uses) and the deferring
requirement. No caller in the package passes `do_vec=True`, so no result the suite covers
can change.

## The dispositions

| File | Sites | `extended` | `guarded` | `unreachable` |
| --- | ---: | ---: | ---: | ---: |
| `_legacy/H0andS.py` | 54 | 0 | 0 | 54 |
| `_h0ands.py` | 52 | 32 | 20 | 0 |
| `_stress.py` | 20 | 0 | 2 | 18 |
| `_coulomb_matrix.py` | 17 | 0 | 1 | 16 |
| `_forces.py` | 5 | 0 | 5 | 0 |
| `Structure.py` | 3 | 3 | 0 | 0 |
| `ESDriver.py` | 2 | 0 | 2 | 0 |
| `_spin.py` | 2 | 0 | 2 | 0 |
| `Constants.py` | 1 | 1 | 0 | 0 |
| `_atomic_density_matrix.py` | 1 | 0 | 0 | 1 |
| **total** | **157** | **36** | **32** | **89** |

`unreachable` is established, never asserted from reading.
`test_every_unreachable_row_states_how_it_was_established` requires each such row's
evidence to name an upstream guard or an import search, and two dedicated tests hold the
two import-based claims:

- `_legacy/H0andS.py` (54 rows): no import statement anywhere in `src/` references it, and
  the directory has no `__init__.py`, so it is not an importable package. The test matches
  import statements only, because this module and the sweep both *name* `_legacy` in prose
  and a substring search would flag the audit that established the property.
- `_atomic_density_matrix.py`: its only production importer is `_legacy/H0andS.py`, itself
  unreachable, which additionally calls it with five positional arguments against a
  six-parameter signature. The live `D0` comes from
  `Structure._atomic_density_matrix_from_shells`.

`src/dftorch/sedacs/` returns **zero** records and that is recorded as a checked-and-empty
result, gated by `test_sedacs_result_is_stated`, which also asserts that if a sedacs site
ever appears it must have a row. `_slater_koster_pair.py` returns zero *code* records --
all its matches are docstrings and message constants.

## Verification

**206 passed / 0 failed / 0 errors / 0 skipped**, across all 17 test files. Entering
baseline was 169; the 37 new tests are the whole difference and none was lost.

The full suite exceeds the executor's command cap, so it is run per file and summed --
a measurement constraint, not a hang, and the same method plan 05-05 used.

| File | Result | | File | Result |
| --- | --- | --- | --- | --- |
| `test_skf_metadata_oracle.py` | 56 | | `test_f_orbital_skf.py` | 16 |
| `test_orbital_count_guards.py` | **37** | | `test_simple_format_regression.py` | 11 |
| `test_shell_resolved_u.py` | 27 | | `test_shell_count_parsing.py` | 7 |
| `test_radial_grid.py` | 21 | | `test_eu_n_scan.py` | 6 |
| `test_spin_guard.py` | 5 | | `test_single_shot_energy.py` | 5 |
| `test_dtype_contract.py` | 4 | | `test_public_api_contract.py` | 4 |
| `test_import.py` | 3 | | `test_scf.py`, `test_io.py`, `test_public_api.py`, `test_nearestneighborlist.py` | 1 each |

The three end-to-end gates that matter for this plan:

- `test_eu_n_scan.py` reproduces its recorded 21-point curve, so no refusal landed on a
  path the supported f workflow travels.
- `test_simple_format_regression.py` and `test_shell_count_parsing.py` are green, so no
  f-free calculation acquired a failure mode.
- `test_f_pair_reaches_single_system_sk_assembly` measures the Eu f / N block of `H0` at
  0.606 eV and of `S` at 0.0279 -- non-zero, which is the direct evidence that the
  `extended` rows in `_h0ands.py` really route f pairs into assembly rather than dropping
  them. A zero block is what the silent drop produced and is indistinguishable from a
  correct result by shape, finiteness or symmetry.

### The flattening claim was measured, and it partly disconfirmed itself

The plan and the phase research both assert that a line-oriented sweep undercounts
`_h0ands.py` by roughly half and that flattening fixes it. Measured on the live file with
the pair-level form `(a == x) & (b == y)` -- the shape the plan-time research swept for:

| How the pattern is applied | Masks found |
| --- | ---: |
| per physical line, the way a `grep` pipeline works | 9 |
| after `flatten_source` | 25 |

So the undercount is real and worse than recorded (the plan said 12 versus 22). **But the
second half of the claim does not survive measurement.** This sweep's own patterns separate
their parts with `\s*`, which already crosses newlines, so matching the whole file is
equivalent to flattening it: 157 records either way, 52 in `_h0ands.py` either way. The
flattening is therefore not what makes this sweep complete -- writing the patterns with
`\s*` is.

It is kept, and the reason is written into both the sweep docstring and the inventory
rather than left implied: it makes the guarantee structural instead of dependent on every
future pattern author remembering to use `\s*` instead of a literal space, and it is what
makes a matched snippet single-line and therefore usable as an inventory key. The first
version of `test_flattening_is_load_bearing` asserted the claim as given and **failed**,
which is how the caveat was found; it now measures the two numbers on the live file and
fails if `_h0ands.py` is ever reformatted so that nothing wraps.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] The sweep swept its own error messages instead of its own code**

- **Found during:** Task 1, first run of the sweep.
- **Issue:** `blank_comments_and_strings` blanked `tokenize.COMMENT` and `tokenize.STRING`
  only. On Python 3.12+ an f-string is emitted as `FSTRING_START`/`MIDDLE`/`END`, not
  `STRING`, so the guards' own f-string messages were matched as code. The sweep reported
  `ESDriver.py:63` and `:103` (message text reading `n_orb == 16`) and **missed** the real
  guard comparisons `counts == 16` at `:60` and `:95`.
- **Fix:** added the three `FSTRING_*` types via `getattr`, so the module still works on
  3.11 where they do not exist. Also added `counts` to the orbital-count identifier set --
  it is the local name all three existing `_require_*` guards bind `n_orb[TYPE]` to, and
  without it the guards would have been invisible to their own audit.
- **Verification:** `test_prose_mentions_are_excluded_and_counted` asserts the ESDriver
  site set is exactly `{60, 95}` and names 63/103 in its failure message as the
  fingerprint of the regression.
- **Commit:** `6e69da2`

**2. [Rule 2 - Missing critical] The `shell-count` and `basis-layout-literal` families**

- **Found during:** Task 1, checking the plan's identifier set against known defects.
- **Issue:** the plan's set is orbital-count identifiers only. The Phase 4 shell-resolved
  Coulomb defect that produced `FShellResolvedCoulombUnsupportedError` lives at
  `_coulomb_matrix.py:816-824` and tests `max_ang`, never `n_orb`. An `n_orb`-only sweep
  would have been complete by its own definition and blind to the class of defect the
  project has already had.
- **Fix:** two extra families. They found the only two real gaps in the audit, neither of
  which mentions `n_orb`: the truncated `[0, 1, 3, 5]` tables (`basis-layout-literal`) and
  `forces_spin`'s 1/2/3-shell masks (`shell-count`).
- **Verification:** 19 shell-count and 8 basis-layout rows in the inventory; the six
  `needs-action` rows were all in those two families.
- **Commit:** `6e69da2`

**3. [Rule 2 - Missing critical] Guards added to `_spin.py` and `_forces.py`, which the plan did not list**

- **Found during:** Task 2.
- **Issue:** the plan's Task 2 `<files>` predicts the gaps in `_h0ands.py`, `_stress.py`,
  `_coulomb_matrix.py`, `_coulomb_matrix_batch.py` and `ESDriver.py`. Measurement put every
  one of those at `extended`, `guarded` or `unreachable` already -- Phases 3 and 4 had done
  that work -- and put the real gaps in `_spin.py` and `_forces.py`, which the plan does not
  mention. The plan anticipated this ("Expected shape, to be confirmed rather than assumed").
- **Fix:** the two `_require_*` guards described above, in the modules that own the sites,
  per D-04's rule.
- **Scope check:** these modules belong to Phase 9 (MD) and Phase 7-8 (derivatives) work,
  but adding a *refusal* is not new physics and is exactly what the phase fence permits --
  the same treatment the batched `FAngularFormulaSourceError` sites already have.
- **Commit:** `1aade8c`

**4. [Rule 1 - Bug] `test_flattening_is_load_bearing` asserted a claim that measurement contradicted**

- **Found during:** Task 2, first run of the new test module.
- **Issue:** the test was written to the plan's claim -- reflow a comparison, show a
  line-oriented match count lower than a flattened one. It failed 2 == 2, because the
  pattern's `\s*` separators already cross newlines. The plan's claim is true of a *per-line*
  sweep and false of whole-text matching, and the two had been conflated.
- **Fix:** the test now measures per-line versus flattened on the real `_h0ands.py` with the
  pair-level pattern (9 versus 25) and the caveat is written into the sweep docstring, the
  inventory and this summary. Nothing was weakened to make a test pass.
- **Commit:** `1aade8c`

**5. [Rule 1 - Bug] Two self-referential test failures in the audit's own gates**

- **Found during:** Task 2.
- **Issue:** `test_legacy_h0ands_has_no_importer` used a substring search and flagged
  `tests/orbital_count_sweep.py` and `tests/test_orbital_count_guards.py`, which mention
  `_legacy` in prose -- the audit failing on the documentation that established the
  property. Separately, one `unreachable` row's evidence read "Same import search as above"
  and did not itself name a search, so `test_every_unreachable_row_states_how_it_was_established`
  rejected it.
- **Fix:** the importer test now matches import statements in `src/` only; the row's evidence
  now states the search rather than referring to it. The second fix is the right one: a row
  whose evidence only points elsewhere is not self-contained.
- **Commit:** `1aade8c`

---

**Total deviations:** 5 auto-fixed (2 bugs in the new sweep, 2 missing-critical additions,
1 self-referential test defect).
**Impact:** no scope creep and no physics change. Deviations 1, 2 and 3 each expanded what
the audit covers, and 2 and 3 are causally linked -- the extra families are what found the
gaps that required the extra guards. Deviation 4 corrected a claim inherited from the plan
rather than working around it.

## Issues Encountered

**One out-of-scope finding, logged rather than fixed.**
`docs/LIBRARY-OUTPUT-INVENTORY.md` (plan 05-05) states that
`tests/test_verbose_flag.py::test_inventory_covers_every_print` gates its completeness.
Neither the file nor the test exists. This matters because `05-VALIDATION.md` identifies
the completeness gate as the one documentation property in Phase 5 with a real automatable
check -- which this plan's inventory implements and that one only claims. Recorded in
`deferred-items.md` item 4 (`f4ae884`) and in `.planning/WINDOWS.md` as an `unmet-truth`
entry, with `tests/test_orbital_count_guards.py` named as the directly reusable shape.
Not fixed here: it is plan 05-05's deliverable, and the scope boundary says log it.

## Known Stubs

None. No stub, placeholder, `TODO`, skipped test or unrun `<verify>` was left behind. Every
`<automated>` verification in the plan was executed, and the plan's `xfail(strict=True)`
escape hatch for an unreachable guard was **not used**: all eight guarded sites are reached
by a real test, including the two whose refusals sit mid-function
(`H0_and_S_vectorized_batch` and `_stress._pair_grad_from_sk`), which were reached with the
smallest synthetic tensors that get there rather than declared untestable.

No temporary source mutation was left in the tree; the two probe geometry files written
during investigation were deleted in the same command that created them.

## Threat mitigations

| Threat | Disposition | Evidence |
| --- | --- | --- |
| T-05-25 an unguarded site returning a well-shaped matrix with zeros where f values belong | mitigated | every swept site dispositioned; `test_inventory_has_no_unresolved_sites` forbids leaving one open; `test_every_guarded_site_raises_for_f` proves all 8 refusals reachable rather than merely present |
| T-05-26 an inventory complete on the day it is written and stale a month later | mitigated | `test_inventory_covers_every_site` re-runs the sweep at test time and matches both directions; `test_every_guard_case_is_claimed_by_at_least_one_row` closes the same loop for guard cases |
| T-05-27 an over-eager guard breaking f-free calculations | mitigated | both new guards return early unless an f atom / f shell is present; `test_guards_ignore_f_free_systems` over all 8 cases, `test_ch4_runs_end_to_end_after_the_guards`, and the whole 206-test suite |
| T-05-28 a site extended to 16 without the capability to compute a correct f result | mitigated | nothing was extended. All six open sites were guarded; the inventory records that widening was a one-character change in each case and states the two reasons it would have been wrong |
| T-05-29 a refusal message interpolating a filesystem path | mitigated | `test_refusal_messages_leak_no_path` over all 8 cases. A bare "no forward slash" check would be wrong -- the shared constants legitimately contain `dH0/dS` -- so it asserts on the actual SKF directory, geometry file, tmp dir, `.skf`, `.xyz`, backslash and drive-letter forms |
| T-05-30 the sweep undercounting because it is line-oriented | mitigated | flattening applied and measured (9 versus 25 per-line at pair level); `test_sweep_finds_the_wrapped_pair_masks_in_h0ands` requires >= 20; the caveat that `\s*` already crosses newlines is recorded rather than hidden |
| T-05-SC package installs | not applicable | nothing was installed |

## Threat Flags

None. This plan adds no network I/O, no deserialisation of untrusted data, no subprocess,
no dynamic import and no privilege transition. The three new messages are bounded to
orbital counts, capability names and requirement ids by construction.

## Next Phase Readiness

- **The audit is the work list for Phases 6-9.** Every `guarded` row names the requirement
  that will lift it: PHY-01/DRV-01 (f derivatives), PHY-02 (f stress), PHY-04 (batched f
  and batched k-space), D-12 (spin). Lifting a refusal now has a test that will notice.
- **`_forces.forces_spin` and `_spin.get_h_spin*` are now hard blockers for f MD.** Phase 9
  cannot run a spin-polarized f trajectory without either implementing the f spin coupling
  or removing these guards deliberately. That was already true in substance; it is now true
  loudly.
- **The `basis-layout-literal` family will need extending.** It matches an enumerated set of
  literal spellings, stated as a limitation in the sweep docstring and the inventory. A phase
  that writes a new layout table with an unlisted spelling must add it to
  `BASIS_LAYOUT_SEQUENCES`, or the sweep will not see it.
- **`docs/LIBRARY-OUTPUT-INVENTORY.md`'s missing gate is open** (deferred item 4, WINDOWS
  entry 2) and belongs to whoever revisits D-02.

## Self-Check: PASSED

Files verified present on disk:

- `FOUND: tests/orbital_count_sweep.py`
- `FOUND: docs/ORBITAL-COUNT-INVENTORY.md`
- `FOUND: tests/test_orbital_count_guards.py`
- `FOUND: src/dftorch/_spin.py` (modified)
- `FOUND: src/dftorch/_forces.py` (modified)
- `FOUND: src/dftorch/_coulomb_matrix_batch.py` (modified)

Commits verified in `git log`:

- `FOUND: 6e69da2` feat(05-06): sweep every orbital-count site and write the inventory
- `FOUND: 1aade8c` feat(05-06): guard every open orbital-count site and gate the inventory
- `FOUND: f4ae884` docs(05-06): log the missing test_verbose_flag.py gate

---
*Phase: 05-regression-safety-and-support-policy-cleanup*
*Completed: 2026-08-03*

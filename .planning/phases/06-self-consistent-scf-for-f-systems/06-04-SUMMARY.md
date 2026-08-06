---
phase: 06-self-consistent-scf-for-f-systems
plan: 04
subsystem: docs
tags: [dftb, scf, energy-definitions, verdict, documentation, hubbard-u, pytest]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "decision D-11, which created the one-pass energy definition, and the pinned reference EU_N_REFERENCE_E_TOT that this plan's document scopes to that path"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 01
    provides: "the Krylov-off interim that makes the settled Eu-N number reproducible, and scf_iter_count"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 02
    provides: "the sixteen-block shell-resolved Coulomb matrix the third settled number rests on"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 03
    provides: "the per-orbital-group charge loop, which produced a settled energy the plan text did not know about and this document had to report"
provides:
  - "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md: the D-6.03 written verdict, with all numbers re-measured 2026-08-06"
  - "tools/check_verdict_doc.py: a stdlib-only gate that fails if the document loses a required anchor OR if a cited source line stops containing what the document says it contains"
  - "tests/test_energy_definitions_f.py: four tests holding the verdict's claims, freezing no value"
  - "A recorded defect note above Constants.py's self.U = Parameter(US, ...) assignment"
  - "A dated ROADMAP note closing the D-6.06 conditional"
affects: [06-05, phase-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A document that cites source line numbers ships with a checker that re-verifies those citations against the tree, so the document cannot drift out of agreement with the code it describes"
    - "When a recorded defect is referred to by a bare line number, inserting the record above it invalidates the name; replace the number with the symbol rather than with a newer number"
    - "Measurements quoted in a prose verdict, never in an assertion; the tests hold signs, exact zeros, inequalities and ratios only"

key-files:
  created:
    - docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md
    - tools/check_verdict_doc.py
    - tests/test_energy_definitions_f.py
  modified:
    - src/dftorch/Constants.py
    - .planning/ROADMAP.md
    - tests/test_shell_resolved_scf_f.py

key-decisions:
  - "The verdict lives in docs/, not in a phase summary, because it is a fact about what two code paths compute rather than a fact about this phase."
  - "The checker verifies cited line numbers against the source, beyond the plan's 'completeness gate'. The plan's own threat model names 'verdict document -> code' as the trust boundary that matters, and both cited sites had already moved once during this phase."
  - "Constants.py's assignment is documented, not fixed. Adding the comment moved it off line 232, so the bare line number was retired as an identifier everywhere rather than replaced with a newer bare number."
  - "The shell-resolved settled energy (-5.639887 eV, a gap of 11.87 eV) is reported alongside the default one. Reporting only the default would have made the document misleading the day after plan 06-03 landed."

requirements-completed: [SCC-01]

coverage:
  - id: D1
    description: "A written verdict exists on the one-pass versus settled energy difference, and says which of the two answers D-6.03 allows is true: a difference of definition, not a defect"
    requirement: SCC-01
    verification:
      - kind: doc
        ref: "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md#the-verdict"
        status: pass
      - kind: unit
        ref: "tools/check_verdict_doc.py (verdict anchor, required to appear before the second section)"
        status: pass
    human_judgment: true
  - id: D2
    description: "The verdict accounts for where the difference actually sits: most of it in the band-structure term, not in the electron-repulsion term"
    requirement: SCC-01
    verification:
      - kind: doc
        ref: "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md#where-the-difference-actually-sits (three-way split table, 84.2 percent)"
        status: pass
      - kind: unit
        ref: "tools/check_verdict_doc.py (requires both the numeric share and the words)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The verdict names the mechanism in plain words and points at the exact line of code that creates it, with line numbers confirmed against the current tree"
    requirement: SCC-01
    verification:
      - kind: unit
        ref: "tools/check_verdict_doc.py (five source citations re-verified against the files on every run)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The one-pass path carries no electron-repulsion term at all and the settled path carries one"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_energy_definitions_f.py#test_the_one_pass_path_carries_no_electron_repulsion_term"
        status: pass
    human_judgment: false
  - id: D5
    description: "The two paths move very different amounts of charge, which is the mechanism behind the band term's share of the difference"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_energy_definitions_f.py#test_the_two_paths_move_very_different_amounts_of_charge"
        status: pass
    human_judgment: false
  - id: D6
    description: "Every nitrogen on-site level lies below every europium one, which is the structural cause of the runaway one-pass transfer"
    requirement: SCC-01
    verification:
      - kind: unit
        ref: "tests/test_energy_definitions_f.py#test_every_nitrogen_level_lies_below_every_europium_level"
        status: pass
    human_judgment: false
  - id: D7
    description: "The Phase 4 pinned one-pass energy remains valid and is stated to be valid for the one-pass path only, with a tripwire against a future change making the two paths agree"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_energy_definitions_f.py#test_the_phase_four_pin_is_not_the_settled_answer"
        status: pass
      - kind: integration
        ref: "tests/test_single_shot_energy.py#test_eu_n_single_shot_reference_energy"
        status: pass
    human_judgment: false
  - id: D8
    description: "The s-group electron-repulsion strength defect is recorded where a reader of that line will see it, with its measured size and with the shell-resolved path named as its remedy"
    requirement: SCC-01
    verification:
      - kind: unit
        ref: "tests/test_shell_resolved_scf_f.py#test_the_per_atom_strength_still_comes_from_the_s_group"
        status: pass
      - kind: unit
        ref: "tools/check_verdict_doc.py (Constants.py citation)"
        status: pass
    human_judgment: false
  - id: D9
    description: "The roadmap no longer describes two of this phase's three requirements as conditional on a diagnosis that has since been made"
    requirement: SCC-01
    verification:
      - kind: doc
        ref: ".planning/ROADMAP.md Phase 6 scope note (2026-08-04)"
        status: pass
    human_judgment: true

# Metrics
duration: ~80min
completed: 2026-08-06
status: complete
---

# Phase 6 Plan 04: The Verdict on the 10.6 eV Difference Summary

**The 10.6 eV difference between the one-pass and settled Eu-N energies is a difference of definition, not a defect - and the obvious explanation is wrong, because 84 percent of it sits in the band-structure term and only 16 percent in the electron-repulsion term the settled path adds.**

## Performance

- **Duration:** ~80 min, one session
- **Completed:** 2026-08-06
- **Tasks:** 3
- **Files:** 3 created, 3 modified. No file deleted in any of the three commits (checked with `git diff --diff-filter=D` after each).

## Every number was re-measured, and one of them had moved

The plan quoted -17.51 eV and -6.92 eV from measurements taken before plan 06-03 changed which
Hubbard strength europium's f electrons are charged at. Both were re-measured from live code at
commit `cef305d` before a word of the verdict was written.

| Quantity | Plan text | Re-measured 2026-08-06 | Verdict |
|---|---|---|---|
| one-pass total | -17.51 | **-17.510444** | reproduces exactly |
| settled total, default path | -6.92 | **-6.920251** | reproduces exactly |
| band term, one pass -> settled | -16.948 -> -8.030 | **-16.948082 -> -8.030033** | reproduces |
| electron repulsion, settled | +1.699 | **+1.698980** | reproduces |
| band share of the gap | 84 % | **84.2 %** | reproduces |
| charge off Eu, one pass | "about 2.55" | **2.702732** | **plan figure was from 2.40 A, not 2.655 A** |
| charge off Eu, settled | "about 0.62" | **0.599583** | close; the ratio is 4.51, not 4.1 |

**Plan 06-03 did not move the default numbers, because its work is opt-in.** The per-orbital-group
charge loop is selected by `MAGNETIC_HUBBARD_LDEP`, which is off by default, so the default settled
path is byte-for-byte the path the plan text described. What 06-03 did create is **a third number**,
and the document would have been misleading the day after it landed if it reported only two:

| Path | Total (eV) | Band (eV) | Electron repulsion (eV) | Charge off Eu | Passes |
|---|---|---|---|---|---|
| one pass | -17.510444 | -16.948082 | 0 (exactly) | 2.702732 | n/a |
| settled, per atom (default) | -6.920251 | -8.030033 | +1.698980 | 0.599583 | 28 |
| settled, per orbital group | -5.639887 | -6.096938 | +1.201020 | 0.498650 | 80 |

Against the finer charge description the gap is **11.87 eV, not 10.59 eV**, and the band term's
share rises from 84 to 91 percent. The verdict and the mechanism are unchanged - if anything
sharper, since charging europium's f electrons at their real 13.61 eV makes the electrostatic
penalty bite sooner and hands back even more band energy. The document says all of this in a
section of its own and warns explicitly against reading 11.87 as a correction to 10.59.

The 40 A dissociation figures that go into the `Constants.py` comment were re-measured too, and
both reproduce the research document exactly: charge off europium 0.2887 with the s strength and
0.2021 with the f strength substituted, energies -1.96171 and -1.73157 eV.

## Task Commits

1. **Task 1: Write the verdict** - `477dcc5`. `docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md` (301 lines) and `tools/check_verdict_doc.py` (325 lines).
2. **Task 2: Hold the claims with tests** - `6a9ba7b`. `tests/test_energy_definitions_f.py`, four tests.
3. **Task 3: Record the defect, close the roadmap tension** - `997c089`. `src/dftorch/Constants.py` (comment only), `.planning/ROADMAP.md` (scoped note), plus the line-number follow-ups described under deviations.

## Accomplishments

- **The verdict is stated first and stated plainly**, in one paragraph before any argument, and it
  is one of the two answers D-6.03 allows. It carries its confidence level (HIGH) and says what
  that rests on: a checkable `None` argument, a measured three-way split, and measured charge
  transfer differing by more than a factor of four.
- **The section D-6.03 warned about is the longest one.** An explanation that says only "the
  settled path includes electrostatics" accounts for 1.70 eV of a 10.59 eV difference. The
  document says so as a number, as a percentage and in words, and rejects that explanation
  by name.
- **The mechanism is traceable rather than asserted.** Two `None` arguments at
  `ESDriver.py:1030-1031` select the `Ecoul = 0` arm at `_energy.py:181-182`. A reader can open
  both files and check.
- **The document cannot silently go stale.** `tools/check_verdict_doc.py` re-reads all five cited
  source locations and fails if any stops containing what the document claims. Both failure
  branches were exercised deliberately before the commit - a citation the document does not make,
  and a citation whose source content no longer matches - and both fire with a message naming the
  drift and printing the current source lines.
- **The tests freeze nothing.** Every assertion is an exact zero, a sign, an inequality or a ratio.
  `grep -c "17.510444238744924" tests/test_energy_definitions_f.py` returns 0: the Phase 4 pin is
  imported from `tests/test_single_shot_energy.py`, so the suite still holds one copy of it.
- **Full suite: 254 passed, 0 failed** in 372 s (was 250). +4, exactly the new module.

## Files Created/Modified

- `docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md` (+301, new) - the verdict. Opens with a vocabulary table
  defining band-structure energy, electron-repulsion energy, nuclear repulsion, per-atom charge and
  convergence in plain words before any of them is used, because the two repulsion terms are easy
  to confuse and only one of them changes between the paths.
- `tools/check_verdict_doc.py` (+325, new) - 11 prose anchors, 5 verified source citations, an
  ASCII byte scan, a first-section check on the verdict, and a forbidden-content scan.
- `tests/test_energy_definitions_f.py` (+434, new) - four tests.
- `src/dftorch/Constants.py` (+55/-0) - comment lines only; verified mechanically, see below.
- `.planning/ROADMAP.md` (+33/-1 from this plan) - one scoped note.
- `tests/test_shell_resolved_scf_f.py` (+7/-4) - three stale `Constants.py:232` prose references,
  see deviation 3.

## Verification Evidence

| Gate | Result |
|---|---|
| `uv run pytest` (whole suite) | **254 passed, 0 failed** in 372.35 s - above the 250 baseline |
| `uv run pytest tests/test_energy_definitions_f.py -x` | 4 passed |
| `uv run pytest tests/test_energy_definitions_f.py tests/test_single_shot_energy.py -x` | 9 passed - the Phase 4 pin is still green |
| `uv run pytest tests/test_shell_resolved_scf_f.py tests/test_shell_resolved_u.py -x` | 43 passed |
| `python tools/check_verdict_doc.py` | exit 0 |
| ASCII byte scan, `docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md` | exit 0 |
| ASCII byte scan, `tools/check_verdict_doc.py` | exit 0 |
| ASCII byte scan, `tests/test_energy_definitions_f.py` | exit 0 |
| `grep -c "17.510444238744924" tests/test_energy_definitions_f.py` | 0 |
| `git diff src/dftorch/Constants.py`, lines added that are not comments | **none** - filtered the `+` lines for anything that is not a `#` line or blank; the result was empty |
| `grep -c "^### Phase " .planning/ROADMAP.md` | **10 before, 10 after** |
| Plan checkmarks for 06-01, 06-02, 06-03 in ROADMAP | all three still `[x]` |
| `python -c "... 'MAGNETIC_HUBBARD_LDEP' in s and 'D-6.10' in s"` on `Constants` source | exit 0 |
| Post-commit deletion check on all three commits | no files deleted |

## Prohibition audit

All four of the plan's prohibitions hold.

1. **No settled energy or charge is written into a test or into the verdict document as a reference
   value.** The document quotes measurements in prose with their source and date named. The test
   module's only module-level numeric constants are `CHARGE_TRANSFER_RATIO_FLOOR = 3.0` and
   `MINIMUM_GAP_TO_THE_PHASE_FOUR_PIN = 1.0`, plus the two orbital counts 16 and 4 used as an
   ordering guard. The one imported number, `EU_N_REFERENCE_E_TOT`, is the Phase 4 **one-pass** pin,
   which D-6.08 does not cover and which is used in a *floor on a difference*, not an equality.
2. **The assignment in `Constants.py` is not changed.** Comment lines only, verified by filtering
   the diff rather than by eye.
3. **The verdict does not present the difference as unexplained or pending.** It reaches a stated
   verdict in its first paragraph and has a "What this document does not claim" section that is
   about scope, not about unfinished diagnosis.
4. **No test gates the settled binding curve on the Phase 4 tolerance band.** No test in this plan
   touches the binding curve at all.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale fact in the plan] Both cited line numbers had moved, by 235 and 50 lines**

- **Found during:** Task 1, at the "confirm each before editing" step the plan itself demands.
- **Issue:** the plan says to cite `ESDriver.py:783-792` and `_energy.py:131-132`. Neither is
  correct against the current tree. Plans 06-01 through 06-03 added code above both sites.
  `ESDriver.py:783-792` today lands in the **open-shell** branch (`get_h_spin`, `get_spin_energy`),
  and `_energy.py:131-132` lands inside plan 06-03's **shell-resolved argument check**. Citing
  either would have pointed a reader at unrelated code while sounding authoritative.
- **Fix:** the confirmed current locations are cited instead - the `energy()` call at
  `ESDriver.py:1018-1041`, its two `None` arguments at `ESDriver.py:1030-1031`, the reason comment
  at `ESDriver.py:958-973`, and the zero arm at `_energy.py:181-182`. The document carries a short
  "A note on the line numbers" paragraph saying the plan-time numbers are stale and what is at them
  now, so a reader cross-checking against the plan is not confused.
- **Committed in:** `477dcc5`

**2. [Rule 2 - Missing critical functionality] The checker verifies the citations, not just their presence**

- **Found during:** Task 1, immediately after deviation 1.
- **Issue:** the plan specifies a checker that "exits non-zero if any of the required anchors is
  absent" - a text-presence gate. That would have passed happily on a document citing
  `ESDriver.py:783-792`, which is exactly the failure that had just been caught by hand. The plan's
  own threat model names `verdict document -> code` as the boundary that matters and says a
  document that drifts out of agreement with the code "is worse than no document, because it is
  trusted".
- **Fix:** `REQUIRED_CITATIONS` holds five `(path, first line, last line, expected text, why)`
  tuples. The checker asserts both that the document makes the citation and that the source file at
  those lines still contains the expected text, printing the current source lines on failure.
- **Verified by driving both failure branches deliberately** rather than by assuming they work: a
  citation the document does not make is reported as "the document does not cite ...", and a
  citation whose content no longer matches prints "`_energy.py:181-182` no longer contains ..."
  followed by lines 181-182 as they currently read.
- **Committed in:** `477dcc5`

**3. [Rule 1 - Bug introduced by the plan's own instruction] Adding the comment moved the line it is about**

- **Found during:** Task 3, immediately after the edit.
- **Issue:** the plan says to add the comment "immediately above the assignment at line 232". Doing
  so pushes the assignment to line 291. Four places then referred to a line that no longer holds
  what they said - `tools/check_verdict_doc.py`'s own citation, two prose references in the verdict
  document, and three prose references in `tests/test_shell_resolved_scf_f.py` (a module docstring,
  a test docstring and an assertion message; none of them an assertion *value*, so no test broke,
  which is precisely why this would have gone unnoticed).
- **Fix:** the bare line number was **retired as an identifier** rather than replaced with a newer
  bare number, since a newer number would break the next time anyone edits above it. Every
  reference now names the assignment `self.U = torch.nn.Parameter(US, ...)`, notes that D-6.10
  calls it "Constants.py:232", and points at the checker as the thing that tracks its current
  location. The comment itself carries a paragraph explaining this, so someone arriving from a
  D-6.10 reference is not left hunting.
- **Files modified:** `src/dftorch/Constants.py`, `docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md`,
  `tools/check_verdict_doc.py`, `tests/test_shell_resolved_scf_f.py`
- **Committed in:** `997c089`

### Acceptance criterion that was unsatisfiable on arrival

**`src/dftorch/Constants.py` does not decode cleanly as ASCII, and did not before this plan.** The
criterion asks for `max(bytes) < 128` to exit 0. Measured on the file as it stood at commit
`cef305d`, before this plan touched it: **291 non-ASCII bytes**, from pre-existing box-drawing
section separators such as `# --- DFTB3: Hubbard derivatives dU/dq ---` written with U+2500. The
criterion could not have been met by any edit short of reformatting other phases' comments, which
is out of scope and is the churn plan 06-02 was burned by.

What was done instead is the substance of the rule: **the added comment introduces zero non-ASCII
bytes.** The count is 291 before and 291 after. My first draft did use box-drawing characters for
its own header and footer, matching the surrounding house style; those were caught by the byte
count and rewritten as `---` before the commit.

## Known limitations, carried forward deliberately

- **The verdict is a claim about definitions, and it validates no settled number.** -6.920251 eV
  and -5.639887 eV appear as evidence, not as approved answers. D-6.08 froze nothing this phase and
  the settled curve's gate is a human looking at a graph - plan 06-05.
- **The document's citation gate protects five locations, not every claim.** It cannot tell whether
  the verdict is *correct*; it can only tell whether the document still says what it set out to say
  and whether its citations still land where it claims. The physics claims are held by
  `tests/test_energy_definitions_f.py`, and the split table's numbers are held by neither - they
  are prose evidence by design, because turning them into assertions is what D-6.08 forbids. If
  the split ever changes materially, nothing automated will notice; the document's measurement date
  is stated so a reader knows how old it is.
- **`tools/` is a new top-level directory.** Nothing else lives there yet and no test sweeps it.
- **Task 2 has no RED commit.** It is marked `tdd="true"` in the plan but adds no production code -
  the behaviour it describes is what Phase 4 D-11 and plan 06-01 already shipped. A failing-first
  commit would have had nothing to make pass, so one was not faked; the commit message says so.
  The plan-level TDD gate sequence (a `test(...)` commit followed by a `feat(...)` commit) therefore
  does not apply to this plan, which ships no `feat`.

## Observation recorded, not fixed

**`src/dftorch/Constants.py:5` imports `numpy as np` and never uses it.** Surfaced by the editor's
diagnostics while the comment was being added. It predates this plan, is in no way caused by it,
and `ruff` is not installed in this environment (`uv run ruff check` reports `program not found`),
so it is logged rather than fixed - fixing an unrelated import inside this plan's diff is the
scope creep the phase has already had to revert once.

## Threat coverage

| Threat | Disposition |
|---|---|
| T-06-20 (a verdict that concludes nothing) | The verdict is the first paragraph; `check_verdict_doc.py` fails if the phrase is missing **or if it appears only after the second heading** |
| T-06-21 (an explanation accounting only for electron repulsion) | The three-way split table plus the share stated in words; the checker requires both `84.2 %` and `84 percent`, so deleting either fails |
| T-06-22 (the Phase 4 pin read as the settled answer, or the two made to agree) | The document scopes the pin to the one-pass path; `test_the_phase_four_pin_is_not_the_settled_answer` is a floor on a difference and fires on agreement |
| T-06-23 (the s-group defect silently fixed later) | Recorded at the assignment itself with its measured size; `test_the_per_atom_strength_still_comes_from_the_s_group` (plan 06-03) turns an edit into a visible decision |
| T-06-24 (a whole-file ROADMAP rewrite destroying other phases) | Scoped `Edit`, never a `Write`; phase-heading count 10 before and 10 after, and all three existing plan checkmarks verified intact |
| T-06-25 (a measured number leaking into an assertion) | Audited above; the pinned literal grep returns 0 and every assertion is a zero, sign, inequality or ratio |
| T-06-26 (a filesystem path interpolated into the comment or document) | Both are hand-written prose. The only paths either contains are repository-relative source references written by hand (`docs/...`, `tests/...`), which is what the plan's action asked for; nothing is interpolated from the filesystem at runtime |
| T-06-SC (package-manager installs) | Not applicable - nothing was installed. `tools/check_verdict_doc.py` imports `pathlib` and `sys` only |

## Threat Flags

None. This plan adds one document, one standard-library script that only reads files inside the
repository, one test module, and comment text. It opens no network endpoint, no authentication
path and no new file-access pattern, and changes no schema.

## Known Stubs

None. No placeholder, empty return or "coming soon" text was introduced. The verdict document's
"What this document does not claim" section is a scope statement about validated numbers, not a
stub - the claims it declines to make belong to plan 06-05's human gate by design.

## Next Phase Readiness

**Ready for plan 06-05.** That plan recomputes the Eu-N binding curves and puts the graph in front
of a human. Two things from here bear on it:

- **It should overlay the default settled curve, and it should know that a third curve exists.**
  The per-orbital-group path gives a different energy at every separation (-5.64 vs -6.92 eV at
  2.655 A) and takes 80 passes against 28. Whether the graph shows two curves or three is 06-05's
  call, but showing the default one without saying which charge description it used would repeat
  the exact confusion this plan's document exists to prevent.
- **The verdict document is the written half of the phase's human gate.** `06-VALIDATION.md`'s
  second manual-only row is discharged by it; the first row, human judgement of the freshly
  computed graph, remains 06-05's blocking checkpoint and is untouched here.

## Self-Check: PASSED

All three created files and all three modified files exist on disk, as does this summary. All three
task commits are present in `git log`: `477dcc5`, `6a9ba7b`, `997c089`. The full suite was re-run
to completion after the last edit and reported 254 passed in 372.35 s, and
`python tools/check_verdict_doc.py` was re-run after the final line-number change and exited 0.

---
*Phase: 06-self-consistent-scf-for-f-systems*
*Plan: 04*
*Completed: 2026-08-06*

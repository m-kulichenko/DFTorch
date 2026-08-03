---
plan: 05-07
phase: 05-regression-safety-and-support-policy-cleanup
requirements: [CLN-01, CLN-02, CLN-04, CLN-05, REG-01]
status: complete
completed: 2026-08-03
---

# 05-07 Summary — Basis metadata map, prototype branches, f support matrix

## Recovery note

The executor was terminated mid-flight by a session limit after committing Task 1, with Tasks
2 and 3 having produced their documents but not committed them. The orchestrator recovered
rather than re-dispatching: both documents were inspected for completeness, the machine half
they claim was run, and they were committed as `0c9dbd9`. No work was redone and none lost.

This is the second recovery this phase (05-05 was the first). Both were survivable because
every task committed the moment it was done.

## Commits

| Commit | Task |
|---|---|
| `0ce95a2` | Map the basis metadata and pin the duplicated tables (CLN-02) |
| `0c9dbd9` | Record f support status and prototype branches (CLN-01, CLN-04) |

## CLN-02 — a map plus consistency tests, not a refactor

`05-CONTEXT.md` lists a single-source basis module as a **Deferred Idea**, so this plan did
not build one. What it did instead is assert cross-module relations that nothing previously
asserted — `Constants.shell_dim` against `Structure.SHELL_DIMS`, and the `_CHANNELS` /
`AO_LABEL_TEMPLATE` orderings against each other.

The argument for why those relations need asserting is concrete rather than theoretical:
plan 05-06 found `n_orb_per_shell = torch.tensor([0, 1, 3, 5])` in `_spin` and `_forces` —
the per-shell AO table truncated one entry short of f — sitting alongside a correct
`Constants.shell_dim = [0, 1, 3, 5, 7]`. Two copies of the same table, one wrong, nothing
comparing them. That is exactly the failure mode CLN-02 exists to make visible.

## CLN-04 — status established from code, not from the roadmap

`docs/F-SUPPORT-STATUS.md` sets a rule worth preserving: a capability is **deferred** because
a named guard refuses it today at a stated line and a test drives a real f system into that
guard — not because the roadmap intends to deliver it in a later phase. Where 05-06 already
wrote such a test, the document names the test rather than re-verifying by hand; where none
exists, the row says so.

**No row is `unsupported`.** Every capability CLN-04 names refuses explicitly. That is the
finding, not an omission, and it is the direct product of D-04: 05-06 dispositioned 157
orbital-count sites and guarded the six that were open.

The matrix cross-checks against the four exception classes in `_slater_koster_pair.py`, whose
taxonomy `test_no_new_f_exception_class_was_defined` pins at four — so the document is
checked against code, not against prose.

## CLN-01 — prototype branches, each falsifiable by hand

`docs/PROTOTYPE-BRANCHES.md` records every deliberate temporary path with a file, a line, a
requirement id and the phase that lifts it. Its framing matters: this is structure holding
something up, not oversight, and it should not be tidied away by someone who has not read
what it is holding.

Examples it records: the additive `R_orb` export kept alive for four out-of-scope consumers;
the truncated `n_orb_per_shell` table left behind a refusal rather than widened past
parameters that do not exist (no `spinw.txt` ships for the f fixtures); the verbose flag's
deliberately noisy default.

**Honest about its own limits.** `05-VALIDATION.md` states CLN-01 admits no automation beyond
"the section exists and names files that exist." Both documents say so rather than implying a
stronger gate.

## Verification

- `tests/test_support_documentation.py` — 4 passed
- `tests/test_orbital_count_guards.py` — 37 passed
- `tests/test_verbose_flag.py` — 6 passed

Run per-file and summed; the full suite exceeds the orchestrator's 10-minute command cap,
which is a measurement constraint rather than a hang. Suite grew 72 → 200+ across this phase.

## Self-Check: PASSED

Both claimed commits resolve in `git log`. `docs/F-SUPPORT-STATUS.md` (211 lines) and
`docs/PROTOTYPE-BRANCHES.md` (226 lines) exist. All three gating test files pass.

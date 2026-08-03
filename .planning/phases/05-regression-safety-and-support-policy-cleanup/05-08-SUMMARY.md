---
plan: 05-08
phase: 05-regression-safety-and-support-policy-cleanup
requirements: [CLN-01, CLN-04, CLN-05]
status: complete
disposition: discharged-with-scope-noted
completed: 2026-08-03
---

# 05-08 Summary — Human sign-off on the phase records

## Disposition: DISCHARGED, with the scope of what was actually reviewed stated plainly

The plan asked the reviewer for four checks. **Two were retracted by the orchestrator as
bogus, one was already discharged, and one was real and acted on.** Recording that honestly
rather than claiming a four-part sign-off that did not happen.

### Item 1 — read a row from each inventory: RETRACTED

The orchestrator asked whether `dropped` rows in `docs/SCRIPT-PY-PORT-INVENTORY.md` "name a
test that covers what was dropped, or just assert the check was redundant." On inspection the
question was malformed. That document defines `dropped` as "no behavior assertion is lost; the
evidence states why" — the category exists for scaffolding that carried no invariant
(`find_project_root`, `ensure_fake_dftorch_package`, `assert_int_metadata`). There is no test
to name because there is nothing to cover, and every row does give a specific reason. Items
that carried invariants are marked `ported` or `covered` and do name tests.

**Not reviewed by the human, because the check was withdrawn as incorrect.**

### Item 2 — trigger a refusal and judge its message: REAL, reviewed, acted on

The orchestrator's instruction (`pytest ... -v`) was also wrong: passing tests print nothing,
so the reviewer ran it, saw eight passes and no message. The message was then produced
directly from source and reviewed.

The orchestrator claimed the message failed to tell a user what to do. **That claim was wrong**
— it was made after reading only the `_spin`-specific prefix, not the shared
`F_SPIN_POLARIZATION_UNSUPPORTED_MESSAGE` appended to it, which already carries a workaround
("run the system closed-shell by omitting UNRESTRICTED... a defined calculation rather than a
wrong one").

The genuine gap, once the whole message was read, was provenance: nothing let a maintainer
trace *which* decision deferred spin or *which* requirement lifts it. Fixed in `8889cda` —
the message now names phase 4 decision D-12, requirement SPN-01, and `docs/F-SUPPORT-STATUS.md`.

**Reviewed by the human, who authorised the change.**

### Item 3 — check the baseline ordering: ALREADY DISCHARGED

The orchestrator had already run `git log --oneline --reverse` and read the result before
asking. `130dbea` (plan 05-02's baseline) is the first Phase 5 commit, preceding every change
it guards, so the unchanged-behaviour claim rests on a baseline measured beforehand. There was
nothing left for a reviewer to do and it should not have been listed as a task.

Worth recording from that log: `56091af`, an unrelated commit, sits between 05-01's work and
everything after it, and is what silently deleted the per-pair lookup wiring.

### Item 4 — read F-SUPPORT-STATUS's REG-02 row: TARGET MISIDENTIFIED

**There is no REG-02 row in `docs/F-SUPPORT-STATUS.md`.** That document has four sections
(support matrix, exception classes, what is supported today, known open issues) and mentions
REG-02 once in passing. The orchestrator invented the row.

The REG-02 narrowing does exist, in `docs/REG-02-NOTEBOOK-BASELINE.md`, which opens with
"**Status: satisfied in substance, not in form**" and carries both "Why the notebook itself is
not executed" and "What this does NOT cover". So the substance the check was after is present
and plainly stated — in a different file than the one named.

**Confirmed present by the orchestrator; not reviewed by the human, since the pointer was wrong.**

## What this means for the phase

Phase 5's work is committed, pushed, and green. The records exist and their automated gates
pass. What this checkpoint did **not** achieve is its actual purpose: an independent human read
of whether the inventory rows *say* anything, as opposed to merely being counted. Two of the
four prompts for that read were defective and were withdrawn rather than answered.

That is a limitation of this checkpoint, not of the underlying work, and it is recorded here so
a later reader does not mistake "phase sealed" for "every record was independently read."

If that read matters later, the useful version is narrow: open
`docs/ORBITAL-COUNT-INVENTORY.md`, pick any row marked `unreachable`, and ask whether it says
*how* unreachability was established or merely asserts it. That is the one question none of the
automated gates can answer.

## Verification at sign-off

- Suite: **216 passed / 0 failed** across 18 files (entering baseline was 72)
- `.planning/WINDOWS.md`: `open_count: 0`, `fixed_count: 2`
- All 11 phase requirements (REG-01..06, CLN-01..05) marked complete
- Pushed: `c7fb003..8889cda` to `origin/f_orbital_initial`, 0 commits ahead

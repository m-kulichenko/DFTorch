---
phase: 04-scf-and-reference-simulation-validation
plan: 05
subsystem: testing
tags: [validation, human-verify, checkpoint, f-orbitals, energy-scan, dftb, sign-off]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "plan 04-04's measured Eu-N binding curve and the passing SIM-05 band gate — the object of this review"
  - phase: 04-scf-and-reference-simulation-validation
    provides: "plan 04-01's forward(do_scf=False) branch — the code path the curve was recomputed through at review time"
provides:
  - "A recorded human verdict on the physical plausibility of the Eu-N minimum: APPROVED"
  - "Both manual-only verifications in 04-VALIDATION.md discharged with recorded verdicts"
  - "An explicit, human-accepted disposition on the non-asymptotic long-range tail of the curve"
  - "Recorded acknowledgement that the four unsampled limitations are consequences of D-11/D-12/D-14/D-20/D-22, not new discoveries"
affects: [deferred self-consistent-SCF phase, deferred spin-polarized-f phase, phase 05 regression-safety, milestone audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A checkpoint review recomputes the observable from live code rather than replaying the recorded table, so the sign-off is on the current tree and not on a stale artifact"
    - "A two-panel plot (full range + well zoom) when the feature under review is two orders of magnitude smaller than the dominant term on the same axis"

key-files:
  created:
    - .planning/phases/04-scf-and-reference-simulation-validation/04-05-SUMMARY.md
  modified:
    - .planning/phases/04-scf-and-reference-simulation-validation/04-VALIDATION.md

key-decisions:
  - "SIM-05 disposition is APPROVED: the Eu-N curve is a single clean well with its minimum at 2.40 A, inside the 2.655 A +/-20% band, and a human judged it physically plausible"
  - "The band is NOT changed. Disposition is approved, not widen-band, so no new CONTEXT.md decision is required and EU_N_TARGET_ANGSTROM / EU_N_BAND_FRACTION remain untouched (threat T-04-11)"
  - "D-24's short-side escalation branch was NOT triggered as a failure — the minimum landed inside the band. The second manual-only row is discharged as not-triggered, not as exercised"
  - "The curve not reaching a dissociation asymptote by 3.60 A was explicitly surfaced to the human and explicitly accepted as expected for single-shot non-SCC energy over this range — reviewed-and-accepted, not a carried-forward open concern"

patterns-established:
  - "A manual-only validation row is discharged with one of {verdict recorded, not triggered}, and not-triggered is stated as such rather than dressed up as exercised"

requirements-completed: [SIM-05]

coverage:
  - id: D1
    description: "Physical plausibility of the located Eu-N minimum judged by a person reading the actual curve, not inferred from a passing band assertion (04-VALIDATION.md manual-only row 1)"
    requirement: SIM-05
    verification:
      - kind: manual_procedural
        ref: "human review at the 04-05 blocking checkpoint of a two-panel plot of the binding curve (eu_n_binding_curve.png, repo root, untracked), recomputed from live code at review time"
        status: pass
      - kind: other
        ref: "uv run pytest -q at review time — 72 passed / 0 failed, exit 0"
        status: pass
    human_judgment: true
    rationale: "The 2.655 A target is a mean over dative Eu-N bonds in crowded 8-9-coordinate Eu-Bp complexes while the case is an isolated diatomic; no predicate can weigh those against each other. Discharged at this plan with disposition APPROVED — a verifier should confirm the recorded verdict rather than re-derive it."
  - id: D2
    description: "D-24's asymmetric short-side rule adjudicated by a human rather than a machine concluding the f implementation is broken (04-VALIDATION.md manual-only row 2)"
    requirement: SIM-05
    verification:
      - kind: manual_procedural
        ref: "human review at the 04-05 blocking checkpoint — minimum at 2.40 A is inside [2.124, 3.186] A, so the short-side branch was not triggered"
        status: pass
    human_judgment: true
    rationale: "Discharged as NOT TRIGGERED: the minimum landed inside the band, so no short-side adjudication was required. Recorded as a legitimate disposition, not as an exercised branch. The branch's machinery was separately mutation-verified in plan 04-04 (M5)."
  - id: D3
    description: "The four limitations 04-VALIDATION.md records as unsampled by this phase are explicitly acknowledged, with their originating decisions named"
    requirement: SIM-05
    verification:
      - kind: manual_procedural
        ref: "human accepted all four at the 04-05 checkpoint; recorded in this summary under '## The four unsampled limitations'"
        status: pass
    human_judgment: true
    rationale: "Acceptance of known unsampled risk is a judgement about what the project is willing to ship, not a testable property."

# Metrics
duration: 5min
completed: 2026-07-29
status: complete
---

# Phase 4 Plan 05: SIM-05 Human Sign-Off Summary

**A human read the actual Eu-N binding curve, judged it a single clean well with its minimum at 2.40 Å inside the 2.655 Å ±20% band, and APPROVED it — with the non-asymptotic long-range tail explicitly surfaced and explicitly accepted, the band left untouched, and both of `04-VALIDATION.md`'s manual-only verifications discharged.**

## Performance

- **Duration:** 5 min (recording only — the checkpoint review itself preceded this agent)
- **Started:** 2026-07-29T22:49:00Z
- **Completed:** 2026-07-29T22:54:11Z
- **Tasks:** 1 (a `checkpoint:human-verify`, presented and answered)
- **Files modified:** 2 (both under `.planning/`; no source or test file touched)

---

## Disposition: APPROVED

Of the plan's three permitted dispositions — **approved** / investigate / widen-band-with-new-decision — the recorded verdict is **approved**.

Consequences, stated explicitly because they are prohibitions the plan carries:

- **No new decision is appended to `04-CONTEXT.md`.** A new decision is required only for a widen-band disposition.
- **No band constant is edited.** `EU_N_TARGET_ANGSTROM` (2.655) and `EU_N_BAND_FRACTION` (0.20) in `tests/test_eu_n_scan.py` are unchanged and were confirmed unchanged after the review (`git diff --stat -- tests/` is empty). This is exactly what threat T-04-11 guards against: a band quietly widened under cover of a sign-off.
- **No code was changed by this plan at all.**

---

## What the human actually reviewed

A **two-panel PNG plot** of the binding curve — `eu_n_binding_curve.png` at the repository root, untracked, generated at review time by **recomputing the curve from live code rather than replaying the recorded table** in `04-04-SUMMARY.md`. The sign-off is therefore on the current tree, not on a stale artifact.

- **Left panel:** the full 1.60–3.60 Å scan.
- **Right panel:** a zoom on the well (r ≥ 2.05 Å). The zoom was necessary because the ~0.83 eV well depth is invisible against the ~14 eV repulsive wall on a single linear axis — a reviewer looking only at the full-range panel would see a wall and a flat line, and could not judge well shape at all.
- Both panels carry the band edges, the 2.655 Å target, and the bulk rock-salt EuN reference strip.

The plot was kept at the human's choice. It remains **untracked and uncommitted**; it is not deleted, moved, or added to `.gitignore`.

## Observed values

Verified **independently by the orchestrator at review time**, not merely copied from `04-04-SUMMARY.md`:

| Property | Observed |
|---|---|
| **Located minimum** | **2.40 Å**, grid index **8 of 20** |
| Energy at the minimum | **-17.64144156 eV** |
| Strictly interior (not clipped at either grid edge)? | **yes** |
| Inside the D-24 band [2.124, 3.186] Å? | **yes** |
| Strictly monotone **decreasing into** the minimum? | **yes** |
| Strictly monotone **increasing out of** the minimum? | **yes** |
| Unique on the grid — no tie to break? | **yes** |
| Full suite at review time | **72 passed / 0 failed** |

## Curve-shape verdict: SINGLE CLEAN WELL

The plan requires one of four shape verdicts — **single well** / monotonic / multi-well / flat-within-numerical-noise. The recorded verdict is **single clean well**: not monotonic, not multi-well, and not flat within numerical noise across the band.

This matters because it is the precondition for the band assertion meaning anything. A band assertion over a monotonic or noise-flat curve would pass while proving nothing; the human read confirms there is a real turning point for the band to be asserted about.

## D-24 asymmetric rule: application

The minimum at 2.40 Å is **inside** the band, so **the short-side escalation branch was not triggered as a failure.** No adjudication of a short-of-band result was required.

What was noted about the location:

- 2.40 Å sits **below** the 2.655 Å target — **the direction D-24 anticipates.** The target is a mean over dative Eu-N bonds in crowded 8–9-coordinate Eu-Bp complexes; the test case is an isolated diatomic, which should be shorter.
- It also **coincides with the independent bulk rock-salt EuN reference (~2.45–2.50 Å).**
- **Two reference points agree**, in the direction the physics predicts.

**No band change was requested.** Therefore no new `CONTEXT.md` decision is required, and the band constants must remain untouched.

## Concern surfaced and accepted: the long-range tail

One concern was **explicitly surfaced to the human and explicitly accepted.**

**The concern.** The curve is still climbing at 3.60 Å — it does not reach a dissociation asymptote within the scanned range, whereas a true diatomic potential should level off toward the separated-atom limit.

**The choice offered.** Three options were put to the human: (a) approve with the asymptote accepted; (b) approve but log it as a carried-forward concern; (c) hold and extend the scan to 6–8 Å before deciding.

**The choice made.** **Approve — the asymptote is fine, expected for single-shot non-SCC energy over this range.**

Recorded status: **reviewed and accepted.** Not an unexamined gap, and **not** a carried-forward open concern. It is written down here so a future reader knows the tail was looked at and dispositioned on purpose, rather than missed.

## The four unsampled limitations

All four are **acknowledged and accepted.** Each is recorded with *why* it exists — and in every case the answer is that it is a direct consequence of a decision this user locked earlier in the phase, **not a new discovery at review time**:

1. **The ±20% band absorbs an f-block error that shifts the minimum by up to 15%.** From **D-20 / D-22**, which mandate a loose band and forbid tightening it.
2. **Single-shot is not SCF, so charge-self-consistency bugs cannot surface this phase by construction.** From **D-11**.
3. **Eu 4f⁷ is genuinely open-shell but is treated closed-shell.** From **D-12**, which defers spin and makes the spin path raise instead.
4. **The shell-resolved f plumbing is validated but never consumed by the single-shot path.** From **D-14**, which called for exactly this deliberate de-risking.

## Manual-only verifications: discharge

Both rows in `04-VALIDATION.md` § "Manual-Only Verifications" are now discharged with a recorded verdict:

| Row | Requirement | Verdict |
|---|---|---|
| **Physical plausibility of the located minimum** | SIM-05 | **DISCHARGED — APPROVED.** Single clean well; minimum 2.40 Å, interior, inside the band; monotone in and out; unique. Judged plausible for an isolated diatomic, agreeing with both the coordination-mean target (in D-24's predicted direction) and the independent bulk rock-salt EuN reference. |
| **Short-side band failure** | SIM-05 | **DISCHARGED — NOT TRIGGERED.** The minimum landed inside the band, so no short-side result existed to adjudicate. Recorded as not-triggered, which is a legitimate disposition; **it is not claimed that the branch was exercised.** The branch's own machinery was separately proven to fire correctly by plan 04-04's M5 mutation. |

`04-VALIDATION.md` was updated to record both verdicts and to mark the `04-05 T1` row of the Per-Task Verification Map green. Its `wave_0_complete` flag was flipped to `true` after confirming on disk that all six Wave 0 artifacts exist and the suite is green.

**`nyquist_compliant` was deliberately left `false`, and `status` left `draft`.** Setting `nyquist_compliant: true` asserts every item in the Validation Sign-Off checklist, several of which this plan did not verify (sampling continuity across tasks, watch-mode flags, feedback-latency measurement). `status: validated` is set by `validate-phase` § 6 per the file's own lifecycle comment, not by an executor. Neither flag is this plan's to claim.

## Files Created/Modified

- `.planning/phases/04-scf-and-reference-simulation-validation/04-05-SUMMARY.md` — new. This file: the recorded verdict, which is this plan's sole required artifact.
- `.planning/phases/04-scf-and-reference-simulation-validation/04-VALIDATION.md` — modified. Both manual-only rows given recorded verdicts; a `### Manual-Only Verification Discharge` subsection added pointing at this summary; the `04-05 T1` map row marked green; `wave_0_complete: true`. `status` and `nyquist_compliant` untouched.

**Not modified, deliberately:**

- `tests/test_eu_n_scan.py` — the band constants. Confirmed unchanged (T-04-11).
- Any file under `src/` or `tests/`. `git diff --stat -- tests/` is empty.
- `eu_n_binding_curve.png` — kept at the human's request, left untracked and uncommitted.
- `src/dftorch/_bond_integral.py`, `src/dftorch/_io.py`, `src/dftorch.egg-info/*` — pre-existing uncommitted edits that predate this phase. Left untouched, as in plan 04-04.

## Decisions Made

- **Approved rather than widen-band or investigate.** The minimum is interior, inside the band, and the curve is a clean single well; two independent references agree in the predicted direction. Nothing met the plan's bar for either alternative disposition.
- **The long-range tail is dispositioned as accepted, not carried forward.** Recording it as an open concern would have misrepresented a decision the human actually made.
- **Row 2 is discharged as not-triggered.** Claiming the short-side branch was exercised would be false; the plan's `<done>` criterion is a recorded verdict, and not-triggered is one.
- **`nyquist_compliant` and `status` left alone in `04-VALIDATION.md`** — see above. Flipping them would assert checks this plan did not run.

## Deviations from Plan

None — plan executed exactly as written. One task, a blocking `checkpoint:human-verify`, was presented and answered; the verdict is recorded above. No deviation rule fired because no code was executed or changed.

## Issues Encountered

- **The well is invisible on a single linear axis.** The ~0.83 eV well depth against a ~14 eV repulsive wall meant a naive single-panel plot could not support the shape judgement the plan asks for. Resolved by the two-panel presentation (full range plus a zoom from 2.05 Å), which is what the human actually reviewed. Worth recording because any future reviewer plotting this curve will hit the same problem.
- **The suite count line carries no summary text in this shell.** `addopts = ["-ra", "-q"]` prints a bare progress line. Confirmed as before by exit code 0 plus a character census: 72 dots, zero `F`/`E`/`s`/`x` markers.

## Threat Flags

None. This plan changed no code, read no external input, crossed no trust boundary, and produced two markdown files.

| Threat | Disposition | Outcome |
|---|---|---|
| T-04-11 — the band widened during review without a recorded decision | mitigate | **Mitigated and verified.** The disposition is approved, not widen-band, so no band change was warranted; `EU_N_TARGET_ANGSTROM` and `EU_N_BAND_FRACTION` were confirmed still 2.655 / 0.20 after review, and `git diff --stat -- tests/` is empty. |
| T-04-SC — package installs | mitigate / not applicable | **Remained inactive.** No packages installed. |

## Known Stubs

None. This plan's artifact is a recorded verdict and it is complete: no placeholder values, no TODO/FIXME markers, no skipped tests. The plan's `<verification>` block had two items and both were executed — the human read is recorded above, and `uv run pytest -q` was run at review time (72 passed / 0 failed, exit 0).

The four unsampled limitations above are **not** stubs. They are accepted scope boundaries traceable to D-11, D-12, D-14 and D-20/D-22, already carried in `04-VALIDATION.md` § Sampling Risk and in `tests/f_orbital_data/README-EU-N-CASE.md`.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Phase 4 is complete on both halves of its goal.** All five plans have summaries, the suite is 72 passed / 0 failed, and SIM-01 through SIM-05 are discharged — SIM-05 now including the human judgement that no predicate could express.

Ready for phase verification (`/gsd-verify-work`) and then phase 5 (regression safety and support policy cleanup), with these constraints intact and unchanged by this plan:

- **The band must not be edited by a later phase either.** Approval was of the *measurement*, not a licence to retune the tolerance. `test_tolerance_band_is_a_fraction_of_the_target` enforces the form and the ±10% floor mechanically.
- **Do not switch a Coulomb term on to "improve" the curve.** First-iterate Mulliken charges were measured in plan 04-01 to destroy the well entirely.
- **Do not replace the scan with an optimizer.** f derivatives still raise `FDerivativeUnsupportedError`.
- **The accepted long-range tail is the natural first item for the SCF phase to revisit** — extending the scan to 6–8 Å becomes meaningful once self-consistency lands. It is accepted here, not forgotten.
- Spin stays refused for f systems; the closed-shell treatment of Eu 4f⁷ is an approximation the band absorbs.
- Only the Eu-N pair is exercised, along a single direction. Seven of nine SKF fixtures stay untouched by design.
- `_bond_integral` still exports a single `R_orb` (the longest grid) for all pair types while the fixture directory mixes radial grids.

## Self-Check: PASSED

- Both claimed files exist on disk: `04-05-SUMMARY.md` (created) and `04-VALIDATION.md` (modified).
- **No test file was modified:** `git diff --stat -- tests/` is empty.
- **Band constants confirmed unchanged:** `EU_N_TARGET_ANGSTROM = 2.655` (line 99) and `EU_N_BAND_FRACTION = 0.20` (line 100) in `tests/test_eu_n_scan.py`.
- **No source file was modified by this plan.** `git diff -- src/` shows only the pre-existing `_bond_integral.py`, `_io.py` and `egg-info` edits that predate this phase and were present at plan start.
- `eu_n_binding_curve.png` still present at the repo root and still untracked, as the human chose.
- No commit hash is claimed for a task: this plan's only task was a checkpoint and produced no code commit.

---
*Phase: 04-scf-and-reference-simulation-validation*
*Completed: 2026-07-29*

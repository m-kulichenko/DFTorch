---
phase: 06-self-consistent-scf-for-f-systems
plan: 05
subsystem: testing
tags: [dftb, scf, binding-curve, matplotlib, figure, human-verdict, eu-n, shell-resolved]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "the approved one-pass Eu-N binding curve this figure overlays, the 21-point 1.60-3.60 A grid, and the two-panel lesson that a single linear energy axis cannot support a shape judgement"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 01
    provides: "structure.scf_iter_count, the only machine-readable signal of which separations the loop gave up on, and the Krylov switch-off that makes the per-atom curve settle at all 21"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 02
    provides: "the sixteen-block shell-resolved Coulomb matrix the third curve rests on"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 03
    provides: "the per-orbital-group charge loop behind MAGNETIC_HUBBARD_LDEP, which is the third curve"
  - phase: 06-self-consistent-scf-for-f-systems
    plan: 04
    provides: "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md, the written verdict that explains why the one-pass and settled curves sit ~10 eV apart and must not be expected to agree"
provides:
  - "experiments/eu_n_scf_binding_curve.py: one command that recomputes all three Eu-N curves from live code and draws them"
  - "figures/eu_n_scf_binding_curve.png: the two-panel figure the user judged (regenerated at review time, not committed)"
  - "matplotlib as a development-time dependency in pyproject.toml's dev extra, deliberately not a runtime one"
  - "The recorded human verdict on the settled binding curve: a qualified close, not an approval"
affects: [phase-07, phase-verification, shell-resolved-scf, gsd-ship]

# Tech tracking
tech-stack:
  added: [matplotlib>=3.11.1 (dev extra only)]
  patterns:
    - "A verification figure is regenerated from live code at review time and is not committed; the script that draws it is the artifact, the PNG is output"
    - "Where a charge loop gave up is shaded as a vertical band across both panels, not marked on the point - a marker on a runaway point disappears the moment the point leaves the panel"
    - "The energy panel's vertical range is set by the separations that settled, so a +359 eV runaway cannot flatten the wells the human is being asked to judge"
    - "One legend for a stacked-panel figure, placed in whichever panel has genuinely empty space; a legend is never allowed to sit on the curve being judged"

key-files:
  created:
    - experiments/eu_n_scf_binding_curve.py
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "matplotlib is a development-time dependency, in the dev extra, not a runtime one. The user's reasoning: the library must produce the data the figures need, not the figures. This reverses the standing position recorded in experiments/diatomic_scans/README.md."
  - "All three curves are shown with the 9 gave-up separations shaded, rather than hiding the failing one. The user chose this explicitly: 'Record it and show all three.'"
  - "The phase closes with the shell-resolved defect open, on the user's ruling, rather than being blocked by D-6.02's looks-wrong rule or reverted."
  - "The figure is not committed. The six existing Phase 4/5 figures are untracked too; the plan's own artifact table calls it 'regenerated at review time', and a committed PNG is a replay by another name."
  - "The energy panel carries no legend. Measured against these curves there is no empty region in it large enough for a five-entry box, and every placement buried part of the curve under judgement."

patterns-established:
  - "Human-gate figures: the shape being judged is never obscured, by a legend or by autoscaling around a runaway"
  - "A checkpoint answer is recorded in the user's own words, including when it is negative"

requirements-completed: [SCC-01, SCC-02, SCC-03]

coverage:
  - id: D1
    description: "One command recomputes all three Eu-N curves from live code at review time and writes the figure - nothing replayed from a recorded table or cached file"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "uv run python experiments/eu_n_scf_binding_curve.py (exit 0; 21 rows; figure written 2026-08-07T17:31:30Z)"
        status: pass
      - kind: unit
        ref: "grep -c 'json\\|\\.load(' experiments/eu_n_scf_binding_curve.py -> 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "The approved one-pass curve is drawn beside the settled one, so the human judges a change rather than an isolated shape"
    requirement: SCC-01
    verification:
      - kind: manual_procedural
        ref: "figures/eu_n_scf_binding_curve.png, upper panel, three labelled series on one axis"
        status: pass
    human_judgment: true
    rationale: "Whether the overlay supports the comparison is a judgement about a picture; the user made it."
  - id: D3
    description: "Every separation where the charge loop gave up is visibly marked, so a nonsensical point cannot be read as a physical feature"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "9 separations shaded on both panels from structure.scf_iter_count == -1; converged-of-21 count in the legend"
        status: pass
    human_judgment: false
  - id: D4
    description: "The figure shows how much charge moves between the atoms, where a runaway is legible even when the energy scale hides it"
    requirement: SCC-01
    verification:
      - kind: manual_procedural
        ref: "figures/eu_n_scf_binding_curve.png, lower panel, full vertical range"
        status: pass
    human_judgment: true
    rationale: "The panel exists so a human can see a runaway; that it does so is a visual judgement."
  - id: D5
    description: "The per-orbital-group result this phase built is on the plot, so the human sees the deliverable rather than a description of it"
    requirement: SCC-02
    verification:
      - kind: manual_procedural
        ref: "figures/eu_n_scf_binding_curve.png, red series - and it is what the user objected to"
        status: pass
    human_judgment: true
    rationale: "Shown and judged. The verdict on it was negative; see the verdict section."
  - id: D6
    description: "A human has looked at the freshly computed settled binding curve and given a verdict, recorded verbatim"
    requirement: SCC-03
    verification:
      - kind: manual_procedural
        ref: "06-05-SUMMARY.md#the-verdict-verbatim (2026-08-07)"
        status: pass
    human_judgment: true
    rationale: "Decision D-6.02 makes human judgement of a freshly computed graph the phase's completion bar, in place of a numeric one. No assertion substitutes."

# Metrics
duration: ~135min across two sessions
completed: 2026-08-07
status: complete
verdict: qualified-close
---

# Phase 6 Plan 05: The Human Looks at the Graph Summary

**The settled per-atom curve is a single clean well and settles at all 21 separations, where Phase 5 ran away at 9 of them - but the per-orbital-group curve the phase was built for spikes and loses charge control at long separation, the user said so, and the phase closes with that defect open rather than approved or reverted.**

## Performance

- **Duration:** ~135 min across two sessions (2026-08-06 and 2026-08-07). The second session existed only because matplotlib was absent and the first session raised that as a blocking checkpoint rather than installing it.
- **Completed:** 2026-08-07
- **Tasks:** 2
- **Files:** 1 created, 2 modified. No file deleted in any commit (`git diff --diff-filter=D` after each).

## The verdict, verbatim

The user looked at `figures/eu_n_scf_binding_curve.png`, regenerated from live code at
2026-08-07T17:31:30Z from the tree at commit `487b347`, and said:

> "Okay, the red graph looks very bad, we have a random spike in the energy and terrible charge
> convergence at longer distances. Something is probably wrong."

Then, after being shown what Phase 6 had changed and what a revert would cost:

> "Let's mark phase 6 as complete with known issues. Write the summary doc. We'll change up what we
> do in future phases"

**This is not an approval.** It is also not a rejection of the phase. It is a **qualified close**: the
phase ships, and the shell-resolved path ships broken and labelled.

### Reading it precisely

- **The complaint is specific to the red curve** - "settled, one charge per orbital group", the
  path selected by `MAGNETIC_HUBBARD_LDEP`. Two things were named: a random spike in the energy, and
  terrible charge convergence at longer distances. Both are real and both are in the table below.
- **The navy curve - settled, one charge per atom - was not objected to.** It was not praised
  either. Nothing in the verdict addresses it, and it should not be recorded as approved.
- **The user declined to revert.** They were told what a revert would cost and chose to close with
  the defect recorded and to change approach in later phases.
- **This overrides decision D-6.02's default.** D-6.02 says a looks-wrong verdict fails the phase
  gate and is not a finding to record and move past, and this plan's acceptance criteria repeat
  that. The user, who set that bar, has now set it aside for this phase in their own words. That is
  a new user decision, recorded here; it is not the executor deciding a negative verdict was
  survivable. It has not been written into `06-CONTEXT.md`, which would need a new numbered
  decision entry.

### The shape judgement, in the user's terms

| Series | Shape | Lowest point | Converged |
|---|---|---|---|
| One pass, charge never revisited | single well (Phase 4's approved shape, unchanged) | 2.40 A, -17.641442 eV | n/a - runs no loop |
| Settled, one charge per atom | single well | 2.00 A, -10.997697 eV | **21 of 21** |
| Settled, one charge per orbital group | **"very bad" - a well with a random spike out of it, then loss of control** | 2.00 A, -10.114978 eV | **12 of 21** |

The user did not use the words "single well", "straight slope", "two wells" or "flat within noise"
for any series. The words they used about the red one were "very bad", "a random spike in the
energy", "terrible charge convergence at longer distances" and "something is probably wrong". Those
are recorded above rather than mapped onto the plan's vocabulary, because mapping them would be
paraphrase.

### The short lowest point was not raised

The plan anticipated the user might object that both settled curves bottom out at 2.00 A, shorter
than Phase 4's tolerance band starting at 2.124 A. **They did not raise it.** Accordingly
`EU_N_TARGET_ANGSTROM` and `EU_N_BAND_FRACTION` are untouched, as decision D-24 requires, and
nothing in this plan tests any curve against that band
(`grep -n "2.124\|3.186\|EU_N_BAND" experiments/eu_n_scf_binding_curve.py` returns nothing).

### `06-VALIDATION.md`'s first manual-only row

**Discharged.** "The self-consistent Eu-N binding curve looks right", SCC-01 under D-6.02, was put
to a human at a blocking checkpoint on a figure regenerated at review time, with both mandated
curves overlaid and every non-converged separation marked. The answer came back negative on the
third series and the phase closes anyway on the user's explicit instruction. The row is answered,
not skipped - which is the distinction that row exists to protect.

## The measurement the verdict was given on

One run, 2026-08-07T17:30:45Z to 17:31:30Z, 63 points (21 separations, three treatments of charge),
under torch 2.10.0+cpu, numpy 2.4.4, scipy 1.17.1, Python 3.13.0 - the same stack every number in
this phase was measured against.

```
  r (A) |     E one pass |    E per atom |   E per group |  qN one pass |  qN per atom | qN per group | it per atom | it per group
  ---------------------------------------------------------------------------------------------------------------------------------
   1.60 |      -3.006103 |     -0.410807 |      0.777377 |     1.397735 |     0.542033 |     0.488704 |          13 |           25
   1.70 |      -9.687629 |     -6.409461 |     -5.334785 |     1.589884 |     0.569039 |     0.506615 |          14 |           19
   1.80 |     -13.470390 |     -9.439071 |     -8.463193 |     1.777708 |     0.590154 |     0.517681 |          14 |           21
   1.90 |     -15.557092 |    -10.719673 |     -9.810340 |     1.954469 |     0.605770 |     0.524208 |          17 |           31
   2.00 |     -16.677763 |    -10.997697 |    -10.114978 |     2.114550 |     0.617180 |     0.528211 |          19 |           31
   2.10 |     -17.257014 |    -10.717228 |     -9.821207 |     2.254049 |     0.625941 |     0.531009 |          20 |           33
   2.20 |     -17.518339 |    -10.129834 |     -9.182658 |     2.371850 |     0.632416 |     0.533433 |          27 |           28
   2.30 |     -17.620714 |     -9.424637 |     21.773461 |     2.469645 |     0.635120 |     2.135192 |          28 |           -1
   2.40 |     -17.641442 |     -8.696156 |     -7.558421 |     2.550571 |     0.632262 |     0.536892 |          29 |           87
   2.50 |     -17.612784 |     -7.977710 |     -6.743243 |     2.617993 |     0.623639 |     0.530642 |          29 |           56
   2.60 |     -17.551797 |     -7.283848 |     -6.001806 |     2.674945 |     0.609577 |     0.512112 |          29 |           60
   2.70 |     -17.475283 |     -6.636423 |     -5.369967 |     2.723913 |     0.590532 |     0.487134 |          27 |           86
   2.80 |     -17.400885 |     -6.059714 |     -4.851072 |     2.766746 |     0.569492 |     0.462496 |          28 |           44
   2.90 |     -17.334547 |     -5.557098 |    176.271048 |     2.804670 |     0.549568 |     3.018754 |          29 |           -1
   3.00 |     -17.247612 |     -5.090483 |    106.753814 |     2.838387 |     0.531997 |    -2.841331 |          27 |           -1
   3.10 |     -17.150273 |     -4.659414 |     34.744279 |     2.868223 |     0.516887 |     2.873463 |          29 |           -1
   3.20 |     -17.062560 |     -4.275004 |    181.271731 |     2.894300 |     0.503945 |     3.006501 |          29 |           -1
   3.30 |     -16.985005 |     -3.931544 |    349.450294 |     2.916688 |     0.492788 |    -4.999168 |          28 |           -1
   3.40 |     -16.917980 |     -3.625795 |    142.790841 |     2.935499 |     0.483059 |    -3.015004 |          28 |           -1
   3.50 |     -16.861443 |     -3.356250 |    356.261310 |     2.950945 |     0.474450 |    -5.000018 |          28 |           -1
   3.60 |     -16.814891 |     -3.122492 |    359.317218 |     2.963334 |     0.466688 |    -5.000011 |          28 |           -1

  Settled, one charge per atom: converged at 21 of 21 separations
  Settled, one charge per orbital group: converged at 12 of 21 separations
    gave up at: 2.30 A, 2.90 A, 3.00 A, 3.10 A, 3.20 A, 3.30 A, 3.40 A, 3.50 A, 3.60 A
```

Energies in eV, charges in electrons with positive meaning electrons moved onto nitrogen, and the
last two columns are how many passes the repeat-until-settled loop took, with `-1` meaning it gave
up. These reproduce the first session's measurement exactly, digit for digit.

The user's "random spike" is the `21.773461` at 2.30 A, sitting between `-9.182658` and `-7.558421`.
Their "terrible charge convergence at longer distances" is the last eight rows, where the charge on
nitrogen swings between +3.0 and -5.0 electrons instead of drifting gently near +0.5.

## What was and was not affected by this

**The default path is untouched.** The per-orbital-group loop is opt-in, gated at a single `if` in
`src/dftorch/ESDriver.py:195` (`if getattr(const, "magnetic_hubbard_ldep", False)`) and defaulting
to `False`. Nothing runs it unless asked. The default-path numbers plan 06-04 re-measured are
unchanged: **-17.510444 eV** one pass and **-6.920251 eV** settled per atom.

**The per-atom settled curve is the phase's working result.** 21 of 21 separations settle, against
Phase 5's 9 runaways with energies at +198 eV. That is what plan 06-01's Krylov switch-off bought,
and the figure shows it.

## Task Commits

1. **Task 1: recompute all three curves and render the figure** - `936ae8e` (the script, 774 lines),
   `24f67b3` (matplotlib into the dev extra), `4edbff6` (the script's docstring pointed at where
   matplotlib now lives), `487b347` (the legend moved off the curve it was covering).
2. **Task 2: the human looks at the graph** - no code. The verdict is this document.

## Accomplishments

- **One command recomputes everything and draws it.** `uv run python
  experiments/eu_n_scf_binding_curve.py` builds `Constants` once per charge resolution, runs 63
  physics calculations, prints the 21-row table with both tallies, and writes the figure. About 45
  seconds. Nothing is read from a recorded table, a cached JSON file or a research document:
  `grep -c "json\|\.load("` on the script returns 0.
- **The failures are shown, not hidden.** All nine separations where the loop gave up are shaded as
  vertical bands across both panels, each failing point carries an X, and each settled series'
  legend entry carries its converged-of-21 count. This came from the user's own ruling on the open
  question - "record it and show all three" - and it is what let them see the defect and name it in
  one sentence.
- **The energy panel's vertical range is set by the points that settled.** A +359 eV runaway on
  autoscale would compress every well into a band a few pixels tall, which would have destroyed the
  exact judgement the figure exists to support. Runaway points visibly leave the panel instead, and
  the shading is what tells the reader they left rather than being missing.
- **matplotlib is now a development-time dependency and not a runtime one**, on the user's ruling.
  Installed with `uv add --optional dev matplotlib`; `pyproject.toml`'s dev extra now reads
  `["pytest>=7.4", "ruff", "mypy", "pre-commit", "matplotlib>=3.11.1"]`. The numerical stack was
  re-checked afterwards and is unchanged.
- **Full suite: 254 passed, 0 failed**, run immediately before the checkpoint (371.49 s), and again
  after the first session's work (353.81 s). The install added no regression.

## Files Created/Modified

- `experiments/eu_n_scf_binding_curve.py` (790 lines, new) - the script. Pure ASCII in the source
  and in every string it emits ("A" for Angstrom, "->" for an arrow), verified by a byte scan. It
  checks for matplotlib *before* spending any time on physics, so a bare interpreter costs the
  reader two seconds rather than two minutes.
- `pyproject.toml`, `uv.lock` - matplotlib into the dev extra.
- `figures/eu_n_scf_binding_curve.png` - **written but deliberately not committed.** The six
  existing Phase 4 and Phase 5 figures in that directory are untracked too, and the plan's own
  artifact table describes this one as "regenerated at review time". A committed PNG is a replay by
  another name, which is the thing threat T-06-28 exists to prevent.

## Verification Evidence

| Gate | Result |
|---|---|
| `uv run python experiments/eu_n_scf_binding_curve.py` | **exit 0**, path printed |
| Printed table row count, first and last | **21 rows**, 1.60 A to 3.60 A |
| Tally line per settled series | both present - "converged at 21 of 21", "converged at 12 of 21" |
| `figures/eu_n_scf_binding_curve.png` mtime vs run start | written 17:31:30Z, run started 17:30:45Z - **regenerated** |
| `grep -c "json\|\.load(" experiments/eu_n_scf_binding_curve.py` | **0** |
| `grep -n "2.124\|3.186\|EU_N_BAND" experiments/eu_n_scf_binding_curve.py` | **no lines** |
| ASCII byte scan on the script | exit 0 |
| `git status --short src/` after the run | **empty** - `VERBOSE_LIBRARY_OUTPUT` lives only in the script's own parameter dict, no library default touched |
| `uv run ruff check experiments/eu_n_scf_binding_curve.py` | All checks passed |
| `uv run pytest` immediately before the checkpoint | **254 passed, 0 failed** in 371.49 s |
| Post-commit deletion check on `487b347` | no files deleted |

## Prohibition audit

All four of the plan's prohibitions hold.

1. **Nothing is replayed.** Every point in the table and on the figure came from the one run at
   17:30:45Z. The grep for file-loading calls returns 0, and the script contains no path to a data
   file other than the geometry files it writes itself into a temporary directory.
2. **No settled value is written into a test, and no test gates the settled curve on the Phase 4
   band.** This plan adds no test at all. The band constants do not appear in the script.
3. **The figure carries no prose explanation box.** It has a title, two panel titles, axis labels
   and one legend. The narrative went into the message to the user, which is where the user's own
   standing instruction says it belongs.
4. **A looks-wrong verdict was not recorded and stepped over.** It was put back to the user, who was
   shown what a revert would cost and made an explicit decision to close with the defect open. The
   verdict is quoted here in full, in their words, at the top of this document rather than buried.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in the deliverable] The legend was sitting on top of the curve being judged**

- **Found during:** Task 1, by rendering the figure and looking at it.
- **Issue:** the energy panel's legend box covered the settled per-atom curve from 2.40 to 2.80 A -
  the shoulder climbing out of the well. That is the exact stretch a human needs to see to answer
  "is this a single clean well or something else", which is this phase's completion bar. The earlier
  code had already pinned it centre-right specifically to keep `"best"` from dropping it top-left
  over the short-separation climb; both corners are occupied. Measured against these curves, neither
  panel has an empty region large enough for a five-entry box.
- **Fix:** one legend for the whole figure, in the charge panel's lower left, which is genuinely
  empty because the settled curves run flat near 0.6 e well above it. The panels share an x-axis and
  use identical colours, so it reads for both, and the energy panel is left completely clear. The
  gave-up band entry and the coordination-mean line entry moved with it; nothing was dropped.
- **Verified:** by re-running the whole thing end to end and looking at the result. Numbers
  unchanged - this touched drawing only.
- **Committed in:** `487b347`

**2. [Rule 4 - escalated, then ruled on] matplotlib was missing**

The first session hit this and, correctly, did not install it - the plan's own threat model
(`T-06-SC`) says to stop and raise a package install rather than perform one. It was raised as a
blocking human checkpoint. The user ruled that matplotlib is a development-time tool for producing
verification figures and belongs in the dev extra, not in the runtime dependencies. Recorded in
`24f67b3` in their own words. This reverses the standing position in
`experiments/diatomic_scans/README.md`, which said the project venv has no matplotlib and plotting
runs in a separate interpreter through JSON; that two-step existed to avoid touching the
environment, and the user has now decided the figures are part of development.

### Deviations already documented in the first session's commit

The vertical-range choice (limits set from the settled points, so a runaway cannot flatten the
wells) and the shading of gave-up separations as vertical bands rather than markers on off-scale
points are both departures from a literal reading of the plan's "distinct marker on the failing
point". Both were kept: a marker on a point that has left the panel is invisible, which would have
defeated decision D-6.09 while appearing to satisfy it. The markers are drawn as well, where the
point is on-panel.

## Known limitations, carried forward deliberately

- **The per-orbital-group charge loop does not converge at long separation.** Recorded as
  `.planning/WINDOWS.md` **entry 6**, kind `unmet-truth`, `src/dftorch/_scf.py`, status **open**.
  It settles at 12 of 21 Eu-N separations, giving up at 2.30 A and at every separation from 2.90 to
  3.60 A, running the charge on nitrogen to between +3.0 and -5.0 electrons and the energy to
  +359 eV. The per-atom loop settles 21 of 21. **This is the defect the user named and chose to
  close the phase with.** It is open in the ledger, so `/gsd-ship` will block on it until someone
  fixes it or waives it with a reason.
- **A hypothesis about the cause, offered as a hypothesis and not a finding.** All nine failures sit
  at long separation, where the two atoms barely interact, and the runaway charges are near-integers
  - +3.0, -2.8, -5.0, and plan 06-03 saw exactly 10.00 electrons into the d group and -7.00 out of
  the f group. That pattern looks like near-degenerate shells trading whole electrons between
  iterations rather than numerical drift: europium's f shell is half full and sits at the frontier,
  the bond splits those levels at short separation, and they collapse back together as the atoms
  part. Per-atom resolution averages over all four shells and never sees it. No electronic-
  temperature or level-broadening knob was found in `Constants.py`, which is what normally damps
  this. **Nobody has tested any of that.** Anyone picking this up should test it before assuming it,
  and should not treat this paragraph as a diagnosis.
- **No new work was opened to fix the oscillation.** The user deferred it to a later phase
  explicitly: "We'll change up what we do in future phases."
- **The navy curve is not approved, merely unobjected-to.** The verdict says nothing about it. If a
  later phase needs the per-atom settled curve blessed, that is a fresh human read, not an
  inheritance from this one.
- **No number from this phase is frozen anywhere.** Decision D-6.08 stands. The Phase 4 one-pass pin
  `EU_N_REFERENCE_E_TOT = -17.510444238744924` is the only pinned Eu-N energy in the suite and it is
  untouched and green.
- **`experiments/diatomic_scans/` is untracked**, so the script's house style is inlined as a
  fallback and the shared `style.py` is used only when present. A fresh checkout can still draw the
  figure.

## Threat coverage

| Threat | Disposition |
|---|---|
| T-06-27 (a gave-up separation reading as a physical feature) | Nine shaded bands on both panels, X markers on every on-panel failing point, converged-of-21 in the legend. The user read it correctly and named the failure unprompted, which is the strongest evidence this mitigation works |
| T-06-28 (the figure replayed from a recorded table) | Computed live in one 45-second run; grep for file-loading calls returns 0; the figure's mtime is later than the run's start; the run timestamp is recorded above; the PNG is deliberately not committed |
| T-06-29 (a looks-wrong verdict logged and stepped over) | The verdict is quoted verbatim at the top of this document, labelled a qualified close rather than an approval, and the user's override of D-6.02 is recorded as a user decision rather than presented as compliance |
| T-06-30 (the checkpoint skipped by a runtime with skip_checkpoints) | It was not skipped. The executor stopped, returned the figure path and the table, and waited |
| T-06-31 (the short lowest point presented as a pass or a failure) | It was presented as an observation with both its context and its caveat, and the user did not raise it. Nothing tests against the band |
| T-06-32 (the tolerance band quietly widened or re-centred) | `EU_N_TARGET_ANGSTROM` and `EU_N_BAND_FRACTION` untouched; neither constant nor its values appear anywhere in this plan's diff |
| T-06-33 (an absolute path in shared output) | Accepted by the plan. The script prints the figure path deliberately, so the human can find it; this is developer-facing local output |
| T-06-SC (package-manager installs) | **Fired, and handled as designed.** The plan asserted matplotlib was already available; it was not. The executor stopped and raised it rather than installing, the user ruled, and the install then happened under that ruling |

## Threat Flags

None. This plan adds one script that writes temporary geometry files into a `TemporaryDirectory`,
reads Slater-Koster data already in the repository, and writes one PNG. It opens no network
endpoint, no authentication path and no new file-access pattern, and changes no schema.

## Known Stubs

None. Nothing in the script returns a placeholder, an empty list or "coming soon" text. The one
thing that could be mistaken for a stub - the `except Exception` that records a separation and keeps
going - reports every caught failure in the printed tally by design, so a bad point cannot cost a
human their review while also not being swallowed. It caught nothing in this run.

## Next Phase Readiness

**Phase 6 closes with known issues, on the user's instruction.** What a following phase should know:

- **The shell-resolved path is the open item**, and the user has said the approach changes rather
  than that this specific bug gets patched. `.planning/WINDOWS.md` entry 6 is where it lives.
- **The default path is safe to build on.** It is gated at `ESDriver.py:195`, defaults to `False`,
  and its numbers are unchanged from before Phase 6.
- **The figure script is reusable.** Point it at another pair of atoms by changing the geometry
  writer and the grid; the three-series structure, the live-recompute guarantee and the gave-up
  shading come along.
- **Damping is the untested lead.** No electronic-temperature or level-broadening control exists in
  `Constants.py`. If the hypothesis above is right, that absence is where the fix would go.

## Self-Check: PASSED

`experiments/eu_n_scf_binding_curve.py`, `figures/eu_n_scf_binding_curve.png`, `pyproject.toml`,
`uv.lock` and this summary all exist on disk. All four commits are present in `git log`: `936ae8e`,
`24f67b3`, `4edbff6`, `487b347`. The full suite was re-run to completion immediately before the
checkpoint and reported 254 passed, 0 failed, and the figure script was re-run end to end after the
last edit and exited 0.

---
*Phase: 06-self-consistent-scf-for-f-systems*
*Plan: 05*
*Completed: 2026-08-07*
*Verdict: qualified close - the settled per-atom curve was not objected to, the per-orbital-group curve was called "very bad", and the user chose to close the phase with that defect open.*

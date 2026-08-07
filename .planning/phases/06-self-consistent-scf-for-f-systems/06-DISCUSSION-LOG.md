# Phase 6: Self-Consistent SCF for f Systems - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 06-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-04
**Phase:** 6-self-consistent-scf-for-f-systems
**Areas discussed:** The 10.6 eV energy gap, Fine-grained electron description, The
did-not-converge flag, What the new test freezes

---

## How this discussion went, recorded because it changed the phase

The gray areas were presented, then **live measurement changed what they were about.** Before
asking anything substantive, `ESDriver.forward(do_scf=True)` was run on the Eu-N diatomic. It
converged, which contradicted the roadmap's premise that self-consistency still had to be
switched on. On the strength of that single point, the discussion was opened with the claim
"the loop already works."

That claim was then **falsified by the full 21-point scan**: the loop runs away at 9 of 21
separations, returning energies up to +198 eV with Eu having absorbed all five of N's valence
electrons. The 2.655 A point sits in a narrow working pocket. The correction was made to the
user before any decision was taken.

A candidate root cause was also raised and then **disproven by test** in the same session:
Eu's per-atom charge penalty is read from its s shell (5.714 eV) though 7 of its 9 valence
electrons are in the f shell (13.606 eV). Substituting the f value changed the failure pattern
but did not restore convergence, and the minimum stayed at 2.00 A, outside the band. Recorded
in CONTEXT.md as a separate defect, explicitly not as the cause.

**Register correction mid-discussion.** After the first gray-area presentation the user
stopped the workflow: "I'm confused — for the rest of the chat remember this. Assume I don't
know your context or plans - I have a small familiarity with the codebase only. I don't
understand complicated jargon. Explain everything rigorously, but intuitively, while avoiding
complicated jargon." The four areas were unchanged; they were re-presented in plain language
with physics terms defined at first use and scale given for every number. Saved to persistent
memory as `explain-intuitively-not-in-jargon`.

---

## Area selection

| Option | Description | Selected |
|--------|-------------|----------|
| The 10.6 eV energy gap | Why the self-consistent energy (-6.92 eV) differs so far from the single-shot energy (-17.51 eV) | ✓ |
| Fine-grained electrons: how far | Whether to rebuild the loop to track electrons per orbital group, write the 7 missing f pieces without a consumer, or defer | ✓ |
| The did-not-converge flag | Which loops get a real flag, warning vs print, whether non-convergence fails the suite | ✓ |
| What the new test freezes | Whether the test only checks the loop closed, or also locks in -6.92 eV and the 0.6-electron transfer | ✓ |

**User's choice:** all four.
**Notes:** The first presentation of these four was rejected on register grounds (see above),
not on content. The same four were selected after re-presentation.

---

## Completion bar for the phase

Asked as: "Given the charge loop runs away at 9 of 21 distances, what does Phase 6 have to
achieve to be called done?"

| Option | Description | Selected |
|--------|-------------|----------|
| Pass the same physics test the shortcut passes | Fix the instability until the self-consistent minimum also lands inside 2.12-3.19 A with all 21 distances converging. Highest bar; open-ended cost | |
| Make all 21 distances converge; report wherever the minimum lands | No distance returns garbage, but the minimum is not gated on the band | |
| Converge at 2.655 A only; refuse honestly everywhere else | Smallest scope, consistent with earlier phases' handling of unfinished capability | partly |
| Diagnose first, then decide the bar | Find the cause, then come back with what a fix would cost before committing | |

**User's choice:** free-text, closest to option 3 but not identical — *"We should converge
around 2.655 A, doesn't have to be exactly there. The final test should be me looking at the
newly generated SCF graph and telling you whether it looks good or not. I definitely also want
to investigate the 10.6 ev gap. We should add a didn't converge flag that is returned by the
SCF loop, similar to how GMRES outputs number of iterations converged, or -1 if not converged
in scipy. If I verify that the SCF loop looks like it's working for f-elements from the EU-N
bond energies and stuff, then that should be good enough for the testing phase - do not freeze
these numbers in yet, we don't know if there's a bug."*

**Notes:** This single answer settled three of the four selected areas at once — the bar
(D-6.01), the gate (D-6.02), the gap investigation (D-6.03), the flag's shape (D-6.04), and
the no-freezing rule (D-6.08). The user's answer went beyond option 3 by adding the human-plot
gate and the scipy-style return value, neither of which was offered in the options. The two
higher bars were declined: Phase 6 does **not** promise the self-consistent curve passes the
Eu-N band check.

---

## Fine-grained (shell-resolved) electron description

| Option | Description | Selected |
|--------|-------------|----------|
| Diagnose first, pull it in only if it's the cause | If the coarse per-atom description is what causes the runaway, building the fine-grained path IS the fix and belongs here. Otherwise it becomes its own phase. Moves the roadmap's promise | ✓ |
| Build it in Phase 6 regardless | Delivers what the roadmap promised, but layers a substantial rebuild on an unsolved bug and changes the machinery underneath the thing being debugged | |
| Keep it out of Phase 6 entirely | Cleanest boundary, but strands the phase if the coarse description IS the cause | |

**User's choice:** Diagnose first, pull it in only if it's the cause.
**Notes:** The user was shown, before choosing, that this moves two of the three requirements
the roadmap assigns to Phase 6 (the seven f Coulomb blocks, and the fine-grained plumbing
being consumed). They were also shown the real cost if the condition fires: measurement 7
found the only shell-resolved charge path in the SCF machinery lives inside the open-shell
routine that Phase 4 D-12 refuses for f, so this is not "write seven missing blocks" but
"build a shell-resolved charge path inside the closed-shell loop, where none exists." The
roadmap was deliberately **not** edited — the tension is recorded in CONTEXT.md
`<open_tensions>` 1 to resolve when diagnosis lands.

---

## Scan range for the graph

| Option | Description | Selected |
|--------|-------------|----------|
| Extend to about 6 A | At large separation the energy must flatten to a constant — a correctness check the current range cannot show, and the region where the runaway is worst. Phase 4 flagged it as worth doing once self-consistency lands | |
| Keep 1.60-3.60 A, the same 21 points | Directly comparable to the Phase 4 curve already approved, point for point | ✓ |
| Both | Same 21 points plus an extended run; compute cost is seconds | |

**User's choice:** Keep 1.60-3.60 A, the same 21 points.
**Notes:** Comparability against the already-approved curve won over the extra physics check.
The extension is preserved in CONTEXT.md `<deferred>` with the reason it matters, for whichever
phase declares the self-consistent path trustworthy.

---

## Derived decisions shown to the user and not corrected

These were not asked as questions. They were presented as explicitly-labelled derivations
before the wrap-up, so the user had the chance to reject them.

| Derived | From | Status |
|---|---|---|
| A non-converged distance away from 2.655 A does not fail the test suite | D-6.01 (only that neighbourhood must converge) + D-6.08 (nothing frozen) | not corrected → D-6.05 |
| The graph shows both curves overlaid with non-converged points marked | You cannot judge "does this look right" without the approved reference curve, or read a +198 eV point without knowing the solver gave up | not corrected → D-6.09 |
| The 10.6 eV investigation produces a written verdict, not a note | "Investigate" needs a deliverable to be checkable | not corrected → D-6.03 |

---

## Wrap-up

| Option | Description | Selected |
|--------|-------------|----------|
| Write the phase document | Write up the decisions, the roadmap tension, the measurements, and the file locations | ✓ |
| Explore more gray areas | Offered: whether all four SCF loops get the new return value or only the one Eu uses (it changes what existing callers unpack), and whether diagnosis is one task or several | |

**User's choice:** Write the phase document.
**Notes:** The two set-aside items were named explicitly rather than dropped silently. Both
moved to CONTEXT.md `### Claude's Discretion`.

---

## Claude's Discretion

- Whether all four SCF loops (`SCFx:158`, `scf_x_os:557`, `SCFx_batch:977`,
  `delta_scf_x_os:1233`) get the returned convergence result, or only the closed-shell
  single-system one the f path uses. Offered to the user and declined as too small.
- The concrete shape of the returned convergence result, subject to the stated semantics
  (iteration count on success, `-1` on failure).
- Whether the diagnosis work is one task or several, and how it stages against the fix.
  Offered to the user and declined as too small.
- How the graph is produced and where it is written.
- Whether the 10.6 eV verdict lives in its own document or inside the phase summary.
- Whether `Constants.py:232`'s s-shell Hubbard U is fixed, guarded, or documented in place.
- Ordering and wave structure.

## Deferred Ideas

- Extending the scan past 3.60 A to ~6 A.
- Requiring the self-consistent Eu-N minimum to land inside the `[2.124, 3.186]` A band.
- Fixing `Constants.py:232` so the per-atom Hubbard U comes from the shell where the electrons
  actually are.
- Spin-polarized / open-shell treatment of Eu 4f7 — noting that spin and shell-resolved work
  are more entangled in this codebase than the requirement list suggests.
- The SEDACS force path consuming `dS` with no f guard.
- Migrating library output to a logging module; flipping `VERBOSE_LIBRARY_OUTPUT` to quiet.

No scope creep was raised during this discussion — every topic stayed inside the
self-consistency boundary.

# Decisions Needed — Phases 6-9

**Written:** 2026-07-29
**Why this file exists:** Phase 5 is fully planned and ready to execute. Phases 6-9 cannot be
planned without design calls that are genuinely yours. Everything answerable from the code or
from prior locked decisions has already been decided and recorded below as **PRE-DECIDED** —
those need no input. Only the **OPEN** items need you.

**How to use this:** answer the OPEN items inline (edit this file, or just reply with the
numbers and your picks). Each has a recommendation. Answering them unblocks
`/gsd-discuss-phase 6` … `9` — or lets me skip discuss entirely and go straight to
`/gsd-plan-phase`, since the answers *are* what discuss-phase would gather.

**Nothing here blocks Phase 5.** Run `/gsd-execute-phase 5` whenever you want.

---

## Verified technical facts (no input needed — recorded so research isn't redone)

| Fact | Evidence |
|---|---|
| MD **requires** the batch path | `MDXL.__init__` and `MDXLBatch.__init__` both take `ESDriverBatch` (`MD.py:38`, `MD.py:1216`) |
| Batched f H0/S currently refuses | `FAngularFormulaSourceError`, raised in `_h0ands.py` batch path — masks cover only `n_orb ∈ {1,4,9}` |
| f derivatives are switched off at the source | `F_ANGULAR_DERIVATIVES_AVAILABLE = False` (`_slater_koster_pair.py:256`) |
| Derivative guards fire in 3 places | `ESDriver.py:864` (`calc_forces`), `ESDriver.py:1840` (`ESDriverBatch.calc_forces`), `_stress.py:194` |
| Force machinery for spd already exists | `_forces.py`: `Forces`, `forces_spin`, `Forces_PME`, `forces_shadow`, `forces_shadow_pme`; plus `_forces_batch.py` |
| SCF entry points | `_scf.py`: `SCFx:154`, `scf_x_os:543`, `SCFx_batch:954`, `delta_scf_x_os:1200`; `_AndersonMixer:32` |
| SCF non-convergence today just prints | `print("Did not converge")` at `_scf.py:507`, `:889`, `:1184`, then continues. `MaxIt=50` |
| Shell-resolved Coulomb f blocks are blocked on SCF | `(n_shells, n_shells)` matrix vs `(Nats, Nats)` consumed by `energy()`/`SCFx` — nothing can validate the values until shell-resolved charges flow through SCF |
| The CH4 SCF **diverges** | residual grows 0.155 → 0.389 → 0.466 over 25 iterations. Pre-existing, unasserted, the one open `.planning/WINDOWS.md` entry — **blocks `/gsd-ship`** |

---

## PHASE 6 — Self-Consistent SCF for f Systems

### PRE-DECIDED (from prior locked decisions — no input needed)
- **Non-convergence warns and returns the last iterate with a flag, never raises.** Phase 4
  D-13 already settled this. Today the code prints and continues; Phase 6 makes it a real
  flag on the structure.
- **Phase 4's single-shot path stays available and unbroken.** It is pinned by
  `tests/test_single_shot_energy.py` with `e_tot = -17.510444238744924`.
- **`MAGNETIC_HUBBARD_LDEP` is the shell-resolved gate.** Phase 4 D-23. No `SHELL_RESOLVED` flag.

### OPEN

**6.1 — The CH4 SCF divergence: fix in Phase 6, or before it?**
The residual *grows*, which is divergence, not slow convergence. It's f-free, so it's a
pre-existing bug in the shared SCF machinery your f work will now depend on.
- **(a) Fix it first, as Phase 6's opening task** *(recommended)* — you're about to build f SCF
  on top of this machinery; starting from a known-diverging baseline means you can't tell your
  bug from its bug. Also clears the `/gsd-ship` blocker.
- **(b) Fix it as a separate small phase before 6** — cleanest separation, one more boundary.
- **(c) Waive it and proceed** — record a reason in the ledger; f SCF may not touch the same path.

**Answer:** We should fix it, but first I need to see the error and figure out why the error is happening. 

**6.2 — What is the convergence acceptance criterion for f systems?**
Phase 4 validated energy at a loose ±20% band. SCF needs a convergence claim.
- **(a) Charge residual below a fixed tolerance within MaxIt, on the Eu-N diatomic** *(recommended)*
  — smallest thing that proves the loop closes. Concrete and cheap.
- **(b) The above, plus the converged energy is within a band of the single-shot value** — adds
  a physics sanity check, but the two need not be close, so the band would be arbitrary.
- **(c) Something else** — e.g. a larger test system, or a dq-vs-iteration curve you eyeball.

**Answer:** What does the current system use to test convergence acceptance criterion?

**6.3 — Does Phase 6 also need spin, or does closed-shell suffice?**
Eu 4f⁷ is genuinely open-shell. Phase 4 D-12 defers spin to v2 and makes it raise. A converged
closed-shell SCF on an open-shell ion is a real physics compromise.
- **(a) Closed-shell only; spin stays deferred** *(recommended)* — consistent with D-12, keeps
  Phase 6 scoped. Record explicitly that the converged number is not physically complete.
- **(b) Pull spin forward into Phase 6** — physically right, substantially bigger phase.

**Answer:** Lets do A, we will have to get open shell working at some point soon

---

## PHASE 7 — f Angular Derivatives

### PRE-DECIDED
- Derivatives must be **checked against finite differences of the Phase 3 angular values**, not
  merely "exist" — a sign or AO-ordering error would otherwise pass silently. Already written
  into ROADMAP Phase 7 success criterion 3.
- The derivative source basis gets **recorded the way Phase 3 source-locked the angular
  values** (`03-SOURCE-LOCK.md` pattern).

### OPEN

**7.1 — Where do the f angular derivative formulas come from?**
This is the single biggest unknown in 6-9. Phase 3 source-locked the *values* to Takegahara
1980. There's a note in my memory that Sharma's Table I covers only ~20 of ~112 f entries, and
that the cubic f set in `Structure.py` can't be reached from tesseral SK tables by permutation
and sign alone — so the derivative source may not be a simple lookup.
- **(a) Analytic differentiation of the Phase 3 source-locked expressions** *(recommended)* —
  you already own those formulas; differentiating them is self-consistent and needs no new
  external source. Verify against finite differences.
- **(b) Find and source-lock a published derivative table** — authoritative if one exists
  covering all f blocks, but Phase 3's experience suggests published f tables are partial.
- **(c) Autodiff the existing angular code** — cheapest to write; risk is it's slow in the hot
  path and gives you no independent check (differentiating the same code you're testing).

**Answer:** Let's do A and C. Also, the table should have all the angular f derivative formulas - if not, specify me the order in which you want them, and I can try to get them put in for you in a format you would preffer - let me know what that is. 

**7.2 — Does Phase 7 ship derivatives only, or derivatives + the guard removal?**
- **(a) Derivatives + narrow the guards to what genuinely remains unsupported** *(recommended)*
  — leaving a blanket refusal in place while correct derivatives exist is confusing.
- **(b) Derivatives only; guards come down in Phase 8 with forces** — keeps 7 tightly scoped
  and means nothing claims to work until forces validate it.

**Answer:** A

---

## PHASE 8 — f Forces and Stress

### PRE-DECIDED
- Forces validated against **finite differences of the total energy**; stress against finite
  differences or a documented reference.
- Geometry optimization of an f system should run — the capability Phase 4 explicitly could
  not use (D-19 forced an energy scan instead).

### OPEN

**8.1 — Which force path(s) must work for f?**
`_forces.py` has five entry points: `Forces`, `forces_spin`, `Forces_PME`, `forces_shadow`,
`forces_shadow_pme`.
- **(a) `Forces` only; the rest keep explicit refusals** *(recommended)* — smallest honest
  scope. `forces_spin` is deferred anyway by D-12; PME and shadow paths can wait for a real need.
- **(b) `Forces` + `Forces_PME`** — needed if you want periodic f systems, which are currently
  v2 (PHY-08).
- **(c) All five** — largest, and several have no f test case to validate against.

**Answer:** A

**8.2 — Does the mixed-grid radial fix belong here after all?**
Phase 5 now owns both the guard and the per-pair lookup (your call). If Phase 5's execution
shows the interface change is riskier than expected, Phase 8 is the natural place to finish it,
since it's already reworking the stress path that consumes the same interface.
- **(a) Leave it in Phase 5 as decided** *(recommended unless Phase 5 execution says otherwise)*
- **(b) Pre-authorize moving the lookup rewrite to Phase 8 if Phase 5 hits trouble** — lets me
  descope without stopping to ask.

**Answer:** It should have been fixed in phase 5, if it hasn't, definitely fix it here. 

---

## PHASE 9 — f Molecular Dynamics

### PRE-DECIDED
- **Batched f H0/S is on the critical path.** `MDXL` requires `ESDriverBatch`. This was
  previously written as "conditional"; corrected 2026-07-29 against `MD.py:38`.
- Energy conservation over a trajectory, within a documented tolerance for the chosen
  integrator and timestep, is the acceptance criterion.

### OPEN

**9.1 — Batched f H0/S: its own phase, or inside Phase 9?**
This is not small — the batch path masks only `n_orb ∈ {1,4,9}` and would silently drop
16-orbital atoms from every off-diagonal block, which is why it refuses today.
- **(a) Split it out as Phase 9a, before MD** *(recommended)* — it's a substantial capability
  with its own validation story (batched H0/S must match single-system H0/S per structure),
  and bundling it makes Phase 9 two phases wearing one hat.
- **(b) Keep it inside Phase 9** — fewer boundaries, but Phase 9 becomes large and its
  acceptance criterion (energy conservation) can't be reached until the batch work lands.

**Answer:** A

**9.2 — What MD system, and how long a trajectory?**
Eu-N diatomic is the only validated f case. A diatomic MD run is a weak test (two atoms, one
vibrational mode) but cheap; anything larger has no reference.
- **(a) Eu-N diatomic, short NVE run, check energy conservation** *(recommended)* — consistent
  with every prior phase's "smallest honest test" pattern.
- **(b) A small Eu-containing cluster** — more representative, but nothing validates it.
- **(c) You have a specific system in mind** — say which.

**Answer:** A

---

## Cross-cutting

**X.1 — Ordering.** Current roadmap is 5 → 6 → 7 → 8 → 9 (with 9a inserted if you pick 9.1a).
Phase 7 (derivatives) has no user-visible output on its own; 8 is where forces become usable.
If you'd rather see something working sooner, 7 and 8 could merge. Recommendation: **keep them
split** — the finite-difference check on derivatives is exactly the gate that catches sign
errors, and merging tempts skipping it.

**Answer:** Yeah let's test it after 8

**X.2 — How autonomous should 6-9 run?** Phase 4 ran fully autonomous except its human
checkpoint. Phase 5 is planned the same way (one `checkpoint:decision` for REG-02, one
`checkpoint:human-verify` at the end).
- **(a) Same pattern — autonomous with human checkpoints at physics judgments** *(recommended)*
- **(b) More checkpoints** — e.g. review each phase's plans before execution
- **(c) Fewer** — run straight through

**Answer:** I'll let you know when we get to each phase, just ask

---

## What happens once you answer

1. I fold your answers into `0N-CONTEXT.md` for each phase (that is exactly what
   `/gsd-discuss-phase` would produce, so it can be skipped).
2. `/gsd-plan-phase 6` … `9` run without further input.
3. Phase 5 can execute at any time, independently — it is not waiting on any of this.

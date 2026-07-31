# Phase 4: SCF and Reference Simulation Validation - Discussion Log

**Date:** 2026-07-29
**Mode:** default (interactive), questions bundled per area

> Human reference only. Downstream agents read `04-CONTEXT.md`, not this file.

## Gray Areas Presented

1. Reference case definition — *not initially selected; user supplied it mid-discussion*
2. SCF support boundary — **selected**
3. f-shell electrostatics model — **selected**
4. The PME float64 blocker — **selected**

---

## Area: SCF Support Boundary

**Q1. How far must the SCF path genuinely go for Phase 4 to count as complete?**
- Converged SCF + observables — full self-consistency reproducing reference observables
- Converged SCF, derivatives error — SCF converges, forces/stress still raise
- Single-shot energy, no self-consistency — diagonalize once, no charge self-consistency

→ **Selected: Single-shot energy, no self-consistency** (D-11)

**Q2. Eu 4f7 is open-shell. How should spin polarization be handled?**
- Explicitly deferred, error out
- Closed-shell, documented caveat
- Support spin polarization

→ **Selected: Explicitly deferred, error out** (D-12)

**Q3. When SCF fails to converge on an f system, what should happen?**
- Raise a named error
- Warn and return last iterate
- You decide

→ **Selected: Warn and return last iterate** (D-13)

*Note raised:* with no self-consistency (D-11), non-convergence cannot arise in Phase 4's
own path. Recorded as governing the existing SCF machinery whenever exercised, rather than
discarded as dead guidance.

*Tension raised:* single-shot makes SIM-05 ("reproduce reference observables") hard to
satisfy as written. Recorded in `<open_tensions>` rather than silently narrowing the
roadmap.

---

## Area: f-Shell Electrostatics Model

Scouting surfaced beforehand: `Hubbard_U_sr` at `Structure.py:435` built from
`shell_present` (already 4-shell after Phase 2), plus a `SHELL_RESOLVED` / l-dependent-U
option in `Constants.py`.

**Q1. Given single-shot, how far should f-shell electrostatics actually be built out?**
- Plumb and validate f dims
- Plumb shapes only
- Minimum not to crash

→ **Selected: Plumb and validate f dims** (D-14)

**Q2. Where should f-shell Hubbard U values come from?**
- Extended SKF files
- SKF with explicit error if absent
- You decide

→ **Selected: Extended SKF files** (D-15)

**Q3. Should the f shell get its own l-dependent U, or share a per-atom value?**
- Shell-resolved (l-dependent)
- Single per-atom U
- Support both, config-gated

→ **Selected: Support both, config-gated** (D-16)

---

## Area: PME float64 Blocker

Root cause diagnosed during discussion and shown to the user: `PME_torch.py:261`
`g_mask = (m_2 > 0).float()` forces float32 while `E_G` / `metric` are float64; the einsum
on line 264 then raises. Verified to be the only `.float()` in `src/dftorch/ewald_pme/`.

Also surfaced: the failing test is f-free, making this nominally REG-01 / Phase 5 territory
rather than f-orbital work.

**Q1. Where should the PME fix land?**
- Fix in Phase 4
- Defer to Phase 5
- Fix now, outside the phase

→ **Selected: Fix in Phase 4** (D-17)

**Q2. How much should the fix cover beyond the one line?**
- One line plus a regression test
- One line only
- Broader dtype audit

→ **Selected: Broader dtype audit** (D-18)

---

## Area: Reference Case (user-initiated, mid-discussion)

User message: *"for the reference case, right now we don't have anything like that, we're
just going to test maybe against a Europium Nitrogen bond and see if we can predict the
length correctly, more of a common sense test"*

Claude flagged in response: a bond-length check is reachable **without forces** if done as
an energy scan over separation; via geometry optimization it would hit the
`FDerivativeUnsupportedError` wall from Phase 3 D-09. The scan approach is the one
compatible with D-11.

**Q1. What system should the Eu-N bond-length check use?**
- Isolated Eu-N diatomic
- EuN rock-salt crystal
- You decide

→ **Selected: Isolated Eu-N diatomic** (D-19)

**Q2. What counts as "predicting the length correctly"?**
- Loose sanity band (~10-20%)
- Order-of-magnitude only
- Tight quantitative

→ **Selected: Loose sanity band** (D-20)

---

## Deferred Ideas Captured

- Self-consistent SCF for f systems
- Spin-polarized / collinear-spin support for open-shell 4f
- f derivatives (forces, stress, MD) — carried from Phase 3 D-09
- Ga-containing and periodic EuN crystal cases

## Claude's Discretion

- Scan range, step count, uniform vs adaptive
- Dtype audit scoping and staging
- Whether D-13's warning reuses an existing channel

## Open Tensions Recorded (need user's call at planning)

1. SIM-04 wording assumes a reference paper that does not exist
2. SIM-05 cannot be met as written under D-11 + D-19
3. `tests/f_orbital_data/` Ga files go unused this phase

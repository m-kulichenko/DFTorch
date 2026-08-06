# Single-Shot vs Self-Consistent: Why Two Energies for the Same Molecule Differ by 10.6 eV

**Written:** plan 06-04. **All numbers below re-measured 2026-08-06** against the current
tree, on branch `f_orbital_initial`, at commit `cef305d`.
**Machine half:** `tests/test_energy_definitions_f.py` (four tests) and
`tools/check_verdict_doc.py` (a completeness and line-number gate on this document).

Pure ASCII throughout: "A" means Angstrom, "->" means an arrow. This repository has been
bitten before by replacement characters on a Windows console.

---

## The verdict

**The 10.6 eV difference is a difference of definition, not a defect.** The two paths are not two attempts at the
same number; they compute two different quantities, and one of them - the single-pass path -
deliberately omits an entire energy term by construction. Comparing them is like comparing a
car's kerb weight with its weight carrying passengers: both are correct, neither is wrong, and
subtracting one from the other measures the passengers, not an error. Confidence: **HIGH**.
It rests on three things that are checkable rather than argued - the exact `None` argument in
the source that switches the missing term off (cited below, line numbers confirmed against the
current tree), the measured three-way split of where the difference sits, and the measured
charge transfer on each path, which differ by more than a factor of four.

No future work should try to make the two numbers agree. If they ever do agree, something has
broken.

---

## Vocabulary, before any of it is used

This document uses five terms of art. Each is defined here in plain words and then used
consistently.

| Term | Plain words |
|---|---|
| **band-structure energy** (`structure.e_band0`) | The energy the electrons gain by spreading out over the molecule instead of sitting on their own atoms. It is the reward for bonding. Big and negative when bonding is strong. |
| **electron-repulsion energy** (`structure.e_coul`) | The electrostatic price paid for having moved charge off one atom and onto another. Two lumps of like charge repel; if a calculation moves electrons, it must pay for the resulting separation of positive and negative. Always zero or positive. In DFTB this is often called the "charge-fluctuation" or "SCC" term. |
| **nuclear repulsion** (`structure.e_repulsion`) | A short-range pairwise term that keeps atoms from collapsing into each other. It is a fixed function of the geometry and knows nothing about where the electrons went. **It is a different thing from electron repulsion and does not change between the two paths.** |
| **electron count per atom** (`structure.q`) | A Mulliken charge: population minus nuclear charge. **Negative means the atom lost electrons.** In Eu-N, europium comes out negative on both paths - it is the metal, and it donates. |
| **settling / convergence** | The self-consistent path guesses where the electrons are, solves, sees where they moved to, and repeats. "Settled" means the answer stopped moving. `structure.scf_iter_count` is the number of passes it took, or the literal `-1` if it gave up. |

---

## The two paths

Both start from the same molecule, the same geometry, and the same parameter files. The only
difference is the `do_scf` argument to the driver.

**One pass (`do_scf=False`).** Put the electrons in their free-atom arrangement, solve the
system once, read off the energy. Fast, no loop, no possibility of failing to converge. This is
what Phase 4 validated and what the pinned reference energy belongs to.

**Settled (`do_scf=True`).** Solve, look at where the electrons ended up, rebuild the problem
with the electrons in their new places, solve again, and keep going until the answer stops
changing. At Eu-N 2.655 A this takes **28 passes** against a cap of 100.

---

## What each path actually computes, and the exact line that makes them different

The one-pass path reports **band-structure energy plus nuclear repulsion plus an entropy term,
and an electron-repulsion energy of exactly zero.** The zero is not a rounding artefact and not
an oversight. It is written into the call:

```
src/dftorch/ESDriver.py:1018-1041   the single-shot call to energy()
src/dftorch/ESDriver.py:1030-1031   the two arguments that do it:
                                        None,  # C: no Coulomb matrix (D-11)
                                        None,  # dq_p1: no PME charge response (D-11)
```

Handing `None` for both selects the first arm of a four-way branch inside `energy()`:

```
src/dftorch/_energy.py:181-182      elif C is None and dq_p1 is None:
                                        Ecoul = 0
```

That is the whole mechanism. One `None`, one branch, one term set to zero.

The settled path passes a real Coulomb matrix instead, so the same branch selects an arm that
computes a real electron-repulsion energy - and, just as importantly, the repeat-until-settled
loop *minimises against* that term while it runs, so the electrons end up somewhere different.

**Why the zero is deliberate.** This is Phase 4 decision **D-11**, and the reasoning is stated
in the source at `ESDriver.py:958-973`. The charges available on the one-pass path are
first-pass Mulliken charges with no self-consistency behind them. Feeding those into an
electrostatic energy would not produce a one-pass energy; it would produce a single broken
self-consistent step, and it destroys the binding curve.

**A note on the line numbers.** Plan 06-04 was written against `ESDriver.py:783-792` and
`_energy.py:131-132`. Those numbers are now stale - plans 06-01 through 06-03 added code above
both sites. `ESDriver.py:783-792` today lands in the open-shell branch, and `_energy.py:131-132`
lands inside the shell-resolved argument check. The citations above are the confirmed current
ones, and `tools/check_verdict_doc.py` re-confirms them on every run, so this document cannot
quietly drift out of agreement with the code it describes.

---

## Where the difference actually sits

This is the section that matters most, because the obvious explanation - "the settled path adds
an electron-repulsion term, and that term is the difference" - is **wrong**, and it is wrong by
a large margin. The electron-repulsion term accounts for only about a sixth of the gap.

Measured at Eu-N 2.655 A, 2026-08-06, default parameters (`COUL_METHOD="FULL"`,
`T_ELECTRONIC=1000.0`, `RCUT_ELECTRONIC=10.0`, `RCUT_REPULSIVE=6.0`, `CHARGE=0`, float64, CPU):

| Term | One pass (eV) | Settled (eV) | Change (eV) | Share of the gap |
|---|---|---|---|---|
| **Total** | **-17.510444** | **-6.920251** | **+10.590193** | 100 % |
| band-structure energy | -16.948082 | -8.030033 | +8.918049 | **84.2 %** |
| electron-repulsion energy | 0 (exactly) | +1.698980 | +1.698980 | 16.0 % |
| entropy term | -0.721272 | -0.748107 | -0.026835 | -0.3 % |
| nuclear repulsion | +0.158909 | +0.158909 | 0 | 0 % |

Stated plainly: **84 percent of the 10.6 eV difference is in the band-structure term, and only
16 percent is in the newly present electron-repulsion term.** The shares add to slightly over
100 percent because the entropy term moves the other way by a quarter of a tenth of an eV.

An explanation that says only "the settled path includes electrostatics" is therefore
incomplete and is rejected. Electrostatics accounts for 1.70 eV of a 10.59 eV difference. The
other 8.92 eV is the band term, and the reason the band term moved is the subject of the next
section.

For scale: a typical chemical bond is worth 2 to 5 eV. The 8.92 eV shift in the band term is
larger than a chemical bond on its own.

---

## The mechanism, in plain words

The band-structure term did not move because anyone changed how it is computed. It moved
because **the electrons are somewhere else.**

| | One pass | Settled |
|---|---|---|
| electrons moved off europium | **2.702732** | **0.599583** |
| ratio | 4.5 times more on the one-pass path | |

Without an electron-repulsion penalty, the one-pass calculation has no reason not to pour
charge onto nitrogen. It moves about 2.70 electrons across and banks the full band-structure
reward for doing so, while paying nothing at all for having created a large separation of
positive and negative charge. That is a very good deal, and it produces a very low - and
misleadingly low - total energy.

Switch the penalty on, and the deal stops being good after about six tenths of an electron. The
transfer collapses to 0.60 electrons, which gives back most of the band-structure reward. That
returned reward is the +8.92 eV.

So the difference is not a correction applied to the same answer. **It is a different
arrangement of the electrons**, with a different band energy to match, and the +1.70 eV of
explicit electrostatics is only the visible part of the change.

---

## Why the transfer runs away when the penalty is off

The one-pass runaway is not random. It is forced by the parameter set, and it is worth showing
because it makes clear that this is textbook method behaviour rather than a bug in this code.

Each orbital in a DFTB calculation has an on-site energy - roughly, how tightly that atom holds
an electron in that orbital. Electrons fill the lowest levels first, across the whole molecule,
against one shared filling level. Read straight from the parameter files
(`tests/f_orbital_data/`, loaded into `Constants.Es/Ep/Ed/Ef`):

| Atom | Orbital group | On-site energy (eV) |
|---|---|---|
| Nitrogen | 2s | -21.5623 |
| Nitrogen | 2p | -6.8355 |
| Europium | 4f | -1.52112 |
| Europium | 5d | -0.67212 |
| Europium | 6s | -2.63406 |
| Europium | 6p | +0.53606 |

**Every nitrogen level lies below every europium level.** The highest nitrogen level, 2p at
-6.8355 eV, is still 4.20 eV below the lowest europium level, 6s at -2.63406 eV. Measured
against europium's shallowest *occupied* level, the 4f at -1.52112 eV, the mismatch is 5.31 eV.

With one shared filling level and no charge penalty, nitrogen therefore fills completely, at any
separation. At 10 A - where the atoms are far enough apart to be independent - the one-pass path
puts exactly -3.0000 electrons on europium: nitrogen holds a filled octet and europium holds
what is left.

This is the classic charge-transfer failure of non-self-consistent tight binding, and it is
**exactly the reason the self-consistent method was invented.** It is not specific to f
orbitals and not specific to this implementation: the same runaway was measured on the one-pass
path for CO, PN, SO, CS and CN- in the unrelated mio-1-1 parameter set (`06-RESEARCH.md`,
section "Cross-check against the mio-1-1 set"), reaching -2.0 to -3.0 electrons in every case.

**And it is not a parsing bug.** Every field of the Eu and N parameter files was cross-checked
against what the code loaded - on-site energies, Hubbard strengths, shell occupations, and the
descending-angular-momentum ordering - and all of them matched (`06-RESEARCH.md`, section
"Parameter audit - parsing verified correct"). This is not a repeat of the Phase 5 phantom-shell
parsing bug.

---

## What this means for the Phase 4 pinned reference

`tests/test_single_shot_energy.py` pins two numbers:

```
EU_N_REFERENCE_E_TOT   = -17.510444238744924
EU_N_REFERENCE_E_BAND0 = -16.948081509141794
```

**Both remain valid, and both belong to the one-pass path only.** They were reproduced exactly
in the measurement above.

A reader who compares `EU_N_REFERENCE_E_TOT` against a settled energy is comparing two
different observables. That is the single most likely way for someone to draw a wrong
conclusion from this codebase, and it is the reason this document exists. The pin is a
regression guard on the *definition* of the one-pass energy: if it ever moves, the one-pass
definition drifted, and Phase 4's binding-curve work drifted with it.

`tests/test_energy_definitions_f.py::test_the_phase_four_pin_is_not_the_settled_answer` exists
to make the confusion loud. It requires the settled total to differ from the pin by more than
1 eV. The observed difference is 10.59 eV, so the test has an order of magnitude of headroom;
it is not a value pin, it is a tripwire that fires if some future change makes the two paths
agree - because agreement would mean D-11 had been silently undone.

---

## A third number, added 2026-08-06 by plan 06-03

Plan 06-03 introduced a finer description of charge: instead of one number per atom, the
settled loop can now track one number per orbital group, so europium's f electrons are charged
at the f group's own strength (13.61 eV) rather than at its s group's (5.71 eV). It is opt-in,
selected by the `MAGNETIC_HUBBARD_LDEP` parameter key, and **off by default**.

That means there are now two settled energies, not one, and this document would be misleading if
it reported only the default. Measured at the same geometry on the same day:

| Path | Total (eV) | Band (eV) | Electron repulsion (eV) | Electrons off Eu | Passes |
|---|---|---|---|---|---|
| one pass | -17.510444 | -16.948082 | 0 (exactly) | 2.702732 | n/a |
| settled, per atom (default) | -6.920251 | -8.030033 | +1.698980 | 0.599583 | 28 |
| settled, per orbital group | -5.639887 | -6.096938 | +1.201020 | 0.498650 | 80 |

Against the finer description the gap is **11.87 eV**, not 10.59 eV, and the band term's share
rises from 84 percent to 91 percent. The verdict is unchanged and the mechanism is unchanged -
if anything it is sharper, because charging europium's f electrons at their real, larger
strength makes the electrostatic penalty bite sooner, cuts the transfer further, and hands back
even more of the band reward.

**Do not read the 11.87 eV number as a correction to the 10.59 eV number.** They are two
settled energies from two different charge descriptions, and neither has been validated against
an external reference. Decision D-6.08 froze no settled value this phase.

---

## One further finding, recorded and accepted

Neither path reaches genuinely neutral atoms when the two atoms are pulled far apart. This was
investigated, and the two failures have different causes and different standing.

**The one-pass path pins at exactly -3.0000 electrons on europium**, at 6 A and at 40 A alike.
That is the non-self-consistent filling behaviour described above, working as designed. It is
expected, not a defect.

**The settled path converges to about -0.29 electrons.** This is DFTB2's known
fractional-charge-at-dissociation error. It was checked against a closed-form model rather than
accepted on faith: at large separation the charge-transfer energy is
`E(d) = -gap * d + 0.5 * (U_Eu + U_N) * d^2`, minimised at `d = gap / (U_Eu + U_N)`. With
`gap = 5.3144 eV` and `U_N = 13.3336 eV`, the model predicts -0.2790 electrons against europium's
5.71 eV strength and -0.1973 against its 13.61 eV one. Re-measured at 40 A on 2026-08-06, the
code gives **-0.2887** and **-0.2021** respectively - agreement to 3.5 percent and 2.4 percent,
with the residual attributable to thermal smearing at `T_ELECTRONIC = 1000 K`.

The mechanism is therefore identified, and the important consequence is that **it does not go
to zero for any choice of strength**. A better strength shrinks the residual from 0.29 to 0.20
electrons and no further.

**A human ruling of 2026-08-04 accepted that residual as within the method's limits and opened
no investigation.** This is recorded here as a closed decision, not as an outstanding item.
Reaching exact neutrality at dissociation would require the europium on-site energies to line
up with nitrogen's, or a fragment-based or DFTB+U treatment - a parameter-set and method
question, not an implementation defect.

---

## What this document does not claim

- **It does not validate the settled numbers.** Decision D-6.08 froze no reference values this
  phase, deliberately. -6.920251 eV and -5.639887 eV are quoted here as evidence for a claim
  about *definitions*, not as approved answers. The settled binding curve's own gate is a human
  looking at a graph, and that is plan 06-05's job, not this document's.
- **It does not claim the settled minimum is in the right place.** The self-consistent curve's
  minimum sits at 2.00 A, short of the Phase 4 band of [2.124, 3.186] A. That is escalated to
  human judgement by the rule in `tests/f_orbital_data/README-EU-N-CASE.md` and is explicitly
  not evidence that the f implementation is broken.
- **It does not claim the one-pass path is wrong.** It is a well-defined, correctly implemented
  quantity that Phase 4 validated. It is simply not the same quantity as the settled energy.
- **It does not settle the Krylov accelerator.** The convergence accelerator diverges on this
  system and is switched off for f systems in the interim by a human ruling of 2026-08-04;
  repairing it belongs to a later phase. That is a separate matter from anything above - the
  measurements here were all taken with the accelerator off, and the plain mixer reaches the
  same fixed point.

---

## How this document is kept from going stale

A document that describes code is trusted, so a document that drifts out of agreement with the
code is worse than no document at all. Three things guard against that:

1. **`tools/check_verdict_doc.py`** - a standard-library-only script that reads this file and
   fails if any required anchor is missing, if the ASCII rule is broken, if an assertion or test
   code has crept in, or - the important one - **if a cited source line no longer contains what
   this document says it contains.** It is a completeness and citation gate, not a physics
   check.
2. **`tests/test_energy_definitions_f.py`** - four tests holding the load-bearing claims: that
   the one-pass path carries no electron-repulsion term at all and the settled path does, that
   the two paths move very different amounts of charge, that every nitrogen level lies below
   every europium level, and that the Phase 4 pin is not the settled answer. Every assertion is
   an exact zero, a sign, an inequality or a ratio. **No energy or charge measured above appears
   in any assertion**; those numbers live here, as prose evidence, and nowhere else.
3. **`src/dftorch/Constants.py:291`** - the assignment
   `self.U = torch.nn.Parameter(US, ...)`, which charges every element its s group's
   electron-repulsion strength - carries a recorded note on that defect immediately above it,
   so an editor of that line meets the record without having to find this file. Decision D-6.10
   refers to it as "Constants.py:232", which is where it sat before plan 06-04 inserted the note
   above it; search for the assignment rather than the number.

---

## Related records

- `06-RESEARCH.md`, section "Blocking Diagnosis (2026-08-04)" - the original investigation, its
  objections and their measurements.
- Phase 4 decision D-11 - the decision that created the one-pass definition.
- Phase 6 decision D-6.03 - the decision requiring this written verdict, including its warning
  that an explanation accounting only for electron repulsion is incomplete.
- Phase 6 decision D-6.08 - no reference numbers frozen this phase.
- Phase 6 decision D-6.10 - the per-atom Hubbard strength defect. D-6.10 names it
  `Constants.py:232`; the recorded note now sits above it and the assignment is at
  `Constants.py:291`.

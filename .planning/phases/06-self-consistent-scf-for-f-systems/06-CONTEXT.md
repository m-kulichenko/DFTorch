# Phase 6: Self-Consistent SCF for f Systems - Context

**Gathered:** 2026-08-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the self-consistent charge loop **actually work** for the Eu-N f system, and give the
loop an honest way to say it failed.

**This is a debugging phase, not a wiring phase.** The roadmap was written on the assumption
that self-consistency still had to be switched on for f systems. It is already switched on —
nothing refuses `ESDriver.forward(do_scf=True)` for a closed-shell f system on the per-atom
Coulomb path. The problem is that the loop **runs away** at most separations. See
`<measurements>` below for the numbers; they are the reason this phase's shape changed.

In scope:
- Diagnose and fix the charge-loop instability far enough that the loop converges in the
  neighbourhood of 2.655 A.
- Investigate the 10.6 eV difference between the single-shot energy and the self-consistent
  energy, and produce a written verdict on whether it is real physics or a defect.
- Give the SCF loop a returned convergence result (iteration count, or -1) instead of a
  printed line.
- Recompute the 21-point Eu-N binding curve self-consistently and put it in front of a human
  for judgement.

Out of scope unless diagnosis pulls it in:
- The shell-resolved (per-orbital-group) charge path and its seven missing f Coulomb blocks —
  conditional, see D-6.06 and `<open_tensions>`.
- Spin polarization / open-shell treatment of Eu 4f7 — stays deferred (Phase 4 D-12).
- f derivatives (Phase 7), forces and stress (Phase 8), batched H0/S (Phase 8.1), MD (Phase 9).
- Extending the scan range past 3.60 A (D-6.07 keeps the Phase 4 grid).

</domain>

<measurements>
## What was measured during this discussion (2026-08-04, branch `f_orbital_initial`)

Every number below was produced by running the installed package, not read off a document.
All of it is reproducible with `Constants` / `Structure` / `ESDriver` on the Eu-N geometry
that `tests/test_single_shot_energy.py::_write_eu_n_xyz` writes, with
`SKFPATH = tests/f_orbital_data/`, `COUL_METHOD = "FULL"`, `T_ELECTRONIC = 1000.0`,
`CHARGE = 0`, under `torch.set_default_dtype(torch.float64)`.

### 1. Nothing refuses the self-consistent path for f

`ESDriver.forward(do_scf=True)` on Eu-N at 2.655 A runs to completion. 17 iterations,
finishing `Res = 0.000000000, dEc = 0.000000000`, exiting through the `zero norm_dr` branch.
`FShellResolvedCoulombUnsupportedError` fires **only** when `MAGNETIC_HUBBARD_LDEP` is set;
with it unset (the default) the per-atom Coulomb path runs.

So ROADMAP Phase 6 success criterion 1 — "runs a real self-consistent charge loop to
convergence, not the Phase 4 single-shot path" — is **already true at one separation**. It is
not true across the range, which is the actual problem.

### 2. At 2.655 A the energy is 10.6 eV away from the single-shot value

| quantity | single-shot (`do_scf=False`) | self-consistent (`do_scf=True`) |
|---|---|---|
| `e_tot` | -17.510444238744924 eV (pinned) | -6.920250682037466 eV |
| `e_band0` | -16.948081509141794 eV | -8.030035994009149 eV |
| `e_coul` | exactly 0 by decision D-11 | +1.6989833033969992 eV |
| `q` (Eu, N) | diagnostic only, q_Eu ~ -2.7 | (-0.5996, +0.5996) |

A chemical bond is worth 2-5 eV, so 10.6 eV is not a self-consistency correction; it is a
different answer.

### 3. The converged charge direction is right, and the sign convention is pinned

`q = population - Znuc` (`src/dftorch/_scf.py:527-528`), so **positive `q` means excess
electrons**, i.e. chemically negative. Eu-N gives `q_Eu = -0.5996` (Eu is the cation) and
`q_N = +0.5996` (N is the anion), which is the correct direction.

Pinned independently against CH4 on `tests/data_skf_mio-1-1`, where carbon must be the anion:
`q = [+0.3026, -0.0755, -0.0703, -0.0706, -0.0862]` for `TYPE = [6, 1, 1, 1, 1]`. Carbon is
positive under this convention, so the convention reads as stated. CH4 converged in
**6 iterations**, `Res = 0.000000007` — confirming Phase 5 commit `4dbffaa` really did fix the
old divergence, so decision 6.1 in `.planning/DECISIONS-NEEDED.md` stays discharged.

### 4. The loop runs away at 9 of the 21 scan separations

Same 21-point grid as `tests/test_eu_n_scan.py` (1.60 to 3.60 A, step 0.10). Whole scan takes
14.5 s including one shared `Constants` build.

| sep / A | single-shot / eV | self-consistent / eV | q_Eu |
|---|---|---|---|
| 1.60 | -3.00610298 | -0.41080677 | -0.5420 |
| 1.70 | -9.68762902 | -6.40946123 | -0.5690 |
| 1.80 | -13.47038969 | -9.43907061 | -0.5902 |
| 1.90 | -15.55709234 | -10.71967277 | -0.6058 |
| 2.00 | -16.67776282 | **-10.99769731** | -0.6172 |
| 2.10 | -17.25701437 | -10.71722809 | -0.6259 |
| 2.20 | -17.51833874 | -10.12983441 | -0.6324 |
| 2.30 | -17.62071417 | -9.42463734 | -0.6351 |
| 2.40 | **-17.64144156** | **+20.20614310** | **-2.8950** |
| 2.50 | -17.61278392 | **+18.17268447** | **-2.7992** |
| 2.60 | -17.55179734 | -7.28384781 | -0.6096 |
| 2.70 | -17.47528328 | -6.63642308 | -0.5905 |
| 2.80 | -17.40088533 | **+29.60144899** | **-3.0364** |
| 2.90 | -17.33454679 | -5.55709844 | -0.5496 |
| 3.00 | -17.24761204 | -5.09048283 | -0.5320 |
| 3.10 | -17.15027346 | **+185.61852280** | **+4.9949** |
| 3.20 | -17.06256033 | **+188.33500969** | **+4.9968** |
| 3.30 | -16.98500529 | **+190.91136613** | **+4.9979** |
| 3.40 | -16.91798038 | **+193.36941770** | **+4.9987** |
| 3.50 | -16.86144284 | **+195.72386055** | **+4.9992** |
| 3.60 | -16.81489147 | **+197.98495321** | **+4.9995** |

Each bolded self-consistent row printed `Did not converge` and then returned the number
anyway.

Read the `q_Eu` column, not just the energies. Two distinct runaway modes:

- **`q_Eu` -> +5.0** at 3.10 A and beyond. Eu has absorbed **all five** of N's valence
  electrons (`Znuc = [9, 5]`). N is stripped bare. The single-shot energy at the same
  separations is entirely sane (-17.2 to -16.8 eV), so H0/S are fine — the failure is purely
  in the charge loop.
- **`q_Eu` -> -3.0** at 2.40, 2.50 and 2.80 A. Eu sheds three electrons instead.

**Consequence for the project's only physics gate:** the single-shot minimum is 2.40 A,
interior and inside the D-24 band `[2.124, 3.186]` A. The self-consistent minimum is
**2.00 A — outside the band**. Turning self-consistency on currently makes the project fail
the check it currently passes.

The 2.655 A point in measurement 2 sits inside a narrow working pocket (2.60-2.70 A both
converge sanely). Any conclusion drawn from that single separation alone is unsafe.

### 5. Eu's per-atom charge penalty is taken from the wrong shell — real defect, NOT the cause

`Constants.py:232` is `self.U = torch.nn.Parameter(US, ...)` — the per-atom Hubbard U is the
**s-shell** value, unconditionally, for every element. Measured for Eu (Z=63):

| shell | Hubbard U / eV |
|---|---|
| s (`const.U`, the one used) | 5.7143911124999995 |
| p (`const.Up`) | 5.1701633875 |
| d (`const.Ud`) | 6.8028465625 |
| **f (`const.Uf`)** | **13.605693125** |

Eu's reference occupation is `el_per_shell = [2.0, 0.0, 0.0, 7.0]` for shells `[s, p, d, f]`
— **seven of its nine valence electrons are in the f shell**, whose penalty is 2.4x the value
the code actually charges it. For N all three of s/p/d read 13.3335792625, identical, which
is why no existing element ever exposed this.

**Tested and disproven as the cause.** Overriding `const.U[63] = const.Uf[63]` in memory and
re-running the full scan: the failure pattern changes (`q_Eu -> -3.0` runaway now starts at
2.50 A and there is no `+5.0` mode) but the loop still fails from 2.50 A outward and the
minimum is **still 2.00 A, still outside the band** (`E = -10.33951844` eV). So this is a
separate genuine defect for f elements, not the instability's root cause. Record it; do not
present it as the fix.

### 6. Non-convergence is invisible to a caller

`structure.scf_converged` does not exist. The only signal is `print("Did not converge")` at
`_scf.py:520-521`, `:911-912`, `:1216-1217`, `:1551-1552`. Measurement 4 is what this costs:
nine confidently wrong energies, up to +198 eV against a true scale near -17 eV, with nothing
machine-readable to distinguish them from the twelve good ones.

### 7. The shell-resolved charge path exists only inside the spin routine

`grep '_sr'` over the closed-shell `SCFx` (`_scf.py:158-556`) returns **nothing**. Every
shell-resolved charge symbol in the SCF machinery (`q_spin_sr`, `net_spin_sr`) lives in
`scf_x_os` (`_scf.py:557-976`, built at `:699-739`, mixed at `:809-856`) — the **open-shell**
routine, which Phase 4 D-12 makes refuse for f systems.

So "shell-resolved charges consumed by the self-consistent path" cannot reuse the consumer
that exists. It means building a new shell-resolved charge path inside the closed-shell loop.
That is the cost estimate behind D-6.06.

</measurements>

<decisions>
## Implementation Decisions

Numbered `D-6.xx` deliberately: Phase 3 used D-01..D-10, Phase 4 used D-11..D-24, and Phase 5
restarted at D-01, so a bare `D-01` is already ambiguous across this project. The phase-scoped
prefix cannot collide.

### Completion bar

- **D-6.01:** The loop must **converge in the neighbourhood of 2.655 A** — not necessarily at
  exactly that separation. Separations elsewhere in the 1.60-3.60 A range may still fail to
  converge; that is accepted for this phase, provided they are honestly reported (D-6.04) and
  visible on the graph (D-6.09).
  - Rejected alternatives, in the user's hearing: requiring all 21 separations to converge,
    and requiring the self-consistent minimum to land inside the D-24 band `[2.124, 3.186]` A.
    Both were offered and neither was chosen.
  - **This means Phase 6 does NOT promise the self-consistent path passes the Eu-N band
    check.** Measurement 4 shows it currently fails it (minimum at 2.00 A). Do not write a
    test that gates on the band for the self-consistent curve; that is not this phase's bar.

- **D-6.02:** The **final gate is human judgement of a freshly computed graph.** A human looks
  at the self-consistent Eu-N binding curve and says whether it looks right. This follows the
  Phase 4 plan 04-05 precedent exactly (`.planning/phases/04-scf-and-reference-simulation-
  validation/04-05-PLAN.md` and `04-05-SUMMARY.md`) — a `checkpoint:human-verify` on a plot
  recomputed from live code, never from a recorded table.
  - The user's words: "The final test should be me looking at the newly generated SCF graph
    and telling you whether it looks good or not."
  - If the human's verdict is that it looks wrong, that is the phase failing its gate, not a
    finding to record and move past.

### The 10.6 eV single-shot / self-consistent difference

- **D-6.03:** The 10.6 eV difference **gets investigated**, and the investigation's deliverable
  is a **written verdict**, not a note: either "this difference is real physics, here is the
  mechanism" or "this is a defect, here is where it lives".
  - The user's words: "I definitely also want to investigate the 10.6 ev gap."
  - Measurement 2 is the whole of what is currently known. Note that a large part of the
    difference is in the band energy (-16.948 -> -8.030 eV), not in the new Coulomb term
    (+1.699 eV) — so an explanation that only accounts for `e_coul` is incomplete.
  - Nothing about the verdict is pre-decided. If it is a defect, whether the fix lands in
    Phase 6 or gets surfaced is a call for whoever plans this phase, given D-6.01's bar.

### Reporting non-convergence

- **D-6.04:** The SCF loop **returns** its convergence result rather than printing it:
  **the iteration count when it converged, and `-1` when it did not.** Modelled on how
  scipy's iterative solvers report an `info` value alongside the answer.
  - The user's words: "similar to how GMRES outputs number of iterations converged, or -1 if
    not converged in scipy."
  - This supersedes the *shape* Phase 4 D-13 left open, and keeps D-13's *substance* intact:
    non-convergence still warns and still returns the last iterate, and still never raises.
  - **Implementation constraint planning must handle:** `SCFx` currently returns a 14-value
    tuple, unpacked positionally at `ESDriver.py:663-678`. Adding a returned value changes
    that tuple for every caller. `_scf.py` has four loops — `SCFx:158`, `scf_x_os:557`,
    `SCFx_batch:977`, `delta_scf_x_os:1233` — and all four carry the same
    `print("Did not converge")` pattern.
  - **Reversibility:** costly — undoing it means reverting a positional return contract at
    every unpack site, and `ESDriver` is not necessarily the only caller. Prefer a shape that
    can be extended once rather than repeatedly.

- **D-6.05:** A separation that fails to converge **away from 2.655 A does not fail the test
  suite.** It is reported through D-6.04's `-1` and marked on D-6.09's graph.
  - **Derived, not the user's words.** It follows from D-6.01 (only the 2.655 A neighbourhood
    must converge) and D-6.08 (nothing gets frozen). It was shown to the user as a derived
    reading and not corrected. If planning finds this untenable, raise it rather than
    reinterpreting it.

### Shell-resolved (per-orbital-group) charges — CONDITIONAL

- **D-6.06:** The shell-resolved charge path and its seven missing f Coulomb blocks are
  **conditional on diagnosis.** Diagnose the runaway first. If the cause is that the coarse
  per-atom charge description is too crude for a 4f element, then building the shell-resolved
  path **is** the fix and belongs in Phase 6. If the cause lies elsewhere, it becomes its own
  phase and Phase 6 does not touch it.
  - Rejected: building it in Phase 6 regardless (the user was told it would mean layering a
    substantial rebuild on top of an unsolved bug, changing the machinery underneath the thing
    being debugged); and ruling it out of Phase 6 unconditionally (which would strand the
    phase if the coarse description *is* the cause).
  - The cost, if it is pulled in, is measurement 7: not "write seven missing blocks" but
    "build a shell-resolved charge path inside the closed-shell loop, where none exists".
  - See `<open_tensions>` — this decision moves two of the three requirements the roadmap
    assigns to Phase 6.

### The graph

- **D-6.07:** The scan stays **1.60 to 3.60 A, the same 21 points** as
  `tests/test_eu_n_scan.py`, so the new curve is comparable point-for-point against the one
  the human already approved in Phase 4.
  - Rejected: extending to ~6 A (offered because at large separation the energy must flatten
    to a constant, which is a correctness check the current range cannot show, and because
    `04-VALIDATION.md` explicitly flagged it as worth doing "once self-consistency lands"),
    and doing both ranges. Neither was chosen. The extension moves to `<deferred>`.

- **D-6.08:** **No numbers get frozen as reference values this phase.** Not
  `-6.920250682037466` eV, not `q = (-0.5996, +0.5996)`, not any point on the new curve.
  - The user's words: "do not freeze these numbers in yet, we don't know if there's a bug."
  - This is the opposite of the Phase 4 pattern (`EU_N_REFERENCE_E_TOT` in
    `tests/test_single_shot_energy.py`) and is deliberate: pinning a value produced by a loop
    that runs away at 9 of 21 separations would carve the bug into the suite.
  - The Phase 4 single-shot pin `-17.510444238744924` is untouched and must stay green.

- **D-6.09:** The graph shows **both curves overlaid** — single-shot and self-consistent —
  with the non-converged separations **visibly marked**.
  - **Derived, not the user's words.** Shown to the user as a derived reading and not
    corrected. Rationale: the human cannot judge "does this look right" without the approved
    reference curve beside it, and cannot interpret a `+198 eV` point without knowing the
    solver gave up there.

### Eu's charge penalty

- **D-6.10:** Record that `Constants.py:232` takes the per-atom Hubbard U from the **s shell**
  for every element, which is wrong for Eu (5.714 eV charged against an atom with 7 of 9
  valence electrons in an f shell whose penalty is 13.606 eV). Record it as a **genuine
  separate defect that is NOT the instability's cause** — measurement 5 tested the substitution
  and the loop still failed.
  - Whether Phase 6 also *fixes* it is left to planning. It is not required by D-6.01's bar,
    and fixing it changes numbers for any future f element, so it is not a free tidy-up.
  - Do not let a later reader mistake this for the diagnosis. It was tested and it is not.

### Claude's Discretion

- Whether all four SCF loops get D-6.04's returned convergence result, or only the closed-shell
  single-system `SCFx` that the f path uses. Consistency argues for all four; the positional
  return-tuple churn argues for restraint. Either is acceptable; state which and why.
- The concrete shape of the returned convergence result (extra tuple element, small dataclass,
  a mirrored attribute on `structure` in addition to the return value) subject to D-6.04's
  semantics: iteration count on success, `-1` on failure.
- Whether the diagnosis work is one task or several, and how it is staged relative to the
  fix.
- How the graph is produced and where it is written (`docs/assets/` already exists and holds
  Phase 4's plot artefacts).
- Whether the 10.6 eV verdict (D-6.03) lives in its own document or inside the phase summary.
- Whether `Constants.py:232` (D-6.10) is fixed, guarded, or documented in place.
- Ordering and wave structure across all of the above.

</decisions>

<open_tensions>
## Open Tensions — Requirements vs. Decisions

**These are live, not resolved. Whoever plans this phase must see the promise and the
condition together.**

1. **Two of Phase 6's three requirements may not be met by Phase 6.** D-6.06 makes the
   shell-resolved work conditional on diagnosis. That directly affects:
   - `.planning/REQUIREMENTS.md` **SCC-02** — "The seven f angular blocks of the shell-resolved
     Coulomb matrix (s-f, f-s, p-f, f-p, d-f, f-d, f-f) are implemented and validated,
     replacing `FShellResolvedCoulombUnsupportedError`."
   - `.planning/REQUIREMENTS.md` **SCC-03** — "The shell-resolved f plumbing validated but
     unconsumed in Phase 4 (D-14) is actually consumed by the self-consistent path."
   - `.planning/ROADMAP.md` Phase 6 success criteria **3** and **4**, which say the same
     things.

   **Deliberately not resolved by editing the roadmap.** Rewriting it now would be a guess
   about what diagnosis will find. It resolves the moment the diagnosis in D-6.03 / D-6.06
   lands, and whoever does that work owes the roadmap and requirements an update at that
   point.

2. **ROADMAP Phase 6 success criterion 1 is already satisfied at one separation, and the
   roadmap's framing of this phase is stale.** The criterion reads "A supported f-containing
   system runs a real self-consistent charge loop to convergence, not the Phase 4 single-shot
   path." Measurement 1 shows that is already true at 2.655 A with no code change. The
   roadmap's scope note (dated 2026-08-02) says "Phase 6 therefore starts from a working
   f-free SCF baseline and does not need to open with a repair task" — that is correct about
   *f-free* CH4 and misleading about the phase, because the **f** loop does need a repair
   task. Read the scope note for the CH4 history, not for this phase's shape.

3. **`MAGNETIC_HUBBARD_LDEP` is the gate for the shell-resolved path (Phase 4 D-23), and it
   currently makes an f system raise.** If D-6.06's condition fires and the shell-resolved
   path gets built, that flag's behaviour changes for f systems. If it does not fire, the flag
   keeps refusing and `docs/F-SUPPORT-STATUS.md` §2's row for
   `FShellResolvedCoulombUnsupportedError` stays accurate as written. Either way the document
   needs a look before the phase closes — `tests/test_support_documentation.py` enumerates the
   exception classes at test time, so a class removed without a matching document edit fails
   the suite.

</open_tensions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` §"Self-Consistent Charge for f Systems" — SCC-01, SCC-02, SCC-03
  (lines 76-78). SCC-02's own text already records that it is blocked on shell-resolved charges
  flowing through SCF.
- `.planning/ROADMAP.md` §"Phase 6" — goal, the 2026-08-02 scope note, and five success
  criteria. **See `<open_tensions>` 1 and 2 before treating criteria 1, 3 or 4 as achievable
  as written.**
- `.planning/DECISIONS-NEEDED.md` §"PHASE 6" — the user's own answers to 6.1 (CH4 divergence,
  discharged), 6.2 (convergence criterion) and 6.3 (closed-shell only). Also X.2: "I'll let you
  know when we get to each phase, just ask" — the autonomy level for this phase is still an
  open ask.

### Prior phase decisions this phase must not contradict
- `.planning/phases/04-scf-and-reference-simulation-validation/04-CONTEXT.md` — D-11
  (single-shot only, `e_coul` exactly 0), **D-12** (spin refuses; still binding),
  **D-13** (non-convergence warns and returns the last iterate, never raises — D-6.04 sets its
  shape), D-14 (shell-resolved plumbing built but unconsumed), D-15 (f Hubbard U from the
  extended SKF headers), **D-23** (`MAGNETIC_HUBBARD_LDEP` is the shell-resolved gate; there is
  no `SHELL_RESOLVED` flag), D-24 (the Eu-N target 2.655 A and the +/-20% band, which
  **may not be tightened or widened**).
- `.planning/phases/04-scf-and-reference-simulation-validation/04-VALIDATION.md` §"Sampling
  Risk" — the four things a green single-shot scan does not sample. Item 2 ("single-shot is not
  SCF, so charge-self-consistency bugs cannot surface by construction") is precisely the bug
  this phase surfaced.
- `.planning/phases/04-scf-and-reference-simulation-validation/04-05-PLAN.md` and
  `04-05-SUMMARY.md` — the human-plot-review checkpoint pattern D-6.02 reuses.
- `.planning/phases/05-regression-safety-and-support-policy-cleanup/05-CONTEXT.md` — D-02
  (library output is gated behind `VERBOSE_LIBRARY_OUTPUT`, **noisy by default, deliberately**;
  set it `False` in diagnostics), D-01 (per-pair radial grid lookup).
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md` — the Takegahara
  1980 f angular formula source lock and AO convention. H0/S are not in question here
  (measurement 4 shows single-shot is sane at every separation), but do not disturb it.

### The f support policy, which this phase can invalidate
- `docs/F-SUPPORT-STATUS.md` §2 — the four-class f exception taxonomy and what removes each.
  §4.1 records the CH4 history. **§3's "what is supported today" table says
  `ESDriver.forward(do_scf=False)`; measurement 1 shows `do_scf=True` also runs. That table
  needs a look.**
- `tests/test_support_documentation.py` — enumerates the exception classes from the module at
  test time, so removing a class without editing the document fails the suite.

### Source files this phase touches
- `src/dftorch/_scf.py:158` `SCFx` (closed-shell single-system — the loop the f path uses);
  `:393-395` the convergence criterion (`ResNorm <= SCF_TOL` **and** `dEc <= SCF_TOL * 100`,
  defaults `1e-6` and `100` iterations); `:520-521` the `Did not converge` print;
  `:527-528` the charge definition `q = population - Znuc`.
- `src/dftorch/_scf.py:557` `scf_x_os`, `:977` `SCFx_batch`, `:1233` `delta_scf_x_os` — the
  other three loops, each with the same print at `:911-912`, `:1216-1217`, `:1551-1552`.
- `src/dftorch/_scf.py:36` `_AndersonMixer` and `:370-384` the mixer setup. The residual
  excursion in measurement 1 (0.156 at iteration 9, then 4.430 at iteration 11 when
  `rank: 0, Fel = 0.000000` first prints) is where the pre-Krylov mixer hands over — a natural
  first place to look.
- `src/dftorch/ESDriver.py:663-678` — the 14-value positional unpack of `SCFx`, which D-6.04
  changes.
- `src/dftorch/ESDriver.py:159-232` — `forward`'s signature, its docstring's account of what
  `do_scf=False` means, and `_require_closed_shell_f_system` as the first statement.
- `src/dftorch/ESDriver.py:322-323` and `:400-430` — where `structure.C_sr` / `dCC_sr` are set
  to `None` and where the shell-resolved matrix is built behind `_select_coulomb_hubbard`.
- `src/dftorch/Constants.py:232` — `self.U = US`, D-6.10.
- `src/dftorch/_coulomb_matrix.py:690` `_require_no_f_shell_resolved_coulomb`, raising at
  `:715`; `:723` `ewald_real_space_vectorized_sr`, the function whose seven f blocks are
  missing.
- `src/dftorch/_slater_koster_pair.py:291` `FShellResolvedCoulombUnsupportedError` and its
  message constant, which already states the blocking argument in full. (Note:
  `docs/F-SUPPORT-STATUS.md` §2 cites this class at `:289`, two lines off. Verified `:291`
  on 2026-08-04.)
- `src/dftorch/Structure.py:435` and `:651` — `Hubbard_U_sr` construction.

### Tests that define the current contract
- `tests/test_single_shot_energy.py` — the Eu-N harness to copy (`_write_eu_n_xyz`,
  `_run_single_shot`, `run_with_float64`) and the pin `EU_N_REFERENCE_E_TOT =
  -17.510444238744924` that must stay green.
- `tests/test_eu_n_scan.py` — the 21-point grid D-6.07 keeps, and the three-branch failure
  message pattern (no interior minimum / short of band / long of band) worth imitating.
- `tests/test_scf.py` — the CH4 f-free SCF smoke test. Measured green: converges in 6
  iterations.
- `tests/test_shell_resolved_u.py` — 27 tests over the shell-resolved Hubbard/charge shapes
  that Phase 4 D-14 built and never consumed.
- Suite baseline entering this phase: **216 passed / 0 failed** across 18 files. Canonical
  invocation is `uv run pytest`, not bare `pytest`.

### Stale — read with caution
- `.planning/codebase/*.md` — mapped 2026-07-20, before Phases 1-5. Verify anything from these
  against current source.
- `.planning/DECISIONS-NEEDED.md`'s fact table says `print("Did not converge")` is at
  `_scf.py:507` and `MaxIt=50`. Both are stale: the print is at `:520-521` and the default cap
  is `SCF_MAX_ITER = 100`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The Eu-N test harness** in `tests/test_single_shot_energy.py` — geometry writer, parameter
  dict, and the `run_with_float64` module-reset wrapper. Every f test module in this project
  uses this shape; do not invent a new one.
- **`tests/test_eu_n_scan.py`'s scan structure** — reuses one `Constants` instance across all
  21 points, verified bit-identical to per-point construction (max abs diff 0.000e+00). This is
  why the scan runs in 14.5 s instead of ~46 s. Reuse it.
- **`VERBOSE_LIBRARY_OUTPUT: False`** (`_tools.py:148` `library_output_enabled`) makes
  diagnostics readable. The noisy default is deliberate per Phase 5 D-02 and must not be
  flipped globally.
- **`_AndersonMixer`** (`_scf.py:36-90`) is already parameterised by `ANDERSON_DEPTH` and
  `ANDERSON_ALPHA` via `dftorch_params`, so mixing behaviour can be probed without editing
  code.
- **`docs/assets/`** already holds Phase 4's plot artefacts — a home for D-6.09's graph.
- **The four-class f exception taxonomy** (`_slater_koster_pair.py:173`, `:234`, `:259`,
  `:291`). Phase 5 pinned it at exactly four names via
  `test_no_new_f_exception_class_was_defined`. Adding a fifth class fails that test; do it
  deliberately or not at all.

### Established Patterns
- **Fail loudly, never silently zero.** Set in Phase 3, reinforced in Phases 4 and 5.
  Measurement 6 is the one place the codebase currently violates its own rule — a wrong number
  is returned with only a printed line — and D-6.04 is what closes it.
- **Named exception raised as the first statement of the guarded method** —
  `_require_f_derivatives` (`ESDriver.py:40`), `_require_closed_shell_f_system`
  (`ESDriver.py:68`), `_require_no_f_shell_resolved_coulomb` (`_coulomb_matrix.py:690`). Each
  tolerates a structure with no `const` so it can be called unconditionally.
- **A physics claim gets a human on a plot recomputed from live code, never from a recorded
  table.** Phase 4 plan 04-05. D-6.02 repeats it.
- **Failure messages carry the diagnosis, not just the verdict** —
  `tests/test_eu_n_scan.py`'s three branches each print the located separation, the band edges
  and the full curve. Also: test modules are pure ASCII, because em dashes rendered as
  replacement characters on a cp1252 console and defeated exactly that requirement.
- Tests run under `torch.set_default_dtype(torch.float64)` with a module-reset fixture.

### Integration Points
- D-6.04 changes `SCFx`'s return tuple, consumed positionally at `ESDriver.py:663-678`. Any
  other caller of the four SCF loops is in the blast radius.
- The diagnosis in D-6.03 spans the charge loop, the mixer, the occupation/Fermi step
  (`dm_fermi_x` called at `_scf.py:352`) and the per-atom Coulomb matrix — it is not localised
  to one module, so a plan that scopes it to one file is probably wrong.
- If D-6.06's condition fires, the shell-resolved work touches `_coulomb_matrix.py:723`
  (seven blocks), `_scf.py:158-556` (a new shell-resolved charge path), `ESDriver.py:400-430`
  (`C_sr` gains a consumer) and `docs/F-SUPPORT-STATUS.md` plus
  `tests/test_support_documentation.py`.

</code_context>

<specifics>
## Specific Ideas

- **The user is the reviewer, and their instrument is a graph.** D-6.02 is not a formality —
  they asked for it in place of a numeric gate, having already done exactly this once for the
  Phase 4 curve. Build the graph for a human eye, not for a log file.
- **The `-1` came from the user, unprompted, with a named precedent** — scipy's iterative
  solvers. Honour the semantics they stated (iteration count on success, `-1` on failure)
  rather than substituting a boolean, which is what Phase 4 D-13's "convergence flag" wording
  would otherwise have suggested.
- **This user rejects requirement IDs used as substitutes for substance.** Recorded in Phase 5
  and reconfirmed here: during this discussion they stopped the workflow to say "assume I don't
  know your context or plans — I have a small familiarity with the codebase only. I don't
  understand complicated jargon." **Downstream agents must describe changes by what the code
  does, define terms of art in plain words at first use, and give scale for numbers** (e.g.
  "10.6 eV, where a chemical bond is 2-5 eV").
- **The user was shown two rejected options for D-6.06 and picked the conditional one
  knowingly**, including the argument that building the shell-resolved path while the bug is
  unsolved changes the machinery underneath the thing being debugged. Do not re-open it as if
  the roadmap's promise settles it.
- Environment: `pytest` 9.1.1, `uv` 0.12.0, `dftorch` installed editable, suite 216 passed /
  0 failed. No CUDA driver on this machine (`Warp CUDA warning` on every run, harmless). A
  spinw.txt warning prints unconditionally and is expected — `tests/f_orbital_data` ships no
  spin parameters.

</specifics>

<deferred>
## Deferred Ideas

- **Extending the scan past 3.60 A (to ~6 A)** — offered and declined for this phase (D-6.07).
  Still the strongest available correctness check on the self-consistent curve: at large
  separation the two atoms barely interact so the energy must flatten to a constant, and
  measurement 4 shows this is exactly where the runaway is worst (+198 eV at 3.60 A, with Eu
  having swallowed all five of N's electrons). `04-VALIDATION.md` flagged it as worth doing
  "once self-consistency lands". Revisit after the runaway is fixed.
- **Requiring the self-consistent Eu-N minimum to land inside the `[2.124, 3.186]` A band** —
  explicitly not this phase's bar (D-6.01). It is the natural bar for whichever phase declares
  the self-consistent path trustworthy.
- **Fixing `Constants.py:232` so the per-atom Hubbard U comes from the shell where the
  electrons actually are** — D-6.10 records it; whether Phase 6 fixes it is planning's call.
  If not fixed here it needs an owner, because it silently mischarges every future f element.
- **Spin-polarized / open-shell treatment of Eu 4f7** — still deferred (Phase 4 D-12). Worth
  noting that the only existing shell-resolved charge path lives in the open-shell routine
  (measurement 7), so spin and shell-resolved work are more entangled in this codebase than
  the requirement list suggests.
- **The `SEDACS` force path consuming `dS` with no f guard** — `docs/F-SUPPORT-STATUS.md` §4.2,
  owned by PHY-05, v2. Untouched here.
- **The two numeric-drift items in Phase 5's `deferred-items.md`** (`water8_mio_full` at
  5e-13 eV, `CH4_DH0_ABS_SUM` at 3.4e-13) — unrelated to this phase.
- **Migrating library output to a logging module** and **flipping `VERBOSE_LIBRARY_OUTPUT` to
  quiet by default** — carried from Phase 5's deferred list, unchanged.

</deferred>

---

*Phase: 6-Self-Consistent SCF for f Systems*
*Context gathered: 2026-08-04*

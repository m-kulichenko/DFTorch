# Phase 4: SCF and Reference Simulation Validation - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Get f-containing **single-system** calculations onto the supported energy path, extend
shell-resolved charge/Hubbard/Coulomb data to carry the f shell, and validate the f-orbital
machinery against a common-sense physical check rather than a published reference.

Covers SIM-01 through SIM-05. Single-system only (D-01 carried forward). Batch, MD, SEDACS,
ML-SK remain out of scope.

</domain>

<decisions>
## Implementation Decisions

### SCF Support Boundary

- **D-11:** Phase 4's completion bar is **single-shot energy with no charge
  self-consistency** — build H0/S, diagonalize once, report band/total energy. This
  satisfies SIM-01 and the "explicit unsupported-mode error" branch of SIM-02. Real
  self-consistent SCF for f systems is deferred beyond this phase.
  — **Reversibility:** reversible — turning self-consistency on later is additive; nothing
  here forecloses it.

- **D-12:** Spin polarization is **explicitly deferred and must error out**. Eu 4f7 is
  open-shell, so a spin-polarized f system raises a named unsupported-mode error rather
  than silently returning a closed-shell number. Consistent with SIM-02's closed-shell
  wording and with the fail-loudly stance established in Phase 3.
  — **Reversibility:** reversible — the guard is a single named exception to remove.

- **D-13:** When an SCF loop **is** exercised and fails to converge, **warn and return the
  last iterate** with a convergence flag, rather than raising. Note this does not arise in
  Phase 4's own single-shot path (D-11); it governs the existing SCF machinery whenever it
  is run, and pre-decides the behaviour for the later phase that turns self-consistency on
  for f systems.

### f-Shell Electrostatics

- **D-14:** **Plumb and validate** f dimensions in charge / Hubbard / Coulomb structures —
  extend the shapes *and* add tests asserting correct values, even though the single-shot
  path (D-11) will not consume them this phase. Fully satisfies SIM-03 and de-risks the
  later SCF work.

- **D-15:** f-shell Hubbard U values come from the **extended-format SKF files**, sourced
  the same way as the existing Us/Up/Ud values. Research must first confirm the Eu/Ga/N SKF
  headers actually carry an f entry; if they do not, that is a blocker to surface, not
  something to zero-fill.
  — **Reversibility:** costly — changing the U source later touches every call site that
  threads `Hubbard_U` / `Hubbard_U_sr` through `ESDriver`.

- **D-16:** Support **both** shell-resolved (l-dependent) and single per-atom U,
  **config-gated**. Default to whichever the Eu-N test case needs. Both paths must be
  tested. — **AMENDED by D-23 below: the flag is `MAGNETIC_HUBBARD_LDEP`, not
  `SHELL_RESOLVED`.**

- **D-23 (amends D-16, resolved 2026-07-29 at plan-phase):** D-16 named a `SHELL_RESOLVED`
  flag in `Constants.py`. **That flag does not exist** — it appears nowhere in `src/` or
  `tests/`. The gate is instead the **existing `MAGNETIC_HUBBARD_LDEP`** key
  (`Constants.py:58` → `self.magnetic_hubbard_ldep`), whose docstring at `Constants.py:39`
  already reads "Use shell-dependent (l-dependent) Hubbard U parameters" — exactly D-16's
  intent. Today its only effect is at `Constants.py:138`, selecting shell- vs. atom-resolved
  **spin W matrix** from `spinw.txt`; this phase extends it to also gate the Coulomb-path
  Hubbard U so the implementation finally matches the documented behaviour.
  - **Do NOT introduce a `SHELL_RESOLVED` flag.** One knob, not two competing ones.
  - Known wart, accepted: the name says "magnetic", a misnomer in the non-spin case, and
    the single flag now controls two things (spin W matrix and Hubbard U). Accepted over a
    rename to keep the blast radius off the existing spin/SOC call sites and `script.py`
    defaults.
  - **Reversibility:** reversible — renaming later is a mechanical change with an alias.

### PME dtype Blocker

- **D-17:** The PME float32/float64 fix **lands in Phase 4**, not deferred to Phase 5.
  Accepted as a deliberate scope stretch: the bug is f-free and pre-existing (so nominally
  REG-01 territory), but it fails `tests/test_scf.py`, which is the smoke gate for exactly
  the path this phase works on. Leaving it red could mask a genuine f-related SCF break.

- **D-18:** Fix scope is a **broader dtype audit**, not just the one line. Sweep the
  codebase for hardcoded `.float()` / `.double()` / dtype assumptions and fix the class of
  bug. Confirmed root cause: `PME_torch.py:261` does `g_mask = (m_2 > 0).float()`, which is
  unconditionally float32 while `E_G` and `metric` are float64 under the project's default
  dtype; the einsum on line 264 then fails. It is currently the **only** `.float()` in
  `src/dftorch/ewald_pme/` — the audit's job is to find the rest of the class elsewhere.

### Reference / Validation Case

- **D-19:** There is **no reference paper**. Validation is a common-sense physical check:
  an **isolated Eu-N diatomic**, scanning the interatomic separation and locating the
  energy minimum.
  — **Critical constraint:** this MUST be done as an **energy scan**, not geometry
  optimization. Geometry optimization needs forces, and `calc_forces` raises
  `FDerivativeUnsupportedError` for any f system (Phase 3, D-09). An energy scan needs only
  energy, so it is compatible with D-11's single-shot decision. A planner that reaches for
  an optimizer here will hit a hard wall.

- **D-20:** Success tolerance is a **loose sanity band** — the minimum lands within roughly
  10-20% of the known Eu-N distance. Appropriate given single-shot energy without
  self-consistency and a closed-shell treatment of an open-shell 4f ion. Do not tighten
  this to a few percent; the method cannot support that claim, and a tight test would fail
  for reasons unrelated to whether the f implementation is correct.

- **D-24 (target value + band, user-supplied 2026-07-29 at plan-phase):** The Eu-N target
  separation is **2.655 Å**, and the band is **±20% → [2.12, 3.19] Å**. Use the percentage
  form, never a hardcoded absolute half-width — earlier research prose mis-stated the band
  as "2.4 ± 0.2 Å = [1.92, 2.88]", which is arithmetically wrong and would silently tighten
  the test to ±8%.
  - **Provenance:** user-supplied table of Eu-N bond lengths in Eu-Bp coordination
    complexes across four ligand variants (Bp, Bp^Me, Bp^Me2, Bp^CF3). Mean of the 16 Eu-N
    entries = 2.655 Å (range 2.606-2.716, spread ~4%). Eu-O rows excluded. This **replaces**
    the earlier unsourced ~2.4-2.5 Å model recollection (RESEARCH.md assumption A3), which
    is now superseded and must not be used.
  - **Known caveat, accepted by the user:** these are dative Eu-N bonds in a crowded
    8-9-coordinate sphere, not an isolated diatomic. A gas-phase Eu-N diatomic would
    plausibly be shorter. Bulk rock-salt EuN (~2.45-2.5 Å) also falls inside the band, so
    two independent reference points agree. The user's call: this is a common-sense test and
    the ±20% band absorbs the difference.
  - **Asymmetric failure interpretation (REQUIRED in the test and its docstring):** a
    minimum landing **short** of the band (below ~2.12 Å) is plausibly correct diatomic
    physics, NOT evidence the f implementation is broken — surface it for human judgement.
    A minimum landing **long** of the band, or **no interior minimum at all**, is a genuine
    red flag. The scan range must extend low enough to observe a short minimum rather than
    clipping it at the band edge.

### Claude's Discretion

- Whether the diatomic scan is uniform or adaptive, and the range/step count — pick
  something that resolves a minimum without excessive cost.
- How the dtype audit is scoped and staged across files.
- Whether the non-convergence warning (D-13) reuses an existing warning channel.

</decisions>

<open_tensions>
## Open Tensions — Requirements vs. Decisions

**RESOLVED 2026-07-29 by the user at plan-phase. These are now binding, not open.**

1. **SIM-04 — RESOLVED: reworded to the Eu-N diatomic case.** There is no reference paper
   (D-19), so SIM-04 no longer refers to one. It now reads: "The isolated Eu-N diatomic
   validation case represented by `tests/f_orbital_data/` is documented with geometry, SKF
   parameter files, observables (energy vs. interatomic separation), units, and
   tolerances." REQUIREMENTS.md and ROADMAP.md success criterion 4 were updated to match.
   The documentation work is still owed by this phase — it was rescoped, not dropped.

   — **D-21:** SIM-04 documents the Eu-N diatomic case, not a reference paper.

2. **SIM-05 — RESOLVED: rescoped to the bond-length sanity check.** With single-shot
   non-self-consistent energy (D-11), a closed-shell treatment of an open-shell 4f ion, and
   no reference paper (D-19), reference-grade reproduction is not claimable. SIM-05 now
   reads: "The supported f-orbital Eu-N energy scan locates an energy minimum within the
   agreed loose sanity band (~10-20%) of the known Eu-N separation." This is D-20's band,
   promoted to the requirement. REQUIREMENTS.md and ROADMAP.md success criterion 5 were
   updated to match. Phase 4 therefore owes a real, passing validation gate.

   — **D-22:** SIM-05 is the loose-band Eu-N minimum check (D-20), not reference-paper
   reproduction. Do not tighten the band.

3. **The `tests/f_orbital_data/` fixtures are broader than the test case.** Nine SKF files
   cover all Eu/Ga/N pairs, but D-19 only exercises Eu-N. The Ga files go unused this
   phase. Not a problem, but worth noting so a planner does not invent Ga work to justify
   them. **Still advisory — do not plan Ga work.**

</open_tensions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 3 foundations (what this phase builds on)
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-CONTEXT.md` — decisions
  D-01..D-10, especially D-01 (single-system only) and D-09 (derivatives deferred)
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md` — the f angular
  formula source lock (Takegahara 1980) and the AO convention mapping
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-01-SUMMARY.md` — what Phase 3
  actually shipped, including the guards that now block f derivative paths
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-VERIFICATION.md` — 8/8 verdict
  and recorded verification debt

### Source files this phase touches
- `src/dftorch/ESDriver.py` — energy path; `Hubbard_U` / `n_shells_per_atom` thread through
  ~14 call sites; `calc_forces` already guards f systems
- `src/dftorch/Structure.py:435` — `Hubbard_U_sr` shell-resolved construction from
  `shell_present` (Phase 2 widened this to 4 shells)
- `src/dftorch/Constants.py` — `SHELL_RESOLVED` / `magnetic_hubbard_ldep` flags, `shell_dim`,
  SKF parameter loading
- `src/dftorch/ewald_pme/PME_torch.py:261` — the `.float()` dtype bug (D-17, D-18)

### Requirements
- `.planning/REQUIREMENTS.md` §"SCF and Full Simulation Validation" — SIM-01..SIM-05
- `.planning/ROADMAP.md` §"Phase 4" — goal and success criteria

### External physics source
- No external spec for this phase. The f angular formulas were source-locked in Phase 3;
  the Eu-N bond length target is a literature/common-knowledge value, not a paper the
  project tracks.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Hubbard_U_sr` / `shell_present` / `n_shells_per_atom` in `Structure.py` — shell-resolved
  scaffolding already exists and Phase 2 already widened it to 4 shells. D-14 extends and
  validates rather than building from scratch.
- `MAGNETIC_HUBBARD_LDEP` config flag in `Constants.py:58` — D-16's config gate already has a
  home. (This line originally named `SHELL_RESOLVED`; that flag does not exist — corrected
  per D-23.)
- `tests/test_f_orbital_skf.py` — 16 passing tests, including the Takegahara orthogonality
  gate. The Eu-N scan test should live alongside these.
- `tests/f_orbital_data/` — 9 Eu/Ga/N SKF files. Only the Eu-N pair is needed for D-19.

### Established Patterns
- **Fail loudly, never silently zero.** Phase 3 set this precedent with
  `FAngularFormulaSourceError` and `FDerivativeUnsupportedError`. D-12 and D-15 follow it.
- **Named exceptions per unsupported mode**, raised as the first statement of the guarded
  method — see `ESDriver.calc_forces`.
- Tests run under `torch.set_default_dtype(torch.float64)` with a module-reset harness.

### Integration Points
- The single-shot energy path (D-11) consumes the H0/S assembly Phase 3 completed.
- Any f system reaching `calc_forces`, `calc_stress`, or the batch path already raises. The
  Eu-N scan must route around these, not through them (D-19).

</code_context>

<specifics>
## Specific Ideas

- The validation is deliberately a **"common sense test"** in the user's words — does the
  code predict a sane Eu-N bond length? Not a publication-grade reproduction. Planning
  should not inflate this into a full benchmarking exercise.
- Environment note: `pytest` 9.1.1 and `uv` 0.12.0 are now installed, and `dftorch` is
  installed editable. The canonical `uv run python -m pytest ...` command works — the Phase
  3 verification debt around unrunnable tests is closed.

</specifics>

<deferred>
## Deferred Ideas

- **Self-consistent SCF for f systems** — deferred out of Phase 4 by D-11. Needs its own
  phase; D-13 already pre-decides its non-convergence behaviour.
- **Spin-polarized / collinear-spin support for open-shell 4f** — raised while discussing
  Eu 4f7 (D-12). Physically the right treatment for rare earths, and a substantial new
  capability. Own phase.
- **f derivatives (forces, stress, MD)** — carried forward as deferred from Phase 3 D-09.
- **Ga-containing and periodic EuN crystal cases** — the SKF set supports them; D-19 chose
  the isolated diatomic instead. Periodic bulk would additionally pull in k-points and
  Ewald/PME.

</deferred>

---

*Phase: 4-SCF and Reference Simulation Validation*
*Context gathered: 2026-07-29*

# Phase 5: Regression Safety and Support Policy Cleanup - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the working f-orbital prototype **supportable and regression-safe** without changing
physics: lock existing simple-format behavior behind tests, make every unsupported f mode
fail explicitly, remove prototype scaffolding, and correct the shared-radial-grid lookup.

Covers REG-01..REG-06 and CLN-01..CLN-05.

**No new physics.** Self-consistent SCF (Phase 6), f derivatives (Phase 7), forces/stress
(Phase 8), and MD (Phase 9) are all out of scope. Spin polarization, SEDACS, ML-SK,
Ga/crystal cases remain v2.

**Sizing note, recorded honestly:** two decisions below (D-01 and D-04) deliberately expand
this phase past a light stabilization pass. D-01 changes an interface with multiple
consumers; D-04 is a full-codebase audit. The user was shown the cost of each and chose the
larger scope both times. Planning should reflect the real size rather than treating Phase 5
as a quick cleanup.

</domain>

<decisions>
## Implementation Decisions

### Radial Grid Lookup

- **D-01:** Fix the shared-radial-grid problem **completely in Phase 5** — both the guard
  and the real per-pair lookup. Do not defer the fix to Phase 8.
  - **The problem:** `_bond_integral.get_skf_tensors` already stores each element pair's own
    radial grid in `R_tensor` (shape `(n_pairs, 1301)`), but then exports a single global
    `R_orb` chosen as *the longest grid seen* (`_bond_integral.py:1064-1065`).
    `_h0ands.H0_and_S_vectorized` computes the interpolation knot from that one array for
    every pair type (`_h0ands.py:556-558`, and the same pattern at `:243-245`). So a pair
    whose real grid differs is interpolated against another pair's ruler.
  - **The fix:** index each pair against its own row of `R_tensor` instead of the shared
    `R_orb`. The data already exists; it is the *lookup* that is wrongly shared.
  - **Plus the guard (REG-05):** loading an `SKFPATH` whose files do not share a radial grid
    step must fail with an explicit error naming the offending files. Keep the guard even
    though D-01 makes mixed loads correct — it catches genuinely incompatible parameter sets.
  - **Same-step / different-length is BENIGN and must NOT be rejected.** Knots coincide, so
    `idx`/`dx` are already right and the tail is zero-padded. `3ob-3-1` has exactly this
    (C-C 650 points, Br-Br 850 points, both step 0.02). Rejecting it would break a working
    parameter set. **Only a differing grid *step* is the hazard.**
  - **Currently inert, verified 2026-07-29:** all nine `tests/f_orbital_data` fixtures load
    to identical 483-point grids (step 0.0211670884 Å = 0.04 Bohr). Phase 4's Eu-N numbers
    are NOT suspect on these grounds. The Phase 3 deferred note claiming these fixtures
    "genuinely differ" is **wrong** and should not be trusted.
  - **When it would bite:** the f dataset is 0.04 Bohr; `mio-1-1`, `3ob-3-1`, `pbc-0-3` and
    `trans3d-0-1` are all 0.02. Any single `SKFPATH` mixing them — e.g. an Eu complex with
    C/H/O ligands — is currently wrong by a factor of two.
  - **Reversibility:** costly — `R_orb` and `coeffs_tensor` are threaded together into the
    ML-SK path and the stress path as well as H0/S, so changing the indexing changes an
    interface with several consumers. Whether to keep exporting `R_orb` additively for
    backward compatibility, or replace it, is left to planning.

### Library Output

- **D-02:** Gate the ~112 library `print()` calls behind a **verbose flag**, and **default
  it to today's noisy behavior**. Do not restructure to a logging module, and do not flip
  the default to quiet.
  - **Rationale for the noisy default:** REG-02 and REG-03 require existing simple-format
    behavior and the tutorial notebook to be unchanged. A quiet default changes what every
    existing caller sees. Preserving byte-identical output is the conservative choice for a
    phase whose whole purpose is regression safety.
  - **Consequence, accepted:** callers who want silence must opt in, and this phase does not
    eliminate the output noise — it only makes it suppressible.
  - **Count:** ~112 across 15 runtime modules once `script.py`'s 57 are removed by D-03.
    Heaviest: `_scf.py` (25), `MD.py` (17), `_h0ands.py` (13), `_xl_tools.py` (10),
    `_coulomb_matrix.py` (8), `ESDriver.py` (7).
  - **Reversibility:** reversible — flipping the default later is a one-line change, and
    migrating to a logger afterwards remains open.

### Prototype Scaffolding

- **D-03:** **Port `src/dftorch/script.py`'s unique checks into `tests/`, then delete it.**
  1656 lines of pre-pytest validation currently ship inside the runtime package, and CI
  never invokes it.
  - **What must be preserved before deleting:** `parse_expected_homonuclear_metadata()`
    builds its expectations by **independently re-parsing the SKF headers**, so it never
    uses the production parser as its own oracle. That property is the real asset and the
    72-test pytest suite does not have it. Port that pattern; do not just delete it.
  - **Do NOT carry over** the fake-package dynamic-import machinery
    (`ensure_fake_dftorch_package()` / `load_dftorch_module()`). It diverges from installed-
    package behavior, and `dftorch` is installed editable so normal imports work.
  - `ConstantsTest` (`Constants.py:211`) is the related stale parallel constants table.
    Whether it goes with `script.py` is left to planning, but it must not be left claiming
    to be a test oracle while bypassing `get_skf_tensors()`.

### Orbital-Count Assumptions

- **D-04:** **Audit every hardcoded `1/4/9` orbital-count site and, per site, either extend
  it to 16 or add an explicit refusal.** End with a written inventory so no site is left
  silently unhandled.
  - **Why not document-only:** a silently unhandled site is exactly the failure mode Phases
    3 and 4 kept catching — a well-shaped matrix with zeros where f values belong. An
    inventory alone leaves that live.
  - **Scale:** 33 `n_orb` comparisons in `_h0ands.py` alone; only 9 sites anywhere currently
    handle `== 16`.
  - Guarding is cheap and always allowed; extend only where f is genuinely supported by this
    phase's scope. A site that would need Phase 6-9 capability gets a refusal, not an
    extension.
  - **Reversibility:** reversible per site — each is a local guard or a local widening.

### Claude's Discretion

- Where the verbose flag lives (almost certainly a `dftorch_params` key, consistent with
  `MAGNETIC_HUBBARD_LDEP` / `DFTB3`) and what it is named.
- Whether genuine failure warnings (e.g. the `spinw.txt` load failure at `Constants.py:149`)
  stay unconditional rather than being gated by the verbose flag.
- Whether D-01 keeps exporting `R_orb` additively for backward compatibility or replaces it.
- How the ported `script.py` checks are organized across test files.
- Whether `ConstantsTest` is deleted, moved under `tests/`, or documented in place.
- How the D-04 inventory is formatted and where it lives.
- Ordering and wave structure across all of the above.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` §"Regression Safety" — REG-01..REG-06 (REG-05/REG-06 added
  2026-07-29 for the radial-grid work described in D-01)
- `.planning/REQUIREMENTS.md` §"Prototype Cleanup and Support Policy" — CLN-01..CLN-05
- `.planning/ROADMAP.md` §"Phase 5" — goal and six success criteria

### Prior phase decisions this phase must not contradict
- `.planning/phases/04-scf-and-reference-simulation-validation/04-CONTEXT.md` — D-11..D-24,
  especially D-11 (single-shot only), D-12 (spin refuses), D-23 (`MAGNETIC_HUBBARD_LDEP` is
  the shell-resolved gate; **no `SHELL_RESOLVED` flag exists**), D-24 (Eu-N band)
- `.planning/phases/04-scf-and-reference-simulation-validation/04-VALIDATION.md` §"Sampling
  Risk" — the four limitations Phase 4 did not sample
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-CONTEXT.md` — D-01 (single-system
  only), D-09 (derivatives deferred)
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md` — the Takegahara
  1980 f angular formula source lock and AO convention mapping

### The radial-grid issue (D-01)
- `.planning/phases/03-h0-s-routing-and-f-angular-blocks/deferred-items.md` §3 — the original
  write-up. **Its claim that `tests/f_orbital_data` grids "genuinely differ" is factually
  wrong** (verified 2026-07-29: all nine load to identical 483-point grids). Read it for the
  mechanism, not for the fixture claim.
- `src/dftorch/_bond_integral.py:1055-1080` — `R_orb_master` selection
- `src/dftorch/_h0ands.py:243-245` and `:556-558` — the two `searchsorted` lookup sites

### Source files this phase touches
- `src/dftorch/script.py` — 1656 lines, D-03
- `src/dftorch/Constants.py:211` — `ConstantsTest`, D-03
- `src/dftorch/_h0ands.py` — 33 orbital-count comparisons, D-04
- 15 runtime modules carrying ~112 `print()` calls, D-02

### Stale — read with caution
- `.planning/codebase/*.md` — mapped 2026-07-20, **before Phases 1-4**. Several gaps it
  reports as open (pytest coverage for f validation, angular f formula correctness) were
  closed by Phases 1-4. The codebase-drift gate flags 454 structural elements since mapping.
  Verify anything from these maps against current source before acting on it.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **72 passing pytest tests** across `test_f_orbital_skf.py` (16), `test_shell_resolved_u.py`
  (27), `test_eu_n_scan.py` (6), `test_single_shot_energy.py` (5), `test_spin_guard.py` (5),
  `test_dtype_contract.py` (4), `test_scf.py` (1), plus others. This is the regression net
  D-01..D-04 lean on.
- **Four named exception classes** in `src/dftorch/_slater_koster_pair.py`:
  `FAngularFormulaSourceError` (173), `FDerivativeUnsupportedError` (234),
  `FSpinPolarizationUnsupportedError` (259), `FShellResolvedCoulombUnsupportedError` (289).
  D-04's refusals should follow this established shape, not invent a new one.
- **`R_tensor`** already holds per-pair radial grids — D-01 does not need new data, only a
  corrected lookup.
- `tests/f_orbital_data/README-EU-N-CASE.md` — the Phase 4 case document pattern, a good
  model for D-04's inventory.

### Established Patterns
- **Fail loudly, never silently zero.** Set in Phase 3, reinforced in Phase 4. D-04's guards
  are a direct continuation.
- **Named exception raised as the first statement of the guarded method** — see
  `_require_f_derivatives` (`ESDriver.py:39-65`) and `_require_closed_shell_f_system`
  (`ESDriver.py:67-100`).
- Tests run under `torch.set_default_dtype(torch.float64)` with a module-reset fixture;
  `tests/test_f_orbital_skf.py` is the template.
- Canonical test command is `uv run pytest` (CI: `.github/workflows/tests.yml:56`). Bare
  `pytest` is not the canonical invocation.

### Integration Points
- D-01 touches the `R_orb` / `coeffs_tensor` pair consumed by H0/S, the ML-SK path, and the
  stress path.
- D-02 touches 15 modules but should not alter any numeric result.
- D-03 removes a file from the shipped package — check `pyproject.toml` packaging and any
  console-script entry point before deleting.
- D-04 touches routing predicates in `_h0ands.py` and wherever else `n_orb` is compared.

</code_context>

<specifics>
## Specific Ideas

- The user hit the print noise directly during Phase 4 execution — every diagnostic command
  this session had to be piped through `grep -v` to read its own output. D-02 is motivated by
  felt pain, not theory.
- The user explicitly asked what "REG-06" meant and rejected being given bare requirement
  IDs without the underlying substance. **Downstream agents should describe changes in terms
  of what the code does, not by ID alone.**
- Environment: `pytest` 9.1.1, `uv` 0.12.0, `dftorch` installed editable, full suite green at
  72 passed / 0 failed. `ruff` is declared as a dev extra but is **not installed**, and
  repo-wide `ruff format --check` likely already fails on pre-existing files.

</specifics>

<deferred>
## Deferred Ideas

- **Migrating library output to a logging module** — D-02 chose a verbose flag instead. A
  proper logger (with levels, handlers, and per-module control) remains the better long-term
  shape and is a natural follow-up once the flag exists.
- **Flipping the verbose default to quiet** — deliberately not done in Phase 5 to protect
  REG-02/REG-03. Reconsider once the tutorial-notebook regression baseline is locked.
- **Centralizing duplicated basis metadata** (`_CHANNELS`, `MAX_SHELLS`, `shell_dim`,
  `AO_LABEL_TEMPLATE` across `_bond_integral.py`, `Constants.py`, `Structure.py`) — CLN-02
  may partially cover this; a full single-source basis module is larger and can stand alone.
- **Splitting legacy CSV parameter loaders out of `_bond_integral.py`** — flagged in the
  2026-07-20 concerns map, not selected for discussion.
- **`wfc.hsd` shell-presence override coverage** and **malformed / skipped-shell SKF
  fixtures** — real test gaps from the concerns map, no fixtures exist for either.
- **The CH4 SCF non-convergence** in `tests/test_scf.py` (residual *growing*: 0.155 → 0.389
  → 0.466). Pre-existing, unasserted, governed by Phase 4 D-13, and recorded as the one open
  entry in `.planning/WINDOWS.md` — which **blocks `/gsd-ship`** until fixed or waived.
  Belongs to Phase 6 (SCF).

</deferred>

---

*Phase: 5-Regression Safety and Support Policy Cleanup*
*Context gathered: 2026-07-29*

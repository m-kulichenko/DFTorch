# Phase 2: Constants and Structure Basis Metadata - Context

**Gathered:** 2026-07-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase makes DFTorch calculation state consistently represent `s/p/d/f`
basis metadata after Phase 1's normalized SKF parser output is loaded, and
before Phase 3's H0/S angular Slater-Koster assembly consumes it. The phase
should prove `Constants`, `Structure`, and `StructureBatch` expose correct
spdf shell, orbital, onsite, Hubbard, reference-density, AO-range, and
shell-range metadata for f-containing systems while preserving existing
f-free metadata behavior.

This phase may inspect or assert the shapes and metadata fields that H0/S will
consume, but it should not execute H0/S assembly. Full f-containing H0/S
matrix construction and angular block formulas belong to Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Validation Gate
- **D-01:** Phase 2 is complete when `Constants`, `Structure`, `StructureBatch`,
  and pytest/script checks prove the full spdf metadata contract is correct.
- **D-02:** Phase 2 should not call H0/S assembly. It may inspect/assert the
  metadata and tensor shapes that H0/S will consume, but executable H0/S
  routing and f angular blocks are Phase 3 work.
- **D-03:** The Phase 2 gate should include a minimal f-containing structure
  state that proves metadata dimensions and routing inputs are sane before
  Phase 3 attempts matrix assembly.

### Metadata Contract
- **D-04:** Treat the full explicit `Constants`, `Structure`, and
  `StructureBatch` metadata surface as the stable downstream contract for this
  phase, not only the minimum H0/S inputs.
- **D-05:** `Constants` must expose and validate `shell_dim`, `n_orb`,
  `max_ang`, `max_ang_occ`, `n_s`, `n_p`, `n_d`, `n_f`, `Es`, `Ep`, `Ed`,
  `Ef`, `U`, `Up`, `Ud`, `Uf`, `shell_present`, `coeffs_tensor`, and
  `pair_lookup`.
- **D-06:** `Structure` must expose and validate `n_orbitals_per_atom`,
  `HDIM`, `H_INDEX_START`, `H_INDEX_END`, `shell_present`, `has_s`,
  `has_p`, `has_d`, `has_f`, shell local/global AO starts and ends, AO labels,
  AO shell types, onsite diagonal entries, `D0`, `Hubbard_U_sr`,
  `shell_types`, `el_per_shell`, `n_shells_per_atom`, `H_INDEX_START_U`, and
  `H_INDEX_END_U`.
- **D-07:** `StructureBatch` must mirror the single-structure contract wherever
  applicable, including padded rows and flattened/global offsets.
- **D-08:** Shared shell dimensions should be represented as
  `[0, 1, 3, 5, 7]` wherever basis metadata is consumed.

### Regression Scope
- **D-09:** Phase 2 should protect existing f-free behavior through metadata
  regression plus existing pytest smoke, not through full tutorial numerical
  comparison.
- **D-10:** Existing s-only, sp, and spd systems should keep unchanged `HDIM`,
  AO ranges, shell metadata, onsite diagonal, `D0`, and shell arrays.
- **D-11:** Current smoke tests should remain green, including import, IO,
  nearest-neighbor, SCF smoke, and public API contract tests where useful.
- **D-12:** Numerical comparison for `experiments/1_tutorial.ipynb` should be
  deferred until the full H0/S and SCF path is runnable, likely in the
  end-to-end validation or regression phases.

### the agent's Discretion
- The agent may decide whether to keep Phase 2 checks primarily in
  `src/dftorch/script.py`, pytest wrappers, or both, as long as the result is
  easy to run and human-readable.
- The agent may choose the smallest f-free fixture set that proves s-only, sp,
  and spd metadata regression without slowing the prototype unnecessarily.
- The agent may add explicit guardrails or assertions for remaining 1/4/9/16
  assumptions when they affect Phase 2 metadata, while leaving broad cleanup
  for Phase 5.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Project goal, active requirements, constraints,
  prototype-first preference, and tutorial parity deferral.
- `.planning/REQUIREMENTS.md` — Phase 2 requirements `BAS-01` through
  `BAS-06`, plus Phase 3/5 boundaries that keep H0/S and tutorial parity out
  of this phase.
- `.planning/ROADMAP.md` — Phase 2 success criteria and dependency boundary
  after Phase 1 and before Phase 3.
- `.planning/STATE.md` — Current project position and known blockers.

### Prior Phase Contract
- `.planning/phases/01-skf-canonicalization-and-spline-validation/01-CONTEXT.md`
  — Locked parser decisions: downstream code consumes normalized 40-channel
  tensors and unsupported basis layouts fail explicitly.
- `.planning/phases/01-skf-canonicalization-and-spline-validation/01-VERIFICATION.md`
  — Verified Phase 1 parser/spline behavior, compact/dashed support, f metadata
  parsing, and pytest coverage.

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Current
  `_bond_integral` -> `Constants` -> `Structure` -> `ESDriver` data flow.
- `.planning/codebase/STRUCTURE.md` — Key file locations for parser metadata,
  Constants registration, Structure AO bookkeeping, and tests.
- `.planning/codebase/TESTING.md` — Validation harness and pytest patterns.

### Source Files
- `src/dftorch/_bond_integral.py` — Source of `N_F`, `EF`, `UF`,
  `SHELL_PRESENT`, 40-channel spline tensors, and pair lookup inputs consumed
  by `Constants`.
- `src/dftorch/Constants.py` — Registers SKF-derived pair tensors and element
  metadata including f-shell fields and `shell_dim`.
- `src/dftorch/Structure.py` — Builds single and batched AO/shell metadata,
  onsite diagonal, Hubbard arrays, occupations, and `D0`.
- `src/dftorch/script.py` — Current scoped f-orbital validation harness for
  parser, constants, and structure metadata.
- `tests/test_f_orbital_skf.py` — Pytest wrapper for the Phase 1 parser/spline
  validation gate; useful pattern for exposing Phase 2 checks through pytest.
- `tests/f_orbital_data/` — Real Eu/Ga/N SKF fixture suite used for f-containing
  parser, constants, and structure validation.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/dftorch/_bond_integral.py:get_skf_tensors()` already returns f-shell
  metadata tensors (`N_F`, `EF`, `UF`, `SHELL_PRESENT`) alongside 40-channel
  spline tensors.
- `src/dftorch/Constants.py` already registers `shell_dim`, `n_f`, `Ef`, `Uf`,
  and `shell_present` as device-aware parameters.
- `src/dftorch/Structure.py` already defines `SHELL_DIMS = (1, 3, 5, 7)`,
  fixed local AO starts, AO labels through f functions, shell masks, single
  `D0`, and batched `D0` helpers.
- `src/dftorch/script.py` already contains independent expected-value helpers
  for metadata, Constants checks, single Structure checks, and StructureBatch
  checks against `tests/f_orbital_data`.
- `tests/test_f_orbital_skf.py` shows how to expose the validation script to
  pytest while forcing CPU `torch.float64` behavior.

### Established Patterns
- Normalize external SKF dialect differences at the parser boundary; downstream
  tensors should use the 40-channel representation without branching on raw
  simple vs extended files.
- Represent shell presence explicitly with `shell_present`, rather than
  inferring layout only from `max_ang`.
- Use nested shell layouts only: s, sp, spd, or spdf. Skipped-shell basis
  layouts should fail explicitly.
- Keep AO ordering atom-major and shell-nested: s, p, d, f with local starts
  `0`, `1`, `4`, `9`.
- Prefer explicit shape/value assertions and small helper functions over clever
  abstractions during the prototype.

### Integration Points
- `Constants.__init__` consumes `get_skf_tensors(TYPE, SKFPATH)` and registers
  returned tensors.
- `Structure.__init__` consumes `const.n_orb`, `const.shell_present`,
  `const.Ef`, `const.Uf`, `const.n_f`, and `const.shell_dim` to build the
  metadata contract that H0/S and SCF code will later consume.
- `StructureBatch.__init__` should stay aligned with `Structure.__init__`,
  including padded rows and global flattened offsets.
- `ESDriver.py` is a downstream consumer for Phase 3; Phase 2 should prepare
  its inputs but avoid executing H0/S.

</code_context>

<specifics>
## Specific Ideas

- The user selected a Phase 2 gate that verifies metadata and H0/S input shapes
  without executing H0/S assembly.
- The user wants both the full explicit metadata contract and SCF-oriented
  metadata contract verified in Phase 2.
- The user explicitly deferred `experiments/1_tutorial.ipynb` numerical
  comparison until H0/S and SCF are runnable, because tutorial parity is not a
  meaningful Phase 2 gate.

</specifics>

<deferred>
## Deferred Ideas

- Execute f-containing H0/S assembly and implement f angular Slater-Koster
  blocks in Phase 3.
- Compare `experiments/1_tutorial.ipynb` numerical outputs after H0/S and SCF
  are runnable, in the later full simulation/regression phases.
- Broad cleanup of remaining 1/4/9/16 assumptions, support policy, and
  unsupported f modes belongs mainly to Phase 5 unless a Phase 2 assumption
  directly blocks metadata correctness.

</deferred>

---

*Phase: 2-Constants and Structure Basis Metadata*
*Context gathered: 2026-07-21*

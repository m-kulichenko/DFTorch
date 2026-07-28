# Phase 1: SKF Canonicalization and Spline Validation - Context

**Gathered:** 2026-07-20
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase verifies and hardens the SKF parsing boundary. DFTorch should load simple 20-column and extended 40-column `.skf` files, normalize both into one 40-channel internal representation, and validate spline reconstruction against the `tests/f_orbital_data/` fixture suite before downstream constants, structure, or H0/S code relies on the data.

</domain>

<decisions>
## Implementation Decisions

### Parser Contract
- **D-01:** Downstream code should see only normalized 40-channel tensors. Raw 20-column/simple and 40-column/extended differences belong inside `src/dftorch/_bond_integral.py`.
- **D-02:** Simple-format files should map existing s/p/d channels into the universal order and zero-fill f-only channels.
- **D-03:** The existing `_bond_integral.py` changes appear to already implement this direction and should be treated as the starting point for Phase 1 planning, not redesigned from scratch.

### Validation Harness
- **D-04:** Keep Phase 1 validation in `src/dftorch/script.py` for now. The script is the working prototype checkpoint and should remain easy to run and inspect.
- **D-05:** Do not require immediate pytest conversion in Phase 1. Pytest migration can be revisited later when the prototype stabilizes.

### Fixture Scope
- **D-06:** The Phase 1 test suite is only `tests/f_orbital_data/`.
- **D-07:** Phase 1 should validate the provided f-orbital data suite, including simple and extended SKF layouts present there.

### Parser Strictness
- **D-08:** Unsupported basis layouts should fail explicitly.
- **D-09:** The parser should still tolerate known DFTB+ file-format quirks where doing so is compatible with an unambiguous interpretation.

### the agent's Discretion
- The agent may choose the exact assertions and helper boundaries inside `src/dftorch/script.py`, as long as the script remains human-readable and exercises the parser behavior above.
- The agent may document current parser behavior directly from `src/dftorch/_bond_integral.py` when creating the Phase 1 plan.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Project goal, active requirements, constraints, and prototype-first preference.
- `.planning/REQUIREMENTS.md` — Phase 1 requirements `SKF-01` through `SKF-07` and `SPL-01` through `SPL-04`.
- `.planning/ROADMAP.md` — Phase 1 success criteria and phase boundaries.
- `.planning/STATE.md` — Current project position and known blockers.

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — Current parser, constants, structure, and driver data flow.
- `.planning/codebase/TESTING.md` — Current validation-script conventions and fixture layout.
- `.planning/codebase/CONCERNS.md` — Known parser/test gaps and fragile assumptions.

### Source Files
- `src/dftorch/_bond_integral.py` — SKF parser, 40-channel order, simple-to-extended normalization, filename resolution, homonuclear metadata, and skipped-shell validation.
- `src/dftorch/script.py` — Current f-orbital validation harness.
- `tests/f_orbital_data/` — The Phase 1 fixture suite.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/dftorch/_bond_integral.py:_CHANNELS` defines the 40-channel internal order.
- `src/dftorch/_bond_integral.py:_SIMPLE_CHANNELS` and `_SIMPLE_TO_EXTENDED` define the legacy simple-format mapping into the canonical representation.
- `src/dftorch/_bond_integral.py:_normalize_skf_row()` accepts 20-column or 40-column electronic rows and emits canonical 40-channel rows.
- `src/dftorch/_bond_integral.py:_resolve_skf_path()` supports dashed and compact pair filenames.
- `src/dftorch/_bond_integral.py:_validate_nested_shells()` rejects unsupported skipped-shell layouts.
- `src/dftorch/script.py` already has parser/spline, constants, and structure validation helpers against the f-orbital fixtures.

### Established Patterns
- Normalize external file formats at the parser boundary; downstream tensors should not branch on raw SKF format.
- Keep basis support nested as s, sp, spd, or spdf.
- Prefer explicit shape and metadata assertions over clever abstractions during the prototype.
- Use CPU `torch.float64` for validation-script checks.

### Integration Points
- `Constants` consumes `get_skf_tensors()` output from `_bond_integral.py`.
- `Structure` and later H0/S code depend on the shape and shell metadata emitted by `Constants`.
- Phase 1 should stop at the parser/spline boundary plus the existing validation harness; full H0/S formula support belongs to later phases.

</code_context>

<specifics>
## Specific Ideas

- The user explicitly wants `src/dftorch/_bond_integral.py` read before planning because many parser changes should already be present.
- The user prefers local implementation and later selective GitHub push of necessary codebase files only; GSD planning files should stay local.

</specifics>

<deferred>
## Deferred Ideas

- Convert the `src/dftorch/script.py` checks into pytest tests after the prototype stabilizes.
- Broaden validation beyond `tests/f_orbital_data/` in a later regression or cleanup phase if needed.

</deferred>

---

*Phase: 1-SKF Canonicalization and Spline Validation*
*Context gathered: 2026-07-20*

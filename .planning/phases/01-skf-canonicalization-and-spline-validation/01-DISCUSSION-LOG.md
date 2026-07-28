# Phase 1: SKF Canonicalization and Spline Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-20
**Phase:** 1-SKF Canonicalization and Spline Validation
**Areas discussed:** Parser Contract, Validation Scope, Fixture Coverage, Error Strictness

---

## Parser Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Normalized only | Downstream code sees only canonical 40-channel tensors. | ✓ |
| Preserve raw metadata too | Keep source-format metadata for debugging alongside normalized tensors. | |

**User's choice:** Expose only the normalized 40-channel tensors.
**Notes:** User noted that `_bond_integral.py` should already implement these changes.

---

## Validation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Keep script only | Continue using `src/dftorch/script.py` as the prototype validation checkpoint. | ✓ |
| Convert to pytest | Move core checks under `tests/` now. | |
| Do both | Keep script for debugging and add pytest for regression. | |

**User's choice:** Keep only `src/dftorch/script.py` for now.
**Notes:** Pytest was explained as the standard Python test runner. The user chose the faster prototype path.

---

## Fixture Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| f_orbital_data only | Use only the uploaded `tests/f_orbital_data/` suite. | ✓ |
| Add existing simple fixtures | Also validate current non-f simple-format datasets. | |
| Add malformed fixtures | Add negative parser fixtures now. | |

**User's choice:** The test suite is only the `f_orbital_data` suite.
**Notes:** Later phases may add broader regression coverage.

---

## Error Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Strict unsupported-layout errors | Fail on unsupported basis layouts. | ✓ |
| DFTB+ quirk tolerance | Accept known DFTB+ quirks when interpretation remains clear. | ✓ |
| Loose parsing | Accept uncertain layouts with warnings. | |

**User's choice:** Fail on unsupported layout, but handle known DFTB+ quirks.
**Notes:** This supports strict scientific behavior while staying compatible with real parameter files.

---

## the agent's Discretion

- Choose exact `script.py` assertion structure and helper details.
- Treat the existing `_bond_integral.py` implementation as current context when planning Phase 1.

## Deferred Ideas

- Convert script validation to pytest after the prototype stabilizes.
- Broaden fixture coverage beyond `tests/f_orbital_data/` later if needed.

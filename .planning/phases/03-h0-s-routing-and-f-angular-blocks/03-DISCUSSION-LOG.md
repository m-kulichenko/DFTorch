# Phase 3: H0/S Routing and f Angular Blocks - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-27
**Phase:** 3-H0/S Routing and f Angular Blocks
**Areas discussed:** Prototype Slice, Formula Source, AO Convention, Test Oracle, Routing Scope

---

## Prototype Slice

| Option | Description | Selected |
|--------|-------------|----------|
| Single system first | Implement `ESDriver` / `H0_and_S_vectorized` path before batch support. | ✓ |
| Include batch routing too | Include `StructureBatch`, `ESDriverBatch`, and batched H0/S in this phase. | |

**User's choice:** Single system first; do not worry about batches.
**Notes:** Batch f support is deferred to later phases.

---

## Formula Source

| Option | Description | Selected |
|--------|-------------|----------|
| Sharma paper | Implement f Slater-Koster formulas from the Sharma f-electron paper. | ✓ |
| Table-driven/generator helper | Use a more abstract generated/table-based representation for auditability. | |
| Other literature search | Research conventions first without locking a paper source. | |

**User's choice:** Use the Sharma paper.
**Notes:** User can send the paper if needed. Quick lookup identified likely source as “Slater-Koster tables for f electrons,” DOI `10.1088/0022-3719/13/4/016`.

---

## AO Convention

| Option | Description | Selected |
|--------|-------------|----------|
| Lock current order immediately | Treat current `Structure.py` f labels as already canonical. | |
| Verify before planning | Check paper/DFTB convention against current labels before implementation planning. | ✓ |

**User's choice:** Verify before planning.
**Notes:** Current labels are `fx3`, `fy3`, `fz3`, `fx_y2_z2`, `fy_z2_x2`, `fz_x2_y2`, `fxyz`.

---

## Test Oracle

| Option | Description | Selected |
|--------|-------------|----------|
| Finite/symmetric H0/S only | Prove matrix construction shape and numerical sanity. | |
| Add selected angular tests | Include known-direction formula checks. | ✓ |
| Compare selected block entries | Add hand-calculated paper-formula checks for specific H0/S entries. | ✓ |

**User's choice:** More tests are better.
**Notes:** Completion should include minimal Eu-containing H0/S build, finite/symmetric checks, selected axis-direction angular tests, selected hand-calculated block entries, and existing simple-format smoke tests.

---

## Routing Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Explicitly support all related paths now | Include gradients, batch, forces, stress, MD. | |
| Defer unsupported paths | Keep Phase 3 focused on single-system H0/S and defer wider modes. | ✓ |

**User's choice:** Defer gradients, batch, forces, and stress for later phases.
**Notes:** Phase 3 may add explicit unsupported guards if needed to avoid silent failures, but should not implement deferred modes.

---

## the agent's Discretion

- Keep the prototype simple and human-troubleshootable.
- Prefer explicit tests and formula readability over premature abstraction.

## Deferred Ideas

- Batch f routing.
- f gradients, forces, stress, MD, SEDACS, ML-SK.
- Full SCF/reference-paper validation in Phase 4.

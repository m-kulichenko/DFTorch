# Phase 2: Constants and Structure Basis Metadata - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-21
**Phase:** 2-Constants and Structure Basis Metadata
**Areas discussed:** Validation gate, Metadata contract, Regression scope

---

## Initial Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Validation gate | Decide whether Phase 2 should finish when Constants/Structure metadata tests pass, leaving H0/S use for Phase 3. | ✓ |
| Metadata contract | Decide exactly which Constants and Structure fields must be treated as the stable spdf contract downstream. | |
| Regression scope | Decide how much simple-format s/sp/spd regression proof is required in this phase versus Phase 5. | |

**User's choice:** Validation gate.
**Notes:** The discussion started with the stopping point for Phase 2 because it determines how far planning should go before Phase 3.

---

## Validation Gate

| Option | Description | Selected |
|--------|-------------|----------|
| Metadata only | Pass when Constants, Structure, StructureBatch, and tests prove f metadata, AO ranges, shell ranges, onsite diagonal, Hubbard arrays, and D0 are correct. H0/S waits for Phase 3. | |
| Metadata plus smoke H0/S shape | Also require a minimal f-containing structure to reach the H0/S call boundary, even if f angular blocks are still unsupported or guarded. | ✓ |
| Metadata plus no-op driver guard | Require ESDriver to explicitly reject f-containing H0/S assembly with a clear Phase 3-not-implemented error. | |

**User's choice:** Metadata plus smoke H0/S shape.
**Notes:** Follow-up clarified that Phase 2 should inspect/verify H0/S-consumed metadata and shapes, not execute H0/S.

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit unsupported error | Smoke test may pass by reaching a clear NotImplementedError or ValueError after confirming metadata and matrix dimensions are sane. | |
| Finite partial matrix | Require H0/S to return finite matrices for non-f blocks while f-containing blocks are guarded or skipped. | |
| No H0/S execution yet | Only inspect the inputs that would be passed to H0/S, without calling the H0/S assembler. | ✓ |

**User's choice:** No H0/S execution yet.
**Notes:** Full H0/S execution and f angular Slater-Koster formulas remain Phase 3.

---

## Metadata Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Full explicit contract | Lock all Constants, Structure, StructureBatch fields, including labels and both single/batch global offsets. | ✓ |
| Minimal H0/S inputs only | Lock only fields needed for H0/S assembly. | |
| SCF-oriented contract | Also prioritize D0, Hubbard_U_sr, el_per_shell, shell_types, and n_shells_per_atom. | ✓ |

**User's choice:** Full explicit contract and SCF-oriented contract.
**Notes:** The user wants the whole metadata surface validated because shape/order bugs here would be painful to debug in Phase 3.

---

## Regression Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Metadata regression only | Verify existing s-only, sp, and spd systems keep metadata behavior; deeper tutorial outputs stay for Phase 5. | |
| Metadata plus existing pytest smoke | Do metadata regression and require current import/IO/neighbor-list/SCF/public API smoke tests to pass. | ✓ |
| Tutorial comparison now | Also run or reproduce experiments/1_tutorial.ipynb outputs in Phase 2. | |

**User's choice:** Metadata plus existing pytest smoke.
**Notes:** The user explicitly reasoned that tutorial numerical comparison must wait until the SCF loop and H0/S path are completed, so it should not gate Phase 2.

---

## the agent's Discretion

- The agent may choose how to expose Phase 2 validation through `src/dftorch/script.py`, pytest, or both.
- The agent may choose the smallest f-free fixture set sufficient to prove metadata regression for s-only, sp, and spd systems.

## Deferred Ideas

- Execute f-containing H0/S assembly in Phase 3.
- Compare `experiments/1_tutorial.ipynb` numerical outputs after H0/S and SCF are runnable.

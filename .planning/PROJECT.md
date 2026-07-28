# DFTorch F-Orbital Support

## What This Is

DFTorch is an existing PyTorch-based DFTB codebase that reads Slater-Koster parameter files, builds Hamiltonian and overlap matrices, runs SCF calculations, and supports simulation workflows through `Constants`, `Structure`, `ESDriver`, and related kernels. This project extends DFTorch to support f-orbital Slater-Koster data and calculations while preserving the behavior of existing simple-format s/p and s/p/d workflows.

The immediate goal is a working, troubleshootable prototype that can parse both simple and extended `.skf` files into a universal internal representation, carry f-orbital data through constants and H0/S construction, and ultimately reproduce reference-paper simulations represented by the `tests/f_orbital_data/` fixtures.

## Core Value

DFTorch can run scientifically valid f-orbital DFTB simulations without changing numerical results for existing simple-format calculations.

## Requirements

### Validated

- ✓ DFTorch exposes user-facing calculation objects for constants, structures, electronic-structure drivers, MD, and optimization — existing
- ✓ Existing workflows read SKF parameter files through `src/dftorch/_bond_integral.py` and load them into `src/dftorch/Constants.py` — existing
- ✓ Existing simple-format SKF datasets support current s/p and s/p/d calculations — existing
- ✓ Existing SCF and force smoke tests exercise the `Constants` → `Structure` → `ESDriver` path with bundled simple-format SKF data — existing
- ✓ Existing tutorial workflows in `experiments/1_tutorial.ipynb` provide a regression reference for simple-format behavior — existing

### Active

- [ ] Parse both simple and extended `.skf` files in `src/dftorch/_bond_integral.py`.
- [ ] Convert simple-format files into the same universal internal channel representation used for extended-format files.
- [ ] Follow the existing no-d-orbital handling pattern for s/p elements, and add the analogous handling for elements with s/p/d orbitals but no f orbitals.
- [ ] Add spline-coefficient regression tests that reproduce original SKF channel values from `tests/f_orbital_data/`.
- [ ] Update `src/dftorch/script.py` or an equivalent test harness so f-orbital SKF parsing can be checked by running cubic spline coefficients against source SKF values.
- [ ] Update `src/dftorch/Constants.py` so expanded channel and orbital data are stored correctly for f-orbital and non-f-orbital element pairs.
- [ ] Add f-orbital angular Slater-Koster transformation formulas.
- [ ] Wire f-orbital transformations through H0/S construction in `src/dftorch/ESDriver.py`, `src/dftorch/_h0ands.py`, and related dependencies.
- [ ] Validate full f-orbital simulation behavior against the reference-paper cases represented by `tests/f_orbital_data/`.
- [ ] Preserve existing simple-format simulation outputs, especially the calculations in `experiments/1_tutorial.ipynb`.

### Out of Scope

- Rewriting the whole electronic-structure driver — the f-orbital path should extend the current architecture first, then cleanups can follow.
- A perfectly clean abstraction layer in the first prototype — technical debt should be minimized and documented, but working scientific behavior comes first.
- Changing public API names or tutorial behavior unless required for f-orbital support.
- Broad performance optimization of f-orbital kernels before correctness and regression checks are in place.
- Reworking unrelated PME, SEDACS, GBSA, D3, or MD internals unless they block the f-orbital simulation path.

## Context

This is a brownfield extension of an existing scientific Python package. The relevant codebase map is in `.planning/codebase/`, especially:

- `.planning/codebase/ARCHITECTURE.md` for the `Constants` → `Structure` → `ESDriver` data flow.
- `.planning/codebase/STRUCTURE.md` for key implementation locations.
- `.planning/codebase/TESTING.md` for current pytest conventions and fixture layout.
- `.planning/codebase/CONCERNS.md` for fragile areas and regression risks.

The implementation will likely touch:

- `src/dftorch/_bond_integral.py` for SKF parsing, spline construction, and channel normalization.
- `src/dftorch/Constants.py` for storing expanded orbital/channel tensors.
- `src/dftorch/_h0ands.py` and `src/dftorch/_slater_koster_pair.py` for H0/S assembly and pair interpolation.
- `src/dftorch/ESDriver.py` for driver wiring and any shape assumptions.
- `src/dftorch/_legacy/H0andS.py` if legacy formulas are still used as references or compatibility paths.
- `src/dftorch/script.py` and tests under `tests/` for spline and simulation validation.
- `tests/f_orbital_data/` for extended/simple f-orbital SKF fixtures from the reference-paper dataset.
- `experiments/1_tutorial.ipynb` as a simple-format regression reference.

The user wants a working prototype as soon as possible. The code should remain easy for a human to troubleshoot: prefer explicit data-shape conversions, clear channel ordering, small testable helpers, and simple branching over clever hidden abstractions. Cleanup is expected after the prototype demonstrates correctness.

## Constraints

- **Scientific correctness**: The f-orbital path must reproduce source SKF values through spline evaluation and ultimately match reference-paper simulation behavior.
- **Regression safety**: Existing simple-format calculations, especially tutorial outputs, must remain numerically unchanged within appropriate tolerances.
- **Data representation**: Simple and extended `.skf` formats must converge to one universal internal representation before downstream code consumes them.
- **Incremental validation**: First checkpoint is `src/dftorch/_bond_integral.py`; later checkpoints are `Constants`, Slater-Koster angular transforms, H0/S construction, then full simulation.
- **Troubleshooting**: Implementations should be simple enough to inspect by hand when channel order, orbital count, or tensor shape bugs appear.
- **Prototype-first**: Some temporary technical debt is acceptable if it is localized, documented, and does not obscure scientific assumptions.
- **Compatibility**: Avoid public API churn and preserve existing simple s/p and s/p/d behavior.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target full f-orbital simulation support as the final milestone | The reference dataset comes from a paper with simulations, and reproducing those results is the validity test for the codebase | — Pending |
| Start with `src/dftorch/_bond_integral.py` parsing and spline validation | Channel normalization is the foundation for every later constants and H0/S change | — Pending |
| Use a universal internal channel representation | Extended SKF files have more channels than simple files; downstream code should not branch on raw file format | — Pending |
| Preserve simple-format tutorial behavior as a regression guard | Existing simple-format users should not see changed calculations from f-orbital support | — Pending |
| Optimize for a working prototype before deep cleanup | The project needs f-orbital capability soon, but the prototype must remain easy to debug | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-17 after initialization*

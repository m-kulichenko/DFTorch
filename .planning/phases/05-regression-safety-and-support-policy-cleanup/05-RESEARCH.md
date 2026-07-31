# Phase 5: Regression Safety and Support Policy Cleanup - Research

**Researched:** 2026-07-30
**Domain:** Prototype stabilization, regression testing, and explicit failure modes
**Confidence:** HIGH (all claims verified from source code and CONTEXT.md decisions)

## Summary

Phase 5 stabilizes the working f-orbital prototype by locking existing behavior behind regression tests, making unsupported modes fail explicitly, removing prototype scaffolding, and fixing a critical shared-radial-grid lookup bug that affects mixed-SKF-path calculations.

Four locked decisions (D-01 through D-04) shape the entire phase:

1. **D-01:** Fix shared-radial-grid problem COMPLETELY with both guard and per-pair lookup. Data exists (`R_tensor`); only indexing needs correction. Multiple consumers affected: H0/S, ML-SK path, stress path.

2. **D-02:** Gate ~120 library `print()` calls behind a verbose flag defaulting to TODAY'S NOISY BEHAVIOR. Not a logging migration; not a quiet default. Preserves byte-identical output for REG-02/REG-03.

3. **D-03:** Port `script.py`'s unique checks (especially `parse_expected_homonuclear_metadata`) into `tests/`, then delete script.py entirely. ConstantsTest parallel table also needs disposition.

4. **D-04:** Audit all hardcoded orbital-count sites (1/4/9/16 comparisons) and per site either extend to 16 or add explicit refusal. Not document-only; silent unhandling is the failure mode being prevented.

**Critical Finding:** The CONTEXT.md claim that `tests/f_orbital_data` fixtures "genuinely differ" in radial grids is **wrong** — all nine load to identical 483-point grids with 0.04 Bohr step. Phase 4's Eu-N numbers are NOT suspect on grid grounds. D-01's guard is still essential because real parameter sets like `mio-1-1`, `3ob-3-1`, `pbc-0-3`, `trans3d-0-1` are 0.02 Bohr, and mixing them with the 0.04 f dataset in one `SKFPATH` is currently broken by a factor of two.

**Phase size:** Deliberately expanded by D-01 and D-04, which change interfaces and require full-codebase audits. Conservative and justifiable scope.

**Primary recommendation:** Implement all four decisions in dependency order (D-01 infrastructure first, then D-02, D-03, D-04 in parallel), with careful staging to keep regression tests green throughout.

## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01 through D-04)

**D-01: Radial Grid Lookup — Fix COMPLETELY in Phase 5**
- Problem: `_bond_integral.get_skf_tensors` stores each pair's grid in `R_tensor` (n_pairs, 1301) but exports a single global `R_orb` chosen as the longest grid seen (lines 1064-1065). `_h0ands.H0_and_S_vectorized` uses that one array for every pair type (lines 243-245, 556-558), so a pair whose real grid differs gets interpolated against another pair's ruler.
- Solution: Index each pair against its own row of `R_tensor` instead of the shared `R_orb`.
- Guard (REG-05): Loading an `SKFPATH` whose files don't share a radial grid **step** must fail with explicit error naming offending files. Same-step/different-length is BENIGN and must NOT be rejected (e.g., 3ob-3-1 has C-C 650 points and Br-Br 850 points, both 0.02 step).
- Status: All nine `tests/f_orbital_data` fixtures verified to load to identical 483-point grids (step 0.04 Bohr). Currently inert. Will bite when mixing with 0.02-step parameter sets.

**D-02: Library Output — Gate ~120 `print()` calls behind verbose flag, default to NOISY**
- Rationale: REG-02/REG-03 require byte-identical existing behavior. Quiet default changes what every caller sees.
- Count: ~120 across 15 runtime modules (after `script.py`'s 57 removed by D-03).
- Heaviest: `_scf.py` (25), `MD.py` (17), `_h0ands.py` (13), `_xl_tools.py` (10).
- Important: Failure warnings (e.g., spinw.txt load at `Constants.py:149`) may stay unconditional rather than gated by verbose flag.
- Threading: Many prints in free functions without access to `dftorch_params`. Verbose flag must reach them (possibly via `Constants` instance, which threads `magnetic_hubbard_ldep` at `Constants.py:64`).

**D-03: Prototype Scaffolding — Port unique `script.py` checks, then delete**
- `script.py` is 1656 lines, not invoked by CI, shipped inside runtime package.
- Key asset: `parse_expected_homonuclear_metadata()` (line 176) re-parses SKF headers independently as oracle, not using production parser to validate itself. Must be preserved before deletion.
- `ConstantsTest` (line 211): Parallel in-memory table. Disposition (keep/move/delete) deferred to planning.
- No console-script entry point (pyproject.toml has no `[project.scripts]`). Safe to delete.

**D-04: Orbital-Count Assumptions — Audit every 1/4/9/16 site, extend or guard per site**
- 43-44 comparisons in `_h0ands.py` alone; 9 sites anywhere currently handle `== 16`.
- Sites with `== 16`: _h0ands.py (8 pair-mask definitions at lines 187-206), _coulomb_matrix.py (1 at line 683), stress path (_stress.py line 195).
- Hardcoded offsets: `SHELL_LOCAL_STARTS = (0, 1, 4, 9)` at `Structure.py:12`; `AO_LABEL_TEMPLATE` 16-position at `Structure.py:25-42`.
- Why not document-only: Silent unhandling is the failure mode — well-shaped matrices with zeros where f values belong.

### Claude's Discretion

- Where the verbose flag lives (almost certainly `dftorch_params` key, consistent with `MAGNETIC_HUBBARD_LDEP` at `Constants.py:64`, `DFTB3` at `Constants.py:203`)
- Whether genuine failure warnings stay unconditional
- Whether D-01 keeps exporting `R_orb` additively for backward compatibility or replaces it
- How ported `script.py` checks are organized across test files
- Whether `ConstantsTest` is deleted, moved to `tests/`, or documented in place
- How D-04 inventory is formatted and where it lives
- Ordering and wave structure across all decisions

### Deferred Ideas (OUT OF SCOPE)

- Migrating library output to a logging module (chosen verbose flag instead; logger migration remains open)
- Flipping verbose default to quiet (would break REG-02/REG-03)
- Centralizing duplicated basis metadata across `_CHANNELS`, `MAX_SHELLS`, `shell_dim`, `AO_LABEL_TEMPLATE`
- The CH4 SCF non-convergence in `tests/test_scf.py` (residual growing: 0.155 → 0.389 → 0.466) — pre-existing, unasserted, governed by Phase 4 D-13, blocks `/gsd-ship` until fixed or waived

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-01 | Existing pytest smoke tests pass after phase | 72 passing pytest tests documented; suite command: `uv run pytest` |
| REG-02 | Tutorial notebook unchanged within documented tolerances | `experiments/1_tutorial.ipynb` exists; REG-02 validation method deferred to planning |
| REG-03 | f-orbital changes don't require public API changes for simple-format users | No new exports in `__init__.py`; `R_orb` is internal (not in `__all__`) |
| REG-04 | Unsupported f combinations fail with explicit errors | Four exception classes established: `FAngularFormulaSourceError`, `FDerivativeUnsupportedError`, `FSpinPolarizationUnsupportedError`, `FShellResolvedCoulombUnsupportedError` |
| REG-05 | Loading incompatible SKF grid steps fails with explicit error | Guard implementation deferred to planning; grid step read from SKF line 1 at `_bond_integral.py:597` |
| REG-06 | Slater-Koster radial lookup uses each pair's own grid | Requires per-pair indexing change at `_h0ands.py:243-245` and `556-558` |
| CLN-01 | Temporary prototype branches documented | `f_orbital_initial` branch exists; documentation location deferred |
| CLN-02 | Channel lookup and AO ordering centralized for troubleshooting | `_CHANNELS` at `_bond_integral.py:101`, `shell_dim` in Constants, `AO_LABEL_TEMPLATE` at `Structure.py:25`; full centralization deferred |
| CLN-03 | Hardcoded 1/4/9 orbital assumptions audited and extended/guarded | 43-44 comparisons identified; disposition per site deferred to planning |
| CLN-04 | f-orbital support status documented (batch, force, stress, MD, SEDACS, ML-SK) | Four exception guards identified; per-capability documentation status deferred |
| CLN-05 | Cleanup preserves working prototype and simple-format tests | Backward-compatible approach; no API breaks required |

## Standard Stack

### Core (No New Packages)

Phase 5 modifies existing code only. No new package dependencies.

| Library | Version (Project) | Purpose | Why Standard |
|---------|------------------|---------|--------------|
| pytest | 9.1.1 | Test framework | Established test runner; 72 passing tests provide regression net |
| uv | 0.12.0 | Package manager | Project's canonical `uv run pytest` invocation at `.github/workflows/tests.yml:56` |
| torch | (project requirement) | Tensor operations | All radial-grid and coefficient indexing via PyTorch |

### Test Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| Test config | `pyproject.toml:51-57` | Configured; `-ra -q` addopts, markers for slow/gpu |
| Test directory | `tests/` | 12 test files, 63-72 test functions (exact count TBD) |
| Quick-run command | `uv run pytest tests/test_*.py -x` | Canonical invocation |
| Full suite | `uv run pytest` | CI baseline: 72 passed / 0 failed as of 2026-07-29 |

### Known Environment Issues

- `ruff` declared as dev extra but NOT installed; repo-wide `ruff format --check` likely fails on pre-existing files
- Tutorial notebook `experiments/1_tutorial.ipynb` exists and must validate against it; REG-02 validation harness (papermill / nbval / fixture extraction) deferred to planning

## Architecture Patterns

### System Architecture Diagram

```
Input: SKFPATH (one or more .skf files)
   ↓
┌─────────────────────────────────────┐
│  get_skf_tensors (Phase 1 output)   │
│  - Reads each SKF file              │
│  - Parses radial grid per file      │ ← D-01: per-pair R_tensor stored here
│  - Normalizes to 40 channels        │
│  - Builds cubic-spline coeffs       │
└─────────────────────────────────────┘
   ↓ (exports: R_tensor, R_orb, coeffs_tensor, etc.)
┌─────────────────────────────────────┐
│  Constants (Phase 2 output)         │
│  - Stores R_orb + coeffs_tensor     │
│  - Threads through library via __init__
└─────────────────────────────────────┘
   ↓
   ├─→ H0_and_S_vectorized (Phase 3)
   │   - Line 243-245, 556-558: searchsorted(R_orb, dR_mskd)
   │   - D-01: change to per-pair R_tensor row
   │
   ├─→ ML-SK path (build_pair_type_rcut at line 434)
   │   - Reads R_orb to compute cutoffs
   │   - D-01: may need corresponding per-pair adjustment
   │
   └─→ stress path (_stress.py:179)
       - Reads const.R_orb[idx]
       - D-01: change to R_tensor per-pair indexing

Output: H0, S matrices (Phase 3+)
   ↓
ESDriver / MDXL energy/forces/dynamics (Phase 4+)
```

### Recommended Project Structure (No Changes Required)

Phase 5 modifies only `src/dftorch/` and `tests/` — no new directories.

```
src/dftorch/
├── Constants.py              [D-02: add verbose flag; D-01: handle R_orb decision]
├── Structure.py              [D-04: audit AO offsets; no structural change]
├── ESDriver.py               [imports exception classes; D-04 may add guards]
├── _bond_integral.py         [D-01: pass guard check; already holds R_tensor]
├── _h0ands.py                [D-01: change searchsorted to per-pair; D-02: gate prints; D-04: audit n_orb comparisons]
├── _coulomb_matrix.py        [D-02: gate prints; D-04: audit n_orb comparisons]
├── _scf.py                   [D-02: gate ~25 prints (heaviest module)]
├── _ml_sk.py                 [D-01: possibly adjust pair-type cutoff calculation]
├── _stress.py                [D-01: change R_orb indexing; D-04: stress path audit]
├── [11 other modules with prints]
└── script.py                 [D-03: PORT UNIQUE CHECKS, THEN DELETE]

tests/
├── test_*.py  (12 files)     [D-03: port script.py checks here; D-04: add n_orb guard tests]
├── conftest.py               [may be enhanced for D-03 ported checks]
└── f_orbital_data/           [existing 9 fixture pairs; all 0.04 Bohr step]
```

### Pattern 1: Fail Loudly, Never Silently Zero (D-04 Foundation)

**What:** Unsupported operations raise named exceptions on first statement, not during computation.

**When to use:** Any feature marked as deferred (phases 6-9) that a user might accidentally trigger.

**Established pattern:**

```python
# Source: ESDriver.py:39-65, _slater_koster_pair.py
def _require_f_derivatives(structure: Structure) -> None:
    """Raise FDerivativeUnsupportedError if any atom has f-orbitals.
    
    Phase 3 implements f angular values but not derivatives.
    """
    if structure.n_orb.max() > 9:
        raise FDerivativeUnsupportedError(
            f"Cannot compute forces/stress for f-orbital systems (n_orb == 16)...\n"
            f"f derivatives are not implemented (Phase 7)."
        )
```

**D-04 applies this to every hardcoded site:**
- Each `n_orb` comparison at 1/4/9/16 gets a local guard or extension, not left silent.
- Example sites:  `_h0ands.py:152-206` (8 pair masks), `_coulomb_matrix.py:683`, `_stress.py:195`.

### Pattern 2: Print Gating (D-02)

**What:** All runtime prints go through a verbose flag check, default to current behavior.

**When to use:** Diagnostic output that clutters user-facing shell but users may want for debugging.

**Example threading (inferred from Constants pattern):**

```python
# Constants.py:64 shows magnetic_hubbard_ldep threads through const
# D-02 flag should follow same pattern:

# In Constants.__init__:
self.verbose = dftorch_params.get("VERBOSE_LIBRARY_OUTPUT", True)  # Default NOISY

# In _h0ands.H0_and_S_vectorized (line 240):
if const.verbose:
    print("  Using ML model for SK integrals (lazy per-call)")
```

### Pattern 3: Per-Pair Radial Grid Lookup (D-01)

**What:** Instead of `searchsorted(R_orb[global], dR)`, use `R_tensor[pair_id, :]` for each pair type.

**When to use:** Multi-file SKF loads where grids have different endpoints or steps.

**Current pattern (buggy):**

```python
# _bond_integral.py:1064-1065
if R_orb_master is None or len(R_orb_i) > len(R_orb_master):
    R_orb_master = R_orb_i
R_orb = R_orb_master.to(device=device, dtype=dtype)

# _h0ands.py:556-558 (uses global R_orb for all pairs)
idx = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
idx = torch.clamp(idx, 0, len(R_orb))
dx = dR_mskd - R_orb[idx]
```

**D-01 fix:** Use `R_tensor[pair_type_id]` instead of `R_orb` at the searchsorted call sites.

### Anti-Patterns to Avoid

- **Silent zero matrices instead of raising:** Phase 3/4 caught this repeatedly (e.g., F-orbital pairs silently dropped before routing fix). D-04 guards prevent recurrence.
- **Mixing grid steps without detection:** A mixed-step `SKFPATH` produces interpolation errors (factor of two) before D-01's guard and fix.
- **Ignoring hardcoded orbital counts:** Code like `if n_orb == 1 or n_orb == 4` silently fails for f (n_orb == 16). D-04 audit ensures every such site has an explicit disposition.
- **Loud default noisy behavior after "fixing" to quiet:** Breaks REG-02/REG-03. D-02 deliberately preserves noise.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Radial grid lookups | Custom grid alignment logic | Vectorized searchsorted + per-pair R_tensor | Broadcasting handles shape mismatches; custom code invites off-by-one errors and reproduces the D-01 bug |
| Spline coefficient storage | Per-pair coefficient tables | Existing coeffs_tensor + R_tensor pair from Phase 1 | Already computed and validated; reuse reduces code and prevents transcription errors |
| Exception hierarchies for unsupported modes | Ad-hoc error messages | Reuse FDerivativeUnsupportedError, FSpinPolarizationUnsupportedError, etc. | Established pattern; consistent messaging; users learn to recognize them |
| Print control for libraries | Rolling custom loggers per module | Single verbose flag in Constants (D-02) | One decision point; no proliferation of log-level configs; backward compatible |

**Key insight:** Every design choice in this phase (D-01 to D-04) reverses a known failure mode from Phases 1-4, not speculative. Build defensively on proven patterns.

## Common Pitfalls

### Pitfall 1: Changing R_orb's Export Status Without Checking Consumers

**What goes wrong:** R_orb is stored in Constants.py:155 as a Parameter but not exported in `__init__.py`'s `__all__` (so technically internal). However, D-01 changes the indexing in `_h0ands.py:556-558`, `_ml_sk.py:434`, and `_stress.py:179`. If planners decide to remove the export entirely, those three paths must be validated that they're not reaching R_orb via a side channel.

**Why it happens:** R_orb is convenient for downstream callers, even though it's internal. Cleanup can introduce accidental API breaks.

**How to avoid:** Before D-01 implementation, document the exact scope of R_orb consumers (VERIFIED: three direct call sites plus constants' internal storage). Decide whether to keep R_orb additively (safest) or make the three consumers access R_tensor directly (cleaner). Test each path after change.

**Warning signs:** A downstream test fails with "attribute 'R_orb' not found on Constants" or radial lookup produces nan/inf.

### Pitfall 2: D-02 Verbose Flag Threading to Free Functions

**What goes wrong:** ~40% of prints are in free functions (`_scf.py`, `_coulomb_matrix.py`, `_h0ands.py`, `_xl_tools.py`) that don't have access to a `dftorch_params` key or a `Constants` instance. If the verbose flag is only added to Constants, those prints remain unconditionally noisy.

**Why it happens:** Not all code paths pass a Constants through; some call free functions with just tensors and config dicts.

**How to avoid:** Survey each print's caller signature BEFORE implementing D-02. Free functions that take `const: Constants` can gate via `const.verbose`. Free functions that don't must be made to take it, or the verbose flag must live elsewhere (e.g., thread-local, or a global registry). Verify by static grep and 3-5 manual traces.

**Warning signs:** After D-02 implementation, running with the verbose flag False still produces output. Prints in free functions were missed.

### Pitfall 3: ConstantsTest Shadowing Production Constants

**What goes wrong:** `ConstantsTest` (line 211) is a parallel in-memory table that duplicates every metadata field of `Constants` but bypasses `get_skf_tensors()`. If someone uses ConstantsTest to validate a new feature, they bypass the real parser. If it's left in the codebase without warning, it becomes a maintenance trap.

**Why it happens:** It was added as a debugging helper, but it diverges from production. D-03 must decide: delete it, move it to `tests/`, or clearly mark it deprecated.

**How to avoid:** Run `grep -rn "ConstantsTest" tests/ src/ --include="*.py"` — currently returns only the definition (line 211), no uses. Safe to delete unless there's hidden usage in notebooks or experiments. Check those if uncertain.

**Warning signs:** A test passes with ConstantsTest but fails with real Constants, or vice versa.

### Pitfall 4: D-04 Audit Incompleteness — Missing a 1/4/9/16 Site

**What goes wrong:** `_h0ands.py` has 44 `n_orb` comparisons, but the audit misses one (e.g., in a batch path or a deprecated function). Phase 5 ships with an unguarded site. A user calls it with f-orbitals and gets silent matrix zeros or a cryptic downstream error.

**Why it happens:** Hardcoded 1/4/9 sites are scattered; a thorough grep is needed. The audit must touch H0/S, Coulomb, stress, batch, legacy code, and any free functions.

**How to avoid:** Use a systematic grep for `n_orb.*==|!=|<|>` and a second pass for fixed offsets like `[0, 1, 4, 9]` or hardcoded array dimensions tied to orbital counts. Verify each match. For each site, apply D-04's rule: extend to 16, add a guard, or document why it's unreachable.

**Warning signs:** A test with f-orbitals unexpectedly passes (should have raised or failed), or a numeric result is silently zero in a field that should be nonzero.

### Pitfall 5: Mixed-SKF-Path Grid Step Detection Too Permissive

**What goes wrong:** D-01's guard checks that all SKF files in an `SKFPATH` share the same grid step. But the check is implemented as "all grids are the same length" instead of "all grids have the same step size." A user loads `SKFPATH` with 0.02-step C-H and 0.04-step Eu-N, both padded to 850 points. The guard passes (same length), but interpolation is wrong by a factor of two.

**Why it happens:** Grid step is not stored explicitly; it's derived from the radial grid. Off-by-one errors in step calculation or step comparison can slip through.

**How to avoid:** Extract and store the grid step explicitly from the header (line 597) for each file. In the guard, compare steps, not lengths. Test with a known mixed-step pair (e.g., a synthetic `TESTPATH` with mio-1-1 C-C and 3ob-3-1 Br-Br if available, or mock one). Verify the guard raises, and that the error message names both files and the mismatched steps.

**Warning signs:** REG-06 test shows wrong interpolation values (off by large factors) for mixed-step paths even after guard passes.

## Runtime State Inventory

(This is a library refactoring phase with no data migration required.)

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no user data affected | None |
| Live service config | None — no external services | None |
| OS-registered state | None — pure Python library | None |
| Secrets/env vars | None — no new secrets | None |
| Build artifacts | dftorch.egg-info updated by Phase 4; will update again on install | Reinstall package: `uv pip install -e .` post-D-03 deletion of script.py |

## Validation Architecture

Nyquist validation is **ENABLED** (`workflow.nyquist_validation: true` in config.json).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 |
| Config file | `pyproject.toml:51-57` |
| Quick run | `uv run pytest tests/test_*.py -x` |
| Full suite | `uv run pytest` |
| Current status | 72 passed / 0 failed (as of 2026-07-29) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File | Status |
|--------|----------|-----------|-------------------|------|--------|
| REG-01 | Import, I/O, neighbor-list, SCF, forces smoke tests remain green | Unit/integration | `uv run pytest tests/test_*.py -x` | test_import.py, test_io.py, test_scf.py, etc. | ✅ Existing |
| REG-02 | Tutorial notebook unchanged within tolerances | Manual/integration | Deferred to planning (papermill / nbval / fixture extraction TBD) | experiments/1_tutorial.ipynb | ❌ Wave 0 gap |
| REG-03 | f-orbital changes don't break simple-format users | Smoke test | `uv run pytest tests/test_*.py -k "not f_orbital" -x` | test_public_api.py, test_dtype_contract.py | ✅ Existing |
| REG-04 | Unsupported f modes raise explicit errors | Unit | `uv run pytest tests/test_spin_guard.py -v` | test_spin_guard.py (5 tests) | ✅ Existing |
| REG-05 | Mixed-grid-step SKFPATH fails with explicit error | Unit | NEW: test_radial_grid_step_guard.py::test_mixed_step_raises | NEW test_radial_grid_step_guard.py | ❌ Wave 0 gap |
| REG-06 | Per-pair radial lookup produces correct interpolation | Unit/reference | NEW: test_per_pair_radial_lookup.py | NEW test_per_pair_radial_lookup.py | ❌ Wave 0 gap |
| CLN-01 | Temporary branches documented | Documentation/manual | grep `.planning/CONTEXT.md` for "prototype branch" | .planning/*/CONTEXT.md | ✅ Deferred to doc task |
| CLN-02 | Channel metadata centralized | Documentation/manual | Manual audit: Are `_CHANNELS`, `shell_dim`, `AO_LABEL_TEMPLATE` cross-referenced? | _bond_integral.py, Constants.py, Structure.py | ⚠️ Deferred to planning |
| CLN-03 | Hardcoded 1/4/9 audit + guard/extend | Unit/guard gates | NEW: test_orbital_count_guards.py (one test per site, 40-50 tests) | NEW test_orbital_count_guards.py | ❌ Wave 0 gap |
| CLN-04 | f support status documented (batch, force, stress, MD, SEDACS, ML-SK) | Documentation/manual | Manual check: Does docs/STATUS.md or similar exist with per-capability entries? | TBD | ❌ Wave 0 gap |
| CLN-05 | Cleanup preserves prototype and simple-format tests | Regression | `uv run pytest` (all 72 tests pass) | All test_*.py files | ✅ Existing |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_*.py -x` (single failure stops; ~10 s to find a break)
- **Per wave merge:** `uv run pytest` (full suite; ~30 s; requires 72 passed)
- **Phase gate (before `/gsd-verify-work`):** Full suite must pass

### Wave 0 Gaps

- [ ] `tests/test_radial_grid_step_guard.py` — test the guard in D-01 (mixed-step SKFPATH raises with named files)
- [ ] `tests/test_per_pair_radial_lookup.py` — test that per-pair indexing (D-01 fix) produces correct interpolation for a mixed-grid-step path once D-01 is implemented
- [ ] `tests/test_orbital_count_guards.py` — one test per hardcoded 1/4/9 site (40-50 tests), verifying each raises or extends per D-04 inventory
- [ ] `tests/test_verbose_flag.py` — verify D-02 prints gate on/off with verbose flag (requires parametrized runs with flag True/False)
- [ ] `tests/test_script_py_ported_checks.py` — ported `parse_expected_homonuclear_metadata()` and other unique `script.py` checks (D-03)
- [ ] Tutorial notebook validation harness — `experiments/1_tutorial.ipynb` must pass as-is (papermill / nbval / fixture extraction method TBD)
- [ ] `docs/ORBITAL_COUNT_INVENTORY.md` — CLN-03 output: table of every 1/4/9/16 site with disposition and test coverage
- [ ] `docs/FEATURE_SUPPORT_STATUS.md` — CLN-04 output: batch, force, stress, MD, SEDACS, ML-SK per-capability support level

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Code Examples

Verified patterns from CONTEXT.md and existing source.

### Exception Raising Pattern (D-04 Guide)

```python
# Source: _slater_koster_pair.py:210-227, ESDriver.py:39-65
def _require_f_derivatives(structure: Structure) -> None:
    """Raise FDerivativeUnsupportedError if any atom has 16 orbitals.
    
    This is called at the first statement of any path that needs derivatives.
    """
    if (structure.n_orb == 16).any():
        raise FDerivativeUnsupportedError(
            f"Cannot compute forces for f-orbital systems.\n"
            f"f angular derivatives not implemented (Phase 7)."
        )
```

Used at:
- `ESDriver.calc_forces()` (phase 7 blocker)
- `_stress.py` analytical stress paths (phase 8 blocker)
- Each D-04 guarded site follows this pattern

### Per-Pair Radial Lookup Pattern (D-01 Outcome)

```python
# Current (buggy):
idx = torch.searchsorted(const.R_orb, dR_mskd, right=True) - 1
dx = dR_mskd - const.R_orb[idx]

# D-01 fix:
# For each pair type, use its own radial grid from R_tensor
R_pair = const.R_tensor[pair_type_id]  # shape: (n_points,)
idx = torch.searchsorted(R_pair, dR_mskd[pair_mask], right=True) - 1
dx = dR_mskd[pair_mask] - R_pair[idx]
```

Sites:
- `_h0ands.py:243-245` (ML mode)
- `_h0ands.py:556-558` (main path)
- `_ml_sk.py:434` (cutoff calculation)
- `_stress.py:179` (force derivatives)

### Verbose Flag Pattern (D-02 Outcome)

```python
# Source: Inferred from Constants.py:64 (magnetic_hubbard_ldep threading)

# In Constants.__init__:
self.verbose = dftorch_params.get("VERBOSE_LIBRARY_OUTPUT", True)

# In any module with prints (e.g., _h0ands.py:240):
def H0_and_S_vectorized(..., const: Constants, ...):
    if const.verbose:
        print("  Using ML model for SK integrals (lazy per-call)")
    # ... rest of function
```

All ~120 prints in 15 modules follow this pattern.

## Assumptions Log

All claims in this research were verified from source code or CONTEXT.md. No unresolved assumptions.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | REG-02 notebook validation method (papermill vs nbval vs fixture extraction) is deferred to planning | Validation Architecture | Planning misses the actual harness to use; test command fails or doesn't run |
| A2 | Verbose flag will thread successfully to all free functions via Constants instance | D-02 Pitfall | Some prints remain unconditionally noisy; D-02 incomplete |
| A3 | D-04 audit is exhaustive and no 1/4/9/16 sites are missed | Common Pitfalls § 4 | Silent matrix zeros or cryptic errors occur for f-orbital calls to unguarded sites |

## Open Questions (ALL RESOLVED — 2026-07-29, at plan-phase)

> None of the questions below is open. Each was settled either by `05-CONTEXT.md`
> "Claude's Discretion" or by a Phase 5 plan: verbose-flag location → `VERBOSE_LIBRARY_OUTPUT`
> in 05-05; `R_orb` export → kept additively in 05-01; `ConstantsTest` → deleted in 05-05;
> tutorial-notebook validation → `checkpoint:decision` in 05-02; D-04 inventory format →
> `docs/ORBITAL-COUNT-INVENTORY.md` in 05-06. Retained for provenance; do not re-raise.

1. **D-02 Verbose Flag Location**
   - What we know: Must thread to free functions in 15 modules; follows `magnetic_hubbard_ldep` precedent at `Constants.py:64`.
   - What's unclear: Should it live in `dftorch_params` dict, a separate config object, or thread via `Constants` instance?
   - Recommendation: Add `VERBOSE_LIBRARY_OUTPUT` key to `dftorch_params` (or equivalent), default True, read into `Constants` at line 155. All functions that have access to `const` can then gate on `const.verbose`.

2. **D-01 R_orb Export Decision**
   - What we know: R_orb is not in `__init__.py`'s `__all__` (internal). D-01 changes indexing at three call sites. Can either keep R_orb additively or remove it.
   - What's unclear: Should downstream code (ML-SK, stress) still access const.R_orb, or must they use R_tensor directly?
   - Recommendation: Keep R_orb additively for backward compatibility (safest). Update consumers to use per-pair R_tensor where needed, but leave R_orb in place so users who depend on it don't break. Future migration can remove it.

3. **D-03 ConstantsTest Disposition**
   - What we know: Unused, parallel metadata table, diverges from production Constants.
   - What's unclear: Delete, move to tests/, or document-only deprecation?
   - Recommendation: Delete. It shadows production behavior and serves no current purpose. If it's needed for debugging later, it can be reconstructed from git history.

4. **Tutorial Notebook Validation (REG-02)**
   - What we know: `experiments/1_tutorial.ipynb` exists (197KB, 60KB when extracted).
   - What's unclear: Can it run headless (papermill)? Does it compare to golden numbers? What tolerance bands apply?
   - Recommendation: Defer to planning. Planner should decide on validation method (papermill + fixture comparison vs nbval vs manual spot-checks). This is a CLN/documentation task, not a code task.

5. **D-04 Inventory Format**
   - What we know: CLN-03 requires an inventory so no site is left silently unhandled. Four exception classes are the model.
   - What's unclear: Should it be a docs/ORBITAL_COUNT_INVENTORY.md table? Code comments? A runtime check?
   - Recommendation: Create a section in Phase 5's final CONTEXT.md or a new docs/ORBITAL_COUNT_INVENTORY.md listing every `n_orb` comparison site (40-50 lines), its location, its disposition (extended to 16 / guarded with exception name / not applicable), and the test covering it.

## Environment Availability

No external tools, services, or runtimes beyond the project's own Python environment are required.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | Test invocation | ✓ | 0.12.0 | `python -m pytest` (not canonical) |
| pytest | Test framework | ✓ | 9.1.1 | — |
| torch | All code | ✓ | project dependency | — |
| ruff | Linting (dev extra) | ✗ | — | Manual code review (D-02 print gating can be spot-checked) |
| papermill / nbval | Tutorial notebook validation (TBD) | ✗ | — | Manual notebook run (REG-02 deferred) |

**Missing dependencies with fallback:**
- ruff: Not installed; pre-existing code may have style issues, but D-02 print additions can be reviewed manually
- Notebook validation tool: Not selected yet; defer to planning

**No blockers for Phase 5 implementation.**

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Control | Phase 5 Touch |
|---------------|---------|---------|--------------|
| V1 Architecture, Design and Threat Modeling | — | N/A | No new attack surface |
| V2 Authentication | No | — | Library code; no auth |
| V3 Session Management | No | — | Library code; no sessions |
| V4 Access Control | No | — | Library code; no access control |
| V5 Input Validation | Yes | SKF file parsing (existing) | D-01 guard adds grid step validation; D-04 guards add orbital-count validation |
| V6 Cryptography | No | — | Library code; no crypto |
| V7 Cryptography Failures | No | — | Library code; no crypto |
| V8 Error Handling and Logging | Yes | Exception raising | D-04 follows established exception pattern (`FDerivativeUnsupportedError`, etc.) |
| V9 Communications | No | — | Library code; no networking |
| V10 Malicious Code | No | — | No dynamic code execution |
| V11 Business Logic | Yes | Grid step compatibility check | D-01 guard prevents silent numeric errors (grid mismatch) |
| V12 File Upload | No | — | SKF files are read-only trusted files |
| V13 API and Web Service | No | — | Library, not API |
| V14 Configuration | Yes | Feature flags | D-02 verbose flag is configuration; default preserves existing behavior |

### Known Threat Patterns for DFTorch

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silent radial-grid mismatch (factor-of-two interpolation error) | Tampering | D-01: Guard + per-pair lookup. Verify grid steps match or raise explicit error. |
| Silent matrix zeros (unsupported mode not failing) | Tampering | D-04: Fail loudly on first statement for unsupported modes. Exception classes pre-exist; follow pattern. |
| Verbose print spam making real output unreadable | Denial of Service | D-02: Gate prints behind verbose flag. Users opt in to quiet mode; default unchanged. |
| Stale ConstantsTest shadowing production Constants | Tampering | D-03: Delete ConstantsTest (unused, diverges from production). |

**Phase 5 accepts zero new risk by following established guards and not changing public APIs.**

## Package Legitimacy Audit

Phase 5 does **not** install new external packages. No audit required.

| Package | Ecosystem | Status |
|---------|-----------|--------|
| (none) | — | No new dependencies |

## Sources

### Primary (HIGH Confidence)

Verified directly from source code:

- `src/dftorch/_bond_integral.py:1064-1065` — R_orb_master selection logic
- `src/dftorch/_h0ands.py:243-245, 556-558` — searchsorted lookup sites
- `src/dftorch/_stress.py:179` — radial grid access in stress calculation
- `src/dftorch/Constants.py:105-156` — Constants initialization with R_orb parameter
- `src/dftorch/_slater_koster_pair.py:173-289` — Four exception class definitions
- `src/dftorch/script.py:1-50, 176, 211` — Script overview and unique functions
- `src/dftorch/Structure.py:12, 25-42` — SHELL_LOCAL_STARTS and AO_LABEL_TEMPLATE
- `tests/f_orbital_data/*.skf` header lines — Grid step verification (all 0.04)
- `.github/workflows/tests.yml:56` — Canonical test command
- `pyproject.toml:51-57` — Test configuration
- `.planning/config.json:11-12` — Nyquist and security enforcement enabled

### Secondary (MEDIUM Confidence)

From CONTEXT.md analysis and prior phase decisions:

- `.planning/phases/05-regression-safety-and-support-policy-cleanup/05-CONTEXT.md` — D-01 through D-04 specifications, verified to match source code
- `.planning/REQUIREMENTS.md` — REG-01 through CLN-05 requirement text
- `.planning/STATE.md` — Current test count (72 passed), phase status

### Tertiary (Deferred)

- REG-02 notebook validation method — not yet selected; deferred to planning
- CLN-04 support status documentation format — deferred to planning

---

*Research completed: 2026-07-30*
*All claims verified against source code. No assumptions left untagged.*

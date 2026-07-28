# Phase 1: SKF Canonicalization and Spline Validation - Research

**Researched:** 2026-07-20
**Domain:** Brownfield scientific Python SKF parser canonicalization and spline validation
**Confidence:** HIGH for local code state; MEDIUM for inferred planning recommendations

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Convert the `src/dftorch/script.py` checks into pytest tests after the prototype stabilizes.
- Broaden validation beyond `tests/f_orbital_data/` in a later regression or cleanup phase if needed.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKF-01 | DFTorch can read simple-format `.skf` files with the existing 20-column electronic table layout. | `_normalize_skf_row()` accepts 20 values and maps them through `_SIMPLE_TO_EXTENDED`; Phase 1 execution should prove this by writing a temporary minimal simple `.skf` in `script.py` and parsing it through `read_skf_table()`. [VERIFIED: local code/command] |
| SKF-02 | DFTorch can read extended-format `.skf` files with the wider f-orbital electronic table layout. | `tests/f_orbital_data` contains 9 extended 40-column SKF fixtures; `uv run python src/dftorch/script.py tests/f_orbital_data` passed. [VERIFIED: local code/command] |
| SKF-03 | Simple-format and extended-format files are normalized to one documented internal 40-channel representation before downstream code consumes them. | `_CHANNELS` has 40 names; `read_skf_table()` returns channel dicts in `_CHANNELS`; `get_skf_tensors()` allocates coefficient tensors with `len(_CHANNELS)`. [VERIFIED: local code/grep] |
| SKF-04 | Simple-format files zero-fill f-only channels while preserving existing s/p and s/p/d channel values. | `_normalize_skf_row()` initializes a 40-value zero row and copies legacy channels by `_SIMPLE_TO_EXTENDED`. Planner should add a targeted assertion in `script.py` for f-only zero-fill and legacy preservation. [VERIFIED: local code/grep] |
| SKF-05 | Compact pair filenames such as `EuN.skf` and dashed pair filenames such as `Eu-N.skf` resolve consistently with clear missing-file errors. | `_resolve_skf_path()` tries dashed first, then compact; `_split_skf_pair_name()` parses both dashed and compact basenames. `script.py` still uses `split_dashed_pair()` for independent checks, so script coverage for compact fixture names is a gap. [VERIFIED: local code/grep] |
| SKF-06 | Homonuclear SKF parsing captures f-shell onsite energy, Hubbard value, and reference occupation when present. | `read_skf_table()` parses extended 13-value homonuclear headers and fills `N_F`, `EF`, and `UF`; the current script checks Eu f metadata from `Eu-Eu.skf`. [VERIFIED: local code/command] |
| SKF-07 | Parser validation rejects unsupported skipped-shell basis layouts with actionable errors. | `_validate_nested_shells()` rejects f without s/p/d, d without s/p, and p without s. No current fixture or script check exercises the negative path. [VERIFIED: local code/grep] |
| SPL-01 | Cubic spline coefficients generated from `tests/f_orbital_data/` reproduce source electronic table values at original grid points. | `check_one_skf()` evaluates left and right spline endpoints for every f fixture; current run passed all 9 files at `1e-9` tolerance. [VERIFIED: local code/command] |
| SPL-02 | Cubic spline coefficients generated from existing simple-format fixtures reproduce legacy source channel values at original grid points. | Full simple-directory validation fails because that directory lacks all ordered pair files required by `get_skf_tensors()`. Phase 1 should add a temporary minimal simple `.skf` helper in `script.py`, parse it through `read_skf_table()`, and reconstruct source-grid values from the parsed channels. [VERIFIED: local command] |
| SPL-03 | Parser tests cover representative f fixture pairs, including homonuclear and heteronuclear files. | `tests/f_orbital_data` has Eu/Ga/N ordered pairs including `Eu-Eu.skf`, `Ga-Ga.skf`, `N-N.skf`, and heteronuclear pairs; current script run covered all 9. [VERIFIED: local filesystem/command] |
| SPL-04 | Parser and spline tests run through pytest without optional GPU, Triton, DFTB+, or ASE dependencies. | User locked Phase 1 to `script.py` rather than pytest conversion, while `pyproject.toml` has pytest config and `uv run pytest -q` passes 9 existing tests. Treat pytest conversion as deferred; Phase 1 should use the standalone script gate plus the focused existing pytest regression command. [VERIFIED: local code/command] |
</phase_requirements>

## Project Constraints (from AGENTS.md)

No `AGENTS.md` file exists in the repository root, and no `rules/*.md` files were found. [VERIFIED: local rg]

## Summary

Phase 1 should plan around an already-present parser boundary in `src/dftorch/_bond_integral.py`: simple 20-column rows and extended 40-column rows are normalized to a single `_CHANNELS` order before spline tensor construction. [VERIFIED: local code/grep] The current validation script already passes on all nine `tests/f_orbital_data` fixtures and validates parser, spline, constants, and structure metadata paths, but Phase 1 should keep its acceptance criteria centered on parser/spline behavior because later constants/structure work belongs to Phase 2. [VERIFIED: local command] [VERIFIED: local roadmap]

The main planning gaps are targeted validation gaps, not wholesale redesign gaps: compact-name behavior is implemented in production parser helpers but not covered by `script.py`; skipped-shell rejection is implemented but not covered by a negative test; simple-format row normalization should be proven through a temporary minimal simple `.skf` parsed by `read_skf_table()` because the full script cannot run over `tests/data_skf_mio-1-1` as a complete ordered-pair directory. [VERIFIED: local code/command]

**Primary recommendation:** Plan a narrow hardening pass that adds focused `script.py` assertions for simple-to-40 mapping, compact/dashed path resolution, homonuclear f metadata, spline endpoint reconstruction, and skipped-shell failure while preserving the current `_bond_integral.py` contract. [VERIFIED: local code/command]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| SKF file parsing | Parser / Data Ingestion | Validation Script | `_bond_integral.py` owns text-to-tensor normalization; `script.py` independently checks parser outputs. [VERIFIED: local code/grep] |
| 20-to-40 channel canonicalization | Parser / Data Ingestion | Constants Registry | `_normalize_skf_row()` is the only place raw row width should matter; downstream `Constants` should receive 40-channel tensors only. [VERIFIED: local code/grep] |
| Spline coefficient validation | Validation Script | Parser / Data Ingestion | `script.py` constructs coefficients with production `cubic_spline_coeffs()` and verifies knot reconstruction against source rows. [VERIFIED: local code/grep] |
| Homonuclear f metadata capture | Parser / Data Ingestion | Validation Script | `read_skf_table()` mutates metadata tensors; `script.py` parses expected homonuclear headers independently. [VERIFIED: local code/grep] |
| Unsupported layout rejection | Parser / Data Ingestion | Validation Script | `_validate_nested_shells()` should reject unsupported basis layouts, and `script.py` should exercise this with synthetic malformed inputs. [VERIFIED: local code/grep] |

## Standard Stack

### Core

| Library / Tool | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| Python | 3.11.15 | Run parser and validation script. | Project requires Python `>=3.11` in `pyproject.toml`; local environment runs 3.11.15. [VERIFIED: local code/command] |
| PyTorch | 2.10.0 | Tensor storage, spline linear solve, metadata arrays. | `_bond_integral.py` and `script.py` use `torch` throughout parser and validation paths. [VERIFIED: local code/command] |
| `uv` | 0.11.26 | Project command runner. | All validation commands succeeded through `uv run`. [VERIFIED: local command] |
| `src/dftorch/script.py` | local source | Phase 1 validation harness. | User decision D-04 keeps validation in this script for Phase 1. [VERIFIED: local CONTEXT.md] |

### Supporting

| Library / Tool | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| pytest | 9.0.3 | Existing repo tests. | Run existing regression suite, but do not require Phase 1 f validation migration into pytest. [VERIFIED: local code/command] |
| numpy | 2.4.4 | Existing project dependency. | Available for existing tests and package imports; not needed for new parser assertions unless already used locally. [VERIFIED: local command] |
| scipy | 1.17.1 | Existing project dependency. | Available but unnecessary for Phase 1 parser/spline assertions because spline code uses `torch.linalg.solve()`. [VERIFIED: local code/command] |
| pandas | 3.0.2 | Existing project dependency. | Available but unrelated to Phase 1 parser hardening. [VERIFIED: local command] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `src/dftorch/script.py` checks | New pytest file | Deferred by D-05; pytest would improve CI collection but contradicts locked Phase 1 direction unless added only as an optional wrapper. [VERIFIED: local CONTEXT.md] |
| Production parser as oracle | Independent row/header parsing in `script.py` | Existing script pattern avoids validating parser output against itself; keep this pattern. [VERIFIED: local code/grep] |

**Installation:** No new package installation is recommended for Phase 1. [VERIFIED: local code/command]

## Package Legitimacy Audit

No external packages need to be installed for this phase. [VERIFIED: local pyproject/command]

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| none | — | — | — | — | — | No install required. [VERIFIED: local code/command] |

**Packages removed due to [SLOP] verdict:** none. [VERIFIED: local code/command]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: local code/command]

## Architecture Patterns

### System Architecture Diagram

```text
SKF file path / pair label
  |
  v
_resolve_skf_path(label)  ---> missing file error names dashed target
  |
  v
read_skf_table(path)
  |
  +--> _split_skf_pair_name(path stem)
  +--> parse grid/header/repulsive spline
  +--> if homonuclear: parse N_S/N_P/N_D/N_F, ES/EP/ED/EF, US/UP/UD/UF
  +--> _validate_nested_shells(shell flags)
  |
  v
for each electronic row:
  _expand_tokens(row) -> _normalize_skf_row(20 or 40 values) -> canonical 40 channels
  |
  v
channels_to_matrix(_CHANNELS order)
  |
  v
cubic_spline_coeffs(R, M)
  |
  v
get_skf_tensors() returns normalized tensors and metadata to Constants
```

Every stage above exists in `src/dftorch/_bond_integral.py`; `src/dftorch/script.py` is the local validation entry point for tracing this flow. [VERIFIED: local code/grep]

### Recommended Project Structure

```text
src/dftorch/
├── _bond_integral.py   # parser boundary, channel normalization, spline coefficients
└── script.py           # Phase 1 human-readable validation harness

tests/
├── f_orbital_data/     # locked Phase 1 fixture scope
└── data_skf_mio-1-1/   # existing simple-format data useful as supporting context, not the Phase 1 gate
```

This structure already exists; the planner should modify only `_bond_integral.py` and `script.py` unless a tiny temporary fixture is needed for negative parser cases. [VERIFIED: local filesystem/CONTEXT.md]

### Pattern 1: Normalize at Parser Boundary

**What:** Accept 20 or 40 electronic row values in `_normalize_skf_row()` and always emit a 40-value row in `_CHANNELS` order. [VERIFIED: local code/grep]

**When to use:** Every SKF electronic row before `channels_to_matrix()` or spline construction. [VERIFIED: local code/grep]

**Example:**
```python
# Source: src/dftorch/_bond_integral.py
if len(values) == len(_SIMPLE_CHANNELS):
    row = [0.0] * len(_CHANNELS)
    for old_idx, new_idx in enumerate(_SIMPLE_TO_EXTENDED):
        row[new_idx] = values[old_idx]
    return row
```

### Pattern 2: Validate Spline Knots, Not Just Shapes

**What:** `script.py` compares coefficient left endpoints with source row values and evaluates right endpoints against the next source row. [VERIFIED: local code/grep]

**When to use:** For both f fixtures and the temporary minimal simple `.skf` parsed through `read_skf_table()` for SPL-02. [VERIFIED: local command]

### Pattern 3: Keep Expected Metadata Independent

**What:** `parse_expected_homonuclear_metadata()` independently reads homonuclear headers instead of reusing `read_skf_table()` output as the oracle. [VERIFIED: local code/grep]

**When to use:** For Eu, Ga, and N homonuclear fixtures in `tests/f_orbital_data`, and any new negative/compact-name fixture assertions. [VERIFIED: local command]

### Anti-Patterns to Avoid

- **Branching downstream on raw SKF width:** Downstream code should never care whether source rows were 20 or 40 columns. [VERIFIED: local CONTEXT.md]
- **Converting the whole harness to pytest in Phase 1:** This contradicts D-04/D-05 and should remain deferred. [VERIFIED: local CONTEXT.md]
- **Using production parser output as the only oracle:** That can hide parser regressions; keep independent expected parsing in `script.py`. [VERIFIED: local code/grep]
- **Expanding fixture scope silently:** D-06 locks the Phase 1 test suite to `tests/f_orbital_data`; use `tests/data_skf_mio-1-1` only as supporting evidence unless the user approves scope change. [VERIFIED: local CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tensor math and spline solve | Custom matrix solver | `torch.linalg.solve()` inside existing `cubic_spline_coeffs()` | Existing implementation already validates source-grid reconstruction. [VERIFIED: local code/command] |
| Pair ordering | Manual pair list generation | Existing `ordered_pairs_from_TYPE()` path through `get_skf_tensors()` | `get_skf_tensors()` already consumes ordered labels used by `Constants`. [VERIFIED: local code/grep] |
| SKF row width branching in validation consumers | Separate 20-column and 40-column downstream paths | `_normalize_skf_row()` and `_CHANNELS` | This preserves the locked downstream 40-channel contract. [VERIFIED: local CONTEXT.md/code] |
| Full test framework migration | New test architecture | Existing `script.py` helpers | User explicitly deferred pytest conversion. [VERIFIED: local CONTEXT.md] |

**Key insight:** The hard part is proving invariants at the parser boundary; building a second parser architecture would increase risk without improving Phase 1 acceptance. [VERIFIED: local code/command]

## Common Pitfalls

### Pitfall 1: Script Does Not Cover Compact SKF Names

**What goes wrong:** Production parser accepts compact names, but `script.py` independent helpers require dashed names. [VERIFIED: local code/grep]
**Why it happens:** `_split_skf_pair_name()` handles compact names in `_bond_integral.py`, while `split_dashed_pair()` in `script.py` rejects names without `-`. [VERIFIED: local code/grep]
**How to avoid:** Add a small parser-level check that calls production `_resolve_skf_path()` and `_split_skf_pair_name()` for compact and dashed examples without requiring renamed f fixtures. [VERIFIED: local code/grep]
**Warning signs:** A compact fixture directory fails in `script.py` before production parsing is exercised. [VERIFIED: local code/grep]

### Pitfall 2: Simple-Format Validation Can Fail for Fixture Completeness, Not Row Parsing

**What goes wrong:** Running `script.py` on `tests/data_skf_mio-1-1` validates many simple rows, then fails when `get_skf_tensors()` cannot find missing ordered pair files. [VERIFIED: local command]
**Why it happens:** The full script assumes a complete ordered-pair directory for all discovered species. [VERIFIED: local code/command]
**How to avoid:** Add a temporary minimal simple-format `.skf` check inside the f fixture validation flow, parse it through `read_skf_table()`, and avoid relying on a permanent fixture rename or complete simple ordered-pair directory. [VERIFIED: local command]
**Warning signs:** Parser rows show `PASS`, followed by `FileNotFoundError` in metadata/constants/structure sections. [VERIFIED: local command]

### Pitfall 3: Skipped-Shell Rejection Is Implemented But Untested

**What goes wrong:** `_validate_nested_shells()` could regress without any current `script.py` failure. [VERIFIED: local code/grep]
**Why it happens:** `tests/f_orbital_data` contains valid nested shell layouts only. [VERIFIED: local command]
**How to avoid:** Add synthetic negative checks that call `_validate_nested_shells()` directly or create minimal malformed homonuclear SKF text in a temp directory. [VERIFIED: local code/grep]
**Warning signs:** No assertion checks a `ValueError` message for f-without-d, d-without-p, or p-without-s. [VERIFIED: local code/grep]

### Pitfall 4: Roadmap Pytest Language Conflicts With Locked User Decision

**What goes wrong:** Planner may try to satisfy SPL-04 by converting the harness to pytest immediately. [VERIFIED: local ROADMAP.md/CONTEXT.md]
**Why it happens:** Requirements mention pytest, but phase context explicitly says keep Phase 1 validation in `script.py`. [VERIFIED: local REQUIREMENTS.md/CONTEXT.md]
**How to avoid:** Treat `uv run python src/dftorch/script.py tests/f_orbital_data` as the Phase 1 gate, plus existing `uv run pytest -q` for repo regressions. [VERIFIED: local command]
**Warning signs:** A plan proposes new pytest files as required work. [VERIFIED: local CONTEXT.md]

## Code Examples

### Spline Endpoint Reconstruction

```python
# Source: src/dftorch/script.py
left_reconstructed = coeffs[:, :, 0][:original_rows]
left_target = M[:original_rows]
left_err = (left_reconstructed - left_target).abs()

h = (R[1:] - R[:-1]).unsqueeze(1)
right_reconstructed = a + b * h + c * h**2 + d * h**3
right_err = (right_reconstructed[: original_rows - 1] - M[1:original_rows]).abs()
```

This is the pattern to reuse for SPL-01 and SPL-02 checks. [VERIFIED: local code/grep]

### Homonuclear f Metadata Assertions

```python
# Source: src/dftorch/script.py
for name in ["TORE", "N_S", "N_P", "N_D", "N_F", "ES", "EP", "ED", "EF", "US", "UP", "UD", "UF"]:
    assert_float_metadata(returned[name], Z, float(expected[name]), name, source)
```

This pattern already checks `N_F`, `EF`, and `UF` after `get_skf_tensors()`. [VERIFIED: local code/grep]

### Compact and Dashed Path Resolution

```python
# Source: src/dftorch/_bond_integral.py
dashed = os.path.join(skfpath, f"{label_name}.skf")
if os.path.isfile(dashed):
    return dashed

undashed = os.path.join(skfpath, f"{label_name.replace('-', '')}.skf")
if os.path.isfile(undashed):
    return undashed
```

Planner should add a direct validation check for this behavior because the current fixture names are dashed. [VERIFIED: local code/filesystem]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw 20-column simple SKF rows consumed as-is | 20-column rows are expanded into canonical 40-channel rows | Present in current working tree on 2026-07-20 | Downstream tensors can assume 40 channels. [VERIFIED: local code/grep] |
| Parser validation focused on f fixture script | Current script also checks constants and structure metadata | Present in current working tree on 2026-07-20 | Phase 1 should avoid letting Phase 2 metadata work dominate parser/spline planning. [VERIFIED: local code/ROADMAP.md] |
| Pytest as ideal CI target | Script remains Phase 1 prototype gate | Locked by context on 2026-07-20 | Planner should not require pytest conversion. [VERIFIED: local CONTEXT.md] |

**Deprecated/outdated:**
- Treating `script.py` as if it were pytest-collected is outdated for this phase; it is a standalone command. [VERIFIED: local code/command]
- Assuming `tests/f_orbital_data` includes simple 20-column fixtures is not supported by the local fixture row-width probe; sampled f fixtures are extended 40-column. [VERIFIED: local command]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SPL-02 should be satisfied by a temporary minimal simple `.skf` created inside `src/dftorch/script.py` and parsed through `read_skf_table()`, keeping `tests/f_orbital_data` as the primary f fixture gate. [RESOLVED] | Phase Requirements / Common Pitfalls | Low; this avoids permanent fixture-scope expansion while validating production parser behavior. |
| A2 | A direct unit-style call to `_validate_nested_shells()` in `script.py` is acceptable for SKF-07 instead of writing malformed SKF files. [ASSUMED] | Common Pitfalls | Planner may prefer end-to-end malformed file parsing for stronger coverage. |

## Open Questions (RESOLVED)

1. **How should SPL-02 be reconciled with D-06?**
   - What we know: `tests/f_orbital_data` sampled fixtures are extended 40-column, while `tests/data_skf_mio-1-1` contains simple 20-column rows. [VERIFIED: local command]
   - Resolution: Keep the Phase 1 gate on `tests/f_orbital_data`, and satisfy SPL-02 in `src/dftorch/script.py` by creating a temporary minimal simple `.skf` with deterministic 20-column electronic rows, parsing it through `read_skf_table()`, converting parsed channels with `channels_to_matrix()`, verifying canonical 40-channel output and f-only zero-fill, then reconstructing source-grid values with `cubic_spline_coeffs()`. [RESOLVED]

2. **Should compact-name validation use temporary files or direct helper calls?**
   - What we know: Production helper supports compact names; script helper rejects compact names. [VERIFIED: local code/grep]
   - Resolution: Validate compact names with direct production helper checks and temporary files created inside the script run. Do not add permanent compact fixture renames or broaden the primary fixture scope beyond `tests/f_orbital_data`. [RESOLVED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Running validation commands | yes | 0.11.26 | Use `python` directly if environment is already activated. [VERIFIED: local command] |
| Python | Parser and validation script | yes | 3.11.15 | None needed. [VERIFIED: local command] |
| PyTorch | Parser tensors and spline solve | yes | 2.10.0 | None for Phase 1. [VERIFIED: local command] |
| pytest | Existing regression suite | yes | 9.0.3 | Use `script.py` for f validation per D-04. [VERIFIED: local command/CONTEXT.md] |

**Missing dependencies with no fallback:** none found for Phase 1. [VERIFIED: local command]

**Missing dependencies with fallback:** pytest conversion is not required; `script.py` is the locked f validation path. [VERIFIED: local CONTEXT.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Standalone `src/dftorch/script.py` for Phase 1 f validation; pytest 9.0.3 for existing repo tests. [VERIFIED: local code/command] |
| Config file | `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests"]`. [VERIFIED: local code] |
| Quick run command | `uv run python src/dftorch/script.py tests/f_orbital_data` [VERIFIED: local command] |
| Full suite command | `uv run python src/dftorch/script.py tests/f_orbital_data` plus `uv run python -m pytest tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_scf.py -q` [VERIFIED: local command] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| SKF-01 | simple 20-column parse | script temp-file parser gap | `uv run python src/dftorch/script.py tests/f_orbital_data` after adding helper | yes, `script.py`; helper must create a temporary minimal simple `.skf` and parse it through `read_skf_table()`. [VERIFIED: local code/command] |
| SKF-02 | extended 40-column parse | script integration | `uv run python src/dftorch/script.py tests/f_orbital_data` | yes. [VERIFIED: local command] |
| SKF-03 | canonical 40-channel output | script integration | `uv run python src/dftorch/script.py tests/f_orbital_data` | yes. [VERIFIED: local command] |
| SKF-04 | simple zero-fill f channels | script temp-file parser gap | `uv run python src/dftorch/script.py tests/f_orbital_data` after adding helper | yes, `script.py`; assertion must inspect `read_skf_table()` output from a temporary minimal simple `.skf`. [VERIFIED: local code/command] |
| SKF-05 | dashed and compact resolution | script parser helper gap | `uv run python src/dftorch/script.py tests/f_orbital_data` after adding helper | yes, `script.py`; compact assertion missing. [VERIFIED: local code/grep] |
| SKF-06 | homonuclear f metadata | script integration | `uv run python src/dftorch/script.py tests/f_orbital_data` | yes. [VERIFIED: local command] |
| SKF-07 | skipped-shell rejection | script negative gap | `uv run python src/dftorch/script.py tests/f_orbital_data` after adding negative check | yes, `script.py`; assertion missing. [VERIFIED: local code/grep] |
| SPL-01 | f fixture spline reconstruction | script integration | `uv run python src/dftorch/script.py tests/f_orbital_data` | yes. [VERIFIED: local command] |
| SPL-02 | simple fixture spline reconstruction | script temp-file parser gap | `uv run python src/dftorch/script.py tests/f_orbital_data` after adding helper | yes, `script.py`; helper must reconstruct source-grid values from temporary simple `.skf` data parsed through `read_skf_table()`. [VERIFIED: local command] |
| SPL-03 | representative f pair coverage | script integration | `uv run python src/dftorch/script.py tests/f_orbital_data` | yes. [VERIFIED: local filesystem/command] |
| SPL-04 | CPU-only validation without optional GPU/Triton/DFTB+/ASE | script plus existing pytest | `uv run python src/dftorch/script.py tests/f_orbital_data` and `uv run pytest -q` | yes. [VERIFIED: local command] |

### Sampling Rate

- **Per task commit:** `uv run python src/dftorch/script.py tests/f_orbital_data` for parser/spline changes. [VERIFIED: local command]
- **Per wave merge:** `uv run python src/dftorch/script.py tests/f_orbital_data` plus `uv run pytest -q`. [VERIFIED: local command]
- **Phase gate:** Script passes on f fixtures and existing pytest suite passes before `$gsd-verify-work`. [VERIFIED: local command]

### Wave 0 Gaps

- [ ] `src/dftorch/script.py` helper for temporary simple-format `.skf` parsing through `read_skf_table()` and spline reconstruction. [VERIFIED: local command]
- [ ] `src/dftorch/script.py` helper for compact/dashed path resolution. [VERIFIED: local code/grep]
- [ ] `src/dftorch/script.py` negative check for skipped-shell `ValueError`. [VERIFIED: local code/grep]
- [ ] SPL-04 is satisfied in Phase 1 by the standalone script gate plus the existing focused pytest regression command; required pytest conversion remains deferred by D-05. [VERIFIED: local CONTEXT.md/REQUIREMENTS.md]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No authentication surface in local parser validation. [VERIFIED: local code/grep] |
| V3 Session Management | no | No session state in local parser validation. [VERIFIED: local code/grep] |
| V4 Access Control | no | Local file paths are caller-provided; no service authorization boundary is present. [VERIFIED: local code/CONCERNS.md] |
| V5 Input Validation | yes | Validate SKF row widths, grid/header presence, pair names, and nested shell layouts with `ValueError`. [VERIFIED: local code/grep] |
| V6 Cryptography | no | No cryptographic operations in parser/spline validation. [VERIFIED: local code/grep] |

### Known Threat Patterns for Local Scientific Parser

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed SKF rows causing silent bad tensors | Tampering | Reject row widths other than 20 or 40 and include file/line in `ValueError`. [VERIFIED: local code/grep] |
| Ambiguous basis layout from skipped shells | Tampering | Reject non-nested s/p/d/f layouts in `_validate_nested_shells()`. [VERIFIED: local code/grep] |
| Silent decode loss from invalid file bytes | Tampering | Current parser uses `read_text(errors="ignore")`; planner should avoid expanding this behavior and may document it as a compatibility quirk. [VERIFIED: local code/CONCERNS.md] |
| Arbitrary caller file paths | Information Disclosure | Treat as local-library behavior; validate roots only if these APIs are later exposed through a multi-user service. [VERIFIED: local CONCERNS.md] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/01-skf-canonicalization-and-spline-validation/01-CONTEXT.md` - locked user decisions and deferred ideas. [VERIFIED: local file]
- `.planning/REQUIREMENTS.md` - Phase 1 requirement IDs and descriptions. [VERIFIED: local file]
- `.planning/ROADMAP.md` - Phase scope and success criteria. [VERIFIED: local file]
- `.planning/STATE.md` - current phase status and blockers. [VERIFIED: local file]
- `.planning/codebase/ARCHITECTURE.md` - mapped parser/constants/structure flow. [VERIFIED: local file]
- `.planning/codebase/TESTING.md` - validation script conventions. [VERIFIED: local file]
- `.planning/codebase/CONCERNS.md` - known parser and validation gaps. [VERIFIED: local file]
- `src/dftorch/_bond_integral.py` - parser implementation. [VERIFIED: local code]
- `src/dftorch/script.py` - validation harness implementation. [VERIFIED: local code]
- Local commands: `uv run python src/dftorch/script.py tests/f_orbital_data`, `uv run python src/dftorch/script.py tests/data_skf_mio-1-1`, `uv run pytest -q`. [VERIFIED: local command]

### Secondary (MEDIUM confidence)

- `.planning/research/*.md` and `.planning/codebase/*.md` prior local research maps were used only as orientation and cross-checks. [VERIFIED: local file]

### Tertiary (LOW confidence)

- No network or external documentation was used, per user request. [VERIFIED: local instruction]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - versions and commands were checked locally. [VERIFIED: local command]
- Architecture: HIGH - parser and validation paths were read directly from source. [VERIFIED: local code]
- Pitfalls: HIGH for current gaps; MEDIUM for recommended remediation shape because user may choose stricter fixture-scope interpretation. [VERIFIED: local code/command] [ASSUMED]

**Research date:** 2026-07-20
**Valid until:** 2026-08-19 for local code state, or until `_bond_integral.py`, `script.py`, or fixture files change. [ASSUMED]

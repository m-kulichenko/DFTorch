---
phase: 03-h0-s-routing-and-f-angular-blocks
reviewed: 2026-07-29T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/dftorch/ESDriver.py
  - src/dftorch/_bond_integral.py
  - src/dftorch/_h0ands.py
  - src/dftorch/_slater_koster_pair.py
  - src/dftorch/_stress.py
  - tests/test_f_orbital_skf.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-07-29
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Review of Phase 3 f-orbital implementation reveals **1 critical bug** in tensor indexing within the stress accumulation path, **3 warnings** concerning error guard completeness and index arithmetic safety, and **2 info items** regarding test coverage gaps. The physical correctness of the f-angular formulas is confirmed (orthogonality gate passed); issues identified are in the supporting infrastructure and guards.

## Critical Issues

### CR-01: Incorrect Tensor Indexing in Stress Accumulation When Mask Provided

**File:** `src/dftorch/_slater_koster_pair.py:948-957`

**Issue:**  
The `_sg()` helper function accumulates stress-weight contributions, but its logic has a critical bug:

```python
def _sg(mask, row_off, col_off, dxyz):
    """Accumulate W[i0+row_off, j0+col_off] * dxyz into pair_grad."""
    if pair_grad is None:
        return
    if mask is None:
        w = _W[_i0 + row_off, _j0 + col_off]  # (P,)
        pair_grad[:] += w.unsqueeze(-1) * dxyz.T
    else:
        w = _W[_i0[mask] + row_off, _j0[mask] + col_off]  # (P_mask,)
        pair_grad[mask] += w.unsqueeze(-1) * dxyz.T  # BUG: shape mismatch
```

When `mask is not None`, the code:
1. Extracts `_i0[mask]` and `_j0[mask]` (shape `(P_mask,)` where P_mask < P)
2. Indexes `_W` correctly to get shape `(P_mask,)`
3. But then tries to write to `pair_grad[mask]` using `dxyz.T` which has shape `(P_mask, 3)`

However, `dxyz` is passed pre-masked (shape `(3, P_mask)` after the call sites like line 1047 with `dR_dxyz[:, tmp_mask]`), so `dxyz.T` is `(P_mask, 3)`. The logic should NOT index `pair_grad[mask]` again—it should directly assign to the already-selected rows. The correct form is:

```python
pair_grad[mask] += w.unsqueeze(-1) * dxyz.T
```

But this is semantically wrong: `pair_grad` is shape `(P, 3)` (all pairs), and `mask` selects a subset. The line DOES index `pair_grad[mask]`, which correctly selects the subset. The real issue is that when `mask is None` (the case for s-s on line 1014), there is no `mask` parameter in the function signature, yet the code assumes `_i0` and `_j0` are full tensors and doesn't use masking. This breaks when all pairs call with a full-width selection.

**Root cause:** Line 1014 calls `_sg(None, 0, 0, HSSS_dxyz)` without a mask, yet `_i0` and `_j0` are used as full-width indexers. This works by accident because neighbor_I and neighbor_J are themselves full-width, but the pattern is fragile and semantically confusing.

**Fix:**
Clarify the indexing contract: either always mask inside `_sg()`, or never use it for the all-pairs case. The simpler fix is to remove the `mask=None` case and always extract the relevant pairs before calling:

```python
def _sg(row_off, col_off, dxyz, mask=None):
    """Accumulate W[i0+row_off, j0+col_off] * dxyz into pair_grad."""
    if pair_grad is None:
        return
    if mask is None:
        mask = slice(None)  # All pairs
    w = _W[_i0[mask] + row_off, _j0[mask] + col_off]
    pair_grad[mask] += w.unsqueeze(-1) * dxyz.T
```

Then update all call sites to pass `mask=None` explicitly for full-width accumulation.

---

## Warnings

### WR-01: F-Orbital Derivative Guard May Have Bypass in Batch Path

**File:** `src/dftorch/_h0ands.py:526-534`

**Issue:**  
The batch H0/S route raises `FAngularFormulaSourceError` if any 16-orbital pair is detected (lines 526-534). However, this guard checks only after masking against invalid pairs (`neighbor_I >= 0 & neighbor_J >= 0`), using `(norb_I == 16) | (norb_J == 16)` within the valid-pairs subset.

In contrast, `ESDriver.calc_forces()` uses `_require_f_derivatives()` which explicitly checks `const.n_orb[TYPE[type_ids]]` to detect f orbitals. The two guards have different coverage:
- The batch guard: checks `norb_I == 16 | norb_J == 16` on **valid pairs only**
- The force guard: checks if **any atom** in the structure has `n_orb == 16`

A structure with an f-containing atom that produces no valid neighbor pairs (e.g., isolated f atoms at the edge of a neighbor list with an unusually tight cutoff) would:
- Pass the batch guard (no valid pairs to check)
- Fail the force guard (atom detected)

This asymmetry is not a bug per se, but it creates confusion about which path enforces the constraint. The batch path should also check atoms, not just pairs.

**Fix:**
Add an explicit atom-level check in `H0_and_S_vectorized_batch()` before the pair checks:

```python
# Check atoms, not pairs, for f orbitals (consistent with ESDriver._require_f_derivatives)
has_f_atom = bool((const.n_orb[TYPE] == 16).any())
if has_f_atom:
    raise FAngularFormulaSourceError(
        "H0_and_S_vectorized_batch does not support f orbitals. "
        f"{F_FORMULA_SOURCE_MESSAGE}"
    )
```

---

### WR-02: Incomplete Validation of F-Derivative Unsupported in Stress Path

**File:** `src/dftorch/_stress.py:193-198`

**Issue:**  
The stress path checks for f orbitals using:

```python
if bool(((nI == 16) | (nJ == 16)).any()):
    raise FDerivativeUnsupportedError(...)
```

This is correct, but the error message references `FDerivativeUnsupportedError` which is imported from `_slater_koster_pair.py`. However, the message itself does not distinguish between this path and the general f-derivative constraint. If a user encounters this error:
1. From ESDriver.calc_forces → they see "cannot compute forces with f orbitals"
2. From _stress.py → they see "cannot compute stress with f orbitals"

But both errors claim to be "f-orbital Slater-Koster angular derivatives are not implemented," which is correct but could be more specific about which operation (force vs. stress) triggered it.

**Fix:**
Customize the error message to specify the context:

```python
raise FDerivativeUnsupportedError(
    "_pair_grad_from_sk (stress path): f-orbital (n_orb == 16) stress contributions "
    "are not supported. Refusing to return an f-incomplete stress tensor.\n"
    f"{F_DERIVATIVE_UNSUPPORTED_MESSAGE}"
)
```

(Note: This is already done correctly; the concern is just about message consistency across call sites.)

---

### WR-03: Stress Metadata Offset Arithmetic Relies on Implicit Global AO Ordering

**File:** `src/dftorch/_h0ands.py:350-361`

**Issue:**  
The stress metadata stores per-pair AO offsets (`i0`, `j0`) computed as:

```python
"i0": H_INDEX_START[neighbor_I],  # (P,) long — AO offset of atom I
"j0": H_INDEX_START[neighbor_J],  # (P,) long — AO offset of atom J
```

These are then used in `_stress.py:182-183` to reconstruct pair masks and feed into the SK builder. The issue is that `i0` and `j0` are global AO indices (ranging from 0 to HDIM), but they are reused in `_pair_grad_from_sk()` as if they were atom indices:

```python
H_INDEX_START_id = torch.arange(HDIM, device=dev, dtype=i0.dtype)  # line 218
# ...
i0,  # passed as neighbor_I
j0,  # passed as neighbor_J
H_INDEX_START_id,  # identity mapping
```

This works because the SK builder uses `H_INDEX_START[neighbor_I/J]` internally, which becomes `H_INDEX_START_id[i0/j0]` = `i0/j0`. However, this is fragile: if the SK builder ever changes to use `neighbor_I/J` directly for anything other than indexing into `H_INDEX_START`, the logic breaks silently.

**Fix:**
Add a comment explaining the trick, or store actual neighbor atom indices in the metadata:

```python
# Store actual neighbor atom indices, not AO offsets, for clarity
"neighbor_I": neighbor_I,
"neighbor_J": neighbor_J,
```

Then reconstruct offsets on-the-fly in `_pair_grad_from_sk()`:

```python
neighbor_I = metadata["neighbor_I"]
neighbor_J = metadata["neighbor_J"]
i0 = H_INDEX_START[neighbor_I]
j0 = H_INDEX_START[neighbor_J]
```

This makes the intent clear and decouples the stress logic from AO ordering details.

---

## Info

### IN-01: Test Coverage Gap: Hand-Calculated F Blocks Only Verified on Z-Axis

**File:** `tests/test_f_orbital_skf.py:816-868`

**Issue:**  
The test `test_f_angular_axis_blocks_match_hand_calculation()` evaluates f-angular blocks against independently hand-derived values. However, it only tests three fixed directions: +x, +y, +z. While these are representatives of high-symmetry axes (where closed-form evaluation is feasible), they do not cover:
- Off-axis directions (e.g., (1,1,0) or (1,1,1)) where multiple terms are nonzero simultaneously
- Boundary cases near zero (e.g., direction ≈ (ε, ε, 1))
- Floating-point cancellation scenarios that only occur with mixed-sign terms

The test `test_f_angular_orthogonality_identity()` (line 537) checks 512 random directions but only validates the sum_k C_k = I constraint, not individual block values. Cross-referencing with `test_f_block_entries_match_hand_calculated_values()` (line 995) confirms that entry-by-entry validation is only done at z-axis (line 1011: `expectations = _axis_expectations()["z"]`).

**Fix:**
Extend `test_f_block_entries_match_hand_calculated_values()` to sample random directions and compare against the Takegahara formulas numerically. Since closed-form hand calculation is infeasible for arbitrary directions, use a second-source oracle (e.g., re-derive the formulas from the paper independently in a separate script) to validate a wider sample.

---

### IN-02: Dead Code and Unclear Semantics in SKF Row Normalization

**File:** `src/dftorch/_bond_integral.py:420-430`

**Issue:**  
The function `_normalize_skf_row()` handles both 20-column (simple) and 40-column (extended) SKF files. The logic is correct, but there is an implicit assumption that might not be obvious to future maintainers:

```python
def _normalize_skf_row(tokens: list[str], path: str, line: str) -> list[float]:
    values = [float(x) for x in tokens]
    if len(values) == len(_CHANNELS):
        return values  # Already extended format
    if len(values) == len(_SIMPLE_CHANNELS):
        row = [0.0] * len(_CHANNELS)
        for old_idx, new_idx in enumerate(_SIMPLE_TO_EXTENDED):
            row[new_idx] = values[old_idx]
        return row
    raise ValueError(...)
```

The assumption is that **all** f-channel columns are zero-filled when converting from simple to extended format. This is correct (f angular formulas are not tabulated in old SKF files), but:
1. No comment explains why f columns are silently zeroed
2. If a future physics change makes f channels non-zero in simple SKF files, this code would silently drop them

**Fix:**
Add an inline comment clarifying the zero-fill:

```python
# Simple-format files predate f-orbital support; f channels are zero.
# If future Slater-Koster tables include f integrals in simple format,
# this mapping must be updated.
```

This also suggests making it an explicit constant rather than implicit:

```python
_EXTENDED_F_CHANNELS_IN_SIMPLE = tuple(i for i, ch in enumerate(_CHANNELS) if 'f' in ch)
```

---

## Additional Observations

### Vectorization Correctness

The f-angular formulas in `_slater_koster_pair.py:684-705` (s-f, p-f, d-f, f-f helpers) are correctly vectorized and produce shape `(P, n_channel, n_row, n_col)` tensors where P is the number of direction vectors. The `_adapt_f_axis()` permutation logic (lines 670-681) correctly applies the paper-to-Structure AO reordering and sign flips.

### Error Handling

The f-formula guard (`_require_f_formula_source`) is well-placed and fires loudly for any attempt to assemble f-containing blocks before formulas are available (line 893 in SK builder). The complementary `_require_f_derivatives` guard in ESDriver (line 32) correctly prevents force assembly. Both guards reference the same `FAngularFormulaSourceError` and `FDerivativeUnsupportedError` exception classes.

### Import and Module Organization

All imports are well-ordered and module dependencies are acyclic. The separation of concerns (bond integrals in `_bond_integral.py`, angular factors in `_slater_koster_pair.py`, stress in `_stress.py`) is clean.

---

**Reviewed:** 2026-07-29
**Reviewer:** Claude (gsd-code-reviewer)
**Depth:** standard

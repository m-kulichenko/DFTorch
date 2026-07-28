# Phase 3 Pattern Map: H0/S Routing and f Angular Blocks

## Closest Existing Patterns

- `src/dftorch/_h0ands.py:12-33` is the single-system H0/S entry point to extend. It accepts prebuilt neighbor tensors, `const.n_orb`, `H_INDEX_START`, `IJ_pair_type`/`JI_pair_type`, `R_orb`, and `coeffs_tensor`, then calls the vectorized Slater-Koster routine twice: H with `SH_shift=0` at `src/dftorch/_h0ands.py:226-254`, S with `SH_shift=1` at `src/dftorch/_h0ands.py:265-293`.
- `src/dftorch/_h0ands.py:151-178` is the exact routing analog. It classifies atom pairs by orbital count masks for 1, 4, and 9 orbitals. Add 16-orbital masks here for f-containing single-system pairs without changing existing `HH/HX/XH/XX/HY/XY/YH/YX/YY` semantics.
- `src/dftorch/_h0ands.py:256-301` is the assembly post-processing pattern. H0 adds onsite `diagonal` and is symmetrized; S is divided by `27.21138625`, gets identity, and is symmetrized. Preserve this for f blocks.
- `src/dftorch/_slater_koster_pair.py:194-210` is the radial channel access pattern. `_get_val_dR(...)` applies `channel + SH_shift * 10` and returns `(value, dvalue_dR)`. This is currently stale for 40-channel extended tensors; Phase 3 should route f channels by the canonical indices in `_bond_integral.py`, not by assuming 10-channel H/S blocks.
- `src/dftorch/_slater_koster_pair.py:410-471` is the s-d block analog: build `tmp_mask`, slice `idx_row/idx_col`, gather one radial channel, compute angular entries, `index_add_` into AO offsets, then compute derivative expressions and `_sg(...)`.
- `src/dftorch/_slater_koster_pair.py:473-535` and `src/dftorch/_slater_koster_pair.py:854-910` are p-d / d-p direction analogs. Note the reverse-direction branch uses `JI_pair_type` and negated p-d angular values.
- `src/dftorch/_slater_koster_pair.py:1174-1245` is the d-d analog for same-shell multi-channel blocks. f-f should mirror this style: gather sigma/pi/delta/phi channels, compute named matrix entries in local AO order, and write each with explicit row/column offsets.
- `src/dftorch/Structure.py:11-24` and `src/dftorch/Structure.py:214-223` define the locked local AO ordering: `s, px, py, pz, dxy, dyz, dzx, dx2_y2, dz2, fx3, fy3, fz3, fx_y2_z2, fy_z2_x2, fz_x2_y2, fxyz`.
- `src/dftorch/Structure.py:333-411` is the single-structure AO layout source consumed by H0/S: `n_orbitals_per_atom`, `H_INDEX_START`, `H_INDEX_END`, shell flags, onsite diagonal, `HDIM`, shell types, and labels.
- `src/dftorch/_bond_integral.py:101-141` is the canonical 40-channel order. f channels are `Hff0..3` at 0-3, `Hdf0..2` at 4-6, `Hpf0..1` at 12-13, `Hsf0` at 18, and overlap equivalents at 22-25, 26-28, 32-33, 38.
- `src/dftorch/Constants.py:101-181` registers `coeffs_tensor`, `pair_lookup`, `n_orb`, `n_f`, `Ef`, `Uf`, and `shell_present` from `get_skf_tensors`; tests should use these attributes instead of rebuilding parser state.

## Files to Modify

- `src/dftorch/_h0ands.py`: add single-system 16-orbital pair masks and pass them to the SK routine. Leave `H0_and_S_vectorized_batch` unchanged or add an explicit f unsupported guard; Phase 3 defers batch f paths.
- `src/dftorch/_slater_koster_pair.py`: extend `Slater_Koster_Pair_SKF_vectorized` for f-containing blocks only: s-f, f-s, p-f, f-p, d-f, f-d, f-f. Preserve existing non-f writes and derivatives. Prefer explicit formulas and named tensors like existing s-d/p-d/d-d sections.
- `tests/test_f_orbital_skf.py`: add Phase 3 tests around the existing float64/module-reset harness and real `tests/f_orbital_data` fixtures. Keep current parser/constants/structure tests intact.
- Possibly `src/dftorch/script.py`: only if adding reusable validation helpers for H0/S construction. Existing helper patterns already cover synthetic XYZ, `Constants`, `Structure`, and assertions.

## Test Patterns to Reuse

- Reuse `tests/test_f_orbital_skf.py:9-23` to force `torch.float64` and restore imported `dftorch` modules after each validation.
- Reuse `tests/test_f_orbital_skf.py:26-35` to load `src/dftorch/script.py` directly for lightweight validation helpers.
- Reuse `tests/test_f_orbital_skf.py:38-42` and `src/dftorch/script.py:980-985` for tiny synthetic XYZ files.
- Reuse `src/dftorch/script.py:1180-1193` to construct `Constants` from `tests/f_orbital_data`, and `tests/test_f_orbital_skf.py:100-113` to construct a single `Structure`.
- Reuse `src/dftorch/script.py:988-1066` and `src/dftorch/script.py:1311-1325` style: assert exact tensor shapes/metadata first, then compare concrete values with readable assertion messages.
- Add targeted Phase 3 tests for x/y/z directions using one Eu-containing single system. Minimum checks: finite H0/S/dH0/dS, symmetry after assembly, preserved simple-format f-free regression, and selected hand-computed f block entries from Sharma formulas.

## Cautions

- AO offsets are fixed: f starts at local offset 9 and spans offsets 9-15. Do not infer f order from channel order; use `Structure.py` labels and verify Sharma ordering/signs before coding formulas.
- The current SK docstring describes 10-channel H/S blocks, but `_bond_integral.py` now returns 40 channels. Existing `_get_val_dR(channel + SH_shift * 10)` maps simple H/S channels, not extended f channels. f routing likely needs an explicit channel-index helper for `H*` and `S*` names.
- Direction matters. Existing p-s and d-p reverse paths use `JI_pair_type` and sign changes; f-s, f-p, and f-d need the same deliberate convention, not blind transposition.
- Derivatives are part of the single-system return contract. If f derivative formulas are not implemented, add explicit unsupported behavior for f gradients rather than returning silent zeros. Non-gradient H0/S construction must still produce finite matrices.
- Do not touch batch f support except to guard against silent misuse. `H0_and_S_vectorized_batch` still only routes 1/4/9-orbital masks at `src/dftorch/_h0ands.py:468-477`.
- Preserve existing non-f masks and regression behavior. f additions should be additive around 16-orbital masks and f-only formulas.

# Phase 3 Research: H0/S Routing and f Angular Blocks

## Summary

Phase 3 should be a small single-system H0/S assembly slice: route all pairs where either atom has `const.n_orb == 16`, add auditable f angular block helpers for `s-f`, `p-f`, `d-f`, and `f-f`, and prove finite symmetric H0/S matrices without attempting SCF, batch, gradients, forces, stress, MD, SEDACS, ML-SK, or performance work. [VERIFIED: codebase grep]

The planner should include one prerequisite correction before f formulas: `_slater_koster_pair.py` still documents and implements old `channel + SH_shift * 10` channel addressing, while `_bond_integral.py` now stores 40 channels as named H channels followed by named S channels. Use named channel indices from `_bond_integral._CHANNELS` or a local canonical mapping instead of numeric offsets. [VERIFIED: src/dftorch/_bond_integral.py:101] [VERIFIED: src/dftorch/_slater_koster_pair.py:97] [VERIFIED: src/dftorch/_slater_koster_pair.py:201]

## Formula Source and Convention

The DOI requested in the phase context, `10.1088/0022-3719/13/4/016`, resolves to "Slater-Koster tables for f electrons" in Journal of Physics C: Solid State Physics 13(4), pages 583-588, published 1980-02-10. The accessible metadata names K. Takegahara, Y. Aoki, and A. Yanase as authors; it did not identify Sharma as an author. [CITED: https://cir.nii.ac.jp/crid/1360855568798449280?lang=en]

The accessible abstract confirms the paper represents f orbitals by cubic harmonics and includes two-centre `s-f`, `p-f`, `d-f`, and `f-f` overlap/energy integral tables. The full formula tables, normalization, row/column ordering, and sign convention were not extractable from the IOP/CiNii HTML or the IOP PDF endpoint in this session; the endpoint returned issue HTML rather than a readable article PDF. [CITED: https://cir.nii.ac.jp/crid/1360855568798449280?lang=en] [VERIFIED: local curl]

Current `Structure.py` fixes the f AO order as `fx3`, `fy3`, `fz3`, `fx_y2_z2`, `fy_z2_x2`, `fz_x2_y2`, `fxyz`, with comments defining the polynomial labels. This is a codebase convention, not a verified match to the paper table order. [VERIFIED: src/dftorch/Structure.py:15]

Smallest correct adaptation: keep `Structure.py` AO order stable, implement Sharma/Takegahara table formulas in their native paper order, then apply an explicit permutation/sign vector into Structure order at the helper boundary. If the supplied paper table order is identical, the adapter is identity; if not, only the adapter changes. Do not silently relabel `Structure.py` or transpose f blocks. [ASSUMED]

What the user must supply before formula implementation can be scientifically locked: the article PDF or extracted table pages showing the cubic harmonic definitions, f AO ordering, and all `sf`, `pf`, `df`, and `ff` formulas. Without that, implementation can only be a scaffold plus convention tests, not a trusted f angular formula implementation. [VERIFIED: local curl] [ASSUMED]

## Existing Code Routing

`src/dftorch/_bond_integral.py` defines the canonical 40-channel order: H channels `Hff0` through `Hss0`, then S channels `Sff0` through `Sss0`; spline tensors are stored as `(n_pairs, npts, 40, 4)` and populated via `channels_to_matrix()`. [VERIFIED: src/dftorch/_bond_integral.py:101] [VERIFIED: src/dftorch/_bond_integral.py:988] [VERIFIED: src/dftorch/_bond_integral.py:1045]

`src/dftorch/Constants.py` registers `coeffs_tensor`, `pair_lookup`, `n_orb`, `max_ang`, `n_f`, `Ef`, `Uf`, and `shell_present` from `get_skf_tensors()`. Phase 2 found no production metadata drift in `Constants.py` or `Structure.py`. [VERIFIED: src/dftorch/Constants.py] [VERIFIED: .planning/phases/02-constants-and-structure-basis-metadata/02-01-SUMMARY.md]

`src/dftorch/Structure.py` already builds atom-major AO starts/ends from `const.n_orb[self.TYPE]`, exposes `shell_present`, `has_f`, `shell_ao_start`, `shell_ao_end`, `ao_labels`, and a 16-position AO template. [VERIFIED: src/dftorch/Structure.py:25] [VERIFIED: src/dftorch/Structure.py:326]

`src/dftorch/_h0ands.py::H0_and_S_vectorized()` currently creates masks only for 1-, 4-, and 9-orbital pair classes: `HH`, `HX`, `XH`, `XX`, `HY`, `XY`, `YH`, `YX`, and `YY`. No mask includes `n_orb == 16`, so f-containing pair angular terms are currently omitted by routing. [VERIFIED: src/dftorch/_h0ands.py:151]

`H0_and_S_vectorized()` calls `Slater_Koster_Pair_SKF_vectorized()` twice, once for H and once for S, then reshapes and explicitly symmetrizes both matrices. This symmetry pass can hide one-sided block errors, so tests must inspect selected unsymmetrized block entries or direct helper outputs too. [VERIFIED: src/dftorch/_h0ands.py:226] [VERIFIED: src/dftorch/_h0ands.py:258] [VERIFIED: src/dftorch/_h0ands.py:265]

`src/dftorch/ESDriver.py::forward()` is the single-system integration entry: it builds the electronic neighbor list, calls `H0_and_S_vectorized()`, then proceeds into overlap inverse, repulsion, Coulomb, SCF, and forces. For Phase 3, tests should call `H0_and_S_vectorized()` directly for H0/S proof and only use `ESDriver` smoke paths if unsupported downstream f modes are guarded. [VERIFIED: src/dftorch/ESDriver.py:100] [VERIFIED: src/dftorch/ESDriver.py:114]

Batch routing in `_h0ands.py::H0_and_S_vectorized_batch()` mirrors the 1/4/9 mask limitation and is explicitly deferred for this phase. [VERIFIED: src/dftorch/_h0ands.py:469] [VERIFIED: .planning/phases/03-h0-s-routing-and-f-angular-blocks/03-CONTEXT.md]

## Implementation Approach

1. Add a named-channel helper for SK spline evaluation, e.g. `channel_index("Hsf0")` / `channel_index("Ssf0")`, and route H versus S by prefix instead of `SH_shift * 10`. This is required to preserve 1-, 4-, and 9-orbital behavior under the 40-channel representation. [VERIFIED: src/dftorch/_bond_integral.py:101] [VERIFIED: src/dftorch/_slater_koster_pair.py:201]

2. Keep current non-f masks and formulas intact in behavior, but add f-aware pair classes in the single-system path: `HZ/ZH`, `XZ/ZX`, `YZ/ZY`, and `ZZ`, where `Z` means `n_orb == 16`. Existing `HH/HX/XH/XX/HY/XY/YH/YX/YY` routes should continue to exercise the existing s/p/d formulas. [VERIFIED: src/dftorch/_h0ands.py:151] [ASSUMED]

3. Add f block helpers in `_slater_koster_pair.py` or a small adjacent module, grouped by angular pair family: `sf/fs`, `pf/fp`, `df/fd`, `ff`. Keep formulas explicit, table-shaped, and indexed by the fixed local AO offsets: s `0`, p `1:4`, d `4:9`, f `9:16`. [VERIFIED: src/dftorch/Structure.py:12] [ASSUMED]

4. First implement value assembly for H0/S. Do not silently provide physically wrong f derivatives for later force/stress paths. Either implement derivatives consistently with the formula helpers or guard downstream f-containing force/stress/MD paths with explicit unsupported errors until a later phase. [VERIFIED: src/dftorch/_h0ands.py:261] [VERIFIED: .planning/phases/03-h0-s-routing-and-f-angular-blocks/03-CONTEXT.md]

5. For formula convention, add a small constant such as `STRUCTURE_F_AO_ORDER` and `PAPER_F_AO_ORDER`, plus `paper_to_structure_f_permutation` and `paper_to_structure_f_sign`. Populate only after the paper table is available. This makes any mismatch visible and keeps the adaptation minimal. [ASSUMED]

6. Keep tests CPU/pytest-only and reuse the existing Phase 1/2 validation harness style in `tests/test_f_orbital_skf.py`, which already wraps parser, constants, structure, and simple-format metadata gates while restoring `dftorch` modules. [VERIFIED: tests/test_f_orbital_skf.py:9] [VERIFIED: tests/test_f_orbital_skf.py:120]

## Test Strategy

Add a direct single-system H0/S construction test using the f fixtures in `tests/f_orbital_data/` and a minimal Eu-containing XYZ, preferably `N/Ga/Eu` or a two-atom Eu-containing pair whose SKF files exist. Assert H0 and S shapes equal `(structure.HDIM, structure.HDIM)`, all entries are finite, S has identity diagonal contribution, and H0/S are symmetric after assembly. [VERIFIED: tests/test_f_orbital_skf.py:156] [ASSUMED]

Add a regression test over representative f-free systems from existing simple-format fixtures to prove H/X/Y routing is preserved for 1-, 4-, and 9-orbital atoms. At minimum, run the existing `test_simple_format_f_free_metadata_regression` and add one H0/S route check that records selected old s/p/d block entries before and after the channel-lookup fix. [VERIFIED: tests/test_f_orbital_skf.py:180] [ASSUMED]

Add angular formula tests for x-axis `(L,M,N)=(1,0,0)`, y-axis `(0,1,0)`, and z-axis `(0,0,1)`. These should call the f-block helper directly with synthetic radial integrals, not only full matrix assembly, so signs/order fail before final symmetrization hides them. [VERIFIED: src/dftorch/_h0ands.py:258] [ASSUMED]

Add selected hand-calculated block-entry checks from the supplied paper formulas for each family: one `s-f`, one `p-f`, one `d-f`, and several `f-f` entries. These checks are blocked until the paper table/order is provided or extracted. [CITED: https://cir.nii.ac.jp/crid/1360855568798449280?lang=en] [ASSUMED]

Add atom-order tests for an `I->J` and `J->I` f pair to verify `IJ_pair_type` versus `JI_pair_type` use, parity signs for reversed odd-l couplings, and final matrix symmetry. Existing `p-s` and `d-p` code uses opposite pair-type lookups and explicit signs, so f routes need equivalent coverage. [VERIFIED: src/dftorch/_slater_koster_pair.py:266] [VERIFIED: src/dftorch/_slater_koster_pair.py:755] [ASSUMED]

Run the focused gate after implementation: `uv run python -m pytest tests/test_f_orbital_skf.py -q`, plus the existing import/IO/neighbor/SCF smoke set used by Phase 2 if f-free routing changed. [VERIFIED: .planning/phases/02-constants-and-structure-basis-metadata/02-01-SUMMARY.md]

## Risks and Open Questions

- Paper access blocker: the DOI metadata and abstract are verified, but the actual formula tables/order/sign convention were not accessible in this session. The user needs to provide the PDF or extracted table pages before formulas can be treated as scientifically verified. [CITED: https://cir.nii.ac.jp/crid/1360855568798449280?lang=en] [VERIFIED: local curl]

- Author/source mismatch: the requested "Sharma" source appears to correspond to the same DOI/title but accessible metadata lists Takegahara/Aoki/Yanase. Confirm whether "Sharma" is a shorthand/misremembered source or whether a different Sharma paper should be used. [CITED: https://cir.nii.ac.jp/crid/1360855568798449280?lang=en]

- Channel indexing risk: current SK evaluation still assumes 10-channel H/S blocks, which conflicts with the 40-channel order. Fix this before adding f formulas, or tests may pass shape/symmetry while using wrong radial integrals. [VERIFIED: src/dftorch/_bond_integral.py:101] [VERIFIED: src/dftorch/_slater_koster_pair.py:97]

- Derivative risk: `H0_and_S_vectorized()` returns `dH0` and `dS`, and downstream force/stress paths may consume them. If Phase 3 does not implement f derivatives, later f force/stress/MD paths need explicit unsupported guards to avoid silent wrong physics. [VERIFIED: src/dftorch/_h0ands.py:261] [VERIFIED: .planning/phases/03-h0-s-routing-and-f-angular-blocks/03-CONTEXT.md]

- Symmetry masking risk: final H0/S symmetrization can hide one-sided f block placement bugs. Direct helper and pre-symmetry selected-entry tests are required. [VERIFIED: src/dftorch/_h0ands.py:258]


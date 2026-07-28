# Codebase Concerns

**Analysis Date:** 2026-07-20
**Scope:** `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`, `tests/f_orbital_data`
**last_mapped_commit:** `e824543a0b411dcf52462ee55db5362c360e7780`

## Tech Debt

**F-orbital basis metadata is duplicated across parser, constants, and structure layers:**
- Issue: The s/p/d/f basis shape is encoded in multiple places: `_CHANNELS` in `src/dftorch/_bond_integral.py`, `MAX_SHELLS` and metadata tensors in `src/dftorch/_bond_integral.py`, `shell_dim` in `src/dftorch/Constants.py`, and AO/shell templates in `src/dftorch/Structure.py`.
- Files: `src/dftorch/_bond_integral.py:101`, `src/dftorch/_bond_integral.py:947`, `src/dftorch/Constants.py:65`, `src/dftorch/Structure.py:11`
- Impact: Changes to orbital ordering, shell count, or channel layout require synchronized edits across several modules. A mismatch can silently corrupt Hamiltonian channel lookup, AO indexing, shell-resolved Hubbard data, or onsite diagonals.
- Fix approach: Promote a single basis metadata module that defines shell dimensions, shell IDs, AO labels, channel names, and simple-to-extended channel mapping. Import that module from `src/dftorch/_bond_integral.py`, `src/dftorch/Constants.py`, and `src/dftorch/Structure.py`.

**`ConstantsTest` is a large stale parallel constants table:**
- Issue: `ConstantsTest` hard-codes element labels and legacy orbital metadata separately from the real SKF loader, then infers `shell_present` from `max_ang`.
- Files: `src/dftorch/Constants.py:205`, `src/dftorch/Constants.py:1078`
- Impact: Tests or debugging code using `ConstantsTest` can disagree with the f-orbital SKF path. The class does not exercise `get_skf_tensors()` and can hide parser/structure integration bugs.
- Fix approach: Replace `ConstantsTest` with a small fixture builder that consumes `tests/f_orbital_data` through `get_skf_tensors()`, or move it under tests with explicit limitations documented in the fixture name.

**Validation code is a standalone script instead of collected tests:**
- Issue: F-orbital validation lives in `src/dftorch/script.py` and uses manual dynamic imports, print-based reporting, and temporary files instead of normal test discovery.
- Files: `src/dftorch/script.py:1`, `src/dftorch/script.py:72`, `src/dftorch/script.py:1232`
- Impact: CI can pass without running the f-orbital validation unless a workflow explicitly invokes `python src/dftorch/script.py tests/f_orbital_data`. The validation is also packaged with runtime source code.
- Fix approach: Move the checks into `tests/` as pytest tests, keep shared fixture helpers in a test utility module, and leave `src/dftorch/script.py` only as a thin optional CLI wrapper if needed.

**Runtime modules print diagnostics directly:**
- Issue: `Constants` prints SOC and DFTB3 status during initialization, and `ESDriver` prints GBSA/Hessian timings from compute paths.
- Files: `src/dftorch/Constants.py:142`, `src/dftorch/Constants.py:197`, `src/dftorch/ESDriver.py:1011`, `src/dftorch/ESDriver.py:1434`, `src/dftorch/ESDriver.py:1612`
- Impact: Library callers and tests cannot consistently suppress output. Long-running workflows mix numerical results with stdout diagnostics, and tests need output capture for deterministic assertions.
- Fix approach: Route messages through a shared logger or a `verbose`/diagnostics object. Keep `Constants` and `ESDriver` silent by default.

**Legacy bond-integral code remains mixed with the SKF parser:**
- Issue: Old CSV-style parameter loaders and legacy vectorized bond-integral helpers share `src/dftorch/_bond_integral.py` with the f-orbital SKF parser.
- Files: `src/dftorch/_bond_integral.py:178`, `src/dftorch/_bond_integral.py:245`, `src/dftorch/_bond_integral.py:300`, `src/dftorch/_bond_integral.py:547`
- Impact: The module has multiple unrelated parameter-loading models, making it harder to isolate f-orbital parser behavior and test only the modern SKF path.
- Fix approach: Split legacy CSV parameter support into a separate compatibility module or remove it if no scoped callers use it. Keep `src/dftorch/_bond_integral.py` focused on SKF parsing, channel normalization, and spline tensor construction.

## Known Bugs

**Batched PME Coulomb is explicitly unsupported:**
- Symptoms: Batched `ESDriverBatch` calculations raise `ValueError("Batched PME Coulomb not implemented.")` in both energy and force paths.
- Files: `src/dftorch/ESDriver.py:1279`, `src/dftorch/ESDriver.py:1568`
- Trigger: `ESDriverBatch.forward()` or `ESDriverBatch.calc_forces()` with `dftorch_params["COUL_METHOD"] == "PME"`.
- Workaround: Use non-batched `ESDriver` for PME calculations, or use a non-PME Coulomb method for batched calculations.

**Full off-diagonal DFTB3 is unsupported with PME:**
- Symptoms: Single-structure `ESDriver` raises `NotImplementedError` when PME is selected with off-diagonal DFTB3 enabled.
- Files: `src/dftorch/ESDriver.py:167`, `src/dftorch/ESDriver.py:172`
- Trigger: `COUL_METHOD == "PME"`, `structure.dU_dq is not None`, and `dftb3_diagonal_only` is not true.
- Workaround: Set `dftb3_diagonal_only=True` for PME, or use `COUL_METHOD="FULL"` for full off-diagonal third-order DFTB.

**Validation script only supports dashed SKF names for its independent checks:**
- Symptoms: The runtime parser supports dashed and compact SKF pair filenames, but the validator's `split_dashed_pair()` rejects compact names.
- Files: `src/dftorch/_bond_integral.py:433`, `src/dftorch/_bond_integral.py:465`, `src/dftorch/script.py:116`
- Trigger: Running `src/dftorch/script.py` against compact fixture names such as `EuN.skf` instead of dashed names such as `Eu-N.skf`.
- Workaround: Keep `tests/f_orbital_data` filenames dashed when using `src/dftorch/script.py`.

## Security Considerations

**Caller-provided parameter paths are read without path controls:**
- Risk: `Constants` accepts `SKFPATH` and `FILENAME`, then reads coordinate files, SKF files, optional `spinw.txt`, optional `hubbard_derivative.txt`, and optional `wfc.hsd` from caller-controlled paths.
- Files: `src/dftorch/Constants.py:57`, `src/dftorch/Constants.py:71`, `src/dftorch/Constants.py:127`, `src/dftorch/Constants.py:185`, `src/dftorch/_bond_integral.py:586`, `src/dftorch/_bond_integral.py:883`
- Current mitigation: Normal local filesystem permissions apply. The scoped code is a local scientific library path, not a service boundary.
- Recommendations: If these APIs are exposed through a server, notebook hub, or workflow runner with untrusted users, validate allowed roots for `SKFPATH` and structure filenames before constructing `Constants`.

**SKF parser ignores decode errors:**
- Risk: `read_skf_table()` and `read_wfc_hsd()` use `Path.read_text(errors="ignore")`, so invalid bytes are silently dropped before numeric parsing.
- Files: `src/dftorch/_bond_integral.py:586`, `src/dftorch/_bond_integral.py:883`, `src/dftorch/script.py:108`
- Current mitigation: Numeric row length and required block checks catch many malformed files after decoding.
- Recommendations: Use explicit encoding with strict decode by default, or report ignored-decode behavior in an opt-in compatibility path for legacy parameter files.

## Performance Bottlenecks

**Spline coefficient construction solves dense systems for every ordered pair:**
- Problem: `get_skf_tensors()` loops through all ordered element pairs and calls `cubic_spline_coeffs()`, which builds and solves a dense `n x n` linear system for each channel matrix.
- Files: `src/dftorch/_bond_integral.py:779`, `src/dftorch/_bond_integral.py:1020`, `src/dftorch/_bond_integral.py:1045`
- Cause: The spline builder handles all 40 channels generically and uses `torch.linalg.solve()` on a dense matrix for each SKF table.
- Improvement path: Use a tridiagonal spline solver or cache coefficient tensors by SKF path and dtype/device. Keep the dense implementation as a reference test path for small fixtures.

**SKF tensors allocate fixed-size padding independent of actual table sizes:**
- Problem: `get_skf_tensors()` allocates `(n_pairs, 1300, 40, 4)` coefficient storage and `(n_pairs, 500, 6)` repulsive storage even when fixture files contain fewer rows.
- Files: `src/dftorch/_bond_integral.py:982`, `src/dftorch/_bond_integral.py:987`, `src/dftorch/_bond_integral.py:994`
- Cause: The loader uses fixed maximum dimensions instead of sizing tensors from parsed tables.
- Improvement path: Parse table metadata first to size tensors exactly, or store per-pair ragged tensors with explicit lengths. If fixed padding remains required by downstream kernels, validate max row counts and document the cap.

**Structure batch assembly has Python loops over batch elements and shell local indices:**
- Problem: `StructureBatch` fills per-structure padded diagonals, AO shell types, labels, and D0 with Python loops.
- Files: `src/dftorch/Structure.py:186`, `src/dftorch/Structure.py:608`, `src/dftorch/Structure.py:668`
- Cause: Batch structures have variable `HDIM`, so the implementation flattens and then copies into padded tensors row by row.
- Improvement path: Keep the current code for readability until profiling shows it matters. For larger f-orbital batches, vectorize padded scatter operations and leave Python-only AO label generation outside hot paths.

## Fragile Areas

**Nested-shell-only assumption is a hard architectural constraint:**
- Files: `src/dftorch/_bond_integral.py:496`, `src/dftorch/Structure.py:11`, `src/dftorch/Structure.py:90`
- Why fragile: The parser rejects f-shell bases unless s, p, and d shells are also present. The structure layer relies on fixed local starts `(0, 1, 4, 9)` and fixed 16-position AO templates.
- Safe modification: Preserve contiguous s/p/d/f shells unless the AO indexing model is redesigned. Add negative tests for skipped-shell parameter files before changing `_validate_nested_shells()`.
- Test coverage: `src/dftorch/script.py` validates N, Ga, and Eu fixture metadata, but it does not include malformed skipped-shell SKF fixtures.

**Parser-inferred shell presence can conflict with `wfc.hsd` overrides:**
- Files: `src/dftorch/_bond_integral.py:665`, `src/dftorch/_bond_integral.py:869`, `src/dftorch/_bond_integral.py:1060`
- Why fragile: Homonuclear SKF headers set occupations, onsite energies, Hubbard values, and shell presence; optional `wfc.hsd` later overrides only `SHELL_PRESENT`, `N_ORB`, `MAX_ANG`, and `MAX_ANG_OCC`.
- Safe modification: When adding `wfc.hsd` support or new parameter sets, verify that occupation tensors (`N_S`, `N_P`, `N_D`, `N_F`) remain consistent with any overridden shell presence.
- Test coverage: `tests/f_orbital_data` does not include `wfc.hsd`, so the override path is not covered by the scoped fixture set.

**AO ordering is a shared scientific invariant with no central assertion:**
- Files: `src/dftorch/_bond_integral.py:101`, `src/dftorch/Structure.py:25`, `src/dftorch/Structure.py:384`
- Why fragile: The channel order and AO label order must match downstream Slater-Koster formulas outside this scoped map. Local tests validate metadata and spline reconstruction, but they do not validate angular f-orbital Hamiltonian formulas.
- Safe modification: Treat `_CHANNELS` and `AO_LABEL_TEMPLATE` as compatibility contracts. Add explicit tests that compare f-channel Hamiltonian/overlap blocks against a trusted DFTB+ or analytical reference before changing order.
- Test coverage: `src/dftorch/script.py:1155` states that angular Slater-Koster formulas are outside its validation scope.

## Scaling Limits

**Element metadata tensors are capped at atomic number 119:**
- Current capacity: `get_skf_tensors()` allocates atomic metadata tensors with length 120.
- Limit: Elements or pseudo-elements with identifiers above 119 cannot be represented without changing tensor sizes and symbol lookup.
- Scaling path: Size metadata tensors from the element table or from `max(TYPE) + 1`, and validate all symbols before allocation.
- Files: `src/dftorch/_bond_integral.py:998`, `src/dftorch/_bond_integral.py:1003`, `src/dftorch/_bond_integral.py:1016`

**Scoped f-orbital fixtures are sizeable and data-heavy:**
- Current capacity: `tests/f_orbital_data` is about 1.7 MB and contains nine `.skf` files for N/Ga/Eu ordered pairs plus `.DS_Store`.
- Limit: Adding more lanthanide or actinide fixture matrices can increase repository size quickly.
- Scaling path: Keep one minimal f-orbital smoke fixture in `tests/f_orbital_data`, move larger parameter sets to optional test assets, and remove non-fixture files such as `tests/f_orbital_data/.DS_Store`.
- Files: `tests/f_orbital_data/Eu-Eu.skf`, `tests/f_orbital_data/Ga-Ga.skf`, `tests/f_orbital_data/N-N.skf`, `tests/f_orbital_data/.DS_Store`

**F-orbital AO dimensions amplify dense matrix costs:**
- Current capacity: The scoped validator covers a small N/Ga/Eu synthetic structure with 24 AOs.
- Limit: Eu contributes 16 AOs per atom, so dense Hamiltonian, overlap, Coulomb, and SCF matrices grow quickly with f-heavy systems.
- Scaling path: Add benchmark tests for f-heavy structures before optimizing dense paths, and prefer sparse/block representations for large f-electron systems.
- Files: `src/dftorch/Structure.py:334`, `src/dftorch/Structure.py:408`, `src/dftorch/script.py:1053`

## Dependencies at Risk

**Parameter-file semantics depend on DFTB+ SKF conventions:**
- Risk: The parser assumes 20-column simple rows or 40-column extended rows, homonuclear header layouts of 10 or 13 values, a `Spline` block, and DFTB+ compact repetition tokens.
- Impact: Valid parameter files with other extensions, encoding, comments, shell conventions, or row layouts can fail to load or be normalized incorrectly.
- Migration plan: Add fixture coverage for each accepted SKF dialect and reject unsupported dialects with structured errors. Keep `_normalize_skf_row()` and `read_skf_table()` as the only code paths for row interpretation.
- Files: `src/dftorch/_bond_integral.py:397`, `src/dftorch/_bond_integral.py:547`, `tests/f_orbital_data/Eu-Eu.skf`

**F-orbital validation depends on direct source import mechanics:**
- Risk: `src/dftorch/script.py` creates a fake `dftorch` package and loads modules by file path.
- Impact: The script can diverge from installed-package import behavior and does not catch issues in package initialization or public exports.
- Migration plan: Convert the script checks to pytest tests that import installed modules normally where possible; use isolated direct imports only for targeted parser unit tests.
- Files: `src/dftorch/script.py:72`, `src/dftorch/script.py:86`

## Missing Critical Features

**Batched PME for f-orbital structures:**
- Problem: Batched PME energy and force paths raise `ValueError`.
- Blocks: Efficient batched periodic f-orbital calculations using PME electrostatics.
- Files: `src/dftorch/ESDriver.py:1279`, `src/dftorch/ESDriver.py:1568`

**Full PME off-diagonal DFTB3 for f-orbital systems:**
- Problem: PME supports only diagonal-only DFTB3 in the scoped driver path.
- Blocks: Full off-diagonal third-order DFTB with PME for f-shell parameter sets.
- Files: `src/dftorch/ESDriver.py:172`, `src/dftorch/ESDriver.py:179`

**Reference validation for angular f-orbital Hamiltonian/overlap formulas:**
- Problem: The scoped validator checks SKF parsing, spline reproduction, constants exposure, and AO bookkeeping, but not the angular Slater-Koster formulas that consume f channels.
- Blocks: High-confidence changes to `_CHANNELS`, AO ordering, and downstream f-orbital Hamiltonian assembly.
- Files: `src/dftorch/script.py:1155`, `src/dftorch/_bond_integral.py:101`, `src/dftorch/Structure.py:25`

## Test Coverage Gaps

**`wfc.hsd` shell-presence override path:**
- What's not tested: `read_wfc_hsd()` overriding `SHELL_PRESENT`, `N_ORB`, `MAX_ANG`, and `MAX_ANG_OCC` after SKF header parsing.
- Files: `src/dftorch/_bond_integral.py:869`, `src/dftorch/_bond_integral.py:1060`, `tests/f_orbital_data`
- Risk: Parameter sets that rely on `wfc.hsd` can produce inconsistent shell metadata or occupations without the scoped fixture script detecting it.
- Priority: High

**Malformed and skipped-shell SKF fixtures:**
- What's not tested: Empty files, malformed grid lines, invalid row widths, missing `Spline` blocks, unsupported angular momentum, and skipped-shell f/d/p bases.
- Files: `src/dftorch/_bond_integral.py:428`, `src/dftorch/_bond_integral.py:522`, `src/dftorch/_bond_integral.py:702`, `tests/f_orbital_data`
- Risk: Parser error handling can regress or become less actionable as more parameter dialects are added.
- Priority: High

**Normal test runner coverage for f-orbital validation:**
- What's not tested: The f-orbital checks under `src/dftorch/script.py` are not represented as standard test files under `tests/`.
- Files: `src/dftorch/script.py:556`, `src/dftorch/script.py:724`, `src/dftorch/script.py:1143`
- Risk: Parser, constants, and structure regressions can ship if CI does not invoke the standalone script.
- Priority: High

**Angular f-channel formula correctness:**
- What's not tested: Hamiltonian/overlap block values that consume `Hff*`, `Hdf*`, `Hpf*`, `Hsf*`, `Sff*`, `Sdf*`, `Spf*`, and `Ssf*` channels.
- Files: `src/dftorch/_bond_integral.py:101`, `src/dftorch/Structure.py:25`, `src/dftorch/script.py:1155`
- Risk: SKF parser and AO bookkeeping can pass while f-orbital electronic structure values are wrong.
- Priority: High

---

*Concerns audit: 2026-07-20*

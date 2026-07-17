# Codebase Concerns

**Analysis Date:** 2026-07-17

## Tech Debt

**Large numerical modules with mixed responsibilities:**
- Issue: Core scientific kernels, orchestration, logging, configuration branching, and feature-specific paths are concentrated in very large files.
- Files: `src/dftorch/_slater_koster_pair.py`, `src/dftorch/MD.py`, `src/dftorch/_gbsa.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_scf.py`, `src/dftorch/ewald_pme/ewald_torch.py`
- Impact: Small behavior changes have a large review surface. Shared scientific invariants are hard to isolate, and tests need broad fixtures to cover a narrow change.
- Fix approach: Extract feature-specific kernels and state builders behind small functions with tensor-shape contracts. Start with `src/dftorch/ESDriver.py` GBSA proxy creation and PME setup, then split `src/dftorch/MD.py` timing/logging from integrator state transitions.

**PME and neighbor-list TODOs encode unresolved assumptions:**
- Issue: The PME/Triton and neighbor-list code contains explicit assumptions about k-vector shape, block sizes, dummy indices, non-periodic systems, and dynamic neighbor limits.
- Files: `src/dftorch/ewald_pme/ewald_triton.py`, `src/dftorch/ewald_pme/ewald_torch.py`, `src/dftorch/ewald_pme/neighbor_list.py`, `src/dftorch/ewald_pme/util.py`
- Impact: PME behavior is fragile across small systems, CPU/GPU layout differences, PyTorch compile modes, and new hardware. The code can silently rely on assumptions that are not validated at API boundaries.
- Fix approach: Convert TODO assumptions into explicit guards and tests. Add shape checks for `kvecs`, explicit dummy-index handling, parameterized CPU/GPU equivalence tests, and dynamic `max_nbr_limit` calculation in `src/dftorch/ewald_pme/neighbor_list.py`.

**Torch compile support is inconsistent:**
- Issue: `src/dftorch/_tools.py` gates `_maybe_compile()` behind `DFTORCH_ENABLE_COMPILE`, but several PME and neighbor-list functions use direct `@torch.compile` decorators, while force kernels disable compile due incorrect results.
- Files: `src/dftorch/_tools.py`, `src/dftorch/ewald_pme/ewald_torch.py`, `src/dftorch/ewald_pme/neighbor_list.py`, `src/dftorch/ewald_pme/PME_torch.py`, `src/dftorch/_forces.py`, `src/dftorch/_forces_batch.py`
- Impact: Tests disable TorchDynamo, so compiled paths and known Inductor-sensitive force paths are not exercised in CI. Users can see different behavior in eager mode, compiled PyTorch kernels, and Triton kernels.
- Fix approach: Route compile decisions through `src/dftorch/_tools.py` only. Add separate compile-enabled tests for representative PME, force, and neighbor-list cases, marked `slow` or `gpu` as needed.

**Runtime output is coupled to computation:**
- Issue: Many core functions print timing and progress directly rather than using a shared verbosity/logger interface.
- Files: `src/dftorch/_scf.py`, `src/dftorch/MD.py`, `src/dftorch/_coulomb_matrix.py`, `src/dftorch/_coulomb_matrix_batch.py`, `src/dftorch/_h0ands.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`
- Impact: Library users cannot consistently suppress output, tests cannot assert diagnostics cleanly, and performance-critical loops carry I/O side effects.
- Fix approach: Introduce a small logging/timing helper and pass verbosity through existing driver parameters. Keep numerical functions pure unless explicit debug output is requested.

**Development artifacts and generated files are present in the working tree:**
- Issue: Profiling reports are tracked, and local/generated artifacts appear in the working tree.
- Files: `experiments/nsight_run.nsys-rep`, `experiments/nsight_reports/nsight_run.nsys-rep`, `.gitignore`, `src/dftorch.egg-info`, `src/dftorch/__pycache__`, `src/dftorch/ewald_pme/__pycache__`, `.DS_Store`, `tests/.DS_Store`, `node-v24.18.0.pkg`
- Impact: Repository size and review noise grow. Generated files can mask packaging issues and confuse clean-checkout behavior.
- Fix approach: Remove tracked profiling artifacts if they are not release assets. Add `.venv/`, `.DS_Store`, `*.nsys-rep`, and installer packages to `.gitignore`; keep benchmark outputs in an ignored artifact directory.

## Known Bugs

**PME k-space matrix is an empty implementation:**
- Symptoms: Calling `ewald_kspace_matrix()` returns `None` because the function body is `pass`.
- Files: `src/dftorch/ewald_pme/ewald_torch.py`
- Trigger: Any code path that expects an explicit k-space matrix from `ewald_kspace_matrix()`.
- Workaround: Use the existing energy/force PME functions instead of the matrix helper.

**Full off-diagonal DFTB3 is unsupported with PME:**
- Symptoms: `ESDriver` raises `NotImplementedError` when `COUL_METHOD == "PME"` and non-diagonal DFTB3 data is present.
- Files: `src/dftorch/ESDriver.py`
- Trigger: PME calculations with full off-diagonal third-order DFTB.
- Workaround: Set `dftb3_diagonal_only=True` or use `COUL_METHOD="FULL"`.

**Zero-charge derivative workaround changes PME derivative math:**
- Symptoms: Zero charges are replaced with `1.0` before charge-derivative division in k-space PME.
- Files: `src/dftorch/ewald_pme/ewald_torch.py`
- Trigger: PME derivative calculations with atoms whose charge value is exactly zero.
- Workaround: Avoid relying on k-space `de_dq` for zero-charge systems until the derivative expression is rewritten without division by charge.

**Neighbor-list backend changes semantics for some flags:**
- Symptoms: The Alchemi backend notes that `remove_self_neigh` is silently ignored and `min_image_only` is handled by post-processing.
- Files: `src/dftorch/_nearestneighborlist.py`
- Trigger: `vectorized_nearestneighborlist(..., use_alchemi=True)` with `remove_self_neigh` or `min_image_only` expectations.
- Workaround: Prefer the default backend for correctness-sensitive cases until parity tests cover these flags.

## Security Considerations

**PyTorch checkpoint loading allows pickle execution:**
- Risk: `torch.load(..., weights_only=False)` can execute pickle payloads from an untrusted model checkpoint.
- Files: `src/dftorch/_ml_sk.py`
- Current mitigation: Not detected in code; callers provide the checkpoint path.
- Recommendations: Use `weights_only=True` where possible, validate checkpoint schema explicitly, and document that model files must be trusted if object loading remains required.

**User-provided file paths are read and appended without sandboxing:**
- Risk: Library APIs read coordinate/parameter files and append output files at caller-provided paths. This is appropriate for a local scientific library, but unsafe if exposed through a service layer without path controls.
- Files: `src/dftorch/_io.py`, `src/dftorch/_tools.py`, `src/dftorch/_gbsa.py`, `src/dftorch/_bond_integral.py`
- Current mitigation: Not detected in code; normal Python file permissions apply.
- Recommendations: If wrapping DFTorch in an API or notebook service, validate path roots before calling `read_xyz`, `read_pdb`, `read_hubbard_derivs`, `load_spin_constants`, or output writers in `src/dftorch/_io.py`.

**CI pins action majors but uses moving tool versions:**
- Risk: `astral-sh/setup-uv@v3` installs `version: "latest"` in release workflows, so dependency resolution tooling can change outside code review.
- Files: `.github/workflows/release.yml`, `.github/workflows/tests.yml`
- Current mitigation: `uv.lock` pins Python packages.
- Recommendations: Pin `uv` to a specific version in CI workflows and update it intentionally.

## Performance Bottlenecks

**Default neighbor list can allocate O(N^2 x 27) tensors:**
- Problem: The documented default backend builds brute-force periodic distance matrices unless Alchemi is selected.
- Files: `src/dftorch/_nearestneighborlist.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_scf.py`
- Cause: `vectorized_nearestneighborlist()` defaults `use_alchemi` to module global `USE_ALCHEMI=False`; many call sites pass `use_triton=False` or do not pass an accelerated backend.
- Improvement path: Promote cell-list neighbor generation for large systems, keep brute force for small test cases, and add size-based backend selection with correctness parity tests.

**Finite-difference Hessian scales with 6N displaced evaluations:**
- Problem: `_hessian_fd()` performs central-difference batches across every coordinate degree of freedom.
- Files: `src/dftorch/ESDriver.py`
- Cause: Hessian construction uses displaced `StructureBatch` calculations rather than analytic second derivatives.
- Improvement path: Document the scaling limit, expose progress and memory estimates, and prefer analytic or block-sparse approaches for larger systems.

**Triton autotune search is broad and manually chosen:**
- Problem: `get_autotune_config()` returns many block-size combinations with a TODO that options need more tuning for modern GPUs.
- Files: `src/dftorch/ewald_pme/ewald_triton.py`
- Cause: Kernel configuration is static and not tied to benchmark data or device capability.
- Improvement path: Add benchmark fixtures for representative PME sizes, narrow autotune configs by device class, and persist validated configs in tests or documentation.

**K-space matrix construction uses nested `torch.vmap`:**
- Problem: `ewald_real_matrix()` constructs a dense matrix using nested vectorization over all atoms.
- Files: `src/dftorch/ewald_pme/ewald_torch.py`
- Cause: Matrix assembly is dense by design and builds per-atom reductions for every target index.
- Improvement path: Use sparse COO accumulation from neighbor indices for systems where the real-space matrix is needed.

## Fragile Areas

**Electronic structure driver owns many optional feature combinations:**
- Files: `src/dftorch/ESDriver.py`
- Why fragile: PME, DFTB3, GBSA, batch mode, Hessian mode, force calculations, and third-order setup share one driver. Several branches mutate `structure` by attaching optional attributes such as `thirdorder`, `gbsa_batch`, and `thirdorder_batch`.
- Safe modification: Add tests for each feature branch before editing. Keep structure mutations explicit and grouped near feature setup.
- Test coverage: Existing tests in `tests/test_scf.py` exercise one small CPU PME smoke path only.

**PME backend selection depends on import and device state:**
- Files: `src/dftorch/ewald_pme/__init__.py`, `src/dftorch/ewald_pme/ewald_triton.py`, `src/dftorch/ewald_pme/ewald_torch.py`
- Why fragile: Triton import is conditional on `torch.cuda.is_available()`, k-vector transposition depends on active backend, and CPU fallback handles different layouts.
- Safe modification: Add backend parity tests that compare CPU Torch PME and GPU/Triton PME for the same small cells.
- Test coverage: No GPU or Triton tests are present under `tests/`.

**SEDACS integration has optional distributed dependencies and direct CUDA assumptions:**
- Files: `src/dftorch/sedacs/sedacs_interface.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`, `pyproject.toml`
- Why fragile: The optional extra pulls `mpi4py`, `numba`, `toml`, and `scikit-learn`, while implementation code directly uses distributed partitioning, CUDA device selection, and synchronization.
- Safe modification: Keep SEDACS changes behind import guards and add small CPU-only partition tests plus optional MPI/GPU CI jobs.
- Test coverage: No tests exercise `src/dftorch/sedacs/`.

**Public API and README disagree on exported names:**
- Files: `README.md`, `src/dftorch/__init__.py`, `tests/test_public_api.py`, `tests/test_public_api_contract.py`
- Why fragile: The README lists `Optimizer` as a supported public import, while `src/dftorch/__init__.py` exports `GeoOpt` from `src/dftorch/Optimizer.py`.
- Safe modification: Treat `src/dftorch/__init__.py` and `tests/test_public_api_contract.py` as the source of truth, or add a compatibility alias if `Optimizer` is intended public API.
- Test coverage: Public API tests do not assert the README-listed `Optimizer` name.

## Scaling Limits

**Memory usage grows quickly with atom count in default paths:**
- Current capacity: README demonstrates large simulations through accelerated paths, but default neighbor-list and some Coulomb paths allocate dense matrices.
- Limit: Brute-force neighbor lists and full Coulomb matrix paths become impractical as `N` grows.
- Scaling path: Use Alchemi/cell-list neighbor backends, PME electrostatics, sparse accumulation, and benchmarked batch sizes for large systems.

**Test suite validates only small CPU examples:**
- Current capacity: Tests cover import, public API, XYZ read, nearest-neighbor smoke, and one CH4 CPU SCF/force smoke test.
- Limit: GPU, Triton, SEDACS, batch structures, GBSA, D3, Delta-SCF, stress, MD, optimizer, and compile-enabled paths can regress without CI failures.
- Scaling path: Add layered tests: small deterministic unit tests for math helpers, CPU integration tests for feature branches, and marked GPU/slow tests for accelerated kernels.

**Repository data footprint is dominated by experiment assets:**
- Current capacity: `experiments/` is about 100 MB locally, mostly Slater-Koster data under `experiments/sk_orig`.
- Limit: Clone and CI context size grow with every additional dataset or profiling artifact.
- Scaling path: Keep minimal fixtures under `tests/`, move large experiment assets to external downloads or release artifacts, and document expected data locations.

## Dependencies at Risk

**Unbounded runtime dependencies:**
- Risk: `pyproject.toml` specifies `numpy`, `scipy`, `torch`, `pandas`, and `nvalchemi-toolkit-ops` without version bounds.
- Impact: Major dependency changes can alter tensor behavior, binary compatibility, or GPU support.
- Migration plan: Add tested lower and upper bounds in `pyproject.toml`, keep `uv.lock` updated through CI, and document supported PyTorch/CUDA combinations.

**Optional GPU and distributed dependencies are hard to validate in CI:**
- Risk: Triton, CUDA, Alchemi/Warp, MPI, and Numba paths are environment-sensitive.
- Impact: Accelerated and distributed workflows can break while CPU smoke tests pass.
- Migration plan: Add optional workflow jobs keyed by markers `gpu`, `slow`, and `sedacs`; keep CPU tests independent of GPU imports.

**Model loading depends on checkpoint schema stability:**
- Risk: `load_ml_sk_model()` expects keys such as `model_config`, `n_species`, `model_state_dict`, `Z_to_idx`, and optional `rcut_by_Z_pair`.
- Impact: Older or externally trained checkpoints can fail at runtime or load partially because `strict=False` is used for model state.
- Migration plan: Version checkpoint schema, validate required keys before constructing the model, and warn when `strict=False` skips or ignores weights.

## Missing Critical Features

**Full PME support for off-diagonal DFTB3:**
- Problem: PME supports only diagonal-only DFTB3 in `src/dftorch/ESDriver.py`.
- Blocks: Full off-diagonal third-order DFTB calculations with PME electrostatics.

**Completed k-space matrix helper:**
- Problem: `ewald_kspace_matrix()` is not implemented in `src/dftorch/ewald_pme/ewald_torch.py`.
- Blocks: Any feature that needs explicit k-space Coulomb matrix assembly from the PME module.

**Structured validation for input parameters:**
- Problem: Driver and constants parameters are plain dictionaries with many implicit keys and mode combinations.
- Blocks: Early, actionable error messages for invalid `COUL_METHOD`, cutoff, solvent, DFTB3, ML-SK, and batch-mode configurations.

**Stable benchmark suite:**
- Problem: Performance claims and backend choices are not tied to executable benchmark tests.
- Blocks: Confident optimization of `src/dftorch/ewald_pme/`, `src/dftorch/_nearestneighborlist.py`, `src/dftorch/_scf.py`, and `src/dftorch/MD.py`.

## Test Coverage Gaps

**GPU/Triton PME paths:**
- What's not tested: Triton kernels, CUDA-only backend selection, k-vector transposition, low-memory k-space paths, and GPU force parity.
- Files: `src/dftorch/ewald_pme/__init__.py`, `src/dftorch/ewald_pme/ewald_triton.py`, `src/dftorch/ewald_pme/ewald_torch.py`, `tests/`
- Risk: GPU regressions and CPU/GPU numerical drift can ship unnoticed.
- Priority: High

**Force and stress correctness:**
- What's not tested: Analytical force decomposition, PME forces, shadow forces, batch forces, stress tensors, and compile-disabled force kernels.
- Files: `src/dftorch/_forces.py`, `src/dftorch/_forces_batch.py`, `src/dftorch/_stress.py`, `tests/test_scf.py`
- Risk: Scientific results can change while smoke tests only assert finite tensors.
- Priority: High

**Feature branches beyond the CH4 smoke case:**
- What's not tested: GBSA/ALPB, D3(BJ), Delta-SCF, DFTB3 full and diagonal modes, spin/open-shell paths, ML-SK, optimizer, MD, and batch structures.
- Files: `src/dftorch/_gbsa.py`, `src/dftorch/_dftd3.py`, `src/dftorch/_thirdorder.py`, `src/dftorch/_spin.py`, `src/dftorch/_ml_sk.py`, `src/dftorch/Optimizer.py`, `src/dftorch/MD.py`, `tests/`
- Risk: Optional capabilities listed in `README.md` can regress without failures.
- Priority: High

**Input parsing edge cases:**
- What's not tested: PDB parsing, CRYST1 cell handling, malformed XYZ/PDB files, trajectory metadata parsing, spin constant parsing, and Hubbard derivative parsing.
- Files: `src/dftorch/_io.py`, `src/dftorch/_tools.py`, `tests/test_io.py`
- Risk: User input errors surface as late numerical failures rather than clear parser errors.
- Priority: Medium

**SEDACS distributed workflow:**
- What's not tested: Graph partitioning, halo exchange assumptions, MPI integration, and SEDACS MD/SCF loops.
- Files: `src/dftorch/sedacs/sedacs_interface.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`, `tests/`
- Risk: Distributed simulations can fail independently of the main CPU package tests.
- Priority: Medium

**Packaging and clean-checkout hygiene:**
- What's not tested: Source distribution contents, wheel contents, absence of generated artifacts, and README public import examples.
- Files: `pyproject.toml`, `README.md`, `src/dftorch/__init__.py`, `.gitignore`, `.github/workflows/release.yml`
- Risk: Published packages can miss data files, include unintended artifacts, or advertise imports that are not exported.
- Priority: Medium

---

*Concerns audit: 2026-07-17*

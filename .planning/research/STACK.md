# Stack Research: f-orbital DFTB/SKF Support

**Project:** DFTorch f-orbital support  
**Research type:** Stack dimension  
**Researched:** 2026-07-17  
**Overall confidence:** HIGH for repo-grounded stack choices; MEDIUM for external SKF/math references; LOW where only forum material exists.

## Recommendation

Keep the implementation inside the current Python/PyTorch stack. Do not introduce a new chemistry framework, parser generator, symbolic algebra layer, JAX backend, compiled extension, or DFTB+ runtime dependency for the prototype. The fastest reliable path is to extend the existing SKF ingestion and tensor kernels in place, using the bundled `tests/f_orbital_data/` files and DFTB+/paper-derived fixtures as validation oracles.

The stack should be:

| Area | Use | Why | Confidence |
|------|-----|-----|------------|
| Runtime math | PyTorch tensors, current dtype/device conventions | Existing `Constants`, `Structure`, `ESDriver`, `_h0ands.py`, and `_slater_koster_pair.py` are already tensor-first and device-aware. Reusing this avoids a second numerical contract. | HIGH |
| SKF parsing | Current `src/dftorch/_bond_integral.py` helpers | The file already contains channel normalization scaffolding for 20-column simple SKF rows and 40-column extended rows, plus f-shell metadata (`N_F`, `EF`, `UF`, `SHELL_PRESENT`). Build on it rather than replacing it. | HIGH |
| Storage | Extend `Constants.py` with f-shell fields | `Constants` is the package parameter boundary. Add `n_f`, `Ef`, `Uf`, and a 16-orbital shell dimension path there before touching driver math. | HIGH |
| H/S construction | Extend `_h0ands.py` and `_slater_koster_pair.py` masks from `1/4/9` to include `16` | Current pair masks are hard-coded for s, sp, spd atom sizes. f support requires explicit masks and block placement for spdf atoms. | HIGH |
| Tests | `pytest`, real fixtures, `torch.float64`, CPU-first | Existing tests already disable TorchDynamo and use bundled SKF fixtures. Add exact parser/channel tests before simulation tests. | HIGH |
| External oracle | DFTB+ only as an offline reference generator | DFTB+ is the canonical implementation for SKF semantics, but requiring it at test/runtime would make local and CI setup brittle. Store reference outputs instead. | MEDIUM |
| Optional validation helper | ASE only for one-off DFTB+ output extraction scripts | ASE can parse DFTB+ outputs such as energy, forces, charges, and eigenvalues, but it should not become a package dependency. | MEDIUM |

## Most Relevant Code Areas

### 1. SKF ingestion: `src/dftorch/_bond_integral.py`

Start here. The project already has the right direction:

- `_CHANNELS` is a 40-channel order including `ff`, `df`, `pf`, `sf`, and the overlap equivalents.
- `_SIMPLE_CHANNELS` maps the older 20-channel s/p/d rows into the 40-channel representation.
- `_normalize_skf_row()` accepts either 20 or 40 electronic values.
- `read_skf_table()` detects extended files by an initial `@`, parses homonuclear metadata, and carries `N_F`, `EF`, `UF`, and `SHELL_PRESENT`.
- `_resolve_skf_path()` supports both dashed names (`Eu-N.skf`) and compact names (`EuN.skf`), matching `tests/f_orbital_data/`.

Recommendation: preserve this universal 40-channel internal representation and make it the only downstream contract. Simple SKF files should become zero-filled f-channel tensors at parse time. No later layer should branch on raw file format.

Prototype checkpoints:

1. Parse every file in `tests/f_orbital_data/`.
2. Assert expected electronic table width after expansion: 40 channels.
3. Assert simple files still map old channels into the same indices.
4. Evaluate cubic splines at source grid points and compare against raw SKF rows.
5. Assert f-shell metadata for Eu-like homonuclear files: `n_orb == 16`, f occupation fields present, and `max_ang` supports f.

### 2. Constants storage: `src/dftorch/Constants.py`

`Constants.__init__()` currently unpacks `N_D`, `ES`, `EP`, `ED`, `US`, `UP`, and `UD` but does not yet register f analogs in the visible storage path. Extend the returned tuple and registered parameters deliberately:

- `self.n_f`
- `self.Ef`
- `self.Uf`
- `self.shell_present` if needed for diagnostics and validation
- `self.shell_dim = [0, 1, 3, 5, 7]` or an equivalent representation that can express f shell size

Keep the current `torch.nn.Parameter(..., requires_grad=False)` pattern for non-trainable constants. Respect `GRAD_PARAM` for learnable on-site/Hubbard values only if existing s/p/d fields do.

Do not hide missing f support behind defaults. If an f-channel file is parsed but `Constants` cannot represent it, raise a precise error during construction.

### 3. Structure indexing: `src/dftorch/Structure.py`

Structure already derives orbital offsets from `const.n_orb`. That is the right abstraction. The risky parts are boolean shell assumptions:

- `has_p = const.n_orb >= 4`
- `has_d = const.n_orb == 9`

Extend these to support `16` for f while preserving the existing offset ordering. The expected prototype orbital layout should remain nested:

```text
s:      0
p:      1..3
d:      4..8
f:      9..15
```

This matches the current assumption that shell offsets are contiguous. Reject non-nested bases for now; supporting skipped shells would require a larger offset model.

### 4. H0/S and Slater-Koster angular transforms

`src/dftorch/_h0ands.py` computes direction cosines, radial spline interval indices via `torch.searchsorted`, and pair masks based on `const.n_orb`. It currently recognizes only:

```text
1 = s
4 = sp
9 = spd
```

Add `16 = spdf` explicitly. The pair-mask expansion should be boring and visible, not clever:

```text
H/Z-style naming today: H=s, X=sp, Y=spd
Recommended next: add F or Z for spdf and enumerate masks that touch 16
```

`src/dftorch/_slater_koster_pair.py` is the most fragile area. It already contains long explicit s/p/d angular formula blocks and derivative propagation. For f support, do not try to abstract the whole file first. Add f blocks in a narrow, testable style:

- s-f: 1 channel
- p-f: 2 channels
- d-f: 3 channels
- f-f: 4 channels
- duplicate the path for H and S using the existing `SH_shift` convention
- include derivative terms at the same time as value terms if forces/stress are in scope for the phase

The channel order should stay aligned to `_CHANNELS`:

```text
Hff0 Hff1 Hff2 Hff3
Hdf0 Hdf1 Hdf2
Hdd0 Hdd1 Hdd2
Hpf0 Hpf1
Hpd0 Hpd1
Hpp0 Hpp1
Hsf0 Hsd0 Hsp0 Hss0
Sff0 Sff1 Sff2 Sff3
Sdf0 Sdf1 Sdf2
Sdd0 Sdd1 Sdd2
Spf0 Spf1
Spd0 Spd1
Spp0 Spp1
Ssf0 Ssd0 Ssp0 Sss0
```

## File-Format References

Use these as implementation references, in this order:

| Reference | Use | Notes | Confidence |
|-----------|-----|-------|------------|
| DFTB.org parameter introduction | General semantics of ordered pair SKF files, H/S tables, repulsive terms, homonuclear atomic data | Good source for high-level format constraints and pair-file expectations. | MEDIUM |
| Slater-Koster file format v1.0 technical guide (`slakoformat`) | Simple vs extended format, `@` marker, homonuclear/heteronuclear layouts, repulsive spline format | The guide states extended format supports angular momentum up to f; use for parser layout. | MEDIUM |
| DFTB+ Recipes first calculation | MaxAngularMomentum requirement, ordered pair files, DFTB+ runtime expectations | Important because SKF files historically do not fully encode included orbitals. | MEDIUM |
| Existing fixtures in `tests/f_orbital_data/` | Ground truth for this project | The fixtures are the concrete input surface; they outrank generic docs when deciding parser behavior. | HIGH |
| Existing simple fixtures in `tests/data_skf_mio-1-1/` and tutorial workflows | Regression guard for old behavior | Simple-format results must remain unchanged. | HIGH |

Important format facts for the roadmap:

- Extended SKF files begin with `@`.
- Simple electronic rows contain 20 H/S values for s/p/d channels.
- Extended electronic rows contain 40 H/S values including f channels.
- Homonuclear files carry atomic metadata; heteronuclear files do not.
- Ordered pair files are not interchangeable: `A-B.skf` and `B-A.skf` can differ.
- Max angular momentum is a calculation input in DFTB+ because SKF format history makes automatic inference unsafe.

## Mathematical References

Use Slater-Koster angular transformations, but treat orbital ordering as a project-level convention that must be tested.

Recommended source stack:

| Reference | Use | Confidence |
|-----------|-----|------------|
| Slater and Koster, Phys. Rev. 94, 1498 (1954) | Original two-center angular decomposition; conceptual basis for sigma/pi/delta/phi channels | MEDIUM |
| DFTB+ source behavior / DFTB+ generated H/S references | Practical oracle for signs, order, and conventions | MEDIUM |
| DFTB+ user-list discussion on p/d tesseral ordering | Warning that DFTB+ internal order is not naive `px, py, pz`; useful for avoiding sign/order bugs | LOW by itself, MEDIUM when confirmed by fixtures |
| Reference-paper fixture outputs for `tests/f_orbital_data/` | Final scientific validation target | HIGH once captured in repo |

Do not copy f formulas from an arbitrary table without proving the basis ordering. The current DFTorch s/p/d implementation appears to use offsets:

```text
s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
```

DFTB+ materials often discuss tesseral ordering differently. The roadmap should include a small axis-aligned pair test suite where direction cosines are `(1,0,0)`, `(0,1,0)`, and `(0,0,1)`. These tests make sign and ordering mistakes obvious before full molecular simulations.

## Validation Stack

Use layered validation. Do not start with full SCF as the first correctness signal.

| Layer | Tool | What to Assert | Confidence |
|-------|------|----------------|------------|
| Parser unit tests | `pytest`, `torch.float64`, raw fixture files | row counts, channel counts, metadata, pair path resolution, simple-to-extended channel mapping | HIGH |
| Spline reconstruction | `pytest`, existing cubic coefficients | evaluating coefficients at grid knots reproduces source SKF channel values within tight tolerance | HIGH |
| Constants integration | `Constants` with f fixtures | registered tensor shapes, f fields, no change to simple fixtures | HIGH |
| Angular block tests | direct calls into `_slater_koster_pair.py` or a small helper | known axis-aligned H/S blocks select the correct sigma/pi/delta/phi channels | MEDIUM |
| H0/S matrix tests | `Constants` -> `Structure` -> `_h0ands.py` | matrix dimensions include 16-orbital atoms; symmetry and overlap diagonal behavior remain correct | HIGH |
| External reference tests | stored DFTB+/paper fixture values | energies, charges, eigenvalues, selected H/S entries, or full simulation outputs | MEDIUM until references are captured |

Suggested commands:

```bash
uv run pytest tests/test_f_orbital_skf.py -q
uv run pytest tests/test_scf.py -q
uv run pytest -m "not slow"
uv run ruff check src/dftorch tests
```

For reference generation, use DFTB+ outside the normal test run. ASE may help parse `detailed.out`, `results.tag`, forces, charges, energies, and eigenvalues, but store the extracted values as fixtures. Do not require ASE or DFTB+ for regular CI.

## What Not To Introduce

| Avoid | Why |
|-------|-----|
| A new parser generator | SKF parsing is line-oriented and already partially implemented; a parser generator adds surface area without reducing the hard part, which is format convention and validation. |
| SymPy-generated runtime formulas | Symbolic generation can help offline, but runtime formulas should be explicit PyTorch expressions so derivatives, shapes, and sign conventions are inspectable. |
| JAX/Numba/C++/Triton for f blocks | Correctness is the active risk, not speed. A compiled path would make debugging channel/order mistakes harder. |
| A new orbital abstraction layer before f works | The codebase is long and explicit. Introduce only small helpers that remove real duplication after tests pass. |
| DFTB+ as a test dependency | Useful oracle, poor required dependency. Store reference artifacts instead. |
| Mixing parameter sets | DFTB.org warns parameter sets generally should not be mixed; use one consistent fixture set per validation case. |
| Public API churn | This milestone extends internals; users should still construct `Constants`, `Structure`, and `ESDriver` the same way. |

## Prototype Plan

1. **Parser and channel contract**
   - Finish/verify 20-to-40 channel normalization in `_bond_integral.py`.
   - Add tests that compare raw fixture rows to normalized channels.
   - Confirm homonuclear f metadata and simple-file zero-filled f channels.

2. **Constants and structure metadata**
   - Register f fields in `Constants.py`.
   - Extend shell/orbital counts through `Structure.py`.
   - Add shape tests for systems containing Eu/Ga/N fixtures.

3. **H/S block dimensions without full f formulas**
   - Teach `_h0ands.py` about `n_orb == 16`.
   - Fail clearly if an f pair reaches `_slater_koster_pair.py` before formulas are implemented.
   - This creates a clean checkpoint and prevents silent wrong matrices.

4. **f angular transformations**
   - Add explicit s-f, p-f, d-f, f-f value blocks and derivative blocks.
   - Validate axis-aligned cases first.
   - Then validate selected fixture-derived H/S entries.

5. **Simulation reference**
   - Run one minimal f-orbital reference case from stored paper/DFTB+ values.
   - Preserve `tests/test_scf.py` and simple-format tutorial outputs.

## Troubleshooting Guidance

When results are wrong, debug in this order:

1. Raw SKF row token count and format detection.
2. Channel index mapping in `_CHANNELS`.
3. Pair direction: `IJ_pair_type` vs `JI_pair_type`.
4. Orbital offsets from `H_INDEX_START`.
5. Axis-aligned angular block values.
6. Matrix symmetry enforcement in `_h0ands.py`.
7. SCF settings and charge convergence.

Avoid diagnosing SCF energies before parser, spline, and H/S block tests pass. Most f-orbital failures will be channel order, sign, basis ordering, or shape bugs.

## Sources

- DFTorch planning/codebase docs: `.planning/PROJECT.md`, `.planning/codebase/STACK.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/TESTING.md` (HIGH).
- DFTorch source: `src/dftorch/_bond_integral.py`, `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/_h0ands.py`, `src/dftorch/_slater_koster_pair.py`, `pyproject.toml` (HIGH).
- DFTB.org parameter introduction: https://dftb.org/parameters/introduction.html (MEDIUM).
- DFTB+ Recipes first calculation: https://dftbplus-recipes.readthedocs.io/en/stable/basics/firstcalc.html (MEDIUM).
- Slater-Koster file format technical guide mirror: https://studylib.net/doc/26065153/slakoformat (MEDIUM; use because the original PDF was referenced by DFTB.org/user-list material but not directly fetchable here).
- DFTB+ user-list format discussion: https://mailman.zfn.uni-bremen.de/pipermail/dftb-plus-user/2019/002918.html (LOW alone; use only as a convention warning).
- ASE DFTB calculator source docs: https://ase-lib.org/_modules/ase/calculators/dftb.html (MEDIUM for optional reference-output extraction).

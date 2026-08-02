# Library Output Inventory

Every `print` call shipped inside `src/dftorch/`, classified. Produced for phase 5
decision **D-02**: gate the library's status chatter behind one documented flag while
keeping today's noisy output as the default.

The flag is the `dftorch_params` key **`VERBOSE_LIBRARY_OUTPUT`**, default `True`.
It is read by `dftorch._tools.library_output_enabled` and exposed as
`Constants.verbose_output`.

**The default is noisy on purpose.** D-02 explicitly rejected both a quiet default and
a migration to `logging`. REG-02 and REG-03 require existing simple-format callers and
the tutorial notebook to see unchanged output, so this phase makes the noise
*suppressible*, not absent. Do not "improve" the default.

## How this table was produced

The authoritative occurrence count, re-runnable at any time:

```bash
grep -rn "print(" src/dftorch --include="*.py" | wc -l
```

At the time of writing this returns **182**, and this table has **182** rows. No line in
the package contains two `print(` occurrences, so the line count and the occurrence
count coincide. `tests/test_verbose_flag.py::test_inventory_covers_every_print` walks
the tree, counts the same occurrences, and asserts the two are equal, so a print added
by a later phase without a row here fails the suite rather than accumulating quietly.

The `File`, `Line` and `Printed text` columns are extracted mechanically from source by
walking `src/dftorch/**/*.py` and taking every line containing the substring `print(`:

```python
import pathlib
for p in sorted(pathlib.Path("src/dftorch").rglob("*.py")):
    if "__pycache__" in p.parts:
        continue
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "print(" in line:
            yield p.as_posix(), i + 1, line.strip()
```

The `Classification`, `Why` and `Gated by` columns are the judgement this document
exists to record. Line numbers are accurate as of the end of plan 05-05; re-run the
snippet above if you need current ones.

**Counting caveat.** The gate is a *textual* substring count, so prose that writes the
six characters `print(` inside a docstring or comment would add to the count and appear
to break the test. This is deliberate: the textual rule also captures commented-out
prints, which an AST-based count would silently drop. Write ``print`` rather than
``print()`` in prose within this package.

## Classification vocabulary

Exactly one of four values per row.

| Value | Meaning | Action taken |
| --- | --- | --- |
| `already-gated` | An existing per-call `verbose` or `debug` argument, defaulting to off, already governs it. Silent today. | **Code untouched.** Not re-gated onto the new flag, which would have switched it *on*. |
| `status` | Progress, timing, banner or informational output. | Gated on the new flag. Prints by default. |
| `warning` | A genuine failure, non-convergence, degeneracy or silent-degradation notice. | **Left unconditional.** Survives `VERBOSE_LIBRARY_OUTPUT=False`. |
| `not-runtime` | Not on any calculation path in this installation. | Code untouched. |

### Totals

| Classification | Count |
| --- | ---: |
| `already-gated` | 28 |
| `status` | 56 |
| `warning` | 22 |
| `not-runtime` | 76 |
| **total** | **182** |

### Corrections to the plan-time counts

`05-05-PLAN.md` states the `already-gated` rows should number **27**, from a table
measured on 2026-07-30 over the top-level `src/dftorch/*.py` modules using the heuristic
"an `if verbose` guard appears within the three preceding lines". The measured figure
here is **28**. The whole difference is accounted for:

| Adjustment | Delta | Detail |
| --- | ---: | --- |
| Plan-time false positives | **-2** | `_coulomb_matrix.py:608` and `_coulomb_matrix_batch.py:463` are **unconditional**. Each is preceded by a *commented-out* line reading `# if verbose: print('   LMAX:', LMAX)`, which the three-line text heuristic read as a live guard. Both print today, and both are classified `status` here and gated. |
| `if debug:` sites | **+3** | `_dm_fermi.py:80`, `:113` and `:120` sit under `if debug:` in `dm_fermi(..., debug: bool = False)`. The plan's table counted them as unconditional. They are silent today by virtue of an existing per-call argument that defaults to off, which is precisely what the `already-gated` class exists to protect, so they are classified `already-gated` and left untouched. |

27 - 2 + 3 = 28. The property the criterion protects -- *do not switch on a print that is
off today* -- is preserved exactly: the two reclassified-down rows were never off, and
the three reclassified-up rows remain off.

The plan's table also totals **112** prints rather than 182, because it covers only the
top-level `src/dftorch/*.py` modules and excludes commented-out occurrences. This
document covers the whole tree, including `_legacy/`, `ewald_pme/` and `sedacs/`, because
that is what the row-count test walks.

Two further corrections to the plan's text, both in the direction of less work:

- `_xl_tools.calc_q` and `calc_dq` contain **no prints at all**, and
  `kernel_update_lr` / `kernel_update_lr_os` / `kernel_update_lr_batch` **all** take
  `dftorch_params`. The plan predicted these functions would need a new keyword
  parameter threaded through; none do.
- `MD.MDXL.step` and `MD.MDXLBatch.step` take `dftorch_params`, and every one of
  `MD.py`'s 17 prints lives in one of them. The plan expected to reach them via
  `self.const`.

## Reason codes

Every `warning` and `not-runtime` row carries one of these. `status` and
`already-gated` rows carry one too, for symmetry.

| Code | Applies to | Reason |
| --- | --- | --- |
| `R-STATUS` | `status` | Progress, timing or informational output with no failure content. Suppressing it loses nothing a user needs to trust the result. |
| `R-GATED-VERBOSE` | `already-gated` | Governed by an existing `if verbose:` whose parameter defaults to `False`. Silent today; must stay silent. Re-gating it onto `VERBOSE_LIBRARY_OUTPUT` (default `True`) would turn it **on** and change observed output. |
| `R-GATED-DEBUG` | `already-gated` | Governed by `if debug:` in `_dm_fermi.dm_fermi`, whose `debug: bool = False` parameter makes it silent today. Same reasoning as `R-GATED-VERBOSE`. |
| `R-NOCONV` | `warning` | A non-convergence report: an iteration count reached its maximum, or a residual never fell below tolerance. A user who opts into quiet must still learn their calculation did not converge. |
| `R-DEGEN` | `warning` | A numerical-degeneracy notice in the Krylov acceleration (`zero norm_dr` / `zero norm_vi`). It precedes a `break` that truncates the subspace, so the answer that follows was computed differently from the one the user asked for. |
| `R-DEGRADE` | `warning` | Silent degradation: the calculation continues, but with different physics or a different method than requested. Suppressing it would let a user believe they got what they asked for. |
| `R-UNIMPL` | `warning` | Announces an unimplemented capability and then continues or returns. See the note below -- flagged for plan 05-06. |
| `R-NOOP` | `warning` | Reports that the routine did nothing at all and returned early. Suppressing it would leave a user believing an optimisation ran. |
| `R-COMMENT` | `not-runtime` | The occurrence is inside a `#` comment. It emits nothing. Counted because the sweep is textual; classified `not-runtime` because there is no runtime behaviour to gate. |
| `R-LEGACY` | `not-runtime` | In `src/dftorch/_legacy/`, established unreachable by plan 05-04: the directory has no `__init__.py`, and `grep -rn "H0andS\|_legacy" src tests` outside the directory itself returns only a `pyproject.toml` line that *excludes* it from ruff, plus a generated `SOURCES.txt` entry. |
| `R-SCRIPT` | `not-runtime` | In `_patch_sk.py`, a module-level source-rewriting utility, not a calculation path. Established by measurement: `import dftorch._patch_sk` raises `FileNotFoundError` on the hardcoded absolute path `/home/maxim/Projects/DFTB/DFTorch/src/dftorch/...` baked into the module body. It cannot be imported on this machine at all, let alone called during a calculation. |
| `R-PRINTER` | `not-runtime` | `_tools.list_global_tensors` is an explicitly-invoked diagnostic printer whose printed output *is* its return value -- it returns `None`. `grep -rn "list_global_tensors" src tests` finds only the definition, so it is never called from a calculation path. Gating it would make the function silently useless when called. |
| `R-SEDACS` | `not-runtime` | In `src/dftorch/sedacs/`, which requires the external `sedacs` distribution. Established by measurement: `import dftorch.sedacs` raises `ModuleNotFoundError: No module named 'sedacs'` in this environment; no module outside `src/dftorch/sedacs/` imports it (the eight `grep` hits elsewhere are all docstring or comment prose); and no test references it. See the caveat below. |

## Notes on specific rows

### `_coulomb_matrix_batch.py` -- unimplemented vectorized k-space (flagged for 05-06)

The row classified `R-UNIMPL` announces that vectorized k-space is not implemented for
batched data and then executes a bare `return`, handing the caller `None` where two
tensors were expected. That is a silent-degradation site rather than output, and it is
exactly the pattern decision **D-04** exists to find: announcing an unimplemented
capability and proceeding anyway.

It is classified `warning` and left unconditional here. **Converting it to an explicit
refusal is plan 05-06's call**, to be made with the rest of the orbital-count audit in
view, not a change to make in passing during an output-gating plan.

### The `sedacs` subpackage caveat

`R-SEDACS` establishes that these 35 prints are unreachable *in this installation*. A
user who installs the external `sedacs` distribution would reach them, and they would be
`status` output at that point. They are not gated here because:

- the subpackage cannot be imported, so no change to it could be executed by any test in
  this repository -- gating it would be an unverifiable edit;
- SEDACS is explicitly out of Phase 5's scope (`05-CONTEXT.md` lists it under v2).

If SEDACS support is ever brought in-tree, these rows are the work list.

### `Optimizer.py` -- two warnings, four status

`GeoOpt.run` opens with a guard that reports "No free atoms and no cell relaxation" and
returns immediately (`R-NOOP`), and closes with "Not converged after N steps"
(`R-NOCONV`). Both stay unconditional. The four in between -- the step banner, the
per-step info line, the convergence confirmation and the step timing -- are `status`.

## Inventory

| File | Line | Printed text | Classification | Why | Gated by |
| --- | ---: | --- | --- | --- | --- |
| `src/dftorch/Constants.py` | 193 | `print( "Warning: could not load spinw.txt file for spin-orbit coupling. Proceeding without SOC." )` | `warning` | R-DEGRADE | - |
| `src/dftorch/Constants.py` | 256 | `print(f"DFTB3: {self.dftb3}")` | `status` | R-STATUS | self.verbose_output |
| `src/dftorch/ESDriver.py` | 585 | `# print(f"GS charges: {structure.q}")` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/ESDriver.py` | 588 | `# print("detecting delta SCF calculation")` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/ESDriver.py` | 638 | `# print(f"ES charges: {structure.q}")` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/ESDriver.py` | 1310 | `print("Reference GBSA computed (frozen geometry approximation)")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/ESDriver.py` | 1316 | `print( f"FD Hessian ({mode}): {N} atoms, {n3} DOFs, δ = {delta} Å\n" f" batch_size={batch_size} → {dofs_p...` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/ESDriver.py` | 1400 | `print( f" batch {batch_idx + 1}/{n_batches} DOFs {dof_start}-{dof_end - 1}" f" ({elapsed:.0f}s elapsed, ~...` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/ESDriver.py` | 1415 | `print( f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min) " f"max\|H-Hᵀ\| = {sym_err:.4e}" )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/ESDriver.py` | 1733 | `print(f"GBSA initialization time: {toc - tic:.2f} seconds")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/ESDriver.py` | 1914 | `print(f"GBSA SASA gradient calculation time: {toc - tic:.2f} seconds")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/ESDriver.py` | 1921 | `print(f"GBSA Born gradient calculation time: {toc - tic:.2f} seconds")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 486 | `print( "########## Step = {:} ##########".format( md_step, ) )` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 658 | `print("H0: {:.3f} s".format(time.perf_counter() - tic2_1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 857 | `print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 934 | `print("KER: {:.3f} s".format(time.perf_counter() - tic3))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1174 | `print("F AND E: {:.3f} s".format(time.perf_counter() - tic4))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1183 | `print( "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, NS = {:.4f}, ResErr = {:.6f}{}, t = {:.1...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1196 | `print( "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}{}, t = {:.1f} s".format(...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1207 | `print(torch.cuda.memory_allocated() / 1e9, "GB\n")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1208 | `print()` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1575 | `print( "########## Step = {:} ##########".format( md_step, ) )` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1709 | `print("H0: {:.3f} s".format(time.perf_counter() - tic2_1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1796 | `print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 1850 | `print("KER: {:.3f} s".format(time.perf_counter() - tic3))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 2005 | `print("F AND E: {:.3f} s".format(time.perf_counter() - tic4))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 2014 | `print( "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}{}, t = {:.2f} s".format(...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 2025 | `print(torch.cuda.memory_allocated() / 1e9, "GB\n")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/MD.py` | 2026 | `print()` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/Optimizer.py` | 340 | `print("GeoOpt: No free atoms and no cell relaxation — nothing to optimise.")` | `warning` | R-NOOP | - |
| `src/dftorch/Optimizer.py` | 373 | `print(f"═══════════ GeoOpt step {step} ═══════════")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/Optimizer.py` | 422 | `print(info)` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/Optimizer.py` | 441 | `print(f"\n✓ Converged in {step + 1} steps.")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/Optimizer.py` | 503 | `print(f" step time: {dt:.2f} s\n")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/Optimizer.py` | 506 | `print( f"\n✗ Not converged after {max_steps} steps. " f"Fmax = {fmax_val:.6f} eV/Å" )` | `warning` | R-NOCONV | - |
| `src/dftorch/_coulomb_matrix.py` | 163 | `print("coulomb_matrix_vectorized")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix.py` | 165 | `print(" Do Coulomb Real")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_coulomb_matrix.py` | 194 | `print(" Coulomb_Real t {:.1f} s".format(time.perf_counter() - start_time1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix.py` | 202 | `print(" Doing Coulomb k")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_coulomb_matrix.py` | 206 | `print(" Coulomb_k t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix.py` | 557 | `print(" init L,M,N,K")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix.py` | 607 | `# if verbose: print(' LMAX:', LMAX)` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_coulomb_matrix.py` | 608 | `print(" LMAX:", LMAX)` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix.py` | 611 | `print(" ", L)` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_coulomb_matrix_batch.py` | 459 | `print("vectorized k-space is not implemented for batched data")` | `warning` | R-UNIMPL | - |
| `src/dftorch/_coulomb_matrix_batch.py` | 462 | `# if verbose: print(' LMAX:', LMAX)` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_coulomb_matrix_batch.py` | 463 | `print(" LMAX:", LMAX)` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_coulomb_matrix_batch.py` | 466 | `print(" ", L)` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_coulomb_matrix_batch.py` | 482 | `# print(K2, KCUTOFF2)` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_dm_fermi.py` | 80 | `print(" eigh {:.1f} s".format(time.perf_counter() - start_time1))` | `already-gated` | R-GATED-DEBUG | - |
| `src/dftorch/_dm_fermi.py` | 106 | `print( "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format( MaxIt, OccErr ) )` | `warning` | R-NOCONV | - |
| `src/dftorch/_dm_fermi.py` | 113 | `print(" dm ptr {:.1f} s".format(time.perf_counter() - start_time1))` | `already-gated` | R-GATED-DEBUG | - |
| `src/dftorch/_dm_fermi.py` | 120 | `print(" v*p0*v.T {:.1f} s".format(time.perf_counter() - start_time1))` | `already-gated` | R-GATED-DEBUG | - |
| `src/dftorch/_dm_fermi_x.py` | 108 | `print( "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format( MaxIt, occ_err_val ) )` | `warning` | R-NOCONV | - |
| `src/dftorch/_dm_fermi_x.py` | 179 | `print( "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format( MaxIt, occ_err_val ) )` | `warning` | R-NOCONV | - |
| `src/dftorch/_dm_fermi_x.py` | 251 | `print( "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format( MaxIt, occ_err_val ) )` | `warning` | R-NOCONV | - |
| `src/dftorch/_dm_fermi_x.py` | 321 | `print( "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format( MaxIt, occ_err_val ) )` | `warning` | R-NOCONV | - |
| `src/dftorch/_dm_fermi_x.py` | 384 | `print( "Warning: dm_fermi_batch_degen did not converge in {} iterations, occ error = {}".format( MaxIt, o...` | `warning` | R-NOCONV | - |
| `src/dftorch/_h0ands.py` | 219 | `print("H0_and_S")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 224 | `print(" Do H off-diag")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 318 | `print( " t <dR and pair mask> {:.1f} s\n".format( time.perf_counter() - start_time3 ) )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 336 | `print(" Using ML model for SK integrals (lazy per-call)")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 361 | `print(" t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 364 | `print(" Do H and S")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 474 | `print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 576 | `print("H0_and_S")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 583 | `print(" Do H off-diag")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 659 | `print( " t <dR and pair mask> {:.1f} s\n".format( time.perf_counter() - start_time3 ) )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 671 | `print(" t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 674 | `print(" Do H and S")` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_h0ands.py` | 761 | `print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_kernel_fermi.py` | 72 | `print("Building kernel row ", J + 1, " of ", Nr_atoms)` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_legacy/H0andS.py` | 122 | `print("H0_and_S")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 127 | `print(" Do H off-diag")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 193 | `print( " t <dR and pair mask> {:.1f} s\n".format( time.perf_counter() - start_time3 ) )` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 205 | `print(" t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 209 | `print(" Do H and S")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 286 | `print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 388 | `print("H0_and_S")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 395 | `print(" Do H off-diag")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 457 | `print( " t <dR and pair mask> {:.1f} s\n".format( time.perf_counter() - start_time3 ) )` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 469 | `print(" t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 473 | `print(" Do H and S")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 560 | `print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 621 | `print("H0_and_S")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 640 | `print(" Load H integral params")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 695 | `# print(' Do H diagonal')` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 736 | `print(" Do H off-diag")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 813 | `print(" Do H Slater-Koster")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 847 | `print(" Load S integral params")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 865 | `print(" Do S Slater-Koster")` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 902 | `# print(D0-D0_)` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 903 | `# print(torch.max(abs(D0-D0_)))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_legacy/H0andS.py` | 907 | `print(" t {:.1f} s\n".format(time.perf_counter() - start_time1))` | `not-runtime` | R-LEGACY | - |
| `src/dftorch/_nearestneighborlist.py` | 278 | `print( " t <neighbor list, alchemi> {:.2f} s\n".format( time.perf_counter() - start_time1 ) )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_nearestneighborlist.py` | 402 | `print( " t <neighbor list> {:.2f} s\n".format(time.perf_counter() - start_time1) )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_nearestneighborlist.py` | 618 | `print( f" t <batched neighbor list> {time.perf_counter() - start_time1:.3f} s (B={B}, N={N})" )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_nearestneighborlist.py` | 657 | `print( f" t <batched neighbor list> {time.perf_counter() - start_time1:.3f} s (B={B}, N={N})" )` | `already-gated` | R-GATED-VERBOSE | - |
| `src/dftorch/_patch_sk.py` | 55 | `print(f"First function ends at line {first_func_end}")` | `not-runtime` | R-SCRIPT | - |
| `src/dftorch/_patch_sk.py` | 159 | `print( f" Block at line {coeffs_line}: ch={channel}, var={var_name}, dir={direction}, mask={mask_expr}, h...` | `not-runtime` | R-SCRIPT | - |
| `src/dftorch/_patch_sk.py` | 167 | `print(f"\nFound {len(blocks)} polynomial blocks in first function")` | `not-runtime` | R-SCRIPT | - |
| `src/dftorch/_patch_sk.py` | 194 | `print(f"\nWrote patched file ({len(lines)} lines)")` | `not-runtime` | R-SCRIPT | - |
| `src/dftorch/_scf.py` | 267 | `print("### Do _scf ###")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 330 | `print(" Initial dm_fermi")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 345 | `print(" Initial mu = {:.4f}".format(mu0.item()))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 379 | `print("\nStarting cycle")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 386 | `print("Iter {}".format(it))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 500 | `# print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), tor...` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 501 | `print( "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format( ResNorm.item(), dEc.item(), time.perf_counter...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 507 | `print("Did not converge")` | `warning` | R-NOCONV | - |
| `src/dftorch/_scf.py` | 586 | `print("### Do _scf ###")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 654 | `print(" Initial dm_fermi")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 709 | `# print(f" Broken symmetry: perturbed shells {atom_shells.tolist()} " # f"on atom {most_polarizable_atom....` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 746 | `print("\nStarting cycle")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 753 | `print("Iter {}".format(it))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 882 | `# print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), tor...` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 883 | `print( "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format( ResNorm.item(), dEc.item(), time.perf_counter...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 889 | `print("Did not converge")` | `warning` | R-NOCONV | - |
| `src/dftorch/_scf.py` | 1041 | `print(" Initial mu", mu0)` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1055 | `print(" Using q_init (skipping initial dm_fermi)")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1083 | `print("\nStarting cycle")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1088 | `print("Iter {}".format(it))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1180 | `print(f"Batch {b}: Res = {rval:.3e}, dEc = {dval:.3e}")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1181 | `# print(f"t = {elapsed:.2f} s")` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 1184 | `print("Did not converge")` | `warning` | R-NOCONV | - |
| `src/dftorch/_scf.py` | 1244 | `print("### Do Delta_scf ###")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1312 | `print(" Initial dm_fermi")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1355 | `# print(f" Broken symmetry: perturbed shells {atom_shells.tolist()} " # f"on atom {most_polarizable_atom....` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 1379 | `print("\nStarting cycle")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1386 | `print("Iter {}".format(it))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1503 | `# print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), tor...` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/_scf.py` | 1504 | `print( "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format( ResNorm.item(), dEc.item(), time.perf_counter...` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_scf.py` | 1510 | `print("Did not converge")` | `warning` | R-NOCONV | - |
| `src/dftorch/_tools.py` | 304 | `print( f"{name:>24} size={bytes_ / 1e6:9.2f} MB shape={tuple(t.shape)} dtype={t.dtype} device={t.device} ...` | `not-runtime` | R-PRINTER | - |
| `src/dftorch/_tools.py` | 307 | `print(f"Total tensors: {len(rows)} total size={total / 1e6:.2f} MB")` | `not-runtime` | R-PRINTER | - |
| `src/dftorch/_tools.py` | 376 | `print( f"[{context}] Non-periodic system with COUL_METHOD='PME' detected. " "Switching to COUL_METHOD='FU...` | `warning` | R-DEGRADE | - |
| `src/dftorch/_xl_tools.py` | 563 | `print("zero norm_dr")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 576 | `print("zero norm_vi")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 649 | `print(" rank: {:}, Fel = {:.6f}".format(krylov_rank, Fel.item()))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_xl_tools.py` | 781 | `print("zero norm_dr")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 916 | `print(" rank: {:}, Fel = {:.6f}".format(krylov_rank, Fel.item()))` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_xl_tools.py` | 1003 | `print("zero norm_dr")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 1011 | `print("zero norm_dr")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 1027 | `print("zero norm_vi")` | `warning` | R-DEGEN | - |
| `src/dftorch/_xl_tools.py` | 1113 | `print(f" rank: {krylov_rank}, batch {b}, Fel = {val:.6f}")` | `status` | R-STATUS | *(task 2)* |
| `src/dftorch/_xl_tools.py` | 1114 | `print(f" Not converged: {active_mask.sum()}")` | `warning` | R-NOCONV | - |
| `src/dftorch/ewald_pme/neighbor_list.py` | 235 | `# print(dist.abs().max(), self.reneighbor_cnt)` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/sedacs/MD.py` | 195 | `print("########## Step = {:} ##########".format(md_step))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 198 | `print("Graph repartition")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 212 | `# print("Rank", dist.get_rank(), "has core and core-halo size:", core_size[i].item(), len(ch[i]))` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/sedacs/MD.py` | 225 | `print( f"CH sizes (all ranks): max={int(g_max)}, min={int(g_min)}, avg={g_sum / g_cnt:.1f}" )` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 426 | `print("PME: {:.3f} s".format(time.perf_counter() - tic2_1))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 513 | `print("mu0: {:.3f} s".format(time.perf_counter() - tic2_1))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 586 | `print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 611 | `print("KER: {:.3f} s".format(time.perf_counter() - tic2_1))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 697 | `print("F AND E: {:.3f} s".format(time.perf_counter() - tic2_1))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/MD.py` | 700 | `print( comm_string + ", t = {:.1f} s".format( time.perf_counter() - start_time, ) )` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 69 | `print("Initial mu", mu0)` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 104 | `print("\nSCF iteration", scf_iter)` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 111 | `# print( # "Rank", # dist.get_rank(), # "has core and core-halo size:", # core_size[i].item(),` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/sedacs/SCF.py` | 130 | `print( f" CH sizes (all ranks): max={int(g_max)}, min={int(g_min)}, avg={g_sum / g_cnt:.1f}" )` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 160 | `print("scf error = {:.9f}".format(scf_error.item()))` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 211 | `print(timing)` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 213 | `print("TOTAL:", timing["TOTAL"])` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 215 | `print("Doing Band Energy...")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 225 | `print("Doing Repulsion Energy...")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 242 | `print("Doing Coulomb Energy...")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 266 | `print(f"{'Energy Components':─^52}")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 267 | `print(f" {'E_band':22s} {e_band0 * Ha:12.6f} Ha ({e_band0:12.6f} eV)")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 268 | `print(f" {'E_coulomb':22s} {e_coul * Ha:12.6f} Ha ({e_coul:12.6f} eV)")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 269 | `print( f" {'E_repulsion':22s} {e_repulsion * Ha:12.6f} Ha ({e_repulsion:12.6f} eV)" )` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 272 | `print( f" {'E_entropy (-TS)':22s} {e_entropy * Ha:12.6f} Ha ({e_entropy:12.6f} eV)" )` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/sedacs/SCF.py` | 275 | `# print(f" {'net spin':22s} {structure1.net_spin_sr.sum().item():12.4f}")` | `not-runtime` | R-COMMENT | - |
| `src/dftorch/sedacs/SCF.py` | 276 | `# print(f" {'E_spin':22s} {structure1.e_spin.item()*Ha:12.6f} Ha ({structure1.e_spin.item():12.6f} eV)")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 277 | `print(f"{'─' * 52}")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/SCF.py` | 291 | `print(f" {'E_total (F=E-TS)':22s} {e_tot * Ha:12.6f} Ha ({e_tot:12.6f} eV)")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 148 | `print(f"Graph partitioning time: {time.perf_counter() - tick:.1f} s")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 152 | `print("\nCore and halos indices for every part:")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 158 | `print( "Rank", dist.get_rank(), "has core and core-halo size:", n_core, len(core_halo), )` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 725 | `print("zero norm_dr")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 738 | `print("zero norm_vi")` | `not-runtime` | R-SEDACS | - |
| `src/dftorch/sedacs/sedacs_interface.py` | 921 | `print(" Krylov rank: {:}, Fel = {:.7f}".format(krylov_rank, Fel.item()))` | `not-runtime` | R-SEDACS | - |

# Phase 03 — Deferred / Out-of-Scope Items

Discovered during execution of `03-01-PLAN.md`. Logged, not fixed (scope boundary).

## 1. `tests/test_scf.py::test_energy_smoke_import_and_call` fails on this machine

- **Status:** Pre-existing. Reproduced identically against a pristine `git archive HEAD`
  export, so it is **not** caused by Phase 3 changes.
- **Failure:** `RuntimeError: expected scalar type Double but found Float` raised from
  `src/dftorch/ewald_pme/PME_torch.py:264` in `calculate_PME_kspace_stress`
  (`torch.einsum("ijk,ijk,ijkab->ab", E_G, g_mask, metric)`).
- **Cause (unconfirmed):** the PME k-space `metric` tensor is built as float32 while the
  test sets `torch.set_default_dtype(torch.float64)`. Likely a torch-version behavior
  change; this environment has torch 2.11.0+cpu.
- **Why deferred:** unrelated to H0/S routing or SK channel addressing; fixing PME dtype
  handling is outside the Phase 3 boundary.

## 2. Local environment cannot run the plan's literal verification command

- `uv` is not installed on this machine and `pytest` is not installed in any available
  interpreter (Python 3.13 / 3.12). `dftorch` is also not installed as a package.
- Prior phase summaries were produced on a different machine (the plan's own
  `<execution_context>` references `/Users/abavera/.codex/...`).
- Installing `pytest`/`uv` is a package-manager install, which this plan's threat register
  (`T-03-SC`) and the executor's deviation rules exclude from auto-fix. Left for the user.

## 3. A single `R_orb` grid is shared by every pair type

- `_bond_integral.get_skf_tensors` keeps a per-pair `R_tensor` but exports one global
  `R_orb` (`R_orb_master`, the *longest* grid seen). `_h0ands.H0_and_S_vectorized` then
  computes `idx`/`dx` from that single grid for all pair types.
- In `tests/f_orbital_data` the grids genuinely differ: `Eu-*` and `N-Eu` are tabulated on
  `0.04 A / 433` points while `Ga-Ga`, `Ga-N`, `N-Ga` and `N-N` use the simple format with a
  different grid. `R_orb` ends up 483 points long, so for the 433-point tables the interval
  index and the `dx` offset are taken from the wrong grid.
- **Impact:** radial *values* for mixed-grid SKF directories can be evaluated at the wrong
  knot. Phase 3's tests therefore assert the *angular* structure and AO placement, deriving
  their expected radial factor from the same `const.R_orb` + `coeffs_tensor` lookup the
  implementation uses, so they stay valid either way.
- **Why deferred:** this is parser/`_bond_integral` territory (Phase 1 contract), not H0/S
  routing, and fixing it means changing the `coeffs_tensor`/`R_orb` interface used by the
  ML path and the stress path as well.

## 4. Extremely large short-range spline coefficients in the extended f tables

- After the leading-zero truncation fix (see the Phase 3 summary, deviation Rule 3), the
  `Eu-Eu` cubic coefficients peak at ~1.4e8 around `r = 0.635 A`, immediately after the 13
  padded zero rows end at `r = 0.56 A`.
- Evaluated values at physical bond lengths (2-5 A) are well behaved, so this does not
  affect Phase 3 results, but a spline fitted straight across the zero/data boundary is
  numerically delicate and may deserve a smoothing or a hard inner cutoff.
- **Why deferred:** short-range spline conditioning is a `_bond_integral` concern.

## 5. `src/dftorch/_patch_sk.py` still encodes the legacy channel scheme

- `_patch_sk.py` is a code-generation/patch helper that rewrites `_slater_koster_pair.py`
  and emits `ch = channel + SH_shift * 10`.
- It is not imported at runtime, so it cannot corrupt results today, but re-running it
  would reintroduce the channel-addressing bug fixed in `edf73e4`.
- Deferred: not in this plan's file scope.

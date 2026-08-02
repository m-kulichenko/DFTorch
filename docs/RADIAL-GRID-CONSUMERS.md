# Radial Grid Consumers: disposition record

**Requirement:** REG-06 (per-pair Slater-Koster radial lookup), decision D-01.
**Written:** plan 05-04, measured 2026-08-01.
**Companion:** REG-05's mixed-step refusal, `_bond_integral._require_uniform_grid_step`.

## Why this document exists

`_bond_integral.get_skf_tensors` stores every element pair's own radial grid in
`R_tensor`, but also exports a single global `R_orb` chosen as *the longest grid seen in
the directory*. Any consumer that derives a spline knot from `R_orb` is measuring every
pair against one pair's ruler.

Plan 05-01 fixed the live single-system H0/S lookup and kept `R_orb` exported
additively, so the other consumers kept working unchanged. "Kept working unchanged" is
not a disposition — it is the absence of one, and an unconverted consumer with no record
is later assumed correct by whoever reads it next (threat T-05-18). This file gives every
remaining consumer an explicit verdict with a reason.

## How to check this table is still complete

```
grep -rn "R_orb" src/dftorch --include="*.py"
```

**69 hits at the time of writing**, distributed as:

| File | Hits |
|---|---|
| `src/dftorch/_bond_integral.py` | 19 |
| `src/dftorch/_h0ands.py` | 15 |
| `src/dftorch/_ml_sk.py` | 14 |
| `src/dftorch/_legacy/H0andS.py` | 13 |
| `src/dftorch/Constants.py` | 3 |
| `src/dftorch/ESDriver.py` | 2 |
| `src/dftorch/_stress.py` | 2 |
| `src/dftorch/sedacs/sedacs_interface.py` | 1 |
| **Total** | **69** |

Most hits are prose (docstrings, parameter descriptions, the explanatory comments plans
05-01 and 05-04 added) or are the producer side inside `get_skf_tensors`. The table below
covers the **9 read sites** — every place that indexes the exported grid, measures it, or
hands it to something that will. If the grep count moves, re-classify the new lines and
add a row here.

## Read sites and dispositions

| # | Site | What it reads the grid for | Disposition | Reason |
|---|---|---|---|---|
| 1 | `ESDriver.py:266-286` | passes `const.R_orb` into the single-system `H0_and_S_vectorized` | **converted** | The same call now also passes `R_tensor=const.R_tensor` and `n_grid=const.n_grid`, which selects `_pair_knot_lookup`. `R_orb` is still passed but is not consumed on that branch. See the correction note below: this wiring had been reverted and was restored by plan 05-04. |
| 2 | `_h0ands.py:353-355` | `searchsorted(R_orb, dR_mskd)` / `clamp` / `dR_mskd - R_orb[idx]`, the single-system fallback branch | **converted** | Live callers take the per-pair branch at `:350-351`. The global expression is retained verbatim as the `R_tensor is None or n_grid is None` fallback so out-of-scope callers (row 7) keep running instead of failing on a missing argument. Retained, not used. |
| 3 | `ESDriver.py:1527` | passes `const.R_orb` into `H0_and_S_vectorized_batch` | **deferred** — **PHY-04, Phase 8.1** | The batched path refuses f systems outright, so it cannot currently reach a directory whose grids differ in the way that matters. Converting it means widening `H0_and_S_vectorized_batch`'s signature, which plan 05-01 explicitly scoped out. `_pair_knot_lookup` is module-level and directly reusable when batch f support lands. |
| 4 | `_h0ands.py:666-668` | the batched knot lookup itself | **deferred** — **PHY-04, Phase 8.1** | Same path as row 3, seen from the callee side. |
| 5 | `_stress.py:183` | `dx = dR - const.R_orb[idx]`, using an `idx` produced upstream and handed over in `metadata` | **deferred** — **PHY-02, Phase 8** | A derivative-consuming path that already raises `FDerivativeUnsupportedError` for any 16-orbital atom ten lines later (`_stress.py:197`). It also does not compute `idx` itself, so converting it requires converting whoever fills `metadata["idx"]` at the same time; that is a joint change belonging to the f stress work. |
| 6 | `_ml_sk.py:489-500` | `build_pair_type_rcut`: `dr = R_orb[1] - R_orb[0]`, `R_orb[-1]`, and the output `dtype`/`device` | **deferred** — **PHY-06, v2** | The cutoff table is a per-pair-*type* quantity derived from a single step. For a directory whose files share one step -- which REG-05's guard now *enforces* -- taking that step from the global grid gives the same answer as taking it from any pair's row, so the result is already correct rather than merely tolerated. ML-SK f support is PHY-06 and unscheduled. |
| 7 | `sedacs/sedacs_interface.py:523` | passes `ch_structure.const.R_orb` into `H0_and_S_vectorized` | **deferred** — **PHY-05, v2** | SEDACS is v2 scope. Plan 05-01's `None` defaults on `R_tensor`/`n_grid` are what keep this call site working untouched: it omits both and therefore runs row 2's retained global expression. |
| 8 | `_legacy/H0andS.py:200-202` | a dead copy of the global knot lookup | **unreachable** | Established by `grep -rn "H0andS\|_legacy" src tests` (excluding the directory itself): the only hits are `pyproject.toml:61`, which *excludes* the directory from ruff, and a generated `SOURCES.txt` entry. The directory has no `__init__.py`. Nothing imports it. |
| 9 | `_legacy/H0andS.py:464-466` | a second dead copy | **unreachable** | Same evidence as row 8. |

Not read sites, listed so the classification is reproducible rather than assumed:

- `_bond_integral.py` (19 hits) — the **producer**. `R_orb_i` is the per-file grid as read;
  `R_orb_master` selects the longest; `R_orb` is returned. Nothing here consumes the
  exported grid.
- `Constants.py:109` and `:164` — **plumbing**: the tuple unpack from `get_skf_tensors`
  and the `torch.nn.Parameter` registration. `Constants.py:159` is a comment.
- `_h0ands.py:21, 106, 156, 165, 168, 341, 494, 530, 534` and `_ml_sk.py:434, 438-481, 548`
  and `_legacy/H0andS.py:30, 76, 80, 306, 342, 346, 585` and `_stress.py:179` — signatures,
  parameter documentation and comments.

## Correction: row 1 was NOT converted when this audit began

Plan 05-01's summary states that `ESDriver`'s single-system caller supplies `R_tensor` and
`n_grid`. It did when 05-01 landed (`1d5acce`). Commit `56091af` — a commit whose stated
purpose was an independent SKF test oracle — deleted both keyword arguments along with
their comment, as an unrelated side effect.

Nothing failed. All thirteen of plan 05-01's tests stayed green, because every one of them
exercises `_pair_knot_lookup` or `H0_and_S_vectorized` directly and none goes through the
driver. So REG-06 was implemented, tested, marked complete, and then **inert in every real
calculation** for eleven commits: `H0_and_S_vectorized` takes its global fallback whenever
either argument is `None`.

Plan 05-04 restored the two arguments and added
`tests/test_radial_grid.py::test_esdriver_supplies_the_per_pair_grid_arguments`, which
records the keyword arguments the driver actually passes. Confirmed by mutation: deleting
the two lines again makes that test fail and names the missing argument.

The general lesson is worth more than the specific fix. A `None`-defaulted fallback that
silently produces a plausible answer cannot be protected by tests that call the callee
directly. It needs a test that watches the **wiring**.

## Units, settled by measurement

The grid and the pair distances are **both in Angstrom**. The SKF file declares its step in
Bohr; `_bond_integral.read_skf_table` builds the grid as
`arange(1, npts_pad + 1) * step * BOHR_TO_ANGSTROM`, and nothing downstream converts back.

`_ml_sk.build_pair_type_rcut`'s docstring previously claimed `R_orb` was Angstrom while
`dR_mskd` was Bohr, and described the comparison as deliberately mixed-unit. That was
wrong, and it was contradicted eleven lines below in the same file. It was settled by
measurement, not argument: a CH4 C-H separation of 1.0566812742799978 A against mio-1-1's
0.0105835442 A step lands on `idx = 98` where `R_orb[98] = 1.0477708758` A, the correct
physical knot for that bond. Under the Bohr reading the knot would sit at roughly twice
the true separation.

Plan 05-04 rewrote that docstring, recorded the superseded claim inside it rather than
deleting it, and gated both halves with
`tests/test_radial_grid.py::test_rcut_values_unchanged_after_docstring_fix`.

## What REG-05's guard changes for the deferred rows

Rows 3-7 all read a single global grid. Before plan 05-04, a directory mixing radial grid
*steps* would have given each of them a ruler belonging to one arbitrary pair. That
directory can no longer be loaded at all: `_require_uniform_grid_step` refuses it in
`get_skf_tensors`, naming the offending files and both steps.

This does not make the deferrals unnecessary — a mixed-*length* directory is legitimate,
loads, and still leaves rows 3-7 clamping against the longest grid. It does bound their
consequence to the benign case, which is why deferring them is defensible rather than
merely convenient.

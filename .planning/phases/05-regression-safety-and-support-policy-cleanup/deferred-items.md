# Phase 5 deferred items

Out-of-scope discoveries logged during execution. Nothing here was fixed.

---

## 1. `water8_mio_full` does not reproduce its pinned energy bit-exactly

**Found during:** plan 05-01, Task 2 verification (2026-07-31).
**Owner:** plan 05-02's baseline (`tests/test_simple_format_regression.py:246`).
**Status:** open, out of scope for 05-01. Not a defect introduced by D-01.

`tests/test_simple_format_regression.py` pins the 8-water case at

    e_tot = -110.65745489032126 eV     (measured on commit db63487)

The value measured on this machine on 2026-07-31 is

    e_tot = -110.65745489032182 eV     (delta -5.542233338928781e-13 eV)

That is 5e-15 relative, comfortably inside 05-02's own `1e-8` eV energy band,
so `test_total_energy_is_pinned[water8_mio_full]` passes.

**This was verified NOT to be caused by plan 05-01.** Three independent checks:

1. With the per-pair lookup active and with it disabled (`const.n_grid = None`,
   which selects the verbatim pre-D-01 global expression), the energy is the
   same value to the last bit: `perpair_eq_global = True`.
2. Both paths are repeatable across repeated runs in one process.
3. A pristine copy of `src/` reconstructed from commit `742f8cc` (identical
   source to `130dbea`, since 742f8cc is test-only) and imported via
   `PYTHONPATH` also measures `-110.65745489032182`, confirmed by
   `hasattr(const, "n_grid") == False`.

The two O2 cases (`o2_mio_unrestricted`, `o2_3ob_dftb3`) reproduce their pinned
energies bit-exactly both before and after this plan.

**Likely cause:** an environment-level numeric difference between the machine
state at `db63487` and now. `water8_mio_full` is the only case of the three that
is 24 atoms with a periodic cell and `COUL_METHOD = "FULL"`, so it is the only
one whose energy passes through a large symmetric eigensolve and a full Coulomb
sum, both of which are sensitive to BLAS/LAPACK threading and kernel selection.
The two 2-atom cases are far too small to expose that.

**What should happen:** whoever owns the REG-02 baseline should decide whether
to re-measure the literal on the current environment (and record the reason), or
to leave it and document that this case is reproducible only to its stated
`1e-8` eV band rather than bit-exactly. Do NOT widen the band; the band is
already correct and already passes.

---

## 2. The CH4 derivative checksum misses its exact literal by 3e-13

**Found during:** plan 05-03, Task 1 and Task 2 full-suite verification
(2026-07-31).
**Owner:** plan 05-01's exact radial-lookup regression gate
(`tests/test_radial_grid.py:184`).
**Status:** open, pre-existing and out of scope for 05-03. Not caused by the
inventory document or deletion of `src/dftorch/script.py`.

Both of these tests fail at the same assertion:

- `test_ch4_h0_s_checksums_are_unchanged`
- `test_ch4_h0_s_bit_identical_after_per_pair_lookup`

The recorded literal and value measured on this machine are:

    CH4_DH0_ABS_SUM = 777.8475256952822
    measured         = 777.8475256952825
    absolute delta   = 3.410605131648481e-13

The failure reproduces when the two tests are run in isolation. The per-pair
and global-fallback derivative tensors still compare bit-identical to each
other inside `test_ch4_h0_s_bit_identical_after_per_pair_lookup`; only the
historical scalar literal differs. Plan 05-03 changed no Hamiltonian,
derivative, or radial-grid source, so changing either the literal or production
arithmetic here would cross the plan boundary.

**What should happen:** plan 05-01's regression owner should determine which
environment produced the original scalar, then either restore that environment
or re-record the literal with an explicit reproducibility decision. Plan 05-03
must not weaken or rewrite a bit-identity gate it does not own.

---

## 3. D-02's pre-count of remaining runtime `print()` calls is stale

**Found during:** plan 05-03, Task 2 deletion verification (2026-07-31).
**Owner:** plan 05-05 (D-02 library-output classification).
**Status:** open plan-time drift; out of scope for 05-03.

Deleting `src/dftorch/script.py` removed exactly 57 `print(` occurrences, as
promised. The count that remains at the current baseline is not the 112 stated
in 05-CONTEXT.md:

    git show 4d158dc:src/dftorch/script.py | rg -o 'print\(' | wc -l
    57

    rg -o 'print\(' src/dftorch/*.py | wc -l
    124 across 15 top-level runtime modules

    git ls-files 'src/dftorch/**/*.py' 'src/dftorch/*.py' | xargs rg -o 'print\(' | wc -l
    182 across 20 runtime modules when subpackages are included

The context's module count of 15 shows that 112 was intended to describe the
top-level modules; that like-for-like count is now 124. No unrelated print site
was edited in 05-03 because D-02 assigns their classification and gating to
05-05.

**What should happen:** plan 05-05 must regenerate its inventory from the
current tree and use 124, not 112, as its top-level starting count. It should
also state explicitly whether `_legacy/`, `sedacs/`, and `ewald_pme/` are in its
scope; including them raises the recursive baseline to 182.

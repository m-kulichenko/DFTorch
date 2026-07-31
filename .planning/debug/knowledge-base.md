# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## ch4-checksum-ulps — exact scalar reduction rejected an unchanged dH0 tensor
- **Date:** 2026-07-31
- **Error patterns:** dh0_abs, exact float equality, CH4_DH0_ABS_SUM, 777.8475256952825, 777.8475256952822, checksum reduction
- **Root cause:** Exact equality on `dH0.abs().sum()` treated a non-associative floating-point reduction as an elementwise bit-identity oracle; identical tensor bytes can reduce to values a few ULPs apart.
- **Fix:** Keep `torch.equal` tensor gates exact and compare historical scalar reductions with zero relative tolerance and an explicit four-ULP absolute bound.
- **Files changed:** tests/test_radial_grid.py
---

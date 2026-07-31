# Phase 5 Discussion Log

**Date:** 2026-07-29
**Mode:** default (interactive)

> Human reference only — audits and retrospectives. Downstream agents read `05-CONTEXT.md`,
> not this file.

## Areas Presented

Four gray areas were offered; the user selected **all four**.

1. Radial grid lookup scope
2. `src/dftorch/script.py` disposition
3. `print()` noise policy
4. CLN-03 depth on hardcoded `1/4/9` orbital counts

## Area 1 — Radial grid lookup

**First attempt failed.** The question was posed as "where should REG-06 land?" The user
replied: *"Wait IM confused, be specific, what is REG-06? This is random naming I don't know
the meaning of. What's the fix?"* — a fair objection, since that ID had been minted less
than an hour earlier in the same session.

**Re-asked concretely**, describing the mechanism: each pair's grid is already stored in
`R_tensor`, but `_bond_integral.py:1064-65` exports one global `R_orb` (the longest grid) and
`_h0ands.py:556-558` derives every pair's interpolation knot from it.

| Options presented | Selected |
|---|---|
| Refuse now, support in Phase 8 *(recommended)* | |
| **Do both in Phase 5** | ✅ |
| Refuse now, support in its own phase | |

**Note:** the cost of the larger scope was stated twice — it changes an interface consumed by
the ML-SK and stress paths, in a phase otherwise meant to stabilize. The user chose it anyway.
Recorded as D-01 with a `costly` reversibility rating.

## Area 2 — `script.py` disposition

| Options presented | Selected |
|---|---|
| **Port unique checks to `tests/`, then delete** *(recommended)* | ✅ |
| Move wholesale to `tests/` | |
| Keep as a thin CLI wrapper | |
| Leave it alone this phase | |

**Note:** the deciding detail was that `parse_expected_homonuclear_metadata()` builds
expectations by independently re-parsing SKF headers, so it never uses the production parser
as its own oracle — a property the 72-test pytest suite lacks. Recorded as D-03.

## Area 3 — `print()` noise policy

**First attempt failed.** The question cited "169 print() calls" without showing any. The user
replied: *"I'm not sure what print statements you are talking about, be more specific"* —
again fair.

**Re-asked** with the actual lines (`Constants.py:203` → `DFTB3: False`,
`_coulomb_matrix.py:163` → `coulomb_matrix_vectorized`, `_h0ands.py:123` → `H0_and_S`, etc.),
and with a self-correction: 57 of the 169 live in `script.py`, which D-03 deletes, so the real
target is ~112 across 15 modules.

| Options presented | Selected |
|---|---|
| Full logging migration, silent by default *(recommended)* | |
| Only the modules that spam f runs | |
| **Keep prints, gate on a verbose flag** | ✅ |
| Document as debt, change nothing | |

**Follow-up asked** — what should the flag default to?

| Options presented | Selected |
|---|---|
| Default quiet *(recommended)* | |
| **Default noisy — preserve today's behavior exactly** | ✅ |
| Quiet by default, except timing lines | |

**Note:** the noisy default is the conservative choice for a regression-safety phase — it
keeps REG-02/REG-03 (tutorial notebook, existing simple-format users) strictly unaffected. The
accepted consequence is that this phase makes the noise *suppressible* but does not remove it.
Recorded as D-02.

## Area 4 — CLN-03 depth

| Options presented | Selected |
|---|---|
| **Audit all, extend or guard each** *(recommended)* | ✅ |
| Guard-only — refuse, don't extend | |
| Inventory and document, no code change | |

Recorded as D-04.

## Claude's Discretion

- Verbose flag name and location (likely a `dftorch_params` key)
- Whether genuine failure warnings stay unconditional rather than verbose-gated
- Whether D-01 keeps exporting `R_orb` additively or replaces it
- Organization of the ported `script.py` checks across test files
- Fate of `ConstantsTest`
- Format and location of the D-04 inventory
- Ordering and wave structure

## Deferred

Migration to a real logging module; flipping the verbose default to quiet; centralizing
duplicated basis metadata; splitting legacy CSV loaders out of `_bond_integral.py`;
`wfc.hsd` override coverage; malformed / skipped-shell SKF fixtures; the CH4 SCF
non-convergence (Phase 6, and currently the one open `.planning/WINDOWS.md` entry blocking
`/gsd-ship`).

## Process note worth keeping

Two of four questions had to be re-asked because they referenced invented requirement IDs or
unshown code instead of the substance. Recorded in `05-CONTEXT.md` `<specifics>` so downstream
agents describe changes by what the code does, not by ID.

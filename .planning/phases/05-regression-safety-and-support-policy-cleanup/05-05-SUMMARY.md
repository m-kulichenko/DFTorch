---
plan: 05-05
phase: 05-regression-safety-and-support-policy-cleanup
requirements: [REG-01, CLN-05]
status: complete
completed: 2026-08-02
---

# 05-05 Summary — Verbose flag (D-02) and ConstantsTest removal

## Recovery note

The executor for this plan was terminated mid-flight by a session limit, after committing
Task 1 but before committing Task 2 or starting Task 3. The orchestrator recovered it rather
than re-dispatching: the partial working tree was inspected, verified coherent (all eight
modules imported, the D-02 contract measured directly), regression-checked, and committed as
`15a092e`. Task 3 was then completed. No work was redone and none was lost.

## Commits

| Commit | Task |
|---|---|
| `da302dd` | Classify every library print; add the `VERBOSE_LIBRARY_OUTPUT` flag |
| `15a092e` | Gate status chatter across eight modules (recovered partial work) |
| `ea3915a` | Remove the dead `ConstantsTest` table |

## D-02: the default is deliberately noisy

The flag defaults to `True`, so a caller who never sets `VERBOSE_LIBRARY_OUTPUT` sees exactly
the output they saw before it existed. This was the user's explicit choice, and it is the
conservative one for a regression-safety phase: REG-02 and REG-03 require existing
simple-format callers and the tutorial notebook to be unaffected, and a quiet default would
have been a behavior change. Migrating to a logging module and defaulting to quiet were both
presented and both rejected.

**Accepted consequence, stated plainly:** this phase makes the noise *suppressible*. It does
not remove it. Anyone wanting silence must opt in.

### Measured (CH4 + mio-1-1, `COUL_METHOD=FULL`)

| Setting | stdout | Lines | `e_tot` |
|---|---|---|---|
| default (key absent) | 518 chars | 30 | -23.619608184971 |
| `VERBOSE_LIBRARY_OUTPUT=False` | 13 chars | 1 | -23.619608184971 |

Energy identical. 97% of chatter suppressed. The single surviving line under quiet is
`zero norm_dr` from `_xl_tools.py` — a genuine warning, correctly unconditional.

### What was deliberately left alone

- **Failure and non-convergence warnings stay unconditional**: the `cnt == MaxIt` prints,
  `_xl_tools`' `zero norm_dr` and `Not converged`, and the `spinw.txt` load failure at
  `Constants.py`. Gating these would trade noise for silent wrongness.
- **Prints already behind an existing per-call `verbose` argument** were not re-gated. They
  default to `False` and are silent today; routing them onto the new flag would have switched
  them *on*, which is the opposite of the intent.
- `_h0ands.py` and `_nearestneighborlist.py` needed no change.

Each entry point reads the flag once per call rather than per print.

## CLN-05: ConstantsTest removed

911 lines of dead parallel constants table. It hard-coded element labels and legacy orbital
metadata separately from the real SKF loader and inferred `shell_present` from `max_ang`,
never calling `get_skf_tensors`. Verified dead — no references anywhere in `src/` or `tests/`
outside its own file. Plan 05-03 deliberately left it for this plan, which owns `Constants.py`.

Worth noting why this mattered more than a line count: the shell-presence defect fixed in
`4dbffaa` is exactly the class of bug a stale parallel metadata table makes harder to see.
The SKF header oracle (`24fb531`) is the honest replacement — it re-derives expectations from
the files themselves rather than from a hand-maintained copy.

## Verification

Regression-checked after each commit. Green at every step:

- `tests/test_scf.py`, `tests/test_simple_format_regression.py`,
  `tests/test_single_shot_energy.py`, `tests/test_eu_n_scan.py`,
  `tests/test_shell_count_parsing.py` — all pass after the gating commit
- `tests/test_skf_metadata_oracle.py` (56), `tests/test_public_api_contract.py`,
  `tests/test_import.py` — all pass after the ConstantsTest removal
- `hasattr(dftorch.Constants, 'ConstantsTest')` is `False`

The full suite exceeds the orchestrator's 10-minute command cap, so it is verified per-file
and summed rather than in one run. That is a measurement constraint, not a hang — see the
note on suite growth (72 -> 169+ tests) recorded during this phase.

## Self-Check: PASSED

All three claimed commits resolve in `git log`. `src/dftorch/Constants.py` is 262 lines
(was 1174). All eight gated modules import.

---
schema_version: 1
open_count: 3
waived_count: 0
fixed_count: 3
total_count: 6
last_updated: 2026-08-07T20:34:05.190Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 04 | unrun-verify | tests/test_scf.py |  | CH4 SCF does not converge in 25 iterations and the residual grows (0.155 -> 0.389 -> 0.466); pre-existing, unasserted, governed by D-13 and out of Phase 4 scope | fixed |  | 2026-07-29T19:12:19.328Z | 2026-08-02T19:17:07.552Z |
| 2 | 05 | unmet-truth | docs/LIBRARY-OUTPUT-INVENTORY.md |  | The library-output inventory claims a completeness gate at tests/test_verbose_flag.py::test_inventory_covers_every_print; neither the file nor the test exists, so a print added by a later phase fails nothing | fixed |  | 2026-08-03T02:42:11.528Z | 2026-08-03T02:52:13.554Z |
| 3 | 6 | unmet-truth | src/dftorch/Constants.py |  | Constants.py does not decode as pure ASCII (291 non-ASCII bytes, pre-existing box-drawing separators), so plan 06-04's ASCII acceptance criterion for this file was unsatisfiable on arrival. Plan 06-04 added zero non-ASCII bytes; the pre-existing ones are untouched. | open |  | 2026-08-06T20:19:42.790Z |  |
| 4 | 6 | lint-warning | src/dftorch/Constants.py | 5 | import numpy as np is never used. Pre-existing, unrelated to plan 06-04, left unfixed to keep the plan's diff scoped; ruff is not installed in this environment. | open |  | 2026-08-06T20:19:45.827Z |  |
| 5 | 06 | unrun-verify | experiments/eu_n_scf_binding_curve.py |  | The end-to-end command 'uv run python experiments/eu_n_scf_binding_curve.py' has never been executed: matplotlib is absent from the project environment, so the render half is unrun. Computation half verified live; render half smoke-tested only. | fixed |  | 2026-08-06T20:42:29.810Z | 2026-08-07T20:34:05.190Z |
| 6 | 06 | unmet-truth | src/dftorch/_scf.py |  | The per-orbital-group charge loop (MAGNETIC_HUBBARD_LDEP) settles at only 12 of 21 Eu-N separations: it gives up at 2.30 A and at every separation from 2.90 to 3.60 A, running charge to plus or minus 3 to 5 electrons and energies to +359 eV. The per-atom loop settles 21 of 21. | open |  | 2026-08-06T20:42:32.163Z |  |

````json
[
  {
    "id": 1,
    "kind": "unrun-verify",
    "phase": "04",
    "file": "tests/test_scf.py",
    "line": null,
    "description": "CH4 SCF does not converge in 25 iterations and the residual grows (0.155 -> 0.389 -> 0.466); pre-existing, unasserted, governed by D-13 and out of Phase 4 scope",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-07-29T19:12:19.328Z",
    "resolved_at": "2026-08-02T19:17:07.552Z"
  },
  {
    "id": 2,
    "kind": "unmet-truth",
    "phase": "05",
    "file": "docs/LIBRARY-OUTPUT-INVENTORY.md",
    "line": null,
    "description": "The library-output inventory claims a completeness gate at tests/test_verbose_flag.py::test_inventory_covers_every_print; neither the file nor the test exists, so a print added by a later phase fails nothing",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-03T02:42:11.528Z",
    "resolved_at": "2026-08-03T02:52:13.554Z"
  },
  {
    "id": 3,
    "kind": "unmet-truth",
    "phase": "6",
    "file": "src/dftorch/Constants.py",
    "line": null,
    "description": "Constants.py does not decode as pure ASCII (291 non-ASCII bytes, pre-existing box-drawing separators), so plan 06-04's ASCII acceptance criterion for this file was unsatisfiable on arrival. Plan 06-04 added zero non-ASCII bytes; the pre-existing ones are untouched.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-06T20:19:42.790Z",
    "resolved_at": null
  },
  {
    "id": 4,
    "kind": "lint-warning",
    "phase": "6",
    "file": "src/dftorch/Constants.py",
    "line": 5,
    "description": "import numpy as np is never used. Pre-existing, unrelated to plan 06-04, left unfixed to keep the plan's diff scoped; ruff is not installed in this environment.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-06T20:19:45.827Z",
    "resolved_at": null
  },
  {
    "id": 5,
    "kind": "unrun-verify",
    "phase": "06",
    "file": "experiments/eu_n_scf_binding_curve.py",
    "line": null,
    "description": "The end-to-end command 'uv run python experiments/eu_n_scf_binding_curve.py' has never been executed: matplotlib is absent from the project environment, so the render half is unrun. Computation half verified live; render half smoke-tested only.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-06T20:42:29.810Z",
    "resolved_at": "2026-08-07T20:34:05.190Z"
  },
  {
    "id": 6,
    "kind": "unmet-truth",
    "phase": "06",
    "file": "src/dftorch/_scf.py",
    "line": null,
    "description": "The per-orbital-group charge loop (MAGNETIC_HUBBARD_LDEP) settles at only 12 of 21 Eu-N separations: it gives up at 2.30 A and at every separation from 2.90 to 3.60 A, running charge to plus or minus 3 to 5 electrons and energies to +359 eV. The per-atom loop settles 21 of 21.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-06T20:42:32.163Z",
    "resolved_at": null
  }
]
````

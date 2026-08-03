---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 2
total_count: 2
last_updated: 2026-08-03T02:52:13.554Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 04 | unrun-verify | tests/test_scf.py |  | CH4 SCF does not converge in 25 iterations and the residual grows (0.155 -> 0.389 -> 0.466); pre-existing, unasserted, governed by D-13 and out of Phase 4 scope | fixed |  | 2026-07-29T19:12:19.328Z | 2026-08-02T19:17:07.552Z |
| 2 | 05 | unmet-truth | docs/LIBRARY-OUTPUT-INVENTORY.md |  | The library-output inventory claims a completeness gate at tests/test_verbose_flag.py::test_inventory_covers_every_print; neither the file nor the test exists, so a print added by a later phase fails nothing | fixed |  | 2026-08-03T02:42:11.528Z | 2026-08-03T02:52:13.554Z |

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
  }
]
````

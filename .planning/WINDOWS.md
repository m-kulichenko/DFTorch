---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-07-29T19:12:19.328Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 04 | unrun-verify | tests/test_scf.py |  | CH4 SCF does not converge in 25 iterations and the residual grows (0.155 -> 0.389 -> 0.466); pre-existing, unasserted, governed by D-13 and out of Phase 4 scope | open |  | 2026-07-29T19:12:19.328Z |  |

````json
[
  {
    "id": 1,
    "kind": "unrun-verify",
    "phase": "04",
    "file": "tests/test_scf.py",
    "line": null,
    "description": "CH4 SCF does not converge in 25 iterations and the residual grows (0.155 -> 0.389 -> 0.466); pre-existing, unasserted, governed by D-13 and out of Phase 4 scope",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-07-29T19:12:19.328Z",
    "resolved_at": null
  }
]
````

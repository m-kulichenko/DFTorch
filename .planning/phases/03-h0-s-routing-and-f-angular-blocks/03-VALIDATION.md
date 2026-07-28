---
phase: 03
slug: h0-s-routing-and-f-angular-blocks
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-28
---

# Phase 03 - Validation Strategy

> Per-phase validation contract for single-system f-orbital H0/S assembly.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml |
| Quick run command | `uv run python -m pytest tests/test_f_orbital_skf.py -q` |
| Full suite command | `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` |
| Estimated runtime | ~30 seconds |

---

## Sampling Rate

- After every task commit: run `uv run python -m pytest tests/test_f_orbital_skf.py -q`.
- After every plan wave: run the full suite command above.
- Before `$gsd-verify-work`: full suite must be green.
- Max feedback latency: 30 seconds for the focused gate.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | HSK-01, HSK-02 | N/A | N/A | unit/regression | `uv run python -m pytest tests/test_f_orbital_skf.py -q` | Wave 0 | pending |
| 03-01-02 | 01 | 1 | HSK-03, HSK-04 | N/A | N/A | unit | `uv run python -m pytest tests/test_f_orbital_skf.py -q` | Wave 0 | pending |
| 03-01-03 | 01 | 1 | HSK-05, HSK-06 | N/A | N/A | unit/integration | `uv run python -m pytest tests/test_f_orbital_skf.py -q` | Wave 0 | pending |
| 03-01-04 | 01 | 1 | HSK-07, HSK-08 | N/A | N/A | regression | `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` | Wave 0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/test_f_orbital_skf.py` contains targeted H0/S tests before execution begins.
- [ ] Existing f-orbital parser/constants/structure tests remain intact.
- [ ] Tests use real `tests/f_orbital_data` fixtures and synthetic XYZ files only for geometry setup.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| External paper convention lock | HSK-04 | Formula table/order/sign source was not extractable from public metadata in research. | User supplies the paper PDF or extracted tables before final formula implementation is treated as scientifically locked. |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 30 seconds for focused checks.
- [x] `nyquist_compliant: true` set in frontmatter.

Approval: pending

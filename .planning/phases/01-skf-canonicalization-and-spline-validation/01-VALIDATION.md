---
phase: 1
slug: skf-canonicalization-and-spline-validation
# status lifecycle: draft (seeded by plan-phase) -> validated (set by validate-phase)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-20
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for parser and spline checkpoint work.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `src/dftorch/script.py` validation harness |
| **Config file** | none |
| **Quick run command** | `uv run python src/dftorch/script.py tests/f_orbital_data` |
| **Full suite command** | `uv run python src/dftorch/script.py tests/f_orbital_data` plus `uv run python -m pytest tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_scf.py -q` |
| **Estimated runtime** | ~30-90 seconds |

---

## Sampling Rate

- **After every task touching `_bond_integral.py` or `script.py`:** Run `uv run python src/dftorch/script.py tests/f_orbital_data`
- **After the plan wave:** Run `uv run python src/dftorch/script.py tests/f_orbital_data` and the focused pytest regression command.
- **Before `$gsd-verify-work`:** Script validation and focused pytest regression must be green.
- **Max feedback latency:** ~90 seconds for the scoped commands.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | SKF-01, SKF-02, SKF-03, SKF-04 | — | N/A | script/invariant | `uv run python src/dftorch/script.py tests/f_orbital_data` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | SKF-05, SKF-07 | — | N/A | script/invariant | `uv run python src/dftorch/script.py tests/f_orbital_data` | ✅ | ⬜ pending |
| 01-01-03 | 01 | 1 | SKF-06, SPL-01, SPL-02, SPL-03, SPL-04 | — | N/A | script/regression | `uv run python src/dftorch/script.py tests/f_orbital_data` plus `uv run python -m pytest tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_scf.py -q` | ✅ | ⬜ pending |

*Status: pending until execution updates this artifact.*

---

## Wave 0 Requirements

Existing infrastructure covers Phase 1. No new test framework setup is required.

---

## Manual-Only Verifications

All Phase 1 behaviors should be covered by automated script output.

---

## Validation Sign-Off

- [ ] All parser/spline tasks have automated script verification.
- [ ] No task relies on manual inspection alone.
- [ ] `src/dftorch/script.py` remains the Phase 1 f-orbital checkpoint harness.
- [ ] Existing focused pytest regression remains green.
- [ ] `nyquist_compliant: true` set in frontmatter after validation review.

**Approval:** pending

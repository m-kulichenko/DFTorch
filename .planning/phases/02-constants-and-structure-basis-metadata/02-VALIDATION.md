---
phase: 02
slug: constants-and-structure-basis-metadata
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-24
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest via `uv run python -m pytest`; standalone validation via `src/dftorch/script.py` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run python -m pytest tests/test_f_orbital_skf.py -q` |
| **Full suite command** | `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` plus `uv run python src/dftorch/script.py tests/f_orbital_data` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run python -m pytest tests/test_f_orbital_skf.py -q`
- **After every plan wave:** Run the full suite command above.
- **Before `$gsd-verify-work`:** Full suite and standalone script must be green.
- **Max feedback latency:** 30 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | BAS-01, BAS-02, BAS-04, BAS-05 | — | N/A | integration | `uv run python -m pytest tests/test_f_orbital_skf.py -q` | ✅ | ⬜ pending |
| 02-01-02 | 01 | 1 | BAS-03, BAS-06 | — | N/A | regression | `uv run python -m pytest tests/test_f_orbital_skf.py -q` | ✅ | ⬜ pending |
| 02-01-03 | 01 | 1 | BAS-01 through BAS-06 | — | N/A | smoke | `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` plus `uv run python src/dftorch/script.py tests/f_orbital_data` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing pytest infrastructure covers all Phase 2 requirements.
- `tests/test_f_orbital_skf.py` already exists and should be extended rather than creating a parallel framework.
- The standalone script gate already exists and remains the human-readable validation path for parser, Constants, Structure, and StructureBatch metadata.

---

## Manual-Only Verifications

All Phase 2 behaviors have automated verification. H0/S execution and tutorial numerical parity are explicitly deferred.

---

## Validation Sign-Off

- [x] All tasks have automated verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all MISSING references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30s.
- [x] `nyquist_compliant: true` set in frontmatter after validation passes.

**Approval:** approved 2026-07-27

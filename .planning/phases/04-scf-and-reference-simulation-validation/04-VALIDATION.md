---
phase: 4
slug: scf-and-reference-simulation-validation
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: true
created: 2026-07-29
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Seeded from `04-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1, under a `torch.set_default_dtype(torch.float64)` harness with a module-reset fixture |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`, line 51) |
| **Quick run command** | `uv run pytest tests/test_scf.py -x -q` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated runtime** | ~30 s quick · ~2–5 min full (CPU) |

**Invocation note:** CI runs `uv run pytest -q` (`.github/workflows/tests.yml:56`). Use that
form. Bare `pytest` (as written in RESEARCH.md) is not the canonical invocation.

---

## Sampling Rate

- **After every task commit:** `uv run pytest tests/test_scf.py -x -q`
- **After every plan wave:** `uv run pytest -q`
- **Before `/gsd-verify-work`:** full suite green, including the Phase 3 regression gate in
  `tests/test_f_orbital_skf.py`
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

> Task IDs are filled in by the planner. Requirement → behavior mapping is fixed below.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01 T1 | 04-01 | 1 | SIM-01 | — | N/A | unit | `uv run pytest tests/test_single_shot_energy.py -q` | ❌ W0 | ⬜ pending |
| 04-01 T1 | 04-01 | 1 | SIM-02 (single-shot energy) | — | N/A | unit + integration | `uv run pytest tests/test_single_shot_energy.py -q` | ❌ W0 | ⬜ pending |
| 04-01 T2 | 04-01 | 1 | SIM-02 (spin guard) | T-04-02 | Named exception, not a silent closed-shell number and not a bare `IndexError` | unit | `uv run pytest tests/test_spin_guard.py -q` | ❌ W0 | ⬜ pending |
| 04-02 T1 | 04-02 | 1 | SIM-02 (smoke gate) / REG-01 | — | N/A | regression | `uv run pytest tests/test_scf.py -q` | ✅ exists, currently RED | ⬜ pending |
| 04-02 T2 | 04-02 | 1 | SIM-02 (dtype audit, D-18) | T-04-04 | Sort key stays injective over `(i, j)`, so nearest-image dedup cannot silently keep the wrong image | unit | `uv run pytest tests/test_dtype_contract.py -q` | ❌ W0 | ⬜ pending |
| 04-03 T1 | 04-03 | 2 | SIM-03 (data) | — | N/A | unit | `uv run pytest tests/test_shell_resolved_u.py -q` | ❌ W0 | ⬜ pending |
| 04-03 T2 | 04-03 | 2 | SIM-03 (gate + guard) | T-04-05, T-04-06 | Named exception instead of a shell-resolved Coulomb matrix whose f, p and d rows are silently zero | unit | `uv run pytest tests/test_shell_resolved_u.py -q` | ❌ W0 | ⬜ pending |
| 04-04 T1 | 04-04 | 3 | SIM-04 | — | N/A | docs gate | section check on `tests/f_orbital_data/README-EU-N-CASE.md` | ❌ W0 | ⬜ pending |
| 04-04 T2 | 04-04 | 3 | SIM-05 | T-04-08, T-04-09 | Band derived as a fraction of the target; three-branch asymmetric failure message | integration | `uv run pytest tests/test_eu_n_scan.py -q` | ❌ W0 | ⬜ pending |
| 04-05 T1 | 04-05 | 4 | SIM-05 (manual) | T-04-11 | Band changes require a recorded decision, never a silent constant edit | manual | — (blocking human-verify; see Manual-Only Verifications below) | n/a | ✅ green — **discharged, APPROVED** (`04-05-SUMMARY.md`) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Phase 3 regression gate** (`uv run pytest tests/test_f_orbital_skf.py -q`) is asserted as a
non-regression criterion by every task in plans 04-01 and 04-03, so it is not a separate row.

**Note on `tests/test_scf.py`:** it is the *authoritative* red→green gate for the PME dtype
defect. A standalone unit call to `calculate_PME_kspace_stress` was measured at plan time to
pass even with the bug present (12 Å and 25 Å boxes, grad and no-grad), so the contract test in
`tests/test_dtype_contract.py` pins the dtype contract but must not be treated as the
reproducer.

---

## Wave 0 Requirements

- [x] `tests/test_single_shot_energy.py` — asserts `forward(do_scf=False)` populates
      `structure.e_tot` for an f-containing system (SIM-02 supported branch)
- [x] `tests/test_spin_guard.py` — asserts the named spin-polarization exception is raised
      (SIM-02 unsupported branch, D-12)
- [x] `tests/test_shell_resolved_u.py` — asserts `Hubbard_U_sr` carries the f dimension and
      correct **values**, under both `MAGNETIC_HUBBARD_LDEP` settings (SIM-03, D-14, D-23)
- [x] `tests/test_dtype_contract.py` — asserts the min-image sort key stays injective at
      realistic system sizes, and pins the PME k-space stress dtype contract (D-18)
- [x] `tests/test_eu_n_scan.py` — Eu-N energy scan; band-derivation guard, grid-brackets-band
      guard, finite/float64 curve, interior-minimum gate, uniqueness, single-well shape
      (SIM-05)
- [x] `tests/f_orbital_data/README-EU-N-CASE.md` — the durable case document: geometry, SKF
      parameter files, observable, units, tolerance, provenance and the D-24 asymmetric
      failure rule (SIM-04)

All six confirmed present on disk at 04-05 review time with the full suite green
(72 passed / 0 failed), so `wave_0_complete: true`. `nyquist_compliant` is deliberately left
`false` and `status` left `draft`: the Validation Sign-Off checklist below asserts items plan
04-05 did not verify (sampling continuity, watch-mode flags, measured feedback latency), and
`status: validated` is set by `validate-phase` § 6, not by an executor.

Existing infrastructure (`tests/test_f_orbital_skf.py`, `tests/test_scf.py`, the float64
harness and module-reset fixture) covers the rest — no framework install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Verdict |
|----------|-------------|------------|-------------------|---------|
| Physical plausibility of the located minimum | SIM-05 | The target (2.655 Å, D-24) is a mean over Eu-N **coordination** bonds, while the test case is an isolated **diatomic**. A green test confirms the ballpark, not the physics. | Sanity-check the scan curve shape: single well, energy rising on both sides, no monotonic drift. If the curve has no interior minimum, the band assertion is meaningless regardless of pass/fail. | ✅ **DISCHARGED — APPROVED** (04-05) |
| Short-side band failure | SIM-05 | A minimum below ~2.12 Å is plausibly correct diatomic physics (shorter than a crowded 8-9-coordinate dative bond), not an f-implementation bug. Only a human can tell those apart. | If the scan minimum falls short of the band, do NOT conclude the f code is broken. Compare against bulk rock-salt EuN (~2.45-2.5 Å) and the curve shape, then decide whether to widen the band or investigate. | ✅ **DISCHARGED — NOT TRIGGERED** (04-05) |

### Manual-Only Verification Discharge (plan 04-05, 2026-07-29)

Full record: `04-05-SUMMARY.md`. Summarised here so the discharge is visible from the
validation contract itself.

**Row 1 — Physical plausibility of the located minimum: DISCHARGED, disposition APPROVED.**
A human reviewed a two-panel plot of the binding curve (full 1.60–3.60 Å scan plus a zoom
from 2.05 Å, because the ~0.83 eV well is invisible against the ~14 eV repulsive wall on one
linear axis), **recomputed from live code at review time** rather than replayed from the
recorded table. Verified independently at review time: minimum **2.40 Å**, E_tot
**-17.64144156 eV**, grid index **8 of 20**; strictly interior; inside [2.124, 3.186] Å;
strictly monotone decreasing in and increasing out; unique on the grid; full suite
**72 passed / 0 failed**. Curve-shape verdict: **single clean well** — not monotonic, not
multi-well, not flat within numerical noise.

**Row 2 — Short-side band failure: DISCHARGED as NOT TRIGGERED.** The minimum landed inside
the band, so no short-side result existed to adjudicate. This is a legitimate disposition;
**it is not claimed that the branch was exercised.** Plan 04-04's M5 mutation separately
proved the branch fires with the correct message when forced.

**D-24 note.** 2.40 Å sits *below* the 2.655 Å target — the direction D-24 anticipates for an
isolated diatomic versus a coordination-sphere mean — and coincides with the independent bulk
rock-salt EuN reference (~2.45–2.50 Å). Two reference points agree. **No band change was
requested, so no new `04-CONTEXT.md` decision is required and the band constants remain
untouched** (threat T-04-11; `EU_N_TARGET_ANGSTROM` = 2.655 and `EU_N_BAND_FRACTION` = 0.20
confirmed unchanged after review).

**One concern surfaced and explicitly accepted.** The curve is still climbing at 3.60 Å and
does not reach a dissociation asymptote within the scanned range. The human was offered
approve-as-is / approve-but-carry-forward / hold-and-extend-to-6-8 Å, and chose **approve —
expected for single-shot non-SCC energy over this range**. Recorded as
**reviewed-and-accepted**, not as an unexamined gap and not as an open concern.

**All six § Sampling Risk items were reviewed; the four that this phase leaves unsampled are
accepted**, each traceable to an earlier locked decision rather than being a new discovery:
±20% band absorbing a 15% f-block error (D-20/D-22), single-shot ≠ SCF (D-11), closed-shell
treatment of open-shell Eu 4f⁷ (D-12), and validated-but-unconsumed shell-resolved f plumbing
(D-14).

---

## Sampling Risk — what a passing suite still would not catch

1. **The Eu-N reference separation is a coordination-chemistry mean, not a diatomic
   measurement.** *(Corrected at plan time: RESEARCH.md A3's unsourced ~2.4–2.5 Å recall is
   superseded by D-24.)* The 2.655 Å target is the mean of 16 Eu-N bond lengths in Eu-Bp
   coordination complexes — dative bonds in a crowded 8–9-coordinate sphere, not an isolated
   diatomic. A green SIM-05 means "the minimum is near a related but non-identical reference",
   not "the physics is right." This is exactly why D-24 makes the failure interpretation
   asymmetric.
2. **Loose band absorbs real errors.** D-20/D-22 mandate a ~10–20% band. An f-block bug
   shifting the minimum by 15% passes. The band is a smoke test, not a correctness proof.
3. **Single-shot ≠ SCF.** D-11 ships one diagonalization. Charge-self-consistency bugs
   cannot surface this phase by construction.
4. **Closed-shell treatment of an open-shell ion.** Eu 4f⁷ is genuinely open-shell; D-12
   defers spin. Any observable sensitive to spin polarization is untested here.
5. **D-14 plumbing is validated but unexercised.** Shell-resolved U shapes and values get
   tests, but the single-shot path (D-11) never consumes them — so integration between the
   two is unsampled until the later SCF phase.
6. **Only the Eu-N pair is exercised.** Seven of nine SKF fixtures go untouched; Ga paths
   and periodic cases are out of scope and therefore unsampled.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

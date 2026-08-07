---
phase: 06
slug: self-consistent-scf-for-f-systems
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-04
per_task_map_filled: 2026-08-04
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

Seeded from `06-RESEARCH.md` §"Validation Architecture". The per-task map is filled by the
planner once plan/task IDs exist; everything else below is measured, not assumed.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 (suite baseline entering this phase: 216 passed / 0 failed, 18 files) |
| **Config file** | none — no pytest.ini or setup.cfg; defaults apply |
| **Quick run command** | `uv run pytest tests/test_scf.py -x` (CH4 f-free SCF, converges in 6 iterations, ~10 s) |
| **Full suite command** | `uv run pytest` (18 files, ~45 s including the 14.5 s Eu-N 21-point scan) |
| **Estimated runtime** | ~45 seconds full, ~10 seconds quick |

Canonical invocation is `uv run pytest`, **not** bare `pytest`.

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_scf_convergence_f.py -x` (focused f
  convergence, ~5 s) once that module exists; until then, `uv run pytest tests/test_scf.py -x`
- **After every plan wave:** Run `uv run pytest` (full suite, including the 21-point scan and every
  Phase 4/5 regression gate)
- **Before `/gsd-verify-work`:** Full suite must be green, and the Phase 4 single-shot pin
  `EU_N_REFERENCE_E_TOT = -17.510444238744924` must still hold (D-6.08 keeps it untouched)
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | SCC-01 | T-06-02, T-06-03 | A loop that exhausted its cap cannot present as converged; the Krylov switch-off cannot leak into an f-free run | integration (tracer) | `uv run pytest tests/test_scf_convergence_f.py tests/test_scf.py tests/test_single_shot_energy.py -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | SCC-01 | T-06-01 | A 14-to-15 element positional return cannot silently shift an attribute at an unfound call site | unit + wiring | `uv run pytest tests/test_scf_convergence_f.py tests/test_scf.py -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | SCC-01 | T-06-03, T-06-04 | Giving up returns `-1`, still returns the last iterate, raises nothing, and freezes no number | unit | `uv run pytest tests/test_scf_convergence_f.py -x` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 2 | SCC-02 | T-06-07, T-06-08, T-06-09 | An added-but-unreachable block, a widened mask changing an f-free result, and a row offset landing on the wrong atom are each caught | unit | `uv run pytest tests/test_shell_resolved_coulomb_f.py -x` | ❌ W0 | ⬜ pending |
| 06-02-02 | 02 | 2 | SCC-02 | T-06-10, T-06-11, T-06-12 | A refusal cannot be removed from code while a published document still claims it fires; the taxonomy cannot shrink | integration + doc | `uv run pytest tests/test_shell_resolved_u.py tests/test_orbital_count_guards.py tests/test_support_documentation.py tests/test_shell_resolved_coulomb_f.py -x` | ✅ (3 of 4 exist) | ⬜ pending |
| 06-03-01 | 03 | 3 | SCC-03 | T-06-13, T-06-14, T-06-15, T-06-16, T-06-17 | A shell-resolved request is never served silently by the per-atom matrix; the finer data provably reaches the answer | integration | `uv run pytest tests/test_shell_resolved_scf_f.py -x` | ❌ W0 | ⬜ pending |
| 06-03-02 | 03 | 3 | SCC-03 | T-06-18, T-06-23 | The f group is charged at the f rate; editing `Constants.py:232` becomes a visible decision, not a tidy-up | unit | `uv run pytest tests/test_shell_resolved_scf_f.py -x` | ❌ W0 | ⬜ pending |
| 06-04-01 | 04 | 3 | SCC-01 (D-6.03) | T-06-20, T-06-21 | The verdict states a conclusion and accounts for the band-energy share, not only the Coulomb term | doc completeness gate | `python tools/check_verdict_doc.py` | ❌ W0 | ⬜ pending |
| 06-04-02 | 04 | 3 | SCC-01 (D-6.03) | T-06-22, T-06-25 | The verdict's claims are held without freezing a value; the Phase 4 pin cannot be read as the settled answer | unit | `uv run pytest tests/test_energy_definitions_f.py tests/test_single_shot_energy.py -x` | ❌ W0 | ⬜ pending |
| 06-04-03 | 04 | 3 | SCC-01 (D-6.10) | T-06-23, T-06-24 | The recorded defect stays recorded and unfixed; a scoped roadmap edit cannot destroy other phases | regression | `uv run pytest tests/test_shell_resolved_scf_f.py tests/test_shell_resolved_u.py -x` | ❌ W0 | ⬜ pending |
| 06-05-01 | 05 | 4 | SCC-01, SCC-02, SCC-03 | T-06-27, T-06-28, T-06-31 | The figure is recomputed live, marks every non-converged separation, and applies no band gate | script run | `uv run python experiments/eu_n_scf_binding_curve.py` | ❌ W0 | ⬜ pending |
| 06-05-02 | 05 | 4 | SCC-01 (D-6.02) | T-06-29, T-06-30, T-06-32 | A looks-wrong verdict blocks the phase; the checkpoint cannot be skipped away | **manual only** — see below | *(no automated substitute exists by design)* | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Sampling continuity check:** every task except `06-05-02` carries an `<automated>` verify. The one
manual task is the phase's final gate, mandated as human-only by decision D-6.02 and recorded in
"Manual-Only Verifications" below. No three consecutive tasks lack automated feedback.

**Threat IDs referenced above** are defined in the `<threat_model>` block of the plan that owns the
task. `T-06-SC` (package-manager installs) appears in every plan and is dispositioned
not-applicable throughout: this phase installs nothing.

---

## Wave 0 Requirements

- [ ] `tests/test_scf_convergence_f.py` (plan 06-01) — new module covering SCC-01: convergence across
      the 2.655 A neighbourhood (2.60 / 2.655 / 2.70), the returned convergence result (iteration
      count on success, `-1` on failure, D-6.04), honest reporting when the iteration cap is
      exhausted, and the f-free control proving the Krylov switch-off is f-scoped
- [ ] `tests/test_shell_resolved_coulomb_f.py` (plan 06-02) — new module covering SCC-02: the seven f
      angular blocks (s-f, f-s, p-f, f-p, d-f, f-d, f-f) proven on a **europium-europium** pair,
      which is the only fixture arrangement reaching all seven, validated by the equal-strength
      reduction identity against the per-atom builder rather than by any recorded number
- [ ] `tests/test_shell_resolved_scf_f.py` (plan 06-03) — new module covering SCC-03: per-orbital-group
      charges actually consumed by the closed-shell loop, the two resolutions agreeing when summed,
      and the converged answer measurably differing from the per-atom one
- [ ] `tests/test_energy_definitions_f.py` (plan 06-04) — new module holding the D-6.03 verdict's
      claims without freezing a value
- [ ] `tools/check_verdict_doc.py` (plan 06-04) — new standard-library-only completeness gate on the
      verdict document
- [ ] `experiments/eu_n_scf_binding_curve.py` (plan 06-05) — new script recomputing all three curves
      from live code for the D-6.02 human checkpoint
- [x] A carrier for the returned convergence result — **decided**: a fifteenth element on the existing
      `SCFx` return tuple, a plain `int`, unpacked into `structure.scf_iter_count`; all four SCF loops
      change. Rationale and rejected alternatives are recorded in plan 06-01.
- [ ] `tests/test_support_documentation.py` and `tests/test_orbital_count_guards.py` — stay green.
      `test_no_new_f_exception_class_was_defined` pins the f taxonomy at exactly four names and must
      pass **unedited**, which is why plan 06-02 keeps `FShellResolvedCoulombUnsupportedError`
      defined while retiring the refusal, and edits `docs/F-SUPPORT-STATUS.md` and
      `docs/ORBITAL-COUNT-INVENTORY.md` in the same task as the code change
- [x] matplotlib — already available and already used by `experiments/diatomic_scans/`; no install

*Existing infrastructure covers f-free CH4 and every Phase 4 single-shot gate. Phase 6 adds only
f-specific SCF convergence coverage and the shell-resolved Coulomb coverage.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The self-consistent Eu-N binding curve "looks right" | SCC-01 (D-6.02) | The user set human judgement of a freshly computed graph as the phase's final gate, in place of a numeric one. No assertion substitutes for it. | Recompute the 21-point curve (1.60-3.60 A, step 0.10) from live code — never from a recorded table. Overlay single-shot and self-consistent curves with non-converged separations visibly marked (D-6.09). Present to the user at a `checkpoint:human-verify`. A verdict of "looks wrong" fails the phase gate; it is not a finding to record and move past. |
| Written verdict on the single-shot / self-consistent energy difference | SCC-01 (D-6.03) | The deliverable is an argued explanation — either "real physics, here is the mechanism" or "a defect, here is where it lives". Prose cannot be asserted. | Human reads the verdict document or summary section. It must account for the band-energy share of the difference, not only the Coulomb term. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — the single exception is
      `06-05-02`, the D-6.02 human checkpoint, which is manual **by the user's explicit design** and
      is carried as a manual-only row above
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references — five new files, each named with the plan that creates it
- [x] No watch-mode flags — every command is a single-shot `uv run pytest ... -x` or a script run
- [x] Feedback latency < 45s — the focused per-task commands run in about 5-15 s; the full suite is
      about 45 s; the plan 06-05 script is about a minute and runs once, at the checkpoint
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** per-task map filled by the planner 2026-08-04. Status flips to `validated` when the
phase is verified.

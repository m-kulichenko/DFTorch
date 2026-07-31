---
phase: 5
slug: regression-safety-and-support-policy-cleanup
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-29
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Seeded from `05-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1, `torch.set_default_dtype(torch.float64)` harness + module-reset fixture |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/test_scf.py tests/test_f_orbital_skf.py -q` |
| **Full suite command** | `uv run pytest -q` |
| **Baseline entering Phase 5** | **72 passed / 0 failed** |
| **Estimated runtime** | ~50 s full suite |

**Invocation note:** CI runs `uv run pytest -q` (`.github/workflows/tests.yml:56`). Bare
`pytest` is not canonical. `ruff` is declared as a dev extra but is **not installed** —
repo-wide `ruff format --check` likely already fails on pre-existing files; do not treat that
as a Phase 5 regression.

---

## Sampling Rate

- **After every task commit:** `uv run pytest tests/test_scf.py tests/test_f_orbital_skf.py -q`
- **After every plan wave:** `uv run pytest -q` — must stay at **≥72 passed / 0 failed**
- **Before `/gsd-verify-work`:** full suite green
- **Max feedback latency:** 50 seconds

**The 72-test baseline is this phase's primary safety net.** Every decision (D-01 interface
change, D-02 across 15 modules, D-03 file deletion, D-04 across 40+ sites) is a refactor whose
correctness claim rests on that suite staying green. A drop below 72 is a regression, not a
test-count change, unless a task explicitly documents why a test was removed or renamed.

---

## Per-Task Verification Map

> Task IDs filled in by the planner. Requirement → behavior mapping fixed below.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-T1 | 05-01 | 1 | REG-06 (characterization before rewrite) | unit | `uv run pytest tests/test_radial_grid.py -q` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | REG-06 (per-pair radial lookup), REG-01 (bit-identical) | unit + integration | `uv run pytest tests/test_radial_grid.py -q` | ❌ W0 | ⬜ pending |
| 05-02-T1 | 05-02 | 1 | REG-02 (disposition) | checkpoint:decision | n/a — blocking human decision, see Risk 1 | n/a | ⬜ pending |
| 05-02-T2 | 05-02 | 1 | REG-02 (simple-format baseline) | regression | `uv run pytest tests/test_simple_format_regression.py -q` | ❌ W0 | ⬜ pending |
| 05-03-T1 | 05-03 | 1 | CLN-05 (D-03 ported independent oracle) | unit | `uv run pytest tests/test_skf_metadata_oracle.py -q` | ❌ W0 | ⬜ pending |
| 05-03-T2 | 05-03 | 1 | CLN-05 (script.py removed) | contract | `uv run python -c "import importlib.util; assert importlib.util.find_spec('dftorch.script') is None"` | ✅ | ⬜ pending |
| 05-04-T1 | 05-04 | 2 | REG-05 (mixed-step guard; same-step/diff-length NOT rejected) | unit | `uv run pytest tests/test_radial_grid.py -q` | ❌ W0 | ⬜ pending |
| 05-04-T2 | 05-04 | 2 | REG-06 (remaining consumer dispositions) | docs gate + unit | grep-vs-`docs/RADIAL-GRID-CONSUMERS.md` row match | ❌ W0 | ⬜ pending |
| 05-04-T3 | 05-04 | 2 | REG-03 (no public API change) | unit | `uv run pytest tests/test_public_api_contract.py -q` | ✅ exists | ⬜ pending |
| 05-05-T1 | 05-05 | 2 | REG-01 (D-02 flag exists, default noisy) | unit | `uv run python -c "from dftorch._tools import library_output_enabled; assert library_output_enabled({}) is True"` | ❌ W0 | ⬜ pending |
| 05-05-T2 | 05-05 | 2 | REG-01 (default output byte-identical) | unit (capsys) | `uv run pytest tests/test_verbose_flag.py -q` | ❌ W0 | ⬜ pending |
| 05-05-T3 | 05-05 | 2 | CLN-05 (ConstantsTest removed) | contract | `uv run python -c "import dftorch.Constants as C; assert not hasattr(C,'ConstantsTest')"` | ✅ | ⬜ pending |
| 05-06-T1 | 05-06 | 3 | CLN-03 (inventory complete vs sweep) | docs gate | `uv run pytest tests/test_orbital_count_guards.py -q` | ❌ W0 | ⬜ pending |
| 05-06-T2 | 05-06 | 3 | REG-04 (explicit refusals, reachable) | unit | `uv run pytest tests/test_orbital_count_guards.py -q` | ❌ W0 | ⬜ pending |
| 05-07-T1 | 05-07 | 4 | CLN-02 (basis metadata map + consistency) | unit | `uv run pytest tests/test_support_documentation.py -q` | ❌ W0 | ⬜ pending |
| 05-07-T2 | 05-07 | 4 | CLN-01, CLN-04 (branches + support matrix) | docs gate + unit | `uv run pytest tests/test_support_documentation.py -q` | ❌ W0 | ⬜ pending |
| 05-07-T3 | 05-07 | 4 | REG-01, CLN-05 (requirement ledger, full suite) | regression | `uv run pytest -q` | ✅ 72 green | ⬜ pending |
| 05-08-T1 | 05-08 | 5 | CLN-01, CLN-04, CLN-05 (human sign-off) | checkpoint:human-verify | n/a — manual, covers Risk 2/4/5 | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Two manual-only rows, both deliberate.** `05-02-T1` is a `checkpoint:decision` because REG-02
cannot be met as written here (see the Risk-1 update below). `05-08-T1` is a
`checkpoint:human-verify` because every other gate in this phase is a count or a presence check,
and none of them can judge whether an inventory row says anything. No task is manual-only for
lack of effort.

---

## Wave 0 Requirements

All are created by the plans that consume them; none is left as an unowned gap.

- [ ] `tests/test_radial_grid.py` — 05-01 T1 (characterization pins), 05-01 T2 (per-pair lookup,
      bit-identity), 05-04 T1 (mixed-step refusal; same-step/different-length accepted)
- [ ] `write_mixed_grid_skf_pair` in `tests/test_radial_grid.py` — 05-01 T1. A synthetic
      mixed-grid SKF fixture generator; none exists in the tree, and all nine current fixtures
      share one grid, so the D-01 fix and the REG-05 guard are **untestable against real data**.
      Documented in `tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md`.
- [ ] `tests/test_simple_format_regression.py` — 05-02 T2, gated behind the REG-02 decision
- [ ] `tests/test_skf_metadata_oracle.py` — 05-03 T1, the ported
      `parse_expected_homonuclear_metadata()` independent-oracle checks
- [ ] `tests/test_verbose_flag.py` — 05-05 T2, `capsys` assertion that default output is
      **unchanged** and that the flag suppresses status chatter but not warnings
- [ ] `tests/orbital_count_sweep.py` + `tests/test_orbital_count_guards.py` — 05-06 T1/T2, the
      sweep, the inventory completeness gate, and one reachability test per guarded site
- [ ] `tests/test_support_documentation.py` — 05-07 T1/T2, basis-table consistency and the
      exception-vs-support-matrix cross-check

---

## Documentation-Deliverable Verification (CLN-01, CLN-02, CLN-04, D-04 inventory)

Several requirements are documents, not behavior. Verify them honestly rather than pretending
they have unit tests:

| Deliverable | How it is actually checked |
|---|---|
| D-04 orbital-count inventory | **Completeness against a grep.** Every `n_orb` comparison site found by a scripted sweep must appear in the inventory with a disposition. A site in code but absent from the inventory fails the check. This is the one documentation item with a real automatable gate. |
| CLN-01 prototype branches documented | Section-presence check plus a reviewer reading it. No automation beyond "the section exists and names files that exist." |
| CLN-02 basis metadata centralization | If code changes: tests. If documentation only: section presence. State which was chosen. |
| CLN-04 support-status matrix | Every one of batch / force / stress / MD / SEDACS / ML-SK appears with a status, and every named exception in `_slater_koster_pair.py` is reflected. Cross-checkable against the exception classes. |

---

## Sampling Risk — what a passing suite still would not catch

1. **REG-02 has no harness, and the notebook demonstrably cannot run here.** ~~Options are
   papermill, nbval, or extracting reference numbers into a fixture — unresolved at seeding
   time.~~ **Resolved at planning time, 2026-07-30, by measurement.** Three independent
   obstacles, each verified:
   - Notebook cell 5, the first computational cell, sets `"FILENAME": "COORD.pdb"`, and
     `experiments/COORD.pdb` **does not exist in the repository**.
   - Cell 14 sets `device = "cuda"` with no availability guard; there is no CUDA driver here.
     (Cells 17 and 25 do guard correctly, so this is one cell, not a general problem.)
   - nbformat, nbclient, papermill, nbval, ipykernel and matplotlib are **all absent** from the
     environment.

   The stored outputs in the notebook's 23 output-bearing cells came from unknown hardware
   running unknown code, so they are not a usable baseline either. Plan 05-02 therefore raises
   this as a `checkpoint:decision` with three costed options rather than inventing a harness,
   and runs in wave 1 so that whatever baseline is chosen is captured before D-01, D-02 and D-04
   edit any code. **REG-02 is not dropped and is not silently rescoped.**

2. **D-01 and REG-05 are untestable against real fixtures.** All nine `tests/f_orbital_data`
   files share one grid (verified: identical 483-point arrays, step 0.0211670884 Å). The bug
   the fix targets cannot be reproduced from existing data — a synthetic mixed-grid fixture
   must be constructed, so the test validates the *implementation's* model of the problem,
   not an observed failure.

3. **~~The Å/Bohr mixed-unit convention is in D-01's blast radius.~~ DOWNGRADED — the docstring
   was wrong.** `_ml_sk.py:437-450` documents `R_orb` as Ångström and `dR_mskd` as Bohr, and
   claims `searchsorted` compares mixed-unit arrays. **Measured at planning time, 2026-07-30:
   both sides are Ångström.** Three independent confirmations:
   - `_bond_integral.py:713-718` builds the grid as
     `arange(1, npts_pad+1) * step * BOHR_TO_ANGSTROM` — Ångström by construction.
   - Instrumenting `torch.searchsorted` on CH4 + mio-1-1 captures `dR_mskd` in
     `[1.05668, 1.88755]`, and `idx = 98` lands on grid point `R[98] = 1.04777` Å — the correct
     physical knot. Under the Bohr reading the knot would sit at roughly twice the true
     separation and CH4 would not produce a sane energy.
   - `_ml_sk.py:522` annotates `dR_mskd` as Å, contradicting its own file's docstring 85 lines
     earlier.

   The per-pair rewrite therefore does not need to reason about units at all. Plan 05-04 corrects
   the docstring and records that it was corrected against measurement. **The characterization
   test is still planned and still required** (05-01 Task 1, before 05-01 Task 2's rewrite) —
   the rewrite still has to be proven behaviour-preserving, and the real hazard turned out to be
   elsewhere: `R_tensor` rows are zero-padded past `len(R_orb_i)` and therefore **not monotonic**,
   which makes `torch.searchsorted` against a row undefined. That is 05-01's concrete blocker
   and is tracked as threat T-05-03.

4. **D-02's noisy default means the flag's off-path is barely exercised.** Preserving today's
   output (the user's explicit choice) means the default path is what all 72 tests see; the
   quiet path gets only whatever targeted tests are written for it.

5. **Deleting `script.py` removes an independent oracle unless the port is faithful.** Its
   value is that `parse_expected_homonuclear_metadata()` never uses the production parser to
   check itself. A port that reuses production parsing silently destroys that property while
   appearing to preserve coverage.

6. **A green suite does not prove REG-03.** "No public API changes" needs an explicit
   import-surface assertion; the existing tests would not notice a removed export they never
   imported.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] Suite ≥ 72 passed / 0 failed at every wave boundary
- [ ] The Å/Bohr convention has a characterization test before D-01's rewrite
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

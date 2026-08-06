# f-Orbital Support Status

**Requirement:** CLN-04 -- "Batch, force, stress, MD, SEDACS and ML-SK f-orbital support
status is explicitly documented as supported, deferred, or unsupported."
**Written:** plan 05-07, measured 2026-08-03 on branch `f_orbital_initial`.
**Machine half:** `tests/test_support_documentation.py`, two tests.

Companion records: `docs/PROTOTYPE-BRANCHES.md` (what the temporary structure is waiting on),
`docs/ORBITAL-COUNT-INVENTORY.md` (all 157 dispositioned orbital-count sites),
`docs/RADIAL-GRID-CONSUMERS.md` (the grid-lookup consumers).

---

## How a status in this document was established

**From the code, never from the roadmap.** A capability the roadmap intends to deliver in
Phase 8 is not "deferred" because the roadmap says so; it is deferred because a named guard
refuses it today, at a stated line, and a test drives a real f system into that guard and
observes the refusal. Where plan 05-06 already wrote such a test, this document names the test
rather than re-verifying by hand. Where no test exists, the row says so.

The vocabulary, used consistently below:

| Word | Means |
|---|---|
| **supported** | works today for a system containing a 16-orbital (f) atom, and something in the suite exercises it |
| **deferred** | refuses explicitly today, and a requirement exists that will lift the refusal |
| **unsupported** | neither works nor refuses -- would produce a result whose f content is wrong |

**No row below is `unsupported`, and that is the finding, not an omission.** Every one of the
six capabilities CLN-04 names refuses explicitly. That is the direct product of decision D-04
(plan 05-06 dispositioned 157 orbital-count sites and guarded the six that were open) and of
the Phase 3 "fail loudly, never silently zero" rule. One genuinely unsupported path was found
while writing this document and it is *not* one of the six; see §4.

---

## 1. The support matrix

<!-- support-matrix -->

| Capability | Status | What a user meets today | Requirement that lifts it | Owning phase | Evidence |
|---|---|---|---|---|---|
| **batch** | deferred | `FAngularFormulaSourceError` from `_h0ands.py:637`, raised before any block is assembled when any pair has `n_orb == 16` | PHY-04 | Phase 8.1 | `tests/test_orbital_count_guards.py::test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| **force** | deferred | `FDerivativeUnsupportedError` from `ESDriver.py:61` (`_require_f_derivatives`, the first statement of both `ESDriver.calc_forces` and `ESDriverBatch.calc_forces`); the spin force path refuses separately at `_forces.py:49` | DRV-01, DRV-02, then PHY-01 | Phase 7 (formulas), Phase 8 (validation) | `test_every_guarded_site_raises_for_f[ESDriver.calc_forces]` and `[forces_spin]` |
| **stress** | deferred | `FDerivativeUnsupportedError` from `_stress.py:198`, inside `_pair_grad_from_sk` | PHY-02 | Phase 8 | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| **MD** | deferred | No f guard of its own. `MDXL.__init__` and `MDXLBatch.__init__` both require an `ESDriverBatch` (`MD.py:42`, `MD.py:1232`), so an f trajectory meets the **batch** refusal above first; the spin builders reached from `MD.py:745`, `:794` and `:1103` refuse separately | PHY-03, on top of PHY-04 and DRV-01 | Phase 9 | inherited: `[H0_and_S_vectorized_batch]`, `[get_h_spin]`, `[forces_spin]`. **No end-to-end f MD test exists**; the inheritance is established by reading the two `__init__` signatures |
| **SEDACS** | deferred | `ModuleNotFoundError: No module named 'sedacs'` -- the external `sedacs` package is not a dependency of this project, so `dftorch.sedacs.sedacs_interface` does not import here at all (probed 2026-08-03) | PHY-05 | v2, unscheduled | import probe, reproducible with `python -c "import dftorch.sedacs.sedacs_interface"`. **See §4: if the package were installed, the force path has no f guard** |
| **ML-SK** | deferred | plain `NotImplementedError` from `_slater_koster_pair.py:1048`, naming the requested channel. The ML head is addressed through the legacy 10-channel numbering (`_ml_sk.py:66-78`), whose keys carry `l` in `{0, 1, 2}` only, so no f channel can be expressed | PHY-06 | v2, unscheduled | established by reading `_slater_koster_pair.py:1041-1052`. **Not covered by a test** -- no test in `tests/` loads an ML-SK model |

### Notes the table cannot carry

**Why `MD` and `SEDACS` are `deferred` rather than `unsupported`.** Neither has an f guard
written *for it*. MD inherits two that fire before any f number is produced, which is the
substance of a refusal even though no line in `MD.py` mentions f. SEDACS cannot be reached at
all in this environment. Neither can currently return a wrong f answer, which is the property
`unsupported` would be recording.

**Why ML-SK's refusal is a plain `NotImplementedError` and not one of the four `F*` classes.**
Plan 05-06 pinned the f exception taxonomy at exactly four names
(`test_no_new_f_exception_class_was_defined`) so that the support policy stays reviewable as one
list. The ML-SK refusal predates that audit, is not an f-capability boundary in the same sense
(the ML head has no f channels at all, trained or otherwise), and adding a fifth class for it
would split the taxonomy. Recorded rather than changed: this plan modifies no production code.

**`v2, unscheduled` is a real status, not a soft "later".** `.planning/REQUIREMENTS.md`
§"v2 Requirements" reads "Tracked but not in current roadmap unless needed", and §"Out of
Scope" excludes "broad PME, GBSA, D3, SEDACS, or ML-SK expansion" except where it blocks the
single-system f path. Nothing schedules PHY-05 or PHY-06.

---

## 2. Every named exception class

The f-unsupported exception taxonomy lives in one module,
`src/dftorch/_slater_koster_pair.py`, deliberately, so that it can be reviewed as a single
support policy (see the class docstring of `FShellResolvedCoulombUnsupportedError`).
**Stale claim, recorded not fixed.** The preceding paragraph used to end by asserting that
`tests/test_support_documentation.py::test_every_named_exception_appears_in_support_matrix`
enumerates the classes from the module at test time, so a class added later without a row
here fails the suite. **No test of that name exists in the repository.**
`tests/test_support_documentation.py` defines four tests and none of them is it, so this
document has no completeness gate. Noticed while retiring the Coulomb refusal in phase 6
plan 06-02 and left for whoever owns the documentation gates: it is the same class of stale
claim already logged in `.planning/STATE.md` against `docs/LIBRARY-OUTPUT-INVENTORY.md`.
What does hold today is
`tests/test_orbital_count_guards.py::test_no_new_f_exception_class_was_defined`, which pins
the taxonomy at exactly these four names -- so a *fifth* class fails the suite, even though
a fourth-class row going missing from this table would not.

| Exception class | Defined at | What it guards | Raised at | Requirement that removes it |
|---|---|---|---|---|
| `FAngularFormulaSourceError` | `_slater_koster_pair.py:173` | reaching an f-containing SK pair class where the source-locked angular formulas are not wired in. Today that is the **batched** H0/S path only | `_h0ands.py:637`; the generic form `_require_f_formula_source` (`_slater_koster_pair.py:209`, called at `:960`) is inert on the single-system path because `F_ANGULAR_FORMULAS_AVAILABLE` is set `True` at `:785` | PHY-04 (batch), Phase 8.1 |
| `FDerivativeUnsupportedError` | `_slater_koster_pair.py:234` | any path that consumes f `dH0`/`dS`, which are exactly zero in every f block. Zero is a legal-looking derivative, so the consumer must refuse rather than integrate it | `ESDriver.py:61` (forces, both drivers), `_stress.py:198` (stress), `_forces.py:49` (the spin force path) | DRV-01 then DRV-02, Phase 7 |
| `FSpinPolarizationUnsupportedError` | `_slater_koster_pair.py:259` | a spin-polarized (open-shell) request for an f system. Eu 4f7 is genuinely open-shell, so a closed-shell number would differ in physics, not accuracy | `ESDriver.py:101` (`_require_closed_shell_f_system`, first statement of `ESDriver.forward`, so it fires for `do_scf=False` too), `_spin.py:45` (the shell-resolved spin builders reached directly from MD) | SPN-01, v2 |
| `FShellResolvedCoulombUnsupportedError` | `_slater_koster_pair.py:291` | **RETIRED.** It guarded the shell-resolved Coulomb matrix, whose pair masks tested `max_ang` against 1, 2 and 3 only, so an f element's non-s rows and columns came back exactly zero in a finite, correctly shaped matrix | nowhere -- no production module raises it, watched by `tests/test_shell_resolved_coulomb_f.py::test_the_retired_refusal_is_no_longer_raised_by_production_code` | **retired by SCC-02, Phase 6** |

**On the retired row.** `ewald_real_space_vectorized_sr` now builds all sixteen ordered
shell-pair blocks, the seven involving f included, so the reason to refuse is gone. The
class name is deliberately **kept** rather than deleted: plan 05-06 pinned this taxonomy at
exactly four names (`test_no_new_f_exception_class_was_defined`) so the support policy reads
as one list, and dropping a name would shrink it to three and lose the record of what used
to be unsupported. Its docstring and message constant now record the retirement, what
retired it, and what would justify raising it again -- an element whose orbital groups are
not a contiguous run starting at s, or a shell-resolved path with an incomplete block set.

Two further refusals exist that are **not** part of this taxonomy, listed so the reader who
meets one can find it:

| Refusal | Raised at | Why it is not an `F*` class |
|---|---|---|
| `NotImplementedError` -- batched vectorized k-space Ewald | `_coulomb_matrix_batch.py:503` | Nothing in that branch reads `n_orb`, `max_ang` or the basis layout. It is a batched k-space gap (PHY-04, Phase 8.1), classified **out** of D-04's row set by plan 05-06 and resolved anyway because it previously printed an apology and returned `None`. Call with `do_vec=False`, the default. |
| `NotImplementedError` -- ML-SK has no f channel | `_slater_koster_pair.py:1048` | See §1. |
| `SKFRadialGridStepMismatchError` | `_bond_integral.py:1088` | Subclasses `ValueError`, not `NotImplementedError`, and deliberately: the four `F*` classes mark capability that could land later, whereas an `SKFPATH` whose files disagree on radial grid *step* is not coherent as a parameter set at all. REG-05, satisfied. |

---

## 3. What is supported today

**Single-system, closed-shell f energy on a uniform-grid `SKFPATH`, single-shot or
self-consistent.**

| Property | Value |
|---|---|
| Entry point | `ESDriver.forward(do_scf=False)` **and `forward(do_scf=True)`** -- the self-consistent charge loop settles for the validated case as of phase 6 plan 06-01 (SCC-01) |
| System | one structure at a time (not `StructureBatch`) |
| Occupation | closed shell only |
| Energy | band + repulsion; `e_coul` is 0 on the single-shot path (decision D-11) |
| Coulomb, per-atom | fully implemented for f |
| Coulomb, shell-resolved | **now builds for an f system.** `MAGNETIC_HUBBARD_LDEP` set makes `ESDriver.forward` populate `structure.C_sr` and `structure.dCC_sr` with all sixteen shell-pair blocks (SCC-02). It is still built *alongside* the per-atom matrix and has no consumer: `energy()` and `SCFx` take `(Nats, Nats)` with per-atom charges, and threading shell-resolved charges is deferred (D-11) |
| Parameters | an `SKFPATH` whose files share one radial grid step (REG-05); mixed *lengths* are fine, mixed *steps* refuse |
| Validated case | the isolated **Eu-N diatomic**, `tests/f_orbital_data/` |

**Convergence caveat, carried from plan 06-01.** The self-consistent loop settles for Eu-N
because the Krylov accelerator is switched off for f systems by
`ESDriver._krylov_params_for_f_interim`. The accelerator itself is not repaired, only
stood down; `structure.krylov_disabled_for_f` says which path a result came from, and
`structure.scf_iter_count` is the pass count, or `-1` if the loop gave up.

H0/S assembly for f is real, not zero-filled: `tests/test_orbital_count_guards.py::
test_f_pair_reaches_single_system_sk_assembly` measures the Eu f / N block of `H0` at
0.606 eV and of `S` at 0.0279. A zero block is what the silent-drop failure produced and is
indistinguishable from a correct result by shape, finiteness or symmetry, which is why the
measurement exists.

The case document is `tests/f_orbital_data/README-EU-N-CASE.md`; the gate is
`tests/test_eu_n_scan.py`, a 21-point scan whose minimum sits at 2.40 A, interior to the grid
and inside the `2.655 A +/- 20%` band.

### The validation depth, stated because a capability list without one is misleading

`04-VALIDATION.md` records four things a green Eu-N scan still does not sample. All four were
reviewed and accepted at the plan 04-05 human sign-off, and each traces to an earlier locked
decision rather than being an open discovery:

1. **The band absorbs real errors.** D-20/D-22 mandate a ~10-20% band; an f-block bug shifting
   the minimum by 15% passes. It is a smoke test, not a correctness proof. The band may not be
   tightened or widened (threat T-04-08).
2. **Single-shot is not SCF.** D-11 ships one diagonalization, so charge-self-consistency bugs
   cannot surface by construction. Phase 6 (SCC-01) is where they can.
3. **A genuinely open-shell ion is treated closed-shell.** Eu 4f7 is open-shell; D-12 defers
   spin, so any observable sensitive to spin polarization is untested.
4. **The shell-resolved f plumbing is validated but never consumed.** D-14 built and tested the
   shapes and values; the single-shot path does not read them, so the integration between the
   two is unsampled until SCC-03.

A fifth, from the same section: **only the Eu-N pair is exercised.** Seven of the nine SKF
fixtures go untouched; the Ga paths and all periodic cases are out of scope (PHY-08, v2).

And one more, recorded at sign-off and explicitly accepted rather than left implicit: the
Eu-N curve is still climbing at 3.60 A and does not reach a dissociation asymptote inside the
scanned range. Expected for single-shot non-SCC energy over this range; extending the scan
becomes meaningful once self-consistency lands.

---

## 4. Known open issues

### 4.1 CH4 SCF does not converge, and the residual grows

`tests/test_scf.py`'s CH4 case does not converge within `SCF_MAX_ITER`, and the residual
**grows** rather than stalling: `0.155 -> 0.389 -> 0.466`.

- **Governed by** Phase 4 decision **D-13**: non-convergence warns and returns the last
  iterate with a convergence flag, rather than raising. So the behaviour is decided, not
  accidental -- but the divergence itself is not.
- **Belongs to Phase 6 (SCC-01).** It is an SCF defect on an **f-free** system, so it is not
  an f-support question at all; it is the SCF loop this milestone never made self-consistent.
- **Not fixed, worked around or asserted here.** This plan modifies no production code.
- **Ledger status.** It was `.planning/WINDOWS.md` entry 1, recorded in Phase 4 and marked
  `fixed` on 2026-08-02 by commit `4dbffaa` -- which found that a `Ep = 0.000039` placeholder
  in mio-1-1's `H-H.skf` gave hydrogen four orbitals instead of one, making the overlap matrix
  indefinite at -0.159 so that `S^(-1/2)` was undefined and CH4 could never converge. CH4 now
  converges in 6 iterations. The entry is closed and the ledger's `open_count` is 0.

  It is recorded here anyway, and deliberately: the *worked example* is the most useful thing
  this milestone produced about silent basis assumptions, and the SCF loop itself is still
  single-shot for f systems regardless.

### 4.2 SEDACS's force path consumes `dS` with no f guard

Found while writing this document, on 2026-08-03. **Not fixed** -- this is a documentation
plan and the prohibition is explicit; recorded and surfaced instead.

`sedacs/sedacs_interface.py` reaches Slater-Koster assembly by calling `H0_and_S_vectorized`
directly (`:507`), bypassing `ESDriver`. `get_forces_on_rank` then consumes
`ch_structure.dS` at `:671` and `:675-676`. Neither call site invokes
`ESDriver._require_f_derivatives`, and `grep -rn "raise F" src/dftorch/sedacs/` returns
nothing. So a 16-orbital atom in a SEDACS partition would contribute forces assembled from
the exactly-zero f derivative blocks, with no refusal -- which is the `FDerivativeUnsupportedError`
failure mode reproduced one module away from its guard.

**Bounded, currently, by two facts:**

- The external `sedacs` package is absent, so the module does not import here (§1).
- Plan 05-06's sweep of `src/dftorch/sedacs/` returned **zero** orbital-count records, and
  that zero is gated by `test_sedacs_result_is_stated`. The sweep matched hardcoded 1/4/9/16
  comparisons; this site has none, because it takes `n_orb` from a tensor shape
  (`ch_structure.dS.shape[1]`). The audit is not wrong -- the site is genuinely outside its
  row set -- but the gap is real.

**Owner:** PHY-05, v2. Whoever schedules SEDACS f support must add
`_require_f_derivatives` at the SEDACS force entry point, or state why not.

### 4.3 Two open items in `deferred-items.md` that touch numbers, not support

Neither is an f-support question; both are recorded so this document is not read as claiming
the numeric baseline is unqualified.

- **Item 1:** `water8_mio_full` reproduces its pinned energy to 5e-13 eV rather than
  bit-exactly -- inside its own 1e-8 eV band, and verified not to be caused by the D-01
  radial-lookup change. Owner: the REG-02 baseline.
- **Item 2:** the CH4 derivative checksum `CH4_DH0_ABS_SUM` misses its literal by 3.4e-13.
  Owner: plan 05-01's radial-lookup regression gate.

Both are environment-level numeric drift against literals recorded on an earlier machine
state, not behaviour changes.

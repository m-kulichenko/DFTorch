---
status: resolved
trigger: "During Phase 05 execute-phase post-wave gating, tests/test_radial_grid.py::test_ch4_h0_s_checksums_are_unchanged fails because dh0_abs is 777.8475256952825 while CH4_DH0_ABS_SUM is 777.8475256952822. The difference is about 3.4e-13. User selected Fix now before Wave 2."
created: 2026-07-31T21:14:45Z
updated: 2026-07-31T21:52:35Z
---

## Current Focus

hypothesis: "CONFIRMED: exact equality against a scalar floating-point reduction is an invalid bit-identity oracle; the tensor is unchanged, but reduction order/environment changes its last bits"
test: complete
expecting: complete
next_action: none; session archived and knowledge-base pattern recorded

reasoning_checkpoint:
  hypothesis: Exact scalar equality fails because non-associative reduction order changes the final two ULPs even when every dH0 tensor byte is unchanged.
  confirming_evidence:
    - HEAD and 742f8cc dH0 raw-byte SHA-256 digests are identical.
    - Equivalent reductions of the same tensor span several ULPs, and the pinned literal is one of those results.
    - All global-versus-per-pair torch.equal assertions pass before the scalar assertion fails.
  falsification_test: If the focused tests still fail after allowing exactly four ULPs, or any torch.equal assertion fails, the root-cause model is incomplete and the patch must be reverted/reinvestigated.
  fix_rationale: A four-ULP absolute-only bound removes reduction-order false positives without changing production arithmetic or weakening the exact elementwise identity gate.
  blind_spots: Four ULPs is evidence-based for the observed equivalent reductions on the current stack; it is intentionally not claimed as a universal cross-platform bound for arbitrary tensor sizes.

## Symptoms

expected: The CH4 radial-grid checksum regression gate passes while still detecting meaningful Hamiltonian drift.
actual: Exact float equality fails by approximately 3.4e-13; Plan 05-03 did not touch tests/test_radial_grid.py or the Hamiltonian code path.
errors: "assert 777.8475256952825 == 777.8475256952822 at tests/test_radial_grid.py:787"
reproduction: "From /Users/abavera/DFTorch run uv run python -m pytest tests/test_radial_grid.py::test_ch4_h0_s_checksums_are_unchanged -q --tb=short."
started: Reproduces at current HEAD after Plan 05-03 and was reproduced before Plan 05-03 source changes; relevant recent commits include 1d5acce/742f8cc for Plan 05-01 and earlier H0/S work.

## Eliminated

- hypothesis: Recent D-01 per-pair lookup caused a real elementwise dH0 drift.
  evidence: Both tests reached the literal assertion. In test_ch4_h0_s_bit_identical_after_per_pair_lookup, all torch.equal checks, including dH0_global versus dH0_per_pair, passed before dh0_abs failed.
  timestamp: 2026-07-31T21:23:03Z
- hypothesis: The dH0 absolute-sum computation is nondeterministic within a fixed process.
  evidence: Ten independent CH4 assemblies in one process all returned exactly 777.8475256952825.
  timestamp: 2026-07-31T21:24:01Z
- hypothesis: Process initialization or CPU scheduling makes the reduction nondeterministic across fresh Python processes.
  evidence: Five fresh pytest processes all reported exactly 777.8475256952825.
  timestamp: 2026-07-31T21:25:13Z
- hypothesis: Current Phase 05 source changes caused the flat dH0 checksum to move from the 742f8cc literal.
  evidence: The exact source and test from commit 742f8cc, extracted outside the working tree and run under the current environment, fail identically at 777.8475256952825 versus 777.8475256952822.
  timestamp: 2026-07-31T21:32:41Z
- hypothesis: Current global-fallback dH0 differs elementwise from commit 742f8cc despite the same scalar failure.
  evidence: Current HEAD and the isolated 742f8cc snapshot produced identical SHA-256 digest e0bfb08f682d4692e1153556120cc36b2b2186a5bb635396cb658cc309868156 over contiguous dH0 bytes.
  timestamp: 2026-07-31T21:36:28Z

## Evidence

- timestamp: 2026-07-31T21:16:03Z
  checked: .planning/debug/knowledge-base.md
  found: No knowledge base exists.
  implication: There is no prior recorded diagnosis to bias or accelerate this investigation.
- timestamp: 2026-07-31T21:17:18Z
  checked: repository search for CH4 checksum symbols and H0/S call sites
  found: tests/test_radial_grid.py pins CH4_DH0_ABS_SUM and asserts exact equality in two tests; the value is documented as recorded on the pre-D-01 path. Phase 05-03 summary independently records the same 3.410605131648481e-13 mismatch and says it predates 05-03 changes.
  implication: The regression concerns a scalar reduction oracle shared by two gates, not a 05-03 source path; both true algorithmic drift and last-bit reduction sensitivity remain possible.
- timestamp: 2026-07-31T21:19:32Z
  checked: tests/test_radial_grid.py lines 1-1200
  found: The literal is the exact float64 result of dH0.abs().sum().item(). The post-D-01 test separately checks torch.equal(dH0_global, dH0_per_pair) before checking the literal. The comment intentionally chose == to detect any movement, but this exactness applies to a newly executed reduction, not to stored tensor bytes.
  implication: The test has a robust elementwise oracle for D-01 and a distinct scalar-reduction oracle whose last bits can move without any element moving; the failing scalar alone does not establish Hamiltonian drift.
- timestamp: 2026-07-31T21:20:12Z
  checked: remainder of tests/test_radial_grid.py and attempted dftorch/_h0ands.py
  found: The test module has no additional checksum mechanics. The package is not rooted at dftorch/ in the repository, so the attempted implementation path did not exist.
  implication: The test reading is complete; source layout must be resolved rather than assumed.
- timestamp: 2026-07-31T21:22:07Z
  checked: complete src/dftorch/_h0ands.py
  found: The only D-01 branch chooses idx_use/dx_use; all downstream Slater-Koster assembly, reshape, and antisymmetrization are shared. The per-pair test calls the same function twice and compares all four output tensors with torch.equal before reducing them.
  implication: A passing dH0 torch.equal assertion is a direct falsification of D-01-induced element drift, while an exact scalar failure afterward isolates the issue to the pinned literal/reduction boundary.
- timestamp: 2026-07-31T21:23:03Z
  checked: original checksum test and post-D-01 bit-identity test at current HEAD
  found: Both fail only at dh0_abs == CH4_DH0_ABS_SUM with 777.8475256952825 versus 777.8475256952822. The post-D-01 test passed torch.equal for H0, S, dH0, and dS first.
  implication: The lookup rewrite does not move any dH0 element. The failing gate is the reduction literal, not the Hamiltonian/derivative tensor comparison.
- timestamp: 2026-07-31T21:24:01Z
  checked: ten same-process CH4 dH0 assemblies under run_with_float64
  found: Every dH0.abs().sum().item() result was 777.8475256952825.
  implication: The mismatch is deterministic within this process; it is not a race or intermittent assembly failure.
- timestamp: 2026-07-31T21:25:13Z
  checked: five fresh-process reproductions
  found: Every process reported 777.8475256952825 against 777.8475256952822.
  implication: This is stable environment-specific arithmetic, not nondeterminism from process initialization or scheduling.
- timestamp: 2026-07-31T21:27:02Z
  checked: equivalent reductions of one fixed float64 dH0 tensor with shape 3x20x20
  found: Flat torch sum is 777.8475256952825; staged torch axis reductions are exactly the pinned 777.8475256952822; math.fsum and NumPy give 777.8475256952823; Python forward/reverse sums give 777.8475256952819 and 777.8475256952818. The actual binary difference from the literal is 2 ULP (2.2737367544323206e-13), despite the planning note's decimal-string delta claim.
  implication: The exact same tensor admits several mathematically equivalent checksums spanning a handful of ULPs, and the pinned value is itself one of them. Exact scalar equality is therefore reduction-order-sensitive and cannot by itself prove elementwise Hamiltonian drift.
- timestamp: 2026-07-31T21:28:33Z
  checked: git log and blame for the test and H0/S path
  found: Commit 742f8cc introduced the literal and the flat dH0.abs().sum().item() assertion together; commit 1d5acce is the only later commit touching tests/test_radial_grid.py or src/dftorch/_h0ands.py. No later Phase 05-03 commit appears in the path history.
  implication: The remaining real-drift branch is narrowly limited to 1d5acce's D-01 implementation or environment drift since 742f8cc was recorded.
- timestamp: 2026-07-31T21:30:46Z
  checked: diff from 742f8cc through HEAD for the CH4 H0/dH0 path
  found: 1d5acce added a per-pair lookup branch while preserving the old global fallback expression verbatim. The failing first test explicitly selects that fallback. No downstream Slater-Koster, reshape, or dH0 antisymmetrization arithmetic changed, and no later commit touched _slater_koster_pair.py.
  implication: Source-drift evidence is absent; running the exact 742f8cc snapshot under the current environment is the narrow counterfactual that can eliminate it conclusively.
- timestamp: 2026-07-31T21:32:41Z
  checked: isolated historical snapshot at commit 742f8cc
  found: Its original test at historical line 774 reports the same flat-reduction value 777.8475256952825 against the same literal 777.8475256952822.
  implication: Current Phase 05 source changes are conclusively not required to trigger the failure. The defect is in treating one floating-point reduction result as a bit-portable historical oracle.
- timestamp: 2026-07-31T21:34:40Z
  checked: dependency history, current numeric environment, and complete 05-01 summary
  found: pyproject.toml and uv.lock are unchanged since 742f8cc; current environment is Python 3.11.15, macOS arm64, torch 2.10.0, NumPy 2.4.4. The summary says 96 tests passed and describes exact reductions as a pre-change oracle, but the exact historical snapshot now contradicts that claim under the locked environment.
  implication: The summary cannot establish that the literal is portable or even that the flat assertion was validated in this exact runtime configuration. A byte digest comparison can still settle whether source history moved dH0 elements.
- timestamp: 2026-07-31T21:36:28Z
  checked: SHA-256 of raw contiguous dH0 float64 bytes for HEAD and 742f8cc
  found: Both digests are exactly e0bfb08f682d4692e1153556120cc36b2b2186a5bb635396cb658cc309868156, and both flat sums are 777.8475256952825.
  implication: There is no elementwise Hamiltonian derivative drift from the checksum-introducing commit to HEAD in the current environment. Only the exact scalar oracle is wrong/unstable.
- timestamp: 2026-07-31T21:45:03Z
  checked: test-only fix in tests/test_radial_grid.py
  found: Added _assert_ch4_checksum with rel_tol=0.0 and abs_tol=4*math.ulp(expected); routed all historical CH4 scalar reductions through it; preserved all four torch.equal assertions unchanged; updated provenance comments and docstrings.
  implication: The patch targets only the invalid scalar-oracle semantics and leaves production Hamiltonian code plus elementwise identity enforcement intact.
- timestamp: 2026-07-31T21:45:48Z
  checked: two previously failing CH4 checksum tests
  found: Both passed. The post-D-01 test therefore also passed all unchanged torch.equal checks for H0, S, dH0, and dS.
  implication: The original failure is fixed without weakening elementwise bit identity.
- timestamp: 2026-07-31T21:46:30Z
  checked: complete tests/test_radial_grid.py module
  found: 13 tests passed.
  implication: The checksum fix does not regress adjacent radial-grid characterization, lookup, or fixture behavior.
- timestamp: 2026-07-31T21:47:17Z
  checked: explicit helper boundary probe using math.nextafter
  found: A checksum exactly four ULP above CH4_DH0_ABS_SUM passed; five ULP above raised AssertionError.
  implication: The implementation enforces the requested bound exactly and continues to reject movement beyond the measured reduction-noise envelope.
- timestamp: 2026-07-31T21:48:42Z
  checked: full project pytest suite
  found: 152 tests passed (progress reached 100% with no failures or errors).
  implication: The test-only checksum repair is compatible with the complete current project behavior.
- timestamp: 2026-07-31T21:49:37Z
  checked: final diff and git diff --check
  found: The only intentional source diff is tests/test_radial_grid.py (53 insertions, 23 deletions); git diff --check reported no whitespace errors. Generated __pycache__ files and the debug directory remain separate worktree state.
  implication: The code change is ready for an atomic single-file commit without staging unrelated artifacts.
- timestamp: 2026-07-31T21:51:10Z
  checked: atomic code commit with repository hooks
  found: Commit f7575d4 (fix(05-03): tolerate checksum reduction ULPs) succeeded and contains only tests/test_radial_grid.py.
  implication: The verified fix is recorded without unrelated generated caches or debug artifacts in the code commit.

## Resolution

root_cause: The test uses exact Python-float equality on dH0.abs().sum(), a non-associative floating-point reduction, as if it were an elementwise bit-identity oracle. The same unchanged 3x20x20 float64 tensor yields several last-bit results under equivalent summation orders, and the exact 742f8cc tensor bytes equal HEAD while both flat reductions produce 777.8475256952825. Therefore CH4_DH0_ABS_SUM=777.8475256952822 is an over-strict or mismatched reduction literal, not evidence of Hamiltonian drift.
fix: Added a test-only four-ULP scalar-checksum helper with zero relative tolerance, applied to the historical CH4 H0/S/dH0/dS reductions in both gates; all elementwise torch.equal checks remain exact.
verification: Focused checksum tests passed; all 13 radial-grid tests passed; four-ULP boundary accepted and five-ULP boundary rejected; full 152-test suite passed; committed as f7575d4.
files_changed: [tests/test_radial_grid.py]

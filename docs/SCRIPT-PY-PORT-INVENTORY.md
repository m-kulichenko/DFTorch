# `script.py` Port Inventory

This inventory records the disposition of the validation code that shipped in
`src/dftorch/script.py` before Phase 05 plan 03 deleted it.  The source audited
here is commit `56091af03bc915e984234574da05acebdaea7889`.

The preservation rule is narrow but strict: checks that assert behavior must
remain executable under pytest; the raw-SKF metadata oracle must remain
independent of the production parser; import-by-filesystem and command-line
reporting scaffolding must not move into the test suite merely to preserve code.

## Mechanical sweep

The authoritative row list was generated before deletion with:

```text
grep -n '^def \(check\|run\|expected\|parse\)_' src/dftorch/script.py
```

After deletion the same list can be regenerated from history with:

```text
git show 56091af:src/dftorch/script.py | grep -n '^def \(check\|run\|expected\|parse\)_'
```

The command emits 28 functions.  Exactly 28 rows appear in the table below.
The narrower plan-time fact claiming 30 `check_`/`run_` functions was stale:
the audited file has 16, confirmed with
`grep -c '^def \(check\|run\)_' src/dftorch/script.py`.

Disposition meanings:

- `ported`: Phase 05-03 has a named pytest test or independent test helper that
  carries the check.
- `covered`: a named test that existed at the audit baseline already exercised
  the same check.
- `dropped`: no behavior assertion is lost; the evidence states why the item is
  plumbing or reporting rather than a check.

| Function | Line | What it checks | Disposition | Evidence |
|---|---:|---|---|---|
| `expected_shell_metadata` | 154 | Derives orbital count and highest present/occupied shell from presence and occupation alone. | ported | `tests/skf_header_oracle.py::expected_shell_metadata`; exercised by `test_shell_metadata_matches_independent_derivation`. |
| `parse_expected_homonuclear_metadata` | 176 | Re-parses raw homonuclear SKF header tokens into onsite, Hubbard, occupation, and shell metadata without using the production parser. | ported | `tests/skf_header_oracle.py::parse_expected_homonuclear_metadata`; exercised by `test_oracle_parses_both_header_formats`, `test_onsite_energies_match_independent_parse`, `test_hubbard_values_match_independent_parse`, and `test_reference_occupations_match_independent_parse`. |
| `check_metadata_from_single_read` | 426 | Compares metadata returned by one production SKF read with independent header expectations. | covered | `test_f_orbital_skf_parser_and_spline_gate` calls the test-only `run_bond_integral_tests`, which performs this comparison; the new independent assertions also run in `test_skf_metadata_oracle.py`. |
| `check_one_skf` | 445 | Checks a fixture parses to 40 channels and reconstructs every source-grid electronic row within tolerance. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_simple_canonical_channels` | 505 | Checks the simple 20-column layout maps to the canonical 40-channel layout with non-simple channels zero-filled. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_simple_homonuclear_metadata` | 561 | Checks simple-format homonuclear metadata and the absence of an f shell. | covered | `test_f_orbital_skf_parser_and_spline_gate`; the independent counterpart is also exercised by `test_oracle_parses_both_header_formats`. |
| `check_extended_fixture_channel_width` | 604 | Checks extended fixtures retain all 40 channels. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_pair_name_and_path_helpers` | 617 | Checks compact/dashed pair names and path resolution, including two-letter symbols. | covered | `test_f_orbital_skf_parser_and_spline_gate` and `test_compact_only_f_orbital_skf_directory`. |
| `check_skipped_shell_errors` | 649 | Checks non-nested shell layouts fail explicitly. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_simple_spline_reconstruction` | 667 | Checks a minimal simple SKF reconstructs its source rows through cubic splines. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_source_grid_reconstruction` | 690 | Checks cubic spline values reproduce source-grid channel values and the added cutoff row. | covered | `test_f_orbital_skf_parser_and_spline_gate`. |
| `check_get_skf_tensors_metadata` | 751 | Checks aggregated `get_skf_tensors` metadata for every element against homonuclear files. | covered | `test_f_orbital_skf_parser_and_spline_gate`; raw-header independence is separately pinned by `test_onsite_energies_match_independent_parse`, `test_hubbard_values_match_independent_parse`, and `test_reference_occupations_match_independent_parse`. |
| `run_bond_integral_tests` | 841 | Runs the parser, canonical-channel, filename, shell-validation, metadata, and spline gates as one suite. | covered | `test_f_orbital_skf_parser_and_spline_gate` and `test_compact_only_f_orbital_skf_directory`. |
| `check_constants_against_expected` | 993 | Checks `Constants` orbital counts, shell flags, onsite energies, Hubbard values, and occupations against fixture expectations. | covered | `test_f_orbital_constants_metadata_gate`; independent raw-header comparisons are also carried by the four parametrized metadata tests in `test_skf_metadata_oracle.py`. |
| `run_constants_tests` | 1084 | Builds `Constants` for the fixture element set and invokes the constants metadata gate. | covered | `test_f_orbital_constants_metadata_gate`. |
| `expected_metadata_by_element` | 1201 | Builds per-element expected metadata for structure layout derivation. | ported | `tests/test_skf_metadata_oracle.py::_expected_eu_n_layout`; exercised by `test_structure_ao_layout_matches_expected_derivation` and `test_batch_structure_layout_matches_single_structure`. |
| `expected_local_starts` | 1215 | Derives local shell starts, using -1 for absent shells. | ported | `tests/skf_header_oracle.py::expected_local_starts`; exercised by `test_structure_ao_layout_matches_expected_derivation`. |
| `expected_local_ends` | 1219 | Derives inclusive local shell ends. | ported | `tests/skf_header_oracle.py::expected_local_ends`; exercised by `test_structure_ao_layout_matches_expected_derivation`. |
| `expected_ao_labels` | 1226 | Expands present shells into the independently written AO label order. | ported | `tests/skf_header_oracle.py::expected_ao_labels`; exercised by `test_basis_tables_match_expected_tables` and `test_structure_ao_layout_matches_expected_derivation`. |
| `expected_ao_shell_types` | 1236 | Expands present shells into AO shell-type identifiers. | ported | `tests/skf_header_oracle.py::expected_ao_shell_types`; exercised by `test_basis_tables_match_expected_tables` and `test_structure_ao_layout_matches_expected_derivation`. |
| `expected_diagonal_values` | 1246 | Expands shell onsite energies over each shell's AO dimension. | covered | `test_f_orbital_structure_metadata_gate` exercises the test-only structure layout gate that compares `Structure.diagonal`. |
| `expected_d0_values` | 1256 | Expands reference occupations into the closed-shell AO density diagonal. | covered | `test_f_orbital_structure_metadata_gate` exercises the test-only structure layout gate that compares `Structure.D0`. |
| `expected_shell_hubbard_values` | 1267 | Selects Hubbard values for present shells in shell order. | covered | `test_f_orbital_structure_metadata_gate` compares `Structure.Hubbard_U_sr`; `test_eu_n_hubbard_u_sr_values` supplies an additional direct gate. |
| `expected_el_per_shell_values` | 1273 | Selects reference occupations for present shells. | covered | `test_f_orbital_structure_metadata_gate` compares `Structure.el_per_shell`; `test_eu_n_electrons_per_shell` supplies an additional direct gate. |
| `expected_shell_type_values` | 1279 | Selects shell identifiers for present shells. | covered | `test_f_orbital_structure_metadata_gate` compares `Structure.shell_types`; `test_eu_n_shell_types_label_the_f_shell` supplies an additional direct gate. |
| `check_single_structure_layout` | 1316 | Checks atom-major AO ranges, shell ranges, labels, onsite diagonal, reference density, Hubbard values, and shell occupations. | covered | `test_f_orbital_structure_metadata_gate`; the AO placement subset is independently re-derived by `test_structure_ao_layout_matches_expected_derivation`. |
| `check_batch_structure_layout` | 1418 | Checks `StructureBatch` per-member AO/shell layout, global offsets, diagonals, and padding. | ported | `test_batch_structure_layout_matches_single_structure`; it uses two f-containing Eu-N members, so no f-free narrowing was needed. |
| `run_structure_tests` | 1539 | Builds single and batched structures and runs the complete structure metadata gate. | covered | `test_f_orbital_structure_metadata_gate`. |

## Independent expectation tables

These are data assets rather than functions, so they are intentionally outside
the 28-row grep equality above.

| Symbol | Line | What it checks | Disposition | Evidence |
|---|---:|---|---|---|
| `EXPECTED_SHELL_DIMS` | 1143 | AO count per s/p/d/f shell. | ported | `tests/skf_header_oracle.py`; `test_basis_tables_match_expected_tables`. |
| `EXPECTED_SHELL_LOCAL_STARTS` | 1144 | Local start of each shell in a 16-AO basis. | ported | `tests/skf_header_oracle.py`; `test_basis_tables_match_expected_tables`. |
| `EXPECTED_SHELL_TYPE_IDS` | 1145 | Integer identifiers for s/p/d/f shells. | ported | `tests/skf_header_oracle.py`; `test_basis_tables_match_expected_tables`. |
| `EXPECTED_AO_LABEL_TEMPLATE` | 1146 | Independent written AO label order, including the seven cubic f harmonics. | ported | `tests/skf_header_oracle.py`; `test_basis_tables_match_expected_tables`. |
| `EXPECTED_AO_SHELL_TEMPLATE` | 1164 | Independent AO-to-shell mapping. | ported | `tests/skf_header_oracle.py`; `test_basis_tables_match_expected_tables`. |

## Supporting and reporting functions

For completeness, this second table dispositions every remaining top-level
function in the deleted file.  These rows are deliberately outside the
mechanical check/derivation count because none is a check function.

| Function | Line | Role | Disposition | Evidence or reason |
|---|---:|---|---|---|
| `find_project_root` | 59 | Locates the repository for a directly executed runtime script. | dropped | Pytest resolves fixtures from `Path(__file__)`; no runtime-script root search remains necessary. |
| `ensure_fake_dftorch_package` | 73 | Injects a synthetic `dftorch` package into `sys.modules`. | dropped | D-03 forbids fake-package machinery because it diverges from installed-package behavior. Tests use the editable installation. |
| `load_dftorch_module` | 87 | Imports package files directly by filesystem path. | dropped | D-03 forbids the filesystem loader. `tests/skf_validation_support.py::import_dftorch_module` delegates to normal `importlib.import_module`. |
| `read_data_lines` | 108 | Reads non-comment SKF data lines. | ported | `tests/skf_header_oracle.py::read_data_lines`, using only `pathlib`. |
| `split_skf_pair` | 117 | Splits compact and dashed pair names. | ported | `tests/skf_header_oracle.py::split_skf_pair_name_independently`; unlike the old helper it does not call the production splitter. |
| `resolve_homonuclear_skf` | 122 | Locates a dashed or compact homonuclear fixture. | ported | `tests/skf_header_oracle.py::resolve_homonuclear_skf_independently`; it does not call the production resolver. |
| `get_original_electronic_row_count` | 126 | Derives source electronic-row count from raw grid metadata. | ported | `tests/skf_header_oracle.py::original_electronic_row_count`; exercised through `test_f_orbital_skf_parser_and_spline_gate`. |
| `assert_int_metadata` | 269 | Hand-written integer comparator. | dropped | It checks nothing independently; pytest assertion rewriting provides the failure report at each metadata assertion. |
| `assert_float_metadata` | 275 | Hand-written tolerance comparator. | dropped | It checks nothing independently; the port uses direct `abs(got - expected) <= ATOL` assertions. |
| `assert_bool_metadata` | 291 | Hand-written boolean-list comparator. | dropped | It checks nothing independently; the port uses direct list equality assertions. |
| `max_error_location` | 297 | Formats the largest spline error for script output. | dropped | Diagnostic/reporting scaffolding, not an invariant; pytest failures identify the relevant case and delta. |
| `format_skf_row` | 305 | Formats generated fixture rows. | dropped | No independent assertion; retained parser coverage uses a pre-existing test-only fixture writer. |
| `write_minimal_simple_skf` | 309 | Writes a temporary simple-format parser fixture. | dropped | It is fixture plumbing, not a check. The behavior remains covered by `test_f_orbital_skf_parser_and_spline_gate` through its existing test-only support module. |
| `read_skf_as_matrix` | 334 | Threads mutable metadata tensors through production parsing helpers. | dropped | It is production-parser plumbing and is prohibited from the independent oracle; the existing parser gate still exercises that path directly. |
| `make_metadata_tensors` | 355 | Allocates production parser output tensors. | dropped | No independent behavior; used only by the existing parser gate's test support. |
| `metadata_tuple_to_dict` | 403 | Names the production parser's metadata tuple fields. | dropped | No independent behavior; raw-header expectations now live in a dictionary produced without the production tuple. |
| `collect_elements_from_skf_dir` | 738 | Collects fixture element symbols. | ported | `tests/skf_header_oracle.py::collect_elements_independently`, with its own element table and filename splitter. |
| `write_test_xyz` | 984 | Writes synthetic coordinates for Constants/Structure construction. | dropped | Fixture plumbing was replaced by `_write_single_element_xyz` and `_write_eu_n_xyz`; it asserts nothing itself. |
| `build_test_constants` | 1185 | Constructs `Constants` for structure checks. | dropped | Fixture plumbing was replaced by `_build_constants`; it asserts nothing itself. |
| `assert_list_equal` | 1284 | Hand-written list comparator. | dropped | It checks nothing independently; the port uses plain pytest assertions. |
| `assert_tensor_int_list` | 1289 | Hand-written integer tensor comparator. | dropped | It checks nothing independently; the port compares `.tolist()` results directly. |
| `assert_tensor_bool_list` | 1295 | Hand-written boolean tensor comparator. | dropped | It checks nothing independently; the port compares `.tolist()` results directly. |
| `assert_tensor_float_list` | 1301 | Hand-written float tensor comparator. | dropped | It checks nothing independently; the port uses direct per-value tolerance assertions. |
| `main` | 1628 | CLI timing, reporting, and direct execution wrapper. | dropped | Pytest owns collection, reporting, and exit status; no runtime-package CLI or entry point exists. |

## Independence backstop

The independent parser and derivations live in `tests/skf_header_oracle.py`, a
standard-library-only module with its own element table, filename splitter,
header parser, and AO expectation tables.  The production `Constants` and
`Structure` objects are imported only by `tests/test_skf_metadata_oracle.py`.

`test_oracle_does_not_use_production_parser` inspects each oracle helper's
source plus the oracle module AST.  It rejects imports from `dftorch` and the
production entry points `get_skf_tensors`, `read_skf_table`,
`channels_to_matrix`, `cubic_spline_coeffs`, `_split_skf_pair_name`,
`_resolve_skf_path`, and `_validate_nested_shells`.  This is the machine gate
that prevents the expected values from silently converging on the code under
test.

## Deletion precondition

The runtime file may be deleted only after all of the following are green:

```text
uv run pytest tests/test_skf_metadata_oracle.py -q
uv run pytest -q
```

The Task 2 deletion must remain a separate later commit.  `build/lib/dftorch/script.py`
is a stale build artifact and is not an import, package declaration, or workflow
reference; Phase 05-03 intentionally leaves it untouched.

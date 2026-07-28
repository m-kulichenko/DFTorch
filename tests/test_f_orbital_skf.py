import importlib.util
import shutil
import sys
from pathlib import Path

import torch


def run_with_float64(fn):
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        for name in [name for name in sys.modules if name == "dftorch" or name.startswith("dftorch.")]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def load_validation_script():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "dftorch" / "script.py"
    spec = importlib.util.spec_from_file_location("dftorch_phase1_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation script from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_xyz(path: Path, elements: list[str], spacing: float = 1.5) -> None:
    """Write atoms evenly spaced along +x, so (L, M, N) = (1, 0, 0) for every pair."""
    lines = [str(len(elements)), "metadata regression"]
    for idx, sym in enumerate(elements):
        lines.append(f"{sym} {spacing * idx:.8f} 0.00000000 0.00000000")
    path.write_text("\n".join(lines) + "\n")


def write_simple_skf(
    path: Path,
    *,
    homonuclear: bool,
    ed: float = 0.0,
    ep: float = 0.0,
    es: float = -0.1,
    ud: float = 0.0,
    up: float = 0.0,
    us: float = 0.01,
    fd: float = 0.0,
    fp: float = 0.0,
    fs: float = 1.0,
) -> None:
    source_rows = [
        [0.01 * (row + 1) + 0.001 * (col + 1) for col in range(20)]
        for row in range(3)
    ]
    lines = ["0.20 4"]
    if homonuclear:
        lines.append(
            f"{ed:.8f} {ep:.8f} {es:.8f} 0.0 "
            f"{ud:.8f} {up:.8f} {us:.8f} {fd:.8f} {fp:.8f} {fs:.8f}"
        )
    lines.extend(
        [
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
            *[" ".join(f"{value:.8f}" for value in row) for row in source_rows],
            "Spline",
            "2 2.0",
            "0.0 0.0 0.0",
            "0.0 1.0 0.0 0.0 0.0 0.0",
            "1.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_structure(validation, project_root: Path, skf_dir: Path, xyz_path: Path, const):
    """Instantiate a single Structure from an already-written XYZ file."""
    structure_mod = validation.load_dftorch_module(project_root, "Structure")
    return structure_mod.Structure(
        {
            "SKFPATH": str(skf_dir),
            "FILENAME": str(xyz_path),
            "T_ELECTRONIC": 1000.0,
            "CHARGE": 0,
            "GRAD_XYZ": False,
            "GRAD_CELL": False,
        },
        const,
        device="cpu",
        ignore_spin=True,
    )


def build_h0_and_s(validation, project_root: Path, skf_dir: Path, elements: list[str], xyz_path: Path, *, rcut: float = 10.0, spacing: float = 1.5):
    """Directly assemble single-system H0/S for ``elements`` without running SCF.

    Mirrors the ``ESDriver.forward`` call sequence (neighbor list, then
    ``H0_and_S_vectorized``) but skips overlap inversion, repulsion, Coulomb,
    SCF, and forces so the H0/S contract can be tested in isolation.
    """
    const = validation.build_test_constants(project_root, skf_dir, None, elements, xyz_path)
    # Constants only needs the species list; rewrite the geometry so callers can
    # pick a separation that lies inside the fixture's radial grid.
    write_xyz(xyz_path, elements, spacing=spacing)
    struct = build_structure(validation, project_root, skf_dir, xyz_path, const)

    nnl_mod = validation.load_dftorch_module(project_root, "_nearestneighborlist")
    h0ands_mod = validation.load_dftorch_module(project_root, "_h0ands")

    (
        _,
        _,
        nnRx,
        nnRy,
        nnRz,
        nnType,
        _,
        _,
        neighbor_I,
        neighbor_J,
        IJ_pair_type,
        JI_pair_type,
    ) = nnl_mod.vectorized_nearestneighborlist(
        struct.TYPE,
        struct.RX,
        struct.RY,
        struct.RZ,
        struct.cell,
        rcut,
        struct.Nats,
        const,
        upper_tri_only=False,
    )

    H0, dH0, S, dS = h0ands_mod.H0_and_S_vectorized(
        struct.TYPE,
        struct.RX,
        struct.RY,
        struct.RZ,
        struct.diagonal,
        struct.H_INDEX_START,
        nnRx,
        nnRy,
        nnRz,
        nnType,
        const,
        neighbor_I,
        neighbor_J,
        IJ_pair_type,
        JI_pair_type,
        const.R_orb,
        const.coeffs_tensor,
    )
    return const, struct, H0, dH0, S, dS


def find_element_with_shells(validation, skf_dir: Path, bond, shell_present: list[bool]) -> str:
    for sym in validation.collect_elements_from_skf_dir(skf_dir, bond):
        metadata = validation.parse_expected_homonuclear_metadata(
            validation.resolve_homonuclear_skf(skf_dir, sym, bond),
            bond,
        )
        if metadata is not None and list(metadata["SHELL_PRESENT"]) == shell_present:
            return sym
    raise AssertionError(f"No element with shell_present={shell_present} found in {skf_dir}")


def assert_simple_format_metadata_case(validation, project_root, bond, skf_dir: Path, tmp_path: Path, element: str) -> None:
    metadata = validation.expected_metadata_by_element(skf_dir, bond, [element])
    xyz_path = tmp_path / f"{element}_metadata.xyz"
    write_xyz(xyz_path, [element])
    const = validation.build_test_constants(project_root, skf_dir, bond, [element], xyz_path)

    structure_mod = validation.load_dftorch_module(project_root, "Structure")
    struct = structure_mod.Structure(
        {
            "SKFPATH": str(skf_dir),
            "FILENAME": str(xyz_path),
            "T_ELECTRONIC": 1000.0,
            "CHARGE": 0,
            "GRAD_XYZ": False,
            "GRAD_CELL": False,
        },
        const,
        device="cpu",
        ignore_spin=True,
    )

    validation.check_constants_against_expected(const, skf_dir, bond, [element])
    validation.check_single_structure_layout(struct, [element], metadata)


def test_f_orbital_skf_parser_and_spline_gate():
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_bond_integral_tests(
            skf_dir,
            bond,
            torch.device("cpu"),
            torch.float64,
        )

    assert run_with_float64(check) == []

def test_compact_only_f_orbital_skf_directory(tmp_path):
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        source_dir = project_root / "tests" / "f_orbital_data"

        for source in source_dir.glob("*.skf"):
            compact_name = source.stem.replace("-", "") + source.suffix
            shutil.copyfile(source, tmp_path / compact_name)

        return validation.run_bond_integral_tests(
            tmp_path,
            bond,
            torch.device("cpu"),
            torch.float64,
        )

    assert run_with_float64(check) == []


def test_f_orbital_constants_metadata_gate():
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_constants_tests(project_root, skf_dir, bond)

    assert run_with_float64(check) == []


def test_f_orbital_structure_metadata_gate():
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_structure_tests(project_root, skf_dir, bond)

    assert run_with_float64(check) == []


def expected_ss_channel_value(validation, project_root, const, struct, channel_name: str):
    """Independently evaluate one radial channel for the single I-J atom pair.

    Returns the mean of the I->J and J->I spline values, which is what the
    symmetrized H0/S s-s entry must equal for a two-atom system.
    """
    sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")
    channel = sk_mod.sk_channel_index(channel_name)

    dR = torch.sqrt(
        (struct.RX[1] - struct.RX[0]) ** 2
        + (struct.RY[1] - struct.RY[0]) ** 2
        + (struct.RZ[1] - struct.RZ[0]) ** 2
    )
    idx = torch.clamp(
        torch.searchsorted(const.R_orb, dR.reshape(1), right=True) - 1,
        0,
        len(const.R_orb),
    )
    dx = dR - const.R_orb[idx][0]

    type_0 = int(struct.TYPE[0])
    type_1 = int(struct.TYPE[1])
    pair_IJ = int(const.pair_lookup[type_0, type_1])
    pair_JI = int(const.pair_lookup[type_1, type_0])

    values = []
    for pair_type in (pair_IJ, pair_JI):
        cs = const.coeffs_tensor[pair_type, idx[0], channel]
        values.append(cs[0] + cs[1] * dx + cs[2] * dx**2 + cs[3] * dx**3)
    return 0.5 * (values[0] + values[1])


def test_sk_channel_lookup_matches_bond_integral_order():
    """HSK-01/HSK-02 guard: SK channel names must track _bond_integral._CHANNELS."""

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")

        assert list(sk_mod.SK_CHANNEL_NAMES) == list(bond._CHANNELS), (
            "SK_CHANNEL_NAMES drifted from _bond_integral._CHANNELS"
        )
        for index, name in enumerate(bond._CHANNELS):
            assert sk_mod.sk_channel_index(name) == index, name

        # SH_shift selects a name prefix, never a numeric block offset.
        assert sk_mod.sk_channel_name("ss0", 0) == "Hss0"
        assert sk_mod.sk_channel_name("ss0", 1) == "Sss0"

        # The legacy `channel + SH_shift * 10` layout would have addressed the
        # s-s Hamiltonian at 9 and the s-s overlap at 19. Under the canonical
        # 40-channel order those are Hdd2 and Hss0 respectively, i.e. the old
        # scheme silently read the wrong integrals.
        assert sk_mod.sk_channel_index("Hss0") == 19
        assert sk_mod.sk_channel_index("Sss0") == 39
        assert bond._CHANNELS[9] == "Hdd2"
        assert bond._CHANNELS[19] == "Hss0"

        # Every base name used by the s/p/d formulas must resolve in both blocks.
        for base in ("ss0", "sp0", "sd0", "pp0", "pp1", "pd0", "pd1", "dd0", "dd1", "dd2"):
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, 0))
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, 1))

        # f channels exist in the table even though the angular formulas do not.
        for base in ("sf0", "pf0", "pf1", "df0", "df1", "df2", "ff0", "ff1", "ff2", "ff3"):
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, 0))
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, 1))

        try:
            sk_mod.sk_channel_index("Hzz9")
        except KeyError:
            pass
        else:
            raise AssertionError("Unknown channel names must raise KeyError")

        assert sk_mod.F_ANGULAR_FORMULAS_AVAILABLE is False, (
            "f angular formulas must stay disabled until the source lock lands"
        )
        assert sk_mod.PAPER_F_AO_ORDER is None
        assert sk_mod.PAPER_TO_STRUCTURE_F_PERMUTATION is None
        assert sk_mod.PAPER_TO_STRUCTURE_F_SIGN is None

        structure_mod = validation.load_dftorch_module(project_root, "Structure")
        assert (
            tuple(sk_mod.STRUCTURE_F_AO_ORDER)
            == tuple(structure_mod.AO_LABEL_TEMPLATE[9:16])
        ), "STRUCTURE_F_AO_ORDER must mirror Structure.AO_LABEL_TEMPLATE offsets 9..15"

        return []

    assert run_with_float64(check) == []


def test_f_free_h0_s_routing_regression(tmp_path):
    """HSK-02/HSK-07: 1-, 4-, and 9-orbital H0/S routes stay intact and correct."""

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        simple_dir = project_root / "tests" / "data_skf_mio-1-1"

        sp_element = find_element_with_shells(validation, simple_dir, bond, [True, True, False, False])
        spd_element = find_element_with_shells(validation, simple_dir, bond, [True, True, True, False])

        # Repository fixtures have no clean s-only element (mio's H declares a p
        # shell), so the 1-orbital route uses the same synthetic simple-format
        # SKF oracle as the Phase 2 metadata regression.
        s_only_dir = tmp_path / "s_only_h0s_skf"
        s_only_dir.mkdir()
        write_simple_skf(s_only_dir / "H-H.skf", homonuclear=True)

        # The synthetic s-only table is tabulated on a 0.2 A grid with 3 points,
        # so its atoms must sit inside that range for the spline to be non-zero.
        cases = [
            ("s_only", s_only_dir, ["H", "H"], 1, 0.30),
            ("sp", simple_dir, [sp_element, sp_element], 4, 1.50),
            ("spd", simple_dir, [spd_element, spd_element], 9, 1.50),
        ]

        for label, skf_dir, elements, expected_n_orb, spacing in cases:
            xyz_path = tmp_path / f"h0s_{label}.xyz"
            const, struct, H0, dH0, S, dS = build_h0_and_s(
                validation, project_root, skf_dir, elements, xyz_path, spacing=spacing
            )

            assert int(const.n_orb[struct.TYPE[0]]) == expected_n_orb, (
                f"{label}: expected n_orb {expected_n_orb}, got {int(const.n_orb[struct.TYPE[0]])}"
            )
            assert H0.shape == (struct.HDIM, struct.HDIM), f"{label}: H0 shape {tuple(H0.shape)}"
            assert S.shape == (struct.HDIM, struct.HDIM), f"{label}: S shape {tuple(S.shape)}"
            assert torch.isfinite(H0).all(), f"{label}: H0 has non-finite entries"
            assert torch.isfinite(S).all(), f"{label}: S has non-finite entries"
            assert torch.isfinite(dH0).all(), f"{label}: dH0 has non-finite entries"
            assert torch.isfinite(dS).all(), f"{label}: dS has non-finite entries"
            assert torch.allclose(H0, H0.transpose(0, 1)), f"{label}: H0 not symmetric"
            assert torch.allclose(S, S.transpose(0, 1)), f"{label}: S not symmetric"

            # Off-diagonal s-s coupling must be non-trivial. The stale
            # `channel + SH_shift * 10` addressing read Hdd2 here, which is
            # identically zero for s-only and sp elements.
            i0 = int(struct.H_INDEX_START[0])
            j0 = int(struct.H_INDEX_START[1])
            assert abs(float(H0[i0, j0])) > 0.0, (
                f"{label}: s-s H0 coupling is exactly zero, the radial channel lookup is wrong"
            )

            expected_h_ss = expected_ss_channel_value(
                validation, project_root, const, struct, "Hss0"
            )
            assert torch.allclose(H0[i0, j0], expected_h_ss), (
                f"{label}: H0 s-s entry {float(H0[i0, j0])} != Hss0 spline value "
                f"{float(expected_h_ss)}"
            )

            expected_s_ss = expected_ss_channel_value(
                validation, project_root, const, struct, "Sss0"
            ) / 27.21138625
            assert torch.allclose(S[i0, j0], expected_s_ss), (
                f"{label}: S s-s entry {float(S[i0, j0])} != Sss0 spline value "
                f"{float(expected_s_ss)}"
            )

            # S must still carry the AO identity contribution on its diagonal.
            assert float(S[i0, i0]) > 0.9, f"{label}: S diagonal lost its identity term"

        return []

    assert run_with_float64(check) == []


def test_eu_containing_h0_s_reaches_f_formula_source_boundary(tmp_path):
    """HSK-01: 16-orbital pairs are routed, then stopped at the f source lock."""

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        skf_dir = project_root / "tests" / "f_orbital_data"
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")

        for label, elements in (("Eu_N", ["Eu", "N"]), ("Eu_Eu", ["Eu", "Eu"])):
            xyz_path = tmp_path / f"h0s_{label}.xyz"
            try:
                build_h0_and_s(validation, project_root, skf_dir, elements, xyz_path)
            except sk_mod.FAngularFormulaSourceError as exc:
                message = str(exc)
                assert "n_orb == 16" in message, message
                assert "source-lock" in message or "source-locked" in message, message
                assert "sf/pf/df/ff" in message, message
            else:
                raise AssertionError(
                    f"{label}: f-containing H0/S assembly silently succeeded; f pairs "
                    "must reach the explicit formula-source boundary"
                )

        return []

    assert run_with_float64(check) == []


def test_f_containing_derivative_paths_are_guarded():
    """HSK-01 / D-09: f pairs must not silently contribute zero derivatives."""

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")
        h0ands_mod = validation.load_dftorch_module(project_root, "_h0ands")
        stress_mod = validation.load_dftorch_module(project_root, "_stress")

        assert h0ands_mod.FAngularFormulaSourceError is sk_mod.FAngularFormulaSourceError
        assert stress_mod.FAngularFormulaSourceError is sk_mod.FAngularFormulaSourceError

        # The batch H0/S route and the analytical stress route both reconstruct
        # only the 1/4/9 orbital masks, so both must reject n_orb == 16 rather
        # than dropping those pairs.
        import inspect

        for module, func_name in (
            (h0ands_mod, "H0_and_S_vectorized_batch"),
            (stress_mod, "_pair_grad_from_sk"),
        ):
            source = inspect.getsource(getattr(module, func_name))
            assert "FAngularFormulaSourceError" in source, (
                f"{func_name} has no explicit f guard"
            )
            assert "16" in source, f"{func_name} does not test for n_orb == 16"

        return []

    assert run_with_float64(check) == []


def test_simple_format_f_free_metadata_regression(tmp_path):
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")

        s_only_dir = tmp_path / "s_only_skf"
        s_only_dir.mkdir()
        write_simple_skf(s_only_dir / "H-H.skf", homonuclear=True)
        assert_simple_format_metadata_case(validation, project_root, bond, s_only_dir, tmp_path, "H")

        simple_dir = project_root / "tests" / "data_skf_mio-1-1"
        sp_element = find_element_with_shells(validation, simple_dir, bond, [True, True, False, False])
        spd_element = find_element_with_shells(validation, simple_dir, bond, [True, True, True, False])

        assert_simple_format_metadata_case(validation, project_root, bond, simple_dir, tmp_path, sp_element)
        assert_simple_format_metadata_case(validation, project_root, bond, simple_dir, tmp_path, spd_element)

        return []

    assert run_with_float64(check) == []

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

        structure_mod = validation.load_dftorch_module(project_root, "Structure")
        assert (
            tuple(sk_mod.STRUCTURE_F_AO_ORDER)
            == tuple(structure_mod.AO_LABEL_TEMPLATE[9:16])
        ), "STRUCTURE_F_AO_ORDER must mirror Structure.AO_LABEL_TEMPLATE offsets 9..15"

        return []

    assert run_with_float64(check) == []


def test_f_angular_formula_source_lock():
    """HSK-03..HSK-06 / D-03, D-04, D-05: the approved source lock is recorded.

    Checkpoint 03-01-02 required a human to supply and confirm the f-electron
    Slater-Koster tables before any formula could be hard-coded.  This test
    pins the recorded provenance and the paper-to-Structure adapter so a later
    edit cannot quietly swap the source or reorder the f block.
    """

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")
        structure_mod = validation.load_dftorch_module(project_root, "Structure")

        source = sk_mod.F_FORMULA_SOURCE
        assert source["doi"] == "10.1088/0022-3719/13/4/016", source["doi"]
        assert source["title"] == "Slater-Koster tables for f electrons"
        assert "Takegahara" in source["authors"], source["authors"]
        assert "Aoki" in source["authors"] and "Yanase" in source["authors"]
        assert "13 (1980) 583-588" in source["journal"], source["journal"]
        assert "03-SOURCE-LOCK.md" in source["record"], source["record"]

        record = project_root / source["record"]
        assert record.is_file(), f"source-lock record missing at {record}"
        record_text = record.read_text(encoding="utf-8")
        assert source["doi"] in record_text
        assert "APPROVED" in record_text

        paper = tuple(sk_mod.PAPER_F_AO_ORDER)
        struct = tuple(sk_mod.STRUCTURE_F_AO_ORDER)
        perm = tuple(sk_mod.PAPER_TO_STRUCTURE_F_PERMUTATION)
        sign = tuple(sk_mod.PAPER_TO_STRUCTURE_F_SIGN)

        assert len(paper) == 7 and len(perm) == 7 and len(sign) == 7
        assert sorted(perm) == list(range(7)), f"adapter is not a permutation: {perm}"
        assert set(paper) == set(struct), "paper and Structure f bases differ"

        # The paper prints xyz (A_2u) first; Structure.py keeps it last.
        assert paper[0] == "fxyz"
        assert struct[-1] == "fxyz"
        assert perm == (1, 2, 3, 4, 5, 6, 0), perm

        # Applying the adapter to the paper order must reproduce Structure order.
        assert tuple(paper[perm[i]] for i in range(7)) == struct

        # Same Cartesian polynomials and normalisation => no sign flips.
        assert sign == (1.0,) * 7, sign

        # Structure.py's AO order is locked and must not have moved.
        assert tuple(structure_mod.AO_LABEL_TEMPLATE) == (
            "s",
            "px",
            "py",
            "pz",
            "dxy",
            "dyz",
            "dzx",
            "dx2_y2",
            "dz2",
            "fx3",
            "fy3",
            "fz3",
            "fx_y2_z2",
            "fy_z2_x2",
            "fz_x2_y2",
            "fxyz",
        )

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


def random_unit_directions(count: int, seed: int = 20260728):
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    return vectors[:, 0], vectors[:, 1], vectors[:, 2], vectors


def f_angular_channel_matrices(sk_mod, L, M, N):
    """Return (sf, pf, df, ff) as ``(P, n_channel, n_row, n_col)`` tensors."""
    return tuple(
        helper(L, M, N).permute(3, 0, 1, 2)
        for helper in (
            sk_mod.f_angular_sf,
            sk_mod.f_angular_pf,
            sk_mod.f_angular_df,
            sk_mod.f_angular_ff,
        )
    )


def test_f_angular_orthogonality_identity():
    """HSK-03..HSK-06: the paper's own correctness gate, equations (14)-(15).

    Takegahara p586: setting every two-centre integral of a shell pair to 1
    must reproduce the identity, "This orthogonal relation is useful in
    checking the results."

    Concretely this makes the coefficient matrix of each channel an orthogonal
    projector, which additionally ties the s-f, p-f and d-f tables to the f-f
    table: for the shell pair (j', 3) the channel matrix C_k obeys
    ``C_k^T C_k == P_k^(3)`` (the f-f channel projector) and the induced row
    projectors ``C_k C_k^T`` must sum to the identity on the j' shell.

    A mis-transcribed coefficient, a wrong sign, a wrong cyclic image or a
    misplaced AO breaks this. Fix the transcription, never this test.
    """

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")

        L, M, N, vectors = random_unit_directions(512)
        sf, pf, df, ff = f_angular_channel_matrices(sk_mod, L, M, N)

        tol = 1e-10
        eye7 = torch.eye(7, dtype=torch.float64)
        eye5 = torch.eye(5, dtype=torch.float64)
        eye3 = torch.eye(3, dtype=torch.float64)

        def close(actual, expected, label):
            error = float((actual - expected).abs().max())
            assert error < tol, f"{label}: max error {error:.3e}"

        # --- equations (14)-(15) for the f shell ------------------------
        close(ff.sum(dim=1), eye7, "sum_k C_k(ff) != I7")

        # --- each f-f channel matrix is an orthogonal projector ---------
        for k, rank in enumerate((1, 2, 2, 2)):  # sigma, pi, delta, phi
            block = ff[:, k]
            close(block, block.transpose(1, 2), f"ff C_{k} not symmetric")
            close(block @ block, block, f"ff C_{k} not idempotent")
            trace = torch.diagonal(block, dim1=1, dim2=2).sum(dim=1)
            close(trace, torch.full_like(trace, float(rank)), f"tr(ff C_{k})")
            for other in range(k + 1, 4):
                close(
                    block @ ff[:, other],
                    torch.zeros_like(block),
                    f"ff C_{k} C_{other} != 0",
                )

        # --- cross-shell blocks project onto the same f-shell subspaces --
        for label, matrices, n_channel in (
            ("sf", sf, 1),
            ("pf", pf, 2),
            ("df", df, 3),
        ):
            for k in range(n_channel):
                block = matrices[:, k]
                close(
                    block.transpose(1, 2) @ block,
                    ff[:, k],
                    f"{label} C_{k}^T C_{k} != ff C_{k}",
                )

        # --- and the induced s / p / d projectors are complete ----------
        close(
            (sf[:, 0] @ sf[:, 0].transpose(1, 2)).reshape(-1),
            torch.ones(sf.shape[0], dtype=torch.float64),
            "sf C_0 C_0^T != 1",
        )
        close(
            sum(pf[:, k] @ pf[:, k].transpose(1, 2) for k in range(2)),
            eye3,
            "sum_k P_k(p) != I3",
        )
        close(
            sum(df[:, k] @ df[:, k].transpose(1, 2) for k in range(3)),
            eye5,
            "sum_k P_k(d) != I5",
        )

        # --- independent oracles from the classic s/p/d Slater-Koster rows
        # p sigma column is just the direction cosine vector.
        close(
            pf[:, 0] @ pf[:, 0].transpose(1, 2),
            vectors.unsqueeze(2) * vectors.unsqueeze(1),
            "p sigma projector != outer(l,m,n)",
        )
        # d sigma column is the s-d row already implemented for 9-orbital atoms,
        # in Structure order (dxy, dyz, dzx, dx2_y2, dz2).
        root3 = 3.0**0.5
        sd = torch.stack(
            [
                root3 * L * M,
                root3 * M * N,
                root3 * N * L,
                0.5 * root3 * (L * L - M * M),
                N * N - 0.5 * (L * L + M * M),
            ],
            dim=1,
        )
        close(
            df[:, 0] @ df[:, 0].transpose(1, 2),
            sd.unsqueeze(2) * sd.unsqueeze(1),
            "d sigma projector != outer(s-d row)",
        )
        # f sigma column is the s-f row.
        sf_row = sf[:, 0, 0]
        close(
            ff[:, 0],
            sf_row.unsqueeze(2) * sf_row.unsqueeze(1),
            "f sigma projector != outer(s-f row)",
        )

        return []

    assert run_with_float64(check) == []


def test_f_angular_parity_under_direction_reversal():
    """HSK-08: reversing the bond flips odd-degree angular polynomials.

    E(-l, -m, -n) = (-1) ** (l_a + l_b) E(l, m, n) is what lets the SK routine
    reuse one table for both the I->J and the J->I halves of a pair, with the
    sign conventions the s-p and d-s blocks already use.
    """

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")

        L, M, N, _ = random_unit_directions(256, seed=99)
        cases = (
            ("sf", sk_mod.f_angular_sf, -1.0),  # (-1) ** (0 + 3)
            ("pf", sk_mod.f_angular_pf, 1.0),  # (-1) ** (1 + 3)
            ("df", sk_mod.f_angular_df, -1.0),  # (-1) ** (2 + 3)
            ("ff", sk_mod.f_angular_ff, 1.0),  # (-1) ** (3 + 3)
        )
        for label, helper, parity in cases:
            forward = helper(L, M, N)
            reverse = helper(-L, -M, -N)
            error = float((reverse - parity * forward).abs().max())
            assert error < 1e-11, f"{label}: parity {parity:+.0f} violated by {error:.3e}"

        return []

    assert run_with_float64(check) == []


def test_eu_containing_h0_s_is_finite_shaped_and_symmetric(tmp_path):
    """HSK-01 / HSK-07 / D-06: single-system f-containing H0/S actually builds.

    Before Phase 3 every 16-orbital neighbor pair fell through the 1/4/9 masks
    and was silently omitted; after the tracer it reached an explicit boundary;
    now the source-locked angular blocks make it produce real numbers.
    """

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        skf_dir = project_root / "tests" / "f_orbital_data"

        for label, elements, spacing in (
            ("Eu_N", ["Eu", "N"], 2.4),
            ("Eu_Ga", ["Eu", "Ga"], 2.8),
            ("Eu_Eu", ["Eu", "Eu"], 3.4),
            ("Eu_N_Ga", ["Eu", "N", "Ga"], 2.4),
        ):
            xyz_path = tmp_path / f"h0s_{label}.xyz"
            const, struct, H0, dH0, S, dS = build_h0_and_s(
                validation, project_root, skf_dir, elements, xyz_path, spacing=spacing
            )

            assert int(const.n_orb[struct.TYPE[0]]) == 16, (
                f"{label}: Eu must be a 16-orbital element"
            )
            assert H0.shape == (struct.HDIM, struct.HDIM), f"{label}: {tuple(H0.shape)}"
            assert S.shape == (struct.HDIM, struct.HDIM), f"{label}: {tuple(S.shape)}"
            assert torch.isfinite(H0).all(), f"{label}: H0 has non-finite entries"
            assert torch.isfinite(S).all(), f"{label}: S has non-finite entries"
            assert torch.allclose(H0, H0.transpose(0, 1)), f"{label}: H0 not symmetric"
            assert torch.allclose(S, S.transpose(0, 1)), f"{label}: S not symmetric"
            assert torch.isfinite(dH0).all(), f"{label}: dH0 has non-finite entries"
            assert torch.isfinite(dS).all(), f"{label}: dS has non-finite entries"

            # S keeps its AO identity contribution on the diagonal.
            assert torch.allclose(
                torch.diagonal(S), torch.ones_like(torch.diagonal(S))
            ), f"{label}: S diagonal lost its identity term"

            # The f rows/columns must not be all zero any more: that was exactly
            # the silent-drop failure mode this phase exists to remove.
            i0 = int(struct.H_INDEX_START[0])
            f_rows = H0[i0 + 9 : i0 + 16, :]
            off_diagonal = f_rows.clone()
            off_diagonal[:, i0 + 9 : i0 + 16] -= torch.diag(
                torch.diagonal(H0)[i0 + 9 : i0 + 16]
            )
            assert float(off_diagonal.abs().max()) > 0.0, (
                f"{label}: every off-diagonal f entry of H0 is zero, so the f "
                "blocks are still being dropped"
            )
            f_rows_S = S[i0 + 9 : i0 + 16, :].clone()
            f_rows_S[:, i0 + 9 : i0 + 16] -= torch.eye(7, dtype=S.dtype)
            assert float(f_rows_S.abs().max()) > 0.0, (
                f"{label}: every off-diagonal f entry of S is zero"
            )

        return []

    assert run_with_float64(check) == []


def test_f_containing_derivative_paths_are_guarded():
    """HSK-01 / D-09: f pairs must not silently contribute zero derivatives.

    Phase 3 delivers f angular *values* only. dH0/dS therefore carry exact
    zeros in every f block, which is a perfectly plausible-looking gradient,
    so every derivative consumer has to refuse f-containing systems.
    """

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")
        h0ands_mod = validation.load_dftorch_module(project_root, "_h0ands")
        stress_mod = validation.load_dftorch_module(project_root, "_stress")
        esdriver_mod = validation.load_dftorch_module(project_root, "ESDriver")

        assert h0ands_mod.FAngularFormulaSourceError is sk_mod.FAngularFormulaSourceError
        assert (
            stress_mod.FDerivativeUnsupportedError
            is sk_mod.FDerivativeUnsupportedError
        )
        assert issubclass(sk_mod.FDerivativeUnsupportedError, NotImplementedError)

        # Values exist; derivatives explicitly do not.
        assert sk_mod.F_ANGULAR_FORMULAS_AVAILABLE is True
        assert sk_mod.F_ANGULAR_DERIVATIVES_AVAILABLE is False, (
            "flip this only once f angular derivatives are implemented AND the "
            "guards below are removed"
        )

        import inspect

        # The batch H0/S route reconstructs only the 1/4/9 orbital masks, so it
        # must reject n_orb == 16 rather than dropping those pairs outright.
        source = inspect.getsource(h0ands_mod.H0_and_S_vectorized_batch)
        assert "FAngularFormulaSourceError" in source
        assert "16" in source

        # Analytical stress consumes dH0/dS.
        source = inspect.getsource(stress_mod._pair_grad_from_sk)
        assert "FDerivativeUnsupportedError" in source
        assert "16" in source

        # Both force entry points guard before assembling any force term.
        guard = inspect.getsource(esdriver_mod._require_f_derivatives)
        assert "FDerivativeUnsupportedError" in guard
        assert "16" in guard
        for cls in (esdriver_mod.ESDriver, esdriver_mod.ESDriverBatch):
            source = inspect.getsource(cls.calc_forces)
            assert "_require_f_derivatives" in source, (
                f"{cls.__name__}.calc_forces has no f derivative guard"
            )

        return []

    assert run_with_float64(check) == []


def test_f_derivative_guard_rejects_f_systems():
    """D-09: the shared force guard fires for f elements and not otherwise."""

    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        sk_mod = validation.load_dftorch_module(project_root, "_slater_koster_pair")
        esdriver_mod = validation.load_dftorch_module(project_root, "ESDriver")

        class _Struct:
            def __init__(self, types):
                self.TYPE = torch.tensor(types, dtype=torch.long)

        class _Const:
            n_orb = torch.tensor([1, 4, 9, 16], dtype=torch.long)

        const = _Const()

        # No f atom -> guard is a no-op.
        esdriver_mod._require_f_derivatives(_Struct([0, 1, 2]), const, "unit-test")

        # An f atom anywhere -> named refusal.
        try:
            esdriver_mod._require_f_derivatives(_Struct([2, 3]), const, "unit-test")
        except sk_mod.FDerivativeUnsupportedError as exc:
            assert "n_orb == 16" in str(exc), str(exc)
        else:
            raise AssertionError("f derivative guard did not fire for an f system")

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

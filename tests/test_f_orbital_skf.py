import shutil
import sys
from pathlib import Path

import torch

# The checks these tests drive used to live in `src/dftorch/script.py`, which
# this module loaded by filesystem path via `importlib.util.spec_from_file_location`.
# Phase 5 plan 03 (decision D-03) moved them into `tests/` and deleted that file,
# so they are now imported like any other test helper.  `skf_validation_support`
# imports torch and the standard library only; nothing under `dftorch` is
# imported until a test asks for it, which is what keeps the float64 harness
# below meaningful.
import skf_validation_support as validation


def _purge_dftorch_modules() -> None:
    for name in [
        name
        for name in sys.modules
        if name == "dftorch" or name.startswith("dftorch.")
    ]:
        sys.modules.pop(name, None)


def run_with_float64(fn):
    """Run ``fn`` with float64 defaults and a freshly imported dftorch.

    The purge on ENTRY matters and is not decoration.  Module-level state inside
    `dftorch` is built at import time under whatever `torch.get_default_dtype()`
    happens to be, and the loader this module used before plan 05-03 re-executed
    every module file on every call, so it always got a float64 import for free.
    A plain `importlib.import_module` returns whatever is already cached, so the
    harness now has to guarantee the same thing itself: purge first, set float64,
    then let the test import.  Without the entry purge these tests would silently
    depend on which test module ran before them.
    """
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    _purge_dftorch_modules()
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        _purge_dftorch_modules()
        sys.modules.update(previous_modules)


def write_xyz(
    path: Path, elements: list[str], spacing: float = 1.5, axis: str = "x"
) -> None:
    """Write atoms evenly spaced along ``axis``.

    With the default ``axis="x"`` every ordered pair (I, J) with J after I has
    direction cosines (L, M, N) = (1, 0, 0), which is what the metadata and
    channel regressions rely on. ``axis="z"`` gives (0, 0, 1) and is what the
    hand-calculated f block tests use, because the Takegahara formulas collapse
    to clean closed forms there.
    """
    offsets = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    lines = [str(len(elements)), "metadata regression"]
    for idx, sym in enumerate(elements):
        step = spacing * idx
        lines.append(
            f"{sym} {step * offsets[0]:.8f} {step * offsets[1]:.8f} "
            f"{step * offsets[2]:.8f}"
        )
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
    structure_mod = validation.import_dftorch_module("Structure")
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


def build_h0_and_s(validation, project_root: Path, skf_dir: Path, elements: list[str], xyz_path: Path, *, rcut: float = 10.0, spacing: float = 1.5, axis: str = "x"):
    """Directly assemble single-system H0/S for ``elements`` without running SCF.

    Mirrors the ``ESDriver.forward`` call sequence (neighbor list, then
    ``H0_and_S_vectorized``) but skips overlap inversion, repulsion, Coulomb,
    SCF, and forces so the H0/S contract can be tested in isolation.
    """
    const = validation.build_test_constants(skf_dir, elements, xyz_path)
    # Constants only needs the species list; rewrite the geometry so callers can
    # pick a separation that lies inside the fixture's radial grid.
    write_xyz(xyz_path, elements, spacing=spacing, axis=axis)
    struct = build_structure(validation, project_root, skf_dir, xyz_path, const)

    nnl_mod = validation.import_dftorch_module("_nearestneighborlist")
    h0ands_mod = validation.import_dftorch_module("_h0ands")

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
    for sym in validation.collect_elements_independently(skf_dir):
        metadata = validation.parse_expected_homonuclear_metadata(
            validation.resolve_homonuclear_skf_independently(skf_dir, sym),
        )
        if metadata is not None and list(metadata["SHELL_PRESENT"]) == shell_present:
            return sym
    raise AssertionError(f"No element with shell_present={shell_present} found in {skf_dir}")


def assert_simple_format_metadata_case(validation, project_root, bond, skf_dir: Path, tmp_path: Path, element: str) -> None:
    metadata = validation.expected_metadata_by_element(skf_dir, [element])
    xyz_path = tmp_path / f"{element}_metadata.xyz"
    write_xyz(xyz_path, [element])
    const = validation.build_test_constants(skf_dir, [element], xyz_path)

    structure_mod = validation.import_dftorch_module("Structure")
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

    validation.check_constants_against_expected(const, skf_dir, [element])
    validation.check_single_structure_layout(struct, [element], metadata)


def test_f_orbital_skf_parser_and_spline_gate():
    def check():
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
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
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
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
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_constants_tests(project_root, skf_dir)

    assert run_with_float64(check) == []


def test_f_orbital_structure_metadata_gate():
    def check():
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_structure_tests(project_root, skf_dir)

    assert run_with_float64(check) == []


def expected_ss_channel_value(validation, project_root, const, struct, channel_name: str):
    """Independently evaluate one radial channel for the single I-J atom pair.

    Returns the mean of the I->J and J->I spline values, which is what the
    symmetrized H0/S s-s entry must equal for a two-atom system.
    """
    sk_mod = validation.import_dftorch_module("_slater_koster_pair")
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
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")

        assert list(sk_mod.SK_CHANNEL_NAMES) == list(bond._CHANNELS), (
            "SK_CHANNEL_NAMES drifted from _bond_integral._CHANNELS"
        )
        for index, name in enumerate(bond._CHANNELS):
            assert sk_mod.sk_channel_index(name) == index, name

        # SH_shift is the name prefix itself, never a numeric block offset.
        assert sk_mod.sk_channel_name("ss0", "H") == "Hss0"
        assert sk_mod.sk_channel_name("ss0", "S") == "Sss0"

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
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, "H"))
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, "S"))

        # f channels exist in the table even though the angular formulas do not.
        for base in ("sf0", "pf0", "pf1", "df0", "df1", "df2", "ff0", "ff1", "ff2", "ff3"):
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, "H"))
            sk_mod.sk_channel_index(sk_mod.sk_channel_name(base, "S"))

        try:
            sk_mod.sk_channel_index("Hzz9")
        except KeyError:
            pass
        else:
            raise AssertionError("Unknown channel names must raise KeyError")

        structure_mod = validation.import_dftorch_module("Structure")
        assert (
            tuple(sk_mod.STRUCTURE_F_AO_ORDER)
            == tuple(structure_mod.AO_LABEL_TEMPLATE[9:16])
        ), "STRUCTURE_F_AO_ORDER must mirror Structure.AO_LABEL_TEMPLATE offsets 9..15"

        return []

    assert run_with_float64(check) == []


def test_f_ao_order_is_locked():
    """HSK-03..HSK-06 / D-03, D-04, D-05: the f AO order must not move.

    The angular tables are written directly in ``STRUCTURE_F_AO_ORDER``, so the
    index of each f orbital is load-bearing in every formula.  This test pins
    that order against ``Structure.AO_LABEL_TEMPLATE`` so a later edit cannot
    quietly reorder the f block out from under the tables.
    """

    def check():
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")
        structure_mod = validation.import_dftorch_module("Structure")

        struct = tuple(sk_mod.STRUCTURE_F_AO_ORDER)
        assert struct == (
            "fx3",
            "fy3",
            "fz3",
            "fx_y2_z2",
            "fy_z2_x2",
            "fz_x2_y2",
            "fxyz",
        ), struct

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
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")
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


def f_angular_gradient_blocks(sk_mod, L, M, N):
    """Return the four derivative blocks in their native ``(3, ...)`` layout."""
    return tuple(
        helper(L, M, N)
        for helper in (
            sk_mod.f_angular_sf_grad,
            sk_mod.f_angular_pf_grad,
            sk_mod.f_angular_df_grad,
            sk_mod.f_angular_ff_grad,
        )
    )


def test_f_angular_derivatives_match_autograd_and_finite_differences():
    """The analytic f angular derivatives agree with two independent references.

    The derivative tables are differentiated by hand from the value tables, so
    they can drift from them silently: nothing about a wrong polynomial looks
    wrong. Two references pin them.

    Automatic differentiation is the tight one. It runs the *value* functions
    and differentiates the actual operations performed, so agreement to machine
    precision means the analytic form matches the value table exactly rather
    than approximately.

    Central finite differences are the loose but assumption-free one: they never
    touch the derivative code path at all, so they would catch a shared mistake
    in which the analytic form and autograd agree with each other but not with
    the function's real slope. Their accuracy is limited to about ``h**2`` plus
    rounding, hence the far weaker tolerance.
    """

    def check():
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")
        L, M, N, vectors = random_unit_directions(256)

        value_helpers = (
            sk_mod.f_angular_sf,
            sk_mod.f_angular_pf,
            sk_mod.f_angular_df,
            sk_mod.f_angular_ff,
        )
        grads = f_angular_gradient_blocks(sk_mod, L, M, N)
        labels = ("sf", "pf", "df", "ff")

        for label, value_fn, analytic in zip(labels, value_helpers, grads):
            # Shape: the value block with a leading axis of 3 for d/dL, d/dM, d/dN.
            expected_shape = (3,) + tuple(value_fn(L, M, N).shape)
            assert tuple(analytic.shape) == expected_shape, (
                f"{label}: derivative block has shape {tuple(analytic.shape)}, "
                f"expected {expected_shape}"
            )

            # --- reference 1: automatic differentiation ------------------
            gl = L.clone().requires_grad_(True)
            gm = M.clone().requires_grad_(True)
            gn = N.clone().requires_grad_(True)
            value = value_fn(gl, gm, gn)
            # One random cotangent contracts the whole block in a single
            # backward pass, which tests every cell at once rather than
            # sampling a few.
            cotangent = torch.randn(
                value.shape,
                generator=torch.Generator().manual_seed(11),
                dtype=torch.float64,
            )
            ref = torch.stack(
                torch.autograd.grad((value * cotangent).sum(), (gl, gm, gn))
            )
            got = (analytic * cotangent.unsqueeze(0)).sum(
                dim=tuple(range(1, analytic.dim() - 1))
            )
            error = float((ref - got).abs().max())
            assert error < 1e-11, f"{label}: analytic vs autograd max error {error:.3e}"

            # --- reference 2: central finite differences -----------------
            step = 1e-6
            for axis in range(3):
                shifted = vectors.clone()
                shifted[:, axis] += step
                up = value_fn(shifted[:, 0], shifted[:, 1], shifted[:, 2])
                shifted = vectors.clone()
                shifted[:, axis] -= step
                down = value_fn(shifted[:, 0], shifted[:, 1], shifted[:, 2])
                numeric = (up - down) / (2 * step)
                error = float((analytic[axis] - numeric).abs().max())
                assert error < 1e-6, (
                    f"{label}: analytic vs finite difference on axis {axis} "
                    f"max error {error:.3e}"
                )

        return []

    assert run_with_float64(check) == []


def test_f_angular_derivatives_satisfy_the_differentiated_orthogonality_gate():
    """Differentiating the orthogonality gate constrains the derivative tables.

    ``test_f_angular_orthogonality_identity`` pins three facts about the values:
    ``sum_k C_k(ff)`` is the 7x7 identity, each f-f channel matrix is an
    idempotent projector, and for a cross-shell pair the channel matrices obey
    ``sum_k C_k C_k^T == I`` and ``C_k^T C_k == P_k`` with ``P_k`` the f-f
    channel projector.

    Each of those is constant as the bond turns, so differentiating along the
    unit sphere gives an identity the derivative tables must satisfy. This is
    an independent check in the strongest sense: it never differentiates the
    value functions, so it shares no machinery with autograd or with finite
    differences. It also *couples the blocks* -- differentiating
    ``C_k^T C_k == P_k`` ties the s-f, p-f and d-f derivative tables to the f-f
    derivative table, so a mistake confined to one of them still shows up.

    A tangential displacement is used throughout: L, M and N are not free
    variables but obey ``L^2 + M^2 + N^2 == 1``, so only motion that preserves
    the unit length is meaningful, and the three partials must be contracted
    together rather than read one at a time.
    """

    def check():
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")
        L, M, N, vectors = random_unit_directions(256, seed=20260812)
        radial = vectors.transpose(0, 1)  # (3, P)

        # Any direction orthogonal to the radial one keeps the vector on the
        # unit sphere to first order.
        tangent = torch.randn(
            radial.shape,
            generator=torch.Generator().manual_seed(5),
            dtype=torch.float64,
        )
        tangent = tangent - (tangent * radial).sum(0, keepdim=True) * radial

        sf, pf, df, ff = (
            sk_mod.f_angular_sf(L, M, N),
            sk_mod.f_angular_pf(L, M, N),
            sk_mod.f_angular_df(L, M, N),
            sk_mod.f_angular_ff(L, M, N),
        )
        g_sf, g_pf, g_df, g_ff = f_angular_gradient_blocks(sk_mod, L, M, N)

        def directional(grad_block):
            """Contract the three partials against the tangential step."""
            return torch.einsum("dk...p,dp->k...p", grad_block, tangent)

        d_ff = directional(g_ff)
        tol = 1e-10

        # --- sum_k C_k(ff) is the identity, so its derivative vanishes ---
        error = float(d_ff.sum(dim=0).abs().max())
        assert error < tol, f"d/dt[sum_k C_k(ff)] != 0: {error:.3e}"

        # --- each channel projector obeys P P == P, so d(P P) == dP ------
        for k in range(4):
            projector, d_projector = ff[k], d_ff[k]
            lhs = torch.einsum("abp,bcp->acp", d_projector, projector) + torch.einsum(
                "abp,bcp->acp", projector, d_projector
            )
            error = float((lhs - d_projector).abs().max())
            assert error < tol, f"d(P{k} P{k}) != dP{k}: {error:.3e}"

        # --- the cross-shell relations, differentiated -------------------
        for label, values, grad_block, n_channel in (
            ("sf", sf, g_sf, 1),
            ("pf", pf, g_pf, 2),
            ("df", df, g_df, 3),
        ):
            d_values = directional(grad_block)

            # sum_k C_k C_k^T is the identity on the lower shell.
            lhs = torch.einsum("kabp,kcbp->acp", d_values, values) + torch.einsum(
                "kabp,kcbp->acp", values, d_values
            )
            error = float(lhs.abs().max())
            assert error < tol, (
                f"{label}: d/dt[sum_k C_k C_k^T] != 0: {error:.3e}"
            )

            # C_k^T C_k is the f-f channel projector -- this is the relation
            # that couples this block's derivatives to the f-f block's.
            lhs = torch.einsum("kabp,kacp->kbcp", d_values, values) + torch.einsum(
                "kabp,kacp->kbcp", values, d_values
            )
            error = float((lhs - d_ff[:n_channel]).abs().max())
            assert error < tol, (
                f"{label}: d[C_k^T C_k] != dP_k, which means the {label} and ff "
                f"derivative tables disagree: {error:.3e}"
            )

        return []

    assert run_with_float64(check) == []


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
        project_root = validation.PROJECT_ROOT
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")

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


# ---------------------------------------------------------------------------
# Hand-calculated angular constants (D-07 / D-08 / HSK-08)
# ---------------------------------------------------------------------------
# Evaluated by hand from the printed Takegahara Table 2 entries plus the p586
# cyclic rule E_{sA,sB}(l, m, n) = E_{A,B}(m, n, l), then reordered from paper
# order into STRUCTURE_F_AO_ORDER:
#
#     0 fx3        1 fy3        2 fz3
#     3 fx_y2_z2   4 fy_z2_x2   5 fz_x2_y2   6 fxyz
#
# Along a coordinate axis two of the three direction cosines vanish and almost
# every term drops out, so these matrices can be re-derived on paper in a few
# lines. Each block is additionally consistent with the projector identities
# checked in test_f_angular_orthogonality_identity (e.g. the p-f pi rows below
# have norm 3/8 + 5/8 = 1 and the s/p/d row projectors sum to the identity).

_A = 3.0 / 8.0
_B = 5.0 / 8.0
_R38 = (3.0 / 8.0) ** 0.5
_R58 = (5.0 / 8.0) ** 0.5
_R3_2 = (3.0**0.5) / 2.0
_R15_8 = (15.0**0.5) / 8.0


def _axis_expectations():
    """Hand-derived angular blocks keyed by axis.

    Each entry maps a shell pair to a list of ``(channel, row, col, value)``;
    every unlisted position must be exactly zero.
    """
    return {
        # --------------------------------------------------------- +z axis
        # (l, m, n) = (0, 0, 1)
        "z": {
            # E_s,z(5z^2-3r^2) = 1/2 * n(5n^2 - 3) = 1; all other columns carry
            # a factor l, m, or (m^2 - n^2)*l and vanish.
            "sf": [(0, 0, 2, 1.0)],
            # px: E_x,x(5x^2-3r^2) = -sqrt(3/8)(5l^2-1)(l^2-1) = -sqrt(3/8) (pi)
            #     E_x,x(y^2-z^2)   = -sqrt(5/8)(3l^2-1)(m^2-n^2) = -sqrt(5/8)
            # py: cyclic image; pz: E_z,z(5z^2-3r^2) = 1/2 n^2(5n^2-3) = sigma.
            "pf": [
                (1, 0, 0, -_R38),
                (1, 0, 3, -_R58),
                (1, 1, 1, -_R38),
                (1, 1, 4, +_R58),
                (0, 2, 2, 1.0),
            ],
            # dxy couples only to xyz: E_xy,xyz = n(3l^2m^2 + 2n^2 - 1) = delta.
            # dyz / dzx are its cyclic images; the E_g rows keep one delta each:
            # E_x2-y2,z(x^2-y^2) = 1/4 n[3(l^2-m^2)^2 + 8n^2 - 4] = 1,
            # E_3z2-r2,z(5z^2-3r^2) = 1/4 n(3n^2-1)(5n^2-3) = sigma.
            "df": [
                (2, 0, 6, 1.0),
                (1, 1, 1, -_R38),
                (1, 1, 4, +_R58),
                (1, 2, 0, -_R38),
                (1, 2, 3, -_R58),
                (2, 3, 5, 1.0),
                (0, 4, 2, 1.0),
            ],
            # f-f. E_z(5z^2-3r^2),z(5z^2-3r^2) = 1/4 n^2(5n^2-3)^2 = sigma;
            # E_xyz,xyz and E_z(x^2-y^2),z(x^2-y^2) collapse to delta; the
            # T_1u/T_2u pairs perpendicular to z mix pi and phi in a rank-1
            # 2x2 block with trace 1.
            "ff": [
                (1, 0, 0, _A),
                (3, 0, 0, _B),
                (1, 0, 3, +_R15_8),
                (3, 0, 3, -_R15_8),
                (1, 3, 0, +_R15_8),
                (3, 3, 0, -_R15_8),
                (1, 3, 3, _B),
                (3, 3, 3, _A),
                (1, 1, 1, _A),
                (3, 1, 1, _B),
                (1, 1, 4, -_R15_8),
                (3, 1, 4, +_R15_8),
                (1, 4, 1, -_R15_8),
                (3, 4, 1, +_R15_8),
                (1, 4, 4, _B),
                (3, 4, 4, _A),
                (0, 2, 2, 1.0),
                (2, 5, 5, 1.0),
                (2, 6, 6, 1.0),
            ],
        },
        # --------------------------------------------------------- +x axis
        # (l, m, n) = (1, 0, 0)
        "x": {
            "sf": [(0, 0, 0, 1.0)],
            "pf": [
                (0, 0, 0, 1.0),
                (1, 1, 1, -_R38),
                (1, 1, 4, -_R58),
                (1, 2, 2, -_R38),
                (1, 2, 5, +_R58),
            ],
            # E_xy,y(5y^2-3r^2) = -sqrt(3/8) l(5m^2-1)(2m^2-1) = -sqrt(3/8);
            # E_xy,y(z^2-x^2)   = -sqrt(5/8) l[(6m^2-1)(n^2-l^2) + 2m^2]
            #                   = -sqrt(5/8);
            # E_x2-y2,x(5x^2-3r^2) = 1/4 sqrt(3) l(l^2-m^2)(5l^2-3) = sqrt(3)/2;
            # E_x2-y2,x(y^2-z^2)   = 1/4 l[3(l^2-m^2)(m^2-n^2) - 4l^2 + 2] = -1/2;
            # E_3z2-r2,x(5x^2-3r^2) = 1/4 l(3n^2-1)(5l^2-3) = -1/2;
            # E_3z2-r2,x(y^2-z^2)   = 1/4 sqrt(3) l[(3n^2-1)(m^2-n^2) - 4l^2 + 2]
            #                       = -sqrt(3)/2.
            "df": [
                (1, 0, 1, -_R38),
                (1, 0, 4, -_R58),
                (2, 1, 6, 1.0),
                (1, 2, 2, -_R38),
                (1, 2, 5, +_R58),
                (0, 3, 0, +_R3_2),
                (2, 3, 3, -0.5),
                (0, 4, 0, -0.5),
                (2, 4, 3, -_R3_2),
            ],
            # Selected f-f entries only; the full block is pinned at +z.
            "ff_selected": [
                # E_x(5x^2-3r^2),x(5x^2-3r^2) = 1/4 l^2(5l^2-3)^2 = sigma
                ((0, 0), {0: 1.0}),
                # E_z(5z^2-3r^2),z(5z^2-3r^2) = 3/8 (pi) + 5/8 (phi)
                ((2, 2), {1: _A, 3: _B}),
                # E_z(x^2-y^2),z(x^2-y^2) = 5/8 (pi) + 3/8 (phi)
                ((5, 5), {1: _B, 3: _A}),
                # E_z(x^2-y^2),z(5z^2-3r^2)
                #   = -sqrt(15)/8 (pi) + sqrt(15)/8 (phi)
                ((5, 2), {1: -_R15_8, 3: +_R15_8}),
                # E_xyz,xyz = [1 - 4(l^2m^2 + m^2n^2 + n^2l^2) + 9l^2m^2n^2] = delta
                ((6, 6), {2: 1.0}),
            ],
        },
        # --------------------------------------------------------- +y axis
        # (l, m, n) = (0, 1, 0)
        "y": {
            "sf": [(0, 0, 1, 1.0)],
            "pf": [
                (1, 0, 0, -_R38),
                (1, 0, 3, +_R58),
                (0, 1, 1, 1.0),
                (1, 2, 2, -_R38),
                (1, 2, 5, -_R58),
            ],
        },
    }


def assert_block_equals(block, entries, label, n_channel, n_row, n_col):
    """``block`` is ``(channel, row, col, P)``; ``entries`` lists the non-zeros."""
    expected = torch.zeros(n_channel, n_row, n_col, dtype=torch.float64)
    for channel, row, col, value in entries:
        expected[channel, row, col] = value
    for pair in range(block.shape[-1]):
        error = float((block[..., pair] - expected).abs().max())
        assert error < 1e-12, (
            f"{label}: hand-calculated block mismatch, max error {error:.3e}\n"
            f"got:\n{block[..., pair]}\nexpected:\n{expected}"
        )


def test_f_angular_axis_blocks_match_hand_calculation():
    """D-07 / D-08 / HSK-08: x, y and z axis blocks against hand calculation.

    Independent of the implementation: every constant here was evaluated by
    hand from the printed table entries (see the comments above), so agreement
    means two separate derivations of the same physics coincide.
    """

    def check():
        project_root = validation.PROJECT_ROOT
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")

        axes = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        shapes = {"sf": (1, 1, 7), "pf": (2, 3, 7), "df": (3, 5, 7), "ff": (4, 7, 7)}
        helpers = {
            "sf": sk_mod.f_angular_sf,
            "pf": sk_mod.f_angular_pf,
            "df": sk_mod.f_angular_df,
            "ff": sk_mod.f_angular_ff,
        }
        expectations = _axis_expectations()

        for axis, (lx, my, nz) in axes.items():
            L = torch.tensor([lx], dtype=torch.float64)
            M = torch.tensor([my], dtype=torch.float64)
            N = torch.tensor([nz], dtype=torch.float64)
            for key, entries in expectations[axis].items():
                if key == "ff_selected":
                    block = helpers["ff"](L, M, N)[..., 0]
                    for (row, col), channels in entries:
                        for channel in range(4):
                            expected = channels.get(channel, 0.0)
                            actual = float(block[channel, row, col])
                            assert abs(actual - expected) < 1e-12, (
                                f"{axis}-axis ff[{channel}, {row}, {col}]: "
                                f"{actual} != {expected}"
                            )
                    continue
                assert_block_equals(
                    helpers[key](L, M, N),
                    entries,
                    f"{axis}-axis {key}",
                    *shapes[key],
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
        project_root = validation.PROJECT_ROOT
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")

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
        project_root = validation.PROJECT_ROOT
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


AO_SHELL_L = tuple([0] + [1] * 3 + [2] * 5 + [3] * 7)


def directed_channel_value(validation, project_root, const, struct, i, j, channel_name):
    """Evaluate one radial channel for the ordered pair (TYPE[i] -> TYPE[j]).

    Uses exactly the interval/offset lookup that ``H0_and_S_vectorized`` uses
    (``const.R_orb`` plus ``const.coeffs_tensor``), so the test isolates the
    angular formulas and the AO placement rather than the radial grid.
    """
    sk_mod = validation.import_dftorch_module("_slater_koster_pair")
    channel = sk_mod.sk_channel_index(channel_name)

    dR = torch.sqrt(
        (struct.RX[j] - struct.RX[i]) ** 2
        + (struct.RY[j] - struct.RY[i]) ** 2
        + (struct.RZ[j] - struct.RZ[i]) ** 2
    )
    idx = torch.clamp(
        torch.searchsorted(const.R_orb, dR.reshape(1), right=True) - 1,
        0,
        len(const.R_orb),
    )
    dx = dR - const.R_orb[idx][0]
    pair_type = int(const.pair_lookup[int(struct.TYPE[i]), int(struct.TYPE[j])])
    cs = const.coeffs_tensor[pair_type, idx[0], channel]
    return cs[0] + cs[1] * dx + cs[2] * dx**2 + cs[3] * dx**3


def test_f_block_entries_match_hand_calculated_values(tmp_path):
    """HSK-03..HSK-06 / HSK-08 / D-08: assembled H0/S f entries, entry by entry.

    A two-atom system aligned with +z makes the angular factors the closed-form
    constants pinned by test_f_angular_axis_blocks_match_hand_calculation, so
    every f entry of H0 and S must equal ``sum_k angular_k * radial_k`` with the
    radial value read from the named 40-channel table. This is what catches a
    block written at the wrong AO offset, through the wrong channel name, from
    the wrong pair direction, or with the wrong reverse-direction sign.
    """

    def check():
        project_root = validation.PROJECT_ROOT
        skf_dir = project_root / "tests" / "f_orbital_data"

        expectations = _axis_expectations()["z"]

        def dense(entries, n_channel, n_row, n_col):
            out = torch.zeros(n_channel, n_row, n_col, dtype=torch.float64)
            for channel, row, col, value in entries:
                out[channel, row, col] = value
            return out

        angular = {
            "sf": dense(expectations["sf"], 1, 1, 7),
            "pf": dense(expectations["pf"], 2, 3, 7),
            "df": dense(expectations["df"], 3, 5, 7),
            "ff": dense(expectations["ff"], 4, 7, 7),
        }

        # (label, elements, spacing, blocks to check)
        # ``blocks`` entries are
        #   (angular key, channel bases, low-shell AO offset, parity, radial dir)
        # where parity is (-1) ** (l_low + 3) and the radial direction says which
        # ordered pair supplies the two-centre integrals: the f atom is atom 0,
        # so a block with f on the *row* reads the (partner -> f) file.
        cases = [
            (
                "Eu_N",
                ["Eu", "N"],
                2.4,
                [
                    ("sf", ("sf0",), 0, -1.0),
                    ("pf", ("pf0", "pf1"), 1, +1.0),
                ],
            ),
            (
                "Eu_Eu",
                ["Eu", "Eu"],
                3.4,
                [
                    ("sf", ("sf0",), 0, -1.0),
                    ("pf", ("pf0", "pf1"), 1, +1.0),
                    ("df", ("df0", "df1", "df2"), 4, -1.0),
                    ("ff", ("ff0", "ff1", "ff2", "ff3"), 9, +1.0),
                ],
            ),
        ]

        for label, elements, spacing, blocks in cases:
            xyz_path = tmp_path / f"handcalc_{label}.xyz"
            const, struct, H0, dH0, S, dS = build_h0_and_s(
                validation,
                project_root,
                skf_dir,
                elements,
                xyz_path,
                spacing=spacing,
                axis="z",
            )
            i0 = int(struct.H_INDEX_START[0])  # Eu (16 orbitals)
            j0 = int(struct.H_INDEX_START[1])  # partner

            for matrix, prefix, scale in (
                (H0, "H", 1.0),
                (S, "S", 1.0 / 27.21138625),
            ):
                for key, bases, low_offset, parity in blocks:
                    coefficients = angular[key]
                    radial = [
                        directed_channel_value(
                            validation,
                            project_root,
                            const,
                            struct,
                            1,
                            0,
                            f"{prefix}{base}",
                        )
                        for base in bases
                    ]
                    n_low = coefficients.shape[1]
                    largest = 0.0
                    for a in range(n_low):
                        for b in range(7):
                            expected = parity * scale * sum(
                                float(coefficients[k, a, b]) * float(radial[k])
                                for k in range(len(bases))
                            )
                            actual = float(matrix[i0 + 9 + b, j0 + low_offset + a])
                            assert abs(actual - expected) < 1e-9, (
                                f"{label} {prefix} {key}[row f{b}, col {low_offset}+{a}]"
                                f": {actual} != {expected}"
                            )
                            largest = max(largest, abs(expected))
                    # An all-zero block would satisfy the comparison vacuously.
                    assert largest > 1e-6, (
                        f"{label} {prefix} {key}: every predicted entry is zero, "
                        "so this block proves nothing"
                    )

        return []

    assert run_with_float64(check) == []


def test_f_containing_pair_atom_order_reversal(tmp_path):
    """HSK-08 / D-07: swapping the two atoms flips odd-parity blocks only.

    Building Eu-N and N-Eu with both geometries along +z puts the Eu -> N bond
    along +z in one case and along -z in the other, so every AO block must pick
    up (-1) ** (l_a + l_b). This is the convention the reverse-direction writes
    in Slater_Koster_Pair_SKF_vectorized rely on.
    """

    def check():
        project_root = validation.PROJECT_ROOT
        skf_dir = project_root / "tests" / "f_orbital_data"
        spacing = 2.4

        _, struct_a, H0_a, _, S_a, _ = build_h0_and_s(
            validation,
            project_root,
            skf_dir,
            ["Eu", "N"],
            tmp_path / "order_eu_n.xyz",
            spacing=spacing,
            axis="z",
        )
        _, struct_b, H0_b, _, S_b, _ = build_h0_and_s(
            validation,
            project_root,
            skf_dir,
            ["N", "Eu"],
            tmp_path / "order_n_eu.xyz",
            spacing=spacing,
            axis="z",
        )

        # Eu is atom 0 in case A and atom 1 in case B.
        eu_a = int(struct_a.H_INDEX_START[0])
        n_a = int(struct_a.H_INDEX_START[1])
        n_b = int(struct_b.H_INDEX_START[0])
        eu_b = int(struct_b.H_INDEX_START[1])

        assert struct_a.HDIM == struct_b.HDIM == 20

        checked = 0
        nontrivial = 0
        for matrix_a, matrix_b, label in ((H0_a, H0_b, "H0"), (S_a, S_b, "S")):
            for eu_ao in range(16):
                for n_ao in range(4):
                    parity = (-1.0) ** (AO_SHELL_L[eu_ao] + AO_SHELL_L[n_ao])
                    value_a = float(matrix_a[eu_a + eu_ao, n_a + n_ao])
                    value_b = float(matrix_b[eu_b + eu_ao, n_b + n_ao])
                    assert abs(value_b - parity * value_a) < 1e-9, (
                        f"{label}[Eu {eu_ao}, N {n_ao}]: reversing the bond gave "
                        f"{value_b} but parity {parity:+.0f} predicts "
                        f"{parity * value_a}"
                    )
                    checked += 1
                    if abs(value_a) > 1e-6:
                        nontrivial += 1

        assert checked == 2 * 16 * 4
        assert nontrivial > 0, "the Eu-N coupling block is entirely zero"

        # Diagonal on-site terms are order-independent.
        assert torch.allclose(
            torch.diagonal(H0_a)[eu_a : eu_a + 16],
            torch.diagonal(H0_b)[eu_b : eu_b + 16],
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
        project_root = validation.PROJECT_ROOT
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")
        h0ands_mod = validation.import_dftorch_module("_h0ands")
        stress_mod = validation.import_dftorch_module("_stress")
        esdriver_mod = validation.import_dftorch_module("ESDriver")

        assert (
            stress_mod.FDerivativeUnsupportedError
            is sk_mod.FDerivativeUnsupportedError
        )
        assert issubclass(sk_mod.FDerivativeUnsupportedError, NotImplementedError)

        # Values exist; derivatives explicitly do not.
        assert sk_mod.F_ANGULAR_DERIVATIVES_AVAILABLE is False, (
            "flip this only once f angular derivatives are implemented AND the "
            "guards below are removed"
        )

        import inspect

        # The batch H0/S route reconstructs only the 1/4/9 orbital masks, so it
        # must reject n_orb == 16 rather than dropping those pairs outright.
        source = inspect.getsource(h0ands_mod.H0_and_S_vectorized_batch)
        assert "NotImplementedError" in source
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
        project_root = validation.PROJECT_ROOT
        sk_mod = validation.import_dftorch_module("_slater_koster_pair")
        esdriver_mod = validation.import_dftorch_module("ESDriver")

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
        project_root = validation.PROJECT_ROOT
        bond = validation.import_dftorch_module("_bond_integral")

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

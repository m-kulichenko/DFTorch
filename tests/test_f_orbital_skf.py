import importlib.util
import shutil
from pathlib import Path

import torch


def run_with_float64(fn):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)


def load_validation_script():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "dftorch" / "script.py"
    spec = importlib.util.spec_from_file_location("dftorch_phase1_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation script from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_xyz(path: Path, elements: list[str]) -> None:
    lines = [str(len(elements)), "metadata regression"]
    for idx, sym in enumerate(elements):
        lines.append(f"{sym} {1.5 * idx:.8f} 0.00000000 0.00000000")
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

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

"""Validation script for the f-orbital loader/constants/structure changes.

Run from the project root with:

    python src/dftorch/script.py tests/f_orbital_data

or, if your f-orbital SKF files live in tests/f_orbital_data:

    python src/dftorch/script.py

What this script checks, in order:

1. _bond_integral.py
   - Reads every .skf file in the given directory.
   - Normalizes the electronic table to 40 Slater-Koster channels.
   - Builds cubic-spline coefficients.
   - Evaluates the spline back at the original SKF grid points.
   - Confirms the spline reproduces the SKF data to numerical precision.
   - Checks homonuclear metadata such as N_F, EF, UF, and SHELL_PRESENT.

2. Constants.py
   - Creates a tiny synthetic XYZ file containing one atom of each element
     found in the SKF directory.
   - Instantiates Constants using that XYZ file.
   - Confirms Constants exposes the new f-shell data from _bond_integral.py.

3. Structure.py
   - Instantiates Structure from the same synthetic XYZ file.
   - Instantiates StructureBatch from two synthetic XYZ files with the same atoms
     in opposite orders.
   - Checks that atom-major AO indexing is correct for s, p, d, and f shells.
   - Checks shell starts/ends, global AO starts/ends, onsite diagonal entries,
     initial atomic density D0, shell-resolved Hubbard data, and AO labels.

The synthetic XYZ coordinates are not meant to be chemically meaningful. They
only exist so Constants.py and Structure.py can read a species list.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch


ATOL = 1.0e-9
EV_PER_HARTREE = 27.21138625


# =============================================================================
# Project/module loading
# =============================================================================


def find_project_root() -> Path:
    """Find the project root from either project-root/script.py or src/dftorch/script.py."""
    here = Path(__file__).resolve()

    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "dftorch" / "_bond_integral.py").is_file():
            return candidate

    if here.parent.name == "dftorch" and here.parent.parent.name == "src":
        return here.parent.parent.parent

    raise RuntimeError("Could not find project root containing src/dftorch/_bond_integral.py")


def ensure_fake_dftorch_package(project_root: Path) -> Path:
    """
    Make src/dftorch importable as a package without importing dftorch/__init__.py.

    This lets us test individual files such as _bond_integral.py and Constants.py
    without triggering unrelated package-level imports.
    """
    package_dir = project_root / "src" / "dftorch"
    fake_pkg = types.ModuleType("dftorch")
    fake_pkg.__path__ = [str(package_dir)]
    sys.modules["dftorch"] = fake_pkg
    return package_dir


def load_dftorch_module(project_root: Path, module_basename: str):
    """Load src/dftorch/{module_basename}.py as dftorch.{module_basename}."""
    package_dir = ensure_fake_dftorch_package(project_root)
    module_path = package_dir / f"{module_basename}.py"
    module_name = f"dftorch.{module_basename}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Shared SKF parsing helpers used by the tests
# =============================================================================


def read_data_lines(skf_path: Path) -> list[str]:
    lines = skf_path.read_text(errors="ignore").splitlines()
    return [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]


def split_skf_pair(skf_path: Path, bond) -> tuple[str, str]:
    """Split dashed or compact SKF basenames using the production parser helper."""
    return bond._split_skf_pair_name(skf_path.stem)


def resolve_homonuclear_skf(skf_dir: Path, sym: str, bond) -> Path:
    return Path(bond._resolve_skf_path(str(skf_dir), f"{sym}-{sym}"))


def get_original_electronic_row_count(skf_path: Path) -> int:
    """
    Return the number of electronic table rows actually present in the SKF file.

    The SKF header gives nGridPoints. The parser reads nGridPoints - 1 electronic
    rows, then appends one zero cutoff row internally. Here we count only the
    original rows in the file.
    """
    data_lines = read_data_lines(skf_path)
    if not data_lines:
        raise ValueError(f"Empty SKF file: {skf_path}")

    extended = data_lines[0].startswith("@")
    grid_idx = 1 if extended else 0

    first = data_lines[grid_idx].replace(",", " ").split()
    if len(first) < 2:
        raise ValueError(f"Malformed grid line in {skf_path}: {data_lines[grid_idx]}")

    npts_read = int(first[1])
    return npts_read - 1


# =============================================================================
# Metadata expectations independently parsed from homonuclear SKF headers
# =============================================================================


def expected_shell_metadata(shell_present: list[bool], shell_occ: list[float]) -> tuple[int, int, int]:
    """
    Return expected (N_ORB, MAX_ANG, MAX_ANG_OCC).

    The code stores MAX_ANG as l + 1, not l.
    """
    n_orb = 0
    max_ang = 0
    max_ang_occ = 0

    for l, present in enumerate(shell_present):
        if present:
            n_orb += 2 * l + 1
            max_ang = max(max_ang, l + 1)

    for l, occ in enumerate(shell_occ):
        if occ != 0.0:
            max_ang_occ = max(max_ang_occ, l + 1)

    return n_orb, max_ang, max_ang_occ


def parse_expected_homonuclear_metadata(skf_path: Path, bond) -> dict[str, object] | None:
    """
    Independently parse the homonuclear SKF header and compute expected metadata.

    Heteronuclear files return None because they do not define element-level
    onsite/Hubbard/reference-occupation data.
    """
    elem_a, elem_b = split_skf_pair(skf_path, bond)
    if elem_a != elem_b:
        return None

    data_lines = read_data_lines(skf_path)
    extended = data_lines[0].startswith("@")
    grid_idx = 1 if extended else 0

    header_tokens = data_lines[grid_idx + 1].replace(",", " ").split()

    if extended:
        if len(header_tokens) < 13:
            raise ValueError(f"Expected 13 extended header values in {skf_path}")
        (
            Ef,
            Ed,
            Ep,
            Es,
            _SPE,
            Uf,
            Ud,
            Up,
            Us,
            ff,
            fd,
            fp,
            fs,
        ) = (float(x) for x in header_tokens[:13])
    else:
        if len(header_tokens) < 10:
            raise ValueError(f"Expected 10 simple header values in {skf_path}")
        (
            Ed,
            Ep,
            Es,
            _SPE,
            Ud,
            Up,
            Us,
            fd,
            fp,
            fs,
        ) = (float(x) for x in header_tokens[:10])
        Ef = 0.0
        Uf = 0.0
        ff = 0.0

    shell_present = [
        Es != 0.0 or fs != 0.0,
        Ep != 0.0 or fp != 0.0,
        Ed != 0.0 or fd != 0.0,
        Ef != 0.0 or ff != 0.0,
    ]
    shell_occ = [fs, fp, fd, ff]
    n_orb, max_ang, max_ang_occ = expected_shell_metadata(shell_present, shell_occ)

    Z = bond.symbol_to_number[elem_a]

    return {
        "element": elem_a,
        "Z": Z,
        "N_ORB": n_orb,
        "MAX_ANG": max_ang,
        "MAX_ANG_OCC": max_ang_occ,
        "TORE": fs + fp + fd + ff,
        "N_S": fs,
        "N_P": fp,
        "N_D": fd,
        "N_F": ff,
        "ES": Es * EV_PER_HARTREE,
        "EP": Ep * EV_PER_HARTREE,
        "ED": Ed * EV_PER_HARTREE,
        "EF": Ef * EV_PER_HARTREE,
        "US": Us * EV_PER_HARTREE,
        "UP": Up * EV_PER_HARTREE,
        "UD": Ud * EV_PER_HARTREE,
        "UF": Uf * EV_PER_HARTREE,
        "SHELL_PRESENT": shell_present,
    }


# =============================================================================
# Assertion helpers
# =============================================================================


def assert_int_metadata(actual: torch.Tensor, index: int, expected: int, name: str, source: str) -> None:
    got = int(actual[index].item())
    if got != expected:
        raise AssertionError(f"{source}: {name}[{index}] expected {expected}, got {got}")


def assert_float_metadata(
    actual: torch.Tensor,
    index: int,
    expected: float,
    name: str,
    source: str,
    atol: float = ATOL,
) -> None:
    got = float(actual[index].item())
    err = abs(got - expected)
    if err > atol:
        raise AssertionError(
            f"{source}: {name}[{index}] expected {expected:.16e}, got {got:.16e}, err={err:.3e}"
        )


def assert_bool_metadata(actual: torch.Tensor, index: int, expected: list[bool], name: str, source: str) -> None:
    got = [bool(x) for x in actual[index].detach().cpu().tolist()]
    if got != expected:
        raise AssertionError(f"{source}: {name}[{index}] expected {expected}, got {got}")


def max_error_location(err: torch.Tensor, channel_names: list[str]) -> tuple[float, int, str]:
    flat_idx = int(torch.argmax(err).item())
    n_channels = err.shape[1]
    row_idx = flat_idx // n_channels
    channel_idx = flat_idx % n_channels
    return float(err[row_idx, channel_idx].item()), row_idx, channel_names[channel_idx]


def format_skf_row(values: list[float]) -> str:
    return " ".join(f"{value:.8f}" for value in values)


def write_minimal_simple_skf(path: Path, *, homonuclear: bool = False) -> list[list[float]]:
    """Write a tiny simple-format SKF file and return its source electronic rows."""
    source_rows = [
        [0.01 * (row + 1) + 0.001 * (col + 1) for col in range(20)]
        for row in range(3)
    ]
    lines = ["0.20 4"]
    if homonuclear:
        # Simple homonuclear order: Ed Ep Es SPE Ud Up Us fd fp fs.
        lines.append("-0.30 -0.20 -0.10 0.0 0.03 0.02 0.01 0.30 0.20 0.10")
    lines.extend(
        [
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
            *[format_skf_row(row) for row in source_rows],
            "Spline",
            "2 2.0",
            "0.0 0.0 0.0",
            "0.0 1.0 0.0 0.0 0.0 0.0",
            "1.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return source_rows


def read_skf_as_matrix(
    skf_path: Path,
    bond,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = make_metadata_tensors(device, dtype)
    R, channels, _R_rep, _rep_splines, _close_exp = bond.read_skf_table(
        str(skf_path),
        *metadata,
        device=device,
        dtype=dtype,
    )
    return R, bond.channels_to_matrix(channels)


# =============================================================================
# _bond_integral.py tests
# =============================================================================


def make_metadata_tensors(device: torch.device, dtype: torch.dtype):
    """Allocate the mutable metadata tensors expected by read_skf_table()."""
    n_elements = 120
    max_shells = 4

    N_ORB = torch.zeros(n_elements, dtype=torch.int64, device=device)
    MAX_ANG = torch.zeros(n_elements, dtype=torch.int64, device=device)
    MAX_ANG_OCC = torch.zeros(n_elements, dtype=torch.int64, device=device)

    TORE = torch.zeros(n_elements, dtype=dtype, device=device)
    N_S = torch.zeros(n_elements, dtype=dtype, device=device)
    N_P = torch.zeros(n_elements, dtype=dtype, device=device)
    N_D = torch.zeros(n_elements, dtype=dtype, device=device)
    N_F = torch.zeros(n_elements, dtype=dtype, device=device)

    ES = torch.zeros(n_elements, dtype=dtype, device=device)
    EP = torch.zeros(n_elements, dtype=dtype, device=device)
    ED = torch.zeros(n_elements, dtype=dtype, device=device)
    EF = torch.zeros(n_elements, dtype=dtype, device=device)

    US = torch.zeros(n_elements, dtype=dtype, device=device)
    UP = torch.zeros(n_elements, dtype=dtype, device=device)
    UD = torch.zeros(n_elements, dtype=dtype, device=device)
    UF = torch.zeros(n_elements, dtype=dtype, device=device)

    SHELL_PRESENT = torch.zeros((n_elements, max_shells), dtype=torch.bool, device=device)

    return (
        N_ORB,
        MAX_ANG,
        MAX_ANG_OCC,
        TORE,
        N_S,
        N_P,
        N_D,
        N_F,
        ES,
        EP,
        ED,
        EF,
        US,
        UP,
        UD,
        UF,
        SHELL_PRESENT,
    )


def metadata_tuple_to_dict(metadata: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    names = [
        "N_ORB",
        "MAX_ANG",
        "MAX_ANG_OCC",
        "TORE",
        "N_S",
        "N_P",
        "N_D",
        "N_F",
        "ES",
        "EP",
        "ED",
        "EF",
        "US",
        "UP",
        "UD",
        "UF",
        "SHELL_PRESENT",
    ]
    return dict(zip(names, metadata))


def check_metadata_from_single_read(skf_path: Path, bond, metadata_dict: dict[str, torch.Tensor]) -> dict | None:
    expected = parse_expected_homonuclear_metadata(skf_path, bond)
    if expected is None:
        return None

    Z = int(expected["Z"])
    source = skf_path.name

    assert_int_metadata(metadata_dict["N_ORB"], Z, int(expected["N_ORB"]), "N_ORB", source)
    assert_int_metadata(metadata_dict["MAX_ANG"], Z, int(expected["MAX_ANG"]), "MAX_ANG", source)
    assert_int_metadata(metadata_dict["MAX_ANG_OCC"], Z, int(expected["MAX_ANG_OCC"]), "MAX_ANG_OCC", source)

    for name in ["TORE", "N_S", "N_P", "N_D", "N_F", "ES", "EP", "ED", "EF", "US", "UP", "UD", "UF"]:
        assert_float_metadata(metadata_dict[name], Z, float(expected[name]), name, source)

    assert_bool_metadata(metadata_dict["SHELL_PRESENT"], Z, list(expected["SHELL_PRESENT"]), "SHELL_PRESENT", source)
    return expected


def check_one_skf(skf_path: Path, bond, device: torch.device, dtype: torch.dtype) -> dict:
    metadata = make_metadata_tensors(device, dtype)
    metadata_dict = metadata_tuple_to_dict(metadata)

    R, channels, _R_rep, _rep_splines, _close_exp = bond.read_skf_table(
        str(skf_path),
        *metadata,
        device=device,
        dtype=dtype,
    )

    M = bond.channels_to_matrix(channels)
    coeffs = bond.cubic_spline_coeffs(R, M)
    original_rows = get_original_electronic_row_count(skf_path)

    a = coeffs[:, :, 0]
    b = coeffs[:, :, 1]
    c = coeffs[:, :, 2]
    d = coeffs[:, :, 3]

    left_reconstructed = a[:original_rows]
    left_target = M[:original_rows]
    left_err = (left_reconstructed - left_target).abs()

    h = (R[1:] - R[:-1]).unsqueeze(1)
    right_reconstructed = a + b * h + c * h**2 + d * h**3

    if original_rows > 1:
        right_err = (right_reconstructed[: original_rows - 1] - M[1:original_rows]).abs()
    else:
        right_err = torch.zeros((0, M.shape[1]), dtype=dtype, device=device)

    left_max, left_row, left_ch = max_error_location(left_err, bond._CHANNELS)

    if right_err.numel() > 0:
        right_max, right_row, right_ch = max_error_location(right_err, bond._CHANNELS)
        right_target_row = right_row + 1
    else:
        right_max, right_row, right_target_row, right_ch = 0.0, 0, 0, "none"

    expected_metadata = check_metadata_from_single_read(skf_path, bond, metadata_dict)
    metadata_checked = expected_metadata is not None
    passed = left_max <= ATOL and right_max <= ATOL

    return {
        "file": skf_path.name,
        "original_rows": original_rows,
        "channels": M.shape[1],
        "left_max": left_max,
        "left_row": left_row,
        "left_channel": left_ch,
        "right_max": right_max,
        "right_target_row": right_target_row,
        "right_channel": right_ch,
        "metadata_checked": metadata_checked,
        "metadata": expected_metadata,
        "passed": passed,
    }


def check_simple_canonical_channels(
    bond,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        skf_path = Path(tmp) / "C-N.skf"
        source_rows = write_minimal_simple_skf(skf_path)
        _R, M = read_skf_as_matrix(skf_path, bond, device, dtype)

    if len(bond._CHANNELS) != 40:
        raise AssertionError(f"_CHANNELS expected 40 entries, got {len(bond._CHANNELS)}")
    if len(bond._SIMPLE_CHANNELS) != 20:
        raise AssertionError(f"_SIMPLE_CHANNELS expected 20 entries, got {len(bond._SIMPLE_CHANNELS)}")
    if len(bond._SIMPLE_TO_EXTENDED) != len(bond._SIMPLE_CHANNELS):
        raise AssertionError("_SIMPLE_TO_EXTENDED length does not match _SIMPLE_CHANNELS")
    if M.shape[1] != len(bond._CHANNELS):
        raise AssertionError(f"simple SKF matrix expected {len(bond._CHANNELS)} channels, got {M.shape[1]}")

    for row_idx, source_row in enumerate(source_rows):
        for simple_idx, channel_name in enumerate(bond._SIMPLE_CHANNELS):
            channel_idx = bond._CHANNELS.index(channel_name)
            mapped_idx = int(bond._SIMPLE_TO_EXTENDED[simple_idx])
            if mapped_idx != channel_idx:
                raise AssertionError(
                    f"_SIMPLE_TO_EXTENDED[{simple_idx}] for {channel_name} expected {channel_idx}, got {mapped_idx}"
                )
            got = float(M[row_idx, channel_idx].item())
            expected = source_row[simple_idx] * EV_PER_HARTREE
            err = abs(got - expected)
            if err > ATOL:
                raise AssertionError(
                    f"simple row {row_idx} channel {channel_name} expected {expected:.16e}, "
                    f"got {got:.16e}, err={err:.3e}"
                )

    simple_channels = set(bond._SIMPLE_CHANNELS)
    for channel_idx, channel_name in enumerate(bond._CHANNELS):
        if channel_name in simple_channels:
            continue
        values = M[: len(source_rows), channel_idx].abs()
        max_abs = float(values.max().item())
        if max_abs > ATOL:
            raise AssertionError(
                f"simple SKF non-simple channel {channel_name} expected zero, max_abs={max_abs:.3e}"
            )

    return {
        "file": skf_path.name,
        "rows": len(source_rows),
        "channels": M.shape[1],
        "mapped_channels": len(bond._SIMPLE_CHANNELS),
        "zero_filled_channels": len(bond._CHANNELS) - len(bond._SIMPLE_CHANNELS),
    }


def check_simple_homonuclear_metadata(
    bond,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        skf_path = Path(tmp) / "C-C.skf"
        write_minimal_simple_skf(skf_path, homonuclear=True)
        metadata = make_metadata_tensors(device, dtype)
        metadata_dict = metadata_tuple_to_dict(metadata)
        bond.read_skf_table(str(skf_path), *metadata, device=device, dtype=dtype)
        expected = parse_expected_homonuclear_metadata(skf_path, bond)

    if expected is None:
        raise AssertionError("temporary simple homonuclear SKF did not produce expected metadata")

    Z = int(expected["Z"])
    source = skf_path.name

    assert_int_metadata(metadata_dict["N_ORB"], Z, int(expected["N_ORB"]), "N_ORB", source)
    assert_int_metadata(metadata_dict["MAX_ANG"], Z, int(expected["MAX_ANG"]), "MAX_ANG", source)
    assert_int_metadata(metadata_dict["MAX_ANG_OCC"], Z, int(expected["MAX_ANG_OCC"]), "MAX_ANG_OCC", source)

    for name in ["TORE", "N_S", "N_P", "N_D", "N_F", "ES", "EP", "ED", "EF", "US", "UP", "UD", "UF"]:
        assert_float_metadata(metadata_dict[name], Z, float(expected[name]), name, source)

    assert_bool_metadata(metadata_dict["SHELL_PRESENT"], Z, list(expected["SHELL_PRESENT"]), "SHELL_PRESENT", source)

    if float(metadata_dict["N_F"][Z].item()) != 0.0:
        raise AssertionError("simple homonuclear N_F should be zero")
    if float(metadata_dict["EF"][Z].item()) != 0.0:
        raise AssertionError("simple homonuclear EF should be zero")
    if float(metadata_dict["UF"][Z].item()) != 0.0:
        raise AssertionError("simple homonuclear UF should be zero")

    return {
        "file": skf_path.name,
        "N_ORB": int(metadata_dict["N_ORB"][Z].item()),
        "N_F": float(metadata_dict["N_F"][Z].item()),
        "SHELL_PRESENT": [bool(x) for x in metadata_dict["SHELL_PRESENT"][Z].detach().cpu().tolist()],
    }


def check_extended_fixture_channel_width(
    skf_dir: Path,
    bond,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    skf_path = sorted(skf_dir.glob("*.skf"))[0]
    _R, M = read_skf_as_matrix(skf_path, bond, device, dtype)
    if M.shape[1] != len(bond._CHANNELS):
        raise AssertionError(f"{skf_path.name}: expected {len(bond._CHANNELS)} channels, got {M.shape[1]}")
    return {"file": skf_path.name, "channels": M.shape[1]}


def check_pair_name_and_path_helpers(bond) -> dict:
    if bond._split_skf_pair_name("Eu-Ga") != ("Eu", "Ga"):
        raise AssertionError("dashed pair name Eu-Ga did not parse to ('Eu', 'Ga')")
    if bond._split_skf_pair_name("EuGa") != ("Eu", "Ga"):
        raise AssertionError("compact pair name EuGa did not parse to ('Eu', 'Ga')")
    if bond._split_skf_pair_name("NN") != ("N", "N"):
        raise AssertionError("compact pair name NN did not parse to ('N', 'N')")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        dashed = tmp_dir / "Eu-Ga.skf"
        compact = tmp_dir / "EuGa.skf"
        dashed.write_text("dashed\n")
        compact.write_text("compact\n")

        resolved = Path(bond._resolve_skf_path(str(tmp_dir), "Eu-Ga"))
        if resolved != dashed:
            raise AssertionError(f"resolver expected dashed path {dashed}, got {resolved}")

        dashed.unlink()
        resolved = Path(bond._resolve_skf_path(str(tmp_dir), "Eu-Ga"))
        if resolved != compact:
            raise AssertionError(f"resolver expected compact fallback {compact}, got {resolved}")

        compact.unlink()
        resolved = Path(bond._resolve_skf_path(str(tmp_dir), "Eu-Ga"))
        if resolved != dashed:
            raise AssertionError(f"resolver expected missing dashed target {dashed}, got {resolved}")

    return {"pairs_checked": ["Eu-Ga", "EuGa", "NN"], "resolver_cases": 3}


def check_skipped_shell_errors(bond) -> dict:
    cases = [
        ("p-without-s", ("N", False, True, False, False, "p_without_s.skf"), "p-shell basis requires an s shell"),
        ("d-without-p", ("Ga", True, False, True, False, "d_without_p.skf"), "d-shell basis requires nested s/p/d shells"),
        ("f-without-d", ("Eu", True, True, False, True, "f_without_d.skf"), "f-shell basis requires nested s/p/d/f shells"),
    ]
    for name, args, expected_fragment in cases:
        try:
            bond._validate_nested_shells(*args)
        except ValueError as exc:
            message = str(exc)
            if expected_fragment not in message:
                raise AssertionError(f"{name}: expected message containing {expected_fragment!r}, got {message!r}") from exc
        else:
            raise AssertionError(f"{name}: expected ValueError")
    return {"cases_checked": [name for name, _args, _fragment in cases]}


def check_simple_spline_reconstruction(
    bond,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        skf_path = Path(tmp) / "C-N.skf"
        source_rows = write_minimal_simple_skf(skf_path)
        R, M = read_skf_as_matrix(skf_path, bond, device, dtype)

    if M.shape[1] != len(bond._CHANNELS):
        raise AssertionError(f"simple spline matrix expected {len(bond._CHANNELS)} channels, got {M.shape[1]}")

    return check_source_grid_reconstruction(
        "temporary-simple-format.skf",
        bond,
        R,
        M,
        len(source_rows),
        bond._CHANNELS,
    )


def check_source_grid_reconstruction(
    name: str,
    bond,
    R: torch.Tensor,
    M: torch.Tensor,
    original_rows: int,
    channel_names: list[str],
) -> dict:
    coeffs = bond.cubic_spline_coeffs(R, M)
    a = coeffs[:, :, 0]
    b = coeffs[:, :, 1]
    c = coeffs[:, :, 2]
    d = coeffs[:, :, 3]

    left_err = (a[:original_rows] - M[:original_rows]).abs()
    h = (R[1:] - R[:-1]).unsqueeze(1)
    right_reconstructed = a + b * h + c * h**2 + d * h**3
    if original_rows > 1:
        right_err = (right_reconstructed[: original_rows - 1] - M[1:original_rows]).abs()
    else:
        right_err = torch.zeros((0, M.shape[1]), dtype=M.dtype, device=M.device)

    left_max, left_row, left_channel = max_error_location(left_err, channel_names)
    if right_err.numel() > 0:
        right_max, right_row, right_channel = max_error_location(right_err, channel_names)
        right_target_row = right_row + 1
    else:
        right_max, right_target_row, right_channel = 0.0, 0, "none"

    if left_max > ATOL or right_max > ATOL:
        raise AssertionError(
            f"{name}: source-grid reconstruction failed: "
            f"left_max={left_max:.3e} at row {left_row} channel {left_channel}, "
            f"right_max={right_max:.3e} at target row {right_target_row} channel {right_channel}"
        )

    return {
        "file": name,
        "original_rows": original_rows,
        "left_max": left_max,
        "left_row": left_row,
        "left_channel": left_channel,
        "right_max": right_max,
        "right_target_row": right_target_row,
        "right_channel": right_channel,
    }


def collect_elements_from_skf_dir(skf_dir: Path, bond) -> list[str]:
    elements = set()
    for skf_path in skf_dir.glob("*.skf"):
        elem_a, elem_b = split_skf_pair(skf_path, bond)
        if elem_a not in bond.symbol_to_number:
            raise ValueError(f"Unknown element {elem_a} in {skf_path.name}")
        if elem_b not in bond.symbol_to_number:
            raise ValueError(f"Unknown element {elem_b} in {skf_path.name}")
        elements.add(elem_a)
        elements.add(elem_b)
    return sorted(elements, key=lambda sym: bond.symbol_to_number[sym])


def check_get_skf_tensors_metadata(skf_dir: Path, bond, device: torch.device) -> None:
    elements = collect_elements_from_skf_dir(skf_dir, bond)
    TYPE = torch.tensor([bond.symbol_to_number[sym] for sym in elements], dtype=torch.long, device=device)

    (
        _R_tensor,
        _R_orb,
        # Per-pair tabulated grid lengths, added to the get_skf_tensors return
        # tuple by plan 05-01 (decision D-01, requirement REG-06). This file is
        # loaded and executed by tests/test_f_orbital_skf.py via
        # load_validation_script(), so a stale unpacking here is NOT inert.
        _n_grid,
        coeffs_tensor,
        _R_rep_tensor,
        _rep_splines_tensor,
        _close_exp_tensor,
        N_ORB,
        MAX_ANG,
        MAX_ANG_OCC,
        TORE,
        N_S,
        N_P,
        N_D,
        N_F,
        ES,
        EP,
        ED,
        EF,
        US,
        UP,
        UD,
        UF,
        SHELL_PRESENT,
    ) = bond.get_skf_tensors(TYPE, str(skf_dir))

    if coeffs_tensor.shape[2] != 40:
        raise AssertionError(f"get_skf_tensors coeffs_tensor expected 40 channels, got {coeffs_tensor.shape[2]}")

    returned = {
        "N_ORB": N_ORB,
        "MAX_ANG": MAX_ANG,
        "MAX_ANG_OCC": MAX_ANG_OCC,
        "TORE": TORE,
        "N_S": N_S,
        "N_P": N_P,
        "N_D": N_D,
        "N_F": N_F,
        "ES": ES,
        "EP": EP,
        "ED": ED,
        "EF": EF,
        "US": US,
        "UP": UP,
        "UD": UD,
        "UF": UF,
        "SHELL_PRESENT": SHELL_PRESENT,
    }

    checked = 0
    for sym in elements:
        homonuclear = resolve_homonuclear_skf(skf_dir, sym, bond)
        if not homonuclear.is_file():
            continue
        expected = parse_expected_homonuclear_metadata(homonuclear, bond)
        if expected is None:
            continue

        Z = int(expected["Z"])
        source = f"get_skf_tensors/{homonuclear.name}"

        assert_int_metadata(returned["N_ORB"], Z, int(expected["N_ORB"]), "N_ORB", source)
        assert_int_metadata(returned["MAX_ANG"], Z, int(expected["MAX_ANG"]), "MAX_ANG", source)
        assert_int_metadata(returned["MAX_ANG_OCC"], Z, int(expected["MAX_ANG_OCC"]), "MAX_ANG_OCC", source)

        for name in ["TORE", "N_S", "N_P", "N_D", "N_F", "ES", "EP", "ED", "EF", "US", "UP", "UD", "UF"]:
            assert_float_metadata(returned[name], Z, float(expected[name]), name, source)

        assert_bool_metadata(returned["SHELL_PRESENT"], Z, list(expected["SHELL_PRESENT"]), "SHELL_PRESENT", source)
        checked += 1

    if checked == 0:
        raise AssertionError("No homonuclear SKF files were available for get_skf_tensors metadata checks")

    print()
    print("get_skf_tensors metadata check:")
    print(f"  elements from SKF directory: {elements}")
    print(f"  coeffs_tensor channels: {coeffs_tensor.shape[2]}")
    print(f"  homonuclear metadata entries checked: {checked}")


def run_bond_integral_tests(skf_dir: Path, bond, device: torch.device, dtype: torch.dtype) -> list[dict]:
    skf_files = sorted(skf_dir.glob("*.skf"))
    if not skf_files:
        raise FileNotFoundError(f"No .skf files found in: {skf_dir}")

    print(f"Testing _bond_integral.py for {len(skf_files)} SKF files")
    print(f"SKF directory: {skf_dir}")
    print(f"Tolerance: {ATOL:.1e}")
    print()

    failures = []
    metadata_count = 0

    try:
        simple_result = check_simple_canonical_channels(bond, device, dtype)
        print(
            "PASS temporary simple-format canonicalization: "
            f"rows={simple_result['rows']}, "
            f"channels={simple_result['channels']}, "
            f"mapped_channels={simple_result['mapped_channels']}, "
            f"zero_filled_channels={simple_result['zero_filled_channels']}"
        )
    except Exception as exc:
        failures.append({"file": "temporary-simple-format.skf", "error": repr(exc)})
        print(f"FAIL temporary simple-format canonicalization: {exc!r}")

    try:
        simple_meta_result = check_simple_homonuclear_metadata(bond, device, dtype)
        print(
            "PASS temporary simple-format homonuclear metadata: "
            f"{simple_meta_result['file']} "
            f"N_ORB={simple_meta_result['N_ORB']}, "
            f"N_F={simple_meta_result['N_F']}, "
            f"SHELL_PRESENT={simple_meta_result['SHELL_PRESENT']}"
        )
    except Exception as exc:
        failures.append({"file": "temporary-simple-homonuclear-metadata", "error": repr(exc)})
        print(f"FAIL temporary simple-format homonuclear metadata: {exc!r}")

    try:
        extended_result = check_extended_fixture_channel_width(skf_dir, bond, device, dtype)
        print(
            "PASS extended fixture parser boundary: "
            f"{extended_result['file']} channels={extended_result['channels']}"
        )
    except Exception as exc:
        failures.append({"file": "extended-fixture-channel-width", "error": repr(exc)})
        print(f"FAIL extended fixture parser boundary: {exc!r}")

    try:
        pair_result = check_pair_name_and_path_helpers(bond)
        print(
            "PASS compact/dashed SKF helper behavior: "
            f"pairs={pair_result['pairs_checked']}, "
            f"resolver_cases={pair_result['resolver_cases']}"
        )
    except Exception as exc:
        failures.append({"file": "compact-dashed-resolver", "error": repr(exc)})
        print(f"FAIL compact/dashed SKF helper behavior: {exc!r}")

    try:
        skipped_result = check_skipped_shell_errors(bond)
        print(f"PASS skipped-shell rejection: cases={skipped_result['cases_checked']}")
    except Exception as exc:
        failures.append({"file": "skipped-shell-rejection", "error": repr(exc)})
        print(f"FAIL skipped-shell rejection: {exc!r}")

    try:
        simple_spline_result = check_simple_spline_reconstruction(bond, device, dtype)
        print(
            "PASS temporary simple-format source-grid spline reconstruction: "
            f"rows={simple_spline_result['original_rows']}, "
            f"channels={len(bond._CHANNELS)}, "
            f"left_max={simple_spline_result['left_max']:.3e} "
            f"at row {simple_spline_result['left_row']} channel {simple_spline_result['left_channel']}, "
            f"right_max={simple_spline_result['right_max']:.3e} "
            f"at target row {simple_spline_result['right_target_row']} "
            f"channel {simple_spline_result['right_channel']}"
        )
    except Exception as exc:
        failures.append({"file": "temporary-simple-format-spline", "error": repr(exc)})
        print(f"FAIL temporary simple-format source-grid spline reconstruction: {exc!r}")

    print()
    print(f"Source-grid spline reconstruction for {len(skf_files)} f-orbital SKF fixtures:")

    for skf_path in skf_files:
        try:
            result = check_one_skf(skf_path, bond, device, dtype)
        except Exception as exc:
            failures.append({"file": skf_path.name, "error": repr(exc)})
            print(f"FAIL {skf_path.name}: {exc!r}")
            continue

        status = "PASS" if result["passed"] else "FAIL"
        meta_status = "metadata=yes" if result["metadata_checked"] else "metadata=no"

        print(
            f"{status} {result['file']}: "
            f"rows={result['original_rows']}, "
            f"channels={result['channels']}, "
            f"left_max={result['left_max']:.3e} "
            f"at row {result['left_row']} channel {result['left_channel']}, "
            f"right_max={result['right_max']:.3e} "
            f"at target row {result['right_target_row']} channel {result['right_channel']}, "
            f"{meta_status}"
        )

        if result["metadata_checked"]:
            md = result["metadata"]
            metadata_count += 1
            print(
                f"    {md['element']}: "
                f"N_ORB={md['N_ORB']}, "
                f"MAX_ANG={md['MAX_ANG']}, "
                f"MAX_ANG_OCC={md['MAX_ANG_OCC']}, "
                f"SHELL_PRESENT={md['SHELL_PRESENT']}, "
                f"N=[s={md['N_S']}, p={md['N_P']}, d={md['N_D']}, f={md['N_F']}], "
                f"EF={md['EF']:.8e}, "
                f"UF={md['UF']:.8e}"
            )

        if not result["passed"]:
            failures.append(result)

    if metadata_count == 0:
        failures.append({"file": "metadata", "error": "No homonuclear SKF metadata checks were run"})

    try:
        check_get_skf_tensors_metadata(skf_dir, bond, device)
    except Exception as exc:
        failures.append({"file": "get_skf_tensors", "error": repr(exc)})
        print()
        print(f"FAIL get_skf_tensors metadata check: {exc!r}")

    return failures


# =============================================================================
# Constants.py tests
# =============================================================================


def write_test_xyz(path: Path, elements: list[str]) -> None:
    """Write a tiny synthetic XYZ containing one atom of each element."""
    lines = [str(len(elements)), "synthetic Constants.py test system"]
    for i, sym in enumerate(elements):
        # Coordinates only need to be parseable; Constants uses the species list.
        lines.append(f"{sym} {1.5 * i:.8f} 0.00000000 0.00000000")
    path.write_text("\n".join(lines) + "\n")


def check_constants_against_expected(const, skf_dir: Path, bond, elements: list[str]) -> None:
    """Check that Constants exposes the SKF metadata as registered attributes."""
    if list(const.shell_dim.detach().cpu().tolist()) != [0, 1, 3, 5, 7]:
        raise AssertionError(f"Constants.shell_dim expected [0, 1, 3, 5, 7], got {const.shell_dim.tolist()}")

    required_attrs = [
        "n_orb",
        "max_ang",
        "max_ang_occ",
        "tore",
        "n_s",
        "n_p",
        "n_d",
        "n_f",
        "Es",
        "Ep",
        "Ed",
        "Ef",
        "U",
        "Up",
        "Ud",
        "Uf",
        "shell_present",
        "coeffs_tensor",
        "pair_lookup",
    ]
    for attr in required_attrs:
        if not hasattr(const, attr):
            raise AssertionError(f"Constants is missing expected attribute: {attr}")

    if const.coeffs_tensor.shape[2] != 40:
        raise AssertionError(f"Constants.coeffs_tensor expected 40 channels, got {const.coeffs_tensor.shape[2]}")

    # All ordered pairs among the elements in the synthetic XYZ should be present.
    for elem_a in elements:
        for elem_b in elements:
            za = bond.symbol_to_number[elem_a]
            zb = bond.symbol_to_number[elem_b]
            idx = int(const.pair_lookup[za, zb].item())
            if idx < 0:
                raise AssertionError(f"Constants.pair_lookup missing pair {elem_a}-{elem_b}")

    checked = 0
    for sym in elements:
        homonuclear = resolve_homonuclear_skf(skf_dir, sym, bond)
        if not homonuclear.is_file():
            continue

        expected = parse_expected_homonuclear_metadata(homonuclear, bond)
        if expected is None:
            continue

        Z = int(expected["Z"])
        source = f"Constants/{homonuclear.name}"

        assert_int_metadata(const.n_orb, Z, int(expected["N_ORB"]), "n_orb", source)
        assert_int_metadata(const.max_ang, Z, int(expected["MAX_ANG"]), "max_ang", source)
        assert_int_metadata(const.max_ang_occ, Z, int(expected["MAX_ANG_OCC"]), "max_ang_occ", source)

        float_attr_to_expected = {
            "tore": "TORE",
            "n_s": "N_S",
            "n_p": "N_P",
            "n_d": "N_D",
            "n_f": "N_F",
            "Es": "ES",
            "Ep": "EP",
            "Ed": "ED",
            "Ef": "EF",
            "U": "US",
            "Up": "UP",
            "Ud": "UD",
            "Uf": "UF",
        }
        for attr, expected_name in float_attr_to_expected.items():
            assert_float_metadata(getattr(const, attr), Z, float(expected[expected_name]), attr, source)

        assert_bool_metadata(const.shell_present, Z, list(expected["SHELL_PRESENT"]), "shell_present", source)
        checked += 1

    if checked == 0:
        raise AssertionError("No Constants homonuclear metadata entries were checked")

    print()
    print("Constants.py check:")
    print(f"  synthetic XYZ elements: {elements}")
    print(f"  coeffs_tensor channels: {const.coeffs_tensor.shape[2]}")
    print(f"  ordered pair entries checked: {len(elements) * len(elements)}")
    print(f"  homonuclear metadata entries checked: {checked}")


def run_constants_tests(project_root: Path, skf_dir: Path, bond) -> list[dict]:
    failures = []

    try:
        constants_mod = load_dftorch_module(project_root, "Constants")
        elements = collect_elements_from_skf_dir(skf_dir, bond)

        tmp_dir = project_root / ".tmp_f_orbital_validation"
        tmp_dir.mkdir(exist_ok=True)
        xyz_path = tmp_dir / "N_Ga_Eu_constants_test.xyz"
        write_test_xyz(xyz_path, elements)

        const = constants_mod.Constants(
            {
                "SKFPATH": str(skf_dir),
                "FILENAME": str(xyz_path),
                "DFTB3": False,
                "MAGNETIC_HUBBARD_LDEP": False,
                "GRAD_PARAM": False,
            }
        )

        check_constants_against_expected(const, skf_dir, bond, elements)
    except Exception as exc:
        failures.append({"file": "Constants.py", "error": repr(exc)})
        print()
        print(f"FAIL Constants.py check: {exc!r}")

    return failures


# =============================================================================
# STRUCTURE.PY TESTS START HERE
# =============================================================================
#
# The functions below test the file Structure.py.  These are deliberately
# independent of the Slater-Koster angular transformation code.  Structure.py
# does not build H/S matrix elements; it only decides where each atom and shell
# lives in the AO vector.  That means we can test it before editing
# _slater_koster_pair.py.
#
# The main idea being tested is:
#
#     atom i has element Z
#     Constants.shell_present[Z] tells us which shells exist
#     Structure.py converts that into actual AO positions
#
# For the fixed local AO layout:
#
#     s:    local AO 0
#     p:    local AOs 1,2,3
#     d:    local AOs 4,5,6,7,8
#     f:    local AOs 9,10,11,12,13,14,15
#
# So if an atom has spdf, it contributes 16 orbitals.
# If an atom has spd, it contributes 9 orbitals.
# If an atom has sp, it contributes 4 orbitals.
# =============================================================================


EXPECTED_SHELL_DIMS = [1, 3, 5, 7]
EXPECTED_SHELL_LOCAL_STARTS = [0, 1, 4, 9]
EXPECTED_SHELL_TYPE_IDS = [1, 2, 3, 4]
EXPECTED_AO_LABEL_TEMPLATE = [
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
]
EXPECTED_AO_SHELL_TEMPLATE = [
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]


def build_test_constants(project_root: Path, skf_dir: Path, bond, elements: list[str], xyz_path: Path):
    """Create a Constants object using a synthetic XYZ containing the requested elements."""
    constants_mod = load_dftorch_module(project_root, "Constants")
    write_test_xyz(xyz_path, elements)
    const = constants_mod.Constants(
        {
            "SKFPATH": str(skf_dir),
            "FILENAME": str(xyz_path),
            "DFTB3": False,
            "MAGNETIC_HUBBARD_LDEP": False,
            "GRAD_PARAM": False,
        }
    )
    return const


def expected_metadata_by_element(skf_dir: Path, bond, elements: list[str]) -> dict[str, dict[str, object]]:
    """Return independently parsed homonuclear metadata for every test element."""
    out: dict[str, dict[str, object]] = {}
    for sym in elements:
        homonuclear = resolve_homonuclear_skf(skf_dir, sym, bond)
        if not homonuclear.is_file():
            raise AssertionError(f"Missing homonuclear SKF needed for Structure test: {homonuclear.name}")
        expected = parse_expected_homonuclear_metadata(homonuclear, bond)
        if expected is None:
            raise AssertionError(f"Could not parse metadata from {homonuclear.name}")
        out[sym] = expected
    return out


def expected_local_starts(shell_present: list[bool]) -> list[int]:
    return [start if present else -1 for start, present in zip(EXPECTED_SHELL_LOCAL_STARTS, shell_present)]


def expected_local_ends(shell_present: list[bool]) -> list[int]:
    return [
        start + dim - 1 if present else -1
        for start, dim, present in zip(EXPECTED_SHELL_LOCAL_STARTS, EXPECTED_SHELL_DIMS, shell_present)
    ]


def expected_ao_labels(shell_present: list[bool]) -> list[str]:
    labels: list[str] = []
    cursor = 0
    for present, dim in zip(shell_present, EXPECTED_SHELL_DIMS):
        if present:
            labels.extend(EXPECTED_AO_LABEL_TEMPLATE[cursor : cursor + dim])
        cursor += dim
    return labels


def expected_ao_shell_types(shell_present: list[bool]) -> list[int]:
    shell_types: list[int] = []
    cursor = 0
    for present, dim in zip(shell_present, EXPECTED_SHELL_DIMS):
        if present:
            shell_types.extend(EXPECTED_AO_SHELL_TEMPLATE[cursor : cursor + dim])
        cursor += dim
    return shell_types


def expected_diagonal_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    energies = [float(md["ES"]), float(md["EP"]), float(md["ED"]), float(md["EF"])]
    vals: list[float] = []
    for present, energy, dim in zip(shell_present, energies, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([energy] * dim)
    return vals


def expected_d0_values(md: dict[str, object]) -> list[float]:
    """Expected Structure.D0 values after the 0.5 closed-shell factor."""
    shell_present = list(md["SHELL_PRESENT"])
    occs = [float(md["N_S"]), float(md["N_P"]), float(md["N_D"]), float(md["N_F"])]
    vals: list[float] = []
    for present, occ, dim in zip(shell_present, occs, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([0.5 * occ / float(dim)] * dim)
    return vals


def expected_shell_hubbard_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    vals = [float(md["US"]), float(md["UP"]), float(md["UD"]), float(md["UF"])]
    return [v for present, v in zip(shell_present, vals) if present]


def expected_el_per_shell_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    vals = [float(md["N_S"]), float(md["N_P"]), float(md["N_D"]), float(md["N_F"])]
    return [v for present, v in zip(shell_present, vals) if present]


def expected_shell_type_values(md: dict[str, object]) -> list[int]:
    shell_present = list(md["SHELL_PRESENT"])
    return [sid for present, sid in zip(shell_present, EXPECTED_SHELL_TYPE_IDS) if present]


def assert_list_equal(got: list, expected: list, name: str) -> None:
    if got != expected:
        raise AssertionError(f"{name} expected {expected}, got {got}")


def assert_tensor_int_list(tensor: torch.Tensor, expected: list[int], name: str) -> None:
    got = [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    if got != expected:
        raise AssertionError(f"{name} expected {expected}, got {got}")


def assert_tensor_bool_list(tensor: torch.Tensor, expected: list[bool], name: str) -> None:
    got = [bool(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    if got != expected:
        raise AssertionError(f"{name} expected {expected}, got {got}")


def assert_tensor_float_list(
    tensor: torch.Tensor,
    expected: list[float],
    name: str,
    atol: float = ATOL,
) -> None:
    got = [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    if len(got) != len(expected):
        raise AssertionError(f"{name} expected length {len(expected)}, got {len(got)}")
    for i, (g, e) in enumerate(zip(got, expected)):
        err = abs(g - e)
        if err > atol:
            raise AssertionError(f"{name}[{i}] expected {e:.16e}, got {g:.16e}, err={err:.3e}")


def check_single_structure_layout(struct, elements: list[str], expected_by_element: dict[str, dict[str, object]]) -> None:
    """Check Structure's atom-major AO and shell metadata.

    Example for elements [N, Ga, Eu]:

        N  contributes n_N  orbitals starting at H_INDEX_START[0] = 0
        Ga contributes n_Ga orbitals starting after N
        Eu contributes n_Eu orbitals starting after N + Ga

    For each atom, Structure.py must also know where each shell begins.
    If Eu is spdf and starts globally at AO 13, then its f shell starts at
    13 + 9 = 22.  That kind of offset bookkeeping is what this test checks.
    """
    expected_types = [int(expected_by_element[sym]["Z"]) for sym in elements]
    expected_n_orb = [int(expected_by_element[sym]["N_ORB"]) for sym in elements]
    expected_h_start = [0]
    for n_orb in expected_n_orb[:-1]:
        expected_h_start.append(expected_h_start[-1] + n_orb)
    expected_h_end = [start + n_orb - 1 for start, n_orb in zip(expected_h_start, expected_n_orb)]
    expected_hdim = sum(expected_n_orb)

    assert_tensor_int_list(struct.TYPE, expected_types, "Structure.TYPE")
    assert_tensor_int_list(struct.n_orbitals_per_atom, expected_n_orb, "Structure.n_orbitals_per_atom")
    assert_tensor_int_list(struct.H_INDEX_START, expected_h_start, "Structure.H_INDEX_START")
    assert_tensor_int_list(struct.H_INDEX_END, expected_h_end, "Structure.H_INDEX_END")
    if int(struct.HDIM) != expected_hdim:
        raise AssertionError(f"Structure.HDIM expected {expected_hdim}, got {struct.HDIM}")
    if int(struct.diagonal.shape[0]) != expected_hdim:
        raise AssertionError(f"Structure.diagonal length expected {expected_hdim}, got {struct.diagonal.shape[0]}")

    expected_shell_present: list[list[bool]] = []
    expected_shell_local_start: list[list[int]] = []
    expected_shell_local_end: list[list[int]] = []
    expected_shell_ao_start: list[list[int]] = []
    expected_shell_ao_end: list[list[int]] = []
    expected_labels: list[str] = []
    expected_ao_types: list[int] = []
    expected_diagonal: list[float] = []
    expected_d0: list[float] = []
    expected_n_shells: list[int] = []
    expected_shell_u: list[float] = []
    expected_shell_types: list[int] = []
    expected_el_per_shell: list[float] = []

    for atom_idx, sym in enumerate(elements):
        md = expected_by_element[sym]
        shell_present = list(md["SHELL_PRESENT"])
        local_start = expected_local_starts(shell_present)
        local_end = expected_local_ends(shell_present)
        h_start = expected_h_start[atom_idx]
        ao_start = [s + h_start if s >= 0 else -1 for s in local_start]
        ao_end = [e + h_start if e >= 0 else -1 for e in local_end]

        expected_shell_present.append(shell_present)
        expected_shell_local_start.append(local_start)
        expected_shell_local_end.append(local_end)
        expected_shell_ao_start.append(ao_start)
        expected_shell_ao_end.append(ao_end)
        expected_labels.extend(expected_ao_labels(shell_present))
        expected_ao_types.extend(expected_ao_shell_types(shell_present))
        expected_diagonal.extend(expected_diagonal_values(md))
        expected_d0.extend(expected_d0_values(md))
        expected_n_shells.append(sum(1 for present in shell_present if present))
        expected_shell_u.extend(expected_shell_hubbard_values(md))
        expected_shell_types.extend(expected_shell_type_values(md))
        expected_el_per_shell.extend(expected_el_per_shell_values(md))

    assert_tensor_bool_list(struct.shell_present, [x for row in expected_shell_present for x in row], "Structure.shell_present")
    assert_tensor_bool_list(struct.has_s, [row[0] for row in expected_shell_present], "Structure.has_s")
    assert_tensor_bool_list(struct.has_p, [row[1] for row in expected_shell_present], "Structure.has_p")
    assert_tensor_bool_list(struct.has_d, [row[2] for row in expected_shell_present], "Structure.has_d")
    assert_tensor_bool_list(struct.has_f, [row[3] for row in expected_shell_present], "Structure.has_f")

    assert_tensor_int_list(struct.shell_local_start, [x for row in expected_shell_local_start for x in row], "Structure.shell_local_start")
    assert_tensor_int_list(struct.shell_local_end, [x for row in expected_shell_local_end for x in row], "Structure.shell_local_end")
    assert_tensor_int_list(struct.shell_ao_start, [x for row in expected_shell_ao_start for x in row], "Structure.shell_ao_start")
    assert_tensor_int_list(struct.shell_ao_end, [x for row in expected_shell_ao_end for x in row], "Structure.shell_ao_end")

    assert_list_equal(struct.ao_labels, expected_labels, "Structure.ao_labels")
    assert_tensor_int_list(struct.ao_shell_types, expected_ao_types, "Structure.ao_shell_types")
    assert_tensor_float_list(struct.diagonal, expected_diagonal, "Structure.diagonal")
    assert_tensor_float_list(struct.D0, expected_d0, "Structure.D0")

    assert_tensor_int_list(struct.n_shells_per_atom, expected_n_shells, "Structure.n_shells_per_atom")
    expected_h_start_u = [0]
    for n_shells in expected_n_shells[:-1]:
        expected_h_start_u.append(expected_h_start_u[-1] + n_shells)
    expected_h_end_u = [start + n_shells - 1 for start, n_shells in zip(expected_h_start_u, expected_n_shells)]
    assert_tensor_int_list(struct.H_INDEX_START_U, expected_h_start_u, "Structure.H_INDEX_START_U")
    assert_tensor_int_list(struct.H_INDEX_END_U, expected_h_end_u, "Structure.H_INDEX_END_U")
    assert_tensor_float_list(struct.Hubbard_U_sr, expected_shell_u, "Structure.Hubbard_U_sr")
    assert_tensor_int_list(struct.shell_types, expected_shell_types, "Structure.shell_types")
    assert_tensor_float_list(struct.el_per_shell, expected_el_per_shell, "Structure.el_per_shell")

    print()
    print("Structure.py single-structure check:")
    print(f"  elements: {elements}")
    print(f"  n_orbitals_per_atom: {expected_n_orb}")
    print(f"  HDIM: {expected_hdim}")
    print(f"  n_shells_per_atom: {expected_n_shells}")


def check_batch_structure_layout(batch_struct, batch_elements: list[list[str]], expected_by_element: dict[str, dict[str, object]]) -> None:
    """Check StructureBatch for two structures with different atom orders.

    StructureBatch stores each molecule in a padded row.  This test checks that
    the real AO region of each row is correct and that any padding after the
    molecule is zero.
    """
    batch_size = len(batch_elements)
    expected_hdim_struct: list[int] = []
    expected_types_flat: list[int] = []
    expected_n_orb_flat: list[int] = []
    expected_h_start_flat: list[int] = []
    expected_h_end_flat: list[int] = []
    expected_h_start_global_flat: list[int] = []
    expected_h_end_global_flat: list[int] = []
    expected_shell_present_flat: list[bool] = []
    expected_shell_ao_start_flat: list[int] = []
    expected_shell_ao_end_flat: list[int] = []
    expected_shell_ao_start_global_flat: list[int] = []
    expected_shell_ao_end_global_flat: list[int] = []
    expected_n_shells_flat: list[int] = []
    expected_h_start_u_flat: list[int] = []
    expected_h_end_u_flat: list[int] = []
    expected_h_start_u_global_flat: list[int] = []
    expected_h_end_u_global_flat: list[int] = []
    expected_labels: list[list[str]] = []
    expected_diagonal_rows: list[list[float]] = []
    expected_d0_rows: list[list[float]] = []

    max_hdim = 0
    global_ao_offset = 0
    global_shell_offset = 0
    for elements in batch_elements:
        n_orb_row = [int(expected_by_element[sym]["N_ORB"]) for sym in elements]
        h_start_row = [0]
        for n_orb in n_orb_row[:-1]:
            h_start_row.append(h_start_row[-1] + n_orb)
        h_end_row = [start + n_orb - 1 for start, n_orb in zip(h_start_row, n_orb_row)]
        hdim = sum(n_orb_row)
        max_hdim = max(max_hdim, hdim)
        expected_hdim_struct.append(hdim)
        expected_types_flat.extend(int(expected_by_element[sym]["Z"]) for sym in elements)
        expected_n_orb_flat.extend(n_orb_row)
        expected_h_start_flat.extend(h_start_row)
        expected_h_end_flat.extend(h_end_row)
        expected_h_start_global_flat.extend([start + global_ao_offset for start in h_start_row])
        expected_h_end_global_flat.extend([end + global_ao_offset for end in h_end_row])

        labels_row: list[str] = []
        diagonal_row: list[float] = []
        d0_row: list[float] = []
        n_shells_row: list[int] = []
        for atom_idx, sym in enumerate(elements):
            md = expected_by_element[sym]
            shell_present = list(md["SHELL_PRESENT"])
            local_start = expected_local_starts(shell_present)
            local_end = expected_local_ends(shell_present)
            h_start = h_start_row[atom_idx]
            h_start_global = h_start + global_ao_offset
            expected_shell_present_flat.extend(shell_present)
            expected_shell_ao_start_flat.extend([s + h_start if s >= 0 else -1 for s in local_start])
            expected_shell_ao_end_flat.extend([e + h_start if e >= 0 else -1 for e in local_end])
            expected_shell_ao_start_global_flat.extend([s + h_start_global if s >= 0 else -1 for s in local_start])
            expected_shell_ao_end_global_flat.extend([e + h_start_global if e >= 0 else -1 for e in local_end])
            n_shells = sum(1 for present in shell_present if present)
            expected_n_shells_flat.append(n_shells)
            n_shells_row.append(n_shells)
            labels_row.extend(expected_ao_labels(shell_present))
            diagonal_row.extend(expected_diagonal_values(md))
            d0_row.extend(expected_d0_values(md))
        h_start_u_row = [0]
        for n_shells in n_shells_row[:-1]:
            h_start_u_row.append(h_start_u_row[-1] + n_shells)
        h_end_u_row = [start + n_shells - 1 for start, n_shells in zip(h_start_u_row, n_shells_row)]
        expected_h_start_u_flat.extend(h_start_u_row)
        expected_h_end_u_flat.extend(h_end_u_row)
        expected_h_start_u_global_flat.extend([start + global_shell_offset for start in h_start_u_row])
        expected_h_end_u_global_flat.extend([end + global_shell_offset for end in h_end_u_row])
        expected_labels.append(labels_row)
        expected_diagonal_rows.append(diagonal_row)
        expected_d0_rows.append(d0_row)
        global_ao_offset += hdim
        global_shell_offset += sum(n_shells_row)

    if int(batch_struct.batch_size) != batch_size:
        raise AssertionError(f"StructureBatch.batch_size expected {batch_size}, got {batch_struct.batch_size}")
    assert_tensor_int_list(batch_struct.TYPE, expected_types_flat, "StructureBatch.TYPE")
    assert_tensor_int_list(batch_struct.n_orbitals_per_atom, expected_n_orb_flat, "StructureBatch.n_orbitals_per_atom")
    assert_tensor_int_list(batch_struct.H_INDEX_START, expected_h_start_flat, "StructureBatch.H_INDEX_START")
    assert_tensor_int_list(batch_struct.H_INDEX_END, expected_h_end_flat, "StructureBatch.H_INDEX_END")
    assert_tensor_int_list(batch_struct.H_INDEX_START_GLOBAL, expected_h_start_global_flat, "StructureBatch.H_INDEX_START_GLOBAL")
    assert_tensor_int_list(batch_struct.H_INDEX_END_GLOBAL, expected_h_end_global_flat, "StructureBatch.H_INDEX_END_GLOBAL")
    assert_tensor_int_list(batch_struct.HDIM_struct, expected_hdim_struct, "StructureBatch.HDIM_struct")

    assert_tensor_bool_list(batch_struct.shell_present, expected_shell_present_flat, "StructureBatch.shell_present")
    assert_tensor_int_list(batch_struct.shell_ao_start, expected_shell_ao_start_flat, "StructureBatch.shell_ao_start")
    assert_tensor_int_list(batch_struct.shell_ao_end, expected_shell_ao_end_flat, "StructureBatch.shell_ao_end")
    assert_tensor_int_list(batch_struct.shell_ao_start_global, expected_shell_ao_start_global_flat, "StructureBatch.shell_ao_start_global")
    assert_tensor_int_list(batch_struct.shell_ao_end_global, expected_shell_ao_end_global_flat, "StructureBatch.shell_ao_end_global")
    assert_tensor_int_list(batch_struct.n_shells_per_atom, expected_n_shells_flat, "StructureBatch.n_shells_per_atom")
    assert_tensor_int_list(batch_struct.H_INDEX_START_U, expected_h_start_u_flat, "StructureBatch.H_INDEX_START_U")
    assert_tensor_int_list(batch_struct.H_INDEX_END_U, expected_h_end_u_flat, "StructureBatch.H_INDEX_END_U")
    assert_tensor_int_list(batch_struct.H_INDEX_START_U_GLOBAL, expected_h_start_u_global_flat, "StructureBatch.H_INDEX_START_U_GLOBAL")
    assert_tensor_int_list(batch_struct.H_INDEX_END_U_GLOBAL, expected_h_end_u_global_flat, "StructureBatch.H_INDEX_END_U_GLOBAL")

    for batch_idx, labels in enumerate(expected_labels):
        assert_list_equal(batch_struct.ao_labels[batch_idx], labels, f"StructureBatch.ao_labels[{batch_idx}]")
        hdim = expected_hdim_struct[batch_idx]
        assert_tensor_float_list(batch_struct.diagonal[batch_idx, :hdim], expected_diagonal_rows[batch_idx], f"StructureBatch.diagonal[{batch_idx}]")
        assert_tensor_float_list(batch_struct.D0[batch_idx, :hdim], expected_d0_rows[batch_idx], f"StructureBatch.D0[{batch_idx}]")
        if hdim < max_hdim:
            assert_tensor_float_list(batch_struct.diagonal[batch_idx, hdim:], [0.0] * (max_hdim - hdim), f"StructureBatch.diagonal padding[{batch_idx}]")
            assert_tensor_float_list(batch_struct.D0[batch_idx, hdim:], [0.0] * (max_hdim - hdim), f"StructureBatch.D0 padding[{batch_idx}]")

    print()
    print("Structure.py batch check:")
    print(f"  batch elements: {batch_elements}")
    print(f"  HDIM per structure: {expected_hdim_struct}")
    print(f"  max padded HDIM: {max_hdim}")


def run_structure_tests(project_root: Path, skf_dir: Path, bond) -> list[dict]:
    """Run all Structure.py tests.

    This function creates exactly the objects that downstream DFTB code will use:

        Constants(...)
        Structure(...)
        StructureBatch(...)

    Then it checks that their internal AO/shell bookkeeping is mathematically
    consistent with the SKF headers.

    This test does not check the angular Slater-Koster formulas.  Those belong
    to _slater_koster_pair.py.  Here we only check indexing and metadata.
    """
    print()
    print("Testing Structure.py AO and shell bookkeeping")
    print("  checks: single Structure and batched StructureBatch")
    print("  scope: shell flags, AO starts/ends, onsite diagonal, D0, labels")

    failures = []
    try:
        # Use every element present in the SKF directory.  For your current
        # f-orbital test set this should be N, Ga, and Eu.
        elements = collect_elements_from_skf_dir(skf_dir, bond)

        # Read N-N.skf, Ga-Ga.skf, Eu-Eu.skf independently.  This gives the
        # ground truth for shell presence, shell occupations, onsite energies,
        # Hubbard U values, and the expected number of AOs per atom.
        expected_by_element = expected_metadata_by_element(skf_dir, bond, elements)

        # Build a tiny XYZ file like:
        #
        #     N   0.0 0.0 0.0
        #     Ga  1.5 0.0 0.0
        #     Eu  3.0 0.0 0.0
        #
        # The coordinates are arbitrary.  The species list is what matters.
        tmp_dir = project_root / ".tmp_f_orbital_validation"
        tmp_dir.mkdir(exist_ok=True)
        xyz_path = tmp_dir / "N_Ga_Eu_structure_test.xyz"

        # Constants reads the synthetic XYZ, loads all ordered SKF pairs, and
        # stores shell metadata.  Structure then consumes this Constants object.
        const = build_test_constants(project_root, skf_dir, bond, elements, xyz_path)

        structure_mod = load_dftorch_module(project_root, "Structure")
        params = {
            "SKFPATH": str(skf_dir),
            "FILENAME": str(xyz_path),
            "T_ELECTRONIC": 1000.0,
            "CHARGE": 0,
            "GRAD_XYZ": False,
            "GRAD_CELL": False,
        }
        # Single-structure test:
        # verifies atom-major AO indexing for one N/Ga/Eu structure.
        struct = structure_mod.Structure(params, const, device="cpu", ignore_spin=True)
        check_single_structure_layout(struct, elements, expected_by_element)

        # Batch test:
        # use the same atoms in reverse order for the second structure.
        # This catches mistakes where code accidentally assumes that atoms
        # always appear in one particular order.
        reversed_elements = list(reversed(elements))
        xyz_path_reversed = tmp_dir / "Eu_Ga_N_structure_test.xyz"
        write_test_xyz(xyz_path_reversed, reversed_elements)
        batch_params = {
            "SKFPATH": str(skf_dir),
            "FILENAME": [str(xyz_path), str(xyz_path_reversed)],
            "T_ELECTRONIC": 1000.0,
            "CHARGE": 0,
            "GRAD_XYZ": False,
            "GRAD_CELL": False,
        }
        batch_struct = structure_mod.StructureBatch(batch_params, const, device="cpu", ignore_spin=True)
        check_batch_structure_layout(batch_struct, [elements, reversed_elements], expected_by_element)
    except Exception as exc:
        failures.append({"file": "Structure.py", "error": repr(exc)})
        print()
        print(f"FAIL Structure.py check: {exc!r}")
    return failures


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    torch.set_default_dtype(torch.float64)

    project_root = find_project_root()
    bond = load_dftorch_module(project_root, "_bond_integral")

    if len(sys.argv) > 1:
        skf_dir = Path(sys.argv[1]).resolve()
    else:
        skf_dir = project_root / "tests" / "f_orbital_data"

    if not skf_dir.is_dir():
        raise FileNotFoundError(f"SKF directory not found: {skf_dir}")

    device = torch.device("cpu")
    dtype = torch.float64

    failures = []
    failures.extend(run_bond_integral_tests(skf_dir, bond, device, dtype))
    failures.extend(run_constants_tests(project_root, skf_dir, bond))
    failures.extend(run_structure_tests(project_root, skf_dir, bond))

    print()
    if failures:
        print("FAILED CHECKS:")
        for result in failures:
            print(f"  {result}")
        raise AssertionError(f"{len(failures)} test check(s) failed")

    print("All _bond_integral.py, Constants.py, and Structure.py validation tests passed.")


if __name__ == "__main__":
    main()
